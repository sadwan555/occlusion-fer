import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import occlusion_fer.stage_a_artifacts as cli_module
from occlusion_fer.mask_manifest import MASKED_CONDITIONS, ManifestWriteResult
from occlusion_fer.stage_a_artifacts import CliServices, main
from occlusion_fer.training_mean import (
    ArtifactWriteResult,
    TrainingMeanArtifact,
    canonical_artifact_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
STAGE_A_CONFIG = REPOSITORY_ROOT / "configs" / "fer2013_stage_a.yaml"
PROCESS_TIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "occlusion_fer.stage_a_artifacts",
            *(str(argument) for argument in arguments),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _tiny_csv_bytes() -> bytes:
    pixels = " ".join(["0"] * (48 * 48))
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["emotion", "pixels", "Usage"])
    writer.writerow(["0", pixels, "Training"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PublicTest"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PrivateTest"])
    return buffer.getvalue().encode("utf-8")


def _write_tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_fer2013.csv"
    path.write_bytes(_tiny_csv_bytes())
    return path


def _artifact(
    *,
    dataset_sha256: str = "a" * 64,
    official_counts: bool = True,
) -> TrainingMeanArtifact:
    sample_count = 28709 if official_counts else 1
    return TrainingMeanArtifact(
        schema_version=1,
        dataset_name="fer2013",
        split="train",
        dataset_sha256=dataset_sha256,
        sample_count=sample_count,
        pixel_count=sample_count * 48 * 48,
        raw_pixel_sum=0,
        raw_training_mean=0.0,
        source_pixel_range=(0, 255),
        normalized_pixel_range=(0.0, 1.0),
        accumulator_dtype="uint64",
        mean_algorithm_version="training-mean-v1",
    )


def _write_bound_tiny_artifact(tmp_path: Path, csv_path: Path) -> Path:
    dataset_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    artifact = _artifact(
        dataset_sha256=dataset_sha256,
        official_counts=False,
    )
    path = tmp_path / "training_mean.json"
    path.write_bytes(canonical_artifact_bytes(artifact))
    return path


def _process_event(stderr: str) -> dict[str, object]:
    lines = stderr.splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert set(event) == {"command", "created_at_utc", "event", "operation"}
    assert event["event"] == "stage-a-artifact-run"
    assert PROCESS_TIME_PATTERN.fullmatch(event["created_at_utc"])
    assert isinstance(event["command"], list)
    return event


def _unused_training_service(
    data_path: Path,
    target_path: Path,
) -> tuple[TrainingMeanArtifact, ArtifactWriteResult]:
    raise AssertionError("training service must not run")


def _unused_manifest_service(
    data_path: Path,
    mean_artifact_path: Path,
    target_path: Path,
) -> tuple[TrainingMeanArtifact, ManifestWriteResult]:
    raise AssertionError("manifest service must not run")


def test_help_exposes_only_two_fixed_subcommands() -> None:
    result = _run_cli("--help")

    assert result.returncode == 0
    assert "training-mean" in result.stdout
    assert "validation-manifest" in result.stdout
    for forbidden in (
        "--split",
        "--seed",
        "--condition",
        "--ratio",
        "--type",
        "--checkpoint",
        "--model",
        "--labels",
        "--metrics",
        "--test",
        "--private-test",
    ):
        assert forbidden not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "arguments",
    [
        ("training-mean",),
        ("training-mean", "--config", "config.yaml"),
        (
            "training-mean",
            "--config",
            "config.yaml",
            "--data-path",
            "data.csv",
        ),
        ("validation-manifest",),
        (
            "validation-manifest",
            "--config",
            "config.yaml",
            "--data-path",
            "data.csv",
            "--output-dir",
            "outputs",
        ),
    ],
)
def test_missing_required_arguments_exit_two(arguments: tuple[str, ...]) -> None:
    result = _run_cli(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert "required" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("subcommand", ["unknown", "test", "PrivateTest"])
def test_unknown_or_forbidden_subcommand_is_rejected_before_config_read(
    tmp_path: Path,
    subcommand: str,
) -> None:
    output_dir = tmp_path / "outputs"

    result = _run_cli(
        subcommand,
        "--config",
        tmp_path / "missing.yaml",
        "--data-path",
        tmp_path / "missing.csv",
        "--output-dir",
        output_dir,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "invalid choice" in result.stderr
    assert "Configuration file not found" not in result.stderr
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--split", "test"),
        ("--seed", "1"),
        ("--condition", "upper_face_0.20"),
        ("--ratio", "0.20"),
        ("--checkpoint", "best.pt"),
        ("--model", "resnet18"),
        ("--metrics", "accuracy"),
    ],
)
def test_forbidden_option_is_rejected_before_config_read(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    output_dir = tmp_path / "outputs"

    result = _run_cli(
        "training-mean",
        "--config",
        tmp_path / "missing.yaml",
        "--data-path",
        tmp_path / "missing.csv",
        "--output-dir",
        output_dir,
        option,
        value,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "unrecognized arguments" in result.stderr
    assert "Configuration file not found" not in result.stderr
    assert not output_dir.exists()


def test_training_success_uses_fixed_target_and_exact_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    data_path = tmp_path / "fer2013.csv"
    data_path.touch()
    output_dir = tmp_path / "nested" / "artifacts"
    calls: list[tuple[str, Path]] = []
    artifact = _artifact()

    def config_loader(path: Path) -> object:
        calls.append(("config", path))
        return object()

    def training_service(
        actual_data_path: Path,
        target_path: Path,
    ) -> tuple[TrainingMeanArtifact, ArtifactWriteResult]:
        assert output_dir.is_dir()
        calls.append(("data", actual_data_path))
        calls.append(("target", target_path))
        data = canonical_artifact_bytes(artifact)
        target_path.write_bytes(data)
        return artifact, ArtifactWriteResult(
            path=target_path,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    services = CliServices(
        config_loader=config_loader,
        training_mean=training_service,
        validation_manifest=_unused_manifest_service,
    )

    exit_status = main(
        [
            "training-mean",
            "--config",
            str(config_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        services=services,
    )
    captured = capsys.readouterr()
    target = output_dir / "training_mean.json"
    expected = {
        "artifact_path": str(target),
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "dataset_sha256": "a" * 64,
        "operation": "training-mean",
        "pixel_count": 66145536,
        "sample_count": 28709,
        "status": "written",
    }

    assert exit_status == 0
    assert captured.out == json.dumps(
        expected,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    assert calls == [
        ("config", config_path),
        ("data", data_path),
        ("target", target),
    ]
    event = _process_event(captured.err)
    assert event["operation"] == "training-mean"
    assert event["command"][1] == "training-mean"
    artifact_bytes = target.read_bytes()
    assert str(target).encode("utf-8") not in artifact_bytes
    assert b"created_at_utc" not in artifact_bytes
    assert b"command" not in artifact_bytes


@pytest.mark.parametrize("status", ["created", "consistent"])
def test_validation_success_uses_fixed_target_and_exact_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    data_path = tmp_path / "fer2013.csv"
    data_path.touch()
    mean_path = tmp_path / "training_mean.json"
    mean_path.touch()
    output_dir = tmp_path / "artifacts"
    artifact = _artifact()
    calls: list[tuple[str, Path]] = []

    def config_loader(path: Path) -> object:
        calls.append(("config", path))
        return object()

    def manifest_service(
        actual_data_path: Path,
        actual_mean_path: Path,
        target_path: Path,
    ) -> tuple[TrainingMeanArtifact, ManifestWriteResult]:
        assert output_dir.is_dir()
        calls.extend(
            (
                ("data", actual_data_path),
                ("mean", actual_mean_path),
                ("target", target_path),
            )
        )
        data = b"controlled canonical manifest\n"
        target_path.write_bytes(data)
        return artifact, ManifestWriteResult(
            path=target_path,
            sha256=hashlib.sha256(data).hexdigest(),
            status=status,
        )

    services = CliServices(
        config_loader=config_loader,
        training_mean=_unused_training_service,
        validation_manifest=manifest_service,
    )

    exit_status = main(
        [
            "validation-manifest",
            "--config",
            str(config_path),
            "--data-path",
            str(data_path),
            "--mean-artifact",
            str(mean_path),
            "--output-dir",
            str(output_dir),
        ],
        services=services,
    )
    captured = capsys.readouterr()
    target = output_dir / "validation_mask_manifest.csv"
    expected_counts = {condition: 3589 for condition in MASKED_CONDITIONS}
    expected = {
        "artifact_path": str(target),
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "condition_counts": expected_counts,
        "dataset_sha256": "a" * 64,
        "operation": "validation-manifest",
        "row_count": 32301,
        "sample_count": 3589,
        "status": status,
    }

    assert exit_status == 0
    assert captured.out == json.dumps(
        expected,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    assert calls == [
        ("config", config_path),
        ("data", data_path),
        ("mean", mean_path),
        ("target", target),
    ]
    event = _process_event(captured.err)
    assert event["operation"] == "validation-manifest"
    assert event["command"][1] == "validation-manifest"
    manifest_bytes = target.read_bytes()
    assert str(target).encode("utf-8") not in manifest_bytes
    assert b"created_at_utc" not in manifest_bytes
    assert b"command" not in manifest_bytes


def test_repeated_validation_dispatch_reports_consistent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    data_path = tmp_path / "fer2013.csv"
    data_path.touch()
    mean_path = tmp_path / "training_mean.json"
    mean_path.touch()
    output_dir = tmp_path / "artifacts"
    artifact = _artifact()

    def manifest_service(
        actual_data_path: Path,
        actual_mean_path: Path,
        target_path: Path,
    ) -> tuple[TrainingMeanArtifact, ManifestWriteResult]:
        status = "consistent" if target_path.exists() else "created"
        data = b"controlled canonical manifest\n"
        if status == "created":
            target_path.write_bytes(data)
        return artifact, ManifestWriteResult(
            path=target_path,
            sha256=hashlib.sha256(data).hexdigest(),
            status=status,
        )

    services = CliServices(
        config_loader=lambda path: object(),
        training_mean=_unused_training_service,
        validation_manifest=manifest_service,
    )
    arguments = [
        "validation-manifest",
        "--config",
        str(config_path),
        "--data-path",
        str(data_path),
        "--mean-artifact",
        str(mean_path),
        "--output-dir",
        str(output_dir),
    ]

    assert main(arguments, services=services) == 0
    first_output = json.loads(capsys.readouterr().out)
    assert main(arguments, services=services) == 0
    second_output = json.loads(capsys.readouterr().out)

    assert first_output["status"] == "created"
    assert second_output["status"] == "consistent"
    assert first_output["artifact_sha256"] == second_output["artifact_sha256"]


def test_invalid_config_prevents_data_service_and_output_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.touch()
    data_path = tmp_path / "fer2013.csv"
    data_path.touch()
    output_dir = tmp_path / "artifacts"
    calls: list[Path] = []

    def failing_config_loader(path: Path) -> object:
        calls.append(path)
        raise ValueError("invalid locked configuration")

    services = CliServices(
        config_loader=failing_config_loader,
        training_mean=_unused_training_service,
        validation_manifest=_unused_manifest_service,
    )

    exit_status = main(
        [
            "training-mean",
            "--config",
            str(config_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        services=services,
    )
    captured = capsys.readouterr()

    assert exit_status == 1
    assert calls == [config_path]
    assert captured.out == ""
    assert "invalid locked configuration" in captured.err
    assert "Traceback" not in captured.err
    assert not output_dir.exists()


def test_missing_data_path_fails_before_output_or_service(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    output_dir = tmp_path / "artifacts"
    config_calls: list[Path] = []

    def config_loader(path: Path) -> object:
        config_calls.append(path)
        return object()

    services = CliServices(
        config_loader=config_loader,
        training_mean=_unused_training_service,
        validation_manifest=_unused_manifest_service,
    )

    exit_status = main(
        [
            "training-mean",
            "--config",
            str(config_path),
            "--data-path",
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(output_dir),
        ],
        services=services,
    )
    captured = capsys.readouterr()

    assert exit_status == 1
    assert config_calls == [config_path]
    assert captured.out == ""
    assert "data path" in captured.err.lower()
    assert not output_dir.exists()


def test_service_failure_returns_one_without_success_summary_or_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    data_path = tmp_path / "fer2013.csv"
    data_path.touch()
    output_dir = tmp_path / "artifacts"

    def failing_service(
        actual_data_path: Path,
        target_path: Path,
    ) -> tuple[TrainingMeanArtifact, ArtifactWriteResult]:
        raise RuntimeError("controlled service failure")

    services = CliServices(
        config_loader=lambda path: object(),
        training_mean=failing_service,
        validation_manifest=_unused_manifest_service,
    )

    exit_status = main(
        [
            "training-mean",
            "--config",
            str(config_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        services=services,
    )
    captured = capsys.readouterr()

    assert exit_status == 1
    assert captured.out == ""
    assert "controlled service failure" in captured.err
    assert "Traceback" not in captured.err
    assert not (output_dir / "training_mean.json").exists()


def test_default_services_bind_only_existing_formal_artifact_apis() -> None:
    assert cli_module.DEFAULT_SERVICES.config_loader is cli_module.load_occlusion_config
    assert cli_module.DEFAULT_SERVICES.training_mean is cli_module._run_training_mean
    assert (
        cli_module.DEFAULT_SERVICES.validation_manifest
        is cli_module._run_validation_manifest
    )

    source = Path(cli_module.__file__).read_text(encoding="utf-8")
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_roots.isdisjoint(
        {
            "torch",
            "torchvision",
            "model",
            "train",
            "evaluation",
            "metrics",
            "checkpoint",
        }
    )


def test_tiny_csv_training_command_fails_official_count_without_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "training-output"

    result = _run_cli(
        "training-mean",
        "--config",
        STAGE_A_CONFIG,
        "--data-path",
        _write_tiny_csv(tmp_path),
        "--output-dir",
        output_dir,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "28709" in result.stderr
    assert "Traceback" not in result.stderr
    assert output_dir.is_dir()
    assert not (output_dir / "training_mean.json").exists()


def test_tiny_csv_validation_command_fails_official_count_without_manifest(
    tmp_path: Path,
) -> None:
    csv_path = _write_tiny_csv(tmp_path)
    mean_path = _write_bound_tiny_artifact(tmp_path, csv_path)
    output_dir = tmp_path / "manifest-output"

    result = _run_cli(
        "validation-manifest",
        "--config",
        STAGE_A_CONFIG,
        "--data-path",
        csv_path,
        "--mean-artifact",
        mean_path,
        "--output-dir",
        output_dir,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "3589" in result.stderr
    assert "Traceback" not in result.stderr
    assert output_dir.is_dir()
    assert not (output_dir / "validation_mask_manifest.csv").exists()
