import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import occlusion_fer.preflight as preflight


PIXELS = " ".join(str(index % 256) for index in range(48 * 48))


def write_csv(
    tmp_path: Path,
    *,
    train_labels: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 0),
    validation_labels: tuple[int, ...] = (0, 1, 2, 3),
    test_labels: tuple[int, ...] = (4, 5),
) -> Path:
    csv_path = tmp_path / "fer2013.csv"
    rows = [
        *(dict(emotion=label, pixels=PIXELS, Usage="Training") for label in train_labels),
        *(
            dict(emotion=label, pixels=PIXELS, Usage="PublicTest")
            for label in validation_labels
        ),
        *(
            dict(emotion=label, pixels=PIXELS, Usage="PrivateTest")
            for label in test_labels
        ),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_config(
    tmp_path: Path,
    *,
    data_path: str | None = None,
    output_directory: str | None = None,
    pretrained: bool = True,
) -> Path:
    selected_data_path = data_path or str(write_csv(tmp_path))
    selected_output = output_directory or str(tmp_path / "output")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""\
project:
  name: occlusion-fer
dataset:
  name: fer2013
  path: {selected_data_path}
  image_size: 112
  num_classes: 7
model:
  name: resnet18
  pretrained: {str(pretrained).lower()}
training:
  mode: clean
  seed: 42
  epochs: 1
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 0
  device: auto
output:
  directory: {selected_output}
""",
        encoding="utf-8",
    )
    return config_path


def parse_args(config_path: Path, *extra: str) -> object:
    return preflight.build_parser().parse_args(
        ["--config", str(config_path), *extra]
    )


class TinyForwardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.inference_mode_enabled = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.inference_mode_enabled = torch.is_inference_mode_enabled()
        return self.weight * torch.ones(images.shape[0], 7, device=images.device)


def test_parser_supports_every_preflight_override(tmp_path: Path) -> None:
    args = parse_args(
        write_config(tmp_path),
        "--data-path",
        "/runtime/fer2013.csv",
        "--output-dir",
        "/runtime/output",
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--max-train-samples",
        "32",
        "--max-validation-samples",
        "16",
        "--skip-model-forward",
    )

    assert args.data_path == "/runtime/fer2013.csv"
    assert args.output_dir == "/runtime/output"
    assert args.device == "cpu"
    assert args.batch_size == 8
    assert args.max_train_samples == 32
    assert args.max_validation_samples == 16
    assert args.skip_model_forward is True


def test_parser_leaves_optional_overrides_unset(tmp_path: Path) -> None:
    args = parse_args(write_config(tmp_path))

    assert args.device is None
    assert args.batch_size is None
    assert args.max_train_samples is None
    assert args.max_validation_samples is None
    assert args.skip_model_forward is False


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--device", "gpu"),
        ("--batch-size", "0"),
        ("--batch-size", "-1"),
        ("--max-train-samples", "0"),
        ("--max-validation-samples", "-2"),
    ],
)
def test_parser_rejects_invalid_bounded_values(
    tmp_path: Path, option: str, value: str
) -> None:
    with pytest.raises(SystemExit):
        parse_args(write_config(tmp_path), option, value)


@pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda"])
def test_parser_accepts_supported_devices(tmp_path: Path, device: str) -> None:
    args = parse_args(write_config(tmp_path), "--device", device)

    assert args.device == device


def test_placeholder_data_path_fails_before_model_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called = False

    def forbidden_model(**_: object) -> nn.Module:
        nonlocal called
        called = True
        raise AssertionError("model must not be created")

    monkeypatch.setattr(preflight, "create_resnet18", forbidden_model)
    args = parse_args(
        write_config(tmp_path, data_path="/path/to/fer2013.csv"),
        "--device",
        "cpu",
    )

    result = preflight.run_preflight(args)

    assert result.success is False
    assert result.failed_stage == "data file"
    assert called is False
    assert "placeholder" in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["missing", "directory", "empty"])
def test_invalid_data_file_fails_with_clear_reason(
    tmp_path: Path, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    data_path = tmp_path / kind
    if kind == "directory":
        data_path.mkdir()
    elif kind == "empty":
        data_path.write_bytes(b"")
    args = parse_args(
        write_config(tmp_path, data_path=str(data_path)),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is False
    assert result.failed_stage == "data file"
    assert "PREFLIGHT FAILED" in output
    assert "reason=" in output


def test_unreadable_data_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = write_csv(tmp_path)
    real_access = os.access

    def selective_access(path: object, mode: int) -> bool:
        if Path(path) == csv_path:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(preflight.os, "access", selective_access)

    with pytest.raises(preflight.PreflightError, match="not readable"):
        preflight.validate_data_path(csv_path)


def test_training_preflight_counts_and_class_names_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse_args(
        write_config(tmp_path),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert "total_samples=12" in output
    assert "train_samples=8" in output
    assert "validation_samples=4" in output
    assert "test_samples=" not in output
    assert "0:angry=2" in output
    assert "6:neutral=1" in output


def test_preflight_does_not_parse_or_report_private_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = write_csv(tmp_path)
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        if row["Usage"] == "PrivateTest":
            row["emotion"] = "not parsed"
            row["pixels"] = "not parsed"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["emotion", "pixels", "Usage"]
        )
        writer.writeheader()
        writer.writerows(rows)

    config_path = write_config(tmp_path, data_path=str(csv_path))
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  num_classes: 7\n",
            """  num_classes: 7
  augmentation:
    type: mild_affine
    horizontal_flip_probability: 0.5
    affine_probability: 0.5
    degrees: 7.0
    translate: [0.05, 0.05]
    scale: [0.97, 1.03]
    interpolation: bilinear
    fill: 0.0
""",
            1,
        ),
        encoding="utf-8",
    )
    args = parse_args(
        config_path,
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert "test_samples=" not in output
    assert "test_class_counts=" not in output


def test_e3_preflight_uses_unsmoothed_validation_and_skips_private_test(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = write_csv(tmp_path)
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    for row in rows:
        if row["Usage"] == "PrivateTest":
            row["emotion"] = "not parsed"
            row["pixels"] = "not parsed"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["emotion", "pixels", "Usage"]
        )
        writer.writeheader()
        writer.writerows(rows)

    config_path = write_config(tmp_path, data_path=str(csv_path))
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  device: auto\n",
            """  device: auto
  loss:
    type: cross_entropy
    label_smoothing: 0.1
""",
            1,
        ),
        encoding="utf-8",
    )
    args = parse_args(
        config_path,
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert "test_samples=" not in output
    assert "test_class_counts=" not in output


def test_missing_classes_are_warnings_not_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = write_csv(
        tmp_path,
        train_labels=(0, 0),
        validation_labels=(1,),
        test_labels=(2,),
    )
    args = parse_args(
        write_config(tmp_path, data_path=str(csv_path)),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert len(result.warnings) == 2
    assert "WARNINGS" in output
    assert "train split missing classes" in output
    assert "validation split missing classes" in output
    assert "test split missing classes" not in output


def test_sample_limits_only_change_loader_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse_args(
        write_config(tmp_path),
        "--device",
        "cpu",
        "--max-train-samples",
        "3",
        "--max-validation-samples",
        "2",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert "total_samples=12" in output
    assert "train_samples=8" in output
    assert "PREFLIGHT SAMPLE LIMIT — NOT A FORMAL EXPERIMENT" in output
    assert "actual_train_samples=3" in output
    assert "actual_validation_samples=2" in output


def test_first_batches_are_unshuffled_normalized_and_finite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse_args(
        write_config(tmp_path),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert "train_batch_shape=(2, 3, 112, 112)" in output
    assert "validation_batch_shape=(2, 3, 112, 112)" in output
    assert "train_image_dtype=torch.float32" in output
    assert "train_label_dtype=torch.int64" in output
    assert "train_sample_ids=[2, 3]" in output
    assert "train_images_finite=True" in output


def test_inspect_batch_rejects_wrong_image_shape() -> None:
    batch = (
        torch.zeros(2, 1, 112, 112),
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
    )

    with pytest.raises(preflight.PreflightError, match="shape"):
        preflight.inspect_batch("train", batch, image_size=112)


@pytest.mark.parametrize(
    ("labels", "sample_ids", "message"),
    [
        (torch.tensor([0.0, 1.0]), torch.tensor([2, 3]), "labels.*torch.int64"),
        (torch.tensor([0, 1]), torch.tensor([2.0, 3.0]), "sample IDs.*torch.int64"),
        (torch.tensor([0, 7]), torch.tensor([2, 3]), "labels.*0.*6"),
    ],
)
def test_inspect_batch_rejects_invalid_labels_or_ids(
    labels: torch.Tensor, sample_ids: torch.Tensor, message: str
) -> None:
    batch = (torch.zeros(2, 3, 112, 112), labels, sample_ids)

    with pytest.raises(preflight.PreflightError, match=message):
        preflight.inspect_batch("train", batch, image_size=112)


def test_inspect_batch_rejects_non_finite_images() -> None:
    images = torch.zeros(2, 3, 112, 112)
    images[0, 0, 0, 0] = float("nan")

    with pytest.raises(preflight.PreflightError, match="non-finite"):
        preflight.inspect_batch(
            "train", (images, torch.tensor([0, 1]), torch.tensor([2, 3])), 112
        )


def test_forward_forces_random_weights_eval_and_inference_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received_pretrained: list[bool] = []
    model = TinyForwardModel()

    def create_tiny(*, num_classes: int, pretrained: bool) -> nn.Module:
        assert num_classes == 7
        received_pretrained.append(pretrained)
        return model

    monkeypatch.setattr(preflight, "create_resnet18", create_tiny)
    args = parse_args(write_config(tmp_path, pretrained=True), "--device", "cpu")

    result = preflight.run_preflight(args)

    assert result.success is True
    assert received_pretrained == [False]
    assert model.training is False
    assert model.inference_mode_enabled is True
    assert model.weight.grad is None


@pytest.mark.parametrize(
    ("class_count", "fill_value", "message"),
    [
        (6, 0.0, "logits.*shape"),
        (7, float("nan"), "logits.*non-finite"),
    ],
)
def test_forward_rejects_wrong_or_non_finite_logits(
    monkeypatch: pytest.MonkeyPatch,
    class_count: int,
    fill_value: float,
    message: str,
) -> None:
    class InvalidModel(nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.full((images.shape[0], class_count), fill_value)

    monkeypatch.setattr(
        preflight,
        "create_resnet18",
        lambda **_: InvalidModel(),
    )

    with pytest.raises(preflight.PreflightError, match=message):
        preflight.run_model_forward_check(
            torch.zeros(2, 3, 112, 112),
            torch.device("cpu"),
            num_classes=7,
        )


def test_skip_forward_does_not_create_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden_model(**_: object) -> nn.Module:
        raise AssertionError("model creation was not skipped")

    monkeypatch.setattr(preflight, "create_resnet18", forbidden_model)
    args = parse_args(
        write_config(tmp_path),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    assert result.success is True
    assert "model_forward=SKIPPED" in capsys.readouterr().out


def test_output_check_creates_directory_and_removes_only_its_marker(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "nested" / "output"

    preflight.check_output_directory(output_directory)

    assert output_directory.is_dir()
    assert list(output_directory.iterdir()) == []


def test_output_check_preserves_existing_marker_file(tmp_path: Path) -> None:
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    marker = output_directory / ".preflight_write_test"
    marker.write_text("belongs to user", encoding="utf-8")

    with pytest.raises(preflight.PreflightError, match="already exists"):
        preflight.check_output_directory(output_directory)

    assert marker.read_text(encoding="utf-8") == "belongs to user"


def test_environment_information_contains_required_versions_and_devices() -> None:
    info = preflight.collect_environment_info(
        requested_device="cpu", selected_device=torch.device("cpu")
    )

    assert {
        "python_version",
        "operating_system",
        "architecture",
        "torch_version",
        "torchvision_version",
        "numpy_version",
        "requested_device",
        "selected_device",
        "cuda_available",
        "mps_available",
    } <= info.keys()
    assert info["requested_device"] == "cpu"
    assert info["selected_device"] == "cpu"


def test_success_prints_pass_summary_and_returns_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse_args(
        write_config(tmp_path),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is True
    assert result.failed_stage is None
    assert "environment=PASS" in output
    assert "data=PASS" in output
    assert "batches=PASS" in output
    assert "output_directory=PASS" in output
    assert output.rstrip().endswith("PREFLIGHT PASSED")


def test_failure_prints_stage_reason_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = parse_args(
        write_config(tmp_path, data_path=str(tmp_path / "missing.csv")),
        "--device",
        "cpu",
        "--skip-model-forward",
    )

    result = preflight.run_preflight(args)

    output = capsys.readouterr().out
    assert result.success is False
    assert "PREFLIGHT FAILED" in output
    assert "failed_stage=data file" in output
    assert "reason=" in output
    assert "Traceback" not in output


def test_module_cli_runs_real_resnet_forward_without_artifacts(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, pretrained=True)
    output_directory = tmp_path / "cli-output"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "occlusion_fer.preflight",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_directory),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-train-samples",
            "2",
            "--max-validation-samples",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "selected_device=cpu" in completed.stdout
    assert "model_weights=pretrained_false_for_preflight" in completed.stdout
    assert "model_logits_shape=(2, 7)" in completed.stdout
    assert "model_forward=PASS" in completed.stdout
    assert "PREFLIGHT PASSED" in completed.stdout
    assert list(output_directory.iterdir()) == []
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))
    assert not list(tmp_path.rglob("*.ckpt"))


def test_module_cli_returns_nonzero_for_missing_data(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path, data_path=str(tmp_path / "not-present.csv")
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "occlusion_fer.preflight",
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--skip-model-forward",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "PREFLIGHT FAILED" in completed.stdout
    assert "failed_stage=data file" in completed.stdout
    assert "Traceback" not in completed.stderr
