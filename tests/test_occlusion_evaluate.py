from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import pytest
from torch import nn

from occlusion_fer.mask_manifest import (
    build_manifest_v2_rows,
    make_manifest_v2_envelope,
    write_manifest_v2_create_or_verify,
)
from occlusion_fer.occlusion import V2_MASKED_CONDITIONS
from occlusion_fer.occlusion_evaluate import (
    OcclusionEvaluationError,
    evaluate_publictest_source,
    evaluate_clean_and_nine_conditions,
    load_masked_evaluation_checkpoint,
    main,
    validate_publictest_route,
    write_occlusion_evaluation_artifacts,
)
from occlusion_fer.permitted_splits import (
    SplitRecord,
    load_stage_b_source,
    make_permitted_splits,
)
from occlusion_fer.training_mean import (
    calculate_training_mean_v2,
    training_mean_v2_sha256,
    write_training_mean_v2,
)


def _record(sample_id: int, value: int) -> SplitRecord:
    return SplitRecord(sample_id, sample_id % 7, (value,) * (48 * 48))


def _manifest(sample_ids: list[int]):
    splits = make_permitted_splits(
        [_record(1, 1)],
        [_record(sample_id, sample_id) for sample_id in sample_ids],
    )
    mean = calculate_training_mean_v2(splits.training)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        sample_ids,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )
    return splits, mean, rows, envelope


def _clean_checkpoint_payload() -> dict[str, object]:
    return {
        "model_state_dict": {"synthetic": torch.zeros(1)},
        "resolved_config": {
            "dataset": {"image_size": 224},
            "training": {"mode": "clean"},
            "occlusion": None,
        },
    }


def _provenance() -> dict[str, object]:
    return {
        "checkpoint_sha256": "a" * 64,
        "training_commit": "training-commit",
        "evaluation_commit": "evaluation-commit",
        "git_dirty": True,
        "resolved_config": {"training": {"mode": "clean"}},
        "training_dataset_sha256": "b" * 64,
        "publictest_dataset_sha256": "c" * 64,
        "training_mean_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "protocol": "occlusion-v2-224",
        "image_height": 224,
        "image_width": 224,
        "seed": 42,
        "run_role": "screening_masked_evaluation",
    }


def _synthetic_cli_fixture(tmp_path: Path):
    source_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["7"] * (48 * 48))
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerow([1, pixels, "Training"])
        writer.writerow([2, pixels, "PublicTest"])
        writer.writerow(["DO NOT PARSE", "DO NOT PARSE", "PrivateTest"])

    splits = load_stage_b_source(source_path)
    mean = calculate_training_mean_v2(splits.training)
    mean_path = tmp_path / "training_mean_v2.json"
    write_training_mean_v2(mean_path, mean)
    rows = build_manifest_v2_rows(
        [record.sample_id for record in splits.publictest.records],
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=training_mean_v2_sha256(mean),
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=training_mean_v2_sha256(mean),
    )
    manifest_path = tmp_path / "publictest_manifest_v2.csv"
    write_manifest_v2_create_or_verify(
        manifest_path,
        manifest_path.with_suffix(".json"),
        rows,
        envelope,
    )
    config_path = tmp_path / "evaluation.yaml"
    template = Path(
        "configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml"
    ).read_text(encoding="utf-8")
    config_path.write_text(
        template
        .replace("  path: /path/to/fer2013.csv", f"  path: {source_path}")
        .replace("  pretrained: true", "  pretrained: false")
        .replace("  num_workers: 4", "  num_workers: 0")
        .replace("  device: cuda", "  device: cpu")
        .replace("/path/to/training_mean_v2.json", str(mean_path))
        .replace("/path/to/publictest_manifest_v2.csv", str(manifest_path)),
        encoding="utf-8",
    )
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_config": {
                "dataset": {"image_size": 224},
                "training": {"mode": "clean"},
                "occlusion": None,
            },
            "seed": 42,
        },
        checkpoint_path,
    )
    return source_path, mean_path, manifest_path, config_path, checkpoint_path, model


def test_occlusion_evaluator_accepts_usage_routed_csv_and_rejects_private_route(
    tmp_path,
) -> None:
    with pytest.raises(OcclusionEvaluationError, match="PrivateTest"):
        validate_publictest_route(split="PrivateTest", private_test=True)
    validate_publictest_route(
        split="PublicTest",
        dataset_path=tmp_path / "combined.csv",
    )


def test_masked_evaluator_strict_loads_checkpoint_and_returns_file_identity(tmp_path) -> None:
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_config": {
                "dataset": {"image_size": 224},
                "training": {"mode": "clean"},
                "occlusion": None,
            },
        },
        checkpoint,
    )
    restored = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    payload, digest = load_masked_evaluation_checkpoint(restored, checkpoint)
    assert payload["resolved_config"]["dataset"]["image_size"] == 224
    assert digest == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_evaluator_writes_clean_and_nine_condition_artifacts(tmp_path) -> None:
    sample_ids = [2, 3, 4]
    splits, mean, rows, envelope = _manifest(sample_ids)
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    results = evaluate_clean_and_nine_conditions(
        model,
        torch.zeros(3, 3, 224, 224),
        torch.tensor([2, 3, 4]),
        torch.tensor(sample_ids),
        torch.device("cpu"),
        training_mean_artifact=mean,
        training_dataset_sha256=splits.training.dataset_sha256,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        manifest_rows=rows,
        manifest_envelope=envelope,
        checkpoint_payload=_clean_checkpoint_payload(),
        require_official_manifest=False,
    )
    paths = write_occlusion_evaluation_artifacts(
        tmp_path,
        results,
        provenance=_provenance(),
    )
    assert tuple(results) == ("clean",) + V2_MASKED_CONDITIONS
    assert len(paths) == 60
    assert all(path.is_file() for path in paths.values())


def test_evaluator_rejects_manifest_runtime_sample_mismatch() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    with pytest.raises(OcclusionEvaluationError, match="sample IDs"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([3]),
            torch.tensor([3]),
            torch.device("cpu"),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=splits.publictest.dataset_sha256,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )


def test_mixed_checkpoint_mean_identity_must_match_manifest() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    mixed_payload = {
        "model_state_dict": {"synthetic": torch.zeros(1)},
        "resolved_config": {
            "dataset": {"image_size": 224},
            "training": {"mode": "mixed"},
            "occlusion": {
                "protocol": {
                    "algorithm_version": "occlusion-v2-224",
                    "mean_algorithm_version": "training-mean-v2",
                    "image_size": 224,
                },
            },
            "occlusion_runtime": {
                "training_mean_sha256": "f" * 64,
                "training_dataset_sha256": splits.training.dataset_sha256,
                "publictest_dataset_sha256": splits.publictest.dataset_sha256,
                "source_routing_version": "combined-usage-routing-v1",
            },
        },
    }
    with pytest.raises(OcclusionEvaluationError, match="mean identity"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=splits.publictest.dataset_sha256,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=mixed_payload,
            require_official_manifest=False,
        )


def test_mixed_checkpoint_publictest_identity_must_match_manifest() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    mixed_payload = {
        "model_state_dict": {"synthetic": torch.zeros(1)},
        "resolved_config": {
            "dataset": {"image_size": 224},
            "training": {"mode": "mixed"},
            "occlusion": {
                "protocol": {
                    "algorithm_version": "occlusion-v2-224",
                    "mean_algorithm_version": "training-mean-v2",
                    "image_size": 224,
                },
            },
            "occlusion_runtime": {
                "training_mean_sha256": envelope.training_mean_artifact_sha256,
                "training_dataset_sha256": splits.training.dataset_sha256,
                "publictest_dataset_sha256": "0" * 64,
                "source_routing_version": "combined-usage-routing-v1",
            },
        },
    }
    with pytest.raises(OcclusionEvaluationError, match="PublicTest identity"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=envelope.publictest_dataset_sha256,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=mixed_payload,
            require_official_manifest=False,
        )


def test_mixed_checkpoint_training_identity_must_match_runtime() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    mixed_payload = {
        "model_state_dict": {"synthetic": torch.zeros(1)},
        "resolved_config": {
            "dataset": {"image_size": 224},
            "training": {"mode": "mixed"},
            "occlusion": {
                "protocol": {
                    "algorithm_version": "occlusion-v2-224",
                    "mean_algorithm_version": "training-mean-v2",
                    "image_size": 224,
                },
            },
            "occlusion_runtime": {
                "training_mean_sha256": envelope.training_mean_artifact_sha256,
                "training_dataset_sha256": "0" * 64,
                "publictest_dataset_sha256": splits.publictest.dataset_sha256,
                "source_routing_version": "combined-usage-routing-v1",
            },
        },
    }
    with pytest.raises(OcclusionEvaluationError, match="Training identity"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=splits.publictest.dataset_sha256,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=mixed_payload,
            require_official_manifest=False,
        )


def test_formal_evaluator_rejects_unbound_fill_override() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    with pytest.raises(TypeError, match="fill_vector"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            fill_vector=(999.0, 999.0, 999.0),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=envelope.publictest_dataset_sha256,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )


def test_publictest_source_identity_changes_when_content_changes() -> None:
    first = make_permitted_splits([_record(1, 1)], [_record(2, 2)])
    second = make_permitted_splits([_record(1, 1)], [_record(2, 3)])
    assert first.publictest.dataset_sha256 != second.publictest.dataset_sha256


def test_publictest_label_only_identity_change_fails_before_forward() -> None:
    pixels = (7,) * (48 * 48)
    training = [SplitRecord(1, 1, pixels)]
    manifest_source = make_permitted_splits(
        training,
        [SplitRecord(2, 2, pixels)],
    )
    runtime_source = make_permitted_splits(
        training,
        [SplitRecord(2, 3, pixels)],
    )
    assert manifest_source.publictest.dataset_sha256 != (
        runtime_source.publictest.dataset_sha256
    )

    mean = calculate_training_mean_v2(runtime_source.training)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [2],
        publictest_dataset_sha256=manifest_source.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=manifest_source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )

    class ForwardSpy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(self, images):
            self.forward_calls += 1
            return torch.zeros(images.shape[0], 7)

    model = ForwardSpy()
    with pytest.raises(OcclusionEvaluationError, match="runtime PublicTest identity"):
        evaluate_publictest_source(
            model,
            runtime_source,
            torch.device("cpu"),
            image_size=224,
            batch_size=1,
            num_workers=0,
            training_mean_artifact=mean,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )
    assert model.forward_calls == 0


def test_evaluator_provenance_publictest_identity_must_match_runtime() -> None:
    splits, mean, rows, envelope = _manifest([2])
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    provenance = {
        **_provenance(),
        "training_dataset_sha256": splits.training.dataset_sha256,
        "publictest_dataset_sha256": "0" * 64,
        "training_mean_sha256": training_mean_v2_sha256(mean),
        "manifest_sha256": envelope.manifest_sha256,
    }
    with pytest.raises(OcclusionEvaluationError, match="provenance.*PublicTest"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            training_mean_artifact=mean,
            training_dataset_sha256=splits.training.dataset_sha256,
            publictest_dataset_sha256=envelope.publictest_dataset_sha256,
            evaluation_provenance=provenance,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )


def test_publictest_source_requires_exact_content_identity_before_forward() -> None:
    source = make_permitted_splits(
        [_record(1, 1)],
        [_record(2, 3)],
    )
    manifest_source = make_permitted_splits(
        [_record(1, 1)],
        [_record(2, 2)],
    )
    mean = calculate_training_mean_v2(source.training)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [2],
        publictest_dataset_sha256=manifest_source.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=manifest_source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )

    class MustNotForward(nn.Module):
        def forward(self, images):  # pragma: no cover - failure is the assertion
            raise AssertionError("model forward must not run on an identity mismatch")

    with pytest.raises(OcclusionEvaluationError, match="runtime PublicTest identity"):
        evaluate_publictest_source(
            MustNotForward(),
            source,
            torch.device("cpu"),
            image_size=224,
            batch_size=1,
            num_workers=0,
            training_mean_artifact=mean,
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )


def test_publictest_source_exact_identity_evaluates_all_conditions() -> None:
    source = make_permitted_splits([_record(1, 1)], [_record(2, 2)])
    mean = calculate_training_mean_v2(source.training)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [2],
        publictest_dataset_sha256=source.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(3, 7),
    )
    results = evaluate_publictest_source(
        model,
        source,
        torch.device("cpu"),
        image_size=224,
        batch_size=1,
        num_workers=0,
        training_mean_artifact=mean,
        manifest_rows=rows,
        manifest_envelope=envelope,
        checkpoint_payload=_clean_checkpoint_payload(),
        require_official_manifest=False,
    )
    assert tuple(results) == ("clean",) + V2_MASKED_CONDITIONS


def test_cli_help_is_a_real_argparse_entrypoint() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": "src",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "occlusion_fer.occlusion_evaluate", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--training-mean" in completed.stdout


def test_cli_synthetic_end_to_end_writes_complete_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    source_path, mean_path, manifest_path, config_path, checkpoint_path, _ = (
        _synthetic_cli_fixture(tmp_path)
    )
    import occlusion_fer.occlusion_evaluate as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "create_resnet18",
        lambda num_classes, pretrained: nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, num_classes)
        ),
    )
    output_dir = tmp_path / "evaluation"
    assert main([
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--training-mean", str(mean_path),
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--batch-size", "1",
        "--num-workers", "0",
        "--allow-nonofficial",
    ]) == 0
    captured = capsys.readouterr()
    assert '"status": "completed"' in captured.out
    assert (output_dir / "evaluation_provenance.json").is_file()
    assert len(tuple((output_dir / "conditions").glob("*_metrics.json"))) == 10
    provenance = json.loads(
        (output_dir / "evaluation_provenance.json").read_text(encoding="utf-8")
    )
    source = load_stage_b_source(source_path)
    assert provenance["publictest_dataset_sha256"] == source.publictest.dataset_sha256
    assert provenance["source_routing_version"] == "combined-usage-routing-v1"
    assert provenance["training_mean_sha256"] == training_mean_v2_sha256(
        calculate_training_mean_v2(source.training)
    )


def test_cli_does_not_overwrite_existing_evaluation_directory(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    marker = output_dir / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    assert main([
        "--config", str(tmp_path / "missing.yaml"),
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--output-dir", str(output_dir),
        "--allow-nonofficial",
    ]) == 1
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / "failure.json").exists()


def test_cli_failure_returns_nonzero_and_records_failure(tmp_path) -> None:
    output_dir = tmp_path / "failed"
    assert main([
        "--config", str(tmp_path / "missing.yaml"),
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--output-dir", str(output_dir),
        "--allow-nonofficial",
    ]) == 1
    payload = json.loads(
        (output_dir / "failure.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["stage"] == "publictest_evaluation"


def test_cli_mid_artifact_failure_is_incomplete_and_not_overwritten(
    tmp_path, monkeypatch
) -> None:
    source_path, mean_path, manifest_path, config_path, checkpoint_path, _ = (
        _synthetic_cli_fixture(tmp_path)
    )
    import occlusion_fer.occlusion_evaluate as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "create_resnet18",
        lambda num_classes, pretrained: nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, num_classes)
        ),
    )
    real_writer = evaluator_module.write_evaluation_artifacts
    write_calls = 0

    def fail_during_second_condition(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("synthetic mid-artifact write failure")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(
        evaluator_module,
        "write_evaluation_artifacts",
        fail_during_second_condition,
    )
    output_dir = tmp_path / "partial-evaluation"
    arguments = [
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--training-mean", str(mean_path),
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--batch-size", "1",
        "--num-workers", "0",
        "--allow-nonofficial",
    ]

    assert main(arguments) == 1
    assert write_calls == 2
    assert tuple((output_dir / "conditions").iterdir())
    failure = json.loads(
        (output_dir / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert not (output_dir / "evaluation_provenance.json").exists()

    before_retry = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert main(arguments) == 1
    after_retry = {
        path.relative_to(output_dir): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after_retry == before_retry
