from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
from dataclasses import replace
from argparse import Namespace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from occlusion_fer.checkpoint_compat import (
    CheckpointCompatibilityError,
    validate_checkpoint_for_route,
)
from occlusion_fer.config import ConfigError, load_config
from occlusion_fer.occlusion import (
    V2_MASKED_CONDITIONS,
    apply_evaluation_batch_v2,
    evaluation_geometry_v2,
    select_training_condition_v2,
    normalized_fill_vector_v2,
)
from occlusion_fer.occlusion_evaluate import OcclusionEvaluationError, validate_publictest_route
from occlusion_fer.occlusion_evaluate import (
    evaluate_clean_and_nine_conditions,
    write_occlusion_evaluation_artifacts,
)
from occlusion_fer.permitted_splits import (
    PermittedSplitError,
    SplitRecord,
    make_permitted_splits,
    reject_combined_dataset_path,
)
from occlusion_fer.training_mean import (
    calculate_training_mean_v2,
    training_mean_v2_sha256,
    write_training_mean_v2,
)
from occlusion_fer.mask_manifest import (
    build_manifest_v2_rows,
    load_manifest_v2,
    manifest_v2_sha256,
    make_manifest_v2_envelope,
    ManifestConflictError,
    write_manifest_v2_create_or_verify,
)
from occlusion_fer.train import train_one_epoch
from occlusion_fer.preflight import run_preflight


def _record(sample_id: int, value: int) -> SplitRecord:
    return SplitRecord(sample_id, sample_id % 7, (value,) * (48 * 48))


def test_v2_hash_payload_and_geometry_are_224() -> None:
    from occlusion_fer.mask_hash import build_v2_training_decision_payload

    assert build_v2_training_decision_payload(7, 42, 1, "apply") == [
        "fer2013", "train", 7, 42, 1, 224, 224, "occlusion-v2-224", "apply"
    ]
    assert evaluation_geometry_v2(7, "upper_face_0.20").masked_pixel_count == 10080
    assert evaluation_geometry_v2(7, "random_rectangle_0.40").height == 142


def test_v2_mask_is_immutable_and_deterministic() -> None:
    clean = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(2, 3, 224, 224)
    original = clean.clone()
    first, metadata = apply_evaluation_batch_v2(
        clean, torch.tensor([1, 2]), "random_rectangle_0.30", (0.1, 0.2, 0.3)
    )
    second, metadata_again = apply_evaluation_batch_v2(
        clean, torch.tensor([1, 2]), "random_rectangle_0.30", (0.1, 0.2, 0.3)
    )
    assert torch.equal(clean, original)
    assert torch.equal(first, second)
    assert metadata == metadata_again
    assert all(item.image_height == 224 and item.total_pixel_count == 50176 for item in metadata)


def test_v2_selection_does_not_depend_on_batch_order() -> None:
    forward = [select_training_condition_v2(i, 42, 1) for i in range(1, 12)]
    reverse = {i: select_training_condition_v2(i, 42, 1) for i in reversed(range(1, 12))}
    assert forward == [reverse[i] for i in range(1, 12)]


def test_v2_selection_respects_general_clean_probability_edges() -> None:
    assert select_training_condition_v2(1, 42, 1, clean_probability=0.0) is not None
    assert select_training_condition_v2(1, 42, 1, clean_probability=1.0) is None


def test_mixed_train_masks_only_selected_samples_and_records_counts() -> None:
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    sample_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=4)
    model = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 7))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    result = train_one_epoch(
        model, loader, optimizer, torch.device("cpu"),
        training_seed=42, epoch=1, fill_vector=(0.1, 0.2, 0.3),
    )
    assert sum(result.condition_counts.values()) == 4
    assert result.sample_count == 4


def test_permitted_sources_are_split_specific_and_combined_path_is_refused() -> None:
    splits = make_permitted_splits([_record(1, 2)], [_record(2, 3)])
    assert splits.training.dataset_sha256 != splits.publictest.dataset_sha256
    with pytest.raises(PermittedSplitError, match="before opening"):
        reject_combined_dataset_path("/path/to/fer2013.csv", occlusion_enabled=True)
    reject_combined_dataset_path(
        "/path/to/permitted_splits.json", occlusion_enabled=True
    )


def test_mean_and_manifest_digests_are_external_to_payload() -> None:
    splits = make_permitted_splits([_record(1, 2)], [_record(2, 3)])
    mean = calculate_training_mean_v2(splits.training)
    mean_digest = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [2],
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_digest,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_digest,
    )
    assert len(rows) == 9
    assert envelope.manifest_sha256 == manifest_v2_sha256(rows)
    assert "manifest_sha256" not in str(rows[0])


def test_v2_fill_rejects_wrong_mean_provenance_and_accepts_uint64_mean() -> None:
    splits = make_permitted_splits([_record(1, 2)], [_record(2, 3)])
    mean = calculate_training_mean_v2(splits.training)
    assert mean.accumulator_dtype == "uint64"
    assert len(normalized_fill_vector_v2(mean)) == 3
    with pytest.raises(ValueError, match="SHA"):
        normalized_fill_vector_v2(
            mean,
            training_dataset_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="source size"):
        normalized_fill_vector_v2(
            replace(mean, source_image_height=112),
        )


def test_v2_manifest_sidecar_is_create_or_verify_and_fail_closed(tmp_path) -> None:
    splits = make_permitted_splits([_record(1, 2)], [_record(2, 3)])
    mean = calculate_training_mean_v2(splits.training)
    mean_digest = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [2],
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_digest,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_digest,
    )
    csv_path = tmp_path / "manifest.csv"
    sidecar_path = tmp_path / "manifest.json"
    first = write_manifest_v2_create_or_verify(csv_path, sidecar_path, rows, envelope)
    second = write_manifest_v2_create_or_verify(csv_path, sidecar_path, rows, envelope)
    assert first.status == "created"
    assert second.status == "consistent"
    loaded_rows, loaded_envelope = load_manifest_v2(
        csv_path,
        sidecar_path,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_digest,
    )
    assert loaded_rows == rows
    assert loaded_envelope == envelope
    original = csv_path.read_bytes()
    csv_path.write_bytes(b"conflict\n")
    with pytest.raises(ManifestConflictError):
        write_manifest_v2_create_or_verify(csv_path, sidecar_path, rows, envelope)
    assert csv_path.read_bytes() != original


def test_evaluator_writes_ten_condition_artifacts_and_paired_drops(tmp_path) -> None:
    splits = make_permitted_splits([_record(1, 2)], [_record(i, i % 7) for i in range(2, 6)])
    mean = calculate_training_mean_v2(splits.training)
    mean_digest = training_mean_v2_sha256(mean)
    sample_ids = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    labels = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    images = torch.zeros(4, 3, 224, 224)
    model = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(3, 7))
    rows = build_manifest_v2_rows(
        [2, 3, 4, 5],
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_digest,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_digest,
    )
    results = evaluate_clean_and_nine_conditions(
        model,
        images,
        labels,
        sample_ids,
        torch.device("cpu"),
        fill_vector=(0.0, 0.0, 0.0),
        manifest_rows=rows,
        manifest_envelope=envelope,
        checkpoint_payload={
            "model_state_dict": {"synthetic": torch.zeros(1)},
            "resolved_config": {
                "dataset": {"image_size": 224},
                "training": {"mode": "clean"},
                "occlusion": None,
            },
        },
        require_official_manifest=False,
    )
    paths = write_occlusion_evaluation_artifacts(
        tmp_path,
        results,
        provenance={
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
        },
    )
    assert tuple(results) == ("clean",) + V2_MASKED_CONDITIONS
    assert len(paths) == 1 + 10 * 5 + 9
    assert (tmp_path / "evaluation_provenance.json").is_file()
    assert (tmp_path / "conditions" / "random_rectangle_0.40_paired_drop.csv").is_file()


def test_v2_routes_reject_private_and_old_checkpoint_identity() -> None:
    with pytest.raises(OcclusionEvaluationError):
        validate_publictest_route(split="PrivateTest", private_test=True)
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_for_route(
            {"model_state_dict": {}}, route="masked"
        )
    validate_checkpoint_for_route(
        {
            "model_state_dict": {},
            "resolved_config": {
                "dataset": {"image_size": 224},
                "training": {"mode": "clean"},
                "occlusion": None,
            },
        },
        route="masked",
    )
    with pytest.raises(CheckpointCompatibilityError):
        validate_checkpoint_for_route(
            {
                "model_state_dict": {},
                "resolved_config": {
                    "dataset": {"image_size": 112},
                    "training": {"mode": "clean"},
                    "occlusion": None,
                },
            },
            route="masked",
        )
    with pytest.raises(CheckpointCompatibilityError, match="occlusion-v2-224"):
        validate_checkpoint_for_route(
            {
                "model_state_dict": {},
                "resolved_config": {
                    "dataset": {"image_size": 224},
                    "training": {"mode": "clean"},
                    "occlusion": {
                        "protocol": {"algorithm_version": "occlusion-v1"},
                    },
                },
            },
            route="masked",
        )


def test_preflight_rejects_combined_path_before_opening() -> None:
    result = run_preflight(Namespace(
        config="configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml",
        data_path="/path/to/combined-fer2013.csv",
        output_dir=None,
        device="cpu",
        batch_size=None,
        max_train_samples=None,
        max_validation_samples=None,
        skip_model_forward=True,
    ))
    assert result.success is False
    assert result.failed_stage == "configuration"
    assert "before opening" in (result.error or "")


def test_mixed_preflight_consumes_permitted_splits_artifact(tmp_path) -> None:
    artifact = tmp_path / "permitted_splits.json"
    record = lambda sample_id, pixel_value: {
        "sample_id": sample_id,
        "label": sample_id % 7,
        "pixels": [pixel_value] * (48 * 48),
    }
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "Training": [record(1, 0), record(2, 1)],
                "PublicTest": [record(3, 2), record(4, 3)],
            }
        ),
        encoding="utf-8",
    )
    splits = make_permitted_splits(
        [_record(1, 0), _record(2, 1)],
        [_record(3, 2), _record(4, 3)],
    )
    mean = calculate_training_mean_v2(splits.training)
    mean_path = tmp_path / "training_mean_v2.json"
    write_training_mean_v2(mean_path, mean)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [3, 4],
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )
    manifest_path = tmp_path / "publictest_manifest_v2.csv"
    write_manifest_v2_create_or_verify(
        manifest_path,
        manifest_path.with_suffix(".json"),
        rows,
        envelope,
    )
    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(
        Path("configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml")
        .read_text(encoding="utf-8")
        .replace("  num_workers: 4\n", "  num_workers: 0\n", 1)
        .replace("/path/to/training_mean_v2.json", str(mean_path))
        .replace("/path/to/publictest_manifest_v2.csv", str(manifest_path)),
        encoding="utf-8",
    )
    result = run_preflight(Namespace(
        config=str(config_path),
        data_path=str(artifact),
        output_dir=str(tmp_path / "preflight"),
        device="cpu",
        batch_size=2,
        max_train_samples=1,
        max_validation_samples=1,
        skip_model_forward=True,
    ))
    assert result.success is True


def test_mixed_config_resolves_and_clean_e7_stays_clean() -> None:
    clean = load_config("configs/experiments/fer2013_resnet18_e7_high_resolution_longer.yaml")
    mixed = load_config("configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml")
    assert clean.occlusion is None
    assert clean.training.mode == "clean"
    assert mixed.training.mode == "mixed"
    assert mixed.occlusion is not None
    with pytest.raises(ConfigError):
        load_config("configs/fer2013_stage_a.yaml")


def test_formal_clean_and_mixed_configs_share_locked_e7_recipe() -> None:
    clean = load_config("configs/experiments/fer2013_resnet18_e7_clean.yaml")
    mixed = load_config("configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml")

    assert clean.project.run_role == "formal_clean"
    assert mixed.project.run_role == "formal_mixed"
    assert clean.training.mode == "clean"
    assert mixed.training.mode == "mixed"
    assert clean.occlusion is None
    assert mixed.occlusion is not None

    assert clean.dataset.image_size == mixed.dataset.image_size == 224
    assert clean.dataset.num_classes == mixed.dataset.num_classes == 7
    assert clean.dataset.augmentation == mixed.dataset.augmentation
    assert clean.model == mixed.model

    for attribute in (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "num_workers",
        "device",
        "loss",
        "scheduler",
        "early_stopping",
    ):
        assert getattr(clean.training, attribute) == getattr(mixed.training, attribute)


def test_v2_config_rejects_protocol_drift_before_data_access(tmp_path: Path) -> None:
    template = Path(
        "configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml"
    ).read_text(encoding="utf-8")

    wrong_sampling = tmp_path / "wrong_sampling.yaml"
    wrong_sampling.write_text(
        template.replace("clean_probability: 0.5", "clean_probability: 0.6"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="locked to 0.5"):
        load_config(wrong_sampling)

    wrong_size = tmp_path / "wrong_size.yaml"
    wrong_size.write_text(
        template.replace("  image_size: 224\n", "  image_size: 112\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="dataset.image_size=224"):
        load_config(wrong_size)

    missing_splits = tmp_path / "missing_splits.yaml"
    missing_splits.write_text(
        template.replace("  permitted_splits: [Training, PublicTest]\n", ""),
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError, match=r"permitted_splits=\[Training, PublicTest\]"
    ):
        load_config(missing_splits)
