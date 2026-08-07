from __future__ import annotations

import hashlib

import torch
import pytest
from torch import nn

from occlusion_fer.mask_manifest import (
    build_manifest_v2_rows,
    make_manifest_v2_envelope,
)
from occlusion_fer.occlusion import V2_MASKED_CONDITIONS
from occlusion_fer.occlusion_evaluate import (
    OcclusionEvaluationError,
    evaluate_clean_and_nine_conditions,
    load_masked_evaluation_checkpoint,
    validate_publictest_route,
    write_occlusion_evaluation_artifacts,
)
from occlusion_fer.permitted_splits import (
    SplitRecord,
    make_permitted_splits,
)
from occlusion_fer.training_mean import (
    calculate_training_mean_v2,
    training_mean_v2_sha256,
)


def _record(sample_id: int, value: int) -> SplitRecord:
    return SplitRecord(sample_id, sample_id % 7, (value,) * (48 * 48))


def _manifest(sample_ids: list[int]):
    splits = make_permitted_splits(
        [_record(1, 1)],
        [_record(sample_id, sample_id) for sample_id in sample_ids],
    )
    mean_sha = training_mean_v2_sha256(
        calculate_training_mean_v2(splits.training)
    )
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
    return rows, envelope


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
    rows, envelope = _manifest(sample_ids)
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
        fill_vector=(0.0, 0.0, 0.0),
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
    rows, envelope = _manifest([2])
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
            fill_vector=(0.0, 0.0, 0.0),
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=_clean_checkpoint_payload(),
            require_official_manifest=False,
        )


def test_mixed_checkpoint_mean_identity_must_match_manifest() -> None:
    rows, envelope = _manifest([2])
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
            "occlusion_runtime": {"training_mean_sha256": "f" * 64},
        },
    }
    with pytest.raises(OcclusionEvaluationError, match="mean identity"):
        evaluate_clean_and_nine_conditions(
            model,
            torch.zeros(1, 3, 224, 224),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.device("cpu"),
            fill_vector=(0.0, 0.0, 0.0),
            manifest_rows=rows,
            manifest_envelope=envelope,
            checkpoint_payload=mixed_payload,
            require_official_manifest=False,
        )
