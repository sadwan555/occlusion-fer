from dataclasses import replace
from pathlib import Path

import pytest
import torch

from occlusion_fer.formal_checkpoints import (
    FORMAL_CHECKPOINTS,
    CheckpointValidationError,
    checkpoint_for_model_id,
    validate_checkpoint_payload,
    validate_registry,
)


EXPECTED = {
    "CLEAN-42": ("clean", 42, 34, "c0f8266c4c3fdceaeb85b0e8195bdbf01a3d5b4b9abafa1bd1b71479121b9499"),
    "CLEAN-123": ("clean", 123, 47, "7ff8526c34d92ce15d8c2c2a092ef04b64064f3d884c3c3d70a8f859bd44835a"),
    "CLEAN-2026": ("clean", 2026, 48, "9e3fa67218522e8fd06f61e30c8a4761cccc731c679e46a73d82c9d3e61409c6"),
    "MIXED-42": ("mixed", 42, 37, "97f1772ae823a6c2c99eae3fcaeb3ceb857344927a3ea8bc60c3ce7b36f0e455"),
    "MIXED-123": ("mixed", 123, 50, "e46c4c4a5191343480566735c9b7879f805ad9686504abc5315c07a1168348ac"),
    "MIXED-2026": ("mixed", 2026, 46, "45a4d3719568ba790c6400b89db001fad66a184c4542305090daa703e8f5dbbf"),
}


def _payload(model_id: str) -> dict[str, object]:
    spec = checkpoint_for_model_id(model_id)
    return {
        "epoch": spec.best_epoch,
        "seed": spec.seed,
        "model_state_dict": {
            "conv1.weight": torch.zeros(64, 3, 7, 7),
            "fc.weight": torch.zeros(7, 512),
            "fc.bias": torch.zeros(7),
        },
        "resolved_config": {
            "dataset": {"image_size": 224, "num_classes": 7},
            "model": {"name": "resnet18", "pretrained": True},
            "training": {"mode": spec.strategy, "seed": spec.seed},
        },
    }


def test_registry_contains_exactly_the_six_frozen_models() -> None:
    assert len(FORMAL_CHECKPOINTS) == 6
    assert {item.model_id for item in FORMAL_CHECKPOINTS} == set(EXPECTED)
    for model_id, expected in EXPECTED.items():
        item = checkpoint_for_model_id(model_id)
        assert (item.strategy, item.seed, item.best_epoch, item.sha256) == expected
        assert item.image_size == 224
        assert item.architecture == "resnet18"
        assert not Path(item.archive_path).is_absolute()
        assert len(item.sha256) == 64
        int(item.sha256, 16)


def test_registry_rejects_duplicate_model_id() -> None:
    with pytest.raises(CheckpointValidationError, match="duplicate"):
        validate_registry(FORMAL_CHECKPOINTS + (FORMAL_CHECKPOINTS[0],))


def test_unapproved_checkpoint_id_is_rejected() -> None:
    with pytest.raises(CheckpointValidationError, match="approved"):
        checkpoint_for_model_id("CLEAN-999")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 999),
        ("best_epoch", 1),
        ("image_size", 112),
        ("architecture", "small_cnn"),
        ("sha256", "0" * 64),
    ],
)
def test_registry_rejects_modified_frozen_entry(field: str, value: object) -> None:
    changed = replace(FORMAL_CHECKPOINTS[0], **{field: value})
    with pytest.raises(CheckpointValidationError):
        validate_registry((changed,) + FORMAL_CHECKPOINTS[1:])


def test_checkpoint_payload_binds_metadata_and_resnet18_shapes() -> None:
    spec = checkpoint_for_model_id("MIXED-42")
    validate_checkpoint_payload(_payload(spec.model_id), spec, spec.sha256)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(seed=123), "seed"),
        (lambda payload: payload.update(epoch=1), "epoch"),
        (
            lambda payload: payload["resolved_config"]["dataset"].update(image_size=112),
            "224",
        ),
        (
            lambda payload: payload["model_state_dict"].update(
                {"fc.weight": torch.zeros(6, 512)}
            ),
            "7 classes",
        ),
    ],
)
def test_checkpoint_payload_rejects_wrong_binding(mutation, message: str) -> None:
    spec = checkpoint_for_model_id("MIXED-42")
    payload = _payload(spec.model_id)
    mutation(payload)
    with pytest.raises(CheckpointValidationError, match=message):
        validate_checkpoint_payload(payload, spec, spec.sha256)


def test_checkpoint_payload_rejects_wrong_checkpoint_sha() -> None:
    spec = checkpoint_for_model_id("CLEAN-42")
    with pytest.raises(CheckpointValidationError, match="SHA"):
        validate_checkpoint_payload(_payload(spec.model_id), spec, "0" * 64)
