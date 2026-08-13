"""Immutable identities for the six approved formal ResNet-18 checkpoints."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CLEAN_ARCHIVE = "fer2013-clean-e7-formal-4cb1e0f.tar"
_MIXED_ARCHIVE = "experiment-3-mixed-training-results.tar"


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint is not one of the six frozen formal models."""


@dataclass(frozen=True)
class FormalCheckpoint:
    model_id: str
    strategy: str
    seed: int
    best_epoch: int
    sha256: str
    archive_path: str
    archive_member: str
    training_commit: str
    image_size: int = 224
    architecture: str = "resnet18"
    num_classes: int = 7
    checkpoint_role: str = "best"


FORMAL_CHECKPOINTS = (
    FormalCheckpoint(
        "CLEAN-42",
        "clean",
        42,
        34,
        "c0f8266c4c3fdceaeb85b0e8195bdbf01a3d5b4b9abafa1bd1b71479121b9499",
        _CLEAN_ARCHIVE,
        "formal-e7-4cb1e0f/seed42/best.pt",
        "4cb1e0ffe4b55efc090a45cfed560b28f50b9509",
    ),
    FormalCheckpoint(
        "CLEAN-123",
        "clean",
        123,
        47,
        "7ff8526c34d92ce15d8c2c2a092ef04b64064f3d884c3c3d70a8f859bd44835a",
        _CLEAN_ARCHIVE,
        "formal-e7-4cb1e0f/seed123/best.pt",
        "4cb1e0ffe4b55efc090a45cfed560b28f50b9509",
    ),
    FormalCheckpoint(
        "CLEAN-2026",
        "clean",
        2026,
        48,
        "9e3fa67218522e8fd06f61e30c8a4761cccc731c679e46a73d82c9d3e61409c6",
        _CLEAN_ARCHIVE,
        "formal-e7-4cb1e0f/seed2026/best.pt",
        "4cb1e0ffe4b55efc090a45cfed560b28f50b9509",
    ),
    FormalCheckpoint(
        "MIXED-42",
        "mixed",
        42,
        37,
        "97f1772ae823a6c2c99eae3fcaeb3ceb857344927a3ea8bc60c3ce7b36f0e455",
        _MIXED_ARCHIVE,
        "mixed/seed42/best.pt",
        "c1c9187aa2ddf7dd84906c7f139ad9a750ef202d",
    ),
    FormalCheckpoint(
        "MIXED-123",
        "mixed",
        123,
        50,
        "e46c4c4a5191343480566735c9b7879f805ad9686504abc5315c07a1168348ac",
        _MIXED_ARCHIVE,
        "mixed/seed123/best.pt",
        "c1c9187aa2ddf7dd84906c7f139ad9a750ef202d",
    ),
    FormalCheckpoint(
        "MIXED-2026",
        "mixed",
        2026,
        46,
        "45a4d3719568ba790c6400b89db001fad66a184c4542305090daa703e8f5dbbf",
        _MIXED_ARCHIVE,
        "mixed/seed2026/best.pt",
        "c1c9187aa2ddf7dd84906c7f139ad9a750ef202d",
    ),
)
_FROZEN_BY_ID = {item.model_id: item for item in FORMAL_CHECKPOINTS}


def validate_registry(
    registry: Sequence[FormalCheckpoint],
) -> tuple[FormalCheckpoint, ...]:
    """Require an exact, ordered copy of the approved six-entry registry."""
    if isinstance(registry, (str, bytes)) or not isinstance(registry, Sequence):
        raise CheckpointValidationError(
            "checkpoint registry must be a sequence"
        )
    items = tuple(registry)
    if any(not isinstance(item, FormalCheckpoint) for item in items):
        raise CheckpointValidationError(
            "checkpoint registry must contain FormalCheckpoint entries"
        )
    model_ids = [item.model_id for item in items]
    if len(set(model_ids)) != len(model_ids):
        raise CheckpointValidationError(
            "checkpoint registry contains a duplicate model ID"
        )
    if len(items) != 6:
        raise CheckpointValidationError(
            "checkpoint registry must contain exactly six approved entries"
        )
    if set(model_ids) != set(_FROZEN_BY_ID):
        raise CheckpointValidationError(
            "checkpoint registry contains an unapproved model ID"
        )
    for item in items:
        if item != _FROZEN_BY_ID[item.model_id]:
            raise CheckpointValidationError(
                f"checkpoint registry entry {item.model_id} differs from "
                "the frozen formal identity"
            )
        if _SHA256_PATTERN.fullmatch(item.sha256) is None:
            raise CheckpointValidationError(
                f"checkpoint {item.model_id} SHA-256 is invalid"
            )
    return items


def checkpoint_for_model_id(model_id: str) -> FormalCheckpoint:
    """Return a frozen entry and reject every unapproved model ID."""
    if type(model_id) is not str:
        raise CheckpointValidationError("model ID must be a string")
    try:
        return _FROZEN_BY_ID[model_id]
    except KeyError as exc:
        raise CheckpointValidationError(
            f"model ID is not approved for final evaluation: {model_id!r}"
        ) from exc


def sha256_file(path: str | Path) -> str:
    """Hash a regular file without interpreting it."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint file not found: {checkpoint_path}"
        )
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_payload(
    payload: Mapping[str, object],
    spec: FormalCheckpoint,
    actual_sha256: str,
) -> Mapping[str, Tensor]:
    """Bind metadata and key ResNet-18 tensor shapes to one frozen entry."""
    checkpoint_for_model_id(spec.model_id)
    if spec != _FROZEN_BY_ID[spec.model_id]:
        raise CheckpointValidationError(
            f"checkpoint spec {spec.model_id} is not the frozen identity"
        )
    if actual_sha256 != spec.sha256:
        raise CheckpointValidationError(
            f"checkpoint SHA mismatch for {spec.model_id}"
        )
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    if payload.get("seed") != spec.seed:
        raise CheckpointValidationError(
            f"checkpoint seed mismatch for {spec.model_id}"
        )
    epoch = payload.get("epoch")
    if epoch != spec.best_epoch:
        raise CheckpointValidationError(
            f"checkpoint best epoch mismatch for {spec.model_id}"
        )

    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        raise CheckpointValidationError(
            "checkpoint must contain resolved_config provenance"
        )
    dataset = resolved.get("dataset")
    model = resolved.get("model")
    training = resolved.get("training")
    if not isinstance(dataset, Mapping) or not isinstance(model, Mapping):
        raise CheckpointValidationError(
            "checkpoint resolved_config is missing dataset/model metadata"
        )
    if not isinstance(training, Mapping):
        raise CheckpointValidationError(
            "checkpoint resolved_config is missing training metadata"
        )
    if dataset.get("image_size") != 224:
        raise CheckpointValidationError(
            "formal checkpoint must bind image_size=224"
        )
    if dataset.get("num_classes") != 7:
        raise CheckpointValidationError(
            "formal checkpoint must bind exactly 7 classes"
        )
    if model.get("name") != "resnet18":
        raise CheckpointValidationError(
            "formal checkpoint architecture must be resnet18"
        )
    if model.get("pretrained") is not True:
        raise CheckpointValidationError(
            "formal checkpoint must bind the ImageNet-pretrained lineage"
        )
    if training.get("seed") != spec.seed:
        raise CheckpointValidationError(
            "checkpoint resolved training seed is incompatible"
        )
    mode = training.get("mode")
    allowed_modes = (
        {"clean", "clean-only"} if spec.strategy == "clean"
        else {"mixed", "mixed_occlusion"}
    )
    if mode not in allowed_modes:
        raise CheckpointValidationError(
            "checkpoint resolved training strategy is incompatible"
        )

    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointValidationError(
            "checkpoint must contain a non-empty model_state_dict"
        )
    for name, value in state.items():
        if type(name) is not str or not isinstance(value, Tensor):
            raise CheckpointValidationError(
                "checkpoint state must map parameter names to tensors"
            )
        if (
            (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        ):
            raise CheckpointValidationError(
                f"checkpoint contains non-finite tensor {name}"
            )
    if tuple(getattr(state.get("conv1.weight"), "shape", ())) != (64, 3, 7, 7):
        raise CheckpointValidationError(
            "checkpoint does not contain the standard ResNet-18 stem"
        )
    if tuple(getattr(state.get("fc.weight"), "shape", ())) != (7, 512):
        raise CheckpointValidationError(
            "checkpoint fc.weight must have 7 classes"
        )
    if tuple(getattr(state.get("fc.bias"), "shape", ())) != (7,):
        raise CheckpointValidationError(
            "checkpoint fc.bias must have 7 classes"
        )
    return state


def load_and_validate_checkpoint(
    path: str | Path,
    spec: FormalCheckpoint,
) -> tuple[Mapping[str, object], str]:
    """Read one staged checkpoint and enforce its complete frozen binding."""
    checkpoint_path = Path(path).expanduser()
    actual_sha256 = sha256_file(checkpoint_path)
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointValidationError(
            f"could not load checkpoint {checkpoint_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    validate_checkpoint_payload(payload, spec, actual_sha256)
    return payload, actual_sha256


validate_registry(FORMAL_CHECKPOINTS)
