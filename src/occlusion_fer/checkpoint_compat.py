"""Fail-closed checkpoint identity checks for clean and v2 routes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint lacks the metadata required by its route."""


def load_checkpoint_payload(
    path: str | Path,
) -> tuple[Mapping[str, object], str]:
    """Load a project checkpoint read-only and return its exact file SHA-256."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"could not load checkpoint: {checkpoint_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("checkpoint payload must be a mapping")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointCompatibilityError(
            "checkpoint requires a non-empty model_state_dict"
        )
    for name, value in state.items():
        if type(name) is not str or not isinstance(value, Tensor):
            raise CheckpointCompatibilityError(
                "model_state_dict must map parameter names to tensors"
            )
        if (
            (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        ):
            raise CheckpointCompatibilityError(
                f"checkpoint contains non-finite tensor for {name}"
            )
    return payload, digest


def strict_load_checkpoint_for_route(
    model: nn.Module,
    path: str | Path,
    *,
    route: str,
) -> tuple[Mapping[str, object], str]:
    """Validate route metadata and strict-load the exact checkpoint state."""
    if not isinstance(model, nn.Module):
        raise CheckpointCompatibilityError("model must be a torch.nn.Module")
    payload, digest = load_checkpoint_payload(path)
    validate_checkpoint_for_route(payload, route=route)
    state = payload["model_state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            "checkpoint model_state_dict is incompatible with the evaluation model"
        ) from exc
    return payload, digest


def validate_checkpoint_for_route(
    payload: Mapping[str, object], *, route: str
) -> None:
    if route not in {"clean", "mixed", "masked"}:
        raise CheckpointCompatibilityError("route must be clean, mixed, or masked")
    if "model_state_dict" not in payload:
        raise CheckpointCompatibilityError("checkpoint is missing model_state_dict")
    resolved = payload.get("resolved_config")
    if route == "clean":
        return
    if not isinstance(resolved, Mapping):
        raise CheckpointCompatibilityError("v2 route requires resolved_config metadata")
    training = resolved.get("training")
    occlusion = resolved.get("occlusion")
    dataset = resolved.get("dataset")
    if not isinstance(training, Mapping) or not isinstance(dataset, Mapping):
        raise CheckpointCompatibilityError("v2 route requires training and dataset metadata")
    if dataset.get("image_size") != 224:
        raise CheckpointCompatibilityError("v2 route rejects non-224 checkpoint identity")
    if route == "masked" and occlusion is None and training.get("mode") == "clean":
        return
    if not isinstance(occlusion, Mapping):
        raise CheckpointCompatibilityError("v2 route requires occlusion metadata")
    protocol = occlusion.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("algorithm_version") != "occlusion-v2-224":
        raise CheckpointCompatibilityError(
            "v2 route requires occlusion-v2-224 and rejects v1/112 checkpoint identity"
        )
    if protocol.get("image_size") != 224 or protocol.get("mean_algorithm_version") != "training-mean-v2":
        raise CheckpointCompatibilityError("v2 checkpoint protocol identity is incompatible")
    if route == "masked" and training.get("mode") not in {"clean", "mixed"}:
        raise CheckpointCompatibilityError("masked route requires clean or mixed training mode")
    if route == "mixed" and training.get("mode") != "mixed":
        raise CheckpointCompatibilityError("mixed route requires training.mode=mixed")


def checkpoint_provenance(
    *,
    route: str,
    seed: int,
    training_commit: str,
    dirty: bool,
    training_dataset_sha256: str | None = None,
    publictest_dataset_sha256: str | None = None,
    mean_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, object]:
    if type(seed) is not int or seed < 0:
        raise CheckpointCompatibilityError("seed must be a non-negative integer")
    return {
        "route": route,
        "seed": seed,
        "training_commit": training_commit,
        "git_dirty": bool(dirty),
        "training_dataset_sha256": training_dataset_sha256,
        "publictest_dataset_sha256": publictest_dataset_sha256,
        "training_mean_sha256": mean_sha256,
        "manifest_sha256": manifest_sha256,
        "protocol": "occlusion-v2-224" if route != "clean" else None,
    }
