"""Fail-closed preflight checks for the future PrivateTest final run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from occlusion_fer.data import Fer2013Data, load_fer2013_csv
from occlusion_fer.formal_checkpoints import (
    FORMAL_CHECKPOINTS,
    load_and_validate_checkpoint,
)
from occlusion_fer.private_manifest import (
    PRIVATE_DATASET_SHA256,
    TRAINING_MEAN_ARTIFACT_SHA256,
    PrivateManifestRow,
    PrivateManifestSidecar,
    validate_manifest_binding,
)
from occlusion_fer.private_plan import FinalEvaluationPlan
from occlusion_fer.training_mean import (
    TrainingMeanError,
    load_training_mean_v2,
    training_mean_v2_sha256,
    validate_training_mean_v2,
)


TRAINING_DATASET_SHA256 = (
    "42eb71ae81c749a40426583445c1fda1db904fc73ec6798b07596e01ebb89a4c"
)
RAW_TRAINING_MEAN = 0.5077425080522144


class PrivatePreflightError(ValueError):
    """Raised before any final inference when an identity check fails."""


@dataclass(frozen=True)
class PrivateSourceIdentity:
    data: Fer2013Data
    sample_ids: tuple[int, ...]
    canonical_sha256: str


def ensure_empty_output_root(path: str | Path) -> Path:
    """Allow an absent or empty directory, while never creating or clearing it."""
    output_path = Path(path).expanduser()
    if output_path.exists():
        if not output_path.is_dir():
            raise PrivatePreflightError(
                f"final output root is not a directory: {output_path}"
            )
        if any(output_path.iterdir()):
            raise FileExistsError(
                f"final output root is not empty: {output_path}"
            )
    elif not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"final output parent does not exist: {output_path.parent}"
        )
    return output_path


def inspect_private_source(path: str | Path) -> PrivateSourceIdentity:
    """Load only PrivateTest and compute the Stage 8 canonical split identity."""
    data = load_fer2013_csv(path, include_splits=("test",))
    records = sorted(data.records, key=lambda item: item.sample_id)
    canonical_lines = (
        json.dumps(
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "pixels": [int(pixel) for pixel in record.image.reshape(-1)],
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        for record in records
    )
    canonical_bytes = ("\n".join(canonical_lines) + "\n").encode("utf-8")
    return PrivateSourceIdentity(
        data=data,
        sample_ids=tuple(record.sample_id for record in records),
        canonical_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )


def validate_training_mean_artifact(path: str | Path) -> None:
    """Validate the canonical Stage 8 envelope and its frozen identity."""
    try:
        envelope = load_training_mean_v2(Path(path).expanduser())
        validate_training_mean_v2(
            envelope,
            training_dataset_sha256=TRAINING_DATASET_SHA256,
            consumer_image_size=224,
        )
        identity = training_mean_v2_sha256(envelope)
        if identity != TRAINING_MEAN_ARTIFACT_SHA256:
            raise PrivatePreflightError(
                "Training mean artifact identity SHA mismatch"
            )
    except (OSError, TrainingMeanError) as exc:
        raise PrivatePreflightError(str(exc)) from exc
    if envelope.raw_training_mean != RAW_TRAINING_MEAN:
        raise PrivatePreflightError("Training mean raw value mismatch")


def preflight_checkpoints(plan: FinalEvaluationPlan) -> None:
    """Load and bind all six files before any dataset loader is evaluated."""
    if len(plan.checkpoints) != 6:
        raise PrivatePreflightError("final plan must contain six checkpoints")
    for planned, frozen in zip(
        plan.checkpoints, FORMAL_CHECKPOINTS, strict=True
    ):
        if planned.model_id != frozen.model_id:
            raise PrivatePreflightError(
                "final plan checkpoint order is incompatible"
            )
        load_and_validate_checkpoint(planned.path, frozen)


def preflight_private_final(
    plan: FinalEvaluationPlan,
    source: PrivateSourceIdentity,
    *,
    manifest_rows: tuple[PrivateManifestRow, ...],
    manifest_sidecar: PrivateManifestSidecar,
    training_mean_path: str | Path,
    output_root: str | Path,
    require_official: bool,
) -> None:
    """Run every fail-closed identity check without model inference."""
    try:
        plan.validate()
        ensure_empty_output_root(output_root)
        preflight_checkpoints(plan)
        validate_training_mean_artifact(training_mean_path)
        if require_official:
            validate_official_private_source(source)
        validate_manifest_binding(
            manifest_rows,
            manifest_sidecar,
            expected_dataset_sha256=(
                plan.private.canonical_sha256
                if require_official
                else source.canonical_sha256
            ),
            expected_sample_ids=source.sample_ids,
            require_official=require_official,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, PrivatePreflightError):
            raise
        raise PrivatePreflightError(str(exc)) from exc


def validate_official_private_source(
    source: PrivateSourceIdentity,
) -> None:
    if source.data.train_count != 0 or source.data.validation_count != 0:
        raise PrivatePreflightError(
            "private final source must materialize only test/PrivateTest"
        )
    if source.data.test_count != 3589 or len(source.sample_ids) != 3589:
        raise PrivatePreflightError(
            "official PrivateTest sample count must be 3589"
        )
    if source.canonical_sha256 != PRIVATE_DATASET_SHA256:
        raise PrivatePreflightError(
            "official PrivateTest canonical SHA mismatch"
        )
