"""Canonical immutable plan for one FER2013 PrivateTest final evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from occlusion_fer.formal_checkpoints import (
    FORMAL_CHECKPOINTS,
    FormalCheckpoint,
    validate_registry,
)
from occlusion_fer.private_manifest import (
    ALGORITHM_VERSION,
    CONDITION_ORDER,
    COORDINATE_CONVENTION,
    EVALUATION_MASK_SEED,
    IMAGE_SIZE,
    INTERNAL_SPLIT,
    MASKED_CONDITIONS,
    NORMALIZED_FILL,
    OFFICIAL_PRIVATE_SAMPLE_COUNT,
    OFFICIAL_USAGE,
    PRIVATE_DATASET_SHA256,
    ROUNDING_RULE,
    TRAINING_MEAN_ARTIFACT_SHA256,
    TRAINING_MEAN_RAW,
)


PLAN_SCHEMA_VERSION = 1
_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinalPlanError(ValueError):
    """Raised when a final evaluation plan is mutable or incomplete."""


@dataclass(frozen=True)
class PrivateIdentity:
    official_usage: str
    internal_split: str
    count: int
    canonical_sha256: str


@dataclass(frozen=True)
class ProtocolIdentity:
    algorithm_version: str
    image_size: int
    mask_seed: int
    occlusion_types: tuple[str, ...]
    ratios: tuple[str, ...]
    condition_order: tuple[str, ...]
    coordinate_convention: str
    rounding_rule: str


@dataclass(frozen=True)
class TrainingMeanIdentity:
    raw_mean: float
    artifact_sha256: str
    normalized_fill: tuple[float, float, float]


@dataclass(frozen=True)
class PlannedCheckpoint:
    model_id: str
    strategy: str
    seed: int
    best_epoch: int
    sha256: str
    path: str
    image_size: int
    architecture: str
    training_commit: str


@dataclass(frozen=True)
class FinalEvaluationPlan:
    schema_version: int
    created_at: str
    dataset_name: str
    private: PrivateIdentity
    protocol: ProtocolIdentity
    training_mean: TrainingMeanIdentity
    checkpoints: tuple[PlannedCheckpoint, ...]
    output_root: str
    git_commit: str
    git_dirty: bool

    def validate(self) -> None:
        validate_final_plan(self)


def build_final_plan(
    *,
    checkpoint_paths: Mapping[str, str],
    output_root: str,
    git_commit: str,
    git_dirty: bool,
    created_at: str,
) -> FinalEvaluationPlan:
    """Build a plan only when all six approved checkpoint paths are explicit."""
    if not isinstance(checkpoint_paths, Mapping):
        raise FinalPlanError("checkpoint paths must be a mapping")
    if set(checkpoint_paths) != {
        item.model_id for item in FORMAL_CHECKPOINTS
    }:
        raise FinalPlanError(
            "final plan requires paths for exactly the six approved checkpoints"
        )
    checkpoints = tuple(
        _planned_checkpoint(item, checkpoint_paths[item.model_id])
        for item in FORMAL_CHECKPOINTS
    )
    plan = FinalEvaluationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        created_at=created_at,
        dataset_name="fer2013",
        private=PrivateIdentity(
            OFFICIAL_USAGE,
            INTERNAL_SPLIT,
            OFFICIAL_PRIVATE_SAMPLE_COUNT,
            PRIVATE_DATASET_SHA256,
        ),
        protocol=ProtocolIdentity(
            ALGORITHM_VERSION,
            IMAGE_SIZE,
            EVALUATION_MASK_SEED,
            ("upper_face", "lower_face", "random_rectangle"),
            ("0.20", "0.30", "0.40"),
            CONDITION_ORDER,
            COORDINATE_CONVENTION,
            ROUNDING_RULE,
        ),
        training_mean=TrainingMeanIdentity(
            TRAINING_MEAN_RAW,
            TRAINING_MEAN_ARTIFACT_SHA256,
            NORMALIZED_FILL,
        ),
        checkpoints=checkpoints,
        output_root=output_root,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    validate_final_plan(plan)
    return plan


def validate_final_plan(plan: FinalEvaluationPlan) -> None:
    if not isinstance(plan, FinalEvaluationPlan):
        raise FinalPlanError("plan must be a FinalEvaluationPlan")
    if plan.schema_version != PLAN_SCHEMA_VERSION:
        raise FinalPlanError("final plan schema version is incompatible")
    if type(plan.created_at) is not str or not plan.created_at.strip():
        raise FinalPlanError("final plan created_at must be non-empty")
    if plan.dataset_name != "fer2013":
        raise FinalPlanError("final plan dataset must be fer2013")
    if plan.private != PrivateIdentity(
        OFFICIAL_USAGE,
        INTERNAL_SPLIT,
        OFFICIAL_PRIVATE_SAMPLE_COUNT,
        PRIVATE_DATASET_SHA256,
    ):
        raise FinalPlanError("final plan PrivateTest identity is incompatible")
    if plan.protocol != ProtocolIdentity(
        ALGORITHM_VERSION,
        IMAGE_SIZE,
        EVALUATION_MASK_SEED,
        ("upper_face", "lower_face", "random_rectangle"),
        ("0.20", "0.30", "0.40"),
        ("clean", *MASKED_CONDITIONS),
        COORDINATE_CONVENTION,
        ROUNDING_RULE,
    ):
        raise FinalPlanError("final plan scientific protocol is incompatible")
    if plan.training_mean != TrainingMeanIdentity(
        TRAINING_MEAN_RAW,
        TRAINING_MEAN_ARTIFACT_SHA256,
        NORMALIZED_FILL,
    ):
        raise FinalPlanError("final plan Training mean is incompatible")
    if len(plan.checkpoints) != 6:
        raise FinalPlanError(
            "final plan must contain exactly six checkpoints"
        )
    validate_registry(FORMAL_CHECKPOINTS)
    for planned, frozen in zip(
        plan.checkpoints, FORMAL_CHECKPOINTS, strict=True
    ):
        expected = _planned_checkpoint(frozen, planned.path)
        if planned != expected:
            raise FinalPlanError(
                f"final plan checkpoint {planned.model_id} is not frozen"
            )
    if type(plan.output_root) is not str or not plan.output_root.strip():
        raise FinalPlanError("final plan output root must be non-empty")
    if (
        type(plan.git_commit) is not str
        or _SHA1_PATTERN.fullmatch(plan.git_commit) is None
    ):
        raise FinalPlanError("final plan git commit must be a 40-character SHA")
    if type(plan.git_dirty) is not bool:
        raise FinalPlanError("final plan git dirty state must be boolean")


def canonical_plan_bytes(plan: FinalEvaluationPlan) -> bytes:
    validate_final_plan(plan)
    return (
        json.dumps(
            asdict(plan),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def plan_sha256(plan: FinalEvaluationPlan) -> str:
    return hashlib.sha256(canonical_plan_bytes(plan)).hexdigest()


def write_final_plan(path: str | Path, plan: FinalEvaluationPlan) -> str:
    """Create a self-hashed plan and never overwrite an existing target."""
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError(f"final plan already exists: {target}")
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"final plan parent directory does not exist: {target.parent}"
        )
    digest = plan_sha256(plan)
    envelope = {
        "plan": asdict(plan),
        "plan_sha256": digest,
    }
    data = (
        json.dumps(
            envelope,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with target.open("xb") as plan_file:
        plan_file.write(data)
    return digest


def load_final_plan(
    path: str | Path,
) -> tuple[FinalEvaluationPlan, str]:
    """Load a canonical self-hashed plan and reject byte or field changes."""
    plan_path = Path(path).expanduser()
    if not plan_path.is_file():
        raise FileNotFoundError(f"final plan not found: {plan_path}")
    try:
        data = plan_path.read_bytes()
        envelope = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalPlanError("final plan must be UTF-8 JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "plan",
        "plan_sha256",
    }:
        raise FinalPlanError("final plan envelope fields are incompatible")
    try:
        plan = _plan_from_mapping(envelope["plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalPlanError("final plan fields are malformed") from exc
    digest = plan_sha256(plan)
    if envelope["plan_sha256"] != digest:
        raise FinalPlanError("final plan SHA mismatch")
    canonical_envelope = (
        json.dumps(
            {"plan": asdict(plan), "plan_sha256": digest},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if data != canonical_envelope:
        raise FinalPlanError("final plan bytes are not canonical")
    return plan, digest


def _planned_checkpoint(
    item: FormalCheckpoint,
    path: object,
) -> PlannedCheckpoint:
    if type(path) is not str or not path.strip():
        raise FinalPlanError(
            f"checkpoint path for {item.model_id} must be non-empty"
        )
    return PlannedCheckpoint(
        item.model_id,
        item.strategy,
        item.seed,
        item.best_epoch,
        item.sha256,
        path,
        item.image_size,
        item.architecture,
        item.training_commit,
    )


def _plan_from_mapping(value: object) -> FinalEvaluationPlan:
    if not isinstance(value, dict):
        raise FinalPlanError("final plan body must be a mapping")
    private = PrivateIdentity(**value["private"])
    protocol_value = dict(value["protocol"])
    protocol_value["occlusion_types"] = tuple(
        protocol_value["occlusion_types"]
    )
    protocol_value["ratios"] = tuple(protocol_value["ratios"])
    protocol_value["condition_order"] = tuple(
        protocol_value["condition_order"]
    )
    protocol = ProtocolIdentity(**protocol_value)
    mean_value = dict(value["training_mean"])
    mean_value["normalized_fill"] = tuple(mean_value["normalized_fill"])
    training_mean = TrainingMeanIdentity(**mean_value)
    checkpoints = tuple(
        PlannedCheckpoint(**item) for item in value["checkpoints"]
    )
    plan = FinalEvaluationPlan(
        value["schema_version"],
        value["created_at"],
        value["dataset_name"],
        private,
        protocol,
        training_mean,
        checkpoints,
        value["output_root"],
        value["git_commit"],
        value["git_dirty"],
    )
    validate_final_plan(plan)
    return plan
