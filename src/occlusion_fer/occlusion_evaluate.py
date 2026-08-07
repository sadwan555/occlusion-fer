"""PublicTest-only evaluator for clean and fixed v2 occlusion conditions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from occlusion_fer.evaluation import EvaluationResult, evaluate
from occlusion_fer.artifacts import (
    write_csv_atomic,
    write_evaluation_artifacts,
    write_failure_artifact,
    write_json_atomic,
)
from occlusion_fer.checkpoint_compat import (
    strict_load_checkpoint_for_route,
    validate_checkpoint_for_route,
)
from occlusion_fer.mask_manifest import (
    ManifestV2Envelope,
    ManifestV2Row,
    validate_manifest_v2,
)
from occlusion_fer.occlusion import (
    V2_MASKED_CONDITIONS,
    apply_evaluation_batch_v2,
)
from occlusion_fer.permitted_splits import reject_combined_dataset_path


class OcclusionEvaluationError(RuntimeError):
    """Raised when a masked evaluation request is unsafe or incomplete."""


CONDITION_ORDER = ("clean",) + V2_MASKED_CONDITIONS
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "checkpoint_sha256",
        "training_commit",
        "evaluation_commit",
        "git_dirty",
        "resolved_config",
        "training_dataset_sha256",
        "publictest_dataset_sha256",
        "training_mean_sha256",
        "manifest_sha256",
        "protocol",
        "image_height",
        "image_width",
        "seed",
        "run_role",
    }
)


def validate_publictest_route(
    *,
    split: str,
    dataset_path: str | Path | None = None,
    private_test: bool = False,
    manifest_algorithm_version: str = "occlusion-v2-224",
) -> None:
    if private_test or split in {"test", "PrivateTest", "private_test"}:
        raise OcclusionEvaluationError("occlusion evaluator has no PrivateTest route")
    if split not in {"validation", "PublicTest"}:
        raise OcclusionEvaluationError("occlusion evaluator requires PublicTest/validation")
    if dataset_path is not None:
        reject_combined_dataset_path(dataset_path, occlusion_enabled=True)
    if manifest_algorithm_version != "occlusion-v2-224":
        raise OcclusionEvaluationError("occlusion evaluator requires occlusion-v2-224 manifest")


def evaluate_condition(
    model: nn.Module,
    clean_images: Tensor,
    labels: Tensor,
    sample_ids: Tensor,
    device: torch.device,
    *,
    condition: str,
    fill_vector: Sequence[float],
) -> EvaluationResult:
    loader = DataLoader(
        TensorDataset(clean_images, labels, sample_ids),
        batch_size=max(1, min(128, clean_images.shape[0])),
        shuffle=False,
    )
    transform = None
    if condition != "clean":
        def transform(images, labels, sample_ids):
            masked_images, _ = apply_evaluation_batch_v2(
                images, sample_ids, condition, fill_vector
            )
            return masked_images, labels, sample_ids

    return evaluate(
        model,
        loader,
        device,
        split="validation",
        condition=condition,
        batch_transform=transform,
    )


def evaluate_clean_and_nine_conditions(
    model: nn.Module,
    clean_images: Tensor,
    labels: Tensor,
    sample_ids: Tensor,
    device: torch.device,
    *,
    fill_vector: Sequence[float],
    split: str = "validation",
    dataset_path: str | Path | None = None,
    private_test: bool = False,
    manifest_rows: Sequence[ManifestV2Row] | None = None,
    manifest_envelope: ManifestV2Envelope | None = None,
    checkpoint_payload: Mapping[str, object] | None = None,
    require_official_manifest: bool = True,
) -> dict[str, EvaluationResult]:
    validate_publictest_route(
        split=split,
        dataset_path=dataset_path,
        private_test=private_test,
    )
    if manifest_rows is None or manifest_envelope is None:
        raise OcclusionEvaluationError(
            "masked evaluation requires a validated v2 manifest and sidecar"
        )
    if checkpoint_payload is None:
        raise OcclusionEvaluationError(
            "masked evaluation requires checkpoint provenance metadata"
        )
    if (manifest_rows is None) != (manifest_envelope is None):
        raise OcclusionEvaluationError(
            "manifest rows and v2 sidecar envelope must be provided together"
        )
    if manifest_rows is not None and manifest_envelope is not None:
        validate_manifest_v2(
            manifest_rows,
            manifest_envelope,
            require_official=require_official_manifest,
        )
        manifest_ids = {
            row.sample_id for row in manifest_rows
        }
        runtime_ids = {
            int(sample_id) for sample_id in sample_ids.detach().cpu().tolist()
        }
        if manifest_ids != runtime_ids:
            raise OcclusionEvaluationError(
                "manifest sample IDs do not match the PublicTest evaluation batch"
            )
    validate_masked_checkpoint_payload(checkpoint_payload)
    _validate_checkpoint_manifest_binding(checkpoint_payload, manifest_envelope)
    results: dict[str, EvaluationResult] = {}
    for condition in CONDITION_ORDER:
        results[condition] = evaluate_condition(
            model, clean_images, labels, sample_ids, device,
            condition=condition, fill_vector=fill_vector,
        )
    return results


def write_occlusion_evaluation_artifacts(
    output_directory: str | Path,
    results: Mapping[str, EvaluationResult],
    *,
    provenance: Mapping[str, object],
) -> dict[str, Path]:
    """Write all ten condition artifacts only after a complete evaluation."""
    if tuple(results) != CONDITION_ORDER:
        raise OcclusionEvaluationError(
            "evaluation results must contain clean plus the nine conditions in canonical order"
        )
    if any(not isinstance(result, EvaluationResult) for result in results.values()):
        raise OcclusionEvaluationError("evaluation results contain an invalid result")
    missing = _REQUIRED_PROVENANCE_FIELDS.difference(provenance)
    if missing:
        raise OcclusionEvaluationError(
            "evaluation provenance is missing: " + ", ".join(sorted(missing))
        )
    _validate_evaluation_provenance(provenance)
    clean = results["clean"]
    if not clean.predictions:
        raise OcclusionEvaluationError("clean evaluation predictions must not be empty")
    output_path = Path(output_directory).expanduser()
    paths: dict[str, Path] = {}
    for condition in CONDITION_ORDER:
        condition_paths = write_evaluation_artifacts(
            output_path,
            Path("conditions") / condition,
            results[condition],
        )
        paths.update({f"{condition}_{name}": path for name, path in condition_paths.items()})
        paths[f"{condition}_provenance"] = write_json_atomic(
            output_path / "conditions" / f"{condition}_provenance.json",
            {**dict(provenance), "condition": condition},
        )
        if condition != "clean":
            drop_rows = paired_clean_to_occluded_drop(clean, results[condition])
            paths[f"{condition}_paired_drop"] = write_csv_atomic(
                output_path / "conditions" / f"{condition}_paired_drop.csv",
                (
                    "sample_id",
                    "condition",
                    "clean_correct",
                    "occluded_correct",
                    "correctness_drop",
                ),
                drop_rows,
            )
    paths["evaluation_provenance"] = write_json_atomic(
        output_path / "evaluation_provenance.json",
        {**dict(provenance), "conditions": list(CONDITION_ORDER)},
    )
    return paths


def write_occlusion_evaluation_failure(
    output_directory: str | Path,
    *,
    stage: str,
    exception: BaseException,
) -> Path:
    """Record a failed masked evaluation without claiming partial completion."""
    return write_failure_artifact(
        output_directory,
        stage=stage,
        exception=exception,
    )


def validate_masked_checkpoint_payload(payload: Mapping[str, object]) -> None:
    """Require a clean E7 or v2 mixed checkpoint before masked evaluation."""
    try:
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("checkpoint requires a non-empty model_state_dict")
        validate_checkpoint_for_route(payload, route="masked")
    except Exception as exc:
        raise OcclusionEvaluationError(
            f"checkpoint is not loadable for masked PublicTest evaluation: {exc}"
        ) from exc


def load_masked_evaluation_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, object], str]:
    """Strict-load a clean E7 or v2 mixed checkpoint for PublicTest masking."""
    try:
        return strict_load_checkpoint_for_route(
            model,
            checkpoint_path,
            route="masked",
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OcclusionEvaluationError(
            f"masked evaluation checkpoint is incompatible: {exc}"
        ) from exc


def _validate_evaluation_provenance(provenance: Mapping[str, object]) -> None:
    for field_name in (
        "checkpoint_sha256",
        "training_dataset_sha256",
        "publictest_dataset_sha256",
        "training_mean_sha256",
        "manifest_sha256",
    ):
        value = provenance[field_name]
        if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
            raise OcclusionEvaluationError(
                f"evaluation provenance {field_name} must be lowercase SHA-256"
            )
    if type(provenance["git_dirty"]) is not bool:
        raise OcclusionEvaluationError("evaluation provenance git_dirty must be a bool")
    if not isinstance(provenance["resolved_config"], Mapping):
        raise OcclusionEvaluationError("evaluation provenance resolved_config must be a mapping")
    if provenance["protocol"] != "occlusion-v2-224":
        raise OcclusionEvaluationError("evaluation provenance protocol must be occlusion-v2-224")
    if provenance["image_height"] != 224 or provenance["image_width"] != 224:
        raise OcclusionEvaluationError("evaluation provenance dimensions must be 224x224")
    if type(provenance["seed"]) is not int or provenance["seed"] < 0:
        raise OcclusionEvaluationError("evaluation provenance seed must be non-negative")
    if type(provenance["run_role"]) is not str or not provenance["run_role"]:
        raise OcclusionEvaluationError("evaluation provenance run_role must be non-empty")


def _validate_checkpoint_manifest_binding(
    payload: Mapping[str, object],
    envelope: ManifestV2Envelope,
) -> None:
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, Mapping):
        return
    training = resolved.get("training")
    if not isinstance(training, Mapping) or training.get("mode") != "mixed":
        return
    runtime = resolved.get("occlusion_runtime")
    if not isinstance(runtime, Mapping):
        raise OcclusionEvaluationError(
            "mixed checkpoint is missing occlusion_runtime provenance"
        )
    if runtime.get("training_mean_sha256") != envelope.training_mean_artifact_sha256:
        raise OcclusionEvaluationError(
            "mixed checkpoint mean identity does not match the evaluation manifest"
        )


def paired_clean_to_occluded_drop(
    clean: EvaluationResult,
    masked: EvaluationResult,
) -> tuple[dict[str, object], ...]:
    clean_by_id = {record.sample_id: record for record in clean.predictions}
    rows: list[dict[str, object]] = []
    for record in masked.predictions:
        baseline = clean_by_id.get(record.sample_id)
        if baseline is None:
            raise OcclusionEvaluationError(
                f"sample_id {record.sample_id} missing from clean evaluation"
            )
        rows.append({
            "sample_id": record.sample_id,
            "condition": record.condition,
            "clean_correct": baseline.correct,
            "occluded_correct": record.correct,
            "correctness_drop": int(baseline.correct) - int(record.correct),
        })
    return tuple(rows)
