"""PublicTest-only evaluator for clean and fixed v2 occlusion conditions."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import re
from pathlib import Path
import subprocess
import sys

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
from occlusion_fer.config import load_config
from occlusion_fer.models import create_resnet18
from occlusion_fer.mask_manifest import (
    ManifestV2Envelope,
    ManifestV2Row,
    load_manifest_v2,
    validate_manifest_v2,
)
from occlusion_fer.occlusion import (
    V2_MASKED_CONDITIONS,
    apply_evaluation_batch_v2,
    normalized_fill_vector_v2,
)
from occlusion_fer.permitted_splits import (
    PermittedSplits,
    STAGE_B_SOURCE_ROUTING_VERSION,
    load_stage_b_source,
    permitted_splits_to_data,
    stage_b_source_kind,
    validate_official_stage_b_sources,
)
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import select_device
from occlusion_fer.training_mean import (
    TrainingMeanV2Envelope,
    load_training_mean_v2,
    training_mean_v2_sha256,
    validate_training_mean_v2,
)


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
        stage_b_source_kind(dataset_path)
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
    training_mean_artifact: TrainingMeanV2Envelope | str | Path,
    training_dataset_sha256: str,
    publictest_dataset_sha256: str,
    split: str = "validation",
    dataset_path: str | Path | None = None,
    private_test: bool = False,
    manifest_rows: Sequence[ManifestV2Row] | None = None,
    manifest_envelope: ManifestV2Envelope | None = None,
    checkpoint_payload: Mapping[str, object] | None = None,
    evaluation_provenance: Mapping[str, object] | None = None,
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
    assert manifest_rows is not None and manifest_envelope is not None
    validate_manifest_v2(
        manifest_rows,
        manifest_envelope,
        require_official=require_official_manifest,
    )
    fill_vector = _prepare_evaluation_bindings(
        manifest_envelope=manifest_envelope,
        checkpoint_payload=checkpoint_payload,
        training_mean_artifact=training_mean_artifact,
        training_dataset_sha256=training_dataset_sha256,
        publictest_dataset_sha256=publictest_dataset_sha256,
        evaluation_provenance=evaluation_provenance,
    )
    manifest_ids = {row.sample_id for row in manifest_rows}
    runtime_ids = {
        int(sample_id) for sample_id in sample_ids.detach().cpu().tolist()
    }
    if manifest_ids != runtime_ids:
        raise OcclusionEvaluationError(
            "manifest sample IDs do not match the PublicTest evaluation batch"
        )
    validate_masked_checkpoint_payload(checkpoint_payload)
    results: dict[str, EvaluationResult] = {}
    for condition in CONDITION_ORDER:
        results[condition] = evaluate_condition(
            model, clean_images, labels, sample_ids, device,
            condition=condition, fill_vector=fill_vector,
        )
    return results


def evaluate_publictest_source(
    model: nn.Module,
    source: PermittedSplits,
    device: torch.device,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    training_mean_artifact: TrainingMeanV2Envelope | str | Path,
    manifest_rows: Sequence[ManifestV2Row],
    manifest_envelope: ManifestV2Envelope,
    checkpoint_payload: Mapping[str, object],
    evaluation_provenance: Mapping[str, object] | None = None,
    amp_enabled: bool = False,
    require_official_manifest: bool = True,
) -> dict[str, EvaluationResult]:
    """Evaluate a validated Usage-routed Training/PublicTest source."""
    if not isinstance(source, PermittedSplits):
        raise OcclusionEvaluationError("evaluation source must be PermittedSplits")
    if type(image_size) is not int or image_size != 224:
        raise OcclusionEvaluationError("PublicTest v2 evaluation requires image_size=224")
    if type(batch_size) is not int or batch_size <= 0:
        raise OcclusionEvaluationError("evaluation batch_size must be positive")
    if type(num_workers) is not int or num_workers < 0:
        raise OcclusionEvaluationError("evaluation num_workers must be non-negative")
    validate_publictest_route(split="PublicTest")
    data = permitted_splits_to_data(source)
    dataset = Fer2013TorchDataset(
        data,
        split="validation",
        image_size=image_size,
        normalize_imagenet=True,
    )
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
    )
    runtime_ids = {record.sample_id for record in source.publictest.records}
    manifest_ids = {row.sample_id for row in manifest_rows}
    if runtime_ids != manifest_ids:
        raise OcclusionEvaluationError(
            "manifest sample IDs do not match the Usage-routed PublicTest source"
        )
    validate_manifest_v2(
        manifest_rows,
        manifest_envelope,
        require_official=require_official_manifest,
    )
    fill_vector = _prepare_evaluation_bindings(
        manifest_envelope=manifest_envelope,
        checkpoint_payload=checkpoint_payload,
        training_mean_artifact=training_mean_artifact,
        training_dataset_sha256=source.training.dataset_sha256,
        publictest_dataset_sha256=source.publictest.dataset_sha256,
        evaluation_provenance=evaluation_provenance,
    )
    validate_masked_checkpoint_payload(checkpoint_payload)
    return _evaluate_loader_conditions(
        model,
        loader,
        device,
        fill_vector=fill_vector,
        amp_enabled=amp_enabled,
    )


def _evaluate_loader_conditions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    fill_vector: Sequence[float],
    amp_enabled: bool,
) -> dict[str, EvaluationResult]:
    results: dict[str, EvaluationResult] = {}
    for condition in CONDITION_ORDER:
        transform = None
        if condition != "clean":
            def transform(images, labels, sample_ids, *, _condition=condition):
                masked_images, _ = apply_evaluation_batch_v2(
                    images, sample_ids, _condition, fill_vector
                )
                return masked_images, labels, sample_ids

        results[condition] = evaluate(
            model,
            loader,
            device,
            split="validation",
            condition=condition,
            amp_enabled=amp_enabled,
            batch_transform=transform,
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
    if output_path.exists():
        if not output_path.is_dir():
            raise OcclusionEvaluationError(
                f"evaluation output path is not a directory: {output_path}"
            )
        if any(output_path.iterdir()):
            raise OcclusionEvaluationError(
                f"evaluation output directory already contains artifacts: {output_path}"
            )
    else:
        output_path.mkdir(parents=True, exist_ok=False)
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
    routing_version = provenance.get("source_routing_version")
    if routing_version is not None and routing_version != STAGE_B_SOURCE_ROUTING_VERSION:
        raise OcclusionEvaluationError(
            "evaluation provenance source routing identity is incompatible"
        )
    training_dirty = provenance.get("training_git_dirty")
    if training_dirty is not None and type(training_dirty) is not bool:
        raise OcclusionEvaluationError(
            "evaluation provenance training_git_dirty must be a bool"
        )


def _prepare_evaluation_bindings(
    *,
    manifest_envelope: ManifestV2Envelope,
    checkpoint_payload: Mapping[str, object],
    training_mean_artifact: TrainingMeanV2Envelope | str | Path,
    training_dataset_sha256: str,
    publictest_dataset_sha256: str,
    evaluation_provenance: Mapping[str, object] | None,
) -> tuple[float, float, float]:
    _validate_sha256(training_dataset_sha256, "training_dataset_sha256")
    _validate_sha256(publictest_dataset_sha256, "publictest_dataset_sha256")
    if manifest_envelope.publictest_dataset_sha256 != publictest_dataset_sha256:
        raise OcclusionEvaluationError(
            "runtime PublicTest identity does not match the evaluation manifest"
        )
    if isinstance(training_mean_artifact, (str, Path)):
        try:
            mean_artifact = load_training_mean_v2(training_mean_artifact)
        except Exception as exc:
            raise OcclusionEvaluationError(
                f"could not load Training mean v2 artifact: {exc}"
            ) from exc
    elif isinstance(training_mean_artifact, TrainingMeanV2Envelope):
        mean_artifact = training_mean_artifact
    else:
        raise OcclusionEvaluationError(
            "training_mean_artifact must be a TrainingMeanV2Envelope or path"
        )
    try:
        validate_training_mean_v2(
            mean_artifact,
            training_dataset_sha256=training_dataset_sha256,
            consumer_image_size=224,
        )
        mean_sha256 = training_mean_v2_sha256(mean_artifact)
        fill_vector = normalized_fill_vector_v2(
            mean_artifact,
            training_dataset_sha256=training_dataset_sha256,
        )
    except Exception as exc:
        raise OcclusionEvaluationError(
            f"Training mean v2 provenance is incompatible: {exc}"
        ) from exc
    if mean_sha256 != manifest_envelope.training_mean_artifact_sha256:
        raise OcclusionEvaluationError(
            "Training mean identity does not match the evaluation manifest"
        )
    _validate_checkpoint_manifest_binding(
        checkpoint_payload,
        manifest_envelope,
        training_dataset_sha256=training_dataset_sha256,
        publictest_dataset_sha256=publictest_dataset_sha256,
    )
    if evaluation_provenance is not None:
        missing = _REQUIRED_PROVENANCE_FIELDS.difference(evaluation_provenance)
        if missing:
            raise OcclusionEvaluationError(
                "evaluation provenance is missing: " + ", ".join(sorted(missing))
            )
        _validate_evaluation_provenance(evaluation_provenance)
        if evaluation_provenance.get("publictest_dataset_sha256") != publictest_dataset_sha256:
            raise OcclusionEvaluationError(
                "evaluation provenance PublicTest identity does not match runtime source"
            )
        if evaluation_provenance.get("training_dataset_sha256") != training_dataset_sha256:
            raise OcclusionEvaluationError(
                "evaluation provenance Training identity does not match runtime source"
            )
        if evaluation_provenance.get("training_mean_sha256") != mean_sha256:
            raise OcclusionEvaluationError(
                "evaluation provenance mean identity does not match the mean artifact"
            )
        if evaluation_provenance.get("manifest_sha256") != manifest_envelope.manifest_sha256:
            raise OcclusionEvaluationError(
                "evaluation provenance manifest identity does not match the manifest"
            )
    return fill_vector


def _validate_sha256(value: object, field_name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise OcclusionEvaluationError(
            f"{field_name} must be a lowercase SHA-256"
        )


def _validate_checkpoint_manifest_binding(
    payload: Mapping[str, object],
    envelope: ManifestV2Envelope,
    *,
    training_dataset_sha256: str,
    publictest_dataset_sha256: str,
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
    if runtime.get("training_dataset_sha256") != training_dataset_sha256:
        raise OcclusionEvaluationError(
            "mixed checkpoint Training identity does not match the runtime source"
        )
    if runtime.get("publictest_dataset_sha256") != envelope.publictest_dataset_sha256:
        raise OcclusionEvaluationError(
            "mixed checkpoint PublicTest identity does not match the evaluation manifest"
        )
    if runtime.get("publictest_dataset_sha256") != publictest_dataset_sha256:
        raise OcclusionEvaluationError(
            "mixed checkpoint PublicTest identity does not match the runtime source"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m occlusion_fer.occlusion_evaluate",
        description=(
            "Evaluate one checkpoint on FER2013 PublicTest clean plus the nine "
            "fixed occlusion-v2-224 conditions."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", "--dataset", dest="data_path", type=Path)
    parser.add_argument("--training-mean", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", required=True, type=Path
    )
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--training-commit")
    parser.add_argument(
        "--allow-nonofficial",
        action="store_true",
        help="allow non-official synthetic source and manifest fixtures",
    )
    return parser


def run_publictest_evaluation(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_directory: str | Path,
    data_path: str | Path | None = None,
    training_mean_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    device_name: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    amp_enabled: bool = False,
    training_commit: str | None = None,
    allow_nonofficial: bool = False,
) -> dict[str, Path]:
    """Run the complete fail-closed PublicTest evaluation workflow."""
    output_path = _reserve_evaluation_output_directory(output_directory)
    config = load_config(config_path)
    if config.occlusion is None:
        raise OcclusionEvaluationError(
            "PublicTest occlusion evaluation requires an occlusion v2 config"
        )
    if config.dataset.image_size != 224:
        raise OcclusionEvaluationError("PublicTest v2 evaluation requires image_size=224")
    resolved_data_path = Path(data_path or config.dataset.path).expanduser()
    mean_value = training_mean_path or config.occlusion.artifacts.training_mean
    manifest_value = manifest_path or config.occlusion.artifacts.manifest
    if mean_value is None or manifest_value is None:
        raise OcclusionEvaluationError(
            "evaluation requires Training mean and PublicTest manifest paths"
        )
    resolved_mean_path = Path(mean_value).expanduser()
    resolved_manifest_path = Path(manifest_value).expanduser()
    source = load_stage_b_source(resolved_data_path)
    require_official = not allow_nonofficial
    if require_official:
        validate_official_stage_b_sources(source)
    mean_artifact = load_training_mean_v2(resolved_mean_path)
    validate_training_mean_v2(
        mean_artifact,
        training_dataset_sha256=source.training.dataset_sha256,
        consumer_image_size=224,
    )
    mean_sha256 = training_mean_v2_sha256(mean_artifact)
    manifest_sidecar = resolved_manifest_path.with_suffix(".json")
    manifest_rows, manifest_envelope = load_manifest_v2(
        resolved_manifest_path,
        manifest_sidecar,
        publictest_dataset_sha256=source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha256,
        require_official=require_official,
    )
    model = create_resnet18(num_classes=config.dataset.num_classes, pretrained=False)
    device = select_device(device_name or config.training.device)
    model = model.to(device)
    checkpoint_payload, checkpoint_sha256 = load_masked_evaluation_checkpoint(
        model,
        checkpoint_path,
    )
    evaluation_commit, evaluation_dirty = _git_identity(Path.cwd())
    resolved_training_commit, training_dirty = _training_identity(
        checkpoint_path,
        checkpoint_payload,
        explicit_commit=training_commit,
        fallback_commit=evaluation_commit,
        allow_nonofficial=allow_nonofficial,
    )
    checkpoint_seed = checkpoint_payload.get("seed", config.training.seed)
    if type(checkpoint_seed) is not int or checkpoint_seed < 0:
        raise OcclusionEvaluationError("checkpoint seed must be a non-negative integer")
    provenance = {
        "checkpoint_sha256": checkpoint_sha256,
        "training_commit": resolved_training_commit,
        "evaluation_commit": evaluation_commit,
        "git_dirty": evaluation_dirty,
        "resolved_config": asdict(config),
        "training_dataset_sha256": source.training.dataset_sha256,
        "publictest_dataset_sha256": source.publictest.dataset_sha256,
        "training_mean_sha256": mean_sha256,
        "manifest_sha256": manifest_envelope.manifest_sha256,
        "protocol": "occlusion-v2-224",
        "image_height": 224,
        "image_width": 224,
        "seed": checkpoint_seed,
        "run_role": (
            "synthetic_masked_evaluation"
            if allow_nonofficial else "formal_masked_evaluation"
        ),
        "source_routing_version": STAGE_B_SOURCE_ROUTING_VERSION,
        "training_git_dirty": training_dirty,
    }
    _validate_evaluation_provenance(provenance)
    results = evaluate_publictest_source(
        model,
        source,
        device,
        image_size=config.dataset.image_size,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        num_workers=(
            config.training.num_workers if num_workers is None else num_workers
        ),
        training_mean_artifact=mean_artifact,
        manifest_rows=manifest_rows,
        manifest_envelope=manifest_envelope,
        checkpoint_payload=checkpoint_payload,
        evaluation_provenance=provenance,
        amp_enabled=amp_enabled,
        require_official_manifest=require_official,
    )
    return write_occlusion_evaluation_artifacts(
        output_path,
        results,
        provenance=provenance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output_dir).expanduser()
    may_write_failure = not (
        output_path.exists()
        and (not output_path.is_dir() or any(output_path.iterdir()))
    )
    try:
        paths = run_publictest_evaluation(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_directory=args.output_dir,
            data_path=args.data_path,
            training_mean_path=args.training_mean,
            manifest_path=args.manifest,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp_enabled=args.amp,
            training_commit=args.training_commit,
            allow_nonofficial=args.allow_nonofficial,
        )
    except Exception as exc:
        if may_write_failure:
            try:
                _make_failure_directory(output_path)
                write_occlusion_evaluation_failure(
                    output_path,
                    stage="publictest_evaluation",
                    exception=exc,
                )
            except Exception as artifact_exc:
                print(f"failure_artifact_error={artifact_exc}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"status": "completed", "artifacts": sorted(str(path) for path in paths.values())},
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _reserve_evaluation_output_directory(output_directory: str | Path) -> Path:
    path = Path(output_directory).expanduser()
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"evaluation output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"evaluation output directory already contains artifacts: {path}"
            )
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _make_failure_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"evaluation output path is not a directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _git_identity(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OcclusionEvaluationError(
            f"could not collect Git provenance from {repository_root}"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise OcclusionEvaluationError("Git HEAD is not a full commit identity")
    return commit, bool(dirty)


def _training_identity(
    checkpoint_path: str | Path,
    checkpoint_payload: Mapping[str, object],
    *,
    explicit_commit: str | None,
    fallback_commit: str,
    allow_nonofficial: bool,
) -> tuple[str, bool]:
    if explicit_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", explicit_commit) is None:
            raise OcclusionEvaluationError(
                "training_commit must be a full lowercase Git commit"
            )
        return explicit_commit, False
    checkpoint = Path(checkpoint_path).expanduser()
    candidates = (checkpoint.parent / "run_metadata.json", checkpoint.parent.parent / "run_metadata.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, Mapping)
            and isinstance(payload.get("git_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", payload["git_commit"])
        ):
            return payload["git_commit"], bool(payload.get("git_dirty", False))
    payload_commit = checkpoint_payload.get("training_commit")
    if (
        isinstance(payload_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", payload_commit)
    ):
        return payload_commit, bool(checkpoint_payload.get("training_git_dirty", False))
    if allow_nonofficial:
        return fallback_commit, True
    raise OcclusionEvaluationError(
        "formal evaluation requires training Git provenance beside the checkpoint "
        "or --training-commit"
    )


if __name__ == "__main__":
    raise SystemExit(main())
