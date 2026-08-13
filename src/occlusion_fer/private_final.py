"""Fail-closed FER2013 PrivateTest final evaluation infrastructure."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch import nn

from occlusion_fer.artifacts import (
    write_evaluation_artifacts_to_directory,
    write_failure_artifact,
    write_json_atomic,
)
from occlusion_fer.evaluation import Batch, EvaluationResult, evaluate
from occlusion_fer.formal_checkpoints import (
    FORMAL_CHECKPOINTS,
    FormalCheckpoint,
    checkpoint_for_model_id,
    load_and_validate_checkpoint,
)
from occlusion_fer.private_manifest import (
    CONDITION_ORDER,
    INTERNAL_SPLIT,
    NORMALIZED_FILL,
    OFFICIAL_USAGE,
    PrivateManifestRow,
    PrivateManifestSidecar,
    apply_manifest_batch,
    build_manifest_lookup,
    load_manifest_artifacts,
    validate_manifest_binding,
)
from occlusion_fer.private_plan import load_final_plan
from occlusion_fer.private_preflight import ensure_empty_output_root


class PrivateFinalError(ValueError):
    """Raised before or during the explicit final-only evaluation route."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m occlusion_fer.private_final",
        description=(
            "Run the frozen FER2013 PrivateTest plan. This command is "
            "deliberately unavailable without an explicit confirmation."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--training-mean", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--confirm-private-test", action="store_true")
    return parser.parse_args(argv)


def require_private_confirmation(confirm_private_test: bool) -> None:
    if confirm_private_test is not True:
        raise PrivateFinalError(
            "PrivateTest final evaluation requires --confirm-private-test"
        )


def validate_private_route(internal_split: str, official_usage: str) -> None:
    if internal_split != INTERNAL_SPLIT or official_usage != OFFICIAL_USAGE:
        raise PrivateFinalError(
            "private final route allows only test/PrivateTest"
        )


def validate_checkpoint_binding(
    spec: FormalCheckpoint,
    actual_sha256: str,
) -> None:
    frozen = checkpoint_for_model_id(spec.model_id)
    if spec != frozen or actual_sha256 != frozen.sha256:
        raise PrivateFinalError(
            f"checkpoint SHA or frozen identity mismatch for {spec.model_id}"
        )


def evaluate_private_conditions(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    *,
    manifest_rows: Sequence[PrivateManifestRow],
    manifest_sidecar: PrivateManifestSidecar,
    expected_dataset_sha256: str | None = None,
    require_official_manifest: bool = True,
    amp_enabled: bool = False,
) -> dict[str, EvaluationResult]:
    """Evaluate clean plus nine masks without optimizer, backward, or mutation."""
    if not isinstance(model, nn.Module):
        raise PrivateFinalError("model must be a torch module")
    iterator = iter(loader)
    if iterator is loader:
        raise PrivateFinalError(
            "private evaluation loader must be re-iterable for ten conditions"
        )
    sample_id_batches = tuple(
        ids.detach().cpu().tolist() for _, _, ids in iterator
    )
    if not sample_id_batches:
        raise PrivateFinalError("private evaluation loader is empty")
    sample_ids = tuple(
        int(sample_id)
        for ids in sample_id_batches
        for sample_id in ids
    )
    validate_manifest_binding(
        manifest_rows,
        manifest_sidecar,
        expected_dataset_sha256=(
            manifest_sidecar.private_dataset_sha256
            if expected_dataset_sha256 is None
            else expected_dataset_sha256
        ),
        expected_sample_ids=sample_ids,
        require_official=require_official_manifest,
    )
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    manifest_lookup = build_manifest_lookup(manifest_rows)
    results: dict[str, EvaluationResult] = {}
    model.eval()
    for condition in CONDITION_ORDER:
        condition_loader: Iterable[Batch]
        if condition == "clean":
            condition_loader = loader
        else:
            condition_loader = (
                (
                    apply_manifest_batch(
                        images,
                        ids,
                        manifest_lookup,
                        condition,
                        NORMALIZED_FILL,
                    ),
                    labels,
                    ids,
                )
                for images, labels, ids in loader
            )
        results[condition] = evaluate(
            model,
            condition_loader,
            device,
            split=INTERNAL_SPLIT,
            condition=condition,
            amp_enabled=amp_enabled,
        )
    for name, tensor in model.state_dict().items():
        if name not in before or not torch.equal(
            tensor.detach().cpu(), before[name]
        ):
            raise PrivateFinalError(
                f"model parameter changed during final inference: {name}"
            )
        if tensor.grad is not None:
            raise PrivateFinalError(
                f"model parameter gradient was created during inference: {name}"
            )
    if tuple(results) != CONDITION_ORDER:
        raise PrivateFinalError(
            "private final results must contain exactly ten conditions"
        )
    return results


def run_private_final_evaluation(
    *,
    plan_path: str | Path,
    data_path: str | Path,
    manifest_path: str | Path,
    training_mean_path: str | Path,
    output_root: str | Path | None = None,
    device: torch.device | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    amp_enabled: bool = False,
    confirm_private_test: bool,
    require_official: bool = True,
    allow_nonofficial: bool = False,
) -> dict[str, Path]:
    """Execute the frozen route when explicitly invoked after human approval.

    ``allow_nonofficial`` is reserved for synthetic fixtures. The CLI keeps the
    official execution disabled during this infrastructure-only stage.
    """
    require_private_confirmation(confirm_private_test)
    plan, plan_sha = load_final_plan(plan_path)
    validate_private_route(plan.private.internal_split, plan.private.official_usage)
    target_root = Path(output_root or plan.output_root).expanduser()
    if target_root != Path(plan.output_root).expanduser():
        raise PrivateFinalError("runtime output root does not match the plan")
    ensure_empty_output_root(target_root)
    if allow_nonofficial and require_official:
        raise PrivateFinalError("synthetic route must disable official checks")
    if not allow_nonofficial and plan.private.canonical_sha256 != (
        "4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634"
    ):
        raise PrivateFinalError("plan does not bind the official PrivateTest SHA")

    target_root.mkdir(parents=True, exist_ok=target_root.exists())
    try:
        from occlusion_fer.private_preflight import (
            inspect_private_source,
            preflight_checkpoints,
            validate_training_mean_artifact,
        )

        preflight_checkpoints(plan)
        validate_training_mean_artifact(training_mean_path)
        identity = inspect_private_source(data_path)
        source = identity.data
        sample_ids = tuple(
            record.sample_id for record in source.records if record.split == "test"
        )
        if require_official:
            if source.test_count != 3589:
                raise PrivateFinalError("PrivateTest count must be 3589")
            if identity.canonical_sha256 != (
                "4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634"
            ):
                raise PrivateFinalError("PrivateTest canonical SHA mismatch")
        rows, sidecar = load_manifest_artifacts(
            manifest_path,
            Path(manifest_path).with_suffix(".json"),
        )
        validate_manifest_binding(
            rows,
            sidecar,
            expected_dataset_sha256=(
                plan.private.canonical_sha256
                if require_official
                else sidecar.private_dataset_sha256
            ),
            expected_sample_ids=sample_ids,
            require_official=require_official,
        )
        from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader

        dataset = Fer2013TorchDataset(
            source,
            split="test",
            image_size=224,
            normalize_imagenet=True,
        )
        resolved_device = device or torch.device("cpu")
        loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            num_workers=num_workers,
            pin_memory=resolved_device.type == "cuda",
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
        )
        for directory in ("protocol", "clean", "mixed", "summary"):
            (target_root / directory).mkdir()
        shutil.copyfile(
            Path(plan_path),
            target_root / "protocol" / "final_evaluation_plan.json",
        )
        shutil.copyfile(
            Path(manifest_path),
            target_root / "protocol" / "privatetest_manifest_v2.csv",
        )
        shutil.copyfile(
            Path(manifest_path).with_suffix(".json"),
            target_root / "protocol" / "privatetest_manifest_v2.json",
        )
        metric_values: dict[tuple[str, int, str], dict[str, float]] = {}
        artifact_paths: dict[str, Path] = {}
        for planned, frozen in zip(plan.checkpoints, FORMAL_CHECKPOINTS, strict=True):
            payload, checkpoint_sha = load_and_validate_checkpoint(planned.path, frozen)
            validate_checkpoint_binding(frozen, checkpoint_sha)
            from occlusion_fer.models import create_resnet18

            model = create_resnet18(num_classes=7, pretrained=False)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model = model.to(resolved_device)
            results = evaluate_private_conditions(
                model,
                loader,
                resolved_device,
                manifest_rows=rows,
                manifest_sidecar=sidecar,
                expected_dataset_sha256=sidecar.private_dataset_sha256,
                require_official_manifest=require_official,
                amp_enabled=amp_enabled,
            )
            conditions_root = target_root / planned.strategy / f"seed{planned.seed}" / "conditions"
            conditions_root.mkdir(parents=True, exist_ok=False)
            for condition, result in results.items():
                condition_root = conditions_root / condition
                artifact_paths.update(
                    {
                        f"{planned.model_id}:{condition}:{name}": path
                        for name, path in write_evaluation_artifacts_to_directory(
                            condition_root, result
                        ).items()
                    }
                )
                condition_rows = [row for row in rows if row.condition == condition]
                first = condition_rows[0] if condition_rows else None
                artifact_paths[f"{planned.model_id}:{condition}:provenance"] = write_json_atomic(
                    condition_root / "condition_provenance.json",
                    {
                        "plan_sha256": plan_sha,
                        "private_dataset_sha256": sidecar.private_dataset_sha256,
                        "manifest_sha256": sidecar.manifest_sha256,
                        "training_mean_sha256": sidecar.training_mean_artifact_sha256,
                        "checkpoint_sha256": checkpoint_sha,
                        "model_id": planned.model_id,
                        "strategy": planned.strategy,
                        "seed": planned.seed,
                        "best_epoch": planned.best_epoch,
                        "split": INTERNAL_SPLIT,
                        "official_usage": OFFICIAL_USAGE,
                        "condition": condition,
                        "protocol": sidecar.algorithm_version,
                        "image_size": sidecar.image_size,
                        "target_ratio": None if first is None else first.target_ratio,
                        "actual_ratio": None if first is None else first.actual_ratio,
                        "git_commit": plan.git_commit,
                        "git_dirty": plan.git_dirty,
                    },
                )
                metric_values[(planned.strategy, planned.seed, condition)] = {
                    "accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                }

        from occlusion_fer.private_aggregate import write_aggregate_artifacts

        artifact_paths.update(
            {
                f"summary:{name}": path
                for name, path in write_aggregate_artifacts(target_root, metric_values).items()
            }
        )
        artifact_manifest_path = target_root / "summary" / "artifact_manifest.json"
        artifact_manifest_rows = []
        for path in sorted(
            (candidate for candidate in target_root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(target_root).as_posix(),
        ):
            if path == artifact_manifest_path:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as artifact_file:
                for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            artifact_manifest_rows.append(
                {
                    "path": path.relative_to(target_root).as_posix(),
                    "sha256": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        artifact_paths["summary:artifact_manifest"] = write_json_atomic(
            artifact_manifest_path,
            {
                "schema_version": 1,
                "plan_sha256": plan_sha,
                "artifacts": artifact_manifest_rows,
            },
        )
        return artifact_paths
    except Exception as exc:
        if target_root.exists() and target_root.is_dir():
            try:
                write_failure_artifact(
                    target_root,
                    stage="private_final_evaluation",
                    exception=exc,
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run only after the explicit confirmation gate and frozen plan checks."""
    args = parse_args(argv)
    run_private_final_evaluation(
        plan_path=args.plan,
        data_path=args.data_path,
        manifest_path=args.manifest,
        training_mean_path=args.training_mean,
        device=(
            torch.device(args.device)
            if args.device != "auto"
            else torch.device("cpu")
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp_enabled=args.amp,
        confirm_private_test=args.confirm_private_test,
        require_official=True,
        allow_nonofficial=False,
        output_root=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
