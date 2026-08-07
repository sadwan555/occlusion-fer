"""Read-only server readiness checks for the FER2013 training pipeline."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor

from occlusion_fer.config import AppConfig, load_config
from occlusion_fer.data import (
    FER2013_LABEL_NAMES,
    Fer2013Data,
    Fer2013Split,
    load_fer2013_csv,
)
from occlusion_fer.models import create_resnet18
from occlusion_fer.mask_manifest import load_manifest_v2
from occlusion_fer.permitted_splits import (
    PermittedSplits,
    load_stage_b_source,
    permitted_splits_to_data,
    stage_b_source_kind,
    validate_official_stage_b_sources,
)
from occlusion_fer.training_mean import (
    load_training_mean_v2,
    training_mean_v2_sha256,
    validate_training_mean_v2,
)
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import (
    apply_config_overrides,
    limit_dataset,
    select_device,
)


PLACEHOLDER_DATA_PATH = "/path/to/fer2013.csv"
PREFLIGHT_SPLITS: tuple[Fer2013Split, ...] = ("train", "validation")
Batch = tuple[Tensor, Tensor, Tensor]


class PreflightError(RuntimeError):
    """Raised when a necessary preflight check cannot pass."""


@dataclass(frozen=True)
class PreflightResult:
    """Machine-readable result returned by :func:`run_preflight`."""

    success: bool
    warnings: tuple[str, ...] = ()
    failed_stage: str | None = None
    error: str | None = None


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for server preflight checks."""
    parser = argparse.ArgumentParser(
        description=(
            "Check FER2013 data, environment, batches, model forward, and "
            "output access without training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=_positive_integer)
    parser.add_argument("--max-train-samples", type=_positive_integer)
    parser.add_argument("--max-validation-samples", type=_positive_integer)
    parser.add_argument("--skip-model-forward", action="store_true")
    return parser


def collect_environment_info(
    *, requested_device: str, selected_device: torch.device
) -> dict[str, str]:
    """Collect version and accelerator diagnostics without changing state."""
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    info = {
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "requested_device": requested_device,
        "selected_device": str(selected_device),
        "cuda_available": str(torch.cuda.is_available()),
        "mps_available": str(mps_available),
    }
    if torch.cuda.is_available():
        info["torch_cuda_version"] = str(torch.version.cuda)
        try:
            info["cuda_device_count"] = str(torch.cuda.device_count())
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                info[f"cuda_device_{index}_name"] = properties.name
                info[f"cuda_device_{index}_memory_bytes"] = str(
                    properties.total_memory
                )
        except (AssertionError, RuntimeError) as exc:
            info["cuda_details_warning"] = str(exc)
    return info


def validate_data_path(path: str | Path) -> Path:
    """Validate that a non-placeholder FER2013 CSV is readable and nonempty."""
    raw_path = str(path)
    if not raw_path.strip():
        raise PreflightError("FER2013 data path must not be empty")
    if raw_path == PLACEHOLDER_DATA_PATH:
        raise PreflightError(
            "FER2013 data path is still the placeholder; provide --data-path"
        )

    data_path = Path(raw_path).expanduser()
    if not data_path.exists():
        raise PreflightError(f"FER2013 data file does not exist: {data_path}")
    if not data_path.is_file():
        raise PreflightError(f"FER2013 data path is not a file: {data_path}")
    if data_path.stat().st_size == 0:
        raise PreflightError(f"FER2013 data file is empty: {data_path}")
    if not os.access(data_path, os.R_OK):
        raise PreflightError(f"FER2013 data file is not readable: {data_path}")
    try:
        with data_path.open("rb") as data_file:
            if not data_file.read(1):
                raise PreflightError(f"FER2013 data file is empty: {data_path}")
    except OSError as exc:
        raise PreflightError(
            f"Could not read FER2013 data file: {data_path}: {exc}"
        ) from exc
    return data_path


def count_split_labels(data: Fer2013Data) -> dict[str, dict[int, int]]:
    """Count labels only for the splits allowed during training preflight."""
    counts = {
        split: {label: 0 for label in range(len(FER2013_LABEL_NAMES))}
        for split in PREFLIGHT_SPLITS
    }
    for record in data.records:
        if record.split in counts:
            counts[record.split][record.label] += 1
    return counts


def inspect_batch(name: str, batch: Batch, image_size: int) -> Tensor:
    """Validate and report one deterministic DataLoader batch."""
    images, labels, sample_ids = batch
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise PreflightError(f"{name} images must be a four-dimensional tensor")
    expected_shape = (images.shape[0], 3, image_size, image_size)
    if tuple(images.shape) != expected_shape:
        raise PreflightError(
            f"{name} image batch shape must be {expected_shape}; "
            f"got {tuple(images.shape)}"
        )
    if images.shape[0] == 0:
        raise PreflightError(f"{name} batch must not be empty")
    if images.dtype != torch.float32:
        raise PreflightError(
            f"{name} images must use torch.float32; got {images.dtype}"
        )
    if not torch.isfinite(images).all().item():
        raise PreflightError(f"{name} images contain non-finite values")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise PreflightError(f"{name} labels must be one-dimensional")
    if labels.dtype != torch.int64:
        raise PreflightError(
            f"{name} labels must use torch.int64; got {labels.dtype}"
        )
    if not isinstance(sample_ids, Tensor) or sample_ids.ndim != 1:
        raise PreflightError(f"{name} sample IDs must be one-dimensional")
    if sample_ids.dtype != torch.int64:
        raise PreflightError(
            f"{name} sample IDs must use torch.int64; got {sample_ids.dtype}"
        )
    batch_size = images.shape[0]
    if labels.shape[0] != batch_size or sample_ids.shape[0] != batch_size:
        raise PreflightError(
            f"{name} images, labels, and sample IDs must have equal batch size"
        )
    if torch.any((labels < 0) | (labels > 6)).item():
        raise PreflightError(f"{name} labels must be between 0 and 6")

    print(f"{name}_batch_shape={tuple(images.shape)}")
    print(f"{name}_actual_batch_size={batch_size}")
    print(f"{name}_image_dtype={images.dtype}")
    print(f"{name}_label_dtype={labels.dtype}")
    print(f"{name}_sample_id_dtype={sample_ids.dtype}")
    print(f"{name}_image_device={images.device}")
    print(f"{name}_label_device={labels.device}")
    print(f"{name}_sample_id_device={sample_ids.device}")
    print(f"{name}_images_finite=True")
    print(f"{name}_labels={labels.tolist()}")
    print(f"{name}_sample_ids={sample_ids.tolist()}")
    return images


def check_output_directory(output_directory: str | Path) -> Path:
    """Check create/write/read/delete access without touching existing files."""
    raw_path = str(output_directory)
    if not raw_path.strip():
        raise PreflightError("output directory must not be empty")
    output_path = Path(raw_path).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreflightError(
            f"Could not create output directory: {output_path}: {exc}"
        ) from exc
    if not output_path.is_dir():
        raise PreflightError(f"output path is not a directory: {output_path}")

    marker_path = output_path / ".preflight_write_test"
    if marker_path.exists():
        raise PreflightError(
            f"preflight marker already exists and will not be overwritten: "
            f"{marker_path}"
        )

    marker_created = False
    try:
        with marker_path.open("x", encoding="utf-8") as marker_file:
            marker_created = True
            marker_file.write("preflight write test")
        if marker_path.read_text(encoding="utf-8") != "preflight write test":
            raise PreflightError(
                f"output write test content could not be verified: {marker_path}"
            )
    except OSError as exc:
        raise PreflightError(
            f"Could not write and read output directory: {output_path}: {exc}"
        ) from exc
    finally:
        if marker_created:
            try:
                marker_path.unlink()
            except OSError as exc:
                raise PreflightError(
                    f"Could not remove preflight marker: {marker_path}: {exc}"
                ) from exc
    return output_path


def run_model_forward_check(
    images: Tensor, device: torch.device, *, num_classes: int
) -> None:
    """Run one randomly initialized ResNet-18 inference batch."""
    print("model_weights=pretrained_false_for_preflight")
    started = time.perf_counter()
    model = create_resnet18(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.eval()
    device_images = images.to(device)
    try:
        with torch.inference_mode():
            logits = model(device_images)
        expected_shape = (images.shape[0], num_classes)
        if not isinstance(logits, Tensor) or tuple(logits.shape) != expected_shape:
            actual_shape = getattr(logits, "shape", None)
            raise PreflightError(
                f"model logits shape must be {expected_shape}; got {actual_shape}"
            )
        if not logits.is_floating_point():
            raise PreflightError("model logits must use a floating-point dtype")
        if not torch.isfinite(logits).all().item():
            raise PreflightError("model logits contain non-finite values")
        print(f"model_logits_shape={tuple(logits.shape)}")
        print(f"model_logits_dtype={logits.dtype}")
        print(f"model_logits_device={logits.device}")
        print("model_logits_finite=True")
        print(f"model_forward_seconds={time.perf_counter() - started:.6f}")
    finally:
        if "logits" in locals():
            del logits
        del device_images
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _resolved_config(args: argparse.Namespace) -> AppConfig:
    config = load_config(args.config)
    config = apply_config_overrides(
        config,
        data_path=args.data_path,
        output_directory=args.output_dir,
    )
    training = config.training
    if args.device is not None:
        training = replace(training, device=args.device)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    return replace(config, training=training)


def _validate_loaded_records(data: Fer2013Data) -> None:
    for record in data.records:
        if record.image.shape != (48, 48):
            raise PreflightError(
                f"sample {record.sample_id} image shape is {record.image.shape}"
            )
        if record.image.dtype != np.uint8:
            raise PreflightError(
                f"sample {record.sample_id} image dtype is {record.image.dtype}"
            )
        if record.image.size == 0:
            raise PreflightError(f"sample {record.sample_id} image is empty")
        if int(record.image.min()) < 0 or int(record.image.max()) > 255:
            raise PreflightError(
                f"sample {record.sample_id} pixels must be between 0 and 255"
            )
        if type(record.label) is not int or not 0 <= record.label <= 6:
            raise PreflightError(
                f"sample {record.sample_id} label must be between 0 and 6"
            )
        if record.label_name != FER2013_LABEL_NAMES[record.label]:
            raise PreflightError(
                f"sample {record.sample_id} label name does not match its label"
            )


def _print_environment(info: Mapping[str, str]) -> None:
    print("=== ENVIRONMENT ===")
    for name, value in info.items():
        print(f"{name}={value}")


def _print_data_summary(
    data: Fer2013Data, split_counts: Mapping[str, Mapping[int, int]]
) -> tuple[str, ...]:
    print("=== FER2013 DATA ===")
    print(f"total_samples={len(data.records)}")
    print(f"train_samples={data.train_count}")
    print(f"validation_samples={data.validation_count}")
    warnings: list[str] = []
    for split in PREFLIGHT_SPLITS:
        formatted_counts = ", ".join(
            f"{label}:{FER2013_LABEL_NAMES[label]}={split_counts[split][label]}"
            for label in range(len(FER2013_LABEL_NAMES))
        )
        print(f"{split}_class_counts={formatted_counts}")
        missing = [
            f"{label}:{FER2013_LABEL_NAMES[label]}"
            for label, count in split_counts[split].items()
            if count == 0
        ]
        if missing:
            warnings.append(
                f"{split} split missing classes: {', '.join(missing)}"
            )
    return tuple(warnings)


def _print_success_summary(
    warnings: tuple[str, ...], *, model_forward_skipped: bool
) -> None:
    print("=== PREFLIGHT SUMMARY ===")
    print("environment=PASS")
    print("data_file=PASS")
    print("data=PASS")
    print("batches=PASS")
    print(
        "model_forward=SKIPPED"
        if model_forward_skipped
        else "model_forward=PASS"
    )
    print("output_directory=PASS")
    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("WARNINGS=none")
    print("PREFLIGHT PASSED")


def run_preflight(args: argparse.Namespace) -> PreflightResult:
    """Run all checks and convert expected failures into a clear result."""
    stage = "configuration"
    warnings: tuple[str, ...] = ()
    try:
        config = _resolved_config(args)
        stage_b_enabled = (
            config.training.mode == "mixed" or config.occlusion is not None
        )
        if config.training.mode == "mixed" or config.occlusion is not None:
            if config.dataset.image_size != 224:
                raise PreflightError(
                    "occlusion-enabled preflight requires image_size=224"
                )
            if config.training.mode == "mixed":
                if config.dataset.permitted_splits != ("Training", "PublicTest"):
                    raise PreflightError(
                        "mixed preflight requires "
                        "dataset.permitted_splits=[Training, PublicTest]"
                    )
            stage_b_source_kind(config.dataset.path)

        stage = "environment"
        requested_device = config.training.device
        selected_device = select_device(requested_device)
        _print_environment(
            collect_environment_info(
                requested_device=requested_device,
                selected_device=selected_device,
            )
        )

        stage = "data file"
        data_path = validate_data_path(config.dataset.path)
        print(f"data_path={data_path.resolve()}")
        if stage_b_enabled:
            print(f"stage_b_source_kind={stage_b_source_kind(data_path)}")

        stage = "data parsing"
        permitted_source = (
            load_stage_b_source(data_path) if stage_b_enabled else None
        )
        if config.training.mode == "mixed" and permitted_source is not None:
            if config.project.run_role == "formal_mixed":
                validate_official_stage_b_sources(permitted_source)
            stage = "artifact compatibility"
            _validate_mixed_artifacts(config, permitted_source)
            stage = "data parsing"
        data = (
            permitted_splits_to_data(permitted_source)
            if permitted_source is not None
            else load_fer2013_csv(
                data_path,
                include_splits=PREFLIGHT_SPLITS,
            )
        )
        _validate_loaded_records(data)
        split_counts = count_split_labels(data)
        warnings = _print_data_summary(data, split_counts)

        stage = "batch preparation"
        train_dataset = Fer2013TorchDataset(
            data,
            split="train",
            image_size=config.dataset.image_size,
            normalize_imagenet=True,
            augmentation=config.dataset.augmentation,
        )
        validation_dataset = Fer2013TorchDataset(
            data,
            split="validation",
            image_size=config.dataset.image_size,
            normalize_imagenet=True,
        )
        train_dataset = limit_dataset(train_dataset, args.max_train_samples)
        validation_dataset = limit_dataset(
            validation_dataset, args.max_validation_samples
        )
        if (
            args.max_train_samples is not None
            or args.max_validation_samples is not None
        ):
            print("PREFLIGHT SAMPLE LIMIT — NOT A FORMAL EXPERIMENT")
        print(f"actual_train_samples={len(train_dataset)}")
        print(f"actual_validation_samples={len(validation_dataset)}")
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        validation_loader = create_dataloader(
            validation_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            seed=config.training.seed,
            num_workers=config.training.num_workers,
        )
        train_images = inspect_batch(
            "train", next(iter(train_loader)), config.dataset.image_size
        )
        inspect_batch(
            "validation",
            next(iter(validation_loader)),
            config.dataset.image_size,
        )

        if not args.skip_model_forward:
            stage = "model forward"
            run_model_forward_check(
                train_images,
                selected_device,
                num_classes=config.dataset.num_classes,
            )

        stage = "output directory"
        output_path = check_output_directory(config.output.directory)
        print(f"output_path={output_path.resolve()}")

        _print_success_summary(
            warnings, model_forward_skipped=args.skip_model_forward
        )
        return PreflightResult(success=True, warnings=warnings)
    except Exception as exc:
        print("PREFLIGHT FAILED")
        print(f"failed_stage={stage}")
        print(f"reason={exc}")
        return PreflightResult(
            success=False,
            warnings=warnings,
            failed_stage=stage,
            error=str(exc),
        )


def _validate_mixed_artifacts(
    config: AppConfig, permitted_source: PermittedSplits
) -> None:
    if config.occlusion is None:
        raise PreflightError("mixed preflight requires an occlusion v2 configuration")
    mean_path = config.occlusion.artifacts.training_mean
    manifest_path = config.occlusion.artifacts.manifest
    if mean_path is None or manifest_path is None:
        raise PreflightError("mixed preflight requires mean and manifest artifacts")
    mean = load_training_mean_v2(mean_path)
    training_source = permitted_source.training
    validate_training_mean_v2(
        mean,
        training_dataset_sha256=training_source.dataset_sha256,
    )
    mean_sha = training_mean_v2_sha256(mean)
    manifest_csv = Path(manifest_path).expanduser()
    manifest_sidecar = manifest_csv.with_suffix(".json")
    load_manifest_v2(
        manifest_csv,
        manifest_sidecar,
        publictest_dataset_sha256=permitted_source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
        require_official=config.project.run_role == "formal_mixed",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run preflight from the command line and return a process exit code."""
    args = build_parser().parse_args(argv)
    result = run_preflight(args)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
