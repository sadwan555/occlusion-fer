"""Reproducible clean ResNet-18 training entry point for FER2013."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

from occlusion_fer.config import (
    AppConfig,
    EarlyStoppingConfig,
    LossConfig,
    TrainingConfig,
    load_config,
)
from occlusion_fer.data import load_fer2013_csv
from occlusion_fer.evaluation import EvaluationResult, evaluate
from occlusion_fer.artifacts import (
    collect_run_metadata,
    utc_now,
    write_evaluation_artifacts,
    write_failure_artifact,
    write_history_artifacts,
    write_json_atomic,
    write_resolved_config,
)
from occlusion_fer.models import create_resnet18
from occlusion_fer.losses import build_training_criterion
from occlusion_fer.schedulers import EpochLearningRateScheduler
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.occlusion import (
    apply_training_batch_v2,
    normalized_fill_vector_v2,
    select_training_condition_v2,
)
from occlusion_fer.permitted_splits import (
    is_permitted_splits_artifact_path,
    load_permitted_splits,
    permitted_splits_to_data,
    reject_combined_dataset_path,
)
from occlusion_fer.training_mean import (
    load_training_mean_v2,
    training_mean_v2_sha256,
)


Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class TrainingResult:
    average_loss: float
    accuracy: float
    sample_count: int
    condition_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class EarlyStopping:
    """Count consecutive epochs without strict macro-F1 improvement."""

    config: EarlyStoppingConfig
    epochs_without_improvement: int = 0

    def update(self, *, improved: bool) -> bool:
        if type(improved) is not bool:
            raise ValueError("improved must be a bool")
        if not self.config.enabled:
            return False
        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.config.patience


def is_better_validation_macro_f1(candidate: float, best: float) -> bool:
    """Return whether a finite validation macro-F1 strictly improves best."""
    if (
        type(candidate) not in (int, float)
        or not math.isfinite(candidate)
        or not 0.0 <= candidate <= 1.0
    ):
        raise ValueError(
            "candidate macro-F1 must be finite and between 0 and 1"
        )
    if (
        type(best) not in (int, float)
        or not math.isfinite(best)
        or not (best == -1.0 or 0.0 <= best <= 1.0)
    ):
        raise ValueError(
            "best macro-F1 must be finite and between -1 and 1"
        )
    return candidate > best


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch compute device."""
    allowed = ("auto", "cpu", "mps", "cuda")
    if type(device_name) is not str or device_name not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"device_name must be one of {choices}")

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if device_name == "mps":
        if not _mps_is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    training_seed: int | None = None,
    epoch: int | None = None,
    fill_vector: Sequence[float] | None = None,
    clean_probability: float = 0.5,
) -> TrainingResult:
    """Run one training epoch and return sample-weighted loss."""
    _validate_amp(amp_enabled, device)
    if amp_enabled and scaler is None:
        raise ValueError("AMP training requires a GradScaler")
    if criterion is None:
        criterion = build_training_criterion(LossConfig())
    if not isinstance(criterion, nn.Module):
        raise ValueError("criterion must be a torch.nn.Module")
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    sample_count = 0
    non_blocking = device.type == "cuda"
    mixed_context = (training_seed is not None) or (epoch is not None) or (fill_vector is not None)
    if mixed_context and (training_seed is None or epoch is None or fill_vector is None):
        raise ValueError("mixed masking requires training_seed, epoch, and fill_vector")
    condition_counts: dict[str, int] = {}

    for images, labels, sample_ids in loader:
        batch_size = _validate_batch(images, labels, sample_ids)
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        sample_ids = sample_ids.to(device, non_blocking=non_blocking)

        if mixed_context:
            selections = tuple(
                select_training_condition_v2(
                    int(sample_id), training_seed, epoch,
                    clean_probability=clean_probability,
                )
                for sample_id in sample_ids.detach().cpu().tolist()
            )
            masked_images = images.clone()
            for condition in sorted({item for item in selections if item is not None}):
                indices = torch.tensor(
                    [index for index, item in enumerate(selections) if item == condition],
                    dtype=torch.int64,
                    device=device,
                )
                condition_images = images.index_select(0, indices)
                condition_ids = sample_ids.index_select(0, indices)
                masked, _ = apply_training_batch_v2(
                    condition_images,
                    condition_ids,
                    condition,
                    fill_vector,
                    training_seed=training_seed,
                    epoch=epoch,
                )
                masked_images.index_copy_(0, indices, masked)
            images = masked_images
            for selected in selections:
                key = "clean" if selected is None else selected
                condition_counts[key] = condition_counts.get(key, 0) + 1

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            logits = model(images)
            _validate_logits(logits, batch_size)
            loss = criterion(logits, labels)
        loss_value = _validated_loss_value(loss)
        correct_predictions += int(
            (logits.detach().argmax(dim=1) == labels).sum().item()
        )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss_value * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("training loader has no samples")
    return TrainingResult(
        average_loss=total_loss / sample_count,
        accuracy=correct_predictions / sample_count,
        sample_count=sample_count,
        condition_counts=condition_counts,
    )


def limit_dataset(
    dataset: Dataset[Any], max_samples: int | None
) -> Dataset[Any]:
    """Use the first requested samples without crossing dataset boundaries."""
    if max_samples is None:
        return dataset
    if type(max_samples) is not int or max_samples <= 0:
        raise ValueError("max_samples must be a positive integer or None")
    actual_count = min(max_samples, len(dataset))
    if actual_count == len(dataset):
        return dataset
    return Subset(dataset, range(actual_count))


def build_optimizer(
    model: nn.Module, training: TrainingConfig
) -> torch.optim.AdamW:
    """Build the single AdamW optimizer from the resolved training config."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )


def apply_config_overrides(
    config: AppConfig,
    *,
    data_path: str | None = None,
    output_directory: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> AppConfig:
    """Return a configuration copy with one-run command-line overrides."""
    dataset = config.dataset
    output = config.output
    training = config.training

    if data_path is not None:
        if type(data_path) is not str or not data_path.strip():
            raise ValueError("data_path override must be a non-empty string")
        dataset = replace(dataset, path=data_path)
    if output_directory is not None:
        if type(output_directory) is not str or not output_directory.strip():
            raise ValueError(
                "output_directory override must be a non-empty string"
            )
        output = replace(output, directory=output_directory)
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed override must be a non-negative integer")
        training = replace(training, seed=seed)
    if device is not None:
        allowed_devices = ("auto", "cpu", "mps", "cuda")
        if type(device) is not str or device not in allowed_devices:
            choices = ", ".join(allowed_devices)
            raise ValueError(
                f"device override must be one of {choices}"
            )
        training = replace(training, device=device)
    if epochs is not None:
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs override must be a positive integer")
        training = replace(training, epochs=epochs)
    if batch_size is not None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size override must be a positive integer")
        training = replace(training, batch_size=batch_size)
    if num_workers is not None:
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError(
                "num_workers override must be a non-negative integer"
            )
        training = replace(training, num_workers=num_workers)

    if not 0 <= training.scheduler.warmup_epochs < training.epochs:
        raise ValueError(
            "scheduler warmup_epochs must satisfy "
            "0 <= warmup_epochs < epochs after overrides"
        )

    return replace(
        config,
        dataset=dataset,
        training=training,
        output=output,
    )


def load_config_with_overrides(
    config_path: str | Path,
    *,
    data_path: str | None = None,
    output_directory: str | None = None,
    seed: int | None = None,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> AppConfig:
    """Load YAML and apply temporary runtime overrides without editing it."""
    config = load_config(config_path)
    return apply_config_overrides(
        config,
        data_path=data_path,
        output_directory=output_directory,
        seed=seed,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def save_checkpoint(
    *,
    output_directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_macro_f1: float,
    validation_accuracy: float,
    validation_macro_f1: float,
    seed: int,
    device: torch.device,
    resolved_config: Mapping[str, object],
    checkpoint_name: str = "best.pt",
    scaler: torch.amp.GradScaler | None = None,
    scheduler: EpochLearningRateScheduler | None = None,
) -> Path:
    """Save a reloadable best or last training checkpoint."""
    if checkpoint_name not in {"best.pt", "last.pt"}:
        raise ValueError("checkpoint_name must be 'best.pt' or 'last.pt'")
    output_path = Path(output_directory).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Could not create output directory: {output_path}"
        ) from exc

    checkpoint_path = output_path / checkpoint_name
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "seed": seed,
        "device": str(device),
        "resolved_config": dict(resolved_config),
    }
    if scaler is not None:
        payload["grad_scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    try:
        torch.save(payload, checkpoint_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not save checkpoint: {checkpoint_path}"
        ) from exc

    resolved_path = checkpoint_path.resolve()
    print(f"saved_checkpoint={resolved_path}")
    return resolved_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a clean or bounded smoke run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible FER2013 clean ResNet-18 training or a bounded "
            "smoke test."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser.parse_args(argv)


def run_training(
    config: AppConfig,
    *,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    amp_enabled: bool = False,
) -> None:
    """Run clean training and preserve bounded failure information."""
    if config.training.mode == "mixed":
        if not _uses_permitted_splits_artifact(config):
            reject_combined_dataset_path(config.dataset.path, occlusion_enabled=True)
        _resolve_permitted_splits_path(config)
    else:
        _resolve_fer2013_data_path(config)
    output_path = _reserve_output_directory(config.output.directory)
    try:
        _run_training(
            config,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            amp_enabled=amp_enabled,
        )
    except Exception as exc:
        if output_path.is_dir():
            try:
                _record_run_failure(output_path, exc)
            except Exception as artifact_exc:
                print(f"failure_artifact_error={artifact_exc}")
        raise


def _run_training(
    config: AppConfig,
    *,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    amp_enabled: bool = False,
) -> None:
    """Execute the approved clean train/validation workflow."""
    set_seed(config.training.seed)
    device = select_device(config.training.device)
    _validate_amp(amp_enabled, device)
    _print_device_diagnostics(config.training.device, device)
    print(f"amp_enabled={amp_enabled}")

    if config.training.mode == "mixed":
        data = permitted_splits_to_data(
            load_permitted_splits(_resolve_permitted_splits_path(config))
        )
    else:
        data_path = _resolve_fer2013_data_path(config)
        data = load_fer2013_csv(
            data_path,
            include_splits=("train", "validation"),
        )
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
    train_dataset = limit_dataset(train_dataset, max_train_samples)
    validation_dataset = limit_dataset(
        validation_dataset, max_validation_samples
    )

    if max_train_samples is not None or max_validation_samples is not None:
        print("SMOKE TEST — NOT A FORMAL EXPERIMENT")
    print(f"actual_train_samples={len(train_dataset)}")
    print(f"actual_validation_samples={len(validation_dataset)}")

    mixed_fill_vector: Sequence[float] | None = None
    mixed_mean_sha256: str | None = None
    if config.training.mode == "mixed":
        if config.occlusion is None or config.occlusion.artifacts.training_mean is None:
            raise ValueError("mixed training requires occlusion.artifacts.training_mean")
        mean_artifact = load_training_mean_v2(
            config.occlusion.artifacts.training_mean
        )
        mixed_fill_vector = normalized_fill_vector_v2(mean_artifact)
        mixed_mean_sha256 = training_mean_v2_sha256(mean_artifact)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=2,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=2,
    )
    resolved_config = asdict(config)
    resolved_config["smoke_limits"] = {
        "max_train_samples": max_train_samples,
        "max_validation_samples": max_validation_samples,
        "actual_train_samples": len(train_dataset),
        "actual_validation_samples": len(validation_dataset),
    }
    resolved_config["runtime"] = {
        "amp_enabled": amp_enabled,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.training.num_workers > 0,
        "prefetch_factor": 2 if config.training.num_workers > 0 else None,
    }
    if mixed_mean_sha256 is not None:
        resolved_config["occlusion_runtime"] = {
            "training_mean_sha256": mixed_mean_sha256,
        }
    protocol_identity = _training_protocol_identity(config)
    if mixed_mean_sha256 is not None:
        protocol_identity["training_mean_sha256"] = mixed_mean_sha256
    output_path = Path(config.output.directory).expanduser()
    resolved_config_path = write_resolved_config(output_path, resolved_config)
    started_at_utc = utc_now()
    run_metadata_path = output_path / "run_metadata.json"
    running_metadata = collect_run_metadata(
        status="running",
        started_at_utc=started_at_utc,
        finished_at_utc=None,
        seed=config.training.seed,
        training_mode=config.training.mode,
        requested_device=config.training.device,
        selected_device=str(device),
        amp_enabled=amp_enabled,
        repository_root=Path.cwd(),
        output_directory=output_path,
        artifact_paths={"resolved_config": resolved_config_path},
        cuda_device_name=_cuda_device_name(device),
        run_role=config.project.run_role,
        protocol_identity=protocol_identity,
    )
    write_json_atomic(run_metadata_path, running_metadata)

    model = create_resnet18(
        num_classes=config.dataset.num_classes,
        pretrained=config.model.pretrained,
    ).to(device)
    optimizer = build_optimizer(model, config.training)
    training_criterion = build_training_criterion(config.training.loss)
    scheduler = (
        None
        if config.training.scheduler.type == "none"
        else EpochLearningRateScheduler(
            optimizer,
            config.training.scheduler,
            base_learning_rate=config.training.learning_rate,
            total_epochs=config.training.epochs,
        )
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    )
    early_stopping = EarlyStopping(config.training.early_stopping)

    best_validation_macro_f1 = -1.0
    history: list[dict[str, object]] = []
    best_evaluation_artifacts: dict[str, Path] = {}
    last_evaluation_artifacts: dict[str, Path] = {}
    history_json_path: Path | None = None
    history_csv_path: Path | None = None
    last_checkpoint: Path | None = None

    for epoch in range(1, config.training.epochs + 1):
        if scheduler is not None:
            learning_rates = scheduler.set_epoch(epoch)
        else:
            learning_rates = tuple(
                float(group["lr"]) for group in optimizer.param_groups
            )
        learning_rate = learning_rates[0]
        _reset_peak_memory(device)
        _synchronize_device(device)
        epoch_started = time.perf_counter()
        training_started = epoch_started
        training_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion=training_criterion,
            amp_enabled=amp_enabled,
            scaler=scaler,
            training_seed=config.training.seed if config.training.mode == "mixed" else None,
            epoch=epoch if config.training.mode == "mixed" else None,
            fill_vector=mixed_fill_vector,
            clean_probability=(
                config.occlusion.sampling.clean_probability
                if config.occlusion is not None else 0.5
            ),
        )
        _synchronize_device(device)
        training_seconds = time.perf_counter() - training_started

        validation_started = time.perf_counter()
        validation_result = evaluate(
            model,
            validation_loader,
            device,
            split="validation",
            condition="clean",
            amp_enabled=amp_enabled,
        )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        train_samples_per_second = (
            training_result.sample_count / training_seconds
        )
        cuda_peak_memory_bytes = _peak_memory_bytes(device)

        last_evaluation_artifacts = write_evaluation_artifacts(
            output_path,
            "validation/last",
            validation_result,
        )
        updated_best_checkpoint = is_better_validation_macro_f1(
            validation_result.macro_f1,
            best_validation_macro_f1,
        )
        should_stop_early = early_stopping.update(
            improved=updated_best_checkpoint
        )
        saved_checkpoint: Path | None = None
        if updated_best_checkpoint:
            best_validation_macro_f1 = validation_result.macro_f1
            best_evaluation_artifacts = write_evaluation_artifacts(
                output_path,
                "validation/best",
                validation_result,
            )
            saved_checkpoint = save_checkpoint(
                output_directory=config.output.directory,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_macro_f1=best_validation_macro_f1,
                validation_accuracy=validation_result.accuracy,
                validation_macro_f1=validation_result.macro_f1,
                seed=config.training.seed,
                device=device,
                resolved_config=resolved_config,
                checkpoint_name="best.pt",
                scaler=scaler,
                scheduler=scheduler,
            )

        last_checkpoint = save_checkpoint(
            output_directory=config.output.directory,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_macro_f1=best_validation_macro_f1,
            validation_accuracy=validation_result.accuracy,
            validation_macro_f1=validation_result.macro_f1,
            seed=config.training.seed,
            device=device,
            resolved_config=resolved_config,
            checkpoint_name="last.pt",
            scaler=scaler,
            scheduler=scheduler,
        )
        history.append(
            {
                "epoch": epoch,
                "train_samples": training_result.sample_count,
                "train_loss": training_result.average_loss,
                "train_accuracy": training_result.accuracy,
                "learning_rate": learning_rate,
                "validation_samples": validation_result.sample_count,
                "validation_loss": validation_result.average_loss,
                "validation_accuracy": validation_result.accuracy,
                "validation_macro_f1": validation_result.macro_f1,
                "train_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_second": train_samples_per_second,
                "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
                "updated_best_checkpoint": updated_best_checkpoint,
                "train_condition_counts": training_result.condition_counts,
            }
        )
        history_json_path, history_csv_path = write_history_artifacts(
            output_path, history
        )

        print(f"Epoch {epoch}/{config.training.epochs}")
        print(f"train_samples={training_result.sample_count}")
        print(f"train_loss={training_result.average_loss:.4f}")
        print(f"train_accuracy={training_result.accuracy:.4f}")
        print(f"learning_rate={learning_rate:.12g}")
        print(f"validation_samples={validation_result.sample_count}")
        print(f"validation_loss={validation_result.average_loss:.4f}")
        print(f"validation_accuracy={validation_result.accuracy:.4f}")
        print(f"validation_macro_f1={validation_result.macro_f1:.4f}")
        print(f"train_seconds={training_seconds:.6f}")
        print(f"validation_seconds={validation_seconds:.6f}")
        print(f"epoch_seconds={epoch_seconds:.6f}")
        print(
            "train_samples_per_second="
            f"{train_samples_per_second:.2f}"
        )
        print(f"cuda_peak_memory_bytes={cuda_peak_memory_bytes}")
        if saved_checkpoint is None:
            print("saved_best_checkpoint=none")
        else:
            print(f"saved_best_checkpoint={saved_checkpoint}")
        print(f"saved_last_checkpoint={last_checkpoint}")
        print(f"saved_history={history_json_path}")
        if should_stop_early:
            print(
                "early_stopping_triggered=True "
                f"patience={config.training.early_stopping.patience}"
            )
            break

    if history_json_path is None or history_csv_path is None:
        raise RuntimeError("training completed without history artifacts")
    if last_checkpoint is None or not best_evaluation_artifacts:
        raise RuntimeError("training completed without required checkpoints")
    artifact_paths: dict[str, Path] = {
        "resolved_config": resolved_config_path,
        "history_json": history_json_path,
        "history_csv": history_csv_path,
        "best_checkpoint": output_path / "best.pt",
        "last_checkpoint": last_checkpoint,
    }
    artifact_paths.update(
        {
            f"validation_best_{name}": path
            for name, path in best_evaluation_artifacts.items()
        }
    )
    artifact_paths.update(
        {
            f"validation_last_{name}": path
            for name, path in last_evaluation_artifacts.items()
        }
    )
    completed_metadata = collect_run_metadata(
        status="completed",
        started_at_utc=started_at_utc,
        finished_at_utc=utc_now(),
        seed=config.training.seed,
        training_mode=config.training.mode,
        requested_device=config.training.device,
        selected_device=str(device),
        amp_enabled=amp_enabled,
        repository_root=Path.cwd(),
        output_directory=output_path,
        artifact_paths=artifact_paths,
        cuda_device_name=_cuda_device_name(device),
        run_role=config.project.run_role,
        protocol_identity=protocol_identity,
    )
    write_json_atomic(run_metadata_path, completed_metadata)


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and run one command-line training job."""
    args = parse_args(argv)
    config = load_config_with_overrides(
        args.config,
        data_path=args.data_path,
        output_directory=args.output_dir,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    run_training(
        config,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        amp_enabled=args.amp,
    )


def _training_protocol_identity(config: AppConfig) -> dict[str, object]:
    """Return bounded protocol provenance without embedding source data."""
    if config.occlusion is None:
        return {
            "training_mode": config.training.mode,
            "image_size": config.dataset.image_size,
        }
    protocol = config.occlusion.protocol
    return {
        "training_mode": config.training.mode,
        "algorithm_version": protocol.algorithm_version,
        "mean_algorithm_version": protocol.mean_algorithm_version,
        "manifest_schema_version": protocol.manifest_schema_version,
        "image_size": protocol.image_size,
        "types": list(protocol.types),
        "ratios": list(protocol.ratios),
        "evaluation_mask_seed": protocol.evaluation_mask_seed,
        "sampling_clean_probability": config.occlusion.sampling.clean_probability,
        "training_mean_artifact": config.occlusion.artifacts.training_mean,
        "manifest": config.occlusion.artifacts.manifest,
    }


def _validate_batch(
    images: Tensor, labels: Tensor, sample_ids: Tensor
) -> int:
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise ValueError("images must be a four-dimensional Tensor")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional Tensor")
    try:
        sample_id_count = len(sample_ids)
    except TypeError as exc:
        raise ValueError("sample_ids must contain one value per sample") from exc
    batch_size = images.shape[0]
    if labels.shape[0] != batch_size or sample_id_count != batch_size:
        raise ValueError(
            "batch size must match for images, labels, and sample_ids"
        )
    return batch_size


def _validate_logits(logits: Tensor, batch_size: int) -> None:
    if not isinstance(logits, Tensor) or logits.ndim != 2:
        raise ValueError("logits must be a two-dimensional Tensor")
    if logits.shape[0] != batch_size:
        raise ValueError("logits batch size must match the input batch size")
    if logits.shape[1] != 7:
        raise ValueError("logits must contain scores for exactly 7 classes")


def _validated_loss_value(loss: Tensor) -> float:
    if loss.ndim != 0:
        raise ValueError("loss must be a scalar")
    value = float(loss.detach().item())
    if not math.isfinite(value):
        raise ValueError("loss must be finite")
    return value


def _validate_amp(amp_enabled: bool, device: torch.device) -> None:
    if type(amp_enabled) is not bool:
        raise ValueError("amp_enabled must be a bool")
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP can only be enabled for a CUDA device")


def _autocast_context(amp_enabled: bool):
    if amp_enabled:
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )
    return nullcontext()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _record_run_failure(
    output_directory: Path, exception: BaseException
) -> None:
    timestamp = utc_now()
    failure_path = write_failure_artifact(
        output_directory,
        stage="training",
        exception=exception,
        timestamp_utc=timestamp,
    )
    metadata_path = output_directory / "run_metadata.json"
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not update failed run metadata: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Run metadata must be a mapping: {metadata_path}")
    metadata["status"] = "failed"
    metadata["finished_at_utc"] = timestamp
    artifacts = metadata.setdefault("artifacts", {})
    if isinstance(artifacts, dict):
        artifacts["failure"] = failure_path.relative_to(
            output_directory.resolve()
        ).as_posix()
    write_json_atomic(metadata_path, metadata)


def _resolve_fer2013_data_path(config: AppConfig) -> Path:
    if config.dataset.path == "/path/to/fer2013.csv":
        raise FileNotFoundError(
            "FER2013 data path is still the placeholder; provide --data-path"
        )
    data_path = Path(config.dataset.path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"FER2013 data path does not exist or is not a file: {data_path}"
        )
    return data_path


def _uses_permitted_splits_artifact(config: AppConfig) -> bool:
    return (
        config.training.mode == "mixed"
        and config.dataset.permitted_splits == ("Training", "PublicTest")
        and is_permitted_splits_artifact_path(config.dataset.path)
    )


def _resolve_permitted_splits_path(config: AppConfig) -> Path:
    if config.dataset.permitted_splits != ("Training", "PublicTest"):
        raise ValueError(
            "mixed training requires dataset.permitted_splits=[Training, PublicTest]"
        )
    if not is_permitted_splits_artifact_path(config.dataset.path):
        raise ValueError(
            "mixed training requires an explicit permitted-splits .json artifact path"
        )
    source_path = Path(config.dataset.path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"permitted-splits artifact does not exist or is not a file: {source_path}"
        )
    return source_path


def _reserve_output_directory(output_directory: str | Path) -> Path:
    output_path = Path(output_directory).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            "Training requires a new output directory and will not overwrite "
            f"existing evidence: {output_path}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not create output directory: {output_path}"
        ) from exc
    return output_path


def _cuda_device_name(device: torch.device) -> str | None:
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_name(device)


def _mps_is_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _print_device_diagnostics(
    requested_device: str, selected_device: torch.device
) -> None:
    print(f"requested_device={requested_device}")
    print(f"selected_device={selected_device}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"mps_available={_mps_is_available()}")
    if selected_device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(selected_device)}")


if __name__ == "__main__":
    main()
