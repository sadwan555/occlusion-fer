"""Minimal ResNet-18 training entry point for FER2013 smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset

from occlusion_fer.config import AppConfig, load_config
from occlusion_fer.data import load_fer2013_csv
from occlusion_fer.models import create_resnet18
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader


Batch = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True)
class TrainingResult:
    average_loss: float
    sample_count: int


@dataclass(frozen=True)
class EvaluationResult:
    average_loss: float
    accuracy: float
    sample_count: int


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
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainingResult:
    """Run one training epoch and return sample-weighted loss."""
    _validate_amp(amp_enabled, device)
    if amp_enabled and scaler is None:
        raise ValueError("AMP training requires a GradScaler")
    model.train()
    total_loss = 0.0
    sample_count = 0
    non_blocking = device.type == "cuda"

    for images, labels, sample_ids in loader:
        batch_size = _validate_batch(images, labels, sample_ids)
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(amp_enabled):
            logits = model(images)
            _validate_logits(logits, batch_size)
            loss = F.cross_entropy(logits, labels)
        loss_value = _validated_loss_value(loss)
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
        sample_count=sample_count,
    )


def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> EvaluationResult:
    """Evaluate without parameter updates and return loss and accuracy."""
    _validate_amp(amp_enabled, device)
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for images, labels, sample_ids in loader:
            batch_size = _validate_batch(images, labels, sample_ids)
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            with _autocast_context(amp_enabled):
                logits = model(images)
                _validate_logits(logits, batch_size)
                loss = F.cross_entropy(logits, labels)
            loss_value = _validated_loss_value(loss)

            total_loss += loss_value * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("validation loader has no samples")
    accuracy = correct_count / sample_count
    return EvaluationResult(
        average_loss=total_loss / sample_count,
        accuracy=accuracy,
        sample_count=sample_count,
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


def apply_config_overrides(
    config: AppConfig,
    *,
    data_path: str | None = None,
    output_directory: str | None = None,
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
    best_validation_accuracy: float,
    seed: int,
    device: torch.device,
    resolved_config: Mapping[str, object],
    checkpoint_name: str = "best.pt",
    scaler: torch.amp.GradScaler | None = None,
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
        "best_validation_accuracy": best_validation_accuracy,
        "seed": seed,
        "device": str(device),
        "resolved_config": dict(resolved_config),
    }
    if scaler is not None:
        payload["grad_scaler_state_dict"] = scaler.state_dict()
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
    """Parse command-line arguments for a smoke training run."""
    parser = argparse.ArgumentParser(
        description="Run a minimal FER2013 ResNet-18 training smoke test."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
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
    """Execute the approved clean train/validation smoke workflow."""
    set_seed(config.training.seed)
    device = select_device(config.training.device)
    _validate_amp(amp_enabled, device)
    _print_device_diagnostics(config.training.device, device)
    print(f"amp_enabled={amp_enabled}")

    if config.dataset.path == "/path/to/fer2013.csv":
        raise FileNotFoundError(
            "FER2013 data path is still the placeholder; provide --data-path"
        )
    data_path = Path(config.dataset.path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"FER2013 data path does not exist or is not a file: {data_path}"
        )

    data = load_fer2013_csv(data_path)
    train_dataset = Fer2013TorchDataset(
        data,
        split="train",
        image_size=config.dataset.image_size,
        normalize_imagenet=True,
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
    model = create_resnet18(
        num_classes=config.dataset.num_classes,
        pretrained=config.model.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
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
    best_validation_accuracy = -1.0
    history: list[dict[str, int | float]] = []

    for epoch in range(1, config.training.epochs + 1):
        _reset_peak_memory(device)
        _synchronize_device(device)
        epoch_started = time.perf_counter()
        training_started = epoch_started
        training_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        _synchronize_device(device)
        training_seconds = time.perf_counter() - training_started

        validation_started = time.perf_counter()
        validation_result = evaluate(
            model,
            validation_loader,
            device,
            amp_enabled=amp_enabled,
        )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        train_samples_per_second = (
            training_result.sample_count / training_seconds
        )
        cuda_peak_memory_bytes = _peak_memory_bytes(device)

        saved_checkpoint: Path | None = None
        if validation_result.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_result.accuracy
            saved_checkpoint = save_checkpoint(
                output_directory=config.output.directory,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_validation_accuracy=best_validation_accuracy,
                seed=config.training.seed,
                device=device,
                resolved_config=resolved_config,
                checkpoint_name="best.pt",
                scaler=scaler,
            )

        last_checkpoint = save_checkpoint(
            output_directory=config.output.directory,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_validation_accuracy=best_validation_accuracy,
            seed=config.training.seed,
            device=device,
            resolved_config=resolved_config,
            checkpoint_name="last.pt",
            scaler=scaler,
        )
        history.append(
            {
                "epoch": epoch,
                "train_samples": training_result.sample_count,
                "train_loss": training_result.average_loss,
                "validation_samples": validation_result.sample_count,
                "validation_loss": validation_result.average_loss,
                "validation_accuracy": validation_result.accuracy,
                "train_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_second": train_samples_per_second,
                "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
            }
        )
        history_path = _save_training_history(
            config.output.directory, history
        )

        print(f"Epoch {epoch}/{config.training.epochs}")
        print(f"train_samples={training_result.sample_count}")
        print(f"train_loss={training_result.average_loss:.4f}")
        print(f"validation_samples={validation_result.sample_count}")
        print(f"validation_loss={validation_result.average_loss:.4f}")
        print(f"validation_accuracy={validation_result.accuracy:.4f}")
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
        print(f"saved_history={history_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and run one command-line training job."""
    args = parse_args(argv)
    config = load_config_with_overrides(
        args.config,
        data_path=args.data_path,
        output_directory=args.output_dir,
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


def _save_training_history(
    output_directory: str | Path,
    history: Sequence[Mapping[str, int | float]],
) -> Path:
    output_path = Path(output_directory).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        history_path = output_path / "history.json"
        history_path.write_text(
            json.dumps(list(history), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not save training history: {output_path / 'history.json'}"
        ) from exc
    return history_path.resolve()


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
