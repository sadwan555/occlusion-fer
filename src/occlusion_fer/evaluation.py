"""Reusable inference evaluation and prediction collection."""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.metrics import PerClassMetrics, compute_classification_metrics


Batch = tuple[Tensor, Tensor, Tensor]
VALID_EVALUATION_SPLITS = ("validation", "test")


@dataclass(frozen=True)
class PredictionRecord:
    """One traceable model prediction without embedding source image data."""

    sample_id: int
    split: str
    condition: str
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    correct: bool
    predicted_confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Loss, metrics, and predictions for one evaluation condition."""

    average_loss: float
    accuracy: float
    macro_f1: float
    sample_count: int
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    predictions: tuple[PredictionRecord, ...]


def evaluate(
    model: nn.Module,
    loader: Iterable[Batch],
    device: torch.device,
    *,
    split: str = "validation",
    condition: str = "clean",
    amp_enabled: bool = False,
) -> EvaluationResult:
    """Evaluate without updates and retain paper-ready per-sample predictions."""
    _validate_evaluation_request(split, condition, amp_enabled, device)
    model.eval()
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    predictions: list[PredictionRecord] = []
    seen_sample_ids: set[int] = set()
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
            probabilities = torch.softmax(logits.detach().float(), dim=1).cpu()
            if not torch.isfinite(probabilities).all().item():
                raise ValueError("evaluation probabilities must be finite")
            batch_true_labels = labels.detach().cpu().tolist()
            batch_predicted_labels = probabilities.argmax(dim=1).tolist()
            batch_sample_ids = sample_ids.detach().cpu().tolist()

            for row_index in range(batch_size):
                sample_id = int(batch_sample_ids[row_index])
                if sample_id in seen_sample_ids:
                    raise ValueError(
                        f"evaluation contains duplicate sample ID {sample_id}"
                    )
                seen_sample_ids.add(sample_id)
                true_label = int(batch_true_labels[row_index])
                predicted_label = int(batch_predicted_labels[row_index])
                row_probabilities = tuple(
                    float(value) for value in probabilities[row_index].tolist()
                )
                confidence = row_probabilities[predicted_label]
                if not math.isfinite(confidence):
                    raise ValueError("predicted confidence must be finite")
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
                predictions.append(
                    PredictionRecord(
                        sample_id=sample_id,
                        split=split,
                        condition=condition,
                        true_label=true_label,
                        true_label_name=FER2013_LABEL_NAMES[true_label],
                        predicted_label=predicted_label,
                        predicted_label_name=(
                            FER2013_LABEL_NAMES[predicted_label]
                        ),
                        correct=true_label == predicted_label,
                        predicted_confidence=confidence,
                        probabilities=row_probabilities,
                    )
                )

            total_loss += loss_value * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError(f"{split} loader has no samples")
    metrics = compute_classification_metrics(true_labels, predicted_labels)
    return EvaluationResult(
        average_loss=total_loss / sample_count,
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        sample_count=sample_count,
        per_class=metrics.per_class,
        confusion_matrix=metrics.confusion_matrix,
        predictions=tuple(predictions),
    )


def _validate_evaluation_request(
    split: object,
    condition: object,
    amp_enabled: object,
    device: torch.device,
) -> None:
    if type(split) is not str or split not in VALID_EVALUATION_SPLITS:
        allowed = ", ".join(VALID_EVALUATION_SPLITS)
        raise ValueError(f"split must be one of {allowed}")
    if type(condition) is not str or not condition.strip():
        raise ValueError("condition must be a non-empty string")
    if type(amp_enabled) is not bool:
        raise ValueError("amp_enabled must be a bool")
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP can only be enabled for a CUDA device")


def _validate_batch(
    images: Tensor, labels: Tensor, sample_ids: Tensor
) -> int:
    if not isinstance(images, Tensor) or images.ndim != 4:
        raise ValueError("images must be a four-dimensional Tensor")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional Tensor")
    if not isinstance(sample_ids, Tensor) or sample_ids.ndim != 1:
        raise ValueError("sample_ids must be a one-dimensional Tensor")
    batch_size = images.shape[0]
    if labels.shape[0] != batch_size or sample_ids.shape[0] != batch_size:
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


def _autocast_context(amp_enabled: bool):
    if amp_enabled:
        return torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        )
    return nullcontext()
