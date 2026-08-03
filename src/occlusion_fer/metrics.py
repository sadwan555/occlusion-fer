"""Deterministic classification metrics for FER2013 experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from occlusion_fer.data import FER2013_LABEL_NAMES


@dataclass(frozen=True)
class PerClassMetrics:
    """Precision, recall, F1, and support for one ordered class."""

    label: int
    label_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate and classwise metrics for one classification condition."""

    accuracy: float
    macro_f1: float
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    sample_count: int


def compute_classification_metrics(
    true_labels: Sequence[object],
    predicted_labels: Sequence[object],
    *,
    num_classes: int = 7,
    label_names: Sequence[str] = FER2013_LABEL_NAMES,
) -> ClassificationMetrics:
    """Compute fixed-order metrics with true rows and predicted columns."""
    if type(num_classes) is not int or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if len(label_names) != num_classes:
        raise ValueError("label_names length must match num_classes")
    if any(type(name) is not str or not name for name in label_names):
        raise ValueError("label_names must contain non-empty strings")
    if len(true_labels) == 0:
        raise ValueError("classification labels must not be empty")
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true and predicted labels must contain the same number")

    true_array = _validated_label_array(
        true_labels, "true_labels", num_classes
    )
    predicted_array = _validated_label_array(
        predicted_labels, "predicted_labels", num_classes
    )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (true_array, predicted_array), 1)

    true_positives = np.diag(confusion).astype(np.float64)
    predicted_counts = confusion.sum(axis=0, dtype=np.int64)
    supports = confusion.sum(axis=1, dtype=np.int64)
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positives,
        supports,
        out=np.zeros(num_classes, dtype=np.float64),
        where=supports != 0,
    )
    precision_recall_sum = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        precision_recall_sum,
        out=np.zeros(num_classes, dtype=np.float64),
        where=precision_recall_sum != 0.0,
    )

    accuracy = float(true_positives.sum() / len(true_array))
    macro_f1 = float(f1.mean())
    values = np.concatenate(([accuracy, macro_f1], precision, recall, f1))
    if not np.isfinite(values).all():
        raise ValueError("classification metrics must be finite")

    per_class = tuple(
        PerClassMetrics(
            label=label,
            label_name=label_names[label],
            precision=float(precision[label]),
            recall=float(recall[label]),
            f1=float(f1[label]),
            support=int(supports[label]),
        )
        for label in range(num_classes)
    )
    confusion_tuple = tuple(
        tuple(int(value) for value in row) for row in confusion.tolist()
    )
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=confusion_tuple,
        sample_count=len(true_array),
    )


def _validated_label_array(
    labels: Sequence[object], field_name: str, num_classes: int
) -> NDArray[np.int64]:
    normalized: list[int] = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError(f"{field_name} must contain integers")
        normalized_label = int(label)
        if not 0 <= normalized_label < num_classes:
            upper = num_classes - 1
            raise ValueError(
                f"{field_name} values must be between 0 and {upper}"
            )
        normalized.append(normalized_label)
    return np.asarray(normalized, dtype=np.int64)
