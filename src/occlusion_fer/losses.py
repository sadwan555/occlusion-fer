"""Explicit training and evaluation loss construction."""

from __future__ import annotations

import math

from torch import nn

from occlusion_fer.config import LossConfig


def build_training_criterion(loss_config: LossConfig) -> nn.CrossEntropyLoss:
    """Build the configured training criterion."""
    _validate_loss_config(loss_config)
    return nn.CrossEntropyLoss(
        label_smoothing=loss_config.label_smoothing,
    )


def build_evaluation_criterion() -> nn.CrossEntropyLoss:
    """Build the unsmoothed criterion shared by validation and final test."""
    return nn.CrossEntropyLoss(label_smoothing=0.0)


def _validate_loss_config(loss_config: LossConfig) -> None:
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if loss_config.type != "cross_entropy":
        raise ValueError("loss type must be cross_entropy")
    if (
        type(loss_config.label_smoothing) not in (int, float)
        or not math.isfinite(float(loss_config.label_smoothing))
        or not 0.0 <= loss_config.label_smoothing <= 1.0
    ):
        raise ValueError("label_smoothing must be finite and between 0 and 1")
