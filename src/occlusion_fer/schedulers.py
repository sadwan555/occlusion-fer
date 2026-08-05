"""Explicit epoch-level learning-rate schedules."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from torch.optim import Optimizer

from occlusion_fer.config import SchedulerConfig


def learning_rate_for_epoch(
    schedule: SchedulerConfig,
    *,
    base_learning_rate: float,
    total_epochs: int,
    epoch: int,
) -> float:
    """Return the learning rate used for every update in a 1-based epoch."""
    _validate_schedule_inputs(
        schedule,
        base_learning_rate=base_learning_rate,
        total_epochs=total_epochs,
        epoch=epoch,
    )
    if schedule.type == "none":
        return float(base_learning_rate)

    if schedule.warmup_epochs > 0 and epoch <= schedule.warmup_epochs:
        if schedule.warmup_epochs == 1:
            return float(base_learning_rate)
        progress = (epoch - 1) / (schedule.warmup_epochs - 1)
        factor = schedule.warmup_start_factor + progress * (
            1.0 - schedule.warmup_start_factor
        )
        return float(base_learning_rate * factor)

    decay_epochs = total_epochs - schedule.warmup_epochs
    if schedule.warmup_epochs == 0:
        if total_epochs == 1:
            return float(base_learning_rate)
        progress = (epoch - 1) / (total_epochs - 1)
    else:
        progress = (epoch - schedule.warmup_epochs) / decay_epochs
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        schedule.min_learning_rate
        + (base_learning_rate - schedule.min_learning_rate) * cosine
    )


class EpochLearningRateScheduler:
    """Set one deterministic LR per epoch while preserving group LR ratios."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedule: SchedulerConfig,
        *,
        base_learning_rate: float,
        total_epochs: int,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain at least one parameter group")
        if type(base_learning_rate) not in (int, float) or base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be greater than 0")
        self.optimizer = optimizer
        self.schedule = schedule
        self.base_learning_rate = float(base_learning_rate)
        self.total_epochs = total_epochs
        self.base_lrs = tuple(
            float(group["lr"]) for group in optimizer.param_groups
        )
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> tuple[float, ...]:
        """Set and return the learning rates used by the requested epoch."""
        primary_lr = learning_rate_for_epoch(
            self.schedule,
            base_learning_rate=self.base_learning_rate,
            total_epochs=self.total_epochs,
            epoch=epoch,
        )
        multiplier = primary_lr / self.base_learning_rate
        learning_rates = tuple(base_lr * multiplier for base_lr in self.base_lrs)
        for group, learning_rate in zip(
            self.optimizer.param_groups, learning_rates, strict=True
        ):
            group["lr"] = learning_rate
        self.current_epoch = epoch
        return learning_rates

    def state_dict(self) -> dict[str, Any]:
        """Return sufficient state for checkpoint provenance."""
        return {
            "schedule": asdict(self.schedule),
            "base_learning_rate": self.base_learning_rate,
            "total_epochs": self.total_epochs,
            "base_lrs": list(self.base_lrs),
            "current_epoch": self.current_epoch,
        }


def _validate_schedule_inputs(
    schedule: SchedulerConfig,
    *,
    base_learning_rate: float,
    total_epochs: int,
    epoch: int,
) -> None:
    if not isinstance(schedule, SchedulerConfig):
        raise TypeError("schedule must be a SchedulerConfig")
    if schedule.type not in {"none", "warmup_cosine"}:
        raise ValueError("scheduler type must be none or warmup_cosine")
    if type(base_learning_rate) not in (int, float) or base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be greater than 0")
    if type(total_epochs) is not int or total_epochs <= 0:
        raise ValueError("total_epochs must be a positive integer")
    if type(epoch) is not int or not 1 <= epoch <= total_epochs:
        raise ValueError(f"epoch must be between 1 and {total_epochs}")
    if (
        type(schedule.warmup_epochs) is not int
        or not 0 <= schedule.warmup_epochs < total_epochs
    ):
        raise ValueError("warmup_epochs must be between 0 and total_epochs - 1")
    if not 0 < schedule.warmup_start_factor <= 1:
        raise ValueError("warmup_start_factor must be greater than 0 and at most 1")
    if not 0 <= schedule.min_learning_rate <= base_learning_rate:
        raise ValueError(
            "min_learning_rate must be between 0 and base_learning_rate"
        )
