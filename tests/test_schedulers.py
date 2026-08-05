import math

import pytest
import torch

from occlusion_fer.config import SchedulerConfig
from occlusion_fer.schedulers import (
    EpochLearningRateScheduler,
    learning_rate_for_epoch,
)


BASE_LR = 1e-4
MIN_LR = 1e-6
TOTAL_EPOCHS = 30
E1_SCHEDULE = SchedulerConfig(
    type="warmup_cosine",
    warmup_epochs=3,
    warmup_start_factor=0.1,
    min_learning_rate=MIN_LR,
)


def e1_learning_rate(epoch: int) -> float:
    return learning_rate_for_epoch(
        E1_SCHEDULE,
        base_learning_rate=BASE_LR,
        total_epochs=TOTAL_EPOCHS,
        epoch=epoch,
    )


def test_e1_learning_rate_key_epochs_have_locked_values() -> None:
    expected_epoch_4 = MIN_LR + (BASE_LR - MIN_LR) * 0.5 * (
        1.0 + math.cos(math.pi / 27.0)
    )
    expected_epoch_15 = MIN_LR + (BASE_LR - MIN_LR) * 0.5 * (
        1.0 + math.cos(math.pi * 12.0 / 27.0)
    )

    assert e1_learning_rate(1) == pytest.approx(1e-5)
    assert e1_learning_rate(2) == pytest.approx(5.5e-5)
    assert e1_learning_rate(3) == pytest.approx(1e-4)
    assert e1_learning_rate(4) == pytest.approx(expected_epoch_4)
    assert e1_learning_rate(15) == pytest.approx(expected_epoch_15)
    assert e1_learning_rate(30) == pytest.approx(1e-6)


def test_e1_learning_rate_is_monotonic_in_each_phase() -> None:
    warmup = [e1_learning_rate(epoch) for epoch in range(1, 4)]
    cosine = [e1_learning_rate(epoch) for epoch in range(4, 31)]

    assert warmup == sorted(warmup)
    assert cosine == sorted(cosine, reverse=True)


def test_none_schedule_keeps_base_learning_rate_for_every_epoch() -> None:
    schedule = SchedulerConfig(
        type="none",
        warmup_epochs=0,
        warmup_start_factor=0.1,
        min_learning_rate=0.0,
    )

    values = [
        learning_rate_for_epoch(
            schedule,
            base_learning_rate=BASE_LR,
            total_epochs=TOTAL_EPOCHS,
            epoch=epoch,
        )
        for epoch in range(1, TOTAL_EPOCHS + 1)
    ]

    assert values == [BASE_LR] * TOTAL_EPOCHS


def test_epoch_scheduler_sets_current_epoch_lr_and_preserves_group_ratios() -> None:
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [first], "lr": BASE_LR},
            {"params": [second], "lr": BASE_LR * 0.25},
        ]
    )
    scheduler = EpochLearningRateScheduler(
        optimizer,
        E1_SCHEDULE,
        base_learning_rate=BASE_LR,
        total_epochs=TOTAL_EPOCHS,
    )

    learning_rates = scheduler.set_epoch(1)

    assert learning_rates == pytest.approx((1e-5, 2.5e-6))
    assert tuple(group["lr"] for group in optimizer.param_groups) == pytest.approx(
        learning_rates
    )
    assert scheduler.state_dict()["current_epoch"] == 1


@pytest.mark.parametrize("epoch", [0, 31])
def test_learning_rate_rejects_epoch_outside_budget(epoch: int) -> None:
    with pytest.raises(ValueError, match=r"epoch.*1.*30"):
        e1_learning_rate(epoch)
