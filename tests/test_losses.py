import pytest
import torch
from torch.nn import functional as F

from occlusion_fer.config import LossConfig
from occlusion_fer.losses import (
    build_evaluation_criterion,
    build_training_criterion,
)


def fixed_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0, 0.0, 0.25, -0.5, 1.0],
            [-0.5, 2.5, 0.0, 0.75, -1.0, 0.5, 0.25],
            [0.0, -0.5, 1.5, 0.5, 0.25, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 5], dtype=torch.long)
    return logits, targets


def test_training_criterion_matches_locked_label_smoothing() -> None:
    logits, targets = fixed_inputs()
    criterion = build_training_criterion(
        LossConfig(type="cross_entropy", label_smoothing=0.1)
    )

    actual = criterion(logits, targets)
    expected = F.cross_entropy(
        logits, targets, label_smoothing=0.1
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_evaluation_criterion_is_unsmoothed_cross_entropy() -> None:
    logits, targets = fixed_inputs()
    criterion = build_evaluation_criterion()

    actual = criterion(logits, targets)
    expected = F.cross_entropy(
        logits, targets, label_smoothing=0.0
    )

    assert criterion.label_smoothing == pytest.approx(0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_zero_smoothing_matches_legacy_loss_gradient_and_rng() -> None:
    logits, targets = fixed_inputs()
    new_logits = logits.clone().requires_grad_(True)
    legacy_logits = logits.clone().requires_grad_(True)
    torch.manual_seed(1234)
    before = torch.get_rng_state()

    new_loss = build_training_criterion(
        LossConfig(type="cross_entropy", label_smoothing=0.0)
    )(new_logits, targets)
    new_loss.backward()
    after = torch.get_rng_state()
    legacy_loss = F.cross_entropy(legacy_logits, targets)
    legacy_loss.backward()

    torch.testing.assert_close(new_loss, legacy_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        new_logits.grad, legacy_logits.grad, rtol=0.0, atol=0.0
    )
    assert torch.equal(before, after)
