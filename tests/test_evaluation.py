import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import occlusion_fer.evaluation as evaluation_module
from occlusion_fer.evaluation import evaluate


class LookupModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("lookup_logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        indices = images[:, 0, 0, 0].to(dtype=torch.long)
        return self.lookup_logits[indices]


def make_loader(
    *,
    labels: tuple[int, ...] = (0, 1, 2),
    sample_ids: tuple[int, ...] = (10, 11, 12),
) -> DataLoader:
    images = torch.zeros(len(labels), 3, 2, 2)
    images[:, 0, 0, 0] = torch.arange(len(labels))
    return DataLoader(
        TensorDataset(
            images,
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(sample_ids, dtype=torch.long),
        ),
        batch_size=2,
        shuffle=False,
    )


def make_model() -> LookupModel:
    return LookupModel(
        torch.tensor(
            [
                [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )


def test_evaluate_collects_paper_ready_prediction_records() -> None:
    result = evaluate(
        make_model(),
        make_loader(),
        torch.device("cpu"),
        split="validation",
        condition="clean",
    )

    assert result.sample_count == 3
    assert result.accuracy == pytest.approx(2.0 / 3.0)
    assert len(result.per_class) == 7
    assert len(result.confusion_matrix) == 7
    assert len(result.predictions) == 3
    first, second, third = result.predictions
    assert first.sample_id == 10
    assert first.split == "validation"
    assert first.condition == "clean"
    assert first.true_label == 0
    assert first.true_label_name == "angry"
    assert first.predicted_label == 0
    assert first.predicted_label_name == "angry"
    assert first.correct is True
    assert first.predicted_confidence == pytest.approx(max(first.probabilities))
    assert sum(first.probabilities) == pytest.approx(1.0)
    assert len(first.probabilities) == 7
    assert second.predicted_label == 2
    assert second.correct is False
    assert third.sample_id == 12


def test_evaluate_metrics_match_prediction_records() -> None:
    result = evaluate(make_model(), make_loader(), torch.device("cpu"))

    assert result.confusion_matrix[0][0] == 1
    assert result.confusion_matrix[1][2] == 1
    assert result.confusion_matrix[2][2] == 1
    assert result.macro_f1 == pytest.approx((1.0 + 0.0 + 2.0 / 3.0) / 7.0)
    assert np.isfinite(result.average_loss)


def test_evaluate_always_uses_unsmoothed_cross_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []
    original_builder = evaluation_module.build_evaluation_criterion

    def recording_builder() -> nn.CrossEntropyLoss:
        criterion = original_builder()
        observed.append(float(criterion.label_smoothing))
        return criterion

    monkeypatch.setattr(
        evaluation_module, "build_evaluation_criterion", recording_builder
    )
    model = make_model()
    loader = make_loader()
    result = evaluate(model, loader, torch.device("cpu"))
    expected = torch.nn.functional.cross_entropy(
        model.lookup_logits,
        torch.tensor([0, 1, 2]),
        label_smoothing=0.0,
    )

    assert observed == [0.0]
    assert result.average_loss == pytest.approx(float(expected.item()))


def test_evaluate_rejects_duplicate_sample_ids() -> None:
    with pytest.raises(ValueError, match=r"duplicate sample ID.*10"):
        evaluate(
            make_model(),
            make_loader(sample_ids=(10, 10, 12)),
            torch.device("cpu"),
        )


def test_evaluate_rejects_non_seven_class_logits() -> None:
    model = LookupModel(torch.zeros(3, 6))

    with pytest.raises(ValueError, match=r"exactly 7 classes"):
        evaluate(model, make_loader(), torch.device("cpu"))


def test_evaluate_rejects_non_finite_logits_before_writing_predictions() -> None:
    logits = torch.zeros(3, 7)
    logits[0, 0] = float("inf")

    with pytest.raises(ValueError, match=r"loss.*finite"):
        evaluate(LookupModel(logits), make_loader(), torch.device("cpu"))


@pytest.mark.parametrize("split", ["", "train", "PublicTest", 1])
def test_evaluate_rejects_invalid_split(split: object) -> None:
    with pytest.raises(ValueError, match=r"split.*validation.*test"):
        evaluate(
            make_model(),
            make_loader(),
            torch.device("cpu"),
            split=split,
        )


@pytest.mark.parametrize("condition", ["", "  ", 1, None])
def test_evaluate_rejects_invalid_condition(condition: object) -> None:
    with pytest.raises(ValueError, match=r"condition.*non-empty string"):
        evaluate(
            make_model(),
            make_loader(),
            torch.device("cpu"),
            condition=condition,
        )
