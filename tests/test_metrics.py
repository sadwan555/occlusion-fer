import numpy as np
import pytest

from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.metrics import compute_classification_metrics


def test_compute_metrics_uses_true_rows_and_predicted_columns() -> None:
    result = compute_classification_metrics([0, 0, 1, 2], [0, 1, 1, 1])

    assert result.confusion_matrix[0][0] == 1
    assert result.confusion_matrix[0][1] == 1
    assert result.confusion_matrix[1][1] == 1
    assert result.confusion_matrix[2][1] == 1
    assert result.accuracy == pytest.approx(0.5)
    assert result.sample_count == 4


def test_compute_metrics_returns_ordered_per_class_values() -> None:
    result = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    )

    angry = result.per_class[0]
    disgust = result.per_class[1]
    assert angry.label == 0
    assert angry.label_name == "angry"
    assert angry.precision == pytest.approx(1.0)
    assert angry.recall == pytest.approx(0.5)
    assert angry.f1 == pytest.approx(2.0 / 3.0)
    assert angry.support == 2
    assert disgust.label == 1
    assert disgust.label_name == "disgust"
    assert disgust.precision == pytest.approx(2.0 / 3.0)
    assert disgust.recall == pytest.approx(1.0)
    assert disgust.f1 == pytest.approx(0.8)
    assert disgust.support == 2


def test_compute_metrics_macro_f1_always_averages_all_seven_classes() -> None:
    result = compute_classification_metrics([0, 0], [0, 0])

    assert len(result.per_class) == 7
    assert result.per_class[0].f1 == pytest.approx(1.0)
    assert all(item.f1 == 0.0 for item in result.per_class[1:])
    assert result.macro_f1 == pytest.approx(1.0 / 7.0)


def test_compute_metrics_outputs_are_finite_with_zero_denominators() -> None:
    result = compute_classification_metrics([0, 1, 2], [0, 0, 0])

    values = [result.accuracy, result.macro_f1]
    for item in result.per_class:
        values.extend([item.precision, item.recall, item.f1])
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    ("true_labels", "predicted_labels", "message"),
    [
        ([], [], "must not be empty"),
        ([0], [], "same number"),
        ([0, 7], [0, 1], "between 0 and 6"),
        ([0, 1], [0, -1], "between 0 and 6"),
        ([0, True], [0, 1], "integers"),
        ([0, 1], [0, 1.0], "integers"),
    ],
)
def test_compute_metrics_rejects_invalid_labels(
    true_labels: list[object],
    predicted_labels: list[object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_classification_metrics(true_labels, predicted_labels)


def test_compute_metrics_rejects_invalid_num_classes() -> None:
    with pytest.raises(ValueError, match=r"num_classes.*positive integer"):
        compute_classification_metrics([0], [0], num_classes=True)


def test_compute_metrics_rejects_wrong_label_name_count() -> None:
    with pytest.raises(ValueError, match=r"label_names.*num_classes"):
        compute_classification_metrics(
            [0],
            [0],
            label_names=FER2013_LABEL_NAMES[:-1],
        )
