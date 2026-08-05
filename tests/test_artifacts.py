import csv
import json
from pathlib import Path

import pytest
import yaml

from occlusion_fer.artifacts import (
    collect_run_metadata,
    write_csv_atomic,
    write_evaluation_artifacts,
    write_failure_artifact,
    write_history_artifacts,
    write_json_atomic,
    write_resolved_config,
)
from occlusion_fer.evaluation import EvaluationResult, PredictionRecord
from occlusion_fer.metrics import compute_classification_metrics


def make_evaluation_result() -> EvaluationResult:
    true_labels = [0, 1, 2]
    predicted_labels = [0, 2, 2]
    metrics = compute_classification_metrics(true_labels, predicted_labels)
    probabilities = (
        (0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0),
        (0.1, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0),
    )
    predictions = tuple(
        PredictionRecord(
            sample_id=100 + index,
            split="validation",
            condition="clean",
            true_label=true_label,
            true_label_name=("angry", "disgust", "fear")[true_label],
            predicted_label=predicted_label,
            predicted_label_name=("angry", "disgust", "fear")[
                predicted_label
            ],
            correct=true_label == predicted_label,
            predicted_confidence=max(probabilities[index]),
            probabilities=probabilities[index],
        )
        for index, (true_label, predicted_label) in enumerate(
            zip(true_labels, predicted_labels, strict=True)
        )
    )
    return EvaluationResult(
        average_loss=0.75,
        accuracy=metrics.accuracy,
        macro_f1=metrics.macro_f1,
        sample_count=metrics.sample_count,
        per_class=metrics.per_class,
        confusion_matrix=metrics.confusion_matrix,
        predictions=predictions,
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def test_atomic_json_and_csv_writers_replace_complete_targets(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "nested" / "result.json"
    csv_path = tmp_path / "nested" / "result.csv"
    json_path.parent.mkdir(parents=True)
    json_path.write_text("old", encoding="utf-8")
    csv_path.write_text("old", encoding="utf-8")

    resolved_json = write_json_atomic(json_path, {"value": 2})
    resolved_csv = write_csv_atomic(
        csv_path,
        ("name", "value"),
        ({"name": "macro_f1", "value": 0.5},),
    )

    assert resolved_json == json_path.resolve()
    assert resolved_csv == csv_path.resolve()
    assert json_path.read_text(encoding="utf-8") == '{\n  "value": 2\n}\n'
    assert csv_path.read_text(encoding="utf-8") == (
        "name,value\nmacro_f1,0.5\n"
    )
    assert not list(json_path.parent.glob("*.tmp"))


def test_write_evaluation_artifacts_uses_stable_paper_schema(
    tmp_path: Path,
) -> None:
    paths = write_evaluation_artifacts(
        tmp_path,
        "validation/best",
        make_evaluation_result(),
    )

    assert set(paths) == {
        "metrics",
        "per_class_metrics",
        "confusion_matrix",
        "predictions",
    }
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["metric_schema_version"] == 1
    assert metrics["split"] == "validation"
    assert metrics["condition"] == "clean"
    assert metrics["sample_count"] == 3
    assert metrics["loss"] == pytest.approx(0.75)
    assert metrics["accuracy"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_f1"] == pytest.approx(
        make_evaluation_result().macro_f1
    )
    assert [item["label_name"] for item in metrics["label_order"]] == [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]
    assert len(metrics["per_class"]) == 7
    assert len(metrics["confusion_matrix"]) == 7

    per_class_fields, per_class_rows = read_csv(paths["per_class_metrics"])
    assert per_class_fields == [
        "label",
        "label_name",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    assert [row["label_name"] for row in per_class_rows] == [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    confusion_fields, confusion_rows = read_csv(paths["confusion_matrix"])
    assert confusion_fields == [
        "true_label",
        "true_label_name",
        "predicted_angry",
        "predicted_disgust",
        "predicted_fear",
        "predicted_happy",
        "predicted_sad",
        "predicted_surprise",
        "predicted_neutral",
    ]
    assert confusion_rows[1]["predicted_fear"] == "1"

    prediction_fields, prediction_rows = read_csv(paths["predictions"])
    assert prediction_fields[:9] == [
        "sample_id",
        "split",
        "condition",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "correct",
        "predicted_confidence",
    ]
    assert prediction_fields[9:] == [
        "probability_angry",
        "probability_disgust",
        "probability_fear",
        "probability_happy",
        "probability_sad",
        "probability_surprise",
        "probability_neutral",
    ]
    assert [row["sample_id"] for row in prediction_rows] == [
        "100",
        "101",
        "102",
    ]


def test_write_history_artifacts_keeps_json_and_csv_consistent(
    tmp_path: Path,
) -> None:
    history = [
        {
            "epoch": 1,
            "train_loss": 1.2,
            "validation_accuracy": 0.5,
            "validation_macro_f1": 0.4,
        },
        {
            "epoch": 2,
            "train_loss": 0.9,
            "validation_accuracy": 0.6,
            "validation_macro_f1": 0.55,
        },
    ]

    json_path, csv_path = write_history_artifacts(tmp_path, history)

    assert json.loads(json_path.read_text(encoding="utf-8")) == history
    fields, rows = read_csv(csv_path)
    assert fields == list(history[0])
    assert rows[1]["validation_macro_f1"] == "0.55"


def test_write_history_artifacts_persists_train_accuracy_and_learning_rate(
    tmp_path: Path,
) -> None:
    history = [
        {
            "epoch": 1,
            "train_samples": 5,
            "train_loss": 1.25,
            "train_accuracy": 0.6,
            "learning_rate": 0.00001,
            "validation_samples": 2,
            "validation_loss": 1.1,
            "validation_accuracy": 0.5,
            "validation_macro_f1": 0.4,
        }
    ]

    json_path, csv_path = write_history_artifacts(tmp_path, history)

    assert json.loads(json_path.read_text(encoding="utf-8")) == history
    fields, rows = read_csv(csv_path)
    assert fields == list(history[0])
    assert rows[0]["train_accuracy"] == "0.6"
    assert rows[0]["learning_rate"] == "1e-05"


def test_write_resolved_config_outputs_yaml(tmp_path: Path) -> None:
    path = write_resolved_config(
        tmp_path,
        {
            "training": {"seed": 42, "batch_size": 128},
            "runtime": {"amp_enabled": True},
        },
    )

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["training"]["seed"] == 42
    assert loaded["runtime"]["amp_enabled"] is True


def test_collect_run_metadata_records_git_and_runtime_without_secrets(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "history.json"
    metadata = collect_run_metadata(
        status="completed",
        started_at_utc="2026-08-03T12:00:00Z",
        finished_at_utc="2026-08-03T12:01:00Z",
        seed=42,
        training_mode="clean",
        requested_device="auto",
        selected_device="cpu",
        amp_enabled=False,
        repository_root=Path.cwd(),
        output_directory=tmp_path,
        artifact_paths={"history": artifact_path},
    )

    assert metadata["run_metadata_schema_version"] == 1
    assert metadata["status"] == "completed"
    assert metadata["seed"] == 42
    assert len(metadata["git_commit"]) == 40
    assert type(metadata["git_dirty"]) is bool
    assert metadata["artifacts"] == {"history": "history.json"}
    serialized = json.dumps(metadata).lower()
    assert "token" not in serialized
    assert "password" not in serialized


def test_failure_artifact_is_minimal_and_does_not_store_traceback(
    tmp_path: Path,
) -> None:
    path = write_failure_artifact(
        tmp_path,
        stage="training",
        exception=RuntimeError("synthetic failure"),
        timestamp_utc="2026-08-03T12:00:00Z",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "failure_schema_version": 1,
        "status": "failed",
        "stage": "training",
        "exception_type": "RuntimeError",
        "message": "synthetic failure",
        "timestamp_utc": "2026-08-03T12:00:00Z",
    }
    assert "traceback" not in payload


def test_write_history_rejects_empty_or_inconsistent_rows(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"history.*not be empty"):
        write_history_artifacts(tmp_path, [])
    with pytest.raises(ValueError, match=r"same fields"):
        write_history_artifacts(tmp_path, [{"epoch": 1}, {"epoch": 2, "x": 3}])
