"""Atomic, reproducible experiment artifacts for training and evaluation."""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml

from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.evaluation import EvaluationResult


METRIC_SCHEMA_VERSION = 1
RUN_METADATA_SCHEMA_VERSION = 1
FAILURE_SCHEMA_VERSION = 1


def write_json_atomic(path: str | Path, payload: object) -> Path:
    """Write newline-terminated JSON without exposing partial targets."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return _write_text_atomic(path, text + "\n")


def write_csv_atomic(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    """Write a deterministic UTF-8 CSV through an atomic replacement."""
    if not fieldnames or any(
        type(field) is not str or not field for field in fieldnames
    ):
        raise ValueError("CSV fieldnames must contain non-empty strings")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return _write_text_atomic(path, buffer.getvalue())


def write_resolved_config(
    output_directory: str | Path,
    resolved_config: Mapping[str, object],
) -> Path:
    """Write the post-override run configuration as portable YAML."""
    text = yaml.safe_dump(
        dict(resolved_config),
        sort_keys=False,
        allow_unicode=True,
    )
    return _write_text_atomic(
        Path(output_directory).expanduser() / "resolved_config.yaml",
        text,
    )


def write_history_artifacts(
    output_directory: str | Path,
    history: Sequence[Mapping[str, object]],
) -> tuple[Path, Path]:
    """Write equivalent JSON and CSV training histories."""
    if not history:
        raise ValueError("history must not be empty")
    normalized = [dict(row) for row in history]
    fieldnames = tuple(normalized[0])
    if not fieldnames:
        raise ValueError("history rows must not be empty")
    if any(tuple(row) != fieldnames for row in normalized[1:]):
        raise ValueError("history rows must contain the same fields in order")
    output_path = Path(output_directory).expanduser()
    json_path = write_json_atomic(output_path / "history.json", normalized)
    csv_path = write_csv_atomic(
        output_path / "history.csv", fieldnames, normalized
    )
    return json_path, csv_path


def write_evaluation_artifacts(
    output_directory: str | Path,
    prefix: str | Path,
    result: EvaluationResult,
) -> dict[str, Path]:
    """Write metrics, classwise values, matrix, and predictions for one result."""
    if not result.predictions:
        raise ValueError("evaluation predictions must not be empty")
    split = result.predictions[0].split
    condition = result.predictions[0].condition
    if any(
        prediction.split != split or prediction.condition != condition
        for prediction in result.predictions
    ):
        raise ValueError(
            "evaluation predictions must share one split and condition"
        )
    if len(result.per_class) != len(FER2013_LABEL_NAMES):
        raise ValueError("evaluation must contain metrics for exactly 7 classes")
    if len(result.confusion_matrix) != len(FER2013_LABEL_NAMES) or any(
        len(row) != len(FER2013_LABEL_NAMES)
        for row in result.confusion_matrix
    ):
        raise ValueError("evaluation confusion matrix must have shape 7 by 7")

    base_path = Path(output_directory).expanduser() / Path(prefix)
    metrics_path = base_path.parent / f"{base_path.name}_metrics.json"
    per_class_path = (
        base_path.parent / f"{base_path.name}_per_class_metrics.csv"
    )
    confusion_path = (
        base_path.parent / f"{base_path.name}_confusion_matrix.csv"
    )
    predictions_path = (
        base_path.parent / f"{base_path.name}_predictions.csv"
    )

    label_order = [
        {"label": label, "label_name": label_name}
        for label, label_name in enumerate(FER2013_LABEL_NAMES)
    ]
    metrics_payload = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "split": split,
        "condition": condition,
        "sample_count": result.sample_count,
        "loss": result.average_loss,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "label_order": label_order,
        "per_class": [asdict(item) for item in result.per_class],
        "confusion_matrix": [list(row) for row in result.confusion_matrix],
    }
    resolved_metrics = write_json_atomic(metrics_path, metrics_payload)

    per_class_fields = (
        "label",
        "label_name",
        "precision",
        "recall",
        "f1",
        "support",
    )
    resolved_per_class = write_csv_atomic(
        per_class_path,
        per_class_fields,
        (asdict(item) for item in result.per_class),
    )

    predicted_fields = tuple(
        f"predicted_{label_name}" for label_name in FER2013_LABEL_NAMES
    )
    confusion_fields = (
        "true_label",
        "true_label_name",
        *predicted_fields,
    )
    confusion_rows = []
    for true_label, row in enumerate(result.confusion_matrix):
        csv_row: dict[str, object] = {
            "true_label": true_label,
            "true_label_name": FER2013_LABEL_NAMES[true_label],
        }
        csv_row.update(dict(zip(predicted_fields, row, strict=True)))
        confusion_rows.append(csv_row)
    resolved_confusion = write_csv_atomic(
        confusion_path,
        confusion_fields,
        confusion_rows,
    )

    base_prediction_fields = (
        "sample_id",
        "split",
        "condition",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "correct",
        "predicted_confidence",
    )
    probability_fields = tuple(
        f"probability_{label_name}" for label_name in FER2013_LABEL_NAMES
    )
    prediction_rows: list[dict[str, object]] = []
    for prediction in result.predictions:
        if len(prediction.probabilities) != len(FER2013_LABEL_NAMES):
            raise ValueError(
                "prediction probabilities must contain exactly 7 values"
            )
        row = {
            "sample_id": prediction.sample_id,
            "split": prediction.split,
            "condition": prediction.condition,
            "true_label": prediction.true_label,
            "true_label_name": prediction.true_label_name,
            "predicted_label": prediction.predicted_label,
            "predicted_label_name": prediction.predicted_label_name,
            "correct": prediction.correct,
            "predicted_confidence": prediction.predicted_confidence,
        }
        row.update(
            dict(zip(probability_fields, prediction.probabilities, strict=True))
        )
        prediction_rows.append(row)
    resolved_predictions = write_csv_atomic(
        predictions_path,
        (*base_prediction_fields, *probability_fields),
        prediction_rows,
    )
    return {
        "metrics": resolved_metrics,
        "per_class_metrics": resolved_per_class,
        "confusion_matrix": resolved_confusion,
        "predictions": resolved_predictions,
    }


def write_evaluation_artifacts_to_directory(
    directory: str | Path,
    result: EvaluationResult,
) -> dict[str, Path]:
    """Write one private-final condition with exact, unprefixed filenames."""
    target = Path(directory).expanduser()
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(
                f"evaluation condition directory is not empty: {target}"
            )
    else:
        target.mkdir(parents=True, exist_ok=False)
    staged = write_evaluation_artifacts(target, "evaluation", result)
    names = {
        "metrics": "metrics.json",
        "per_class_metrics": "per_class_metrics.csv",
        "confusion_matrix": "confusion_matrix.csv",
        "predictions": "predictions.csv",
    }
    resolved: dict[str, Path] = {}
    for artifact_name, source in staged.items():
        destination = target / names[artifact_name]
        if destination.exists():
            raise FileExistsError(
                f"evaluation artifact already exists: {destination}"
            )
        source.rename(destination)
        resolved[artifact_name] = destination
    return resolved


def collect_run_metadata(
    *,
    status: str,
    started_at_utc: str,
    finished_at_utc: str | None,
    seed: int,
    training_mode: str,
    requested_device: str,
    selected_device: str,
    amp_enabled: bool,
    repository_root: str | Path,
    output_directory: str | Path,
    artifact_paths: Mapping[str, str | Path],
    cuda_device_name: str | None = None,
) -> dict[str, Any]:
    """Collect bounded provenance without credentials or environment dumps."""
    if status not in {"running", "completed", "failed"}:
        raise ValueError("run status must be running, completed, or failed")
    repository_path = Path(repository_root).expanduser().resolve()
    output_path = Path(output_directory).expanduser().resolve()
    commit = _run_git(repository_path, "rev-parse", "HEAD")
    dirty_output = _run_git(repository_path, "status", "--porcelain")
    relative_artifacts: dict[str, str] = {}
    for name, artifact_path in artifact_paths.items():
        resolved_artifact = Path(artifact_path).expanduser().resolve()
        try:
            relative = resolved_artifact.relative_to(output_path)
        except ValueError as exc:
            raise ValueError(
                "artifact paths must be inside the output directory"
            ) from exc
        relative_artifacts[name] = relative.as_posix()

    return {
        "run_metadata_schema_version": RUN_METADATA_SCHEMA_VERSION,
        "status": status,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "seed": seed,
        "training_mode": training_mode,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "cuda_device_name": cuda_device_name,
        "amp_enabled": amp_enabled,
        "git_commit": commit,
        "git_dirty": bool(dirty_output),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "executable": Path(sys.executable).name,
        "artifacts": relative_artifacts,
    }


def write_failure_artifact(
    output_directory: str | Path,
    *,
    stage: str,
    exception: BaseException,
    timestamp_utc: str | None = None,
) -> Path:
    """Write a minimal failure record without traceback or environment data."""
    if type(stage) is not str or not stage.strip():
        raise ValueError("failure stage must be a non-empty string")
    payload = {
        "failure_schema_version": FAILURE_SCHEMA_VERSION,
        "status": "failed",
        "stage": stage,
        "exception_type": type(exception).__name__,
        "message": str(exception),
        "timestamp_utc": timestamp_utc or utc_now(),
    }
    return write_json_atomic(
        Path(output_directory).expanduser() / "failure.json", payload
    )


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_text_atomic(path: str | Path, text: str) -> Path:
    target = Path(path).expanduser()
    temporary_path: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(target)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not write artifact: {target}") from exc
    return target.resolve()


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not collect Git metadata from: {repository_root}"
        ) from exc
    return completed.stdout.strip()
