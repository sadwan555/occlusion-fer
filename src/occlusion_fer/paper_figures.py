"""Reproducible figures for the formal FER2013 clean-only baseline."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "occlusion-fer-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

from occlusion_fer.data import FER2013_LABEL_NAMES


EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_FORMAL_COMMIT = "da889bdd818cab6403767f2f6c7d5391d8317324"
EXPECTED_OCCLUSION_EVALUATION_COMMIT = "c1c9187aa2ddf7dd84906c7f139ad9a750ef202d"
EXPECTED_OCCLUSION_TRAINING_COMMIT = "4cb1e0ffe4b55efc090a45cfed560b28f50b9509"
EXPECTED_TRAIN_SAMPLES = 28709
EXPECTED_VALIDATION_SAMPLES = 3589
EXPECTED_PUBLICTEST_SUPPORT = (467, 56, 496, 895, 653, 415, 607)
LABEL_NAMES = tuple(FER2013_LABEL_NAMES)
EXPECTED_BEST_EPOCHS = {42: 30, 123: 30, 2026: 23}
CLASSWISE_CONDITIONS = (
    "clean",
    "upper_face_0.20",
    "upper_face_0.30",
    "upper_face_0.40",
    "lower_face_0.20",
    "lower_face_0.30",
    "lower_face_0.40",
    "random_rectangle_0.20",
    "random_rectangle_0.30",
    "random_rectangle_0.40",
)
CLASSWISE_OCCLUDED_CONDITIONS = CLASSWISE_CONDITIONS[1:]

HISTORY_REQUIRED_COLUMNS = (
    "epoch",
    "train_samples",
    "train_loss",
    "validation_samples",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "train_seconds",
    "validation_seconds",
    "epoch_seconds",
    "train_samples_per_second",
    "cuda_peak_memory_bytes",
    "updated_best_checkpoint",
)
PER_CLASS_REQUIRED_COLUMNS = (
    "label",
    "label_name",
    "precision",
    "recall",
    "f1",
    "support",
)
CONFUSION_REQUIRED_COLUMNS = (
    "true_label",
    "true_label_name",
    "predicted_angry",
    "predicted_disgust",
    "predicted_fear",
    "predicted_happy",
    "predicted_sad",
    "predicted_surprise",
    "predicted_neutral",
)
NUMERIC_HISTORY_COLUMNS = tuple(
    column
    for column in HISTORY_REQUIRED_COLUMNS
    if column not in {"epoch", "updated_best_checkpoint"}
)
PREDICTED_COLUMNS = tuple(
    f"predicted_{label_name}" for label_name in LABEL_NAMES
)
FIGURE_FORMATS = ("pdf", "svg", "png")
PROTOCOL_FIGURE_STEM = "occlusion_protocol_examples"
PROTOCOL_OUTPUT_NAMES = frozenset(
    {
        f"{PROTOCOL_FIGURE_STEM}.pdf",
        f"{PROTOCOL_FIGURE_STEM}.png",
        f"{PROTOCOL_FIGURE_STEM}_metadata.json",
    }
)
FIGURE_STEMS = (
    "clean_training_validation_curves",
    "clean_per_class_f1",
    "clean_validation_confusion_matrix",
    "fer2013_training_publictest_distribution",
)
SUMMARY_TABLE_NAMES = (
    "summary_metrics",
    "per_class_f1_summary",
    "mean_normalized_confusion_matrix",
    "class_distribution",
)
EXPECTED_OUTPUT_NAMES = {
    *(f"{stem}.{extension}" for stem in FIGURE_STEMS for extension in FIGURE_FORMATS),
    *(f"{name}.csv" for name in SUMMARY_TABLE_NAMES),
    "generation_manifest.json",
}
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.0
SEED_STYLES = {
    42: {"color": "#426A8C", "linestyle": "-", "marker": "o"},
    123: {"color": "#A66A43", "linestyle": "--", "marker": "s"},
    2026: {"color": "#5A8F70", "linestyle": "-.", "marker": "^"},
}


class PaperFigureError(ValueError):
    """Raised when formal evidence cannot be validated for plotting."""


@dataclass(frozen=True)
class RunEvidence:
    seed: int
    run_dir: Path
    history: pd.DataFrame
    best_metrics: dict[str, Any]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    best_epoch: int
    input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ClasswiseEvidence:
    """Validated per-class evidence from the formal Stage 8 archives."""

    stage8_archive: Path
    baseline_archive: Path
    per_class: pd.DataFrame
    source_members: tuple[str, ...]


def _fail(message: str) -> None:
    raise PaperFigureError(message)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        _fail(f"required input file does not exist: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_require_file(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read JSON input {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON input must contain an object: {path}")
    return value


def _require_keys(mapping: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        _fail(f"{context} is missing required field(s): {', '.join(missing)}")


def _finite_numeric(values: pd.Series, context: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        _fail(f"{context} contains NaN, infinite, or non-numeric values")
    return numeric


def _integer_series(values: pd.Series, context: str) -> pd.Series:
    numeric = _finite_numeric(values, context)
    if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric)).all():
        _fail(f"{context} must contain integers")
    return numeric.astype("int64")


def _check_label_order(
    label_ids: Sequence[int], label_names: Sequence[str], context: str
) -> None:
    if list(label_ids) != list(range(len(LABEL_NAMES))):
        _fail(f"{context} label IDs must be 0 through 6 in order")
    if list(label_names) != list(LABEL_NAMES):
        _fail(
            f"{context} label order must be "
            f"{', '.join(LABEL_NAMES)}"
        )


def _check_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        _fail(f"{context} does not match its source value")


def _parse_bool_series(values: pd.Series, context: str) -> pd.Series:
    normalized = values.map(lambda value: str(value).strip().lower())
    allowed = {"true", "false"}
    if not normalized.isin(allowed).all():
        _fail(f"{context} must contain only true/false values")
    return normalized.eq("true")


def _validate_history(path: Path, seed: int) -> tuple[pd.DataFrame, int]:
    try:
        history = pd.read_csv(_require_file(path))
    except (OSError, pd.errors.ParserError) as exc:
        _fail(f"could not read history CSV {path}: {exc}")
    missing = [column for column in HISTORY_REQUIRED_COLUMNS if column not in history]
    if missing:
        _fail(f"{path} is missing history field(s): {', '.join(missing)}")
    if len(history) != 30:
        _fail(f"{path} must contain exactly 30 epochs; got {len(history)}")
    history = history.copy()
    history["epoch"] = _integer_series(history["epoch"], f"{path}: epoch")
    if history["epoch"].tolist() != list(range(1, 31)):
        _fail(f"{path} epoch values must be exactly 1 through 30")
    for column in NUMERIC_HISTORY_COLUMNS:
        history[column] = _finite_numeric(history[column], f"{path}: {column}")
    history["updated_best_checkpoint"] = _parse_bool_series(
        history["updated_best_checkpoint"], f"{path}: updated_best_checkpoint"
    )
    best_rows = history.loc[history["updated_best_checkpoint"]]
    if best_rows.empty:
        _fail(f"{path} has no updated_best_checkpoint=True row")
    best_epoch = int(best_rows.iloc[-1]["epoch"])
    expected_epoch = EXPECTED_BEST_EPOCHS[seed]
    if best_epoch != expected_epoch:
        _fail(
            f"seed {seed} best epoch is {best_epoch}, expected {expected_epoch}"
        )
    return history, best_epoch


def _validate_best_metrics(path: Path) -> dict[str, Any]:
    metrics = _read_json(path)
    _require_keys(
        metrics,
        ("condition", "split", "sample_count", "accuracy", "macro_f1", "loss"),
        str(path),
    )
    if metrics["condition"] != "clean":
        _fail(f"{path} condition must be clean")
    if metrics["split"] != "validation":
        _fail(f"{path} split must be validation/PublicTest")
    if metrics["sample_count"] != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{path} sample_count must be {EXPECTED_VALIDATION_SAMPLES}")
    for field in ("accuracy", "macro_f1", "loss"):
        value = metrics[field]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            _fail(f"{path} field {field} must be finite")
    label_order = metrics.get("label_order")
    if not isinstance(label_order, list) or len(label_order) != 7:
        _fail(f"{path} label_order must contain seven entries")
    try:
        label_ids = [int(item["label"]) for item in label_order]
        label_names = [str(item["label_name"]) for item in label_order]
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"{path} has malformed label_order: {exc}")
    _check_label_order(label_ids, label_names, str(path))
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 7:
        _fail(f"{path} confusion_matrix must be 7 by 7")
    if any(
        not isinstance(row, list) or len(row) != 7 for row in matrix
    ):
        _fail(f"{path} confusion_matrix must be 7 by 7")
    for row in matrix:
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                _fail(
                    f"{path} confusion_matrix must contain non-negative integers"
                )
    return metrics


def _validate_per_class(path: Path, metrics: dict[str, Any]) -> pd.DataFrame:
    try:
        per_class = pd.read_csv(_require_file(path))
    except (OSError, pd.errors.ParserError) as exc:
        _fail(f"could not read per-class CSV {path}: {exc}")
    missing = [column for column in PER_CLASS_REQUIRED_COLUMNS if column not in per_class]
    if missing:
        _fail(f"{path} is missing per-class field(s): {', '.join(missing)}")
    if len(per_class) != 7:
        _fail(f"{path} must contain seven classes")
    per_class = per_class.copy()
    per_class["label"] = _integer_series(per_class["label"], f"{path}: label")
    _check_label_order(
        per_class["label"].tolist(),
        per_class["label_name"].astype(str).tolist(),
        str(path),
    )
    for column in ("precision", "recall", "f1"):
        per_class[column] = _finite_numeric(per_class[column], f"{path}: {column}")
        if ((per_class[column] < 0) | (per_class[column] > 1)).any():
            _fail(f"{path}: {column} must be between 0 and 1")
    per_class["support"] = _integer_series(per_class["support"], f"{path}: support")
    if (per_class["support"] < 0).any():
        _fail(f"{path}: support cannot be negative")
    supports = tuple(int(value) for value in per_class["support"])
    if supports != EXPECTED_PUBLICTEST_SUPPORT:
        _fail(f"{path} has unexpected PublicTest support: {supports}")
    if int(per_class["support"].sum()) != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{path} support total must be {EXPECTED_VALIDATION_SAMPLES}")
    embedded = metrics.get("per_class")
    if not isinstance(embedded, list) or len(embedded) != 7:
        _fail(f"{path} cannot cross-check missing embedded per_class metrics")
    for index, row in per_class.iterrows():
        source = embedded[index]
        if not isinstance(source, dict):
            _fail(f"{path} has malformed embedded per_class metrics")
        _require_keys(
            source,
            ("label", "label_name", "precision", "recall", "f1", "support"),
            f"{path} embedded per_class entry",
        )
        if int(source["label"]) != int(row["label"]) or str(
            source["label_name"]
        ) != str(row["label_name"]):
            _fail(f"{path} label metadata disagrees with embedded per_class")
        for field in ("precision", "recall", "f1"):
            _check_close(
                float(row[field]),
                float(source[field]),
                f"{path} {field} for label {row['label_name']}",
            )
        if int(row["support"]) != int(source["support"]):
            _fail(f"{path} support disagrees with embedded per_class")
    return per_class


def _validate_confusion(
    path: Path, metrics: dict[str, Any], per_class: pd.DataFrame
) -> pd.DataFrame:
    try:
        confusion = pd.read_csv(_require_file(path))
    except (OSError, pd.errors.ParserError) as exc:
        _fail(f"could not read confusion CSV {path}: {exc}")
    missing = [column for column in CONFUSION_REQUIRED_COLUMNS if column not in confusion]
    if missing:
        _fail(f"{path} is missing confusion field(s): {', '.join(missing)}")
    if len(confusion) != 7:
        _fail(f"{path} must contain seven true-label rows")
    confusion = confusion.copy()
    confusion["true_label"] = _integer_series(
        confusion["true_label"], f"{path}: true_label"
    )
    _check_label_order(
        confusion["true_label"].tolist(),
        confusion["true_label_name"].astype(str).tolist(),
        str(path),
    )
    matrix = np.empty((7, 7), dtype=np.int64)
    for column_index, column in enumerate(PREDICTED_COLUMNS):
        values = _integer_series(confusion[column], f"{path}: {column}")
        if (values < 0).any():
            _fail(f"{path}: confusion values cannot be negative")
        matrix[:, column_index] = values.to_numpy()
    expected_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(matrix, expected_matrix):
        _fail(f"{path} does not match best_metrics.json confusion_matrix")
    row_sums = matrix.sum(axis=1)
    supports = per_class["support"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_sums, supports):
        _fail(f"{path} row sums do not match per-class support")
    if int(matrix.sum()) != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{path} confusion total must be {EXPECTED_VALIDATION_SAMPLES}")
    return confusion


def load_run_evidence(run_dir: str | Path, expected_seed: int | None = None) -> RunEvidence:
    """Load and fail closed on one formal clean-only run."""
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        _fail(f"run directory does not exist: {run_path}")
    metadata_path = _require_file(run_path / "run_metadata.json")
    config_path = _require_file(run_path / "resolved_config.yaml")
    history_path = _require_file(run_path / "history.csv")
    metrics_path = _require_file(run_path / "validation" / "best_metrics.json")
    per_class_path = _require_file(
        run_path / "validation" / "best_per_class_metrics.csv"
    )
    confusion_path = _require_file(
        run_path / "validation" / "best_confusion_matrix.csv"
    )

    metadata = _read_json(metadata_path)
    _require_keys(
        metadata,
        ("seed", "status", "training_mode", "git_commit"),
        str(metadata_path),
    )
    try:
        seed = int(metadata["seed"])
    except (TypeError, ValueError) as exc:
        _fail(f"{metadata_path} seed is invalid: {exc}")
    if expected_seed is not None and seed != expected_seed:
        _fail(f"{run_path} metadata seed {seed} does not match expected {expected_seed}")
    if seed not in EXPECTED_SEEDS:
        _fail(f"unexpected formal seed {seed}; expected {EXPECTED_SEEDS}")
    if metadata["status"] != "completed":
        _fail(f"{metadata_path} status must be completed")
    if metadata["training_mode"] != "clean":
        _fail(f"{metadata_path} training_mode must be clean")
    if metadata["git_commit"] != EXPECTED_FORMAL_COMMIT:
        _fail(f"{metadata_path} git_commit does not match the formal experiment commit")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _fail(f"could not read resolved config {config_path}: {exc}")
    if not isinstance(config, dict):
        _fail(f"resolved config must contain a mapping: {config_path}")
    training_config = config.get("training")
    if not isinstance(training_config, dict):
        _fail(f"resolved config must contain a training mapping: {config_path}")
    config_seed = training_config.get("seed")
    if config_seed != seed:
        _fail(f"{config_path} seed does not match run_metadata.json")

    history, best_epoch = _validate_history(history_path, seed)
    metrics = _validate_best_metrics(metrics_path)
    per_class = _validate_per_class(per_class_path, metrics)
    confusion = _validate_confusion(confusion_path, metrics, per_class)
    history_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    for history_field, metric_field in (
        ("validation_accuracy", "accuracy"),
        ("validation_macro_f1", "macro_f1"),
        ("validation_loss", "loss"),
    ):
        _check_close(
            float(history_row[history_field]),
            float(metrics[metric_field]),
            f"seed {seed} best metrics vs history",
        )
    return RunEvidence(
        seed=seed,
        run_dir=run_path,
        history=history,
        best_metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        best_epoch=best_epoch,
        input_paths=(
            metadata_path,
            config_path,
            history_path,
            metrics_path,
            per_class_path,
            confusion_path,
        ),
    )


def _tar_member_index(archive: tarfile.TarFile) -> dict[str, list[tarfile.TarInfo]]:
    members: dict[str, list[tarfile.TarInfo]] = {}
    for member in archive.getmembers():
        members.setdefault(member.name, []).append(member)
    return members


def _read_tar_bytes(
    archive: tarfile.TarFile,
    members: dict[str, list[tarfile.TarInfo]],
    member_name: str,
) -> bytes:
    matches = members.get(member_name, [])
    if len(matches) != 1:
        _fail(
            f"archive must contain exactly one {member_name}; found {len(matches)}"
        )
    handle = archive.extractfile(matches[0])
    if handle is None:
        _fail(f"archive member is not a readable file: {member_name}")
    try:
        return handle.read()
    except OSError as exc:
        _fail(f"could not read archive member {member_name}: {exc}")


def _read_tar_csv(
    archive: tarfile.TarFile,
    members: dict[str, list[tarfile.TarInfo]],
    member_name: str,
) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(_read_tar_bytes(archive, members, member_name)))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        _fail(f"could not read CSV archive member {member_name}: {exc}")


def _read_tar_json(
    archive: tarfile.TarFile,
    members: dict[str, list[tarfile.TarInfo]],
    member_name: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_tar_bytes(archive, members, member_name).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"could not read JSON archive member {member_name}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON archive member must contain an object: {member_name}")
    return value


def _validate_classwise_per_class(
    per_class: pd.DataFrame, context: str
) -> pd.DataFrame:
    missing = [column for column in PER_CLASS_REQUIRED_COLUMNS if column not in per_class]
    if missing:
        _fail(f"{context} is missing per-class field(s): {', '.join(missing)}")
    if len(per_class) != len(LABEL_NAMES):
        _fail(f"{context} must contain exactly seven classes")
    result = per_class.loc[:, list(PER_CLASS_REQUIRED_COLUMNS)].copy()
    result["label"] = _integer_series(result["label"], f"{context}: label")
    _check_label_order(
        result["label"].tolist(),
        result["label_name"].astype(str).tolist(),
        context,
    )
    for column in ("precision", "recall", "f1"):
        result[column] = _finite_numeric(result[column], f"{context}: {column}")
        if ((result[column] < 0) | (result[column] > 1)).any():
            _fail(f"{context}: {column} must be between 0 and 1")
    result["support"] = _integer_series(
        result["support"], f"{context}: support"
    )
    supports = tuple(int(value) for value in result["support"])
    if supports != EXPECTED_PUBLICTEST_SUPPORT:
        _fail(f"{context} has unexpected PublicTest support: {supports}")
    return result


def _classwise_confusion_matrix(
    confusion: pd.DataFrame, context: str
) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [column for column in CONFUSION_REQUIRED_COLUMNS if column not in confusion]
    if missing:
        _fail(f"{context} is missing confusion field(s): {', '.join(missing)}")
    if len(confusion) != len(LABEL_NAMES):
        _fail(f"{context} must contain exactly seven true-label rows")
    result = confusion.loc[:, list(CONFUSION_REQUIRED_COLUMNS)].copy()
    result["true_label"] = _integer_series(
        result["true_label"], f"{context}: true_label"
    )
    _check_label_order(
        result["true_label"].tolist(),
        result["true_label_name"].astype(str).tolist(),
        context,
    )
    matrix = np.empty((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    for index, column in enumerate(PREDICTED_COLUMNS):
        values = _integer_series(result[column], f"{context}: {column}")
        if (values < 0).any():
            _fail(f"{context}: confusion values cannot be negative")
        result[column] = values
        matrix[:, index] = values.to_numpy(dtype=np.int64)
    if int(matrix.sum()) != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{context} confusion total must be {EXPECTED_VALIDATION_SAMPLES}")
    if tuple(int(value) for value in matrix.sum(axis=1)) != EXPECTED_PUBLICTEST_SUPPORT:
        _fail(f"{context} confusion row totals do not match PublicTest support")
    return result, matrix


def _validate_classwise_predictions(
    predictions: pd.DataFrame, condition: str, context: str
) -> tuple[pd.DataFrame, np.ndarray]:
    required = (
        "sample_id",
        "split",
        "condition",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "correct",
    )
    missing = [column for column in required if column not in predictions]
    if missing:
        _fail(f"{context} is missing prediction field(s): {', '.join(missing)}")
    if len(predictions) != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{context} must contain {EXPECTED_VALIDATION_SAMPLES} predictions")
    result = predictions.copy()
    result["sample_id"] = _integer_series(
        result["sample_id"], f"{context}: sample_id"
    )
    if result["sample_id"].duplicated().any():
        _fail(f"{context} contains duplicate sample IDs")
    if set(result["split"].astype(str)) != {"validation"}:
        _fail(f"{context} split must be validation/PublicTest")
    if set(result["condition"].astype(str)) != {condition}:
        _fail(f"{context} condition must be {condition}")
    for field in ("true_label", "predicted_label"):
        result[field] = _integer_series(result[field], f"{context}: {field}")
        if ((result[field] < 0) | (result[field] >= len(LABEL_NAMES))).any():
            _fail(f"{context}: {field} must be between 0 and 6")
    expected_true_names = result["true_label"].map(dict(enumerate(LABEL_NAMES)))
    expected_predicted_names = result["predicted_label"].map(
        dict(enumerate(LABEL_NAMES))
    )
    if not result["true_label_name"].astype(str).equals(expected_true_names):
        _fail(f"{context} true-label names do not match label IDs")
    if not result["predicted_label_name"].astype(str).equals(
        expected_predicted_names
    ):
        _fail(f"{context} predicted-label names do not match label IDs")
    parsed_correct = _parse_bool_series(result["correct"], f"{context}: correct")
    expected_correct = result["true_label"].eq(result["predicted_label"])
    if not parsed_correct.equals(expected_correct):
        _fail(f"{context} correct flags disagree with labels")
    result["correct"] = parsed_correct
    matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    np.add.at(
        matrix,
        (
            result["true_label"].to_numpy(dtype=np.int64),
            result["predicted_label"].to_numpy(dtype=np.int64),
        ),
        1,
    )
    return result, matrix


def _metrics_from_confusion(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    true_positive = np.diag(values)
    predicted_totals = values.sum(axis=0)
    true_totals = values.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_totals,
        out=np.zeros_like(true_positive),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        true_positive,
        true_totals,
        out=np.zeros_like(true_positive),
        where=true_totals != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    return precision, recall, f1


def _validate_condition_metrics(
    metrics: dict[str, Any],
    condition: str,
    per_class: pd.DataFrame,
    matrix: np.ndarray,
    context: str,
) -> None:
    _require_keys(
        metrics,
        ("condition", "split", "sample_count", "confusion_matrix", "per_class"),
        context,
    )
    if metrics["condition"] != condition:
        _fail(f"{context} condition must be {condition}")
    if metrics["split"] != "validation":
        _fail(f"{context} split must be validation/PublicTest")
    if metrics["sample_count"] != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"{context} sample_count must be {EXPECTED_VALIDATION_SAMPLES}")
    try:
        embedded_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        _fail(f"{context} has malformed confusion_matrix: {exc}")
    if not np.array_equal(embedded_matrix, matrix):
        _fail(f"{context} confusion_matrix disagrees with CSV counts")
    embedded_per_class = metrics["per_class"]
    if not isinstance(embedded_per_class, list) or len(embedded_per_class) != 7:
        _fail(f"{context} per_class must contain exactly seven entries")
    for index, row in per_class.iterrows():
        source = embedded_per_class[index]
        if not isinstance(source, dict):
            _fail(f"{context} has malformed per_class entry")
        _require_keys(source, PER_CLASS_REQUIRED_COLUMNS, f"{context} per_class entry")
        if int(source["label"]) != int(row["label"]) or str(
            source["label_name"]
        ) != str(row["label_name"]):
            _fail(f"{context} per_class labels disagree with the CSV")
        for field in ("precision", "recall", "f1"):
            _check_close(
                float(source[field]),
                float(row[field]),
                f"{context} {field} for {row['label_name']}",
            )
        if int(source["support"]) != int(row["support"]):
            _fail(f"{context} support disagrees with the CSV")


def _assert_frames_identical(
    actual: pd.DataFrame, expected: pd.DataFrame, context: str
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        _fail(f"{context} do not match exactly: {exc}")


def _validate_stage8_provenance(provenance: dict[str, Any], seed: int, context: str) -> None:
    _require_keys(
        provenance,
        (
            "seed",
            "conditions",
            "evaluation_commit",
            "training_commit",
            "git_dirty",
            "training_git_dirty",
            "run_role",
        ),
        context,
    )
    if provenance["seed"] != seed:
        _fail(f"{context} seed must be {seed}")
    if tuple(provenance["conditions"]) != CLASSWISE_CONDITIONS:
        _fail(f"{context} condition order does not match the formal protocol")
    if provenance["evaluation_commit"] != EXPECTED_OCCLUSION_EVALUATION_COMMIT:
        _fail(f"{context} evaluation_commit is not the locked Stage 8 commit")
    if provenance["training_commit"] != EXPECTED_OCCLUSION_TRAINING_COMMIT:
        _fail(f"{context} training_commit is not the locked clean-training commit")
    if provenance["git_dirty"] is not False or provenance["training_git_dirty"] is not False:
        _fail(f"{context} formal evaluation and training must both be clean Git states")
    if provenance["run_role"] != "formal_masked_evaluation":
        _fail(f"{context} run_role must be formal_masked_evaluation")


def _validate_baseline_metadata(metadata: dict[str, Any], seed: int, context: str) -> None:
    _require_keys(
        metadata,
        ("seed", "status", "training_mode", "git_commit", "git_dirty"),
        context,
    )
    if metadata["seed"] != seed:
        _fail(f"{context} seed must be {seed}")
    if metadata["status"] != "completed" or metadata["training_mode"] != "clean":
        _fail(f"{context} must describe a completed clean-training run")
    if metadata["git_commit"] != EXPECTED_OCCLUSION_TRAINING_COMMIT:
        _fail(f"{context} git_commit is not the locked clean-training commit")
    if metadata["git_dirty"] is not False:
        _fail(f"{context} formal training Git state must be clean")


def load_classwise_evidence(
    stage8_archive: str | Path, baseline_archive: str | Path
) -> ClasswiseEvidence:
    """Load Stage 8 class metrics and fail closed on any evidence mismatch."""
    stage8_path = _require_file(Path(stage8_archive))
    baseline_path = _require_file(Path(baseline_archive))
    if stage8_path == baseline_path:
        _fail("Stage 8 and clean-baseline archives must be distinct")
    rows: list[dict[str, Any]] = []
    source_members: list[str] = []
    try:
        with tarfile.open(stage8_path, mode="r:*") as stage8, tarfile.open(
            baseline_path, mode="r:*"
        ) as baseline:
            stage8_members = _tar_member_index(stage8)
            baseline_members = _tar_member_index(baseline)
            for seed in EXPECTED_SEEDS:
                provenance_name = f"clean/seed{seed}/evaluation_provenance.json"
                provenance = _read_tar_json(
                    stage8, stage8_members, provenance_name
                )
                _validate_stage8_provenance(provenance, seed, provenance_name)
                source_members.append(provenance_name)

                baseline_prefix = f"formal-e7-4cb1e0f/seed{seed}"
                metadata_name = f"{baseline_prefix}/run_metadata.json"
                metadata = _read_tar_json(baseline, baseline_members, metadata_name)
                _validate_baseline_metadata(metadata, seed, metadata_name)
                source_members.append(metadata_name)

                canonical_sample_ids: np.ndarray | None = None
                canonical_true_labels: np.ndarray | None = None
                stage8_clean: tuple[
                    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
                ] | None = None
                for condition in CLASSWISE_CONDITIONS:
                    prefix = f"clean/seed{seed}/conditions/{condition}"
                    per_class_name = f"{prefix}_per_class_metrics.csv"
                    confusion_name = f"{prefix}_confusion_matrix.csv"
                    predictions_name = f"{prefix}_predictions.csv"
                    metrics_name = f"{prefix}_metrics.json"
                    per_class = _validate_classwise_per_class(
                        _read_tar_csv(stage8, stage8_members, per_class_name),
                        per_class_name,
                    )
                    confusion, confusion_matrix = _classwise_confusion_matrix(
                        _read_tar_csv(stage8, stage8_members, confusion_name),
                        confusion_name,
                    )
                    predictions, reconstructed_matrix = (
                        _validate_classwise_predictions(
                            _read_tar_csv(
                                stage8, stage8_members, predictions_name
                            ),
                            condition,
                            predictions_name,
                        )
                    )
                    if not np.array_equal(confusion_matrix, reconstructed_matrix):
                        _fail(
                            f"{confusion_name} cannot be reconstructed from "
                            f"{predictions_name}"
                        )
                    precision, recall, f1 = _metrics_from_confusion(confusion_matrix)
                    for field, calculated in (
                        ("precision", precision),
                        ("recall", recall),
                        ("f1", f1),
                    ):
                        if not np.allclose(
                            per_class[field].to_numpy(dtype=float),
                            calculated,
                            rtol=1e-12,
                            atol=1e-12,
                        ):
                            _fail(
                                f"{per_class_name} {field} cannot be reconstructed "
                                "from sample-level predictions"
                            )
                    metrics = _read_tar_json(stage8, stage8_members, metrics_name)
                    _validate_condition_metrics(
                        metrics,
                        condition,
                        per_class,
                        confusion_matrix,
                        metrics_name,
                    )
                    sample_ids = predictions["sample_id"].to_numpy(dtype=np.int64)
                    true_labels = predictions["true_label"].to_numpy(dtype=np.int64)
                    if canonical_sample_ids is None:
                        canonical_sample_ids = sample_ids
                        canonical_true_labels = true_labels
                    elif not np.array_equal(
                        sample_ids, canonical_sample_ids
                    ) or not np.array_equal(true_labels, canonical_true_labels):
                        _fail(
                            f"seed {seed} sample order or true labels differ across conditions"
                        )
                    for row in per_class.to_dict(orient="records"):
                        rows.append(
                            {
                                "seed": seed,
                                "condition": condition,
                                **row,
                            }
                        )
                    source_members.extend(
                        (
                            per_class_name,
                            confusion_name,
                            predictions_name,
                            metrics_name,
                        )
                    )
                    if condition == "clean":
                        stage8_clean = (
                            per_class,
                            confusion,
                            predictions,
                            metrics,
                        )

                if stage8_clean is None:
                    _fail(f"seed {seed} is missing the clean Stage 8 condition")
                baseline_names = {
                    "per_class": f"{baseline_prefix}/validation/best_per_class_metrics.csv",
                    "confusion": f"{baseline_prefix}/validation/best_confusion_matrix.csv",
                    "predictions": f"{baseline_prefix}/validation/best_predictions.csv",
                    "metrics": f"{baseline_prefix}/validation/best_metrics.json",
                }
                baseline_per_class = _validate_classwise_per_class(
                    _read_tar_csv(
                        baseline, baseline_members, baseline_names["per_class"]
                    ),
                    baseline_names["per_class"],
                )
                baseline_confusion, _ = _classwise_confusion_matrix(
                    _read_tar_csv(
                        baseline, baseline_members, baseline_names["confusion"]
                    ),
                    baseline_names["confusion"],
                )
                baseline_predictions, _ = _validate_classwise_predictions(
                    _read_tar_csv(
                        baseline, baseline_members, baseline_names["predictions"]
                    ),
                    "clean",
                    baseline_names["predictions"],
                )
                baseline_metrics = _read_tar_json(
                    baseline, baseline_members, baseline_names["metrics"]
                )
                _assert_frames_identical(
                    stage8_clean[0],
                    baseline_per_class,
                    f"seed {seed} Stage 8 and baseline clean per-class metrics",
                )
                _assert_frames_identical(
                    stage8_clean[1],
                    baseline_confusion,
                    f"seed {seed} Stage 8 and baseline clean confusion matrices",
                )
                _assert_frames_identical(
                    stage8_clean[2],
                    baseline_predictions,
                    f"seed {seed} Stage 8 and baseline clean predictions",
                )
                if stage8_clean[3] != baseline_metrics:
                    _fail(
                        f"seed {seed} Stage 8 and baseline clean metrics JSON do not match"
                    )
                source_members.extend(baseline_names.values())
    except (OSError, tarfile.TarError) as exc:
        _fail(f"could not read formal archive: {exc}")

    evidence = pd.DataFrame(rows)
    expected_rows = len(EXPECTED_SEEDS) * len(CLASSWISE_CONDITIONS) * len(LABEL_NAMES)
    if len(evidence) != expected_rows:
        _fail(f"class-wise evidence must contain {expected_rows} rows")
    return ClasswiseEvidence(
        stage8_archive=stage8_path,
        baseline_archive=baseline_path,
        per_class=evidence,
        source_members=tuple(source_members),
    )


def _parse_class_counts(raw: str, context: str) -> tuple[tuple[int, str, int], ...]:
    entries = []
    for item in raw.split(", "):
        match = re.fullmatch(r"(\d+):([A-Za-z]+)=(\d+)", item.strip())
        if match is None:
            _fail(f"{context} contains malformed class-count entry: {item!r}")
        entries.append((int(match.group(1)), match.group(2), int(match.group(3))))
    if len(entries) != 7:
        _fail(f"{context} must contain exactly seven class-count entries")
    _check_label_order(
        [entry[0] for entry in entries],
        [entry[1] for entry in entries],
        context,
    )
    if any(entry[2] < 0 for entry in entries):
        _fail(f"{context} cannot contain negative counts")
    return tuple(entries)


def _single_log_value(text: str, key: str) -> int:
    matches = re.findall(rf"(?m)^{re.escape(key)}=(\d+)$", text)
    if len(matches) != 1:
        _fail(f"preflight log must contain exactly one {key}=... line")
    return int(matches[0])


def parse_preflight_log(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse the Training/PublicTest class-count lines from preflight evidence."""
    log_path = _require_file(Path(path))
    text = log_path.read_text(encoding="utf-8")
    if "SMOKE TEST" in text or "clean-smoke" in text:
        _fail(f"smoke-test evidence is not allowed: {log_path}")
    train_total = _single_log_value(text, "train_samples")
    validation_total = _single_log_value(text, "validation_samples")
    train_match = re.findall(r"(?m)^train_class_counts=(.+)$", text)
    validation_match = re.findall(r"(?m)^validation_class_counts=(.+)$", text)
    if len(train_match) != 1 or len(validation_match) != 1:
        _fail("preflight log must contain one train_class_counts and one validation_class_counts line")
    train_entries = _parse_class_counts(train_match[0], "train_class_counts")
    validation_entries = _parse_class_counts(
        validation_match[0], "validation_class_counts"
    )
    if train_total != EXPECTED_TRAIN_SAMPLES:
        _fail(f"preflight train_samples must be {EXPECTED_TRAIN_SAMPLES}")
    if validation_total != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"preflight validation_samples must be {EXPECTED_VALIDATION_SAMPLES}")
    if sum(entry[2] for entry in train_entries) != train_total:
        _fail("preflight Training class counts do not sum to train_samples")
    if sum(entry[2] for entry in validation_entries) != validation_total:
        _fail("preflight PublicTest class counts do not sum to validation_samples")
    if tuple(entry[2] for entry in validation_entries) != EXPECTED_PUBLICTEST_SUPPORT:
        _fail("preflight PublicTest counts do not match the formal support contract")

    rows = []
    for split, entries, total in (
        ("Training", train_entries, train_total),
        ("PublicTest", validation_entries, validation_total),
    ):
        for label, label_name, count in entries:
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "label_name": label_name,
                    "count": count,
                    "percentage": 100.0 * count / total,
                }
            )
    return pd.DataFrame(rows), {
        "Training": train_total,
        "PublicTest": validation_total,
    }


def _validate_runs(runs: Sequence[RunEvidence]) -> None:
    seeds = [run.seed for run in runs]
    if sorted(seeds) != list(EXPECTED_SEEDS):
        _fail(f"formal run seeds must be exactly {EXPECTED_SEEDS}; got {seeds}")
    supports = [tuple(run.per_class["support"]) for run in runs]
    if any(support != supports[0] for support in supports[1:]):
        _fail("PublicTest support differs across formal runs")


def compute_summaries(
    runs: Sequence[RunEvidence], distribution: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Compute all intermediate tables using source data and ddof=1."""
    _validate_runs(runs)
    ordered_runs = sorted(runs, key=lambda run: run.seed)
    summary_metrics = pd.DataFrame(
        [
            {
                "seed": run.seed,
                "best_epoch": run.best_epoch,
                "accuracy": float(run.best_metrics["accuracy"]),
                "macro_f1": float(run.best_metrics["macro_f1"]),
                "loss": float(run.best_metrics["loss"]),
            }
            for run in ordered_runs
        ]
    )

    per_class_rows = []
    for label_index, label_name in enumerate(LABEL_NAMES):
        values = pd.Series(
            {
                f"f1_seed{run.seed}": float(run.per_class.iloc[label_index]["f1"])
                for run in ordered_runs
            },
            dtype="float64",
        )
        per_class_rows.append(
            {
                "label": label_index,
                "label_name": label_name,
                "support": int(ordered_runs[0].per_class.iloc[label_index]["support"]),
                **values.to_dict(),
                "f1_mean": float(values.mean()),
                "f1_sample_sd": float(values.std(ddof=1)),
            }
        )
    per_class_summary = pd.DataFrame(per_class_rows)

    normalized_matrices = []
    for run in ordered_runs:
        matrix = np.asarray(run.best_metrics["confusion_matrix"], dtype=float)
        try:
            normalized_matrices.append(row_normalize_confusion(matrix))
        except ValueError as exc:
            _fail(f"seed {run.seed} confusion matrix cannot be normalized: {exc}")
    mean_normalized = np.mean(np.stack(normalized_matrices, axis=0), axis=0)
    confusion_rows = []
    for index, label_name in enumerate(LABEL_NAMES):
        confusion_rows.append(
            {
                "true_label": index,
                "true_label_name": label_name,
                **{
                    f"predicted_{predicted_name}": float(mean_normalized[index, column])
                    for column, predicted_name in enumerate(LABEL_NAMES)
                },
            }
        )
    confusion_summary = pd.DataFrame(confusion_rows)
    return {
        "summary_metrics": summary_metrics,
        "per_class_f1_summary": per_class_summary,
        "mean_normalized_confusion_matrix": confusion_summary,
        "class_distribution": distribution.copy(),
    }


def _ordered_classwise_pivot(
    summary: pd.DataFrame, value_column: str, conditions: Sequence[str]
) -> pd.DataFrame:
    pivoted = summary.pivot(
        index=["label", "label_name"],
        columns="condition",
        values=value_column,
    )
    expected_index = pd.MultiIndex.from_tuples(
        list(enumerate(LABEL_NAMES)), names=("label", "label_name")
    )
    pivoted = pivoted.reindex(index=expected_index, columns=list(conditions))
    if pivoted.isna().any().any():
        _fail("class-wise summary is incomplete after seed aggregation")
    pivoted.columns.name = None
    return pivoted.reset_index()


def compute_classwise_summaries(
    evidence: ClasswiseEvidence,
) -> dict[str, pd.DataFrame]:
    """Compute seed means and paired sample SDs for class-wise F1 and drops."""
    frame = evidence.per_class.copy()
    seed_counts = frame.groupby(["condition", "label"])["seed"].nunique()
    if not seed_counts.eq(len(EXPECTED_SEEDS)).all():
        _fail("every class-condition pair must contain all three formal seeds")
    f1_aggregate = (
        frame.groupby(["condition", "label", "label_name"], sort=False)["f1"]
        .agg(f1_mean="mean", f1_sample_sd=lambda values: values.std(ddof=1))
        .reset_index()
    )
    f1_summary = _ordered_classwise_pivot(
        f1_aggregate, "f1_mean", CLASSWISE_CONDITIONS
    )
    f1_std = _ordered_classwise_pivot(
        f1_aggregate, "f1_sample_sd", CLASSWISE_CONDITIONS
    )

    clean = frame.loc[
        frame["condition"] == "clean", ["seed", "label", "label_name", "f1"]
    ].rename(columns={"f1": "clean_f1"})
    occluded = frame.loc[
        frame["condition"].isin(CLASSWISE_OCCLUDED_CONDITIONS),
        ["seed", "condition", "label", "label_name", "f1"],
    ]
    paired = occluded.merge(
        clean,
        on=["seed", "label", "label_name"],
        how="left",
        validate="many_to_one",
    )
    if paired["clean_f1"].isna().any():
        _fail("an occluded per-class result has no same-seed clean reference")
    paired["f1_drop"] = paired["clean_f1"] - paired["f1"]
    drop_aggregate = (
        paired.groupby(["condition", "label", "label_name"], sort=False)["f1_drop"]
        .agg(
            drop_mean="mean",
            drop_sample_sd=lambda values: values.std(ddof=1),
        )
        .reset_index()
    )
    drop_summary = _ordered_classwise_pivot(
        drop_aggregate, "drop_mean", CLASSWISE_OCCLUDED_CONDITIONS
    )
    drop_std = _ordered_classwise_pivot(
        drop_aggregate, "drop_sample_sd", CLASSWISE_OCCLUDED_CONDITIONS
    )
    return {
        "classwise_f1_summary": f1_summary,
        "classwise_f1_std": f1_std,
        "classwise_drop_from_clean_summary": drop_summary,
        "classwise_drop_from_clean_std": drop_std,
    }


def row_normalize_confusion(matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return a row-normalized copy of a non-empty confusion matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion matrix must be square")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("confusion matrix must contain finite non-negative values")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("confusion matrix cannot contain an empty row")
    return values / totals


def raw_curve_points(run: RunEvidence, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unmodified epoch and metric copies for a learning-curve panel."""
    allowed = {
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    }
    if column not in allowed:
        raise ValueError(f"unsupported learning-curve column: {column}")
    return (
        run.history["epoch"].to_numpy(dtype=float, copy=True),
        run.history[column].to_numpy(dtype=float, copy=True),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _ensure_independent_output(output_dir: Path, runs: Sequence[RunEvidence]) -> Path:
    output_path = output_dir.expanduser().resolve()
    for run in runs:
        if (
            output_path == run.run_dir
            or output_path.is_relative_to(run.run_dir)
            or run.run_dir.is_relative_to(output_path)
        ):
            _fail("output-dir must be independent from every formal run directory")
    output_path.mkdir(parents=True, exist_ok=True)
    unexpected = sorted(
        child.name
        for child in output_path.iterdir()
        if child.name not in EXPECTED_OUTPUT_NAMES
    )
    if unexpected:
        _fail(
            "output-dir contains unexpected existing entries: "
            + ", ".join(unexpected)
        )
    return output_path


def _use_paper_style() -> None:
    style_path = Path(__file__).with_name("paper.mplstyle")
    if not style_path.is_file():
        _fail(f"paper style file is missing: {style_path}")
    plt.style.use(str(style_path))


def _save_figure(figure: plt.Figure, stem: str, output_dir: Path) -> list[str]:
    output_names = []
    for extension in FIGURE_FORMATS:
        path = output_dir / f"{stem}.{extension}"
        if extension == "png":
            figure.savefig(path, format=extension, dpi=300)
        else:
            figure.savefig(path, format=extension)
        if not path.is_file() or path.stat().st_size == 0:
            _fail(f"figure export failed or is empty: {path}")
        output_names.append(path.name)
    plt.close(figure)
    return output_names


def _ensure_protocol_output(output_dir: Path) -> Path:
    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        _fail(f"protocol output path is not a directory: {output_path}")
    unexpected = sorted(
        child.name
        for child in output_path.iterdir()
        if child.name not in PROTOCOL_OUTPUT_NAMES
    )
    if unexpected:
        _fail(
            "protocol output directory contains unexpected existing entries: "
            + ", ".join(unexpected)
        )
    return output_path


def _protocol_display_image(image: Any) -> np.ndarray:
    from occlusion_fer.torch_data import IMAGENET_MEAN, IMAGENET_STD

    import torch

    means = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    stds = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    raw = image.detach().cpu() * stds + means
    if not torch.allclose(raw[0], raw[1], atol=1e-6, rtol=0) or not torch.allclose(
        raw[0], raw[2], atol=1e-6, rtol=0
    ):
        _fail("inverse-normalized FER2013 channels are not identical grayscale")
    return raw[0].clamp(0, 1).numpy()


def _plot_occlusion_protocol(
    clean_image: Any,
    masked_images: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    row_specs = (
        ("Upper-face", "upper_face"),
        ("Lower-face", "lower_face"),
        ("Random\nrectangle", "random_rectangle"),
    )
    severities = ("0.20", "0.30", "0.40")
    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.25))
    grid = figure.add_gridspec(
        3,
        5,
        width_ratios=(1.28, 0.42, 1.0, 1.0, 1.0),
        left=0.015,
        right=0.995,
        bottom=0.025,
        top=0.885,
        wspace=0.10,
        hspace=0.10,
    )

    original_axis = figure.add_subplot(grid[:, 0])
    original_axis.imshow(
        _protocol_display_image(clean_image),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    original_axis.set_xticks([])
    original_axis.set_yticks([])
    original_axis.set_title("Original / Unmasked", fontsize=9.2, fontweight="bold", pad=8)

    label_axes = []
    image_axes: list[list[Any]] = []
    for row_index, (display_name, condition_prefix) in enumerate(row_specs):
        label_axis = figure.add_subplot(grid[row_index, 1])
        label_axis.axis("off")
        label_axis.text(
            0.5,
            0.5,
            display_name,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            linespacing=1.05,
        )
        label_axes.append(label_axis)

        row_axes = []
        for column_index, severity in enumerate(severities):
            axis = figure.add_subplot(grid[row_index, column_index + 2])
            condition = f"{condition_prefix}_{severity}"
            axis.imshow(
                _protocol_display_image(masked_images[condition]),
                cmap="gray",
                vmin=0,
                vmax=1,
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(severity, fontsize=9.0, fontweight="bold", pad=8)
            row_axes.append(axis)
        image_axes.append(row_axes)

    figure.canvas.draw()
    severity_center = sum(
        axis.get_position().x0 + axis.get_position().width / 2
        for axis in image_axes[0]
    ) / len(image_axes[0])
    label_center = label_axes[0].get_position().x0 + label_axes[0].get_position().width / 2
    figure.text(
        label_center,
        0.955,
        "Occlusion type",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#596872",
    )
    figure.text(
        severity_center,
        0.955,
        "Severity (target ratio)",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#596872",
    )

    pdf_path = output_dir / f"{PROTOCOL_FIGURE_STEM}.pdf"
    png_path = output_dir / f"{PROTOCOL_FIGURE_STEM}.png"
    figure.savefig(pdf_path, format="pdf")
    figure.savefig(png_path, format="png", dpi=300)
    plt.close(figure)
    for path in (pdf_path, png_path):
        if not path.is_file() or path.stat().st_size == 0:
            _fail(f"protocol figure export failed or is empty: {path}")
    return pdf_path, png_path


def generate_occlusion_protocol_examples(
    data_path: str | Path,
    occlusion_config_path: str | Path,
    mean_artifact_path: str | Path,
    mask_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    sample_id: int,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate a non-result protocol figure from the locked masking pipeline."""
    from dataclasses import asdict

    import torch

    from occlusion_fer.config import load_occlusion_config
    from occlusion_fer.data import load_fer2013_csv
    from occlusion_fer.mask_manifest import (
        generate_official_manifest_rows,
        manifest_sha256,
    )
    from occlusion_fer.occlusion import (
        MASKED_CONDITIONS,
        apply_evaluation_mask,
        normalized_fill_vector,
    )
    from occlusion_fer.torch_data import Fer2013TorchDataset
    from occlusion_fer.training_mean import (
        load_training_mean_artifact,
        validate_artifact_dataset,
    )

    if type(sample_id) is not int or sample_id <= 0:
        _fail("protocol sample_id must be a positive integer")
    resolved_inputs = {
        "data": _require_file(Path(data_path)),
        "occlusion_config": _require_file(Path(occlusion_config_path)),
        "training_mean_artifact": _require_file(Path(mean_artifact_path)),
        "evaluation_mask_manifest": _require_file(Path(mask_manifest_path)),
    }
    config = load_occlusion_config(resolved_inputs["occlusion_config"])
    artifact = load_training_mean_artifact(resolved_inputs["training_mean_artifact"])
    validate_artifact_dataset(artifact, resolved_inputs["data"])
    manifest_rows = generate_official_manifest_rows(
        resolved_inputs["data"], resolved_inputs["training_mean_artifact"]
    )
    expected_manifest_sha256 = manifest_sha256(manifest_rows)
    if _sha256(resolved_inputs["evaluation_mask_manifest"]) != expected_manifest_sha256:
        _fail("evaluation mask manifest does not match the locked official rows")
    sample_manifest_rows = {
        row.condition: row
        for row in manifest_rows
        if row.sample_id == sample_id
    }
    if tuple(sample_manifest_rows) != MASKED_CONDITIONS:
        _fail("protocol sample is missing one or more canonical manifest rows")

    data = load_fer2013_csv(resolved_inputs["data"], include_splits=("validation",))
    matching_records = [record for record in data.records if record.sample_id == sample_id]
    if len(matching_records) != 1:
        _fail("protocol sample_id must identify exactly one PublicTest sample")
    record = matching_records[0]
    dataset = Fer2013TorchDataset(
        data,
        "validation",
        image_size=config.image_size,
        normalize_imagenet=True,
    )
    dataset_index = next(
        index
        for index, candidate in enumerate(data.records)
        if candidate.sample_id == sample_id
    )
    clean_image, tensor_label, tensor_sample_id = dataset[dataset_index]
    if tensor_label != record.label or tensor_sample_id != sample_id:
        _fail("protocol sample identity changed during tensor conversion")
    clean_snapshot = clean_image.clone()
    fill_vector = normalized_fill_vector(artifact.raw_training_mean)
    masked_images: dict[str, Any] = {}
    mask_metadata = []
    for condition in MASKED_CONDITIONS:
        masked_image, metadata = apply_evaluation_mask(
            clean_image,
            sample_id,
            condition,
            fill_vector,
        )
        manifest_row = sample_manifest_rows[condition]
        for field_name in (
            "occlusion_type",
            "target_ratio",
            "actual_ratio",
            "top",
            "left",
            "height",
            "width",
            "masked_pixel_count",
            "total_pixel_count",
            "evaluation_mask_seed",
            "coordinate_convention",
        ):
            metadata_field = "seed" if field_name == "evaluation_mask_seed" else field_name
            if getattr(metadata, metadata_field) != getattr(manifest_row, field_name):
                _fail(
                    f"mask metadata field {metadata_field} does not match manifest "
                    f"for {condition}"
                )
        masked_images[condition] = masked_image
        row_metadata = asdict(metadata)
        row_metadata["mask_identifier"] = (
            f"validation:{sample_id}:{condition}:seed={metadata.seed}:"
            f"{metadata.algorithm_version}"
        )
        row_metadata["manifest_row_key"] = {
            "sample_id": sample_id,
            "condition": condition,
        }
        row_metadata["mask_application_function"] = (
            "occlusion_fer.occlusion.apply_evaluation_mask"
        )
        mask_metadata.append(row_metadata)
    if not torch.equal(clean_image, clean_snapshot):
        _fail("source image changed while generating protocol masks")

    output_path = _ensure_protocol_output(Path(output_dir))
    _use_paper_style()
    pdf_path, png_path = _plot_occlusion_protocol(
        clean_image, masked_images, output_path
    )
    module_path = Path(__file__).resolve()
    style_path = module_path.with_name("paper.mplstyle")
    command = shlex.join(
        [sys.executable, "-m", "occlusion_fer.paper_figures", *argv]
    )
    python_path = os.environ.get("PYTHONPATH")
    if python_path:
        command = f"PYTHONPATH={shlex.quote(python_path)} {command}"
    metadata = {
        "metadata_schema_version": 1,
        "figure_type": "methodology_protocol",
        "contains_experimental_results": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "sample": {
            "sample_identifier": sample_id,
            "identifier_definition": "physical one-based CSV line number including header",
            "split": "validation",
            "official_split": "PublicTest",
            "dataset_label": record.label,
            "dataset_label_name": record.label_name,
            "source_image_shape": list(record.image.shape),
            "source_image_dtype": str(record.image.dtype),
            "source_image_sha256": hashlib.sha256(record.image.tobytes()).hexdigest(),
            "model_input_shape": list(clean_image.shape),
        },
        "original": {
            "mask_type": "original_unmasked",
            "severity": None,
            "seed": None,
            "mask_identifier": None,
        },
        "protocol": {
            "algorithm_version": config.algorithm_version,
            "image_size": config.image_size,
            "occlusion_types": list(config.types),
            "target_ratios": list(config.ratios),
            "fill_source": config.fill_source,
            "raw_training_mean": artifact.raw_training_mean,
            "evaluation_mask_seed": config.evaluation_mask_seed,
            "coordinate_convention": mask_metadata[0]["coordinate_convention"],
            "image_preparation_function": (
                "occlusion_fer.torch_data.Fer2013TorchDataset"
            ),
            "mask_application_function": (
                "occlusion_fer.occlusion.apply_evaluation_mask"
            ),
            "geometry_function": "occlusion_fer.occlusion.evaluation_geometry",
            "random_rectangle_hash_payload_function": (
                "occlusion_fer.mask_hash.build_evaluation_coordinate_payload"
            ),
            "random_rectangle_coordinate_function": (
                "occlusion_fer.mask_hash.coordinate_from_u64"
            ),
            "occlusion_config_source": str(resolved_inputs["occlusion_config"]),
            "training_mean_source": str(resolved_inputs["training_mean_artifact"]),
            "evaluation_manifest_source": str(
                resolved_inputs["evaluation_mask_manifest"]
            ),
        },
        "masks": mask_metadata,
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in resolved_inputs.items()
        },
        "generator": {
            "function": (
                "occlusion_fer.paper_figures.generate_occlusion_protocol_examples"
            ),
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "style_path": str(style_path),
            "style_sha256": _sha256(style_path),
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "dirty": bool(_git_value(["status", "--porcelain"])),
        },
        "outputs": {
            pdf_path.name: _sha256(pdf_path),
            png_path.name: _sha256(png_path),
        },
    }
    metadata_path = output_path / f"{PROTOCOL_FIGURE_STEM}_metadata.json"
    _write_json(metadata_path, metadata)
    print(f"sample_id={sample_id}")
    print("split=validation/PublicTest")
    print(f"output_dir={output_path}")
    print(f"generated_files={len(PROTOCOL_OUTPUT_NAMES)}")
    return metadata


def _plot_curves(runs: Sequence[RunEvidence], output_dir: Path) -> list[str]:
    ordered_runs = sorted(runs, key=lambda run: run.seed)
    figure, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_WIDTH, 5.6))
    panels = (
        (axes[0, 0], "train_loss", "Training loss", "Loss"),
        (axes[0, 1], "validation_loss", "PublicTest loss", "Loss"),
        (axes[1, 0], "validation_accuracy", "PublicTest accuracy", "Accuracy"),
        (axes[1, 1], "validation_macro_f1", "PublicTest macro-F1", "Macro-F1"),
    )
    handles = []
    labels = []
    for axis, column, title, ylabel in panels:
        for run in ordered_runs:
            style = SEED_STYLES[run.seed]
            epochs, values = raw_curve_points(run, column)
            line = axis.plot(
                epochs,
                values,
                label=f"seed {run.seed}",
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=5,
                markersize=3.2,
                linewidth=1.2,
            )[0]
            if axis is axes[0, 0]:
                handles.append(line)
                labels.append(f"seed {run.seed}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_xlim(1, 30)
        if column in {"validation_accuracy", "validation_macro_f1"}:
            axis.set_ylim(0, 1)
        axis.grid(True)
    for run in ordered_runs:
        value = float(
            run.history.loc[
                run.history["epoch"] == run.best_epoch, "validation_macro_f1"
            ].iloc[0]
        )
        style = SEED_STYLES[run.seed]
        axes[1, 1].scatter(
            [run.best_epoch],
            [value],
            color=style["color"],
            marker="*",
            s=52,
            zorder=4,
            edgecolors="black",
            linewidths=0.35,
        )
        axes[1, 1].annotate(
            f"best e{run.best_epoch}",
            (run.best_epoch, value),
            xytext={42: (-6, 10), 123: (-6, -13), 2026: (5, 6)}[run.seed],
            textcoords="offset points",
            ha="right" if run.best_epoch == 30 else "left",
            color=style["color"],
            fontsize=7,
        )
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(figure, "clean_training_validation_curves", output_dir)


def _plot_per_class_f1(
    runs: Sequence[RunEvidence], summary: pd.DataFrame, output_dir: Path
) -> list[str]:
    ordered_runs = sorted(runs, key=lambda run: run.seed)
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.8))
    x = np.arange(len(LABEL_NAMES), dtype=float)
    offsets = (-0.24, -0.08, 0.08)
    for offset, run in zip(offsets, ordered_runs):
        style = SEED_STYLES[run.seed]
        axis.scatter(
            x + offset,
            run.per_class["f1"].to_numpy(dtype=float),
            color=style["color"],
            marker=style["marker"],
            s=26,
            label=f"seed {run.seed}",
            zorder=3,
        )
    axis.errorbar(
        x + 0.24,
        summary["f1_mean"],
        yerr=summary["f1_sample_sd"],
        fmt="o",
        color="#222222",
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=5,
        capsize=3,
        linewidth=1.1,
        label="mean ± sample SD",
        zorder=4,
    )
    axis.set_xticks(x, LABEL_NAMES)
    axis.set_xlabel("FER2013 label")
    axis.set_ylabel("F1 score")
    axis.set_ylim(0, 1)
    axis.grid(True, axis="y")
    axis.legend(frameon=False, ncol=4, loc="upper center")
    figure.tight_layout()
    return _save_figure(figure, "clean_per_class_f1", output_dir)


def _plot_confusion(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    matrix = summary.loc[:, list(PREDICTED_COLUMNS)].to_numpy(dtype=float) * 100.0
    figure, axis = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH * 1.65, SINGLE_COLUMN_WIDTH * 1.45)
    )
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    axis.set_xticks(np.arange(7), LABEL_NAMES, rotation=35, ha="right")
    axis.set_yticks(np.arange(7), LABEL_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Clean baseline validation confusion matrix")
    for row in range(7):
        for column in range(7):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value >= 50 else "black",
                fontsize=7.3,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of true-label row (%)")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(100))
    figure.tight_layout()
    return _save_figure(figure, "clean_validation_confusion_matrix", output_dir)


def _plot_distribution(
    distribution: pd.DataFrame, totals: dict[str, int], output_dir: Path
) -> list[str]:
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.0))
    x = np.arange(len(LABEL_NAMES), dtype=float)
    width = 0.36
    colors = ("#5B7893", "#B07A55")
    for offset, split, color in zip((-width / 2, width / 2), ("Training", "PublicTest"), colors):
        frame = distribution.loc[distribution["split"] == split].sort_values("label")
        bars = axis.bar(
            x + offset,
            frame["percentage"],
            width,
            label=f"{split} (n={totals[split]:,})",
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
        )
        for bar, count in zip(bars, frame["count"]):
            axis.annotate(
                f"{int(count):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )
    axis.set_xticks(x, LABEL_NAMES)
    axis.set_xlabel("FER2013 label")
    axis.set_ylabel("Share of split (%)")
    axis.set_ylim(0, max(distribution["percentage"]) * 1.25)
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.grid(True, axis="y")
    axis.legend(frameon=False)
    figure.tight_layout()
    return _save_figure(
        figure, "fer2013_training_publictest_distribution", output_dir
    )


def _heatmap_text_color(image: matplotlib.image.AxesImage, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.52 else "white"


def _format_classwise_heatmap_axes(
    axis: plt.Axes, *, include_clean: bool
) -> None:
    column_count = len(CLASSWISE_CONDITIONS if include_clean else CLASSWISE_OCCLUDED_CONDITIONS)
    axis.set_yticks(np.arange(len(LABEL_NAMES)), LABEL_NAMES)
    axis.set_ylabel("FER2013 class")
    if include_clean:
        ratio_labels = ("", "0.20", "0.30", "0.40", "0.20", "0.30", "0.40", "0.20", "0.30", "0.40")
        group_positions = (0, 2, 5, 8)
        group_labels = ("Clean", "Upper face", "Lower face", "Random rectangle")
        separators = (0.5, 3.5, 6.5)
    else:
        ratio_labels = ("0.20", "0.30", "0.40") * 3
        group_positions = (1, 4, 7)
        group_labels = ("Upper face", "Lower face", "Random rectangle")
        separators = (2.5, 5.5)
    axis.set_xticks(np.arange(column_count), ratio_labels)
    axis.set_xlabel("Target occlusion ratio")
    group_axis = axis.secondary_xaxis("top")
    group_axis.set_xticks(group_positions, group_labels)
    group_axis.tick_params(axis="x", length=0, pad=4, labelsize=8)
    group_axis.spines["top"].set_visible(False)
    axis.set_xticks(np.arange(-0.5, column_count, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(LABEL_NAMES), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.75, alpha=0.9)
    axis.tick_params(which="minor", bottom=False, left=False)
    for position in separators:
        axis.axvline(position, color="#30363B", linewidth=1.0)


def _plot_classwise_f1_heatmap(
    summary: pd.DataFrame, output_dir: Path
) -> list[str]:
    matrix = summary.loc[:, list(CLASSWISE_CONDITIONS)].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.45))
    normalization = Normalize(vmin=0.0, vmax=1.0)
    image = axis.imshow(
        matrix,
        cmap="cividis",
        norm=normalization,
        aspect="auto",
        interpolation="nearest",
    )
    _format_classwise_heatmap_axes(axis, include_clean=True)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=_heatmap_text_color(image, value),
                fontsize=6.8,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean F1 across seeds")
    figure.tight_layout()
    return _save_figure(figure, "classwise_f1_heatmap", output_dir)


def _plot_classwise_drop_heatmap(
    summary: pd.DataFrame, output_dir: Path
) -> list[str]:
    matrix = summary.loc[:, list(CLASSWISE_OCCLUDED_CONDITIONS)].to_numpy(
        dtype=float
    )
    figure, axis = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 4.45))
    minimum = float(np.min(matrix))
    maximum = float(np.max(matrix))
    if minimum < 0:
        limit = max(abs(minimum), abs(maximum), 0.01)
        normalization: Normalize = TwoSlopeNorm(
            vmin=-limit, vcenter=0.0, vmax=limit
        )
        color_map = "RdBu_r"
    else:
        normalization = Normalize(vmin=0.0, vmax=max(maximum, 0.01))
        color_map = "magma"
    image = axis.imshow(
        matrix,
        cmap=color_map,
        norm=normalization,
        aspect="auto",
        interpolation="nearest",
    )
    _format_classwise_heatmap_axes(axis, include_clean=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=_heatmap_text_color(image, value),
                fontsize=6.8,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Mean F1 drop (clean - occluded)")
    figure.tight_layout()
    return _save_figure(
        figure, "classwise_drop_from_clean_heatmap", output_dir
    )


def _build_classwise_manifest(
    evidence: ClasswiseEvidence,
    output_dir: Path,
    output_files: Sequence[str],
    argv: Sequence[str],
) -> dict[str, Any]:
    style_path = Path(__file__).with_name("paper.mplstyle").resolve()
    module_path = Path(__file__).resolve()
    command = shlex.join(
        [sys.executable, "-m", "occlusion_fer.paper_figures", *argv]
    )
    python_path = os.environ.get("PYTHONPATH")
    if python_path:
        command = f"PYTHONPATH={shlex.quote(python_path)} {command}"
    git_status = _git_value(["status", "--porcelain"])
    return {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "current_git_commit": _git_value(["rev-parse", "HEAD"]),
        "current_git_dirty": bool(git_status),
        "generator": {
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "style_path": str(style_path),
            "style_sha256": _sha256(style_path),
        },
        "formal_commits": {
            "clean_training": EXPECTED_OCCLUSION_TRAINING_COMMIT,
            "stage8_evaluation": EXPECTED_OCCLUSION_EVALUATION_COMMIT,
        },
        "input_archives": {
            str(evidence.stage8_archive): _sha256(evidence.stage8_archive),
            str(evidence.baseline_archive): _sha256(evidence.baseline_archive),
        },
        "source_archive_members": list(evidence.source_members),
        "seeds": list(EXPECTED_SEEDS),
        "split": "validation",
        "conditions": list(CLASSWISE_CONDITIONS),
        "label_order": list(LABEL_NAMES),
        "metric": "per-class F1",
        "aggregation": {
            "f1": "arithmetic mean across formal seeds",
            "drop": "same-seed clean F1 minus occluded F1, then arithmetic mean",
            "standard_deviation": "sample SD across formal seeds (ddof=1)",
        },
        "clean_alignment": (
            "Stage 8 clean per-class metrics, confusion matrices, metrics JSON, "
            "and predictions exactly match each same-seed clean baseline best checkpoint"
        ),
        "evidence_checks": {
            "per_class_metrics_reconstructed_from_predictions": True,
            "confusion_counts_reconstructed_from_predictions": True,
            "sample_count_per_seed_condition": EXPECTED_VALIDATION_SAMPLES,
            "private_test_read": False,
        },
        "output_dir": str(output_dir),
        "output_files": sorted(output_files),
    }


def generate_classwise_heatmaps(
    stage8_archive: str | Path,
    baseline_archive: str | Path,
    output_dir: str | Path,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate formal archives and generate class-wise F1 heatmaps and tables."""
    evidence = load_classwise_evidence(stage8_archive, baseline_archive)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tables = compute_classwise_summaries(evidence)
    for name, table in tables.items():
        destination = output_path / f"{name}.csv"
        if destination.exists() and not destination.is_file():
            _fail(f"output destination is not a file: {destination}")
        table.to_csv(destination, index=False, float_format="%.10f")

    _use_paper_style()
    output_files = [f"{name}.csv" for name in tables]
    output_files.extend(
        _plot_classwise_f1_heatmap(
            tables["classwise_f1_summary"], output_path
        )
    )
    output_files.extend(
        _plot_classwise_drop_heatmap(
            tables["classwise_drop_from_clean_summary"], output_path
        )
    )
    manifest = _build_classwise_manifest(
        evidence, output_path, output_files, argv
    )
    manifest_path = output_path / "classwise_generation_manifest.json"
    output_files.append(manifest_path.name)
    manifest["output_files"] = sorted(set(output_files))
    _write_json(manifest_path, manifest)

    maximum = (
        tables["classwise_drop_from_clean_summary"]
        .set_index("label_name")
        .loc[:, list(CLASSWISE_OCCLUDED_CONDITIONS)]
        .stack()
    )
    label_name, condition = maximum.idxmax()
    print(
        "largest_mean_classwise_drop="
        f"{label_name},{condition},{maximum.max():.10f}"
    )
    print(f"output_dir={output_path}")
    print(f"generated_files={len(manifest['output_files'])}")
    return {"evidence": evidence, "tables": tables, "manifest": manifest}


def _print_summaries(
    tables: dict[str, pd.DataFrame], totals: dict[str, int]
) -> None:
    print("=== summary_metrics ===")
    print(tables["summary_metrics"].to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    overall = tables["summary_metrics"][["accuracy", "macro_f1", "loss"]]
    print("=== overall mean and sample SD (ddof=1) ===")
    for column in overall:
        values = pd.Series(overall[column], dtype="float64")
        print(
            f"{column}: mean={values.mean():.8f} "
            f"sample_sd={values.std(ddof=1):.8f}"
        )
    print("=== per-class F1 mean and sample SD (ddof=1) ===")
    print(
        tables["per_class_f1_summary"][
            ["label_name", "f1_mean", "f1_sample_sd"]
        ].to_string(index=False, float_format=lambda v: f"{v:.8f}")
    )
    print("=== mean row-normalized confusion matrix ===")
    print(
        tables["mean_normalized_confusion_matrix"]
        .loc[:, ["true_label_name", *PREDICTED_COLUMNS]]
        .to_string(index=False, float_format=lambda v: f"{v:.8f}")
    )
    print("=== class distribution ===")
    print(tables["class_distribution"].to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print(f"split totals: Training={totals['Training']} PublicTest={totals['PublicTest']}")


def _build_manifest(
    runs: Sequence[RunEvidence],
    preflight_path: Path,
    output_dir: Path,
    output_files: Sequence[str],
    argv: Sequence[str],
) -> dict[str, Any]:
    input_paths = [path for run in runs for path in run.input_paths]
    input_paths.append(preflight_path)
    input_hashes = {str(path): _sha256(path) for path in sorted(set(input_paths))}
    style_path = Path(__file__).with_name("paper.mplstyle").resolve()
    module_path = Path(__file__).resolve()
    command = shlex.join(
        [sys.executable, "-m", "occlusion_fer.paper_figures", *argv]
    )
    python_path = os.environ.get("PYTHONPATH")
    if python_path:
        command = f"PYTHONPATH={shlex.quote(python_path)} {command}"
    git_status = _git_value(["status", "--porcelain"])
    return {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "git_commit": EXPECTED_FORMAL_COMMIT,
        "current_git_commit": _git_value(["rev-parse", "HEAD"]),
        "current_git_dirty": bool(git_status),
        "generator": {
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "style_path": str(style_path),
            "style_sha256": _sha256(style_path),
        },
        "formal_run_dirs": [str(run.run_dir) for run in sorted(runs, key=lambda item: item.seed)],
        "preflight_log": str(preflight_path),
        "input_sha256": input_hashes,
        "seeds": list(EXPECTED_SEEDS),
        "split": "validation",
        "class_distribution_splits": ["Training", "PublicTest"],
        "condition": "clean",
        "smoke_test_read": False,
        "statistical_rules": {
            "cross_seed_standard_deviation": "pandas Series.std(ddof=1)",
            "curve_processing": "raw epoch values; no smoothing/interpolation/fitting",
            "confusion_matrix": "row-normalize each seed matrix, then arithmetic mean",
        },
        "output_dir": str(output_dir),
        "output_files": sorted(output_files),
    }


def generate_figures(
    run_dirs: Sequence[str | Path],
    preflight_log: str | Path,
    output_dir: str | Path,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate evidence, write summaries, and export all formal figures."""
    if len(run_dirs) != 3:
        _fail("exactly three --run-dir arguments are required")
    normalized_run_dirs = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(set(normalized_run_dirs)) != 3:
        _fail("the three --run-dir arguments must be distinct")
    runs = [
        load_run_evidence(path)
        for path in normalized_run_dirs
    ]
    _validate_runs(runs)
    distribution, totals = parse_preflight_log(preflight_log)
    if tuple(
        distribution.loc[distribution["split"] == "PublicTest"]
        .sort_values("label")["count"]
    ) != EXPECTED_PUBLICTEST_SUPPORT:
        _fail("preflight PublicTest support does not match formal run support")
    output_path = _ensure_independent_output(Path(output_dir), runs)
    tables = compute_summaries(runs, distribution)
    for name, table in tables.items():
        table.to_csv(output_path / f"{name}.csv", index=False)

    _use_paper_style()
    output_files = [f"{name}.csv" for name in SUMMARY_TABLE_NAMES]
    output_files.extend(_plot_curves(runs, output_path))
    output_files.extend(
        _plot_per_class_f1(runs, tables["per_class_f1_summary"], output_path)
    )
    output_files.extend(
        _plot_confusion(tables["mean_normalized_confusion_matrix"], output_path)
    )
    output_files.extend(_plot_distribution(distribution, totals, output_path))
    manifest = _build_manifest(
        runs,
        Path(preflight_log).expanduser().resolve(),
        output_path,
        output_files,
        argv,
    )
    manifest_path = output_path / "generation_manifest.json"
    _write_json(manifest_path, manifest)
    output_files.append(manifest_path.name)
    manifest["output_files"] = sorted(set(output_files))
    _write_json(manifest_path, manifest)
    _print_summaries(tables, totals)
    print(f"output_dir={output_path}")
    print(f"generated_files={len(manifest['output_files'])}")
    return {"tables": tables, "totals": totals, "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible FER2013 result figures or the locked "
            "synthetic-occlusion protocol figure."
        )
    )
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="generate the methodology/protocol figure instead of result figures",
    )
    parser.add_argument(
        "--stage8-core",
        action="store_true",
        help="generate all locked Stage 8 core paper figures and audit files",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        metavar="DIR",
        help="one formal clean-only run directory; pass exactly three times",
    )
    parser.add_argument("--preflight-log", metavar="FILE")
    parser.add_argument(
        "--classwise-stage8-archive",
        metavar="FILE",
        help="formal Stage 8 clean-occlusion evaluation archive",
    )
    parser.add_argument(
        "--classwise-baseline-archive",
        metavar="FILE",
        help="formal E7 clean-baseline archive used to verify clean alignment",
    )
    parser.add_argument("--data-path", metavar="FILE")
    parser.add_argument("--occlusion-config", metavar="FILE")
    parser.add_argument("--mean-artifact", metavar="FILE")
    parser.add_argument("--mask-manifest", metavar="FILE")
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--stage8-clean-results", metavar="DIR")
    parser.add_argument("--stage8-mixed-training-results", metavar="DIR")
    parser.add_argument("--stage8-mixed-evaluation-results", metavar="DIR")
    parser.add_argument("--stage8-clean-archive", metavar="FILE")
    parser.add_argument("--stage8-baseline-archive", metavar="FILE")
    parser.add_argument("--stage8-protocol-dir", metavar="DIR")
    parser.add_argument("--stage8-protocol-code-root", metavar="DIR")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    classwise_requested = bool(
        args.classwise_stage8_archive or args.classwise_baseline_archive
    )
    protocol_values = {
        "--data-path": args.data_path,
        "--occlusion-config": args.occlusion_config,
        "--mean-artifact": args.mean_artifact,
        "--mask-manifest": args.mask_manifest,
        "--sample-id": args.sample_id,
    }
    stage8_values = {
        "--stage8-clean-results": args.stage8_clean_results,
        "--stage8-mixed-training-results": args.stage8_mixed_training_results,
        "--stage8-mixed-evaluation-results": args.stage8_mixed_evaluation_results,
        "--stage8-clean-archive": args.stage8_clean_archive,
        "--stage8-baseline-archive": args.stage8_baseline_archive,
        "--stage8-protocol-dir": args.stage8_protocol_dir,
        "--stage8-protocol-code-root": args.stage8_protocol_code_root,
        "--data-path": args.data_path,
        "--sample-id": args.sample_id,
    }
    stage8_exclusive_values = {
        name: value
        for name, value in stage8_values.items()
        if name not in {"--data-path", "--sample-id"}
    }
    if args.stage8_core:
        if args.protocol or classwise_requested or args.run_dir or args.preflight_log:
            parser.error(
                "--stage8-core cannot be combined with another figure-generation mode"
            )
        if args.occlusion_config or args.mean_artifact or args.mask_manifest:
            parser.error(
                "--stage8-core resolves locked protocol artifacts from "
                "--stage8-protocol-dir"
            )
        missing = [name for name, value in stage8_values.items() if value is None]
        if missing:
            parser.error("--stage8-core requires " + ", ".join(missing))
    elif any(value is not None for value in stage8_exclusive_values.values()):
        parser.error("Stage 8 core input arguments require --stage8-core")
    elif args.protocol:
        if classwise_requested or args.run_dir or args.preflight_log:
            parser.error(
                "--protocol cannot be combined with result-figure or class-wise inputs"
            )
        missing = [name for name, value in protocol_values.items() if value is None]
        if missing:
            parser.error("--protocol requires " + ", ".join(missing))
    elif any(value is not None for value in protocol_values.values()):
        parser.error("protocol input arguments require --protocol")
    elif classwise_requested:
        if not args.classwise_stage8_archive or not args.classwise_baseline_archive:
            parser.error(
                "class-wise mode requires both --classwise-stage8-archive "
                "and --classwise-baseline-archive"
            )
        if args.run_dir or args.preflight_log:
            parser.error(
                "class-wise archive arguments cannot be combined with "
                "--run-dir or --preflight-log"
            )
    else:
        if args.run_dir is None or len(args.run_dir) != 3:
            parser.error("--run-dir must be provided exactly three times")
        if args.preflight_log is None:
            parser.error("--preflight-log is required with --run-dir")
    try:
        command_argv = tuple(sys.argv[1:] if argv is None else argv)
        if args.stage8_core:
            from occlusion_fer.paper_stage8 import generate_stage8_core_figures

            generate_stage8_core_figures(
                clean_results_dir=args.stage8_clean_results,
                mixed_training_dir=args.stage8_mixed_training_results,
                mixed_evaluation_dir=args.stage8_mixed_evaluation_results,
                clean_archive=args.stage8_clean_archive,
                baseline_archive=args.stage8_baseline_archive,
                protocol_dir=args.stage8_protocol_dir,
                data_path=args.data_path,
                protocol_code_root=args.stage8_protocol_code_root,
                output_dir=args.output_dir,
                sample_id=args.sample_id,
                argv=command_argv,
            )
        elif args.protocol:
            generate_occlusion_protocol_examples(
                args.data_path,
                args.occlusion_config,
                args.mean_artifact,
                args.mask_manifest,
                args.output_dir,
                sample_id=args.sample_id,
                argv=command_argv,
            )
        elif classwise_requested:
            generate_classwise_heatmaps(
                args.classwise_stage8_archive,
                args.classwise_baseline_archive,
                args.output_dir,
                argv=command_argv,
            )
        else:
            generate_figures(
                args.run_dir,
                args.preflight_log,
                args.output_dir,
                argv=command_argv,
            )
    except (PaperFigureError, OSError, ValueError) as exc:
        print(f"paper_figures error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
