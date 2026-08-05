"""Reproducible figures for the formal FER2013 clean-only baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import yaml

from occlusion_fer.data import FER2013_LABEL_NAMES


EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_FORMAL_COMMIT = "da889bdd818cab6403767f2f6c7d5391d8317324"
EXPECTED_TRAIN_SAMPLES = 28709
EXPECTED_VALIDATION_SAMPLES = 3589
EXPECTED_PUBLICTEST_SUPPORT = (467, 56, 496, 895, 653, 415, 607)
LABEL_NAMES = tuple(FER2013_LABEL_NAMES)
EXPECTED_BEST_EPOCHS = {42: 30, 123: 30, 2026: 23}

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
        description="Generate reproducible FER2013 clean-only paper figures."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        metavar="DIR",
        help="one formal clean-only run directory; pass exactly three times",
    )
    parser.add_argument("--preflight-log", required=True, metavar="FILE")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.run_dir) != 3:
        parser.error("--run-dir must be provided exactly three times")
    try:
        generate_figures(
            args.run_dir,
            args.preflight_log,
            args.output_dir,
            argv=tuple(sys.argv[1:] if argv is None else argv),
        )
    except (PaperFigureError, OSError, ValueError) as exc:
        print(f"paper_figures error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
