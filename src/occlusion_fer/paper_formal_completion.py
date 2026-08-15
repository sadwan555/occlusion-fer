"""Generate missing formal paper figures from locked Stage 8 evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "occlusion-fer-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from occlusion_fer.paper_framework import generate_framework


EXPECTED_SEEDS = (42, 123, 2026)
EXPECTED_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
EXPECTED_ROOT = "formal-e7-4cb1e0f"
EXPECTED_TRAINING_COMMIT = "4cb1e0ffe4b55efc090a45cfed560b28f50b9509"
EXPECTED_EVALUATION_COMMIT = "c1c9187aa2ddf7dd84906c7f139ad9a750ef202d"
EXPECTED_IMAGE_SIZE = 224
EXPECTED_EPOCHS = 50
EXPECTED_VALIDATION_SAMPLES = 3589
EXPECTED_PROTOCOL = "occlusion-v2-224"
EXPECTED_CONDITIONS = (
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
SUMMARY_STEM = "experiment1_clean_baseline_summary"
CURVES_STEM = "experiment1_training_curves"
CONFUSION_STEM = "experiment1_clean_confusion_matrix"
AUDIT_NAME = "experiment1_formal_source_audit.json"
MANIFEST_NAME = "formal_completion_generation_manifest.json"
REJECTED_SOURCE_TOKENS = ("da889bd", "synthetic", "gradcam_112")
HISTORY_COLUMNS = (
    "epoch",
    "train_samples",
    "train_loss",
    "validation_samples",
    "validation_accuracy",
    "validation_macro_f1",
    "updated_best_checkpoint",
)

SEED_COLORS = {
    42: "#356A8A",
    123: "#A45C40",
    2026: "#4F7651",
}


class FormalCompletionError(ValueError):
    """Raised when formal paper evidence fails closed validation."""


@dataclass(frozen=True)
class FormalRunEvidence:
    seed: int
    history: pd.DataFrame
    best_epoch: int
    best_metrics: dict[str, Any]
    last_metrics: dict[str, Any]
    per_class: pd.DataFrame
    confusion: np.ndarray
    config: dict[str, Any]
    run_metadata: dict[str, Any]
    checkpoint_sha256: str
    checkpoint_metadata: dict[str, Any]
    source_members: dict[str, str]
    source_sha256: dict[str, str]
    evaluation_provenance_path: Path
    evaluation_provenance: dict[str, Any]


def _fail(message: str) -> None:
    raise FormalCompletionError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonformal_path(path: Path, role: str) -> None:
    lowered = str(path).lower()
    token = next((item for item in REJECTED_SOURCE_TOKENS if item in lowered), None)
    if token is not None:
        _fail(f"{role} uses rejected legacy or synthetic token {token!r}: {path}")


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError:
        _fail(f"formal archive member is missing: {name}")
    if not member.isfile() or member.size <= 0:
        _fail(f"formal archive member is not a non-empty file: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        _fail(f"formal archive member cannot be read: {name}")
    return stream.read()


def _json_bytes(payload: bytes, source: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON in {source}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {source}")
    return value


def _yaml_bytes(payload: bytes, source: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        _fail(f"invalid YAML in {source}: {exc}")
    if not isinstance(value, dict):
        _fail(f"YAML root must be a mapping: {source}")
    return value


def _csv_bytes(payload: bytes, source: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        _fail(f"invalid CSV in {source}: {exc}")
    if frame.empty:
        _fail(f"CSV is empty: {source}")
    return frame


def _nested(mapping: dict[str, Any], path: str, expected: Any) -> None:
    value: Any = mapping
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            _fail(f"missing required field {path}")
        value = value[key]
    if value != expected:
        _fail(f"{path} is {value!r}, expected {expected!r}")


def _checkpoint_metadata(payload: bytes, seed: int) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as exc:  # torch surfaces several format-specific errors
        _fail(f"seed {seed} best checkpoint cannot be read: {exc}")
    if not isinstance(checkpoint, dict):
        _fail(f"seed {seed} checkpoint root must be a mapping")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, dict):
        _fail(f"seed {seed} checkpoint lacks resolved_config metadata")
    _nested(resolved, "dataset.image_size", EXPECTED_IMAGE_SIZE)
    _nested(resolved, "model.name", "resnet18")
    _nested(resolved, "model.pretrained", True)
    _nested(resolved, "training.mode", "clean")
    _nested(resolved, "training.seed", seed)
    _nested(resolved, "training.epochs", EXPECTED_EPOCHS)
    return {
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "validation_accuracy": checkpoint.get("validation_accuracy"),
        "validation_macro_f1": checkpoint.get("validation_macro_f1"),
        "best_validation_macro_f1": checkpoint.get("best_validation_macro_f1"),
        "image_size": resolved["dataset"]["image_size"],
        "model": resolved["model"]["name"],
        "pretrained": resolved["model"]["pretrained"],
        "training_mode": resolved["training"]["mode"],
        "epochs": resolved["training"]["epochs"],
    }


def _validate_history(history: pd.DataFrame, seed: int) -> int:
    missing = [name for name in HISTORY_COLUMNS if name not in history.columns]
    if missing:
        _fail(f"seed {seed} history is missing columns: {missing}")
    if len(history) != EXPECTED_EPOCHS:
        _fail(
            f"seed {seed} history has {len(history)} epochs; "
            f"expected {EXPECTED_EPOCHS}"
        )
    epochs = history["epoch"].to_numpy(dtype=int)
    if not np.array_equal(epochs, np.arange(1, EXPECTED_EPOCHS + 1)):
        _fail(f"seed {seed} history epochs are not exactly 1..{EXPECTED_EPOCHS}")
    numeric = history.loc[
        :, ["train_loss", "validation_accuracy", "validation_macro_f1"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        _fail(f"seed {seed} history contains a missing or non-finite plotted value")
    if set(history["train_samples"].astype(int)) != {28709}:
        _fail(f"seed {seed} history has unexpected Training sample counts")
    if set(history["validation_samples"].astype(int)) != {
        EXPECTED_VALIDATION_SAMPLES
    }:
        _fail(f"seed {seed} history has unexpected PublicTest sample counts")
    flags = history["updated_best_checkpoint"].map(
        lambda value: value is True or str(value).lower() == "true"
    )
    best_rows = history.loc[flags]
    if best_rows.empty:
        _fail(f"seed {seed} history has no best-checkpoint update")
    return int(best_rows.iloc[-1]["epoch"])


def _validate_metrics(
    metrics: dict[str, Any], per_class: pd.DataFrame, confusion: np.ndarray, seed: int
) -> None:
    _nested(metrics, "condition", "clean")
    _nested(metrics, "split", "validation")
    _nested(metrics, "sample_count", EXPECTED_VALIDATION_SAMPLES)
    if list(per_class["label_name"].astype(str)) != list(EXPECTED_LABELS):
        _fail(f"seed {seed} per-class label order is not the locked FER2013 order")
    if list(per_class["label"].astype(int)) != list(range(len(EXPECTED_LABELS))):
        _fail(f"seed {seed} per-class label IDs are not 0..6")
    if confusion.shape != (7, 7):
        _fail(f"seed {seed} confusion matrix shape is {confusion.shape}, expected (7, 7)")
    if not np.isfinite(confusion).all() or (confusion < 0).any():
        _fail(f"seed {seed} confusion matrix has invalid values")
    if int(confusion.sum()) != EXPECTED_VALIDATION_SAMPLES:
        _fail(f"seed {seed} confusion matrix does not contain 3,589 samples")
    support = per_class["support"].to_numpy(dtype=int)
    if not np.array_equal(confusion.sum(axis=1).astype(int), support):
        _fail(f"seed {seed} confusion rows do not match per-class support")
    embedded = np.asarray(metrics.get("confusion_matrix"), dtype=float)
    if not np.array_equal(embedded, confusion):
        _fail(f"seed {seed} metrics and confusion CSV disagree")
    labels = [item.get("label_name") for item in metrics.get("label_order", [])]
    if labels != list(EXPECTED_LABELS):
        _fail(f"seed {seed} metrics label order is not locked")


def _validate_config_and_metadata(
    config: dict[str, Any], metadata: dict[str, Any], seed: int
) -> None:
    for path, expected in (
        ("project.name", "occlusion-fer"),
        ("project.experiment_name", "e7_high_resolution_longer"),
        ("dataset.name", "fer2013"),
        ("dataset.image_size", EXPECTED_IMAGE_SIZE),
        ("dataset.num_classes", 7),
        ("model.name", "resnet18"),
        ("model.pretrained", True),
        ("training.mode", "clean"),
        ("training.seed", seed),
        ("training.epochs", EXPECTED_EPOCHS),
        ("training.batch_size", 128),
        ("training.learning_rate", 0.0001),
        ("training.weight_decay", 0.001),
    ):
        _nested(config, path, expected)
    for key, expected in (
        ("status", "completed"),
        ("seed", seed),
        ("training_mode", "clean"),
        ("git_commit", EXPECTED_TRAINING_COMMIT),
        ("git_dirty", False),
    ):
        if metadata.get(key) != expected:
            _fail(
                f"seed {seed} run_metadata.{key} is {metadata.get(key)!r}, "
                f"expected {expected!r}"
            )


def _validate_evaluation_provenance(
    provenance: dict[str, Any], checkpoint_sha256: str, seed: int
) -> None:
    required = {
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "training_commit": EXPECTED_TRAINING_COMMIT,
        "evaluation_commit": EXPECTED_EVALUATION_COMMIT,
        "training_git_dirty": False,
        "git_dirty": False,
        "image_height": EXPECTED_IMAGE_SIZE,
        "image_width": EXPECTED_IMAGE_SIZE,
        "protocol": EXPECTED_PROTOCOL,
        "conditions": list(EXPECTED_CONDITIONS),
    }
    for key, expected in required.items():
        if provenance.get(key) != expected:
            _fail(
                f"seed {seed} evaluation provenance {key} is "
                f"{provenance.get(key)!r}, expected {expected!r}"
            )
    _nested(provenance, "resolved_config.dataset.image_size", EXPECTED_IMAGE_SIZE)
    _nested(provenance, "resolved_config.dataset.permitted_splits", ["Training", "PublicTest"])
    _nested(provenance, "resolved_config.model.name", "resnet18")
    _nested(provenance, "resolved_config.model.pretrained", True)
    _nested(provenance, "resolved_config.training.mode", "clean")
    _nested(provenance, "resolved_config.training.seed", seed)
    _nested(provenance, "resolved_config.training.epochs", EXPECTED_EPOCHS)
    _nested(provenance, "resolved_config.occlusion.evaluation.split", "validation")
    _nested(
        provenance,
        "resolved_config.occlusion.protocol.algorithm_version",
        EXPECTED_PROTOCOL,
    )


def load_formal_lineage(
    clean_archive: str | Path, experiment2_clean_root: str | Path
) -> tuple[FormalRunEvidence, ...]:
    """Load and cross-check the current formal clean lineage, failing closed."""
    archive_path = Path(clean_archive).expanduser().resolve()
    evaluation_root = Path(experiment2_clean_root).expanduser().resolve()
    _reject_nonformal_path(archive_path, "clean archive")
    _reject_nonformal_path(evaluation_root, "Experiment 2 clean root")
    if not archive_path.is_file():
        _fail(f"clean archive is missing: {archive_path}")
    if not evaluation_root.is_dir():
        _fail(f"Experiment 2 clean root is missing: {evaluation_root}")

    runs: list[FormalRunEvidence] = []
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except tarfile.TarError as exc:
        _fail(f"clean archive is not readable: {archive_path}: {exc}")
    with archive:
        top_levels = {name.name.split("/", 1)[0] for name in archive.getmembers()}
        if top_levels != {EXPECTED_ROOT}:
            _fail(
                f"clean archive root is {sorted(top_levels)}, expected only {EXPECTED_ROOT}"
            )
        for seed in EXPECTED_SEEDS:
            base = f"{EXPECTED_ROOT}/seed{seed}"
            members = {
                "checkpoint": f"{base}/best.pt",
                "history": f"{base}/history.csv",
                "resolved_config": f"{base}/resolved_config.yaml",
                "run_metadata": f"{base}/run_metadata.json",
                "best_metrics": f"{base}/validation/best_metrics.json",
                "last_metrics": f"{base}/validation/last_metrics.json",
                "per_class": f"{base}/validation/best_per_class_metrics.csv",
                "confusion": f"{base}/validation/best_confusion_matrix.csv",
                "predictions": f"{base}/validation/best_predictions.csv",
            }
            payloads = {
                role: _read_member(archive, member) for role, member in members.items()
            }
            config = _yaml_bytes(payloads["resolved_config"], members["resolved_config"])
            metadata = _json_bytes(payloads["run_metadata"], members["run_metadata"])
            history = _csv_bytes(payloads["history"], members["history"])
            best_metrics = _json_bytes(payloads["best_metrics"], members["best_metrics"])
            last_metrics = _json_bytes(payloads["last_metrics"], members["last_metrics"])
            per_class = _csv_bytes(payloads["per_class"], members["per_class"])
            confusion_frame = _csv_bytes(
                payloads["confusion"], members["confusion"]
            )
            predicted_columns = [f"predicted_{label}" for label in EXPECTED_LABELS]
            required_confusion_columns = [
                "true_label",
                "true_label_name",
                *predicted_columns,
            ]
            if list(confusion_frame.columns) != required_confusion_columns:
                _fail(f"seed {seed} confusion CSV columns are not locked")
            if list(confusion_frame["true_label"].astype(int)) != list(range(7)):
                _fail(f"seed {seed} confusion true-label IDs are not 0..6")
            if list(confusion_frame["true_label_name"].astype(str)) != list(
                EXPECTED_LABELS
            ):
                _fail(f"seed {seed} confusion true-label order is not locked")
            confusion = confusion_frame.loc[:, predicted_columns].to_numpy(dtype=float)
            _validate_config_and_metadata(config, metadata, seed)
            best_epoch = _validate_history(history, seed)
            _validate_metrics(best_metrics, per_class, confusion, seed)
            checkpoint_sha256 = _sha256_bytes(payloads["checkpoint"])
            checkpoint_metadata = _checkpoint_metadata(payloads["checkpoint"], seed)
            if checkpoint_metadata["seed"] != seed:
                _fail(f"seed {seed} checkpoint seed metadata disagrees")
            if checkpoint_metadata["epoch"] != best_epoch:
                _fail(f"seed {seed} checkpoint epoch disagrees with history")
            best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
            for field, history_field in (
                ("validation_accuracy", "validation_accuracy"),
                ("validation_macro_f1", "validation_macro_f1"),
            ):
                values = (
                    float(checkpoint_metadata[field]),
                    float(best_metrics[field.replace("validation_", "")]),
                    float(best_row[history_field]),
                )
                if not np.allclose(values, values[0], atol=1e-12, rtol=0.0):
                    _fail(f"seed {seed} {field} disagrees across checkpoint/history/metrics")

            provenance_path = evaluation_root / f"seed{seed}" / "evaluation_provenance.json"
            if not provenance_path.is_file():
                _fail(f"Experiment 2 provenance is missing: {provenance_path}")
            provenance = _json_bytes(provenance_path.read_bytes(), str(provenance_path))
            _validate_evaluation_provenance(provenance, checkpoint_sha256, seed)

            alignment = {
                "best_metrics": "clean_metrics.json",
                "per_class": "clean_per_class_metrics.csv",
                "confusion": "clean_confusion_matrix.csv",
                "predictions": "clean_predictions.csv",
            }
            conditions_dir = evaluation_root / f"seed{seed}" / "conditions"
            for source_role, clean_name in alignment.items():
                stage8_path = conditions_dir / clean_name
                if not stage8_path.is_file():
                    _fail(f"Experiment 2 clean evidence is missing: {stage8_path}")
                if _sha256_path(stage8_path) != _sha256_bytes(payloads[source_role]):
                    _fail(
                        f"seed {seed} Experiment 1 {source_role} does not match "
                        f"Experiment 2 clean evaluation"
                    )

            runs.append(
                FormalRunEvidence(
                    seed=seed,
                    history=history,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    last_metrics=last_metrics,
                    per_class=per_class,
                    confusion=confusion,
                    config=config,
                    run_metadata=metadata,
                    checkpoint_sha256=checkpoint_sha256,
                    checkpoint_metadata=checkpoint_metadata,
                    source_members=members,
                    source_sha256={
                        role: _sha256_bytes(payload) for role, payload in payloads.items()
                    },
                    evaluation_provenance_path=provenance_path,
                    evaluation_provenance=provenance,
                )
            )
    if tuple(run.seed for run in runs) != EXPECTED_SEEDS:
        _fail(f"formal lineage must contain exactly seeds {EXPECTED_SEEDS}")
    return tuple(runs)


def sample_summary(values: Sequence[float]) -> tuple[float, float]:
    """Return arithmetic mean and sample SD for the locked three-seed design."""
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        _fail("sample summary requires exactly three finite seed values")
    return float(array.mean()), float(array.std(ddof=1))


def build_summary_table(runs: Sequence[FormalRunEvidence]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    series: list[tuple[str, str, list[float]]] = []
    for metric in ("accuracy", "macro_f1"):
        values = [float(run.best_metrics[metric]) for run in runs]
        series.append((metric, "", values))
    for label in EXPECTED_LABELS:
        values = [
            float(run.per_class.set_index("label_name").loc[label, "f1"])
            for run in runs
        ]
        series.append(("per_class_f1", label, values))
    for metric, label, values in series:
        for run, value in zip(runs, values):
            rows.append(
                {
                    "seed": str(run.seed),
                    "metric": metric,
                    "class": label,
                    "value": value,
                    "summary": "individual",
                }
            )
        mean, sample_sd = sample_summary(values)
        rows.extend(
            (
                {
                    "seed": "mean",
                    "metric": metric,
                    "class": label,
                    "value": mean,
                    "summary": "mean",
                },
                {
                    "seed": "sample_sd",
                    "metric": metric,
                    "class": label,
                    "value": sample_sd,
                    "summary": "sample_sd_ddof_1",
                },
            )
        )
    return pd.DataFrame(rows, columns=["seed", "metric", "class", "value", "summary"])


def build_training_table(runs: Sequence[FormalRunEvidence]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for source, metric in (
            ("train_loss", "training_loss"),
            ("validation_accuracy", "validation_accuracy"),
            ("validation_macro_f1", "validation_macro_f1"),
        ):
            for record in run.history.loc[:, ["epoch", source]].to_dict("records"):
                rows.append(
                    {
                        "seed": run.seed,
                        "epoch": int(record["epoch"]),
                        "metric": metric,
                        "value": float(record[source]),
                        "best_checkpoint_epoch": run.best_epoch,
                        "checkpoint_selection_metric": "clean PublicTest macro-F1",
                    }
                )
    return pd.DataFrame(rows)


def build_confusion_table(runs: Sequence[FormalRunEvidence]) -> pd.DataFrame:
    matrices = []
    for run in runs:
        row_totals = run.confusion.sum(axis=1, keepdims=True)
        if (row_totals == 0).any():
            _fail(f"seed {run.seed} confusion matrix contains an empty true class")
        matrices.append(run.confusion / row_totals)
    stack = np.stack(matrices)
    mean = stack.mean(axis=0)
    sample_sd = stack.std(axis=0, ddof=1)
    rows = []
    for true_index, true_label in enumerate(EXPECTED_LABELS):
        for predicted_index, predicted_label in enumerate(EXPECTED_LABELS):
            rows.append(
                {
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "mean_fraction": float(mean[true_index, predicted_index]),
                    "mean_percent": float(mean[true_index, predicted_index] * 100.0),
                    "sample_sd_fraction": float(
                        sample_sd[true_index, predicted_index]
                    ),
                    **{
                        f"seed{run.seed}_fraction": float(
                            stack[index, true_index, predicted_index]
                        )
                        for index, run in enumerate(runs)
                    },
                }
            )
    return pd.DataFrame(rows)


def _style_path() -> Path:
    path = Path(__file__).with_name("paper.mplstyle")
    if not path.is_file():
        _fail(f"paper style is missing: {path}")
    return path


def _save_figure(figure: Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        options: dict[str, Any] = {
            "format": extension,
            "bbox_inches": "tight",
            "pad_inches": 0.04,
        }
        if extension == "png":
            options["dpi"] = 300
        else:
            options["metadata"] = {
                "Title": stem,
                "Creator": "occlusion_fer.paper_formal_completion",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(path, **options)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_summary(
    runs: Sequence[FormalRunEvidence],
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    split_label: str = "PublicTest",
    stem: str = SUMMARY_STEM,
) -> list[Path]:
    with plt.style.context(_style_path()):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
        overall_axis, class_axis = axes
        x_positions = np.arange(2, dtype=float)
        offsets = (-0.10, 0.0, 0.10)
        for x, metric, label in zip(
            x_positions, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1")
        ):
            rows = summary.loc[
                (summary["metric"] == metric)
                & (summary["summary"] == "individual")
            ]
            for offset, run, value in zip(offsets, runs, rows["value"]):
                overall_axis.scatter(
                    x + offset,
                    value,
                    s=27,
                    color=SEED_COLORS[run.seed],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                    label=f"Seed {run.seed}" if x == 0 else None,
                )
            mean = float(
                summary.loc[
                    (summary["metric"] == metric) & (summary["summary"] == "mean"),
                    "value",
                ].iloc[0]
            )
            sd = float(
                summary.loc[
                    (summary["metric"] == metric)
                    & (summary["summary"] == "sample_sd_ddof_1"),
                    "value",
                ].iloc[0]
            )
            overall_axis.errorbar(
                x,
                mean,
                yerr=sd,
                color="#20272D",
                marker="D",
                markersize=5,
                capsize=4,
                linewidth=1.2,
                zorder=5,
                label="Mean +/- sample SD" if x == 0 else None,
            )
        overall_axis.set_xticks(x_positions, ("Accuracy", "Macro-F1"))
        overall_axis.set_ylim(0.65, 0.73)
        overall_axis.set_ylabel("Score")
        overall_axis.set_title(f"A  Overall clean performance ({split_label})")
        overall_axis.grid(axis="y")
        overall_axis.legend(frameon=False, loc="lower right", fontsize=6.8)
        overall_axis.text(
            0.02,
            0.98,
            "n = 3 seeds",
            transform=overall_axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#596872",
        )

        class_x = np.arange(len(EXPECTED_LABELS), dtype=float)
        for class_index, label in enumerate(EXPECTED_LABELS):
            rows = summary.loc[
                (summary["metric"] == "per_class_f1")
                & (summary["class"] == label)
            ]
            individual = rows.loc[rows["summary"] == "individual", "value"].to_numpy()
            for offset, run, value in zip(offsets, runs, individual):
                class_axis.scatter(
                    class_index + offset,
                    value,
                    s=16,
                    color=SEED_COLORS[run.seed],
                    alpha=0.72,
                    edgecolor="none",
                    zorder=3,
                )
            mean = float(rows.loc[rows["summary"] == "mean", "value"].iloc[0])
            sd = float(
                rows.loc[rows["summary"] == "sample_sd_ddof_1", "value"].iloc[0]
            )
            class_axis.errorbar(
                class_index,
                mean,
                yerr=sd,
                color="#20272D",
                marker="D",
                markersize=4,
                capsize=3,
                linewidth=1.0,
                zorder=4,
            )
        class_axis.set_xticks(class_x, EXPECTED_LABELS, rotation=30, ha="right")
        class_axis.set_ylim(0.50, 0.91)
        class_axis.set_ylabel("F1")
        class_axis.set_title(f"B  Per-class clean F1 ({split_label})")
        class_axis.grid(axis="y")
        figure.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.22, wspace=0.30)
        return _save_figure(figure, output_dir, stem)


def _plot_training_curves(
    runs: Sequence[FormalRunEvidence], output_dir: Path
) -> list[Path]:
    with plt.style.context(_style_path()):
        figure, axes = plt.subplots(1, 3, figsize=(8.8, 3.0), sharex=True)
        panels = (
            ("train_loss", "A  Training loss", "Loss"),
            ("validation_accuracy", "B  Validation Accuracy", "Accuracy"),
            ("validation_macro_f1", "C  Validation Macro-F1", "Macro-F1"),
        )
        for axis, (column, title, ylabel) in zip(axes, panels):
            for run in runs:
                epochs = run.history["epoch"].to_numpy(dtype=int)
                values = run.history[column].to_numpy(dtype=float)
                axis.plot(
                    epochs,
                    values,
                    color=SEED_COLORS[run.seed],
                    linewidth=1.25,
                    label=f"Seed {run.seed} (best e{run.best_epoch})",
                )
                best_value = float(
                    run.history.loc[run.history["epoch"] == run.best_epoch, column].iloc[0]
                )
                axis.scatter(
                    [run.best_epoch],
                    [best_value],
                    marker="*",
                    s=45,
                    color=SEED_COLORS[run.seed],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=4,
                )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_xlim(1, EXPECTED_EPOCHS)
            axis.grid(alpha=0.65)
        axes[2].legend(frameon=False, fontsize=6.6, loc="best")
        figure.text(
            0.5,
            0.015,
            "Star: selected checkpoint; selection metric = clean PublicTest macro-F1",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#596872",
        )
        figure.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.20, wspace=0.30)
        return _save_figure(figure, output_dir, CURVES_STEM)


def _plot_confusion(
    confusion: pd.DataFrame,
    output_dir: Path,
    *,
    split_label: str = "PublicTest",
    stem: str = CONFUSION_STEM,
) -> list[Path]:
    matrix = confusion.pivot(
        index="true_class", columns="predicted_class", values="mean_fraction"
    ).loc[list(EXPECTED_LABELS), list(EXPECTED_LABELS)].to_numpy(dtype=float)
    with plt.style.context(_style_path()):
        figure, axis = plt.subplots(figsize=(5.0, 4.35))
        image = axis.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0)
        axis.set_xticks(
            np.arange(7), EXPECTED_LABELS, rotation=35, ha="right", rotation_mode="anchor"
        )
        axis.set_yticks(np.arange(7), EXPECTED_LABELS)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"Clean {split_label} confusion (mean across 3 seeds)")
        for row in range(7):
            for column in range(7):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{100.0 * value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="white" if value < 0.48 else "black",
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
        colorbar.set_label("Row-normalized fraction")
        figure.subplots_adjust(left=0.15, right=0.90, top=0.90, bottom=0.20)
        return _save_figure(figure, output_dir, stem)


def _audit_payload(
    archive_path: Path,
    evaluation_root: Path,
    runs: Sequence[FormalRunEvidence],
) -> dict[str, Any]:
    return {
        "audit_schema_version": 1,
        "status": "verified_current_formal_paper_lineage",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_lineage": {
            "name": "FER2013 clean formal E7 / Stage 8",
            "input_size": [EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE],
            "model": "ImageNet-pretrained ResNet-18 with standard stem",
            "training_mode": "clean-only",
            "training_commit": EXPECTED_TRAINING_COMMIT,
            "training_epochs": EXPECTED_EPOCHS,
            "batch_size": 128,
            "optimizer": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.001,
            "checkpoint_selection": "strict clean PublicTest macro-F1 improvement",
            "validation_split": "PublicTest (internal split name: validation)",
            "occlusion_protocol": EXPECTED_PROTOCOL,
            "evaluation_commit": EXPECTED_EVALUATION_COMMIT,
            "conditions": list(EXPECTED_CONDITIONS),
            "private_test_status": "reserved; not materialized for these paper results",
        },
        "experiment_relationship": {
            "experiment_1": "trains the three clean-only E7 checkpoints",
            "experiment_2": "reuses those exact best checkpoints and evaluates clean plus nine occlusion conditions",
            "experiment_3": "trains three new mixed models and evaluates the same ten conditions",
        },
        "legacy_lineage": {
            "input_size": [112, 112],
            "epochs": 30,
            "commit": "da889bdd818cab6403767f2f6c7d5391d8317324",
            "status": "legacy; rejected by this loader and excluded from formal paper figures",
        },
        "source_archive": {
            "path": str(archive_path),
            "sha256": _sha256_path(archive_path),
            "root": EXPECTED_ROOT,
        },
        "experiment2_clean_root": str(evaluation_root),
        "seeds": [
            {
                "seed": run.seed,
                "training_completed": run.run_metadata["status"] == "completed",
                "formal_run": True,
                "checkpoint_path": f"{archive_path}::{run.source_members['checkpoint']}",
                "checkpoint_sha256": run.checkpoint_sha256,
                "history_path": f"{archive_path}::{run.source_members['history']}",
                "history_sha256": run.source_sha256["history"],
                "resolved_config_path": f"{archive_path}::{run.source_members['resolved_config']}",
                "resolved_config_sha256": run.source_sha256["resolved_config"],
                "run_metadata_path": f"{archive_path}::{run.source_members['run_metadata']}",
                "input_size": EXPECTED_IMAGE_SIZE,
                "epoch_count": len(run.history),
                "selected_best_epoch": run.best_epoch,
                "validation_split": "PublicTest (validation)",
                "best_metrics": {
                    "accuracy": run.best_metrics["accuracy"],
                    "macro_f1": run.best_metrics["macro_f1"],
                    "loss": run.best_metrics["loss"],
                },
                "last_epoch_metrics": {
                    "accuracy": run.last_metrics["accuracy"],
                    "macro_f1": run.last_metrics["macro_f1"],
                    "loss": run.last_metrics["loss"],
                },
                "checkpoint_metadata": run.checkpoint_metadata,
                "evaluation_provenance_path": str(run.evaluation_provenance_path),
                "provenance_status": {
                    "checkpoint_sha_matches_experiment2": True,
                    "clean_metrics_match_experiment2": True,
                    "clean_per_class_match_experiment2": True,
                    "clean_confusion_match_experiment2": True,
                    "clean_predictions_match_experiment2": True,
                    "training_git_dirty": False,
                    "evaluation_git_dirty": False,
                },
            }
            for run in runs
        ],
        "evidence_checks": {
            "all_three_seeds_present": True,
            "sample_sd_ddof": 1,
            "class_order": list(EXPECTED_LABELS),
            "legacy_da889bd_inputs_read": False,
            "synthetic_inputs_read": False,
            "obsolete_gradcam_inputs_read": False,
            "training_run": False,
            "evaluation_run": False,
            "gradcam_run": False,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_formal_completion(
    clean_archive: str | Path,
    experiment2_clean_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Validate formal evidence, then generate only the missing paper figures."""
    archive_path = Path(clean_archive).expanduser().resolve()
    evaluation_root = Path(experiment2_clean_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        _fail(f"output path is not a directory: {destination}")

    runs = load_formal_lineage(archive_path, evaluation_root)
    summary = build_summary_table(runs)
    training = build_training_table(runs)
    confusion = build_confusion_table(runs)
    audit = _audit_payload(archive_path, evaluation_root, runs)

    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / f"{SUMMARY_STEM}.csv"
    curves_path = destination / f"{CURVES_STEM}.csv"
    confusion_path = destination / f"{CONFUSION_STEM}.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    training.to_csv(curves_path, index=False, float_format="%.10f")
    confusion.to_csv(confusion_path, index=False, float_format="%.10f")
    audit_path = destination / AUDIT_NAME
    _write_json(audit_path, audit)

    outputs: list[Path] = [summary_path, curves_path, confusion_path, audit_path]
    outputs.extend(_plot_summary(runs, summary, destination))
    outputs.extend(_plot_training_curves(runs, destination))
    outputs.extend(_plot_confusion(confusion, destination))
    outputs.extend(generate_framework(destination))

    manifest = {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "occlusion_fer.paper_formal_completion",
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "pandas_version": pd.__version__,
        "formal_source_audit": str(audit_path),
        "formal_source_audit_sha256": _sha256_path(audit_path),
        "statistics": {
            "seed_count": 3,
            "standard_deviation": "sample SD (ddof=1)",
            "confusion": "row-normalize within seed, then arithmetic mean across seeds",
        },
        "outputs": {
            path.name: _sha256_path(path) for path in sorted(outputs)
        },
        "prohibited_operations": {
            "training_run": False,
            "evaluation_run": False,
            "gradcam_run": False,
            "source_artifacts_modified": False,
        },
    }
    manifest_path = destination / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    return {"runs": runs, "audit": audit, "manifest": manifest, "outputs": outputs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate missing formal Experiment 1 and framework figures."
    )
    parser.add_argument("--clean-archive", required=True, metavar="FILE")
    parser.add_argument("--experiment2-clean-root", required=True, metavar="DIR")
    parser.add_argument("--output-dir", required=True, metavar="DIR")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_formal_completion(
            args.clean_archive, args.experiment2_clean_root, args.output_dir
        )
    except (FormalCompletionError, OSError, tarfile.TarError) as exc:
        print(f"paper_formal_completion error: {exc}", file=sys.stderr)
        return 1
    print(f"formal_lineage=224x224/{EXPECTED_PROTOCOL}")
    print(f"seeds={','.join(str(seed) for seed in EXPECTED_SEEDS)}")
    print(f"output_dir={Path(args.output_dir).expanduser().resolve()}")
    print(f"generated_files={len(result['outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
