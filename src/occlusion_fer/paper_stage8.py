"""Stage 8 paper figures from locked clean-only and mixed-training evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from occlusion_fer.paper_figures import (
    CLASSWISE_CONDITIONS,
    CLASSWISE_OCCLUDED_CONDITIONS,
    DOUBLE_COLUMN_WIDTH,
    EXPECTED_OCCLUSION_EVALUATION_COMMIT,
    EXPECTED_OCCLUSION_TRAINING_COMMIT,
    EXPECTED_SEEDS,
    EXPECTED_VALIDATION_SAMPLES,
    LABEL_NAMES,
    PaperFigureError,
    _assert_frames_identical,
    _build_classwise_manifest,
    _check_label_order,
    _classwise_confusion_matrix,
    _fail,
    _git_value,
    _plot_classwise_drop_heatmap,
    _plot_classwise_f1_heatmap,
    _plot_occlusion_protocol,
    _read_json,
    _require_file,
    _require_keys,
    _sha256,
    _use_paper_style,
    _validate_classwise_per_class,
    _validate_classwise_predictions,
    _validate_condition_metrics,
    _write_json,
    compute_classwise_summaries,
    load_classwise_evidence,
)


METHODS = ("Clean-only", "Mixed training")
METRICS = ("accuracy", "macro_f1")
METRIC_LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
OCCLUSION_TYPES = ("upper_face", "lower_face", "random_rectangle")
OCCLUSION_TYPE_LABELS = {
    "upper_face": "Upper-face",
    "lower_face": "Lower-face",
    "random_rectangle": "Random-rectangle",
}
SEVERITIES = (0.20, 0.30, 0.40)
METHOD_STYLES = {
    "Clean-only": {"color": "#394B59", "marker": "o", "linestyle": "-"},
    "Mixed training": {"color": "#238B57", "marker": "s", "linestyle": "--"},
}
OCCLUSION_STYLES = {
    "upper_face": {"color": "#1F77B4", "marker": "o"},
    "lower_face": {"color": "#D55E00", "marker": "s"},
    "random_rectangle": {"color": "#238B57", "marker": "^"},
}
FIGURE_STEMS = (
    "occlusion_protocol_examples",
    "occlusion_severity_accuracy",
    "occlusion_severity_macro_f1",
    "occlusion_severity_performance",
    "classwise_f1_heatmap",
    "classwise_drop_from_clean_heatmap",
    "experiment3_by_condition_comparison",
    "experiment3_clean_performance_comparison",
    "experiment3_occluded_robustness_summary",
    "experiment3_robustness_gain_heatmap",
)


@dataclass(frozen=True)
class Stage8Evidence:
    clean_results_dir: Path
    mixed_training_dir: Path
    mixed_evaluation_dir: Path
    metrics: pd.DataFrame
    source_files: tuple[Path, ...]
    provenance: tuple[dict[str, Any], ...]


def _require_directory(path: str | Path, context: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        _fail(f"{context} directory does not exist: {resolved}")
    return resolved


def _read_csv(path: Path, context: str) -> pd.DataFrame:
    source = _require_file(path)
    try:
        return pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as exc:
        _fail(f"could not read {context} CSV {source}: {exc}")


def _condition_parts(condition: str) -> tuple[str, float]:
    if condition == "clean":
        return "clean", 0.0
    for occlusion_type in OCCLUSION_TYPES:
        prefix = f"{occlusion_type}_"
        if condition.startswith(prefix):
            severity = float(condition.removeprefix(prefix))
            if severity not in SEVERITIES:
                break
            return occlusion_type, severity
    _fail(f"unsupported formal condition: {condition}")


def _validate_evaluation_provenance(
    provenance: dict[str, Any],
    *,
    seed: int,
    training_mode: str,
    context: str,
) -> None:
    required = (
        "checkpoint_sha256",
        "conditions",
        "evaluation_commit",
        "git_dirty",
        "image_height",
        "image_width",
        "manifest_sha256",
        "protocol",
        "publictest_dataset_sha256",
        "resolved_config",
        "run_role",
        "seed",
        "training_commit",
        "training_dataset_sha256",
        "training_git_dirty",
        "training_mean_sha256",
    )
    _require_keys(provenance, required, context)
    if provenance["seed"] != seed:
        _fail(f"{context} seed must be {seed}")
    if tuple(provenance["conditions"]) != CLASSWISE_CONDITIONS:
        _fail(f"{context} does not contain the locked ten-condition order")
    if provenance["evaluation_commit"] != EXPECTED_OCCLUSION_EVALUATION_COMMIT:
        _fail(f"{context} evaluation commit is not the locked Stage 8 commit")
    expected_training_commit = (
        EXPECTED_OCCLUSION_TRAINING_COMMIT
        if training_mode == "clean"
        else EXPECTED_OCCLUSION_EVALUATION_COMMIT
    )
    if provenance["training_commit"] != expected_training_commit:
        _fail(f"{context} training commit does not match {training_mode} training")
    if provenance["git_dirty"] is not False or provenance["training_git_dirty"] is not False:
        _fail(f"{context} formal evaluation and training Git states must be clean")
    if provenance["run_role"] != "formal_masked_evaluation":
        _fail(f"{context} run_role must be formal_masked_evaluation")
    if provenance["protocol"] != "occlusion-v2-224":
        _fail(f"{context} protocol must be occlusion-v2-224")
    if (provenance["image_height"], provenance["image_width"]) != (224, 224):
        _fail(f"{context} consumer image size must be 224x224")
    config = provenance["resolved_config"]
    if not isinstance(config, dict):
        _fail(f"{context} resolved_config must be an object")
    try:
        configured_mode = config["training"]["mode"]
        configured_seed = config["training"]["seed"]
        configured_conditions = tuple(config["occlusion"]["evaluation"]["conditions"])
        configured_split = config["occlusion"]["evaluation"]["split"]
        protocol = config["occlusion"]["protocol"]
    except (KeyError, TypeError) as exc:
        _fail(f"{context} resolved_config is missing locked evaluation fields: {exc}")
    if configured_mode != training_mode or configured_seed != seed:
        _fail(f"{context} resolved training mode or seed is inconsistent")
    if configured_conditions != CLASSWISE_CONDITIONS or configured_split != "validation":
        _fail(f"{context} resolved evaluation conditions or split are inconsistent")
    expected_protocol = {
        "algorithm_version": "occlusion-v2-224",
        "image_size": 224,
        "types": list(OCCLUSION_TYPES),
        "ratios": ["0.20", "0.30", "0.40"],
        "fill_source": "training_split_global_mean",
        "evaluation_mask_seed": 20260804,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            _fail(f"{context} protocol field {key} is not locked to {expected!r}")


def _validate_condition_provenance(
    condition_provenance: dict[str, Any],
    evaluation_provenance: dict[str, Any],
    condition: str,
    context: str,
) -> None:
    if condition_provenance.get("condition") != condition:
        _fail(f"{context} condition does not match its filename")
    for key in (
        "checkpoint_sha256",
        "evaluation_commit",
        "git_dirty",
        "image_height",
        "image_width",
        "manifest_sha256",
        "protocol",
        "publictest_dataset_sha256",
        "resolved_config",
        "run_role",
        "seed",
        "training_commit",
        "training_dataset_sha256",
        "training_git_dirty",
        "training_mean_sha256",
    ):
        if condition_provenance.get(key) != evaluation_provenance.get(key):
            _fail(f"{context} field {key} differs from evaluation_provenance.json")


def _validate_mixed_training_run(
    run_dir: Path,
    provenance: dict[str, Any],
    seed: int,
    supplemental_clean_metrics: dict[str, Any],
) -> tuple[Path, ...]:
    metadata_path = _require_file(run_dir / "run_metadata.json")
    best_path = _require_file(run_dir / "best.pt")
    best_metrics_path = _require_file(run_dir / "validation" / "best_metrics.json")
    metadata = _read_json(metadata_path)
    _require_keys(
        metadata,
        ("seed", "status", "training_mode", "git_commit", "git_dirty"),
        str(metadata_path),
    )
    if (
        metadata["seed"] != seed
        or metadata["status"] != "completed"
        or metadata["training_mode"] != "mixed"
        or metadata["git_commit"] != EXPECTED_OCCLUSION_EVALUATION_COMMIT
        or metadata["git_dirty"] is not False
    ):
        _fail(f"{metadata_path} is not the locked completed mixed-training run")
    if _sha256(best_path) != provenance["checkpoint_sha256"]:
        _fail(f"{best_path} SHA-256 does not match supplemental evaluation provenance")
    best_metrics = _read_json(best_metrics_path)
    if best_metrics != supplemental_clean_metrics:
        _fail(
            f"seed {seed} supplemental clean metrics do not exactly match the "
            "mixed-training best validation metrics"
        )
    return metadata_path, best_path, best_metrics_path


def load_stage8_evidence(
    clean_results_dir: str | Path,
    mixed_training_dir: str | Path,
    mixed_evaluation_dir: str | Path,
) -> Stage8Evidence:
    """Load and validate the formal overall-metric evidence for all six runs."""
    clean_root = _require_directory(clean_results_dir, "clean-only Stage 8 results")
    mixed_training_root = _require_directory(mixed_training_dir, "mixed-training results")
    mixed_evaluation_root = _require_directory(
        mixed_evaluation_dir, "mixed supplemental evaluation"
    )
    rows: list[dict[str, Any]] = []
    source_files: list[Path] = []
    provenances: list[dict[str, Any]] = []
    identity_reference: dict[str, Any] | None = None
    sample_reference: tuple[np.ndarray, np.ndarray] | None = None

    for method, training_mode, root in (
        ("Clean-only", "clean", clean_root / "clean"),
        ("Mixed training", "mixed", mixed_evaluation_root),
    ):
        for seed in EXPECTED_SEEDS:
            run_dir = _require_directory(root / f"seed{seed}", f"{method} seed {seed}")
            provenance_path = _require_file(run_dir / "evaluation_provenance.json")
            provenance = _read_json(provenance_path)
            _validate_evaluation_provenance(
                provenance,
                seed=seed,
                training_mode=training_mode,
                context=str(provenance_path),
            )
            current_identity = {
                key: provenance[key]
                for key in (
                    "manifest_sha256",
                    "publictest_dataset_sha256",
                    "training_dataset_sha256",
                    "training_mean_sha256",
                    "protocol",
                    "image_height",
                    "image_width",
                )
            }
            if identity_reference is None:
                identity_reference = current_identity
            elif current_identity != identity_reference:
                _fail(f"{provenance_path} does not share the locked evaluation identity")
            source_files.append(provenance_path)
            provenances.append(provenance)

            clean_metrics_for_training_check: dict[str, Any] | None = None
            for condition in CLASSWISE_CONDITIONS:
                prefix = run_dir / "conditions" / condition
                metrics_path = _require_file(prefix.with_name(f"{condition}_metrics.json"))
                per_class_path = _require_file(
                    prefix.with_name(f"{condition}_per_class_metrics.csv")
                )
                confusion_path = _require_file(
                    prefix.with_name(f"{condition}_confusion_matrix.csv")
                )
                predictions_path = _require_file(
                    prefix.with_name(f"{condition}_predictions.csv")
                )
                condition_provenance_path = _require_file(
                    prefix.with_name(f"{condition}_provenance.json")
                )
                metrics = _read_json(metrics_path)
                per_class = _validate_classwise_per_class(
                    _read_csv(per_class_path, "per-class metrics"), str(per_class_path)
                )
                _, confusion = _classwise_confusion_matrix(
                    _read_csv(confusion_path, "confusion matrix"), str(confusion_path)
                )
                predictions, reconstructed = _validate_classwise_predictions(
                    _read_csv(predictions_path, "predictions"),
                    condition,
                    str(predictions_path),
                )
                if not np.array_equal(confusion, reconstructed):
                    _fail(f"{confusion_path} cannot be reconstructed from predictions")
                _validate_condition_metrics(
                    metrics, condition, per_class, confusion, str(metrics_path)
                )
                condition_provenance = _read_json(condition_provenance_path)
                _validate_condition_provenance(
                    condition_provenance,
                    provenance,
                    condition,
                    str(condition_provenance_path),
                )
                sample_identity = (
                    predictions["sample_id"].to_numpy(dtype=np.int64),
                    predictions["true_label"].to_numpy(dtype=np.int64),
                )
                if sample_reference is None:
                    sample_reference = sample_identity
                elif not np.array_equal(sample_identity[0], sample_reference[0]) or not np.array_equal(
                    sample_identity[1], sample_reference[1]
                ):
                    _fail(f"{predictions_path} sample order or labels differ from formal evidence")
                occlusion_type, severity = _condition_parts(condition)
                rows.append(
                    {
                        "method": method,
                        "training_mode": training_mode,
                        "seed": seed,
                        "condition": condition,
                        "occlusion_type": occlusion_type,
                        "severity": severity,
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                    }
                )
                source_files.extend(
                    (
                        metrics_path,
                        per_class_path,
                        confusion_path,
                        predictions_path,
                        condition_provenance_path,
                    )
                )
                if condition == "clean":
                    clean_metrics_for_training_check = metrics

            if method == "Mixed training":
                if clean_metrics_for_training_check is None:
                    _fail(f"mixed seed {seed} is missing clean metrics")
                source_files.extend(
                    _validate_mixed_training_run(
                        _require_directory(
                            mixed_training_root / "mixed" / f"seed{seed}",
                            f"mixed-training seed {seed}",
                        ),
                        provenance,
                        seed,
                        clean_metrics_for_training_check,
                    )
                )

    metrics_frame = pd.DataFrame(rows)
    expected_rows = len(METHODS) * len(EXPECTED_SEEDS) * len(CLASSWISE_CONDITIONS)
    if len(metrics_frame) != expected_rows:
        _fail(f"Stage 8 evidence must contain exactly {expected_rows} metric rows")
    return Stage8Evidence(
        clean_results_dir=clean_root,
        mixed_training_dir=mixed_training_root,
        mixed_evaluation_dir=mixed_evaluation_root,
        metrics=metrics_frame,
        source_files=tuple(dict.fromkeys(source_files)),
        provenance=tuple(provenances),
    )


def compute_stage8_tables(evidence: Stage8Evidence) -> dict[str, pd.DataFrame]:
    """Compute all requested tables with seed as the unit of replication."""
    comparison_rows: list[dict[str, Any]] = []
    for (method, condition, occlusion_type, severity), group in evidence.metrics.groupby(
        ["method", "condition", "occlusion_type", "severity"], sort=False
    ):
        if tuple(sorted(group["seed"].astype(int))) != EXPECTED_SEEDS:
            _fail(f"{method} {condition} does not contain all three formal seeds")
        for metric in METRICS:
            values = group[metric].astype(float)
            comparison_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "occlusion_type": occlusion_type,
                    "severity": severity,
                    "metric": metric,
                    "mean": values.mean(),
                    "sample_sd": values.std(ddof=1),
                    "n_seeds": len(values),
                }
            )
    comparison = pd.DataFrame(comparison_rows)
    method_order = {method: index for index, method in enumerate(METHODS)}
    condition_order = {condition: index for index, condition in enumerate(CLASSWISE_CONDITIONS)}
    metric_order = {metric: index for index, metric in enumerate(METRICS)}
    comparison = comparison.sort_values(
        ["method", "condition", "metric"],
        key=lambda series: series.map(
            method_order
            if series.name == "method"
            else condition_order
            if series.name == "condition"
            else metric_order
        ),
    ).reset_index(drop=True)

    clean_only_summary = comparison.loc[comparison["method"] == "Clean-only"].copy()
    clean_only_summary.insert(1, "training_strategy", "clean-only")
    clean_only_summary["split"] = "validation/PublicTest"

    occluded = evidence.metrics.loc[evidence.metrics["condition"] != "clean"]
    per_seed_rows: list[dict[str, Any]] = []
    for (method, seed), group in occluded.groupby(["method", "seed"], sort=False):
        if len(group) != len(CLASSWISE_OCCLUDED_CONDITIONS) or set(group["condition"]) != set(
            CLASSWISE_OCCLUDED_CONDITIONS
        ):
            _fail(f"{method} seed {seed} does not contain all nine occluded conditions")
        for metric in METRICS:
            per_seed_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "metric": metric,
                    "seed_mean_across_9_conditions": group[metric].astype(float).mean(),
                }
            )
    per_seed = pd.DataFrame(per_seed_rows)
    robustness_rows: list[dict[str, Any]] = []
    for (method, metric), group in per_seed.groupby(["method", "metric"], sort=False):
        values = group.set_index("seed")["seed_mean_across_9_conditions"]
        robustness_rows.append(
            {
                "method": method,
                "metric": metric,
                "mean": values.mean(),
                "sample_sd": values.std(ddof=1),
                "n_seeds": len(values),
                "seed42_mean": values.loc[42],
                "seed123_mean": values.loc[123],
                "seed2026_mean": values.loc[2026],
                "aggregation": (
                    "unweighted mean across 9 occluded conditions within each seed; "
                    "then mean and sample SD across 3 seeds"
                ),
            }
        )
    robustness = pd.DataFrame(robustness_rows).sort_values(
        ["method", "metric"],
        key=lambda series: series.map(method_order if series.name == "method" else metric_order),
    ).reset_index(drop=True)

    paired = evidence.metrics.pivot_table(
        index=["seed", "condition", "occlusion_type", "severity"],
        columns="method",
        values=list(METRICS),
        aggfunc="first",
    )
    if paired.isna().any().any():
        _fail("mixed-versus-clean evidence is not paired for every seed and condition")
    gain_rows: list[dict[str, Any]] = []
    for condition in CLASSWISE_OCCLUDED_CONDITIONS:
        occlusion_type, severity = _condition_parts(condition)
        condition_rows = paired.xs(condition, level="condition")
        for metric in METRICS:
            gains = (
                condition_rows[(metric, "Mixed training")]
                - condition_rows[(metric, "Clean-only")]
            )
            gains.index = gains.index.get_level_values("seed")
            gain_rows.append(
                {
                    "condition": condition,
                    "occlusion_type": occlusion_type,
                    "severity": severity,
                    "metric": metric,
                    "mean_gain": gains.mean(),
                    "sample_sd": gains.std(ddof=1),
                    "n_seeds": len(gains),
                    "seed42_gain": gains.loc[42],
                    "seed123_gain": gains.loc[123],
                    "seed2026_gain": gains.loc[2026],
                    "definition": "Mixed training minus Clean-only, paired within seed",
                }
            )
    gain = pd.DataFrame(gain_rows)
    return {
        "occlusion_severity_summary": clean_only_summary,
        "experiment3_comparison_summary": comparison,
        "experiment3_robustness_summary": robustness,
        "experiment3_robustness_gain": gain,
    }


def _save_pdf_png(figure: plt.Figure, stem: str, output_dir: Path) -> list[str]:
    names: list[str] = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, format=extension, dpi=300 if extension == "png" else None)
        if not path.is_file() or path.stat().st_size == 0:
            _fail(f"figure export failed or is empty: {path}")
        names.append(path.name)
    plt.close(figure)
    return names


def _format_metric_axis(axis: plt.Axes, metric: str) -> None:
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", alpha=0.25)


def _plot_clean_severity_metric(
    summary: pd.DataFrame, metric: str, output_dir: Path
) -> list[str]:
    figure, axis = plt.subplots(figsize=(3.5, 3.15))
    clean = summary.loc[(summary["condition"] == "clean") & (summary["metric"] == metric)].iloc[0]
    axis.axhline(clean["mean"], color="#58656E", linestyle="--", linewidth=1.1, label="Clean reference")
    axis.axhspan(
        clean["mean"] - clean["sample_sd"],
        clean["mean"] + clean["sample_sd"],
        color="#58656E",
        alpha=0.10,
        linewidth=0,
    )
    for occlusion_type in OCCLUSION_TYPES:
        subset = summary.loc[
            (summary["occlusion_type"] == occlusion_type) & (summary["metric"] == metric)
        ].sort_values("severity")
        style = OCCLUSION_STYLES[occlusion_type]
        axis.errorbar(
            subset["severity"],
            subset["mean"],
            yerr=subset["sample_sd"],
            label=OCCLUSION_TYPE_LABELS[occlusion_type],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.35,
            capsize=2.5,
        )
    axis.set_xticks(SEVERITIES, [f"{value:.2f}" for value in SEVERITIES])
    axis.set_xlabel("Target occlusion ratio")
    _format_metric_axis(axis, metric)
    axis.legend(fontsize=7.2, frameon=False)
    figure.tight_layout()
    return _save_pdf_png(figure, f"occlusion_severity_{metric}", output_dir)


def _draw_clean_severity_panel(axis: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    clean = summary.loc[(summary["condition"] == "clean") & (summary["metric"] == metric)].iloc[0]
    axis.axhline(clean["mean"], color="#58656E", linestyle="--", linewidth=1.0, label="Clean reference")
    axis.axhspan(
        clean["mean"] - clean["sample_sd"],
        clean["mean"] + clean["sample_sd"],
        color="#58656E",
        alpha=0.10,
        linewidth=0,
    )
    for occlusion_type in OCCLUSION_TYPES:
        subset = summary.loc[
            (summary["occlusion_type"] == occlusion_type) & (summary["metric"] == metric)
        ].sort_values("severity")
        style = OCCLUSION_STYLES[occlusion_type]
        axis.errorbar(
            subset["severity"], subset["mean"], yerr=subset["sample_sd"],
            label=OCCLUSION_TYPE_LABELS[occlusion_type], color=style["color"],
            marker=style["marker"], linewidth=1.35, capsize=2.5,
        )
    axis.set_xticks(SEVERITIES, [f"{value:.2f}" for value in SEVERITIES])
    axis.set_xlabel("Target occlusion ratio")
    axis.set_title(METRIC_LABELS[metric])
    _format_metric_axis(axis, metric)


def _plot_clean_severity_performance(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.15), sharex=True)
    for axis, metric in zip(axes, METRICS, strict=True):
        _draw_clean_severity_panel(axis, summary, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.6)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_pdf_png(figure, "occlusion_severity_performance", output_dir)


def _plot_by_condition_comparison(summary: pd.DataFrame, output_dir: Path) -> list[str]:
    figure, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH, 5.35), sharex=True)
    for row, metric in enumerate(METRICS):
        for column, occlusion_type in enumerate(OCCLUSION_TYPES):
            axis = axes[row, column]
            for method in METHODS:
                subset = summary.loc[
                    (summary["method"] == method)
                    & (summary["metric"] == metric)
                    & (summary["occlusion_type"] == occlusion_type)
                ].sort_values("severity")
                style = METHOD_STYLES[method]
                axis.errorbar(
                    subset["severity"], subset["mean"], yerr=subset["sample_sd"],
                    label=method, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.35, capsize=2.2,
                )
            if row == 0:
                axis.set_title(OCCLUSION_TYPE_LABELS[occlusion_type])
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(SEVERITIES, [f"{value:.2f}" for value in SEVERITIES])
            if row == 1:
                axis.set_xlabel("Target ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_pdf_png(figure, "experiment3_by_condition_comparison", output_dir)


def _plot_grouped_metric_bars(
    summary: pd.DataFrame,
    *,
    stem: str,
    output_dir: Path,
    title: str | None = None,
) -> list[str]:
    figure, axis = plt.subplots(figsize=(3.8, 3.25))
    x = np.arange(len(METRICS), dtype=float)
    width = 0.34
    for method_index, method in enumerate(METHODS):
        subset = summary.loc[summary["method"] == method].set_index("metric").loc[list(METRICS)]
        positions = x + (method_index - 0.5) * width
        axis.bar(
            positions,
            subset["mean"],
            width,
            yerr=subset["sample_sd"],
            capsize=3,
            label=method,
            color=METHOD_STYLES[method]["color"],
            alpha=0.92,
        )
    axis.set_xticks(x, [METRIC_LABELS[metric] for metric in METRICS])
    axis.set_ylabel("Score")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", alpha=0.25)
    if title:
        axis.set_title(title)
    upper = float((summary["mean"] + summary["sample_sd"]).max())
    axis.set_ylim(0.0, min(1.0, upper + 0.12))
    axis.legend(loc="upper right", frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_pdf_png(figure, stem, output_dir)


def _plot_gain_heatmap(gain: pd.DataFrame, output_dir: Path) -> list[str]:
    matrices = []
    for metric in METRICS:
        matrix = (
            gain.loc[gain["metric"] == metric]
            .pivot(index="occlusion_type", columns="severity", values="mean_gain")
            .reindex(index=OCCLUSION_TYPES, columns=SEVERITIES)
        )
        matrices.append(matrix)
    limit = max(float(np.abs(matrix.to_numpy()).max()) for matrix in matrices)
    limit = max(limit, 0.001)
    normalization = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 3.05))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.055),
        left=0.13,
        right=0.94,
        bottom=0.18,
        top=0.86,
        wspace=0.58,
    )
    axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]))
    colorbar_axis = figure.add_subplot(grid[0, 2])
    image = None
    for axis, metric, matrix in zip(axes, METRICS, matrices, strict=True):
        image = axis.imshow(matrix, cmap="RdBu", norm=normalization, aspect="auto", interpolation="nearest")
        axis.set_title(f"{METRIC_LABELS[metric]} gain")
        axis.set_xticks(np.arange(3), [f"{value:.2f}" for value in SEVERITIES])
        axis.set_yticks(
            np.arange(3),
            ("Upper-face", "Lower-face", "Random\nrectangle"),
        )
        axis.set_xlabel("Target occlusion ratio")
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row in range(3):
            for column in range(3):
                value = float(matrix.iloc[row, column])
                normalized = normalization(value)
                axis.text(
                    column, row, f"{value:+.3f}", ha="center", va="center",
                    color="white" if normalized < 0.22 or normalized > 0.78 else "#111111",
                    fontsize=7.2,
                )
    if image is None:
        _fail("robustness gain heatmap has no data")
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Mixed training - Clean-only")
    return _save_pdf_png(figure, "experiment3_robustness_gain_heatmap", output_dir)


_PROTOCOL_WORKER = r'''
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from occlusion_fer.config import load_config
from occlusion_fer.mask_manifest import load_manifest_v2
from occlusion_fer.occlusion import (
    V2_MASKED_CONDITIONS,
    apply_evaluation_mask_v2,
    normalized_fill_vector_v2,
)
from occlusion_fer.permitted_splits import (
    load_stage_b_source,
    permitted_splits_to_data,
    validate_official_stage_b_sources,
)
from occlusion_fer.torch_data import Fer2013TorchDataset
from occlusion_fer.training_mean import load_training_mean_v2, training_mean_v2_sha256

data_path, config_path, mean_path, manifest_path, sidecar_path, sample_text, arrays_path, metadata_path = sys.argv[1:]
sample_id = int(sample_text)
config = load_config(config_path)
if config.dataset.image_size != 224 or config.occlusion is None:
    raise ValueError("formal protocol config must use the 224x224 occlusion integration")
protocol = config.occlusion.protocol
if protocol.algorithm_version != "occlusion-v2-224":
    raise ValueError("formal protocol config is not occlusion-v2-224")
source = load_stage_b_source(data_path)
validate_official_stage_b_sources(source)
mean = load_training_mean_v2(mean_path)
mean_sha = training_mean_v2_sha256(mean)
fill_vector = normalized_fill_vector_v2(
    mean, training_dataset_sha256=source.training.dataset_sha256
)
manifest_rows, envelope = load_manifest_v2(
    manifest_path,
    sidecar_path,
    publictest_dataset_sha256=source.publictest.dataset_sha256,
    training_mean_artifact_sha256=mean_sha,
    require_official=True,
)
sample_rows = {row.condition: row for row in manifest_rows if row.sample_id == sample_id}
if tuple(sample_rows) != tuple(V2_MASKED_CONDITIONS):
    raise ValueError("sample is missing one or more locked v2 manifest rows")
matching = [record for record in source.publictest.records if record.sample_id == sample_id]
if len(matching) != 1:
    raise ValueError("sample_id must identify exactly one PublicTest record")
record = matching[0]
dataset = Fer2013TorchDataset(
    permitted_splits_to_data(source), "validation", image_size=224, normalize_imagenet=True
)
dataset_index = next(
    index for index, candidate in enumerate(source.publictest.records)
    if candidate.sample_id == sample_id
)
clean, label, returned_id = dataset[dataset_index]
if label != record.label or returned_id != sample_id:
    raise ValueError("sample identity changed in the formal tensor pipeline")
snapshot = clean.clone()
arrays = {"clean": clean.numpy()}
masks = []
geometry_fields = (
    "occlusion_type", "ratio_token", "actual_ratio", "top", "left", "height", "width",
    "masked_pixel_count", "total_pixel_count", "image_height", "image_width",
)
for condition in V2_MASKED_CONDITIONS:
    masked, metadata = apply_evaluation_mask_v2(clean, sample_id, condition, fill_vector)
    manifest_row = sample_rows[condition]
    for field in geometry_fields:
        if getattr(metadata, field) != getattr(manifest_row, field):
            raise ValueError(f"{condition} {field} differs from the locked manifest")
    if metadata.seed != manifest_row.evaluation_mask_seed:
        raise ValueError(f"{condition} seed differs from the locked manifest")
    arrays[condition] = masked.numpy()
    item = asdict(metadata)
    item["mask_identifier"] = (
        f"validation:{sample_id}:{condition}:seed={metadata.seed}:{metadata.algorithm_version}"
    )
    item["manifest_row_key"] = {"sample_id": sample_id, "condition": condition}
    item["mask_application_function"] = "occlusion_fer.occlusion.apply_evaluation_mask_v2"
    masks.append(item)
if not torch.equal(clean, snapshot):
    raise ValueError("source image changed while applying formal masks")
np.savez_compressed(arrays_path, **arrays)
source_image = np.asarray(record.pixels, dtype=np.uint8).reshape(48, 48)
metadata = {
    "sample": {
        "sample_identifier": sample_id,
        "dataset_label": record.label,
        "model_input_shape": list(clean.shape),
        "source_image_shape": list(source_image.shape),
        "source_image_dtype": str(source_image.dtype),
        "source_image_sha256": hashlib.sha256(source_image.tobytes()).hexdigest(),
    },
    "protocol": {
        "algorithm_version": protocol.algorithm_version,
        "image_size": protocol.image_size,
        "occlusion_types": list(protocol.types),
        "target_ratios": list(protocol.ratios),
        "fill_source": protocol.fill_source,
        "raw_training_mean": mean.raw_training_mean,
        "evaluation_mask_seed": protocol.evaluation_mask_seed,
        "coordinate_convention": protocol.coordinate_convention,
        "manifest_sha256": envelope.manifest_sha256,
        "training_mean_sha256": mean_sha,
        "training_dataset_sha256": source.training.dataset_sha256,
        "publictest_dataset_sha256": source.publictest.dataset_sha256,
    },
    "masks": masks,
}
Path(metadata_path).write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
'''


def generate_stage8_protocol_examples(
    *,
    data_path: str | Path,
    config_path: str | Path,
    mean_path: str | Path,
    manifest_path: str | Path,
    code_root: str | Path,
    sample_id: int,
    output_dir: Path,
    command: str,
) -> dict[str, Any]:
    if type(sample_id) is not int or sample_id <= 0:
        _fail("protocol sample_id must be a positive integer")
    resolved_inputs = {
        "data": _require_file(Path(data_path)),
        "occlusion_config": _require_file(Path(config_path)),
        "training_mean_artifact": _require_file(Path(mean_path)),
        "evaluation_mask_manifest": _require_file(Path(manifest_path)),
    }
    sidecar = resolved_inputs["evaluation_mask_manifest"].with_suffix(".json")
    resolved_inputs["evaluation_mask_manifest_sidecar"] = _require_file(sidecar)
    locked_code_root = _require_directory(code_root, "locked Stage 8 code")
    try:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=locked_code_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        code_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=locked_code_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"could not validate locked Stage 8 code root: {exc}")
    if code_commit != EXPECTED_OCCLUSION_EVALUATION_COMMIT or code_status:
        _fail("Stage 8 protocol code root must be clean at c1c9187")

    with tempfile.TemporaryDirectory(prefix="occlusion-fer-stage8-protocol-") as temporary:
        temporary_path = Path(temporary)
        arrays_path = temporary_path / "protocol_arrays.npz"
        worker_metadata_path = temporary_path / "protocol_metadata.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(locked_code_root / "src")
        worker_command = [
            sys.executable,
            "-c",
            _PROTOCOL_WORKER,
            str(resolved_inputs["data"]),
            str(resolved_inputs["occlusion_config"]),
            str(resolved_inputs["training_mean_artifact"]),
            str(resolved_inputs["evaluation_mask_manifest"]),
            str(resolved_inputs["evaluation_mask_manifest_sidecar"]),
            str(sample_id),
            str(arrays_path),
            str(worker_metadata_path),
        ]
        try:
            completed = subprocess.run(
                worker_command,
                cwd=locked_code_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            _fail(f"formal Stage 8 masking worker failed: {detail}")
        if completed.stderr.strip():
            _fail(f"formal Stage 8 masking worker wrote stderr: {completed.stderr.strip()}")
        worker_metadata = _read_json(worker_metadata_path)
        try:
            import torch

            with np.load(arrays_path) as arrays:
                clean = torch.from_numpy(arrays["clean"].copy())
                masked = {
                    condition: torch.from_numpy(arrays[condition].copy())
                    for condition in CLASSWISE_OCCLUDED_CONDITIONS
                }
        except (OSError, KeyError, ValueError) as exc:
            _fail(f"could not read formal protocol worker arrays: {exc}")

    _use_paper_style()
    pdf_path, png_path = _plot_occlusion_protocol(clean, masked, output_dir)
    sample = dict(worker_metadata["sample"])
    sample.update(
        {
            "identifier_definition": "physical one-based CSV line number including header",
            "split": "validation",
            "official_split": "PublicTest",
            "dataset_label_name": LABEL_NAMES[sample["dataset_label"]],
        }
    )
    metadata = {
        "metadata_schema_version": 2,
        "figure_type": "methodology_protocol",
        "contains_experimental_results": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "sample": sample,
        "original": {
            "mask_type": "original_unmasked",
            "severity": None,
            "seed": None,
            "mask_identifier": None,
        },
        "protocol": {
            **worker_metadata["protocol"],
            "image_preparation_function": "occlusion_fer.torch_data.Fer2013TorchDataset",
            "mask_application_function": "occlusion_fer.occlusion.apply_evaluation_mask_v2",
            "formal_code_root": str(locked_code_root),
            "formal_code_commit": code_commit,
        },
        "masks": worker_metadata["masks"],
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in resolved_inputs.items()
        },
        "outputs": {
            pdf_path.name: _sha256(pdf_path),
            png_path.name: _sha256(png_path),
        },
    }
    metadata_path = output_dir / "occlusion_protocol_examples_metadata.json"
    _write_json(metadata_path, metadata)
    return metadata


def _write_markdown_outputs(output_dir: Path, sources: dict[str, str]) -> None:
    index = f"""# Stage 8 Figure Index

This index is the authoritative map for the core paper figures generated from the locked formal artifacts. The labels Clean-only and Mixed training are the only method names used in comparison figures.

| Suggested no. | File stem | Purpose | Formal data source | Placement |
|---|---|---|---|---|
| Figure 1 | `occlusion_protocol_examples` | Method: locked synthetic-occlusion protocol | `{sources['protocol_dir']}` plus the local FER2013 CSV | Main text |
| Figure 2 | `occlusion_severity_performance` | Main clean-only performance across occlusion type and target ratio | `{sources['clean_results']}` | Main text |
| Figure S1 | `occlusion_severity_accuracy` | Single-metric clean-only accuracy view | `{sources['clean_results']}` | Appendix |
| Figure S2 | `occlusion_severity_macro_f1` | Single-metric clean-only macro-F1 view | `{sources['clean_results']}` | Appendix |
| Figure 3 | `classwise_f1_heatmap` | Class-wise absolute F1 across all ten conditions | `{sources['clean_archive']}` | Main text |
| Figure S3 | `classwise_drop_from_clean_heatmap` | Class-wise clean-to-occluded F1 drop | `{sources['clean_archive']}` | Appendix |
| Figure 4 | `experiment3_by_condition_comparison` | Clean-only versus Mixed training by condition | `{sources['clean_results']}` and `{sources['mixed_evaluation']}` | Main text |
| Figure S4 | `experiment3_clean_performance_comparison` | Clean-condition method comparison | Same formal evaluation trees as Figure 4 | Appendix |
| Figure 5 | `experiment3_occluded_robustness_summary` | Strict nine-condition occluded summary | Same formal evaluation trees as Figure 4 | Main text |
| Figure 6 | `experiment3_robustness_gain_heatmap` | Paired Mixed training minus Clean-only gains | Same formal evaluation trees as Figure 4 | Main text |

Each stem has a PDF and a 300 dpi PNG. The generated CSV tables contain the plotted means and sample standard deviations; no result value is entered manually.
"""
    (output_dir / "figure_index.md").write_text(index, encoding="utf-8")

    captions = """# Draft Figure Captions

## Figure 1: Occlusion protocol examples

Examples of the locked synthetic-occlusion protocol applied to one FER2013 PublicTest image. The same grayscale sample is shown unmasked and with upper-face, lower-face, and random-rectangle masks at target ratios 0.20, 0.30, and 0.40. Masks are generated by the formal `occlusion-v2-224` implementation using the Training-split pixel mean as the fill value.

## Figure 2: Clean-only performance across occlusion severity

Accuracy and macro-F1 of the Clean-only ResNet-18 on FER2013 PublicTest under the three synthetic occlusion types. Points and error bars show the mean and sample standard deviation across seeds 42, 123, and 2026; the dashed line and band show the corresponding clean-condition reference. The panels summarize dataset-defined facial-expression classification performance under the locked evaluation protocol.

## Figure S1: Clean-only accuracy across occlusion severity

Accuracy of the Clean-only model across upper-face, lower-face, and random-rectangle occlusion at the three target ratios. Values are means across the three formal seeds, with error bars denoting sample standard deviation and the dashed reference denoting clean-condition performance.

## Figure S2: Clean-only macro-F1 across occlusion severity

Macro-F1 of the Clean-only model across the nine occluded evaluation conditions. Values are means across the three formal seeds, with sample-standard-deviation error bars and a clean-condition reference.

## Figure 3: Class-wise F1 across evaluation conditions

Mean per-class F1 for the Clean-only model across the clean condition and all nine synthetic-occlusion conditions. Rows follow the fixed FER2013 label order, and each cell is averaged across the three formal seeds. The figure describes dataset-label classification behavior rather than internal emotional state.

## Figure S3: Class-wise F1 drop from clean

Mean within-seed reduction in per-class F1 from the clean condition to each occluded condition for the Clean-only model. Drops are computed for each formal seed before averaging, preserving the paired clean reference for every model run.

## Figure 4: Clean-only and Mixed training by condition

Accuracy and macro-F1 for Clean-only and Mixed training across occlusion type and target ratio on FER2013 PublicTest. Curves show the mean across the three formal seeds and error bars show sample standard deviation. Both methods are evaluated with the same locked masks and evaluation protocol.

## Figure S4: Clean-condition performance comparison

Clean-condition accuracy and macro-F1 for the Clean-only and Mixed training strategies. Bars show the mean across the three formal seeds and error bars show sample standard deviation, isolating clean-input performance from the occluded-condition comparisons.

## Figure 5: Occluded-condition robustness summary

Summary accuracy and macro-F1 across the nine occluded conditions for Clean-only and Mixed training. For each seed, the nine conditions are first averaged without weighting; the displayed mean and sample standard deviation are then computed across the three seed-level averages. This aggregation keeps the formal seed, rather than an individual seed-condition pair, as the unit of replication.

## Figure 6: Paired robustness gain

Mean paired gain from Mixed training relative to Clean-only for every occlusion type, target ratio, and metric. Differences are computed within each seed before averaging across the three formal seeds; positive cells favor Mixed training under the corresponding synthetic condition. The figure reports in-dataset protocol-specific differences and does not imply real-world or cross-dataset robustness.
"""
    (output_dir / "figure_captions_draft.md").write_text(captions, encoding="utf-8")


def _build_figure_manifest(
    output_dir: Path,
    sources: dict[str, str],
    command: str,
) -> dict[str, Any]:
    definitions = {
        "occlusion_protocol_examples": ("method", [sources["protocol_dir"], sources["fer2013_csv"]], []),
        "occlusion_severity_accuracy": ("main_result", [sources["clean_results"]], ["accuracy"]),
        "occlusion_severity_macro_f1": ("main_result", [sources["clean_results"]], ["macro_f1"]),
        "occlusion_severity_performance": ("main_result", [sources["clean_results"]], ["accuracy", "macro_f1"]),
        "classwise_f1_heatmap": ("class_wise", [sources["clean_archive"], sources["baseline_archive"]], ["per_class_f1"]),
        "classwise_drop_from_clean_heatmap": ("class_wise", [sources["clean_archive"], sources["baseline_archive"]], ["per_class_f1_drop"]),
        "experiment3_by_condition_comparison": ("comparison", [sources["clean_results"], sources["mixed_evaluation"], sources["mixed_training"]], ["accuracy", "macro_f1"]),
        "experiment3_clean_performance_comparison": ("comparison", [sources["clean_results"], sources["mixed_evaluation"], sources["mixed_training"]], ["accuracy", "macro_f1"]),
        "experiment3_occluded_robustness_summary": ("comparison", [sources["clean_results"], sources["mixed_evaluation"], sources["mixed_training"]], ["accuracy", "macro_f1"]),
        "experiment3_robustness_gain_heatmap": ("comparison", [sources["clean_results"], sources["mixed_evaluation"], sources["mixed_training"]], ["accuracy_gain", "macro_f1_gain"]),
    }
    figures = []
    for stem in FIGURE_STEMS:
        figure_type, data_sources, metrics = definitions[stem]
        for extension in ("pdf", "png"):
            path = output_dir / f"{stem}.{extension}"
            figures.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "figure_type": figure_type,
                    "data_sources": data_sources,
                    "metrics": metrics,
                    "status": "generated" if path.is_file() and path.stat().st_size > 0 else "missing",
                    "sha256": _sha256(path) if path.is_file() else None,
                }
            )
    style_path = Path(__file__).with_name("paper.mplstyle").resolve()
    module_path = Path(__file__).resolve()
    return {
        "manifest_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "generator": {
            "module_path": str(module_path),
            "module_sha256": _sha256(module_path),
            "style_path": str(style_path),
            "style_sha256": _sha256(style_path),
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
            "pandas_version": pd.__version__,
            "git_commit": _git_value(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git_value(["status", "--porcelain"])),
        },
        "formal_evaluation_commit": EXPECTED_OCCLUSION_EVALUATION_COMMIT,
        "formal_seeds": list(EXPECTED_SEEDS),
        "split": "validation/PublicTest",
        "conditions": list(CLASSWISE_CONDITIONS),
        "figures": figures,
        "tables": sorted(
            path.name for path in output_dir.glob("*.csv") if path.is_file()
        ),
        "documentation": ["figure_index.md", "figure_captions_draft.md"],
    }


def generate_stage8_core_figures(
    *,
    clean_results_dir: str | Path,
    mixed_training_dir: str | Path,
    mixed_evaluation_dir: str | Path,
    clean_archive: str | Path,
    baseline_archive: str | Path,
    protocol_dir: str | Path,
    data_path: str | Path,
    protocol_code_root: str | Path,
    output_dir: str | Path,
    sample_id: int,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Generate every requested Stage 8 paper figure and traceability artifact."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        _fail(f"paper figure output path is not a directory: {output_path}")
    command = shlex.join([sys.executable, "-m", "occlusion_fer.paper_figures", *argv])
    if os.environ.get("PYTHONPATH"):
        command = f"PYTHONPATH={shlex.quote(os.environ['PYTHONPATH'])} {command}"

    evidence = load_stage8_evidence(
        clean_results_dir, mixed_training_dir, mixed_evaluation_dir
    )
    tables = compute_stage8_tables(evidence)
    for name, table in tables.items():
        table.to_csv(output_path / f"{name}.csv", index=False, float_format="%.10f")

    classwise_evidence = load_classwise_evidence(clean_archive, baseline_archive)
    classwise_tables = compute_classwise_summaries(classwise_evidence)
    for name, table in classwise_tables.items():
        table.to_csv(
            output_path / f"{name}.csv", index=False, float_format="%.10f"
        )

    protocol_root = _require_directory(protocol_dir, "Stage 8 protocol artifacts")
    protocol_metadata = generate_stage8_protocol_examples(
        data_path=data_path,
        config_path=protocol_root / "mixed-seed42-occlusion-eval.yaml",
        mean_path=protocol_root / "training_mean_v2.json",
        manifest_path=protocol_root / "publictest_manifest_v2.csv",
        code_root=protocol_code_root,
        sample_id=sample_id,
        output_dir=output_path,
        command=command,
    )

    _use_paper_style()
    output_files: list[str] = []
    output_files.extend(
        _plot_clean_severity_metric(
            tables["occlusion_severity_summary"], "accuracy", output_path
        )
    )
    output_files.extend(
        _plot_clean_severity_metric(
            tables["occlusion_severity_summary"], "macro_f1", output_path
        )
    )
    output_files.extend(
        _plot_clean_severity_performance(
            tables["occlusion_severity_summary"], output_path
        )
    )
    classwise_output_files = [f"{name}.csv" for name in classwise_tables]
    classwise_output_files.extend(
        _plot_classwise_f1_heatmap(classwise_tables["classwise_f1_summary"], output_path)
    )
    classwise_output_files.extend(
        _plot_classwise_drop_heatmap(
            classwise_tables["classwise_drop_from_clean_summary"], output_path
        )
    )
    output_files.extend(classwise_output_files)
    classwise_manifest_path = output_path / "classwise_generation_manifest.json"
    classwise_output_files.append(classwise_manifest_path.name)
    classwise_manifest = _build_classwise_manifest(
        classwise_evidence, output_path, classwise_output_files, argv
    )
    classwise_manifest["output_files"] = sorted(set(classwise_output_files))
    _write_json(classwise_manifest_path, classwise_manifest)
    output_files.extend(
        _plot_by_condition_comparison(
            tables["experiment3_comparison_summary"], output_path
        )
    )
    clean_comparison = tables["experiment3_comparison_summary"].loc[
        tables["experiment3_comparison_summary"]["condition"] == "clean"
    ]
    output_files.extend(
        _plot_grouped_metric_bars(
            clean_comparison,
            stem="experiment3_clean_performance_comparison",
            output_dir=output_path,
            title="Clean condition",
        )
    )
    output_files.extend(
        _plot_grouped_metric_bars(
            tables["experiment3_robustness_summary"],
            stem="experiment3_occluded_robustness_summary",
            output_dir=output_path,
            title="Mean across nine occluded conditions",
        )
    )
    output_files.extend(
        _plot_gain_heatmap(tables["experiment3_robustness_gain"], output_path)
    )

    sources = {
        "clean_results": str(evidence.clean_results_dir),
        "mixed_training": str(evidence.mixed_training_dir),
        "mixed_evaluation": str(evidence.mixed_evaluation_dir),
        "clean_archive": str(_require_file(Path(clean_archive))),
        "baseline_archive": str(_require_file(Path(baseline_archive))),
        "protocol_dir": str(protocol_root),
        "fer2013_csv": str(_require_file(Path(data_path))),
    }
    _write_markdown_outputs(output_path, sources)
    manifest = _build_figure_manifest(output_path, sources, command)
    _write_json(output_path / "figure_manifest.json", manifest)
    if any(item["status"] != "generated" for item in manifest["figures"]):
        _fail("one or more requested figures were not generated")
    print(f"output_dir={output_path}")
    print(f"formal_metric_rows={len(evidence.metrics)}")
    print(f"requested_figure_files={len(manifest['figures'])}")
    print(f"protocol={protocol_metadata['protocol']['algorithm_version']}")
    return {
        "evidence": evidence,
        "tables": {**tables, **classwise_tables},
        "manifest": manifest,
        "output_files": output_files,
    }
