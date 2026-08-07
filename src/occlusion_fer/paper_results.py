"""Strict aggregation and plotting for completed formal PublicTest outputs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from occlusion_fer.artifacts import write_csv_atomic, write_json_atomic
from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.occlusion_evaluate import CONDITION_ORDER
from occlusion_fer.paper_figures import (
    FIGURE_DPI,
    _creation_command,
    _git_identity,
    sha256_file,
)


FORMAL_SEEDS = (42, 123, 2026)
STRATEGY_ORDER = ("clean-only", "mixed")
OFFICIAL_PUBLICTEST_SAMPLE_COUNT = 3589
_IDENTITY_FIELDS = (
    "evaluation_commit",
    "training_dataset_sha256",
    "publictest_dataset_sha256",
    "training_mean_sha256",
    "manifest_sha256",
    "protocol",
    "image_height",
    "image_width",
)
_METRIC_FIELDS = ("accuracy", "macro_f1")
_OCCLUSION_TYPES = ("upper_face", "lower_face", "random_rectangle")
_STRATEGY_COLORS = {"clean-only": "#3366A6", "mixed": "#B54632"}


class PaperResultError(ValueError):
    """Raised when formal result inputs are incomplete or incompatible."""


def export_formal_results(
    evaluation_runs: Mapping[str, Mapping[int, str | Path]],
    output_directory: str | Path,
    *,
    training_runs: Mapping[str, Mapping[int, str | Path]] | None = None,
    require_official: bool = True,
    creation_command: Sequence[str] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Create paper tables and figures from exactly six completed evaluations."""
    _validate_run_map(evaluation_runs, "evaluation_runs")
    if type(require_official) is not bool:
        raise PaperResultError("require_official must be a bool")
    if training_runs is not None:
        _validate_run_map(training_runs, "training_runs")
    output_path = Path(output_directory).expanduser()
    if output_path.exists():
        raise FileExistsError(
            f"paper result output directory already exists: {output_path}"
        )

    input_hashes: dict[str, str] = {}
    run_payloads: dict[tuple[str, int], dict[str, object]] = {}
    shared_identity: dict[str, object] | None = None
    for strategy in STRATEGY_ORDER:
        for seed in FORMAL_SEEDS:
            payload = _load_evaluation_run(
                Path(evaluation_runs[strategy][seed]).expanduser(),
                strategy=strategy,
                seed=seed,
                require_official=require_official,
                input_hashes=input_hashes,
            )
            identity = {
                field: payload["provenance"][field] for field in _IDENTITY_FIELDS
            }
            if shared_identity is None:
                shared_identity = identity
            else:
                for field, expected in shared_identity.items():
                    if identity[field] != expected:
                        raise PaperResultError(
                            f"formal evaluation {field} differs across runs"
                        )
            run_payloads[(strategy, seed)] = payload
    assert shared_identity is not None
    sample_counts = {
        payload["metrics"]["clean"]["sample_count"]
        for payload in run_payloads.values()
    }
    if len(sample_counts) != 1:
        raise PaperResultError("PublicTest sample counts differ across formal runs")

    metric_rows = _condition_metric_rows(run_payloads)
    summary_rows = _condition_summary_rows(metric_rows)
    comparison_rows = _strategy_comparison_rows(metric_rows)
    per_class_by_seed, per_class_summary = _per_class_rows(run_payloads)
    histories: list[dict[str, object]] | None = None
    if training_runs is not None:
        histories = _load_histories(training_runs, input_hashes)

    output_path.mkdir(parents=True, exist_ok=False)
    output_files: list[Path] = []
    output_files.append(
        write_csv_atomic(
            output_path / "condition_metrics_by_seed.csv",
            tuple(metric_rows[0]),
            metric_rows,
        )
    )
    output_files.append(
        write_csv_atomic(
            output_path / "condition_summary.csv",
            tuple(summary_rows[0]),
            summary_rows,
        )
    )
    output_files.append(
        write_csv_atomic(
            output_path / "strategy_comparison.csv",
            tuple(comparison_rows[0]),
            comparison_rows,
        )
    )
    output_files.append(
        write_csv_atomic(
            output_path / "per_class_metrics_by_seed.csv",
            tuple(per_class_by_seed[0]),
            per_class_by_seed,
        )
    )
    output_files.append(
        write_csv_atomic(
            output_path / "per_class_summary.csv",
            tuple(per_class_summary[0]),
            per_class_summary,
        )
    )
    output_files.extend(
        _write_robustness_figures(output_path, metric_rows, summary_rows)
    )
    output_files.extend(
        _write_confusion_figures(output_path, run_payloads)
    )
    if histories is not None:
        output_files.append(
            write_csv_atomic(
                output_path / "training_history_by_seed.csv",
                tuple(histories[0]),
                histories,
            )
        )
        output_files.append(_write_training_curves(output_path, histories))

    commit, dirty = _git_identity(repository_root)
    provenance: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "formal_paper_result_export",
        "strategy_order": list(STRATEGY_ORDER),
        "seed_order": list(FORMAL_SEEDS),
        "condition_order": list(CONDITION_ORDER),
        "label_order": list(FER2013_LABEL_NAMES),
        "aggregation_rule": "mean_and_sample_standard_deviation",
        "drop_rule": "paired_per_seed_clean_minus_occluded_then_aggregate",
        "strategy_delta_rule": "paired_per_seed_mixed_minus_clean_only",
        "shared_evaluation_identity": shared_identity,
        "input_runs": {
            strategy: {
                str(seed): str(Path(evaluation_runs[strategy][seed]).resolve())
                for seed in FORMAL_SEEDS
            }
            for strategy in STRATEGY_ORDER
        },
        "training_runs": (
            {
                strategy: {
                    str(seed): str(Path(training_runs[strategy][seed]).resolve())
                    for seed in FORMAL_SEEDS
                }
                for strategy in STRATEGY_ORDER
            }
            if training_runs is not None
            else None
        ),
        "input_files_sha256": dict(sorted(input_hashes.items())),
        "generation_commit": commit,
        "git_dirty": dirty,
        "creation_command": _creation_command(creation_command),
        "output_files_sha256": {
            str(path.relative_to(output_path)): sha256_file(path)
            for path in sorted(output_files)
        },
    }
    write_json_atomic(output_path / "paper_results_provenance.json", provenance)
    return provenance


def _validate_run_map(value: object, field_name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(STRATEGY_ORDER):
        raise PaperResultError(
            f"{field_name} must contain clean-only and mixed strategies"
        )
    for strategy in STRATEGY_ORDER:
        runs = value[strategy]
        if not isinstance(runs, Mapping) or set(runs) != set(FORMAL_SEEDS):
            raise PaperResultError(
                f"{field_name} {strategy} must contain seeds 42, 123, 2026"
            )


def _load_evaluation_run(
    path: Path,
    *,
    strategy: str,
    seed: int,
    require_official: bool,
    input_hashes: dict[str, str],
) -> dict[str, object]:
    if not path.is_dir():
        raise PaperResultError(f"evaluation run directory not found: {path}")
    if (path / "failure.json").exists():
        raise PaperResultError(f"evaluation run contains failure.json: {path}")
    provenance_path = path / "evaluation_provenance.json"
    provenance = _read_json_mapping(provenance_path, "evaluation provenance")
    input_hashes[str(provenance_path.resolve())] = sha256_file(provenance_path)
    if provenance.get("conditions") != list(CONDITION_ORDER):
        raise PaperResultError("evaluation provenance condition order is incomplete")
    if provenance.get("seed") != seed:
        raise PaperResultError("evaluation provenance seed does not match run mapping")
    if provenance.get("protocol") != "occlusion-v2-224":
        raise PaperResultError("evaluation provenance protocol is not occlusion-v2-224")
    if provenance.get("git_dirty") is not False:
        raise PaperResultError("formal evaluation provenance git_dirty must be false")
    if provenance.get("training_git_dirty") not in (None, False):
        raise PaperResultError("formal training provenance dirty state must be false")
    resolved_config = provenance.get("resolved_config")
    expected_mode = "clean" if strategy == "clean-only" else "mixed"
    if (
        not isinstance(resolved_config, Mapping)
        or not isinstance(resolved_config.get("training"), Mapping)
        or resolved_config["training"].get("mode") != expected_mode
    ):
        raise PaperResultError(
            f"evaluation run does not identify the {strategy} strategy"
        )
    for field in _IDENTITY_FIELDS:
        if field not in provenance:
            raise PaperResultError(f"evaluation provenance is missing {field}")

    metrics: dict[str, dict[str, object]] = {}
    for condition in CONDITION_ORDER:
        metrics_path = path / "conditions" / condition / f"{condition}_metrics.json"
        payload = _read_json_mapping(metrics_path, "condition metrics")
        input_hashes[str(metrics_path.resolve())] = sha256_file(metrics_path)
        _validate_metrics(
            payload,
            condition=condition,
            require_official=require_official,
        )
        metrics[condition] = payload
    sample_counts = {payload["sample_count"] for payload in metrics.values()}
    if len(sample_counts) != 1:
        raise PaperResultError("condition sample counts differ within one run")
    return {"path": path, "provenance": provenance, "metrics": metrics}


def _read_json_mapping(path: Path, description: str) -> dict[str, object]:
    if not path.is_file():
        raise PaperResultError(f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperResultError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise PaperResultError(f"{description} must be a JSON mapping: {path}")
    return payload


def _validate_metrics(
    payload: Mapping[str, object], *, condition: str, require_official: bool
) -> None:
    if payload.get("metric_schema_version") != 1:
        raise PaperResultError("condition metrics schema version must be 1")
    if payload.get("split") != "validation":
        raise PaperResultError(
            "paper result export requires validation/PublicTest metrics and "
            "has no PrivateTest route"
        )
    if payload.get("condition") != condition:
        raise PaperResultError("condition metrics identity does not match its path")
    sample_count = payload.get("sample_count")
    if type(sample_count) is not int or sample_count <= 0:
        raise PaperResultError("condition sample_count must be a positive integer")
    if require_official and sample_count != OFFICIAL_PUBLICTEST_SAMPLE_COUNT:
        raise PaperResultError(
            "official PublicTest condition sample_count must be 3589"
        )
    for field in ("loss", *_METRIC_FIELDS):
        value = payload.get(field)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise PaperResultError(f"condition metric {field} must be finite")
        if field != "loss" and not 0 <= float(value) <= 1:
            raise PaperResultError(f"condition metric {field} must be in [0, 1]")
    expected_label_order = [
        {"label": label, "label_name": name}
        for label, name in enumerate(FER2013_LABEL_NAMES)
    ]
    if payload.get("label_order") != expected_label_order:
        raise PaperResultError("condition metric label order is not canonical")
    per_class = payload.get("per_class")
    if not isinstance(per_class, list) or len(per_class) != 7:
        raise PaperResultError("condition metrics require seven per-class rows")
    support_total = 0
    for label, row in enumerate(per_class):
        if (
            not isinstance(row, Mapping)
            or row.get("label") != label
            or row.get("label_name") != FER2013_LABEL_NAMES[label]
        ):
            raise PaperResultError("per-class metric identity is not canonical")
        support = row.get("support")
        if type(support) is not int or support < 0:
            raise PaperResultError("per-class support must be non-negative")
        support_total += support
        for field in ("precision", "recall", "f1"):
            value = row.get(field)
            if type(value) not in (int, float) or not 0 <= float(value) <= 1:
                raise PaperResultError(f"per-class {field} must be in [0, 1]")
    if support_total != sample_count:
        raise PaperResultError("per-class support does not sum to sample_count")
    matrix = payload.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 7
        or any(not isinstance(row, list) or len(row) != 7 for row in matrix)
        or any(type(value) is not int or value < 0 for row in matrix for value in row)
    ):
        raise PaperResultError("confusion matrix must contain non-negative 7x7 counts")
    if sum(value for row in matrix for value in row) != sample_count:
        raise PaperResultError("confusion matrix does not sum to sample_count")


def _condition_metric_rows(
    run_payloads: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        for seed in FORMAL_SEEDS:
            metrics = run_payloads[(strategy, seed)]["metrics"]
            clean = metrics["clean"]
            for condition in CONDITION_ORDER:
                payload = metrics[condition]
                occlusion_type, ratio = _condition_parts(condition)
                rows.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "condition": condition,
                        "occlusion_type": occlusion_type,
                        "target_ratio": ratio,
                        "sample_count": payload["sample_count"],
                        "loss": payload["loss"],
                        "accuracy": payload["accuracy"],
                        "macro_f1": payload["macro_f1"],
                        "accuracy_drop": float(clean["accuracy"])
                        - float(payload["accuracy"]),
                        "macro_f1_drop": float(clean["macro_f1"])
                        - float(payload["macro_f1"]),
                    }
                )
    return rows


def _condition_summary_rows(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        for condition in CONDITION_ORDER:
            selected = [
                row
                for row in metric_rows
                if row["strategy"] == strategy and row["condition"] == condition
            ]
            occlusion_type, ratio = _condition_parts(condition)
            row: dict[str, object] = {
                "strategy": strategy,
                "condition": condition,
                "occlusion_type": occlusion_type,
                "target_ratio": ratio,
                "seed_count": len(selected),
            }
            for field in (*_METRIC_FIELDS, "accuracy_drop", "macro_f1_drop"):
                values = [float(item[field]) for item in selected]
                row[f"{field}_mean"] = statistics.mean(values)
                row[f"{field}_sample_std"] = statistics.stdev(values)
            rows.append(row)
    return rows


def _strategy_comparison_rows(
    metric_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (row["strategy"], row["seed"], row["condition"]): row
        for row in metric_rows
    }
    rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        accuracy_delta = []
        macro_delta = []
        drop_improvement = []
        macro_drop_improvement = []
        for seed in FORMAL_SEEDS:
            clean_only = lookup[("clean-only", seed, condition)]
            mixed = lookup[("mixed", seed, condition)]
            accuracy_delta.append(
                float(mixed["accuracy"]) - float(clean_only["accuracy"])
            )
            macro_delta.append(
                float(mixed["macro_f1"]) - float(clean_only["macro_f1"])
            )
            drop_improvement.append(
                float(clean_only["accuracy_drop"]) - float(mixed["accuracy_drop"])
            )
            macro_drop_improvement.append(
                float(clean_only["macro_f1_drop"])
                - float(mixed["macro_f1_drop"])
            )
        occlusion_type, ratio = _condition_parts(condition)
        rows.append(
            {
                "condition": condition,
                "occlusion_type": occlusion_type,
                "target_ratio": ratio,
                "seed_count": len(FORMAL_SEEDS),
                "accuracy_delta_mean": statistics.mean(accuracy_delta),
                "accuracy_delta_sample_std": statistics.stdev(accuracy_delta),
                "macro_f1_delta_mean": statistics.mean(macro_delta),
                "macro_f1_delta_sample_std": statistics.stdev(macro_delta),
                "drop_improvement_mean": statistics.mean(drop_improvement),
                "drop_improvement_sample_std": statistics.stdev(drop_improvement),
                "macro_f1_drop_improvement_mean": statistics.mean(
                    macro_drop_improvement
                ),
                "macro_f1_drop_improvement_sample_std": statistics.stdev(
                    macro_drop_improvement
                ),
            }
        )
    return rows


def _per_class_rows(
    run_payloads: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_seed: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        for seed in FORMAL_SEEDS:
            metrics = run_payloads[(strategy, seed)]["metrics"]
            for condition in CONDITION_ORDER:
                for class_row in metrics[condition]["per_class"]:
                    by_seed.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "condition": condition,
                            **dict(class_row),
                        }
                    )
    summary: list[dict[str, object]] = []
    for strategy in STRATEGY_ORDER:
        for condition in CONDITION_ORDER:
            for label, label_name in enumerate(FER2013_LABEL_NAMES):
                selected = [
                    row
                    for row in by_seed
                    if row["strategy"] == strategy
                    and row["condition"] == condition
                    and row["label"] == label
                ]
                row: dict[str, object] = {
                    "strategy": strategy,
                    "condition": condition,
                    "label": label,
                    "label_name": label_name,
                    "seed_count": len(selected),
                }
                for field in ("precision", "recall", "f1", "support"):
                    values = [float(item[field]) for item in selected]
                    row[f"{field}_mean"] = statistics.mean(values)
                    row[f"{field}_sample_std"] = statistics.stdev(values)
                summary.append(row)
    return by_seed, summary


def _load_histories(
    training_runs: Mapping[str, Mapping[int, str | Path]],
    input_hashes: dict[str, str],
) -> list[dict[str, object]]:
    histories: list[dict[str, object]] = []
    epoch_identity: dict[str, tuple[int, ...]] = {}
    required = (
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    )
    for strategy in STRATEGY_ORDER:
        for seed in FORMAL_SEEDS:
            run_path = Path(training_runs[strategy][seed]).expanduser()
            metadata_path = run_path / "run_metadata.json"
            metadata = _read_json_mapping(metadata_path, "training run metadata")
            input_hashes[str(metadata_path.resolve())] = sha256_file(metadata_path)
            expected_mode = "clean" if strategy == "clean-only" else "mixed"
            if (
                metadata.get("status") != "completed"
                or metadata.get("seed") != seed
                or metadata.get("training_mode") != expected_mode
                or metadata.get("git_dirty") is not False
            ):
                raise PaperResultError(
                    f"training run metadata does not match {strategy} seed {seed}"
                )
            path = run_path / "history.csv"
            if not path.is_file():
                raise PaperResultError(f"training history not found: {path}")
            input_hashes[str(path.resolve())] = sha256_file(path)
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None or any(
                        field not in reader.fieldnames for field in required
                    ):
                        raise PaperResultError(
                            f"training history is missing required columns: {path}"
                        )
                    source_rows = list(reader)
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                raise PaperResultError(f"training history is not readable: {path}") from exc
            if not source_rows:
                raise PaperResultError(f"training history is empty: {path}")
            epochs: list[int] = []
            for source_row in source_rows:
                try:
                    epoch = int(source_row["epoch"])
                    values = {
                        field: float(source_row[field]) for field in required[1:]
                    }
                except (TypeError, ValueError) as exc:
                    raise PaperResultError(
                        f"training history contains invalid numeric data: {path}"
                    ) from exc
                if epoch <= 0 or any(
                    not math.isfinite(value) for value in values.values()
                ):
                    raise PaperResultError(
                        f"training history contains invalid numeric data: {path}"
                    )
                epochs.append(epoch)
                histories.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "epoch": epoch,
                        **values,
                    }
                )
            identity = tuple(epochs)
            if strategy in epoch_identity and epoch_identity[strategy] != identity:
                raise PaperResultError(
                    f"{strategy} training histories do not share epoch order"
                )
            epoch_identity[strategy] = identity
    return histories


def _write_robustness_figures(
    output_path: Path,
    metric_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> list[Path]:
    paths: list[Path] = []
    summary_lookup = {
        (row["strategy"], row["condition"]): row for row in summary_rows
    }
    for metric in _METRIC_FIELDS:
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=FIGURE_DPI)
        for axis, occlusion_type in zip(axes, _OCCLUSION_TYPES, strict=True):
            conditions = [
                "clean",
                *(f"{occlusion_type}_{ratio:.2f}" for ratio in (0.2, 0.3, 0.4)),
            ]
            x_values = (0.0, 0.2, 0.3, 0.4)
            for strategy in STRATEGY_ORDER:
                means = [
                    float(summary_lookup[(strategy, condition)][f"{metric}_mean"])
                    for condition in conditions
                ]
                errors = [
                    float(
                        summary_lookup[(strategy, condition)][
                            f"{metric}_sample_std"
                        ]
                    )
                    for condition in conditions
                ]
                axis.errorbar(
                    x_values,
                    means,
                    yerr=errors,
                    color=_STRATEGY_COLORS[strategy],
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    label=strategy,
                )
                for x_value, condition in zip(x_values, conditions, strict=True):
                    raw_values = [
                        float(row[metric])
                        for row in metric_rows
                        if row["strategy"] == strategy
                        and row["condition"] == condition
                    ]
                    axis.scatter(
                        [x_value] * len(raw_values),
                        raw_values,
                        color=_STRATEGY_COLORS[strategy],
                        s=12,
                        alpha=0.45,
                        zorder=3,
                    )
            axis.set_title(occlusion_type.replace("_", " "))
            axis.set_xlabel("Target occlusion ratio")
            axis.set_xticks(x_values, ("Clean", "0.20", "0.30", "0.40"))
            axis.set_ylim(0.0, 1.0)
            axis.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(metric.replace("_", " ").title())
        axes[0].legend(frameon=False, loc="lower left")
        figure.tight_layout()
        path = output_path / f"robustness_{metric}.png"
        figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
        plt.close(figure)
        paths.append(path)
    return paths


def _write_confusion_figures(
    output_path: Path,
    run_payloads: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[Path]:
    directory = output_path / "confusion_matrices"
    directory.mkdir()
    paths: list[Path] = []
    for strategy in STRATEGY_ORDER:
        for condition in CONDITION_ORDER:
            matrices = [
                np.asarray(
                    run_payloads[(strategy, seed)]["metrics"][condition][
                        "confusion_matrix"
                    ],
                    dtype=np.int64,
                )
                for seed in FORMAL_SEEDS
            ]
            counts = np.sum(matrices, axis=0)
            row_totals = counts.sum(axis=1, keepdims=True)
            normalized = np.divide(
                counts,
                row_totals,
                out=np.zeros_like(counts, dtype=np.float64),
                where=row_totals != 0,
            )
            figure, axis = plt.subplots(figsize=(6, 5.5), dpi=FIGURE_DPI)
            image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
            for true_label in range(7):
                for predicted_label in range(7):
                    value = normalized[true_label, predicted_label]
                    axis.text(
                        predicted_label,
                        true_label,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value >= 0.55 else "black",
                    )
            axis.set_xticks(range(7), FER2013_LABEL_NAMES, rotation=40, ha="right")
            axis.set_yticks(range(7), FER2013_LABEL_NAMES)
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("True label")
            axis.set_title(f"{strategy}: {condition}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            figure.tight_layout()
            path = directory / f"{strategy}_{condition}.png"
            figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
            plt.close(figure)
            paths.append(path)
    return paths


def _write_training_curves(
    output_path: Path, histories: Sequence[Mapping[str, object]]
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=FIGURE_DPI)
    for strategy in STRATEGY_ORDER:
        strategy_rows = [row for row in histories if row["strategy"] == strategy]
        epochs = sorted({int(row["epoch"]) for row in strategy_rows})
        for seed in FORMAL_SEEDS:
            seed_rows = [row for row in strategy_rows if row["seed"] == seed]
            axes[0].plot(
                [row["epoch"] for row in seed_rows],
                [row["train_loss"] for row in seed_rows],
                color=_STRATEGY_COLORS[strategy],
                alpha=0.14,
                linewidth=1.0,
            )
            axes[0].plot(
                [row["epoch"] for row in seed_rows],
                [row["validation_loss"] for row in seed_rows],
                color=_STRATEGY_COLORS[strategy],
                alpha=0.24,
                linewidth=1.0,
            )
            axes[1].plot(
                [row["epoch"] for row in seed_rows],
                [row["validation_macro_f1"] for row in seed_rows],
                color=_STRATEGY_COLORS[strategy],
                alpha=0.22,
                linewidth=1.0,
            )
        mean_loss = [
            statistics.mean(
                float(row["validation_loss"])
                for row in strategy_rows
                if row["epoch"] == epoch
            )
            for epoch in epochs
        ]
        mean_train_loss = [
            statistics.mean(
                float(row["train_loss"])
                for row in strategy_rows
                if row["epoch"] == epoch
            )
            for epoch in epochs
        ]
        mean_macro = [
            statistics.mean(
                float(row["validation_macro_f1"])
                for row in strategy_rows
                if row["epoch"] == epoch
            )
            for epoch in epochs
        ]
        axes[0].plot(
            epochs,
            mean_train_loss,
            color=_STRATEGY_COLORS[strategy],
            linewidth=1.7,
            linestyle="--",
            label=f"{strategy} train",
        )
        axes[0].plot(
            epochs,
            mean_loss,
            color=_STRATEGY_COLORS[strategy],
            linewidth=2.0,
            label=f"{strategy} validation",
        )
        axes[1].plot(
            epochs,
            mean_macro,
            color=_STRATEGY_COLORS[strategy],
            linewidth=2.0,
            label=strategy,
        )
    axes[0].set_title("Training and validation loss")
    axes[1].set_title("Validation macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0.0, 1.0)
    figure.tight_layout()
    path = output_path / "training_curves.png"
    figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)
    return path


def _condition_parts(condition: str) -> tuple[str, float]:
    if condition == "clean":
        return "clean", 0.0
    occlusion_type, ratio_token = condition.rsplit("_", 1)
    return occlusion_type, float(ratio_token)
