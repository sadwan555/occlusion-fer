"""Aggregate final clean/masked results without inferential statistics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from pathlib import Path

from occlusion_fer.artifacts import write_csv_atomic
from occlusion_fer.private_manifest import CONDITION_ORDER


SEEDS = (42, 123, 2026)
STRATEGIES = ("clean", "mixed")
METRICS = ("accuracy", "macro_f1")


class PrivateAggregateError(ValueError):
    """Raised when final metrics are missing, duplicated, or non-finite."""


def build_aggregate_rows(
    metrics: Mapping[tuple[str, int, str], Mapping[str, float]],
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    """Return summary, clean-drop, and paired clean/mixed rows."""
    normalized = _validate_metrics(metrics)
    conditions = tuple(
        dict.fromkeys(key[2] for key in normalized)
    )
    if conditions != CONDITION_ORDER:
        raise PrivateAggregateError(
            "aggregate metrics must contain canonical clean plus nine conditions"
        )

    summary: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        for condition in conditions:
            for metric in METRICS:
                values = [
                    normalized[(strategy, seed, condition)][metric]
                    for seed in SEEDS
                ]
                summary.append(
                    {
                        "strategy": strategy,
                        "condition": condition,
                        "metric": metric,
                        "seed42": values[0],
                        "seed123": values[1],
                        "seed2026": values[2],
                        "mean": statistics.fmean(values),
                        "sample_std": statistics.stdev(values),
                    }
                )

    drops: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        for seed in SEEDS:
            baseline = normalized[(strategy, seed, "clean")]
            for condition in conditions:
                if condition == "clean":
                    continue
                masked = normalized[(strategy, seed, condition)]
                drops.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "condition": condition,
                        "accuracy_drop": (
                            baseline["accuracy"] - masked["accuracy"]
                        ),
                        "macro_f1_drop": (
                            baseline["macro_f1"] - masked["macro_f1"]
                        ),
                    }
                )

    paired: list[dict[str, object]] = []
    for seed in SEEDS:
        for condition in conditions:
            clean = normalized[("clean", seed, condition)]
            mixed = normalized[("mixed", seed, condition)]
            paired.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "accuracy_difference": (
                        mixed["accuracy"] - clean["accuracy"]
                    ),
                    "macro_f1_difference": (
                        mixed["macro_f1"] - clean["macro_f1"]
                    ),
                }
            )
    return tuple(summary), tuple(drops), tuple(paired)


def write_aggregate_artifacts(
    output_root: str | Path,
    metrics: Mapping[tuple[str, int, str], Mapping[str, float]],
) -> dict[str, Path]:
    """Create the three frozen aggregate CSV files with no overwrite."""
    root = Path(output_root).expanduser()
    summary_directory = root / "summary"
    if summary_directory.exists():
        if not summary_directory.is_dir() or any(summary_directory.iterdir()):
            raise FileExistsError(
                f"final summary directory is not empty: {summary_directory}"
            )
    else:
        summary_directory.mkdir(parents=True, exist_ok=False)
    summary, drops, paired = build_aggregate_rows(metrics)
    return {
        "strategy_condition_summary": write_csv_atomic(
            summary_directory / "strategy_condition_summary.csv",
            (
                "strategy",
                "condition",
                "metric",
                "seed42",
                "seed123",
                "seed2026",
                "mean",
                "sample_std",
            ),
            summary,
        ),
        "clean_to_occluded_drop": write_csv_atomic(
            summary_directory / "clean_to_occluded_drop.csv",
            (
                "strategy",
                "seed",
                "condition",
                "accuracy_drop",
                "macro_f1_drop",
            ),
            drops,
        ),
        "paired_strategy_differences": write_csv_atomic(
            summary_directory / "paired_strategy_differences.csv",
            (
                "seed",
                "condition",
                "accuracy_difference",
                "macro_f1_difference",
            ),
            paired,
        ),
    }


def _validate_metrics(
    metrics: Mapping[tuple[str, int, str], Mapping[str, float]],
) -> dict[tuple[str, int, str], dict[str, float]]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise PrivateAggregateError("aggregate metrics must be a mapping")
    normalized: dict[tuple[str, int, str], dict[str, float]] = {}
    conditions: list[str] = []
    for key, row in metrics.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or key[0] not in STRATEGIES
            or key[1] not in SEEDS
            or type(key[2]) is not str
            or not key[2]
        ):
            raise PrivateAggregateError(
                "aggregate key must be (strategy, seed, condition)"
            )
        if not isinstance(row, Mapping) or set(row) != set(METRICS):
            raise PrivateAggregateError(
                "aggregate row must contain accuracy and macro_f1"
            )
        values = {}
        for metric in METRICS:
            value = row[metric]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PrivateAggregateError(
                    f"aggregate {metric} must be finite"
                )
            values[metric] = float(value)
        normalized[key] = values
        if key[2] not in conditions:
            conditions.append(key[2])
    expected = {
        (strategy, seed, condition)
        for strategy in STRATEGIES
        for seed in SEEDS
        for condition in conditions
    }
    if set(normalized) != expected:
        raise PrivateAggregateError(
            "aggregate metrics must contain both strategies and all three seeds"
        )
    return normalized
