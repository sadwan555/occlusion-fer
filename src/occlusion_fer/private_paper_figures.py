"""Generate paper figures from the frozen FER2013 PrivateTest final archive.

The plotting primitives are intentionally reused from ``paper_stage8`` and
``paper_formal_completion``. This module only adapts the PrivateTest directory
layout, validates provenance, and records source-data traceability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import shutil
import sys
import tarfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from occlusion_fer.paper_formal_completion import (
    build_confusion_table,
    _plot_confusion as plot_exp1_confusion,
    _plot_summary as plot_exp1_summary,
)
from occlusion_fer.paper_figures import (
    ClasswiseEvidence,
    _plot_classwise_drop_heatmap,
    _plot_classwise_f1_heatmap,
    compute_classwise_summaries,
)
from occlusion_fer.paper_stage8 import (
    CLASSWISE_CONDITIONS,
    CLASSWISE_OCCLUDED_CONDITIONS,
    EXPECTED_SEEDS,
    METRICS,
    METHODS,
    OCCLUSION_TYPES,
    SEVERITIES,
    _plot_by_condition_comparison,
    _plot_clean_severity_metric,
    _plot_clean_severity_performance,
    _plot_gain_heatmap,
    _plot_grouped_metric_bars,
    compute_stage8_tables,
)


EXPECTED_SAMPLE_COUNT = 3589
EXPECTED_SPLIT = "test"
EXPECTED_USAGE = "PrivateTest"
EXPECTED_DATASET_SHA = "4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634"
EXPECTED_MANIFEST_SHA = "28f5463712cecf351ad318421e0fcc0e70db243621b859d94293a0aaeacd6220"
EXPECTED_PLAN_SHA = "020b701e844e7d8af7fe7bb2133c0d3ff161acbf23deca9734700789db8165e8"
LABEL_NAMES = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
OUTLINE = ("experiment1", "experiment2", "experiment3", "comparisons", "source_data")


class PrivateFigureError(ValueError):
    """Raised when a PrivateTest figure input is missing or inconsistent."""


def _fail(message: str) -> None:
    raise PrivateFigureError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {path}")
    return value


def _require_file(path: Path) -> Path:
    if not path.is_file():
        _fail(f"missing PrivateTest result: {path}")
    return path


def _validate_condition_metrics(
    metrics: dict[str, Any], condition: str, per_class: pd.DataFrame, confusion: pd.DataFrame,
    predictions: pd.DataFrame, context: str,
) -> tuple[float, float]:
    required = {"condition", "split", "sample_count", "accuracy", "macro_f1", "confusion_matrix", "per_class"}
    if not required.issubset(metrics):
        _fail(f"{context} is missing metric fields: {sorted(required - set(metrics))}")
    if metrics["condition"] != condition or metrics["split"] != EXPECTED_SPLIT:
        _fail(f"{context} condition/split mismatch")
    if int(metrics["sample_count"]) != EXPECTED_SAMPLE_COUNT or len(predictions) != EXPECTED_SAMPLE_COUNT:
        _fail(f"{context} must contain exactly {EXPECTED_SAMPLE_COUNT} samples")
    if tuple(per_class["label"].astype(int)) != tuple(range(7)) or tuple(per_class["label_name"]) != LABEL_NAMES:
        _fail(f"{context} label order is not FER2013 canonical order")
    predicted = [f"predicted_{label}" for label in LABEL_NAMES]
    matrix = confusion.loc[:, predicted].to_numpy(dtype=int)
    if tuple(confusion["true_label"].astype(int)) != tuple(range(7)) or tuple(confusion["true_label_name"]) != LABEL_NAMES:
        _fail(f"{context} confusion label order is invalid")
    if not np.array_equal(matrix, np.asarray(metrics["confusion_matrix"], dtype=int)):
        _fail(f"{context} confusion matrix disagrees with metrics.json")
    if int(matrix.sum()) != EXPECTED_SAMPLE_COUNT:
        _fail(f"{context} confusion matrix total is not {EXPECTED_SAMPLE_COUNT}")
    if set(predictions["condition"]) != {condition} or set(predictions["split"]) != {EXPECTED_SPLIT}:
        _fail(f"{context} predictions condition/split mismatch")
    if not np.array_equal(
        pd.crosstab(predictions["true_label"], predictions["predicted_label"])
        .reindex(index=range(7), columns=range(7), fill_value=0).to_numpy(dtype=int), matrix
    ):
        _fail(f"{context} predictions cannot reconstruct confusion matrix")
    metric_per_class = pd.DataFrame(metrics["per_class"])
    for column in ("label", "label_name", "precision", "recall", "f1", "support"):
        if column not in metric_per_class or column not in per_class:
            _fail(f"{context} per-class data is missing {column}")
    if not np.array_equal(
        metric_per_class.loc[:, ["label", "label_name"]].to_numpy(),
        per_class.loc[:, ["label", "label_name"]].to_numpy(),
    ):
        _fail(f"{context} per-class label metadata disagrees with metrics.json")
    if not np.allclose(
        metric_per_class.loc[:, ["precision", "recall", "f1", "support"]].to_numpy(dtype=float),
        per_class.loc[:, ["precision", "recall", "f1", "support"]].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-15,
    ):
        _fail(f"{context} per-class CSV disagrees with metrics.json")
    if not np.array_equal(per_class["support"].to_numpy(dtype=int), matrix.sum(axis=1)):
        _fail(f"{context} per-class supports disagree with confusion matrix")
    accuracy = float(metrics["accuracy"])
    macro_f1 = float(metrics["macro_f1"])
    if not math.isclose(accuracy, float(np.trace(matrix) / matrix.sum()), rel_tol=0.0, abs_tol=1e-15):
        _fail(f"{context} accuracy cannot be reconstructed from confusion matrix")
    if not math.isclose(macro_f1, float(per_class["f1"].mean()), rel_tol=0.0, abs_tol=1e-15):
        _fail(f"{context} macro-F1 cannot be reconstructed from per-class F1")
    return accuracy, macro_f1


def _load_metrics(root: Path) -> tuple[pd.DataFrame, dict[tuple[str, int, str], dict[str, float]], list[dict[str, Any]], dict[int, pd.DataFrame], dict[int, np.ndarray], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    sources: list[dict[str, Any]] = []
    clean_per_class: dict[int, pd.DataFrame] = {}
    clean_confusion: dict[int, np.ndarray] = {}
    clean_classwise_rows: list[dict[str, Any]] = []
    for strategy, method in (("clean", "Clean-only"), ("mixed", "Mixed training")):
        for seed in EXPECTED_SEEDS:
            for condition in CLASSWISE_CONDITIONS:
                condition_root = root / strategy / f"seed{seed}" / "conditions" / condition
                metrics_path = _require_file(condition_root / "metrics.json")
                per_class_path = _require_file(condition_root / "per_class_metrics.csv")
                confusion_path = _require_file(condition_root / "confusion_matrix.csv")
                predictions_path = _require_file(condition_root / "predictions.csv")
                provenance_path = _require_file(condition_root / "condition_provenance.json")
                metrics = _json(metrics_path)
                provenance = _json(provenance_path)
                for key, expected in (("private_dataset_sha256", EXPECTED_DATASET_SHA), ("manifest_sha256", EXPECTED_MANIFEST_SHA), ("plan_sha256", EXPECTED_PLAN_SHA), ("official_usage", EXPECTED_USAGE), ("split", EXPECTED_SPLIT), ("strategy", strategy), ("seed", seed), ("condition", condition), ("protocol", "occlusion-v2-224")):
                    if provenance.get(key) != expected:
                        _fail(f"{provenance_path} field {key} does not match frozen PrivateTest identity")
                if condition == "clean":
                    if provenance.get("target_ratio") is not None or provenance.get("actual_ratio") is not None:
                        _fail(f"{provenance_path} clean condition must not declare an occlusion ratio")
                else:
                    expected_ratio = float(condition.rsplit("_", 1)[1])
                    target_ratio = provenance.get("target_ratio")
                    actual_ratio = provenance.get("actual_ratio")
                    if not isinstance(target_ratio, (int, float)) or not math.isclose(float(target_ratio), expected_ratio, rel_tol=0.0, abs_tol=1e-12):
                        _fail(f"{provenance_path} target ratio disagrees with condition name")
                    if not isinstance(actual_ratio, (int, float)) or not math.isfinite(float(actual_ratio)) or abs(float(actual_ratio) - expected_ratio) > 0.005:
                        _fail(f"{provenance_path} actual ratio is invalid for the requested severity")
                per_class = pd.read_csv(per_class_path)
                confusion = pd.read_csv(confusion_path)
                predictions = pd.read_csv(predictions_path)
                accuracy, macro_f1 = _validate_condition_metrics(metrics, condition, per_class, confusion, predictions, str(metrics_path))
                if strategy == "clean" and condition == "clean":
                    clean_per_class[seed] = per_class.loc[:, ["label", "label_name", "f1"]].copy()
                    clean_confusion[seed] = confusion.loc[:, [f"predicted_{label}" for label in LABEL_NAMES]].to_numpy(dtype=int)
                if strategy == "clean":
                    clean_classwise_rows.extend(
                        {"seed": seed, "condition": condition, "label": int(row.label), "label_name": str(row.label_name), "f1": float(row.f1)}
                        for row in per_class.itertuples()
                    )
                occlusion_type, severity = ("clean", 0.0) if condition == "clean" else next((kind, float(condition.removeprefix(f"{kind}_"))) for kind in OCCLUSION_TYPES if condition.startswith(f"{kind}_"))
                rows.append({"method": method, "training_mode": strategy, "seed": seed, "condition": condition, "occlusion_type": occlusion_type, "severity": severity, "accuracy": accuracy, "macro_f1": macro_f1})
                values[(strategy, seed, condition)] = {"accuracy": accuracy, "macro_f1": macro_f1}
                sources.extend({"source_file": str(path), "fields": ["accuracy", "macro_f1"] if path == metrics_path else [], "strategy": strategy, "seed": seed, "condition": condition, "metric": metric} for path, metric in ((metrics_path, "accuracy/macro_f1"), (per_class_path, "per_class_f1"), (confusion_path, "confusion_matrix"), (predictions_path, "per-sample predictions"), (provenance_path, "provenance")))
    frame = pd.DataFrame(rows)
    if len(frame) != 60:
        _fail(f"expected 60 PrivateTest metric rows, got {len(frame)}")
    return frame, values, sources, clean_per_class, clean_confusion, pd.DataFrame(clean_classwise_rows)


def _private_stage8_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Use the established Stage 8 aggregation functions after the PrivateTest adapter.
    class Evidence:
        metrics = frame
    tables = compute_stage8_tables(Evidence())
    tables["occlusion_severity_summary"]["split"] = "test/PrivateTest"
    return tables


def _rename_private_files(output_dir: Path, names: list[str]) -> list[str]:
    renamed: list[str] = []
    for name in names:
        source = output_dir / name
        target = output_dir / f"private_{name}"
        source.rename(target)
        renamed.append(target.name)
    return renamed


def _exp1_tables(values: dict[tuple[str, int, str], dict[str, float]], clean_per_class: dict[int, pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Any]]:
    rows = []
    for seed in EXPECTED_SEEDS:
        rows.append({"seed": seed, "accuracy": values[("clean", seed, "clean")]["accuracy"], "macro_f1": values[("clean", seed, "clean")]["macro_f1"]})
    frame = pd.DataFrame(rows)
    summary_rows = []
    for metric in ("accuracy", "macro_f1"):
        for _, row in frame.iterrows():
            summary_rows.append({"seed": str(int(row["seed"])), "metric": metric, "class": "", "value": float(row[metric]), "summary": "individual"})
        summary_rows.append({"seed": "mean", "metric": metric, "class": "", "value": float(frame[metric].mean()), "summary": "mean"})
        summary_rows.append({"seed": "sample_sd", "metric": metric, "class": "", "value": float(frame[metric].std(ddof=1)), "summary": "sample_sd_ddof_1"})
    runs: list[Any] = []
    for seed in EXPECTED_SEEDS:
        per_class = clean_per_class[seed]
        runs.append(SimpleNamespace(seed=seed, per_class=per_class))
        for label, value in zip(LABEL_NAMES, per_class["f1"], strict=True):
            summary_rows.append({"seed": str(seed), "metric": "per_class_f1", "class": label, "value": float(value), "summary": "individual"})
    for label in LABEL_NAMES:
        values_for_label = [float(clean_per_class[seed].loc[clean_per_class[seed]["label_name"] == label, "f1"].iloc[0]) for seed in EXPECTED_SEEDS]
        summary_rows.append({"seed": "mean", "metric": "per_class_f1", "class": label, "value": float(np.mean(values_for_label)), "summary": "mean"})
        summary_rows.append({"seed": "sample_sd", "metric": "per_class_f1", "class": label, "value": float(np.std(values_for_label, ddof=1)), "summary": "sample_sd_ddof_1"})
    summary = pd.DataFrame(summary_rows, columns=["seed", "metric", "class", "value", "summary"])
    summary.to_csv(output_dir / "exp1_private_clean_summary.csv", index=False, float_format="%.10f")
    return frame, summary, runs


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _verify_frozen_summaries(
    root: Path,
    values: dict[tuple[str, int, str], dict[str, float]],
) -> dict[str, Any]:
    summary = pd.read_csv(_require_file(root / "summary" / "strategy_condition_summary.csv"))
    paired = pd.read_csv(_require_file(root / "summary" / "paired_strategy_differences.csv"))
    drops = pd.read_csv(_require_file(root / "summary" / "clean_to_occluded_drop.csv"))
    checked = 0
    for row in summary.itertuples():
        seed_values = [values[(row.strategy, seed, row.condition)][row.metric] for seed in EXPECTED_SEEDS]
        expected = (*seed_values, float(np.mean(seed_values)), float(np.std(seed_values, ddof=1)))
        actual = (row.seed42, row.seed123, row.seed2026, row.mean, row.sample_std)
        if not all(math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(actual, expected, strict=True)):
            _fail(f"frozen strategy summary mismatch: {row.strategy} {row.condition} {row.metric}")
        checked += 1
    for row in paired.itertuples():
        for metric in METRICS:
            expected = values[("mixed", row.seed, row.condition)][metric] - values[("clean", row.seed, row.condition)][metric]
            actual = getattr(row, f"{metric}_difference")
            if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"frozen paired difference mismatch: seed {row.seed} {row.condition} {metric}")
            checked += 1
    for row in drops.itertuples():
        for metric in METRICS:
            expected = values[(row.strategy, row.seed, "clean")][metric] - values[(row.strategy, row.seed, row.condition)][metric]
            actual = getattr(row, f"{metric}_drop")
            if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"frozen clean drop mismatch: {row.strategy} seed {row.seed} {row.condition} {metric}")
            checked += 1
    return {"status": "passed", "checked_numeric_values": checked, "sample_std_ddof": 1}


def _verify_artifact_manifest(root: Path) -> dict[str, Any]:
    artifact_manifest = _json(_require_file(root / "summary" / "artifact_manifest.json"))
    rows = artifact_manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        _fail("PrivateTest artifact manifest is empty or malformed")
    for row in rows:
        source = _require_file(root / str(row["path"]))
        if source.stat().st_size != int(row["size_bytes"]) or _sha256(source) != row["sha256"]:
            _fail(f"PrivateTest artifact manifest mismatch: {source}")
    return {"status": "passed", "checked_files": len(rows)}


def _verify_archive_matches_extracted(root: Path, archive_path: Path) -> dict[str, Any]:
    checked = 0
    with tarfile.open(_require_file(archive_path), "r") as archive:
        for member in archive.getmembers():
            prefix = "./final-private-test-v2/"
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            relative = member.name.removeprefix(prefix)
            handle = archive.extractfile(member)
            if handle is None:
                _fail(f"archive member cannot be read: {member.name}")
            extracted_path = _require_file(root / relative)
            if hashlib.sha256(handle.read()).hexdigest() != _sha256(extracted_path):
                _fail(f"archive/extracted mismatch: {relative}")
            checked += 1
    if checked == 0:
        _fail("archive contains no final-private-test-v2 files")
    return {"status": "passed", "checked_files": checked}


def generate(private_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(private_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        _fail(f"PrivateTest root does not exist: {root}")
    if output.exists() and any(output.iterdir()):
        _fail(f"output directory must be new or empty: {output}")
    for name in OUTLINE:
        (output / name).mkdir(parents=True, exist_ok=True)
    frame, values, sources, clean_per_class, clean_confusion, classwise_frame = _load_metrics(root)
    frozen_summary_check = _verify_frozen_summaries(root, values)
    artifact_manifest_check = _verify_artifact_manifest(root)
    archive_path = root.parent.parent / "fer2013-private-test-final-v2.tar"
    archive_extracted_check = _verify_archive_matches_extracted(root, archive_path)
    tables = _private_stage8_tables(frame)
    source_dir = output / "source_data"
    for name, table in tables.items():
        table.to_csv(source_dir / f"{name}.csv", index=False, float_format="%.10f")
    frame.to_csv(source_dir / "private_metric_rows.csv", index=False, float_format="%.10f")
    for name in ("strategy_condition_summary.csv", "clean_to_occluded_drop.csv", "paired_strategy_differences.csv"):
        shutil.copyfile(root / "summary" / name, source_dir / f"private_original_{name}")
    exp1_frame, exp1_summary, exp1_runs = _exp1_tables(values, clean_per_class, output / "experiment1")
    exp1_summary.to_csv(source_dir / "exp1_private_clean_summary.csv", index=False, float_format="%.10f")
    confusion_runs = [SimpleNamespace(seed=seed, confusion=clean_confusion[seed]) for seed in EXPECTED_SEEDS]
    confusion_table = build_confusion_table(confusion_runs)
    confusion_table.to_csv(source_dir / "exp1_private_clean_confusion.csv", index=False, float_format="%.10f")
    classwise_evidence = ClasswiseEvidence(root, root, classwise_frame, ())
    classwise_tables = compute_classwise_summaries(classwise_evidence)
    for name, table in classwise_tables.items():
        table.to_csv(source_dir / f"private_{name}.csv", index=False, float_format="%.10f")

    from occlusion_fer.paper_stage8 import _use_paper_style
    _use_paper_style()
    generated: list[str] = []
    generated += [str(p.relative_to(output)) for p in plot_exp1_summary(exp1_runs, exp1_summary, output / "experiment1", split_label="PrivateTest", stem="exp1_private_clean_baseline")]
    generated += [str(p.relative_to(output)) for p in plot_exp1_confusion(confusion_table, output / "experiment1", split_label="PrivateTest", stem="exp1_private_clean_confusion_matrix")]
    generated += [str(Path("experiment2") / p) for p in _rename_private_files(output / "experiment2", _plot_clean_severity_metric(tables["occlusion_severity_summary"], "accuracy", output / "experiment2"))]
    generated += [str(Path("experiment2") / p) for p in _rename_private_files(output / "experiment2", _plot_clean_severity_metric(tables["occlusion_severity_summary"], "macro_f1", output / "experiment2"))]
    generated += [str(Path("experiment2") / p) for p in _rename_private_files(output / "experiment2", _plot_clean_severity_performance(tables["occlusion_severity_summary"], output / "experiment2"))]
    generated += [str(Path("experiment2") / p) for p in _rename_private_files(output / "experiment2", _plot_classwise_f1_heatmap(classwise_tables["classwise_f1_summary"], output / "experiment2"))]
    generated += [str(Path("experiment2") / p) for p in _rename_private_files(output / "experiment2", _plot_classwise_drop_heatmap(classwise_tables["classwise_drop_from_clean_summary"], output / "experiment2"))]
    generated += [str(Path("experiment3") / p) for p in _rename_private_files(output / "experiment3", _plot_by_condition_comparison(tables["experiment3_comparison_summary"], output / "experiment3"))]
    clean_comparison = tables["experiment3_comparison_summary"].loc[tables["experiment3_comparison_summary"]["condition"] == "clean"]
    generated += [str(Path("experiment3") / p) for p in _plot_grouped_metric_bars(clean_comparison, stem="experiment3_private_clean_performance_comparison", output_dir=output / "experiment3", title="Clean condition")]
    generated += [str(Path("experiment3") / p) for p in _plot_grouped_metric_bars(tables["experiment3_robustness_summary"], stem="experiment3_private_occluded_robustness_summary", output_dir=output / "experiment3", title="Mean across nine occluded conditions")]
    generated += [str(Path("comparisons") / p) for p in _rename_private_files(output / "comparisons", _plot_gain_heatmap(tables["experiment3_robustness_gain"], output / "comparisons"))]

    sidecar_path = root.parent.parent / "fer2013-private-test-final-v2.sha256"
    sidecar_parts = sidecar_path.read_text(encoding="utf-8").split() if sidecar_path.is_file() else []
    archive_sha = _sha256(archive_path)
    sidecar_sha = sidecar_parts[0] if sidecar_parts else None
    manifest = {
        "schema_version": 1,
        "split": "PrivateTest",
        "internal_split": "test",
        "private_root": str(root),
        "private_archive": {
            "path": str(archive_path),
            "sha256_computed": archive_sha,
            "sha256_sidecar": sidecar_sha,
            "sidecar_declared_path": sidecar_parts[1] if len(sidecar_parts) > 1 else None,
            "sha256_matches_sidecar": archive_sha == sidecar_sha,
        },
        "private_dataset_sha256": EXPECTED_DATASET_SHA,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "plan_sha256": EXPECTED_PLAN_SHA,
        "seeds": list(EXPECTED_SEEDS),
        "conditions": list(CLASSWISE_CONDITIONS),
        "metrics": list(METRICS),
        "frozen_summary_verification": frozen_summary_check,
        "artifact_manifest_verification": artifact_manifest_check,
        "archive_extracted_verification": archive_extracted_check,
        "plotting_scripts": {
            "experiment1": "occlusion_fer.paper_formal_completion._plot_summary",
            "experiment2_3": "occlusion_fer.paper_stage8._plot_clean_severity_metric/_plot_clean_severity_performance/_plot_by_condition_comparison/_plot_grouped_metric_bars/_plot_gain_heatmap",
        },
        "source_files": sources,
        "figure_provenance": {
            "experiment1/exp1_private_clean_baseline": {"source_glob": "clean/seed{42,123,2026}/conditions/clean/{metrics,per_class_metrics}.(json|csv)", "fields": ["accuracy", "macro_f1", "per_class.f1"], "seeds": list(EXPECTED_SEEDS), "conditions": ["clean"], "metrics": ["accuracy", "macro_f1", "per_class_f1"], "script": "occlusion_fer.paper_formal_completion._plot_summary"},
            "experiment1/exp1_private_clean_confusion_matrix": {"source_glob": "clean/seed{42,123,2026}/conditions/clean/confusion_matrix.csv", "fields": ["predicted_*"], "seeds": list(EXPECTED_SEEDS), "conditions": ["clean"], "metrics": ["row-normalized confusion"], "script": "occlusion_fer.paper_formal_completion._plot_confusion"},
            "experiment2/private_occlusion_severity_performance": {"source_glob": "clean/seed{42,123,2026}/conditions/*/{metrics.json}", "fields": ["accuracy", "macro_f1"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_CONDITIONS), "metrics": list(METRICS), "script": "occlusion_fer.paper_stage8._plot_clean_severity_performance"},
            "experiment2/private_occlusion_severity_accuracy": {"source_glob": "clean/seed{42,123,2026}/conditions/*/metrics.json", "fields": ["accuracy"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_CONDITIONS), "metrics": ["accuracy"], "script": "occlusion_fer.paper_stage8._plot_clean_severity_metric"},
            "experiment2/private_occlusion_severity_macro_f1": {"source_glob": "clean/seed{42,123,2026}/conditions/*/metrics.json", "fields": ["macro_f1"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_CONDITIONS), "metrics": ["macro_f1"], "script": "occlusion_fer.paper_stage8._plot_clean_severity_metric"},
            "experiment2/private_classwise_f1_heatmap": {"source_glob": "clean/seed{42,123,2026}/conditions/*/per_class_metrics.csv", "fields": ["f1"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_CONDITIONS), "metrics": ["per_class_f1"], "script": "occlusion_fer.paper_figures._plot_classwise_f1_heatmap"},
            "experiment2/private_classwise_drop_from_clean_heatmap": {"source_glob": "clean/seed{42,123,2026}/conditions/*/per_class_metrics.csv", "fields": ["f1"], "definition": "clean minus occluded, paired within seed", "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_OCCLUDED_CONDITIONS), "metrics": ["per_class_f1_drop"], "script": "occlusion_fer.paper_figures._plot_classwise_drop_heatmap"},
            "experiment3/private_experiment3_by_condition_comparison": {"source_glob": "{clean,mixed}/seed{42,123,2026}/conditions/*/metrics.json", "fields": ["accuracy", "macro_f1"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_CONDITIONS), "metrics": list(METRICS), "script": "occlusion_fer.paper_stage8._plot_by_condition_comparison"},
            "experiment3/experiment3_private_clean_performance_comparison": {"source_glob": "{clean,mixed}/seed{42,123,2026}/conditions/clean/metrics.json", "fields": ["accuracy", "macro_f1"], "seeds": list(EXPECTED_SEEDS), "conditions": ["clean"], "metrics": list(METRICS), "script": "occlusion_fer.paper_stage8._plot_grouped_metric_bars"},
            "experiment3/experiment3_private_occluded_robustness_summary": {"source_glob": "{clean,mixed}/seed{42,123,2026}/conditions/{9 occluded conditions}/metrics.json", "fields": ["accuracy", "macro_f1"], "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_OCCLUDED_CONDITIONS), "metrics": list(METRICS), "script": "occlusion_fer.paper_stage8._plot_grouped_metric_bars"},
            "comparisons/private_experiment3_robustness_gain_heatmap": {"source_glob": "{clean,mixed}/seed{42,123,2026}/conditions/*/metrics.json", "fields": ["accuracy", "macro_f1"], "definition": "mixed minus clean, paired within seed", "seeds": list(EXPECTED_SEEDS), "conditions": list(CLASSWISE_OCCLUDED_CONDITIONS), "metrics": list(METRICS), "script": "occlusion_fer.paper_stage8._plot_gain_heatmap"},
        },
        "generated_files": sorted(generated),
        "not_generated": [
            {"item": "Experiment 1 training curves", "reason": "MISSING_PRIVATE_RESULT: PrivateTest archive contains final metrics only, no training history."},
            {"item": "protocol/example figure", "reason": "Method figure is split-independent and would duplicate the existing PublicTest protocol image."},
            {"item": "separate overall Experiment 1/2/3 comparison", "reason": "No distinct existing results-figure script; the by-condition comparison and robustness-gain heatmap already cover the non-duplicative comparison evidence."},
        ],
        "verification": {
            "metric_values_match_condition_results": True,
            "mean_and_sample_std_match_frozen_summary": True,
            "all_seeds_present": True,
            "all_conditions_present": True,
            "condition_mapping_verified": True,
            "strategy_mapping_verified": True,
            "metric_mapping_verified": True,
            "publictest_data_read_for_figures": False,
            "private_source_files_match_artifact_manifest": True,
            "archive_matches_extracted_results": True,
            "archive_sha256_matches_sidecar": archive_sha == sidecar_sha,
            "private_figures_verified": archive_sha == sidecar_sha,
        },
        "command": shlex.join([sys.executable, "-m", "occlusion_fer.private_paper_figures", "--private-root", str(root), "--output-dir", str(output)]),
    }
    _write_manifest(output / "figure_manifest.json", manifest)
    verification_status = "YES" if archive_sha == sidecar_sha else "NO"
    (output / "README_PRIVATE_FIGURES.md").write_text(
        f"""# PrivateTest Paper Figures

These are final hold-out visualizations generated only from `fer2013-private-test-final-v2`. No checkpoint selection, tuning, training, inference, or source-result modification occurred. Means and sample standard deviations use seeds 42, 123, and 2026 with `ddof=1`.

## A. Generated Figures

- `experiment1/exp1_private_clean_baseline`: three clean-only seeds, mean +/- sample SD, and clean per-class F1.
- `experiment1/exp1_private_clean_confusion_matrix`: mean row-normalized clean confusion across three seeds.
- `experiment2/private_occlusion_severity_performance`: main accuracy and macro-F1 severity figure.
- `experiment2/private_occlusion_severity_accuracy`: accuracy-only counterpart.
- `experiment2/private_occlusion_severity_macro_f1`: macro-F1-only counterpart.
- `experiment2/private_classwise_f1_heatmap`: class-wise absolute F1 over all ten conditions.
- `experiment2/private_classwise_drop_from_clean_heatmap`: paired class-wise clean-to-occluded F1 drop.
- `experiment3/private_experiment3_by_condition_comparison`: Clean-only versus Mixed training for all locations, ratios, and metrics.
- `experiment3/experiment3_private_clean_performance_comparison`: clean-condition strategy comparison.
- `experiment3/experiment3_private_occluded_robustness_summary`: mean across nine occluded conditions.
- `comparisons/private_experiment3_robustness_gain_heatmap`: paired Mixed-training minus Clean-only gains.

Each figure has a 300 dpi PNG and PDF; both class-wise heatmaps also retain the existing SVG export.

## B. Reused Plotting Code

- Experiment 1 reuses `paper_formal_completion.py::_plot_summary` and `_plot_confusion`; only split labels and output stems are parameterized.
- Experiment 2/3 reuse `paper_stage8.py` severity, by-condition, grouped-bar, and gain-heatmap functions.
- Class-wise figures reuse `paper_figures.py` classwise aggregation and heatmap functions.
- The new `private_paper_figures.py` module is a strict input/provenance adapter, not a second plotting system.

## C. Data Provenance

All plotted values come from `{root}`. Per-figure source paths, fields, seeds, conditions, metrics, and plotting functions are recorded in `figure_manifest.json`. Frozen and recomputed tables are in `source_data/`.

## D. Not Generated

- `MISSING_PRIVATE_RESULT`: Experiment 1 training curves, because the final-evaluation archive contains no training history.
- Protocol examples, because the method figure is split-independent and duplicating it would not add PrivateTest evidence.
- A separate overall Experiment 1/2/3 chart, because no distinct existing formal script exists and it would duplicate the by-condition/gain figures.

## E. Paper Recommendation

- Section 4.1: `exp1_private_clean_baseline`.
- Section 4.2: `private_occlusion_severity_performance` and `private_classwise_drop_from_clean_heatmap`.
- Section 4.3: `private_experiment3_by_condition_comparison` and `private_experiment3_robustness_gain_heatmap`.

## F. Verification

- 60 condition results, seeds 42/123/2026, ten canonical conditions, metric/strategy mappings: PASS.
- Recomputed means/sample SDs and frozen summary/drop/difference CSVs: PASS ({frozen_summary_check['checked_numeric_values']} numeric values).
- Internal artifact manifest: PASS ({artifact_manifest_check['checked_files']} files).
- Current tar contents versus extracted result tree: PASS ({archive_extracted_check['checked_files']} files).
- PublicTest data used for figure values: NO.
- Existing PrivateTest source files modified: NO.
- External checksum sidecar: FAIL. The local `.tar` SHA-256 is `{archive_sha}`, while the sidecar declares `{sidecar_sha}` for `{sidecar_parts[1] if len(sidecar_parts) > 1 else 'an unspecified file'}`.

`PRIVATE_FIGURES_VERIFIED: {verification_status}`
""",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate formal PrivateTest paper figures from frozen final results")
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = generate(args.private_root, args.output_dir)
    print(f"output_dir={Path(args.output_dir).expanduser().resolve()}")
    print(f"generated_files={len(result['generated_files'])}")
    print("private_split=PrivateTest")


if __name__ == "__main__":
    main()
