"""Tests for the fail-closed formal paper-figure pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from occlusion_fer.paper_figures import (
    EXPECTED_BEST_EPOCHS,
    EXPECTED_FORMAL_COMMIT,
    EXPECTED_PUBLICTEST_SUPPORT,
    LABEL_NAMES,
    PaperFigureError,
    compute_summaries,
    generate_figures,
    load_run_evidence,
    parse_preflight_log,
    raw_curve_points,
    row_normalize_confusion,
)


def _write_formal_run(root: Path, seed: int, *, f1: float | None = None) -> Path:
    run = root / f"run-{seed}"
    validation = run / "validation"
    validation.mkdir(parents=True)
    best_epoch = EXPECTED_BEST_EPOCHS[seed]
    f1_value = f1 if f1 is not None else 0.8
    history = pd.DataFrame(
        {
            "epoch": range(1, 31),
            "train_samples": [28709] * 30,
            "train_loss": np.linspace(1.2, 0.01, 30),
            "validation_samples": [3589] * 30,
            "validation_loss": np.linspace(1.1, 1.9, 30),
            "validation_accuracy": np.linspace(0.5, 0.6, 30),
            "validation_macro_f1": np.linspace(0.4, 0.7, 30),
            "train_seconds": [1.0] * 30,
            "validation_seconds": [0.1] * 30,
            "epoch_seconds": [1.1] * 30,
            "train_samples_per_second": [1000.0] * 30,
            "cuda_peak_memory_bytes": [1000] * 30,
            "updated_best_checkpoint": [False] * 30,
        }
    )
    history.loc[best_epoch - 1, "updated_best_checkpoint"] = True
    history.loc[best_epoch - 1, "validation_accuracy"] = 0.6
    history.loc[best_epoch - 1, "validation_loss"] = 1.9
    history.loc[best_epoch - 1, "validation_macro_f1"] = f1_value
    history.to_csv(run / "history.csv", index=False)
    matrix = np.diag(EXPECTED_PUBLICTEST_SUPPORT).tolist()
    per_class = [
        {
            "label": label,
            "label_name": label_name,
            "precision": f1_value,
            "recall": f1_value,
            "f1": f1_value,
            "support": support,
        }
        for label, (label_name, support) in enumerate(
            zip(LABEL_NAMES, EXPECTED_PUBLICTEST_SUPPORT)
        )
    ]
    best_metrics = {
        "accuracy": 0.6,
        "condition": "clean",
        "confusion_matrix": matrix,
        "label_order": [
            {"label": label, "label_name": label_name}
            for label, label_name in enumerate(LABEL_NAMES)
        ],
        "loss": 1.9,
        "macro_f1": f1_value,
        "metric_schema_version": 1,
        "per_class": per_class,
        "sample_count": 3589,
        "split": "validation",
    }
    (validation / "best_metrics.json").write_text(
        json.dumps(best_metrics), encoding="utf-8"
    )
    pd.DataFrame(per_class).to_csv(
        validation / "best_per_class_metrics.csv", index=False
    )
    confusion_rows = []
    for label, label_name in enumerate(LABEL_NAMES):
        row = {
            "true_label": label,
            "true_label_name": label_name,
        }
        row.update({column: matrix[label][index] for index, column in enumerate(
            [f"predicted_{name}" for name in LABEL_NAMES]
        )})
        confusion_rows.append(row)
    pd.DataFrame(confusion_rows).to_csv(
        validation / "best_confusion_matrix.csv", index=False
    )
    (run / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "status": "completed",
                "training_mode": "clean",
                "git_commit": EXPECTED_FORMAL_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump({"training": {"seed": seed}}), encoding="utf-8"
    )
    return run


def _write_preflight(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "train_samples=28709",
                "validation_samples=3589",
                "train_class_counts="
                "0:angry=3995, 1:disgust=436, 2:fear=4097, "
                "3:happy=7215, 4:sad=4830, 5:surprise=3171, 6:neutral=4965",
                "validation_class_counts="
                "0:angry=467, 1:disgust=56, 2:fear=496, "
                "3:happy=895, 4:sad=653, 5:surprise=415, 6:neutral=607",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_formal_run_loads_and_validates(tmp_path: Path) -> None:
    run = _write_formal_run(tmp_path, 42)
    evidence = load_run_evidence(run)
    assert evidence.seed == 42
    assert evidence.best_epoch == 30
    assert evidence.history["epoch"].tolist() == list(range(1, 31))


@pytest.mark.parametrize(
    ("metadata_field", "metadata_value"),
    [("status", "failed"), ("training_mode", "mixed")],
)
def test_run_contract_rejects_metadata_values(
    tmp_path: Path, metadata_field: str, metadata_value: str
) -> None:
    run = _write_formal_run(tmp_path, 42)
    path = run / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[metadata_field] = metadata_value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)


def test_run_contract_rejects_seed_and_config_mismatch(tmp_path: Path) -> None:
    run = _write_formal_run(tmp_path, 42)
    config = {"training": {"seed": 123}}
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)


@pytest.mark.parametrize("field,value", [("condition", "upper_face"), ("split", "test")])
def test_best_metrics_rejects_condition_or_split(
    tmp_path: Path, field: str, value: str
) -> None:
    run = _write_formal_run(tmp_path, 42)
    path = run / "validation" / "best_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics[field] = value
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)


def test_rejects_label_order_missing_field_nan_and_incomplete_epoch(tmp_path: Path) -> None:
    run = _write_formal_run(tmp_path, 42)
    metrics_path = run / "validation" / "best_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["label_order"][0]["label_name"] = "happy"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)

    run = _write_formal_run(tmp_path / "missing", 42)
    history_path = run / "history.csv"
    history = pd.read_csv(history_path).drop(columns=["train_loss"])
    history.to_csv(history_path, index=False)
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)

    run = _write_formal_run(tmp_path / "nan", 42)
    history_path = run / "history.csv"
    history = pd.read_csv(history_path)
    history.loc[0, "validation_loss"] = np.inf
    history.to_csv(history_path, index=False)
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)

    run = _write_formal_run(tmp_path / "epoch", 42)
    history_path = run / "history.csv"
    history = pd.read_csv(history_path).iloc[:-1]
    history.to_csv(history_path, index=False)
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)


def test_rejects_support_and_confusion_mismatches(tmp_path: Path) -> None:
    run = _write_formal_run(tmp_path, 42)
    per_path = run / "validation" / "best_per_class_metrics.csv"
    per_class = pd.read_csv(per_path)
    per_class.loc[0, "support"] = 466
    per_class.to_csv(per_path, index=False)
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)

    run = _write_formal_run(tmp_path / "confusion", 42)
    confusion_path = run / "validation" / "best_confusion_matrix.csv"
    confusion = pd.read_csv(confusion_path)
    confusion.loc[0, "predicted_angry"] = 0
    confusion.to_csv(confusion_path, index=False)
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)

    run = _write_formal_run(tmp_path / "embedded", 42)
    metrics_path = run / "validation" / "best_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["confusion_matrix"][0][0] = 0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(PaperFigureError):
        load_run_evidence(run)


def test_preflight_parser_is_strict_and_checks_totals(tmp_path: Path) -> None:
    path = _write_preflight(tmp_path / "preflight.log")
    distribution, totals = parse_preflight_log(path)
    assert totals == {"Training": 28709, "PublicTest": 3589}
    assert distribution.groupby("split")["count"].sum().to_dict() == {
        "PublicTest": 3589,
        "Training": 28709,
    }
    path.write_text(path.read_text(encoding="utf-8").replace("=3589\n", "=0\n"), encoding="utf-8")
    with pytest.raises(PaperFigureError):
        parse_preflight_log(path)

    malformed = _write_preflight(tmp_path / "malformed.log")
    malformed.write_text(
        malformed.read_text(encoding="utf-8").replace(
            "0:angry=3995", "angry=3995"
        ),
        encoding="utf-8",
    )
    with pytest.raises(PaperFigureError):
        parse_preflight_log(malformed)


def test_ddof_row_normalization_and_raw_curve_values(tmp_path: Path) -> None:
    runs = [
        load_run_evidence(_write_formal_run(tmp_path / str(seed), seed, f1=f1))
        for seed, f1 in ((42, 0.2), (123, 0.4), (2026, 0.6))
    ]
    distribution, _ = parse_preflight_log(_write_preflight(tmp_path / "preflight.log"))
    tables = compute_summaries(runs, distribution)
    row = tables["per_class_f1_summary"].iloc[0]
    assert row["f1_mean"] == pytest.approx(0.4)
    assert row["f1_sample_sd"] == pytest.approx(0.2)
    normalized = row_normalize_confusion([[1, 1], [0, 2]])
    np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.0, 1.0]])
    mean_confusion = tables["mean_normalized_confusion_matrix"].loc[
        :, [f"predicted_{name}" for name in LABEL_NAMES]
    ]
    np.testing.assert_allclose(mean_confusion, np.eye(7))
    np.testing.assert_allclose(
        runs[0].history["train_loss"], np.linspace(1.2, 0.01, 30)
    )
    epochs, plotted_values = raw_curve_points(runs[0], "train_loss")
    np.testing.assert_array_equal(epochs, np.arange(1, 31, dtype=float))
    np.testing.assert_allclose(plotted_values, np.linspace(1.2, 0.01, 30))
    plotted_values[0] = -1
    assert runs[0].history.iloc[0]["train_loss"] == pytest.approx(1.2)


def test_generation_outputs_all_formats_and_manifest(tmp_path: Path) -> None:
    runs = [
        _write_formal_run(tmp_path / str(seed), seed)
        for seed in EXPECTED_BEST_EPOCHS
    ]
    preflight = _write_preflight(tmp_path / "preflight.log")
    output = tmp_path / "figures"
    result = generate_figures(runs, preflight, output, argv=("--test",))
    stems = (
        "clean_training_validation_curves",
        "clean_per_class_f1",
        "clean_validation_confusion_matrix",
        "fer2013_training_publictest_distribution",
    )
    for stem in stems:
        for extension in ("pdf", "svg", "png"):
            assert (output / f"{stem}.{extension}").stat().st_size > 0
    for name in (
        "summary_metrics.csv",
        "per_class_f1_summary.csv",
        "mean_normalized_confusion_matrix.csv",
        "class_distribution.csv",
        "generation_manifest.json",
    ):
        assert (output / name).stat().st_size > 0
    assert result["manifest"]["smoke_test_read"] is False
    assert result["manifest"]["split"] == "validation"
    assert result["manifest"]["class_distribution_splits"] == [
        "Training",
        "PublicTest",
    ]
    assert result["manifest"]["condition"] == "clean"
    assert result["manifest"]["seeds"] == [42, 123, 2026]
    assert "ddof=1" in result["manifest"]["statistical_rules"][
        "cross_seed_standard_deviation"
    ]
    assert len(result["manifest"]["input_sha256"]) == 19
    assert len(result["manifest"]["output_files"]) == 17


def test_generation_rejects_wrong_seed_set(tmp_path: Path) -> None:
    run42 = _write_formal_run(tmp_path / "one", 42)
    run123 = _write_formal_run(tmp_path / "two", 123)
    duplicate42 = _write_formal_run(tmp_path / "three", 42)
    preflight = _write_preflight(tmp_path / "preflight.log")
    with pytest.raises(PaperFigureError):
        generate_figures(
            [run42, run123, duplicate42], preflight, tmp_path / "figures"
        )


def test_output_directory_cannot_be_a_formal_run(tmp_path: Path) -> None:
    runs = [
        _write_formal_run(tmp_path / str(seed), seed)
        for seed in EXPECTED_BEST_EPOCHS
    ]
    preflight = _write_preflight(tmp_path / "preflight.log")
    with pytest.raises(PaperFigureError):
        generate_figures(runs, preflight, runs[0])
