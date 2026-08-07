from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.occlusion_evaluate import CONDITION_ORDER


SEEDS = (42, 123, 2026)


def _condition_accuracy(strategy: str, seed: int, condition_index: int) -> float:
    base = 0.70 if strategy == "clean-only" else 0.72
    return base + SEEDS.index(seed) * 0.01 - condition_index * (
        0.02 if strategy == "clean-only" else 0.01
    )


def _write_evaluation_run(
    path: Path,
    *,
    strategy: str,
    seed: int,
    split: str = "validation",
    manifest_sha: str = "d" * 64,
) -> None:
    mode = "clean" if strategy == "clean-only" else "mixed"
    path.mkdir(parents=True)
    provenance = {
        "checkpoint_sha256": ("a" if strategy == "clean-only" else "b") * 64,
        "training_commit": "training-commit",
        "evaluation_commit": "evaluation-commit",
        "git_dirty": False,
        "training_git_dirty": False,
        "resolved_config": {
            "dataset": {"image_size": 224},
            "training": {"mode": mode, "seed": seed},
        },
        "training_dataset_sha256": "1" * 64,
        "publictest_dataset_sha256": "2" * 64,
        "training_mean_sha256": "3" * 64,
        "manifest_sha256": manifest_sha,
        "protocol": "occlusion-v2-224",
        "image_height": 224,
        "image_width": 224,
        "seed": seed,
        "run_role": f"formal_{mode}",
        "conditions": list(CONDITION_ORDER),
    }
    (path / "evaluation_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    for condition_index, condition in enumerate(CONDITION_ORDER):
        condition_path = path / "conditions" / condition
        condition_path.mkdir(parents=True)
        accuracy = _condition_accuracy(strategy, seed, condition_index)
        macro_f1 = accuracy - 0.05
        per_class = [
            {
                "label": label,
                "label_name": label_name,
                "precision": macro_f1 - label * 0.001,
                "recall": macro_f1 - label * 0.002,
                "f1": macro_f1 - label * 0.0015,
                "support": 10,
            }
            for label, label_name in enumerate(FER2013_LABEL_NAMES)
        ]
        confusion = [[0] * 7 for _ in range(7)]
        for label in range(7):
            confusion[label][label] = 9
            confusion[label][(label + 1) % 7] = 1
        metrics = {
            "metric_schema_version": 1,
            "split": split,
            "condition": condition,
            "sample_count": 70,
            "loss": 1.0 - accuracy,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "label_order": [
                {"label": label, "label_name": name}
                for label, name in enumerate(FER2013_LABEL_NAMES)
            ],
            "per_class": per_class,
            "confusion_matrix": confusion,
        }
        (condition_path / f"{condition}_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )


def _write_training_run(path: Path, *, strategy: str, seed: int) -> None:
    path.mkdir(parents=True)
    (path / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_metadata_schema_version": 1,
                "status": "completed",
                "seed": seed,
                "training_mode": "clean" if strategy == "clean-only" else "mixed",
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    with (path / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_f1",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for epoch in (1, 2, 3):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": 1.5 - epoch * 0.1,
                    "validation_loss": 1.4 - epoch * 0.08,
                    "validation_accuracy": 0.5 + epoch * 0.03,
                    "validation_macro_f1": 0.45 + epoch * 0.03,
                }
            )


def _make_runs(tmp_path: Path):
    evaluation_runs = {"clean-only": {}, "mixed": {}}
    training_runs = {"clean-only": {}, "mixed": {}}
    for strategy in evaluation_runs:
        for seed in SEEDS:
            evaluation_path = tmp_path / "evaluation" / strategy / str(seed)
            training_path = tmp_path / "training" / strategy / str(seed)
            _write_evaluation_run(
                evaluation_path,
                strategy=strategy,
                seed=seed,
            )
            _write_training_run(training_path, strategy=strategy, seed=seed)
            evaluation_runs[strategy][seed] = evaluation_path
            training_runs[strategy][seed] = training_path
    return evaluation_runs, training_runs


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_png(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width >= 1800
        assert image.height >= 900


def test_formal_result_export_writes_complete_tables_and_figures(tmp_path) -> None:
    from occlusion_fer.paper_results import export_formal_results
    from occlusion_fer.paper_figures import sha256_file

    evaluation_runs, training_runs = _make_runs(tmp_path)
    output = tmp_path / "paper-results"
    provenance = export_formal_results(
        evaluation_runs,
        output,
        training_runs=training_runs,
        require_official=False,
        creation_command=["synthetic-result-export"],
    )

    by_seed = _read_csv(output / "condition_metrics_by_seed.csv")
    summary = _read_csv(output / "condition_summary.csv")
    comparison = _read_csv(output / "strategy_comparison.csv")
    per_class = _read_csv(output / "per_class_summary.csv")
    history = _read_csv(output / "training_history_by_seed.csv")
    assert len(by_seed) == 2 * 3 * 10
    assert len(summary) == 2 * 10
    assert len(comparison) == 10
    assert len(per_class) == 2 * 10 * 7
    assert len(history) == 2 * 3 * 3
    clean_summary = next(
        row
        for row in summary
        if row["strategy"] == "clean-only" and row["condition"] == "clean"
    )
    assert float(clean_summary["accuracy_mean"]) == pytest.approx(0.71)
    assert float(clean_summary["accuracy_sample_std"]) == pytest.approx(0.01)
    random_comparison = next(
        row for row in comparison if row["condition"] == "random_rectangle_0.40"
    )
    assert float(random_comparison["accuracy_delta_mean"]) > 0
    assert float(random_comparison["drop_improvement_mean"]) > 0

    for filename in (
        "robustness_accuracy.png",
        "robustness_macro_f1.png",
        "training_curves.png",
    ):
        _assert_png(output / filename)
    confusion_paths = sorted((output / "confusion_matrices").glob("*.png"))
    assert len(confusion_paths) == 20
    _assert_png(confusion_paths[0])
    assert provenance["seed_order"] == list(SEEDS)
    assert provenance["condition_order"] == list(CONDITION_ORDER)
    assert provenance["aggregation_rule"] == "mean_and_sample_standard_deviation"
    assert len(provenance["input_files_sha256"]) == 78
    for relative_path, digest in provenance["output_files_sha256"].items():
        assert sha256_file(output / relative_path) == digest


def test_formal_result_export_rejects_incomplete_or_unsafe_inputs(tmp_path) -> None:
    from occlusion_fer.paper_results import PaperResultError, export_formal_results

    evaluation_runs, _ = _make_runs(tmp_path)
    incomplete = {
        strategy: dict(runs) for strategy, runs in evaluation_runs.items()
    }
    incomplete["mixed"].pop(2026)
    with pytest.raises(PaperResultError, match="42, 123, 2026"):
        export_formal_results(
            incomplete,
            tmp_path / "incomplete",
            require_official=False,
        )

    failed_path = evaluation_runs["mixed"][2026]
    (failed_path / "failure.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PaperResultError, match="failure.json"):
        export_formal_results(
            evaluation_runs,
            tmp_path / "failed",
            require_official=False,
        )
    (failed_path / "failure.json").unlink()

    metrics_path = (
        evaluation_runs["clean-only"][42]
        / "conditions"
        / "clean"
        / "clean_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["split"] = "test"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(PaperResultError, match="PrivateTest|validation"):
        export_formal_results(
            evaluation_runs,
            tmp_path / "private",
            require_official=False,
        )


def test_formal_result_export_rejects_identity_mismatch_and_overwrite(tmp_path) -> None:
    from occlusion_fer.paper_results import PaperResultError, export_formal_results

    evaluation_runs, _ = _make_runs(tmp_path)
    provenance_path = evaluation_runs["mixed"][123] / "evaluation_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["manifest_sha256"] = "e" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(PaperResultError, match="manifest_sha256"):
        export_formal_results(
            evaluation_runs,
            tmp_path / "mismatch",
            require_official=False,
        )

    provenance["manifest_sha256"] = "d" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        export_formal_results(
            evaluation_runs,
            existing,
            require_official=False,
        )
