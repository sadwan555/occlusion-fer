"""Focused tests for the current formal paper completion pipeline."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from occlusion_fer.paper_formal_completion import (
    CONFUSION_STEM,
    CURVES_STEM,
    EXPECTED_CONDITIONS,
    EXPECTED_EVALUATION_COMMIT,
    EXPECTED_LABELS,
    EXPECTED_ROOT,
    EXPECTED_SEEDS,
    EXPECTED_TRAINING_COMMIT,
    FormalCompletionError,
    generate_formal_completion,
    load_formal_lineage,
    sample_summary,
)
from occlusion_fer.paper_framework import FIGURE_STEM


BEST_EPOCHS = {42: 34, 123: 47, 2026: 48}
SUPPORT = (467, 56, 496, 895, 653, 415, 607)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_formal_fixture(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    labels: tuple[str, ...] = EXPECTED_LABELS,
) -> tuple[Path, Path]:
    staging = tmp_path / "staging" / EXPECTED_ROOT
    experiment2 = tmp_path / "experiment2-clean"
    for seed_index, seed in enumerate(seeds):
        run = staging / f"seed{seed}"
        validation = run / "validation"
        validation.mkdir(parents=True)
        best_epoch = BEST_EPOCHS[seed]
        accuracy = 0.70 + seed_index * 0.005
        macro_f1 = 0.69 + seed_index * 0.005
        config = {
            "project": {
                "name": "occlusion-fer",
                "experiment_name": "e7_high_resolution_longer",
            },
            "dataset": {"name": "fer2013", "image_size": 224, "num_classes": 7},
            "model": {"name": "resnet18", "pretrained": True},
            "training": {
                "mode": "clean",
                "seed": seed,
                "epochs": 50,
                "batch_size": 128,
                "learning_rate": 0.0001,
                "weight_decay": 0.001,
            },
        }
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        _write_json(
            run / "run_metadata.json",
            {
                "status": "completed",
                "seed": seed,
                "training_mode": "clean",
                "git_commit": EXPECTED_TRAINING_COMMIT,
                "git_dirty": False,
            },
        )
        history = pd.DataFrame(
            {
                "epoch": np.arange(1, 51),
                "train_samples": 28709,
                "train_loss": np.linspace(1.5, 0.5, 50),
                "validation_samples": 3589,
                "validation_accuracy": np.linspace(0.60, 0.68, 50),
                "validation_macro_f1": np.linspace(0.55, 0.67, 50),
                "updated_best_checkpoint": False,
            }
        )
        history.loc[best_epoch - 1, "updated_best_checkpoint"] = True
        history.loc[best_epoch - 1, "validation_accuracy"] = accuracy
        history.loc[best_epoch - 1, "validation_macro_f1"] = macro_f1
        history.to_csv(run / "history.csv", index=False)

        confusion = np.diag(SUPPORT)
        confusion_frame = pd.DataFrame(
            confusion,
            columns=[f"predicted_{label}" for label in EXPECTED_LABELS],
        )
        confusion_frame.insert(0, "true_label_name", EXPECTED_LABELS)
        confusion_frame.insert(0, "true_label", range(7))
        confusion_frame.to_csv(validation / "best_confusion_matrix.csv", index=False)
        per_class = pd.DataFrame(
            {
                "label": range(7),
                "label_name": labels,
                "precision": np.linspace(0.6, 0.9, 7),
                "recall": np.linspace(0.61, 0.91, 7),
                "f1": np.linspace(0.605, 0.905, 7) + seed_index * 0.001,
                "support": SUPPORT,
            }
        )
        per_class.to_csv(validation / "best_per_class_metrics.csv", index=False)
        label_order = [
            {"label": index, "label_name": label}
            for index, label in enumerate(EXPECTED_LABELS)
        ]
        metrics = {
            "condition": "clean",
            "split": "validation",
            "sample_count": 3589,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "loss": 0.9 + seed_index * 0.01,
            "label_order": label_order,
            "confusion_matrix": confusion.tolist(),
        }
        _write_json(validation / "best_metrics.json", metrics)
        _write_json(
            validation / "last_metrics.json",
            {**metrics, "accuracy": accuracy - 0.01, "macro_f1": macro_f1 - 0.01},
        )
        (validation / "best_predictions.csv").write_text(
            "sample_id,true_label,predicted_label\n1,0,0\n", encoding="utf-8"
        )
        torch.save(
            {
                "seed": seed,
                "epoch": best_epoch,
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
                "best_validation_macro_f1": macro_f1,
                "resolved_config": config,
                "model_state_dict": {},
            },
            run / "best.pt",
        )

        conditions = experiment2 / f"seed{seed}" / "conditions"
        conditions.mkdir(parents=True)
        alignment = {
            "best_metrics.json": "clean_metrics.json",
            "best_per_class_metrics.csv": "clean_per_class_metrics.csv",
            "best_confusion_matrix.csv": "clean_confusion_matrix.csv",
            "best_predictions.csv": "clean_predictions.csv",
        }
        for source_name, clean_name in alignment.items():
            (conditions / clean_name).write_bytes((validation / source_name).read_bytes())
        _write_json(
            experiment2 / f"seed{seed}" / "evaluation_provenance.json",
            {
                "seed": seed,
                "checkpoint_sha256": _sha256(run / "best.pt"),
                "training_commit": EXPECTED_TRAINING_COMMIT,
                "evaluation_commit": EXPECTED_EVALUATION_COMMIT,
                "training_git_dirty": False,
                "git_dirty": False,
                "image_height": 224,
                "image_width": 224,
                "protocol": "occlusion-v2-224",
                "conditions": list(EXPECTED_CONDITIONS),
                "resolved_config": {
                    "dataset": {
                        "image_size": 224,
                        "permitted_splits": ["Training", "PublicTest"],
                    },
                    "model": {"name": "resnet18", "pretrained": True},
                    "training": {"mode": "clean", "seed": seed, "epochs": 50},
                    "occlusion": {
                        "evaluation": {"split": "validation"},
                        "protocol": {"algorithm_version": "occlusion-v2-224"},
                    },
                },
            },
        )

    archive_path = tmp_path / "fer2013-clean-e7-formal.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(staging, arcname=EXPECTED_ROOT)
    return archive_path, experiment2


def test_loader_accepts_only_complete_current_formal_lineage(tmp_path: Path) -> None:
    archive, experiment2 = _build_formal_fixture(tmp_path)
    runs = load_formal_lineage(archive, experiment2)
    assert tuple(run.seed for run in runs) == EXPECTED_SEEDS
    assert tuple(run.best_epoch for run in runs) == (34, 47, 48)
    assert all(run.config["dataset"]["image_size"] == 224 for run in runs)
    assert tuple(runs[0].per_class["label_name"]) == EXPECTED_LABELS


def test_loader_rejects_legacy_and_missing_seed(tmp_path: Path) -> None:
    archive, experiment2 = _build_formal_fixture(tmp_path / "complete")
    legacy = tmp_path / "da889bd-formal.tar"
    legacy.write_bytes(archive.read_bytes())
    with pytest.raises(FormalCompletionError, match="da889bd"):
        load_formal_lineage(legacy, experiment2)

    incomplete, incomplete_evaluation = _build_formal_fixture(
        tmp_path / "incomplete", seeds=(42, 123)
    )
    with pytest.raises(FormalCompletionError, match="seed2026"):
        load_formal_lineage(incomplete, incomplete_evaluation)


def test_sample_summary_uses_ddof_one() -> None:
    mean, sample_sd = sample_summary((0.2, 0.4, 0.8))
    assert mean == pytest.approx(np.mean((0.2, 0.4, 0.8)))
    assert sample_sd == pytest.approx(np.std((0.2, 0.4, 0.8), ddof=1))
    assert sample_sd != pytest.approx(np.std((0.2, 0.4, 0.8), ddof=0))


def test_generation_exports_formal_figures_and_audit(tmp_path: Path) -> None:
    archive, experiment2 = _build_formal_fixture(tmp_path / "inputs")
    output = tmp_path / "paper"
    result = generate_formal_completion(archive, experiment2, output)
    for stem in (
        "experiment1_clean_baseline_summary",
        CURVES_STEM,
        CONFUSION_STEM,
        FIGURE_STEM,
    ):
        assert (output / f"{stem}.pdf").stat().st_size > 0
        assert (output / f"{stem}.png").stat().st_size > 0
    assert (output / f"{FIGURE_STEM}.svg").stat().st_size > 0
    audit = result["audit"]
    assert audit["formal_lineage"]["input_size"] == [224, 224]
    assert audit["evidence_checks"]["sample_sd_ddof"] == 1
    assert audit["evidence_checks"]["obsolete_gradcam_inputs_read"] is False
    assert audit["evidence_checks"]["legacy_da889bd_inputs_read"] is False
    summary = pd.read_csv(output / "experiment1_clean_baseline_summary.csv")
    assert set(summary["summary"]) == {
        "individual",
        "mean",
        "sample_sd_ddof_1",
    }
    assert list(
        summary.loc[
            (summary["metric"] == "per_class_f1")
            & (summary["summary"] == "mean"),
            "class",
        ]
    ) == list(EXPECTED_LABELS)
