"""Tests for formal class-wise paper heatmaps."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import pytest

from occlusion_fer.paper_figures import (
    CLASSWISE_CONDITIONS,
    CLASSWISE_OCCLUDED_CONDITIONS,
    EXPECTED_OCCLUSION_EVALUATION_COMMIT,
    EXPECTED_OCCLUSION_TRAINING_COMMIT,
    EXPECTED_PUBLICTEST_SUPPORT,
    EXPECTED_SEEDS,
    LABEL_NAMES,
    PaperFigureError,
    compute_classwise_summaries,
    generate_classwise_heatmaps,
    load_classwise_evidence,
)


def _add_tar_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    archive.addfile(info, BytesIO(value))


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _condition_artifacts(
    seed: int, condition_index: int, condition: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    true_labels = np.repeat(
        np.arange(len(LABEL_NAMES), dtype=np.int64),
        np.asarray(EXPECTED_PUBLICTEST_SUPPORT, dtype=np.int64),
    )
    predicted_labels = true_labels.copy()
    errors_per_class = condition_index + EXPECTED_SEEDS.index(seed)
    offset = 0
    for label, support in enumerate(EXPECTED_PUBLICTEST_SUPPORT):
        error_count = min(errors_per_class, support)
        predicted_labels[offset : offset + error_count] = (label + 1) % len(
            LABEL_NAMES
        )
        offset += support
    predictions = pd.DataFrame(
        {
            "sample_id": np.arange(len(true_labels), dtype=np.int64) + 28711,
            "split": "validation",
            "condition": condition,
            "true_label": true_labels,
            "true_label_name": [LABEL_NAMES[value] for value in true_labels],
            "predicted_label": predicted_labels,
            "predicted_label_name": [
                LABEL_NAMES[value] for value in predicted_labels
            ],
            "correct": true_labels == predicted_labels,
        }
    )
    matrix = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=np.int64)
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    true_positive = np.diag(matrix).astype(float)
    predicted_totals = matrix.sum(axis=0)
    true_totals = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted_totals,
        out=np.zeros_like(true_positive),
        where=predicted_totals != 0,
    )
    recall = true_positive / true_totals
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    per_class = pd.DataFrame(
        {
            "label": np.arange(len(LABEL_NAMES)),
            "label_name": LABEL_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": EXPECTED_PUBLICTEST_SUPPORT,
        }
    )
    confusion_rows = []
    for label, label_name in enumerate(LABEL_NAMES):
        confusion_rows.append(
            {
                "true_label": label,
                "true_label_name": label_name,
                **{
                    f"predicted_{predicted_name}": matrix[label, predicted_label]
                    for predicted_label, predicted_name in enumerate(LABEL_NAMES)
                },
            }
        )
    confusion = pd.DataFrame(confusion_rows)
    metrics: dict[str, object] = {
        "condition": condition,
        "split": "validation",
        "sample_count": len(predictions),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class.to_dict(orient="records"),
    }
    return per_class, confusion, predictions, metrics


def _write_formal_archives(
    root: Path, *, corrupt_f1: bool = False
) -> tuple[Path, Path]:
    stage8_path = root / "stage8.tar"
    baseline_path = root / "baseline.tar"
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(stage8_path, "w") as stage8, tarfile.open(
        baseline_path, "w"
    ) as baseline:
        for seed in EXPECTED_SEEDS:
            provenance = {
                "seed": seed,
                "conditions": list(CLASSWISE_CONDITIONS),
                "evaluation_commit": EXPECTED_OCCLUSION_EVALUATION_COMMIT,
                "training_commit": EXPECTED_OCCLUSION_TRAINING_COMMIT,
                "git_dirty": False,
                "training_git_dirty": False,
                "run_role": "formal_masked_evaluation",
            }
            _add_tar_bytes(
                stage8,
                f"clean/seed{seed}/evaluation_provenance.json",
                json.dumps(provenance).encode("utf-8"),
            )
            baseline_prefix = f"formal-e7-4cb1e0f/seed{seed}"
            metadata = {
                "seed": seed,
                "status": "completed",
                "training_mode": "clean",
                "git_commit": EXPECTED_OCCLUSION_TRAINING_COMMIT,
                "git_dirty": False,
            }
            _add_tar_bytes(
                baseline,
                f"{baseline_prefix}/run_metadata.json",
                json.dumps(metadata).encode("utf-8"),
            )
            clean_artifacts = None
            for condition_index, condition in enumerate(CLASSWISE_CONDITIONS):
                per_class, confusion, predictions, metrics = _condition_artifacts(
                    seed, condition_index, condition
                )
                if corrupt_f1 and seed == 42 and condition == "upper_face_0.20":
                    per_class.loc[0, "f1"] -= 0.1
                prefix = f"clean/seed{seed}/conditions/{condition}"
                _add_tar_bytes(
                    stage8,
                    f"{prefix}_per_class_metrics.csv",
                    _csv_bytes(per_class),
                )
                _add_tar_bytes(
                    stage8,
                    f"{prefix}_confusion_matrix.csv",
                    _csv_bytes(confusion),
                )
                _add_tar_bytes(
                    stage8,
                    f"{prefix}_predictions.csv",
                    _csv_bytes(predictions),
                )
                _add_tar_bytes(
                    stage8,
                    f"{prefix}_metrics.json",
                    json.dumps(metrics).encode("utf-8"),
                )
                if condition == "clean":
                    clean_artifacts = (per_class, confusion, predictions, metrics)
            assert clean_artifacts is not None
            names_and_values = (
                ("best_per_class_metrics.csv", _csv_bytes(clean_artifacts[0])),
                ("best_confusion_matrix.csv", _csv_bytes(clean_artifacts[1])),
                ("best_predictions.csv", _csv_bytes(clean_artifacts[2])),
                ("best_metrics.json", json.dumps(clean_artifacts[3]).encode("utf-8")),
            )
            for name, value in names_and_values:
                _add_tar_bytes(
                    baseline, f"{baseline_prefix}/validation/{name}", value
                )
    return stage8_path, baseline_path


def test_classwise_archive_aggregation_and_generation(tmp_path: Path) -> None:
    stage8, baseline = _write_formal_archives(tmp_path / "evidence")
    evidence = load_classwise_evidence(stage8, baseline)
    assert len(evidence.per_class) == 210
    summaries = compute_classwise_summaries(evidence)
    f1_summary = summaries["classwise_f1_summary"].set_index("label_name")
    raw_angry_clean = evidence.per_class.loc[
        (evidence.per_class["label_name"] == "angry")
        & (evidence.per_class["condition"] == "clean"),
        "f1",
    ]
    assert f1_summary.loc["angry", "clean"] == pytest.approx(
        raw_angry_clean.mean()
    )
    drop_summary = summaries["classwise_drop_from_clean_summary"].set_index(
        "label_name"
    )
    angry = evidence.per_class.loc[
        evidence.per_class["label_name"] == "angry",
        ["seed", "condition", "f1"],
    ]
    angry_clean = angry.loc[angry["condition"] == "clean"].set_index("seed")["f1"]
    angry_upper = angry.loc[
        angry["condition"] == "upper_face_0.20"
    ].set_index("seed")["f1"]
    assert drop_summary.loc["angry", "upper_face_0.20"] == pytest.approx(
        (angry_clean - angry_upper).mean()
    )

    output = tmp_path / "figures"
    result = generate_classwise_heatmaps(
        stage8, baseline, output, argv=("--classwise-test",)
    )
    required = (
        "classwise_f1_heatmap.pdf",
        "classwise_f1_heatmap.png",
        "classwise_drop_from_clean_heatmap.pdf",
        "classwise_drop_from_clean_heatmap.png",
        "classwise_f1_summary.csv",
        "classwise_drop_from_clean_summary.csv",
        "classwise_f1_std.csv",
        "classwise_drop_from_clean_std.csv",
        "classwise_generation_manifest.json",
    )
    for name in required:
        assert (output / name).stat().st_size > 0
    assert result["manifest"]["seeds"] == list(EXPECTED_SEEDS)
    assert result["manifest"]["label_order"] == list(LABEL_NAMES)
    assert result["manifest"]["evidence_checks"][
        "per_class_metrics_reconstructed_from_predictions"
    ] is True


def test_classwise_archive_rejects_unreconstructable_f1(tmp_path: Path) -> None:
    stage8, baseline = _write_formal_archives(
        tmp_path / "corrupt", corrupt_f1=True
    )
    with pytest.raises(PaperFigureError, match="cannot be reconstructed"):
        load_classwise_evidence(stage8, baseline)
