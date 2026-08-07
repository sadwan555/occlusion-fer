from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

from occlusion_fer.mask_manifest import load_manifest_v2
from occlusion_fer.permitted_splits import load_stage_b_source
from occlusion_fer.stage_b_artifacts import build_stage_b_artifacts
from occlusion_fer.training_mean import (
    load_training_mean_v2,
    training_mean_v2_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_combined_csv(tmp_path: Path) -> Path:
    source = tmp_path / "fer2013.csv"
    pixels = " ".join(["11"] * (48 * 48))
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerow(["0", pixels, "Training"])
        writer.writerow(["NOT PARSED", "DO NOT PARSE", "PrivateTest"])
        writer.writerow(["1", pixels, "PublicTest"])
    return source


def test_builds_bound_v2_artifacts_from_usage_routed_csv(tmp_path) -> None:
    source = _write_combined_csv(tmp_path)
    output = tmp_path / "artifacts"

    summary = build_stage_b_artifacts(
        source,
        output,
        require_official=False,
    )

    splits = load_stage_b_source(source)
    mean = load_training_mean_v2(output / "training_mean_v2.json")
    rows, envelope = load_manifest_v2(
        output / "publictest_manifest_v2.csv",
        output / "publictest_manifest_v2.json",
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=training_mean_v2_sha256(mean),
    )
    assert len(rows) == 9
    assert envelope.row_count == 9
    assert summary["source_kind"] == "combined_csv_usage_routed"
    assert summary["excluded_official_split"] == "PrivateTest"
    assert summary["training_sample_count"] == 1
    assert summary["publictest_sample_count"] == 1
    assert (output / "artifact_generation.json").is_file()


def test_formal_builder_rejects_nonofficial_counts_without_output(tmp_path) -> None:
    source = _write_combined_csv(tmp_path)
    output = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="official Training sample count"):
        build_stage_b_artifacts(source, output)

    assert not output.exists()


def test_builder_refuses_existing_output_directory(tmp_path) -> None:
    source = _write_combined_csv(tmp_path)
    output = tmp_path / "artifacts"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        build_stage_b_artifacts(source, output, require_official=False)


def test_cli_exposes_no_split_or_private_test_controls() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "occlusion_fer.stage_b_artifacts", "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--data-path" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--split" not in result.stdout
    assert "--private-test" not in result.stdout
    assert result.stderr == ""
