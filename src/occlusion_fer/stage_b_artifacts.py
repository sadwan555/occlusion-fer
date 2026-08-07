"""Generate locked Stage B artifacts from FER2013 Training/PublicTest rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from occlusion_fer.artifacts import write_json_atomic
from occlusion_fer.mask_manifest import (
    build_manifest_v2_rows,
    make_manifest_v2_envelope,
    validate_manifest_v2,
    write_manifest_v2_create_or_verify,
)
from occlusion_fer.permitted_splits import (
    STAGE_B_SOURCE_ROUTING_VERSION,
    load_stage_b_source,
    stage_b_source_kind,
    validate_official_stage_b_sources,
)
from occlusion_fer.training_mean import (
    calculate_training_mean_v2,
    training_mean_v2_sha256,
    write_training_mean_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m occlusion_fer.stage_b_artifacts",
        description=(
            "Generate the locked Training mean and PublicTest v2 manifest "
            "without parsing excluded PrivateTest label or pixel fields."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def build_stage_b_artifacts(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    require_official: bool = True,
) -> dict[str, object]:
    """Build a mutually bound mean, manifest, and routing provenance set."""
    if type(require_official) is not bool:
        raise ValueError("require_official must be a bool")
    source_path = Path(data_path).expanduser()
    source_kind = stage_b_source_kind(source_path)
    output_path = Path(output_directory).expanduser()
    if output_path.exists():
        raise FileExistsError(
            f"Stage B artifact output directory already exists: {output_path}"
        )

    splits = load_stage_b_source(source_path)
    if require_official:
        validate_official_stage_b_sources(splits)

    mean = calculate_training_mean_v2(splits.training)
    mean_sha256 = training_mean_v2_sha256(mean)
    publictest_sample_ids = [
        record.sample_id for record in splits.publictest.records
    ]
    rows = build_manifest_v2_rows(
        publictest_sample_ids,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha256,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=splits.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha256,
    )
    validate_manifest_v2(rows, envelope, require_official=require_official)

    output_path.mkdir(parents=True, exist_ok=False)
    mean_path = output_path / "training_mean_v2.json"
    manifest_path = output_path / "publictest_manifest_v2.csv"
    sidecar_path = output_path / "publictest_manifest_v2.json"
    write_training_mean_v2(mean_path, mean)
    manifest_result = write_manifest_v2_create_or_verify(
        manifest_path,
        sidecar_path,
        rows,
        envelope,
    )

    provenance = {
        "schema_version": 1,
        "source_routing_version": STAGE_B_SOURCE_ROUTING_VERSION,
        "source_kind": source_kind,
        "selected_official_splits": ["Training", "PublicTest"],
        "excluded_official_split": "PrivateTest",
        "combined_csv_excluded_row_policy": {
            "routing_field_inspected": "Usage",
            "label_parsed": False,
            "pixels_parsed": False,
            "included_in_split_hashes": False,
        },
        "training_sample_count": splits.training.count,
        "publictest_sample_count": splits.publictest.count,
        "training_dataset_sha256": splits.training.dataset_sha256,
        "publictest_dataset_sha256": splits.publictest.dataset_sha256,
        "training_mean_sha256": mean_sha256,
        "manifest_sha256": manifest_result.sha256,
        "artifacts": {
            "training_mean": mean_path.name,
            "manifest": manifest_path.name,
            "manifest_sidecar": sidecar_path.name,
        },
    }
    provenance_path = output_path / "artifact_generation.json"
    write_json_atomic(provenance_path, provenance)
    return {
        **provenance,
        "output_directory": str(output_path),
        "provenance": provenance_path.name,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_stage_b_artifacts(args.data_path, args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            summary,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
