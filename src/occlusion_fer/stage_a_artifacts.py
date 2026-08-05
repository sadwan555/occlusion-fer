"""Narrow command-line entry points for deterministic Stage A artifacts."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from occlusion_fer.config import load_occlusion_config
from occlusion_fer.mask_manifest import (
    MASKED_CONDITIONS,
    ManifestWriteResult,
    generate_and_write_official_manifest,
)
from occlusion_fer.training_mean import (
    ArtifactWriteResult,
    TrainingMeanArtifact,
    build_training_mean_artifact,
    load_training_mean_artifact,
    validate_artifact_dataset,
    write_training_mean_artifact,
)


OFFICIAL_TRAINING_SAMPLE_COUNT = 28709
OFFICIAL_TRAINING_PIXEL_COUNT = 66145536
OFFICIAL_PUBLICTEST_SAMPLE_COUNT = 3589
OFFICIAL_MANIFEST_ROW_COUNT = 32301
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


TrainingMeanService = Callable[
    [Path, Path], tuple[TrainingMeanArtifact, ArtifactWriteResult]
]
ValidationManifestService = Callable[
    [Path, Path, Path], tuple[TrainingMeanArtifact, ManifestWriteResult]
]
ConfigLoader = Callable[[Path], object]


@dataclass(frozen=True)
class CliServices:
    """Private Python dependency seam; no CLI or environment override exists."""

    config_loader: ConfigLoader
    training_mean: TrainingMeanService
    validation_manifest: ValidationManifestService


def build_parser() -> argparse.ArgumentParser:
    """Build the only two approved Stage A artifact commands."""
    parser = argparse.ArgumentParser(
        prog="python -m occlusion_fer.stage_a_artifacts",
        description="Generate deterministic Stage A artifacts for FER2013.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    training_parser = subparsers.add_parser(
        "training-mean",
        help="write the canonical Training mean artifact",
        allow_abbrev=False,
    )
    training_parser.add_argument("--config", required=True, type=Path)
    training_parser.add_argument("--data-path", required=True, type=Path)
    training_parser.add_argument("--output-dir", required=True, type=Path)

    manifest_parser = subparsers.add_parser(
        "validation-manifest",
        help="write the canonical PublicTest evaluation mask manifest",
        allow_abbrev=False,
    )
    manifest_parser.add_argument("--config", required=True, type=Path)
    manifest_parser.add_argument("--data-path", required=True, type=Path)
    manifest_parser.add_argument("--mean-artifact", required=True, type=Path)
    manifest_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    services: CliServices | None = None,
) -> int:
    """Run one fixed artifact operation and return its process exit status."""
    parser = build_parser()
    parsed = parser.parse_args(argv)
    argument_vector = list(sys.argv[1:] if argv is None else argv)
    if argv is None:
        command = list(sys.argv)
    else:
        command = ["occlusion_fer.stage_a_artifacts", *argument_vector]
    _write_process_event(parsed.operation, command)
    active_services = services or _default_services()

    try:
        active_services.config_loader(parsed.config)
        _require_input_file(parsed.data_path, "data path")
        if parsed.operation == "validation-manifest":
            _require_input_file(parsed.mean_artifact, "mean artifact path")
        output_dir = _create_output_directory(parsed.output_dir)

        if parsed.operation == "training-mean":
            artifact_path = output_dir / "training_mean.json"
            artifact, write_result = active_services.training_mean(
                parsed.data_path,
                artifact_path,
            )
            summary = _training_summary(artifact, write_result, artifact_path)
        else:
            artifact_path = output_dir / "validation_mask_manifest.csv"
            artifact, write_result = active_services.validation_manifest(
                parsed.data_path,
                parsed.mean_artifact,
                artifact_path,
            )
            summary = _validation_summary(artifact, write_result, artifact_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(_json_line(summary), end="")
    return 0


def _default_services() -> CliServices:
    return DEFAULT_SERVICES


def _run_training_mean(
    data_path: Path,
    target_path: Path,
) -> tuple[TrainingMeanArtifact, ArtifactWriteResult]:
    artifact = build_training_mean_artifact(data_path)
    result = write_training_mean_artifact(target_path, artifact)
    return artifact, result


def _run_validation_manifest(
    data_path: Path,
    mean_artifact_path: Path,
    target_path: Path,
) -> tuple[TrainingMeanArtifact, ManifestWriteResult]:
    artifact = load_training_mean_artifact(mean_artifact_path)
    validate_artifact_dataset(artifact, data_path)
    result = generate_and_write_official_manifest(
        data_path,
        mean_artifact_path,
        target_path,
    )
    return artifact, result


def _training_summary(
    artifact: TrainingMeanArtifact,
    result: ArtifactWriteResult,
    target_path: Path,
) -> dict[str, object]:
    _validate_artifact_identity(artifact)
    _validate_result(result, target_path)
    if artifact.sample_count != OFFICIAL_TRAINING_SAMPLE_COUNT:
        raise ValueError("Training mean sample count is not official")
    if artifact.pixel_count != OFFICIAL_TRAINING_PIXEL_COUNT:
        raise ValueError("Training mean pixel count is not official")
    return {
        "operation": "training-mean",
        "status": "written",
        "artifact_path": str(target_path),
        "artifact_sha256": result.sha256,
        "dataset_sha256": artifact.dataset_sha256,
        "sample_count": OFFICIAL_TRAINING_SAMPLE_COUNT,
        "pixel_count": OFFICIAL_TRAINING_PIXEL_COUNT,
    }


def _validation_summary(
    artifact: TrainingMeanArtifact,
    result: ManifestWriteResult,
    target_path: Path,
) -> dict[str, object]:
    _validate_artifact_identity(artifact)
    _validate_manifest_result(result, target_path)
    return {
        "operation": "validation-manifest",
        "status": result.status,
        "artifact_path": str(target_path),
        "artifact_sha256": result.sha256,
        "dataset_sha256": artifact.dataset_sha256,
        "sample_count": OFFICIAL_PUBLICTEST_SAMPLE_COUNT,
        "row_count": OFFICIAL_MANIFEST_ROW_COUNT,
        "condition_counts": {
            condition: OFFICIAL_PUBLICTEST_SAMPLE_COUNT
            for condition in MASKED_CONDITIONS
        },
    }


def _validate_artifact_identity(artifact: TrainingMeanArtifact) -> None:
    if not isinstance(artifact, TrainingMeanArtifact):
        raise TypeError("artifact service returned an invalid TrainingMeanArtifact")
    if artifact.dataset_name != "fer2013" or artifact.split != "train":
        raise ValueError("artifact does not identify FER2013 Training")
    if _SHA256_PATTERN.fullmatch(artifact.dataset_sha256) is None:
        raise ValueError("artifact dataset_sha256 must be lowercase hexadecimal")


def _validate_result(result: ArtifactWriteResult, target_path: Path) -> None:
    if not isinstance(result, ArtifactWriteResult):
        raise TypeError("training service returned an invalid write result")
    if result.path != target_path:
        raise ValueError("training service returned an unexpected target path")
    if _SHA256_PATTERN.fullmatch(result.sha256) is None:
        raise ValueError("artifact_sha256 must be lowercase hexadecimal")


def _validate_manifest_result(
    result: ManifestWriteResult,
    target_path: Path,
) -> None:
    if not isinstance(result, ManifestWriteResult):
        raise TypeError("manifest service returned an invalid write result")
    if result.path != target_path:
        raise ValueError("manifest service returned an unexpected target path")
    if result.status not in {"created", "consistent"}:
        raise ValueError("manifest service returned an invalid status")
    if _SHA256_PATTERN.fullmatch(result.sha256) is None:
        raise ValueError("artifact_sha256 must be lowercase hexadecimal")


def _require_input_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} does not exist or is not a file: {path}"
        )


def _create_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"output directory is not a directory: {path}")
    return path


def _write_process_event(operation: str, command: list[str]) -> None:
    event = {
        "event": "stage-a-artifact-run",
        "created_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "operation": operation,
        "command": command,
    }
    sys.stderr.write(_json_line(event))


def _json_line(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


DEFAULT_SERVICES = CliServices(
    config_loader=load_occlusion_config,
    training_mean=_run_training_mean,
    validation_manifest=_run_validation_manifest,
)


if __name__ == "__main__":
    raise SystemExit(main())
