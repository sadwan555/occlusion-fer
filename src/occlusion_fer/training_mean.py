"""Deterministic Training-split mean artifacts for FER2013."""

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from occlusion_fer.data import Fer2013DataError, load_fer2013_csv


SCHEMA_VERSION = 1
MEAN_ALGORITHM_VERSION = "training-mean-v1"
OFFICIAL_SAMPLE_COUNT = 28709
PIXELS_PER_IMAGE = 48 * 48
OFFICIAL_PIXEL_COUNT = OFFICIAL_SAMPLE_COUNT * PIXELS_PER_IMAGE
_DATASET_NAME = "fer2013"
_SPLIT = "train"
_SOURCE_PIXEL_RANGE = (0, 255)
_NORMALIZED_PIXEL_RANGE = (0.0, 1.0)
_ACCUMULATOR_DTYPE = "uint64"
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_name",
        "split",
        "dataset_sha256",
        "sample_count",
        "pixel_count",
        "raw_pixel_sum",
        "raw_training_mean",
        "source_pixel_range",
        "normalized_pixel_range",
        "accumulator_dtype",
        "mean_algorithm_version",
    }
)


class TrainingMeanError(ValueError):
    """Raised when Training statistics or a mean artifact is invalid."""


@dataclass(frozen=True)
class TrainingStatistics:
    dataset_sha256: str
    sample_count: int
    pixel_count: int
    raw_pixel_sum: int
    raw_training_mean: float
    accumulator_dtype: str


@dataclass(frozen=True)
class TrainingMeanArtifact:
    schema_version: int
    dataset_name: str
    split: str
    dataset_sha256: str
    sample_count: int
    pixel_count: int
    raw_pixel_sum: int
    raw_training_mean: float
    source_pixel_range: tuple[int, int]
    normalized_pixel_range: tuple[float, float]
    accumulator_dtype: str
    mean_algorithm_version: str


@dataclass(frozen=True)
class ArtifactWriteResult:
    path: Path
    sha256: str


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes without interpreting their contents."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"FER2013 CSV file not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_training_statistics(path: str | Path) -> TrainingStatistics:
    """Calculate exact raw-pixel statistics from Training rows only."""
    csv_path = Path(path)
    dataset_sha256 = sha256_file(csv_path)
    try:
        data = load_fer2013_csv(csv_path, include_splits=("train",))
    except Fer2013DataError as exc:
        raise TrainingMeanError(str(exc)) from exc

    accumulator = np.uint64(0)
    pixel_count = 0
    for record in data.records:
        if record.image.dtype != np.uint8 or record.image.shape != (48, 48):
            raise TrainingMeanError(
                "Training images must be uint8 arrays with shape (48, 48)"
            )
        image_sum = record.image.sum(dtype=np.uint64)
        accumulator = _checked_uint64_add(accumulator, image_sum)
        pixel_count += int(record.image.size)

    sample_count = len(data.records)
    if sample_count <= 0 or pixel_count <= 0:
        raise TrainingMeanError("Training split must contain pixels")
    raw_pixel_sum = int(accumulator)
    raw_training_mean = raw_pixel_sum / (255 * pixel_count)
    if not math.isfinite(raw_training_mean) or not 0 <= raw_training_mean <= 1:
        raise TrainingMeanError("raw_training_mean must be finite and between 0 and 1")
    return TrainingStatistics(
        dataset_sha256=dataset_sha256,
        sample_count=sample_count,
        pixel_count=pixel_count,
        raw_pixel_sum=raw_pixel_sum,
        raw_training_mean=raw_training_mean,
        accumulator_dtype=_ACCUMULATOR_DTYPE,
    )


def _checked_uint64_add(current: np.uint64, increment: np.uint64) -> np.uint64:
    """Add two uint64 values after proving the operation cannot overflow."""
    current_value = int(current)
    increment_value = int(increment)
    if current_value > _UINT64_MAX - increment_value:
        raise TrainingMeanError("numpy.uint64 accumulator overflow")
    return np.uint64(current_value + increment_value)


def artifact_from_statistics(statistics: TrainingStatistics) -> TrainingMeanArtifact:
    """Construct a validated canonical artifact from low-level statistics."""
    artifact = TrainingMeanArtifact(
        schema_version=SCHEMA_VERSION,
        dataset_name=_DATASET_NAME,
        split=_SPLIT,
        dataset_sha256=statistics.dataset_sha256,
        sample_count=statistics.sample_count,
        pixel_count=statistics.pixel_count,
        raw_pixel_sum=statistics.raw_pixel_sum,
        raw_training_mean=statistics.raw_training_mean,
        source_pixel_range=_SOURCE_PIXEL_RANGE,
        normalized_pixel_range=_NORMALIZED_PIXEL_RANGE,
        accumulator_dtype=statistics.accumulator_dtype,
        mean_algorithm_version=MEAN_ALGORITHM_VERSION,
    )
    _validate_artifact(artifact)
    return artifact


def build_training_mean_artifact(path: str | Path) -> TrainingMeanArtifact:
    """Build an artifact only when official FER2013 Training counts match."""
    statistics = calculate_training_statistics(path)
    if statistics.sample_count != OFFICIAL_SAMPLE_COUNT:
        raise TrainingMeanError(
            "official Training sample_count must be "
            f"{OFFICIAL_SAMPLE_COUNT}; got {statistics.sample_count}"
        )
    if statistics.pixel_count != OFFICIAL_PIXEL_COUNT:
        raise TrainingMeanError(
            "official Training pixel_count must be "
            f"{OFFICIAL_PIXEL_COUNT}; got {statistics.pixel_count}"
        )
    return artifact_from_statistics(statistics)


def canonical_artifact_bytes(artifact: TrainingMeanArtifact) -> bytes:
    """Serialize a validated artifact as locked canonical JSON bytes."""
    _validate_artifact(artifact)
    mapping = asdict(artifact)
    return json.dumps(
        mapping,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def artifact_sha256(artifact: TrainingMeanArtifact) -> str:
    """Hash the complete canonical artifact bytes."""
    return hashlib.sha256(canonical_artifact_bytes(artifact)).hexdigest()


def load_training_mean_artifact(path: str | Path) -> TrainingMeanArtifact:
    """Load a byte-for-byte canonical Training mean artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Training mean artifact not found: {artifact_path}")
    data = artifact_path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise TrainingMeanError("Training mean artifact must not contain a BOM")
    if b"\r" in data:
        raise TrainingMeanError("Training mean artifact must use LF line endings")
    if not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise TrainingMeanError(
            "Training mean artifact must contain exactly one trailing LF"
        )
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingMeanError("Training mean artifact must be UTF-8") from exc
    try:
        mapping = json.loads(decoded)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TrainingMeanError("Training mean artifact must contain valid JSON") from exc
    artifact = _artifact_from_mapping(mapping)
    try:
        canonical = canonical_artifact_bytes(artifact)
    except (TypeError, ValueError) as exc:
        raise TrainingMeanError("Training mean artifact values are invalid") from exc
    if canonical != data:
        raise TrainingMeanError("Training mean artifact bytes are not canonical")
    return artifact


def validate_artifact_dataset(
    artifact: TrainingMeanArtifact,
    csv_path: str | Path,
) -> str:
    """Require a runtime CSV to match the artifact's exact file identity."""
    _validate_artifact(artifact)
    runtime_sha256 = sha256_file(csv_path)
    if runtime_sha256 != artifact.dataset_sha256:
        raise TrainingMeanError(
            "runtime CSV dataset_sha256 does not match the Training mean artifact"
        )
    return runtime_sha256


def write_training_mean_artifact(
    path: str | Path,
    artifact: TrainingMeanArtifact,
) -> ArtifactWriteResult:
    """Atomically replace a target with complete canonical artifact bytes."""
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Artifact parent directory not found: {parent}")
    data = canonical_artifact_bytes(artifact)
    digest = hashlib.sha256(data).hexdigest()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return ArtifactWriteResult(path=target, sha256=digest)


def _artifact_from_mapping(mapping: object) -> TrainingMeanArtifact:
    if type(mapping) is not dict:
        raise TrainingMeanError("Training mean artifact JSON must be an object")
    fields = set(mapping)
    if fields != _ARTIFACT_FIELDS:
        raise TrainingMeanError("Training mean artifact fields do not match the schema")

    _require_exact_integer(mapping["schema_version"], "schema_version", positive=True)
    _require_exact_integer(mapping["sample_count"], "sample_count", positive=True)
    _require_exact_integer(mapping["pixel_count"], "pixel_count", positive=True)
    _require_exact_integer(mapping["raw_pixel_sum"], "raw_pixel_sum", nonnegative=True)
    raw_mean = mapping["raw_training_mean"]
    if isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
        raise TrainingMeanError("raw_training_mean must be a number")

    artifact = TrainingMeanArtifact(
        schema_version=mapping["schema_version"],
        dataset_name=mapping["dataset_name"],
        split=mapping["split"],
        dataset_sha256=mapping["dataset_sha256"],
        sample_count=mapping["sample_count"],
        pixel_count=mapping["pixel_count"],
        raw_pixel_sum=mapping["raw_pixel_sum"],
        raw_training_mean=float(raw_mean),
        source_pixel_range=tuple(mapping["source_pixel_range"]),
        normalized_pixel_range=tuple(mapping["normalized_pixel_range"]),
        accumulator_dtype=mapping["accumulator_dtype"],
        mean_algorithm_version=mapping["mean_algorithm_version"],
    )
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: TrainingMeanArtifact) -> None:
    if not isinstance(artifact, TrainingMeanArtifact):
        raise TypeError("artifact must be a TrainingMeanArtifact")
    if type(artifact.schema_version) is not int or artifact.schema_version != 1:
        raise TrainingMeanError("schema_version must be integer 1")
    if artifact.dataset_name != _DATASET_NAME:
        raise TrainingMeanError("dataset_name must be 'fer2013'")
    if artifact.split != _SPLIT:
        raise TrainingMeanError("split must be 'train'")
    if type(artifact.dataset_sha256) is not str or not _SHA256_PATTERN.fullmatch(
        artifact.dataset_sha256
    ):
        raise TrainingMeanError("dataset_sha256 must be 64 lowercase hex characters")
    _require_exact_integer(artifact.sample_count, "sample_count", positive=True)
    _require_exact_integer(artifact.pixel_count, "pixel_count", positive=True)
    _require_exact_integer(artifact.raw_pixel_sum, "raw_pixel_sum", nonnegative=True)
    if artifact.pixel_count != artifact.sample_count * PIXELS_PER_IMAGE:
        raise TrainingMeanError("pixel_count must equal sample_count times 2304")
    if artifact.raw_pixel_sum > 255 * artifact.pixel_count:
        raise TrainingMeanError("raw_pixel_sum exceeds the maximum possible pixel sum")
    if isinstance(artifact.raw_training_mean, bool) or not isinstance(
        artifact.raw_training_mean, (int, float)
    ):
        raise TrainingMeanError("raw_training_mean must be a number")
    if not math.isfinite(artifact.raw_training_mean) or not (
        0 <= artifact.raw_training_mean <= 1
    ):
        raise TrainingMeanError("raw_training_mean must be finite and between 0 and 1")
    expected_mean = artifact.raw_pixel_sum / (255 * artifact.pixel_count)
    if abs(artifact.raw_training_mean - expected_mean) > 1e-15:
        raise TrainingMeanError("raw_training_mean is inconsistent with sum and count")
    if (
        type(artifact.source_pixel_range) is not tuple
        or len(artifact.source_pixel_range) != 2
        or any(type(value) is not int for value in artifact.source_pixel_range)
        or artifact.source_pixel_range != _SOURCE_PIXEL_RANGE
    ):
        raise TrainingMeanError("source_pixel_range must be [0, 255]")
    if (
        type(artifact.normalized_pixel_range) is not tuple
        or len(artifact.normalized_pixel_range) != 2
        or any(type(value) is not float for value in artifact.normalized_pixel_range)
        or artifact.normalized_pixel_range != _NORMALIZED_PIXEL_RANGE
    ):
        raise TrainingMeanError("normalized_pixel_range must be [0.0, 1.0]")
    if artifact.accumulator_dtype != _ACCUMULATOR_DTYPE:
        raise TrainingMeanError("accumulator_dtype must be 'uint64'")
    if artifact.mean_algorithm_version != MEAN_ALGORITHM_VERSION:
        raise TrainingMeanError(
            "mean_algorithm_version must be 'training-mean-v1'"
        )


def _require_exact_integer(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if type(value) is not int:
        raise TrainingMeanError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise TrainingMeanError(f"{field_name} must be positive")
    if nonnegative and value < 0:
        raise TrainingMeanError(f"{field_name} must be non-negative")
