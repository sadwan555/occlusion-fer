"""Canonical PublicTest mask manifests for the locked Stage A protocol."""

import csv
import hashlib
import io
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from occlusion_fer.occlusion import (
    IMAGE_SIZE,
    MASKED_CONDITIONS,
    TOTAL_PIXEL_COUNT,
    evaluation_geometry,
    normalized_fill_vector,
    parse_condition,
)
from occlusion_fer.training_mean import (
    TrainingMeanArtifact,
    TrainingMeanError,
    canonical_artifact_bytes,
    load_training_mean_artifact,
    sha256_file,
    validate_artifact_dataset,
)


MANIFEST_COLUMNS = (
    "manifest_schema_version",
    "algorithm_version",
    "dataset_name",
    "split",
    "dataset_sha256",
    "sample_id",
    "condition",
    "occlusion_type",
    "target_ratio",
    "image_height",
    "image_width",
    "top",
    "left",
    "height",
    "width",
    "masked_pixel_count",
    "total_pixel_count",
    "actual_ratio",
    "raw_fill_value",
    "normalized_fill_red",
    "normalized_fill_green",
    "normalized_fill_blue",
    "evaluation_mask_seed",
    "coordinate_convention",
)

MANIFEST_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "occlusion-v1"
DATASET_NAME = "fer2013"
INTERNAL_SPLIT = "validation"
EVALUATION_MASK_SEED = 20260804
COORDINATE_CONVENTION = "half-open:[top,top+height)x[left,left+width)"
OFFICIAL_PUBLICTEST_SAMPLE_COUNT = 3589
OFFICIAL_ROW_COUNT = OFFICIAL_PUBLICTEST_SAMPLE_COUNT * len(MASKED_CONDITIONS)

_REQUIRED_COLUMNS = ("emotion", "pixels", "Usage")
_VALID_USAGES = frozenset(("Training", "PublicTest", "PrivateTest"))
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """Raised when manifest inputs or canonical rows violate the protocol."""


class ManifestConflictError(ManifestError):
    """Raised when a target contains different bytes and cannot be replaced."""


@dataclass(frozen=True)
class ManifestRow:
    manifest_schema_version: int
    algorithm_version: str
    dataset_name: str
    split: str
    dataset_sha256: str
    sample_id: int
    condition: str
    occlusion_type: str
    target_ratio: float
    image_height: int
    image_width: int
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    actual_ratio: float
    raw_fill_value: float
    normalized_fill_red: float
    normalized_fill_green: float
    normalized_fill_blue: float
    evaluation_mask_seed: int
    coordinate_convention: str


@dataclass(frozen=True)
class ManifestWriteResult:
    path: Path
    sha256: str
    status: str


def enumerate_publictest_sample_ids(path: str | Path) -> tuple[int, ...]:
    """Return only PublicTest physical CSV line numbers without parsing samples."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"FER2013 CSV file not found: {csv_path}")

    publictest_ids: list[int] = []
    data_row_count = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ManifestError(f"FER2013 CSV file is empty: {csv_path}") from exc
            column_indices = _required_column_indices(header)
            usage_index = column_indices["Usage"]

            for row in reader:
                data_row_count += 1
                row_number = reader.line_num
                if not row or usage_index >= len(row):
                    raise ManifestError(
                        f"CSV row {row_number}: missing required Usage value"
                    )
                usage = row[usage_index]
                if usage not in _VALID_USAGES:
                    raise ManifestError(
                        f"CSV row {row_number}: unknown Usage value {usage!r}"
                    )
                if usage == "PublicTest":
                    publictest_ids.append(row_number)
    except csv.Error as exc:
        line_number = reader.line_num or 1
        raise ManifestError(
            f"CSV row {line_number}: malformed CSV content: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ManifestError(f"FER2013 CSV file must be UTF-8 text: {csv_path}") from exc

    if data_row_count == 0:
        raise ManifestError(
            f"FER2013 CSV file is empty: it contains no data rows: {csv_path}"
        )
    if not publictest_ids:
        raise ManifestError("FER2013 PublicTest split is empty")
    return tuple(publictest_ids)


def generate_manifest_rows(
    csv_path: str | Path,
    mean_artifact_path: str | Path,
) -> tuple[ManifestRow, ...]:
    """Build synthetic-fixture rows after strict artifact and dataset binding."""
    artifact = _load_and_bind_mean_artifact(csv_path, mean_artifact_path)
    sample_ids = enumerate_publictest_sample_ids(csv_path)
    return _build_manifest_rows(sample_ids, artifact)


def generate_official_manifest_rows(
    csv_path: str | Path,
    mean_artifact_path: str | Path,
) -> tuple[ManifestRow, ...]:
    """Build rows while enforcing the official FER2013 PublicTest cardinality."""
    rows = generate_manifest_rows(csv_path, mean_artifact_path)
    _validate_official_rows(rows)
    return rows


def generate_and_write_manifest(
    csv_path: str | Path,
    mean_artifact_path: str | Path,
    target_path: str | Path,
) -> ManifestWriteResult:
    """Generate low-level rows and create or verify their canonical artifact."""
    rows = generate_manifest_rows(csv_path, mean_artifact_path)
    return write_manifest_create_or_verify(target_path, rows)


def generate_and_write_official_manifest(
    csv_path: str | Path,
    mean_artifact_path: str | Path,
    target_path: str | Path,
) -> ManifestWriteResult:
    """Generate official rows and create or verify their canonical artifact."""
    rows = generate_official_manifest_rows(csv_path, mean_artifact_path)
    return write_manifest_create_or_verify(target_path, rows)


def canonical_manifest_bytes(rows: tuple[ManifestRow, ...]) -> bytes:
    """Validate and serialize complete rows using the locked CSV byte format."""
    canonical_rows = _validate_manifest_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
        doublequote=True,
        escapechar=None,
    )
    writer.writerow(MANIFEST_COLUMNS)
    for row in canonical_rows:
        writer.writerow(_formatted_row(row))
    data = buffer.getvalue().encode("utf-8")
    if not data.endswith(b"\n") or data.endswith(b"\n\n") or b"\r" in data:
        raise ManifestError("canonical manifest must use exactly one final LF")
    return data


def manifest_sha256(rows: tuple[ManifestRow, ...]) -> str:
    """Hash complete canonical manifest bytes."""
    return hashlib.sha256(canonical_manifest_bytes(rows)).hexdigest()


def write_manifest_create_or_verify(
    path: str | Path,
    rows: tuple[ManifestRow, ...],
) -> ManifestWriteResult:
    """Publish with hard-link exclusion, or verify an identical existing file."""
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Manifest parent directory not found: {parent}")

    data = canonical_manifest_bytes(rows)
    digest = hashlib.sha256(data).hexdigest()
    if target.is_symlink():
        raise ManifestError(f"Manifest target must not be a symlink: {target}")
    if target.exists() and not target.is_file():
        raise ManifestError(f"Manifest target must be a regular file: {target}")

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

        if sha256_file(temporary_path) != digest:
            raise ManifestError("temporary manifest SHA-256 verification failed")
        if target.is_symlink():
            raise ManifestError(f"Manifest target must not be a symlink: {target}")
        if target.exists():
            return _verify_existing_target(target, data, digest)
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            return _verify_existing_target(target, data, digest)
        _fsync_directory(parent)
        return ManifestWriteResult(path=target, sha256=digest, status="created")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required_column_indices(header: list[str]) -> dict[str, int]:
    if len(header) != len(set(header)):
        raise ManifestError("FER2013 CSV header contains duplicate columns")
    missing = [column for column in _REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ManifestError(
            "Missing required CSV column(s): " + ", ".join(missing)
        )
    return {column: header.index(column) for column in _REQUIRED_COLUMNS}


def _load_and_bind_mean_artifact(
    csv_path: str | Path,
    mean_artifact_path: str | Path,
) -> TrainingMeanArtifact:
    try:
        artifact = load_training_mean_artifact(mean_artifact_path)
        validate_artifact_dataset(artifact, csv_path)
    except (TrainingMeanError, TypeError, ValueError) as exc:
        raise ManifestError(f"invalid Training mean artifact: {exc}") from exc
    return artifact


def _build_manifest_rows(
    sample_ids: tuple[int, ...],
    artifact: TrainingMeanArtifact,
) -> tuple[ManifestRow, ...]:
    try:
        canonical_artifact_bytes(artifact)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid Training mean artifact: {exc}") from exc
    _validate_sample_ids(sample_ids)
    fill_red, fill_green, fill_blue = normalized_fill_vector(
        artifact.raw_training_mean
    )
    rows: list[ManifestRow] = []
    for condition in MASKED_CONDITIONS:
        condition_spec = parse_condition(condition)
        for sample_id in sample_ids:
            geometry = evaluation_geometry(sample_id, condition)
            rows.append(
                ManifestRow(
                    manifest_schema_version=MANIFEST_SCHEMA_VERSION,
                    algorithm_version=ALGORITHM_VERSION,
                    dataset_name=DATASET_NAME,
                    split=INTERNAL_SPLIT,
                    dataset_sha256=artifact.dataset_sha256,
                    sample_id=sample_id,
                    condition=condition,
                    occlusion_type=condition_spec.occlusion_type,
                    target_ratio=condition_spec.target_ratio,
                    image_height=IMAGE_SIZE,
                    image_width=IMAGE_SIZE,
                    top=geometry.top,
                    left=geometry.left,
                    height=geometry.height,
                    width=geometry.width,
                    masked_pixel_count=geometry.masked_pixel_count,
                    total_pixel_count=geometry.total_pixel_count,
                    actual_ratio=geometry.actual_ratio,
                    raw_fill_value=artifact.raw_training_mean,
                    normalized_fill_red=fill_red,
                    normalized_fill_green=fill_green,
                    normalized_fill_blue=fill_blue,
                    evaluation_mask_seed=EVALUATION_MASK_SEED,
                    coordinate_convention=COORDINATE_CONVENTION,
                )
            )
    return tuple(rows)


def _validate_sample_ids(sample_ids: object) -> tuple[int, ...]:
    if type(sample_ids) is not tuple or not sample_ids:
        raise ManifestError("sample_ids must be a non-empty tuple")
    for sample_id in sample_ids:
        if type(sample_id) is not int or sample_id <= 0:
            raise ManifestError("sample_id values must be positive integers")
    if (
        tuple(sorted(sample_ids)) != sample_ids
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ManifestError("sample_id values must be unique and strictly ascending")
    return sample_ids


def _validate_manifest_rows(rows: object) -> tuple[ManifestRow, ...]:
    if type(rows) is not tuple or not rows:
        raise ManifestError("manifest rows must be a non-empty tuple")
    if any(type(row) is not ManifestRow for row in rows):
        raise ManifestError("manifest rows must contain exact ManifestRow objects")
    canonical_rows = rows
    for row in canonical_rows:
        _validate_manifest_row(row)
    identity = (
        canonical_rows[0].dataset_sha256,
        canonical_rows[0].raw_fill_value,
        canonical_rows[0].normalized_fill_red,
        canonical_rows[0].normalized_fill_green,
        canonical_rows[0].normalized_fill_blue,
    )
    if any(
        (
            row.dataset_sha256,
            row.raw_fill_value,
            row.normalized_fill_red,
            row.normalized_fill_green,
            row.normalized_fill_blue,
        )
        != identity
        for row in canonical_rows[1:]
    ):
        raise ManifestError(
            "all manifest rows must use the same dataset and fill identity"
        )

    condition_counts = Counter(row.condition for row in canonical_rows)
    if tuple(condition_counts) != MASKED_CONDITIONS:
        raise ManifestError("manifest condition order or set is invalid")
    expected_count = condition_counts[MASKED_CONDITIONS[0]]
    if expected_count <= 0 or any(
        condition_counts[condition] != expected_count
        for condition in MASKED_CONDITIONS
    ):
        raise ManifestError("each manifest condition must have the same row count")

    expected_ids: tuple[int, ...] | None = None
    offset = 0
    for condition in MASKED_CONDITIONS:
        group = canonical_rows[offset : offset + expected_count]
        if len(group) != expected_count or any(
            row.condition != condition for row in group
        ):
            raise ManifestError("manifest rows do not follow locked condition order")
        sample_ids = tuple(row.sample_id for row in group)
        _validate_sample_ids(sample_ids)
        if expected_ids is None:
            expected_ids = sample_ids
        elif sample_ids != expected_ids:
            raise ManifestError("all conditions must contain the same sample IDs")
        offset += expected_count
    if offset != len(canonical_rows):
        raise ManifestError("manifest contains unexpected rows")
    return canonical_rows


def _validate_manifest_row(row: ManifestRow) -> None:
    fixed_values = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "dataset_name": DATASET_NAME,
        "split": INTERNAL_SPLIT,
        "image_height": IMAGE_SIZE,
        "image_width": IMAGE_SIZE,
        "total_pixel_count": TOTAL_PIXEL_COUNT,
        "evaluation_mask_seed": EVALUATION_MASK_SEED,
        "coordinate_convention": COORDINATE_CONVENTION,
    }
    for field_name, expected in fixed_values.items():
        if getattr(row, field_name) != expected:
            raise ManifestError(f"manifest {field_name} violates the locked protocol")
    for field_name in (
        "manifest_schema_version",
        "sample_id",
        "image_height",
        "image_width",
        "top",
        "left",
        "height",
        "width",
        "masked_pixel_count",
        "total_pixel_count",
        "evaluation_mask_seed",
    ):
        if type(getattr(row, field_name)) is not int:
            raise ManifestError(f"manifest {field_name} must be an integer")
    if row.sample_id <= 0:
        raise ManifestError("manifest sample_id must be positive")
    if _SHA256_PATTERN.fullmatch(row.dataset_sha256) is None:
        raise ManifestError("manifest dataset_sha256 must be lowercase hexadecimal")

    try:
        condition_spec = parse_condition(row.condition)
        geometry = evaluation_geometry(row.sample_id, row.condition)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid manifest condition or geometry: {exc}") from exc
    if row.occlusion_type != condition_spec.occlusion_type:
        raise ManifestError("manifest occlusion_type does not match condition")
    if row.target_ratio != condition_spec.target_ratio:
        raise ManifestError("manifest target_ratio does not match condition")
    geometry_values = {
        "top": geometry.top,
        "left": geometry.left,
        "height": geometry.height,
        "width": geometry.width,
        "masked_pixel_count": geometry.masked_pixel_count,
        "total_pixel_count": geometry.total_pixel_count,
        "actual_ratio": geometry.actual_ratio,
    }
    for field_name, expected in geometry_values.items():
        if getattr(row, field_name) != expected:
            raise ManifestError(f"manifest {field_name} does not match Task 4 geometry")
    if not (
        0 <= row.top
        and 0 <= row.left
        and row.top + row.height <= IMAGE_SIZE
        and row.left + row.width <= IMAGE_SIZE
    ):
        raise ManifestError("manifest geometry is outside the image")
    if row.masked_pixel_count != row.height * row.width:
        raise ManifestError("manifest masked_pixel_count is inconsistent")

    fill_values = (
        row.raw_fill_value,
        row.normalized_fill_red,
        row.normalized_fill_green,
        row.normalized_fill_blue,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, float)
        for value in fill_values
    ):
        raise ManifestError("manifest fill values must be floats")
    if not all(math.isfinite(value) for value in fill_values):
        raise ManifestError("manifest fill values must be finite")
    try:
        expected_fills = normalized_fill_vector(row.raw_fill_value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid manifest raw_fill_value: {exc}") from exc
    if (
        row.normalized_fill_red,
        row.normalized_fill_green,
        row.normalized_fill_blue,
    ) != expected_fills:
        raise ManifestError("manifest normalized fill values are inconsistent")


def _formatted_row(row: ManifestRow) -> tuple[str, ...]:
    return (
        str(row.manifest_schema_version),
        row.algorithm_version,
        row.dataset_name,
        row.split,
        row.dataset_sha256,
        str(row.sample_id),
        row.condition,
        row.occlusion_type,
        f"{row.target_ratio:.2f}",
        str(row.image_height),
        str(row.image_width),
        str(row.top),
        str(row.left),
        str(row.height),
        str(row.width),
        str(row.masked_pixel_count),
        str(row.total_pixel_count),
        f"{row.actual_ratio:.10f}",
        f"{row.raw_fill_value:.17f}",
        f"{row.normalized_fill_red:.17f}",
        f"{row.normalized_fill_green:.17f}",
        f"{row.normalized_fill_blue:.17f}",
        str(row.evaluation_mask_seed),
        row.coordinate_convention,
    )


def _validate_official_rows(rows: tuple[ManifestRow, ...]) -> None:
    validated = _validate_manifest_rows(rows)
    sample_ids = tuple(
        row.sample_id
        for row in validated
        if row.condition == MASKED_CONDITIONS[0]
    )
    if len(sample_ids) != OFFICIAL_PUBLICTEST_SAMPLE_COUNT:
        raise ManifestError(
            "official PublicTest sample count must be "
            f"{OFFICIAL_PUBLICTEST_SAMPLE_COUNT}; got {len(sample_ids)}"
        )
    if len(validated) != OFFICIAL_ROW_COUNT:
        raise ManifestError(
            f"official manifest row count must be {OFFICIAL_ROW_COUNT}; "
            f"got {len(validated)}"
        )
    for condition in MASKED_CONDITIONS:
        count = sum(row.condition == condition for row in validated)
        if count != OFFICIAL_PUBLICTEST_SAMPLE_COUNT:
            raise ManifestError(
                f"official condition {condition!r} must contain "
                f"{OFFICIAL_PUBLICTEST_SAMPLE_COUNT} rows; got {count}"
            )


def _verify_existing_target(
    target: Path,
    expected_data: bytes,
    expected_sha256: str,
) -> ManifestWriteResult:
    if target.is_symlink():
        raise ManifestError(f"Manifest target must not be a symlink: {target}")
    if not target.is_file():
        raise ManifestError(f"Manifest target must be a regular file: {target}")
    actual_data = target.read_bytes()
    actual_sha256 = sha256_file(target)
    if actual_data != expected_data or actual_sha256 != expected_sha256:
        raise ManifestConflictError(
            f"manifest protocol conflict: existing target differs: {target}"
        )
    return ManifestWriteResult(
        path=target,
        sha256=expected_sha256,
        status="consistent",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
