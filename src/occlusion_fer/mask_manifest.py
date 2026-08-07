"""Canonical PublicTest mask manifests for the locked Stage A protocol."""

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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


V2_MANIFEST_SCHEMA_VERSION = 2
V2_MANIFEST_ALGORITHM_VERSION = "occlusion-v2-224"
V2_MANIFEST_COLUMNS = (
    "schema_version", "algorithm_version", "dataset_name", "split",
    "publictest_dataset_sha256", "sample_id", "condition", "occlusion_type",
    "ratio_token", "image_height", "image_width", "top", "left", "height",
    "width", "masked_pixel_count", "total_pixel_count", "actual_ratio",
    "evaluation_mask_seed", "mean_artifact_sha256",
)
V2_OFFICIAL_PUBLICTEST_SAMPLE_COUNT = 3589
V2_OFFICIAL_ROW_COUNT = V2_OFFICIAL_PUBLICTEST_SAMPLE_COUNT * 9


@dataclass(frozen=True)
class ManifestV2Row:
    schema_version: int
    algorithm_version: str
    dataset_name: str
    split: str
    publictest_dataset_sha256: str
    sample_id: int
    condition: str
    occlusion_type: str
    ratio_token: str
    image_height: int
    image_width: int
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    actual_ratio: float
    evaluation_mask_seed: int
    mean_artifact_sha256: str


@dataclass(frozen=True)
class ManifestV2Envelope:
    schema_version: int
    algorithm_version: str
    dataset_name: str
    split: str
    publictest_dataset_sha256: str
    training_mean_artifact_sha256: str
    image_height: int
    image_width: int
    evaluation_mask_seed: int
    condition_order: tuple[str, ...]
    row_count: int
    manifest_sha256: str


@dataclass(frozen=True)
class ManifestV2WriteResult:
    path: Path
    sidecar_path: Path
    sha256: str
    status: str


def canonical_manifest_v2_bytes(rows: Sequence[ManifestV2Row]) -> bytes:
    from occlusion_fer.occlusion import V2_MASKED_CONDITIONS

    if isinstance(rows, (str, bytes)):
        raise ManifestError("v2 manifest rows must be a sequence")
    validated_rows = tuple(rows)
    if not validated_rows:
        raise ManifestError("v2 manifest must contain rows")
    for row in validated_rows:
        _validate_manifest_v2_row(row)
        if row.condition not in V2_MASKED_CONDITIONS:
            raise ManifestError("manifest v2 contains an unknown condition")
    ordered = sorted(
        validated_rows,
        key=lambda row: (
            row.sample_id,
            V2_MASKED_CONDITIONS.index(row.condition),
        ),
    )
    if any(row.schema_version != 2 or row.algorithm_version != V2_MANIFEST_ALGORITHM_VERSION for row in ordered):
        raise ManifestError("v2 manifest identity mismatch")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=V2_MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in ordered:
        writer.writerow({field: getattr(row, field) for field in V2_MANIFEST_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def canonical_manifest_v2_sidecar_bytes(envelope: ManifestV2Envelope) -> bytes:
    """Serialize sidecar metadata without adding a second digest identity."""
    _validate_manifest_v2_envelope(envelope)
    payload = asdict(envelope)
    payload["condition_order"] = list(envelope.condition_order)
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def manifest_v2_sha256(rows: Sequence[ManifestV2Row]) -> str:
    return hashlib.sha256(canonical_manifest_v2_bytes(rows)).hexdigest()


def build_manifest_v2_rows(
    sample_ids: Sequence[int],
    *,
    publictest_dataset_sha256: str,
    mean_artifact_sha256: str,
) -> tuple[ManifestV2Row, ...]:
    from occlusion_fer.occlusion import V2_MASKED_CONDITIONS, evaluation_geometry_v2, parse_condition_v2

    rows: list[ManifestV2Row] = []
    for sample_id in sample_ids:
        if type(sample_id) is not int or sample_id <= 0:
            raise ManifestError("sample IDs must be positive integers")
        for condition in V2_MASKED_CONDITIONS:
            spec = parse_condition_v2(condition)
            geometry = evaluation_geometry_v2(sample_id, condition)
            rows.append(ManifestV2Row(
                schema_version=2,
                algorithm_version=V2_MANIFEST_ALGORITHM_VERSION,
                dataset_name="fer2013",
                split="validation",
                publictest_dataset_sha256=publictest_dataset_sha256,
                sample_id=sample_id,
                condition=condition,
                occlusion_type=spec.occlusion_type,
                ratio_token=f"{spec.target_ratio:.2f}",
                image_height=224,
                image_width=224,
                top=geometry.top,
                left=geometry.left,
                height=geometry.height,
                width=geometry.width,
                masked_pixel_count=geometry.masked_pixel_count,
                total_pixel_count=geometry.total_pixel_count,
                actual_ratio=geometry.actual_ratio,
                evaluation_mask_seed=20260804,
                mean_artifact_sha256=mean_artifact_sha256,
            ))
    return tuple(rows)


def make_manifest_v2_envelope(
    rows: Sequence[ManifestV2Row],
    *,
    publictest_dataset_sha256: str,
    training_mean_artifact_sha256: str,
) -> ManifestV2Envelope:
    from occlusion_fer.occlusion import V2_MASKED_CONDITIONS

    envelope = ManifestV2Envelope(
        schema_version=2,
        algorithm_version=V2_MANIFEST_ALGORITHM_VERSION,
        dataset_name="fer2013",
        split="validation",
        publictest_dataset_sha256=publictest_dataset_sha256,
        training_mean_artifact_sha256=training_mean_artifact_sha256,
        image_height=224,
        image_width=224,
        evaluation_mask_seed=20260804,
        condition_order=("clean",) + tuple(V2_MASKED_CONDITIONS),
        row_count=len(rows),
        manifest_sha256=manifest_v2_sha256(rows),
    )
    _validate_manifest_v2_envelope(envelope)
    return envelope


def validate_manifest_v2(
    rows: Sequence[ManifestV2Row],
    envelope: ManifestV2Envelope,
    *,
    require_official: bool = False,
) -> None:
    _validate_manifest_v2_envelope(envelope)
    if envelope.manifest_sha256 != manifest_v2_sha256(rows):
        raise ManifestError("manifest v2 digest mismatch")
    if envelope.row_count != len(rows):
        raise ManifestError("manifest v2 row count mismatch")
    if type(require_official) is not bool:
        raise ManifestError("require_official must be a bool")
    if any(
        row.image_height != 224 or row.image_width != 224
        for row in rows
    ):
        raise ManifestError("manifest v2 dimensions must be 224x224")
    if any(
        row.publictest_dataset_sha256 != envelope.publictest_dataset_sha256
        or row.mean_artifact_sha256 != envelope.training_mean_artifact_sha256
        for row in rows
    ):
        raise ManifestError("manifest v2 row identity does not match sidecar")
    keys = {(row.sample_id, row.condition) for row in rows}
    if len(keys) != len(rows):
        raise ManifestError("manifest v2 contains duplicate sample/condition rows")
    from occlusion_fer.occlusion import V2_MASKED_CONDITIONS, evaluation_geometry_v2, parse_condition_v2

    expected_conditions = set(V2_MASKED_CONDITIONS)
    by_sample: dict[int, set[str]] = {}
    for row in rows:
        _validate_manifest_v2_row(row)
        if row.condition not in expected_conditions:
            raise ManifestError("manifest v2 contains an unknown condition")
        by_sample.setdefault(row.sample_id, set()).add(row.condition)
        spec = parse_condition_v2(row.condition)
        geometry = evaluation_geometry_v2(row.sample_id, row.condition)
        if row.occlusion_type != spec.occlusion_type:
            raise ManifestError("manifest v2 occlusion_type does not match condition")
        if row.ratio_token != f"{spec.target_ratio:.2f}":
            raise ManifestError("manifest v2 ratio_token does not match condition")
        expected = (
            geometry.top,
            geometry.left,
            geometry.height,
            geometry.width,
            geometry.masked_pixel_count,
            geometry.total_pixel_count,
            geometry.actual_ratio,
        )
        actual = (
            row.top,
            row.left,
            row.height,
            row.width,
            row.masked_pixel_count,
            row.total_pixel_count,
            row.actual_ratio,
        )
        if actual != expected:
            raise ManifestError("manifest v2 geometry does not match protocol")
    if any(conditions != expected_conditions for conditions in by_sample.values()):
        raise ManifestError("manifest v2 samples must contain all nine conditions")
    if require_official:
        if len(by_sample) != V2_OFFICIAL_PUBLICTEST_SAMPLE_COUNT:
            raise ManifestError(
                "official v2 PublicTest sample count must be "
                f"{V2_OFFICIAL_PUBLICTEST_SAMPLE_COUNT}; got {len(by_sample)}"
            )
        if len(rows) != V2_OFFICIAL_ROW_COUNT:
            raise ManifestError(
                f"official v2 manifest row count must be {V2_OFFICIAL_ROW_COUNT}; got {len(rows)}"
            )


def write_manifest_v2_create_or_verify(
    path: str | Path,
    sidecar_path: str | Path,
    rows: Sequence[ManifestV2Row],
    envelope: ManifestV2Envelope,
) -> ManifestV2WriteResult:
    """Publish a v2 CSV/sidecar pair without overwriting either identity."""
    target = Path(path)
    sidecar = Path(sidecar_path)
    if target.parent != sidecar.parent:
        raise ManifestError("v2 manifest and sidecar must share a parent directory")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Manifest parent directory not found: {target.parent}")
    validate_manifest_v2(rows, envelope)
    csv_data = canonical_manifest_v2_bytes(rows)
    sidecar_data = canonical_manifest_v2_sidecar_bytes(envelope)
    digest = hashlib.sha256(csv_data).hexdigest()
    if envelope.manifest_sha256 != digest:
        raise ManifestError("manifest v2 envelope digest does not match CSV bytes")
    for candidate in (target, sidecar):
        if candidate.is_symlink():
            raise ManifestError(f"v2 manifest target must not be a symlink: {candidate}")
        if candidate.exists() and not candidate.is_file():
            raise ManifestError(f"v2 manifest target must be a regular file: {candidate}")
    target_exists = target.exists()
    sidecar_exists = sidecar.exists()
    if target_exists != sidecar_exists:
        raise ManifestConflictError("v2 manifest and sidecar must exist as a pair")
    if target_exists:
        if target.read_bytes() != csv_data or sidecar.read_bytes() != sidecar_data:
            raise ManifestConflictError("v2 manifest protocol conflict: existing bytes differ")
        return ManifestV2WriteResult(target, sidecar, digest, "consistent")

    temporary_paths: list[Path] = []
    created: list[Path] = []
    try:
        for candidate, data in ((target, csv_data), (sidecar, sidecar_data)):
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{candidate.name}.",
                suffix=".tmp", delete=False,
            ) as temporary_file:
                temporary = Path(temporary_file.name)
                temporary_paths.append(temporary)
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        for temporary, candidate in zip(temporary_paths, (target, sidecar), strict=True):
            try:
                os.link(temporary, candidate)
                created.append(candidate)
            except FileExistsError:
                raise ManifestConflictError(
                    f"v2 manifest target appeared during publish: {candidate}"
                )
        _fsync_directory(target.parent)
    except Exception:
        for candidate in created:
            candidate.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return ManifestV2WriteResult(target, sidecar, digest, "created")


def load_manifest_v2(
    path: str | Path,
    sidecar_path: str | Path,
    *,
    publictest_dataset_sha256: str | None = None,
    training_mean_artifact_sha256: str | None = None,
    require_official: bool = False,
) -> tuple[tuple[ManifestV2Row, ...], ManifestV2Envelope]:
    """Load and validate a v2 pair; never regenerate missing or bad geometry."""
    target = Path(path)
    sidecar = Path(sidecar_path)
    if not target.is_file() or not sidecar.is_file():
        raise ManifestError("v2 manifest requires an existing CSV and sidecar")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("v2 manifest sidecar must be canonical UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ManifestError("v2 manifest sidecar must contain a mapping")
    try:
        envelope_payload = dict(payload)
        envelope_payload["condition_order"] = tuple(
            envelope_payload["condition_order"]
        )
        envelope = ManifestV2Envelope(**envelope_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("v2 manifest sidecar fields are invalid") from exc
    if canonical_manifest_v2_sidecar_bytes(envelope) != sidecar.read_bytes():
        raise ManifestError("v2 manifest sidecar bytes are not canonical")
    if publictest_dataset_sha256 is not None and envelope.publictest_dataset_sha256 != publictest_dataset_sha256:
        raise ManifestError("manifest v2 PublicTest identity does not match expected source")
    if training_mean_artifact_sha256 is not None and envelope.training_mean_artifact_sha256 != training_mean_artifact_sha256:
        raise ManifestError("manifest v2 mean identity does not match expected artifact")
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != V2_MANIFEST_COLUMNS:
                raise ManifestError("manifest v2 CSV columns are not canonical")
            rows = tuple(_manifest_v2_row_from_mapping(row) for row in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ManifestError("manifest v2 CSV is not readable canonical UTF-8") from exc
    if canonical_manifest_v2_bytes(rows) != target.read_bytes():
        raise ManifestError("manifest v2 CSV bytes are not canonical")
    validate_manifest_v2(rows, envelope, require_official=require_official)
    return rows, envelope


def _validate_manifest_v2_envelope(envelope: ManifestV2Envelope) -> None:
    from occlusion_fer.occlusion import V2_MASKED_CONDITIONS

    if not isinstance(envelope, ManifestV2Envelope):
        raise ManifestError("manifest v2 envelope type is invalid")
    if (
        envelope.schema_version != 2
        or envelope.algorithm_version != V2_MANIFEST_ALGORITHM_VERSION
        or envelope.dataset_name != "fer2013"
        or envelope.split != "validation"
        or envelope.image_height != 224
        or envelope.image_width != 224
        or envelope.evaluation_mask_seed != 20260804
    ):
        raise ManifestError("manifest v2 envelope identity is invalid")
    if envelope.condition_order != ("clean",) + tuple(V2_MASKED_CONDITIONS):
        raise ManifestError("manifest v2 condition order is not canonical")
    for field_name in (
        "publictest_dataset_sha256", "training_mean_artifact_sha256", "manifest_sha256"
    ):
        value = getattr(envelope, field_name)
        if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
            raise ManifestError(f"manifest v2 {field_name} must be lowercase SHA-256")
    if type(envelope.row_count) is not int or envelope.row_count <= 0:
        raise ManifestError("manifest v2 row_count must be positive")


def _validate_manifest_v2_row(row: ManifestV2Row) -> None:
    if not isinstance(row, ManifestV2Row):
        raise ManifestError("manifest v2 rows must contain ManifestV2Row values")
    if (
        row.schema_version != 2
        or row.algorithm_version != V2_MANIFEST_ALGORITHM_VERSION
        or row.dataset_name != "fer2013"
        or row.split != "validation"
        or row.image_height != 224
        or row.image_width != 224
        or row.total_pixel_count != 224 * 224
        or row.evaluation_mask_seed != 20260804
    ):
        raise ManifestError("manifest v2 row identity is invalid")
    if type(row.sample_id) is not int or row.sample_id <= 0:
        raise ManifestError("manifest v2 sample_id must be positive")
    for field_name in (
        "publictest_dataset_sha256", "mean_artifact_sha256"
    ):
        value = getattr(row, field_name)
        if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
            raise ManifestError(f"manifest v2 {field_name} must be lowercase SHA-256")
    for field_name in (
        "schema_version", "image_height", "image_width", "top", "left",
        "height", "width", "masked_pixel_count", "total_pixel_count",
        "evaluation_mask_seed",
    ):
        if type(getattr(row, field_name)) is not int:
            raise ManifestError(f"manifest v2 {field_name} must be an integer")
    if not isinstance(row.actual_ratio, float) or not math.isfinite(row.actual_ratio):
        raise ManifestError("manifest v2 actual_ratio must be a finite float")


def _manifest_v2_row_from_mapping(mapping: Mapping[str, str]) -> ManifestV2Row:
    try:
        return ManifestV2Row(
            schema_version=int(mapping["schema_version"]),
            algorithm_version=mapping["algorithm_version"],
            dataset_name=mapping["dataset_name"],
            split=mapping["split"],
            publictest_dataset_sha256=mapping["publictest_dataset_sha256"],
            sample_id=int(mapping["sample_id"]),
            condition=mapping["condition"],
            occlusion_type=mapping["occlusion_type"],
            ratio_token=mapping["ratio_token"],
            image_height=int(mapping["image_height"]),
            image_width=int(mapping["image_width"]),
            top=int(mapping["top"]),
            left=int(mapping["left"]),
            height=int(mapping["height"]),
            width=int(mapping["width"]),
            masked_pixel_count=int(mapping["masked_pixel_count"]),
            total_pixel_count=int(mapping["total_pixel_count"]),
            actual_ratio=float(mapping["actual_ratio"]),
            evaluation_mask_seed=int(mapping["evaluation_mask_seed"]),
            mean_artifact_sha256=mapping["mean_artifact_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("manifest v2 row fields are invalid") from exc
