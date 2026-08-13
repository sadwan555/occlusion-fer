"""PrivateTest-only deterministic masks for the frozen 224 v2 protocol."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import torch
from torch import Tensor

from occlusion_fer.mask_hash import canonical_payload_bytes, coordinate_from_u64
from occlusion_fer.occlusion import normalized_fill_vector, round_half_up


MANIFEST_SCHEMA_VERSION = 2
SIDECAR_SCHEMA_VERSION = 2
ALGORITHM_VERSION = "occlusion-v2-224"
DATASET_NAME = "fer2013"
OFFICIAL_USAGE = "PrivateTest"
INTERNAL_SPLIT = "test"
IMAGE_SIZE = 224
EVALUATION_MASK_SEED = 20260804
COORDINATE_CONVENTION = "half-open:[top,top+height)x[left,left+width)"
ROUNDING_RULE = "round_half_up=floor(x+0.5)"
OFFICIAL_PRIVATE_SAMPLE_COUNT = 3589
OFFICIAL_PRIVATE_SAMPLE_ID_MIN = 32300
OFFICIAL_PRIVATE_SAMPLE_ID_MAX = 35888
PRIVATE_DATASET_SHA256 = (
    "4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634"
)
TRAINING_MEAN_RAW = 0.5077425080522144
TRAINING_MEAN_ARTIFACT_SHA256 = (
    "02ff7194f653c0aa6c53a042beca4b44eac58c12cb2737a6e25e0ccbe79fc1d9"
)
NORMALIZED_FILL = (
    0.09931226223674432,
    0.23099333951881437,
    0.45218892467650845,
)
MASKED_CONDITIONS = (
    "upper_face_0.20",
    "upper_face_0.30",
    "upper_face_0.40",
    "lower_face_0.20",
    "lower_face_0.30",
    "lower_face_0.40",
    "random_rectangle_0.20",
    "random_rectangle_0.30",
    "random_rectangle_0.40",
)
CONDITION_ORDER = ("clean", *MASKED_CONDITIONS)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONDITION_VALUES = {
    f"{occlusion_type}_{ratio_token}": (occlusion_type, ratio_token, ratio)
    for occlusion_type in ("upper_face", "lower_face", "random_rectangle")
    for ratio_token, ratio in (("0.20", 0.20), ("0.30", 0.30), ("0.40", 0.40))
}
MANIFEST_COLUMNS = (
    "manifest_schema_version",
    "algorithm_version",
    "dataset_name",
    "official_usage",
    "internal_split",
    "private_dataset_sha256",
    "sample_id",
    "condition",
    "occlusion_type",
    "ratio_token",
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
    "evaluation_mask_seed",
    "coordinate_convention",
    "geometry_hash",
)


class PrivateManifestError(ValueError):
    """Raised when PrivateTest mask identity is incomplete or incompatible."""


@dataclass(frozen=True)
class PrivateGeometry:
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    actual_ratio: float
    geometry_hash: str


@dataclass(frozen=True)
class PrivateManifestRow:
    manifest_schema_version: int
    algorithm_version: str
    dataset_name: str
    official_usage: str
    internal_split: str
    private_dataset_sha256: str
    sample_id: int
    condition: str
    occlusion_type: str
    ratio_token: str
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
    evaluation_mask_seed: int
    coordinate_convention: str
    geometry_hash: str


@dataclass(frozen=True)
class PrivateManifestSidecar:
    sidecar_schema_version: int
    manifest_schema_version: int
    manifest_sha256: str
    row_count: int
    dataset_name: str
    official_usage: str
    internal_split: str
    private_dataset_sha256: str
    sample_count: int
    sample_id_min: int
    sample_id_max: int
    algorithm_version: str
    image_size: int
    evaluation_mask_seed: int
    condition_order: tuple[str, ...]
    masked_conditions: tuple[str, ...]
    coordinate_convention: str
    rounding_rule: str
    training_mean_raw: float
    training_mean_artifact_sha256: str
    normalized_fill: tuple[float, float, float]


def build_private_coordinate_payload(
    sample_id: int,
    condition: str,
    axis: str,
) -> list[object]:
    """Extend the Stage 8 v2 payload with explicit PrivateTest/test identity."""
    _require_positive_integer(sample_id, "sample_id")
    occlusion_type, ratio_token, _ = _parse_condition(condition)
    if axis not in {"top", "left"}:
        raise PrivateManifestError("axis must be top or left")
    condition_token = f"{occlusion_type}:{ratio_token}"
    return [
        DATASET_NAME,
        INTERNAL_SPLIT,
        OFFICIAL_USAGE,
        sample_id,
        EVALUATION_MASK_SEED,
        IMAGE_SIZE,
        IMAGE_SIZE,
        ALGORITHM_VERSION,
        axis,
        condition_token,
        occlusion_type,
        ratio_token,
        axis,
    ]


def private_geometry(sample_id: int, condition: str) -> PrivateGeometry:
    """Return deterministic v2 geometry in the explicit PrivateTest namespace."""
    _require_positive_integer(sample_id, "sample_id")
    occlusion_type, _, target_ratio = _parse_condition(condition)
    if occlusion_type == "random_rectangle":
        height = width = round_half_up(math.sqrt(target_ratio) * IMAGE_SIZE)
        available_positions = IMAGE_SIZE - height + 1
        top = _coordinate(sample_id, condition, "top", available_positions)
        left = _coordinate(sample_id, condition, "left", available_positions)
    else:
        height = round_half_up(target_ratio * IMAGE_SIZE)
        width = IMAGE_SIZE
        top = 0 if occlusion_type == "upper_face" else IMAGE_SIZE - height
        left = 0
    masked_pixel_count = height * width
    actual_ratio = masked_pixel_count / (IMAGE_SIZE * IMAGE_SIZE)
    geometry_payload = [
        *build_private_coordinate_payload(sample_id, condition, "top")[:-1],
        "geometry",
        top,
        left,
        height,
        width,
        masked_pixel_count,
    ]
    geometry_hash = hashlib.sha256(
        canonical_payload_bytes(geometry_payload)
    ).hexdigest()
    return PrivateGeometry(
        top,
        left,
        height,
        width,
        masked_pixel_count,
        IMAGE_SIZE * IMAGE_SIZE,
        actual_ratio,
        geometry_hash,
    )


def build_manifest_rows(
    sample_ids: Sequence[int],
    private_dataset_sha256: str,
) -> tuple[PrivateManifestRow, ...]:
    """Build nine mask rows per sorted synthetic or official sample ID."""
    ids = _validated_sample_ids(sample_ids)
    _validate_sha256(private_dataset_sha256, "private dataset SHA")
    rows = []
    for sample_id in ids:
        for condition in MASKED_CONDITIONS:
            geometry = private_geometry(sample_id, condition)
            occlusion_type, ratio_token, target_ratio = _parse_condition(condition)
            rows.append(
                PrivateManifestRow(
                    MANIFEST_SCHEMA_VERSION,
                    ALGORITHM_VERSION,
                    DATASET_NAME,
                    OFFICIAL_USAGE,
                    INTERNAL_SPLIT,
                    private_dataset_sha256,
                    sample_id,
                    condition,
                    occlusion_type,
                    ratio_token,
                    target_ratio,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    geometry.top,
                    geometry.left,
                    geometry.height,
                    geometry.width,
                    geometry.masked_pixel_count,
                    geometry.total_pixel_count,
                    geometry.actual_ratio,
                    EVALUATION_MASK_SEED,
                    COORDINATE_CONVENTION,
                    geometry.geometry_hash,
                )
            )
    return tuple(rows)


def canonical_manifest_bytes(rows: Sequence[PrivateManifestRow]) -> bytes:
    """Serialize validated rows using stable Stage 8-style CSV bytes."""
    validated = _validate_rows(rows)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for row in validated:
        writer.writerow(
            (
                row.manifest_schema_version,
                row.algorithm_version,
                row.dataset_name,
                row.official_usage,
                row.internal_split,
                row.private_dataset_sha256,
                row.sample_id,
                row.condition,
                row.occlusion_type,
                row.ratio_token,
                f"{row.target_ratio:.2f}",
                row.image_height,
                row.image_width,
                row.top,
                row.left,
                row.height,
                row.width,
                row.masked_pixel_count,
                row.total_pixel_count,
                f"{row.actual_ratio:.17g}",
                row.evaluation_mask_seed,
                row.coordinate_convention,
                row.geometry_hash,
            )
        )
    return buffer.getvalue().encode("utf-8")


def manifest_sha256(rows: Sequence[PrivateManifestRow]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(rows)).hexdigest()


def build_manifest_sidecar(
    rows: Sequence[PrivateManifestRow],
    private_dataset_sha256: str,
) -> PrivateManifestSidecar:
    """Bind complete mask bytes to PrivateTest identity and frozen protocol."""
    validated = _validate_rows(rows)
    _validate_sha256(private_dataset_sha256, "private dataset SHA")
    sample_ids = tuple(
        dict.fromkeys(row.sample_id for row in validated)
    )
    return PrivateManifestSidecar(
        SIDECAR_SCHEMA_VERSION,
        MANIFEST_SCHEMA_VERSION,
        manifest_sha256(validated),
        len(validated),
        DATASET_NAME,
        OFFICIAL_USAGE,
        INTERNAL_SPLIT,
        private_dataset_sha256,
        len(sample_ids),
        min(sample_ids),
        max(sample_ids),
        ALGORITHM_VERSION,
        IMAGE_SIZE,
        EVALUATION_MASK_SEED,
        CONDITION_ORDER,
        MASKED_CONDITIONS,
        COORDINATE_CONVENTION,
        ROUNDING_RULE,
        TRAINING_MEAN_RAW,
        TRAINING_MEAN_ARTIFACT_SHA256,
        NORMALIZED_FILL,
    )


def canonical_sidecar_bytes(sidecar: PrivateManifestSidecar) -> bytes:
    if not isinstance(sidecar, PrivateManifestSidecar):
        raise PrivateManifestError(
            "manifest sidecar must be a PrivateManifestSidecar"
        )
    payload = asdict(sidecar)
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def write_manifest_artifacts(
    manifest_path: str | Path,
    sidecar_path: str | Path,
    rows: Sequence[PrivateManifestRow],
    sidecar: PrivateManifestSidecar,
) -> tuple[str, str]:
    """Create canonical manifest files while refusing every overwrite."""
    manifest_target = Path(manifest_path).expanduser()
    sidecar_target = Path(sidecar_path).expanduser()
    if manifest_target == sidecar_target:
        raise PrivateManifestError(
            "manifest and sidecar paths must be different"
        )
    for target in (manifest_target, sidecar_target):
        if target.exists():
            raise FileExistsError(
                f"PrivateTest manifest artifact already exists: {target}"
            )
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"manifest parent directory does not exist: {target.parent}"
            )
    validate_manifest_binding(
        rows,
        sidecar,
        expected_dataset_sha256=sidecar.private_dataset_sha256,
        expected_sample_ids=tuple(
            dict.fromkeys(row.sample_id for row in rows)
        ),
        require_official=False,
    )
    manifest_data = canonical_manifest_bytes(rows)
    sidecar_data = canonical_sidecar_bytes(sidecar)
    with manifest_target.open("xb") as manifest_file:
        manifest_file.write(manifest_data)
    try:
        with sidecar_target.open("xb") as sidecar_file:
            sidecar_file.write(sidecar_data)
    except Exception:
        manifest_target.unlink(missing_ok=True)
        raise
    return (
        hashlib.sha256(manifest_data).hexdigest(),
        hashlib.sha256(sidecar_data).hexdigest(),
    )


def load_manifest_artifacts(
    manifest_path: str | Path,
    sidecar_path: str | Path,
) -> tuple[tuple[PrivateManifestRow, ...], PrivateManifestSidecar]:
    """Load only canonical bytes and verify the sidecar's manifest SHA."""
    manifest_source = Path(manifest_path).expanduser()
    sidecar_source = Path(sidecar_path).expanduser()
    if not manifest_source.is_file():
        raise FileNotFoundError(
            f"PrivateTest manifest not found: {manifest_source}"
        )
    if not sidecar_source.is_file():
        raise FileNotFoundError(
            f"PrivateTest manifest sidecar not found: {sidecar_source}"
        )
    try:
        manifest_data = manifest_source.read_bytes()
        text = manifest_data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise PrivateManifestError(
                "PrivateTest manifest columns are incompatible"
            )
        rows = tuple(_row_from_mapping(row) for row in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise PrivateManifestError(
            "PrivateTest manifest must be canonical UTF-8 CSV"
        ) from exc
    if canonical_manifest_bytes(rows) != manifest_data:
        raise PrivateManifestError(
            "PrivateTest manifest bytes are not canonical"
        )
    try:
        sidecar_data = sidecar_source.read_bytes()
        sidecar_mapping = json.loads(sidecar_data)
        sidecar = _sidecar_from_mapping(sidecar_mapping)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivateManifestError(
            "PrivateTest manifest sidecar must be canonical UTF-8 JSON"
        ) from exc
    if canonical_sidecar_bytes(sidecar) != sidecar_data:
        raise PrivateManifestError(
            "PrivateTest manifest sidecar bytes are not canonical"
        )
    if sidecar.manifest_sha256 != hashlib.sha256(manifest_data).hexdigest():
        raise PrivateManifestError("manifest SHA mismatch")
    return rows, sidecar


def validate_manifest_binding(
    rows: Sequence[PrivateManifestRow],
    sidecar: PrivateManifestSidecar,
    *,
    expected_dataset_sha256: str,
    expected_sample_ids: Sequence[int],
    require_official: bool,
) -> None:
    """Fail on any split, dataset, sample, manifest, mean, or protocol mismatch."""
    validated = _validate_rows(rows)
    if not isinstance(sidecar, PrivateManifestSidecar):
        raise PrivateManifestError(
            "manifest sidecar must be a PrivateManifestSidecar"
        )
    expected_ids = _validated_sample_ids(expected_sample_ids)
    actual_ids = tuple(dict.fromkeys(row.sample_id for row in validated))
    if sidecar.official_usage != OFFICIAL_USAGE:
        raise PrivateManifestError("manifest official Usage must be PrivateTest")
    if sidecar.internal_split != INTERNAL_SPLIT:
        raise PrivateManifestError("manifest split must be test")
    if sidecar.private_dataset_sha256 != expected_dataset_sha256:
        raise PrivateManifestError("manifest dataset identity mismatch")
    if sidecar.private_dataset_sha256 != validated[0].private_dataset_sha256:
        raise PrivateManifestError("manifest row dataset identity mismatch")
    if actual_ids != expected_ids:
        raise PrivateManifestError("manifest sample IDs do not match the source")
    if sidecar.manifest_sha256 != manifest_sha256(validated):
        raise PrivateManifestError("manifest SHA mismatch")
    if sidecar.row_count != len(expected_ids) * len(MASKED_CONDITIONS):
        raise PrivateManifestError("manifest row count is incompatible")
    if sidecar.sample_count != len(expected_ids):
        raise PrivateManifestError("manifest sample count is incompatible")
    if sidecar.sample_id_min != min(expected_ids) or sidecar.sample_id_max != max(
        expected_ids
    ):
        raise PrivateManifestError("manifest sample ID bounds are incompatible")
    if (
        sidecar.algorithm_version != ALGORITHM_VERSION
        or sidecar.image_size != IMAGE_SIZE
        or sidecar.evaluation_mask_seed != EVALUATION_MASK_SEED
        or sidecar.masked_conditions != MASKED_CONDITIONS
        or sidecar.condition_order != CONDITION_ORDER
    ):
        raise PrivateManifestError(
            "manifest scientific protocol is incompatible"
        )
    if sidecar.training_mean_artifact_sha256 != TRAINING_MEAN_ARTIFACT_SHA256:
        raise PrivateManifestError("manifest Training mean SHA mismatch")
    if sidecar.training_mean_raw != TRAINING_MEAN_RAW:
        raise PrivateManifestError("manifest Training mean value mismatch")
    if sidecar.normalized_fill != NORMALIZED_FILL:
        raise PrivateManifestError("manifest normalized fill mismatch")
    if require_official:
        if expected_dataset_sha256 != PRIVATE_DATASET_SHA256:
            raise PrivateManifestError(
                "official PrivateTest canonical dataset SHA mismatch"
            )
        if len(expected_ids) != OFFICIAL_PRIVATE_SAMPLE_COUNT:
            raise PrivateManifestError(
                "official PrivateTest sample count must be 3589"
            )
        if expected_ids != tuple(
            range(
                OFFICIAL_PRIVATE_SAMPLE_ID_MIN,
                OFFICIAL_PRIVATE_SAMPLE_ID_MAX + 1,
            )
        ):
            raise PrivateManifestError(
                "official PrivateTest sample IDs must be 32300 through 35888"
            )


def apply_manifest_batch(
    clean_images: Tensor,
    sample_ids: Tensor,
    rows: Sequence[PrivateManifestRow]
    | Mapping[tuple[str, int], PrivateManifestRow],
    condition: str,
    fill_vector: Sequence[Real] = NORMALIZED_FILL,
) -> Tensor:
    """Clone a 224 batch and apply exact precomputed manifest rows."""
    if not isinstance(clean_images, Tensor) or clean_images.ndim != 4:
        raise PrivateManifestError(
            "clean images must have shape [B, 3, 224, 224]"
        )
    if tuple(clean_images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise PrivateManifestError(
            "clean images must have shape [B, 3, 224, 224]"
        )
    if not clean_images.is_floating_point() or not torch.isfinite(
        clean_images
    ).all().item():
        raise PrivateManifestError(
            "clean images must contain finite floating-point values"
        )
    if not isinstance(sample_ids, Tensor) or sample_ids.ndim != 1:
        raise PrivateManifestError("sample IDs must be one-dimensional")
    ids = tuple(int(value) for value in sample_ids.detach().cpu().tolist())
    if len(ids) != clean_images.shape[0] or len(set(ids)) != len(ids):
        raise PrivateManifestError(
            "sample IDs must be unique and match the image batch"
        )
    _parse_condition(condition)
    values = tuple(float(value) for value in fill_vector)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise PrivateManifestError(
            "fill vector must contain three finite values"
        )
    lookup = (
        rows
        if isinstance(rows, Mapping)
        else build_manifest_lookup(rows)
    )
    by_id = {
        sample_id: lookup[(condition, sample_id)]
        for sample_id in ids
        if (condition, sample_id) in lookup
    }
    if set(by_id) != set(ids):
        raise PrivateManifestError(
            "manifest rows do not bind the requested sample IDs"
        )
    masked = clean_images.clone()
    fill = clean_images.new_tensor(values).view(3, 1, 1)
    for index, sample_id in enumerate(ids):
        row = by_id[sample_id]
        masked[
            index,
            :,
            row.top : row.top + row.height,
            row.left : row.left + row.width,
        ] = fill
    return masked


def build_manifest_lookup(
    rows: Sequence[PrivateManifestRow],
) -> Mapping[tuple[str, int], PrivateManifestRow]:
    """Validate once and expose an immutable condition/sample lookup."""
    validated = _validate_rows(rows)
    return MappingProxyType(
        {(row.condition, row.sample_id): row for row in validated}
    )


def _coordinate(
    sample_id: int,
    condition: str,
    axis: str,
    available_positions: int,
) -> int:
    digest = hashlib.sha256(
        canonical_payload_bytes(
            build_private_coordinate_payload(sample_id, condition, axis)
        )
    ).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return coordinate_from_u64(value, available_positions)


def _row_from_mapping(row: dict[str, str]) -> PrivateManifestRow:
    if set(row) != set(MANIFEST_COLUMNS):
        raise PrivateManifestError("PrivateTest manifest row fields are invalid")
    try:
        return PrivateManifestRow(
            manifest_schema_version=int(row["manifest_schema_version"]),
            algorithm_version=row["algorithm_version"],
            dataset_name=row["dataset_name"],
            official_usage=row["official_usage"],
            internal_split=row["internal_split"],
            private_dataset_sha256=row["private_dataset_sha256"],
            sample_id=int(row["sample_id"]),
            condition=row["condition"],
            occlusion_type=row["occlusion_type"],
            ratio_token=row["ratio_token"],
            target_ratio=float(row["target_ratio"]),
            image_height=int(row["image_height"]),
            image_width=int(row["image_width"]),
            top=int(row["top"]),
            left=int(row["left"]),
            height=int(row["height"]),
            width=int(row["width"]),
            masked_pixel_count=int(row["masked_pixel_count"]),
            total_pixel_count=int(row["total_pixel_count"]),
            actual_ratio=float(row["actual_ratio"]),
            evaluation_mask_seed=int(row["evaluation_mask_seed"]),
            coordinate_convention=row["coordinate_convention"],
            geometry_hash=row["geometry_hash"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PrivateManifestError(
            "PrivateTest manifest row contains malformed values"
        ) from exc


def _sidecar_from_mapping(value: object) -> PrivateManifestSidecar:
    if not isinstance(value, dict):
        raise PrivateManifestError(
            "PrivateTest manifest sidecar must be a mapping"
        )
    expected_fields = set(PrivateManifestSidecar.__dataclass_fields__)
    if set(value) != expected_fields:
        raise PrivateManifestError(
            "PrivateTest manifest sidecar fields are incompatible"
        )
    mapping = dict(value)
    mapping["condition_order"] = tuple(mapping["condition_order"])
    mapping["masked_conditions"] = tuple(mapping["masked_conditions"])
    mapping["normalized_fill"] = tuple(mapping["normalized_fill"])
    try:
        return PrivateManifestSidecar(**mapping)
    except TypeError as exc:
        raise PrivateManifestError(
            "PrivateTest manifest sidecar contains malformed values"
        ) from exc


def _parse_condition(condition: str) -> tuple[str, str, float]:
    if type(condition) is not str:
        raise PrivateManifestError("condition must be a string")
    try:
        return _CONDITION_VALUES[condition]
    except KeyError as exc:
        raise PrivateManifestError(
            f"unsupported masked condition: {condition!r}"
        ) from exc


def _validated_sample_ids(sample_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(sample_ids, (str, bytes)) or not isinstance(
        sample_ids, Sequence
    ):
        raise PrivateManifestError("sample IDs must be a sequence")
    ids = tuple(sample_ids)
    if (
        not ids
        or any(type(value) is not int or value <= 0 for value in ids)
        or tuple(sorted(ids)) != ids
        or len(set(ids)) != len(ids)
    ):
        raise PrivateManifestError(
            "sample IDs must be non-empty, unique, positive, and sorted"
        )
    return ids


def _validate_rows(
    rows: Sequence[PrivateManifestRow],
) -> tuple[PrivateManifestRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PrivateManifestError("manifest rows must be a sequence")
    values = tuple(rows)
    if not values or any(
        not isinstance(row, PrivateManifestRow) for row in values
    ):
        raise PrivateManifestError(
            "manifest rows must contain PrivateManifestRow values"
        )
    dataset_sha = values[0].private_dataset_sha256
    sample_ids = tuple(dict.fromkeys(row.sample_id for row in values))
    expected = tuple(
        (condition, sample_id)
        for sample_id in sample_ids
        for condition in MASKED_CONDITIONS
    )
    actual = tuple((row.condition, row.sample_id) for row in values)
    if actual != expected:
        raise PrivateManifestError(
            "manifest rows must use canonical condition/sample order"
        )
    counts = Counter(row.condition for row in values)
    if any(counts[condition] != len(sample_ids) for condition in MASKED_CONDITIONS):
        raise PrivateManifestError(
            "manifest must contain every sample in nine conditions"
        )
    for row in values:
        _validate_sha256(row.private_dataset_sha256, "private dataset SHA")
        if (
            row.private_dataset_sha256 != dataset_sha
            or row.manifest_schema_version != MANIFEST_SCHEMA_VERSION
            or row.algorithm_version != ALGORITHM_VERSION
            or row.dataset_name != DATASET_NAME
            or row.official_usage != OFFICIAL_USAGE
            or row.internal_split != INTERNAL_SPLIT
            or row.image_height != IMAGE_SIZE
            or row.image_width != IMAGE_SIZE
            or row.evaluation_mask_seed != EVALUATION_MASK_SEED
            or row.coordinate_convention != COORDINATE_CONVENTION
        ):
            raise PrivateManifestError(
                "manifest row protocol or split identity is incompatible"
            )
        geometry = private_geometry(row.sample_id, row.condition)
        expected_type, expected_ratio_token, expected_ratio = _parse_condition(
            row.condition
        )
        if (
            row.occlusion_type != expected_type
            or row.ratio_token != expected_ratio_token
            or row.target_ratio != expected_ratio
        ):
            raise PrivateManifestError(
                "manifest condition fields do not match the protocol"
            )
        if (
            row.top,
            row.left,
            row.height,
            row.width,
            row.masked_pixel_count,
            row.total_pixel_count,
            row.actual_ratio,
            row.geometry_hash,
        ) != (
            geometry.top,
            geometry.left,
            geometry.height,
            geometry.width,
            geometry.masked_pixel_count,
            geometry.total_pixel_count,
            geometry.actual_ratio,
            geometry.geometry_hash,
        ):
            raise PrivateManifestError(
                "manifest row geometry is incompatible with the frozen protocol"
            )
    return values


def _require_positive_integer(value: object, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise PrivateManifestError(
            f"{field_name} must be a positive integer"
        )


def _validate_sha256(value: object, field_name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise PrivateManifestError(
            f"{field_name} must be a lowercase SHA-256"
        )


if normalized_fill_vector(TRAINING_MEAN_RAW) != NORMALIZED_FILL:
    raise RuntimeError("frozen normalized Training mean vector is inconsistent")
