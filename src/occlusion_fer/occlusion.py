"""Deterministic geometry for the locked synthetic-occlusion protocol."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from occlusion_fer.mask_hash import (
    build_evaluation_coordinate_payload,
    build_training_coordinate_payload,
    coordinate_from_u64,
    u64_for_payload,
    V2_ALGORITHM_VERSION,
    V2_EVALUATION_MASK_SEED,
    V2_IMAGE_HEIGHT,
    V2_IMAGE_WIDTH,
    V2_RATIO_TOKENS,
    V2_TYPE_TOKENS,
    build_v2_evaluation_coordinate_payload,
    build_v2_training_coordinate_payload,
    v2_coordinate_from_u64,
    u64_for_payload_v2,
)


IMAGE_SIZE = 112
TOTAL_PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE
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
_CONDITION_VALUES = {
    condition: (occlusion_type, ratio)
    for occlusion_type in ("upper_face", "lower_face", "random_rectangle")
    for ratio, condition in zip(
        (0.20, 0.30, 0.40),
        (
            f"{occlusion_type}_0.20",
            f"{occlusion_type}_0.30",
            f"{occlusion_type}_0.40",
        ),
        strict=True,
    )
}
_IMAGENET_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_STDS = (0.229, 0.224, 0.225)
_ALGORITHM_VERSION = "occlusion-v1"
_EVALUATION_MASK_SEED = 20260804
_COORDINATE_CONVENTION = "half-open:[top,top+height)x[left,left+width)"
_INTEGER_DTYPES = frozenset(
    {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
V2_IMAGE_SIZE = V2_IMAGE_HEIGHT
V2_TOTAL_PIXEL_COUNT = V2_IMAGE_HEIGHT * V2_IMAGE_WIDTH
V2_MASKED_CONDITIONS = MASKED_CONDITIONS
V2_TYPE_ORDER = V2_TYPE_TOKENS
V2_RATIO_ORDER = V2_RATIO_TOKENS


@dataclass(frozen=True)
class GeometryV2:
    """Geometry and realized ratio for the fixed 224x224 consumer space."""

    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    actual_ratio: float
    image_height: int = V2_IMAGE_HEIGHT
    image_width: int = V2_IMAGE_WIDTH
    algorithm_version: str = V2_ALGORITHM_VERSION


@dataclass(frozen=True)
class MaskMetadataV2:
    sample_id: int
    condition: str
    occlusion_type: str
    ratio_token: str
    target_ratio: float
    actual_ratio: float
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    image_height: int
    image_width: int
    algorithm_version: str
    context: str
    seed: int
    epoch: int | None
    coordinate_convention: str = _COORDINATE_CONVENTION


@dataclass(frozen=True)
class ConditionSpec:
    occlusion_type: str
    target_ratio: float


@dataclass(frozen=True)
class Geometry:
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    actual_ratio: float


@dataclass(frozen=True)
class MaskMetadata:
    sample_id: int
    condition: str
    occlusion_type: str
    target_ratio: float
    actual_ratio: float
    top: int
    left: int
    height: int
    width: int
    masked_pixel_count: int
    total_pixel_count: int
    algorithm_version: str
    context: str
    seed: int
    epoch: int | None
    coordinate_convention: str


def parse_condition(condition: str) -> ConditionSpec:
    """Parse one exact canonical masked condition."""
    if type(condition) is not str:
        raise TypeError("condition must be a string")
    try:
        occlusion_type, target_ratio = _CONDITION_VALUES[condition]
    except KeyError as exc:
        raise ValueError(f"unsupported masked condition: {condition!r}") from exc
    return ConditionSpec(occlusion_type, target_ratio)


def round_half_up(value: Real) -> int:
    """Round a finite non-negative number using floor(value + 0.5)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("value must be finite")
    if numeric_value < 0:
        raise ValueError("value must be non-negative")
    return math.floor(numeric_value + 0.5)


def normalized_fill_vector(raw_training_mean: Real) -> tuple[float, float, float]:
    """Convert a normalized raw Training mean to ImageNet-normalized fills."""
    if isinstance(raw_training_mean, bool) or not isinstance(raw_training_mean, Real):
        raise TypeError("raw_training_mean must be a real number")
    mean = float(raw_training_mean)
    if not math.isfinite(mean):
        raise ValueError("raw_training_mean must be finite")
    if not 0 <= mean <= 1:
        raise ValueError("raw_training_mean must be between 0 and 1")
    return tuple(
        (mean - channel_mean) / channel_std
        for channel_mean, channel_std in zip(
            _IMAGENET_MEANS,
            _IMAGENET_STDS,
            strict=True,
        )
    )


def normalized_fill_vector_v2(
    mean_artifact: object,
    *,
    training_dataset_sha256: str | None = None,
) -> tuple[float, float, float]:
    """Derive fill only from a matching Training mean v2 envelope."""
    from occlusion_fer.training_mean import (
        TrainingMeanV2Envelope,
        validate_training_mean_v2,
    )

    if not isinstance(mean_artifact, TrainingMeanV2Envelope):
        raise ValueError("v2 fill requires a TrainingMeanV2Envelope")
    expected_sha = (
        mean_artifact.training_dataset_sha256
        if training_dataset_sha256 is None
        else training_dataset_sha256
    )
    try:
        validate_training_mean_v2(
            mean_artifact,
            training_dataset_sha256=expected_sha,
            consumer_image_size=V2_IMAGE_SIZE,
        )
    except ValueError as exc:
        raise ValueError(f"v2 fill rejects incompatible mean provenance: {exc}") from exc
    return normalized_fill_vector(mean_artifact.raw_training_mean)


def evaluation_geometry(sample_id: int, condition: str) -> Geometry:
    """Generate locked validation geometry for one sample and condition."""
    _require_positive_integer(sample_id, "sample_id")
    condition_spec = parse_condition(condition)
    top, left = _evaluation_coordinates(sample_id, condition, condition_spec)
    return _build_geometry(condition_spec, top, left)


def training_geometry(
    sample_id: int,
    condition: str,
    training_seed: int,
    epoch: int,
) -> Geometry:
    """Generate locked Training geometry for an already selected condition."""
    _require_positive_integer(sample_id, "sample_id")
    _require_nonnegative_integer(training_seed, "training_seed")
    _require_positive_integer(epoch, "epoch")
    condition_spec = parse_condition(condition)
    top, left = _training_coordinates(
        sample_id,
        condition,
        condition_spec,
        training_seed,
        epoch,
    )
    return _build_geometry(condition_spec, top, left)


def apply_evaluation_batch(
    clean_images: Tensor,
    sample_ids: Tensor,
    condition: str,
    fill_vector: Sequence[Real],
) -> tuple[Tensor, tuple[MaskMetadata, ...]]:
    """Clone and mask a validation batch using fixed evaluation geometry."""
    ids, condition_spec, fill_values = _validate_batch_inputs(
        clean_images,
        sample_ids,
        condition,
        fill_vector,
        reject_duplicate_ids=True,
    )
    geometries = tuple(evaluation_geometry(sample_id, condition) for sample_id in ids)
    metadata = tuple(
        _build_metadata(
            sample_id,
            condition,
            condition_spec,
            geometry,
            context="validation",
            seed=_EVALUATION_MASK_SEED,
            epoch=None,
        )
        for sample_id, geometry in zip(ids, geometries, strict=True)
    )
    return _clone_and_fill(clean_images, geometries, fill_values), metadata


def apply_training_batch(
    clean_images: Tensor,
    sample_ids: Tensor,
    condition: str,
    fill_vector: Sequence[Real],
    *,
    training_seed: int,
    epoch: int,
) -> tuple[Tensor, tuple[MaskMetadata, ...]]:
    """Clone and mask a Training batch for an already selected condition."""
    _require_nonnegative_integer(training_seed, "training_seed")
    _require_positive_integer(epoch, "epoch")
    ids, condition_spec, fill_values = _validate_batch_inputs(
        clean_images,
        sample_ids,
        condition,
        fill_vector,
        reject_duplicate_ids=False,
    )
    geometries = tuple(
        training_geometry(sample_id, condition, training_seed, epoch)
        for sample_id in ids
    )
    metadata = tuple(
        _build_metadata(
            sample_id,
            condition,
            condition_spec,
            geometry,
            context="train",
            seed=training_seed,
            epoch=epoch,
        )
        for sample_id, geometry in zip(ids, geometries, strict=True)
    )
    return _clone_and_fill(clean_images, geometries, fill_values), metadata


def apply_evaluation_mask(
    clean_image: Tensor,
    sample_id: int,
    condition: str,
    fill_vector: Sequence[Real],
) -> tuple[Tensor, MaskMetadata]:
    """Apply the batch implementation to one validation image."""
    if not isinstance(clean_image, Tensor):
        raise TypeError("clean_image must be a torch.Tensor")
    if tuple(clean_image.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("clean_image must have shape [3, 112, 112]")
    masked_images, metadata = apply_evaluation_batch(
        clean_image.unsqueeze(0),
        torch.tensor([sample_id], dtype=torch.int64),
        condition,
        fill_vector,
    )
    return masked_images[0], metadata[0]


def _evaluation_coordinates(
    sample_id: int,
    condition: str,
    condition_spec: ConditionSpec,
) -> tuple[int, int]:
    if condition_spec.occlusion_type != "random_rectangle":
        return _band_coordinates(condition_spec)
    side = _random_side(condition_spec.target_ratio)
    available_positions = IMAGE_SIZE - side + 1
    return tuple(
        coordinate_from_u64(
            u64_for_payload(
                build_evaluation_coordinate_payload(sample_id, condition, axis)
            ),
            available_positions,
        )
        for axis in ("top", "left")
    )


def _training_coordinates(
    sample_id: int,
    condition: str,
    condition_spec: ConditionSpec,
    training_seed: int,
    epoch: int,
) -> tuple[int, int]:
    if condition_spec.occlusion_type != "random_rectangle":
        return _band_coordinates(condition_spec)
    side = _random_side(condition_spec.target_ratio)
    available_positions = IMAGE_SIZE - side + 1
    return tuple(
        coordinate_from_u64(
            u64_for_payload(
                build_training_coordinate_payload(
                    sample_id,
                    training_seed,
                    epoch,
                    condition,
                    axis,
                )
            ),
            available_positions,
        )
        for axis in ("top", "left")
    )


def _band_coordinates(condition_spec: ConditionSpec) -> tuple[int, int]:
    height = round_half_up(condition_spec.target_ratio * IMAGE_SIZE)
    if condition_spec.occlusion_type == "upper_face":
        return 0, 0
    return IMAGE_SIZE - height, 0


def _build_geometry(
    condition_spec: ConditionSpec,
    top: int,
    left: int,
) -> Geometry:
    if condition_spec.occlusion_type == "random_rectangle":
        height = width = _random_side(condition_spec.target_ratio)
    else:
        height = round_half_up(condition_spec.target_ratio * IMAGE_SIZE)
        width = IMAGE_SIZE
    masked_pixel_count = height * width
    return Geometry(
        top=top,
        left=left,
        height=height,
        width=width,
        masked_pixel_count=masked_pixel_count,
        total_pixel_count=TOTAL_PIXEL_COUNT,
        actual_ratio=masked_pixel_count / TOTAL_PIXEL_COUNT,
    )


def _random_side(target_ratio: float) -> int:
    return round_half_up(math.sqrt(target_ratio) * IMAGE_SIZE)


def _validate_batch_inputs(
    clean_images: object,
    sample_ids: object,
    condition: object,
    fill_vector: object,
    *,
    reject_duplicate_ids: bool,
) -> tuple[list[int], ConditionSpec, tuple[float, float, float]]:
    _validate_clean_images(clean_images)
    ids = _validated_sample_ids(
        sample_ids,
        batch_size=clean_images.shape[0],
        reject_duplicates=reject_duplicate_ids,
    )
    condition_spec = parse_condition(condition)
    fill_values = _validated_fill_vector(fill_vector, clean_images)
    return ids, condition_spec, fill_values


def _validate_clean_images(clean_images: object) -> None:
    if not isinstance(clean_images, Tensor):
        raise TypeError("clean_images must be a torch.Tensor")
    if clean_images.ndim != 4:
        raise ValueError("clean_images must have shape [B, 3, 112, 112]")
    if clean_images.shape[0] <= 0:
        raise ValueError("clean_images batch must be non-empty")
    if tuple(clean_images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("clean_images must have shape [B, 3, 112, 112]")
    if not clean_images.is_floating_point():
        raise TypeError("clean_images must use a floating dtype")
    if not torch.isfinite(clean_images).all().item():
        raise ValueError("clean_images must contain only finite values")


def _validated_sample_ids(
    sample_ids: object,
    *,
    batch_size: int,
    reject_duplicates: bool,
) -> list[int]:
    if not isinstance(sample_ids, Tensor):
        raise TypeError("sample_ids must be a torch.Tensor")
    if sample_ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    if sample_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("sample_ids must use an integer dtype")
    if sample_ids.shape[0] != batch_size:
        raise ValueError("sample_ids length must equal the image batch size")
    ids = [int(value) for value in sample_ids.detach().cpu().tolist()]
    if any(sample_id <= 0 for sample_id in ids):
        raise ValueError("sample_ids must contain only positive integers")
    if reject_duplicates and len(set(ids)) != len(ids):
        raise ValueError("evaluation sample_ids must be unique")
    return ids


def _validated_fill_vector(
    fill_vector: object,
    clean_images: Tensor,
) -> tuple[float, float, float]:
    if isinstance(fill_vector, (str, bytes)) or not isinstance(fill_vector, Sequence):
        raise TypeError("fill_vector must be a sequence of three real numbers")
    if len(fill_vector) != 3:
        raise ValueError("fill_vector must contain exactly three values")
    values: list[float] = []
    for value in fill_vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("fill_vector values must be real numbers")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("fill_vector values must be finite")
        values.append(numeric_value)
    converted = clean_images.new_tensor(values)
    if not torch.isfinite(converted).all().item():
        raise ValueError("fill_vector is not finite in the image dtype")
    return values[0], values[1], values[2]


def _build_metadata(
    sample_id: int,
    condition: str,
    condition_spec: ConditionSpec,
    geometry: Geometry,
    *,
    context: str,
    seed: int,
    epoch: int | None,
) -> MaskMetadata:
    return MaskMetadata(
        sample_id=sample_id,
        condition=condition,
        occlusion_type=condition_spec.occlusion_type,
        target_ratio=condition_spec.target_ratio,
        actual_ratio=geometry.actual_ratio,
        top=geometry.top,
        left=geometry.left,
        height=geometry.height,
        width=geometry.width,
        masked_pixel_count=geometry.masked_pixel_count,
        total_pixel_count=geometry.total_pixel_count,
        algorithm_version=_ALGORITHM_VERSION,
        context=context,
        seed=seed,
        epoch=epoch,
        coordinate_convention=_COORDINATE_CONVENTION,
    )


def _clone_and_fill(
    clean_images: Tensor,
    geometries: tuple[Geometry, ...],
    fill_values: tuple[float, float, float],
) -> Tensor:
    masked_images = clean_images.clone()
    fill = clean_images.new_tensor(fill_values).view(3, 1, 1)
    for sample_index, geometry in enumerate(geometries):
        masked_images[
            sample_index,
            :,
            geometry.top : geometry.top + geometry.height,
            geometry.left : geometry.left + geometry.width,
        ] = fill
    return masked_images


def _require_positive_integer(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _validate_v2_image_size(image_size: object) -> None:
    if type(image_size) is not int:
        raise TypeError("image_size must be an integer")
    if image_size != V2_IMAGE_SIZE:
        raise ValueError("occlusion-v2-224 requires image_size=224")


def parse_condition_v2(condition: str) -> ConditionSpec:
    """Parse a canonical v2 condition while preserving the v1 spelling."""
    return parse_condition(condition)


def select_training_condition_v2(
    sample_id: int,
    training_seed: int,
    epoch: int,
    *,
    clean_probability: float = 0.5,
) -> str | None:
    """Select clean or one fixed v2 condition independent of batch/worker order."""
    _require_positive_integer(sample_id, "sample_id")
    _require_nonnegative_integer(training_seed, "training_seed")
    _require_positive_integer(epoch, "epoch")
    if type(clean_probability) not in (int, float) or not 0 <= clean_probability <= 1:
        raise ValueError("clean_probability must be between 0 and 1")
    from occlusion_fer.mask_hash import (
        apply_from_u64,
        build_v2_training_decision_payload,
        ratio_index_from_u64,
        type_index_from_u64,
    )
    apply_payload = build_v2_training_decision_payload(
        sample_id, training_seed, epoch, "apply"
    )
    # The approved protocol keeps the historical 0.5 predicate. For callers
    # using the general helper outside the locked config, the threshold still
    # represents the probability of applying an occlusion.
    decision = u64_for_payload_v2(apply_payload)
    if clean_probability == 0.5:
        apply = apply_from_u64(decision)
    else:
        mask_probability = 1.0 - float(clean_probability)
        apply = decision < int(mask_probability * (2**64))
    if not apply:
        return None
    type_payload = build_v2_training_decision_payload(
        sample_id, training_seed, epoch, "type"
    )
    ratio_payload = build_v2_training_decision_payload(
        sample_id, training_seed, epoch, "ratio"
    )
    occlusion_type = V2_TYPE_ORDER[type_index_from_u64(u64_for_payload_v2(type_payload))]
    ratio_token = V2_RATIO_ORDER[ratio_index_from_u64(u64_for_payload_v2(ratio_payload))]
    return f"{occlusion_type}_{ratio_token}"


def _v2_ratio_token(condition_spec: ConditionSpec) -> str:
    token = f"{condition_spec.target_ratio:.2f}"
    if token not in V2_RATIO_TOKENS:
        raise ValueError("condition ratio is not a canonical v2 ratio token")
    return token


def _v2_geometry(condition_spec: ConditionSpec, top: int, left: int) -> GeometryV2:
    if condition_spec.occlusion_type == "random_rectangle":
        height = width = round_half_up(
            math.sqrt(condition_spec.target_ratio) * V2_IMAGE_SIZE
        )
    else:
        height = round_half_up(condition_spec.target_ratio * V2_IMAGE_SIZE)
        width = V2_IMAGE_WIDTH
    masked_pixel_count = height * width
    return GeometryV2(
        top=top,
        left=left,
        height=height,
        width=width,
        masked_pixel_count=masked_pixel_count,
        total_pixel_count=V2_TOTAL_PIXEL_COUNT,
        actual_ratio=masked_pixel_count / V2_TOTAL_PIXEL_COUNT,
    )


def _v2_coordinates(
    sample_id: int,
    condition: str,
    condition_spec: ConditionSpec,
    *,
    training_seed: int | None,
    epoch: int | None,
) -> tuple[int, int]:
    if condition_spec.occlusion_type != "random_rectangle":
        if condition_spec.occlusion_type == "upper_face":
            return 0, 0
        side = round_half_up(condition_spec.target_ratio * V2_IMAGE_SIZE)
        return V2_IMAGE_SIZE - side, 0
    side = round_half_up(math.sqrt(condition_spec.target_ratio) * V2_IMAGE_SIZE)
    available_positions = V2_IMAGE_SIZE - side + 1
    ratio_token = _v2_ratio_token(condition_spec)
    condition_token = f"{condition_spec.occlusion_type}:{ratio_token}"
    coordinates: list[int] = []
    for axis in ("top", "left"):
        if training_seed is None:
            payload = build_v2_evaluation_coordinate_payload(
                sample_id, condition_token, condition_spec.occlusion_type,
                ratio_token, axis,
            )
        else:
            if epoch is None:
                raise ValueError("training epoch is required for v2 geometry")
            payload = build_v2_training_coordinate_payload(
                sample_id, training_seed, epoch, condition_token,
                condition_spec.occlusion_type, ratio_token, axis,
            )
        coordinates.append(v2_coordinate_from_u64(
            u64_for_payload_v2(payload), available_positions
        ))
    return coordinates[0], coordinates[1]


def evaluation_geometry_v2(sample_id: int, condition: str) -> GeometryV2:
    _require_positive_integer(sample_id, "sample_id")
    spec = parse_condition_v2(condition)
    top, left = _v2_coordinates(
        sample_id, condition, spec, training_seed=None, epoch=None
    )
    return _v2_geometry(spec, top, left)


def training_geometry_v2(
    sample_id: int, condition: str, training_seed: int, epoch: int
) -> GeometryV2:
    _require_positive_integer(sample_id, "sample_id")
    _require_nonnegative_integer(training_seed, "training_seed")
    _require_positive_integer(epoch, "epoch")
    spec = parse_condition_v2(condition)
    top, left = _v2_coordinates(
        sample_id, condition, spec, training_seed=training_seed, epoch=epoch
    )
    return _v2_geometry(spec, top, left)


def _validate_v2_batch_inputs(
    clean_images: object,
    sample_ids: object,
    condition: object,
    fill_vector: object,
    *,
    reject_duplicate_ids: bool,
) -> tuple[list[int], ConditionSpec, tuple[float, float, float]]:
    _validate_v2_clean_images(clean_images)
    ids = _validated_sample_ids(
        sample_ids,
        batch_size=clean_images.shape[0],
        reject_duplicates=reject_duplicate_ids,
    )
    spec = parse_condition_v2(condition)
    values = _validated_fill_vector(fill_vector, clean_images)
    return ids, spec, values


def _validate_v2_clean_images(clean_images: object) -> None:
    if not isinstance(clean_images, Tensor):
        raise TypeError("clean_images must be a torch.Tensor")
    if clean_images.ndim != 4 or clean_images.shape[0] <= 0:
        raise ValueError("clean_images must have shape [B, 3, 224, 224]")
    if tuple(clean_images.shape) != (
        clean_images.shape[0], 3, V2_IMAGE_HEIGHT, V2_IMAGE_WIDTH
    ):
        raise ValueError("clean_images must have shape [B, 3, 224, 224]")
    if not clean_images.is_floating_point():
        raise TypeError("clean_images must use a floating dtype")
    if not torch.isfinite(clean_images).all().item():
        raise ValueError("clean_images must contain only finite values")


def _v2_metadata(
    sample_id: int,
    condition: str,
    spec: ConditionSpec,
    geometry: GeometryV2,
    *,
    context: str,
    seed: int,
    epoch: int | None,
) -> MaskMetadataV2:
    return MaskMetadataV2(
        sample_id=sample_id,
        condition=condition,
        occlusion_type=spec.occlusion_type,
        ratio_token=_v2_ratio_token(spec),
        target_ratio=spec.target_ratio,
        actual_ratio=geometry.actual_ratio,
        top=geometry.top,
        left=geometry.left,
        height=geometry.height,
        width=geometry.width,
        masked_pixel_count=geometry.masked_pixel_count,
        total_pixel_count=geometry.total_pixel_count,
        image_height=V2_IMAGE_HEIGHT,
        image_width=V2_IMAGE_WIDTH,
        algorithm_version=V2_ALGORITHM_VERSION,
        context=context,
        seed=seed,
        epoch=epoch,
    )


def _clone_and_fill_v2(
    clean_images: Tensor,
    geometries: tuple[GeometryV2, ...],
    fill_values: tuple[float, float, float],
) -> Tensor:
    masked_images = clean_images.clone()
    fill = clean_images.new_tensor(fill_values).view(3, 1, 1)
    for sample_index, geometry in enumerate(geometries):
        masked_images[
            sample_index,
            :,
            geometry.top : geometry.top + geometry.height,
            geometry.left : geometry.left + geometry.width,
        ] = fill
    return masked_images


def apply_evaluation_batch_v2(
    clean_images: Tensor,
    sample_ids: Tensor,
    condition: str,
    fill_vector: Sequence[Real],
) -> tuple[Tensor, tuple[MaskMetadataV2, ...]]:
    """Apply a fixed PublicTest v2 mask without mutating the source batch."""
    ids, spec, values = _validate_v2_batch_inputs(
        clean_images, sample_ids, condition, fill_vector,
        reject_duplicate_ids=True,
    )
    geometries = tuple(evaluation_geometry_v2(sample_id, condition) for sample_id in ids)
    metadata = tuple(
        _v2_metadata(
            sample_id, condition, spec, geometry,
            context="validation", seed=V2_EVALUATION_MASK_SEED, epoch=None,
        )
        for sample_id, geometry in zip(ids, geometries, strict=True)
    )
    return _clone_and_fill_v2(clean_images, geometries, values), metadata


def apply_training_batch_v2(
    clean_images: Tensor,
    sample_ids: Tensor,
    condition: str,
    fill_vector: Sequence[Real],
    *,
    training_seed: int,
    epoch: int,
) -> tuple[Tensor, tuple[MaskMetadataV2, ...]]:
    ids, spec, values = _validate_v2_batch_inputs(
        clean_images, sample_ids, condition, fill_vector,
        reject_duplicate_ids=False,
    )
    geometries = tuple(
        training_geometry_v2(sample_id, condition, training_seed, epoch)
        for sample_id in ids
    )
    metadata = tuple(
        _v2_metadata(
            sample_id, condition, spec, geometry,
            context="train", seed=training_seed, epoch=epoch,
        )
        for sample_id, geometry in zip(ids, geometries, strict=True)
    )
    return _clone_and_fill_v2(clean_images, geometries, values), metadata


def apply_evaluation_mask_v2(
    clean_image: Tensor,
    sample_id: int,
    condition: str,
    fill_vector: Sequence[Real],
) -> tuple[Tensor, MaskMetadataV2]:
    if not isinstance(clean_image, Tensor) or tuple(clean_image.shape) != (
        3, V2_IMAGE_HEIGHT, V2_IMAGE_WIDTH
    ):
        raise ValueError("clean_image must have shape [3, 224, 224]")
    result, metadata = apply_evaluation_batch_v2(
        clean_image.unsqueeze(0),
        torch.tensor([sample_id], dtype=torch.int64, device=clean_image.device),
        condition,
        fill_vector,
    )
    return result[0], metadata[0]
