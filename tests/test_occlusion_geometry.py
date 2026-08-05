import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from occlusion_fer.occlusion import (
    MASKED_CONDITIONS,
    evaluation_geometry,
    normalized_fill_vector,
    parse_condition,
    round_half_up,
    training_geometry,
)


LOCKED_GEOMETRIES = (
    ("upper_face_0.20", 0, 0, 22, 112, 2464),
    ("upper_face_0.30", 0, 0, 34, 112, 3808),
    ("upper_face_0.40", 0, 0, 45, 112, 5040),
    ("lower_face_0.20", 90, 0, 22, 112, 2464),
    ("lower_face_0.30", 78, 0, 34, 112, 3808),
    ("lower_face_0.40", 67, 0, 45, 112, 5040),
    ("random_rectangle_0.20", None, None, 50, 50, 2500),
    ("random_rectangle_0.30", 28, 46, 61, 61, 3721),
    ("random_rectangle_0.40", None, None, 71, 71, 5041),
)


def test_masked_condition_order_is_locked() -> None:
    assert MASKED_CONDITIONS == (
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


@pytest.mark.parametrize(
    ("condition", "expected_type", "expected_ratio"),
    [
        ("upper_face_0.20", "upper_face", 0.20),
        ("lower_face_0.30", "lower_face", 0.30),
        ("random_rectangle_0.40", "random_rectangle", 0.40),
    ],
)
def test_parses_canonical_condition(
    condition: str,
    expected_type: str,
    expected_ratio: float,
) -> None:
    parsed = parse_condition(condition)

    assert parsed.occlusion_type == expected_type
    assert parsed.target_ratio == expected_ratio


@pytest.mark.parametrize(
    "condition",
    [
        "original",
        "clean",
        "upper_face",
        "lower_face",
        "random_rectangle",
        "upper_face_0.2",
        "UPPER_FACE_0.20",
        " upper_face_0.20",
        "upper_face_0.20 ",
        "unknown_0.20",
        "upper_face_0.10",
        "",
        None,
        0.20,
    ],
)
def test_rejects_noncanonical_condition(condition: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_condition(condition)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0),
        (0.49, 0),
        (0.5, 1),
        (1.49, 1),
        (1.5, 2),
        (22.4, 22),
        (33.6, 34),
        (44.8, 45),
    ],
)
def test_round_half_up_uses_locked_rule(value: float, expected: int) -> None:
    assert round_half_up(value) == expected


@pytest.mark.parametrize(
    "value",
    [math.nan, math.inf, -math.inf, -0.01, True, "0.5", None],
)
def test_round_half_up_rejects_invalid_value(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        round_half_up(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("condition", "top", "left", "height", "width", "masked_pixels"),
    LOCKED_GEOMETRIES,
)
def test_evaluation_geometry_matches_locked_dimensions(
    condition: str,
    top: int | None,
    left: int | None,
    height: int,
    width: int,
    masked_pixels: int,
) -> None:
    geometry = evaluation_geometry(28711, condition)

    if top is not None:
        assert geometry.top == top
    if left is not None:
        assert geometry.left == left
    assert geometry.height == height
    assert geometry.width == width
    assert geometry.masked_pixel_count == masked_pixels
    assert geometry.total_pixel_count == 12544
    assert geometry.actual_ratio == masked_pixels / 12544


@pytest.mark.parametrize(
    "condition",
    [
        "upper_face_0.20",
        "upper_face_0.30",
        "upper_face_0.40",
    ],
)
def test_upper_face_geometry_is_top_aligned(condition: str) -> None:
    geometry = evaluation_geometry(28711, condition)

    assert geometry.top == 0
    assert geometry.left == 0
    assert geometry.width == 112


@pytest.mark.parametrize(
    "condition",
    [
        "lower_face_0.20",
        "lower_face_0.30",
        "lower_face_0.40",
    ],
)
def test_lower_face_geometry_is_bottom_aligned(condition: str) -> None:
    geometry = evaluation_geometry(28711, condition)

    assert geometry.top + geometry.height == 112
    assert geometry.left == 0
    assert geometry.width == 112


@pytest.mark.parametrize(
    "condition",
    [
        "random_rectangle_0.20",
        "random_rectangle_0.30",
        "random_rectangle_0.40",
    ],
)
def test_random_geometry_is_a_square_inside_the_image(condition: str) -> None:
    geometry = evaluation_geometry(28711, condition)

    assert geometry.height == geometry.width
    assert 0 <= geometry.top
    assert 0 <= geometry.left
    assert geometry.top + geometry.height <= 112
    assert geometry.left + geometry.width <= 112


def test_evaluation_golden_geometry_is_locked() -> None:
    geometry = evaluation_geometry(28711, "random_rectangle_0.30")

    assert (
        geometry.top,
        geometry.left,
        geometry.height,
        geometry.width,
        geometry.masked_pixel_count,
        geometry.total_pixel_count,
        geometry.actual_ratio,
    ) == (28, 46, 61, 61, 3721, 12544, 3721 / 12544)


def test_evaluation_golden_uint8_mask_sha256_is_locked() -> None:
    geometry = evaluation_geometry(28711, "random_rectangle_0.30")
    mask = torch.zeros((112, 112), dtype=torch.uint8)
    mask[
        geometry.top : geometry.top + geometry.height,
        geometry.left : geometry.left + geometry.width,
    ] = 1

    assert hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest() == (
        "49fe672bc883fb68796d85c9317f68ed0c08ab532d7c82aac04495a5dce8d9fa"
    )


def test_geometry_uses_half_open_pixel_count() -> None:
    geometry = evaluation_geometry(28711, "random_rectangle_0.30")

    assert geometry.masked_pixel_count == (
        (geometry.top + geometry.height - geometry.top)
        * (geometry.left + geometry.width - geometry.left)
    )


def test_geometry_is_immutable() -> None:
    geometry = evaluation_geometry(28711, "upper_face_0.20")

    with pytest.raises(FrozenInstanceError):
        geometry.top = 1


@pytest.mark.parametrize("sample_id", [0, -1, True, 1.5])
def test_evaluation_geometry_rejects_invalid_sample_id(sample_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluation_geometry(
            sample_id,  # type: ignore[arg-type]
            "upper_face_0.20",
        )


def test_evaluation_geometry_does_not_accept_free_protocol_fields() -> None:
    with pytest.raises(TypeError):
        evaluation_geometry(
            28711,
            "random_rectangle_0.30",
            split="test",  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        evaluation_geometry(
            28711,
            "random_rectangle_0.30",
            evaluation_seed=1,  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        evaluation_geometry(
            28711,
            "random_rectangle_0.30",
            label=3,  # type: ignore[call-arg]
        )


def test_formal_geometry_does_not_accept_an_alternate_size() -> None:
    with pytest.raises(TypeError):
        evaluation_geometry(
            28711,
            "upper_face_0.20",
            image_size=48,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("sample_id", "training_seed", "epoch", "top", "left"),
    [
        (28711, 42, 1, 21, 38),
        (28711, 42, 2, 25, 36),
        (28711, 123, 1, 40, 36),
        (28712, 42, 1, 4, 10),
    ],
)
def test_training_random_geometry_matches_fixed_hash_cases(
    sample_id: int,
    training_seed: int,
    epoch: int,
    top: int,
    left: int,
) -> None:
    geometry = training_geometry(
        sample_id,
        "random_rectangle_0.30",
        training_seed,
        epoch,
    )

    assert (geometry.top, geometry.left, geometry.height, geometry.width) == (
        top,
        left,
        61,
        61,
    )


def test_training_geometry_is_repeatable() -> None:
    first = training_geometry(28711, "random_rectangle_0.30", 42, 1)
    second = training_geometry(28711, "random_rectangle_0.30", 42, 1)

    assert first == second


def test_training_band_geometry_is_fixed() -> None:
    first = training_geometry(28711, "lower_face_0.40", 42, 1)
    second = training_geometry(99999, "lower_face_0.40", 2026, 20)

    assert first == second
    assert (first.top, first.left, first.height, first.width) == (67, 0, 45, 112)


@pytest.mark.parametrize(
    ("sample_id", "training_seed", "epoch"),
    [
        (True, 42, 1),
        (28711, True, 1),
        (28711, 42, True),
        (0, 42, 1),
        (28711, -1, 1),
        (28711, 42, 0),
    ],
)
def test_training_geometry_rejects_invalid_integer_fields(
    sample_id: object,
    training_seed: object,
    epoch: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        training_geometry(
            sample_id,  # type: ignore[arg-type]
            "random_rectangle_0.30",
            training_seed,  # type: ignore[arg-type]
            epoch,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("raw_mean", [0.0, 0.5, 1.0])
def test_normalized_fill_vector_uses_imagenet_channel_formula(
    raw_mean: float,
) -> None:
    assert normalized_fill_vector(raw_mean) == pytest.approx(
        (
            (raw_mean - 0.485) / 0.229,
            (raw_mean - 0.456) / 0.224,
            (raw_mean - 0.406) / 0.225,
        )
    )


@pytest.mark.parametrize(
    "raw_mean",
    [math.nan, math.inf, -math.inf, -0.01, 1.01, True, "0.5", None],
)
def test_normalized_fill_vector_rejects_invalid_mean(raw_mean: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalized_fill_vector(raw_mean)  # type: ignore[arg-type]
