import math

import pytest

from occlusion_fer.mask_hash import (
    apply_from_u64,
    build_evaluation_coordinate_payload,
    build_training_coordinate_payload,
    build_training_decision_payload,
    canonical_payload_bytes,
    coordinate_from_u64,
    ratio_index_from_u64,
    sha256_digest,
    type_index_from_u64,
    u64_for_payload,
    u64_from_digest,
)


EVALUATION_TOP_PAYLOAD = [
    "fer2013",
    "validation",
    28711,
    "random_rectangle_0.30",
    20260804,
    "occlusion-v1",
    "top",
]
EVALUATION_LEFT_PAYLOAD = [
    "fer2013",
    "validation",
    28711,
    "random_rectangle_0.30",
    20260804,
    "occlusion-v1",
    "left",
]
TRAINING_APPLY_PAYLOAD = [
    "fer2013",
    "train",
    28711,
    42,
    1,
    "occlusion-v1",
    "apply",
]


def test_serializes_exact_canonical_evaluation_bytes() -> None:
    assert canonical_payload_bytes(EVALUATION_TOP_PAYLOAD) == (
        b'["fer2013","validation",28711,"random_rectangle_0.30",'
        b'20260804,"occlusion-v1","top"]'
    )


def test_canonical_bytes_have_no_trailing_newline() -> None:
    assert not canonical_payload_bytes(EVALUATION_TOP_PAYLOAD).endswith(b"\n")


def test_canonical_bytes_escape_non_ascii_content() -> None:
    assert canonical_payload_bytes(["caf\N{LATIN SMALL LETTER E WITH ACUTE}"]) == (
        b'["caf\\u00e9"]'
    )


def test_canonical_bytes_reject_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_payload_bytes([math.nan])


def test_evaluation_top_golden_digest() -> None:
    assert sha256_digest(EVALUATION_TOP_PAYLOAD).hex() == (
        "1143203e5bd651f03c57bc0d33a9c9b1a67abe3abe59c0fdcaf39ae4972482b2"
    )


def test_evaluation_top_golden_u64() -> None:
    assert u64_for_payload(EVALUATION_TOP_PAYLOAD) == 1243873374285222384


def test_evaluation_left_golden_digest() -> None:
    assert sha256_digest(EVALUATION_LEFT_PAYLOAD).hex() == (
        "02c6f1f2889688fefb763e33845bb00e618be168351c3e9a1dc827806677d34c"
    )


def test_evaluation_left_golden_u64() -> None:
    assert u64_for_payload(EVALUATION_LEFT_PAYLOAD) == 200113257440512254


def test_training_apply_golden_digest() -> None:
    assert sha256_digest(TRAINING_APPLY_PAYLOAD).hex() == (
        "0c8c9e48579be533ab3e13aa635c1c6b442b013366bd5aea4e6026d59cca4bdd"
    )


def test_training_apply_golden_u64() -> None:
    assert u64_for_payload(TRAINING_APPLY_PAYLOAD) == 904271658739295539


def test_training_apply_golden_decision() -> None:
    assert apply_from_u64(u64_for_payload(TRAINING_APPLY_PAYLOAD)) is True


def test_digest_conversion_is_unsigned_big_endian() -> None:
    digest = bytes.fromhex("8000000000000000" + "00" * 24)

    assert u64_from_digest(digest) == 2**63


def test_apply_mapping_uses_locked_half_range_threshold() -> None:
    assert apply_from_u64(2**63 - 1) is True
    assert apply_from_u64(2**63) is False


def test_type_mapping_uses_modulo_three() -> None:
    assert type_index_from_u64(5) == 2


def test_ratio_mapping_uses_modulo_three() -> None:
    assert ratio_index_from_u64(7) == 1


def test_coordinate_mapping_uses_available_position_modulo() -> None:
    assert coordinate_from_u64(1243873374285222384, 52) == 28
    assert coordinate_from_u64(200113257440512254, 52) == 46


def test_builds_exact_training_decision_payload_without_model_data() -> None:
    assert build_training_decision_payload(28711, 42, 1, "apply") == (
        TRAINING_APPLY_PAYLOAD
    )


def test_builds_exact_training_coordinate_payload_without_model_data() -> None:
    assert build_training_coordinate_payload(
        28711,
        42,
        1,
        "random_rectangle_0.30",
        "top",
    ) == [
        "fer2013",
        "train",
        28711,
        42,
        1,
        "random_rectangle_0.30",
        "occlusion-v1",
        "top",
    ]


def test_builds_exact_fixed_validation_payload_without_model_data() -> None:
    assert build_evaluation_coordinate_payload(
        28711,
        "random_rectangle_0.30",
        "top",
    ) == EVALUATION_TOP_PAYLOAD


def test_training_decision_namespaces_are_independent() -> None:
    values = {
        u64_for_payload(build_training_decision_payload(28711, 42, 1, namespace))
        for namespace in ("apply", "type", "ratio")
    }

    assert len(values) == 3


def test_coordinate_axes_are_independent() -> None:
    top = u64_for_payload(
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            "top",
        )
    )
    left = u64_for_payload(
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            "left",
        )
    )

    assert top != left


def test_repeated_calls_return_the_same_value() -> None:
    first = u64_for_payload(build_training_decision_payload(28711, 42, 1, "type"))
    second = u64_for_payload(build_training_decision_payload(28711, 42, 1, "type"))

    assert first == second


def test_changing_sample_id_changes_evaluation_result() -> None:
    first = u64_for_payload(
        build_evaluation_coordinate_payload(28711, "random_rectangle_0.30", "top")
    )
    second = u64_for_payload(
        build_evaluation_coordinate_payload(28712, "random_rectangle_0.30", "top")
    )

    assert first != second


def test_changing_epoch_changes_training_result() -> None:
    first = u64_for_payload(build_training_decision_payload(28711, 42, 1, "apply"))
    second = u64_for_payload(build_training_decision_payload(28711, 42, 2, "apply"))

    assert first != second


def test_changing_seed_changes_training_result() -> None:
    first = u64_for_payload(build_training_decision_payload(28711, 42, 1, "apply"))
    second = u64_for_payload(build_training_decision_payload(28711, 123, 1, "apply"))

    assert first != second


@pytest.mark.parametrize(
    ("sample_id", "training_seed", "epoch"),
    [
        (True, 42, 1),
        (28711, True, 1),
        (28711, 42, True),
    ],
)
def test_training_decision_rejects_booleans_in_integer_fields(
    sample_id: object,
    training_seed: object,
    epoch: object,
) -> None:
    with pytest.raises(TypeError):
        build_training_decision_payload(
            sample_id,  # type: ignore[arg-type]
            training_seed,  # type: ignore[arg-type]
            epoch,  # type: ignore[arg-type]
            "apply",
        )


@pytest.mark.parametrize(
    ("sample_id", "training_seed", "epoch"),
    [
        (True, 42, 1),
        (28711, True, 1),
        (28711, 42, True),
    ],
)
def test_training_coordinate_rejects_booleans_in_integer_fields(
    sample_id: object,
    training_seed: object,
    epoch: object,
) -> None:
    with pytest.raises(TypeError):
        build_training_coordinate_payload(
            sample_id,  # type: ignore[arg-type]
            training_seed,  # type: ignore[arg-type]
            epoch,  # type: ignore[arg-type]
            "random_rectangle_0.30",
            "top",
        )


@pytest.mark.parametrize("sample_id", [0, -1])
def test_training_decision_rejects_nonpositive_sample_id(sample_id: int) -> None:
    with pytest.raises(ValueError):
        build_training_decision_payload(sample_id, 42, 1, "apply")


def test_training_decision_rejects_negative_seed() -> None:
    with pytest.raises(ValueError):
        build_training_decision_payload(28711, -1, 1, "apply")


def test_training_decision_rejects_epoch_zero() -> None:
    with pytest.raises(ValueError):
        build_training_decision_payload(28711, 42, 0, "apply")


@pytest.mark.parametrize("namespace", ["position", "top", "left", "Apply", ""])
def test_training_decision_rejects_invalid_namespace(namespace: str) -> None:
    with pytest.raises(ValueError):
        build_training_decision_payload(28711, 42, 1, namespace)


@pytest.mark.parametrize("axis", ["position", "apply", "Top", ""])
def test_coordinate_builders_reject_invalid_axis(axis: str) -> None:
    with pytest.raises(ValueError):
        build_training_coordinate_payload(
            28711,
            42,
            1,
            "random_rectangle_0.30",
            axis,
        )
    with pytest.raises(ValueError):
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            axis,
        )


@pytest.mark.parametrize("condition", ["", 0, None, True])
def test_coordinate_builders_reject_invalid_condition(condition: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_training_coordinate_payload(
            28711,
            42,
            1,
            condition,  # type: ignore[arg-type]
            "top",
        )
    with pytest.raises((TypeError, ValueError)):
        build_evaluation_coordinate_payload(
            28711,
            condition,  # type: ignore[arg-type]
            "top",
        )


@pytest.mark.parametrize("available_positions", [0, -1, True])
def test_coordinate_mapping_rejects_invalid_available_positions(
    available_positions: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        coordinate_from_u64(
            1243873374285222384,
            available_positions,  # type: ignore[arg-type]
        )


def test_evaluation_builder_rejects_free_split() -> None:
    with pytest.raises(TypeError):
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            "top",
            split="test",  # type: ignore[call-arg]
        )


def test_evaluation_builder_rejects_free_seed() -> None:
    with pytest.raises(TypeError):
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            "top",
            evaluation_seed=1,  # type: ignore[call-arg]
        )


def test_evaluation_builder_rejects_boolean_sample_id() -> None:
    with pytest.raises(TypeError):
        build_evaluation_coordinate_payload(
            True,
            "random_rectangle_0.30",
            "top",
        )


@pytest.mark.parametrize("forbidden_field", ["label", "prediction", "image_content"])
def test_evaluation_builder_rejects_model_or_image_inputs(
    forbidden_field: str,
) -> None:
    with pytest.raises(TypeError):
        build_evaluation_coordinate_payload(
            28711,
            "random_rectangle_0.30",
            "top",
            **{forbidden_field: object()},
        )


def test_digest_conversion_rejects_non_sha256_length() -> None:
    with pytest.raises(ValueError):
        u64_from_digest(b"too short")
