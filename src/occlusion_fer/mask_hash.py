"""Canonical deterministic hashing for the locked occlusion protocol."""

import hashlib
import json


_DATASET_NAME = "fer2013"
_TRAINING_SPLIT = "train"
_EVALUATION_SPLIT = "validation"
_ALGORITHM_VERSION = "occlusion-v1"
_EVALUATION_MASK_SEED = 20260804
_DECISION_NAMESPACES = frozenset({"apply", "type", "ratio"})
_COORDINATE_AXES = frozenset({"top", "left"})
_APPLY_THRESHOLD = 2**63
_TYPE_COUNT = 3
_RATIO_COUNT = 3
_U64_LIMIT = 2**64


def build_training_decision_payload(
    sample_id: int,
    training_seed: int,
    epoch: int,
    namespace: str,
) -> list[object]:
    """Build a locked Training apply, type, or ratio payload."""
    _validate_training_identity(sample_id, training_seed, epoch)
    _require_member(namespace, _DECISION_NAMESPACES, "namespace")
    return [
        _DATASET_NAME,
        _TRAINING_SPLIT,
        sample_id,
        training_seed,
        epoch,
        _ALGORITHM_VERSION,
        namespace,
    ]


def build_training_coordinate_payload(
    sample_id: int,
    training_seed: int,
    epoch: int,
    selected_condition: str,
    axis: str,
) -> list[object]:
    """Build a locked Training top or left coordinate payload."""
    _validate_training_identity(sample_id, training_seed, epoch)
    _require_nonempty_string(selected_condition, "selected_condition")
    _require_member(axis, _COORDINATE_AXES, "axis")
    return [
        _DATASET_NAME,
        _TRAINING_SPLIT,
        sample_id,
        training_seed,
        epoch,
        selected_condition,
        _ALGORITHM_VERSION,
        axis,
    ]


def build_evaluation_coordinate_payload(
    sample_id: int,
    condition: str,
    axis: str,
) -> list[object]:
    """Build a locked validation coordinate payload with the fixed seed."""
    _require_positive_integer(sample_id, "sample_id")
    _require_nonempty_string(condition, "condition")
    _require_member(axis, _COORDINATE_AXES, "axis")
    return [
        _DATASET_NAME,
        _EVALUATION_SPLIT,
        sample_id,
        condition,
        _EVALUATION_MASK_SEED,
        _ALGORITHM_VERSION,
        axis,
    ]


def canonical_payload_bytes(payload: list[object]) -> bytes:
    """Serialize a protocol payload as canonical UTF-8 JSON array bytes."""
    if type(payload) is not list:
        raise TypeError("payload must be a list")
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_digest(payload: list[object]) -> bytes:
    """Return the SHA-256 digest of a canonical payload."""
    return hashlib.sha256(canonical_payload_bytes(payload)).digest()


def u64_from_digest(digest: bytes) -> int:
    """Interpret the first eight SHA-256 digest bytes as unsigned big-endian."""
    if type(digest) is not bytes:
        raise TypeError("digest must be bytes")
    if len(digest) != hashlib.sha256().digest_size:
        raise ValueError("digest must contain exactly 32 bytes")
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def u64_for_payload(payload: list[object]) -> int:
    """Return the locked unsigned 64-bit value for a canonical payload."""
    return u64_from_digest(sha256_digest(payload))


def apply_from_u64(value: int) -> bool:
    """Map a protocol u64 to the locked 0.5 apply decision."""
    _require_u64(value)
    return value < _APPLY_THRESHOLD


def type_index_from_u64(value: int) -> int:
    """Map a protocol u64 to the locked occlusion-type index."""
    _require_u64(value)
    return value % _TYPE_COUNT


def ratio_index_from_u64(value: int) -> int:
    """Map a protocol u64 to the locked occlusion-ratio index."""
    _require_u64(value)
    return value % _RATIO_COUNT


def coordinate_from_u64(value: int, available_positions: int) -> int:
    """Map a protocol u64 to a coordinate within available positions."""
    _require_u64(value)
    _require_positive_integer(available_positions, "available_positions")
    return value % available_positions


def _validate_training_identity(
    sample_id: int,
    training_seed: int,
    epoch: int,
) -> None:
    _require_positive_integer(sample_id, "sample_id")
    _require_nonnegative_integer(training_seed, "training_seed")
    _require_positive_integer(epoch, "epoch")


def _require_positive_integer(value: object, field_name: str) -> None:
    _require_integer(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    _require_integer(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_integer(value: object, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")


def _require_nonempty_string(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must be non-empty")


def _require_member(value: object, allowed: frozenset[str], field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if value not in allowed:
        raise ValueError(f"{field_name} is not supported")


def _require_u64(value: object) -> None:
    _require_integer(value, "value")
    if not 0 <= value < _U64_LIMIT:
        raise ValueError("value must be an unsigned 64-bit integer")
