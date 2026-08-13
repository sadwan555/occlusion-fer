import csv
import hashlib
import json

import pytest

from occlusion_fer.formal_checkpoints import FORMAL_CHECKPOINTS
from occlusion_fer.private_manifest import TRAINING_MEAN_ARTIFACT_SHA256
from occlusion_fer.private_plan import build_final_plan
import occlusion_fer.private_preflight as preflight_module
from occlusion_fer.private_preflight import (
    PrivatePreflightError,
    inspect_private_source,
    preflight_checkpoints,
    validate_training_mean_artifact,
    validate_official_private_source,
)


TRAINING_DATASET_SHA256 = (
    "42eb71ae81c749a40426583445c1fda1db904fc73ec6798b07596e01ebb89a4c"
)
FORMAL_TRAINING_MEAN = {
    "accumulator_dtype": "uint64",
    "artifact_sha256": TRAINING_MEAN_ARTIFACT_SHA256,
    "consumer_image_height": 224,
    "consumer_image_width": 224,
    "dataset_name": "fer2013",
    "fill_domain": "normalized_imagenet_after_resize",
    "mean_algorithm_version": "training-mean-v2",
    "pixel_count": 66145536,
    "raw_pixel_sum": 8564149588,
    "raw_training_mean": 0.5077425080522144,
    "schema_version": 2,
    "source_image_height": 48,
    "source_image_width": 48,
    "split": "train",
    "training_dataset_sha256": TRAINING_DATASET_SHA256,
}


def _canonical_json_bytes(payload) -> bytes:
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


def _envelope_sha256(payload) -> str:
    envelope = dict(payload)
    envelope.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(envelope)).hexdigest()


def _write_training_mean(tmp_path, payload=None, *, trailing_lf=True):
    path = tmp_path / "training_mean_v2.json"
    data = _canonical_json_bytes(payload or FORMAL_TRAINING_MEAN)
    path.write_bytes(data if trailing_lf else data.removesuffix(b"\n"))
    return path


def _write_synthetic_csv(tmp_path):
    path = tmp_path / "synthetic-fer2013.csv"
    pixels = " ".join(["0"] * (48 * 48))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("emotion", "pixels", "Usage"))
        writer.writerow(("0", pixels, "Training"))
        writer.writerow(("1", pixels, "PublicTest"))
        writer.writerow(("2", pixels, "PrivateTest"))
        writer.writerow(("3", pixels, "PrivateTest"))
    return path


def test_inspect_private_source_only_returns_test_identity(tmp_path) -> None:
    path = _write_synthetic_csv(tmp_path)
    identity = inspect_private_source(path)
    assert identity.data.train_count == 0
    assert identity.data.validation_count == 0
    assert identity.data.test_count == 2
    assert identity.sample_ids == (4, 5)
    lines = []
    for record in identity.data.records:
        lines.append(
            json.dumps(
                {
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "pixels": [int(pixel) for pixel in record.image.reshape(-1)],
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    expected = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    assert identity.canonical_sha256 == expected


def test_official_preflight_rejects_synthetic_cardinality(tmp_path) -> None:
    with pytest.raises(PrivatePreflightError, match="3589"):
        validate_official_private_source(inspect_private_source(_write_synthetic_csv(tmp_path)))


def test_checkpoint_preflight_visits_exactly_six_frozen_entries(
    tmp_path, monkeypatch
) -> None:
    paths = {
        item.model_id: str(tmp_path / f"{item.model_id}.pt")
        for item in FORMAL_CHECKPOINTS
    }
    plan = build_final_plan(
        checkpoint_paths=paths,
        output_root=str(tmp_path / "final"),
        git_commit="a" * 40,
        git_dirty=True,
        created_at="2026-08-13T12:00:00Z",
    )
    checked = []

    def fake_validate(path, spec):
        checked.append((str(path), spec.model_id))
        return ({"model_state_dict": {}}, spec.sha256)

    monkeypatch.setattr(
        preflight_module, "load_and_validate_checkpoint", fake_validate
    )
    preflight_checkpoints(plan)
    assert [model_id for _, model_id in checked] == [
        item.model_id for item in FORMAL_CHECKPOINTS
    ]


def test_formal_training_mean_uses_envelope_sha_with_trailing_lf(tmp_path) -> None:
    path = _write_training_mean(tmp_path)

    assert _envelope_sha256(FORMAL_TRAINING_MEAN) == TRAINING_MEAN_ARTIFACT_SHA256
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "becf171033a87fca4445b1081dcc656fbe6dba2c098c38ad1b06e4e20f279295"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() != (
        TRAINING_MEAN_ARTIFACT_SHA256
    )
    validate_training_mean_artifact(path)


def test_training_mean_rejects_changed_embedded_sha(tmp_path) -> None:
    payload = dict(FORMAL_TRAINING_MEAN)
    payload["artifact_sha256"] = "1" + TRAINING_MEAN_ARTIFACT_SHA256[1:]

    with pytest.raises(PrivatePreflightError):
        validate_training_mean_artifact(_write_training_mean(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_training_mean", 0.5, "mean"),
        ("training_dataset_sha256", "0" * 64, "dataset SHA"),
    ],
)
def test_training_mean_rejects_changed_frozen_envelope_field(
    tmp_path, field, value, message
) -> None:
    payload = dict(FORMAL_TRAINING_MEAN)
    payload[field] = value
    payload["artifact_sha256"] = _envelope_sha256(payload)

    with pytest.raises(PrivatePreflightError, match=message):
        validate_training_mean_artifact(_write_training_mean(tmp_path, payload))


def test_training_mean_requires_canonical_trailing_lf(tmp_path) -> None:
    path = _write_training_mean(tmp_path, trailing_lf=False)

    with pytest.raises(PrivatePreflightError, match="canonical LF"):
        validate_training_mean_artifact(path)
