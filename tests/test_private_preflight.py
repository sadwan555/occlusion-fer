import csv
import hashlib
import json

import pytest

from occlusion_fer.formal_checkpoints import FORMAL_CHECKPOINTS
from occlusion_fer.private_plan import build_final_plan
import occlusion_fer.private_preflight as preflight_module
from occlusion_fer.private_preflight import (
    PrivatePreflightError,
    inspect_private_source,
    preflight_checkpoints,
    validate_official_private_source,
)


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
