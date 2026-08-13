from dataclasses import replace

import pytest
import torch

from occlusion_fer.private_manifest import (
    ALGORITHM_VERSION,
    CONDITION_ORDER,
    MASKED_CONDITIONS,
    PrivateManifestError,
    apply_manifest_batch,
    build_manifest_rows,
    build_manifest_sidecar,
    canonical_manifest_bytes,
    canonical_sidecar_bytes,
    manifest_sha256,
    load_manifest_artifacts,
    private_geometry,
    write_manifest_artifacts,
    validate_manifest_binding,
)


DATASET_SHA = "a" * 64
SAMPLE_IDS = (101, 205)


def _manifest():
    rows = build_manifest_rows(SAMPLE_IDS, DATASET_SHA)
    sidecar = build_manifest_sidecar(rows, DATASET_SHA)
    return rows, sidecar


def test_private_manifest_has_nine_conditions_per_synthetic_sample() -> None:
    rows, sidecar = _manifest()
    assert CONDITION_ORDER == ("clean", *MASKED_CONDITIONS)
    assert len(MASKED_CONDITIONS) == 9
    assert len(rows) == len(SAMPLE_IDS) * 9
    assert tuple(row.condition for row in rows[:9]) == MASKED_CONDITIONS
    assert tuple(row.sample_id for row in rows[::9]) == SAMPLE_IDS
    assert {row.official_usage for row in rows} == {"PrivateTest"}
    assert {row.internal_split for row in rows} == {"test"}
    assert {row.algorithm_version for row in rows} == {ALGORITHM_VERSION}
    assert sidecar.row_count == 18


def test_manifest_generation_is_byte_deterministic() -> None:
    first_rows, first_sidecar = _manifest()
    second_rows, second_sidecar = _manifest()
    assert first_rows == second_rows
    assert canonical_manifest_bytes(first_rows) == canonical_manifest_bytes(second_rows)
    assert canonical_sidecar_bytes(first_sidecar) == canonical_sidecar_bytes(second_sidecar)


def test_manifest_disk_pair_round_trips_and_refuses_overwrite(tmp_path) -> None:
    rows, sidecar = _manifest()
    manifest_path = tmp_path / "privatetest_manifest_v2.csv"
    sidecar_path = tmp_path / "privatetest_manifest_v2.json"
    write_manifest_artifacts(manifest_path, sidecar_path, rows, sidecar)
    loaded_rows, loaded_sidecar = load_manifest_artifacts(
        manifest_path, sidecar_path
    )
    assert loaded_rows == rows
    assert loaded_sidecar == sidecar
    with pytest.raises(FileExistsError):
        write_manifest_artifacts(manifest_path, sidecar_path, rows, sidecar)


@pytest.mark.parametrize(
    ("condition", "top", "height", "width"),
    [
        ("upper_face_0.20", 0, 45, 224),
        ("upper_face_0.30", 0, 67, 224),
        ("upper_face_0.40", 0, 90, 224),
        ("lower_face_0.20", 179, 45, 224),
        ("lower_face_0.30", 157, 67, 224),
        ("lower_face_0.40", 134, 90, 224),
        ("random_rectangle_0.20", None, 100, 100),
        ("random_rectangle_0.30", None, 123, 123),
        ("random_rectangle_0.40", None, 142, 142),
    ],
)
def test_private_geometry_preserves_stage8_dimensions(
    condition: str, top: int | None, height: int, width: int
) -> None:
    geometry = private_geometry(101, condition)
    if top is not None:
        assert geometry.top == top
    assert geometry.height == height
    assert geometry.width == width
    assert 0 <= geometry.top <= 224 - height
    assert 0 <= geometry.left <= 224 - width


def test_random_geometry_depends_on_sample_id_but_not_model_seed() -> None:
    first = tuple(
        private_geometry(101, condition)
        for condition in MASKED_CONDITIONS
        if condition.startswith("random_rectangle")
    )
    second = tuple(
        private_geometry(205, condition)
        for condition in MASKED_CONDITIONS
        if condition.startswith("random_rectangle")
    )
    assert first != second
    for model_seed in (42, 123, 2026):
        del model_seed
        assert tuple(
            private_geometry(101, condition)
            for condition in MASKED_CONDITIONS
            if condition.startswith("random_rectangle")
        ) == first


def test_apply_manifest_batch_does_not_modify_source() -> None:
    rows, _ = _manifest()
    source = torch.zeros(2, 3, 224, 224)
    before = source.clone()
    masked = apply_manifest_batch(
        source,
        torch.tensor(SAMPLE_IDS),
        rows,
        "upper_face_0.20",
    )
    assert torch.equal(source, before)
    assert not torch.equal(masked, source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("private_dataset_sha256", "b" * 64, "dataset"),
        ("internal_split", "validation", "split"),
        ("official_usage", "PublicTest", "Usage"),
        ("manifest_sha256", "0" * 64, "manifest SHA"),
    ],
)
def test_manifest_binding_rejects_wrong_sidecar_field(
    field: str, value: object, message: str
) -> None:
    rows, sidecar = _manifest()
    changed = replace(sidecar, **{field: value})
    with pytest.raises(PrivateManifestError, match=message):
        validate_manifest_binding(
            rows,
            changed,
            expected_dataset_sha256=DATASET_SHA,
            expected_sample_ids=SAMPLE_IDS,
            require_official=False,
        )


def test_manifest_binding_rejects_wrong_sample_ids() -> None:
    rows, sidecar = _manifest()
    with pytest.raises(PrivateManifestError, match="sample"):
        validate_manifest_binding(
            rows,
            sidecar,
            expected_dataset_sha256=DATASET_SHA,
            expected_sample_ids=(101, 999),
            require_official=False,
        )


def test_manifest_rejects_changed_geometry() -> None:
    rows, _ = _manifest()
    changed = rows[:-1] + (replace(rows[-1], left=rows[-1].left + 1),)
    with pytest.raises(PrivateManifestError, match="geometry"):
        manifest_sha256(changed)
