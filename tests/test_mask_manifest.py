import csv
import hashlib
import io
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import occlusion_fer.data as data_module
import occlusion_fer.mask_manifest as manifest_module
from occlusion_fer.mask_manifest import (
    MANIFEST_COLUMNS,
    ManifestConflictError,
    ManifestError,
    canonical_manifest_bytes,
    enumerate_publictest_sample_ids,
    generate_and_write_manifest,
    generate_manifest_rows,
    generate_official_manifest_rows,
    manifest_sha256,
    write_manifest_create_or_verify,
)
from occlusion_fer.occlusion import MASKED_CONDITIONS
from occlusion_fer.training_mean import (
    TrainingMeanArtifact,
    canonical_artifact_bytes,
)


EXPECTED_DATASET_SHA256 = (
    "f0b820fa3c66a12d683e60bc89495ab6cd279d5134f809398f48af29cc54eec5"
)
EXPECTED_MANIFEST_SHA256 = (
    "837dbcb0bcfeb75b3ad3ff599a1b9ee7e12979f3cba930d61ee32cefab3612fe"
)
EXPECTED_HEADER = (
    "manifest_schema_version,algorithm_version,dataset_name,split,"
    "dataset_sha256,sample_id,condition,occlusion_type,target_ratio,"
    "image_height,image_width,top,left,height,width,masked_pixel_count,"
    "total_pixel_count,actual_ratio,raw_fill_value,normalized_fill_red,"
    "normalized_fill_green,normalized_fill_blue,evaluation_mask_seed,"
    "coordinate_convention"
)
EXPECTED_FIRST_ROW = (
    "1,occlusion-v1,fer2013,validation,"
    "f0b820fa3c66a12d683e60bc89495ab6cd279d5134f809398f48af29cc54eec5,"
    "3,upper_face_0.20,upper_face,0.20,112,112,0,0,22,112,2464,12544,"
    "0.1964285714,0.00000000000000000,-2.11790393013100431,"
    "-2.03571428571428559,-1.80444444444444452,20260804,"
    '"half-open:[top,top+height)x[left,left+width)"'
)
EXPECTED_RANDOM_ROW = (
    "1,occlusion-v1,fer2013,validation,"
    "f0b820fa3c66a12d683e60bc89495ab6cd279d5134f809398f48af29cc54eec5,"
    "3,random_rectangle_0.20,random_rectangle,0.20,112,112,22,6,50,50,"
    "2500,12544,0.1992984694,0.00000000000000000,-2.11790393013100431,"
    "-2.03571428571428559,-1.80444444444444452,20260804,"
    '"half-open:[top,top+height)x[left,left+width)"'
)
EXPECTED_LAST_ROW = (
    "1,occlusion-v1,fer2013,validation,"
    "f0b820fa3c66a12d683e60bc89495ab6cd279d5134f809398f48af29cc54eec5,"
    "5,random_rectangle_0.40,random_rectangle,0.40,112,112,10,38,71,71,"
    "5041,12544,0.4018654337,0.00000000000000000,-2.11790393013100431,"
    "-2.03571428571428559,-1.80444444444444452,20260804,"
    '"half-open:[top,top+height)x[left,left+width)"'
)


def _combined_csv_bytes() -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["emotion", "pixels", "Usage"])
    writer.writerow(["0", "DO NOT PARSE", "Training"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PublicTest"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PrivateTest"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PublicTest"])
    return buffer.getvalue().encode("utf-8")


def _write_csv(tmp_path: Path, data: bytes | None = None) -> Path:
    path = tmp_path / "fer2013.csv"
    path.write_bytes(_combined_csv_bytes() if data is None else data)
    return path


def _artifact(dataset_sha256: str = EXPECTED_DATASET_SHA256) -> TrainingMeanArtifact:
    return TrainingMeanArtifact(
        schema_version=1,
        dataset_name="fer2013",
        split="train",
        dataset_sha256=dataset_sha256,
        sample_count=1,
        pixel_count=2304,
        raw_pixel_sum=0,
        raw_training_mean=0.0,
        source_pixel_range=(0, 255),
        normalized_pixel_range=(0.0, 1.0),
        accumulator_dtype="uint64",
        mean_algorithm_version="training-mean-v1",
    )


def _write_mean_artifact(
    tmp_path: Path,
    artifact: TrainingMeanArtifact | None = None,
) -> Path:
    path = tmp_path / "training_mean.json"
    path.write_bytes(canonical_artifact_bytes(artifact or _artifact()))
    return path


def _rows(tmp_path: Path):
    csv_path = _write_csv(tmp_path)
    mean_path = _write_mean_artifact(tmp_path)
    return generate_manifest_rows(csv_path, mean_path)


def test_manifest_column_order_is_locked() -> None:
    assert MANIFEST_COLUMNS == (
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


def test_enumerates_only_publictest_original_csv_line_numbers(tmp_path: Path) -> None:
    assert enumerate_publictest_sample_ids(_write_csv(tmp_path)) == (3, 5)


def test_manifest_generation_never_calls_label_or_pixel_parsers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_parser(*args: object, **kwargs: object) -> None:
        raise AssertionError("label and pixel parsers must not be called")

    monkeypatch.setattr(data_module, "_parse_label", forbidden_parser)
    monkeypatch.setattr(data_module, "_parse_image", forbidden_parser)

    rows = generate_manifest_rows(
        _write_csv(tmp_path),
        _write_mean_artifact(tmp_path),
    )

    assert len(rows) == 18
    assert {row.sample_id for row in rows} == {3, 5}


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"emotion,pixels,Usage\n",
        b"emotion,pixels\n0,x\n",
        b"emotion,Usage\n0,Training\n",
        b"pixels,Usage\nx,Training\n",
        b"emotion,pixels,Usage\n0,x\n",
        b"emotion,pixels,Usage\n0,x,Development\n",
    ],
)
def test_enumeration_rejects_empty_or_invalid_csv(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(ManifestError):
        enumerate_publictest_sample_ids(_write_csv(tmp_path, data))


def test_enumeration_rejects_duplicate_header_names(tmp_path: Path) -> None:
    data = b"emotion,pixels,Usage,Usage\n0,x,Training,Training\n"

    with pytest.raises(ManifestError, match="header"):
        enumerate_publictest_sample_ids(_write_csv(tmp_path, data))


def test_enumeration_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        enumerate_publictest_sample_ids(tmp_path / "missing.csv")


def test_generates_nine_conditions_for_each_publictest_id(tmp_path: Path) -> None:
    rows = _rows(tmp_path)

    assert len(rows) == 18
    assert tuple(row.condition for row in rows) == tuple(
        condition for condition in MASKED_CONDITIONS for _ in range(2)
    )
    assert tuple(row.sample_id for row in rows) == (3, 5) * 9


def test_generated_rows_have_locked_protocol_and_geometry(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    first = rows[0]
    random_row = next(
        row
        for row in rows
        if row.sample_id == 3 and row.condition == "random_rectangle_0.20"
    )

    assert first.manifest_schema_version == 1
    assert first.algorithm_version == "occlusion-v1"
    assert first.dataset_name == "fer2013"
    assert first.split == "validation"
    assert first.dataset_sha256 == EXPECTED_DATASET_SHA256
    assert first.image_height == first.image_width == 112
    assert first.total_pixel_count == 12544
    assert first.evaluation_mask_seed == 20260804
    assert first.coordinate_convention == (
        "half-open:[top,top+height)x[left,left+width)"
    )
    assert (random_row.top, random_row.left, random_row.height, random_row.width) == (
        22,
        6,
        50,
        50,
    )


def test_golden_sample_geometry_is_preserved_in_manifest_row() -> None:
    row = next(
        row
        for row in manifest_module._build_manifest_rows((28711,), _artifact())
        if row.condition == "random_rectangle_0.30"
    )

    assert (
        row.top,
        row.left,
        row.height,
        row.width,
        row.masked_pixel_count,
        row.actual_ratio,
    ) == (28, 46, 61, 61, 3721, 3721 / 12544)


def test_manifest_rows_are_immutable(tmp_path: Path) -> None:
    row = _rows(tmp_path)[0]

    with pytest.raises(FrozenInstanceError):
        row.top = 1


def test_row_builder_rejects_duplicate_or_unsorted_sample_ids() -> None:
    with pytest.raises(ManifestError, match="sample_id"):
        manifest_module._build_manifest_rows((), _artifact())
    with pytest.raises(ManifestError, match="sample_id"):
        manifest_module._build_manifest_rows((3, 3), _artifact())
    with pytest.raises(ManifestError, match="sample_id"):
        manifest_module._build_manifest_rows((5, 3), _artifact())


def test_canonical_manifest_bytes_and_sha_are_locked(tmp_path: Path) -> None:
    data = canonical_manifest_bytes(_rows(tmp_path))
    lines = data.decode("utf-8").splitlines()

    assert len(data) == 6095
    assert len(lines) == 19
    assert lines[0] == EXPECTED_HEADER
    assert lines[1] == EXPECTED_FIRST_ROW
    assert lines[13] == EXPECTED_RANDOM_ROW
    assert lines[-1] == EXPECTED_LAST_ROW
    assert data.endswith(b"\n")
    assert b"\r" not in data
    assert hashlib.sha256(data).hexdigest() == EXPECTED_MANIFEST_SHA256
    assert manifest_sha256(_rows(tmp_path)) == EXPECTED_MANIFEST_SHA256


def test_canonical_serialization_is_repeatable(tmp_path: Path) -> None:
    rows = _rows(tmp_path)

    assert canonical_manifest_bytes(rows) == canonical_manifest_bytes(rows)


def test_manifest_contains_no_labels_predictions_metrics_or_paths(
    tmp_path: Path,
) -> None:
    header = canonical_manifest_bytes(_rows(tmp_path)).splitlines()[0].decode("ascii")

    for forbidden in (
        "emotion",
        "label",
        "prediction",
        "accuracy",
        "f1",
        "path",
        "created_at",
    ):
        assert forbidden not in header


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_schema_version", 2),
        ("algorithm_version", "occlusion-v2"),
        ("dataset_name", "other"),
        ("split", "test"),
        ("dataset_sha256", "0" * 64),
        ("sample_id", 0),
        ("condition", "original"),
        ("image_height", 48),
        ("top", -1),
        ("masked_pixel_count", 1),
        ("actual_ratio", float("nan")),
        ("raw_fill_value", float("inf")),
        ("evaluation_mask_seed", 1),
        ("coordinate_convention", "closed"),
    ],
)
def test_serializer_rejects_invalid_manifest_rows(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    row = replace(_rows(tmp_path)[0], **{field: value})

    with pytest.raises(ManifestError):
        canonical_manifest_bytes((row,))


def test_serializer_rejects_empty_rows_or_wrong_object() -> None:
    with pytest.raises(ManifestError):
        canonical_manifest_bytes(())
    with pytest.raises(ManifestError):
        canonical_manifest_bytes((object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_sha256": "a" * 64},
        {
            "raw_fill_value": 0.5,
            "normalized_fill_red": (0.5 - 0.485) / 0.229,
            "normalized_fill_green": (0.5 - 0.456) / 0.224,
            "normalized_fill_blue": (0.5 - 0.406) / 0.225,
        },
    ],
)
def test_serializer_rejects_mixed_dataset_or_fill_identity(
    tmp_path: Path,
    changes: dict[str, object],
) -> None:
    rows = _rows(tmp_path)
    changed = replace(rows[-1], **changes)

    with pytest.raises(ManifestError, match="same"):
        canonical_manifest_bytes(rows[:-1] + (changed,))


def test_generation_rejects_noncanonical_mean_artifact(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    mean_path = tmp_path / "training_mean.json"
    mean_path.write_bytes(canonical_artifact_bytes(_artifact())[:-1] + b" \n")

    with pytest.raises(ManifestError, match="mean artifact"):
        generate_manifest_rows(csv_path, mean_path)


def test_generation_rejects_mean_artifact_bound_to_other_csv(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    mean_path = _write_mean_artifact(tmp_path, _artifact("0" * 64))

    with pytest.raises(ManifestError, match="dataset_sha256"):
        generate_manifest_rows(csv_path, mean_path)


def test_changed_csv_is_rejected_after_mean_artifact_creation(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    mean_path = _write_mean_artifact(tmp_path)
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    with pytest.raises(ManifestError, match="dataset_sha256"):
        generate_manifest_rows(csv_path, mean_path)


def test_generation_does_not_accept_free_protocol_fields(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path)
    mean_path = _write_mean_artifact(tmp_path)

    for keyword in (
        {"split": "test"},
        {"evaluation_mask_seed": 1},
        {"conditions": ("upper_face_0.20",)},
        {"labels": (0,)},
        {"model": object()},
        {"metrics": ("accuracy",)},
    ):
        with pytest.raises(TypeError):
            generate_manifest_rows(
                csv_path,
                mean_path,
                **keyword,  # type: ignore[call-arg]
            )


def test_official_wrapper_rejects_tiny_publictest_count(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="3589"):
        generate_official_manifest_rows(
            _write_csv(tmp_path),
            _write_mean_artifact(tmp_path),
        )


def test_create_or_verify_creates_exact_bytes(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    target = tmp_path / "validation_manifest.csv"

    result = write_manifest_create_or_verify(target, rows)

    assert result.path == target
    assert result.sha256 == EXPECTED_MANIFEST_SHA256
    assert result.status == "created"
    assert target.read_bytes() == canonical_manifest_bytes(rows)


def test_create_or_verify_preserves_identical_existing_inode_and_mtime(
    tmp_path: Path,
) -> None:
    rows = _rows(tmp_path)
    target = tmp_path / "validation_manifest.csv"
    target.write_bytes(canonical_manifest_bytes(rows))
    before = target.stat()

    result = write_manifest_create_or_verify(target, rows)
    after = target.stat()

    assert result.status == "consistent"
    assert after.st_ino == before.st_ino
    assert after.st_mtime_ns == before.st_mtime_ns
    assert not list(tmp_path.glob(".validation_manifest.csv.*.tmp"))


def test_create_or_verify_rejects_conflicting_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "validation_manifest.csv"
    target.write_bytes(b"existing research artifact\n")
    before_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()

    with pytest.raises(ManifestConflictError, match="conflict"):
        write_manifest_create_or_verify(target, _rows(tmp_path))

    assert target.read_bytes() == b"existing research artifact\n"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before_sha256
    assert not list(tmp_path.glob(".validation_manifest.csv.*.tmp"))


@pytest.mark.parametrize("same_bytes", [True, False])
def test_create_or_verify_handles_publish_race_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_bytes: bool,
) -> None:
    rows = _rows(tmp_path)
    expected = canonical_manifest_bytes(rows)
    target = tmp_path / "validation_manifest.csv"

    def racing_link(source: object, destination: object) -> None:
        Path(destination).write_bytes(expected if same_bytes else b"racing artifact\n")
        raise FileExistsError

    monkeypatch.setattr(os, "link", racing_link)

    if same_bytes:
        result = write_manifest_create_or_verify(target, rows)
        assert result.status == "consistent"
        assert target.read_bytes() == expected
    else:
        with pytest.raises(ManifestConflictError, match="conflict"):
            write_manifest_create_or_verify(target, rows)
        assert target.read_bytes() == b"racing artifact\n"
    assert not list(tmp_path.glob(".validation_manifest.csv.*.tmp"))


def test_create_or_verify_cleans_temporary_file_on_publish_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "validation_manifest.csv"

    def failing_link(source: object, destination: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "link", failing_link)

    with pytest.raises(PermissionError, match="denied"):
        write_manifest_create_or_verify(target, _rows(tmp_path))

    assert not target.exists()
    assert not list(tmp_path.glob(".validation_manifest.csv.*.tmp"))


def test_create_or_verify_rejects_missing_parent(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "manifest.csv"

    with pytest.raises(FileNotFoundError):
        write_manifest_create_or_verify(target, _rows(tmp_path))


def test_create_or_verify_rejects_directory_target(tmp_path: Path) -> None:
    target = tmp_path / "manifest.csv"
    target.mkdir()

    with pytest.raises(ManifestError, match="regular file"):
        write_manifest_create_or_verify(target, _rows(tmp_path))


def test_create_or_verify_rejects_symlink_target(tmp_path: Path) -> None:
    real_target = tmp_path / "real.csv"
    real_target.write_bytes(b"do not replace\n")
    target = tmp_path / "manifest.csv"
    target.symlink_to(real_target)

    with pytest.raises(ManifestError, match="symlink"):
        write_manifest_create_or_verify(target, _rows(tmp_path))

    assert real_target.read_bytes() == b"do not replace\n"


def test_generate_and_write_binding_failure_does_not_create_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "validation_manifest.csv"

    with pytest.raises(ManifestError, match="dataset_sha256"):
        generate_and_write_manifest(
            _write_csv(tmp_path),
            _write_mean_artifact(tmp_path, _artifact("0" * 64)),
            target,
        )

    assert not target.exists()


def test_generate_and_write_binding_failure_preserves_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "validation_manifest.csv"
    target.write_bytes(b"prior artifact\n")

    with pytest.raises(ManifestError, match="dataset_sha256"):
        generate_and_write_manifest(
            _write_csv(tmp_path),
            _write_mean_artifact(tmp_path, _artifact("0" * 64)),
            target,
        )

    assert target.read_bytes() == b"prior artifact\n"
