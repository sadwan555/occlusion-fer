import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import torch

import occlusion_fer.data as data_module
import occlusion_fer.training_mean as training_mean_module
from occlusion_fer.occlusion import normalized_fill_vector
from occlusion_fer.training_mean import (
    TrainingMeanArtifact,
    TrainingMeanError,
    artifact_from_statistics,
    artifact_sha256,
    build_training_mean_artifact,
    calculate_training_statistics,
    canonical_artifact_bytes,
    load_training_mean_artifact,
    sha256_file,
    validate_artifact_dataset,
    write_training_mean_artifact,
)


PIXELS_PER_IMAGE = 48 * 48
EXPECTED_DATASET_SHA256 = (
    "ca6a1192afe2edd88de35b53b3c782f734305f6e32651363266a92ad4968d1d2"
)
EXPECTED_ARTIFACT_SHA256 = (
    "3672d07e0c272358214fc7781daf486ac832d46d55ceb0ff814032a1f0a44fcb"
)
EXPECTED_CANONICAL_BYTES = (
    b'{"accumulator_dtype":"uint64","dataset_name":"fer2013",'
    b'"dataset_sha256":"ca6a1192afe2edd88de35b53b3c782f734305f6e32651363266a92ad4968d1d2",'
    b'"mean_algorithm_version":"training-mean-v1",'
    b'"normalized_pixel_range":[0.0,1.0],"pixel_count":4608,'
    b'"raw_pixel_sum":294912,"raw_training_mean":0.25098039215686274,'
    b'"sample_count":2,"schema_version":1,"source_pixel_range":[0,255],'
    b'"split":"train"}\n'
)


def _pixel_string(value: int) -> str:
    return " ".join([str(value)] * PIXELS_PER_IMAGE)


def _combined_csv_bytes(*, line_terminator: str = "\n") -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=line_terminator)
    writer.writerow(["emotion", "pixels", "Usage"])
    writer.writerow(["0", _pixel_string(0), "Training"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PublicTest"])
    writer.writerow(["BROKEN", "DO NOT PARSE", "PrivateTest"])
    writer.writerow(["1", _pixel_string(128), "Training"])
    return buffer.getvalue().encode("utf-8")


def _write_combined_csv(tmp_path: Path, data: bytes | None = None) -> Path:
    path = tmp_path / "synthetic_fer2013.csv"
    path.write_bytes(_combined_csv_bytes() if data is None else data)
    return path


def _tiny_artifact() -> TrainingMeanArtifact:
    return TrainingMeanArtifact(
        schema_version=1,
        dataset_name="fer2013",
        split="train",
        dataset_sha256=EXPECTED_DATASET_SHA256,
        sample_count=2,
        pixel_count=4608,
        raw_pixel_sum=294912,
        raw_training_mean=0.25098039215686274,
        source_pixel_range=(0, 255),
        normalized_pixel_range=(0.0, 1.0),
        accumulator_dtype="uint64",
        mean_algorithm_version="training-mean-v1",
    )


def _mapping_from_golden() -> dict[str, object]:
    return json.loads(EXPECTED_CANONICAL_BYTES)


def _canonical_test_bytes(mapping: dict[str, object]) -> bytes:
    return json.dumps(
        mapping,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _write_artifact_bytes(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "training_mean.json"
    path.write_bytes(data)
    return path


def test_hashes_exact_combined_csv_bytes(tmp_path: Path) -> None:
    path = _write_combined_csv(tmp_path)

    assert sha256_file(path) == EXPECTED_DATASET_SHA256


def test_file_hash_is_repeatable(tmp_path: Path) -> None:
    path = _write_combined_csv(tmp_path)

    assert sha256_file(path) == sha256_file(path)


def test_file_hash_changes_when_one_byte_changes(tmp_path: Path) -> None:
    original = _write_combined_csv(tmp_path)
    changed = tmp_path / "changed.csv"
    changed.write_bytes(_combined_csv_bytes() + b" ")

    assert sha256_file(original) != sha256_file(changed)


def test_file_hash_distinguishes_lf_and_crlf(tmp_path: Path) -> None:
    lf_path = _write_combined_csv(tmp_path)
    crlf_path = tmp_path / "crlf.csv"
    crlf_path.write_bytes(_combined_csv_bytes(line_terminator="\r\n"))

    assert sha256_file(lf_path) != sha256_file(crlf_path)


def test_file_hash_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.csv")


def test_calculates_tiny_training_statistics(tmp_path: Path) -> None:
    statistics = calculate_training_statistics(_write_combined_csv(tmp_path))

    assert statistics.dataset_sha256 == EXPECTED_DATASET_SHA256
    assert statistics.sample_count == 2
    assert statistics.pixel_count == 4608
    assert statistics.raw_pixel_sum == 294912
    assert type(statistics.raw_pixel_sum) is int
    assert statistics.raw_training_mean == 0.25098039215686274
    assert statistics.accumulator_dtype == "uint64"


def test_only_training_rows_reach_label_and_pixel_parsers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_labels: list[str] = []
    parsed_pixels: list[str] = []
    original_label_parser = data_module._parse_label
    original_pixel_parser = data_module._parse_image

    def recording_label_parser(value: str, row_number: int) -> int:
        parsed_labels.append(value)
        return original_label_parser(value, row_number)

    def recording_pixel_parser(value: str, row_number: int) -> np.ndarray:
        parsed_pixels.append(value)
        return original_pixel_parser(value, row_number)

    monkeypatch.setattr(data_module, "_parse_label", recording_label_parser)
    monkeypatch.setattr(data_module, "_parse_image", recording_pixel_parser)

    statistics = calculate_training_statistics(_write_combined_csv(tmp_path))

    assert parsed_labels == ["0", "1"]
    assert parsed_pixels == [_pixel_string(0), _pixel_string(128)]
    assert statistics.raw_pixel_sum == 294912


@pytest.mark.parametrize("excluded_usage", ["PublicTest", "PrivateTest"])
def test_broken_excluded_rows_do_not_affect_training_mean(
    tmp_path: Path,
    excluded_usage: str,
) -> None:
    data = _combined_csv_bytes().replace(
        f"BROKEN,DO NOT PARSE,{excluded_usage}".encode("utf-8"),
        f"INVALID LABEL,INVALID PIXELS,{excluded_usage}".encode("utf-8"),
    )

    statistics = calculate_training_statistics(_write_combined_csv(tmp_path, data))

    assert statistics.raw_pixel_sum == 294912
    assert statistics.sample_count == 2


def test_broken_training_label_fails(tmp_path: Path) -> None:
    data = _combined_csv_bytes().replace(
        b"0," + _pixel_string(0).encode("utf-8") + b",Training",
        b"BROKEN," + _pixel_string(0).encode("utf-8") + b",Training",
    )

    with pytest.raises(TrainingMeanError, match="emotion"):
        calculate_training_statistics(_write_combined_csv(tmp_path, data))


def test_broken_training_pixels_fail(tmp_path: Path) -> None:
    data = _combined_csv_bytes().replace(
        _pixel_string(0).encode("utf-8"),
        b"BROKEN PIXELS",
        1,
    )

    with pytest.raises(TrainingMeanError, match="pixels"):
        calculate_training_statistics(_write_combined_csv(tmp_path, data))


def test_unknown_usage_fails(tmp_path: Path) -> None:
    data = _combined_csv_bytes().replace(b"PublicTest", b"Development", 1)

    with pytest.raises(TrainingMeanError, match="Usage"):
        calculate_training_statistics(_write_combined_csv(tmp_path, data))


def test_missing_usage_value_fails(tmp_path: Path) -> None:
    row = b"2," + _pixel_string(0).encode("utf-8") + b"\n"
    data = _combined_csv_bytes() + row

    with pytest.raises(TrainingMeanError, match="Usage"):
        calculate_training_statistics(_write_combined_csv(tmp_path, data))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"0," + b"0 " * 2303 + b"0,Training\n",
        b"emotion,pixels\n0,0\n",
        b"emotion,Usage\n0,Training\n",
        b"pixels,Usage\n0,Training\n",
        b"emotion,pixels,Usage\n",
    ],
)
def test_rejects_empty_or_incomplete_csv(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(TrainingMeanError):
        calculate_training_statistics(_write_combined_csv(tmp_path, data))


def test_checked_addition_uses_numpy_uint64() -> None:
    result = training_mean_module._checked_uint64_add(
        np.uint64(1),
        np.uint64(2),
    )

    assert isinstance(result, np.uint64)
    assert result == np.uint64(3)


def test_checked_addition_rejects_uint64_overflow() -> None:
    maximum = np.iinfo(np.uint64).max

    with pytest.raises(TrainingMeanError, match="overflow"):
        training_mean_module._checked_uint64_add(
            np.uint64(maximum),
            np.uint64(1),
        )


def test_official_builder_rejects_tiny_fixture_counts(tmp_path: Path) -> None:
    with pytest.raises(TrainingMeanError, match="28709"):
        build_training_mean_artifact(_write_combined_csv(tmp_path))


def test_official_builder_does_not_accept_split_selection(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_training_mean_artifact(
            _write_combined_csv(tmp_path),
            split="validation",  # type: ignore[call-arg]
        )


def test_artifact_from_statistics_has_exact_semantic_fields(tmp_path: Path) -> None:
    artifact = artifact_from_statistics(
        calculate_training_statistics(_write_combined_csv(tmp_path))
    )

    assert artifact == _tiny_artifact()
    assert set(_mapping_from_golden()) == {
        "schema_version",
        "dataset_name",
        "split",
        "dataset_sha256",
        "sample_count",
        "pixel_count",
        "raw_pixel_sum",
        "raw_training_mean",
        "source_pixel_range",
        "normalized_pixel_range",
        "accumulator_dtype",
        "mean_algorithm_version",
    }


def test_artifact_contains_no_event_or_private_data() -> None:
    serialized = EXPECTED_CANONICAL_BYTES.decode("utf-8")

    for forbidden in (
        "created_at_utc",
        "path",
        "PublicTest",
        "PrivateTest",
        "label",
        "pixels",
        "seed",
    ):
        assert forbidden not in serialized


def test_artifact_is_immutable() -> None:
    artifact = _tiny_artifact()

    with pytest.raises(FrozenInstanceError):
        artifact.sample_count = 3


def test_canonical_artifact_bytes_match_golden() -> None:
    assert canonical_artifact_bytes(_tiny_artifact()) == EXPECTED_CANONICAL_BYTES


def test_canonical_artifact_has_exactly_one_lf_and_no_bom_or_crlf() -> None:
    data = canonical_artifact_bytes(_tiny_artifact())

    assert data.endswith(b"\n")
    assert not data.endswith(b"\n\n")
    assert data.count(b"\n") == 1
    assert b"\r\n" not in data
    assert not data.startswith(b"\xef\xbb\xbf")


def test_artifact_sha256_matches_golden() -> None:
    assert artifact_sha256(_tiny_artifact()) == EXPECTED_ARTIFACT_SHA256


def test_canonical_generation_is_repeatable() -> None:
    assert canonical_artifact_bytes(_tiny_artifact()) == canonical_artifact_bytes(
        _tiny_artifact()
    )


def test_loader_accepts_canonical_artifact(tmp_path: Path) -> None:
    loaded = load_training_mean_artifact(
        _write_artifact_bytes(tmp_path, EXPECTED_CANONICAL_BYTES)
    )

    assert loaded == _tiny_artifact()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "missing_field",
        "bool_schema",
        "bool_sample_count",
        "bool_pixel_count",
        "bool_raw_sum",
        "wrong_schema",
        "wrong_dataset",
        "wrong_split",
        "wrong_accumulator",
        "wrong_algorithm",
        "short_digest",
        "uppercase_digest",
        "negative_raw_sum",
        "raw_sum_too_large",
        "mean_out_of_range",
        "mean_mismatch",
        "source_range",
        "normalized_range",
    ],
)
def test_loader_rejects_invalid_semantics(tmp_path: Path, mutation: str) -> None:
    mapping = _mapping_from_golden()
    if mutation == "extra_field":
        mapping["created_at_utc"] = "2026-08-04T00:00:00Z"
    elif mutation == "missing_field":
        del mapping["dataset_name"]
    elif mutation == "bool_schema":
        mapping["schema_version"] = True
    elif mutation == "bool_sample_count":
        mapping["sample_count"] = True
    elif mutation == "bool_pixel_count":
        mapping["pixel_count"] = True
    elif mutation == "bool_raw_sum":
        mapping["raw_pixel_sum"] = False
    elif mutation == "wrong_schema":
        mapping["schema_version"] = 2
    elif mutation == "wrong_dataset":
        mapping["dataset_name"] = "other"
    elif mutation == "wrong_split":
        mapping["split"] = "validation"
    elif mutation == "wrong_accumulator":
        mapping["accumulator_dtype"] = "float64"
    elif mutation == "wrong_algorithm":
        mapping["mean_algorithm_version"] = "training-mean-v2"
    elif mutation == "short_digest":
        mapping["dataset_sha256"] = "0" * 63
    elif mutation == "uppercase_digest":
        mapping["dataset_sha256"] = "A" * 64
    elif mutation == "negative_raw_sum":
        mapping["raw_pixel_sum"] = -1
    elif mutation == "raw_sum_too_large":
        mapping["raw_pixel_sum"] = 255 * 4608 + 1
    elif mutation == "mean_out_of_range":
        mapping["raw_training_mean"] = 1.01
    elif mutation == "mean_mismatch":
        mapping["raw_training_mean"] = 0.5
    elif mutation == "source_range":
        mapping["source_pixel_range"] = [1, 255]
    elif mutation == "normalized_range":
        mapping["normalized_pixel_range"] = [0, 1]

    with pytest.raises(TrainingMeanError):
        load_training_mean_artifact(
            _write_artifact_bytes(tmp_path, _canonical_test_bytes(mapping))
        )


@pytest.mark.parametrize(
    "data",
    [
        EXPECTED_CANONICAL_BYTES[:-1],
        EXPECTED_CANONICAL_BYTES + b"\n",
        EXPECTED_CANONICAL_BYTES.replace(b"\n", b"\r\n"),
        b"\xef\xbb\xbf" + EXPECTED_CANONICAL_BYTES,
        EXPECTED_CANONICAL_BYTES.replace(b'{"accumulator', b'{ "accumulator'),
    ],
)
def test_loader_rejects_noncanonical_bytes(tmp_path: Path, data: bytes) -> None:
    with pytest.raises(TrainingMeanError):
        load_training_mean_artifact(_write_artifact_bytes(tmp_path, data))


def test_loader_rejects_noncanonical_key_order(tmp_path: Path) -> None:
    mapping = _mapping_from_golden()
    reversed_mapping = dict(reversed(tuple(mapping.items())))
    data = json.dumps(
        reversed_mapping,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"

    with pytest.raises(TrainingMeanError):
        load_training_mean_artifact(_write_artifact_bytes(tmp_path, data))


def test_loader_rejects_nonfinite_mean(tmp_path: Path) -> None:
    data = EXPECTED_CANONICAL_BYTES.replace(
        b"0.25098039215686274",
        b"NaN",
    )

    with pytest.raises(TrainingMeanError):
        load_training_mean_artifact(_write_artifact_bytes(tmp_path, data))


def test_runtime_csv_binding_accepts_exact_file(tmp_path: Path) -> None:
    path = _write_combined_csv(tmp_path)

    assert validate_artifact_dataset(_tiny_artifact(), path) == EXPECTED_DATASET_SHA256


def test_runtime_csv_binding_rejects_changed_file(tmp_path: Path) -> None:
    path = _write_combined_csv(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(TrainingMeanError, match="dataset_sha256"):
        validate_artifact_dataset(_tiny_artifact(), path)


def test_writer_publishes_exact_canonical_bytes_and_hash(tmp_path: Path) -> None:
    target = tmp_path / "training_mean.json"

    result = write_training_mean_artifact(target, _tiny_artifact())

    assert result.path == target
    assert result.sha256 == EXPECTED_ARTIFACT_SHA256
    assert target.read_bytes() == EXPECTED_CANONICAL_BYTES


def test_writer_is_repeatable_without_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "training_mean.json"

    first = write_training_mean_artifact(target, _tiny_artifact())
    second = write_training_mean_artifact(target, _tiny_artifact())

    assert first == second
    assert target.read_bytes() == EXPECTED_CANONICAL_BYTES
    assert tuple(tmp_path.iterdir()) == (target,)


def test_writer_atomically_replaces_different_complete_artifact(
    tmp_path: Path,
) -> None:
    target = tmp_path / "training_mean.json"
    target.write_bytes(b"partial old content")
    changed = replace(_tiny_artifact(), dataset_sha256="0" * 64)

    result = write_training_mean_artifact(target, changed)

    assert target.read_bytes() == canonical_artifact_bytes(changed)
    assert result.sha256 == artifact_sha256(changed)
    assert tuple(tmp_path.iterdir()) == (target,)


def test_writer_rejects_missing_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "training_mean.json"

    with pytest.raises(FileNotFoundError):
        write_training_mean_artifact(target, _tiny_artifact())

    assert not target.exists()


def test_normalization_and_normalized_fill_are_equivalent() -> None:
    raw_mean = 0.5
    grayscale = torch.linspace(0.0, 1.0, 112 * 112).reshape(1, 112, 112)
    channel_means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    channel_stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    raw_filled = grayscale.clone()
    raw_filled[:, 10:30, 20:50] = raw_mean
    fill_then_normalize = (raw_filled.expand(3, -1, -1) - channel_means) / channel_stds

    normalize_then_fill = (
        grayscale.expand(3, -1, -1).clone() - channel_means
    ) / channel_stds
    normalized_fill = torch.tensor(normalized_fill_vector(raw_mean)).view(3, 1, 1)
    normalize_then_fill[:, 10:30, 20:50] = normalized_fill

    torch.testing.assert_close(
        normalize_then_fill,
        fill_then_normalize,
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        normalize_then_fill[:, 10:30, 20:50],
        normalized_fill.expand(3, 20, 30),
        rtol=0,
        atol=1e-6,
    )
