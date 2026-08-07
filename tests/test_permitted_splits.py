from __future__ import annotations

import csv
import json

import pytest

from occlusion_fer.permitted_splits import (
    PermittedSplitError,
    SplitRecord,
    load_stage_b_source,
    load_permitted_splits,
    make_permitted_splits,
    permitted_splits_to_data,
    stage_b_source_kind,
    validate_official_stage_b_sources,
)


def _record(sample_id: int, value: int) -> SplitRecord:
    return SplitRecord(sample_id, sample_id % 7, (value,) * (48 * 48))


def test_permitted_json_rejects_extra_split(tmp_path) -> None:
    artifact = tmp_path / "permitted.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "Training": [],
                "PublicTest": [],
                "PrivateTest": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermittedSplitError, match="only schema_version"):
        load_permitted_splits(artifact)


def test_combined_csv_routes_by_usage_without_parsing_private_fields(tmp_path) -> None:
    pixels = " ".join(["7"] * (48 * 48))
    source = tmp_path / "fer2013.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerow(["0", pixels, "Training"])
        writer.writerow(["NOT A LABEL", "DO NOT PARSE", "PrivateTest"])
        writer.writerow(["1", pixels, "PublicTest"])

    splits = load_stage_b_source(source)
    data = permitted_splits_to_data(splits)

    assert stage_b_source_kind(source) == "combined_csv_usage_routed"
    assert [record.sample_id for record in splits.training.records] == [2]
    assert [record.sample_id for record in splits.publictest.records] == [4]
    assert data.train_count == 1
    assert data.validation_count == 1
    assert data.test_count == 0
    assert all(record.split != "test" for record in data.records)


def test_combined_csv_rejects_unknown_usage_even_when_fields_are_excluded(
    tmp_path,
) -> None:
    source = tmp_path / "fer2013.csv"
    source.write_text(
        "emotion,pixels,Usage\nNOT A LABEL,DO NOT PARSE,UnknownTest\n",
        encoding="utf-8",
    )
    with pytest.raises(PermittedSplitError, match="Usage"):
        load_stage_b_source(source)


def test_canonical_identity_is_ordered_and_split_specific() -> None:
    first = make_permitted_splits(
        [_record(2, 2), _record(1, 1)],
        [_record(4, 4), _record(3, 3)],
    )
    second = make_permitted_splits(
        [_record(1, 1), _record(2, 2)],
        [_record(3, 3), _record(4, 4)],
    )
    assert first.training.dataset_sha256 == second.training.dataset_sha256
    assert first.publictest.dataset_sha256 == second.publictest.dataset_sha256
    assert first.training.dataset_sha256 != first.publictest.dataset_sha256


def test_split_sources_reject_overlapping_sample_ids() -> None:
    with pytest.raises(PermittedSplitError, match="disjoint"):
        make_permitted_splits([_record(1, 1)], [_record(1, 2)])


def test_formal_source_gate_rejects_nonofficial_counts() -> None:
    splits = make_permitted_splits([_record(1, 1)], [_record(2, 2)])
    with pytest.raises(PermittedSplitError, match="official Training sample count"):
        validate_official_stage_b_sources(splits)


def test_stage_b_source_path_rejects_ambiguous_suffix() -> None:
    with pytest.raises(PermittedSplitError, match=r"\.csv.*\.json"):
        stage_b_source_kind("/path/to/fer2013.txt")
