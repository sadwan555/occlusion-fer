from __future__ import annotations

import json

import pytest

from occlusion_fer.permitted_splits import (
    PermittedSplitError,
    SplitRecord,
    load_permitted_splits,
    make_permitted_splits,
    reject_combined_dataset_path,
)


def _record(sample_id: int, value: int) -> SplitRecord:
    return SplitRecord(sample_id, sample_id % 7, (value,) * (48 * 48))


def test_rejects_combined_source_and_extra_split(tmp_path) -> None:
    with pytest.raises(PermittedSplitError, match="before opening"):
        reject_combined_dataset_path(
            tmp_path / "does-not-exist.csv",
            occlusion_enabled=True,
        )
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


def test_occlusion_entrypoints_fail_on_combined_dataset_path(tmp_path) -> None:
    source = tmp_path / "combined.csv"
    source.write_text("PrivateTest must never be parsed\n", encoding="utf-8")
    before = source.stat().st_atime_ns
    with pytest.raises(PermittedSplitError, match="before opening"):
        reject_combined_dataset_path(source, occlusion_enabled=True)
    assert source.stat().st_atime_ns == before
