"""Strict split-source identities for Stage B occlusion artifacts.

This module deliberately does not accept a combined FER2013 CSV for Stage B.
The JSON artifact format is small, deterministic, and suitable for synthetic
fixtures or a separately prepared server-side permitted-splits export.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from occlusion_fer.data import (
    FER2013_LABEL_NAMES,
    Fer2013Data,
    Fer2013Record,
)


class PermittedSplitError(ValueError):
    """Raised when a split source violates the Stage B contract."""


@dataclass(frozen=True)
class SplitRecord:
    sample_id: int
    label: int
    pixels: tuple[int, ...]


@dataclass(frozen=True)
class SplitSource:
    split: str
    records: tuple[SplitRecord, ...]
    dataset_sha256: str

    @property
    def count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class PermittedSplits:
    training: SplitSource
    publictest: SplitSource


def canonical_record_bytes(records: Sequence[SplitRecord]) -> bytes:
    """Return ordered compact JSON records with one final LF per record."""
    validated = _validate_records(records)
    lines = [
        json.dumps(
            {"sample_id": r.sample_id, "label": r.label, "pixels": list(r.pixels)},
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        for r in sorted(validated, key=lambda item: item.sample_id)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def dataset_sha256(records: Sequence[SplitRecord]) -> str:
    return hashlib.sha256(canonical_record_bytes(records)).hexdigest()


def make_split_source(split: str, records: Sequence[SplitRecord]) -> SplitSource:
    _require_split(split)
    validated = tuple(_validate_records(records))
    if not validated:
        raise PermittedSplitError(f"{split} source must not be empty")
    return SplitSource(split, validated, dataset_sha256(validated))


def make_permitted_splits(
    training: Sequence[SplitRecord], publictest: Sequence[SplitRecord]
) -> PermittedSplits:
    training_source = make_split_source("Training", training)
    publictest_source = make_split_source("PublicTest", publictest)
    overlap = {
        record.sample_id for record in training_source.records
    }.intersection(record.sample_id for record in publictest_source.records)
    if overlap:
        raise PermittedSplitError(
            "Training and PublicTest sources must be disjoint; overlapping sample IDs: "
            + ", ".join(str(sample_id) for sample_id in sorted(overlap))
        )
    return PermittedSplits(training_source, publictest_source)


def load_permitted_splits(path: str | Path) -> PermittedSplits:
    """Load only the explicit Training/PublicTest artifact schema."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"permitted-splits artifact not found: {source_path}")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermittedSplitError("permitted-splits artifact must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PermittedSplitError("permitted-splits artifact must be a mapping")
    if set(payload) != {"schema_version", "Training", "PublicTest"}:
        raise PermittedSplitError(
            "permitted-splits artifact must contain only schema_version, Training, PublicTest"
        )
    if payload["schema_version"] != 1:
        raise PermittedSplitError("permitted-splits schema_version must be 1")
    training = _records_from_json(payload["Training"], "Training")
    publictest = _records_from_json(payload["PublicTest"], "PublicTest")
    return make_permitted_splits(training, publictest)


def permitted_splits_to_data(splits: PermittedSplits) -> Fer2013Data:
    """Convert validated split records into the existing tensor-pipeline data model."""
    if not isinstance(splits, PermittedSplits):
        raise PermittedSplitError("splits must be a validated PermittedSplits object")
    records: list[Fer2013Record] = []
    class_counts = {label: 0 for label in range(len(FER2013_LABEL_NAMES))}
    for source, split in (
        (splits.training, "train"),
        (splits.publictest, "validation"),
    ):
        for record in source.records:
            image = np.asarray(record.pixels, dtype=np.uint8).reshape((48, 48))
            records.append(
                Fer2013Record(
                    sample_id=record.sample_id,
                    label=record.label,
                    label_name=FER2013_LABEL_NAMES[record.label],
                    split=split,
                    image=image,
                )
            )
            class_counts[record.label] += 1
    return Fer2013Data(
        records=tuple(records),
        train_count=splits.training.count,
        validation_count=splits.publictest.count,
        test_count=0,
        class_counts=class_counts,
    )


def is_permitted_splits_artifact_path(path: str | Path) -> bool:
    """Recognize the explicit JSON source form without opening or hashing it."""
    return Path(path).expanduser().suffix.lower() == ".json"


def reject_combined_dataset_path(
    dataset_path: str | Path, *, occlusion_enabled: bool
) -> None:
    """Reject legacy combined paths before any file open when enabled.

    A permitted-splits JSON is the only path-shaped Stage B source currently
    accepted by the command-line training/preflight routes. The caller still
    validates and loads that artifact separately; this guard only prevents a
    combined CSV from being opened or hashed by an occlusion-enabled route.
    """
    if not occlusion_enabled:
        return
    raw_path = str(dataset_path)
    if not raw_path.strip():
        raise PermittedSplitError("occlusion-enabled runs require explicit split sources")
    if is_permitted_splits_artifact_path(raw_path):
        return
    raise PermittedSplitError(
        "occlusion-enabled Stage B commands reject dataset.path combined CSV before opening; "
        "provide Training-only and PublicTest-only sources or a permitted-splits artifact"
    )


def _records_from_json(value: object, split: str) -> tuple[SplitRecord, ...]:
    if not isinstance(value, list):
        raise PermittedSplitError(f"{split} source must be a list of records")
    records: list[SplitRecord] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != {"sample_id", "label", "pixels"}:
            raise PermittedSplitError(f"{split} record {index} has invalid fields")
        pixels = item["pixels"]
        if not isinstance(pixels, list):
            raise PermittedSplitError(f"{split} record {index} pixels must be a list")
        records.append(SplitRecord(item["sample_id"], item["label"], tuple(pixels)))
    return tuple(records)


def _validate_records(records: Sequence[SplitRecord]) -> tuple[SplitRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise PermittedSplitError("records must be a sequence of SplitRecord")
    validated: list[SplitRecord] = []
    seen: set[int] = set()
    for record in records:
        if not isinstance(record, SplitRecord):
            raise PermittedSplitError("records must contain SplitRecord values")
        if type(record.sample_id) is not int or record.sample_id <= 0:
            raise PermittedSplitError("sample_id must be a positive integer")
        if record.sample_id in seen:
            raise PermittedSplitError(f"duplicate sample_id: {record.sample_id}")
        seen.add(record.sample_id)
        if type(record.label) is not int or not 0 <= record.label <= 6:
            raise PermittedSplitError("label must be an integer from 0 through 6")
        if len(record.pixels) != 48 * 48:
            raise PermittedSplitError("pixels must contain exactly 2304 values")
        if any(type(pixel) is not int or not 0 <= pixel <= 255 for pixel in record.pixels):
            raise PermittedSplitError("pixels must contain uint8 integer values")
        validated.append(record)
    return tuple(validated)


def _require_split(split: str) -> None:
    if split not in {"Training", "PublicTest"}:
        raise PermittedSplitError("only Training and PublicTest sources are permitted")
