"""FER2013 CSV loading and validation."""

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
PIXEL_COUNT = IMAGE_HEIGHT * IMAGE_WIDTH
REQUIRED_COLUMNS = ("emotion", "pixels", "Usage")

FER2013_LABEL_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

Fer2013Split = Literal["train", "validation", "test"]
USAGE_TO_SPLIT: dict[str, Fer2013Split] = {
    "Training": "train",
    "PublicTest": "validation",
    "PrivateTest": "test",
}


class Fer2013DataError(ValueError):
    """Raised when FER2013 CSV content is invalid."""


@dataclass(frozen=True)
class Fer2013Record:
    """One validated FER2013 sample."""

    sample_id: int
    label: int
    label_name: str
    split: Fer2013Split
    image: NDArray[np.uint8]


@dataclass(frozen=True)
class Fer2013Data:
    """Validated FER2013 records and summary counts."""

    records: tuple[Fer2013Record, ...]
    train_count: int
    validation_count: int
    test_count: int
    class_counts: dict[int, int]


def load_fer2013_csv(path: str | Path) -> Fer2013Data:
    """Load and validate all records from a FER2013 CSV file."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"FER2013 CSV file not found: {csv_path}")

    records: list[Fer2013Record] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    class_counts = {label: 0 for label in range(len(FER2013_LABEL_NAMES))}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file, strict=True)
            header = _read_header(reader, csv_path)
            column_indices = _required_column_indices(header)

            for row in reader:
                row_number = reader.line_num
                record = _parse_record(row, row_number, column_indices)
                records.append(record)
                split_counts[record.split] += 1
                class_counts[record.label] += 1
    except csv.Error as exc:
        line_number = reader.line_num or 1
        raise Fer2013DataError(
            f"CSV row {line_number}: malformed CSV content: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise Fer2013DataError(
            f"FER2013 CSV file must be UTF-8 text: {csv_path}"
        ) from exc

    if not records:
        raise Fer2013DataError(
            f"FER2013 CSV file is empty: it contains no data rows: {csv_path}"
        )

    for split, count in split_counts.items():
        if count == 0:
            raise Fer2013DataError(f"FER2013 split '{split}' is empty")

    return Fer2013Data(
        records=tuple(records),
        train_count=split_counts["train"],
        validation_count=split_counts["validation"],
        test_count=split_counts["test"],
        class_counts=class_counts,
    )


def _read_header(reader: Iterator[list[str]], csv_path: Path) -> list[str]:
    try:
        return next(reader)
    except StopIteration as exc:
        raise Fer2013DataError(f"FER2013 CSV file is empty: {csv_path}") from exc


def _required_column_indices(header: list[str]) -> dict[str, int]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise Fer2013DataError(f"Missing required CSV column(s): {missing}")
    return {column: header.index(column) for column in REQUIRED_COLUMNS}


def _parse_record(
    row: list[str], row_number: int, column_indices: dict[str, int]
) -> Fer2013Record:
    if not row:
        raise Fer2013DataError(f"CSV row {row_number}: row must not be empty")

    emotion = _row_value(row, row_number, "emotion", column_indices)
    pixels = _row_value(row, row_number, "pixels", column_indices)
    usage = _row_value(row, row_number, "Usage", column_indices)

    label = _parse_label(emotion, row_number)
    split = _parse_split(usage, row_number)
    image = _parse_image(pixels, row_number)

    return Fer2013Record(
        sample_id=row_number,
        label=label,
        label_name=FER2013_LABEL_NAMES[label],
        split=split,
        image=image,
    )


def _row_value(
    row: list[str],
    row_number: int,
    field: str,
    column_indices: dict[str, int],
) -> str:
    index = column_indices[field]
    if index >= len(row):
        raise Fer2013DataError(
            f"CSV row {row_number}: field '{field}' is missing from the row"
        )
    return row[index]


def _parse_label(value: str, row_number: int) -> int:
    try:
        label = int(value)
    except ValueError as exc:
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'emotion' must be an integer; got {value!r}"
        ) from exc

    if not 0 <= label < len(FER2013_LABEL_NAMES):
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'emotion' must be between 0 and 6; "
            f"got {label}"
        )
    return label


def _parse_split(value: str, row_number: int) -> Fer2013Split:
    try:
        return USAGE_TO_SPLIT[value]
    except KeyError as exc:
        allowed = ", ".join(USAGE_TO_SPLIT)
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'Usage' must be one of {allowed}; "
            f"got {value!r}"
        ) from exc


def _parse_image(value: str, row_number: int) -> NDArray[np.uint8]:
    pixel_tokens = value.split()
    if len(pixel_tokens) != PIXEL_COUNT:
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'pixels' must contain exactly "
            f"{PIXEL_COUNT} integers; got {len(pixel_tokens)}"
        )

    try:
        pixel_values = np.asarray(pixel_tokens, dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'pixels' must contain only integers"
        ) from exc

    if np.any((pixel_values < 0) | (pixel_values > 255)):
        raise Fer2013DataError(
            f"CSV row {row_number}: field 'pixels' values must be between 0 and 255"
        )

    return pixel_values.astype(np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
