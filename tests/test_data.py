import csv
from pathlib import Path

import numpy as np
import pytest

from occlusion_fer.data import Fer2013DataError, load_fer2013_csv


PIXEL_COUNT = 48 * 48
VALID_PIXELS = " ".join(str(index % 256) for index in range(PIXEL_COUNT))


def make_row(
    *,
    emotion: object = 0,
    pixels: object = VALID_PIXELS,
    usage: object = "Training",
) -> dict[str, object]:
    return {"emotion": emotion, "pixels": pixels, "Usage": usage}


def valid_rows() -> list[dict[str, object]]:
    return [
        make_row(emotion=0, usage="Training"),
        make_row(emotion=3, usage="PublicTest"),
        make_row(emotion=6, usage="PrivateTest"),
    ]


def write_csv(
    tmp_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> Path:
    csv_path = tmp_path / "fer2013.csv"
    selected_fields = fieldnames or ["emotion", "pixels", "Usage"]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=selected_fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_loads_valid_csv_and_preserves_row_identity(tmp_path: Path) -> None:
    data = load_fer2013_csv(write_csv(tmp_path, valid_rows()))

    assert len(data.records) == 3
    assert [record.sample_id for record in data.records] == [2, 3, 4]
    assert data.records[0].label == 0
    assert data.records[0].label_name == "angry"
    assert data.records[0].image[0, 0] == 0
    assert data.records[0].image[47, 47] == 255


@pytest.mark.parametrize(
    ("usage", "expected_split"),
    [
        ("Training", "train"),
        ("PublicTest", "validation"),
        ("PrivateTest", "test"),
    ],
)
def test_maps_official_usage_to_split(
    tmp_path: Path, usage: str, expected_split: str
) -> None:
    data = load_fer2013_csv(write_csv(tmp_path, valid_rows()))
    record_by_usage = {
        "Training": data.records[0],
        "PublicTest": data.records[1],
        "PrivateTest": data.records[2],
    }

    assert record_by_usage[usage].split == expected_split


@pytest.mark.parametrize(
    ("label", "expected_name"),
    [
        (0, "angry"),
        (1, "disgust"),
        (2, "fear"),
        (3, "happy"),
        (4, "sad"),
        (5, "surprise"),
        (6, "neutral"),
    ],
)
def test_maps_each_numeric_label_to_its_name(
    tmp_path: Path, label: int, expected_name: str
) -> None:
    usages = ["Training", "PublicTest", "PrivateTest"]
    rows = [
        make_row(emotion=current_label, usage=usages[current_label % 3])
        for current_label in range(7)
    ]

    data = load_fer2013_csv(write_csv(tmp_path, rows))

    assert data.records[label].label == label
    assert data.records[label].label_name == expected_name


def test_image_has_fer2013_shape(tmp_path: Path) -> None:
    data = load_fer2013_csv(write_csv(tmp_path, valid_rows()))

    assert data.records[0].image.shape == (48, 48)


def test_image_uses_uint8_dtype(tmp_path: Path) -> None:
    data = load_fer2013_csv(write_csv(tmp_path, valid_rows()))

    assert data.records[0].image.dtype == np.uint8


def test_counts_records_in_each_split(tmp_path: Path) -> None:
    rows = valid_rows() + [make_row(emotion=1, usage="Training")]

    data = load_fer2013_csv(write_csv(tmp_path, rows))

    assert data.train_count == 2
    assert data.validation_count == 1
    assert data.test_count == 1


def test_counts_records_in_each_class(tmp_path: Path) -> None:
    rows = valid_rows() + [make_row(emotion=0, usage="Training")]

    data = load_fer2013_csv(write_csv(tmp_path, rows))

    assert data.class_counts == {0: 2, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 1}


def test_allows_extra_csv_columns(tmp_path: Path) -> None:
    rows = [dict(row, source="synthetic") for row in valid_rows()]

    data = load_fer2013_csv(
        write_csv(tmp_path, rows, ["emotion", "pixels", "Usage", "source"])
    )

    assert len(data.records) == 3


def test_reports_missing_csv_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="FER2013 CSV file not found"):
        load_fer2013_csv(missing_path)


def test_rejects_empty_file(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    with pytest.raises(Fer2013DataError, match="FER2013 CSV file is empty"):
        load_fer2013_csv(empty_path)


@pytest.mark.parametrize("missing_column", ["emotion", "pixels", "Usage"])
def test_rejects_missing_required_column(
    tmp_path: Path, missing_column: str
) -> None:
    fieldnames = [
        field for field in ["emotion", "pixels", "Usage"] if field != missing_column
    ]

    with pytest.raises(Fer2013DataError, match=missing_column):
        load_fer2013_csv(write_csv(tmp_path, valid_rows(), fieldnames))


def test_rejects_non_integer_emotion(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[0] = make_row(emotion="happy", usage="Training")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*emotion.*integer"):
        load_fer2013_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize("emotion", [-1, 7])
def test_rejects_emotion_outside_supported_range(
    tmp_path: Path, emotion: int
) -> None:
    rows = valid_rows()
    rows[0] = make_row(emotion=emotion, usage="Training")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*emotion.*0.*6"):
        load_fer2013_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize("pixel_count", [PIXEL_COUNT - 1, PIXEL_COUNT + 1])
def test_rejects_incorrect_pixel_count(tmp_path: Path, pixel_count: int) -> None:
    rows = valid_rows()
    rows[0] = make_row(pixels=" ".join(["0"] * pixel_count), usage="Training")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*pixels.*2304"):
        load_fer2013_csv(write_csv(tmp_path, rows))


def test_rejects_non_integer_pixel(tmp_path: Path) -> None:
    pixel_values = ["0"] * PIXEL_COUNT
    pixel_values[10] = "1.5"
    rows = valid_rows()
    rows[0] = make_row(pixels=" ".join(pixel_values), usage="Training")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*pixels.*integers"):
        load_fer2013_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize("invalid_pixel", [-1, 256])
def test_rejects_pixel_outside_uint8_range(
    tmp_path: Path, invalid_pixel: int
) -> None:
    pixel_values = ["0"] * PIXEL_COUNT
    pixel_values[10] = str(invalid_pixel)
    rows = valid_rows()
    rows[0] = make_row(pixels=" ".join(pixel_values), usage="Training")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*pixels.*0.*255"):
        load_fer2013_csv(write_csv(tmp_path, rows))


def test_rejects_unknown_usage(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[0] = make_row(usage="Development")

    with pytest.raises(Fer2013DataError, match=r"CSV row 2.*Usage.*Development"):
        load_fer2013_csv(write_csv(tmp_path, rows))


def test_error_reports_physical_csv_row_number(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[1] = make_row(emotion="not-an-integer", usage="PublicTest")

    with pytest.raises(Fer2013DataError, match=r"CSV row 3.*emotion"):
        load_fer2013_csv(write_csv(tmp_path, rows))


@pytest.mark.parametrize(
    ("missing_usage", "expected_split"),
    [
        ("Training", "train"),
        ("PublicTest", "validation"),
        ("PrivateTest", "test"),
    ],
)
def test_rejects_empty_official_split(
    tmp_path: Path, missing_usage: str, expected_split: str
) -> None:
    rows = [row for row in valid_rows() if row["Usage"] != missing_usage]

    with pytest.raises(Fer2013DataError, match=rf"{expected_split}.*empty"):
        load_fer2013_csv(write_csv(tmp_path, rows))


def test_loads_only_requested_splits_without_parsing_excluded_pixels(
    tmp_path: Path,
) -> None:
    rows = valid_rows()
    rows[-1]["pixels"] = "not parsed for excluded PrivateTest"

    data = load_fer2013_csv(
        write_csv(tmp_path, rows),
        include_splits=("train", "validation"),
    )

    assert {record.split for record in data.records} == {
        "train",
        "validation",
    }
    assert data.train_count == 1
    assert data.validation_count == 1
    assert data.test_count == 0


def test_requested_split_still_validates_its_pixels(tmp_path: Path) -> None:
    rows = valid_rows()
    rows[-1]["pixels"] = "invalid"

    with pytest.raises(Fer2013DataError, match=r"pixels.*2304"):
        load_fer2013_csv(
            write_csv(tmp_path, rows),
            include_splits=("test",),
        )


def test_missing_requested_split_reports_that_split_as_empty(
    tmp_path: Path,
) -> None:
    rows = [row for row in valid_rows() if row["Usage"] != "PrivateTest"]

    with pytest.raises(Fer2013DataError, match=r"split 'test' is empty"):
        load_fer2013_csv(
            write_csv(tmp_path, rows),
            include_splits=("test",),
        )


@pytest.mark.parametrize(
    "include_splits",
    [(), ("invalid",), ("train", "train"), "train"],
)
def test_rejects_invalid_requested_splits(
    tmp_path: Path,
    include_splits: object,
) -> None:
    with pytest.raises(Fer2013DataError, match=r"include_splits"):
        load_fer2013_csv(
            write_csv(tmp_path, valid_rows()),
            include_splits=include_splits,
        )
