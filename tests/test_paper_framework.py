"""Tests for the overall experimental framework figure."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from PIL import Image
import pytest

from occlusion_fer.paper_framework import (
    FIGURE_STEM,
    FrameworkFigureError,
    generate_framework,
)


EXPECTED_VISIBLE_TEXT = (
    "Overall experimental framework",
    "Current formal paper lineage: 224 x 224",
    "FER2013",
    "Official labels",
    "Official splits",
    "Training",
    "PublicTest",
    "PrivateTest: reserved",
    "Pre-processing",
    "Resize: 224 x 224",
    "Locked occlusion protocol",
    "upper_face",
    "lower_face",
    "random_rectangle",
    "Experiment 1 | Clean baseline",
    "Experiment 2 | Occlusion analysis",
    "Reuse Experiment 1 Clean-only best checkpoints",
    "Experiment 3 | Mixed training",
    "Clean-only vs Mixed training comparison",
    "Grad-CAM qualitative analysis",
    "Accuracy",
    "Macro-F1",
)


def test_framework_exports_editable_vectors_and_300_dpi_png(tmp_path: Path) -> None:
    paths = generate_framework(tmp_path)
    assert [path.name for path in paths] == [
        f"{FIGURE_STEM}.pdf",
        f"{FIGURE_STEM}.svg",
        f"{FIGURE_STEM}.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)

    pdf_path, svg_path, png_path = paths
    assert pdf_path.read_bytes().startswith(b"%PDF")
    svg_tree = ElementTree.parse(svg_path)
    svg_text = svg_path.read_text(encoding="utf-8")
    visible_text = " ".join(
        " ".join("".join(element.itertext()).split())
        for element in svg_tree.iter()
        if element.tag.endswith("text")
    )
    for expected in EXPECTED_VISIBLE_TEXT:
        assert expected in visible_text
    assert "112" not in visible_text
    assert "PrivateTest: final" not in visible_text
    assert "<text" in svg_text

    with Image.open(png_path) as image:
        assert image.format == "PNG"
        assert image.width >= 3500
        assert image.height >= 1800
        assert image.width / image.height >= 1.7
        dpi = image.info["dpi"]
        assert dpi[0] == pytest.approx(300, abs=0.01)
        assert dpi[1] == pytest.approx(300, abs=0.01)


def test_framework_rejects_non_directory_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(FrameworkFigureError, match="not a directory"):
        generate_framework(output_path)
