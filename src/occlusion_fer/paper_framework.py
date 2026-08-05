"""Generate the landscape overall experimental framework diagram."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "occlusion-fer-matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle


FIGURE_STEM = "overall_experimental_framework_landscape"
FIGURE_FORMATS = ("pdf", "svg", "png")
FIGURE_SIZE = (12.0, 6.8)

INK = "#20272D"
MUTED = "#596872"
NAVY = "#284D68"
LINE = "#40515E"
PALE_BLUE = "#EAF1F5"
PALE_GRAY = "#F3F5F6"
SUBBOX_EDGE = "#AEBAC2"
WHITE = "#FFFFFF"


class FrameworkFigureError(ValueError):
    """Raised when the framework figure cannot be generated safely."""


def _rectangle(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str = LINE,
    linewidth: float = 0.9,
    zorder: int = 2,
) -> None:
    axes.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            joinstyle="miter",
            zorder=zorder,
        )
    )


def _label(
    axes: Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    weight: str = "normal",
    color: str = INK,
    horizontalalignment: str = "center",
    verticalalignment: str = "center",
    linespacing: float = 1.12,
) -> None:
    axes.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=horizontalalignment,
        va=verticalalignment,
        linespacing=linespacing,
        zorder=5,
    )


def _arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    mutation_scale: float = 10.0,
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=1.05,
            color=LINE,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def _group(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> None:
    header_height = 0.085
    _rectangle(
        axes,
        x,
        y,
        width,
        height,
        facecolor=PALE_GRAY,
        edgecolor=NAVY,
        linewidth=1.0,
    )
    _rectangle(
        axes,
        x,
        y + height - header_height,
        width,
        header_height,
        facecolor=NAVY,
        edgecolor=NAVY,
        linewidth=0.0,
        zorder=3,
    )
    _label(
        axes,
        x + width / 2,
        y + height - header_height / 2,
        title,
        size=11.3,
        weight="bold",
        color=WHITE,
    )


def _inner_module(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    header_height: float = 0.055,
    title_size: float = 8.8,
    body_color: str = WHITE,
) -> None:
    _rectangle(
        axes,
        x,
        y,
        width,
        height,
        facecolor=body_color,
        edgecolor=SUBBOX_EDGE,
        linewidth=0.8,
        zorder=3,
    )
    _rectangle(
        axes,
        x,
        y + height - header_height,
        width,
        header_height,
        facecolor=PALE_BLUE,
        edgecolor=SUBBOX_EDGE,
        linewidth=0.8,
        zorder=4,
    )
    _label(
        axes,
        x + width / 2,
        y + height - header_height / 2,
        title,
        size=title_size,
        weight="bold",
        color=NAVY,
    )


def _content_box(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str = "",
    *,
    title_size: float = 8.8,
    detail_size: float = 7.2,
) -> None:
    _rectangle(
        axes,
        x,
        y,
        width,
        height,
        facecolor=WHITE,
        edgecolor=SUBBOX_EDGE,
        linewidth=0.7,
        zorder=4,
    )
    title_y = y + height * (0.62 if detail else 0.50)
    _label(axes, x + width / 2, title_y, title, size=title_size, weight="bold")
    if detail:
        _label(
            axes,
            x + width / 2,
            y + height * 0.27,
            detail,
            size=detail_size,
            color=MUTED,
        )


def _draw_data_group(axes: Axes) -> None:
    _group(axes, 0.015, 0.120, 0.350, 0.760, "Data Preparation")

    _inner_module(axes, 0.030, 0.335, 0.080, 0.300, "Dataset")
    _label(axes, 0.070, 0.455, "FER2013\nDataset", size=10.0, weight="bold")

    _inner_module(
        axes,
        0.125,
        0.215,
        0.105,
        0.540,
        "Official Data Splits",
        title_size=8.1,
    )
    split_rows = (
        (0.585, "Training", "model fitting"),
        (0.425, "PublicTest", "validation /\ncheckpoint selection"),
        (0.265, "PrivateTest", "final evaluation"),
    )
    for y, name, role in split_rows:
        _content_box(
            axes,
            0.134,
            y,
            0.087,
            0.115,
            name,
            role,
            title_size=8.4,
            detail_size=6.7,
        )

    _inner_module(
        axes,
        0.245,
        0.175,
        0.105,
        0.620,
        "Image Pre-processing",
        title_size=8.1,
        body_color=PALE_BLUE,
    )
    preprocessing_steps = (
        (0.630, "Grayscale image"),
        (0.505, "Resize to\n112\u00d7112"),
        (0.380, "Replicate to\n3 channels"),
        (0.255, "ImageNet\nnormalization"),
    )
    for y, step in preprocessing_steps:
        _content_box(
            axes,
            0.254,
            y,
            0.087,
            0.085,
            step,
            title_size=7.8,
        )
    for start_y, end_y in ((0.630, 0.590), (0.505, 0.465), (0.380, 0.340)):
        _arrow(axes, (0.2975, start_y), (0.2975, end_y), mutation_scale=6.5)

    _arrow(axes, (0.110, 0.485), (0.125, 0.485), mutation_scale=8.0)
    _arrow(axes, (0.230, 0.485), (0.245, 0.485), mutation_scale=8.0)


def _draw_training_group(axes: Axes) -> None:
    _group(axes, 0.385, 0.120, 0.245, 0.760, "Model Training")
    _label(axes, 0.5075, 0.745, "Training Strategies", size=9.5, weight="bold")

    _content_box(
        axes,
        0.403,
        0.535,
        0.095,
        0.150,
        "Clean-only\nBaseline",
        title_size=8.9,
    )
    _content_box(
        axes,
        0.517,
        0.535,
        0.095,
        0.150,
        "Mixed Clean/\nOccluded Training",
        title_size=8.0,
    )

    _inner_module(
        axes,
        0.425,
        0.245,
        0.165,
        0.200,
        "Shared Backbone Model",
        header_height=0.060,
        title_size=8.6,
        body_color=PALE_BLUE,
    )
    _label(
        axes,
        0.5075,
        0.345,
        "ImageNet-pretrained\nResNet-18",
        size=8.6,
        weight="bold",
    )
    _label(
        axes,
        0.5075,
        0.275,
        "same architecture,\ntrained separately",
        size=6.6,
        color=MUTED,
    )

    _arrow(axes, (0.4505, 0.535), (0.472, 0.445), mutation_scale=8.0)
    _arrow(axes, (0.5645, 0.535), (0.543, 0.445), mutation_scale=8.0)


def _draw_evaluation_group(axes: Axes) -> None:
    _group(axes, 0.650, 0.120, 0.335, 0.760, "Evaluation")

    _inner_module(
        axes,
        0.665,
        0.405,
        0.305,
        0.350,
        "Evaluation Conditions (10 total)",
        header_height=0.060,
        title_size=9.4,
    )
    cell_positions = (0.674, 0.748, 0.822, 0.896)
    condition_names = (
        "Clean",
        "Upper-face\nOcclusion",
        "Lower-face\nOcclusion",
        "Random\nRectangular\nOcclusion",
    )
    condition_settings = (
        "1 condition",
        "20% / 30%\n/ 40%",
        "20% / 30%\n/ 40%",
        "20% / 30%\n/ 40%",
    )
    for x, name, setting in zip(cell_positions, condition_names, condition_settings):
        _content_box(
            axes,
            x,
            0.440,
            0.065,
            0.220,
            name,
            setting,
            title_size=7.2,
            detail_size=6.2,
        )

    _inner_module(
        axes,
        0.690,
        0.185,
        0.255,
        0.145,
        "Evaluation Outputs",
        header_height=0.050,
        title_size=8.7,
        body_color=PALE_BLUE,
    )
    metric_centers = (0.722, 0.786, 0.850, 0.914)
    metrics = ("Accuracy", "Macro-F1", "Per-class\nMetrics", "Confusion\nMatrix")
    for center, metric in zip(metric_centers, metrics):
        _label(axes, center, 0.232, metric, size=7.4, weight="bold")
    for divider in (0.754, 0.818, 0.882):
        axes.plot(
            [divider, divider],
            [0.197, 0.272],
            color=SUBBOX_EDGE,
            linewidth=0.6,
            zorder=4,
        )

    _arrow(axes, (0.8175, 0.405), (0.8175, 0.330), mutation_scale=8.0)


def _paper_style_path() -> Path:
    style_path = Path(__file__).with_name("paper.mplstyle")
    if not style_path.is_file():
        raise FrameworkFigureError(f"paper style not found: {style_path}")
    return style_path


def build_framework_figure() -> Figure:
    """Build the landscape framework without reading experiment results."""
    with plt.style.context(_paper_style_path()):
        figure = plt.figure(figsize=FIGURE_SIZE)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.axis("off")

        _draw_data_group(axes)
        _draw_training_group(axes)
        _draw_evaluation_group(axes)
        _arrow(axes, (0.365, 0.500), (0.385, 0.500))
        _arrow(axes, (0.630, 0.500), (0.650, 0.500))
        return figure


def generate_framework(output_dir: str | Path) -> tuple[Path, ...]:
    """Export the landscape framework as PDF, editable SVG, and 300-dpi PNG."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise FrameworkFigureError(f"output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    with plt.style.context(_paper_style_path()):
        figure = build_framework_figure()
        try:
            for extension in FIGURE_FORMATS:
                path = destination / f"{FIGURE_STEM}.{extension}"
                save_options: dict[str, object] = {
                    "format": extension,
                    "bbox_inches": "tight",
                    "pad_inches": 0.04,
                }
                if extension == "png":
                    save_options["dpi"] = 300
                figure.savefig(path, **save_options)
                paths.append(path)
        finally:
            plt.close(figure)
    return tuple(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the landscape overall experimental framework."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Independent directory for the PDF, SVG, and PNG outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = generate_framework(args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
