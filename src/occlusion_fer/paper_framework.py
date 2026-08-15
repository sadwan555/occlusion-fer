"""Generate the formal landscape experimental framework diagram."""

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


FIGURE_STEM = "overall_experimental_framework_formal"
FIGURE_FORMATS = ("pdf", "svg", "png")
FIGURE_SIZE = (12.0, 6.2)

INK = "#20272D"
MUTED = "#596872"
NAVY = "#284D68"
GREEN = "#476A57"
RUST = "#8A5943"
LINE = "#40515E"
PALE_BLUE = "#EAF1F5"
PALE_GREEN = "#EDF3EF"
PALE_RUST = "#F5EFEC"
PALE_GRAY = "#F3F5F6"
WHITE = "#FFFFFF"


class FrameworkFigureError(ValueError):
    """Raised when the framework figure cannot be generated safely."""


def _box(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    body: str,
    accent: str = NAVY,
    facecolor: str = WHITE,
    title_size: float = 9.0,
    body_size: float = 7.5,
) -> None:
    axes.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=accent,
            linewidth=1.0,
            joinstyle="miter",
            zorder=2,
        )
    )
    header_height = min(0.052, height * 0.28)
    axes.add_patch(
        Rectangle(
            (x, y + height - header_height),
            width,
            header_height,
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
            zorder=3,
        )
    )
    axes.text(
        x + width / 2,
        y + height - header_height / 2,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=WHITE,
        zorder=4,
    )
    axes.text(
        x + width / 2,
        y + (height - header_height) / 2,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=INK,
        linespacing=1.26,
        zorder=4,
    )


def _arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    label_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color=LINE,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.0",
            zorder=5,
        )
    )
    if label:
        axes.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=MUTED,
            backgroundcolor=WHITE,
            zorder=6,
        )


def _paper_style_path() -> Path:
    style_path = Path(__file__).with_name("paper.mplstyle")
    if not style_path.is_file():
        raise FrameworkFigureError(f"paper style not found: {style_path}")
    return style_path


def build_framework_figure() -> Figure:
    """Build the framework for the verified 224 x 224 Stage 8 lineage."""
    with plt.style.context(_paper_style_path()):
        figure = plt.figure(figsize=FIGURE_SIZE)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(0.0, 1.0)
        axes.axis("off")

        axes.text(
            0.03,
            0.955,
            "Overall experimental framework",
            fontsize=14,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
        )
        axes.text(
            0.97,
            0.955,
            "Current formal paper lineage: 224 x 224 | PublicTest evidence | PrivateTest reserved",
            fontsize=7.6,
            color=MUTED,
            ha="right",
            va="center",
        )

        _box(
            axes,
            0.03,
            0.705,
            0.16,
            0.175,
            title="FER2013",
            body="Official labels\n7 expression classes",
            facecolor=PALE_BLUE,
        )
        _box(
            axes,
            0.215,
            0.705,
            0.24,
            0.175,
            title="Official splits",
            body=(
                "Training: model fitting\n"
                "PublicTest: validation, checkpoint selection, reported evaluation\n"
                "PrivateTest: reserved"
            ),
            facecolor=PALE_GRAY,
            body_size=7.0,
        )
        _box(
            axes,
            0.48,
            0.705,
            0.20,
            0.175,
            title="Pre-processing",
            body="48 x 48 grayscale -> 3 channels\nResize: 224 x 224\nImageNet normalization",
            facecolor=PALE_BLUE,
            body_size=7.2,
        )
        _box(
            axes,
            0.705,
            0.705,
            0.265,
            0.175,
            title="Locked occlusion protocol",
            body=(
                "occlusion-v2-224 | Training pixel mean fill\n"
                "upper_face | lower_face | random_rectangle\n"
                "Target ratios: 0.20 | 0.30 | 0.40"
            ),
            facecolor=PALE_GRAY,
            body_size=7.1,
        )
        _arrow(axes, (0.19, 0.792), (0.215, 0.792))
        _arrow(axes, (0.455, 0.792), (0.48, 0.792))
        _arrow(axes, (0.68, 0.792), (0.705, 0.792))

        _box(
            axes,
            0.03,
            0.375,
            0.27,
            0.235,
            title="Experiment 1 | Clean baseline",
            body=(
                "Train Clean-only ResNet-18\n"
                "Seeds 42, 123, 2026 | 50 epochs\n"
                "Best checkpoint: clean PublicTest Macro-F1"
            ),
            accent=NAVY,
            facecolor=PALE_BLUE,
            body_size=8.0,
        )
        _box(
            axes,
            0.365,
            0.375,
            0.27,
            0.235,
            title="Experiment 2 | Occlusion analysis",
            body=(
                "Reuse Experiment 1 Clean-only best checkpoints\n"
                "Evaluate clean + 9 occlusion conditions\n"
                "Location and severity analysis"
            ),
            accent=GREEN,
            facecolor=PALE_GREEN,
            body_size=7.8,
        )
        _box(
            axes,
            0.70,
            0.375,
            0.27,
            0.235,
            title="Experiment 3 | Mixed training",
            body=(
                "Train new Mixed models with the same ResNet-18\n"
                "Same seeds and training budget\n"
                "Evaluate the same 10 conditions and masks"
            ),
            accent=RUST,
            facecolor=PALE_RUST,
            body_size=7.8,
        )
        _arrow(axes, (0.165, 0.705), (0.165, 0.610))
        _arrow(
            axes,
            (0.30, 0.492),
            (0.365, 0.492),
            label="same checkpoints",
            label_offset=(0.0, 0.025),
        )
        _arrow(axes, (0.58, 0.705), (0.835, 0.610), label="shared pre-processing")
        _arrow(axes, (0.837, 0.705), (0.835, 0.610))

        _box(
            axes,
            0.20,
            0.105,
            0.50,
            0.16,
            title="Clean-only vs Mixed training comparison",
            body=(
                "Same clean + 9 occlusion conditions and deterministic masks\n"
                "Accuracy | Macro-F1 | per-class metrics | confusion matrices | clean-to-occluded drop"
            ),
            accent=LINE,
            facecolor=PALE_GRAY,
            body_size=7.4,
        )
        _box(
            axes,
            0.755,
            0.105,
            0.215,
            0.16,
            title="Grad-CAM qualitative analysis",
            body="Seed 42 | PublicTest\nSelected clean and occluded conditions",
            accent=NAVY,
            facecolor=PALE_BLUE,
            title_size=8.3,
            body_size=7.3,
        )
        _arrow(axes, (0.50, 0.375), (0.43, 0.265))
        _arrow(axes, (0.835, 0.375), (0.60, 0.265), label="paired conditions")
        _arrow(axes, (0.70, 0.185), (0.755, 0.185))
        return figure


def generate_framework(output_dir: str | Path) -> tuple[Path, ...]:
    """Export the formal framework as PDF, editable SVG, and 300-dpi PNG."""
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
                elif extension == "pdf":
                    save_options["metadata"] = {
                        "Title": FIGURE_STEM,
                        "Creator": "occlusion_fer.paper_framework",
                        "CreationDate": None,
                        "ModDate": None,
                    }
                figure.savefig(path, **save_options)
                paths.append(path)
        finally:
            plt.close(figure)
    return tuple(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the verified formal experimental framework."
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in generate_framework(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
