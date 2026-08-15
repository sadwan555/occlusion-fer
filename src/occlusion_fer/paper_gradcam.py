"""Assemble provenance-locked formal Grad-CAM overlays into a paper panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


FORMAL_IMAGE_SIZE = 224
FORMAL_PROTOCOL = "occlusion-v2-224"
FORMAL_SPLIT = "validation"
FORMAL_OFFICIAL_SPLIT = "PublicTest"
FORMAL_TARGET_LAYER = "model.layer4[-1]"
FORMAL_TARGET_POLICY = "ground_truth"
FORMAL_SAMPLE_ORDER = (
    (0, "angry", 28711),
    (1, "disgust", 28712),
    (2, "fear", 28717),
    (3, "happy", 28715),
    (4, "sad", 28713),
    (5, "surprise", 28727),
    (6, "neutral", 28714),
)
FORMAL_MODEL_ORDER = ("clean", "mixed")
FORMAL_CONDITION_SET = {
    "clean",
    "upper_face_0.40",
    "lower_face_0.40",
    "random_rectangle_0.40",
}
FULL_FIGURE_SIZE = (7.20, 7.15)
PNG_DPI = 300
FULL_FIGURE_STEM = "gradcam_full_comparison"
FIGURE_MANIFEST_NAME = "gradcam_figure_manifest.json"
FIGURE_AUDIT_NAME = "gradcam_figure_audit.md"
COMPACT_SELECTION_RULE = (
    "not generated: the complete 7x8 panel remains legible at standard "
    "double-column width, so predefined option 1 retains all samples"
)


class GradcamFigureError(ValueError):
    """Raised when formal Grad-CAM figure evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class GradcamCell:
    sample_id: int
    label: int
    label_name: str
    model_role: str
    condition: str
    overlay_path: Path
    metadata_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    target_class: int
    target_class_name: str


@dataclass(frozen=True)
class GradcamPanelEvidence:
    source_manifest: Path
    source_manifest_sha256: str
    samples: tuple[tuple[int, str, int], ...]
    model_order: tuple[str, ...]
    condition_order: tuple[str, ...]
    cells: tuple[GradcamCell, ...]
    checkpoint_sha256: Mapping[str, str]

    @property
    def combination_order(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (role, condition)
            for role in self.model_order
            for condition in self.condition_order
        )

    def cell_index(self) -> dict[tuple[int, str, str], GradcamCell]:
        return {
            (cell.sample_id, cell.model_role, cell.condition): cell
            for cell in self.cells
        }


def _fail(message: str) -> None:
    raise GradcamFigureError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{description} must be readable UTF-8 JSON: {path}")
    if not isinstance(payload, Mapping):
        _fail(f"{description} must contain a mapping: {path}")
    return payload


def _require_exact_manifest_identity(manifest: Mapping[str, Any]) -> None:
    expected_scalars = {
        "status": "completed",
        "generated_image_count": 56,
        "generated_metadata_count": 56,
        "split": FORMAL_SPLIT,
        "official_split": FORMAL_OFFICIAL_SPLIT,
        "private_test_materialized": False,
        "image_size": FORMAL_IMAGE_SIZE,
        "occlusion_protocol": FORMAL_PROTOCOL,
        "target_layer": FORMAL_TARGET_LAYER,
        "target_class_policy": FORMAL_TARGET_POLICY,
    }
    for field, expected in expected_scalars.items():
        if manifest.get(field) != expected:
            _fail(
                f"sample manifest {field} mismatch: expected {expected!r}, "
                f"got {manifest.get(field)!r}"
            )
    if manifest.get("input_tensor_shape") != [1, 3, 224, 224]:
        _fail("sample manifest input tensor shape must be [1, 3, 224, 224]")
    if manifest.get("skipped_samples") != []:
        _fail("sample manifest must record zero skipped samples")


def _parse_samples(manifest: Mapping[str, Any]) -> tuple[tuple[int, str, int], ...]:
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        _fail("sample manifest samples must be a list")
    parsed: list[tuple[int, str, int]] = []
    for item in raw_samples:
        if not isinstance(item, Mapping):
            _fail("sample manifest contains an invalid sample")
        if item.get("split") != FORMAL_SPLIT or item.get("official_split") != FORMAL_OFFICIAL_SPLIT:
            _fail("sample manifest contains a non-PublicTest sample")
        label = item.get("label")
        label_name = item.get("label_name")
        sample_id = item.get("sample_id")
        if type(label) is not int or type(label_name) is not str or type(sample_id) is not int:
            _fail("sample manifest sample identity is invalid")
        parsed.append((label, label_name, sample_id))
    if tuple(parsed) != FORMAL_SAMPLE_ORDER:
        _fail(
            "sample manifest does not match the locked canonical samples: "
            f"{tuple(parsed)!r}"
        )
    selected_ids = manifest.get("selected_sample_ids")
    if selected_ids != [sample_id for _, _, sample_id in parsed]:
        _fail("sample manifest selected_sample_ids does not match its sample rows")
    return tuple(parsed)


def _parse_orders(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(FORMAL_MODEL_ORDER):
        _fail("sample manifest must contain clean and mixed checkpoints")
    for role in FORMAL_MODEL_ORDER:
        checkpoint = checkpoints.get(role)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("model_role") != role:
            _fail(f"sample manifest {role} checkpoint identity is invalid")
        digest = checkpoint.get("sha256")
        path = checkpoint.get("path")
        if type(digest) is not str or len(digest) != 64 or type(path) is not str:
            _fail(f"sample manifest {role} checkpoint provenance is invalid")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 4:
        _fail("sample manifest must contain four formal Grad-CAM conditions")
    if len(set(conditions)) != 4 or set(conditions) != FORMAL_CONDITION_SET:
        _fail("sample manifest Grad-CAM conditions do not match the locked formal set")
    if conditions[0] != "clean":
        _fail("sample manifest must place the clean input before masked inputs")
    return FORMAL_MODEL_ORDER, tuple(conditions), checkpoints


def _validate_image(path: Path, root: Path) -> None:
    if not path.is_absolute() or root not in path.parents:
        _fail(f"overlay path must resolve inside the formal output root: {path}")
    if "obsolete" in path.parts:
        _fail(f"obsolete Grad-CAM output is forbidden: {path}")
    if not path.is_file() or path.suffix.lower() != ".png":
        _fail(f"overlay PNG not found: {path}")
    try:
        with Image.open(path) as image:
            if image.size != (FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE):
                _fail(f"overlay must be 224x224: {path}")
            if image.mode != "RGB":
                _fail(f"overlay must use RGB pixels without conversion: {path}")
            image.verify()
    except (OSError, SyntaxError) as exc:
        _fail(f"overlay PNG is not readable: {path}: {exc}")


def _require_metadata_identity(
    item: Mapping[str, Any],
    *,
    expected_sample: tuple[int, str, int],
    checkpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    label, label_name, sample_id = expected_sample
    role = item.get("model_role")
    condition = item.get("condition")
    if role not in FORMAL_MODEL_ORDER or condition not in FORMAL_CONDITION_SET:
        _fail("metadata contains an unknown model-condition combination")
    expected = {
        "sample_id": sample_id,
        "ground_truth_label": label,
        "ground_truth_label_name": label_name,
        "split": FORMAL_SPLIT,
        "official_split": FORMAL_OFFICIAL_SPLIT,
        "input_image_size": FORMAL_IMAGE_SIZE,
        "input_tensor_shape": [1, 3, 224, 224],
        "occlusion_protocol": FORMAL_PROTOCOL,
        "target_layer": FORMAL_TARGET_LAYER,
        "target_class": label,
        "target_class_name": label_name,
        "checkpoint_path": checkpoints[role]["path"],
        "checkpoint_sha256": checkpoints[role]["sha256"],
    }
    for field, value in expected.items():
        if item.get(field) != value:
            _fail(
                f"metadata field {field} mismatch for sample {sample_id}, "
                f"{role}, {condition}"
            )
    mask = item.get("mask")
    if condition == "clean":
        if mask is not None:
            _fail("clean-input metadata must not contain mask geometry")
    elif not isinstance(mask, Mapping) or (
        mask.get("algorithm_version") != FORMAL_PROTOCOL
        or mask.get("context") != FORMAL_SPLIT
        or mask.get("image_height") != FORMAL_IMAGE_SIZE
        or mask.get("image_width") != FORMAL_IMAGE_SIZE
    ):
        _fail("masked metadata is missing formal v2 geometry identity")


def load_gradcam_panel_evidence(
    sample_manifest_path: str | Path,
) -> GradcamPanelEvidence:
    """Load all 56 cells from metadata and fail closed on any provenance drift."""
    manifest_path = Path(sample_manifest_path).expanduser().resolve()
    if "obsolete" in manifest_path.parts:
        _fail("obsolete Grad-CAM manifests cannot be used for paper figures")
    root = manifest_path.parent
    manifest = _load_json(manifest_path, "formal Grad-CAM sample manifest")
    _require_exact_manifest_identity(manifest)
    samples = _parse_samples(manifest)
    model_order, condition_order, checkpoints = _parse_orders(manifest)
    samples_by_id = {sample_id: sample for sample in samples for sample_id in (sample[2],)}

    metadata_paths = tuple(sorted(root.rglob("metadata_*.json")))
    if len(metadata_paths) != 56:
        _fail(f"expected 56 metadata files, found {len(metadata_paths)}")
    cells: list[GradcamCell] = []
    keys: set[tuple[int, str, str]] = set()
    overlays: set[Path] = set()
    for metadata_path in metadata_paths:
        if "obsolete" in metadata_path.parts:
            _fail(f"obsolete metadata is forbidden: {metadata_path}")
        item = _load_json(metadata_path, "Grad-CAM per-image metadata")
        sample_id = item.get("sample_id")
        if type(sample_id) is not int or sample_id not in samples_by_id:
            _fail(f"metadata contains an unselected sample_id: {sample_id!r}")
        _require_metadata_identity(
            item,
            expected_sample=samples_by_id[sample_id],
            checkpoints=checkpoints,
        )
        role = str(item["model_role"])
        condition = str(item["condition"])
        key = (sample_id, role, condition)
        if key in keys:
            _fail(f"duplicate Grad-CAM metadata combination: {key}")
        keys.add(key)
        overlay_value = item.get("overlay_path")
        if type(overlay_value) is not str:
            _fail(f"metadata overlay_path is invalid: {metadata_path}")
        overlay_path = Path(overlay_value).expanduser().resolve()
        _validate_image(overlay_path, root)
        if overlay_path in overlays:
            _fail(f"multiple metadata files reference the same overlay: {overlay_path}")
        overlays.add(overlay_path)
        checkpoint_path = Path(str(item["checkpoint_path"])).expanduser().resolve()
        cells.append(
            GradcamCell(
                sample_id=sample_id,
                label=int(item["ground_truth_label"]),
                label_name=str(item["ground_truth_label_name"]),
                model_role=role,
                condition=condition,
                overlay_path=overlay_path,
                metadata_path=metadata_path.resolve(),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=str(item["checkpoint_sha256"]),
                target_class=int(item["target_class"]),
                target_class_name=str(item["target_class_name"]),
            )
        )
    expected_keys = {
        (sample_id, role, condition)
        for _, _, sample_id in samples
        for role in model_order
        for condition in condition_order
    }
    if keys != expected_keys:
        _fail(
            "Grad-CAM coverage mismatch: "
            f"missing={sorted(expected_keys - keys)!r}, extra={sorted(keys - expected_keys)!r}"
        )
    all_pngs = {path.resolve() for path in root.rglob("*.png")}
    if all_pngs != overlays:
        _fail(
            "formal output PNGs and metadata overlay references are not one-to-one: "
            f"unreferenced={sorted(all_pngs-overlays)!r}, missing={sorted(overlays-all_pngs)!r}"
        )
    ordered_cells = tuple(
        next(cell for cell in cells if (cell.sample_id, cell.model_role, cell.condition) == key)
        for _, _, sample_id in samples
        for key in (
            (sample_id, role, condition)
            for role in model_order
            for condition in condition_order
        )
    )
    return GradcamPanelEvidence(
        source_manifest=manifest_path,
        source_manifest_sha256=_sha256(manifest_path),
        samples=samples,
        model_order=model_order,
        condition_order=condition_order,
        cells=ordered_cells,
        checkpoint_sha256={
            role: str(checkpoints[role]["sha256"]) for role in model_order
        },
    )


def _condition_label(condition: str) -> str:
    if condition == "clean":
        return "Clean\ninput"
    occlusion_type, ratio_token = condition.rsplit("_", 1)
    labels = {
        "upper_face": "Upper face",
        "lower_face": "Lower face",
        "random_rectangle": "Random rect.",
    }
    if occlusion_type not in labels:
        _fail(f"cannot create a short label for condition: {condition}")
    return f"{labels[occlusion_type]}\n{float(ratio_token):.0%}"


def _model_label(role: str) -> str:
    labels = {"clean": "Clean-only model", "mixed": "Mixed-training model"}
    if role not in labels:
        _fail(f"cannot create a model label for role: {role}")
    return labels[role]


def _publish_figure_create_only(
    figure: plt.Figure,
    output_dir: Path,
) -> tuple[Path, Path]:
    targets = (
        output_dir / f"{FULL_FIGURE_STEM}.pdf",
        output_dir / f"{FULL_FIGURE_STEM}.png",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        _fail("figure assembly will not overwrite: " + ", ".join(map(str, existing)))
    temporary_paths: list[Path] = []
    created: list[Path] = []
    try:
        for target in targets:
            with tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix=f".{target.stem}.",
                suffix=target.suffix,
                delete=False,
            ) as temporary_file:
                temporary = Path(temporary_file.name)
            temporary_paths.append(temporary)
            figure.savefig(
                temporary,
                format=target.suffix.lstrip("."),
                dpi=PNG_DPI,
                facecolor="white",
                bbox_inches=None,
                pad_inches=0,
            )
        for temporary, target in zip(temporary_paths, targets, strict=True):
            os.link(temporary, target)
            created.append(target)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        plt.close(figure)
    return targets


def render_full_panel(
    evidence: GradcamPanelEvidence,
    output_dir: str | Path,
    *,
    style_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Assemble unchanged source overlays in the manifest-derived 7x8 order."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    if not output_path.is_dir():
        _fail(f"paper figure output is not a directory: {output_path}")
    selected_style = (
        Path(style_path).expanduser().resolve()
        if style_path is not None
        else Path(__file__).with_name("paper.mplstyle").resolve()
    )
    if not selected_style.is_file():
        raise FileNotFoundError(f"paper style not found: {selected_style}")
    index = evidence.cell_index()
    with plt.style.context(str(selected_style)), matplotlib.rc_context(
        {"savefig.bbox": None, "figure.facecolor": "white"}
    ):
        figure, axes = plt.subplots(
            len(evidence.samples),
            len(evidence.combination_order),
            figsize=FULL_FIGURE_SIZE,
            squeeze=False,
        )
        figure.subplots_adjust(
            left=0.105,
            right=0.995,
            bottom=0.018,
            top=0.885,
            wspace=0.025,
            hspace=0.030,
        )
        for row_index, (_, label_name, sample_id) in enumerate(evidence.samples):
            for column_index, (role, condition) in enumerate(evidence.combination_order):
                axis = axes[row_index, column_index]
                cell = index[(sample_id, role, condition)]
                with Image.open(cell.overlay_path) as image:
                    pixels = np.asarray(image)
                axis.imshow(pixels, interpolation="none", aspect="equal")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_facecolor("white")
                for spine in axis.spines.values():
                    spine.set_visible(True)
                    spine.set_color("0.72")
                    spine.set_linewidth(0.35)
                if row_index == 0:
                    axis.set_title(_condition_label(condition), fontsize=7.0, pad=3.0)
                if column_index == 0:
                    axis.set_ylabel(
                        label_name.capitalize(),
                        rotation=0,
                        ha="right",
                        va="center",
                        fontsize=7.2,
                        labelpad=25,
                    )
        group_size = len(evidence.condition_order)
        for group_index, role in enumerate(evidence.model_order):
            left_axis = axes[0, group_index * group_size]
            right_axis = axes[0, (group_index + 1) * group_size - 1]
            left = left_axis.get_position().x0
            right = right_axis.get_position().x1
            figure.text(
                (left + right) / 2,
                0.970,
                _model_label(role),
                ha="center",
                va="center",
                fontsize=9.0,
                fontweight="semibold",
            )
            figure.add_artist(
                Line2D(
                    [left, right],
                    [0.945, 0.945],
                    transform=figure.transFigure,
                    color="0.35",
                    linewidth=0.55,
                )
            )
        boundary_left = axes[0, group_size - 1].get_position().x1
        boundary_right = axes[0, group_size].get_position().x0
        figure.add_artist(
            Line2D(
                [(boundary_left + boundary_right) / 2] * 2,
                [0.018, 0.945],
                transform=figure.transFigure,
                color="0.55",
                linewidth=0.55,
            )
        )
        return _publish_figure_create_only(figure, output_path)


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        _fail(f"figure assembly will not overwrite: {path}")
    data = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_text_create_only(path: Path, text: str) -> Path:
    if path.exists():
        _fail(f"figure assembly will not overwrite: {path}")
    data = text.encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _manifest_payload(
    evidence: GradcamPanelEvidence,
    figure_paths: Sequence[Path],
    *,
    style_path: Path,
    command: str,
) -> dict[str, Any]:
    source_cells = [
        {
            "sample_id": cell.sample_id,
            "class_label": cell.label,
            "class_name": cell.label_name,
            "model": cell.model_role,
            "condition": cell.condition,
            "source_overlay": str(cell.overlay_path),
            "source_overlay_sha256": _sha256(cell.overlay_path),
            "source_metadata": str(cell.metadata_path),
            "source_metadata_sha256": _sha256(cell.metadata_path),
            "checkpoint_path": str(cell.checkpoint_path),
            "checkpoint_sha256": cell.checkpoint_sha256,
            "image_size": [FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE],
            "split": FORMAL_SPLIT,
            "official_split": FORMAL_OFFICIAL_SPLIT,
            "protocol": FORMAL_PROTOCOL,
            "gradcam_target_layer": FORMAL_TARGET_LAYER,
            "target_class": cell.target_class,
            "target_class_name": cell.target_class_name,
        }
        for cell in evidence.cells
    ]
    return {
        "manifest_schema_version": 1,
        "assembly_timestamp": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "figure_filename": [path.name for path in figure_paths],
        "figures": [
            {
                "filename": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "dpi": PNG_DPI if path.suffix == ".png" else None,
                "figure_size_inches": list(FULL_FIGURE_SIZE),
                "cell_count": 56,
            }
            for path in figure_paths
        ],
        "compact_figure": {
            "generated": False,
            "selection_rule": COMPACT_SELECTION_RULE,
            "selected_sample_ids": [sample_id for _, _, sample_id in evidence.samples],
            "labels": [name for _, name, _ in evidence.samples],
            "source_files": [],
        },
        "source_sample_manifest": str(evidence.source_manifest),
        "source_sample_manifest_sha256": evidence.source_manifest_sha256,
        "selected_sample_ids": [sample_id for _, _, sample_id in evidence.samples],
        "labels": [name for _, name, _ in evidence.samples],
        "row_order": [name for _, name, _ in evidence.samples],
        "column_order": [
            {"model": role, "condition": condition}
            for role, condition in evidence.combination_order
        ],
        "included_model_condition_combinations": [
            {"model": role, "condition": condition}
            for role, condition in evidence.combination_order
        ],
        "checkpoint_sha256": dict(evidence.checkpoint_sha256),
        "image_size": [FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE],
        "split": FORMAL_SPLIT,
        "official_split": FORMAL_OFFICIAL_SPLIT,
        "protocol": FORMAL_PROTOCOL,
        "gradcam_target_layer": FORMAL_TARGET_LAYER,
        "target_class_rule": FORMAL_TARGET_POLICY,
        "source_cells": source_cells,
        "obsolete_outputs_excluded": True,
        "colorbar_included": False,
        "colorbar_reason": (
            "source CAMs are normalized per image; no shared absolute scale is implied"
        ),
        "assembly": {
            "source_overlays_modified": False,
            "cam_recomputed": False,
            "heatmap_reapplied": False,
            "cropping": False,
            "interpolation": "none",
        },
        "generator": {
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": _sha256(Path(__file__).resolve()),
            "style_path": str(style_path),
            "style_sha256": _sha256(style_path),
            "python_version": platform.python_version(),
            "matplotlib_version": matplotlib.__version__,
        },
    }


def _audit_markdown(
    evidence: GradcamPanelEvidence,
    figure_paths: Sequence[Path],
) -> str:
    lines = [
        "# Grad-CAM Figure Audit",
        "",
        f"- Source manifest: `{evidence.source_manifest}`",
        "- Formal overlays found: 56/56",
        "- Formal metadata files found: 56/56",
        "- Duplicate model-condition-sample combinations: 0",
        "- Obsolete 112x112 outputs included: NO",
        "- Image dimensions: all 224x224 RGB",
        "- Locked sample IDs matched exactly: 28711, 28712, 28717, 28715, 28713, 28727, 28714",
        "- Every panel cell traceable to one metadata file and one formal overlay: YES",
        "- Source overlays modified or recomputed: NO",
        "",
        "## Figure Usage",
        "",
        "Full figure files:",
        "",
        *[f"- `{path}`" for path in figure_paths],
        "",
        f"Compact figure: not generated. {COMPACT_SELECTION_RULE.capitalize()}.",
        "",
        "## Full Figure Sources",
        "",
        "| Sample | Class | Model | Condition | Overlay | Metadata |",
        "|---:|---|---|---|---|---|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            (
                str(cell.sample_id),
                cell.label_name,
                cell.model_role,
                cell.condition,
                f"`{cell.overlay_path}`",
                f"`{cell.metadata_path}`",
            )
        )
        + " |"
        for cell in evidence.cells
    )
    lines.extend(("", "## Compact Figure Sources", "", "None; compact figure was not generated.", ""))
    return "\n".join(lines)


def generate_gradcam_paper_figure(
    *,
    sample_manifest_path: str | Path,
    output_dir: str | Path,
    style_path: str | Path | None = None,
    argv: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify formal evidence, assemble the full panel, and write provenance."""
    evidence = load_gradcam_panel_evidence(sample_manifest_path)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    selected_style = (
        Path(style_path).expanduser().resolve()
        if style_path is not None
        else Path(__file__).with_name("paper.mplstyle").resolve()
    )
    reserved = (
        output_path / f"{FULL_FIGURE_STEM}.pdf",
        output_path / f"{FULL_FIGURE_STEM}.png",
        output_path / FIGURE_MANIFEST_NAME,
        output_path / FIGURE_AUDIT_NAME,
    )
    existing = [path for path in reserved if path.exists()]
    if existing:
        _fail("figure assembly will not overwrite: " + ", ".join(map(str, existing)))
    figure_paths = render_full_panel(evidence, output_path, style_path=selected_style)
    command = shlex.join([sys.executable, "-m", "occlusion_fer.paper_gradcam", *argv])
    python_path = os.environ.get("PYTHONPATH")
    if python_path:
        command = f"PYTHONPATH={shlex.quote(python_path)} {command}"
    manifest = _manifest_payload(
        evidence,
        figure_paths,
        style_path=selected_style,
        command=command,
    )
    _write_json_create_only(output_path / FIGURE_MANIFEST_NAME, manifest)
    _write_text_create_only(
        output_path / FIGURE_AUDIT_NAME,
        _audit_markdown(evidence, figure_paths),
    )
    print("formal_overlays=56/56")
    print("formal_metadata=56/56")
    print("duplicates=0")
    print("obsolete_outputs_included=false")
    print("compact_figure_generated=false")
    print(f"full_figure_pdf={figure_paths[0]}")
    print(f"full_figure_png={figure_paths[1]}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the formal 224/v2 Grad-CAM paper comparison panel.",
        allow_abbrev=False,
    )
    parser.add_argument("--sample-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--style")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    generate_gradcam_paper_figure(
        sample_manifest_path=args.sample_manifest,
        output_dir=args.output_dir,
        style_path=args.style,
        argv=raw_argv,
    )


if __name__ == "__main__":
    main()
