"""Reproducible, create-only paper figure exports for FER2013."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from occlusion_fer.data import FER2013_LABEL_NAMES
from occlusion_fer.mask_manifest import (
    ManifestV2Row,
    load_manifest_v2,
)
from occlusion_fer.occlusion import (
    V2_MASKED_CONDITIONS,
    apply_evaluation_mask_v2,
    normalized_fill_vector_v2,
)
from occlusion_fer.permitted_splits import (
    STAGE_B_SOURCE_ROUTING_VERSION,
    PermittedSplits,
    SplitRecord,
    load_stage_b_source,
    permitted_splits_to_data,
    stage_b_source_kind,
    validate_official_stage_b_sources,
)
from occlusion_fer.torch_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Fer2013TorchDataset,
)
from occlusion_fer.training_mean import (
    load_training_mean_v2,
    training_mean_v2_sha256,
    validate_training_mean_v2,
)


CONDITION_ORDER = ("clean",) + tuple(V2_MASKED_CONDITIONS)
PROTOCOL = "occlusion-v2-224"
FIGURE_DPI = 300
DATASET_DISPLAY_SCALE = 12
OCCLUSION_DISPLAY_SCALE = 3

_SPLIT_TO_DATASET_SPLIT = {
    "Training": "train",
    "PublicTest": "validation",
}
_SHORT_CONDITION_LABELS = {
    "clean": "Clean",
    "upper_face_0.20": "Upper 20%",
    "upper_face_0.30": "Upper 30%",
    "upper_face_0.40": "Upper 40%",
    "lower_face_0.20": "Lower 20%",
    "lower_face_0.30": "Lower 30%",
    "lower_face_0.40": "Lower 40%",
    "random_rectangle_0.20": "Random 20%",
    "random_rectangle_0.30": "Random 30%",
    "random_rectangle_0.40": "Random 40%",
}


class PaperFigureError(ValueError):
    """Raised when a paper figure request is incomplete or unsafe."""


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest for one regular file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"file not found for SHA-256: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_split(split: str) -> str:
    """Accept only the two research-development splits permitted for figures."""
    if split == "PrivateTest" or split in {"test", "private_test"}:
        raise PaperFigureError("paper figure tools have no PrivateTest route")
    if split not in _SPLIT_TO_DATASET_SPLIT:
        raise PaperFigureError("split must be Training or PublicTest")
    return split


def export_fer2013_examples(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    split: str = "Training",
    samples_per_class: int = 1,
    require_official: bool = True,
    creation_command: Sequence[str] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Export deterministic raw grayscale examples for all seven classes."""
    split = validate_source_split(split)
    _require_positive_integer(samples_per_class, "samples_per_class")
    _require_bool(require_official, "require_official")
    source_path = Path(data_path).expanduser()
    source_kind = stage_b_source_kind(source_path)
    sources = _load_sources(source_path, require_official=require_official)
    source = sources.training if split == "Training" else sources.publictest
    selected_by_class = _select_examples(source.records, samples_per_class)
    commit, dirty = _git_identity(repository_root)

    selected: list[dict[str, object]] = []
    individual_images: dict[str, Image.Image] = {}
    montage_tiles: list[tuple[str, Image.Image]] = []
    for sample_index in range(samples_per_class):
        for label, label_name in enumerate(FER2013_LABEL_NAMES):
            record = selected_by_class[label][sample_index]
            filename = f"{label_name}_sample_{record.sample_id}.png"
            image = _raw_record_image(record).resize(
                (48 * DATASET_DISPLAY_SCALE, 48 * DATASET_DISPLAY_SCALE),
                resample=Image.Resampling.NEAREST,
            )
            individual_images[filename] = image
            montage_tiles.append((label_name, image))
    for label, label_name in enumerate(FER2013_LABEL_NAMES):
        for record in selected_by_class[label]:
            selected.append(
                {
                    "sample_id": record.sample_id,
                    "label": label,
                    "label_name": label_name,
                    "filename": f"{label_name}_sample_{record.sample_id}.png",
                }
            )

    montage = _montage(
        montage_tiles,
        columns=len(FER2013_LABEL_NAMES),
        tile_size=48 * DATASET_DISPLAY_SCALE,
        font_size=34,
    )
    output_path = _reserve_output_directory(output_directory)
    output_files: list[Path] = []
    for filename, image in individual_images.items():
        output_files.append(_save_png(image, output_path / filename))
    output_files.append(
        _save_png(montage, output_path / "fer2013_class_examples.png")
    )

    provenance: dict[str, object] = {
        "schema_version": 1,
        "figure_type": "fer2013_class_examples",
        "source_path": str(source_path.resolve()),
        "source_kind": source_kind,
        "source_routing_version": STAGE_B_SOURCE_ROUTING_VERSION,
        "source_split": split,
        "excluded_official_split": "PrivateTest",
        "training_dataset_sha256": sources.training.dataset_sha256,
        "publictest_dataset_sha256": sources.publictest.dataset_sha256,
        "selected_split_dataset_sha256": source.dataset_sha256,
        "class_order": list(FER2013_LABEL_NAMES),
        "samples_per_class": samples_per_class,
        "selection_rule": "sample_id_ascending_first_n_per_class",
        "selected_samples": selected,
        "source_image_size": [48, 48],
        "display_scaling_rule": {
            "method": "nearest_neighbor_integer_replication",
            "scale": DATASET_DISPLAY_SCALE,
            "content_adjustments": "none",
        },
        "png_dpi": FIGURE_DPI,
        "generation_commit": commit,
        "git_dirty": dirty,
        "creation_command": _creation_command(creation_command),
        "output_files_sha256": _output_hashes(output_path, output_files),
    }
    _write_json(output_path / "figure_provenance.json", provenance)
    return provenance


def build_formal_occlusion_tensors(
    clean_image: Tensor,
    *,
    sample_id: int,
    fill_vector: Sequence[float],
) -> tuple[dict[str, Tensor], tuple[dict[str, object], ...]]:
    """Apply the authoritative v2 evaluation masker to one cloned image."""
    if not isinstance(clean_image, Tensor):
        raise PaperFigureError("clean_image must be a torch.Tensor")
    if tuple(clean_image.shape) != (3, 224, 224):
        raise PaperFigureError("clean_image must have shape [3, 224, 224]")
    if type(sample_id) is not int or sample_id <= 0:
        raise PaperFigureError("sample_id must be a positive integer")
    tensors = {"clean": clean_image.clone()}
    metadata: list[dict[str, object]] = []
    for condition in V2_MASKED_CONDITIONS:
        masked, condition_metadata = apply_evaluation_mask_v2(
            clean_image,
            sample_id,
            condition,
            fill_vector,
        )
        tensors[condition] = masked
        metadata.append(asdict(condition_metadata))
    return tensors, tuple(metadata)


def export_occlusion_examples(
    data_path: str | Path,
    output_directory: str | Path,
    *,
    split: str,
    sample_id: int,
    training_mean_path: str | Path,
    manifest_path: str | Path | None = None,
    manifest_sidecar_path: str | Path | None = None,
    require_official: bool = True,
    creation_command: Sequence[str] | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Export clean plus nine masks using the exact formal v2 protocol."""
    split = validate_source_split(split)
    _require_positive_integer(sample_id, "sample_id")
    _require_bool(require_official, "require_official")
    source_path = Path(data_path).expanduser()
    source_kind = stage_b_source_kind(source_path)
    sources = _load_sources(source_path, require_official=require_official)
    selected_source = (
        sources.training if split == "Training" else sources.publictest
    )
    selected_record = _record_by_id(selected_source.records, sample_id, split)

    mean_path = Path(training_mean_path).expanduser()
    try:
        mean_artifact = load_training_mean_v2(mean_path)
        validate_training_mean_v2(
            mean_artifact,
            training_dataset_sha256=sources.training.dataset_sha256,
            consumer_image_size=224,
        )
        mean_sha256 = training_mean_v2_sha256(mean_artifact)
        fill_vector = normalized_fill_vector_v2(
            mean_artifact,
            training_dataset_sha256=sources.training.dataset_sha256,
        )
    except Exception as exc:
        raise PaperFigureError(f"Training mean v2 artifact is incompatible: {exc}") from exc

    manifest_rows: tuple[ManifestV2Row, ...] | None = None
    manifest_metadata: dict[str, object] = {
        "manifest_semantic_sha256": None,
        "manifest_file_sha256": None,
        "manifest_sidecar_sha256": None,
    }
    selected_manifest_rows: tuple[ManifestV2Row, ...] = ()
    if split == "PublicTest":
        if manifest_path is None:
            raise PaperFigureError("PublicTest figure export requires a v2 manifest")
        resolved_manifest = Path(manifest_path).expanduser()
        resolved_sidecar = (
            Path(manifest_sidecar_path).expanduser()
            if manifest_sidecar_path is not None
            else resolved_manifest.with_suffix(".json")
        )
        try:
            manifest_rows, envelope = load_manifest_v2(
                resolved_manifest,
                resolved_sidecar,
                publictest_dataset_sha256=sources.publictest.dataset_sha256,
                training_mean_artifact_sha256=mean_sha256,
                require_official=require_official,
            )
        except Exception as exc:
            raise PaperFigureError(f"PublicTest v2 manifest is incompatible: {exc}") from exc
        selected_manifest_rows = tuple(
            row for row in manifest_rows if row.sample_id == sample_id
        )
        if len(selected_manifest_rows) != len(V2_MASKED_CONDITIONS):
            raise PaperFigureError(
                "PublicTest manifest does not contain all nine rows for sample_id"
            )
        manifest_metadata = {
            "manifest_semantic_sha256": envelope.manifest_sha256,
            "manifest_file_sha256": sha256_file(resolved_manifest),
            "manifest_sidecar_sha256": sha256_file(resolved_sidecar),
        }

    clean_image, label = _model_domain_sample(sources, split, sample_id)
    if label != selected_record.label:
        raise PaperFigureError("selected sample label changed during preprocessing")
    formal_tensors, geometry = build_formal_occlusion_tensors(
        clean_image,
        sample_id=sample_id,
        fill_vector=fill_vector,
    )
    if selected_manifest_rows:
        _validate_metadata_against_manifest(geometry, selected_manifest_rows)

    display_images = {
        condition: _model_tensor_to_display_image(formal_tensors[condition]).resize(
            (224 * OCCLUSION_DISPLAY_SCALE, 224 * OCCLUSION_DISPLAY_SCALE),
            resample=Image.Resampling.NEAREST,
        )
        for condition in CONDITION_ORDER
    }
    montage = _montage(
        [
            (_SHORT_CONDITION_LABELS[condition], display_images[condition])
            for condition in CONDITION_ORDER
        ],
        columns=5,
        tile_size=224 * OCCLUSION_DISPLAY_SCALE,
        font_size=42,
    )
    commit, dirty = _git_identity(repository_root)
    output_path = _reserve_output_directory(output_directory)
    output_files: list[Path] = []
    for condition in CONDITION_ORDER:
        output_files.append(
            _save_png(display_images[condition], output_path / f"{condition}.png")
        )
    output_files.append(
        _save_png(montage, output_path / "occlusion_protocol_examples.png")
    )

    provenance: dict[str, object] = {
        "schema_version": 1,
        "figure_type": "occlusion_protocol_examples",
        "source_path": str(source_path.resolve()),
        "source_kind": source_kind,
        "source_routing_version": STAGE_B_SOURCE_ROUTING_VERSION,
        "official_split": split,
        "excluded_official_split": "PrivateTest",
        "sample_id": sample_id,
        "label": selected_record.label,
        "label_name": FER2013_LABEL_NAMES[selected_record.label],
        "protocol": PROTOCOL,
        "image_size": [224, 224],
        "condition_order": list(CONDITION_ORDER),
        "training_dataset_sha256": sources.training.dataset_sha256,
        "publictest_dataset_sha256": (
            sources.publictest.dataset_sha256 if split == "PublicTest" else None
        ),
        "training_mean_semantic_sha256": mean_sha256,
        "training_mean_file_sha256": sha256_file(mean_path),
        **manifest_metadata,
        "geometry_source": (
            "validated_publictest_manifest_v2_and_apply_evaluation_mask_v2"
            if split == "PublicTest"
            else "apply_evaluation_mask_v2"
        ),
        "geometry": list(geometry),
        "formal_mask_fill": {
            "source": "training_split_global_mean",
            "raw_training_mean": mean_artifact.raw_training_mean,
            "model_domain_normalized_fill": list(fill_vector),
        },
        "display_rendering_rule": {
            "sequence": [
                "formal_model_domain_mask",
                "inverse_imagenet_normalization",
                "clip_0_1",
                "round_to_uint8",
                "nearest_neighbor_integer_replication",
            ],
            "scale": OCCLUSION_DISPLAY_SCALE,
            "content_adjustments": "none",
        },
        "png_dpi": FIGURE_DPI,
        "generation_commit": commit,
        "git_dirty": dirty,
        "creation_command": _creation_command(creation_command),
        "output_files_sha256": _output_hashes(output_path, output_files),
    }
    _write_json(output_path / "figure_provenance.json", provenance)
    return provenance


def _load_sources(path: Path, *, require_official: bool) -> PermittedSplits:
    try:
        sources = load_stage_b_source(path)
        if require_official:
            validate_official_stage_b_sources(sources)
        return sources
    except Exception as exc:
        raise PaperFigureError(f"FER2013 figure source is invalid: {exc}") from exc


def _select_examples(
    records: Sequence[SplitRecord], samples_per_class: int
) -> dict[int, tuple[SplitRecord, ...]]:
    selected: dict[int, tuple[SplitRecord, ...]] = {}
    for label, label_name in enumerate(FER2013_LABEL_NAMES):
        candidates = sorted(
            (record for record in records if record.label == label),
            key=lambda record: record.sample_id,
        )
        if len(candidates) < samples_per_class:
            raise PaperFigureError(
                f"class {label_name!r} has fewer than {samples_per_class} sample(s)"
            )
        selected[label] = tuple(candidates[:samples_per_class])
    return selected


def _record_by_id(
    records: Sequence[SplitRecord], sample_id: int, split: str
) -> SplitRecord:
    matches = [record for record in records if record.sample_id == sample_id]
    if len(matches) != 1:
        raise PaperFigureError(
            f"sample_id {sample_id} does not identify exactly one {split} sample"
        )
    return matches[0]


def _raw_record_image(record: SplitRecord) -> Image.Image:
    array = np.asarray(record.pixels, dtype=np.uint8).reshape((48, 48))
    return Image.fromarray(array)


def _model_domain_sample(
    sources: PermittedSplits, split: str, sample_id: int
) -> tuple[Tensor, int]:
    data = permitted_splits_to_data(sources)
    dataset = Fer2013TorchDataset(
        data,
        split=_SPLIT_TO_DATASET_SPLIT[split],
        image_size=224,
        normalize_imagenet=True,
    )
    for index in range(len(dataset)):
        image, label, candidate_id = dataset[index]
        if candidate_id == sample_id:
            return image, label
    raise PaperFigureError(f"sample_id {sample_id} disappeared during preprocessing")


def _validate_metadata_against_manifest(
    geometry: Sequence[Mapping[str, object]],
    rows: Sequence[ManifestV2Row],
) -> None:
    by_condition = {row.condition: row for row in rows}
    fields = (
        "sample_id",
        "condition",
        "occlusion_type",
        "ratio_token",
        "actual_ratio",
        "top",
        "left",
        "height",
        "width",
        "masked_pixel_count",
        "total_pixel_count",
        "image_height",
        "image_width",
    )
    for metadata in geometry:
        condition = metadata["condition"]
        row = by_condition.get(condition)
        if row is None:
            raise PaperFigureError(f"manifest row missing for condition {condition}")
        for field in fields:
            if metadata[field] != getattr(row, field):
                raise PaperFigureError(
                    f"formal mask metadata does not match manifest field {field}"
                )


def _model_tensor_to_display_image(tensor: Tensor) -> Image.Image:
    if not isinstance(tensor, Tensor) or tuple(tensor.shape) != (3, 224, 224):
        raise PaperFigureError("display tensor must have shape [3, 224, 224]")
    values = tensor.detach().to(device="cpu", dtype=torch.float32)
    mean = values.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = values.new_tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (values * std + mean).clamp(0.0, 1.0)
    array = (
        (rgb.permute(1, 2, 0) * 255.0)
        .round()
        .to(dtype=torch.uint8)
        .numpy()
    )
    return Image.fromarray(array)


def _montage(
    tiles: Sequence[tuple[str, Image.Image]],
    *,
    columns: int,
    tile_size: int,
    font_size: int,
) -> Image.Image:
    if not tiles:
        raise PaperFigureError("montage requires at least one tile")
    _require_positive_integer(columns, "columns")
    rows = (len(tiles) + columns - 1) // columns
    margin = 48
    gap = 24
    label_height = max(64, font_size + 24)
    width = 2 * margin + columns * tile_size + (columns - 1) * gap
    height = 2 * margin + rows * (tile_size + label_height) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=font_size)
    for index, (label, image) in enumerate(tiles):
        row, column = divmod(index, columns)
        left = margin + column * (tile_size + gap)
        top = margin + row * (tile_size + label_height + gap)
        rendered = image.convert("RGB")
        if rendered.size != (tile_size, tile_size):
            raise PaperFigureError("montage tile dimensions are inconsistent")
        canvas.paste(rendered, (left, top))
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_left = left + (tile_size - text_width) // 2
        draw.text(
            (text_left, top + tile_size + 12),
            label,
            fill="black",
            font=font,
        )
    return canvas


def _save_png(image: Image.Image, path: Path) -> Path:
    image.save(
        path,
        format="PNG",
        dpi=(FIGURE_DPI, FIGURE_DPI),
        compress_level=9,
        optimize=False,
    )
    with Image.open(path) as reopened:
        reopened.verify()
    return path


def _reserve_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser()
    if output.exists():
        raise FileExistsError(f"paper figure output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _output_hashes(output: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(paths, key=lambda item: item.name)
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    path.write_text(data, encoding="utf-8")


def _git_identity(repository_root: str | Path | None) -> tuple[str, bool]:
    start = Path.cwd() if repository_root is None else Path(repository_root)
    try:
        root_result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=True,
        )
        root = Path(root_result.stdout.strip())
        commit_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PaperFigureError("figure generation requires a readable Git identity") from exc
    commit = commit_result.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise PaperFigureError("Git HEAD is not a full lowercase commit SHA")
    return commit, bool(status_result.stdout)


def _creation_command(value: Sequence[str] | None) -> list[str]:
    command = list(sys.argv if value is None else value)
    if any(type(item) is not str for item in command):
        raise PaperFigureError("creation_command must contain only strings")
    return command


def _require_positive_integer(value: object, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise PaperFigureError(f"{field_name} must be a positive integer")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise PaperFigureError(f"{field_name} must be a bool")
