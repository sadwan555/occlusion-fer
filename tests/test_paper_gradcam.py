from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from occlusion_fer.paper_gradcam import (
    FORMAL_CONDITION_SET,
    FORMAL_SAMPLE_ORDER,
    FULL_FIGURE_SIZE,
    GradcamFigureError,
    load_gradcam_panel_evidence,
    render_full_panel,
)


CONDITIONS = (
    "clean",
    "upper_face_0.40",
    "lower_face_0.40",
    "random_rectangle_0.40",
)
ROLES = ("clean", "mixed")
HASHES = {"clean": "a" * 64, "mixed": "b" * 64}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _formal_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "gradcam_224_v2"
    samples = [
        {
            "label": label,
            "label_name": name,
            "sample_id": sample_id,
            "split": "validation",
            "official_split": "PublicTest",
        }
        for label, name, sample_id in FORMAL_SAMPLE_ORDER
    ]
    manifest = {
        "status": "completed",
        "generated_image_count": 56,
        "generated_metadata_count": 56,
        "split": "validation",
        "official_split": "PublicTest",
        "private_test_materialized": False,
        "image_size": 224,
        "input_tensor_shape": [1, 3, 224, 224],
        "occlusion_protocol": "occlusion-v2-224",
        "target_layer": "model.layer4[-1]",
        "target_class_policy": "ground_truth",
        "skipped_samples": [],
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "samples": samples,
        "conditions": list(CONDITIONS),
        "checkpoints": {
            role: {
                "model_role": role,
                "path": f"/checkpoints/{role}.pt",
                "sha256": HASHES[role],
            }
            for role in ROLES
        },
    }
    manifest_path = root / "sample_manifest.json"
    _write_json(manifest_path, manifest)
    for label, label_name, sample_id in FORMAL_SAMPLE_ORDER:
        for role in ROLES:
            for condition in CONDITIONS:
                token = condition.replace("_face", "").replace("_rectangle", "")
                directory = root / label_name / token
                overlay_path = directory / f"{role}_{condition}.png"
                directory.mkdir(parents=True, exist_ok=True)
                Image.new(
                    "RGB",
                    (224, 224),
                    color=(10 + label, 20 if role == "clean" else 30, 40),
                ).save(overlay_path)
                mask = None
                if condition != "clean":
                    mask = {
                        "algorithm_version": "occlusion-v2-224",
                        "context": "validation",
                        "image_height": 224,
                        "image_width": 224,
                    }
                _write_json(
                    directory / f"metadata_{role}_{condition}.json",
                    {
                        "sample_id": sample_id,
                        "ground_truth_label": label,
                        "ground_truth_label_name": label_name,
                        "split": "validation",
                        "official_split": "PublicTest",
                        "condition": condition,
                        "input_image_size": 224,
                        "input_tensor_shape": [1, 3, 224, 224],
                        "occlusion_protocol": "occlusion-v2-224",
                        "mask": mask,
                        "target_class": label,
                        "target_class_name": label_name,
                        "checkpoint_path": f"/checkpoints/{role}.pt",
                        "checkpoint_sha256": HASHES[role],
                        "model_role": role,
                        "target_layer": "model.layer4[-1]",
                        "overlay_path": str(overlay_path.resolve()),
                    },
                )
    return manifest_path


def test_loader_derives_complete_row_and_column_order_from_metadata(
    tmp_path: Path,
) -> None:
    evidence = load_gradcam_panel_evidence(_formal_fixture(tmp_path))

    assert evidence.samples == FORMAL_SAMPLE_ORDER
    assert evidence.model_order == ROLES
    assert evidence.condition_order == CONDITIONS
    assert set(evidence.condition_order) == FORMAL_CONDITION_SET
    assert evidence.combination_order == tuple(
        (role, condition) for role in ROLES for condition in CONDITIONS
    )
    assert len(evidence.cells) == 56
    assert len(evidence.cell_index()) == 56


def test_loader_rejects_metadata_provenance_mismatch(tmp_path: Path) -> None:
    manifest_path = _formal_fixture(tmp_path)
    metadata_path = next(manifest_path.parent.rglob("metadata_*.json"))
    payload = json.loads(metadata_path.read_text())
    payload["occlusion_protocol"] = "occlusion-v1"
    _write_json(metadata_path, payload)

    with pytest.raises(GradcamFigureError, match="occlusion_protocol"):
        load_gradcam_panel_evidence(manifest_path)


def test_loader_rejects_obsolete_source_overlay(tmp_path: Path) -> None:
    manifest_path = _formal_fixture(tmp_path)
    metadata_path = next(manifest_path.parent.rglob("metadata_*.json"))
    payload = json.loads(metadata_path.read_text())
    old_path = manifest_path.parent / "obsolete" / "wrong.png"
    old_path.parent.mkdir()
    Image.new("RGB", (224, 224)).save(old_path)
    payload["overlay_path"] = str(old_path.resolve())
    _write_json(metadata_path, payload)

    with pytest.raises(GradcamFigureError, match="obsolete"):
        load_gradcam_panel_evidence(manifest_path)


def test_full_panel_is_exact_size_and_create_only(tmp_path: Path) -> None:
    evidence = load_gradcam_panel_evidence(_formal_fixture(tmp_path))
    output_dir = tmp_path / "paper"

    pdf_path, png_path = render_full_panel(evidence, output_dir)

    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    assert FULL_FIGURE_SIZE == (7.20, 7.15)
    with Image.open(png_path) as image:
        assert image.size == (2160, 2145)
        assert image.info["dpi"] == pytest.approx((300, 300), abs=0.1)
    with pytest.raises(GradcamFigureError, match="will not overwrite"):
        render_full_panel(evidence, output_dir)
