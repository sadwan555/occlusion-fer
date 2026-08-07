from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from occlusion_fer.mask_manifest import (
    build_manifest_v2_rows,
    make_manifest_v2_envelope,
    write_manifest_v2_create_or_verify,
)
from occlusion_fer.occlusion import V2_MASKED_CONDITIONS
from occlusion_fer.permitted_splits import (
    SplitRecord,
    make_permitted_splits,
)
from occlusion_fer.training_mean import (
    calculate_training_mean_v2,
    training_mean_v2_sha256,
    write_training_mean_v2,
)


LABEL_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


def _record(sample_id: int, label: int, value: int) -> SplitRecord:
    pixels = tuple((value + index % 13) % 256 for index in range(48 * 48))
    return SplitRecord(sample_id, label, pixels)


def _make_sources(*, training_offset: int = 0):
    training = [
        _record(10 + label, label, 20 + label + training_offset)
        for label in range(7)
    ]
    publictest = [
        _record(100 + label, label, 80 + label)
        for label in range(7)
    ]
    return make_permitted_splits(training, publictest)


def _write_source(path: Path, sources) -> None:
    def records(source):
        return [
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "pixels": list(record.pixels),
            }
            for record in source.records
        ]

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "Training": records(sources.training),
                "PublicTest": records(sources.publictest),
            }
        ),
        encoding="utf-8",
    )


def _write_v2_artifacts(tmp_path: Path, sources):
    mean = calculate_training_mean_v2(sources.training)
    mean_path = tmp_path / "mean.json"
    write_training_mean_v2(mean_path, mean)
    mean_sha = training_mean_v2_sha256(mean)
    rows = build_manifest_v2_rows(
        [record.sample_id for record in sources.publictest.records],
        publictest_dataset_sha256=sources.publictest.dataset_sha256,
        mean_artifact_sha256=mean_sha,
    )
    envelope = make_manifest_v2_envelope(
        rows,
        publictest_dataset_sha256=sources.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha,
    )
    manifest_path = tmp_path / "manifest.csv"
    sidecar_path = tmp_path / "manifest.json"
    write_manifest_v2_create_or_verify(
        manifest_path,
        sidecar_path,
        rows,
        envelope,
    )
    return mean_path, manifest_path, sidecar_path, rows, envelope


def _assert_png(path: Path, *, minimum_width: int = 1) -> None:
    assert path.stat().st_size > 100
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width >= minimum_width
        assert image.height > 0


def test_dataset_export_is_deterministic_complete_and_hashed(tmp_path) -> None:
    from occlusion_fer.paper_figures import export_fer2013_examples, sha256_file

    sources = _make_sources()
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)

    first = export_fer2013_examples(
        source_path,
        tmp_path / "first",
        split="Training",
        samples_per_class=1,
        require_official=False,
        creation_command=["synthetic-dataset-export"],
    )
    second = export_fer2013_examples(
        source_path,
        tmp_path / "second",
        split="Training",
        samples_per_class=1,
        require_official=False,
        creation_command=["synthetic-dataset-export"],
    )

    assert first["source_split"] == "Training"
    assert first["selection_rule"] == "sample_id_ascending_first_n_per_class"
    assert first["class_order"] == list(LABEL_NAMES)
    assert [item["sample_id"] for item in first["selected_samples"]] == list(
        range(10, 17)
    )
    assert first["selected_samples"] == second["selected_samples"]
    assert (tmp_path / "first" / "fer2013_class_examples.png").read_bytes() == (
        tmp_path / "second" / "fer2013_class_examples.png"
    ).read_bytes()
    assert len(first["output_files_sha256"]) == 8
    for relative_path, digest in first["output_files_sha256"].items():
        output = tmp_path / "first" / relative_path
        _assert_png(output, minimum_width=500 if "sample" in relative_path else 3000)
        assert sha256_file(output) == digest
    assert json.loads(
        (tmp_path / "first" / "figure_provenance.json").read_text(
            encoding="utf-8"
        )
    ) == first


def test_dataset_export_publictest_and_fail_closed_boundaries(tmp_path) -> None:
    from occlusion_fer.paper_figures import (
        PaperFigureError,
        export_fer2013_examples,
        validate_source_split,
    )

    sources = _make_sources()
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)
    provenance = export_fer2013_examples(
        source_path,
        tmp_path / "public",
        split="PublicTest",
        require_official=False,
    )
    assert provenance["source_split"] == "PublicTest"
    assert [item["sample_id"] for item in provenance["selected_samples"]] == list(
        range(100, 107)
    )

    with pytest.raises(PaperFigureError, match="PrivateTest"):
        validate_source_split("PrivateTest")
    with pytest.raises(PaperFigureError, match="PrivateTest"):
        export_fer2013_examples(
            source_path,
            tmp_path / "private",
            split="PrivateTest",
            require_official=False,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        export_fer2013_examples(
            source_path,
            tmp_path / "public",
            split="PublicTest",
            require_official=False,
        )


def test_dataset_export_requires_every_class_and_requested_count(tmp_path) -> None:
    from occlusion_fer.paper_figures import PaperFigureError, export_fer2013_examples

    sources = make_permitted_splits(
        [_record(10 + label, label, 20) for label in range(6)],
        [_record(100 + label, label, 80) for label in range(7)],
    )
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)
    with pytest.raises(PaperFigureError, match="neutral"):
        export_fer2013_examples(
            source_path,
            tmp_path / "missing",
            require_official=False,
        )
    with pytest.raises(PaperFigureError, match="2 sample"):
        export_fer2013_examples(
            source_path,
            tmp_path / "too-many",
            samples_per_class=2,
            require_official=False,
        )


def test_figure_source_routes_combined_csv_without_parsing_private_fields(
    tmp_path,
) -> None:
    from occlusion_fer.paper_figures import export_fer2013_examples

    source_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["17"] * (48 * 48))
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("emotion", "pixels", "Usage"))
        for label in range(7):
            writer.writerow((label, pixels, "Training"))
        writer.writerow(("DO NOT PARSE", "DO NOT MATERIALIZE", "PrivateTest"))
        for label in range(7):
            writer.writerow((label, pixels, "PublicTest"))

    provenance = export_fer2013_examples(
        source_path,
        tmp_path / "figure",
        require_official=False,
    )
    assert provenance["source_kind"] == "combined_csv_usage_routed"
    assert provenance["excluded_official_split"] == "PrivateTest"
    assert all(
        item["sample_id"] != 9 for item in provenance["selected_samples"]
    )


def test_occlusion_export_uses_exact_protocol_and_manifest_geometry(tmp_path) -> None:
    from occlusion_fer.paper_figures import export_occlusion_examples, sha256_file

    sources = _make_sources()
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)
    mean_path, manifest_path, sidecar_path, rows, envelope = _write_v2_artifacts(
        tmp_path, sources
    )
    provenance = export_occlusion_examples(
        source_path,
        tmp_path / "occlusion",
        split="PublicTest",
        sample_id=100,
        training_mean_path=mean_path,
        manifest_path=manifest_path,
        manifest_sidecar_path=sidecar_path,
        require_official=False,
        creation_command=["synthetic-occlusion-export"],
    )

    expected_order = ["clean", *V2_MASKED_CONDITIONS]
    assert provenance["condition_order"] == expected_order
    assert provenance["protocol"] == "occlusion-v2-224"
    assert provenance["image_size"] == [224, 224]
    assert provenance["sample_id"] == 100
    assert provenance["official_split"] == "PublicTest"
    assert provenance["manifest_semantic_sha256"] == envelope.manifest_sha256
    assert len(provenance["geometry"]) == 9
    manifest_rows = {row.condition: row for row in rows if row.sample_id == 100}
    for metadata in provenance["geometry"]:
        row = manifest_rows[metadata["condition"]]
        assert (metadata["top"], metadata["left"]) == (row.top, row.left)
        assert (metadata["height"], metadata["width"]) == (
            row.height,
            row.width,
        )
        assert metadata["target_ratio"] in (0.2, 0.3, 0.4)
    assert len(provenance["output_files_sha256"]) == 11
    for relative_path, digest in provenance["output_files_sha256"].items():
        output = tmp_path / "occlusion" / relative_path
        _assert_png(
            output,
            minimum_width=600 if relative_path != "occlusion_protocol_examples.png" else 3000,
        )
        assert sha256_file(output) == digest


def test_occlusion_export_is_stable_and_does_not_mutate_source_tensor(tmp_path) -> None:
    from occlusion_fer.paper_figures import (
        build_formal_occlusion_tensors,
        export_occlusion_examples,
    )
    from occlusion_fer.occlusion import normalized_fill_vector_v2
    from occlusion_fer.training_mean import load_training_mean_v2

    sources = _make_sources()
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)
    mean_path, manifest_path, sidecar_path, _, _ = _write_v2_artifacts(
        tmp_path, sources
    )
    clean = torch.linspace(-1, 1, 3 * 224 * 224).reshape(3, 224, 224)
    original = clean.clone()
    tensors, _ = build_formal_occlusion_tensors(
        clean,
        sample_id=100,
        fill_vector=normalized_fill_vector_v2(load_training_mean_v2(mean_path)),
    )
    assert torch.equal(clean, original)
    assert tuple(tensors) == ("clean", *V2_MASKED_CONDITIONS)
    assert tensors["clean"] is not clean

    first = export_occlusion_examples(
        source_path,
        tmp_path / "first",
        split="PublicTest",
        sample_id=100,
        training_mean_path=mean_path,
        manifest_path=manifest_path,
        manifest_sidecar_path=sidecar_path,
        require_official=False,
    )
    second = export_occlusion_examples(
        source_path,
        tmp_path / "second",
        split="PublicTest",
        sample_id=100,
        training_mean_path=mean_path,
        manifest_path=manifest_path,
        manifest_sidecar_path=sidecar_path,
        require_official=False,
    )
    assert first["geometry"] == second["geometry"]
    for filename in first["output_files_sha256"]:
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()


def test_occlusion_export_rejects_private_missing_or_mismatched_artifacts(
    tmp_path,
) -> None:
    from occlusion_fer.paper_figures import PaperFigureError, export_occlusion_examples

    sources = _make_sources()
    source_path = tmp_path / "permitted.json"
    _write_source(source_path, sources)
    mean_path, manifest_path, sidecar_path, _, _ = _write_v2_artifacts(
        tmp_path, sources
    )
    base = {
        "data_path": source_path,
        "output_directory": tmp_path / "output",
        "sample_id": 100,
        "training_mean_path": mean_path,
        "require_official": False,
    }
    with pytest.raises(PaperFigureError, match="PrivateTest"):
        export_occlusion_examples(**base, split="PrivateTest")
    with pytest.raises(PaperFigureError, match="manifest"):
        export_occlusion_examples(**base, split="PublicTest")

    other_sources = _make_sources(training_offset=30)
    other_source_path = tmp_path / "other.json"
    _write_source(other_source_path, other_sources)
    with pytest.raises(PaperFigureError, match="mean|Training"):
        export_occlusion_examples(
            other_source_path,
            tmp_path / "bad-mean",
            split="Training",
            sample_id=10,
            training_mean_path=mean_path,
            require_official=False,
        )

    wrong_public = make_permitted_splits(
        sources.training.records,
        [_record(200 + label, label, 80) for label in range(7)],
    )
    wrong_path = tmp_path / "wrong-public.json"
    _write_source(wrong_path, wrong_public)
    with pytest.raises(PaperFigureError, match="PublicTest|manifest"):
        export_occlusion_examples(
            wrong_path,
            tmp_path / "bad-manifest",
            split="PublicTest",
            sample_id=200,
            training_mean_path=mean_path,
            manifest_path=manifest_path,
            manifest_sidecar_path=sidecar_path,
            require_official=False,
        )

    existing_output = tmp_path / "output"
    existing_output.mkdir()
    (existing_output / "existing.txt").write_text("reserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        export_occlusion_examples(
            source_path,
            existing_output,
            split="PublicTest",
            sample_id=100,
            training_mean_path=mean_path,
            manifest_path=manifest_path,
            manifest_sidecar_path=sidecar_path,
            require_official=False,
        )


def test_formal_export_signature_has_no_arbitrary_fill_override() -> None:
    from occlusion_fer.paper_figures import export_occlusion_examples

    assert "fill_vector" not in inspect.signature(export_occlusion_examples).parameters
