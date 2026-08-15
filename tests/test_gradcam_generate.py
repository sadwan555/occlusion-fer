from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from occlusion_fer.gradcam import GradCAMResult
from occlusion_fer import gradcam_generate
from occlusion_fer.gradcam_generate import (
    FORMAL_CONDITIONS,
    FORMAL_IMAGE_SIZE,
    FORMAL_PROTOCOL,
    TARGET_LAYER_NAME,
    build_overlay_uint8,
    compute_formal_gradcam,
    parse_args,
    prepare_condition_image,
    resolve_classes,
    resolve_conditions,
    select_first_publictest_samples,
    validate_checkpoint_identity,
    validate_checkpoint_metadata,
    validate_formal_config,
    validate_previous_selection,
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


def _record(sample_id: int, label: int, split: str = "validation") -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        label=label,
        label_name=LABEL_NAMES[label],
        split=split,
    )


def _formal_config(
    *,
    image_size: int = 224,
    protocol: str = "occlusion-v2-224",
    split: str = "validation",
    permitted_splits: tuple[str, ...] = ("Training", "PublicTest"),
) -> SimpleNamespace:
    return SimpleNamespace(
        project=SimpleNamespace(run_role="formal_clean_occlusion_evaluation"),
        dataset=SimpleNamespace(
            image_size=image_size,
            num_classes=7,
            permitted_splits=permitted_splits,
        ),
        model=SimpleNamespace(name="resnet18"),
        training=SimpleNamespace(seed=42),
        occlusion=SimpleNamespace(
            protocol=SimpleNamespace(
                algorithm_version=protocol,
                image_size=image_size,
            ),
            evaluation=SimpleNamespace(split=split),
        ),
    )


def _checkpoint_payload(role: str) -> dict[str, object]:
    resolved: dict[str, object] = {
        "dataset": {"image_size": 224},
        "model": {"name": "resnet18"},
        "training": {"mode": role, "seed": 42},
    }
    if role == "mixed":
        resolved["occlusion"] = {
            "protocol": {
                "algorithm_version": "occlusion-v2-224",
                "image_size": 224,
            }
        }
    return {"resolved_config": resolved, "model_state_dict": {"weight": torch.ones(1)}}


def test_parse_args_defaults_to_cpu_and_formal_output() -> None:
    args = parse_args(
        [
            "--config", "config.yaml",
            "--formal-source-root", "formal-source",
            "--clean-checkpoint", "clean.pt",
            "--mixed-checkpoint", "mixed.pt",
            "--clean-provenance", "clean.json",
            "--mixed-provenance", "mixed.json",
            "--mean-artifact", "mean.json",
            "--manifest", "manifest.csv",
            "--previous-manifest", "old.json",
        ]
    )

    assert args.device == "cpu"
    assert args.output_dir == "outputs/gradcam_224_v2"
    assert args.classes is None
    assert args.conditions is None


def test_resolvers_are_deterministic_and_reject_duplicates() -> None:
    assert resolve_classes(None) == tuple(range(7))
    assert resolve_classes((6, 0, 3)) == (0, 3, 6)
    assert resolve_conditions(None) == FORMAL_CONDITIONS
    assert resolve_conditions(("lower_face_0.40", "clean")) == (
        "clean",
        "lower_face_0.40",
    )

    with pytest.raises(ValueError, match="duplicates"):
        resolve_classes((0, 0))
    with pytest.raises(ValueError, match="duplicates"):
        resolve_conditions(("clean", "clean"))
    with pytest.raises(ValueError, match="unsupported"):
        resolve_conditions(("upper_face_0.20",))


def test_formal_config_accepts_224_v2_publictest_only() -> None:
    validate_formal_config(_formal_config())


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_formal_config(image_size=112), "image_size=224"),
        (_formal_config(protocol="occlusion-v1"), "occlusion-v2-224"),
        (_formal_config(split="test"), "validation/PublicTest"),
        (
            _formal_config(permitted_splits=("Training", "PublicTest", "PrivateTest")),
            "PrivateTest",
        ),
    ],
)
def test_formal_config_rejects_wrong_protocol_or_split(
    config: SimpleNamespace,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_formal_config(config)


def test_selection_uses_only_publictest_and_lowest_sample_id_per_class() -> None:
    records = (
        _record(40, 0),
        _record(12, 0),
        _record(8, 0, split="test"),
        _record(31, 3),
        _record(22, 3),
    )

    selected = select_first_publictest_samples(records, classes=(3, 0))

    assert [(record.label, record.sample_id, record.split) for record in selected] == [
        (0, 12, "validation"),
        (3, 22, "validation"),
    ]


def test_previous_manifest_requires_same_deterministic_sample_ids(tmp_path: Path) -> None:
    old_manifest = tmp_path / "sample_manifest.json"
    old_manifest.write_text(
        json.dumps(
            {
                "split": "validation",
                "official_split": "PublicTest",
                "samples": [
                    {"sample_id": 12, "label": 0},
                    {"sample_id": 22, "label": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = (_record(12, 0), _record(22, 3))

    assert validate_previous_selection(old_manifest, selected) == {0: 12, 3: 22}

    with pytest.raises(ValueError, match="does not match"):
        validate_previous_selection(old_manifest, (_record(13, 0), _record(22, 3)))


def test_prepare_condition_image_uses_only_injected_v2_masker() -> None:
    clean = torch.zeros(3, 224, 224, dtype=torch.float32)
    calls: list[tuple[int, str]] = []

    def v2_masker(
        image: torch.Tensor,
        sample_id: int,
        condition: str,
        fill_vector: tuple[float, float, float],
    ) -> tuple[torch.Tensor, SimpleNamespace]:
        calls.append((sample_id, condition))
        masked = image.clone()
        masked[:, :90, :] = torch.tensor(fill_vector).view(3, 1, 1)
        return masked, SimpleNamespace(algorithm_version=FORMAL_PROTOCOL)

    masked, metadata = prepare_condition_image(
        clean,
        sample_id=28711,
        condition="upper_face_0.40",
        fill_vector=(0.1, 0.2, 0.3),
        v2_masker=v2_masker,
    )

    assert calls == [(28711, "upper_face_0.40")]
    assert tuple(masked.shape) == (3, 224, 224)
    assert metadata.algorithm_version == FORMAL_PROTOCOL
    assert torch.equal(clean, torch.zeros_like(clean))
    source = inspect.getsource(gradcam_generate)
    assert "from occlusion_fer.occlusion import apply_evaluation_mask\n" not in source
    assert "apply_evaluation_mask(" not in source


def test_compute_formal_gradcam_passes_exact_224_batch() -> None:
    image = torch.zeros(3, 224, 224, dtype=torch.float32)
    seen: list[tuple[int, ...]] = []

    def fake_compute(
        model: torch.nn.Module,
        batch: torch.Tensor,
        *,
        target_layer: torch.nn.Module,
        target_class: int,
    ) -> GradCAMResult:
        del model, target_layer
        seen.append(tuple(batch.shape))
        return GradCAMResult(
            cam=torch.zeros(224, 224),
            target_class=target_class,
            predicted_class=target_class,
            target_logit=1.0,
            predicted_probability=0.5,
        )

    result = compute_formal_gradcam(
        torch.nn.Identity(),
        image,
        target_layer=torch.nn.Identity(),
        target_class=3,
        device=torch.device("cpu"),
        compute=fake_compute,
    )

    assert seen == [(1, 3, 224, 224)]
    assert tuple(result.cam.shape) == (224, 224)


def test_overlay_accepts_224_and_is_deterministic_uint8() -> None:
    image = torch.zeros(3, 224, 224, dtype=torch.float32)
    cam = torch.linspace(0.0, 1.0, 224 * 224, dtype=torch.float32).reshape(224, 224)
    image_before = image.clone()
    cam_before = cam.clone()

    first = build_overlay_uint8(image, cam)
    second = build_overlay_uint8(image, cam)

    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.uint8
    assert torch.equal(first, second)
    torch.testing.assert_close(image, image_before, rtol=0, atol=0)
    torch.testing.assert_close(cam, cam_before, rtol=0, atol=0)


def test_checkpoint_identity_fails_closed_on_wrong_clean_sha() -> None:
    expected = "a" * 64
    provenance = {
        "checkpoint_sha256": expected,
        "protocol": FORMAL_PROTOCOL,
        "image_height": FORMAL_IMAGE_SIZE,
        "image_width": FORMAL_IMAGE_SIZE,
        "seed": 42,
        "evaluation_commit": "b" * 40,
    }

    with pytest.raises(ValueError, match="clean checkpoint SHA-256"):
        validate_checkpoint_identity("c" * 64, provenance, role="clean")


@pytest.mark.parametrize("role", ["clean", "mixed"])
def test_checkpoint_metadata_requires_formal_seed42_resnet18_224(role: str) -> None:
    validate_checkpoint_metadata(_checkpoint_payload(role), role=role)

    payload = _checkpoint_payload(role)
    payload["resolved_config"]["dataset"]["image_size"] = 112  # type: ignore[index]
    with pytest.raises(ValueError, match="image_size=224"):
        validate_checkpoint_metadata(payload, role=role)


def test_target_layer_identity_is_fixed() -> None:
    assert TARGET_LAYER_NAME == "model.layer4[-1]"
