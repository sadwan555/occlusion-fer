"""Protocol-locked formal Grad-CAM generation for FER2013 PublicTest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.io import write_png


GRADCAM_SCHEMA_VERSION = 2
FORMAL_IMAGE_SIZE = 224
FORMAL_PROTOCOL = "occlusion-v2-224"
FORMAL_CONDITIONS = (
    "clean",
    "upper_face_0.40",
    "lower_face_0.40",
    "random_rectangle_0.40",
)
CONDITION_DIRECTORIES = {
    "clean": "clean",
    "upper_face_0.40": "upper40",
    "lower_face_0.40": "lower40",
    "random_rectangle_0.40": "random40",
}
MODEL_ROLES = ("clean", "mixed")
TARGET_LAYER_NAME = "model.layer4[-1]"
OFFICIAL_SPLIT_NAME = "PublicTest"
FER2013_LABEL_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _class_token(value: str) -> int:
    normalized = value.strip().lower()
    label_by_name = {
        name: label for label, name in enumerate(FER2013_LABEL_NAMES)
    }
    if normalized in label_by_name:
        return label_by_name[normalized]
    try:
        label = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "class must be a FER2013 label name or integer from 0 to 6"
        ) from exc
    if not 0 <= label < len(FER2013_LABEL_NAMES):
        raise argparse.ArgumentTypeError(
            "class must be a FER2013 label name or integer from 0 to 6"
        )
    return label


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the formal, PublicTest-only Grad-CAM command."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate protocol-matched ground-truth Grad-CAM overlays for "
            "formal FER2013 PublicTest checkpoints."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--formal-source-root", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--clean-checkpoint", required=True)
    parser.add_argument("--mixed-checkpoint", required=True)
    parser.add_argument("--clean-provenance", required=True)
    parser.add_argument("--mixed-provenance", required=True)
    parser.add_argument("--mean-artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sidecar")
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/gradcam_224_v2")
    parser.add_argument(
        "--device",
        choices=("cpu", "auto", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=_class_token,
        metavar="CLASS",
        help="FER2013 label names or integers; defaults to all seven classes",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=FORMAL_CONDITIONS,
        help="defaults to the four approved formal Grad-CAM conditions",
    )
    return parser.parse_args(argv)


def resolve_classes(classes: Sequence[int] | None) -> tuple[int, ...]:
    """Validate a class subset and return it in FER2013 label order."""
    if classes is None:
        return tuple(range(len(FER2013_LABEL_NAMES)))
    resolved = tuple(classes)
    if not resolved:
        raise ValueError("classes must not be empty")
    if any(type(label) is not int or not 0 <= label <= 6 for label in resolved):
        raise ValueError("classes must contain FER2013 integer labels from 0 to 6")
    if len(set(resolved)) != len(resolved):
        raise ValueError("classes must not contain duplicates")
    return tuple(sorted(resolved))


def resolve_conditions(conditions: Sequence[str] | None) -> tuple[str, ...]:
    """Validate an optional subset of the four approved conditions."""
    if conditions is None:
        return FORMAL_CONDITIONS
    resolved = tuple(conditions)
    if not resolved:
        raise ValueError("conditions must not be empty")
    if len(set(resolved)) != len(resolved):
        raise ValueError("conditions must not contain duplicates")
    invalid = [condition for condition in resolved if condition not in FORMAL_CONDITIONS]
    if invalid:
        raise ValueError("unsupported Grad-CAM condition(s): " + ", ".join(invalid))
    return tuple(condition for condition in FORMAL_CONDITIONS if condition in resolved)


def select_first_publictest_samples(
    records: Sequence[Any],
    classes: Sequence[int] | None = None,
) -> tuple[Any, ...]:
    """Choose the lowest-sample-id PublicTest record for each selected class."""
    selected: list[Any] = []
    for label in resolve_classes(classes):
        candidates = sorted(
            (
                record
                for record in records
                if record.split == "validation" and record.label == label
            ),
            key=lambda record: record.sample_id,
        )
        if not candidates:
            raise ValueError(
                "PublicTest has no valid sample for class "
                f"{label} ({FER2013_LABEL_NAMES[label]})"
            )
        selected.append(candidates[0])
    return tuple(selected)


def validate_formal_config(config: object) -> None:
    """Fail closed unless a resolved config is the formal 224/v2 route."""
    dataset = getattr(config, "dataset", None)
    model = getattr(config, "model", None)
    training = getattr(config, "training", None)
    occlusion = getattr(config, "occlusion", None)
    if dataset is None or model is None or training is None or occlusion is None:
        raise ValueError("formal Grad-CAM requires a complete Stage 8 configuration")
    if getattr(dataset, "image_size", None) != FORMAL_IMAGE_SIZE:
        raise ValueError("formal Grad-CAM requires dataset.image_size=224")
    if getattr(dataset, "num_classes", None) != len(FER2013_LABEL_NAMES):
        raise ValueError("formal Grad-CAM requires the seven FER2013 classes")
    permitted = getattr(dataset, "permitted_splits", None)
    if permitted is None or tuple(permitted) != ("Training", "PublicTest"):
        if permitted is not None and "PrivateTest" in permitted:
            raise ValueError("formal Grad-CAM rejects PrivateTest")
        raise ValueError("formal Grad-CAM permits only Training and PublicTest sources")
    if getattr(model, "name", None) != "resnet18":
        raise ValueError("formal Grad-CAM requires model.name=resnet18")
    if getattr(training, "seed", None) != 42:
        raise ValueError("formal Grad-CAM requires the seed-42 formal config")
    protocol = getattr(occlusion, "protocol", None)
    evaluation = getattr(occlusion, "evaluation", None)
    if (
        protocol is None
        or getattr(protocol, "algorithm_version", None) != FORMAL_PROTOCOL
    ):
        raise ValueError(f"formal Grad-CAM requires {FORMAL_PROTOCOL}")
    if getattr(protocol, "image_size", None) != FORMAL_IMAGE_SIZE:
        raise ValueError("formal occlusion protocol requires image_size=224")
    if evaluation is None or getattr(evaluation, "split", None) != "validation":
        raise ValueError("formal Grad-CAM requires validation/PublicTest only")


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def validate_checkpoint_identity(
    actual_sha256: str,
    provenance: Mapping[str, object],
    *,
    role: str,
) -> str:
    """Bind a checkpoint byte-for-byte to formal evaluation provenance."""
    if role not in MODEL_ROLES:
        raise ValueError("checkpoint role must be clean or mixed")
    expected = provenance.get("checkpoint_sha256")
    if type(expected) is not str or _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"{role} provenance checkpoint SHA-256 is invalid")
    if actual_sha256 != expected:
        raise ValueError(
            f"{role} checkpoint SHA-256 does not match formal provenance: "
            f"expected {expected}, got {actual_sha256}"
        )
    if provenance.get("protocol") != FORMAL_PROTOCOL:
        raise ValueError(f"{role} provenance must record {FORMAL_PROTOCOL}")
    if (
        provenance.get("image_height") != FORMAL_IMAGE_SIZE
        or provenance.get("image_width") != FORMAL_IMAGE_SIZE
    ):
        raise ValueError(f"{role} provenance must record 224x224")
    if provenance.get("seed") != 42:
        raise ValueError(f"{role} provenance must record seed 42")
    commit = provenance.get("evaluation_commit")
    if type(commit) is not str or _COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"{role} provenance evaluation commit is invalid")
    return commit


def validate_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Require checkpoint metadata for formal seed-42 ResNet-18 at 224."""
    if role not in MODEL_ROLES:
        raise ValueError("checkpoint role must be clean or mixed")
    resolved = _mapping(payload.get("resolved_config"), "checkpoint resolved_config")
    dataset = _mapping(resolved.get("dataset"), "checkpoint dataset config")
    model = _mapping(resolved.get("model"), "checkpoint model config")
    training = _mapping(resolved.get("training"), "checkpoint training config")
    if dataset.get("image_size") != FORMAL_IMAGE_SIZE:
        raise ValueError(f"{role} checkpoint metadata must record image_size=224")
    if model.get("name") != "resnet18":
        raise ValueError(f"{role} checkpoint metadata must record ResNet-18")
    if training.get("seed") != 42:
        raise ValueError(f"{role} checkpoint metadata must record seed 42")
    if training.get("mode") != role:
        raise ValueError(f"{role} checkpoint metadata has the wrong training mode")
    if role == "mixed":
        occlusion = _mapping(resolved.get("occlusion"), "mixed checkpoint occlusion config")
        protocol = _mapping(occlusion.get("protocol"), "mixed checkpoint protocol")
        if (
            protocol.get("algorithm_version") != FORMAL_PROTOCOL
            or protocol.get("image_size") != FORMAL_IMAGE_SIZE
        ):
            raise ValueError(
                "mixed checkpoint metadata must record occlusion-v2-224 at 224"
            )


def validate_previous_selection(
    previous_manifest_path: str | Path,
    selected: Sequence[Any],
) -> dict[int, int]:
    """Require regenerated samples to match the prior deterministic selection."""
    path = _require_file(previous_manifest_path, "previous Grad-CAM sample manifest")
    payload = _load_json_mapping(path, "previous Grad-CAM sample manifest")
    if payload.get("split") != "validation" or payload.get("official_split") != OFFICIAL_SPLIT_NAME:
        raise ValueError("previous Grad-CAM manifest is not PublicTest-only")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("previous Grad-CAM manifest samples must be a list")
    old_by_label: dict[int, int] = {}
    for item in samples:
        if not isinstance(item, Mapping):
            raise ValueError("previous Grad-CAM manifest sample is invalid")
        label = item.get("label")
        sample_id = item.get("sample_id")
        if type(label) is not int or type(sample_id) is not int or label in old_by_label:
            raise ValueError("previous Grad-CAM manifest sample identity is invalid")
        old_by_label[label] = sample_id
    actual = {record.label: record.sample_id for record in selected}
    expected = {label: old_by_label.get(label) for label in actual}
    if None in expected.values() or actual != expected:
        raise ValueError(
            f"deterministic sample selection does not match previous manifest: "
            f"expected {expected}, got {actual}"
        )
    return actual


def prepare_condition_image(
    clean_image: Tensor,
    sample_id: int,
    condition: str,
    fill_vector: Sequence[float],
    *,
    v2_masker: Callable[[Tensor, int, str, Sequence[float]], tuple[Tensor, object]],
) -> tuple[Tensor, object | None]:
    """Apply only the injected formal v2 evaluator mask implementation."""
    if tuple(clean_image.shape) != (3, FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE):
        raise ValueError("clean Grad-CAM image must have shape [3, 224, 224]")
    if condition == "clean":
        return clean_image.detach().clone(), None
    if condition not in FORMAL_CONDITIONS:
        raise ValueError(f"unsupported Grad-CAM condition: {condition!r}")
    masked, metadata = v2_masker(clean_image, sample_id, condition, fill_vector)
    if tuple(masked.shape) != (3, FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE):
        raise ValueError("v2 masker did not return shape [3, 224, 224]")
    algorithm = getattr(metadata, "algorithm_version", None)
    if algorithm != FORMAL_PROTOCOL:
        raise ValueError("v2 masker metadata does not record occlusion-v2-224")
    return masked, metadata


def compute_formal_gradcam(
    model: nn.Module,
    image: Tensor,
    *,
    target_layer: nn.Module,
    target_class: int,
    device: torch.device,
    compute: Callable[..., object],
) -> object:
    """Validate and pass exactly one [1, 3, 224, 224] input to Grad-CAM."""
    if tuple(image.shape) != (3, FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE):
        raise ValueError("formal Grad-CAM image must have shape [3, 224, 224]")
    batch = image.unsqueeze(0).to(device)
    if tuple(batch.shape) != (1, 3, FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE):
        raise RuntimeError("formal Grad-CAM batch shape changed unexpectedly")
    return compute(
        model,
        batch,
        target_layer=target_layer,
        target_class=target_class,
    )


def build_overlay_uint8(
    normalized_image: Tensor,
    cam: Tensor,
    *,
    maximum_alpha: float = 0.60,
) -> Tensor:
    """Create a deterministic RGB heat overlay for matching H x W inputs."""
    if not isinstance(normalized_image, Tensor) or normalized_image.ndim != 3:
        raise ValueError("normalized_image must have shape [3, H, W]")
    if normalized_image.shape[0] != 3 or min(normalized_image.shape[1:]) <= 0:
        raise ValueError("normalized_image must have shape [3, H, W]")
    expected_cam_shape = tuple(normalized_image.shape[1:])
    if not isinstance(cam, Tensor) or tuple(cam.shape) != expected_cam_shape:
        raise ValueError(f"cam must have shape {expected_cam_shape}")
    if not normalized_image.is_floating_point() or not cam.is_floating_point():
        raise ValueError("normalized_image and cam must be floating tensors")
    if not torch.isfinite(normalized_image).all() or not torch.isfinite(cam).all():
        raise ValueError("normalized_image and cam must contain finite values")
    if cam.min().item() < 0.0 or cam.max().item() > 1.0:
        raise ValueError("cam values must be between 0 and 1")
    if not 0.0 <= maximum_alpha <= 1.0:
        raise ValueError("maximum_alpha must be between 0 and 1")

    image = normalized_image.detach().to(device="cpu", dtype=torch.float32)
    heat = cam.detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    base = (image * std + mean).clamp(0.0, 1.0)
    heat_color = torch.tensor((1.0, 0.12, 0.0), dtype=torch.float32).view(3, 1, 1)
    alpha = heat * maximum_alpha
    overlay = base * (1.0 - alpha) + heat_color * alpha
    return torch.round(overlay.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


def _write_png_atomic(path: Path, image: Tensor) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".png",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        write_png(image, str(temporary_path), compression_level=9)
        temporary_path.replace(path)
        temporary_path = None
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not write Grad-CAM overlay: {path}") from exc
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved.resolve()


def _load_json_mapping(path: Path, description: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} must be readable UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must contain a mapping")
    return payload


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"could not validate formal source repository: {root}") from exc
    return completed.stdout.strip()


def _activate_formal_runtime(
    formal_source_root: str | Path,
    *,
    expected_commit: str,
) -> SimpleNamespace:
    """Load exact formal modules from a clean worktree at the provenance commit."""
    root = Path(formal_source_root).expanduser().resolve()
    package_path = root / "src" / "occlusion_fer"
    if not (package_path / "occlusion_evaluate.py").is_file():
        raise FileNotFoundError(f"formal Stage 8 source package not found: {package_path}")
    actual_commit = _git_output(root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            "formal source revision does not match evaluation provenance: "
            f"expected {expected_commit}, got {actual_commit}"
        )
    dirty = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError("formal source worktree must be clean")

    package = sys.modules.get("occlusion_fer")
    package_search = getattr(package, "__path__", None)
    if package is None or package_search is None:
        raise RuntimeError("occlusion_fer package is not initialized")
    required_names = {
        "occlusion_fer.artifacts",
        "occlusion_fer.checkpoint_compat",
        "occlusion_fer.config",
        "occlusion_fer.data",
        "occlusion_fer.mask_manifest",
        "occlusion_fer.models",
        "occlusion_fer.occlusion",
        "occlusion_fer.occlusion_evaluate",
        "occlusion_fer.permitted_splits",
        "occlusion_fer.torch_data",
        "occlusion_fer.train",
        "occlusion_fer.training_mean",
    }
    conflicts = sorted(name for name in required_names if name in sys.modules)
    if conflicts:
        raise RuntimeError(
            "formal runtime must be activated before project submodules are imported: "
            + ", ".join(conflicts)
        )
    original_paths = list(package_search)
    package_search.insert(0, str(package_path))
    try:
        artifacts = importlib.import_module("occlusion_fer.artifacts")
        checkpoint = importlib.import_module("occlusion_fer.checkpoint_compat")
        data = importlib.import_module("occlusion_fer.data")
        gradcam = importlib.import_module("occlusion_fer.gradcam")
        manifest = importlib.import_module("occlusion_fer.mask_manifest")
        models = importlib.import_module("occlusion_fer.models")
        occlusion = importlib.import_module("occlusion_fer.occlusion")
        evaluation = importlib.import_module("occlusion_fer.occlusion_evaluate")
        splits = importlib.import_module("occlusion_fer.permitted_splits")
        torch_data = importlib.import_module("occlusion_fer.torch_data")
        train = importlib.import_module("occlusion_fer.train")
        training_mean = importlib.import_module("occlusion_fer.training_mean")
    except Exception:
        package_search[:] = original_paths
        raise
    formal_modules = (
        artifacts,
        checkpoint,
        data,
        manifest,
        models,
        occlusion,
        evaluation,
        splits,
        torch_data,
        train,
        training_mean,
    )
    if any(package_path not in Path(module.__file__).resolve().parents for module in formal_modules):
        raise RuntimeError("a Stage 8 runtime module did not load from the formal worktree")
    return SimpleNamespace(
        formal_source_root=root,
        formal_commit=actual_commit,
        write_json_atomic=artifacts.write_json_atomic,
        utc_now=artifacts.utc_now,
        load_config_with_overrides=train.load_config_with_overrides,
        select_device=train.select_device,
        load_stage_b_source=splits.load_stage_b_source,
        validate_official_stage_b_sources=splits.validate_official_stage_b_sources,
        permitted_splits_to_data=splits.permitted_splits_to_data,
        Fer2013TorchDataset=torch_data.Fer2013TorchDataset,
        load_training_mean_v2=training_mean.load_training_mean_v2,
        training_mean_v2_sha256=training_mean.training_mean_v2_sha256,
        load_manifest_v2=manifest.load_manifest_v2,
        apply_evaluation_mask_v2=occlusion.apply_evaluation_mask_v2,
        prepare_evaluation_bindings=evaluation._prepare_evaluation_bindings,
        create_resnet18=models.create_resnet18,
        load_masked_evaluation_checkpoint=evaluation.load_masked_evaluation_checkpoint,
        compute_gradcam=gradcam.compute_gradcam,
        gradcam_source_path=Path(gradcam.__file__).resolve(),
    )


def _target_layer(model: nn.Module) -> nn.Module:
    layer4 = getattr(model, "layer4", None)
    if not isinstance(layer4, nn.Sequential) or len(layer4) == 0:
        raise ValueError("configured model does not expose model.layer4[-1]")
    target = layer4[-1]
    if target is not model.layer4[-1]:
        raise RuntimeError("Grad-CAM target layer identity changed unexpectedly")
    return target


def _reserve_output_directory(path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            "Grad-CAM generation will not overwrite an existing output directory: "
            f"{output_path}"
        ) from exc
    return output_path.resolve()


def _metadata_mapping(value: object) -> dict[str, object]:
    if not is_dataclass(value):
        raise ValueError("v2 mask metadata must be a dataclass")
    return asdict(value)


def _validate_mask_manifest_row(mask: object, row: object) -> dict[str, object]:
    metadata = _metadata_mapping(mask)
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
    differences = {
        field: (metadata.get(field), getattr(row, field, None))
        for field in fields
        if metadata.get(field) != getattr(row, field, None)
    }
    if differences:
        raise ValueError(f"v2 mask geometry does not match formal manifest: {differences}")
    if metadata.get("algorithm_version") != FORMAL_PROTOCOL:
        raise ValueError("v2 mask metadata has the wrong protocol")
    if metadata.get("context") != "validation":
        raise ValueError("v2 mask metadata is not PublicTest evaluation geometry")
    return metadata


def _load_formal_model(
    runtime: SimpleNamespace,
    checkpoint_path: Path,
    provenance: Mapping[str, object],
    *,
    role: str,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, object], str]:
    model = runtime.create_resnet18(num_classes=7, pretrained=False)
    payload, digest = runtime.load_masked_evaluation_checkpoint(model, checkpoint_path)
    validate_checkpoint_identity(digest, provenance, role=role)
    validate_checkpoint_metadata(payload, role=role)
    print(f"{role}_checkpoint_path={checkpoint_path}", flush=True)
    print(f"{role}_checkpoint_sha256={digest}", flush=True)
    return model.to(device).eval(), payload, digest


def _selection_manifest(
    *,
    runtime: SimpleNamespace,
    config_path: Path,
    data_path: Path,
    mean_path: Path,
    mean_sha256: str,
    manifest_path: Path,
    manifest_sidecar_path: Path,
    manifest_sha256: str,
    previous_manifest_path: Path,
    checkpoints: Mapping[str, Path],
    checkpoint_sha256: Mapping[str, str],
    provenance_paths: Mapping[str, Path],
    selected: Sequence[Any],
    conditions: Sequence[str],
    device: torch.device,
) -> dict[str, object]:
    return {
        "schema_version": GRADCAM_SCHEMA_VERSION,
        "status": "running",
        "generation_timestamp": runtime.utc_now(),
        "split": "validation",
        "official_split": OFFICIAL_SPLIT_NAME,
        "private_test_materialized": False,
        "image_size": FORMAL_IMAGE_SIZE,
        "input_tensor_shape": [1, 3, FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE],
        "preprocessing": {
            "config_source": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "source_image_size": [48, 48],
            "resize": [FORMAL_IMAGE_SIZE, FORMAL_IMAGE_SIZE],
            "channel_construction": "replicate grayscale to three channels",
            "normalization": "ImageNet mean/std",
        },
        "data_path": str(data_path),
        "occlusion_protocol": FORMAL_PROTOCOL,
        "training_mean_artifact": {
            "path": str(mean_path),
            "artifact_sha256": mean_sha256,
            "file_sha256": _sha256_file(mean_path),
        },
        "publictest_v2_manifest": {
            "path": str(manifest_path),
            "sidecar_path": str(manifest_sidecar_path),
            "manifest_sha256": manifest_sha256,
            "file_sha256": _sha256_file(manifest_path),
            "sidecar_file_sha256": _sha256_file(manifest_sidecar_path),
        },
        "formal_source": {
            "path": str(runtime.formal_source_root),
            "evaluation_commit": runtime.formal_commit,
            "gradcam_core_path": str(runtime.gradcam_source_path),
        },
        "previous_sample_manifest": {
            "path": str(previous_manifest_path),
            "sha256": _sha256_file(previous_manifest_path),
        },
        "selection_method": (
            "group PublicTest records by class label, sort each group by "
            "sample_id, choose the first valid record; require equality with "
            "the prior 112/v1 manifest"
        ),
        "target_class_policy": "ground_truth",
        "target_layer": TARGET_LAYER_NAME,
        "device": str(device),
        "conditions": list(conditions),
        "checkpoints": {
            role: {
                "model_role": role,
                "path": str(checkpoints[role]),
                "sha256": checkpoint_sha256[role],
                "formal_evaluation_provenance": str(provenance_paths[role]),
            }
            for role in MODEL_ROLES
        },
        "selected_sample_ids": [record.sample_id for record in selected],
        "samples": [
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "label_name": record.label_name,
                "split": record.split,
                "official_split": OFFICIAL_SPLIT_NAME,
            }
            for record in selected
        ],
        "generated_image_count": 0,
        "generated_metadata_count": 0,
        "skipped_samples": [],
    }


def generate_gradcam_outputs(
    config: object,
    *,
    runtime: SimpleNamespace,
    config_path: str | Path,
    clean_checkpoint: str | Path,
    mixed_checkpoint: str | Path,
    clean_provenance_path: str | Path,
    mixed_provenance_path: str | Path,
    mean_artifact_path: str | Path,
    manifest_path: str | Path,
    manifest_sidecar_path: str | Path,
    previous_manifest_path: str | Path,
    output_directory: str | Path,
    classes: Sequence[int] | None = None,
    conditions: Sequence[str] | None = None,
) -> dict[str, object]:
    """Generate protocol-matched overlays without changing formal behavior."""
    validate_formal_config(config)
    selected_classes = resolve_classes(classes)
    selected_conditions = resolve_conditions(conditions)
    config_file = _require_file(config_path, "formal evaluation configuration")
    data_path = _require_file(getattr(config.dataset, "path"), "FER2013 data file")
    mean_path = _require_file(mean_artifact_path, "Training mean v2 artifact")
    v2_manifest_path = _require_file(manifest_path, "PublicTest v2 manifest")
    v2_sidecar_path = _require_file(manifest_sidecar_path, "PublicTest v2 manifest sidecar")
    previous_path = _require_file(previous_manifest_path, "previous Grad-CAM sample manifest")
    checkpoints = {
        "clean": _require_file(clean_checkpoint, "formal clean checkpoint"),
        "mixed": _require_file(mixed_checkpoint, "formal mixed checkpoint"),
    }
    provenance_paths = {
        "clean": _require_file(clean_provenance_path, "clean evaluation provenance"),
        "mixed": _require_file(mixed_provenance_path, "mixed evaluation provenance"),
    }
    provenance = {
        role: _load_json_mapping(provenance_paths[role], f"{role} evaluation provenance")
        for role in MODEL_ROLES
    }
    provenance_commits = {
        validate_checkpoint_identity(
            str(provenance[role].get("checkpoint_sha256")),
            provenance[role],
            role=role,
        )
        for role in MODEL_ROLES
    }
    if provenance_commits != {runtime.formal_commit}:
        raise ValueError("formal evaluation provenance commits do not match runtime source")

    device = runtime.select_device(getattr(config.training, "device"))
    models: dict[str, nn.Module] = {}
    payloads: dict[str, Mapping[str, object]] = {}
    checkpoint_sha256: dict[str, str] = {}
    for role in MODEL_ROLES:
        model, payload, digest = _load_formal_model(
            runtime,
            checkpoints[role],
            provenance[role],
            role=role,
            device=device,
        )
        models[role] = model
        payloads[role] = payload
        checkpoint_sha256[role] = digest

    source = runtime.load_stage_b_source(data_path)
    runtime.validate_official_stage_b_sources(source)
    data = runtime.permitted_splits_to_data(source)
    if data.test_count != 0 or any(record.split == "test" for record in data.records):
        raise ValueError("PrivateTest was materialized by the formal source route")
    selected = select_first_publictest_samples(data.records, selected_classes)
    validate_previous_selection(previous_path, selected)
    dataset = runtime.Fer2013TorchDataset(
        data,
        split="validation",
        image_size=FORMAL_IMAGE_SIZE,
        normalize_imagenet=True,
    )
    validation_records = tuple(record for record in data.records if record.split == "validation")
    dataset_index = {record.sample_id: index for index, record in enumerate(validation_records)}

    mean_artifact = runtime.load_training_mean_v2(mean_path)
    mean_sha256 = runtime.training_mean_v2_sha256(mean_artifact)
    rows, envelope = runtime.load_manifest_v2(
        v2_manifest_path,
        v2_sidecar_path,
        publictest_dataset_sha256=source.publictest.dataset_sha256,
        training_mean_artifact_sha256=mean_sha256,
        require_official=True,
    )
    if envelope.algorithm_version != FORMAL_PROTOCOL:
        raise ValueError("PublicTest manifest does not use occlusion-v2-224")
    if (envelope.image_height, envelope.image_width) != (224, 224):
        raise ValueError("PublicTest manifest does not use 224x224 geometry")
    manifest_rows = {(row.sample_id, row.condition): row for row in rows}
    for record in selected:
        for condition in selected_conditions:
            if condition != "clean" and (record.sample_id, condition) not in manifest_rows:
                raise ValueError(
                    f"formal v2 manifest is missing sample {record.sample_id} {condition}"
                )

    fill_vectors = []
    for role in MODEL_ROLES:
        fill_vectors.append(
            runtime.prepare_evaluation_bindings(
                manifest_envelope=envelope,
                checkpoint_payload=payloads[role],
                training_mean_artifact=mean_artifact,
                training_dataset_sha256=source.training.dataset_sha256,
                publictest_dataset_sha256=source.publictest.dataset_sha256,
                evaluation_provenance=provenance[role],
            )
        )
    if fill_vectors[0] != fill_vectors[1]:
        raise ValueError("formal clean and mixed evaluation fill bindings differ")
    fill_vector = fill_vectors[0]

    output_path = _reserve_output_directory(output_directory)
    manifest = _selection_manifest(
        runtime=runtime,
        config_path=config_file,
        data_path=data_path,
        mean_path=mean_path,
        mean_sha256=mean_sha256,
        manifest_path=v2_manifest_path,
        manifest_sidecar_path=v2_sidecar_path,
        manifest_sha256=envelope.manifest_sha256,
        previous_manifest_path=previous_path,
        checkpoints=checkpoints,
        checkpoint_sha256=checkpoint_sha256,
        provenance_paths=provenance_paths,
        selected=selected,
        conditions=selected_conditions,
        device=device,
    )
    output_manifest_path = output_path / "sample_manifest.json"
    runtime.write_json_atomic(output_manifest_path, manifest)

    generated_count = 0
    metadata_count = 0
    try:
        for role in MODEL_ROLES:
            model = models[role]
            target_layer = _target_layer(model)
            for record in selected:
                clean_image, label, sample_id = dataset[dataset_index[record.sample_id]]
                if label != record.label or sample_id != record.sample_id:
                    raise RuntimeError("PublicTest dataset indexing changed unexpectedly")
                if tuple(clean_image.shape) != (3, 224, 224):
                    raise RuntimeError("formal preprocessing did not produce [3, 224, 224]")
                for condition in selected_conditions:
                    condition_image, mask_metadata = prepare_condition_image(
                        clean_image,
                        sample_id,
                        condition,
                        fill_vector,
                        v2_masker=runtime.apply_evaluation_mask_v2,
                    )
                    mask_payload = None
                    if mask_metadata is not None:
                        mask_payload = _validate_mask_manifest_row(
                            mask_metadata,
                            manifest_rows[(sample_id, condition)],
                        )
                    result = compute_formal_gradcam(
                        model,
                        condition_image,
                        target_layer=target_layer,
                        target_class=label,
                        device=device,
                        compute=runtime.compute_gradcam,
                    )
                    condition_path = output_path / record.label_name / CONDITION_DIRECTORIES[condition]
                    overlay_path = condition_path / f"{role}_model_overlay.png"
                    metadata_path = condition_path / f"metadata_{role}.json"
                    _write_png_atomic(
                        overlay_path,
                        build_overlay_uint8(condition_image, result.cam),
                    )
                    metadata = {
                        "schema_version": GRADCAM_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "ground_truth_label": label,
                        "ground_truth_label_name": record.label_name,
                        "split": "validation",
                        "official_split": OFFICIAL_SPLIT_NAME,
                        "condition": condition,
                        "input_image_size": FORMAL_IMAGE_SIZE,
                        "input_tensor_shape": [1, 3, 224, 224],
                        "occlusion_protocol": FORMAL_PROTOCOL,
                        "mask": mask_payload,
                        "target_class": result.target_class,
                        "target_class_name": FER2013_LABEL_NAMES[result.target_class],
                        "predicted_class": result.predicted_class,
                        "predicted_class_name": FER2013_LABEL_NAMES[result.predicted_class],
                        "predicted_probability": result.predicted_probability,
                        "target_logit": result.target_logit,
                        "checkpoint_path": str(checkpoints[role]),
                        "checkpoint_sha256": checkpoint_sha256[role],
                        "model_role": role,
                        "target_layer": TARGET_LAYER_NAME,
                        "device": str(device),
                        "overlay_path": str(overlay_path.resolve()),
                    }
                    runtime.write_json_atomic(metadata_path, metadata)
                    generated_count += 1
                    metadata_count += 1
                    print(
                        f"generated={generated_count} role={role} "
                        f"sample_id={sample_id} condition={condition}",
                        flush=True,
                    )
        manifest["status"] = "completed"
        manifest["generated_image_count"] = generated_count
        manifest["generated_metadata_count"] = metadata_count
        manifest["completed_timestamp"] = runtime.utc_now()
        runtime.write_json_atomic(output_manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["generated_image_count"] = generated_count
        manifest["generated_metadata_count"] = metadata_count
        manifest["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        runtime.write_json_atomic(output_manifest_path, manifest)
        raise
    finally:
        models.clear()
        payloads.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"split=validation/{OFFICIAL_SPLIT_NAME}")
    print(f"private_test_materialized=false")
    print(f"input_tensor_shape=[1,3,224,224]")
    print(f"occlusion_protocol={FORMAL_PROTOCOL}")
    print(f"device={device}")
    print(f"selected_samples={len(selected)}")
    print(f"generated_images={generated_count}")
    print(f"sample_manifest={output_manifest_path.resolve()}")
    print(f"output_directory={output_path}")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    """Load exact formal inputs and run protocol-locked Grad-CAM generation."""
    args = parse_args(argv)
    clean_provenance_path = _require_file(args.clean_provenance, "clean evaluation provenance")
    mixed_provenance_path = _require_file(args.mixed_provenance, "mixed evaluation provenance")
    clean_provenance = _load_json_mapping(clean_provenance_path, "clean evaluation provenance")
    mixed_provenance = _load_json_mapping(mixed_provenance_path, "mixed evaluation provenance")
    commits = {
        provenance.get("evaluation_commit")
        for provenance in (clean_provenance, mixed_provenance)
    }
    if len(commits) != 1:
        raise ValueError("clean and mixed evaluation provenance commits differ")
    expected_commit = next(iter(commits))
    if type(expected_commit) is not str or _COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("evaluation provenance commit is invalid")
    runtime = _activate_formal_runtime(
        args.formal_source_root,
        expected_commit=expected_commit,
    )
    config = runtime.load_config_with_overrides(
        args.config,
        data_path=args.data_path,
        output_directory=args.output_dir,
        device=args.device,
        num_workers=0,
    )
    sidecar = args.manifest_sidecar or str(Path(args.manifest).with_suffix(".json"))
    generate_gradcam_outputs(
        config,
        runtime=runtime,
        config_path=args.config,
        clean_checkpoint=args.clean_checkpoint,
        mixed_checkpoint=args.mixed_checkpoint,
        clean_provenance_path=clean_provenance_path,
        mixed_provenance_path=mixed_provenance_path,
        mean_artifact_path=args.mean_artifact,
        manifest_path=args.manifest,
        manifest_sidecar_path=sidecar,
        previous_manifest_path=args.previous_manifest,
        output_directory=args.output_dir,
        classes=args.classes,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
