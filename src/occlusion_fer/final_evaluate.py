"""Explicit, read-only FER2013 PrivateTest evaluation entry point."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from occlusion_fer.artifacts import write_evaluation_artifacts
from occlusion_fer.config import AppConfig
from occlusion_fer.data import load_fer2013_csv
from occlusion_fer.evaluation import evaluate
from occlusion_fer.models import create_resnet18
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import load_config_with_overrides, select_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately explicit final-evaluation command."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one locked FER2013 checkpoint on PrivateTest without "
            "training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--confirm-private-test", action="store_true")
    return parser.parse_args(argv)


def load_checkpoint_model_state(
    checkpoint_path: str | Path,
) -> Mapping[str, Tensor]:
    """Load and validate a model state from a project checkpoint."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint must be a mapping containing model_state_dict"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            "checkpoint must contain a non-empty model_state_dict mapping"
        )
    for name, value in state.items():
        if type(name) is not str or not isinstance(value, Tensor):
            raise ValueError(
                "checkpoint model_state_dict must map parameter names to tensors"
            )
        if (
            (value.is_floating_point() or value.is_complex())
            and not torch.isfinite(value).all().item()
        ):
            raise ValueError(
                f"checkpoint contains non-finite tensor for {name}"
            )
    return state


def run_final_evaluation(
    config: AppConfig,
    checkpoint_path: str | Path,
    *,
    confirm_private_test: bool,
    amp_enabled: bool = False,
) -> dict[str, Path]:
    """Evaluate only PrivateTest after an explicit one-way confirmation."""
    if confirm_private_test is not True:
        raise ValueError(
            "PrivateTest evaluation requires --confirm-private-test"
        )

    output_path = Path(config.output.directory).expanduser()
    expected_artifacts = (
        output_path / "final_test" / "clean_metrics.json",
        output_path / "final_test" / "clean_per_class_metrics.csv",
        output_path / "final_test" / "clean_confusion_matrix.csv",
        output_path / "final_test" / "clean_predictions.csv",
    )
    existing_artifacts = [path for path in expected_artifacts if path.exists()]
    if existing_artifacts:
        raise FileExistsError(
            "Final PrivateTest results already exist: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    if config.dataset.path == "/path/to/fer2013.csv":
        raise FileNotFoundError(
            "FER2013 data path is still the placeholder; provide --data-path"
        )
    data_path = Path(config.dataset.path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"FER2013 data path does not exist or is not a file: {data_path}"
        )

    state = load_checkpoint_model_state(checkpoint_path)
    device = select_device(config.training.device)
    data = load_fer2013_csv(data_path, include_splits=("test",))
    test_dataset = Fer2013TorchDataset(
        data,
        split="test",
        image_size=config.dataset.image_size,
        normalize_imagenet=True,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.training.seed,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.training.num_workers > 0,
        prefetch_factor=2,
    )
    model = create_resnet18(
        num_classes=config.dataset.num_classes,
        pretrained=False,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint model_state_dict is incompatible with ResNet-18"
        ) from exc
    model = model.to(device)
    result = evaluate(
        model,
        test_loader,
        device,
        split="test",
        condition="clean",
        amp_enabled=amp_enabled,
    )
    paths = write_evaluation_artifacts(
        output_path,
        "final_test/clean",
        result,
    )
    print(f"final_test_samples={result.sample_count}")
    print(f"final_test_loss={result.average_loss:.4f}")
    print(f"final_test_accuracy={result.accuracy:.4f}")
    print(f"final_test_macro_f1={result.macro_f1:.4f}")
    for name, path in paths.items():
        print(f"saved_{name}={path}")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Load runtime overrides and run one confirmed final evaluation."""
    args = parse_args(argv)
    config = load_config_with_overrides(
        args.config,
        data_path=args.data_path,
        output_directory=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    run_final_evaluation(
        config,
        checkpoint_path=args.checkpoint,
        confirm_private_test=args.confirm_private_test,
        amp_enabled=args.amp,
    )


if __name__ == "__main__":
    main()
