import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from occlusion_fer.artifacts import write_resolved_config
from occlusion_fer.config import load_config
from occlusion_fer.data import load_fer2013_csv
from occlusion_fer.losses import (
    build_evaluation_criterion,
    build_training_criterion,
)
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import build_optimizer


E4_ACCURACY = 0.6870994706046253
E4_MACRO_F1 = 0.6803268061914639
MACRO_F1_FLOOR = E4_MACRO_F1 - 0.0030
FORMAL_ACCURACY_GATE = 0.70
COMPUTE_ORDER = {"E5": 0, "E6": 1, "E7": 2}


@dataclass(frozen=True)
class CandidateResult:
    name: str
    accuracy: float
    macro_f1: float


def passes_protection(result: CandidateResult) -> bool:
    return (
        result.macro_f1 >= MACRO_F1_FLOOR
        and result.accuracy > E4_ACCURACY
    )


def eligible_for_formal_seeds(result: CandidateResult) -> bool:
    return passes_protection(result) and result.accuracy >= FORMAL_ACCURACY_GATE


def select_candidate(results: list[CandidateResult]) -> CandidateResult | None:
    eligible = [result for result in results if eligible_for_formal_seeds(result)]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda result: (
            result.accuracy,
            result.macro_f1,
            -COMPUTE_ORDER[result.name],
        ),
    )


def test_macro_f1_below_raw_tolerance_fails_protection() -> None:
    result = CandidateResult("E5", 0.71, MACRO_F1_FLOOR - 1e-15)

    assert not passes_protection(result)


def test_accuracy_not_strictly_above_e4_fails_protection() -> None:
    result = CandidateResult("E5", E4_ACCURACY, MACRO_F1_FLOOR)

    assert not passes_protection(result)


def test_accuracy_below_point_seven_cannot_enter_formal_seeds() -> None:
    result = CandidateResult("E5", 0.6999999999999999, E4_MACRO_F1)

    assert passes_protection(result)
    assert not eligible_for_formal_seeds(result)


def test_selection_prefers_higher_raw_accuracy() -> None:
    selected = select_candidate(
        [
            CandidateResult("E5", 0.701, 0.69),
            CandidateResult("E6", 0.702, 0.68),
        ]
    )

    assert selected is not None
    assert selected.name == "E6"


def test_selection_prefers_macro_f1_when_accuracy_is_exactly_equal() -> None:
    selected = select_candidate(
        [
            CandidateResult("E5", 0.702, 0.681),
            CandidateResult("E6", 0.702, 0.682),
        ]
    )

    assert selected is not None
    assert selected.name == "E6"


def test_selection_prefers_lower_compute_when_metrics_are_exactly_equal() -> None:
    selected = select_candidate(
        [
            CandidateResult("E7", 0.702, 0.682),
            CandidateResult("E6", 0.702, 0.682),
            CandidateResult("E5", 0.702, 0.682),
        ]
    )

    assert selected is not None
    assert selected.name == "E5"


def test_no_candidate_is_selected_when_formal_gate_is_not_met() -> None:
    selected = select_candidate(
        [
            CandidateResult("E5", 0.699, 0.69),
            CandidateResult("E6", 0.698, 0.69),
            CandidateResult("E7", 0.697, 0.69),
        ]
    )

    assert selected is None


def write_split_restricted_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "synthetic-fer2013.csv"
    pixels = " ".join(["128"] * (48 * 48))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerows(
            [
                (0, pixels, "Training"),
                (1, pixels, "Training"),
                (0, pixels, "PublicTest"),
                (1, pixels, "PublicTest"),
                ("not parsed", "not parsed", "PrivateTest"),
            ]
        )
    return csv_path


@pytest.mark.parametrize(
    ("filename", "image_size", "epochs", "output_directory"),
    [
        (
            "fer2013_resnet18_e5_longer_training.yaml",
            112,
            50,
            "outputs/screening/e5_longer_training",
        ),
        (
            "fer2013_resnet18_e6_high_resolution.yaml",
            224,
            30,
            "outputs/screening/e6_high_resolution",
        ),
        (
            "fer2013_resnet18_e7_high_resolution_longer.yaml",
            224,
            50,
            "outputs/screening/e7_high_resolution_longer",
        ),
    ],
)
def test_second_round_synthetic_config_and_batch_smoke(
    tmp_path: Path,
    filename: str,
    image_size: int,
    epochs: int,
    output_directory: str,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config = load_config(repository_root / "configs" / "experiments" / filename)
    csv_path = write_split_restricted_csv(tmp_path)
    smoke_config = replace(
        config,
        dataset=replace(config.dataset, path=str(csv_path)),
        model=replace(config.model, pretrained=False),
    )
    data = load_fer2013_csv(
        csv_path,
        include_splits=("train", "validation"),
    )
    train_dataset = Fer2013TorchDataset(
        data,
        split="train",
        image_size=smoke_config.dataset.image_size,
        augmentation=smoke_config.dataset.augmentation,
    )
    validation_dataset = Fer2013TorchDataset(
        data,
        split="validation",
        image_size=smoke_config.dataset.image_size,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        seed=smoke_config.training.seed,
        num_workers=0,
    )
    validation_loader = create_dataloader(
        validation_dataset,
        batch_size=2,
        shuffle=False,
        seed=smoke_config.training.seed,
        num_workers=0,
    )

    torch.manual_seed(smoke_config.training.seed)
    train_images, _, _ = next(iter(train_loader))
    validation_images, _, _ = next(iter(validation_loader))
    training_criterion = build_training_criterion(smoke_config.training.loss)
    evaluation_criterion = build_evaluation_criterion()
    optimizer = build_optimizer(nn.Linear(1, 1), smoke_config.training)
    resolved_path = write_resolved_config(
        tmp_path / filename.removesuffix(".yaml"),
        asdict(smoke_config),
    )
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))

    assert data.train_count == 2
    assert data.validation_count == 2
    assert data.test_count == 0
    assert train_images.shape == (2, 3, image_size, image_size)
    assert validation_images.shape == (2, 3, image_size, image_size)
    assert smoke_config.training.epochs == epochs
    assert training_criterion.label_smoothing == pytest.approx(0.1)
    assert evaluation_criterion.label_smoothing == pytest.approx(0.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.001)
    assert smoke_config.training.scheduler.type == "none"
    assert resolved["dataset"]["image_size"] == image_size
    assert resolved["dataset"]["augmentation"]["type"] == "mild_affine"
    assert resolved["training"]["epochs"] == epochs
    assert resolved["training"]["loss"]["label_smoothing"] == pytest.approx(0.1)
    assert resolved["training"]["weight_decay"] == pytest.approx(0.001)
    assert resolved["training"]["scheduler"]["type"] == "none"
    assert resolved["output"]["directory"] == output_directory
    assert not (tmp_path / "final_test").exists()
