import csv
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from occlusion_fer.config import (
    AppConfig,
    DatasetConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
    TrainingConfig,
)
from occlusion_fer.data import Fer2013Data, Fer2013Record
from occlusion_fer.models import create_resnet18
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import (
    apply_config_overrides,
    evaluate,
    limit_dataset,
    load_config_with_overrides,
    parse_args,
    run_training,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
)


def make_model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 7))


def make_loader(sample_count: int = 5, batch_size: int = 2) -> DataLoader:
    generator = torch.Generator().manual_seed(7)
    images = torch.rand(sample_count, 3, 2, 2, generator=generator)
    labels = torch.arange(sample_count, dtype=torch.long) % 7
    sample_ids = torch.arange(100, 100 + sample_count, dtype=torch.long)
    return DataLoader(
        TensorDataset(images, labels, sample_ids),
        batch_size=batch_size,
        shuffle=False,
    )


def make_config() -> AppConfig:
    return AppConfig(
        project=ProjectConfig(name="occlusion-fer"),
        dataset=DatasetConfig(
            name="fer2013",
            path="/path/to/fer2013.csv",
            image_size=112,
            num_classes=7,
        ),
        model=ModelConfig(name="resnet18", pretrained=True),
        training=TrainingConfig(
            mode="clean",
            seed=42,
            epochs=1,
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.0001,
            num_workers=0,
            device="auto",
        ),
        output=OutputConfig(directory="outputs/smoke"),
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_set_seed_accepts_nonnegative_integers(seed: int) -> None:
    set_seed(seed)


def test_set_seed_controls_python_numpy_and_torch() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1))

    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


@pytest.mark.parametrize("seed", [-1, 1.5, True, "42"])
def test_set_seed_rejects_invalid_values(seed: object) -> None:
    with pytest.raises(ValueError, match=r"seed.*non-negative integer"):
        set_seed(seed)


def test_select_device_cpu_always_returns_cpu() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_returns_available_device() -> None:
    selected = select_device("auto")

    assert selected.type in {"cpu", "mps", "cuda"}
    if selected.type == "cuda":
        assert torch.cuda.is_available()
    if selected.type == "mps":
        assert torch.backends.mps.is_available()


def test_select_device_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device("auto") == torch.device("cuda")


def test_select_device_rejects_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"CUDA.*not available"):
        select_device("cuda")


def test_select_device_rejects_unavailable_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"MPS.*not available"):
        select_device("mps")


@pytest.mark.parametrize("device_name", ["gpu", "AUTO", "", 1])
def test_select_device_rejects_invalid_name(device_name: object) -> None:
    with pytest.raises(ValueError, match=r"device_name.*auto.*cpu.*mps.*cuda"):
        select_device(device_name)


def test_train_one_epoch_returns_weighted_loss_and_sample_count() -> None:
    model = make_model()
    loader = make_loader(sample_count=5, batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    images = torch.cat([batch[0] for batch in loader])
    labels = torch.cat([batch[1] for batch in loader])
    with torch.no_grad():
        expected_loss = F.cross_entropy(model(images), labels).item()

    result = train_one_epoch(model, loader, optimizer, torch.device("cpu"))

    assert result.sample_count == 5
    assert result.average_loss == pytest.approx(expected_loss)
    assert np.isfinite(result.average_loss)


def test_train_one_epoch_optimizer_step_changes_parameters() -> None:
    model = make_model()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(model, make_loader(), optimizer, torch.device("cpu"))

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


def test_train_one_epoch_sets_gradients_to_none_before_backward() -> None:
    class RecordingSgd(torch.optim.SGD):
        def __init__(self, parameters: object) -> None:
            super().__init__(parameters, lr=0.0)
            self.recorded_set_to_none: bool | None = None

        def zero_grad(self, set_to_none: bool = False) -> None:
            self.recorded_set_to_none = set_to_none
            super().zero_grad(set_to_none=set_to_none)

    model = make_model()
    optimizer = RecordingSgd(model.parameters())

    train_one_epoch(model, make_loader(), optimizer, torch.device("cpu"))

    assert optimizer.recorded_set_to_none is True


def test_train_one_epoch_rejects_amp_on_cpu() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"AMP.*CUDA"):
        train_one_epoch(
            model,
            make_loader(),
            optimizer,
            torch.device("cpu"),
            amp_enabled=True,
        )


def test_train_one_epoch_rejects_empty_loader() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"training loader.*no samples"):
        train_one_epoch(model, [], optimizer, torch.device("cpu"))


def test_train_one_epoch_rejects_incorrect_images_shape() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [(torch.zeros(2, 12), torch.tensor([0, 1]), torch.tensor([1, 2]))]

    with pytest.raises(ValueError, match=r"images.*four-dimensional"):
        train_one_epoch(model, loader, optimizer, torch.device("cpu"))


def test_train_one_epoch_rejects_incorrect_labels_shape() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        (
            torch.zeros(2, 3, 2, 2),
            torch.tensor([[0], [1]]),
            torch.tensor([1, 2]),
        )
    ]

    with pytest.raises(ValueError, match=r"labels.*one-dimensional"):
        train_one_epoch(model, loader, optimizer, torch.device("cpu"))


def test_train_one_epoch_rejects_mismatched_batch_counts() -> None:
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = [
        (
            torch.zeros(2, 3, 2, 2),
            torch.tensor([0, 1]),
            torch.tensor([1]),
        )
    ]

    with pytest.raises(ValueError, match=r"batch size.*sample_ids"):
        train_one_epoch(model, loader, optimizer, torch.device("cpu"))


def test_train_one_epoch_rejects_wrong_logit_class_count() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 2 * 2, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"logits.*7 classes"):
        train_one_epoch(
            model, make_loader(), optimizer, torch.device("cpu")
        )


def test_train_one_epoch_rejects_non_finite_loss() -> None:
    class NonFiniteModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.scale * torch.full(
                (images.shape[0], 7), float("nan")
            )

    model = NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match=r"loss.*finite"):
        train_one_epoch(model, make_loader(), optimizer, torch.device("cpu"))


def test_evaluate_returns_weighted_loss_accuracy_and_sample_count() -> None:
    model = make_model().eval()
    loader = make_loader(sample_count=5, batch_size=2)
    images = torch.cat([batch[0] for batch in loader])
    labels = torch.cat([batch[1] for batch in loader])
    with torch.inference_mode():
        logits = model(images)
        expected_loss = F.cross_entropy(logits, labels).item()
        expected_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    result = evaluate(model, loader, torch.device("cpu"))

    assert result.sample_count == 5
    assert result.average_loss == pytest.approx(expected_loss)
    assert result.accuracy == pytest.approx(expected_accuracy)
    assert np.isfinite(result.average_loss)
    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_does_not_change_parameters_or_create_gradients() -> None:
    model = make_model()
    model.zero_grad(set_to_none=True)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    evaluate(model, make_loader(), torch.device("cpu"))

    assert all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_rejects_amp_on_cpu() -> None:
    with pytest.raises(ValueError, match=r"AMP.*CUDA"):
        evaluate(
            make_model(),
            make_loader(),
            torch.device("cpu"),
            amp_enabled=True,
        )


def test_evaluate_rejects_empty_loader() -> None:
    with pytest.raises(ValueError, match=r"validation loader.*no samples"):
        evaluate(make_model(), [], torch.device("cpu"))


def test_save_checkpoint_creates_reloadable_best_file(tmp_path: Path) -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    output_directory = tmp_path / "new" / "output"

    checkpoint_path = save_checkpoint(
        output_directory=output_directory,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_validation_accuracy=0.25,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
    )
    loaded = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    assert checkpoint_path == output_directory.resolve() / "best.pt"
    assert checkpoint_path.is_file()
    assert {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_validation_accuracy",
        "seed",
        "device",
        "resolved_config",
    } <= loaded.keys()
    assert loaded["epoch"] == 1
    assert loaded["best_validation_accuracy"] == pytest.approx(0.25)
    assert loaded["seed"] == 42
    assert loaded["device"] == "cpu"
    assert isinstance(loaded["resolved_config"], dict)


def test_save_checkpoint_supports_last_file_and_grad_scaler_state(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    checkpoint_path = save_checkpoint(
        output_directory=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        best_validation_accuracy=0.5,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
        checkpoint_name="last.pt",
        scaler=scaler,
    )
    loaded = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    assert checkpoint_path == tmp_path.resolve() / "last.pt"
    assert loaded["epoch"] == 2
    assert loaded["grad_scaler_state_dict"] == scaler.state_dict()


def test_limit_dataset_none_and_large_limit_use_all_samples() -> None:
    dataset = make_loader().dataset

    assert limit_dataset(dataset, None) is dataset
    assert limit_dataset(dataset, 100) is dataset


def test_limit_dataset_positive_integer_uses_first_samples() -> None:
    dataset = make_loader(sample_count=5).dataset

    limited = limit_dataset(dataset, 3)

    assert len(limited) == 3
    assert [limited[index][2].item() for index in range(3)] == [100, 101, 102]


@pytest.mark.parametrize("max_samples", [0, -1, 1.5, True, "2"])
def test_limit_dataset_rejects_invalid_limit(max_samples: object) -> None:
    with pytest.raises(ValueError, match=r"max_samples.*positive integer"):
        limit_dataset(make_loader().dataset, max_samples)


def test_apply_config_overrides_changes_only_runtime_copy() -> None:
    original = make_config()

    resolved = apply_config_overrides(
        original,
        data_path="/runtime/data.csv",
        output_directory="/runtime/output",
        epochs=3,
        batch_size=64,
        num_workers=4,
    )

    assert resolved.dataset.path == "/runtime/data.csv"
    assert resolved.output.directory == "/runtime/output"
    assert resolved.training.epochs == 3
    assert resolved.training.batch_size == 64
    assert resolved.training.num_workers == 4
    assert original.dataset.path == "/path/to/fer2013.csv"
    assert original.output.directory == "outputs/smoke"
    assert original.training.epochs == 1
    assert original.training.batch_size == 32
    assert original.training.num_workers == 0


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "64"])
def test_apply_config_overrides_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(ValueError, match=r"batch_size.*positive integer"):
        apply_config_overrides(make_config(), batch_size=batch_size)


@pytest.mark.parametrize("num_workers", [-1, 1.5, True, "4"])
def test_apply_config_overrides_rejects_invalid_num_workers(
    num_workers: object,
) -> None:
    with pytest.raises(ValueError, match=r"num_workers.*non-negative integer"):
        apply_config_overrides(make_config(), num_workers=num_workers)


def test_load_config_with_overrides_does_not_modify_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_text = """\
project:
  name: occlusion-fer
dataset:
  name: fer2013
  path: /path/to/fer2013.csv
  image_size: 112
  num_classes: 7
model:
  name: resnet18
  pretrained: true
training:
  mode: clean
  seed: 42
  epochs: 1
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 0
  device: auto
output:
  directory: outputs/smoke
"""
    config_path.write_text(config_text, encoding="utf-8")

    resolved = load_config_with_overrides(
        config_path,
        data_path="/runtime/data.csv",
        output_directory="/runtime/output",
        epochs=2,
        batch_size=64,
        num_workers=4,
    )

    assert resolved.dataset.path == "/runtime/data.csv"
    assert resolved.output.directory == "/runtime/output"
    assert resolved.training.epochs == 2
    assert resolved.training.batch_size == 64
    assert resolved.training.num_workers == 4
    assert config_path.read_text(encoding="utf-8") == config_text


def test_parse_args_supports_all_smoke_overrides() -> None:
    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--data-path",
            "/runtime/data.csv",
            "--output-dir",
            "/runtime/output",
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--num-workers",
            "4",
            "--amp",
            "--max-train-samples",
            "8",
            "--max-validation-samples",
            "4",
        ]
    )

    assert args.config == "config.yaml"
    assert args.data_path == "/runtime/data.csv"
    assert args.output_dir == "/runtime/output"
    assert args.epochs == 2
    assert args.batch_size == 64
    assert args.num_workers == 4
    assert args.amp is True
    assert args.max_train_samples == 8
    assert args.max_validation_samples == 4


def test_run_training_rejects_placeholder_data_path_before_model_creation() -> None:
    with pytest.raises(FileNotFoundError, match=r"placeholder.*--data-path"):
        run_training(make_config())


def test_artificial_cpu_training_validation_checkpoint_flow(tmp_path: Path) -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_result = train_one_epoch(
        model, make_loader(), optimizer, torch.device("cpu")
    )
    validation_result = evaluate(
        model, make_loader(sample_count=3), torch.device("cpu")
    )
    checkpoint_path = save_checkpoint(
        output_directory=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_validation_accuracy=validation_result.accuracy,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
    )

    assert train_result.sample_count == 5
    assert validation_result.sample_count == 3
    assert checkpoint_path == tmp_path.resolve() / "best.pt"
    assert checkpoint_path.is_file()


def test_artificial_fer_resnet_full_training_chain(tmp_path: Path) -> None:
    records = tuple(
        Fer2013Record(
            sample_id=index + 1,
            label=index,
            label_name=("angry", "disgust", "fear", "happy")[index],
            split="train" if index < 2 else "validation",
            image=np.full((48, 48), index * 48, dtype=np.uint8),
        )
        for index in range(4)
    )
    data = Fer2013Data(
        records=records,
        train_count=2,
        validation_count=2,
        test_count=0,
        class_counts={label: int(label < 4) for label in range(7)},
    )
    train_dataset = Fer2013TorchDataset(
        data,
        split="train",
        normalize_imagenet=True,
    )
    validation_dataset = Fer2013TorchDataset(
        data,
        split="validation",
        normalize_imagenet=True,
    )
    train_loader = create_dataloader(
        train_dataset, batch_size=2, shuffle=False, seed=42
    )
    validation_loader = create_dataloader(
        validation_dataset, batch_size=2, shuffle=False, seed=42
    )
    model = create_resnet18(num_classes=7, pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    before = model.fc.weight.detach().clone()

    training_result = train_one_epoch(
        model, train_loader, optimizer, torch.device("cpu")
    )
    validation_result = evaluate(
        model, validation_loader, torch.device("cpu")
    )
    checkpoint_path = save_checkpoint(
        output_directory=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_validation_accuracy=validation_result.accuracy,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
    )

    assert training_result.sample_count == 2
    assert validation_result.sample_count == 2
    assert not torch.equal(before, model.fc.weight)
    assert checkpoint_path == tmp_path.resolve() / "best.pt"
    assert checkpoint_path.is_file()


def test_run_training_saves_best_last_and_history(tmp_path: Path) -> None:
    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["128"] * (48 * 48))
    rows = [
        (0, pixels, "Training"),
        (1, pixels, "Training"),
        (2, pixels, "PublicTest"),
        (3, pixels, "PublicTest"),
        (4, pixels, "PrivateTest"),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerows(rows)

    original = make_config()
    output_directory = tmp_path / "output"
    config = replace(
        original,
        dataset=replace(original.dataset, path=str(csv_path)),
        model=replace(original.model, pretrained=False),
        training=replace(
            original.training,
            batch_size=2,
            num_workers=0,
            device="cpu",
        ),
        output=replace(original.output, directory=str(output_directory)),
    )

    run_training(config)

    best_path = output_directory / "best.pt"
    last_path = output_directory / "last.pt"
    history_path = output_directory / "history.json"
    assert best_path.is_file()
    assert last_path.is_file()
    assert history_path.is_file()

    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(history) == 1
    epoch_record = history[0]
    assert epoch_record["epoch"] == 1
    assert epoch_record["train_samples"] == 2
    assert epoch_record["validation_samples"] == 2
    assert epoch_record["train_seconds"] >= 0.0
    assert epoch_record["validation_seconds"] >= 0.0
    assert epoch_record["epoch_seconds"] >= 0.0
    assert epoch_record["train_samples_per_second"] > 0.0
    assert epoch_record["cuda_peak_memory_bytes"] == 0
    assert np.isfinite(epoch_record["train_loss"])
    assert np.isfinite(epoch_record["validation_loss"])
    assert 0.0 <= epoch_record["validation_accuracy"] <= 1.0

    last_checkpoint = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    runtime = last_checkpoint["resolved_config"]["runtime"]
    assert runtime["amp_enabled"] is False
    assert runtime["pin_memory"] is False
    assert runtime["persistent_workers"] is False
    assert runtime["prefetch_factor"] is None
