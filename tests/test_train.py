import csv
import inspect
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

import occlusion_fer.train as train_module
from occlusion_fer.config import (
    AppConfig,
    DatasetConfig,
    EarlyStoppingConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
    SchedulerConfig,
    TrainingConfig,
)
from occlusion_fer.data import Fer2013Data, Fer2013Record
from occlusion_fer.models import create_resnet18
from occlusion_fer.schedulers import EpochLearningRateScheduler
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader
from occlusion_fer.train import (
    EarlyStopping,
    apply_config_overrides,
    evaluate,
    is_better_validation_macro_f1,
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


def test_macro_f1_checkpoint_rule_requires_strict_improvement() -> None:
    assert is_better_validation_macro_f1(0.61, 0.60)
    assert not is_better_validation_macro_f1(0.60, 0.60)
    assert not is_better_validation_macro_f1(0.59, 0.60)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_macro_f1_checkpoint_rule_rejects_invalid_candidate(
    value: float,
) -> None:
    with pytest.raises(ValueError, match=r"candidate.*finite.*0 and 1"):
        is_better_validation_macro_f1(value, 0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_macro_f1_checkpoint_rule_rejects_invalid_best(value: float) -> None:
    with pytest.raises(ValueError, match=r"best.*finite.*-1 and 1"):
        is_better_validation_macro_f1(0.5, value)


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
        logits = model(images)
        expected_loss = F.cross_entropy(logits, labels).item()
        expected_accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    result = train_one_epoch(model, loader, optimizer, torch.device("cpu"))

    assert result.sample_count == 5
    assert result.average_loss == pytest.approx(expected_loss)
    assert result.accuracy == pytest.approx(expected_accuracy)
    assert np.isfinite(result.average_loss)


def test_train_one_epoch_remains_clean_without_epoch_or_occlusion_inputs() -> None:
    parameters = inspect.signature(train_one_epoch).parameters
    source = inspect.getsource(train_module.train_one_epoch)

    assert tuple(parameters) == (
        "model",
        "loader",
        "optimizer",
        "device",
        "amp_enabled",
        "scaler",
    )
    assert "epoch" not in parameters
    assert "occlusion" not in source
    assert "mask" not in source
    assert "manifest" not in source
    assert "training_mean" not in source
def test_train_accuracy_uses_samples_not_unweighted_batch_means() -> None:
    class IndexedLogits(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = nn.Parameter(
                torch.tensor(
                    [
                        [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0],
                    ]
                )
            )

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            indices = images[:, 0, 0, 0].to(dtype=torch.long)
            return self.logits[indices]

    images = torch.zeros(5, 3, 2, 2)
    images[:, 0, 0, 0] = torch.arange(5)
    labels = torch.tensor([0, 1, 2, 3, 4])
    sample_ids = torch.arange(5)
    loader = DataLoader(
        TensorDataset(images, labels, sample_ids),
        batch_size=2,
        shuffle=False,
    )
    model = IndexedLogits()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    with torch.no_grad():
        expected_loss = F.cross_entropy(model(images), labels).item()

    result = train_one_epoch(model, loader, optimizer, torch.device("cpu"))

    assert result.sample_count == 5
    assert result.accuracy == pytest.approx(3.0 / 5.0)
    assert result.average_loss == pytest.approx(expected_loss)


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
        best_validation_macro_f1=0.25,
        validation_accuracy=0.3,
        validation_macro_f1=0.25,
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
        "best_validation_macro_f1",
        "validation_accuracy",
        "validation_macro_f1",
        "seed",
        "device",
        "resolved_config",
    } <= loaded.keys()
    assert loaded["epoch"] == 1
    assert loaded["best_validation_macro_f1"] == pytest.approx(0.25)
    assert loaded["validation_accuracy"] == pytest.approx(0.3)
    assert loaded["validation_macro_f1"] == pytest.approx(0.25)
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
        best_validation_macro_f1=0.5,
        validation_accuracy=0.6,
        validation_macro_f1=0.5,
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


def test_save_checkpoint_includes_scheduler_state_when_enabled(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = EpochLearningRateScheduler(
        optimizer,
        SchedulerConfig(
            type="warmup_cosine",
            warmup_epochs=3,
            warmup_start_factor=0.1,
            min_learning_rate=0.000001,
        ),
        base_learning_rate=0.0001,
        total_epochs=30,
    )
    scheduler.set_epoch(1)

    checkpoint_path = save_checkpoint(
        output_directory=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_validation_macro_f1=0.5,
        validation_accuracy=0.6,
        validation_macro_f1=0.5,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
        scheduler=scheduler,
    )
    loaded = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    assert loaded["scheduler_state_dict"] == scheduler.state_dict()


def test_save_checkpoint_without_scheduler_keeps_legacy_payload(
    tmp_path: Path,
) -> None:
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    checkpoint_path = save_checkpoint(
        output_directory=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_validation_macro_f1=0.5,
        validation_accuracy=0.6,
        validation_macro_f1=0.5,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
    )
    loaded = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    assert "scheduler_state_dict" not in loaded


def test_early_stopping_disabled_never_stops() -> None:
    early_stopping = EarlyStopping(
        EarlyStoppingConfig(enabled=False, patience=1)
    )

    assert all(
        early_stopping.update(improved=False) is False for _ in range(20)
    )


def test_early_stopping_counts_consecutive_epochs_without_strict_improvement() -> None:
    early_stopping = EarlyStopping(
        EarlyStoppingConfig(enabled=True, patience=2)
    )

    assert early_stopping.update(improved=True) is False
    assert early_stopping.update(improved=False) is False
    assert early_stopping.update(improved=True) is False
    assert early_stopping.update(improved=False) is False
    assert early_stopping.update(improved=False) is True
    assert early_stopping.epochs_without_improvement == 2


def test_early_stopping_preserves_macro_f1_best_epoch() -> None:
    values = [0.50, 0.60, 0.60, 0.59]
    early_stopping = EarlyStopping(
        EarlyStoppingConfig(enabled=True, patience=2)
    )
    best = -1.0
    best_epoch = 0

    for epoch, value in enumerate(values, start=1):
        improved = is_better_validation_macro_f1(value, best)
        if improved:
            best = value
            best_epoch = epoch
        if early_stopping.update(improved=improved):
            break

    assert best == pytest.approx(0.60)
    assert best_epoch == 2
    assert epoch == 4


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
        seed=123,
        device="cuda",
        epochs=3,
        batch_size=64,
        num_workers=4,
    )

    assert resolved.dataset.path == "/runtime/data.csv"
    assert resolved.output.directory == "/runtime/output"
    assert resolved.training.seed == 123
    assert resolved.training.device == "cuda"
    assert resolved.training.epochs == 3
    assert resolved.training.batch_size == 64
    assert resolved.training.num_workers == 4
    assert original.dataset.path == "/path/to/fer2013.csv"
    assert original.output.directory == "outputs/smoke"
    assert original.training.seed == 42
    assert original.training.epochs == 1
    assert original.training.batch_size == 32
    assert original.training.num_workers == 0


def test_apply_config_overrides_revalidates_scheduler_epoch_budget() -> None:
    original = make_config()
    original = replace(
        original,
        training=replace(
            original.training,
            epochs=30,
            scheduler=SchedulerConfig(
                type="warmup_cosine",
                warmup_epochs=3,
                warmup_start_factor=0.1,
                min_learning_rate=0.000001,
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"warmup_epochs.*epochs"):
        apply_config_overrides(original, epochs=3)


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


@pytest.mark.parametrize("seed", [-1, 1.5, True, "123"])
def test_apply_config_overrides_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match=r"seed.*non-negative integer"):
        apply_config_overrides(make_config(), seed=seed)


@pytest.mark.parametrize("device", ["gpu", "CUDA", "", 1])
def test_apply_config_overrides_rejects_invalid_device(device: object) -> None:
    with pytest.raises(ValueError, match=r"device override.*auto.*cpu.*mps.*cuda"):
        apply_config_overrides(make_config(), device=device)


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
        seed=123,
        device="cuda",
        epochs=2,
        batch_size=64,
        num_workers=4,
    )

    assert resolved.dataset.path == "/runtime/data.csv"
    assert resolved.output.directory == "/runtime/output"
    assert resolved.training.seed == 123
    assert resolved.training.device == "cuda"
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
            "--seed",
            "123",
            "--device",
            "cuda",
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
    assert args.seed == 123
    assert args.device == "cuda"
    assert args.epochs == 2
    assert args.batch_size == 64
    assert args.num_workers == 4
    assert args.amp is True
    assert args.max_train_samples == 8
    assert args.max_validation_samples == 4


def test_run_training_rejects_placeholder_data_path_before_model_creation() -> None:
    with pytest.raises(FileNotFoundError, match=r"placeholder.*--data-path"):
        run_training(make_config())


def test_run_training_refuses_existing_output_without_modifying_it(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "fer2013.csv"
    csv_path.write_text("present", encoding="utf-8")
    output_directory = tmp_path / "completed-run"
    output_directory.mkdir()
    metadata_path = output_directory / "run_metadata.json"
    metadata_path.write_text(
        '{"status": "completed"}\n', encoding="utf-8"
    )
    original = make_config()
    config = replace(
        original,
        dataset=replace(original.dataset, path=str(csv_path)),
        output=replace(original.output, directory=str(output_directory)),
    )

    with pytest.raises(FileExistsError, match=r"new output directory"):
        run_training(config)

    assert metadata_path.read_text(encoding="utf-8") == (
        '{"status": "completed"}\n'
    )
    assert not (output_directory / "failure.json").exists()


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
        best_validation_macro_f1=validation_result.macro_f1,
        validation_accuracy=validation_result.accuracy,
        validation_macro_f1=validation_result.macro_f1,
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
        best_validation_macro_f1=validation_result.macro_f1,
        validation_accuracy=validation_result.accuracy,
        validation_macro_f1=validation_result.macro_f1,
        seed=42,
        device=torch.device("cpu"),
        resolved_config=asdict(make_config()),
    )
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    fixed_input = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 112 * 112,
        dtype=torch.float32,
    ).reshape(1, 3, 112, 112)
    with torch.inference_mode():
        output_before_reload = model.eval()(fixed_input)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    reloaded_model = create_resnet18(
        num_classes=7,
        pretrained=False,
    ).eval()
    incompatible = reloaded_model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )
    with torch.inference_mode():
        output_after_reload = reloaded_model(fixed_input)

    assert training_result.sample_count == 2
    assert validation_result.sample_count == 2
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert not torch.equal(before, model.fc.weight)
    assert checkpoint_path == tmp_path.resolve() / "best.pt"
    assert checkpoint_path.is_file()
    assert set(payload["model_state_dict"]) == set(reloaded_model.state_dict())
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(output_before_reload, output_after_reload)


def test_run_training_saves_best_last_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import occlusion_fer.mask_manifest as manifest_module
    import occlusion_fer.occlusion as occlusion_module
    import occlusion_fer.training_mean as training_mean_module

    def forbidden_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("clean smoke training must not use Stage A masking")

    monkeypatch.setattr(
        occlusion_module,
        "apply_training_batch",
        forbidden_call,
    )
    monkeypatch.setattr(
        occlusion_module,
        "apply_evaluation_batch",
        forbidden_call,
    )
    monkeypatch.setattr(
        training_mean_module,
        "load_training_mean_artifact",
        forbidden_call,
    )
    monkeypatch.setattr(
        manifest_module,
        "generate_manifest_rows",
        forbidden_call,
    )
    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["128"] * (48 * 48))
    rows = [
        (0, pixels, "Training"),
        (1, pixels, "Training"),
        (2, pixels, "PublicTest"),
        (3, pixels, "PublicTest"),
        ("not parsed", "not parsed", "PrivateTest"),
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
    history_csv_path = output_directory / "history.csv"
    resolved_config_path = output_directory / "resolved_config.yaml"
    run_metadata_path = output_directory / "run_metadata.json"
    assert best_path.is_file()
    assert last_path.is_file()
    assert history_path.is_file()
    assert history_csv_path.is_file()
    assert resolved_config_path.is_file()
    assert run_metadata_path.is_file()
    for checkpoint_role in ("best", "last"):
        for suffix in (
            "metrics.json",
            "per_class_metrics.csv",
            "confusion_matrix.csv",
            "predictions.csv",
        ):
            assert (
                output_directory
                / "validation"
                / f"{checkpoint_role}_{suffix}"
            ).is_file()

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
    assert 0.0 <= epoch_record["train_accuracy"] <= 1.0
    assert epoch_record["learning_rate"] == pytest.approx(0.0001)
    assert np.isfinite(epoch_record["validation_loss"])
    assert 0.0 <= epoch_record["validation_accuracy"] <= 1.0
    assert 0.0 <= epoch_record["validation_macro_f1"] <= 1.0
    assert epoch_record["updated_best_checkpoint"] is True

    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["seed"] == 42
    assert metadata["training_mode"] == "clean"
    assert metadata["artifacts"]["history_json"] == "history.json"

    with (
        output_directory / "validation" / "best_predictions.csv"
    ).open("r", encoding="utf-8", newline="") as predictions_file:
        prediction_rows = list(csv.DictReader(predictions_file))
    assert {row["sample_id"] for row in prediction_rows} == {"4", "5"}
    assert all(row["split"] == "validation" for row in prediction_rows)

    last_checkpoint = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    runtime = last_checkpoint["resolved_config"]["runtime"]
    assert runtime["amp_enabled"] is False
    assert runtime["pin_memory"] is False
    assert runtime["persistent_workers"] is False
    assert runtime["prefetch_factor"] is None
    assert 0.0 <= last_checkpoint["best_validation_macro_f1"] <= 1.0
    assert 0.0 <= last_checkpoint["validation_macro_f1"] <= 1.0


def test_run_training_records_failure_after_run_artifacts_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "fer2013.csv"
    pixels = " ".join(["128"] * (48 * 48))
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["emotion", "pixels", "Usage"])
        writer.writerows(
            [
                (0, pixels, "Training"),
                (1, pixels, "PublicTest"),
                (2, pixels, "PrivateTest"),
            ]
        )
    original = make_config()
    output_directory = tmp_path / "failed-run"
    config = replace(
        original,
        dataset=replace(original.dataset, path=str(csv_path)),
        model=replace(original.model, pretrained=False),
        training=replace(
            original.training,
            batch_size=1,
            num_workers=0,
            device="cpu",
        ),
        output=replace(original.output, directory=str(output_directory)),
    )

    def fail_training(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("occlusion_fer.train.train_one_epoch", fail_training)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_training(config)

    failure = json.loads(
        (output_directory / "failure.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (output_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert failure["stage"] == "training"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic training failure"
    assert metadata["status"] == "failed"
    assert metadata["artifacts"]["failure"] == "failure.json"


def write_synthetic_training_csv(tmp_path: Path) -> Path:
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
            ]
        )
    return csv_path


def make_spatial_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 7),
    )


@pytest.mark.parametrize(
    ("schedule", "epochs", "expected_lrs", "expect_scheduler_state"),
    [
        (SchedulerConfig(type="none"), 1, [0.0001], False),
        (
            SchedulerConfig(
                type="warmup_cosine",
                warmup_epochs=2,
                warmup_start_factor=0.1,
                min_learning_rate=0.000001,
            ),
            4,
            [0.00001, 0.0001, 0.0000505, 0.000001],
            True,
        ),
    ],
)
def test_synthetic_scheduler_modes_write_current_lr_and_train_accuracy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schedule: SchedulerConfig,
    epochs: int,
    expected_lrs: list[float],
    expect_scheduler_state: bool,
) -> None:
    csv_path = write_synthetic_training_csv(tmp_path)
    output_directory = tmp_path / f"output-{schedule.type}"
    original = make_config()
    config = replace(
        original,
        dataset=replace(original.dataset, path=str(csv_path)),
        model=replace(original.model, pretrained=False),
        training=replace(
            original.training,
            epochs=epochs,
            batch_size=2,
            num_workers=0,
            device="cpu",
            scheduler=schedule,
        ),
        output=replace(original.output, directory=str(output_directory)),
    )
    monkeypatch.setattr(
        "occlusion_fer.train.create_resnet18",
        lambda **kwargs: make_spatial_model(),
    )

    run_training(config)

    history = json.loads(
        (output_directory / "history.json").read_text(encoding="utf-8")
    )
    assert [row["learning_rate"] for row in history] == pytest.approx(
        expected_lrs
    )
    assert all(0.0 <= row["train_accuracy"] <= 1.0 for row in history)
    best = torch.load(
        output_directory / "best.pt", map_location="cpu", weights_only=False
    )
    last = torch.load(
        output_directory / "last.pt", map_location="cpu", weights_only=False
    )
    best_row = max(
        history,
        key=lambda row: (row["validation_macro_f1"], -row["epoch"]),
    )
    assert best["epoch"] == best_row["epoch"]
    assert best["validation_macro_f1"] == pytest.approx(
        best_row["validation_macro_f1"]
    )
    assert ("scheduler_state_dict" in last) is expect_scheduler_state
    if expect_scheduler_state:
        assert last["scheduler_state_dict"]["current_epoch"] == epochs


def test_run_training_early_stops_without_replacing_macro_f1_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = write_synthetic_training_csv(tmp_path)
    output_directory = tmp_path / "early-stopping-output"
    original = make_config()
    config = replace(
        original,
        dataset=replace(original.dataset, path=str(csv_path)),
        model=replace(original.model, pretrained=False),
        training=replace(
            original.training,
            epochs=5,
            batch_size=2,
            num_workers=0,
            device="cpu",
            early_stopping=EarlyStoppingConfig(enabled=True, patience=2),
        ),
        output=replace(original.output, directory=str(output_directory)),
    )
    monkeypatch.setattr(
        "occlusion_fer.train.create_resnet18",
        lambda **kwargs: make_spatial_model(),
    )
    template = evaluate(
        make_model(), make_loader(sample_count=2), torch.device("cpu")
    )
    macro_f1_values = iter([0.50, 0.60, 0.60, 0.59])
    accuracy_values = iter([0.40, 0.50, 0.90, 0.95])

    def controlled_evaluate(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return replace(
            template,
            macro_f1=next(macro_f1_values),
            accuracy=next(accuracy_values),
        )

    monkeypatch.setattr("occlusion_fer.train.evaluate", controlled_evaluate)

    run_training(config)

    history = json.loads(
        (output_directory / "history.json").read_text(encoding="utf-8")
    )
    best = torch.load(
        output_directory / "best.pt", map_location="cpu", weights_only=False
    )
    last = torch.load(
        output_directory / "last.pt", map_location="cpu", weights_only=False
    )
    assert len(history) == 4
    assert history[-1]["validation_accuracy"] == pytest.approx(0.95)
    assert best["epoch"] == 2
    assert best["validation_macro_f1"] == pytest.approx(0.60)
    assert best["validation_accuracy"] == pytest.approx(0.50)
    assert last["epoch"] == 4
