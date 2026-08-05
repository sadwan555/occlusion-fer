"""Configuration loading for occlusion-fer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    experiment_name: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str
    image_size: int
    num_classes: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: bool


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "none"
    warmup_epochs: int = 0
    warmup_start_factor: float = 1.0
    min_learning_rate: float = 0.0


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 1


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    device: str
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )


@dataclass(frozen=True)
class OutputConfig:
    directory: str


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file: {config_path}") from exc

    root = _require_mapping(raw_config, "configuration root")
    project = _require_mapping(_require_field(root, "project", "project"), "project")
    dataset = _require_mapping(_require_field(root, "dataset", "dataset"), "dataset")
    model = _require_mapping(_require_field(root, "model", "model"), "model")
    training = _require_mapping(
        _require_field(root, "training", "training"), "training"
    )
    output = _require_mapping(_require_field(root, "output", "output"), "output")

    project_name = _require_string(project, "name", "project.name")
    experiment_name = _optional_string(
        project,
        "experiment_name",
        "project.experiment_name",
        default=project_name,
    )

    dataset_name = _require_string(dataset, "name", "dataset.name")
    if dataset_name != "fer2013":
        raise ConfigError("dataset.name must be 'fer2013'")

    dataset_path = _require_string(dataset, "path", "dataset.path")
    image_size = _require_integer(dataset, "image_size", "dataset.image_size")
    if image_size <= 0:
        raise ConfigError("dataset.image_size must be a positive integer")

    num_classes = _require_integer(dataset, "num_classes", "dataset.num_classes")
    if num_classes != 7:
        raise ConfigError("dataset.num_classes must be 7")

    model_name = _require_string(model, "name", "model.name")
    if model_name != "resnet18":
        raise ConfigError("model.name must be 'resnet18'")
    pretrained = _require_bool(model, "pretrained", "model.pretrained")

    training_mode = _require_string(training, "mode", "training.mode")
    if training_mode != "clean":
        raise ConfigError("training.mode must be 'clean'")

    seed = _require_nonnegative_integer(training, "seed", "training.seed")
    epochs = _require_positive_integer(training, "epochs", "training.epochs")
    batch_size = _require_positive_integer(
        training, "batch_size", "training.batch_size"
    )
    learning_rate = _require_number(
        training, "learning_rate", "training.learning_rate"
    )
    if learning_rate <= 0:
        raise ConfigError("training.learning_rate must be greater than 0")
    weight_decay = _require_number(
        training, "weight_decay", "training.weight_decay"
    )
    if weight_decay < 0:
        raise ConfigError(
            "training.weight_decay must be greater than or equal to 0"
        )
    num_workers = _require_nonnegative_integer(
        training, "num_workers", "training.num_workers"
    )
    device = _require_string(training, "device", "training.device")
    allowed_devices = ("auto", "cpu", "mps", "cuda")
    if device not in allowed_devices:
        allowed = ", ".join(allowed_devices)
        raise ConfigError(f"training.device must be one of {allowed}")

    scheduler = _optional_mapping(
        training, "scheduler", "training.scheduler"
    )
    scheduler_type = _optional_string(
        scheduler,
        "type",
        "training.scheduler.type",
        default="none",
    )
    allowed_schedulers = ("none", "warmup_cosine")
    if scheduler_type not in allowed_schedulers:
        allowed = ", ".join(allowed_schedulers)
        raise ConfigError(
            f"training.scheduler.type must be one of {allowed}"
        )
    warmup_epochs = _optional_integer(
        scheduler,
        "warmup_epochs",
        "training.scheduler.warmup_epochs",
        default=0,
    )
    if not 0 <= warmup_epochs < epochs:
        raise ConfigError(
            "training.scheduler.warmup_epochs must satisfy "
            "0 <= warmup_epochs < training.epochs"
        )
    warmup_start_factor = _optional_number(
        scheduler,
        "warmup_start_factor",
        "training.scheduler.warmup_start_factor",
        default=1.0,
    )
    if not 0 < warmup_start_factor <= 1:
        raise ConfigError(
            "training.scheduler.warmup_start_factor must be greater than 0 "
            "and less than or equal to 1"
        )
    min_learning_rate = _optional_number(
        scheduler,
        "min_learning_rate",
        "training.scheduler.min_learning_rate",
        default=0.0,
    )
    if not 0 <= min_learning_rate <= learning_rate:
        raise ConfigError(
            "training.scheduler.min_learning_rate must satisfy "
            "0 <= min_learning_rate <= training.learning_rate"
        )

    early_stopping = _optional_mapping(
        training, "early_stopping", "training.early_stopping"
    )
    early_stopping_enabled = _optional_bool(
        early_stopping,
        "enabled",
        "training.early_stopping.enabled",
        default=False,
    )
    early_stopping_patience = _optional_integer(
        early_stopping,
        "patience",
        "training.early_stopping.patience",
        default=1,
    )
    if early_stopping_enabled and early_stopping_patience < 1:
        raise ConfigError(
            "training.early_stopping.patience must be at least 1 when "
            "early stopping is enabled"
        )

    output_directory = _require_string(output, "directory", "output.directory")

    return AppConfig(
        project=ProjectConfig(
            name=project_name,
            experiment_name=experiment_name,
        ),
        dataset=DatasetConfig(
            name=dataset_name,
            path=dataset_path,
            image_size=image_size,
            num_classes=num_classes,
        ),
        model=ModelConfig(name=model_name, pretrained=pretrained),
        training=TrainingConfig(
            mode=training_mode,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            num_workers=num_workers,
            device=device,
            scheduler=SchedulerConfig(
                type=scheduler_type,
                warmup_epochs=warmup_epochs,
                warmup_start_factor=float(warmup_start_factor),
                min_learning_rate=float(min_learning_rate),
            ),
            early_stopping=EarlyStoppingConfig(
                enabled=early_stopping_enabled,
                patience=early_stopping_patience,
            ),
        ),
        output=OutputConfig(directory=output_directory),
    )


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a mapping")
    return value


def _require_field(
    mapping: Mapping[str, object], key: str, field_name: str
) -> object:
    if key not in mapping:
        raise ConfigError(f"Missing required field: {field_name}")
    return mapping[key]


def _require_string(
    mapping: Mapping[str, object], key: str, field_name: str
) -> str:
    value = _require_field(mapping, key, field_name)
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    if not value.strip():
        raise ConfigError(f"{field_name} must not be empty")
    return value


def _require_integer(
    mapping: Mapping[str, object], key: str, field_name: str
) -> int:
    value = _require_field(mapping, key, field_name)
    if type(value) is not int:
        raise ConfigError(f"{field_name} must be an integer")
    return value


def _require_positive_integer(
    mapping: Mapping[str, object], key: str, field_name: str
) -> int:
    value = _require_field(mapping, key, field_name)
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_integer(
    mapping: Mapping[str, object], key: str, field_name: str
) -> int:
    value = _require_field(mapping, key, field_name)
    if type(value) is not int or value < 0:
        raise ConfigError(f"{field_name} must be a non-negative integer")
    return value


def _require_number(
    mapping: Mapping[str, object], key: str, field_name: str
) -> int | float:
    value = _require_field(mapping, key, field_name)
    if type(value) not in (int, float):
        raise ConfigError(f"{field_name} must be a number")
    return value


def _require_bool(
    mapping: Mapping[str, object], key: str, field_name: str
) -> bool:
    value = _require_field(mapping, key, field_name)
    if type(value) is not bool:
        raise ConfigError(f"{field_name} must be a bool")
    return value


def _optional_mapping(
    mapping: Mapping[str, object], key: str, field_name: str
) -> Mapping[str, object]:
    if key not in mapping:
        return {}
    return _require_mapping(mapping[key], field_name)


def _optional_string(
    mapping: Mapping[str, object],
    key: str,
    field_name: str,
    *,
    default: str,
) -> str:
    if key not in mapping:
        return default
    return _require_string(mapping, key, field_name)


def _optional_integer(
    mapping: Mapping[str, object],
    key: str,
    field_name: str,
    *,
    default: int,
) -> int:
    if key not in mapping:
        return default
    return _require_integer(mapping, key, field_name)


def _optional_number(
    mapping: Mapping[str, object],
    key: str,
    field_name: str,
    *,
    default: float,
) -> int | float:
    if key not in mapping:
        return default
    return _require_number(mapping, key, field_name)


def _optional_bool(
    mapping: Mapping[str, object],
    key: str,
    field_name: str,
    *,
    default: bool,
) -> bool:
    if key not in mapping:
        return default
    return _require_bool(mapping, key, field_name)
