"""Configuration loading for occlusion-fer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: str
    image_size: int
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    seed: int


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    dataset: DatasetConfig
    training: TrainingConfig


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
    training = _require_mapping(
        _require_field(root, "training", "training"), "training"
    )

    project_name = _require_string(project, "name", "project.name")

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

    training_mode = _require_string(training, "mode", "training.mode")
    if training_mode != "clean":
        raise ConfigError("training.mode must be 'clean'")

    seed = _require_integer(training, "seed", "training.seed")
    if seed < 0:
        raise ConfigError("training.seed must be a non-negative integer")

    return AppConfig(
        project=ProjectConfig(name=project_name),
        dataset=DatasetConfig(
            name=dataset_name,
            path=dataset_path,
            image_size=image_size,
            num_classes=num_classes,
        ),
        training=TrainingConfig(mode=training_mode, seed=seed),
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
