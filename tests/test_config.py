from pathlib import Path

import pytest

from occlusion_fer.config import ConfigError, load_config


VALID_CONFIG = """\
project:
  name: occlusion-fer
dataset:
  name: fer2013
  path: /path/to/fer2013.csv
  image_size: 112
  num_classes: 7
training:
  mode: clean
  seed: 42
"""


def write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_loads_valid_configuration(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, VALID_CONFIG))

    assert config.project.name == "occlusion-fer"
    assert config.dataset.name == "fer2013"
    assert config.dataset.path == "/path/to/fer2013.csv"
    assert config.dataset.image_size == 112
    assert config.dataset.num_classes == 7
    assert config.training.mode == "clean"
    assert config.training.seed == 42


def test_rejects_missing_dataset_path(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("  path: /path/to/fer2013.csv\n", "")

    with pytest.raises(ConfigError, match=r"dataset\.path"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("image_size_yaml", ["0", "-1", "1.5", "'112'", "true"])
def test_rejects_image_size_that_is_not_a_positive_integer(
    tmp_path: Path, image_size_yaml: str
) -> None:
    content = VALID_CONFIG.replace("image_size: 112", f"image_size: {image_size_yaml}")

    with pytest.raises(ConfigError, match=r"dataset\.image_size"):
        load_config(write_config(tmp_path, content))


def test_rejects_num_classes_other_than_seven(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("num_classes: 7", "num_classes: 8")

    with pytest.raises(ConfigError, match=r"dataset\.num_classes"):
        load_config(write_config(tmp_path, content))


def test_rejects_unsupported_training_mode(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("mode: clean", "mode: mixed")

    with pytest.raises(ConfigError, match=r"training\.mode"):
        load_config(write_config(tmp_path, content))


def test_rejects_negative_seed(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("seed: 42", "seed: -1")

    with pytest.raises(ConfigError, match=r"training\.seed"):
        load_config(write_config(tmp_path, content))


def test_reports_missing_yaml_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(missing_path)


def test_reports_invalid_yaml(tmp_path: Path) -> None:
    content = "project: [\ndataset:"

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write_config(tmp_path, content))
