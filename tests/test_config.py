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
    assert config.model.name == "resnet18"
    assert config.model.pretrained is True
    assert config.training.mode == "clean"
    assert config.training.seed == 42
    assert config.training.epochs == 1
    assert config.training.batch_size == 32
    assert config.training.learning_rate == pytest.approx(0.0001)
    assert config.training.weight_decay == pytest.approx(0.0001)
    assert config.training.num_workers == 0
    assert config.training.device == "auto"
    assert config.output.directory == "outputs/smoke"


def test_repository_smoke_configuration_is_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_config(
        repository_root / "configs" / "fer2013_resnet18_clean.yaml"
    )

    assert config.model.name == "resnet18"
    assert config.output.directory == "outputs/smoke"


def test_rejects_empty_project_name(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("name: occlusion-fer", "name: ' '", 1)

    with pytest.raises(ConfigError, match=r"project\.name"):
        load_config(write_config(tmp_path, content))


def test_rejects_unsupported_dataset_name(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("name: fer2013", "name: another-dataset", 1)

    with pytest.raises(ConfigError, match=r"dataset\.name"):
        load_config(write_config(tmp_path, content))


def test_rejects_missing_dataset_path(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("  path: /path/to/fer2013.csv\n", "")

    with pytest.raises(ConfigError, match=r"dataset\.path"):
        load_config(write_config(tmp_path, content))


def test_rejects_empty_dataset_path(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace(
        "path: /path/to/fer2013.csv", "path: ' '", 1
    )

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


def test_rejects_unsupported_model_name(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("name: resnet18", "name: resnet50", 1)

    with pytest.raises(ConfigError, match=r"model\.name"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("pretrained_yaml", ["'true'", "1", "null"])
def test_rejects_pretrained_that_is_not_bool(
    tmp_path: Path, pretrained_yaml: str
) -> None:
    content = VALID_CONFIG.replace(
        "pretrained: true", f"pretrained: {pretrained_yaml}", 1
    )

    with pytest.raises(ConfigError, match=r"model\.pretrained.*bool"):
        load_config(write_config(tmp_path, content))


def test_rejects_unsupported_training_mode(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("mode: clean", "mode: mixed")

    with pytest.raises(ConfigError, match=r"training\.mode"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("seed_yaml", ["-1", "1.5", "true", "'42'"])
def test_rejects_invalid_seed(tmp_path: Path, seed_yaml: str) -> None:
    content = VALID_CONFIG.replace("seed: 42", f"seed: {seed_yaml}")

    with pytest.raises(
        ConfigError, match=r"training\.seed.*non-negative integer"
    ):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("field", "valid_value"),
    [("epochs", "1"), ("batch_size", "32")],
)
@pytest.mark.parametrize("invalid_value", ["0", "-1", "1.5", "true", "'1'"])
def test_rejects_training_positive_integer_fields(
    tmp_path: Path,
    field: str,
    valid_value: str,
    invalid_value: str,
) -> None:
    content = VALID_CONFIG.replace(
        f"{field}: {valid_value}", f"{field}: {invalid_value}", 1
    )

    with pytest.raises(
        ConfigError, match=rf"training\.{field}.*positive integer"
    ):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("invalid_value", "expected_message"),
    [
        ("0", "greater than 0"),
        ("-1", "greater than 0"),
        ("true", "must be a number"),
        ("'0.1'", "must be a number"),
    ],
)
def test_rejects_invalid_learning_rate(
    tmp_path: Path, invalid_value: str, expected_message: str
) -> None:
    content = VALID_CONFIG.replace(
        "learning_rate: 0.0001", f"learning_rate: {invalid_value}", 1
    )

    with pytest.raises(
        ConfigError, match=rf"training\.learning_rate.*{expected_message}"
    ):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("invalid_value", "expected_message"),
    [
        ("-1", "greater than or equal to 0"),
        ("true", "must be a number"),
        ("'0.1'", "must be a number"),
    ],
)
def test_rejects_invalid_weight_decay(
    tmp_path: Path, invalid_value: str, expected_message: str
) -> None:
    content = VALID_CONFIG.replace(
        "weight_decay: 0.0001", f"weight_decay: {invalid_value}", 1
    )

    with pytest.raises(
        ConfigError, match=rf"training\.weight_decay.*{expected_message}"
    ):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("invalid_value", ["-1", "1.5", "true", "'0'"])
def test_rejects_invalid_num_workers(
    tmp_path: Path, invalid_value: str
) -> None:
    content = VALID_CONFIG.replace(
        "num_workers: 0", f"num_workers: {invalid_value}", 1
    )

    with pytest.raises(
        ConfigError, match=r"training\.num_workers.*non-negative integer"
    ):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("device", ["gpu", "metal", "AUTO"])
def test_rejects_unsupported_device(tmp_path: Path, device: str) -> None:
    content = VALID_CONFIG.replace("device: auto", f"device: {device}", 1)

    with pytest.raises(ConfigError, match=r"training\.device.*auto.*cpu.*mps.*cuda"):
        load_config(write_config(tmp_path, content))


def test_rejects_empty_output_directory(tmp_path: Path) -> None:
    content = VALID_CONFIG.replace("directory: outputs/smoke", "directory: ' '", 1)

    with pytest.raises(ConfigError, match=r"output\.directory"):
        load_config(write_config(tmp_path, content))


def test_reports_missing_yaml_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(missing_path)


def test_reports_invalid_yaml(tmp_path: Path) -> None:
    content = "project: [\ndataset:"

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(write_config(tmp_path, content))
