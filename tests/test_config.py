from dataclasses import asdict
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


def add_training_settings(content: str, settings: str) -> str:
    return content.replace(
        "  device: auto\n",
        f"  device: auto\n{settings}",
        1,
    )


def add_dataset_settings(content: str, settings: str) -> str:
    return content.replace(
        "  num_classes: 7\n",
        f"  num_classes: 7\n{settings}",
        1,
    )


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
    assert config.training.loss.type == "cross_entropy"
    assert config.training.loss.label_smoothing == pytest.approx(0.0)
    assert config.training.num_workers == 0
    assert config.training.device == "auto"
    assert config.training.scheduler.type == "none"
    assert config.training.scheduler.warmup_epochs == 0
    assert config.training.early_stopping.enabled is False
    assert config.dataset.augmentation.type == "none"
    assert config.output.directory == "outputs/smoke"


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 1.0])
def test_loads_cross_entropy_label_smoothing(
    tmp_path: Path, label_smoothing: float
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        f"""\
  loss:
    type: cross_entropy
    label_smoothing: {label_smoothing}
""",
    )

    config = load_config(write_config(tmp_path, content))

    assert config.training.loss.type == "cross_entropy"
    assert config.training.loss.label_smoothing == pytest.approx(
        label_smoothing
    )


@pytest.mark.parametrize(
    "label_smoothing", ["-0.1", "1.1", ".nan", ".inf", "'0.1'"]
)
def test_rejects_invalid_label_smoothing(
    tmp_path: Path, label_smoothing: str
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        f"""\
  loss:
    type: cross_entropy
    label_smoothing: {label_smoothing}
""",
    )

    with pytest.raises(
        ConfigError, match=r"training\.loss\.label_smoothing"
    ):
        load_config(write_config(tmp_path, content))


def test_rejects_unknown_loss_type(tmp_path: Path) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        """\
  loss:
    type: focal
    label_smoothing: 0.1
""",
    )

    with pytest.raises(ConfigError, match=r"training\.loss\.type"):
        load_config(write_config(tmp_path, content))


def test_loads_mild_affine_augmentation(tmp_path: Path) -> None:
    content = add_dataset_settings(
        VALID_CONFIG,
        """\
  augmentation:
    type: mild_affine
    horizontal_flip_probability: 0.5
    affine_probability: 0.5
    degrees: 7.0
    translate: [0.05, 0.05]
    scale: [0.97, 1.03]
    interpolation: bilinear
    fill: 0.0
""",
    )

    config = load_config(write_config(tmp_path, content))

    augmentation = config.dataset.augmentation
    assert augmentation.type == "mild_affine"
    assert augmentation.horizontal_flip_probability == pytest.approx(0.5)
    assert augmentation.affine_probability == pytest.approx(0.5)
    assert augmentation.degrees == pytest.approx(7.0)
    assert augmentation.translate == pytest.approx((0.05, 0.05))
    assert augmentation.scale == pytest.approx((0.97, 1.03))
    assert augmentation.interpolation == "bilinear"
    assert augmentation.fill == pytest.approx(0.0)


def test_rejects_unknown_augmentation_type(tmp_path: Path) -> None:
    content = add_dataset_settings(
        VALID_CONFIG,
        """\
  augmentation:
    type: random_erasing
""",
    )

    with pytest.raises(ConfigError, match=r"dataset\.augmentation\.type"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_flip_probability", -0.1),
        ("horizontal_flip_probability", 1.1),
        ("affine_probability", -0.1),
        ("affine_probability", 1.1),
        ("degrees", -1.0),
        ("fill", "nan"),
    ],
)
def test_rejects_invalid_augmentation_scalar(
    tmp_path: Path, field: str, value: object
) -> None:
    rendered = repr(value) if isinstance(value, str) else str(value)
    content = add_dataset_settings(
        VALID_CONFIG,
        f"""\
  augmentation:
    type: mild_affine
    {field}: {rendered}
""",
    )

    with pytest.raises(ConfigError, match=r"dataset\.augmentation"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("translate", ["[-0.01, 0.05]", "[0.05, 1.01]"])
def test_rejects_augmentation_translation_out_of_range(
    tmp_path: Path, translate: str
) -> None:
    content = add_dataset_settings(
        VALID_CONFIG,
        f"""\
  augmentation:
    type: mild_affine
    translate: {translate}
""",
    )

    with pytest.raises(ConfigError, match=r"dataset\.augmentation\.translate"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("scale", ["[0.0, 1.0]", "[1.03, 0.97]", "[0.9]"])
def test_rejects_augmentation_scale(tmp_path: Path, scale: str) -> None:
    content = add_dataset_settings(
        VALID_CONFIG,
        f"""\
  augmentation:
    type: mild_affine
    scale: {scale}
""",
    )

    with pytest.raises(ConfigError, match=r"dataset\.augmentation\.scale"):
        load_config(write_config(tmp_path, content))


def test_rejects_non_bilinear_augmentation_interpolation(tmp_path: Path) -> None:
    content = add_dataset_settings(
        VALID_CONFIG,
        """\
  augmentation:
    type: mild_affine
    interpolation: nearest
""",
    )

    with pytest.raises(
        ConfigError, match=r"dataset\.augmentation\.interpolation"
    ):
        load_config(write_config(tmp_path, content))


def test_loads_explicit_none_scheduler_without_changing_base_lr(
    tmp_path: Path,
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        """\
  scheduler:
    type: none
    warmup_epochs: 0
    warmup_start_factor: 1.0
    min_learning_rate: 0.0001
  early_stopping:
    enabled: false
    patience: 0
""",
    )

    config = load_config(write_config(tmp_path, content))

    assert config.training.scheduler.type == "none"
    assert config.training.scheduler.min_learning_rate == pytest.approx(0.0001)
    assert config.training.learning_rate == pytest.approx(0.0001)
    assert config.training.early_stopping.enabled is False
    assert config.training.early_stopping.patience == 0


def test_loads_warmup_cosine_scheduler_and_early_stopping(tmp_path: Path) -> None:
    content = add_training_settings(
        VALID_CONFIG.replace(
            "  name: occlusion-fer\n",
            "  name: occlusion-fer\n  experiment_name: e1_warmup_cosine\n",
            1,
        ).replace("epochs: 1", "epochs: 30"),
        """\
  scheduler:
    type: warmup_cosine
    warmup_epochs: 3
    warmup_start_factor: 0.1
    min_learning_rate: 0.000001
  early_stopping:
    enabled: true
    patience: 8
""",
    )

    config = load_config(write_config(tmp_path, content))

    assert config.project.experiment_name == "e1_warmup_cosine"
    assert config.training.scheduler.type == "warmup_cosine"
    assert config.training.scheduler.warmup_epochs == 3
    assert config.training.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert config.training.scheduler.min_learning_rate == pytest.approx(0.000001)
    assert config.training.early_stopping.enabled is True
    assert config.training.early_stopping.patience == 8


def test_rejects_unknown_scheduler_type(tmp_path: Path) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        """\
  scheduler:
    type: one_cycle
""",
    )

    with pytest.raises(ConfigError, match=r"training\.scheduler\.type.*none.*warmup_cosine"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("warmup_epochs", [-1, 30, 31])
def test_rejects_warmup_epochs_outside_epoch_budget(
    tmp_path: Path, warmup_epochs: int
) -> None:
    content = add_training_settings(
        VALID_CONFIG.replace("epochs: 1", "epochs: 30"),
        f"""\
  scheduler:
    type: warmup_cosine
    warmup_epochs: {warmup_epochs}
""",
    )

    with pytest.raises(ConfigError, match=r"training\.scheduler\.warmup_epochs.*epochs"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("start_factor", [0, -0.1, 1.1])
def test_rejects_invalid_warmup_start_factor(
    tmp_path: Path, start_factor: float
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        f"""\
  scheduler:
    type: none
    warmup_start_factor: {start_factor}
""",
    )

    with pytest.raises(ConfigError, match=r"training\.scheduler\.warmup_start_factor"):
        load_config(write_config(tmp_path, content))


def test_rejects_min_learning_rate_above_base_learning_rate(
    tmp_path: Path,
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        """\
  scheduler:
    type: warmup_cosine
    min_learning_rate: 0.0002
""",
    )

    with pytest.raises(ConfigError, match=r"training\.scheduler\.min_learning_rate"):
        load_config(write_config(tmp_path, content))


@pytest.mark.parametrize("patience", [0, -1])
def test_rejects_invalid_enabled_early_stopping_patience(
    tmp_path: Path, patience: int
) -> None:
    content = add_training_settings(
        VALID_CONFIG,
        f"""\
  early_stopping:
    enabled: true
    patience: {patience}
""",
    )

    with pytest.raises(ConfigError, match=r"training\.early_stopping\.patience"):
        load_config(write_config(tmp_path, content))


def test_repository_smoke_configuration_is_valid() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_config(
        repository_root / "configs" / "fer2013_resnet18_clean.yaml"
    )

    assert config.model.name == "resnet18"
    assert config.output.directory == "outputs/smoke"


def test_repository_e0_e1_configs_differ_only_in_locked_fields() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "configs" / "experiments"

    e0 = load_config(config_root / "fer2013_resnet18_e0_baseline.yaml")
    e1 = load_config(
        config_root / "fer2013_resnet18_e1_warmup_cosine.yaml"
    )

    assert e0.project.experiment_name == "e0_baseline"
    assert e1.project.experiment_name == "e1_warmup_cosine"
    assert e0.training.scheduler.type == "none"
    assert e1.training.scheduler.type == "warmup_cosine"
    assert e1.training.scheduler.warmup_epochs == 3
    assert e1.training.scheduler.warmup_start_factor == pytest.approx(0.1)
    assert e1.training.scheduler.min_learning_rate == pytest.approx(1e-6)
    assert e0.training.early_stopping.enabled is False
    assert e1.training.early_stopping.enabled is False
    assert e0.dataset.augmentation.type == "none"
    assert e1.dataset.augmentation.type == "none"
    assert e0.training.loss.label_smoothing == pytest.approx(0.0)
    assert e1.training.loss.label_smoothing == pytest.approx(0.0)

    e0_values = asdict(e0)
    e1_values = asdict(e1)
    for values in (e0_values, e1_values):
        values["project"].pop("experiment_name")
        values["training"].pop("scheduler")
        values["output"].pop("directory")
    assert e0_values == e1_values


def test_repository_e2_differs_from_e0_only_in_allowed_fields() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "configs" / "experiments"

    e0 = load_config(config_root / "fer2013_resnet18_e0_baseline.yaml")
    e2 = load_config(
        config_root / "fer2013_resnet18_e2_mild_augmentation.yaml"
    )
    augmentation = e2.dataset.augmentation
    assert augmentation.type == "mild_affine"
    assert augmentation.horizontal_flip_probability == pytest.approx(0.5)
    assert augmentation.affine_probability == pytest.approx(0.5)
    assert augmentation.degrees == pytest.approx(7.0)
    assert augmentation.translate == pytest.approx((0.05, 0.05))
    assert augmentation.scale == pytest.approx((0.97, 1.03))
    assert augmentation.interpolation == "bilinear"
    assert augmentation.fill == pytest.approx(0.0)

    e0_values = asdict(e0)
    e2_values = asdict(e2)
    for values in (e0_values, e2_values):
        values["project"].pop("experiment_name")
        values["dataset"].pop("augmentation")
        values["output"].pop("directory")
    assert e0_values == e2_values


def test_repository_e3_differs_from_e0_only_in_allowed_fields() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "configs" / "experiments"

    e0 = load_config(config_root / "fer2013_resnet18_e0_baseline.yaml")
    e3 = load_config(config_root / "fer2013_resnet18_e3_regularization.yaml")
    assert e3.project.experiment_name == "e3_regularization"
    assert e3.dataset.augmentation.type == "none"
    assert e3.training.loss.type == "cross_entropy"
    assert e3.training.loss.label_smoothing == pytest.approx(0.1)
    assert e3.training.weight_decay == pytest.approx(0.001)
    assert e3.training.learning_rate == pytest.approx(0.0001)
    assert e3.training.scheduler.type == "none"
    assert e3.training.scheduler.warmup_epochs == 0
    assert e3.training.early_stopping.enabled is False

    e0_values = asdict(e0)
    e3_values = asdict(e3)
    for values in (e0_values, e3_values):
        values["project"].pop("experiment_name")
        values["output"].pop("directory")
        values["training"].pop("weight_decay")
        values["training"]["loss"].pop("label_smoothing")
    assert e0_values == e3_values


def test_repository_e4_combines_e2_and_e3_without_e1_scheduler() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "configs" / "experiments"

    e0 = load_config(config_root / "fer2013_resnet18_e0_baseline.yaml")
    e1 = load_config(
        config_root / "fer2013_resnet18_e1_warmup_cosine.yaml"
    )
    e2 = load_config(
        config_root / "fer2013_resnet18_e2_mild_augmentation.yaml"
    )
    e3 = load_config(config_root / "fer2013_resnet18_e3_regularization.yaml")
    e4 = load_config(config_root / "fer2013_resnet18_e4_combined.yaml")

    assert e4.project.name == "occlusion-fer"
    assert e4.project.experiment_name == "e4_combined"
    assert e4.output.directory == "outputs/screening/e4_combined"

    assert e4.dataset.augmentation == e2.dataset.augmentation
    assert e4.training.loss.type == "cross_entropy"
    assert e4.training.loss.label_smoothing == pytest.approx(0.1)
    assert e4.training.loss.label_smoothing == pytest.approx(
        e3.training.loss.label_smoothing
    )
    assert e4.training.weight_decay == pytest.approx(0.001)
    assert e4.training.weight_decay == pytest.approx(e3.training.weight_decay)

    assert e4.training.scheduler.type == "none"
    assert e4.training.scheduler.warmup_epochs == 0
    assert e4.training.scheduler.warmup_start_factor == pytest.approx(1.0)
    assert e4.training.scheduler.min_learning_rate == pytest.approx(0.0001)
    assert e4.training.scheduler != e1.training.scheduler
    assert e4.training.scheduler == e0.training.scheduler
    assert e4.training.early_stopping == e0.training.early_stopping

    e0_values = asdict(e0)
    e4_values = asdict(e4)
    for values in (e0_values, e4_values):
        values["project"].pop("experiment_name")
        values["dataset"].pop("augmentation")
        values["output"].pop("directory")
        values["training"].pop("weight_decay")
        values["training"]["loss"].pop("label_smoothing")
    assert e0_values == e4_values

    e2_values = asdict(e2)
    e4_values = asdict(e4)
    for values in (e2_values, e4_values):
        values["project"].pop("experiment_name")
        values["output"].pop("directory")
        values["training"].pop("weight_decay")
        values["training"]["loss"].pop("label_smoothing")
    assert e2_values == e4_values

    e3_values = asdict(e3)
    e4_values = asdict(e4)
    for values in (e3_values, e4_values):
        values["project"].pop("experiment_name")
        values["dataset"].pop("augmentation")
        values["output"].pop("directory")
    assert e3_values == e4_values


def test_repository_e0_e1_e2_default_to_unsmoothed_loss() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_root = repository_root / "configs" / "experiments"

    for name in (
        "fer2013_resnet18_e0_baseline.yaml",
        "fer2013_resnet18_e1_warmup_cosine.yaml",
        "fer2013_resnet18_e2_mild_augmentation.yaml",
    ):
        loss = load_config(config_root / name).training.loss
        assert loss.type == "cross_entropy"
        assert loss.label_smoothing == pytest.approx(0.0)


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
