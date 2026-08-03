import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from occlusion_fer.config import load_config
from occlusion_fer.final_evaluate import (
    load_checkpoint_model_state,
    parse_args,
    run_final_evaluation,
)
from occlusion_fer.models import create_resnet18


def write_config(tmp_path: Path, csv_path: Path, output: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""\
project:
  name: occlusion-fer
dataset:
  name: fer2013
  path: {csv_path}
  image_size: 112
  num_classes: 7
model:
  name: resnet18
  pretrained: false
training:
  mode: clean
  seed: 42
  epochs: 1
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 0
  device: cpu
output:
  directory: {output}
""",
        encoding="utf-8",
    )
    return config_path


def write_artificial_csv(tmp_path: Path) -> Path:
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
                (3, pixels, "PrivateTest"),
            ]
        )
    return csv_path


def test_parse_args_supports_explicit_private_test_options() -> None:
    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "best.pt",
            "--data-path",
            "fer2013.csv",
            "--output-dir",
            "outputs/run",
            "--device",
            "cuda",
            "--batch-size",
            "128",
            "--num-workers",
            "4",
            "--amp",
            "--confirm-private-test",
        ]
    )

    assert args.config == "config.yaml"
    assert args.checkpoint == "best.pt"
    assert args.data_path == "fer2013.csv"
    assert args.output_dir == "outputs/run"
    assert args.device == "cuda"
    assert args.batch_size == 128
    assert args.num_workers == 4
    assert args.amp is True
    assert args.confirm_private_test is True


def test_final_evaluation_refuses_before_loading_checkpoint_or_dataset(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "missing.csv"
    output = tmp_path / "output"
    config = load_config(write_config(tmp_path, csv_path, output))

    with pytest.raises(ValueError, match=r"--confirm-private-test"):
        run_final_evaluation(
            config,
            checkpoint_path=tmp_path / "missing.pt",
            confirm_private_test=False,
        )

    assert not output.exists()


def test_load_checkpoint_model_state_accepts_legacy_checkpoint(
    tmp_path: Path,
) -> None:
    model = create_resnet18(num_classes=7, pretrained=False)
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    state = load_checkpoint_model_state(checkpoint_path)

    assert set(state) == set(model.state_dict())


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"model_state_dict": "invalid"}],
)
def test_load_checkpoint_model_state_rejects_malformed_payload(
    tmp_path: Path,
    payload: object,
) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=r"checkpoint.*model_state_dict"):
        load_checkpoint_model_state(checkpoint_path)


def test_load_checkpoint_model_state_rejects_non_finite_tensor(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(
        {"model_state_dict": {"weight": torch.tensor([float("nan")])}},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match=r"non-finite.*weight"):
        load_checkpoint_model_state(checkpoint_path)


def test_final_evaluation_uses_only_private_test_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    csv_path = write_artificial_csv(tmp_path)
    output = tmp_path / "run"
    config = load_config(write_config(tmp_path, csv_path, output))
    config = replace(config, output=replace(config.output, directory=str(output)))
    model = create_resnet18(num_classes=7, pretrained=False)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()

    paths = run_final_evaluation(
        config,
        checkpoint_path=checkpoint_path,
        confirm_private_test=True,
    )

    assert set(paths) == {
        "metrics",
        "per_class_metrics",
        "confusion_matrix",
        "predictions",
    }
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["split"] == "test"
    assert metrics["condition"] == "clean"
    assert metrics["sample_count"] == 2
    with paths["predictions"].open(
        "r", encoding="utf-8", newline=""
    ) as predictions_file:
        prediction_rows = list(csv.DictReader(predictions_file))
    assert {row["sample_id"] for row in prediction_rows} == {"4", "5"}
    assert all(row["split"] == "test" for row in prediction_rows)
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_final_evaluation_refuses_to_overwrite_existing_results(
    tmp_path: Path,
) -> None:
    csv_path = write_artificial_csv(tmp_path)
    output = tmp_path / "run"
    config = load_config(write_config(tmp_path, csv_path, output))
    model = create_resnet18(num_classes=7, pretrained=False)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    existing = output / "final_test" / "clean_metrics.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match=r"already exist"):
        run_final_evaluation(
            config,
            checkpoint_path=checkpoint_path,
            confirm_private_test=True,
        )

    assert existing.read_text(encoding="utf-8") == "preserve"
