import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv_models

from occlusion_fer.data import Fer2013Data, Fer2013Record
from occlusion_fer.models import create_resnet18
from occlusion_fer.torch_data import Fer2013TorchDataset, create_dataloader


@pytest.fixture(scope="module")
def seven_class_model() -> nn.Module:
    return create_resnet18(num_classes=7, pretrained=False).eval()


def test_creates_standard_resnet18_without_pretrained_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resnet18 = tv_models.resnet18
    received_weights: list[object] = []

    def recording_resnet18(*, weights: object) -> nn.Module:
        received_weights.append(weights)
        return original_resnet18(weights=weights)

    monkeypatch.setattr(tv_models, "resnet18", recording_resnet18)

    model = create_resnet18(num_classes=7, pretrained=False)

    assert received_weights == [None]
    assert isinstance(model, tv_models.ResNet)
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert model.maxpool.kernel_size == 3
    assert model.fc.out_features == 7


@pytest.mark.parametrize("batch_size", [1, 4])
def test_resnet18_returns_finite_floating_logits_with_expected_shape(
    seven_class_model: nn.Module, batch_size: int
) -> None:
    images = torch.zeros(batch_size, 3, 112, 112)

    with torch.inference_mode():
        logits = seven_class_model(images)

    assert logits.shape == (batch_size, 7)
    assert logits.is_floating_point()
    assert torch.isfinite(logits).all()


def test_model_contains_no_probability_activation(
    seven_class_model: nn.Module,
) -> None:
    forbidden_types = (nn.Softmax, nn.LogSoftmax, nn.Sigmoid)

    assert not any(
        isinstance(module, forbidden_types)
        for module in seven_class_model.modules()
    )


def test_model_output_is_not_forced_to_sum_to_one() -> None:
    model = create_resnet18(num_classes=7, pretrained=False).eval()
    assert isinstance(model.fc, nn.Linear)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.arange(7, dtype=model.fc.bias.dtype))

    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 112, 112))

    torch.testing.assert_close(logits[0], torch.arange(7, dtype=logits.dtype))
    assert logits.sum().item() == pytest.approx(21.0)


def test_num_classes_controls_output_layer() -> None:
    model = create_resnet18(num_classes=3, pretrained=False).eval()

    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 112, 112))

    assert model.fc.out_features == 3
    assert logits.shape == (2, 3)


@pytest.mark.parametrize("num_classes", [0, -1, 1.5, True])
def test_rejects_invalid_num_classes(num_classes: object) -> None:
    with pytest.raises(ValueError, match=r"num_classes.*positive integer"):
        create_resnet18(num_classes=num_classes, pretrained=False)


@pytest.mark.parametrize("pretrained", [0, 1, "false", None])
def test_rejects_pretrained_that_is_not_bool(pretrained: object) -> None:
    with pytest.raises(ValueError, match=r"pretrained.*bool"):
        create_resnet18(num_classes=7, pretrained=pretrained)


def test_pretrained_weight_failure_is_reported_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def failing_resnet18(*, weights: object) -> nn.Module:
        calls.append(weights)
        raise OSError("simulated download failure")

    monkeypatch.setattr(tv_models, "resnet18", failing_resnet18)

    with pytest.raises(RuntimeError, match=r"pretrained ResNet-18 weights"):
        create_resnet18(num_classes=7, pretrained=True)

    assert calls == [tv_models.ResNet18_Weights.DEFAULT]


def test_small_batch_completes_forward_cross_entropy_and_backward() -> None:
    model = create_resnet18(num_classes=7, pretrained=False)
    logits = model(torch.rand(2, 3, 112, 112))
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_artificial_fer_data_reaches_resnet_loss_and_backward() -> None:
    records = tuple(
        Fer2013Record(
            sample_id=index + 1,
            label=index,
            label_name=("angry", "disgust")[index],
            split="train",
            image=np.full((48, 48), index * 64, dtype=np.uint8),
        )
        for index in range(2)
    )
    data = Fer2013Data(
        records=records,
        train_count=2,
        validation_count=0,
        test_count=0,
        class_counts={label: int(label < 2) for label in range(7)},
    )
    dataset = Fer2013TorchDataset(
        data,
        split="train",
        image_size=112,
        normalize_imagenet=True,
    )
    loader = create_dataloader(
        dataset, batch_size=2, shuffle=False, seed=42
    )
    images, labels, _ = next(iter(loader))
    model = create_resnet18(num_classes=7, pretrained=False)

    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    assert logits.shape == (2, 7)
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None for parameter in model.parameters()
    )
