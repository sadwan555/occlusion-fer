from __future__ import annotations

import pytest
import torch
from torch import nn

from occlusion_fer.gradcam import GradCAMResult, compute_gradcam
from occlusion_fer.models import create_resnet18


class SpatialCNN(nn.Module):
    """Small deterministic classifier with an inspectable spatial target layer."""

    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2, classes, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]], [[0.0]]],
                        [[[0.0]], [[1.0]], [[0.0]]],
                    ]
                )
            )
            self.classifier.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        activation = self.features(image)
        return self.classifier(self.pool(activation).flatten(1))


def make_image() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1.0, 0.0, 0.5], [0.0, 0.5, 1.0], [0.5, 1.0, 0.0]],
                [[0.0, 1.0, 0.5], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]],
                [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
            ]
        ],
        dtype=torch.float32,
    )


def test_compute_gradcam_returns_normalized_spatial_result() -> None:
    model = SpatialCNN().eval()

    result = compute_gradcam(
        model,
        make_image(),
        target_layer=model.features,
        target_class=0,
    )

    assert isinstance(result, GradCAMResult)
    assert result.cam.shape == (3, 3)
    assert result.cam.dtype == torch.float32
    assert torch.isfinite(result.cam).all()
    assert result.cam.min().item() >= 0.0
    assert result.cam.max().item() <= 1.0
    assert result.cam.min().item() == pytest.approx(0.0)
    assert result.cam.max().item() == pytest.approx(1.0)
    assert result.target_class == 0
    assert result.predicted_class == 0
    assert result.target_logit == pytest.approx(0.5)
    assert result.predicted_probability == pytest.approx(0.5)


def test_target_class_changes_gradient_path_and_is_deterministic() -> None:
    model = SpatialCNN().eval()
    image = make_image()

    class_zero = compute_gradcam(
        model, image, target_layer=model.features, target_class=0
    )
    class_one = compute_gradcam(
        model, image, target_layer=model.features, target_class=1
    )
    repeated = compute_gradcam(
        model, image, target_layer=model.features, target_class=0
    )

    assert not torch.equal(class_zero.cam, class_one.cam)
    torch.testing.assert_close(class_zero.cam, repeated.cam, rtol=0, atol=0)


def test_zero_cam_is_finite_and_zero() -> None:
    model = SpatialCNN().eval()
    with torch.no_grad():
        model.classifier.weight.zero_()

    result = compute_gradcam(
        model, make_image(), target_layer=model.features, target_class=0
    )

    assert torch.equal(result.cam, torch.zeros_like(result.cam))
    assert result.predicted_class == 0
    assert result.predicted_probability == pytest.approx(0.5)


def test_rejects_invalid_images_and_target_classes() -> None:
    model = SpatialCNN().eval()
    cases = (
        (make_image().to(torch.int64), "floating"),
        (
            make_image().masked_fill(
                torch.ones_like(make_image(), dtype=torch.bool), float("nan")
            ),
            "finite",
        ),
        (
            make_image().masked_fill(
                torch.ones_like(make_image(), dtype=torch.bool), float("inf")
            ),
            "finite",
        ),
        (make_image()[:, :2], "3 channels"),
        (make_image().repeat(2, 1, 1, 1), "batch size"),
        (make_image().squeeze(0), "shape"),
    )
    for image, message in cases:
        with pytest.raises(ValueError, match=message):
            compute_gradcam(
                model, image, target_layer=model.features, target_class=0
            )

    for target_class in (-1, 2, True, 1.0):
        with pytest.raises(ValueError, match="target_class"):
            compute_gradcam(
                model,
                make_image(),
                target_layer=model.features,
                target_class=target_class,
            )


def test_rejects_invalid_model_outputs_and_target_layer() -> None:
    class BadOutput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Conv2d(3, 2, 1)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.layer(image)

    class VectorOutput(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.mean(dim=(1, 2, 3))

    spatial_model = SpatialCNN().eval()
    with pytest.raises(ValueError, match="logits shape"):
        compute_gradcam(
            BadOutput().eval(),
            make_image(),
            target_layer=spatial_model.features,
            target_class=0,
        )
    with pytest.raises(ValueError, match="logits shape"):
        compute_gradcam(
            VectorOutput().eval(),
            make_image(),
            target_layer=nn.Identity(),
            target_class=0,
        )
    with pytest.raises(ValueError, match="spatial"):
        compute_gradcam(
            spatial_model,
            make_image(),
            target_layer=spatial_model.classifier,
            target_class=0,
        )


def test_parameters_input_and_state_are_unchanged_and_hooks_cleanup() -> None:
    model = SpatialCNN()
    model.features.eval()
    model.classifier.train()
    (model(make_image()).sum()).backward()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    gradients_before = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in model.parameters()
    ]
    image = make_image()
    image_before = image.clone()
    original_states = [module.training for module in model.modules()]

    first = compute_gradcam(
        model, image, target_layer=model.features, target_class=0
    )
    second = compute_gradcam(
        model, image, target_layer=model.features, target_class=0
    )

    assert all(
        torch.equal(parameter, snapshot)
        for parameter, snapshot in zip(model.parameters(), before)
    )
    assert all(
        (parameter.grad is None and snapshot is None)
        or (parameter.grad is not None and snapshot is not None and torch.equal(parameter.grad, snapshot))
        for parameter, snapshot in zip(model.parameters(), gradients_before)
    )
    assert all(module.training == state for module, state in zip(model.modules(), original_states))
    torch.testing.assert_close(image, image_before, rtol=0, atol=0)
    torch.testing.assert_close(first.cam, second.cam, rtol=0, atol=0)
    assert all(
        not getattr(module, "_forward_hooks", {})
        for module in model.modules()
    )


def test_resnet18_layer4_gradcam_on_cpu_without_weights() -> None:
    model = create_resnet18(num_classes=7, pretrained=False).eval()
    result = compute_gradcam(
        model,
        torch.randn(1, 3, 112, 112),
        target_layer=model.layer4[-1],
        target_class=3,
    )

    assert result.cam.shape == (112, 112)
    assert torch.isfinite(result.cam).all()
    assert 0.0 <= result.cam.min().item() <= result.cam.max().item() <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_resnet18_layer4_gradcam_on_cuda() -> None:
    device = torch.device("cuda")
    model = create_resnet18(num_classes=7, pretrained=False).to(device).eval()
    result = compute_gradcam(
        model,
        torch.randn(1, 3, 112, 112, device=device),
        target_layer=model.layer4[-1],
        target_class=3,
    )

    assert result.cam.device == device
    assert result.cam.shape == (112, 112)
    assert torch.isfinite(result.cam).all()
    assert 0.0 <= result.cam.min().item() <= result.cam.max().item() <= 1.0
