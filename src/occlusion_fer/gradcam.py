"""Optional post-hoc Grad-CAM analysis for already-prepared model inputs.

This module deliberately has no dataset, occlusion, checkpoint, or training
dependencies.  Callers choose the target class and provide the target layer.
"""

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["GradCAMResult", "compute_gradcam"]


@dataclass(frozen=True)
class GradCAMResult:
    """Grad-CAM map and metadata for one model input."""

    cam: Tensor
    target_class: int
    predicted_class: int
    target_logit: float
    predicted_probability: float


def _validate_image(image: Tensor) -> tuple[int, int]:
    if not isinstance(image, Tensor):
        raise ValueError("image must be a torch.Tensor")
    if not image.is_floating_point():
        raise ValueError("image must be a floating tensor")
    if image.ndim != 4:
        raise ValueError("image shape must be [1, 3, H, W]")
    if image.shape[0] != 1:
        raise ValueError("image batch size must be 1")
    if image.shape[1] != 3:
        raise ValueError("image must have 3 channels")
    if image.shape[2] <= 0 or image.shape[3] <= 0:
        raise ValueError("image spatial dimensions must be positive")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return int(image.shape[2]), int(image.shape[3])


def _validate_target_class(target_class: int) -> int:
    if not isinstance(target_class, Integral) or isinstance(target_class, bool):
        raise ValueError("target_class must be an integer")
    return int(target_class)


def _restore_training_states(states: dict[nn.Module, bool]) -> None:
    # Assign the flags directly so mixed per-submodule train/eval states survive.
    for module, training in states.items():
        module.training = training


def compute_gradcam(
    model: nn.Module,
    image: Tensor,
    *,
    target_layer: nn.Module,
    target_class: int,
) -> GradCAMResult:
    """Compute a normalized Grad-CAM map for one ``[1, 3, H, W]`` image.

    The model is evaluated without changing its parameters or persistent
    gradients.  ``target_class`` is intentionally explicit; prediction is
    returned only as metadata and never selects the explained class.
    """

    height, width = _validate_image(image)
    target_class = _validate_target_class(target_class)
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if not isinstance(target_layer, nn.Module):
        raise ValueError("target_layer must be a torch.nn.Module")

    training_states = {module: module.training for module in model.modules()}
    activation: Tensor | None = None

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: object,
    ) -> None:
        nonlocal activation
        if not isinstance(output, Tensor):
            raise ValueError("target layer must produce a tensor activation")
        activation = output

    hook = target_layer.register_forward_hook(capture_activation)
    try:
        # A cloned input keeps the caller's tensor untouched and supplies a
        # differentiable path even when all model parameters are frozen.
        input_for_grad = image.detach().clone().requires_grad_(True)
        model.eval()
        try:
            logits = model(input_for_grad)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Grad-CAM model forward failed") from exc

        if (
            not isinstance(logits, Tensor)
            or logits.ndim != 2
            or logits.shape[0] != 1
            or logits.shape[1] <= 0
        ):
            raise ValueError("model logits shape must be [1, num_classes]")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("model logits must be finite floating values")
        if target_class < 0 or target_class >= logits.shape[1]:
            raise ValueError("target_class is outside the model logits range")
        if activation is None:
            raise ValueError("target layer did not produce an activation")
        if (
            activation.ndim != 4
            or activation.shape[0] != 1
            or activation.shape[1] <= 0
            or activation.shape[2] <= 0
            or activation.shape[3] <= 0
        ):
            raise ValueError("target layer must produce a spatial activation")
        if (
            not activation.is_floating_point()
            or not torch.isfinite(activation).all()
        ):
            raise ValueError(
                "target-layer activations must be finite floating values"
            )

        target_logit_tensor = logits[0, target_class]
        try:
            (gradients,) = torch.autograd.grad(
                target_logit_tensor,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError("could not compute gradients for target layer") from exc
        if (
            gradients.shape != activation.shape
            or not torch.isfinite(gradients).all()
        ):
            raise ValueError(
                "target-layer gradients must be finite and match activations"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        raw_cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        resized_cam = F.interpolate(
            raw_cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        if not torch.isfinite(resized_cam).all():
            raise ValueError("Grad-CAM values must be finite")
        minimum = resized_cam.min()
        maximum = resized_cam.max()
        if torch.equal(minimum, maximum):
            cam = torch.zeros_like(resized_cam)
        else:
            cam = (resized_cam - minimum) / (maximum - minimum)
            cam = cam.clamp(0.0, 1.0)
        if not torch.isfinite(cam).all():
            raise ValueError("normalized Grad-CAM values must be finite")

        probabilities = torch.softmax(logits.detach(), dim=1)
        predicted_class = int(torch.argmax(logits.detach(), dim=1).item())
        return GradCAMResult(
            cam=cam.detach(),
            target_class=target_class,
            predicted_class=predicted_class,
            target_logit=float(target_logit_tensor.detach().item()),
            predicted_probability=float(probabilities[0, predicted_class].item()),
        )
    finally:
        hook.remove()
        _restore_training_states(training_states)
