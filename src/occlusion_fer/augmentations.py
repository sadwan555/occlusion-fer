"""Deterministic-by-seed training-only FER2013 augmentations."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from occlusion_fer.config import AugmentationConfig
from occlusion_fer.data import Fer2013Split


class AugmentationError(ValueError):
    """Raised when an augmentation receives invalid input."""


def _draw_unit_interval() -> float:
    """Draw from torch's worker-seeded global generator."""
    return float(torch.rand((), dtype=torch.float32).item())


class IdentityAugmentation:
    """No-op transform that does not consume random numbers."""

    def __call__(self, image: Tensor) -> Tensor:
        return image


class MildAffineAugmentation:
    """Apply the locked E2 flip and affine operations to a [0, 1] tensor."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 3 or image.shape[0] != 1:
            raise AugmentationError("augmentation input must have shape (1, H, W)")
        if image.dtype is not torch.float32:
            raise AugmentationError("augmentation input must use float32")
        if not torch.isfinite(image).all().item():
            raise AugmentationError("augmentation input must be finite")

        if _draw_unit_interval() < self.config.horizontal_flip_probability:
            image = F.hflip(image)
        if _draw_unit_interval() >= self.config.affine_probability:
            return image

        height, width = image.shape[-2:]
        angle = self._uniform(-self.config.degrees, self.config.degrees)
        max_translate_x = math.floor(width * self.config.translate[0])
        max_translate_y = math.floor(height * self.config.translate[1])
        translation = [
            int((2 * self._uniform(0.0, 1.0) - 1) * max_translate_x),
            int((2 * self._uniform(0.0, 1.0) - 1) * max_translate_y),
        ]
        scale = self._uniform(*self.config.scale)
        return F.affine(
            image,
            angle,
            translation,
            scale,
            [0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=self.config.fill,
        )

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return low + (high - low) * _draw_unit_interval()


def build_augmentation(
    *,
    split: Fer2013Split,
    config: AugmentationConfig | None,
) -> Callable[[Tensor], Tensor]:
    """Build train augmentation while forcing evaluation splits to be deterministic."""
    selected = config or AugmentationConfig()
    if not isinstance(selected, AugmentationConfig):
        raise AugmentationError("augmentation config must be AugmentationConfig")
    if split != "train" or selected.type == "none":
        return IdentityAugmentation()
    if selected.type == "mild_affine":
        return MildAffineAugmentation(selected)
    raise AugmentationError(
        f"unsupported augmentation type: {selected.type!r}"
    )
