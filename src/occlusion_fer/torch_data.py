"""PyTorch dataset and data loader helpers for FER2013."""

from __future__ import annotations

import random
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from occlusion_fer.data import Fer2013Data, Fer2013Record, Fer2013Split


VALID_SPLITS: tuple[Fer2013Split, ...] = ("train", "validation", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_MEAN_TENSOR = torch.tensor(
    IMAGENET_MEAN, dtype=torch.float32
).view(3, 1, 1)
_IMAGENET_STD_TENSOR = torch.tensor(
    IMAGENET_STD, dtype=torch.float32
).view(3, 1, 1)
TorchSample = tuple[Tensor, int, int]


class TorchDataError(ValueError):
    """Raised when PyTorch data preparation input is invalid."""


class Fer2013TorchDataset(Dataset[TorchSample]):
    """Expose one FER2013 split as deterministic PyTorch samples."""

    def __init__(
        self,
        data: Fer2013Data,
        split: str,
        image_size: int = 112,
        normalize_imagenet: bool = False,
    ) -> None:
        if split not in VALID_SPLITS:
            allowed = ", ".join(VALID_SPLITS)
            raise TorchDataError(
                f"split must be one of {allowed}; got {split!r}"
            )
        _require_positive_integer(image_size, "image_size")
        if type(normalize_imagenet) is not bool:
            raise TorchDataError("normalize_imagenet must be a bool")

        self.split = cast(Fer2013Split, split)
        self.image_size = image_size
        self.normalize_imagenet = normalize_imagenet
        self._records = tuple(
            record for record in data.records if record.split == self.split
        )
        if not self._records:
            raise TorchDataError(f"split '{self.split}' has no samples")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TorchSample:
        record = self._records[index]
        image = _record_to_tensor(
            record, self.image_size, self.normalize_imagenet
        )
        return image, record.label, record.sample_id


def create_dataloader(
    dataset: Dataset[TorchSample],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader[TorchSample]:
    """Create a DataLoader whose sample order is controlled by an explicit seed."""
    _require_positive_integer(batch_size, "batch_size")
    _require_nonnegative_integer(seed, "seed")
    _require_nonnegative_integer(num_workers, "num_workers")
    _require_bool(pin_memory, "pin_memory")
    _require_bool(persistent_workers, "persistent_workers")
    _require_positive_integer(prefetch_factor, "prefetch_factor")

    generator = torch.Generator()
    generator.manual_seed(seed)
    uses_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and uses_workers,
        prefetch_factor=prefetch_factor if uses_workers else None,
        worker_init_fn=_initialize_worker if uses_workers else None,
    )


def _record_to_tensor(
    record: Fer2013Record,
    image_size: int,
    normalize_imagenet: bool,
) -> Tensor:
    if record.image.shape != (48, 48):
        raise TorchDataError(
            f"sample {record.sample_id} image must have shape (48, 48); "
            f"got {record.image.shape}"
        )
    if type(record.label) is not int or not 0 <= record.label <= 6:
        raise TorchDataError(
            f"sample {record.sample_id} label must be an integer from 0 to 6; "
            f"got {record.label!r}"
        )

    image = torch.from_numpy(record.image).to(dtype=torch.float32)
    image = image / 255.0
    image = F.interpolate(
        image.view(1, 1, 48, 48),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).view(1, image_size, image_size)
    image = image.expand(3, -1, -1).contiguous()
    if normalize_imagenet:
        image = (image - _IMAGENET_MEAN_TENSOR) / _IMAGENET_STD_TENSOR

    if not torch.isfinite(image).all().item():
        raise TorchDataError(
            f"sample {record.sample_id} converted image contains non-finite values"
        )
    return image


def _require_positive_integer(value: object, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TorchDataError(f"{field_name} must be a positive integer")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if type(value) is not int or value < 0:
        raise TorchDataError(f"{field_name} must be a non-negative integer")


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise TorchDataError(f"{field_name} must be a bool")


def _initialize_worker(worker_id: int) -> None:
    """Keep worker CPU use bounded and seed non-PyTorch random generators."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)
