"""PyTorch dataset and data loader helpers for FER2013."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from occlusion_fer.data import Fer2013Data, Fer2013Record, Fer2013Split


VALID_SPLITS: tuple[Fer2013Split, ...] = ("train", "validation", "test")
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
    ) -> None:
        if split not in VALID_SPLITS:
            allowed = ", ".join(VALID_SPLITS)
            raise TorchDataError(
                f"split must be one of {allowed}; got {split!r}"
            )
        _require_positive_integer(image_size, "image_size")

        self.split = cast(Fer2013Split, split)
        self.image_size = image_size
        self._records = tuple(
            record for record in data.records if record.split == self.split
        )
        if not self._records:
            raise TorchDataError(f"split '{self.split}' has no samples")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TorchSample:
        record = self._records[index]
        image = _record_to_tensor(record, self.image_size)
        return image, record.label, record.sample_id


def create_dataloader(
    dataset: Dataset[TorchSample],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[TorchSample]:
    """Create a DataLoader whose sample order is controlled by an explicit seed."""
    _require_positive_integer(batch_size, "batch_size")
    _require_nonnegative_integer(seed, "seed")
    _require_nonnegative_integer(num_workers, "num_workers")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _record_to_tensor(record: Fer2013Record, image_size: int) -> Tensor:
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

    image_array = np.array(record.image, copy=True)
    image = torch.from_numpy(image_array).to(dtype=torch.float32)
    image = image / 255.0
    image = image.unsqueeze(0).repeat(3, 1, 1)
    image = F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

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
