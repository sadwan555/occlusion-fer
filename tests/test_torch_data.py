from collections.abc import Iterable

import numpy as np
import pytest
import torch

import occlusion_fer.torch_data as torch_data_module
from occlusion_fer.data import Fer2013Data, Fer2013Record, Fer2013Split
from occlusion_fer.torch_data import (
    Fer2013TorchDataset,
    TorchDataError,
    create_dataloader,
)


LABEL_NAMES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


def varied_image(offset: int = 0) -> np.ndarray:
    values = np.arange(48 * 48, dtype=np.uint16).reshape(48, 48)
    return ((values + offset) % 256).astype(np.uint8)


def make_record(
    sample_id: int,
    label: int,
    split: Fer2013Split,
    image: np.ndarray | None = None,
) -> Fer2013Record:
    label_name = LABEL_NAMES[label] if type(label) is int and 0 <= label <= 6 else "invalid"
    return Fer2013Record(
        sample_id=sample_id,
        label=label,
        label_name=label_name,
        split=split,
        image=varied_image(sample_id) if image is None else image,
    )


def make_data(records: Iterable[Fer2013Record] | None = None) -> Fer2013Data:
    selected_records = tuple(records) if records is not None else (
        make_record(101, 0, "train"),
        make_record(102, 1, "train"),
        make_record(103, 2, "train"),
        make_record(104, 3, "train"),
        make_record(105, 4, "train"),
        make_record(106, 5, "train"),
        make_record(107, 6, "train"),
        make_record(108, 0, "train"),
        make_record(201, 1, "validation"),
        make_record(202, 2, "validation"),
        make_record(301, 3, "test"),
        make_record(302, 4, "test"),
    )
    return Fer2013Data(
        records=selected_records,
        train_count=sum(record.split == "train" for record in selected_records),
        validation_count=sum(
            record.split == "validation" for record in selected_records
        ),
        test_count=sum(record.split == "test" for record in selected_records),
        class_counts={
            label: sum(record.label == label for record in selected_records)
            for label in range(7)
        },
    )


def sample_ids(dataset: Fer2013TorchDataset) -> list[int]:
    return [dataset[index][2] for index in range(len(dataset))]


def loader_sample_ids(loader: torch.utils.data.DataLoader) -> list[int]:
    batches = [batch_ids for _, _, batch_ids in loader]
    return torch.cat(batches).tolist()


@pytest.mark.parametrize(
    ("split", "expected_ids"),
    [
        ("train", [101, 102, 103, 104, 105, 106, 107, 108]),
        ("validation", [201, 202]),
        ("test", [301, 302]),
    ],
)
def test_dataset_contains_only_requested_split(
    split: Fer2013Split, expected_ids: list[int]
) -> None:
    dataset = Fer2013TorchDataset(make_data(), split=split)

    assert len(dataset) == len(expected_ids)
    assert sample_ids(dataset) == expected_ids


def test_image_has_three_channel_target_shape() -> None:
    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]

    assert image.shape == (3, 112, 112)


def test_image_size_parameter_controls_output_shape() -> None:
    image, _, _ = Fer2013TorchDataset(
        make_data(), split="train", image_size=64
    )[0]

    assert image.shape == (3, 64, 64)


def test_image_uses_float32_dtype() -> None:
    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]

    assert image.dtype == torch.float32


def test_image_pixels_are_scaled_to_zero_one_range() -> None:
    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]

    assert image.min().item() >= 0.0
    assert image.max().item() <= 1.0
    assert torch.unique(image[0]).numel() > 1


def test_resize_uses_interpolation_instead_of_nearest_copying() -> None:
    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]
    rescaled_pixels = image[0] * 255.0

    assert torch.any(rescaled_pixels != rescaled_pixels.round())


def test_resize_interpolates_one_channel_before_replication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interpolate_input_shapes: list[tuple[int, ...]] = []
    original_interpolate = torch_data_module.F.interpolate

    def recording_interpolate(
        image: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        interpolate_input_shapes.append(tuple(image.shape))
        return original_interpolate(image, *args, **kwargs)

    monkeypatch.setattr(
        torch_data_module.F, "interpolate", recording_interpolate
    )

    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]

    assert interpolate_input_shapes == [(1, 1, 48, 48)]
    assert image.shape == (3, 112, 112)


def test_three_channels_have_identical_content() -> None:
    image, _, _ = Fer2013TorchDataset(make_data(), split="train")[0]

    assert torch.equal(image[0], image[1])
    assert torch.equal(image[1], image[2])


def test_imagenet_normalization_has_expected_shape_dtype_and_values() -> None:
    source_image = np.full((48, 48), 255, dtype=np.uint8)
    data = make_data([make_record(101, 0, "train", source_image)])
    image, _, _ = Fer2013TorchDataset(
        data,
        split="train",
        image_size=48,
        normalize_imagenet=True,
    )[0]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    expected = (torch.ones(3, dtype=torch.float32) - mean) / std

    assert image.shape == (3, 48, 48)
    assert image.dtype == torch.float32
    torch.testing.assert_close(image[:, 0, 0], expected)
    assert not torch.equal(image[0], image[1])
    assert not torch.equal(image[1], image[2])


def test_imagenet_normalization_does_not_modify_original_numpy_image() -> None:
    data = make_data()
    original_image = data.records[0].image.copy()
    dataset = Fer2013TorchDataset(
        data, split="train", normalize_imagenet=True
    )

    dataset[0]

    np.testing.assert_array_equal(data.records[0].image, original_image)


def test_explicitly_disabling_imagenet_normalization_preserves_old_behavior() -> None:
    data = make_data()
    default_image, _, _ = Fer2013TorchDataset(data, split="train")[0]
    unnormalized_image, _, _ = Fer2013TorchDataset(
        data, split="train", normalize_imagenet=False
    )[0]

    assert torch.equal(default_image, unnormalized_image)
    assert unnormalized_image.min().item() >= 0.0
    assert unnormalized_image.max().item() <= 1.0


@pytest.mark.parametrize("normalize_imagenet", [0, 1, "true", None])
def test_rejects_normalize_imagenet_that_is_not_bool(
    normalize_imagenet: object,
) -> None:
    with pytest.raises(TorchDataError, match=r"normalize_imagenet.*bool"):
        Fer2013TorchDataset(
            make_data(),
            split="train",
            normalize_imagenet=normalize_imagenet,
        )


def test_sample_preserves_label_and_sample_id() -> None:
    _, label, record_sample_id = Fer2013TorchDataset(
        make_data(), split="validation"
    )[1]

    assert label == 2
    assert record_sample_id == 202


def test_getitem_does_not_modify_original_numpy_image() -> None:
    data = make_data()
    original_image = data.records[0].image.copy()
    dataset = Fer2013TorchDataset(data, split="train")

    dataset[0]

    np.testing.assert_array_equal(data.records[0].image, original_image)


def test_repeated_access_is_deterministic() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    first_image, _, _ = dataset[0]
    second_image, _, _ = dataset[0]

    assert torch.equal(first_image, second_image)


def test_dataloader_combines_samples_into_expected_batch_shapes() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")
    loader = create_dataloader(
        dataset, batch_size=4, shuffle=False, seed=42
    )

    images, labels, batch_sample_ids = next(iter(loader))

    assert images.shape == (4, 3, 112, 112)
    assert labels.shape == (4,)
    assert batch_sample_ids.shape == (4,)


def test_dataloader_labels_use_cross_entropy_dtype() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")
    loader = create_dataloader(
        dataset, batch_size=4, shuffle=False, seed=42
    )

    _, labels, batch_sample_ids = next(iter(loader))

    assert labels.dtype == torch.long
    assert batch_sample_ids.dtype == torch.long


def test_dataloader_configures_transfer_and_worker_options() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=42,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )

    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3


def test_dataloader_disables_worker_only_options_with_zero_workers() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        seed=42,
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=3,
    )
    images, labels, batch_sample_ids = next(iter(loader))

    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None
    assert images.shape == (4, 3, 112, 112)
    assert labels.shape == (4,)
    assert batch_sample_ids.shape == (4,)


def test_same_seed_produces_same_shuffled_order() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")
    first_loader = create_dataloader(
        dataset, batch_size=3, shuffle=True, seed=42
    )
    second_loader = create_dataloader(
        dataset, batch_size=3, shuffle=True, seed=42
    )

    assert loader_sample_ids(first_loader) == loader_sample_ids(second_loader)


def test_different_seeds_produce_different_shuffled_orders() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")
    first_loader = create_dataloader(
        dataset, batch_size=3, shuffle=True, seed=42
    )
    second_loader = create_dataloader(
        dataset, batch_size=3, shuffle=True, seed=123
    )

    assert loader_sample_ids(first_loader) != loader_sample_ids(second_loader)


def test_shuffle_false_preserves_record_order() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")
    loader = create_dataloader(
        dataset, batch_size=3, shuffle=False, seed=42
    )

    assert loader_sample_ids(loader) == [101, 102, 103, 104, 105, 106, 107, 108]


def test_rejects_unsupported_split() -> None:
    with pytest.raises(TorchDataError, match=r"split.*train.*validation.*test"):
        Fer2013TorchDataset(make_data(), split="development")


def test_rejects_empty_requested_split() -> None:
    train_only = make_data([make_record(101, 0, "train")])

    with pytest.raises(TorchDataError, match=r"validation.*no samples"):
        Fer2013TorchDataset(train_only, split="validation")


@pytest.mark.parametrize("image_size", [0, -1, 1.5, True])
def test_rejects_image_size_that_is_not_a_positive_integer(
    image_size: object,
) -> None:
    with pytest.raises(TorchDataError, match=r"image_size.*positive integer"):
        Fer2013TorchDataset(make_data(), split="train", image_size=image_size)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_rejects_batch_size_that_is_not_a_positive_integer(
    batch_size: object,
) -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"batch_size.*positive integer"):
        create_dataloader(
            dataset, batch_size=batch_size, shuffle=False, seed=42
        )


def test_rejects_negative_seed() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"seed.*non-negative integer"):
        create_dataloader(dataset, batch_size=2, shuffle=True, seed=-1)


def test_rejects_negative_num_workers() -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"num_workers.*non-negative integer"):
        create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            num_workers=-1,
        )


@pytest.mark.parametrize("pin_memory", [0, 1, None, "true"])
def test_rejects_pin_memory_that_is_not_bool(pin_memory: object) -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"pin_memory.*bool"):
        create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            pin_memory=pin_memory,
        )


@pytest.mark.parametrize("persistent_workers", [0, 1, None, "true"])
def test_rejects_persistent_workers_that_is_not_bool(
    persistent_workers: object,
) -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"persistent_workers.*bool"):
        create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            persistent_workers=persistent_workers,
        )


@pytest.mark.parametrize("prefetch_factor", [0, -1, 1.5, True, None])
def test_rejects_invalid_prefetch_factor(prefetch_factor: object) -> None:
    dataset = Fer2013TorchDataset(make_data(), split="train")

    with pytest.raises(TorchDataError, match=r"prefetch_factor.*positive integer"):
        create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
            prefetch_factor=prefetch_factor,
        )


@pytest.mark.parametrize(
    "wrong_shape",
    [(48, 47), (47, 48), (48 * 48,), (1, 48, 48)],
)
def test_rejects_record_image_with_wrong_shape(wrong_shape: tuple[int, ...]) -> None:
    invalid_image = np.zeros(wrong_shape, dtype=np.uint8)
    data = make_data([make_record(101, 0, "train", invalid_image)])
    dataset = Fer2013TorchDataset(data, split="train")

    with pytest.raises(TorchDataError, match=r"sample 101.*image.*48, 48"):
        dataset[0]


@pytest.mark.parametrize("invalid_label", [-1, 7, 1.5, True])
def test_rejects_record_with_invalid_label(invalid_label: object) -> None:
    data = make_data([make_record(101, invalid_label, "train")])
    dataset = Fer2013TorchDataset(data, split="train")

    with pytest.raises(TorchDataError, match=r"sample 101.*label.*0.*6"):
        dataset[0]


def test_rejects_non_finite_converted_image() -> None:
    invalid_image = varied_image().astype(np.float32)
    invalid_image[12, 12] = np.nan
    data = make_data([make_record(101, 0, "train", invalid_image)])
    dataset = Fer2013TorchDataset(data, split="train")

    with pytest.raises(TorchDataError, match=r"sample 101.*non-finite"):
        dataset[0]
