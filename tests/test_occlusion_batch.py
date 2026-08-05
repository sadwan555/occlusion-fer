from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from occlusion_fer.occlusion import (
    apply_evaluation_batch,
    apply_evaluation_mask,
    apply_training_batch,
)


FILL_VECTOR = (1.25, -2.0, 0.5)


def _image_for_sample(sample_id: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.full((3, 112, 112), sample_id / 100000.0, dtype=dtype)


def _mask_in_batches(
    sample_ids: list[int],
    batch_size: int,
) -> dict[int, torch.Tensor]:
    outputs: dict[int, torch.Tensor] = {}
    for start in range(0, len(sample_ids), batch_size):
        chunk = sample_ids[start : start + batch_size]
        images = torch.stack([_image_for_sample(sample_id) for sample_id in chunk])
        ids = torch.tensor(chunk, dtype=torch.int64)
        masked, _ = apply_evaluation_batch(
            images,
            ids,
            "random_rectangle_0.30",
            FILL_VECTOR,
        )
        outputs.update(
            (sample_id, masked[index].clone())
            for index, sample_id in enumerate(chunk)
        )
    return outputs


class _ArtificialImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self) -> None:
        self.sample_ids = tuple(range(101, 109))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample_id = self.sample_ids[index]
        return _image_for_sample(sample_id), sample_id


def _mask_with_workers(num_workers: int) -> dict[int, torch.Tensor]:
    loader = DataLoader(
        _ArtificialImageDataset(),
        batch_size=3,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    outputs: dict[int, torch.Tensor] = {}
    for images, sample_ids in loader:
        masked, _ = apply_evaluation_batch(
            images,
            sample_ids,
            "random_rectangle_0.30",
            FILL_VECTOR,
        )
        outputs.update(
            (int(sample_id), masked[index].clone())
            for index, sample_id in enumerate(sample_ids.tolist())
        )
    return outputs


def test_evaluation_batch_clones_without_modifying_source() -> None:
    clean = torch.arange(2 * 3 * 112 * 112, dtype=torch.float32).reshape(
        2, 3, 112, 112
    )
    original = clean.clone()

    masked, _ = apply_evaluation_batch(
        clean,
        torch.tensor([28711, 28712]),
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert torch.equal(clean, original)
    assert masked.untyped_storage().data_ptr() != clean.untyped_storage().data_ptr()


def test_evaluation_batch_fills_only_golden_region_in_all_channels() -> None:
    clean = torch.arange(3 * 112 * 112, dtype=torch.float32).reshape(
        1, 3, 112, 112
    )
    expected = clean.clone()
    expected[0, 0, 28:89, 46:107] = 1.25
    expected[0, 1, 28:89, 46:107] = -2.0
    expected[0, 2, 28:89, 46:107] = 0.5

    masked, _ = apply_evaluation_batch(
        clean,
        torch.tensor([28711]),
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert torch.equal(masked, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_evaluation_batch_preserves_shape_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    clean = torch.zeros((2, 3, 112, 112), dtype=dtype)

    masked, _ = apply_evaluation_batch(
        clean,
        torch.tensor([28711, 28712]),
        "upper_face_0.20",
        FILL_VECTOR,
    )

    assert masked.shape == clean.shape
    assert masked.dtype == clean.dtype
    assert masked.device == clean.device


def test_evaluation_batch_converts_fill_to_image_dtype() -> None:
    clean = torch.zeros((1, 3, 112, 112), dtype=torch.float64)

    masked, _ = apply_evaluation_batch(
        clean,
        torch.tensor([28711]),
        "upper_face_0.20",
        FILL_VECTOR,
    )

    torch.testing.assert_close(
        masked[0, :, 0, 0],
        torch.tensor(FILL_VECTOR, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_evaluation_metadata_is_complete_and_ordered() -> None:
    sample_ids = torch.tensor([28712, 28711], dtype=torch.int64)

    _, metadata = apply_evaluation_batch(
        torch.zeros((2, 3, 112, 112)),
        sample_ids,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert tuple(item.sample_id for item in metadata) == (28712, 28711)
    golden = metadata[1]
    assert (
        golden.condition,
        golden.occlusion_type,
        golden.target_ratio,
        golden.actual_ratio,
        golden.top,
        golden.left,
        golden.height,
        golden.width,
        golden.masked_pixel_count,
        golden.total_pixel_count,
        golden.algorithm_version,
        golden.context,
        golden.seed,
        golden.epoch,
        golden.coordinate_convention,
    ) == (
        "random_rectangle_0.30",
        "random_rectangle",
        0.30,
        3721 / 12544,
        28,
        46,
        61,
        61,
        3721,
        12544,
        "occlusion-v1",
        "validation",
        20260804,
        None,
        "half-open:[top,top+height)x[left,left+width)",
    )


def test_metadata_is_immutable() -> None:
    _, metadata = apply_evaluation_batch(
        torch.zeros((1, 3, 112, 112)),
        torch.tensor([28711]),
        "upper_face_0.20",
        FILL_VECTOR,
    )

    with pytest.raises(FrozenInstanceError):
        metadata[0].top = 1


def test_sample_ids_are_not_modified() -> None:
    sample_ids = torch.tensor([28711, 28712], dtype=torch.int64)
    original_ids = sample_ids.clone()

    apply_evaluation_batch(
        torch.zeros((2, 3, 112, 112)),
        sample_ids,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert torch.equal(sample_ids, original_ids)


def test_batch_reordering_preserves_each_sample_output() -> None:
    first_order = [28711, 28712, 28713]
    second_order = [28713, 28711, 28712]

    first = _mask_in_batches(first_order, batch_size=3)
    second = _mask_in_batches(second_order, batch_size=3)

    assert all(torch.equal(first[sample_id], second[sample_id]) for sample_id in first)


def test_batch_sizes_1_32_and_128_preserve_each_sample_output() -> None:
    sample_ids = list(range(1, 129))
    reference = _mask_in_batches(sample_ids, batch_size=1)

    for batch_size in (32, 128):
        actual = _mask_in_batches(sample_ids, batch_size=batch_size)
        assert all(
            torch.equal(reference[sample_id], actual[sample_id])
            for sample_id in sample_ids
        )


def test_single_sample_and_batch_outputs_are_identical() -> None:
    clean = _image_for_sample(28711)

    single_masked, single_metadata = apply_evaluation_mask(
        clean,
        28711,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )
    batch_masked, batch_metadata = apply_evaluation_batch(
        clean.unsqueeze(0),
        torch.tensor([28711]),
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert torch.equal(single_masked, batch_masked[0])
    assert single_metadata == batch_metadata[0]


def test_num_workers_zero_and_four_preserve_each_sample_output() -> None:
    without_workers = _mask_with_workers(0)
    with_workers = _mask_with_workers(4)

    assert without_workers.keys() == with_workers.keys()
    assert all(
        torch.equal(without_workers[sample_id], with_workers[sample_id])
        for sample_id in without_workers
    )


def test_masking_does_not_consume_or_depend_on_torch_rng() -> None:
    clean = _image_for_sample(28711).unsqueeze(0)
    sample_ids = torch.tensor([28711])

    torch.manual_seed(1)
    state_before = torch.random.get_rng_state().clone()
    first, _ = apply_evaluation_batch(
        clean,
        sample_ids,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )
    state_after = torch.random.get_rng_state()

    torch.manual_seed(999)
    second, _ = apply_evaluation_batch(
        clean,
        sample_ids,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    assert torch.equal(state_before, state_after)
    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_masking_does_not_consume_cuda_rng() -> None:
    device = torch.device("cuda")
    clean = torch.zeros((1, 3, 112, 112), device=device)
    sample_ids = torch.tensor([28711])
    torch.cuda.manual_seed_all(7)
    state_before = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    apply_evaluation_batch(
        clean,
        sample_ids,
        "random_rectangle_0.30",
        FILL_VECTOR,
    )

    state_after = torch.cuda.get_rng_state_all()
    assert all(
        torch.equal(before, after)
        for before, after in zip(state_before, state_after, strict=True)
    )


def test_training_batch_uses_selected_condition_context() -> None:
    masked, metadata = apply_training_batch(
        torch.zeros((1, 3, 112, 112)),
        torch.tensor([28711]),
        "random_rectangle_0.30",
        FILL_VECTOR,
        training_seed=42,
        epoch=1,
    )

    item = metadata[0]
    assert (item.top, item.left, item.context, item.seed, item.epoch) == (
        21,
        38,
        "train",
        42,
        1,
    )
    torch.testing.assert_close(
        masked[0, :, 21, 38],
        torch.tensor(FILL_VECTOR),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "clean_images",
    [
        torch.zeros((3, 112, 112)),
        torch.zeros((1, 1, 3, 112, 112)),
        torch.zeros((1, 1, 112, 112)),
        torch.zeros((1, 4, 112, 112)),
        torch.zeros((1, 3, 48, 48)),
    ],
)
def test_rejects_wrong_image_shape(clean_images: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        apply_evaluation_batch(
            clean_images,
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_non_tensor_images() -> None:
    with pytest.raises(TypeError):
        apply_evaluation_batch(
            [[0.0]],  # type: ignore[arg-type]
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_integer_image_tensor() -> None:
    with pytest.raises(TypeError):
        apply_evaluation_batch(
            torch.zeros((1, 3, 112, 112), dtype=torch.int64),
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError):
        apply_evaluation_batch(
            torch.zeros((0, 3, 112, 112)),
            torch.tensor([], dtype=torch.int64),
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_nonfinite_image_tensor() -> None:
    clean = torch.zeros((1, 3, 112, 112))
    clean[0, 0, 0, 0] = torch.nan

    with pytest.raises(ValueError):
        apply_evaluation_batch(
            clean,
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


@pytest.mark.parametrize(
    "sample_ids",
    [
        [28711],
        torch.tensor([[28711]]),
        torch.tensor([28711.0]),
        torch.tensor([True]),
        torch.tensor([0]),
        torch.tensor([-1]),
    ],
)
def test_rejects_invalid_sample_ids(sample_ids: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_evaluation_batch(
            torch.zeros((1, 3, 112, 112)),
            sample_ids,  # type: ignore[arg-type]
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_mismatched_sample_id_count() -> None:
    with pytest.raises(ValueError):
        apply_evaluation_batch(
            torch.zeros((2, 3, 112, 112)),
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


def test_rejects_duplicate_evaluation_sample_ids() -> None:
    with pytest.raises(ValueError):
        apply_evaluation_batch(
            torch.zeros((2, 3, 112, 112)),
            torch.tensor([28711, 28711]),
            "upper_face_0.20",
            FILL_VECTOR,
        )


@pytest.mark.parametrize(
    "fill_vector",
    [
        (),
        (0.0, 1.0),
        (0.0, 1.0, 2.0, 3.0),
        (0.0, float("nan"), 2.0),
        (0.0, float("inf"), 2.0),
        (0.0, True, 2.0),
        (0.0, "1.0", 2.0),
    ],
)
def test_rejects_invalid_fill_vector(fill_vector: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_evaluation_batch(
            torch.zeros((1, 3, 112, 112)),
            torch.tensor([28711]),
            "upper_face_0.20",
            fill_vector,  # type: ignore[arg-type]
        )


def test_rejects_noncanonical_batch_condition() -> None:
    with pytest.raises(ValueError):
        apply_evaluation_batch(
            torch.zeros((1, 3, 112, 112)),
            torch.tensor([28711]),
            "original",
            FILL_VECTOR,
        )


def test_failed_validation_leaves_source_unchanged() -> None:
    clean = torch.arange(3 * 112 * 112, dtype=torch.float32).reshape(
        1, 3, 112, 112
    )
    original = clean.clone()

    with pytest.raises(ValueError):
        apply_evaluation_batch(
            clean,
            torch.tensor([28711]),
            "upper_face_0.20",
            (0.0, float("nan"), 1.0),
        )

    assert torch.equal(clean, original)


def test_batch_api_does_not_accept_labels() -> None:
    with pytest.raises(TypeError):
        apply_evaluation_batch(
            torch.zeros((1, 3, 112, 112)),
            torch.tensor([28711]),
            "upper_face_0.20",
            FILL_VECTOR,
            labels=torch.tensor([3]),  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("training_seed", "epoch"),
    [(True, 1), (-1, 1), (42, True), (42, 0)],
)
def test_training_batch_rejects_invalid_context_integers(
    training_seed: object,
    epoch: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        apply_training_batch(
            torch.zeros((1, 3, 112, 112)),
            torch.tensor([28711]),
            "random_rectangle_0.30",
            FILL_VECTOR,
            training_seed=training_seed,  # type: ignore[arg-type]
            epoch=epoch,  # type: ignore[arg-type]
        )
