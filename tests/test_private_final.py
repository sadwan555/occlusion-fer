from pathlib import Path

import pytest
import torch
from torch import nn

from occlusion_fer.formal_checkpoints import checkpoint_for_model_id
from occlusion_fer.private_final import (
    PrivateFinalError,
    evaluate_private_conditions,
    parse_args,
    require_private_confirmation,
    validate_checkpoint_binding,
    validate_private_route,
)
from occlusion_fer.private_manifest import (
    CONDITION_ORDER,
    build_manifest_rows,
    build_manifest_sidecar,
)
from occlusion_fer.private_preflight import ensure_empty_output_root


class TrackingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        logits = torch.zeros(images.shape[0], 7, device=images.device)
        logits[:, 0] = self.scale
        return logits


def _loader():
    return [
        (
            torch.zeros(2, 3, 224, 224),
            torch.tensor([0, 1]),
            torch.tensor([101, 205]),
        )
    ]


def test_cli_requires_explicit_plan_and_private_confirmation() -> None:
    args = parse_args(
        [
            "--plan",
            "plan.json",
            "--data-path",
            "synthetic.csv",
            "--training-mean",
            "mean.json",
            "--manifest",
            "manifest.csv",
            "--confirm-private-test",
        ]
    )
    assert args.plan == Path("plan.json")
    assert args.confirm_private_test is True
    with pytest.raises(PrivateFinalError, match="confirm-private-test"):
        require_private_confirmation(False)


@pytest.mark.parametrize(
    ("split", "usage"),
    [
        ("train", "Training"),
        ("validation", "PublicTest"),
        ("test", "PublicTest"),
    ],
)
def test_final_route_allows_only_test_privatetest(split: str, usage: str) -> None:
    with pytest.raises(PrivateFinalError):
        validate_private_route(split, usage)
    validate_private_route("test", "PrivateTest")


def test_output_root_must_be_absent_or_empty(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert ensure_empty_output_root(missing) == missing
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ensure_empty_output_root(empty) == empty
    existing = empty / "result.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_empty_output_root(empty)
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_synthetic_evaluator_is_inference_only_and_preserves_parameters() -> None:
    rows = build_manifest_rows((101, 205), "a" * 64)
    sidecar = build_manifest_sidecar(rows, "a" * 64)
    model = TrackingModel()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    results = evaluate_private_conditions(
        model,
        _loader(),
        torch.device("cpu"),
        manifest_rows=rows,
        manifest_sidecar=sidecar,
        require_official_manifest=False,
    )

    assert tuple(results) == CONDITION_ORDER
    assert all(result.sample_count == 2 for result in results.values())
    assert model.training is False
    assert model.grad_enabled_during_forward
    assert not any(model.grad_enabled_during_forward)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])
        assert tensor.grad is None


def test_checkpoint_sha_binding_fails_loudly() -> None:
    spec = checkpoint_for_model_id("CLEAN-42")
    with pytest.raises(PrivateFinalError, match="checkpoint SHA"):
        validate_checkpoint_binding(spec, "0" * 64)
