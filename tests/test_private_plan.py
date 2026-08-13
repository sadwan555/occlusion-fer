from dataclasses import replace
from pathlib import Path

import pytest

from occlusion_fer.formal_checkpoints import FORMAL_CHECKPOINTS
from occlusion_fer.private_plan import (
    FinalPlanError,
    build_final_plan,
    canonical_plan_bytes,
    load_final_plan,
    plan_sha256,
    write_final_plan,
)


def _plan(tmp_path: Path):
    paths = {
        item.model_id: str(tmp_path / "checkpoints" / f"{item.model_id}.pt")
        for item in FORMAL_CHECKPOINTS
    }
    return build_final_plan(
        checkpoint_paths=paths,
        output_root=str(tmp_path / "final"),
        git_commit="a" * 40,
        git_dirty=True,
        created_at="2026-08-13T12:00:00Z",
    )


def test_plan_locks_scientific_protocol_and_six_checkpoints(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    assert plan.private.official_usage == "PrivateTest"
    assert plan.private.internal_split == "test"
    assert plan.private.count == 3589
    assert plan.private.canonical_sha256 == (
        "4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634"
    )
    assert plan.protocol.image_size == 224
    assert plan.protocol.algorithm_version == "occlusion-v2-224"
    assert plan.protocol.mask_seed == 20260804
    assert len(plan.checkpoints) == 6


def test_plan_sha_is_canonical_and_sensitive_to_fields(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    same = _plan(tmp_path)
    changed = replace(plan, git_dirty=False)
    assert canonical_plan_bytes(plan) == canonical_plan_bytes(same)
    assert plan_sha256(plan) == plan_sha256(same)
    assert plan_sha256(plan) != plan_sha256(changed)


def test_plan_write_is_no_overwrite_and_load_verifies_sha(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    target = tmp_path / "plan.json"
    digest = write_final_plan(target, plan)
    loaded, loaded_digest = load_final_plan(target)
    assert loaded == plan
    assert loaded_digest == digest
    with pytest.raises(FileExistsError):
        write_final_plan(target, plan)

    target.write_bytes(target.read_bytes().replace(b'"git_dirty":true', b'"git_dirty":false'))
    with pytest.raises(FinalPlanError, match="SHA"):
        load_final_plan(target)


def test_plan_rejects_missing_or_unapproved_checkpoint(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(FinalPlanError, match="six"):
        replace(plan, checkpoints=plan.checkpoints[:-1]).validate()
    changed = replace(plan.checkpoints[0], sha256="0" * 64)
    with pytest.raises(FinalPlanError, match="checkpoint"):
        replace(plan, checkpoints=(changed,) + plan.checkpoints[1:]).validate()
