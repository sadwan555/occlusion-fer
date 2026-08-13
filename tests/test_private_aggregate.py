import math

from occlusion_fer.private_aggregate import build_aggregate_rows
from occlusion_fer.private_manifest import CONDITION_ORDER


def _metrics():
    values = {}
    for strategy, offset in (("clean", 0.0), ("mixed", 0.1)):
        for seed, clean_accuracy in zip((42, 123, 2026), (0.6, 0.7, 0.8), strict=True):
            values[(strategy, seed, "clean")] = {
                "accuracy": clean_accuracy + offset,
                "macro_f1": clean_accuracy - 0.1 + offset,
            }
            for condition in CONDITION_ORDER[1:]:
                values[(strategy, seed, condition)] = {
                    "accuracy": clean_accuracy - 0.2 + offset,
                    "macro_f1": clean_accuracy - 0.3 + offset,
                }
    return values


def test_aggregate_uses_three_seed_sample_standard_deviation() -> None:
    summary, _, _ = build_aggregate_rows(_metrics())
    row = next(
        item
        for item in summary
        if item["strategy"] == "clean"
        and item["condition"] == "clean"
        and item["metric"] == "accuracy"
    )
    assert row["seed42"] == 0.6
    assert row["seed123"] == 0.7
    assert row["seed2026"] == 0.8
    assert math.isclose(row["mean"], 0.7)
    assert math.isclose(row["sample_std"], 0.1)


def test_aggregate_computes_clean_drop_and_paired_strategy_difference() -> None:
    _, drops, paired = build_aggregate_rows(_metrics())
    drop = next(
        item
        for item in drops
        if item["strategy"] == "clean"
        and item["seed"] == 42
        and item["condition"] == "upper_face_0.20"
    )
    assert math.isclose(drop["accuracy_drop"], 0.2)
    assert math.isclose(drop["macro_f1_drop"], 0.2)

    difference = next(
        item
        for item in paired
        if item["seed"] == 42 and item["condition"] == "upper_face_0.20"
    )
    assert math.isclose(difference["accuracy_difference"], 0.1)
    assert math.isclose(difference["macro_f1_difference"], 0.1)
