from pathlib import Path

import numpy as np
import pandas as pd

from occlusion_fer.paper_figures import CLASSWISE_CONDITIONS, EXPECTED_SEEDS
from occlusion_fer.paper_stage8 import Stage8Evidence, compute_stage8_tables


def _evidence() -> Stage8Evidence:
    rows = []
    for method_index, method in enumerate(("Clean-only", "Mixed training")):
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            for condition_index, condition in enumerate(CLASSWISE_CONDITIONS):
                if condition == "clean":
                    occlusion_type = "clean"
                    severity = 0.0
                else:
                    occlusion_type, severity_text = condition.rsplit("_", 1)
                    severity = float(severity_text)
                clean_value = 0.50 + 0.10 * seed_index + 0.001 * condition_index
                paired_gain = 0.01 + 0.01 * seed_index
                value = clean_value + (paired_gain if method_index else 0.0)
                rows.append(
                    {
                        "method": method,
                        "training_mode": "mixed" if method_index else "clean",
                        "seed": seed,
                        "condition": condition,
                        "occlusion_type": occlusion_type,
                        "severity": severity,
                        "accuracy": value,
                        "macro_f1": value - 0.02,
                    }
                )
    return Stage8Evidence(
        clean_results_dir=Path("/clean"),
        mixed_training_dir=Path("/mixed-training"),
        mixed_evaluation_dir=Path("/mixed-evaluation"),
        metrics=pd.DataFrame(rows),
        source_files=(),
        provenance=(),
    )


def test_robustness_summary_uses_seed_level_means_before_sample_sd() -> None:
    tables = compute_stage8_tables(_evidence())
    robustness = tables["experiment3_robustness_summary"]
    clean_accuracy = robustness.loc[
        (robustness["method"] == "Clean-only")
        & (robustness["metric"] == "accuracy")
    ].iloc[0]

    expected_seed_means = np.array([0.505, 0.605, 0.705])
    assert np.isclose(clean_accuracy["mean"], expected_seed_means.mean())
    assert np.isclose(clean_accuracy["sample_sd"], expected_seed_means.std(ddof=1))

    twenty_seven_values = _evidence().metrics.loc[
        (_evidence().metrics["method"] == "Clean-only")
        & (_evidence().metrics["condition"] != "clean"),
        "accuracy",
    ]
    assert not np.isclose(clean_accuracy["sample_sd"], twenty_seven_values.std(ddof=1))


def test_robustness_gain_is_paired_within_seed_before_aggregation() -> None:
    gain = compute_stage8_tables(_evidence())["experiment3_robustness_gain"]

    assert len(gain) == 18
    assert np.allclose(gain["mean_gain"], 0.02)
    assert np.allclose(gain["sample_sd"], 0.01)
    assert np.allclose(gain["seed42_gain"], 0.01)
    assert np.allclose(gain["seed123_gain"], 0.02)
    assert np.allclose(gain["seed2026_gain"], 0.03)


def test_comparison_summary_contains_two_methods_ten_conditions_two_metrics() -> None:
    comparison = compute_stage8_tables(_evidence())["experiment3_comparison_summary"]

    assert len(comparison) == 40
    assert set(comparison["method"]) == {"Clean-only", "Mixed training"}
    assert set(comparison["condition"]) == set(CLASSWISE_CONDITIONS)
    assert set(comparison["metric"]) == {"accuracy", "macro_f1"}
    assert set(comparison["n_seeds"]) == {3}
