# Paper Evidence Map

This document maps the completed first-version experiments to paper sections.
It does not replace the source JSON/CSV artifacts and must not be used to invent
or manually edit numerical results.

## Evidence Boundary

Use the phrasing "FER2013 dataset-defined facial-expression label
classification." Do not claim recognition of internal emotion or cognitive
state, a validated classroom application, real-world robustness, cross-dataset
generalization, novelty, or state-of-the-art performance.

The current paper lineage is 224x224 E7/Stage 8 with
`occlusion-v2-224`. The `da889bd` 112x112/30-epoch results, v1 masks, and v1
Grad-CAM images are development history and are excluded from Experiments 1-3.

## Section Mapping

### Method

Describe the official Training/PublicTest/PrivateTest roles, 224x224 grayscale
replication pipeline, ImageNet-pretrained ResNet-18, locked E7 hyperparameters,
three formal seeds, clean-only and mixed strategies, Training-mean fill, and
the nine deterministic masked conditions.

Use resolved configurations, run metadata, protocol artifacts, final evaluation
plan, checkpoint registry, and manifest sidecars as direct evidence. Do not
infer settings from the legacy configs visible on another branch.

### Experiment 1

Report the three clean-only seeds and their aggregate clean-condition results.
Use per-class metrics and confusion matrices from the frozen final output. The
training curves come from the E7 training archives, not from PrivateTest.

### Experiment 2

Report the clean-trained checkpoints under clean and nine occluded conditions.
Compute each performance drop within a seed before aggregating across seeds.
Keep target and actual occlusion ratios distinct.

### Experiment 3

Compare clean-only and mixed checkpoints using the same seed and condition.
Report clean performance separately from masked-condition performance. Any
mixed-minus-clean gain is paired within seed before calculating mean and sample
standard deviation.

### Discussion and Conclusion

Answer RQ1-RQ3 only from the frozen FER2013 results. Separate directly observed
patterns from possible explanations. State the single-dataset, single-backbone,
synthetic-occlusion, fixed-resolution, and three-seed limitations.

## Artifact Mapping

| Paper content | Direct source |
|---|---|
| Training setup | `resolved_config.yaml`, `run_metadata.json` |
| Checkpoint selection | `history.csv`, `validation/best_metrics.json` |
| Overall metrics | per-condition `metrics.json` |
| Per-class results | `per_class_metrics.csv` |
| Confusion matrices | `confusion_matrix.csv` |
| Paired analysis | `predictions.csv`, clean-to-occluded drop tables |
| Protocol identity | Training mean, mask manifest/sidecar, evaluation plan |
| Checkpoint identity | best/last files, registry metadata, SHA-256 |
| Aggregate tables | strategy-condition summary and paired-difference CSVs |
| Figure traceability | figure manifests and exported source-data CSVs |

## Figure Set

The current local figure set includes:

- overall experimental framework;
- occlusion protocol examples;
- Experiment 1 clean baseline and confusion matrix;
- occlusion-severity accuracy/macro-F1 plots;
- class-wise F1 and clean-drop heatmaps;
- clean-only versus mixed condition comparisons;
- paired robustness-gain heatmap;
- a PublicTest Grad-CAM comparison panel.

PrivateTest plots are generated from the frozen final archive only. Grad-CAM is
qualitative and does not replace the quantitative experiments.

## Final Checks

- Use all seeds 42, 123, and 2026.
- Use sample standard deviation (`ddof=1`).
- Keep PublicTest checkpoint-selection evidence separate from PrivateTest final
  evidence.
- Confirm all numbers trace to an immutable JSON/CSV source.
- Do not use smoke, screening, pilot, legacy, or best-seed-only values.
- Retain failed and negative evidence.
- Record any release checksum mismatch rather than silently rewriting it.
