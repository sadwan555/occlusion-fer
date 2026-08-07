# Formal Clean Training and Paper Artifacts Design

> Historical design record. For the current E7 clean/mixed protocol, use
> [`../../experiment_protocol.md`](../../experiment_protocol.md). This record
> predates `occlusion-v2-224` and must not be used to start a formal run.

## Purpose

Prepare the existing FER2013 clean ResNet-18 pipeline for reproducible formal
training and for later thesis writing. The stage must produce trustworthy,
plot-ready evidence without changing the dataset split meanings, seven labels,
model architecture, optimizer, or validated high-performance data-loading path.

The attached `essay.docx` establishes a Methods, Results, Discussion, and
Conclusion structure. This design therefore treats experiment artifacts as a
stable interface for filling those sections later. It does not edit the essay
and does not treat FER2013 labels as verified internal emotions or cognitive
states.

## Stage boundary

This stage covers formal **clean-only** training readiness:

- seven-class accuracy and macro-F1;
- per-class precision, recall, F1, and support;
- 7 x 7 confusion matrices;
- PublicTest checkpoint selection by clean validation macro-F1;
- best and last checkpoints;
- best and last validation predictions and metrics;
- run configuration, environment, Git, timing, and failure metadata;
- an explicit PrivateTest final-evaluation command;
- plot-ready JSON and CSV outputs;
- command-line seed override for the approved seeds 42, 123, and 2026;
- documentation and automated tests.

This stage does not implement synthetic occlusion, mixed training, multi-run
aggregation, statistical testing, or final thesis figures. Those later stages
will consume the stable artifact schema defined here.

## Selected architecture

Use focused modules rather than adding every responsibility to `train.py`:

- `src/occlusion_fer/metrics.py` owns classification metrics and their
  validation.
- `src/occlusion_fer/evaluation.py` owns inference-mode evaluation, prediction
  records, and the reusable evaluation result type. `train.py` re-exports its
  existing `evaluate` name for compatibility.
- `src/occlusion_fer/artifacts.py` owns atomic JSON/CSV artifact writing, run
  metadata, prediction tables, and failure records.
- `src/occlusion_fer/train.py` keeps optimization and orchestration, enriches
  validation results, and selects `best.pt` using macro-F1.
- `src/occlusion_fer/final_evaluate.py` loads one checkpoint and evaluates only
  the official PrivateTest split after explicit confirmation.

Only the existing Python standard library, NumPy, PyYAML, PyTorch, and
torchvision are used. Do not add scikit-learn, pandas, matplotlib, or a GPU
package dependency. Later figure code can read the emitted CSV files on the
Mac without changing the training environment.

## Metric definitions

The confusion matrix has shape `[7, 7]`. Rows are true labels and columns are
predicted labels, both ordered as:

1. angry
2. disgust
3. fear
4. happy
5. sad
6. surprise
7. neutral

For class `c`:

- `precision_c = TP_c / (TP_c + FP_c)`;
- `recall_c = TP_c / (TP_c + FN_c)`;
- `F1_c = 2 * precision_c * recall_c / (precision_c + recall_c)`;
- `support_c` is the number of true samples of class `c`.

A zero denominator produces `0.0`; it never produces NaN or Inf. Macro-F1 is
the arithmetic mean of all seven class F1 values, including a zero for a class
that is absent from a limited smoke-test subset. Accuracy is total correct
predictions divided by total samples.

## Validation and checkpoint rule

Training uses only the official Training split for parameter updates and only
PublicTest for epoch-level validation. PrivateTest is never loaded by
`occlusion_fer.train`.

Every validation pass returns:

- sample-weighted loss;
- accuracy;
- macro-F1;
- per-class metrics;
- confusion matrix;
- one prediction record per sample.

`best.pt` is replaced only when validation macro-F1 is strictly greater than
the previous best. A tie retains the earlier checkpoint. `last.pt` is written
after every epoch. Checkpoints store both current validation metrics and the
best validation macro-F1. Old checkpoints remain usable for evaluation when
they contain `model_state_dict`, even if they lack the new metric fields.

## Prediction schema

Each prediction row contains:

- `sample_id`;
- `split`;
- `condition` (`clean` in this stage);
- `true_label` and `true_label_name`;
- `predicted_label` and `predicted_label_name`;
- `correct`;
- `predicted_confidence`;
- one probability column for each of the seven ordered labels.

Probabilities are calculated from softmax logits and written with enough
decimal precision for later error analysis. Rows remain in DataLoader output
order and sample IDs provide the stable join key. No image data or local path
is embedded in prediction files.

## Run artifact layout

One output directory represents one run and must remain outside Git:

```text
<run-directory>/
  best.pt
  last.pt
  resolved_config.yaml
  run_metadata.json
  history.json
  history.csv
  failure.json                         # only when a run fails
  validation/
    best_metrics.json
    best_per_class_metrics.csv
    best_confusion_matrix.csv
    best_predictions.csv
    last_metrics.json
    last_per_class_metrics.csv
    last_confusion_matrix.csv
    last_predictions.csv
  final_test/
    clean_metrics.json
    clean_per_class_metrics.csv
    clean_confusion_matrix.csv
    clean_predictions.csv
```

`history.json` and `history.csv` include epoch, training/validation losses,
validation accuracy, validation macro-F1, durations, samples per second, peak
CUDA memory, and whether that epoch updated the best checkpoint. These files
support thesis learning-curve figures without parsing terminal logs.

Metric JSON includes the metric definition version, split, condition, sample
count, loss, accuracy, macro-F1, ordered label metadata, per-class metrics, and
the confusion matrix. The parallel CSV files support spreadsheet, Python, R,
and plotting workflows.

Artifact writes use a temporary file followed by replacement so interrupted
writes do not leave a partially written JSON or CSV presented as valid.

## Reproducibility metadata

`resolved_config.yaml` records the post-override configuration used by the run,
including the runtime smoke limits and AMP/DataLoader settings already captured
by the pipeline.

`run_metadata.json` records:

- schema version and run status (`running`, `completed`, or `failed`);
- UTC start and finish timestamps;
- seed and training mode;
- Git commit and dirty state without altering Git;
- Python, NumPy, PyTorch, and torchvision versions;
- operating system, hostname, selected device, and CUDA device name when used;
- command-line runtime settings;
- artifact paths relative to the run directory.

If training fails after the output directory is known, `failure.json` records
the exception type, message, stage, and UTC timestamp. It must not contain
credentials, environment-variable dumps, dataset content, or fabricated
results.

## PrivateTest integrity boundary

PrivateTest evaluation is a separate command:

```text
python -m occlusion_fer.final_evaluate ... --confirm-private-test
```

Without `--confirm-private-test`, the command exits before loading the dataset.
It loads a specified checkpoint, uses only the official PrivateTest split, and
writes the `final_test/clean_*` artifacts. It performs no optimizer update and
does not change `best.pt` or `last.pt`. Documentation instructs the researcher
to run it only after the model, epoch budget, batch size, learning rate, seed
set, and checkpoint rule are locked.

## Configuration and HIVE settings

Preserve the YAML configuration structure and placeholder dataset path. Add a
`--seed` override using the existing `training.seed` field. Do not hard-code a
Mac or HIVE data path.

The validated HIVE performance settings remain explicit runtime overrides:

```text
--batch-size 128 --num-workers 4 --amp
```

This keeps the committed YAML portable while ensuring every checkpoint and
resolved configuration records the actual HIVE settings. Formal clean runs use
seeds 42, 123, and 2026 with identical architecture, optimizer, epoch budget,
batch size, data processing, and checkpoint rule.

## Paper-writing interface

The artifacts support the existing essay outline as follows:

- **Methods 3.1:** `run_metadata.json` and the artifact layout describe the
  reproducible experiment flow.
- **Methods 3.2:** `resolved_config.yaml`, label metadata, and split fields
  document FER2013 processing without exposing the dataset.
- **Methods 3.3:** checkpoint metadata and resolved model/training fields
  document ResNet-18, ImageNet initialization, AdamW, image size, normalization,
  seed, batch size, AMP, and device.
- **Results 4:** history files support learning curves; metric and per-class
  CSVs support tables; confusion matrices support heatmaps; predictions support
  error analysis.
- **Discussion 5:** classwise metrics and stable sample IDs support evidence
  about dataset-label error patterns without claiming true emotion or cognitive
  inference.
- **Conclusion 6:** run status and fixed metric definitions constrain claims to
  controlled FER2013 experiments.

The future occlusion stage must reuse `condition`, label order, metric
definitions, and CSV column names so clean and occluded results can be compared
without ad hoc data conversion.

## Error handling

Fail clearly on invalid label or prediction shapes, values outside 0 through 6,
empty evaluation data, non-finite loss/probability/metric values, duplicate or
missing sample IDs, malformed checkpoints, unwritable artifacts, missing Git
metadata for a claimed formal run, and an unconfirmed PrivateTest request.
Never silently skip a sample, class, artifact, or failed run.

## Testing strategy

Use test-driven development for each behavior:

- exact confusion-matrix orientation and counts;
- accuracy, per-class metrics, and seven-class macro-F1;
- zero-support and zero-prediction classes without non-finite values;
- evaluation prediction records and probability fields;
- strict macro-F1 checkpoint comparison and tie behavior;
- best/last checkpoint payloads and legacy checkpoint loading;
- atomic JSON/CSV artifact content and column order;
- resolved configuration and run metadata;
- failure record creation;
- PrivateTest confirmation before dataset loading;
- PrivateTest-only evaluation and absence of parameter changes;
- seed CLI override;
- artificial CPU smoke training with best/last validation artifacts;
- existing performance settings, AMP, DataLoader, and training tests.

Before completion run the full pytest suite, `compileall`, `pip check`, Git diff
checks, and secret/data/checkpoint scans. No test result is claimed without its
actual command output.

## Documentation changes

Update `README.md`, `SERVER_RUN.md`, and `docs/server_runbook.md` so commands and
artifact descriptions match the implementation. The quick start remains short;
the runbook explains checkpoint selection, the PrivateTest gate, formal seeds,
artifact interpretation, and the distinction between engineering checks and
formal results.

Do not edit `essay.docx` in this stage. A later thesis-writing stage can use the
verified artifacts to fill its empty Methods, Results, Discussion, and
Conclusion sections and correct research-boundary language.
