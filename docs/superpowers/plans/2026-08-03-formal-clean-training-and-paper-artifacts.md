# Formal Clean Training and Paper Artifacts Implementation Plan

> Historical implementation plan. The current formal protocol is
> [`../../experiment_protocol.md`](../../experiment_protocol.md); this plan is
> retained for provenance and does not authorize a new run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make clean FER2013 ResNet-18 runs reproducible, select checkpoints by PublicTest macro-F1, evaluate PrivateTest only through an explicit final command, and emit paper-ready metrics and prediction artifacts.

**Architecture:** Pure metric computation lives in `metrics.py`; inference and prediction collection live in `evaluation.py`; deterministic JSON/CSV output and run provenance live in `artifacts.py`. `train.py` orchestrates optimization and PublicTest checkpoint selection while `final_evaluate.py` is the only entry point that accesses PrivateTest.

**Tech Stack:** Python 3.10+, NumPy, PyYAML, PyTorch 2.5.x, torchvision 0.20.x, pytest.

## Global Constraints

- Preserve official FER2013 Training, PublicTest, and PrivateTest meanings.
- PrivateTest must never be loaded by `occlusion_fer.train`.
- Select `best.pt` using clean PublicTest macro-F1; ties retain the earlier checkpoint.
- Use all seven labels in macro-F1, including zero-F1 for an absent class.
- Preserve ResNet-18, ImageNet normalization, AdamW, and the optimized DataLoader path.
- Do not add runtime dependencies or hard-code Mac/HIVE paths.
- Do not implement occlusion, mixed training, multi-run aggregation, or final thesis figures in this stage.
- Outputs, predictions, checkpoints, logs, and data remain ignored by Git.
- Do not commit or push; repository policy requires a separate explicit request.

---

### Task 1: Seven-class classification metrics

**Files:**
- Create: `src/occlusion_fer/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `PerClassMetrics`, `ClassificationMetrics`, and `compute_classification_metrics(true_labels, predicted_labels, *, num_classes=7, label_names=FER2013_LABEL_NAMES) -> ClassificationMetrics`.
- `ClassificationMetrics.confusion_matrix` is `tuple[tuple[int, ...], ...]`, rows true and columns predicted.

- [ ] **Step 1: Write failing tests for exact counts and orientation**

```python
def test_compute_metrics_uses_true_rows_and_predicted_columns() -> None:
    result = compute_classification_metrics([0, 0, 1, 2], [0, 1, 1, 1])
    assert result.confusion_matrix[0][0] == 1
    assert result.confusion_matrix[0][1] == 1
    assert result.confusion_matrix[1][1] == 1
    assert result.confusion_matrix[2][1] == 1
    assert result.accuracy == pytest.approx(0.5)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_metrics.py -q`

Expected: collection fails because `occlusion_fer.metrics` does not exist.

- [ ] **Step 3: Implement immutable metric result types and exact metric computation**

```python
@dataclass(frozen=True)
class PerClassMetrics:
    label: int
    label_name: str
    precision: float
    recall: float
    f1: float
    support: int

@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    sample_count: int
```

Build a NumPy `int64` confusion matrix with `np.add.at`, validate equal nonzero
lengths and labels in `[0, 6]`, calculate zero-denominator values as `0.0`, and
reject non-finite outputs.

- [ ] **Step 4: Add failing edge-case tests**

Cover seven-class averaging with absent classes, empty input, unequal lengths,
invalid labels, wrong label-name count, non-integer/bool labels, and finite
outputs.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_metrics.py -q`

Expected: all metric tests pass.

- [ ] **Step 6: Review the task diff without committing**

Run: `git diff --check -- src/occlusion_fer/metrics.py tests/test_metrics.py`

---

### Task 2: Reusable evaluation and prediction collection

**Files:**
- Create: `src/occlusion_fer/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `src/occlusion_fer/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `compute_classification_metrics` from Task 1.
- Produces: `PredictionRecord`, `EvaluationResult`, and `evaluate(model, loader, device, *, split="validation", condition="clean", amp_enabled=False) -> EvaluationResult`.
- `train.py` re-exports `EvaluationResult` and `evaluate` so existing imports continue to work.

- [ ] **Step 1: Write failing tests for prediction rows and probabilities**

Use a deterministic model and loader. Assert one prediction per sample, stable
sample IDs, seven probabilities summing to one, true/predicted label names,
`split`, `condition`, and `correct`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: collection fails because `occlusion_fer.evaluation` does not exist.

- [ ] **Step 3: Move inference evaluation into the new module**

```python
@dataclass(frozen=True)
class PredictionRecord:
    sample_id: int
    split: str
    condition: str
    true_label: int
    true_label_name: str
    predicted_label: int
    predicted_label_name: str
    correct: bool
    predicted_confidence: float
    probabilities: tuple[float, ...]

@dataclass(frozen=True)
class EvaluationResult:
    average_loss: float
    accuracy: float
    macro_f1: float
    sample_count: int
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    predictions: tuple[PredictionRecord, ...]
```

Run under `torch.inference_mode()`, retain AMP validation, move logits to CPU,
calculate softmax once per batch, reject duplicate sample IDs, and compute
metrics only after collecting all labels.

- [ ] **Step 4: Re-export the compatibility API from `train.py`**

Remove the old local `EvaluationResult` and `evaluate` implementation and add:

```python
from occlusion_fer.evaluation import EvaluationResult, evaluate
```

- [ ] **Step 5: Add failing validation tests**

Cover empty loader, non-7-class logits, duplicate sample IDs, invalid/non-finite
probabilities, no parameter/gradient changes, and AMP rejection on CPU.

- [ ] **Step 6: Run evaluation and existing train tests**

Run: `python -m pytest tests/test_evaluation.py tests/test_train.py -q`

Expected: all focused and compatibility tests pass.

- [ ] **Step 7: Review the task diff without committing**

Run: `git diff --check -- src/occlusion_fer/evaluation.py src/occlusion_fer/train.py tests/test_evaluation.py tests/test_train.py`

---

### Task 3: Atomic, paper-ready artifact writers

**Files:**
- Create: `src/occlusion_fer/artifacts.py`
- Create: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: `EvaluationResult` and `PredictionRecord` from Task 2.
- Produces:
  - `write_json_atomic(path, payload) -> Path`
  - `write_csv_atomic(path, fieldnames, rows) -> Path`
  - `write_resolved_config(output_directory, resolved_config) -> Path`
  - `write_history_artifacts(output_directory, history) -> tuple[Path, Path]`
  - `write_evaluation_artifacts(output_directory, prefix, result) -> dict[str, Path]`
  - `collect_run_metadata(...) -> dict[str, object]`
  - `write_failure_artifact(output_directory, *, stage, exception) -> Path`

- [ ] **Step 1: Write failing tests for JSON/CSV atomic output**

Assert newline-terminated UTF-8 JSON, deterministic CSV headers, directory
creation, resolved absolute return paths, and replacement of an existing target
without leaving the temporary sibling.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_artifacts.py -q`

Expected: collection fails because `occlusion_fer.artifacts` does not exist.

- [ ] **Step 3: Implement atomic JSON and CSV primitives**

Create the temporary file in the target directory with `tempfile.NamedTemporaryFile`,
flush and `os.fsync`, then replace using `Path.replace`. Convert write failures
to `RuntimeError` naming the intended target.

- [ ] **Step 4: Write failing artifact-schema tests**

Assert:

- `history.json` and `history.csv` carry the same epoch records;
- metric JSON contains schema version, split, condition, accuracy, macro-F1,
  per-class metrics, ordered labels, and confusion matrix;
- per-class CSV uses ordered labels;
- confusion CSV has true-label rows and seven predicted-label columns;
- predictions CSV contains stable base columns plus seven `probability_*`
  columns in FER2013 label order;
- resolved YAML contains runtime overrides;
- failure JSON excludes traceback/environment dumps.

- [ ] **Step 5: Implement the artifact schema**

Use `dataclasses.asdict` for metric records and explicit ordered field lists for
CSV. Include `metric_schema_version = 1` and `condition = "clean"`.

- [ ] **Step 6: Implement reproducibility metadata collection**

Use read-only `subprocess.run` calls for Git commit and `git status --porcelain`.
Record UTC timestamps, Python/platform/hostname, NumPy/PyTorch/torchvision
versions, seed, mode, device, CUDA device name, and relative artifact paths.
Do not capture all environment variables or credentials.

- [ ] **Step 7: Run artifact tests and verify GREEN**

Run: `python -m pytest tests/test_artifacts.py -q`

Expected: all artifact tests pass.

- [ ] **Step 8: Review the task diff without committing**

Run: `git diff --check -- src/occlusion_fer/artifacts.py tests/test_artifacts.py`

---

### Task 4: Macro-F1 checkpoint selection and formal training artifacts

**Files:**
- Modify: `src/occlusion_fer/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: evaluation and artifact interfaces from Tasks 2 and 3.
- Produces:
  - `is_better_validation_macro_f1(candidate: float, best: float) -> bool`
  - checkpoints containing `best_validation_macro_f1`, current validation
    accuracy/macro-F1, and existing model/optimizer/scaler/config fields.

- [ ] **Step 1: Write failing strict-comparison tests**

```python
def test_macro_f1_checkpoint_rule_requires_strict_improvement() -> None:
    assert is_better_validation_macro_f1(0.61, 0.60)
    assert not is_better_validation_macro_f1(0.60, 0.60)
    assert not is_better_validation_macro_f1(0.59, 0.60)
```

Also reject NaN, Inf, and values outside `[0, 1]`.

- [ ] **Step 2: Run focused test and verify RED**

Run: `python -m pytest tests/test_train.py -q -k macro_f1_checkpoint_rule`

Expected: import fails because the selector does not exist.

- [ ] **Step 3: Change checkpoint payload and selection state**

Replace `best_validation_accuracy` state with `best_validation_macro_f1`. Store:

```python
"best_validation_macro_f1": best_validation_macro_f1,
"validation_accuracy": validation_result.accuracy,
"validation_macro_f1": validation_result.macro_f1,
```

Keep model, optimizer, epoch, seed, device, resolved configuration, and optional
GradScaler state. Continue writing `last.pt` every epoch.

- [ ] **Step 4: Integrate best/last validation artifact output**

After every validation pass, write `validation/last_*`. On strict macro-F1
improvement, also write `validation/best_*`. Add validation macro-F1 and
`updated_best_checkpoint` to each history row and write both history formats.

- [ ] **Step 5: Write failing integration tests**

Use a tiny artificial FER2013 CSV and CPU model path. Assert best/last
checkpoints, history JSON/CSV, resolved config, run metadata, and all best/last
validation artifact files. Assert no PrivateTest prediction appears in training
artifacts.

- [ ] **Step 6: Add run lifecycle and failure records**

Write metadata with `running` before training, update it to `completed` after
the final epoch, and write status `failed` plus `failure.json` when orchestration
raises after the output directory is known. Preserve the original exception.

- [ ] **Step 7: Run the train tests and verify GREEN**

Run: `python -m pytest tests/test_train.py -q`

Expected: all train tests pass.

- [ ] **Step 8: Review the task diff without committing**

Run: `git diff --check -- src/occlusion_fer/train.py tests/test_train.py`

---

### Task 5: Seed override and explicit PrivateTest evaluation

**Files:**
- Create: `src/occlusion_fer/final_evaluate.py`
- Create: `tests/test_final_evaluate.py`
- Modify: `src/occlusion_fer/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Produces `python -m occlusion_fer.final_evaluate` with arguments:
  `--config`, `--checkpoint`, `--data-path`, `--output-dir`, `--device`,
  `--batch-size`, `--num-workers`, `--amp`, and `--confirm-private-test`.
- Adds `--seed` to the training CLI and existing config override functions.

- [ ] **Step 1: Write failing seed-override tests**

Assert `apply_config_overrides(seed=123)`, YAML immutability, parser support, and
rejection of negative, bool, float, and string seeds.

- [ ] **Step 2: Run focused seed tests and verify RED**

Run: `python -m pytest tests/test_train.py -q -k seed_override`

Expected: calls fail because no seed override is accepted.

- [ ] **Step 3: Implement the seed override**

Add a validated optional seed through `parse_args`, `load_config_with_overrides`,
and `apply_config_overrides`; update the training config copy only.

- [ ] **Step 4: Write failing PrivateTest gate tests**

Assert missing `--confirm-private-test` exits before `load_fer2013_csv` can be
called. Assert the parser requires checkpoint and output paths.

- [ ] **Step 5: Implement checkpoint loading validation**

Require a mapping containing `model_state_dict`, load with `map_location`, and
give clear errors for a missing file, malformed payload, incompatible model
state, or non-finite stored tensors. Accept legacy performance checkpoints that
lack macro-F1 fields.

- [ ] **Step 6: Implement final clean evaluation**

Load FER2013, build only `split="test"`, create the optimized DataLoader,
instantiate the configured ResNet-18 without downloading weights, load the
checkpoint state, run shared evaluation with `split="test"`, and write only
`final_test/clean_*` artifacts. Do not create an optimizer or update weights.

- [ ] **Step 7: Write CPU integration tests**

Create an artificial CSV with all three official splits and a compatible saved
model checkpoint. Assert output sample IDs come only from PrivateTest and model
parameters remain unchanged. Assert the training entry point never creates a
PrivateTest loader.

- [ ] **Step 8: Run final-evaluation and train tests**

Run: `python -m pytest tests/test_final_evaluate.py tests/test_train.py -q`

Expected: all focused tests pass.

- [ ] **Step 9: Review the task diff without committing**

Run: `git diff --check -- src/occlusion_fer/final_evaluate.py src/occlusion_fer/train.py tests/test_final_evaluate.py tests/test_train.py`

---

### Task 6: Paper-oriented documentation and server commands

**Files:**
- Modify: `README.md`
- Modify: `SERVER_RUN.md`
- Modify: `docs/server_runbook.md`
- Modify: `configs/fer2013_resnet18_clean.yaml` only if comments or values are
  required for consistency; keep its path placeholder and schema unchanged.

**Interfaces:**
- Consumes: actual CLI names and artifact paths implemented in Tasks 4 and 5.
- Produces: concise quick-start commands and detailed formal-run integrity rules.

- [ ] **Step 1: Update README scope and evidence boundary**

Document FER2013 dataset-label classification, official split use, ResNet-18,
clean stage status, module entry points, outputs, and prohibition on true
emotion/confusion claims.

- [ ] **Step 2: Update the five-minute server guide**

Add one validated clean smoke command using `--batch-size 128 --num-workers 4
--amp`, three formal clean commands using seeds `42`, `123`, and `2026`, and the
separate final evaluation command with `--confirm-private-test`. Keep data and
outputs outside Git and do not include Mac paths.

- [ ] **Step 3: Update the detailed runbook**

Explain PublicTest macro-F1 selection, fixed label order, each artifact, how
history/confusion/per-class/prediction CSVs support thesis figures and tables,
the PrivateTest lock, Git cleanliness, failed-run preservation, and HIVE process
checks.

- [ ] **Step 4: Add documentation consistency tests or static assertions**

Extend existing CLI parser tests and use `rg` checks to confirm no Mac absolute
path, token, password, or real CSV/checkpoint is introduced.

- [ ] **Step 5: Run documentation and configuration tests**

Run: `python -m pytest tests/test_config.py tests/test_train.py tests/test_final_evaluate.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Review documentation diff without committing**

Run: `git diff --check -- README.md SERVER_RUN.md docs/server_runbook.md configs/fer2013_resnet18_clean.yaml`

---

### Task 7: Full verification and integrity audit

**Files:**
- Verify all modified and created files.

**Interfaces:**
- Produces no new behavior; proves the stage meets its definition of done.

- [ ] **Step 1: Run the entire automated test suite**

Run: `python -m pytest -q`

Record exact passed, failed, skipped, warnings, and duration.

- [ ] **Step 2: Compile all source and tests**

Run: `python -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 3: Validate installed dependency consistency**

Run: `python -m pip check`

Expected: `No broken requirements found.`

- [ ] **Step 4: Run Git whitespace and scope checks**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff
```

Expected: only approved source, tests, documentation, and design/plan files.

- [ ] **Step 5: Scan for forbidden artifacts and secrets**

Use tracked and untracked file lists plus text searches to confirm there is no
FER2013 CSV/image data, `.pt`/`.pth`/`.ckpt`, output/log/run directory, virtual
environment, cache, `.env`, `kaggle.json`, API token, SSH key, Mac absolute data
path, or teacher-server absolute data path in the diff.

- [ ] **Step 6: Perform a final implementation-to-spec review**

Confirm every design requirement has an implementation and test, PrivateTest is
absent from training, macro-F1 selects best, artifact schemas are stable and
plot-ready, documentation commands match parser help, and no result is
fabricated.

- [ ] **Step 7: Report without committing or pushing**

Summarize modified files, purpose, test evidence, artifact layout, HIVE commands,
paper-writing uses, limitations, and remaining later stages. Stop for user
review before any commit or push.
