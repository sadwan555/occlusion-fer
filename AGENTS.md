# AGENTS.md

## Goal

Build a small, reproducible PyTorch project for studying ResNet-18 facial
expression classification on FER2013 under synthetic facial occlusion.

Work on one approved implementation stage at a time.

## First-version scope

Dataset:

- FER2013 only

Formal model:

- ImageNet-pretrained ResNet-18 only

Sanity model:

- small CNN, used only for pipeline checks

Occlusions:

- upper_face
- lower_face
- random_rectangle

Ratios:

- 0.20
- 0.30
- 0.40

Training strategies:

- clean-only
- mixed clean/occluded

Formal seeds:

- 42
- 123
- 2026

Metrics:

- accuracy
- macro-F1
- confusion matrix
- clean-to-occluded performance drop

## Research boundary

The project predicts dataset-defined facial-expression labels.

Do not claim that the model recognizes:

- true internal emotion;
- confusion or understanding;
- engagement;
- learning outcome;
- another cognitive state.

An online-classroom example is hypothetical HCI motivation only. It is not the
experimental application.

Do not claim novelty, state of the art, real-world robustness, or cross-dataset
generalization from the first-version experiments.

## Data rules

- Never commit FER2013 data or derived images.
- Never hard-code a local data path.
- Read paths from validated YAML configuration.
- Preserve the official Training, PublicTest, and PrivateTest split meanings.
- Never use PrivateTest for hyperparameter tuning or checkpoint selection.
- Retain original labels and mapped label names.
- Fail clearly on a missing path, empty split, malformed image, or invalid
  label.
- Do not silently skip data.

## Model and training rules

- Use the same ResNet-18 architecture for both formal training strategies.
- Replicate grayscale inputs to three channels and resize to 224×224.
- Keep the standard ResNet-18 stem.
- Use the same optimizer, epoch budget, batch size, base augmentation, seed set,
  and checkpoint rule for both strategies.
- Use the locked E7 recipe: mild affine Training augmentation, AdamW with
  learning rate 0.0001 and weight decay 0.001, label smoothing 0.1, no
  scheduler, 50 epochs, batch size 128, four workers, and CUDA AMP.
- Select the best checkpoint using strict improvement in clean PublicTest
  macro-F1; a tie retains the earlier checkpoint.
- Save best and last checkpoints.
- Do not report the small CNN as a research result.
- Do not change formal hyperparameters after inspecting final-test results.
- Treat E0-E7 seed-2026 runs as screening history, not formal evidence.

## Occlusion rules

- Use `occlusion-v2-224`; retain `occlusion-v1`/112 only as immutable
  development history.
- Do not modify source images in place.
- Apply only the approved three occlusion types and three ratios.
- Use the training-split pixel mean as the fill value.
- Record target and actual occlusion ratios.
- Use explicit deterministic seeds.
- All compared checkpoints must use the same final evaluation masks.
- Do not use test labels or test performance to generate masks.
- Use fixed PublicTest evaluation seed 20260804 and a validated v2 manifest.
- Reject combined all-splits CSV inputs before opening them on
  occlusion-enabled Stage B routes.

## Output rules

Each formal run must save:

- resolved configuration;
- seed;
- Git commit and dirty state;
- training history;
- best and last checkpoint;
- validation metrics;
- final per-condition metrics;
- per-sample predictions;
- confusion matrices;
- failure information when applicable.

Outputs, checkpoints, logs, and data must remain ignored by Git.

## Testing rules

Relevant tests must pass before a stage is considered complete.

Required first-version tests include:

- configuration validation;
- label mapping;
- official split mapping;
- split disjointness;
- batch shape;
- deterministic transforms;
- model output shape;
- one-batch forward and backward;
- occlusion position and ratio;
- unchanged source image;
- metric correctness;
- checkpoint save and load;
- smoke training.

Report the exact test command and result. Do not claim a test passed without
running it.

## Deferred features

Do not implement the following without new approval:

- RAF-DB or AffectNet;
- a second backbone;
- landmark or Grad-CAM occlusion;
- real-object occlusion;
- Transformers;
- knowledge distillation;
- reconstruction;
- complex attention modules;
- Hydra;
- complex statistical testing;
- profiling or publication-release infrastructure.

Deferred features are future possibilities, not requirements for the first
version.

## Git and integrity rules

- Preserve unrelated user changes.
- Do not use destructive Git commands.
- Do not commit unless explicitly requested.
- Never commit datasets, weights, outputs, credentials, or local paths.
- Do not fabricate results, references, dataset access, or conclusions.
- Keep failed and negative experiments.
- Report all three formal seeds, not only the best seed.

## Definition of done

A stage is complete only when:

1. only the approved scope was implemented;
2. its tests pass;
3. invalid input and error paths are handled;
4. no data, output, weight, or secret is tracked;
5. the relevant documentation is consistent with the implementation;
6. verification commands and results are reported;
7. remaining limitations are stated.

The complete first version additionally requires six formal ResNet-18 runs,
evaluation on all ten conditions, all per-seed results, aggregate results, and
answers to RQ1–RQ3 without exceeding the evidence boundary.
