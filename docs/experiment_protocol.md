# FER2013 E7 / Occlusion-v2-224 Experiment Protocol

Status: current authoritative protocol for the first-version study.

This document supersedes earlier documents wherever they describe E0-E7
screening, `112 x 112`, or `occlusion-v1` as the current formal experiment.
Those documents remain valid only as development history. Protocol changes
require explicit approval before any additional formal run is started.

## 1. Evidence status

- E0-E7 seed-2026 runs are recipe screening and are not formal three-seed
  evidence.
- The locked recipe is E7 with the `occlusion-v2-224` integration described
  below.
- Stage 7 local engineering validation does not constitute research
  performance evidence; HIVE/Linux validation is still required before formal
  server execution.
- Stage 8 formal training has not started.
- No completed Stage B clean-versus-mixed three-seed result exists yet.
- PrivateTest has not been accessed under this protocol.
- Smoke, synthetic, unit-test, and artificial-training outputs are engineering
  evidence only.

## 2. Research scope and claim boundary

The task is seven-class classification of the labels defined by FER2013:

```text
angry, disgust, fear, happy, sad, surprise, neutral
```

The first version uses only FER2013 and one ImageNet-pretrained, standard-stem
ResNet-18. It compares clean-only training with mixed clean/occluded training
under controlled synthetic occlusion.

The study does not establish recognition of internal emotion, confusion,
understanding, engagement, learning outcome, or another cognitive state. It
does not establish real-world robustness, cross-dataset generalization,
novelty, or state-of-the-art performance. An online-classroom example is
hypothetical HCI motivation only and is not the experimental application.

## 3. Dataset and split contract

The official FER2013 split meanings and expected counts are fixed:

| Official split | Count | Permitted use |
|---|---:|---|
| Training | 28,709 | optimization, training-only augmentation, Training pixel mean |
| PublicTest | 3,589 | clean validation, checkpoint selection, screening, fixed-mask occlusion analysis |
| PrivateTest | 3,589 | one final evaluation after recipe, checkpoints, and masks are locked |

PrivateTest must not be used for recipe selection, hyperparameter tuning,
checkpoint selection, mask generation, or Stage B integration validation.

Occlusion-enabled Stage B routes accept only a Training-only source, a
PublicTest-only source, or a validated permitted-splits artifact containing
exactly those sources. A combined all-splits CSV is rejected before it is
opened by those routes. Dataset identities are computed separately for
Training and PublicTest from canonical sample records; PrivateTest content does
not participate in either identity.

Data paths remain validated YAML values or explicit runtime overrides and must
never be committed as machine-specific paths. Data, derived images,
checkpoints, logs, and outputs remain outside Git.

## 4. Model and preprocessing

The formal model and input pipeline are fixed:

- torchvision ResNet-18 with ImageNet pretrained weights;
- standard ResNet-18 stem and a seven-output classification head;
- 48 x 48 uint8 grayscale pixels converted to float32 and divided by 255;
- bilinear resize to `224 x 224`;
- Training-only mild affine augmentation;
- replication of the grayscale channel to three channels;
- ImageNet channel normalization.

The Training augmentation is:

| Setting | Value |
|---|---:|
| horizontal flip probability | 0.5 |
| affine probability | 0.5 |
| rotation | +/-7 degrees |
| translation | up to 5% per axis |
| scale | [0.97, 1.03] |
| interpolation | bilinear |
| augmentation fill | 0.0 in the pre-normalized image domain |

PublicTest receives no random augmentation. Historical `112 x 112` processing
belongs to E0-E5 and occlusion-v1 development and is not the current formal
input protocol.

## 5. Locked E7 training recipe

Both formal strategies use the same model, optimizer, loss, schedule, budget,
augmentation, runtime settings, seed set, and checkpoint rule:

| Setting | Locked value |
|---|---|
| optimizer | AdamW |
| learning rate | 0.0001 |
| weight decay | 0.001 |
| training loss | cross entropy with label smoothing 0.1 |
| validation loss | cross entropy without label smoothing |
| scheduler | none |
| epochs | 50 |
| batch size | 128 |
| DataLoader workers | 4 |
| early stopping | disabled |
| runtime AMP | enabled on CUDA |
| formal seeds | 42, 123, 2026 |

The repository configurations are:

- clean-only: `configs/experiments/fer2013_resnet18_e7_clean.yaml`;
- mixed: `configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml`.

The YAML seed is a reproducible default. Each formal run must override it with
exactly one member of the formal seed set and use a fresh output directory.
This yields six formal runs: three clean-only and three mixed.

Every epoch evaluates the clean PublicTest split. `best.pt` is replaced only
when clean PublicTest macro-F1 strictly improves; a tie retains the earlier
checkpoint. Masked PublicTest metrics must never select a checkpoint. Both
`best.pt` and `last.pt` are retained. Formal hyperparameters must not change
after any final PrivateTest result is inspected.

## 6. Occlusion-v2-224 protocol

The current algorithm identity is `occlusion-v2-224`. Historical
`occlusion-v1` / 112 artifacts are immutable development records and are
incompatible with v2 consumers.

The nine masked conditions, in canonical order, are:

```text
upper_face_0.20
upper_face_0.30
upper_face_0.40
lower_face_0.20
lower_face_0.30
lower_face_0.40
random_rectangle_0.20
random_rectangle_0.30
random_rectangle_0.40
```

Masks use half-open coordinates. Dimensions use half-up rounding:

| Family | Target ratio | Geometry | Masked pixels | Actual ratio |
|---|---:|---:|---:|---:|
| upper/lower band | 0.20 | 45 x 224 | 10,080 | 0.2008928571 |
| upper/lower band | 0.30 | 67 x 224 | 15,008 | 0.2991071429 |
| upper/lower band | 0.40 | 90 x 224 | 20,160 | 0.4017857143 |
| random rectangle | 0.20 | 100 x 100 | 10,000 | 0.1992984694 |
| random rectangle | 0.30 | 123 x 123 | 15,129 | 0.3015186543 |
| random rectangle | 0.40 | 142 x 142 | 20,164 | 0.4018654337 |

The fill is the Training-split global mean of raw 48 x 48 grayscale pixels,
transformed into the three ImageNet-normalized channel values. The source clean
tensor is cloned before fill; source pixels are not modified in place. Every
artifact records both target and actual ratios plus mean provenance.

Canonical hash payloads use compact UTF-8 JSON, SHA-256, and the unsigned
big-endian integer represented by the first eight digest bytes. They never use
labels, predictions, metrics, Python's built-in hash, worker order, batch
order, or CUDA RNG.

Mixed training makes one deterministic decision for each tuple of sample ID,
formal seed, and one-based epoch:

- 0.5 probability remains clean;
- 0.5 probability is masked;
- for a masked sample, type and ratio are independently uniform over the three
  approved choices;
- random-rectangle top and left coordinates use independent hash namespaces.

PublicTest evaluation uses fixed seed `20260804` and one validated v2 manifest
shared by every clean and mixed checkpoint. The manifest contains
`3,589 x 9 = 32,301` rows. A manifest or mean with the wrong source identity,
protocol, dimensions, order, digest, or provenance is rejected rather than
silently regenerated.

## 7. Evaluation and reporting

Each of the six best checkpoints is evaluated on the same ten PublicTest
conditions: clean plus the nine masked conditions. PublicTest masked evaluation
is analysis only and cannot change the checkpoint, recipe, seed set, or masks.

For every seed, strategy, and condition, retain:

- accuracy and seven-class macro-F1;
- the 7 x 7 confusion matrix, with rows as true labels and columns as predicted
  labels;
- per-class precision, recall, F1, and support;
- per-sample predictions and provenance;
- clean-to-occluded accuracy and macro-F1 drops.

For a metric `M`, the paired per-seed drop is:

```text
drop(seed, condition) = M(seed, clean) - M(seed, condition)
```

Report all three seeds, followed by the mean and sample standard deviation.
Do not select the best seed, omit failed or negative runs, or make complex
significance claims from three seeds.

PrivateTest remains sealed until all six best-checkpoint SHA-256 values, the E7
recipe, v2 mean, v2 mask identities, evaluation code, and reporting plan are
locked. Its eventual execution is one final evaluation batch and must not feed
back into any design choice. The current PublicTest-only occlusion evaluator has
no PrivateTest route.

## 8. Stage gates

1. Preserve E0-E7 outputs as screening history; do not count seed 2026 from
   screening toward the formal runs.
2. Snapshot the Stage B implementation in a clean, reviewable Git commit.
3. Complete HIVE/Linux synthetic and test validation without FER2013 inference.
4. Generate and validate Training/PublicTest v2 source identities, mean, and
   manifest without reading PrivateTest.
5. Run the six Stage 8 formal trainings from one locked commit and retain every
   completed or failed run.
6. Evaluate all six best checkpoints on the same ten PublicTest conditions and
   freeze all provenance.
7. Only then authorize the single final PrivateTest batch.

At the time of this protocol update, gates 2-7 are not complete. No pending
result may be represented by a placeholder number.

## 9. Required run provenance

Every formal run must preserve the resolved configuration, formal seed, Git
commit and dirty state, environment and device metadata, training history,
best and last checkpoints, clean PublicTest checkpoint-selection artifacts,
condition metrics, predictions, confusion matrices, checkpoint/config/artifact
SHA-256 values, and bounded failure information when applicable.

Synthetic fixtures and engineering validation outputs must be physically and
semantically separated from formal research results.

## 10. Deferred scope

RAF-DB, AffectNet, another backbone, landmark or Grad-CAM masking, real-object
occlusion, Transformers, distillation, reconstruction, complex attention,
Hydra, complex statistical testing, and publication-release infrastructure
remain outside the first version unless separately approved.
