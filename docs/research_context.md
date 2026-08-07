# Research Context

## Project

This project studies occlusion-robust static facial expression recognition on
FER2013.

## Research scope

The model predicts one of seven FER2013 dataset labels:

- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

These labels describe dataset-defined facial-expression categories. They must
not be interpreted as verified internal emotion, confusion, understanding,
engagement, or another cognitive state.

An online-classroom example may be used only as hypothetical HCI motivation. It
is not the experimental setting and does not determine the dataset or model.

## First-version objective

The first version trains an ImageNet-pretrained ResNet-18 and measures its
performance under:

- clean images;
- upper-face occlusion;
- lower-face occlusion;
- random rectangular occlusion.

Occlusion ratios are limited to:

- 20%;
- 30%;
- 40%.

A small CNN is used only to verify the training pipeline.

The current formal protocol is `occlusion-v2-224` with the locked E7 recipe.
Earlier E0-E7 seed-2026 runs and `occlusion-v1` / 112 artifacts are development
history, not formal evidence.

## Research questions

RQ1: Do upper-face, lower-face, and random rectangular occlusions produce
different performance drops?

RQ2: How does performance change when the occlusion ratio increases from 20% to
40%?

RQ3: Does mixed clean/occluded training improve occluded performance without
clearly reducing clean performance?

## Dataset

Only FER2013 is included in the first version.

The official split meanings are preserved:

- Training: model training;
- PublicTest: validation and checkpoint selection;
- PrivateTest: final testing only.

Expected official counts are 28,709 Training, 3,589 PublicTest, and 3,589
PrivateTest samples. PrivateTest remains sealed until the recipe, all six best
checkpoints, masks, and reporting plan are locked.

Data must remain outside Git. Paths are supplied through YAML configuration.
Original numeric labels and mapped label names must both be retained.

## Models

Formal model:

- torchvision ResNet-18;
- ImageNet pretrained weights;
- standard ResNet stem;
- seven-class output layer;
- grayscale images replicated to three channels;
- input resized bilinearly to 224×224.

Sanity model:

- a small CNN;
- not included in the formal comparison.

## Training strategies

### Clean-only

Training images receive only the shared basic augmentation.

### Mixed clean/occluded

For each training sample:

- 50% probability of remaining clean;
- 50% probability of receiving an occlusion.

Occlusion type and ratio are sampled uniformly from the approved first-version
conditions.

Both strategies use the same model, data split, optimizer, training budget,
random seeds, checkpoint rule, and evaluation masks.

The shared E7 recipe uses Training-only mild affine augmentation, AdamW with
learning rate 0.0001 and weight decay 0.001, label smoothing 0.1, no scheduler,
50 epochs, batch size 128, four DataLoader workers, and CUDA AMP. Checkpoints
are selected only by strict improvement in clean PublicTest macro-F1.

## Formal runs

Formal experiments use three seeds:

- 42
- 123
- 2026

The evidence set contains six formal ResNet-18 runs: three locked clean-only
runs and three mixed runs. Stage 8 adds only the mixed runs; it must not retrain
or overwrite the locked clean checkpoints.

Stage 8 mixed training has not started. E0-E7 screening runs do not count
toward the formal evidence set, and no formal Stage B result is currently
available.

## Evaluation

Each checkpoint is evaluated on ten conditions:

- clean;
- upper face at 20%, 30%, and 40%;
- lower face at 20%, 30%, and 40%;
- random rectangle at 20%, 30%, and 40%.

All compared checkpoints use the same deterministic evaluation masks.
PublicTest masks use v2 evaluation seed 20260804 and a validated 32,301-row
manifest. Masked PublicTest performance cannot select a checkpoint.

Reported metrics are:

- accuracy;
- macro-F1;
- confusion matrix;
- clean-to-occluded accuracy drop;
- clean-to-occluded macro-F1 drop.

Results report every seed and mean plus standard deviation. No complex
significance claim is made from three seeds.

## Claim boundary

The project may report controlled findings for FER2013 and the specified
synthetic occlusion protocol.

It must not claim:

- recognition of true internal emotion or cognitive state;
- validation of an online-classroom application;
- real-world occlusion robustness;
- cross-dataset generalization;
- algorithmic novelty;
- state-of-the-art performance.

## Deferred work

RAF-DB, AffectNet, a second backbone, landmarks, Grad-CAM masks, real-object
occlusions, Transformers, distillation, reconstruction, complex attention,
Hydra, advanced statistics, and release infrastructure are deferred until the
first version is complete.
