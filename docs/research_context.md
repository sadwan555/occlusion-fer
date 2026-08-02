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

Data must remain outside Git. Paths are supplied through YAML configuration.
Original numeric labels and mapped label names must both be retained.

## Models

Formal model:

- torchvision ResNet-18;
- ImageNet pretrained weights;
- standard ResNet stem;
- seven-class output layer;
- grayscale images replicated to three channels;
- input resized to 112×112.

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

## Formal runs

Formal experiments use three seeds:

- 42
- 123
- 2026

This produces six formal ResNet-18 training runs: three clean-only and three
mixed.

## Evaluation

Each checkpoint is evaluated on ten conditions:

- clean;
- upper face at 20%, 30%, and 40%;
- lower face at 20%, 30%, and 40%;
- random rectangle at 20%, 30%, and 40%.

All compared checkpoints use the same deterministic evaluation masks.

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
