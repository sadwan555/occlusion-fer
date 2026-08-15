# Research Context

## Question

This project studies how controlled synthetic occlusion changes seven-class
FER2013 label classification, and whether mixed clean/occluded training reduces
the resulting performance drop relative to clean-only training.

The research questions are:

- RQ1: Do upper-face, lower-face, and random-rectangle occlusions produce
  different performance drops?
- RQ2: How does performance change as the target occlusion ratio increases from
  0.20 to 0.40?
- RQ3: Does mixed training improve masked-condition performance without a clear
  loss on the clean condition?

## Scope

The first version uses only FER2013 and one ImageNet-pretrained, standard-stem
ResNet-18. The label order is:

```text
angry, disgust, fear, happy, sad, surprise, neutral
```

These are dataset-defined facial-expression categories. They are not verified
internal emotion, confusion, understanding, engagement, learning outcome, or
another cognitive state. An online-classroom example is hypothetical HCI
motivation only and is not the experimental application.

The evidence does not establish real-world robustness, cross-dataset
generalization, novelty, or state-of-the-art performance.

## Dataset Contract

Official FER2013 split meanings are preserved:

| Official split | Count | Use |
|---|---:|---|
| Training | 28,709 | optimization and Training-only pixel mean |
| PublicTest | 3,589 | clean validation, checkpoint selection, fixed-mask analysis |
| PrivateTest | 3,589 | final evaluation after protocol and checkpoints were frozen |

PrivateTest was not used for tuning, checkpoint selection, or mask generation.
Data and derived images remain outside Git, and paths are supplied through
validated configuration or explicit CLI arguments.

## Formal Protocol

- Input: 48x48 grayscale to float, bilinear resize to 224x224, replicate to
  three channels, ImageNet normalization.
- Backbone: ImageNet-pretrained ResNet-18 with the standard stem and seven-class
  output.
- Training strategies: clean-only and mixed clean/occluded.
- Formal seeds: 42, 123, 2026.
- Budget: 50 epochs, batch size 128, AdamW, learning rate 0.0001, weight decay
  0.001, label smoothing 0.1, no scheduler, no early stopping.
- Checkpoint rule: best clean PublicTest macro-F1, plus a last checkpoint.
- Protocol identity: `occlusion-v2-224`.
- Occlusions: upper face, lower face, and random rectangle at target ratios
  0.20, 0.30, and 0.40.
- Fill: Training-split global raw-pixel mean transformed to normalized channel
  values.
- Evaluation mask seed: 20260804.

Mixed training deterministically leaves a sample clean with probability 0.5;
otherwise it samples uniformly from the approved occlusion types and ratios.
All compared checkpoints use the same evaluation masks.

## Evidence Status

The first-version experimental batch is complete:

- three clean-only formal runs;
- three mixed formal runs;
- clean plus nine masked PublicTest conditions for the six best checkpoints;
- clean plus nine masked PrivateTest conditions for the same six frozen best
  checkpoints;
- per-seed metrics, aggregate tables, predictions, confusion matrices,
  provenance, and PrivateTest paper figures.

Formal code identities are `4cb1e0f` for E7 clean training, `c1c9187` for Stage
8 mixed training/PublicTest evaluation, and `7e154ac` for PrivateTest final
evaluation. The earlier `da889bd` 112x112/30-epoch lineage is legacy and must
not be mixed with the current evidence.

## Reporting

Report every formal seed, followed by the mean and sample standard deviation
(`ddof=1`). Report accuracy, macro-F1, confusion matrices, per-class metrics,
and paired clean-to-occluded drops. Failed, negative, and non-best-seed results
must not be hidden.

The small CNN, smoke runs, screening experiments, synthetic fixtures, and
legacy 112/v1 outputs are engineering or historical evidence only.
