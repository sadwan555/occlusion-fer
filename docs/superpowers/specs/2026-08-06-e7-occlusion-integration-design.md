# Approved Stage 5 Design: E7 Occlusion Integration

> Current implementation design, with the executable experiment protocol
> consolidated in [`../../experiment_protocol.md`](../../experiment_protocol.md).
> The design's non-goal statements describe the earlier implementation gate;
> they do not override the current Stage 8 status in the protocol.

Status: approved Stage 4 v2 design, copied outside the repository for later
Stage 6 use. This file is authoritative for implementation planning; it does not
authorize implementation.

## Locked identities and scope

- E7 base: `4cb1e0ffe4b55efc090a45cfed560b28f50b9509`.
- Stage A core commits allowed later: `6efe243eca5ab4b39251fc1a7a11c4c82ebcff28`
  and `78bb35d713c4e76f24bc378965fd6f2549b47b5b`.
- New protocol: `occlusion-v2-224`.
- Historical `occlusion-v1`/112 artifacts remain immutable and fail closed in v2.
- Types: `upper_face`, `lower_face`, `random_rectangle`.
- Ratio tokens: strings `"0.20"`, `"0.30"`, `"0.40"`.
- No new model, dataset, occlusion type/ratio, landmark, attention,
  reconstruction, real-object occlusion, or research claim.

## E7 behavior to preserve

The existing YAML hierarchy is unchanged:

```text
project
dataset
  augmentation
model
training
  loss
  scheduler
  early_stopping
output
```

E7 remains 224x224, ResNet-18 ImageNet-pretrained, mild affine training-only
augmentation, 50 epochs, batch 128, AdamW lr `0.0001`, weight decay `0.001`,
label smoothing `0.1`, scheduler `none`, early stopping disabled, runtime AMP,
and seeds 42/123/2026 for formal runs. Clean best selection is strict `>` on
clean PublicTest macro-F1 (`validation` internally). Clean E7 configs without an
`occlusion` block load unchanged.

Current input order:

```text
48x48 uint8 grayscale -> float32 / 255 -> bilinear resize 224
-> dataset.augmentation only for train -> replicate RGB
-> ImageNet normalization -> device transfer -> existing autocast/model/loss
```

Masking is inserted after device transfer and before the existing E7 autocast
region. The current autocast encloses model forward and loss only; backward,
scaler update, and optimizer step remain outside it.

## Protocol identity and exact hash rules

Every payload is a compact UTF-8 JSON array serialized with:
`ensure_ascii=true`, `allow_nan=false`, `separators=(",", ":")`. Integers remain
JSON integers. Ratios are fixed strings, never JSON floats. SHA-256 covers only
canonical payload bytes. The first eight digest bytes become unsigned big-endian
u64. No label, prediction, metric, Python hash, or CUDA RNG enters a payload.

Training apply:

```json
["fer2013","train",sample_id,training_seed,epoch,224,224,"occlusion-v2-224","apply"]
```

Training type:

```json
["fer2013","train",sample_id,training_seed,epoch,224,224,"occlusion-v2-224","type"]
```

Training ratio:

```json
["fer2013","train",sample_id,training_seed,epoch,224,224,"occlusion-v2-224","ratio"]
```

Training random-rectangle top:

```json
["fer2013","train",sample_id,training_seed,epoch,224,224,"occlusion-v2-224","top",condition_token,occlusion_type,ratio_token,"top"]
```

Training random-rectangle left:

```json
["fer2013","train",sample_id,training_seed,epoch,224,224,"occlusion-v2-224","left",condition_token,occlusion_type,ratio_token,"left"]
```

Validation/PublicTest top and left use the same fields, replacing training seed
and epoch with fixed evaluation seed `20260804`:

```json
["fer2013","validation",sample_id,20260804,224,224,"occlusion-v2-224","top",condition_token,occlusion_type,ratio_token,"top"]
["fer2013","validation",sample_id,20260804,224,224,"occlusion-v2-224","left",condition_token,occlusion_type,ratio_token,"left"]
```

Condition tokens are exact strings such as `random_rectangle:0.20`; type order
is `["upper_face","lower_face","random_rectangle"]`; ratio order is
`["0.20","0.30","0.40"]`. Apply is `u64 < 2**63`; type and ratio are
independent `u64 % 3`; random coordinates are `u64 % (224-side+1)`. Epoch is
one-based and explicit. Top/left are independent namespaces. This preserves the
historical Stage A predicate and modulo mapping while making v2 identity explicit.

## Data-source identity and artifacts

Stage B accepts only a Training-only source, a PublicTest-only source, or a
permitted-splits artifact containing only those two sources. It rejects combined
all-splits CSV before opening it; it never whole-file-hashes, scans, parses,
splits, or reads PrivateTest. If only a combined server CSV exists, Stage B is
blocked.

`training_dataset_sha256` and `publictest_dataset_sha256` are separate SHA-256
identities over canonical UTF-8 LF-delimited compact JSON records in increasing
sample ID order:

```json
{"sample_id":int,"label":int,"pixels":[int,...]}
```

Training mean mathematics remain raw Training-only 48x48 pixels with uint64
accumulation and `training-mean-v1` math. Mean schema v2 records source size,
consumer size 224, preprocessing identity, fill domain, and
`training_dataset_sha256`.

Artifact rule for mean, manifest, and protocol metadata:

```text
semantic canonical payload (no digest field)
-> exact canonical bytes
-> SHA-256
-> envelope/sidecar stores the digest
```

No digest participates in its own digest. The manifest is canonical CSV plus a
metadata sidecar, schema v2, consumer dimensions 224x224, PublicTest identity,
mean artifact SHA, algorithm version, condition order, geometry, pixel counts,
actual ratio, and fill provenance. Expected rows are `3589*9=32301`. The loader
fails closed on identity/version/dimension/mean/hash/order/duplicate/row-count or
canonical-byte mismatch and never silently regenerates geometry.

## Clean, mixed, evaluator, and checkpoint flows

Clean E7 training remains unchanged and does not require mean or manifest artifacts.

Clean-checkpoint occlusion evaluation loads only separate Training/PublicTest
identities, validates one v2 manifest shared across checkpoints, strict-loads the
E7 checkpoint, evaluates clean plus the nine fixed masked conditions with the
existing evaluation criterion/metrics, and writes paired clean-to-occluded drops.

Mixed training requires `training.mode: mixed`, separate permitted sources, and
v2 mean provenance. DataLoader returns E7-preprocessed clean tensors and IDs;
after device transfer, explicit seed/epoch decisions select masks per sample.
Clean samples bypass masking; masked samples clone-and-fill. Validation remains
clean-only and selects best strictly by clean PublicTest macro-F1. Optimizer, loss,
scheduler, early stopping, and existing AMP scope remain unchanged.

The evaluator is a new narrow `src/occlusion_fer/occlusion_evaluate.py`; it does
not import/call `final_evaluate.py` and has no PrivateTest route. Per checkpoint
and condition it writes metrics JSON, predictions CSV, per-class metrics,
confusion matrix, paired drop, and provenance containing checkpoint/training/
evaluation commits, Git dirty state, resolved config, split SHAs, mean SHA,
manifest SHA, protocol version, dimensions, seed, condition, and run role.

Clean E7 checkpoints strict-load without future occlusion fields. Masked v2
evaluation rejects v1/112 artifacts and incompatible split/mean/manifest/config
identity. Mixed checkpoints record mode, recipe identity, protocol, sampling,
mean SHA, resolved config, seed, and training commit.

## Geometry expectations

Using half-up rounding and `224*224=50176`:

| Family | Ratio | Geometry | Pixels |
|---|---:|---:|---:|
| upper/lower band | 0.20 | 45x224 | 10080 |
| upper/lower band | 0.30 | 67x224 | 15008 |
| upper/lower band | 0.40 | 90x224 | 20160 |
| random rectangle | 0.20 | 100x100 | 10000 |
| random rectangle | 0.30 | 123x123 | 15129 |
| random rectangle | 0.40 | 142x142 | 20164 |

Actual ratio is independently computed as `masked_pixel_count/50176`.

## Later Git integration and non-goals

After Stage 6 approval only, create branch `stage-b/e7-occlusion-integration`
and worktree `${INTEGRATION_WORKTREE}`
from the locked E7 ref. Cherry-pick only the two approved Stage A commits, resolve
conflicts in favor of E7 clean behavior, and preserve historical v1/112 files.
Do not touch main, existing worktrees, formal checkpoints/results, or the dirty
test-exit worktree. No commit is permitted without later explicit approval.

Non-goals: formal runs, real PublicTest masked metrics, PrivateTest, final_evaluate,
new models/datasets/occlusions, research claims, push/PR/merge, or repository
documentation before Stage 6.

## Copy target after Stage 6 approval

Only after the new integration worktree exists, copy this approved design to:

```text
docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md
```

The copy must be made in the new integration worktree, never in main or an existing
worktree.
