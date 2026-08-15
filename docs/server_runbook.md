# Formal Experiment and Artifact Runbook

## Purpose

This runbook describes the completed 224x224 E7/Stage 8 lineage and the frozen
PrivateTest final route. It replaces earlier 112x112/30-epoch server guidance.
Historical 112/v1 plans remain in `docs/superpowers/` for provenance only.

The normal post-project operation is artifact verification and figure
regeneration. Do not retrain or rerun PrivateTest merely to reorganize files.

## Fixed Identities

| Stage | Git commit |
|---|---|
| E7 clean formal training | `4cb1e0ffe4b55efc090a45cfed560b28f50b9509` |
| Stage 8 mixed training/PublicTest evaluation | `c1c9187aa2ddf7dd84906c7f139ad9a750ef202d` |
| PrivateTest final evaluation | `7e154aca1e95ef78ea7e3bc8767bcb21ca769335` |

Formal seeds are 42, 123, and 2026. The protocol is `occlusion-v2-224`, the
evaluation mask seed is 20260804, and the condition order is clean followed by
upper/lower/random rectangle at 0.20, 0.30, and 0.40.

## Server Layout

Keep code, data, environments, and results separate:

```bash
export FER_ROOT="${HOME}/anson-fer"
export FER_REPO="${FER_ROOT}/repo/occlusion-fer"
export FER_DATA="${FER_ROOT}/data/fer2013.csv"
export FER_RESULTS="${FER_ROOT}/results"
export FER_ENV="${FER_ROOT}/envs/occlusion-fer"
export FER_CLEAN_WORKTREE="${FER_ROOT}/worktrees/e7-clean"
export FER_STAGE8_WORKTREE="${FER_ROOT}/worktrees/stage8"
export FER_PRIVATE_WORKTREE="${FER_ROOT}/worktrees/private-final"
```

FER2013 data, checkpoints, logs, outputs, generated figures, credentials, and
release archives must remain outside Git.

## Checkout

```bash
git clone git@github.com:sadwan555/occlusion-fer.git "${FER_REPO}"
cd "${FER_REPO}"
git worktree add "${FER_CLEAN_WORKTREE}" 4cb1e0ffe4b55efc090a45cfed560b28f50b9509
git worktree add "${FER_STAGE8_WORKTREE}" c1c9187aa2ddf7dd84906c7f139ad9a750ef202d
git worktree add "${FER_PRIVATE_WORKTREE}" 7e154aca1e95ef78ea7e3bc8767bcb21ca769335
```

Record `git rev-parse HEAD` and `git status --short` before every verification or
reproduction action.

## Environment

Install a CUDA-compatible PyTorch build appropriate for the server before the
editable project install when necessary.

```bash
python3 -m venv "${FER_ENV}"
source "${FER_ENV}/bin/activate"
python -m pip install --upgrade pip
cd "${FER_STAGE8_WORKTREE}"
python -m pip install -e ".[test]"
python -m pip check
PYTHONPATH=src pytest -q
python3 -m compileall -q src tests
```

Do not print credential contents. Do not copy a local absolute data path into a
tracked YAML file; use `--data-path` or a server-local untracked configuration.

## Dataset Preflight

The expected official counts are Training 28,709, PublicTest 3,589, and
PrivateTest 3,589. Training and PublicTest must be used for training/validation
exactly as defined by FER2013.

For E7 clean:

```bash
cd "${FER_CLEAN_WORKTREE}"
python -m occlusion_fer.preflight \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path "${FER_DATA}" \
  --output-dir "${FER_RESULTS}/preflight/e7-clean" \
  --device cuda --batch-size 128
```

Do not proceed if the preflight fails, a split is empty, a source identity is
wrong, or the output directory already contains unrelated artifacts.

## E7 Clean Reproduction

The clean formal config is
`configs/experiments/fer2013_resnet18_e7_clean.yaml` at commit `4cb1e0f`. Run
all three seeds with identical hyperparameters and separate output directories:

```bash
python -m occlusion_fer.train \
  --config configs/experiments/fer2013_resnet18_e7_clean.yaml \
  --data-path "${FER_DATA}" \
  --output-dir "${FER_RESULTS}/formal-clean/seed42" \
  --seed 42 --device cuda --epochs 50 \
  --batch-size 128 --num-workers 4 --amp
```

Repeat only by replacing both the seed and output directory with 123 and 2026.
Preserve every best/last checkpoint, resolved config, history, validation
artifact, metadata file, log, and failure file.

## Stage 8 Artifacts and Mixed Training

Switch to commit `c1c9187`. Generate Training mean v2 and the PublicTest v2
manifest into a new external directory:

```bash
cd "${FER_STAGE8_WORKTREE}"
python -m occlusion_fer.stage_b_artifacts \
  --data-path "${FER_DATA}" \
  --output-dir "${FER_RESULTS}/stage8-protocol-artifacts"
```

The mixed config is
`configs/experiments/fer2013_resnet18_e7_occlusion_mixed.yaml`. Use an untracked
resolved copy whose artifact paths point to the generated Training mean and
PublicTest manifest. Run seeds 42, 123, and 2026 with the same 50-epoch budget,
batch size, optimizer, augmentation, checkpoint rule, and distinct output
directories used for clean training.

The mixed sampler uses 0.5 clean probability and only the approved three types
and three ratios. It must not read PrivateTest labels or pixels.

## PublicTest Evaluation

Use `occlusion_fer.occlusion_evaluate` at `c1c9187` for each of the six best
checkpoints. Every run must use the same v2 Training mean and PublicTest
manifest and a fresh output directory.

```bash
python -m occlusion_fer.occlusion_evaluate \
  --config /path/to/resolved-evaluation.yaml \
  --checkpoint /path/to/approved-best.pt \
  --data-path "${FER_DATA}" \
  --training-mean "${FER_RESULTS}/stage8-protocol-artifacts/training_mean_v2.json" \
  --manifest "${FER_RESULTS}/stage8-protocol-artifacts/publictest_manifest_v2.csv" \
  --output-dir /path/to/new-evaluation-output \
  --device cuda --batch-size 128 --num-workers 4 --amp \
  --training-commit 4cb1e0ffe4b55efc090a45cfed560b28f50b9509
```

The example is for a clean E7 checkpoint. Use
`c1c9187aa2ddf7dd84906c7f139ad9a750ef202d` for a mixed checkpoint. Do not use
`--allow-nonofficial` for formal data.

## PrivateTest Final Route

The existing final evaluation was executed from clean commit `7e154ac` with a
self-hashed plan and six frozen checkpoint identities. Do not rerun it for
cleanup or figure work.

For an independently approved reproduction, all of the following must pass
before inference:

- explicit `--confirm-private-test` gate;
- PrivateTest canonical identity and count;
- `occlusion-v2-224` manifest and sidecar;
- Training mean v2 identity;
- exact SHA-256, seed, epoch, architecture, input size, and role for all six
  best checkpoints;
- new or empty output root;
- plan commit and dirty-state provenance.

The interface is:

```bash
cd "${FER_PRIVATE_WORKTREE}"
python -m occlusion_fer.private_final \
  --plan /path/to/final_evaluation_plan.json \
  --data-path "${FER_DATA}" \
  --training-mean /path/to/training_mean_v2.json \
  --manifest /path/to/privatetest_manifest_v2.csv \
  --device cuda --batch-size 128 --num-workers 4 --amp \
  --confirm-private-test
```

Never change a checkpoint, hyperparameter, mask, or report rule after seeing
PrivateTest performance.

## Artifact Verification

Prefer read-only verification of archived results:

```bash
shasum -a 256 /path/to/release-asset.tar
tar -tf /path/to/release-asset.tar
```

The current local `.tar` files and historical checksum sidecars created for
`.tar.gz` are different byte streams. Do not treat the old `.tar.gz` checksum as
the checksum of an uncompressed `.tar`. See `docs/artifact_inventory.md` before
publishing release assets.

## Figure Regeneration

PrivateTest figures consume frozen result files only:

```bash
cd "${FER_PRIVATE_WORKTREE}"
python -m pip install -e ".[paper]"
python -m occlusion_fer.private_paper_figures \
  --private-root /path/to/extracted/final-private-test-v2 \
  --output-dir /path/to/new-private-test-figures
```

The output directory must be new or empty. Do not modify source results,
checkpoints, manifests, or provenance while producing figures.

## Failure Handling

- Preserve `failure.json`, logs, partial outputs, and the exact command.
- Do not overwrite or silently resume a failed formal run.
- Do not silently skip malformed data or incomplete conditions.
- Use a new clearly named directory only after reviewing the failure.
- Report all three seeds and all ten conditions, including negative results.
