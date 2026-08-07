# Stage 5 Implementation Plan: E7 Occlusion Integration

> Historical planning record. The implementation is now present as local
> uncommitted work in the integration worktree. Use
> [`../../experiment_protocol.md`](../../experiment_protocol.md) for the
> current protocol and Stage 8 status; the planning-only statements below are
> not current repository status.

Status: planning only; approved Stage 4 v2 is the design authority. No task below
has been executed in Stage 5. No repository file, branch, worktree, test, config,
dataset, checkpoint, output, or commit is created by this plan.

Task count: 39 sequential tasks (A1 through P1). Estimated implementation effort:
6-9 engineer-hours, excluding HIVE queue time and any separate investigation of a
macOS test-process hang.

## Execution contract

Future implementation root, created only after Stage 6 approval:

```text
branch: stage-b/e7-occlusion-integration
worktree: ${INTEGRATION_WORKTREE}
base: 4cb1e0ffe4b55efc090a45cfed560b28f50b9509
```

`REPO_ROOT`, `INTEGRATION_WORKTREE`, and `TEST_EXIT_WORKTREE` below are
environment-specific shell variables. They must be resolved outside versioned
configuration and must never be replaced with a committed machine-local path.

Every code task follows TDD: write the named failing test, run the exact failing
command and record its real non-zero exit, make the smallest production change,
run the focused passing command, run the listed regression command, record evidence,
and stop if the expected failure or regression differs. Future test commands use
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider`.

The literal failing, focused-passing, and regression commands for every code task
are written in that task block. A failing command is run before the production
change and must exit non-zero; the same focused command is rerun after the change
and must exit zero. Every command below resolves its shell and filesystem targets.

The existing `${TEST_EXIT_WORKTREE}` dirty
worktree and its `persistent_workers=False` edit are never touched or copied.
Stage B commands reject combined FER2013 CSV before opening it; real server
training is blocked until Training-only and PublicTest-only sources or a compliant
permitted-splits artifact exist.

Each task is designed for approximately 5–15 minutes. Git-only tasks explicitly
use shell guards instead of Python failing tests; all code tasks name an exact
failing test path.

## A. New integration worktree

### A1. Verify isolation and create the new worktree (Git-only, 10 min)

- Purpose: prove the requested branch/path are absent and create the only future
  implementation worktree from the locked E7 ref.
- Repository paths: no repository file changes; target
  `${INTEGRATION_WORKTREE}`.
- Test file: N/A; Git-only guard replaces a Python test.
- Failing check first: `test ! -e "${INTEGRATION_WORKTREE}" && ! git -C "${REPO_ROOT}" show-ref --verify --quiet refs/heads/stage-b/e7-occlusion-integration`; exit 0 proves both targets are absent, and any non-zero result blocks creation.
- Expected failure reason: existing path or branch must block creation.
- Smallest change: after both guards pass, run `git worktree add -b stage-b/e7-occlusion-integration "${INTEGRATION_WORKTREE}" 4cb1e0ffe4b55efc090a45cfed560b28f50b9509`.
- Focused pass: `test "$(git -C "${INTEGRATION_WORKTREE}" rev-parse HEAD)" = 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 && test "$(git -C "${INTEGRATION_WORKTREE}" branch --show-current)" = stage-b/e7-occlusion-integration && test -z "$(git -C "${INTEGRATION_WORKTREE}" status --porcelain=v2 --untracked-files=all)"`.
- Regression: `git -C "${REPO_ROOT}" worktree list --porcelain && git -C "${REPO_ROOT}" status --porcelain=v2 --untracked-files=all && git -C "${TEST_EXIT_WORKTREE}" status --porcelain=v2 --untracked-files=all`.
- Evidence: branch/path absence before; created path; HEAD equals locked ref; all existing status transcripts unchanged.
- Stop condition: stop immediately if branch/path exists, HEAD differs, or any existing worktree changes.
- Prohibited scope: no checkout, reset, clean, restore, stash, commit, or dirty-worktree operation.

### A2. Copy approved documents into the new worktree (Git-only, 5 min)

- Purpose: place the approved design and this plan in the future worktree only.
- Repository paths: create `docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md` and `docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md` under the new worktree.
- Test file: N/A; file-copy integrity guard replaces a Python test.
- Failing check first: `test -f /tmp/occlusion-fer-stage5-approved-design.md && test -f /tmp/occlusion-fer-stage5-implementation-plan.md && test ! -e "${INTEGRATION_WORKTREE}/docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md" && test ! -e "${INTEGRATION_WORKTREE}/docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md"`; a non-zero result blocks copying.
- Expected failure reason: source document not available or destination already exists.
- Smallest change: `mkdir -p "${INTEGRATION_WORKTREE}/docs/superpowers/specs" "${INTEGRATION_WORKTREE}/docs/superpowers/plans" && cp /tmp/occlusion-fer-stage5-approved-design.md "${INTEGRATION_WORKTREE}/docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md" && cp /tmp/occlusion-fer-stage5-implementation-plan.md "${INTEGRATION_WORKTREE}/docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md"`, then compare byte counts and SHA-256; do not edit content.
- Focused pass: `cmp -s /tmp/occlusion-fer-stage5-approved-design.md "${INTEGRATION_WORKTREE}/docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md" && cmp -s /tmp/occlusion-fer-stage5-implementation-plan.md "${INTEGRATION_WORKTREE}/docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md" && shasum -a 256 /tmp/occlusion-fer-stage5-approved-design.md "${INTEGRATION_WORKTREE}/docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md" /tmp/occlusion-fer-stage5-implementation-plan.md "${INTEGRATION_WORKTREE}/docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md" && git -C "${INTEGRATION_WORKTREE}" diff --check`.
- Regression: `git -C "${REPO_ROOT}" status --porcelain=v2 --untracked-files=all && git -C "${TEST_EXIT_WORKTREE}" status --porcelain=v2 --untracked-files=all` and compare both transcripts with the Stage 5 start transcript.
- Evidence: source/destination hashes and destination paths.
- Stop condition: stop on any hash mismatch or destination outside the new worktree.
- Prohibited scope: no repository docs in main or existing worktrees; no implementation edits.

## B. Controlled Stage A integration

### B1. Cherry-pick Stage A core only (Git-only, 15 min)

- Purpose: bring the deterministic Stage A core onto E7 with provenance.
- Repository paths: the new worktree only; expected touched paths originate from commit `6efe243...`.
- Test file: N/A; cherry-pick status/conflict guard replaces a Python test.
- Failing check first: `git -C ${INTEGRATION_WORKTREE} merge-base --is-ancestor 6efe243eca5ab4b39251fc1a7a11c4c82ebcff28 HEAD`; this must exit non-zero because the core commit is not yet in the new worktree history.
- Expected failure reason: commit already present or ref not reachable.
- Smallest change: run `git -C ${INTEGRATION_WORKTREE} cherry-pick 6efe243eca5ab4b39251fc1a7a11c4c82ebcff28` only.
- Focused pass: `git -C ${INTEGRATION_WORKTREE} status --short` and `git -C ${INTEGRATION_WORKTREE} show --stat --oneline HEAD`.
- Regression: `git -C ${INTEGRATION_WORKTREE} diff --exit-code 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 HEAD -- configs/experiments/fer2013_resnet18_e7_high_resolution_longer.yaml src/occlusion_fer/augmentations.py src/occlusion_fer/losses.py src/occlusion_fer/schedulers.py`.
- Evidence: cherry-pick exit code, conflict list if any, HEAD, and changed-file list.
- Stop condition: stop on conflict; do not run broad conflict resolution or continue automatically.
- Prohibited scope: no second cherry-pick, no BASE/paper commits, no dirty test-exit change.

### B2. Cherry-pick the cross-platform Stage A test regression (Git-only, 10 min)

- Purpose: preserve only the approved Stage A cross-platform test correction.
- Repository paths: new worktree `tests/test_torch_data.py` only for this commit.
- Test file: N/A; Git-only conflict guard.
- Failing check first: `git -C ${INTEGRATION_WORKTREE} merge-base --is-ancestor 78bb35d713c4e76f24bc378965fd6f2549b47b5b HEAD`; this must exit non-zero because the commit is not already an ancestor.
- Expected failure reason: the commit must not already be present.
- Smallest change: run `git -C ${INTEGRATION_WORKTREE} cherry-pick 78bb35d713c4e76f24bc378965fd6f2549b47b5b` only after B1 is clean.
- Focused pass: `git -C ${INTEGRATION_WORKTREE} show --stat --oneline HEAD -- tests/test_torch_data.py` and `git -C ${INTEGRATION_WORKTREE} diff HEAD^ HEAD -- tests/test_torch_data.py`.
- Regression: `test "$(git -C ${INTEGRATION_WORKTREE} diff --name-only 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 HEAD | sort)" = "configs/fer2013_stage_a.yaml
src/occlusion_fer/config.py
src/occlusion_fer/mask_hash.py
src/occlusion_fer/mask_manifest.py
src/occlusion_fer/occlusion.py
src/occlusion_fer/stage_a_artifacts.py
src/occlusion_fer/training_mean.py
tests/test_config.py
tests/test_evaluation.py
tests/test_mask_hash.py
tests/test_mask_manifest.py
tests/test_occlusion_batch.py
tests/test_occlusion_geometry.py
tests/test_stage_a_artifacts_cli.py
tests/test_torch_data.py
tests/test_train.py
tests/test_training_mean.py"`.
- Evidence: commit IDs, conflict status, exact diff, HEAD.
- Stop condition: stop if any unrelated file is changed or a conflict touches unapproved history.
- Prohibited scope: never cherry-pick `fix/macos-dataloader-test-exit`, BASE, paper, or WIP commits.

### B3. Audit the post-pick tree (Git-only, 10 min)

- Purpose: ensure controlled migration did not silently add/delete E7 or BASE/paper files.
- Repository paths: new worktree tree only.
- Test file: N/A; tree-diff guard replaces a Python test.
- Failing check first: `test -z "$(git -C ${INTEGRATION_WORKTREE} diff --name-only 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 HEAD | rg -v '^(configs/fer2013_stage_a.yaml|src/occlusion_fer/(config.py|mask_hash.py|mask_manifest.py|occlusion.py|stage_a_artifacts.py|training_mean.py)|tests/test_(config|evaluation|mask_hash|mask_manifest|occlusion_batch|occlusion_geometry|stage_a_artifacts_cli|torch_data|train|training_mean)\.py)$')"`; a non-empty output is an unapproved path and must stop the task.
- Expected failure reason: cherry-pick may expose conflicts or unintended tree drift.
- Smallest change: resolve only documented E7/Stage A overlap, preserving E7 configs and historical Stage A files.
- Focused pass: `git -C ${INTEGRATION_WORKTREE} diff --check 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 HEAD && test -z "$(git -C ${INTEGRATION_WORKTREE} diff --name-only 4cb1e0ffe4b55efc090a45cfed560b28f50b9509 HEAD | rg -v '^(configs/fer2013_stage_a.yaml|src/occlusion_fer/(config.py|mask_hash.py|mask_manifest.py|occlusion.py|stage_a_artifacts.py|training_mean.py)|tests/test_(config|evaluation|mask_hash|mask_manifest|occlusion_batch|occlusion_geometry|stage_a_artifacts_cli|torch_data|train|training_mean)\.py)$')"`.
- Regression: `test -f ${INTEGRATION_WORKTREE}/configs/experiments/fer2013_resnet18_e7_high_resolution_longer.yaml && test -f ${INTEGRATION_WORKTREE}/src/occlusion_fer/augmentations.py && test -f ${INTEGRATION_WORKTREE}/src/occlusion_fer/losses.py && test -f ${INTEGRATION_WORKTREE}/src/occlusion_fer/schedulers.py && rg -n 'from occlusion_fer\.(augmentations|losses|schedulers)' ${INTEGRATION_WORKTREE}/src/occlusion_fer`.
- Evidence: allow-list comparison, diff summary, dirty status.
- Stop condition: stop on an unallowlisted addition/deletion.
- Prohibited scope: no broad refactor or silent historical-file rewrite.

## C. Config compatibility

### C1. Preserve clean E7 parsing (10 min)

- Purpose: lock clean E7 backward compatibility before adding occlusion fields.
- Repository paths: `src/occlusion_fer/config.py`; test `tests/test_config.py`.
- Failing test first: add `test_locked_e7_config_resolves_without_occlusion_block` to `tests/test_config.py`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k locked_e7_config_resolves_without_occlusion_block`.
- Expected failure reason: Stage A cherry-pick may introduce a competing loader or reject the candidate E7 schema.
- Smallest production change: compose loaders so the exact E7 hierarchy remains valid and `occlusion` is optional.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k locked_e7_config_resolves_without_occlusion_block` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py`.
- Evidence: resolved dataclass fields and serialized clean config equal the locked E7 keys.
- Stop condition: stop if clean config needs mean/manifest or changes augmentation/loss location.
- Prohibited scope: no relocation of `dataset.augmentation` or `training.loss`.

### C2. Add additive run role and permitted source fields (10 min)

- Purpose: add provenance role and strict split-source declarations without altering old keys.
- Repository paths: `src/occlusion_fer/config.py`; `tests/test_config.py`.
- Failing test first: add `test_parses_run_role_and_permitted_split_sources`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k parses_run_role_and_permitted_split_sources`.
- Expected failure reason: current parser rejects `project.run_role` and `dataset.permitted_splits` as unknown.
- Smallest production change: add optional `ProjectConfig.run_role` and optional `DatasetConfig.permitted_splits` with strict validation.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k parses_run_role_and_permitted_split_sources` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_data.py`.
- Evidence: old E7 fixture and new split-source fixture both resolve; invalid extra fields fail.
- Stop condition: stop if `dataset.path` semantics change for clean configs.
- Prohibited scope: no combined CSV opening or data access.

### C3. Add one non-overlapping occlusion block and mode validation (15 min)

- Purpose: represent protocol, sampling, artifact paths, and evaluation conditions without duplicate training meanings.
- Repository paths: `src/occlusion_fer/config.py`; `tests/test_config.py`.
- Failing test first: add `test_occlusion_config_requires_complete_v2_identity` and `test_clean_config_does_not_require_occlusion_artifacts`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k 'occlusion_config_requires_complete_v2_identity or clean_config_does_not_require_occlusion_artifacts'`.
- Expected failure reason: no candidate schema currently accepts top-level `occlusion` or `training.mode: mixed`.
- Smallest production change: add optional top-level `occlusion.protocol`, `occlusion.sampling`, `occlusion.artifacts`, `occlusion.evaluation`; retain `training.mode` as the sole mode selector.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py -k 'occlusion_config_requires_complete_v2_identity or clean_config_does_not_require_occlusion_artifacts'` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_preflight.py`.
- Evidence: clean config omits block; mixed/masked configs require complete v2 identity and permitted sources; partial combinations fail closed.
- Stop condition: stop if duplicate `training.occlusion` semantics appear.
- Prohibited scope: no new protocol names or ratio/type values beyond approved design.

## D. Permitted-split source contract

### D1. Validate split-only source records (15 min)

- Purpose: define and reject invalid Training-only/PublicTest-only sources before any model path uses them.
- Repository paths: `src/occlusion_fer/permitted_splits.py` (new); `tests/test_permitted_splits.py` (new).
- Failing test first: add `test_rejects_combined_source_and_extra_split`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py -k rejects_combined_source_and_extra_split`.
- Expected failure reason: no split-specific source loader exists; a combined-source fixture must be refused before opening.
- Smallest production change: implement source metadata/record validation for split name, counts, IDs, 48x48 uint8 range, labels, and no extra split.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py -k rejects_combined_source_and_extra_split` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_data.py tests/test_config.py`.
- Evidence: source refusal occurs before file reads; valid synthetic separate sources load.
- Stop condition: stop if implementation scans a combined source or imports real data.
- Prohibited scope: no auto-splitting or Usage-only fallback.

### D2. Canonical split identities (15 min)

- Purpose: produce separate `training_dataset_sha256` and `publictest_dataset_sha256` over permitted records only.
- Repository paths: `src/occlusion_fer/permitted_splits.py`; `tests/test_permitted_splits.py`.
- Failing test first: add `test_canonical_identity_is_ordered_and_split_specific`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py -k canonical_identity_is_ordered_and_split_specific`.
- Expected failure reason: no canonical record serializer or separate digest fields exist.
- Smallest production change: serialize sorted compact JSON records with final LF and hash each permitted split independently.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py -k canonical_identity_is_ordered_and_split_specific` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py`.
- Evidence: exact bytes, digest, preserved IDs, and count checks recorded.
- Stop condition: stop if path bytes, Usage fields, or non-permitted rows enter either digest.
- Prohibited scope: no opaque whole-file SHA or combined source scan.

### D3. Enforce source contract at Stage B entry points (10 min)

- Purpose: prevent mean/manifest/mixed/evaluator commands from accepting legacy `dataset.path` for occlusion-enabled runs.
- Repository paths: `src/occlusion_fer/training_mean.py`, `src/occlusion_fer/mask_manifest.py`, `src/occlusion_fer/train.py`, `src/occlusion_fer/occlusion_evaluate.py`; tests `tests/test_permitted_splits.py`, `tests/test_preflight.py`.
- Failing test first: add `test_occlusion_entrypoints_fail_on_combined_dataset_path`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py tests/test_preflight.py -k occlusion_entrypoints_fail_on_combined_dataset_path`.
- Expected failure reason: historical code accepts a single path and whole-file hashes it.
- Smallest production change: require validated split-source objects for Stage B operations and fail before open on combined path.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_permitted_splits.py tests/test_preflight.py -k occlusion_entrypoints_fail_on_combined_dataset_path` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_preflight.py tests/test_train.py -k 'clean or synthetic'`.
- Evidence: failure message, no source-open event, no output artifact.
- Stop condition: stop if any Stage B path falls back to historical `sha256_file`.
- Prohibited scope: no real server data access.

## E. Hash v2

### E1. Canonical serializer and non-self-referential digest helpers (10 min)

- Purpose: preserve exact JSON bytes and make digest/envelope separation testable.
- Repository paths: `src/occlusion_fer/hash_v2.py` (new or approved hash module); tests `tests/test_mask_hash.py`.
- Failing test first: add `test_digest_excludes_digest_field_from_canonical_bytes`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k digest_excludes_digest_field_from_canonical_bytes`.
- Expected failure reason: v2 serializer/helper does not exist.
- Smallest production change: implement compact UTF-8 JSON-array serialization and payload digest helper; keep digest outside payload.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k digest_excludes_digest_field_from_canonical_bytes` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py`.
- Evidence: exact bytes, digest, and envelope fields.
- Stop condition: stop if a digest is inserted before hashing.
- Prohibited scope: no golden vector generation from production code.

### E2. Implement exact Training and validation payload builders (15 min)

- Purpose: encode all approved payload fields, tokens, namespaces, and types.
- Repository paths: `src/occlusion_fer/mask_hash.py`; tests `tests/test_mask_hash.py`.
- Failing test first: add `test_v2_payloads_have_exact_order_types_and_tokens`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k v2_payloads_have_exact_order_types_and_tokens`.
- Expected failure reason: v1 builders lack dimensions, v2 version, fixed ratio strings, and explicit top/left namespaces.
- Smallest production change: add v2 builders for apply/type/ratio/training top/left/validation top/left without changing v1 functions.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k v2_payloads_have_exact_order_types_and_tokens` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py`.
- Evidence: literal canonical arrays and field-type assertions.
- Stop condition: stop on float ratio tokens or ambiguous condition strings.
- Prohibited scope: no modification of v1 expected vectors.

### E3. Preserve exact mapping rules and independence (10 min)

- Purpose: prove apply/type/ratio/top/left mappings and coordinate bounds.
- Repository paths: `src/occlusion_fer/mask_hash.py`, `src/occlusion_fer/occlusion.py`; tests `tests/test_mask_hash.py`, `tests/test_occlusion_geometry.py`.
- Failing test first: add `test_v2_mapping_preserves_apply_threshold_modulo_and_bounds`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py tests/test_occlusion_geometry.py -k v2_mapping_preserves_apply_threshold_modulo_and_bounds`.
- Expected failure reason: v2 constants/mappings are not wired.
- Smallest production change: expose v2 mapping helpers and feed explicit condition/ratio tokens to coordinate payloads.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py tests/test_occlusion_geometry.py -k v2_mapping_preserves_apply_threshold_modulo_and_bounds` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py tests/test_occlusion_geometry.py`.
- Evidence: threshold `2**63`, type/ratio index order, top/left bound calculations.
- Stop condition: stop if label, batch order, or worker RNG influences output.
- Prohibited scope: no new types/ratios.

### E4 Independent reference vectors (10 min)

- Purpose: make golden vectors independent of production implementation.
- Repository paths: `tests/reference_hash_v2.py` (test-only reference); `tests/test_mask_hash.py`.
- Failing test first: add `test_v2_golden_vectors_match_independent_reference`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k v2_golden_vectors_match_independent_reference`.
- Expected failure reason: no reference serializer/vectors exist.
- Smallest production change: none; add a minimal reference serializer that does not import production hash/geometry.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py -k v2_golden_vectors_match_independent_reference` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py tests/test_occlusion_geometry.py`.
- Evidence: payload bytes, digest, u64, and coordinates for fixed synthetic identities.
- Stop condition: stop if the reference imports production payload builders.
- Prohibited scope: no real data or CUDA RNG.

## F. 224 geometry

### F1. Parameterize geometry and remove implicit 112 (15 min)

- Purpose: support only the approved 224 consumer geometry while preserving v1 historical constants.
- Repository paths: `src/occlusion_fer/occlusion.py`; tests `tests/test_occlusion_geometry.py`.
- Failing test first: add `test_v2_geometry_matches_all_224_dimensions`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py -k v2_geometry_matches_all_224_dimensions`.
- Expected failure reason: current `IMAGE_SIZE=112` and v1 contracts reject 224.
- Smallest production change: make v2 geometry use validated 224 dimensions and preserve a separate v1 historical path/identity.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py -k v2_geometry_matches_all_224_dimensions` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py`.
- Evidence: 45x224/10080, 67x224/15008, 90x224/20160, 100x100/10000, 123x123/15129, 142x142/20164.
- Stop condition: stop if any implicit 112 constant remains in v2 path.
- Prohibited scope: no ratio/type changes.

### F2. Validate half-open bounds and metadata dimensions (10 min)

- Purpose: ensure geometry metadata and ratios are independently correct.
- Repository paths: `src/occlusion_fer/occlusion.py`; tests `tests/test_occlusion_geometry.py`.
- Failing test first: add `test_v2_metadata_records_dimensions_pixels_and_actual_ratio`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py -k v2_metadata_records_dimensions_pixels_and_actual_ratio`.
- Expected failure reason: metadata currently reports v1 size and ratios.
- Smallest production change: emit v2 consumer dimensions, pixel counts, actual ratio, half-open convention, and condition tokens.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py -k v2_metadata_records_dimensions_pixels_and_actual_ratio` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_geometry.py tests/test_occlusion_batch.py`.
- Evidence: `masked_pixel_count/50176`, top/bottom alignment, random bounds.
- Stop condition: stop if metadata can mix v1 and v2 identity.
- Prohibited scope: no manifest integration yet.

## G. Tensor mask application

### G1. Single-image and batch 224 contracts (10 min)

- Purpose: make v2 mask application accept `[3,224,224]` and `[B,3,224,224]`.
- Repository paths: `src/occlusion_fer/occlusion.py`; tests `tests/test_occlusion_batch.py`.
- Failing test first: add `test_v2_accepts_single_and_batch_224_contract`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py -k v2_accepts_single_and_batch_224_contract`.
- Expected failure reason: current validators require 112.
- Smallest production change: add v2 contract validation while keeping v1 rejection/identity explicit.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py -k v2_accepts_single_and_batch_224_contract` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py`.
- Evidence: shape, dtype, device, finite values, sample ID contract.
- Stop condition: stop if clean E7 path requires mask invocation.
- Prohibited scope: no augmentation changes.

### G2. Clone/fill/immutability and clean bypass (10 min)

- Purpose: prove normalized three-channel fill, source immutability, and clean bypass.
- Repository paths: `src/occlusion_fer/occlusion.py`, `src/occlusion_fer/train.py`; tests `tests/test_occlusion_batch.py`, `tests/test_train.py`.
- Failing test first: add `test_masked_batch_clones_and_clean_batch_bypasses_mask`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py tests/test_train.py -k masked_batch_clones_and_clean_batch_bypasses_mask`.
- Expected failure reason: no mixed boundary exists and current train loop is clean-only.
- Smallest production change: add a pure post-device batch mask seam; clean samples return unchanged tensors.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py tests/test_train.py -k masked_batch_clones_and_clean_batch_bypasses_mask` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py tests/test_train.py`.
- Evidence: source storage pointer unchanged, fill in all channels, no autocast-scope diff.
- Stop condition: stop if mask executes inside autocast or mutates source.
- Prohibited scope: no checkpoint selection changes.

## H. Training mean v2 envelope

### H1. Split-specific raw mean calculation and envelope (15 min)

- Purpose: preserve raw 48x48 Training-only math while adding v2 provenance and non-self-referential digest.
- Repository paths: `src/occlusion_fer/training_mean.py`, `src/occlusion_fer/permitted_splits.py`; tests `tests/test_training_mean.py`, `tests/test_permitted_splits.py`.
- Failing test first: add `test_mean_envelope_digest_excludes_digest_and_private_split`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_permitted_splits.py -k mean_envelope_digest_excludes_digest_and_private_split`.
- Expected failure reason: historical mean hashes whole-file path and has no v2 envelope.
- Smallest production change: calculate only permitted Training records with uint64 accumulation and emit v2 payload plus external digest.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_permitted_splits.py -k mean_envelope_digest_excludes_digest_and_private_split` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_permitted_splits.py`.
- Evidence: exact canonical payload bytes/digest, 28709/66145536 counts, no augmentation contribution.
- Stop condition: stop if PublicTest/PrivateTest records enter sum or identity.
- Prohibited scope: no combined CSV fallback.

### H2. Normalized fill derivation and compatibility rejection (10 min)

- Purpose: derive normalized three-channel fill and reject wrong source/consumer identity.
- Repository paths: `src/occlusion_fer/occlusion.py`, `src/occlusion_fer/training_mean.py`; tests `tests/test_training_mean.py`, `tests/test_occlusion_batch.py`.
- Failing test first: add `test_v2_fill_requires_matching_mean_provenance`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_occlusion_batch.py -k v2_fill_requires_matching_mean_provenance`.
- Expected failure reason: v1 fill helper lacks source/consumer identity checks.
- Smallest production change: validate mean envelope/version/split SHA/source size/consumer size before normalized fill.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_occlusion_batch.py -k v2_fill_requires_matching_mean_provenance` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_training_mean.py tests/test_occlusion_geometry.py tests/test_occlusion_batch.py`.
- Evidence: normalized fill equivalence and fail-closed mismatch messages.
- Stop condition: stop if fill is computed from augmented pixels.
- Prohibited scope: no mean algorithm change beyond provenance/schema.

## I. Manifest v2

### I1. Canonical PublicTest manifest rows and sidecar (15 min)

- Purpose: produce v2 rows from permitted PublicTest identities and separate manifest digest.
- Repository paths: `src/occlusion_fer/mask_manifest.py`; tests `tests/test_mask_manifest.py`.
- Failing test first: add `test_v2_manifest_bytes_exclude_manifest_digest`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py -k v2_manifest_bytes_exclude_manifest_digest`.
- Expected failure reason: current manifest is v1/112 and has no sidecar identity.
- Smallest production change: emit canonical CSV rows with v2 fields and external metadata sidecar.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py -k v2_manifest_bytes_exclude_manifest_digest` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py`.
- Evidence: exact columns/bytes, `3589*9=32301`, condition order, manifest SHA.
- Stop condition: stop if digest is embedded in hashed CSV or rows include PrivateTest.
- Prohibited scope: no real PublicTest source.

### I2. Fail-closed manifest loader (10 min)

- Purpose: validate dimensions, split SHA, mean SHA, protocol version, rows, and canonical bytes before evaluator use.
- Repository paths: `src/occlusion_fer/mask_manifest.py`; tests `tests/test_mask_manifest.py`.
- Failing test first: add `test_v2_manifest_rejects_wrong_identity_and_duplicate_rows`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py -k v2_manifest_rejects_wrong_identity_and_duplicate_rows`.
- Expected failure reason: current loader validates v1 fields only.
- Smallest production change: add explicit v2 loader checks and no-regeneration behavior.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py -k v2_manifest_rejects_wrong_identity_and_duplicate_rows` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py tests/test_permitted_splits.py`.
- Evidence: each mismatch fails with an identified field and leaves existing files unchanged.
- Stop condition: stop if loader silently recomputes geometry.
- Prohibited scope: no evaluator implementation yet.

### I3. Create-or-verify conflict safety (10 min)

- Purpose: preserve atomic create-or-verify semantics with sidecar identity.
- Repository paths: `src/occlusion_fer/mask_manifest.py`, `src/occlusion_fer/stage_a_artifacts.py`; tests `tests/test_mask_manifest.py`, `tests/test_stage_a_artifacts_cli.py`.
- Failing test first: add `test_v2_manifest_sidecar_conflict_does_not_overwrite`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py tests/test_stage_a_artifacts_cli.py -k v2_manifest_sidecar_conflict_does_not_overwrite`.
- Expected failure reason: v2 sidecar is absent and conflict rules are not bound to it.
- Smallest production change: publish CSV and sidecar atomically/consistently; reject mismatched existing bytes.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py tests/test_stage_a_artifacts_cli.py -k v2_manifest_sidecar_conflict_does_not_overwrite` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_manifest.py tests/test_stage_a_artifacts_cli.py`.
- Evidence: statuses `created`/`consistent`, conflict bytes preserved, no temporary files.
- Stop condition: stop on overwrite or partial artifact.
- Prohibited scope: no output in formal results directories.

## J. Mixed training integration

### J1. Deterministic per-sample sampling helper (10 min)

- Purpose: select apply/type/ratio independently without Dataset epoch state.
- Repository paths: `src/occlusion_fer/occlusion.py`, `src/occlusion_fer/train.py`; tests `tests/test_train.py`, `tests/test_occlusion_batch.py`.
- Failing test first: add `test_mixed_selection_is_sample_epoch_and_seed_deterministic`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_occlusion_batch.py -k mixed_selection_is_sample_epoch_and_seed_deterministic`.
- Expected failure reason: current train loop has no mode/context selector.
- Smallest production change: add pure selection helper taking sample ID, seed, one-based epoch and returning clean/condition.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_occlusion_batch.py -k mixed_selection_is_sample_epoch_and_seed_deterministic` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_occlusion_batch.py`.
- Evidence: order/batch-size/worker-independent selections and exact apply/type/ratio mapping.
- Stop condition: stop if helper reads labels or worker RNG.
- Prohibited scope: no dataset mutable epoch.

### J2. Insert mask after device transfer before existing autocast (15 min)

- Purpose: wire mixed masking while preserving E7 AMP/model/loss scope.
- Repository paths: `src/occlusion_fer/train.py`; tests `tests/test_train.py`.
- Failing test first: add `test_mixed_train_masks_before_existing_autocast_and_keeps_clean_path`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py -k mixed_train_masks_before_existing_autocast_and_keeps_clean_path`.
- Expected failure reason: current `train_one_epoch` always forwards unmasked images.
- Smallest production change: pass explicit context, device-transfer clean batch, mask selected samples, then enter unchanged `_autocast_context` block.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py -k mixed_train_masks_before_existing_autocast_and_keeps_clean_path` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_torch_data.py`.
- Evidence: source line/order inspection, clean bypass, autocast scope diff, optimizer/loss unchanged.
- Stop condition: stop if masking is inside autocast or clean mode invokes it.
- Prohibited scope: no AMP redesign or optimizer changes.

### J3. Record realized counts and clean validation (10 min)

- Purpose: make mixed recipe provenance and validation selection explicit.
- Repository paths: `src/occlusion_fer/train.py`, `src/occlusion_fer/artifacts.py`; tests `tests/test_train.py`, `tests/test_artifacts.py`.
- Failing test first: add `test_mixed_history_records_condition_counts_and_clean_selection`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_artifacts.py -k mixed_history_records_condition_counts_and_clean_selection`.
- Expected failure reason: history/metadata currently know only clean samples.
- Smallest production change: record per-epoch counts and keep validation `condition="clean"` with strict `>` best rule.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_artifacts.py -k mixed_history_records_condition_counts_and_clean_selection` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_artifacts.py`.
- Evidence: history fields, best/last payloads, no masked selection metric.
- Stop condition: stop if masked metric can update best checkpoint.
- Prohibited scope: no formal training.

## K. PublicTest masked evaluator

### K1. Narrow evaluator entry and source/manifest guard (15 min)

- Purpose: create PublicTest-only evaluator without coupling to `final_evaluate.py`.
- Repository paths: `src/occlusion_fer/occlusion_evaluate.py` (new); tests `tests/test_occlusion_evaluate.py` (new).
- Failing test first: add `test_occlusion_evaluator_rejects_combined_source_and_private_route`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py -k occlusion_evaluator_rejects_combined_source_and_private_route`.
- Expected failure reason: entry point does not exist and current final evaluator is PrivateTest-specific.
- Smallest production change: add argument/config guard requiring permitted PublicTest source, v2 manifest, and no PrivateTest flag.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py -k occlusion_evaluator_rejects_combined_source_and_private_route` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_final_evaluate.py tests/test_preflight.py`.
- Evidence: no final-evaluate import/call; combined-source refusal; no output on guard failure.
- Stop condition: stop if evaluator can select/parse PrivateTest.
- Prohibited scope: no real checkpoint/data.

### K2. Ten conditions and shared metrics/artifacts (15 min)

- Purpose: evaluate clean plus nine fixed conditions through existing metrics/loss.
- Repository paths: `src/occlusion_fer/occlusion_evaluate.py`, `src/occlusion_fer/evaluation.py`, `src/occlusion_fer/artifacts.py`; tests `tests/test_occlusion_evaluate.py`, `tests/test_evaluation.py`, `tests/test_artifacts.py`.
- Failing test first: add `test_evaluator_writes_clean_and_nine_condition_artifacts`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py tests/test_evaluation.py tests/test_artifacts.py -k evaluator_writes_clean_and_nine_condition_artifacts`.
- Expected failure reason: no condition loop or manifest-backed batch path exists.
- Smallest production change: loop fixed order, validate shared manifest, mask after device transfer, call existing evaluator, write per-condition artifacts.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py tests/test_evaluation.py tests/test_artifacts.py -k evaluator_writes_clean_and_nine_condition_artifacts` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py tests/test_evaluation.py tests/test_artifacts.py`.
- Evidence: ten condition keys, same checkpoint/manifest identity, metrics/predictions/matrix files.
- Stop condition: stop if evaluator regenerates geometry or changes metric criterion.
- Prohibited scope: no PrivateTest or real PublicTest inference.

### K3. Paired drop and failure artifact (10 min)

- Purpose: record sample-paired clean-to-occluded drops and partial-failure evidence.
- Repository paths: `src/occlusion_fer/occlusion_evaluate.py`, `src/occlusion_fer/artifacts.py`; tests `tests/test_occlusion_evaluate.py`.
- Failing test first: add `test_evaluator_writes_paired_drop_and_failure_artifact`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py -k evaluator_writes_paired_drop_and_failure_artifact`.
- Expected failure reason: current artifacts have no paired-drop or evaluator failure contract.
- Smallest production change: join predictions by checkpoint/sample ID and write failure metadata without partial success claims.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py -k evaluator_writes_paired_drop_and_failure_artifact` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py tests/test_artifacts.py`.
- Evidence: paired rows, condition drops, failure JSON, no false completion marker.
- Stop condition: stop if partial conditions are reported as complete.
- Prohibited scope: no research metric generation.

## L. Checkpoint and provenance

### L1. Strict clean and mixed checkpoint compatibility (10 min)

- Purpose: accept current clean model state while requiring v2 metadata for mixed/masked routes.
- Repository paths: `src/occlusion_fer/train.py`, `src/occlusion_fer/final_evaluate.py`, `src/occlusion_fer/occlusion_evaluate.py`; tests `tests/test_train.py`, `tests/test_final_evaluate.py`, `tests/test_occlusion_evaluate.py`.
- Failing test first: add `test_clean_checkpoint_loads_without_occlusion_fields_and_mixed_requires_metadata`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_final_evaluate.py tests/test_occlusion_evaluate.py -k clean_checkpoint_loads_without_occlusion_fields_and_mixed_requires_metadata`.
- Expected failure reason: no route-specific compatibility checks exist.
- Smallest production change: strict model-state load for clean; explicit v2/mixed metadata validation for masked/mixed routes.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_final_evaluate.py tests/test_occlusion_evaluate.py -k clean_checkpoint_loads_without_occlusion_fields_and_mixed_requires_metadata` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_final_evaluate.py tests/test_occlusion_evaluate.py`.
- Evidence: accepted synthetic clean payload, rejected v1/112 and incomplete mixed payloads.
- Stop condition: stop if clean checkpoint loading requires future occlusion fields.
- Prohibited scope: no real checkpoint modification.

### L2. Provenance and split SHA fields (10 min)

- Purpose: record checkpoint/evaluation commits, dirty state, split identities, mean/manifest SHA, protocol, role, and seed class.
- Repository paths: `src/occlusion_fer/artifacts.py`, `src/occlusion_fer/train.py`, `src/occlusion_fer/occlusion_evaluate.py`; tests `tests/test_artifacts.py`, `tests/test_train.py`, `tests/test_occlusion_evaluate.py`.
- Failing test first: add `test_provenance_records_v2_identity_without_digest_self_reference`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_artifacts.py tests/test_train.py tests/test_occlusion_evaluate.py -k provenance_records_v2_identity_without_digest_self_reference`.
- Expected failure reason: current metadata has git/seed but not split-specific v2 identity.
- Smallest production change: extend metadata payload and resolved config references only.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_artifacts.py tests/test_train.py tests/test_occlusion_evaluate.py -k provenance_records_v2_identity_without_digest_self_reference` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_artifacts.py tests/test_train.py tests/test_occlusion_evaluate.py`.
- Evidence: canonical metadata, SHA fields, run role, formal vs screening identity.
- Stop condition: stop if metadata digest participates in its own hash.
- Prohibited scope: no output in formal result directories.

## M. Preflight and guards

### M1. Clean/mixed/masked preflight (15 min)

- Purpose: validate 224 shapes, protocol/artifact compatibility, split-source refusal, output safety, and PrivateTest isolation.
- Repository paths: `src/occlusion_fer/preflight.py`, `src/occlusion_fer/config.py`; tests `tests/test_preflight.py`.
- Failing test first: add `test_preflight_rejects_combined_source_and_wrong_v2_artifact`; run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_preflight.py -k preflight_rejects_combined_source_and_wrong_v2_artifact`.
- Expected failure reason: current preflight checks only clean train/validation and has no split-source/artifact guard.
- Smallest production change: add mode-specific checks while retaining clean E7 preflight output and sample limits.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_preflight.py -k preflight_rejects_combined_source_and_wrong_v2_artifact` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_preflight.py tests/test_config.py tests/test_data.py`.
- Evidence: 224 batch checks, clear blocker, no PrivateTest parse, safe output reservation.
- Stop condition: stop if clean preflight starts requiring mean/manifest or opens combined source.
- Prohibited scope: no real data or training.

## N. Documentation and ignore rules

### N1. Historical/current documentation and `.gitignore` (10 min)

- Purpose: document v2/224 as current design while retaining v1/112 historical records and ignoring artifacts.
- Repository paths: `README.md`, `docs/server_runbook.md`, required historical docs if present, `.gitignore`; tests `tests/test_artifacts.py` (no docs test).
- Failing test first: N/A for documentation-only task; run `git -C ${INTEGRATION_WORKTREE} diff --check` before editing as the structural guard.
- Expected failure reason: no test can fail before text changes; structural guard catches whitespace/path errors.
- Smallest production change: add minimal v2 prerequisite/identity text and ignore rules without changing historical claims.
- Focused pass: `git -C ${INTEGRATION_WORKTREE} diff --check && rg -n 'occlusion-v1|112|occlusion-v2-224|224|combined|Training-only|PublicTest-only' ${INTEGRATION_WORKTREE}/README.md ${INTEGRATION_WORKTREE}/docs ${INTEGRATION_WORKTREE}/.gitignore` with exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_artifacts.py tests/test_config.py`.
- Evidence: diff, ignore patterns, no synthetic result presented as research result.
- Stop condition: stop on silent rewrite of historical 112/v1 or formal result claims.
- Prohibited scope: no broad documentation rewrite.

## O. Verification sequence

### O1. Focused test groups (10 min)

- Purpose: run focused groups after all implementation tasks and preserve real exit codes.
- Repository paths: no production changes; logs outside repository.
- Test file: all named focused files from C–M.
- Failing test first: N/A; implementation tasks already created failing tests. Verify the gate with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_permitted_splits.py tests/test_mask_hash.py tests/test_occlusion_geometry.py tests/test_occlusion_batch.py tests/test_training_mean.py tests/test_mask_manifest.py tests/test_train.py tests/test_occlusion_evaluate.py tests/test_final_evaluate.py tests/test_artifacts.py tests/test_preflight.py -x` and require exit 0.
- Expected failure reason: any focused group may expose unresolved integration conflict.
- Smallest production change: none during verification; return to the owning task on failure.
- Focused pass: run these six literal commands separately, recording each exit: (1) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_data.py tests/test_permitted_splits.py tests/test_preflight.py`; (2) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_mask_hash.py tests/test_occlusion_geometry.py`; (3) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py tests/test_training_mean.py tests/test_mask_manifest.py`; (4) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py tests/test_evaluation.py tests/test_artifacts.py`; (5) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_evaluate.py tests/test_final_evaluate.py tests/test_stage_a_artifacts_cli.py`; (6) `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_torch_data.py tests/test_models.py tests/test_metrics.py`. Each must exit 0.
- Regression: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_paper_figures.py tests/test_paper_framework.py` after groups 1-6 pass.
- Evidence: command, start/end, wall time, stdout/stderr summary, exit, hang/timeout state.
- Stop condition: any non-zero or timeout pauses verification and starts systematic debugging.
- Prohibited scope: no rerun loop without root-cause analysis.

### O2. Static/package checks (10 min)

- Purpose: verify syntax, dependencies, whitespace, and tracked data/weights safety.
- Repository paths: new worktree only; logs outside repository.
- Test file: N/A; shell verification task.
- Failing test first: N/A; prior focused failures are the gate.
- Expected failure reason: compile/import/package or tracked-large-file violation.
- Smallest production change: none; fix only the owning task after failure.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m compileall -q src tests && python3 -m pip check && git diff --check`.
- Regression: `git status --short && git ls-files 'data/**' 'datasets/**' 'outputs/**' 'results/**' 'runs/**' 'checkpoints/**' 'logs/**' '*.pt' '*.pth' '*.ckpt'`.
- Evidence: each real exit code and scan output.
- Stop condition: any syntax, dependency, whitespace, or tracked-artifact failure.
- Prohibited scope: no dependency installation from network and no data cleanup.

### O3. Full suite with macOS hang handling (15 min)

- Purpose: run complete tests and distinguish exit 0 from post-test hang.
- Repository paths: new worktree; logs outside repository.
- Test file: complete `tests/` suite.
- Failing test first: N/A; focused groups are the preceding gate.
- Expected failure reason: regression or macOS multiprocessing `resource_tracker` hang.
- Smallest production change: none during run; preserve exact failure/hang evidence.
- Focused pass: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src perl -e 'alarm 900; exec @ARGV' python3 -m pytest -q -p no:cacheprovider`; exit 0 is PASS, signal/exit 142 is HANG/TIMEOUT, and any other non-zero exit is FAIL.
- Regression: `! git -C ${INTEGRATION_WORKTREE} diff -- tests/test_occlusion_batch.py | rg -q '^\+.*persistent_workers=False'` must exit 0; do not copy the known dirty fix into the integration worktree.
- Evidence: process exit code, timeout signal, passed count only as supplemental output, wall time.
- Stop condition: HANG/TIMEOUT is not success and blocks implementation verification.
- Prohibited scope: no dirty test-exit branch changes.

### O4. HIVE/Linux synthetic verification (15 min)

- Purpose: validate worker lifecycle and synthetic smoke without real data or training.
- Repository paths: new worktree; HIVE logs external.
- Test file: focused DataLoader/determinism/preflight/smoke tests.
- Failing test first: N/A; local suite must pass before HIVE. On the HIVE checkout root, run `git rev-parse HEAD` and require `4cb1e0ffe4b55efc090a45cfed560b28f50b9509` or the documented post-pick descendant before any synthetic command.
- Expected failure reason: environment/dependency/worker lifecycle mismatch.
- Smallest production change: none on HIVE; return with exact blocker.
- Focused pass: execute these literal HIVE commands from the clean integration checkout root: `python3 --version`; `python3 -m pip check`; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_torch_data.py tests/test_occlusion_batch.py tests/test_preflight.py`; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_occlusion_batch.py -k num_workers_zero_and_four_preserve_each_sample_output`; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_train.py -k 'artificial_cpu_training_validation_checkpoint_flow or artificial_fer_resnet_full_training_chain'`; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_preflight.py -k module_cli_runs_real_resnet_forward_without_artifacts`; `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src perl -e 'alarm 900; exec @ARGV' python3 -m pytest -q -p no:cacheprovider`; classify exit 0, non-zero, and signal/timeout separately.
- Regression: before each HIVE command run `git rev-parse --show-toplevel && git status --porcelain=v2 --untracked-files=all && git rev-parse HEAD`; stop if the root, status, or HEAD differs from the recorded checkout.
- Evidence: command timing, stdout/stderr, real exit, timeout/hang, Git identity.
- Stop condition: no real FER2013 training, PublicTest metrics, PrivateTest, or final_evaluate.
- Prohibited scope: no combined CSV transfer or source splitting on HIVE.

## P. Final implementation handoff

### P1. Generate evidence report without commit (10 min)

- Purpose: produce the final implementation/verification handoff only after all approved checks pass.
- Repository paths: future `/tmp/occlusion-fer-e7-occlusion-integration-report.md`; no repository report required before approval.
- Test file: N/A; report integrity is the guard.
- Failing check first: `test ! -e /tmp/occlusion-fer-e7-occlusion-integration-report.md`; existing report must be handled without overwrite.
- Expected failure reason: report path collision or incomplete evidence set.
- Smallest production change: write report outside repository with exact changed files, decisions, commands/exits, limitations, Git status, and SHA.
- Focused pass: `shasum -a 256 /tmp/occlusion-fer-e7-occlusion-integration-report.md && git -C ${INTEGRATION_WORKTREE} rev-parse HEAD && git -C ${INTEGRATION_WORKTREE} branch --show-current && git -C ${INTEGRATION_WORKTREE} status --porcelain=v2 --untracked-files=all`.
- Regression: `git -C ${INTEGRATION_WORKTREE} diff --check && test -z "$(git -C ${INTEGRATION_WORKTREE} ls-files 'data/**' 'datasets/**' 'outputs/**' 'results/**' 'runs/**' 'checkpoints/**' 'logs/**' '*.pt' '*.pth' '*.ckpt')" && git -C ${REPO_ROOT} status --porcelain=v2 --untracked-files=all && git -C ${TEST_EXIT_WORKTREE} status --porcelain=v2 --untracked-files=all`; compare the last two transcripts with the Stage 5 start transcript.
- Evidence: report path/SHA, all command exits, unrun experiments, HIVE/macOS status.
- Stop condition: stop with `IMPLEMENTATION PARTIALLY VERIFIED` or `BLOCKED` if any gate is incomplete.
- Prohibited scope: no commit, push, PR, merge, or final PrivateTest evaluation.

## Acceptance criteria for Stage 6 completion

Stage 6 may not claim verified completion unless it proves:

- clean E7 config, tensor pipeline, optimizer/loss/epochs/batch/augmentation,
  strict clean PublicTest checkpoint selection, and strict model loading remain compatible;
- no v1/112 artifact is accepted by v2 consumers;
- 224 geometry, source immutability, normalized fill, and all nine conditions pass;
- v2 determinism is independent of sample order, batch size, and worker count;
- mixed masking uses explicit one-based epoch and clean validation selection;
- masked performance never selects a checkpoint;
- combined CSV is rejected before opening by every Stage B command;
- PrivateTest is not read, parsed, hashed, inferred, or scored;
- every verification command has a real exit code and a hang/timeout classification;
- a macOS post-test hang is reported as HANG/TIMEOUT, never as success;
- no data, checkpoint, output, secret, or unrelated file is tracked;
- no commit is created without later explicit user approval.

## Copy target after implementation approval

After Stage 6 approval and new worktree creation, copy this plan to:

```text
docs/superpowers/plans/2026-08-06-e7-occlusion-integration.md
```

The approved design copy goes to:

```text
docs/superpowers/specs/2026-08-06-e7-occlusion-integration-design.md
```

Both copies belong only in the new integration worktree, never main or an existing
worktree. Do not commit them.
