# Project Cleanup Report

Date: 2026-08-14

This was a repository and artifact organization pass. No model was trained, no
formal evaluation was regenerated, no checkpoint/manifest/metadata bytes were
edited, and no remote GitHub branch or release was changed.

## Audit Performed

The initial pass was read-only and covered:

```text
git status --short --branch
git status --ignored --short
git log --oneline --decorate -n 30
git ls-files
git branch -a
git worktree list --porcelain
find / rg inventories of source, tests, configs, outputs and paper files
SHA-256 comparison of suspicious duplicate files and release archives
```

The repository has separate linked worktrees for E0-E7 screening, Stage A,
Stage 8, paper tooling, Grad-CAM, and PrivateTest final evaluation. They were
left intact because they are Git history and active branch worktrees, not
untracked duplicate directories.

## Formal Content Preserved

The following evidence remains available and was not edited:

- current source and tests in `src/occlusion_fer/` and `tests/`;
- formal checkpoint registry and PrivateTest final evaluator;
- formal paper adapters and their tests that were already present as working
  tree changes;
- `outputs/gradcam_224_v2/` and its seed-42 checkpoint input copy for
  PublicTest Grad-CAM provenance;
- `03_PAPER/private_test_figures/`, including 60-condition source tables,
  figure manifest, metrics-only figures, and archive verification records;
- `03_PAPER/occlusion-fer-paper-figures/`, including the protocol figure;
- `outputs/gradcam_112_v1_obsolete/` and `outputs/stage_a/`, retained as legacy
  provenance because current manifests refer to their identities;
- external E7, Stage 8, and PrivateTest archives, checkpoints, manifests,
  predictions, summaries, and SHA records described in
  `docs/artifact_inventory.md`;
- `docs/provenance/private_final_training_mean_identity.md`, which records the
  Training-mean v2 identity semantics and the fix that was already part of the
  working tree.

## Organization Applied

### Historical archive

The following top-level reports were moved to the Git-ignored local archive
`local_archive/historical_reports/`:

- `PRIVATE_FINAL_GITHUB_DEPLOYMENT.md`;
- `PRIVATE_FINAL_GITHUB_RELEASE_REPORT.md`;
- `PRIVATE_FINAL_IMPLEMENTATION_REPORT.md`;
- `PRIVATE_TEST_AUDIT_FOR_CHATGPT.md`;
- `RESULTS_FOR_CHATGPT.md`.

They retain audit/provenance value but contain machine-specific paths, stale
intermediate status, or a ChatGPT handoff format and should not be presented as
the main GitHub documentation.

The existing `docs/superpowers/` plans/specs remain in place as development
history because their internal references are part of the historical record.

### Review before deletion

The two pilot directories were moved to:

```text
outputs/review_before_deletion/gradcam_112_v1_pilot_obsolete/
outputs/review_before_deletion/gradcam_224_v2_pilot/
```

They contain 34 files in total (about 428 KB). Hash comparison found eight
image payload hashes from each pilot in its corresponding full run; the pilot
manifests and metadata are not byte-identical because they record a different
selection and generation timestamp. Compatibility symlinks remain at the old
ignored paths so their recorded `overlay_path` values still resolve. Remove
each symlink and review target together only after the historical evidence is no
longer needed.

Known caches (`.pytest_cache`, Python `__pycache__`, and empty `tmp/`) were sent
to the macOS Trash rather than permanently deleted. The local `.venv`, FER2013
data, formal outputs, and paper figures were left intact.

The empty tracked `.gitkeep` placeholders in `configs/`, `scripts/`, `src/`,
and `tests/` were removed because those directories already contain real
tracked files (or no source files in the case of `scripts/`).

## Code and Documentation Decisions

- No duplicate formal algorithm was deleted. `private_paper_figures.py` reuses
  existing plotting functions and is intentionally an input/provenance adapter,
  not a second plotting system.
- Legacy `112`/`occlusion-v1` configs and modules remain available for history
  and compatibility; they are explicitly excluded from current formal results.
- `private_paper_figures.py` no longer has a hard-coded local PrivateTest root.
  `--private-root` is now required, and a test rejects personal `/Users/` and
  `/home/ucla/` paths in source code. This is a portability/path-interface fix,
  not an algorithm or result change.
- `README.md` was rewritten as a technical repository guide. It documents the
  actual 224/v2 protocol, exact Git lineage, result locations, reproduction
  boundaries, and the distinction between tracked source and external assets.
- `SERVER_RUN.md`, `docs/server_runbook.md`, `docs/research_context.md`, and
  `docs/paper_writing_blueprint.md` were updated to the completed E7/Stage 8
  evidence line and to remove stale current-stage claims.
- `docs/artifact_inventory.md` records external archives, local `.tar` hashes,
  known packaging gaps, and a release recommendation.
- `.gitignore` now covers generated paper directories, local cleanup archives,
  review areas, common Python/tool caches, logs, and packaging output. Data,
  weights, outputs, and figures remain ignored.
- `pyproject.toml` now has a factual project description and `requirements.txt`
  explains that dependency groups are declared in `pyproject.toml`.

Pre-existing user modifications to `AGENTS.md`, `src/occlusion_fer/paper_figures.py`,
`src/occlusion_fer/paper_framework.py`, `tests/test_paper_framework.py`, and the
untracked formal paper modules/tests were preserved and not reverted.

## Duplicate and Deprecated Content

Duplicate evidence found:

- pilot image payloads overlap the complete 112/v1 and 224/v2 Grad-CAM runs;
- `exp1_private_clean_summary.csv` exists both as a figure output and source-data
  copy, with identical bytes, and both are retained because the latter is the
  source table referenced by the figure bundle;
- Python cache files were generated duplicates and were trashed.

No whole formal source file was safely identified as a redundant copy. The
legacy single-checkpoint `final_evaluate.py`, Stage A artifacts, 112/v1 configs,
and old Grad-CAM outputs are deprecated relative to Stage 8, but remain
available for provenance rather than being silently removed.

## Path and Integrity Checks

- Full output and paper manifest paths were checked after the pilot move.
- Both compatibility symlinks resolve their original pilot metadata paths.
- Source and user-facing documentation scans found no personal `/Users/` or
  `/home/ucla/` paths; the guard test contains those literal strings only to
  reject them. Placeholder `/path/to/fer2013.csv` values remain intentional
  test/config placeholders.
- No dataset, checkpoint, archive, output, generated figure, or secret is Git
  tracked.
- `git diff --check` passed.
- Current external PrivateTest figure manifest reports internal artifact
  verification passed (306 files), extracted archive verification passed (307
  files), and frozen summary recomputation passed (208 numeric values).
- The external checksum caveat remains: historical sidecars name `.tar.gz`,
  while current local assets are `.tar`; the PrivateTest figure manifest
  therefore correctly records `archive_sha256_matches_sidecar=false`.
- The expected mixed PublicTest supplemental archive was not found; its
  extracted result directory is preserved and documented as a release gap.

## Git State

Branch: `private-final-eval-v2`

HEAD: `7e154aca1e95ef78ea7e3bc8767bcb21ca769335`

Remote operations: none. Commit operations: none. The worktree remains dirty by
design because it contains the user's earlier paper/Grad-CAM changes plus this
cleanup pass. The exact current state is available from:

```bash
git status --short --branch
git status --ignored --short
```

Snapshot at report generation:

```text
M  .gitignore
M  AGENTS.md                         (pre-existing user change)
M  README.md
M  SERVER_RUN.md
D  configs/.gitkeep
M  docs/paper_writing_blueprint.md
M  docs/research_context.md
M  docs/server_runbook.md
M  pyproject.toml
M  requirements.txt
D  scripts/.gitkeep
D  src/.gitkeep
M  src/occlusion_fer/paper_figures.py (pre-existing user change)
M  src/occlusion_fer/paper_framework.py (pre-existing user change)
D  tests/.gitkeep
M  tests/test_paper_framework.py      (pre-existing user change)
?? PROJECT_CLEANUP_REPORT.md
?? docs/artifact_inventory.md
?? docs/provenance/private_final_training_mean_identity.md
?? src/occlusion_fer/{gradcam_generate,paper_formal_completion,paper_gradcam,paper_stage8,private_paper_figures}.py
?? tests/{test_gradcam_generate,test_paper_classwise,test_paper_formal_completion,test_paper_gradcam,test_paper_stage8,test_private_paper_figures}.py
```

## Verification

Executed commands and results:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_private_paper_figures.py
2 passed in 0.51s

PYTHONPATH=src .venv/bin/python -m pytest -q
831 passed, 2 skipped in 36.63s

PYTHONPATH=src .venv/bin/python -m compileall -q src tests
PASS

git diff --check
PASS
```

The compile/test commands recreate ignored caches; those caches were sent to
Trash after verification.

## Recommended Next Action

1. Review the two pilot targets in `outputs/review_before_deletion/` and then
   remove the targets and compatibility symlinks together if the audit history
   is already retained elsewhere.
2. Before a GitHub Release, create fresh checksum files for the exact `.tar`
   or `.tar.gz` bytes that will be uploaded; do not reuse the mismatched old
   sidecars.
3. Decide whether to package the missing mixed PublicTest supplemental archive
   from its preserved extracted results, then verify its member list and hashes.
4. Keep FER2013 data, checkpoints, generated outputs, Grad-CAM overlays, and
   local reports outside the Git tree. No remote branch/release cleanup was
   performed by this task.

## Summary

```text
PROJECT CLEANUP: NEEDS REVIEW

Formal code preserved: YES
Formal experiment results preserved: YES (external archives and ignored local figures)
PrivateTest artifacts preserved: YES
Grad-CAM artifacts preserved: YES (224/v2 formal; 112/v1 legacy retained)
Checkpoints preserved: YES (external archives and verified seed-42 Grad-CAM input copy)
Reproducibility artifacts preserved: YES

Files reorganized: historical reports, provenance documentation, ignored paper/output areas
Files archived: 5 top-level reports under local_archive/historical_reports
Files moved to pending deletion: 2 pilot Grad-CAM directories, 34 files
Duplicate files found: pilot image overlap; one intentional source-table duplicate; caches
Deprecated code found: 112/v1 configs and legacy evaluator paths, retained for provenance

README rewritten: YES
.gitignore updated: YES
Code behavior changed: only explicit PrivateTest figure input path; no algorithm change
Experiment protocol changed: NO

Tests: 831 passed, 2 skipped; compileall and git diff --check passed
Broken paths detected: pilot metadata paths were protected by compatibility symlinks; no final broken paths

Recommended next action: review pending pilot targets, then regenerate release checksums for the exact archive bytes
```
