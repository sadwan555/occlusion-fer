# Project Cleanup Report

Date: 2026-08-14

This was a repository and artifact organization pass. No model was trained, no
formal evaluation was regenerated, no checkpoint/manifest/metadata bytes were
edited, and no remote GitHub branch or release was changed. Exact duplicate
paper and figure payloads outside Git were replaced by canonical symlinks; the
canonical bytes and all provenance records remain available.

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

The repository initially had 11 separate linked worktrees for E0-E7 screening,
Stage A, Stage 8, paper tooling, Grad-CAM, and PrivateTest final evaluation.
Nine clean historical worktrees were removed with `git worktree remove`; the
two dirty worktrees containing uncommitted paper tooling and a macOS test fix
remain under the workspace archive. Their branches and commits were retained.

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
- `outputs/stage_a/`, retained as legacy provenance;
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

The two pilot directories were initially moved to:

```text
outputs/review_before_deletion/gradcam_112_v1_pilot_obsolete/
outputs/review_before_deletion/gradcam_224_v2_pilot/
```

They contained 34 files in total (about 428 KB). The final audit confirmed both
pilots were superseded, excluded from the formal panel, and had no unique
scientific result. The review targets, the obsolete 112-v1 output tree, and
their compatibility path were removed. Historical index text remains as a
record of the retired pilot; no formal manifest depends on its bytes.

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
- `exp1_private_clean_summary.csv` was present both as a figure output and
  source-data copy; the source-data copy is canonical and the figure path is a
  compatibility symlink;
- 64 exact paper/literature duplicate groups were canonicalized. The generation
  output directory owns the bytes; review-center paths are compatibility
  symlinks so existing indexes remain valid;
- Python cache files were generated duplicates and were trashed.

No whole formal source file was safely identified as a redundant copy. The
legacy single-checkpoint `final_evaluate.py`, Stage A artifacts, and 112/v1
configs are deprecated relative to Stage 8 but remain available for provenance.
The obsolete pilot Grad-CAM payloads were removed after review; formal 224/v2
outputs remain available.

## Path and Integrity Checks

- Full output and paper manifest paths were checked after deduplication.
- Numbered review paths resolve through symlinks to the canonical paper figure
  entities; the retired pilot paths are intentionally absent.
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
- Current local archive sidecars now cover the `.tar` bytes, and the frozen
  mixed PublicTest supplemental archive was packaged deterministically from its
  existing 183-file extracted result tree. The archive contains 190 unique
  members: 183 files and 7 directories. Historical `.tar.gz` sidecars were
  preserved rather than overwritten.

## Git State

Branch: `private-final-eval-v2`

HEAD: `7dadb09` (`chore: finalize artifact deduplication and archival cleanup`)

Remote operations: the deduplication commit was pushed to
`origin/private-final-eval-v2` after verification. The exact current state is
available from:

```bash
git status --short --branch
git status --ignored --short
```

Final snapshot at report generation:

```text
## private-final-eval-v2...origin/private-final-eval-v2
```

## Verification

Executed commands and results:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_private_paper_figures.py
2 passed in 0.51s

PYTHONPATH=src .venv/bin/python -m pytest -q
831 passed, 2 skipped in 36.77s

PYTHONPATH=src .venv/bin/python -m compileall -q src tests
PASS

git diff --check
PASS
```

The compile/test commands recreate ignored caches; they remain Git-ignored.

## Recommended Next Action

1. Use the new current checksum files and supplemental archive for any future
   GitHub Release; do not overwrite the legacy sidecars.
2. Keep FER2013 data, checkpoints, generated outputs, Grad-CAM overlays, and
   local reports outside the Git tree. No remote branch or release deletion was
   performed by this task.

## Summary

```text
PROJECT CLEANUP: COMPLETE WITH DEDUPLICATION REVIEW PASSED

Formal code preserved: YES
Formal experiment results preserved: YES (external archives and ignored local figures)
PrivateTest artifacts preserved: YES
Grad-CAM artifacts preserved: YES (224/v2 formal; 112/v1 code/config provenance retained)
Checkpoints preserved: YES (external archives and verified seed-42 Grad-CAM input copy)
Reproducibility artifacts preserved: YES

Files reorganized: historical reports, provenance documentation, ignored paper/output areas
Files archived: 5 top-level reports under local_archive/historical_reports
Files removed from workspace: 2 pilot Grad-CAM directories, 51 obsolete pilot files
Duplicate files found: paper/literature groups were canonicalized with symlinks;
  one PrivateTest source-table duplicate was canonicalized; archive/extracted
  experiment layers were retained for reproducibility
Deprecated code found: 112/v1 configs and legacy evaluator paths, retained for provenance

README rewritten: YES
.gitignore updated: YES
Code behavior changed: only explicit PrivateTest figure input path; no algorithm change
Experiment protocol changed: NO

Tests: 831 passed, 2 skipped; compileall and git diff --check passed
Broken paths detected: none in active code; historical retired pilot paths remain only as text provenance

Recommended next action: use the current checksum files and supplemental archive for any future release; retain the legacy sidecars as historical records
```
