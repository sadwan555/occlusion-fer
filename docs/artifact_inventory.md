# Formal Artifact Inventory

This inventory separates Git-tracked source from external experiment assets.
It records the local `.tar` byte identities observed during project cleanup on
2026-08-14. It does not authorize a remote upload or replace each archive's
internal manifest.

## Git-Tracked Material

Keep these in the repository:

- source code and tests;
- portable configs with placeholder paths;
- protocol and research-boundary documentation;
- checkpoint/manifest validation logic;
- paper-figure generation code and Matplotlib style;
- cleanup and provenance reports that contain no local paths or private data.

Do not track FER2013 data, derived sample images, checkpoints, run outputs,
predictions, logs, generated figures, environments, credentials, or archives.

## Formal External Assets

| Asset | Role | SHA-256 of current local `.tar` |
|---|---|---|
| `fer2013-clean-e7-formal-4cb1e0f.tar` | three E7 clean runs, checkpoints, validation evidence | `b847f212b25010eeebdb464ce8a6fe6ff11c413e1f2e7f6555df2f326ede6f30` |
| `fer2013-clean-e7-selection-4cb1e0f.tar` | E7 selection evidence | `3cea6e57e3554ed83c461951624abd297acb5225ad4d9d2b524939602d543cc6` |
| `occlusion-fer-clean-baseline-e7-4cb1e0f.tar` | clean E7 code snapshot | `a630b2622af06e68114ef35acaeec54314a0c00b4f1f0e63934c7cc565df343b` |
| `experiment-2-clean-occlusion-results.tar` | clean-trained PublicTest ten-condition results | `6f8b97fdc78dfb5058d50145fba2b55989b18afcc7bb594609e58f60a95eef74` |
| `experiment-3-mixed-training-results.tar` | three mixed runs and checkpoints | `f69012e3f2847b9327b561c2b01f79001d9b420febe53575bc3895c41215a330` |
| `stage8-protocol-artifacts.tar` | Training mean v2, PublicTest manifest v2, configs and provenance | `f0a2502a81ac0b619bda4d436d44ecee1ff34c8d0fac2ee99213f0a5f4c24a9a` |
| `fer2013-private-test-final-v2.tar` | 60 final condition results, predictions, summaries, plan and manifest | `59495cff297f37ca4e4b0ab30f614094dcb148b60b7540359d02ca94026b8d66` |

The PrivateTest bundle does not contain raw FER2013 data or model checkpoints.
The clean and mixed training archives retain the six formal checkpoint sources.

## PrivateTest Evidence Checks

The generated PrivateTest figure manifest records:

- 60 condition results across two strategies, three seeds, and ten conditions;
- internal artifact-manifest verification: 306 files passed;
- extracted tree versus final archive: 307 files passed;
- frozen summary recomputation: 208 numeric values passed with `ddof=1`;
- PrivateTest canonical SHA:
  `4ab52c800e8abe786db253bb44a405b711fa81da2103c6e5eb00fa9a8ef3d634`;
- PrivateTest mask manifest SHA:
  `28f5463712cecf351ad318421e0fcc0e70db243621b859d94293a0aaeacd6220`;
- final plan SHA:
  `020b701e844e7d8af7fe7bb2133c0d3ff161acbf23deca9734700789db8165e8`.

These checks cover the source result tree, internal manifest, aggregate
recomputation, and local `.tar` contents. They do not resolve the external
sidecar mismatch below.

## Known Packaging Gaps

1. Existing Stage 8 `SHA256SUMS` entries name `.tar.gz` files, while the current
   local assets are uncompressed `.tar` files. Their byte hashes are expected to
   differ; the old sidecar must not be presented as verification of the `.tar`
   files.
2. The PrivateTest sidecar declares a server `.tar.gz` SHA
   `4f01c7e77799e9f9d0dd745be17ccdc19fb514a8be33be9ec520e424250232be`,
   while the current local `.tar` SHA is `59495c...`. The current figure manifest
   therefore correctly reports `archive_sha256_matches_sidecar=false`.
3. PublicTest mixed-checkpoint evaluation is present as an extracted result
   directory, but the expected
   `experiment-3-mixed-occlusion-evaluation-c1c9187` archive was not found in
   the audited release assets.

These are packaging/provenance issues, not evidence that the contained metrics
were recomputed or changed. Resolve them before calling a release bundle fully
verified.

## GitHub Release Recommendation

Create no remote changes automatically. After review, a release should use
new, immutable filenames and matching checksums generated from the exact bytes
uploaded. Recommended assets are:

- E7 clean formal archive;
- Stage 8 clean-occlusion results archive;
- Stage 8 mixed-training archive;
- Stage 8 protocol-artifacts archive;
- a verified archive of the mixed PublicTest ten-condition evaluation;
- PrivateTest final results archive;
- a metrics-only paper-figure/source-data bundle;
- one release README and one checksum file matching those exact assets.

Checkpoint-containing archives belong in the release or institutional storage,
not the Git tree. FER2013 data and Grad-CAM overlays derived from FER2013 samples
should remain private unless dataset licensing is separately reviewed.
