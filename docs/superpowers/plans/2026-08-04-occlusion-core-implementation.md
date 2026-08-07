# Stage A Occlusion Core Implementation Plan

> Historical Stage A plan. It documents the superseded v1/112 development
> stage; consult [`../../experiment_protocol.md`](../../experiment_protocol.md)
> for current formal work and do not use this plan to launch experiments.

> For implementation: execute one task at a time. Run the focused checks for that task, stop for review, and only then start the next task. This document authorizes only its own plan-only commit under the gate below; it does not authorize Python implementation, dataset processing, metric generation, or an implementation commit.

## Goal

Implement and verify the deterministic occlusion core for the FER2013 ResNet-18 study while preserving the existing clean baseline exactly. Stage A produces the reusable configuration, hashing, geometry, batch API, Training-mean artifact, and PublicTest mask manifest needed by Stage B mixed training and final evaluation.

Stage A must end with evidence that:

- the approved protocol is represented by strict, validated configuration;
- the same sample receives the same mask regardless of batch ordering, batch size, or DataLoader worker count;
- source clean tensors, labels, and sample IDs remain unchanged;
- Training mean and the PublicTest manifest are reproducible and byte-addressable;
- the existing clean training and checkpoint-loading path remains compatible;
- no model performance result is produced from an occluded PublicTest or any PrivateTest data.

## Locked Protocol

The implementation must use the approved design specification at docs/superpowers/specs/2026-08-04-occlusion-core-design.md as its source of truth:

    mask_algorithm_version = occlusion-v1
    mean_algorithm_version = training-mean-v1
    manifest_schema_version = 1
    evaluation_mask_seed = 20260804
    formal input size = 112 x 112
    formal model = ImageNet-pretrained ResNet-18

Fixed facts:

- sample_id is the original FER2013 CSV source-line number.
- Only Training is used to compute the mean artifact.
- Only PublicTest is used to create the Stage A evaluation manifest; internally this is the validation split.
- The exact combined FER2013 CSV may be streamed as opaque bytes solely to compute dataset_sha256. Excluded rows may have only their Usage field inspected so the reader can reject or skip them.
- Training rows may be parsed into complete records. PublicTest rows may expose only the split identity and source-line sample_id needed by the manifest.
- Stage A does not parse or decode PrivateTest labels or pixels, construct PrivateTest records, Dataset objects, or DataLoaders, enumerate PrivateTest sample IDs, generate its manifest, or run inference or metrics on it.
- Evaluation uses seed 20260804; training mask generation uses the formal run seed and epoch.
- Approved types are upper_face, lower_face, and random_rectangle; approved ratios are exactly 0.20, 0.30, and 0.40.
- Every compared checkpoint uses the same fixed evaluation manifest.
- Clean evaluation keeps the existing clean path and never calls the mask function.

## Architecture and Interfaces

Planned module responsibilities:

| Module | Responsibility |
| --- | --- |
| src/occlusion_fer/config.py | Strict Stage A protocol configuration and cross-field validation. |
| src/occlusion_fer/mask_hash.py | Canonical payload serialization, SHA-256, unsigned big-endian u64, and namespace decisions. |
| src/occlusion_fer/occlusion.py | Integer geometry, normalized fill, immutable batch application, and mask metadata. |
| src/occlusion_fer/training_mean.py | Training-split raw-pixel mean and deterministic artifact serialization. |
| src/occlusion_fer/mask_manifest.py | PublicTest manifest generation, schema validation, stable CSV bytes, and create-or-verify behavior. |
| src/occlusion_fer/stage_a_artifacts.py | Narrow fixed-subcommand CLI for reproducible HIVE artifact generation. |
| Existing data.py, torch_data.py, train.py, evaluation.py | Read-only clean-compatibility targets in Stage A; no masked inference integration. |
| src/occlusion_fer/artifacts.py | Shared artifact support only if the canonical writers require a narrowly scoped helper. |

The batch API accepts clean_images, sample_ids, condition or protocol fields, and the validated fill vector. It does not accept labels. It returns masked_images and mask_metadata, preserving tensor shape, dtype, device, and sample-ID values. It explicitly clones clean_images before filling pixels. Geometry decisions are CPU integer operations from canonical hashes; CUDA only fills already determined coordinates. Label preservation is verified at the upper training/evaluation integration boundary, not by passing labels through the occlusion core.

## Global Constraints

- Do not modify source images in place or write derived images to the repository.
- Do not alter the locked clean baseline result or change formal hyperparameters after seeing a final-test result.
- Do not implement mixed training in Stage A.
- Do not run mixed training, final_evaluate, PublicTest occluded accuracy or macro-F1, or any PrivateTest semantic data operation during this plan. Opaque whole-file hashing and Usage-only split filtering are the only permitted contacts with excluded CSV rows.
- Do not commit FER2013 data, manifests containing private local paths, weights, outputs, credentials, or secrets.
- Do not use Python built-in hash or CUDA RNG for protocol decisions.
- Do not use ambiguous round; geometry uses the specified half-up integer rule.
- Keep training.mode: clean as the only accepted Stage A training mode.
- Resolve paths from YAML or CLI input; introduce no hard-coded local data path.
- Preserve unrelated user changes and do not use destructive Git commands.
- Never stash, restore, checkout, reset, amend, delete, or stage an unrelated user file.
- Never use git add -A, git add ., a directory-wide add, or a wildcard add for Stage A.
- Every implementation task adds or updates focused tests before the corresponding code is accepted.

## Plan Document Commit Gate

This revised plan must be committed by itself before Task 1 begins:

1. Record the actual pre-commit HEAD with git rev-parse HEAD.
2. Verify git merge-base --is-ancestor 16c356b HEAD succeeds.
3. Stage only docs/superpowers/plans/2026-08-04-occlusion-core-implementation.md by its exact path.
4. Inspect git diff --cached --name-only and fail unless it contains exactly that one plan path.
5. Commit the plan locally without amending an existing commit and do not push.
6. Report the new plan commit SHA and the complete git status --short output.
7. Stop and wait for explicit approval before Task 1.

## Task 1: Freeze the implementation surface

Files:

- Make no repository modification; the plan file must already be present in its separate approved commit.
- Read, but do not edit, docs/superpowers/specs/2026-08-04-occlusion-core-design.md, src/occlusion_fer/data.py, src/occlusion_fer/config.py, src/occlusion_fer/torch_data.py, src/occlusion_fer/train.py, src/occlusion_fer/evaluation.py, and src/occlusion_fer/artifacts.py.

Acceptance checks during implementation:

- Record the actual starting HEAD with git rev-parse HEAD and the clean/dirty status before code edits.
- Verify with git merge-base --is-ancestor 16c356b HEAD that the approved design revision is an ancestor of the implementation starting point.
- Confirm no data or output files are tracked by Git.
- Confirm official split mapping and CSV-line sample_id behavior with read-only tests before changing shared code.
- Treat the locked design specification as immutable unless a new review approves a protocol change.

## Task 2: Add the strict configuration protocol

Files:

- Modify src/occlusion_fer/config.py only as needed to represent Stage A.
- Add configs/fer2013_stage_a.yaml containing only portable locked protocol values; runtime data and output paths remain CLI overrides or validated environment-provided values.
- Add focused tests under tests/ for valid and invalid Stage A configurations.
- Add or update a minimal YAML fixture under tests/fixtures/ if the existing fixture cannot express the protocol.

Interface and behavior:

- Add an explicit occlusion/evaluation configuration object containing algorithm versions, manifest schema version, evaluation seed, approved type and ratio lists, image size, fill source, and resolved artifact locations.
- Validate dataset_name == fer2013, exact scalar types, finite numeric values, duplicate-free ordered lists, image_size == 112, evaluation_mask_seed == 20260804, and locked version strings.
- Reject missing or unknown fields, booleans in integer positions, unsupported types or ratios, non-finite values, mismatched image size, invalid fill source, and any attempt to select test or PrivateTest for Stage A.
- Keep existing clean configurations loadable and keep training.mode restricted to clean during Stage A.
- Resolve paths from YAML or CLI input; add no hard-coded local data path.

Test-first sequence:

1. Write tests for one valid resolved configuration and each invalid cross-field case.
2. Run focused configuration tests and record the expected failing state.
3. Implement validation and resolved-configuration serialization.
4. Run focused tests again, then existing configuration and preflight tests.
5. Run git diff --check and inspect that no unrelated config behavior changed.

Acceptance evidence:

- A resolved configuration serializes with stable field names and values.
- Clean baseline configuration behavior is unchanged.
- Invalid Stage A requests fail before data or model work begins.

## Task 3: Implement canonical deterministic hashing

Files:

- Add src/occlusion_fer/mask_hash.py.
- Add tests/test_mask_hash.py and a golden-vector fixture if useful.

Interface and behavior:

- Serialize hash payloads as UTF-8 canonical JSON arrays using ensure_ascii=True, allow_nan=False, and separators=(",", ":"). Do not sort array elements, add a prefix or suffix, or add a newline. Element order and JSON types are part of the protocol.
- Distinguish Training and Evaluation payloads exactly as specified. Training apply/type/ratio payloads are ["fer2013","train",sample_id,training_seed,epoch,"occlusion-v1",namespace]; Training coordinate payloads replace namespace with selected_condition,"occlusion-v1","top" or "left"; Evaluation coordinate payloads are ["fer2013","validation",sample_id,condition,20260804,"occlusion-v1","top" or "left"]. Never include labels, predictions, or image content.
- Compute SHA-256 over canonical payload bytes; interpret the first eight digest bytes as an unsigned big-endian integer.
- Expose independent namespaces apply, type, ratio, top, and left; never combine top and left into one position decision.
- Perform decisions on CPU using integer arithmetic. Accept the negligible modulo bias for % 3 as part of occlusion-v1; describe the nine type-ratio probabilities as approximately 0.5 / 9.

Test-first sequence:

1. Add exact serialization, namespace separation, repeatability, and input-sensitivity tests.
2. Add the fixed golden case: sample_id 28711, random_rectangle_0.30, top 28, left 46, and the approved payload/hash values from the design specification.
3. Run focused tests and record the expected failing state.
4. Implement the hash API and rerun focused tests.
5. Run deterministic-transform and seed-related existing tests.

Acceptance evidence:

- Golden vectors match byte-for-byte.
- Reordering a batch or changing worker count cannot alter a sample decision.
- Different namespaces do not share accidental state.

## Task 4: Implement occlusion geometry and the batch API

Files:

- Add src/occlusion_fer/occlusion.py.
- Add tests/test_occlusion_geometry.py, tests/test_occlusion_batch.py, and normalized-fill fixtures.

Interface and behavior:

- Implement only upper_face, lower_face, and random_rectangle.
- Use formal 112 x 112 geometry, half-open intervals, and round_half_up(x) = floor(x + 0.5).
- Lock these dimensions and pixel counts:

  | Condition | Mask dimensions | Mask pixels | Actual ratio |
  | --- | ---: | ---: | ---: |
  | upper/lower 0.20 | 22 x 112 | 2464 | 0.1964285714 |
  | upper/lower 0.30 | 34 x 112 | 3808 | 0.3035714286 |
  | upper/lower 0.40 | 45 x 112 | 5040 | 0.4017857143 |
  | random 0.20 | 50 x 50 | 2500 | 0.1992984694 |
  | random 0.30 | 61 x 61 | 3721 | 0.2966358418 |
  | random 0.40 | 71 x 71 | 5041 | 0.4018654337 |

- Apply the Training mean as a normalized three-channel fill vector after the existing ImageNet normalization path.
- Accept clean_images, sample_ids, condition or protocol fields, and the validated fill vector only. Do not accept labels or predictions.
- Clone clean_images after it is moved to the device; fill all channels in the clone; leave sample_ids unchanged; return target and actual ratios, integer geometry, seed, algorithm version, and condition in metadata.
- Never call this API for original or clean. Do not mutate input, even for a rejected invalid request.
- Validate shape, dtype, device, batch length, sample IDs, and supported conditions before writing tensor values.

Test-first sequence:

1. Add pure geometry tests for all six fixed dimensions, boundary coordinates, half-up behavior, and the golden random rectangle.
2. Add batch tests for unchanged source, all-channel fill, shape/dtype/device/sample-ID preservation, rejection of label-like unsupported inputs, and invalid-input failure. Verify label preservation separately in the clean integration tests.
3. Add invariance tests for batch ordering, batch sizes 1/32/128, worker counts 0/4, and single-sample versus batch processing.
4. Run focused tests and record the expected failing state.
5. Implement geometry and batch application, rerun focused tests, then run the full existing unit suite.

Acceptance evidence:

- Every geometry assertion is integer-derived and matches the locked table.
- Each sample's output is identical across the required batching and worker permutations.
- Clean input and metadata remain unchanged.

## Task 5: Create the deterministic Training mean artifact

Files:

- Add src/occlusion_fer/training_mean.py.
- Add tests/test_training_mean.py.
- Integrate only necessary artifact path or metadata hooks in src/occlusion_fer/artifacts.py.

Interface and behavior:

- Stream the exact combined CSV as opaque bytes before parsing to compute dataset_sha256 over the complete file, including all split bytes. Do not interpret labels or pixels during this hash pass.
- During the semantic pass, inspect Usage to select Training, parse only Training rows into records, and use their raw 48 x 48, 0..255 grayscale pixels. Excluded rows must not become records; PrivateTest labels and pixels must not be parsed or decoded.
- Use numpy.uint64 for pixel accumulation with an explicit proof/check that the maximum possible sum is below numpy.iinfo(numpy.uint64).max. Convert the final integer to Python int only for canonical JSON serialization. Verify expected pixel count 66,145,536 for the official Training split.
- Emit exactly the semantic fields schema_version, dataset_name, split, dataset_sha256, sample_count, pixel_count, raw_pixel_sum, raw_training_mean, source_pixel_range, normalized_pixel_range, accumulator_dtype, and mean_algorithm_version. Keep generation time in run metadata, not the deterministic artifact.
- Write canonical JSON bytes exactly as locked: ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"), UTF-8, and one trailing LF.
- Compute mean_artifact_sha256 from complete artifact bytes. Re-running on identical CSV bytes must reproduce identical bytes and hash.
- Reject empty or malformed Training rows, invalid Training labels, malformed Training pixels, non-finite values, wrong Training counts, or a command requesting any split other than train.

Test-first sequence:

1. Add a tiny combined synthetic CSV fixture for lower-level integer sum, count, split filtering, whole-file SHA-256, malformed Training data, and deterministic-byte tests; keep the production official-count assertion (28,709 samples and 66,145,536 pixels) in the HIVE data-level test.
2. Add a normalization-equivalence test with absolute tolerance 1e-6.
3. Add a spy or monkeypatch test proving an excluded PrivateTest row may expose Usage for skipping but its emotion and pixels are never parsed, decoded, or materialized as a record.
4. Run focused tests and record the expected failing state.
5. Implement the calculator and canonical writer.
6. Rerun focused tests and existing data/split validation tests.

Acceptance evidence:

- Artifact identity is stable across repeated generation and operating-system JSON line-ending differences.
- The recorded mean is explicitly the raw Training-split mean, not an exact mean of resized model input.
- No evaluation split contributes a pixel or semantic record to the calculator; its bytes contribute only to the locked whole-file dataset_sha256.

## Task 6: Generate and verify the PublicTest evaluation manifest

Files:

- Add src/occlusion_fer/mask_manifest.py.
- Add tests/test_mask_manifest.py and a small synthetic manifest fixture.

Interface and behavior:

- Stream the exact combined CSV as opaque bytes when dataset_sha256 is needed. For manifest generation, inspect Usage and source-line position, enumerate only PublicTest/internal validation sample_id values, and do not parse PublicTest labels or pixels.
- PrivateTest rows may expose Usage only so they can be skipped; do not parse their labels or pixels, create records, or enumerate their source-line positions as sample IDs.
- Produce exactly nine type-ratio conditions for every PublicTest sample with evaluation seed 20260804, yielding 3589 x 9 = 32,301 rows for the official split.
- Use this exact column order: manifest_schema_version, algorithm_version, dataset_name, split, dataset_sha256, sample_id, condition, occlusion_type, target_ratio, image_height, image_width, top, left, height, width, masked_pixel_count, total_pixel_count, actual_ratio, raw_fill_value, normalized_fill_red, normalized_fill_green, normalized_fill_blue, evaluation_mask_seed, coordinate_convention.
- Use fixed semantic values for schema version, algorithm version, dataset name, validation split, 112 x 112 dimensions, total pixel count 12,544, seed 20260804, and coordinate convention half-open:[top,top+height)x[left,left+width).
- Use UTF-8/no BOM, comma delimiter, double-quote quoting with minimal quoting and doubled embedded quotes, LF endings with exactly one final LF, and condition then ascending sample_id row order.
- Format target_ratio with two decimal places, actual_ratio with ten, raw_fill_value and normalized channel fills with seventeen; write integers in base 10 without leading zeros and use no locale-dependent formatting.
- Store image_height, image_width, masked_pixel_count, total_pixel_count, target and actual ratios, and all algorithm/version fields required by the exact schema. Record the complete manifest SHA-256 in artifact or run metadata rather than adding non-schema columns.
- Implement create-or-verify: an existing manifest is accepted only when canonical bytes/hash and protocol identity match; never overwrite a different manifest.
- Do not include labels, predictions, accuracy, macro-F1, or any PrivateTest row.

Test-first sequence:

1. Add tests for schema order, row count, split filtering, stable bytes, duplicate/missing IDs, hash/geometry consistency, collision/rewrite refusal, and whole-file dataset SHA-256.
2. Add a spy or monkeypatch test proving PrivateTest labels, pixels, and sample IDs are not materialized during manifest generation.
3. Run focused tests and record the expected failing state.
4. Implement manifest generation and verification.
5. Rerun focused tests, then the complete unit suite.

Acceptance evidence:

- The same PublicTest CSV and mean artifact yield one reproducible manifest hash.
- A changed protocol, row, or byte causes verification failure without replacing the original.
- The manifest contains no model-derived result.

## Task 7: Add the narrow Stage A artifact CLI

Files:

- Add src/occlusion_fer/stage_a_artifacts.py.
- Add tests/test_stage_a_artifacts_cli.py.

Command contract:

    python -m occlusion_fer.stage_a_artifacts training-mean \
      --config configs/fer2013_stage_a.yaml \
      --data-path "$FER2013_CSV" \
      --output-dir "$STAGE_A_OUTPUT_DIR"

    python -m occlusion_fer.stage_a_artifacts validation-manifest \
      --config configs/fer2013_stage_a.yaml \
      --data-path "$FER2013_CSV" \
      --mean-artifact "$STAGE_A_OUTPUT_DIR/training_mean.json" \
      --output-dir "$STAGE_A_OUTPUT_DIR"

Interface and behavior:

- Expose only the fixed subcommands training-mean and validation-manifest. Do not expose a general split parameter.
- Bind training-mean to internal split train and validation-manifest to internal split validation. Reject an unknown subcommand or any attempted test/PrivateTest input before dataset loading.
- Print a structured completion summary containing the artifact path, artifact SHA-256, dataset SHA-256, and sample count; validation-manifest also prints manifest row count and per-condition counts.
- Do not output image data, labels, predictions, losses, accuracy, macro-F1, confusion matrices, or performance drops.
- Do not place machine-specific absolute paths, command lines, or timestamps in either canonical artifact. Record created_at_utc and the executed command only in a separate non-canonical run log or process log.
- Use the module APIs from Tasks 5 and 6; do not duplicate mean, hashing, geometry, or manifest logic inside the CLI.

Test-first sequence:

1. Add subprocess-level tests for both fixed subcommands with synthetic fixtures.
2. Add tests showing there is no accepted split option and test is rejected before any dataset reader is constructed.
3. Add stdout/run-log schema tests and assertions that canonical artifacts contain no local path or timestamp.
4. Run focused CLI tests and record the expected failing state.
5. Implement the narrow dispatcher and rerun focused tests.

Acceptance evidence:

- HIVE has one documented, repeatable entry point for both Stage A artifacts.
- Repeated calls use create-or-verify behavior and report identical canonical hashes.
- The CLI cannot be repurposed for PrivateTest or model inference.

## Task 8: Preserve clean compatibility without masked inference integration

Files:

- Read src/occlusion_fer/evaluation.py, src/occlusion_fer/torch_data.py, and src/occlusion_fer/train.py as compatibility targets; do not modify them in Stage A unless a focused test demonstrates a regression caused by the new protocol modules.
- Modify src/occlusion_fer/artifacts.py only if a narrowly scoped shared helper is required by the canonical writers.
- Do not modify src/occlusion_fer/train.py or train_one_epoch in Stage A; the future explicit epoch argument belongs to Stage B.
- Add compatibility tests under tests/.

Interface and behavior:

- Keep the clean branch behavior compatible: original uses the existing clean tensor, label, and sample_id path and does not instantiate or invoke an occlusion transform.
- Preserve strict loading of existing clean ResNet-18 checkpoints and output shape.
- Do not change train_one_epoch to add epoch-dependent masking in Stage A; the explicit epoch argument belongs to Stage B.
- Verify that evaluation.py original needs no mean artifact and no manifest. Do not add a manifest-reading, masked-image, or masked-model-inference path to evaluation.py in Stage A.
- Preserve current optimizer, epoch budget, batch size, augmentation, seed set, and checkpoint rules.

Test-first sequence:

1. Add regression tests comparing clean tensor hashes, labels, and sample IDs before and after new modules are imported.
2. Add checkpoint save/load and original one-batch forward/backward compatibility tests.
3. Add a test proving the original evaluation branch works without a mean artifact or manifest.
4. Run focused compatibility tests and record the expected failing state.
5. If a focused test exposes a clean-path regression caused by the new modules, make the smallest approved compatibility fix; otherwise leave existing business modules unchanged. Rerun compatibility tests, then existing clean smoke-training tests on a synthetic fixture.
6. Inspect the diff for accidental changes to clean hyperparameters, checkpoint selection, or a new masked inference path.

Acceptance evidence:

- Existing clean checkpoint loading and smoke training remain green.
- No Stage A path silently switches a clean run to an occluded condition.
- No persistent-worker synchronization state is used for mask decisions.

## Task 9: Run the complete unit and integration test matrix

Required checks:

- configuration validation;
- label mapping and official split mapping;
- split disjointness;
- batch shape and deterministic transforms;
- model output shape;
- one-batch forward and backward;
- all six fixed geometry cases and golden hash case;
- unchanged source image;
- metric correctness and checkpoint save/load;
- normalization/fill equivalence;
- batch ordering, batch size, worker count, and single/batch invariants;
- manifest schema, row count, stable bytes, and no-overwrite behavior;
- clean-mode smoke training only.

Implementation-phase commands:

    python -m pytest -q
    python -m compileall -q src tests
    python -m pip check
    git diff --check

The execution report must include exact commands and exit statuses. “Passed” may be reported only after commands have actually run.

## Task 10: Perform HIVE data-level validation only

Prerequisites:

- A validated YAML configuration supplied by the user or HIVE environment.
- FER2013 CSV available through that configuration.
- Tasks 2-9 have passed.

Allowed future checks:

1. Invoke the two fixed stage_a_artifacts subcommands from Task 7. The CLI may stream the exact combined CSV as opaque bytes to compute dataset_sha256 and inspect Usage solely to select or skip rows.
2. Confirm 28,709 parsed Training records and 3,589 enumerated PublicTest sample IDs. Do not report or retain a PrivateTest count or its sample IDs.
3. Compute the Training mean artifact and record its canonical bytes and hash.
4. Generate the PublicTest manifest with seed 20260804 and record its canonical bytes and hash, 32,301-row count, and nine condition counts.
5. Re-run both commands to prove byte-for-byte reproducibility and create-or-verify behavior.
6. Validate the manifest against the HIVE CSV and mean artifact without constructing a model or computing metrics.

Hard stops:

- The exact combined CSV may be streamed as opaque bytes solely to compute dataset_sha256. Excluded rows may have their Usage field inspected only to reject or skip them.
- Do not parse or decode PrivateTest labels or pixels, construct PrivateTest records, Dataset objects, or DataLoaders, enumerate its sample IDs, generate its manifest, or run inference or metrics.
- Do not run occluded PublicTest inference, accuracy, macro-F1, confusion matrix, or clean-to-occluded drop calculation.
- Do not train mixed clean/occluded models or inspect a checkpoint selected using occluded metrics.
- If the approved reader cannot inspect Usage and skip PrivateTest without parsing its emotion or pixels, stop and report the data-boundary failure. Do not weaken the boundary merely to obtain counts.

Acceptance evidence:

- HIVE row counts and hashes agree with the resolved configuration and manifest schema.
- The report records dataset identity, artifact hashes, command lines, environment, and any failure without exposing local data paths or samples.

## Task 11: Assemble final evidence and prepare the Stage A commit

Files:

- Add only the configuration, implementation modules, and tests listed in the reviewed staging allowlist from Tasks 2-10.
- Keep generated artifacts, outputs, checkpoints, datasets, and local paths ignored by Git.

The reviewed Stage A implementation staging allowlist is:

- configs/fer2013_stage_a.yaml
- src/occlusion_fer/config.py
- src/occlusion_fer/mask_hash.py
- src/occlusion_fer/occlusion.py
- src/occlusion_fer/training_mean.py
- src/occlusion_fer/mask_manifest.py
- src/occlusion_fer/stage_a_artifacts.py
- src/occlusion_fer/artifacts.py
- tests/test_config.py
- tests/test_data.py
- tests/test_torch_data.py
- tests/test_train.py
- tests/test_evaluation.py
- tests/test_artifacts.py
- tests/test_mask_hash.py
- tests/test_occlusion_geometry.py
- tests/test_occlusion_batch.py
- tests/test_training_mean.py
- tests/test_mask_manifest.py
- tests/test_stage_a_artifacts_cli.py

If implementation requires any path outside this list, stop and obtain a plan amendment before staging it.

Evidence bundle:

- resolved configuration;
- starting and ending Git commit, dirty-state record, and diff summary;
- exact test commands and exit statuses;
- Training mean artifact and SHA-256;
- PublicTest manifest and SHA-256;
- fixed golden-vector verification;
- HIVE data-level validation report;
- explicit statement that PrivateTest received no semantic parsing, record/Dataset/DataLoader construction, sample-ID enumeration, manifest, inference, or metric; also record that the only permitted contacts were opaque whole-file hashing and Usage-only skipping;
- failure logs for rejected input or failed validation.

Commit gate:

1. Record the actual implementation starting HEAD with git rev-parse HEAD; do not assume it is 16c356b.
2. Verify git merge-base --is-ancestor 16c356b HEAD succeeds and record the result.
3. Run git status --short and inspect every changed path without stashing, restoring, checking out, resetting, or amending anything.
4. Run Task 9 checks and Task 10 HIVE checks.
5. Confirm no prohibited file is tracked with git ls-files and the repository ignore rules.
6. Stage only individual exact paths from the allowlist. Never use git add -A, git add ., a directory, or a wildcard.
7. Run git diff --cached --name-only and fail if the staged set contains README.md, pyproject.toml, src/occlusion_fer/paper.mplstyle, src/occlusion_fer/paper_figures.py, tests/test_paper_figures.py, a dataset, output, checkpoint, the separately committed plan file, or any path outside the allowlist.
8. Review git diff --cached, then create one local Stage A implementation commit only after the user approves the completed evidence; do not push it.
9. Report the commit ID, exact verification commands, results, complete git status --short output, and remaining limitations.

## Stage A Completion Definition

Stage A is complete only when every task's focused and full tests pass, invalid inputs fail clearly, deterministic hashes and artifact bytes are reproduced, clean compatibility is demonstrated, HIVE data-level validation is recorded, no prohibited data or metric is accessed, and the implementation commit contains only the approved scope. Stage B mixed training and all final model metrics remain a separate reviewed stage.
