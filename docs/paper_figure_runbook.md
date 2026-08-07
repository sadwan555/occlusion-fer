# Paper figure and result export runbook

## Purpose and boundary

These commands create visualization and reporting artifacts from already
approved FER2013 sources and completed formal outputs. They do not train a
model, run inference, select a checkpoint, alter a mask, or change any research
result. The final real-data commands are human-only HIVE tasks.

The tools expose only `Training` and `PublicTest`. They have no `PrivateTest`
route. A combined FER2013 CSV is read through the Stage B Usage router: every
row's `Usage` value is inspected, but excluded `PrivateTest` emotion and pixel
fields are not parsed or materialized. A permitted-splits JSON source is also
supported.

Install the declared project dependencies and run from a clean checkout of the
paper-tools commit:

```bash
python -m pip install -e .
export PYTHONPATH=src
git status --short --branch
git rev-parse HEAD
```

Never generate real paper artifacts from a dirty checkout. Keep generated
figures and tables outside the repository, under an ignored output root.

## FER2013 class examples

The dataset montage provides one or more examples of each of the seven
dataset-defined facial-expression labels. Selection is not visual or
performance-based. For each class, records are sorted by stable physical sample
ID and the first `N` are selected. The default is `N=1` and `Training`.

Raw 48x48 grayscale pixels are not enhanced or altered. For print display, each
pixel is replicated by a fixed integer nearest-neighbor scale. PNGs are lossless
and carry 300 dpi metadata.

```bash
PYTHONPATH=src python scripts/paper/export_fer2013_examples.py \
  --data-path "${FER_DATA_CSV}" \
  --split Training \
  --samples-per-class 1 \
  --output-dir "${FER_PAPER_ROOT}/fer2013-examples"
```

`--split PublicTest` is permitted when explicitly needed. `PrivateTest` is not
an accepted option.

Expected files:

```text
fer2013-examples/
  angry_sample_<id>.png
  disgust_sample_<id>.png
  fear_sample_<id>.png
  happy_sample_<id>.png
  sad_sample_<id>.png
  surprise_sample_<id>.png
  neutral_sample_<id>.png
  fer2013_class_examples.png
  figure_provenance.json
```

The sidecar records the exact IDs, label order, selection rule, split hashes,
source routing version, scaling rule, Git identity, creation command, and every
PNG SHA-256.

## Formal occlusion examples

The occlusion montage illustrates the exact `occlusion-v2-224` evaluator order:

```text
clean
upper_face_0.20
upper_face_0.30
upper_face_0.40
lower_face_0.20
lower_face_0.30
lower_face_0.40
random_rectangle_0.20
random_rectangle_0.30
random_rectangle_0.40
```

Choose and record the sample ID before consulting predictions or condition
performance. The CLI deliberately requires `--sample-id`; it never picks a
sample because a model was correct, incorrect, robust, or visually convenient.

For PublicTest, the canonical v2 manifest and sidecar are mandatory. The loader
validates dataset, mean, protocol, condition, digest, and geometry identities.
The exporter then calls `apply_evaluation_mask_v2` and verifies every returned
geometry field against the selected manifest rows. Random-rectangle coordinates
therefore come from the same deterministic protocol/manifest used for formal
evaluation. No rectangle implementation exists in the paper tool.

The fill is derived only from the validated `training_mean_v2.json` through
`normalized_fill_vector_v2`. It is never silently replaced with black or a
caller-provided value. The formal model-domain mask is created first. Only then
is it inverse ImageNet-normalized, clipped to the display range, quantized to
uint8, and nearest-neighbor scaled for the PNG. This display conversion cannot
feed back into evaluation or model results.

```bash
PYTHONPATH=src python scripts/paper/export_occlusion_examples.py \
  --data-path "${FER_DATA_CSV}" \
  --split PublicTest \
  --sample-id <PREDECLARED_PUBLICTEST_SAMPLE_ID> \
  --training-mean "${FER_STAGE_B_ARTIFACTS}/training_mean_v2.json" \
  --manifest "${FER_STAGE_B_ARTIFACTS}/publictest_manifest_v2.csv" \
  --manifest-sidecar "${FER_STAGE_B_ARTIFACTS}/publictest_manifest_v2.json" \
  --output-dir "${FER_PAPER_ROOT}/occlusion-example"
```

Expected files are `clean.png`, one PNG named for each of the nine masked
conditions, `occlusion_protocol_examples.png`, and `figure_provenance.json`.
The sidecar records exact mask coordinates, target and actual ratios, fill and
display rules, split/mean/manifest semantic identities, Git identity, and output
hashes.

## Formal result tables and plots

Run this only after all six PublicTest ten-condition evaluations are complete.
Each directory must contain `evaluation_provenance.json`, must not contain
`failure.json`, and must contain all ten condition metrics. The command requires
both strategies and all formal seeds; it cannot select or omit a seed.

```bash
PYTHONPATH=src python scripts/paper/export_formal_results.py \
  --clean-run "42=${FER_CLEAN_EVAL_ROOT}/seed42" \
  --clean-run "123=${FER_CLEAN_EVAL_ROOT}/seed123" \
  --clean-run "2026=${FER_CLEAN_EVAL_ROOT}/seed2026" \
  --mixed-run "42=${FER_MIXED_EVAL_ROOT}/seed42" \
  --mixed-run "123=${FER_MIXED_EVAL_ROOT}/seed123" \
  --mixed-run "2026=${FER_MIXED_EVAL_ROOT}/seed2026" \
  --clean-training-run "42=${FER_CLEAN_TRAIN_ROOT}/seed42" \
  --clean-training-run "123=${FER_CLEAN_TRAIN_ROOT}/seed123" \
  --clean-training-run "2026=${FER_CLEAN_TRAIN_ROOT}/seed2026" \
  --mixed-training-run "42=${FER_MIXED_TRAIN_ROOT}/seed42" \
  --mixed-training-run "123=${FER_MIXED_TRAIN_ROOT}/seed123" \
  --mixed-training-run "2026=${FER_MIXED_TRAIN_ROOT}/seed2026" \
  --output-dir "${FER_PAPER_ROOT}/formal-results"
```

Training-run options are optional as a complete set. When omitted, all
condition tables, confusion matrices, and robustness plots are still produced;
only `training_history_by_seed.csv` and `training_curves.png` are omitted.

Main outputs:

```text
condition_metrics_by_seed.csv
condition_summary.csv
strategy_comparison.csv
per_class_metrics_by_seed.csv
per_class_summary.csv
robustness_accuracy.png
robustness_macro_f1.png
confusion_matrices/<strategy>_<condition>.png
training_history_by_seed.csv       # when training runs are supplied
training_curves.png                 # when training runs are supplied
paper_results_provenance.json
```

Means and sample standard deviations use all three seeds. Clean-to-occluded
drops are first paired within each seed and only then aggregated. Strategy
deltas are also paired by seed. The PNG curves retain the individual seed
points. The exporter rejects mismatched evaluation commit, datasets, mean,
manifest, protocol, image size, split, label order, sample counts, or dirty
formal provenance.

## Provenance verification

Inspect the recorded identity and recompute output hashes before inserting a
figure into Word:

```bash
python -m json.tool "${FIGURE_DIR}/figure_provenance.json" | less
sha256sum "${FIGURE_DIR}"/*.png

python -m json.tool \
  "${FER_PAPER_ROOT}/formal-results/paper_results_provenance.json" | less
find "${FER_PAPER_ROOT}/formal-results" -type f -name '*.png' \
  -exec sha256sum {} +
```

The JSON `output_files_sha256` entries must equal the recomputed values. For the
occlusion figure, also confirm `protocol=occlusion-v2-224`, the expected sample
ID, ten-condition order, Training/PublicTest hashes, Training mean semantic SHA,
manifest semantic SHA, and `git_dirty=false`.

Output directories are create-only. A rerun must use a new empty path; no tool
silently overwrites an existing directory.

## Suggested captions

Figure X. Example images from the FER2013 dataset representing the seven facial
expression categories. Samples were selected deterministically by ascending
sample ID within each category.

Figure Y. Illustration of the clean input and nine deterministic occlusion
conditions used in the experiments. Masks follow the locked
`occlusion-v2-224` protocol and use the Training-split global pixel mean as the
fill value.

These captions describe the data and method only. They do not claim recognition
of internal emotion, confusion, understanding, engagement, or learning outcome,
and they do not claim real-world robustness, cross-dataset generalization,
novelty, or state-of-the-art performance.
