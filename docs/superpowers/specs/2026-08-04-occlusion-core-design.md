# Synthetic Occlusion Core Design

## 1. Purpose and research boundary

Stage A defines and later implements the reproducible synthetic-occlusion core
for the FER2013 project. It adds protocol validation, the Training-split fill
value artifact, deterministic mask generation, batch-level mask application,
PublicTest mask manifests, and automated verification. The locked mask
algorithm version is `occlusion-v1`.

The formal model remains the ImageNet-pretrained, standard-stem ResNet-18. The
formal input remains a three-channel `112 x 112` tensor. Stage A does not train
a mixed model, evaluate masked classification performance, or access
PrivateTest.

The project predicts FER2013 dataset-defined facial-expression labels. It does
not measure verified internal emotion, confusion, understanding, engagement,
learning outcome, or another cognitive state. The controlled masks do not
establish real-world occlusion robustness, cross-dataset generalization,
novelty, or state-of-the-art performance.

This design preserves the clean baseline trained at Git commit
`da889bdd818cab6403767f2f6c7d5391d8317324`. No existing baseline artifact is
rewritten or replaced.

## 2. Existing repository constraints

The implementation must fit the current repository rather than replace its
working clean pipeline:

- `src/occlusion_fer/data.py` maps the official `Training`, `PublicTest`, and
  `PrivateTest` values to internal `train`, `validation`, and `test` splits.
- `Fer2013Record.sample_id` is the integer CSV source line number. It is already
  stable across shuffle, batch size, subset wrapping, and worker count.
- `src/occlusion_fer/torch_data.py` converts an original `48 x 48` grayscale
  image to `[0, 1]`, resizes it to `112 x 112`, replicates it to three channels,
  and optionally applies ImageNet normalization.
- The formal train and validation datasets request ImageNet normalization.
- `src/occlusion_fer/train.py` moves a completed batch to the selected device
  inside `train_one_epoch` before model inference.
- `train_one_epoch` does not currently receive an epoch argument. Stage B must
  add an explicit one-based epoch argument when it integrates mixed training;
  Stage A must not hide epoch state in the Dataset.
- Formal DataLoaders use persistent workers whenever `num_workers > 0`.
- `src/occlusion_fer/evaluation.py`, not an `evaluate.py` file, owns reusable
  validation and test inference.
- `src/occlusion_fer/artifacts.py` provides replace-style atomic JSON and CSV
  writers. A mask manifest needs a separate create-or-verify writer because a
  different existing manifest must never be replaced.
- `src/occlusion_fer/config.py` currently accepts only `training.mode=clean`.
  Stage A may define and validate the occlusion protocol without enabling
  mixed training. Stage B owns the later `mixed` mode integration.
- The committed YAML retains a portable dataset-path value. Runtime data and
  artifact paths remain externally supplied and validated.
- The existing dependency set is sufficient. Stage A adds no third-party
  runtime or test dependency.
- `outputs/` is already ignored by Git and is the approved root for local audit
  material. Data, derived images, manifests, checkpoints, and logs remain
  untracked.

The integer `sample_id` representation must not change. Dataset identity is the
combination of dataset name, internal split, source-line `sample_id`, and the
SHA-256 of the exact FER2013 CSV bytes. Prediction files produced by the clean
baseline remain join-compatible.

## 3. Terminology

In existing code and artifacts, `clean` means the original FER2013 input with
no synthetic mask added by this project. It does not mean that the source image
is naturally free of eyeglasses, hair, hands, crop loss, or another incidental
occlusion.

Paper prose, table labels, and figure labels should prefer `original`,
`unmasked`, or `without additional synthetic occlusion`. The semantic
selection condition is called `original` in the protocol. The existing
`training.mode=clean` token and legacy clean-baseline artifact value `clean`
remain valid and are not rewritten. Later aggregation must map legacy `clean`
to the semantic `original` condition explicitly.

An occlusion ratio is always a spatial ratio:

```text
actual_ratio = masked spatial pixels / (image_height * image_width)
```

All three channels at a masked spatial position are filled, but channel count
is never included in `masked_pixel_count` or `actual_ratio`.

`upper_face` and `lower_face` are retained configuration names. They are
top-aligned and bottom-aligned full-width image bands, respectively. They only
approximate upper- and lower-face regions. The protocol uses no facial
landmarks and must not claim that it precisely selects eyes, eyebrows, nose,
or mouth.

## 4. Architecture and module boundaries

Stage A implementation should use focused modules with the following future
ownership. This design task does not create them.

- `src/occlusion_fer/occlusion.py` owns approved constants, immutable protocol
  and mask metadata types, half-up geometry, normalized fill construction,
  single-sample masking, and batch-level masking.
- `src/occlusion_fer/mask_hash.py` owns canonical payload serialization,
  SHA-256-to-u64 conversion, decision namespaces, and deterministic integer
  mappings. It must not import CUDA random APIs.
- `src/occlusion_fer/training_mean.py` owns exact Training-only accumulation,
  dataset-file hashing, mean artifact validation, and atomic artifact output.
- `src/occlusion_fer/mask_manifest.py` owns PublicTest manifest rows, canonical
  CSV bytes, sorting, file hashing, and create-or-verify persistence.
- `src/occlusion_fer/config.py` gains the minimal immutable occlusion protocol
  schema and strict field validation while retaining compatibility with the
  current clean configuration.
- `src/occlusion_fer/train.py` remains the optimization orchestrator. Stage B,
  not Stage A, supplies the one-based epoch and training seed to the batch mask
  interface and records realized training-mask counts.
- `src/occlusion_fer/evaluation.py` remains the inference owner. A later stage
  may add a narrow batch-mask integration point after device transfer and
  before model inference without duplicating metric computation.
- `src/occlusion_fer/artifacts.py` remains the owner of existing metric and run
  artifacts. It may consume lineage values but must not become the mask
  geometry or manifest owner.

Expected focused tests are:

- `tests/test_occlusion.py` for geometry, fill behavior, batch invariants, and
  unchanged input tensors;
- `tests/test_mask_hash.py` for canonical bytes, u64 values, namespace
  separation, and deterministic mappings;
- `tests/test_training_mean.py` for Training-only exact accumulation and
  artifact validation;
- `tests/test_mask_manifest.py` for schema bytes, ordering, hashing, and
  conflict-safe persistence;
- minimal additions to `tests/test_config.py`, `tests/test_train.py`, and
  `tests/test_evaluation.py` for integration and clean compatibility.

No unrelated refactor is part of Stage A.

## 5. Data flow

The current clean tensor path remains authoritative:

```text
FER2013 CSV 48 x 48 uint8 grayscale
-> validate pixels, label, split, and source-line sample_id
-> convert pixels to float and scale to [0, 1]
-> bilinear resize to 112 x 112 with align_corners=False
-> replicate to three contiguous channels
-> apply ImageNet normalization
-> Dataset returns clean tensor, label, sample_id
-> DataLoader forms a batch
-> move image and label batches to device
```

The original condition sends the moved clean batch directly to the model. A
masked condition continues as follows:

```text
clean_images on device
-> masked_images = clean_images.clone()
-> generate CPU integer geometry from stable sample IDs and protocol fields
-> fill all three channels at each selected two-dimensional region
-> return masked_images and ordered mask_metadata
-> send masked_images to the model
```

The batch interface must preserve image shape, floating dtype, and device. It
must not change labels or sample IDs. Metadata is returned in the same order as
the input sample IDs. Invalid dimensions, non-finite input tensors, duplicate
evaluation sample IDs, mismatched batch lengths, invalid conditions, and an
incompatible fill vector fail explicitly.

The original branch must not call a mask function, clone the image batch, or
require a Training-mean artifact to run. This keeps the existing clean
behavior and checkpoint compatibility independently testable.

## 6. Training mean artifact

The fill source is the global mean of original FER2013 Training-split grayscale
pixels. It is not the exact mean of resized `112 x 112` model inputs.

The locked algorithm is `training-mean-v1`:

1. Compute the SHA-256 lowercase hexadecimal digest of the exact CSV file
   bytes before parsing.
2. Load only the internal `train` split through the validated FER2013 reader.
3. Require exactly 28,709 Training samples.
4. Require every source image to contain exactly `48 * 48 = 2,304` uint8
   pixels in the inclusive range 0 through 255.
5. Accumulate all raw integer pixels with an unsigned 64-bit accumulator.
6. Require exactly `28,709 * 48 * 48 = 66,145,536` accumulated pixels.
7. Compute the normalized mean once as the exact integer sum divided in
   float64 by `255 * 66,145,536`.
8. Reject a non-finite mean or a mean outside `[0, 1]`.

The raw sum cannot overflow uint64 because its maximum is
`66,145,536 * 255`, which is far below the uint64 limit. No PublicTest or
PrivateTest pixel contributes to the sum.

The JSON artifact has `schema_version=1` and contains exactly these semantic
fields:

```text
schema_version
dataset_name                  = fer2013
split                         = train
dataset_sha256
sample_count                  = 28709
pixel_count                   = 66145536
raw_pixel_sum
raw_training_mean
source_pixel_range            = [0, 255]
normalized_pixel_range        = [0.0, 1.0]
accumulator_dtype             = uint64
mean_algorithm_version        = training-mean-v1
```

Generation time is event metadata, not part of the deterministic mean
artifact. The command's run metadata or log records `created_at_utc` in
ISO-8601 UTC with second precision and a trailing `Z`.

The mean artifact bytes are exactly:

```python
json.dumps(
    artifact,
    ensure_ascii=True,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8") + b"\n"
```

No indentation, additional whitespace, byte-order marker, or additional
newline is permitted. The artifact is written atomically outside Git. The
existing indented JSON artifact helper is not used for these canonical bytes.
A loader validates every fixed field, exact field set, integer count, finite
range, algorithm version, and digest shape. It also hashes the runtime CSV and
requires exact equality with `dataset_sha256`. A mismatch fails before any
mask is generated.

The artifact SHA-256 is computed from the exact closed artifact bytes and is
recorded as `mean_artifact_sha256` wherever the artifact is consumed. The same
FER2013 CSV and `training-mean-v1` algorithm must reproduce identical artifact
bytes and the same SHA-256 on macOS and Linux. The runtime path is supplied
through validated configuration or a command-line override and is never
hard-coded into source or committed YAML.

## 7. Mask geometry

The formal image dimensions are fixed at height 112 and width 112. Protocol
validation rejects a formal `image_size` other than 112. The core may be
internally testable with dimensions supplied by a fixture, but no alternate
dimension is a valid `occlusion-v1` experiment.

The only approved type order is:

```text
0 upper_face
1 lower_face
2 random_rectangle
```

The only approved ratio order is:

```text
0 0.20
1 0.30
2 0.40
```

Numeric configuration values are converted through their decimal string form
and compared with the approved decimal set. Boolean values, non-finite values,
negative values, values greater than one, and unapproved values fail. The
canonical condition names always use two decimal places:

```text
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

The canonical full condition order is semantic `original` followed by those
nine values in the displayed order. A mask manifest contains only the nine
masked conditions.

All dimension conversion uses the explicitly named function:

```text
round_half_up(x) = floor(x + 0.5)
```

`upper_face` is a full-width top band:

```text
height = round_half_up(target_ratio * 112)
width  = 112
top    = 0
left   = 0
```

`lower_face` is a full-width bottom band:

```text
height = round_half_up(target_ratio * 112)
width  = 112
top    = 112 - height
left   = 0
```

`random_rectangle` is a fixed-aspect-ratio square with a deterministic random
position:

```text
side   = round_half_up(sqrt(target_ratio) * 112)
height = side
width  = side
```

Coordinates use zero-based half-open intervals:

```text
top  <= y < top  + height
left <= x < left + width
```

The square must be entirely inside the image:

```text
0 <= top  <= 112 - height
0 <= left <= 112 - width
```

The locked geometries are:

| Condition family | Target ratio | Height x width | Masked pixels | Actual ratio |
|---|---:|---:|---:|---:|
| upper/lower | 0.20 | 22 x 112 | 2,464 | 0.1964285714... |
| upper/lower | 0.30 | 34 x 112 | 3,808 | 0.3035714286... |
| upper/lower | 0.40 | 45 x 112 | 5,040 | 0.4017857143... |
| random | 0.20 | 50 x 50 | 2,500 | 0.1992984694... |
| random | 0.30 | 61 x 61 | 3,721 | 0.2966358418... |
| random | 0.40 | 71 x 71 | 5,041 | 0.4018654337... |

For every row:

```text
masked_pixel_count = height * width
total_pixel_count  = 112 * 112 = 12544
actual_ratio       = masked_pixel_count / total_pixel_count
```

The spatial count is not multiplied by three.

## 8. Deterministic hashing protocol

Python's built-in `hash()` is prohibited. All decisions use canonical JSON,
SHA-256, unsigned big-endian conversion, and integer mappings on CPU.

The canonical payload bytes are exactly:

```python
json.dumps(
    payload,
    ensure_ascii=True,
    allow_nan=False,
    separators=(",", ":"),
).encode("utf-8")
```

No prefix, suffix, byte-order marker, or newline is added. Payloads are JSON
arrays whose element order and JSON types are part of the protocol. Dataset,
split, condition, algorithm version, and namespace elements are strings;
sample ID, seed, and epoch elements are JSON integers.

The decision integer is:

```text
digest = SHA-256(canonical_payload_bytes)
u64    = first 8 digest bytes interpreted as unsigned big-endian
```

Mask geometry never consumes Python, NumPy, PyTorch, DataLoader, or CUDA random
state. CPU integer logic determines all decisions before the device assignment
fills tensor regions.

For an evaluation square:

```text
top  = u64_top  % (image_height - mask_height + 1)
left = u64_left % (image_width  - mask_width + 1)
```

The same mapping is used by a training square. The negligible modulo bias is
accepted because the mapping is simple, deterministic, and auditable.

Changing payload fields, field order, JSON settings, digest selection,
endianness, namespace strings, or integer mappings requires a new mask
algorithm version.

## 9. Training-time mask protocol

Stage A specifies but does not integrate or run mixed training. Stage B applies
this protocol.

The three formal mixed seeds are 42, 123, and 2026. The effective
`training_seed` equals the run's existing `training.seed` and is saved in the
resolved configuration. Epoch is a one-based integer from 1 through the locked
epoch budget.

Before type and ratio are known, each decision uses:

```json
["fer2013","train",sample_id,training_seed,epoch,"occlusion-v1",namespace]
```

The namespace is one of `apply`, `type`, or `ratio`.

Application is an independent Bernoulli decision:

```text
apply mask if u64_apply < 2^63
```

This is exactly probability 0.5 over the u64 space. It does not require an
epoch to contain exactly half masked samples.

Conditional on mask application:

```text
type_index  = u64_type  % 3
ratio_index = u64_ratio % 3
```

The index orders are the locked orders in Section 7. Each of the nine
type-ratio combinations therefore has an unconditional probability
approximately equal to `0.5 / 9`. The negligible modulo bias introduced by
`u64 % 3` is accepted as part of `occlusion-v1`. The label is absent from every
payload and decision.

After the selected condition is canonicalized, random-square coordinates use:

```json
["fer2013","train",sample_id,training_seed,epoch,selected_condition,"occlusion-v1","top"]
```

and:

```json
["fer2013","train",sample_id,training_seed,epoch,selected_condition,"occlusion-v1","left"]
```

Band conditions use their fixed coordinates and do not consume coordinate
hashes. A training mask must be identical for the same training seed, epoch,
sample ID, selected condition, and algorithm version. Different epochs obtain
separate deterministic decisions without storing epoch state in Dataset worker
copies.

Stage B records realized original and nine-condition counts for every epoch,
plus per-class after-the-fact counts. Labels may be read for reporting only;
they never influence application, type, ratio, or coordinates. A full
per-sample, per-epoch training manifest is unnecessary because the protocol can
regenerate every decision.

## 10. Evaluation-time mask protocol

The single locked `evaluation_mask_seed` is the integer `20260804`. Every model,
training strategy, and formal seed uses this value. The internal split string
is `validation` for PublicTest. Stage A does not generate any row for internal
split `test`.

Each random-square coordinate uses:

```json
["fer2013",split,sample_id,condition,20260804,"occlusion-v1","top"]
```

and:

```json
["fer2013",split,sample_id,condition,20260804,"occlusion-v1","left"]
```

Band conditions use fixed geometry and do not require coordinate hashes. The
mask generator accepts sample IDs and condition values, not labels or model
predictions. No evaluation decision may use a true label, predicted label,
confidence, metric, checkpoint identity, training strategy, or training seed.

For the same split, sample ID, condition, evaluation seed, dataset digest, and
algorithm version, the mask is identical across all checkpoints. PublicTest
manifest generation produces geometry only. Stage A must not pass the masked
images through a research model or calculate masked accuracy, macro-F1, loss,
confusion matrices, or performance drops.

## 11. Persistent-worker and batch-level constraints

The current formal DataLoader uses `persistent_workers=True` whenever
`num_workers > 0`. Stage A and later Stage B must not drive epoch-dependent
masking by assigning `dataset.epoch = epoch` in the main process. Dataset
objects held by persistent worker processes may not observe that assignment.

The later training integration must instead pass the one-based epoch explicitly
from the training loop to a batch-level mask function:

- Dataset always returns the existing clean tensor, label, and sample ID;
- `train_one_epoch` receives an explicit epoch argument;
- the batch is moved to the device;
- the batch-level function derives each sample's mask from CPU integer logic;
- the function clones the batch and fills the clone on the device;
- the model receives the masked clone.

This makes the result independent of worker lifetime, shuffle order, DataLoader
read order, batch size, and worker count. Stage A documents this integration
contract but does not change `train_one_epoch` or implement mixed training.

## 12. Batch-level application

The batch API accepts:

- a finite floating image tensor of shape `[batch, 3, 112, 112]` on any
  supported device;
- a one-dimensional integer sample-ID tensor with matching batch length;
- one canonical masked condition for evaluation, or deterministic per-sample
  training decisions for the future Stage B path;
- an immutable validated protocol;
- a validated three-value normalized fill vector.

It returns:

- `masked_images`, with the same shape, dtype, and device as the input images;
- one immutable `MaskMetadata` item per masked sample in input order.

Application begins with behavior equivalent to:

```python
masked_images = clean_images.clone()
```

The clone occurs after the batch is moved to its device and before any mask is
written. The fill vector is converted to the image dtype and device without
changing its semantic channel values. AMP autocast begins only for model
inference; mask application operates on the ordinary input tensor dtype.

For each selected spatial region, all three channels are assigned their
respective normalized fill values. The input `clean_images` tensor must remain
bitwise unchanged after the call. Labels and sample IDs are not inputs to the
pixel-assignment operation except that sample IDs select deterministic
geometry; labels are never accepted by the hash or mask API.

Single-sample and batch APIs must share the same geometry and assignment code.
They must not contain parallel implementations that can drift.

## 13. Configuration schema and validation

The future Stage A configuration model adds an immutable `OcclusionConfig`
with these resolved semantic fields:

```yaml
occlusion:
  algorithm_version: occlusion-v1
  image_size: 112
  types: [upper_face, lower_face, random_rectangle]
  ratios: [0.20, 0.30, 0.40]
  fill_source: training_split_global_mean
  evaluation_mask_seed: 20260804
```

The runtime operation that consumes the mean artifact also receives a
validated artifact path through configuration or a command-line override. No
machine-specific path is committed.

Validation requires exact algorithm version, dataset name `fer2013`, image
size 112, the exact type order, the exact ratio order, the exact fill source,
and evaluation seed 20260804. Missing and unknown fields inside `occlusion`
fail. Unknown types, duplicate types, reordered types, unknown ratios,
duplicate ratios, reordered ratios, booleans used as numbers, non-finite
ratios, negative ratios, ratios greater than one, and any unapproved ratio
fail without fallback.

The existing clean configuration remains loadable without an occlusion section
for the original clean path. An operation that generates a mean artifact,
masked tensor, or manifest requires an explicit validated occlusion protocol
and fails if none is present. Stage A does not permit `training.mode=mixed`;
Stage B later extends the allowed mode while preserving the current clean
validation behavior.

The resolved configuration or operation metadata records:

```text
raw_training_mean
normalized_fill_vector
imagenet_channel_means       = [0.485, 0.456, 0.406]
imagenet_channel_stds        = [0.229, 0.224, 0.225]
image_size                   = 112
dataset_sha256
mean_artifact_sha256
mask_algorithm_version       = occlusion-v1
evaluation_mask_seed         = 20260804
training_seed                = formal run seed when applicable
```

## 14. Manifest schema and atomic persistence

Stage A may generate one PublicTest evaluation mask manifest after protocol
tests pass. It contains 3,589 samples times nine masked conditions, for 32,301
rows. It contains no image bytes, label, prediction, confidence, checkpoint
identity, or metric.

Manifest schema version 1 uses this exact column order:

```text
manifest_schema_version
algorithm_version
dataset_name
split
dataset_sha256
sample_id
condition
occlusion_type
target_ratio
image_height
image_width
top
left
height
width
masked_pixel_count
total_pixel_count
actual_ratio
raw_fill_value
normalized_fill_red
normalized_fill_green
normalized_fill_blue
evaluation_mask_seed
coordinate_convention
```

Fixed semantic values include:

```text
manifest_schema_version = 1
algorithm_version       = occlusion-v1
dataset_name            = fer2013
split                   = validation
image_height            = 112
image_width             = 112
total_pixel_count       = 12544
evaluation_mask_seed    = 20260804
coordinate_convention   = half-open:[top,top+height)x[left,left+width)
```

Rows are ordered by the locked masked-condition order in Section 7 and then by
ascending integer `sample_id`. Sample IDs must be unique within PublicTest.

Canonical manifest bytes use:

- UTF-8 encoding without a byte-order marker;
- comma delimiter;
- double quote as quote character;
- `csv.QUOTE_MINIMAL` behavior;
- doubled embedded quote characters and no escape character;
- the exact header order above;
- LF line endings on every row;
- exactly one LF after the final row;
- `target_ratio` formatted with exactly two digits after the decimal point;
- `actual_ratio` formatted with exactly ten digits after the decimal point;
- raw and normalized fill values formatted with exactly seventeen digits after
  the decimal point;
- integers written in base 10 without leading zeros;
- no locale-dependent formatting.

The SHA-256 lowercase hexadecimal digest is computed from the complete closed
canonical CSV bytes.

Manifest persistence is create-or-verify, not replace-style:

1. Serialize all bytes into a temporary file in the target directory.
2. Flush and close the file, then compute its SHA-256.
3. If the target does not exist, publish the completed temporary file through
   a filesystem operation that fails rather than replacing a concurrently
   created target.
4. If the target exists, hash its exact bytes.
5. If hashes and bytes are identical, remove the temporary file and report the
   existing target as consistent.
6. If they differ, remove the temporary file and raise an explicit protocol
   conflict without modifying the target.
7. Clean up temporary files on all handled failures.

The implementation must account for the check-to-publish race. On the formal
macOS and Linux environments, an exclusive same-filesystem hard-link publish or
another exclusive-create primitive is acceptable. The existing replace-style
CSV helper is not acceptable for this manifest.

Any change to a reproducibility-relevant manifest field requires either a new
manifest schema version or a new mask algorithm version. Changes to geometry,
hashing, fill semantics, condition names or order, or serialization fields that
drive regeneration require a version later than `occlusion-v1`.

## 15. Reproducibility and experiment lineage

Stage A artifacts record their own schema versions, algorithm versions, exact
dataset digest, Git commit, Git dirty state, and artifact digests. No artifact
stores credentials, environment-variable dumps, image pixels, or an absolute
data path.

The Training mean and PublicTest manifest must be generated from the same CSV
digest. A mismatch fails before manifest serialization. The normalized fill
vector is computed as follows for raw normalized mean `m`:

```text
red_fill   = (m - 0.485) / 0.229
green_fill = (m - 0.456) / 0.224
blue_fill  = (m - 0.406) / 0.225
```

The mean artifact records `m`; resolved operation metadata records both `m`
and the three-value vector. Computation uses float64 metadata values and casts
the vector to the destination tensor dtype only at assignment time.

Future Stage D condition results must record:

```text
checkpoint_training_commit
evaluation_commit
checkpoint_sha256
mask_manifest_sha256
dataset_sha256
evaluation_git_dirty
mask_algorithm_version
mean_artifact_sha256
resolved_evaluation_configuration
```

The existing clean checkpoints retain training commit
`da889bdd818cab6403767f2f6c7d5391d8317324`; the later evaluation commit is
recorded separately. A checkpoint is never identified only by filename.

Changing any of the following requires a mask version later than
`occlusion-v1`: canonical payload schema, JSON serialization settings, digest
byte selection, endianness, namespace, decision mapping, type or ratio order,
condition token or order, half-up algorithm, mask dimensions, coordinate
convention, coordinate mapping, spatial-ratio definition, fill source or fill
normalization, or any manifest field that changes mask regeneration.

## 16. Test strategy

Stage A follows the existing pytest style: small deterministic fixtures,
explicit invalid-input assertions, `tmp_path` for artifacts, and CPU-first
tests. No test needs FER2013 data unless it is an explicitly separated HIVE
integration check for the real Training mean or PublicTest manifest.

Geometry and application tests cover:

- `upper_face` changes only the locked top band;
- `lower_face` changes only the locked bottom band;
- every random square is entirely inside the image;
- all nine conditions produce the exact locked dimensions;
- `masked_pixel_count` equals `height * width`;
- `actual_ratio` is derived from two-dimensional spatial pixels;
- all three channels receive the appropriate fill value;
- `clean_images` remains bitwise unchanged;
- output shape, dtype, and device equal the input values;
- labels and sample IDs remain unchanged by integration;
- invalid tensor dimensions, dtype, finite state, fill vector, condition, and
  metadata fail clearly.

Configuration and artifact tests cover:

- unknown, duplicate, missing, and reordered types;
- unknown, duplicate, missing, and reordered ratios;
- negative, greater-than-one, boolean, and non-finite ratios;
- formal image size other than 112;
- dataset SHA-256 mismatch;
- malformed or internally inconsistent Training mean artifacts;
- non-canonical Training mean JSON bytes or an unexpected field;
- different Training mean bytes or SHA-256 for the same fixed input content;
- a manifest target with different bytes;
- unexpected manifest row count or duplicate PublicTest sample IDs.

Hash and determinism tests cover:

- exact canonical payload bytes;
- fixed payload-to-u64 examples;
- unsigned big-endian interpretation;
- namespace separation for `apply`, `type`, `ratio`, `top`, and `left`;
- identical training masks for identical seed, epoch, sample ID, and protocol;
- deterministic epoch-dependent changes over fixed golden cases;
- identical evaluation masks across checkpoint identities and strategies;
- batch reordering without per-sample mask changes;
- batch sizes 1, 32, and 128 without per-sample mask changes;
- `num_workers=0` and `num_workers=4` without per-sample mask changes;
- single-sample and batch interfaces producing equivalent outputs;
- shuffle order without per-sample mask changes;
- CPU coordinate generation remaining independent of CUDA random state.

Clean compatibility tests cover:

- the original condition does not invoke a mask function;
- clean tensors match commit `da889bd` behavior exactly for fixed source
  fixtures and representative real-data audit hashes;
- clean labels and sample IDs remain unchanged;
- old clean checkpoints load strictly in the later evaluator;
- the original clean evaluation path remains usable without a mean artifact;
- no Stage A training or evaluation path loads the internal `test` split.

Preprocessing equivalence tests compare:

```text
normalize(fill(raw_image, raw_training_mean))
```

with:

```text
fill(normalize(raw_image), normalized_fill_vector)
```

using `rtol=0` and `atol=1e-6`. The comparison covers every channel and every
masked position.

Manifest tests lock field order, row order, encoding, quoting, LF endings,
final newline, numeric formatting, exact canonical bytes, and SHA-256. The
same expected bytes are used on macOS and Linux. Rewriting identical content
reports consistency; different content raises a conflict and preserves the
original target.

The future Stage A implementation is not complete until the exact full-suite
test command, compile check, dependency check, diff check, and data/secret/
checkpoint scans are run and their actual results are reported. Smoke checks
are engineering evidence and are not formal research results.

## 17. Golden test specification

Golden cases use artificial `3 x 112 x 112` float32 CPU tensors and fixed
integer sample IDs. At least one case exists for each band ratio and each
random-square ratio. A fixed evaluation case uses internal split `validation`,
evaluation seed 20260804, and algorithm version `occlusion-v1`.

Each golden case locks:

```text
condition
top
left
height
width
masked_pixel_count
total_pixel_count
actual_ratio
SHA-256 of the contiguous CPU uint8 two-dimensional mask bytes
```

The boolean mask is converted to contiguous row-major uint8 values containing
only zero and one before hashing. The final floating tensor bytes are not
hashed. Floating output uses:

```python
torch.testing.assert_close(
    actual,
    expected,
    rtol=0,
    atol=1e-6,
)
```

Golden fixtures also lock at least one canonical JSON byte string and its
expected u64 integer, so serialization or endianness drift fails immediately.

The required evaluation golden case is sample ID 28711 under
`random_rectangle_0.30`. Its top payload is exactly:

```text
["fer2013","validation",28711,"random_rectangle_0.30",20260804,"occlusion-v1","top"]
```

Its locked values are:

```text
top payload SHA-256 = 1143203e5bd651f03c57bc0d33a9c9b1a67abe3abe59c0fdcaf39ae4972482b2
top u64            = 1243873374285222384
top                = 1243873374285222384 % 52 = 28
```

Its left payload is exactly:

```text
["fer2013","validation",28711,"random_rectangle_0.30",20260804,"occlusion-v1","left"]
```

Its locked values are:

```text
left payload SHA-256 = 02c6f1f2889688fefb763e33845bb00e618be168351c3e9a1dc827806677d34c
left u64            = 200113257440512254
left                = 200113257440512254 % 52 = 46
```

The resulting mask uses `top=28`, `left=46`, `height=61`, and `width=61`.
Its 12,544 row-major uint8 bytes contain 3,721 ones and have SHA-256:

```text
49fe672bc883fb68796d85c9317f68ed0c08ab532d7c82aac04495a5dce8d9fa
```

These constants are part of `occlusion-v1`; changing any one requires a
protocol review rather than updating the expected test value to match code.

## 18. Visual audit policy

Stage A may produce a small visual grid from fixed Training sample IDs solely
to verify top, bottom, and random-square placement. The suggested local
directory is `outputs/occlusion-audit/`, which is already covered by the
repository ignore rules.

Audit images:

- are derived FER2013 data and never enter Git;
- are never included in public evidence archives;
- do not modify source arrays or the source CSV;
- are not used for training, mask-shape selection, ratio selection, or
  checkpoint selection;
- do not trigger a protocol change based on model performance;
- contain only the minimum fixed samples required for human geometry review.

Stage A does not modify `.gitignore`. Before any commit, tracked-file and
ignored-output checks must confirm that no audit image, dataset, manifest,
weight, output, or local path was added.

## 19. PrivateTest isolation

Stage A code paths are limited to artificial fixtures, the FER2013 Training
split for the global mean, and PublicTest sample IDs for the validation
manifest. They must not request internal split `test`, open PrivateTest image
rows for mask generation, generate a PrivateTest manifest, invoke
`final_evaluate`, or calculate a PrivateTest metric.

The Stage A manifest command accepts only internal split `validation` and
rejects `test` before data loading. Training-mean generation accepts only
internal split `train`. Unit tests use spies or monkeypatching to prove these
gates execute before a disallowed dataset read.

No label is serialized into a mask manifest or included in a hash payload.
Future PrivateTest masks are generated only in Stage C after all six best
checkpoints and the complete protocol are locked. They use only sample IDs,
split identity, condition, dataset digest, evaluation seed, and algorithm
version, never test labels or performance.

## 20. Stage boundaries and future stages

Stage A contains only:

1. strict occlusion protocol configuration;
2. exact Training global-mean calculation and artifact validation;
3. the three approved geometries and three approved ratios;
4. canonical deterministic hashing and integer decisions;
5. single-sample and batch-level clone-and-fill interfaces;
6. PublicTest geometry manifest generation and conflict-safe persistence;
7. clean-path and old-checkpoint compatibility verification;
8. automated unit, integration, and smoke checks that produce no masked
   research metric;
9. documentation and the lock of `occlusion-v1`.

Stage B adds mixed training for seeds 42, 123, and 2026. It uses the same
ResNet-18, preprocessing, optimizer, batch size, epoch budget, base behavior,
formal seed set, and checkpoint rule as clean-only training. Each epoch selects
`best.pt` only by semantic-original PublicTest macro-F1:

```text
selection_split     = PublicTest
selection_condition = original
selection_metric    = macro_f1
```

It must not select by masked-condition mean, worst masked condition, or a
combined original-and-masked score.

Stage C freezes the three clean-only best checkpoints, three mixed best
checkpoints, the exact checkpoint SHA-256 values, and PrivateTest manifests for
the locked `occlusion-v1` protocol. It does not tune a model or mask from
PrivateTest information.

Stage D evaluates six checkpoints on the semantic-original condition and the
nine locked masked conditions. It emits all per-condition artifacts and
aggregates paired drops without changing training or mask rules.

For metric `M`, each seed's drop is:

```text
drop_seed = M(same_seed, original) - M(same_seed, masked_condition)
```

The report retains each seed and then reports the mean and sample standard
deviation of the three recorded drops. Computing only an aggregate original
mean minus an aggregate masked mean is insufficient as the stored paired
evidence, even though equal seed sets make the two means algebraically equal.

## 21. Explicit non-goals

Stage A does not:

- modify Python business code, configuration, or tests during this design-only
  task;
- implement or run mixed training;
- train any formal model;
- run masked PublicTest accuracy, macro-F1, loss, confusion matrices, or model
  predictions;
- access PrivateTest or generate its manifest;
- invoke final evaluation;
- modify, delete, or replace the clean baseline or its outputs;
- filter FER2013 source images for natural occlusion;
- change official split meanings or sample IDs;
- use labels, predictions, confidence, or performance to generate a mask;
- add facial landmarks, face alignment, Grad-CAM, attention-guided masks,
  real-object occlusion, reconstruction, another dataset, another backbone,
  Transformers, distillation, or complex statistical testing;
- add profiling, deployment, or publication-release infrastructure;
- claim that mean fill removes the artificial rectangular boundary.

Mean-intensity filling reduces the extreme intensity contrast introduced by a
pure black or pure white mask, but a synthetic rectangular boundary remains.
This is a limitation of the controlled protocol.

## 22. Acceptance criteria

The Stage A design is accepted when it satisfies all of the following:

- the only design-task repository change is this specification;
- existing source-line integer sample IDs and prediction joins are preserved;
- the clean Dataset preprocessing path remains authoritative and unchanged;
- formal spatial size is exactly `112 x 112` and ratios count two-dimensional
  pixels once;
- geometry, half-up conversion, dimensions, coordinate intervals, and integer
  position mappings are exact and mutually consistent;
- Training and evaluation payloads are distinct, canonical, and fully typed;
- every random decision has a separate fixed namespace;
- no mask decision reads a label, prediction, confidence, or metric;
- persistent workers cannot hold stale epoch state because epoch-dependent
  masking occurs after DataLoader batching;
- masked application clones the moved batch and preserves input, shape, dtype,
  device, label, and sample ID;
- the Training mean is derived only from 66,145,536 original Training pixels
  and is bound to the runtime CSV digest;
- the Training mean artifact excludes event time, uses canonical JSON bytes,
  and has a stable cross-platform SHA-256 for the same CSV;
- normalized fill metadata and preprocessing equivalence are specified;
- PublicTest manifest canonical bytes, integer facts, SHA-256, and
  create-or-verify persistence are exact;
- `occlusion-v1` upgrade triggers cover every mask-regeneration rule;
- old clean checkpoints remain loadable and original evaluation does not
  require a mean artifact;
- Stage A has no route to internal split `test`, final evaluation, mixed
  training, or masked research metrics;
- audit images and all data-derived outputs remain outside Git;
- implementation-stage tests cover valid behavior, invalid input, error paths,
  invariants, cross-platform bytes, and locked golden cases;
- Stage B, Stage C, and Stage D responsibilities remain separate;
- all documentation stays inside the FER2013, single-ResNet-18, three-mask,
  three-ratio, three-seed first-version boundary.

Before Stage A implementation begins, this specification is the authoritative
protocol. Any requested change to a locked geometry, hashing, fill, condition,
or persistence rule requires explicit approval and a corresponding version
decision before code changes.
