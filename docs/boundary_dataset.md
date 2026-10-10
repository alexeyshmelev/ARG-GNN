# Verified clipped-window boundary dataset

This module provides versioned CPU precomputation and verified loading of the existing `LeafInterfaceBank` schema. It changes no registry, runner or model. Cache geometry is derived only from an observed `GraphSequence`. External ancestry labels and the existing observed nearest-TMRCA probabilities live in a separate parent supervision file. Persistent IDs and source seeds are provenance/matching fields, with no learned ID embeddings.

## Preparation and source binding

`prepare_boundary_dataset(source_manifest_path, output_dir, source_protocol_path=..., windows=..., ...)` returns a `VerifiedBoundaryDataset`. It first binds the exact source manifest and protocol files, checks the embedded protocol against its frozen file, then loads selected independent parents through `arggnn.train_study.load_cases`. That existing loader verifies raw/prepared/metadata hashes, parent directory and split/replicate/seed identity, the prepared original feature prefix and graph/query metadata, and the source reference-adapter configuration.

Preparation additionally binds the prepared file to its declared parent directory, records each entire source-record canonical hash, and checks the metadata master seed against the protocol. Source manifest, protocol, metadata, raw tensor and prepared tensor hashes are retained, together with split, replicate, seeds, master seed, adapter-configuration hash and optional inferred source-parent/inference-seed fields. The raw source tensor contains supervision, so its hash is provenance only and never a numerical bank feature.

The preparation configuration derives `time_column`, `parent_branch_column`, reference columns and time scale from the frozen source reference adapter. Defaults are resistance scale `1e-4`, maximum interior condition `1e12`, full-boundary mean-diagonal normalization, CPU float64 and all prepared leaf features except the declared raw parent-branch column. Optional explicit feature columns must satisfy that same exclusion. Window count defaults to the source protocol. A separately declared count is included in the new preparation hash. Only train and validation splits are supported, with test parents rejected.

Every complete source parent is checked by `precompute_leaf_interfaces` **before any output directory is created**. This checks connected acyclic observed trees, strictly positive decoded branch resistances, root/older-parent consistency, persistent sample leaves, conditioning and observed exhaustive query/reference membership. It includes graphs that have no scoring positions or whose entire fixed tile would later be skipped. An inadmissible inferred graph is an error, never silently dropped, perturbed or extrapolated.

After the parent split and preflight, `simulate.split_fixed_windows` creates each exact tile. It clips edge tiles to genuine observed coverage and skips uncovered or unscored tiles. The module independently checks the source position/label selection and builds a fresh bank and query cache from that clipped sequence. In particular, `interval_weights`, `bag_cqq` and its eigenpairs are computed for the exact clipped tile. Full-parent bag geometry cannot be reused. Original coordinate precision, persistent query order, target-graph mapping and actual/global tile bounds are retained.

## On-disk schema

The schema version is 1. Use a fresh confined directory for every cache version:

```text
cache/
  preparation.json                      # immutable initial incomplete header
  manifest.json                         # written once after full success
  train/rep000/supervision.pt            # labels/baseline/parent metadata
  train/rep000/window000/bank.pt         # observed bank + identity provenance
  train/rep000/window001/bank.pt
  val/rep000/supervision.pt
  ...
```

Bank envelopes have exactly `schema_version`, `bank` and `provenance`. The bank uses `LeafInterfaceBank.as_dict()`, including the exact tile query cache. No label, baseline probability or full parent metadata field is included in the bank envelope. A separate supervision envelope contains parent labels `[P,Q]`, baseline probabilities `[P,Q,K]`, parent positions/query IDs, original retained grid indexes and external parent metadata.

The complete manifest records the source binding, canonical preparation dictionary/hash, implementation/dependency SHA256 map, environment, per-parent source/admissibility/supervision records and per-tile bank hashes/byte counts, identity and metadata. Preparation dependency hashes include this module, the Kron primitive, the source loader/windowing code and observed reference-adapter helpers. Import, preflight and final sealing checks reject implementation changes during a running preparation. Source file hashes are rechecked before sealing. The initial `preparation.json` intentionally remains as an audit record. No partial file is deleted or replaced. A failure leaves an unsealed directory that the loader will not accept. Resume/overwrite is unsupported.

All writes use direct exclusive creation in project-confined fresh paths. Tensor envelopes are serialized to memory before admission, so their exact payload size is known. All tensor loads use `weights_only=True` on CPU. Frozen dataclasses protect schema field assignment. Tensor/metadata mutation after verification remains forbidden by the use convention.

## Verified load and runner API

`load_boundary_dataset(directory, expected_manifest_sha256=..., expected_preparation_sha256=..., expected_source_manifest_sha256=...)` validates the entire selected panel before returning a verified snapshot. Expected hashes are optional for inspection, while a frozen study runner must pass its externally recorded manifest and preparation hashes. The loader checks schema/completion, canonical configuration, current implementation/dependency hashes and source protocol/files, then reuses `train_study.load_cases` for the current source integrity contract.

It compares external supervision exactly with the verified parent. Cached leaf features and packed boundary indices are compared with the observed clipped source leaves. Intervals, positions, query order, target mapping and window bounds must match exactly, including dtype. The loader rederives normalized query/reference blocks and clipped bag geometry from the raw cached response, and verifies eigenbasis orthogonality and response reconstruction. Raw Kron responses are protected by the cache bank hashes and frozen implementation/source bindings. Loading does not repeat the full Schur reduction. Source-independent mathematical reduction checks remain in the Kron primitive tests. No cache hash replaces an externally frozen manifest binding.

The runner-facing API matches `train_boundary.py`:

```python
verified = load_boundary_dataset(
    directory,
    expected_manifest_sha256=frozen_cache_hash,
    expected_preparation_sha256=frozen_preparation_hash,
)
train_cases = load_boundary_cases(verified, "train")
val_cases = load_boundary_cases(verified, "val")
```

`load_boundary_cases(directory, split, expected_manifest_sha256=..., expected_preparation_sha256=...)` also works directly. Passing a verified snapshot avoids repeating source loading and validation for each split. The convenience directory form validates the complete cache on each call.

`BoundaryCase` has `metadata`, `labels`, `baseline_probabilities`, `tiles`, `positions`, `query_node_ids`, `parent_id` and `provenance`. `BoundaryTile` has `bank`, `labels`, `metadata`, `parent_position_indices` and `provenance`. Tile metadata retains `window_index`, actual and global bounds, parent retained-position indices and original grid indexes. The parent identity is source-manifest-hash/split/replicate scoped.

Scatter each tile prediction into `parent_position_indices` to recover the parent's exact retained scoring order. This works even for nonmonotone source positions. Every retained parent coordinate must occur exactly once. Original inferred grid indexes are carried separately from indexes into the already-retained parent tensor. Validation/checkpoint selection remains a parent-level decision owned by the runner. Only the bank is passed to a model. Use `bank.to(device,dtype=model_dtype)` after CPU validation, retaining original coordinate dtype. Device admission and GPU scheduling remain primary-agent responsibilities.

## Storage admission

The module uses a conservative Python `os.scandir`/stat inventory rather than fragile `du` subprocesses. It does not traverse symlink targets or delete files. It counts the larger of logical/allocated bytes per path and may count hard links repeatedly. Non-deletion scan errors abort. A removed entry is skipped because it is absent from the current directory inventory.

Admission enforces the project 500 GB limit, warns at 450 GB, and checks filesystem available bytes. A 1 GB reserved margin covers manifests, directory overhead and routine concurrent project writes. The inventory is refreshed before each parent write group and exact direct-write bytes are accounted between refreshes. Concurrent writers still need their own project guards. This is an admission safeguard, not a distributed allocation lock or process peak-memory measurement.

## CLI

With the existing project environment and a fresh output directory:

```sh
source /workspace-SR003.nfs2/ARG-GNN/scripts/project_env.sh
export CUDA_VISIBLE_DEVICES=''
cd /workspace-SR003.nfs2/ARG-GNN
"$ARGGNN_PYTHON" -m arggnn.boundary_dataset \\
  --source-manifest data/ancient_inferred_common_input_pilot_v3/manifest.json \\
  --source-protocol configs/inferred_common_input_pilot_v3.json \\
  --output data/CHOOSE_A_FRESH_BOUNDARY_CACHE_VERSION \\
  --windows 8 --threads 4 \\
  --expected-source-manifest-sha256 7af4f51f48357ddaf67fe2163988a1f64c567e9cbc0d656d20ebf5dec1c2c4e4 \\
  --expected-source-protocol-sha256 8e4c6f0cbb56baba043c0443f9f1740d549919ccdd32ec39d95d6124345c135f
```

The CLI also supports `--splits`, `--resistance-scale`, `--max-condition` and `--feature-columns`. It emits parent-admissibility, per-tile and final completion records. Completion provides the exact `manifest_sha256` and `preparation_sha256` required by the frozen runner. It creates no CUDA contexts, installs nothing and changes no registry or training protocol.

## Verification record

On 2026-10-10, the module passed **12 bounded CPU tests in 4.86 seconds**. Tests cover oracle and genuinely clipped synthetic observed fixtures, exact tile bag weights, original inferred grid mapping, nonmonotone parent-order scatter, external supervision, weights-only roundtrip, fresh-directory and frozen-hash checks, source/cached tensor corruption, rehashed observed-feature leakage, rehashed full-parent bag reuse, invalid zero branches in an unscored window, incomplete/path-escaped caches and disk admission/warnings. Fixture storage scans are mocked to keep tests bounded. The real scanner is separately exercised on a small confined directory. No original dataset cache, neural training or GPU work was performed by these tests.

A separate read-only audit loaded the existing inferred v3 source panel through `train_study.load_cases`. All **20 parents and 3,684 observed graphs** passed full-parent Kron admissibility, with no decoded zero branch entries. Maximum interior condition was **19,987.970**. This audit used the previously frozen primitive SHA256 `69ead9015c3d6edda5208b3e1807da46bb9fc0e650ba9852da93bd98b9a21f5c`. The primary agent subsequently added analytic single-boundary-component canonicalization, which does not apply to these connected 32-boundary ancestry graphs. Actual cache preparation reruns the preflight against its current dependency snapshot.

Current tested dataset module SHA256: `06f5dfebd40c4a3c39a21c8be7b655e40c422edd1629bdde0b8e75d3c25022a2`. Current Kron dependency SHA256: `09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69`. The source panel audit's raw records are retained below. They establish admissibility only, with no ancestry-performance claim.

```json
[
  {
    "split": "train",
    "replicate": 0,
    "graphs": 210,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 12629.65783239733
  },
  {
    "split": "train",
    "replicate": 1,
    "graphs": 182,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 943.4078765717853
  },
  {
    "split": "train",
    "replicate": 2,
    "graphs": 149,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 19987.970020536297
  },
  {
    "split": "train",
    "replicate": 3,
    "graphs": 202,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 18620.625940095015
  },
  {
    "split": "train",
    "replicate": 4,
    "graphs": 188,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 15990.06525388675
  },
  {
    "split": "train",
    "replicate": 5,
    "graphs": 199,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 1077.0287294020552
  },
  {
    "split": "train",
    "replicate": 6,
    "graphs": 214,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 5382.327510543904
  },
  {
    "split": "train",
    "replicate": 7,
    "graphs": 174,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 775.8586749862103
  },
  {
    "split": "train",
    "replicate": 8,
    "graphs": 170,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 648.6682979870443
  },
  {
    "split": "train",
    "replicate": 9,
    "graphs": 179,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 667.0123762549508
  },
  {
    "split": "train",
    "replicate": 10,
    "graphs": 161,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 10590.059750312006
  },
  {
    "split": "train",
    "replicate": 11,
    "graphs": 172,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 2240.249593436748
  },
  {
    "split": "train",
    "replicate": 12,
    "graphs": 219,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 1229.6215017054556
  },
  {
    "split": "train",
    "replicate": 13,
    "graphs": 196,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 978.2103079878709
  },
  {
    "split": "train",
    "replicate": 14,
    "graphs": 184,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 8764.124403034906
  },
  {
    "split": "train",
    "replicate": 15,
    "graphs": 161,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 1124.5378340778905
  },
  {
    "split": "val",
    "replicate": 0,
    "graphs": 169,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 1056.2204546156581
  },
  {
    "split": "val",
    "replicate": 1,
    "graphs": 177,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 486.5832226686004
  },
  {
    "split": "val",
    "replicate": 2,
    "graphs": 197,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 4067.8081154472416
  },
  {
    "split": "val",
    "replicate": 3,
    "graphs": 181,
    "zero_branch_entries": 0,
    "admissible": true,
    "max_condition": 17836.61956821903
  },
  {
    "parents_checked": 20,
    "graphs_checked": 3684,
    "all_admissible": true
  }
]
```

## Complete-scan CPU precision and consumer-state audit

A read-only probe at 2026-10-10T03:45:10.621741+00:00 verified the primary-agent-created inferred cache `data/boundary_inferred_common_input_pilot_v1`, manifest SHA256 `481e241c3f0aa61f8a62f743e1379f0009fcda795bf315cb1f2f7ad52cf532ed`, preparation SHA256 `c1749a89c2776debadb16a814516a1998057f380fada0c9efe8acafaed284642`. The complete cache had 20 parents and 160 exact tiles. It then evaluated observed **train rep000 (210 graphs)** and **validation rep000 (169 graphs)**, each as a whole observed parent and as the combined eight exact clipped tiles. Prediction/context tensors from tiles were scattered back to original parent coordinate order. No parent label value entered a model or loss.

The probe used the current `BoundaryMemoryModel` with width 32, two graph layers, eight latent channels, four outputs, ordered positive-duration forward/reverse scans and symmetric readout. All three controls used the same seeded learned parameter schema. Reference weights were initialized at seed 42 in the normal float32 construction, promoted to float64, and loaded into the float32 counterpart. Every stored parameter matched **exactly** after promoting the float32 copy, in all 24 comparisons. Original interval/position/window coordinates remained float64 in both variants. The geometry/eigenpairs were precomputed in float64, then cast with the model-facing bank for float32 evaluation.

The label-free loss was the mean of squared logits over every retained parent position, query and output channel. Input gradients are derivatives with respect to the cached `leaf_x` tensors, with observed geometry fixed. They are not derivatives through the CPU observed-tree/reference-statistic preparation. No optimizer update, ancestry-performance score, GPU context or hyperparameter selection occurred.

Nominal rate/damping pairs were `lambda=20, delta=1` and the stiffer `lambda=200, delta=0.1`. For each rate pair and control, the table gives maxima over train/validation and whole-parent/exact-tile modes. Relative infinity error means `max(abs(float32-float64)) / max(max(abs(float64)),1e-12)`, computed within each comparison before taking these maxima. Relative L2 errors and each reference magnitude are retained in the raw record.

| Lambda / delta | Control | Logit max abs | Logit relative infinity | Leaf-gradient max abs | Leaf-gradient relative infinity |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 / 1 | full | 4.401e-7 | 4.061e-7 | 2.459e-10 | 5.526e-7 |
| 20 / 1 | gain_matched_diagonal | 4.247e-7 | 4.216e-7 | 2.882e-10 | 5.396e-7 |
| 20 / 1 | gain_leak_diagonal | 4.452e-7 | 4.165e-7 | 2.845e-10 | 5.490e-7 |
| 200 / 0.1 | full | 4.730e-7 | 4.804e-7 | 2.576e-10 | 5.469e-7 |
| 200 / 0.1 | gain_matched_diagonal | 4.455e-7 | 4.627e-7 | 2.165e-10 | 4.612e-7 |
| 200 / 0.1 | gain_leak_diagonal | 4.501e-7 | 4.612e-7 | 2.333e-10 | 5.083e-7 |

| Lambda / delta | Control | Local context max abs / rel | Forward memory max abs / rel | Backward memory max abs / rel |
| --- | --- | ---: | ---: | ---: |
| 20 / 1 | full | 1.536e-6 / 3.974e-7 | 8.128e-7 / 1.002e-6 | 6.819e-7 / 8.910e-7 |
| 20 / 1 | gain_matched_diagonal | 1.536e-6 / 3.974e-7 | 2.541e-7 / 3.395e-7 | 2.490e-7 / 3.255e-7 |
| 20 / 1 | gain_leak_diagonal | 1.536e-6 / 3.974e-7 | 3.243e-7 / 3.978e-7 | 2.960e-7 / 3.870e-7 |
| 200 / 0.1 | full | 1.536e-6 / 3.974e-7 | 5.709e-7 / 6.471e-7 | 5.289e-7 / 6.045e-7 |
| 200 / 0.1 | gain_matched_diagonal | 1.536e-6 / 3.974e-7 | 3.703e-7 / 4.201e-7 | 3.914e-7 / 4.397e-7 |
| 200 / 0.1 | gain_leak_diagonal | 1.536e-6 / 3.974e-7 | 3.895e-7 / 4.425e-7 | 3.742e-7 / 4.206e-7 |

All outputs, input gradients and local/forward/backward contexts were finite. Across the 24 comparisons, maximum absolute logit error was `4.730e-7`, logit relative infinity error `4.804e-7`, input-gradient absolute error `2.882e-10`, input-gradient relative infinity error `5.526e-7`, and context absolute error `1.536e-6`. These are CPU observations for initialized weights and selected inputs. They do not establish trained-weight, different-hardware or GPU precision guarantees.

The consumer's validation rejected four in-memory corrupted bank states: missing query cache, changed query-ID order, a noncontiguous interval partition and a feature/matrix dtype inconsistent with model parameters. Dataset verification additionally enforces exact observed-source feature/coordinate identity and tile-specific bag reconstruction, which are intentionally not repeated inside every model recurrence. Frozen verified tensors must remain mutation-free after CPU validation.

The final combined primitive/dataset suite passed **33 CPU tests in 6.03 seconds**, including the primary agent's component-wise analytic single-boundary response regressions. A fresh confined pytest directory was used, with no temporary-file deletion or cache overwrite workflow.

Evaluated model SHA256: `a056cd10d1c848d7dabdc1daf646548876fd90e8f489667c46a374ac28cd11dc`. Dataset module SHA256: `06f5dfebd40c4a3c39a21c8be7b655e40c422edd1629bdde0b8e75d3c25022a2`. Primitive SHA256: `09ec6286c185e7aa2b13d76dcbe8fea24b3a1c785a6619ac9bc5464e0de66d69`. PyTorch 2.6.0+cu124, one CPU thread. The complete precision/state-validation record follows:

The complete precision record is retained in [boundary_precision_audit_v1.json](../research/reviews/boundary_precision_audit_v1.json).

