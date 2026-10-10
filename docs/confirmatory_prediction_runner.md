# Frozen observed-only ID prediction cells

This workflow implements one frozen method × training seed × admitted ID parent prediction cell. It accepts only a real validated confirmatory provenance lock, the complete sealed label-blind admission ledger, and one safe observed graph NPZ. It has no target reader, training-loader call, optimizer, gradient computation, temperature fitting, simulation or preparation action. No actual confirmatory lock, input, registry, prediction or configuration is supplied by this implementation.

## Public interfaces

```python
from arggnn.confirmatory_prediction_runner import (
    prediction_schema_values, validate_observed_npz,
    admit_prediction, predict_confirmatory,
)
```

`prediction_schema_values()` returns prospective source-derived metadata and current code pins only. `validate_observed_npz(content, identity, lock_sha256)` validates raw observed bytes and returns a CPU `GraphSequence` with primary grid coordinates reconstructed from the external sealed identity. Custodian validation may call this pure observed validator without model execution.

`admit_prediction(lock_path, expected_lock_sha256, ledger_path, expected_ledger_sha256, method_id, training_seed, parent_id, project_root=ROOT)` first calls the public published-lock reader and full check. It then admits the exact current source-pinned analysis schema, locked method and seed bundle, registry and sealed ledger. Admission must reach the first 160 admitted parents or exhaust all 192 attempts before model scoring. It never opens primary-label or census references, including existing target files. Those paths are metadata only, used to reject observed source substitutions.

The requested observed identity must match the frozen namespace, parent ID, ancestry/mutation/inference seeds, all 64 ordered original query IDs, 32 diploid pair slots, original clipped 128/4096 midpoint grids and eight fixed physical tile edges. At least 120 primary coordinates and 95% physical coverage are required. The selected protocol and observed source are captured by hash/stat. The returned admission object checks in-memory metadata, source bytes, full lock provenance and imported implementation drift before execution and every tile. Full lock byte hashes are verified at admission and publication; between tiles the public lock reader performs its captured-stat/source checks (`full=False`) to avoid repeatedly rereading all development checkpoints.

`predict_confirmatory(admission, output_directory, device="cpu")` requires a fresh in-project directory. CPU mode is explicitly recorded as CPU validation only, with no CUDA execution or admission claim. Actual root GPU scheduling uses device="cuda" through the project launcher. GPU mode requires process title `AI Auto Research Pipeline Test`, strict deterministic policy before context creation, inherited held project lease on physical GPU 2–5, exactly one visible lease UUID and matching runtime UUID. CUDA software/build and device name/compute capability must match the numerical runtime captured in the lock. Actual physical assignment/UUID are recorded. Title, lease and numerical policy are rechecked during prediction; root remains responsible for scheduling, two-GPU limit and NVTOP verification.

## Observed release contract

The identity's `source_artifacts` contains exactly one `{path,sha256}` NPZ. Loading uses `allow_pickle=False`, an exact ZIP member allowlist and bounded compressed/uncompressed size. Its exact arrays are:

| Array | Contract |
|---|---|
| metadata_json | Scalar Unicode JSON with duplicate-key/nonfinite rejection |
| x | Finite original `[N,8]` float32/float64 observed features |
| edge_index | `[2,E]` int64 observed tree edges |
| node_graph, node_ids | `[N]` int64 graph index and stable leaf ID or internal -1 |
| clades | `[N,16]` float32/float64 exact observed-reference landmark descendants |
| intervals | `[G,2]` float64 complete normalized observed physical partition |
| query_node_ids | `[64]` int64 in original sealed query order |
| window_bounds | `[2]` float64 observed coverage, global bp/1e6 |

The header has exactly schema_version=1, kind="sealed_confirmatory_observed_graph_sequence", lock_sha256, namespace, parent_id, feature_names, time_scale=10000.0 and truth_free_observed_inputs=true. Feature names are `reference_ANA`, `reference_CHG`, `reference_WHG`, `reference_EHG`, `is_query_leaf`, `is_leaf`, `log_time`, `log_parent_branch`. Query/internal source labels and extra fields, probabilities, supervision, stored readout coordinates, arbitrary feature channels or object arrays are rejected.

Validation reconstructs all 88 stable terminal sample leaves per tree, observed reference roles (six haplotypes per source), query/leaf flags, prescribed sampling dates, connected acyclic monotone-age topology, age-derived branch channel and exact landmark clades. Landmarks reproduce the existing packer: within each category sort six reference IDs, choose `np.linspace(0,5,4,dtype=int)` indices [0,1,3,5], then use class-major descendant indicators. IDs provide correspondence only. This validates the supplied observed representation; inference provenance and custodian isolation remain responsibilities of the source release workflow. Hash binding is not authentication or file-permission enforcement.

## Frozen factories and independent grids

The runner uses the existing pure selected primary factory interfaces and secondary `make_control`. Constructors run inside `torch.device("cpu")`; no training data loader is called. Actual factory parameter count must equal the frozen choice. Checkpoints load with `weights_only=True`/map_location="cpu". Exact state keys/shapes/dtypes, finite tensors and unchanged fixed buffers are required before strict load. Only the zero-parameter, zero-state deterministic reference smoother may use checkpoint=null. The model stays in eval mode with all parameter gradients disabled, and every forward uses `torch.inference_mode()`.

Implemented primary families are candidate, anchor GRU/transformer, no-fit reference smoother, all-record GRU/transformer, canonical-tree GRU, full-tree ARGMix, GST CA span and GraphSSM ZOH. Implemented controls are gain-matched diagonal, static local, matched scalar-age full and noncontextual. The lock fixes which conditional controls are actually eligible; no method substitution occurs.

Reference features are recomputed from raw F8 on CPU. Boundary families use exact plain leaf/query/reference geometry with raw branch column excluded (F47), full-parent geometry admission before clipping, fixed resistance 1e-4, and age branches 350/1400 with strength 3 when the frozen geometry requires them. Anchors remain 32 per physical tile with three scales. Raw-tree families receive verified F48. Canonical GRU canonicalizes the observed complete parent before fresh augmentation and fixed tiling. Every retained coordinate appears in exactly one tile, and original query/coordinate identity is checked before transfer.

Primary128 preprocessing and forwards finish, and their raw output is copied, before dense4096 preprocessing or forwards begin. Dense readouts never enter a primary forward. There is no combined primary/dense batch or resampled coordinate replacement. CPU tests also repeat primary predictions after the dense pass and require exact equality for every primary family.

## Publications and terminal records

One safe `prediction.npz` per successful cell contains metadata_json, primary_logits, dense_logits, primary_positions_bp, dense_positions_bp and query_ids. Logits retain their original finite float32/float64 values with shape `[retained_positions,64,4]`; positions remain exact float64 physical bp. Class order is ANA/CHG/WHG/EHG. No softmax, temperature scaling, clipping or label join changes stored raw logits.

The exact prediction header matches the source-pinned joined-analysis schema: lock, namespace, parent, method, seed, configuration, family, protocol and checkpoint references, observed identity/grid hashes, class order and no_gradients_or_optimization=true. `startup.json` records a started cell. `terminal.json` is an exclusive hashed success or failure receipt with source/input/runtime pins, CPU/CUDA scope and the output NPZ hash. Partial startup stages are preserved when title/policy/lease/runtime checks fail. Failure receipts never substitute predictions; any NPZ published before a later failure is recorded as an orphan, and the cell remains failed. Existing files are never overwritten. Publication uses an owned in-project tempfile, source recheck, exclusive hard link and pending-file cleanup.

A complete sealed roster is produced separately by the custodian workflow, not by one-cell prediction. It must account for every locked method/seed/admitted-parent cell. Joined analysis admits that seal before labels and refuses unresolved cells. Terminal failures block positive confirmation and remain visible as descriptive sensitivity. CPU-validation receipts must remain labeled as such in operational review.

## CLI and CPU evidence

`--help` is safe and reads no dataset. Actual execution requires explicit existing reviewed artifacts and a fresh output. The standalone script is `scripts/predict_confirmatory.py`; the GPU launcher entry point is `arggnn.confirmatory_prediction_runner` with `--device cuda`. No executable example with invented lock hashes or confirmatory paths is included.

Tests use handmade changing Q64/R24 trees, exact physical grids, real saved synthetic factory states and explicitly fake published-lock admission in isolated unit projects. They check actual forwards for all ten families on both separate grids, all conditional controls, exact primary replay, no-fit smoother, safe observed arrays, metadata/source/ledger rejection, target-read exclusion, strict checkpoint/buffer checks, exclusive output, terminal failure preservation and CUDA guard ordering without CUDA inspection/context. These tests do not constitute real provenance admission, CUDA execution or scientific confirmation.
