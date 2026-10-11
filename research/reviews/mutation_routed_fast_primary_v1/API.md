# Fast Present64 primary validation scoring v1

Source: `src/arggnn/mutation_routed_fast_primary_v1.py`.

This small helper computes only primary Present64 NLL and physical-span accuracy at selection epochs. It reads the original sealed H036 FP32 probability raster and preserves the original C4 class order, Present64 columns, bin edges and represented query-bp denominator. Full class, boundary, ECE, dosage and short-tract diagnostics remain the unchanged original scorer and run once for the final selected checkpoint.

## Driver integration

Initialize the original scientific helpers as `initial.science_v1()` does, including the original `fit.np` and `fit.torch` module globals. Construct once per worker:

```python
validation = FastPrimaryValidationV1(
    parents, bindings, reader, helpers, cfg, backend="numpy"
)
```

At each scheduled selection epoch, finish and seal both VAL probability-only NPZ files, then call:

```python
cells, means = validation.score(val_seals)
selection_nll = initial.selection_nll_v1(means)
```

`val_seals` is the original two-element list of `(parent_id, path_sha_record)` pairs. `reader.read(record)` must verify the sealed bytes against the record. Each archive must contain exactly `C4_probabilities`, `positions_bp`, `bin_edges_bp`, `retained_fixed100bp_indexes` and `query_sample_indexes`.

The helper verifies both current VAL seals before constructing either external target. On the first call only, it obtains unchanged physical C4 class-span mass through `fit.ExternalTargetV1(..., "VAL_after_sealed_prediction", ...).target_mass(...)`. It caches only the primary Present64 mass as a readonly FP32 NumPy array. Keep this score-only state separate from observed input banks, observation-cache manifests, model input, TRAIN targets and loss. No TEST parent is retained.

After training and final checkpoint selection, use the selected checkpoint's original sealed predictions:

```python
full_cells, full_means = validation.full_selected_diagnostics(selected_val_seals)
receipt = validation.receipt()
```

This verifies both selected seals, invokes unchanged `h025.score_v1`, and refuses a second full-diagnostic call in the same object. The caller records full diagnostics alongside the primary selection record and handles any existing checkpoint, storage and completion guards.

## Lower-level API

- `mass_cache_v1(parent, all_query_mass, *, target_provenance, expected_present_queries=64)` selects the original Present64 query columns. The full mass must have original `[R,Q,4]` shape and each selected query/bin must sum to the original physical bin width. Positive mass may not underflow to zero during FP32 conversion.
- `PrimaryMassCacheV1` stores readonly `mass[R,64,4]`, original query columns and float64 edges, original represented query-bp denominator, parent ID and target provenance.
- `verify_seal_v1(record, parent, reader)` verifies unchanged support and returns raw complete FP32 probabilities. It does not floor or renormalize them.
- `score_primary_v1(probabilities, cache, *, backend="numpy", device=None, torch_mass=None)` returns primary accuracy, NLL, infinity status and unchanged support metadata. NumPy FP32 is the default. Optional Torch FP32 requires an explicit device and may receive a separately cached score-only FP32 tensor. Only the Torch CPU path has been checked.

Accuracy is the sum of physical mass in each bin's first fixed-order argmax class divided by the original represented query-bp. NLL is the sum of `-mass * log(probability)` over bins, queries and C4 classes divided by the same denominator. Any zero probability with positive true mass produces `nll=None` and `NLL_is_infinite=True`. A zero with zero mass contributes zero. Neither probability floors nor post-hoc renormalization are used.

`FastPrimaryValidationV1.score` returns one primary cell per VAL parent and the FP32 mean of their primary metrics, compatible with `initial.selection_nll_v1`. It deliberately does not generate the other six overlapping strata during selection.

## Hot loop removed from selection

The existing `full_span_score` in `scripts/train_ancestralpaths_biological_v1.py` calculates confusion, Brier, 15 ECE bins and dosage at lines 559–583, followed by per-query tracts, boundaries and short-tract scans at lines 597–636. Existing `h025.score_v1` repeats that work over seven overlapping strata per VAL parent. This helper replaces selection-time diagnostics with the primary vectorized reductions and leaves those full computations for the final selected checkpoint.

## Verification and practical limits

Handmade unequal-bin mixed-span cases verify accuracy 0.65, analytic NLL, first-class ties, zero-probability handling, readonly cache and NumPy/Torch CPU FP32 agreement. Source inspection checks seal ordering and the absence of full diagnostics in selection.

One authorized readonly development comparison used existing aligned epoch-10 VAL000 predictions after both original VAL seals had been verified. Accuracy was 0.47772974 and NLL 1.24686849, agreeing with the stored legacy score within declared absolute tolerances of 1e-6 accuracy and 1e-5 NLL. Original support was [300,1999681] bp over 64 queries, denominator 127960384 query-bp. The single vectorized CPU score took about 0.15 seconds after about 0.33 seconds of initial target-cache preparation; primary mass occupied 20473856 bytes. Full diagnostics were not rerun and no GPU was used.

These single timings do not establish total training throughput or GPU utilization. FP32 reductions can differ slightly from the original float64 arithmetic. Full raw values, artifact hashes, declared tolerances and checks are in `receipt.json`.
