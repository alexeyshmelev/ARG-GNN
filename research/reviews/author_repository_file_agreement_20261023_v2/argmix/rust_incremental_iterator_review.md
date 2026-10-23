# ARGMix incremental engine and iterator source review

All five assigned source files were read completely: 4,666 lines and 166,642 bytes at commit `c2481248eb0de5fda659d68f59c51e0e27becd20`. Exact file hashes, original line coverage and source sections are in the accompanying JSON. Both binary-lift implementations are covered by the companion Rust review and were not repeated here.

The current shard inference pipeline validates `backend_name=fast` and constructs `FastSubgraphEngine`. Findings in the retained standalone and incremental alternatives must be attributed to those routes. No author module, test, model, compiler or scientific run was executed. Only these two review reports were written.

## Complete file coverage

| Source file | Lines | Purpose |
| --- | ---: | --- |
| `src/rust/ancpath_incremental/src/engine.rs` | 1820 | Incremental per-tree LCA cache, persistent subgraph rows, reference selection and streaming batches |
| `src/rust/ancpath_incremental/src/benchmark.rs` | 98 | Python-exposed incremental extraction throughput benchmark wrapper |
| `src/rust/ancpath_incremental/src/sample_batch.rs` | 231 | Python incremental batch iterator with decoded and flat output forms |
| `src/rust/ancpath/src/sample_iterator.rs` | 1453 | Standalone Python sample and batch iterators with optional interval merging |
| `src/rust/ancpath/src/shard_pool_iterator.rs` | 1064 | Bounded shard worker pool feeding TorchScript inference and optional TSV/HDF5 sinks |

## Semantic dispositions

### RI01: ShardPool inference uses the fast engine, not the standalone or incremental iterators.

Status: `resolved_route_identity`.

Only findings reachable through FastSubgraphEngine affect that inference route. Alternative iterator defects must not be presented as observed fast-route errors.

Source: `src/rust/ancpath/src/shard_pool_iterator.rs` 331-371, 701-708.

### RI02: Incremental linear distance caps before u16 conversion; standalone float conversion saturates. These do not share the fast integer narrowing behavior.

Status: `confirmed_source_semantic_difference`.

With a positive linear distance of 65536 and cap 1000, incremental yields 1000; standalone positive f64-to-u16 conversion saturates at 65535 before min and also yields 1000. The companion review reports that fast u64-to-u16 narrowing yields zero at this boundary.

Do not claim backend equivalence for large distances. The fast engine narrowing observation is owned by the companion Rust review; the incremental source test was read but not run.

Source: `src/rust/ancpath_incremental/src/engine.rs` 1492-1525, 1793-1804; `src/rust/ancpath/src/sample_iterator.rs` 684-709.

### RI03: Standalone reference ranking changes with log_scale_tmrca; incremental ranking remains linear.

Status: `confirmed_source_semantic_difference`.

For reference indices [0, 1] with raw distances [2, 1], standalone log keys are [1, 1] after truncation, so the index tie-break selects reference 0. Linear ranking selects reference 1. This is a hand-worked source example, not an executed experiment.

Log-mode datasets from the standalone iterator are not automatically identical to fast or incremental selections. Matrix values may tie while population identities differ.

Source: `src/rust/ancpath/src/sample_iterator.rs` 46-56, 504-522, 684-709; `src/rust/ancpath_incremental/src/engine.rs` 501-520, 1492-1497.

### RI04: The incremental row-refresh gate can miss a newly tied reference with a preferred index.

Status: `static_counterexample_not_executed`.

Use one requested reference, drop_top_m_nearest=0 and an untouched prediction sample. Old distances [10, 5] select reference 1. Move only reference 0 so new distances are [5, 5]. Deterministic distance/index order should select reference 0. The gate tests new_distance < kth_distance or previous membership: 5 < 5 is false and reference 0 was not selected, so the row can remain stale.

An inactive incremental alternative needs a reference-selection invalidation fix and a dedicated boundary test before it can be claimed equivalent to full rebuilding. No source fix or test execution was performed.

Source: `src/rust/ancpath_incremental/src/engine.rs` 523-537, 1164-1185, 1688-1699.

### RI05: With nearest-reference dropping, moving a dropped reference outward can change the selected set while the incremental refresh gate skips the row.

Status: `static_counterexample_not_executed`.

Use one requested reference, drop_top_m_nearest=1 and an untouched prediction sample. Old distances [1, 2, 3] drop reference 0 and retain reference 1. Move only reference 0 to distance 4, producing [4, 2, 3]. Full selection should now drop reference 1 and retain reference 2. The changed reference 0 is not in the retained set and its new distance 4 is not less than the retained kth distance 2, so the gate can retain reference 1.

Tracking only retained membership and the new kth-distance threshold is insufficient for nonzero drop augmentation. This is separate from source tests that check static selection; those do not exercise this transition.

Source: `src/rust/ancpath_incremental/src/engine.rs` 501-537, 1164-1185, 1702-1749.

### RI06: Standalone Python iterator constructors support ancestor positions. The shard helper rejects ancestors only when merging is enabled.

Status: `resolved_low_level_capability_boundary`.

Upper archive or YAML entrypoint rejection of model.num_ancs is a route-specific restriction; it does not erase the retained low-level implementation. Main scientific configuration must identify the actual entrypoint.

Source: `src/rust/ancpath/src/sample_iterator.rs` 260-370, 537-585, 648-662, 1264-1302, 1344-1382; `src/rust/ancpath/src/shard_pool_iterator.rs` 303-315, 331-345, 649-710; `src/rust/ancpath/src/subgraph_merge_pipeline.rs` 15-38.

### RI07: Inference labels are absent in these iterator outputs; query population metadata is a separate field.

Status: `resolved_output_contract`.

These modules do not recover true ancestry or demonstrate ancestry accuracy. Training supervision must be traced through the separate truth/archive route.

Source: `src/rust/ancpath_incremental/src/sample_batch.rs` 178-231; `src/rust/ancpath/src/sample_iterator.rs` 795-803, 862-876, 1304-1319.

### RI08: Standalone nonmerged sample order is not FIFO; pooled max_count is per shard and shard scheduling can interleave rows.

Status: `resolved_order_and_count_contract`.

Match records with explicit shard, node and interval keys. Do not assume output position or interpret max_count as one global cap.

Source: `src/rust/ancpath/src/sample_iterator.rs` 432-446; `src/rust/ancpath/src/shard_pool_iterator.rs` 202-224, 549-578, 650-655, 878-905, 926-932.

### RI09: IncrementalBenchmarkRunner reports extraction throughput and progress only.

Status: `resolved_benchmark_boundary`.

Rows per second and tree counts are not training, prediction or scoring results. A tree is counted when prepared, so the row_capped_partial predicate alone does not prove every row of the final prepared tree was emitted.

Source: `src/rust/ancpath_incremental/src/benchmark.rs` 21-98; `src/rust/ancpath_incremental/src/engine.rs` 864-891, 1028-1051.

### RI10: Standalone traversal calls next after tree_iterator_at_index, while incremental traversal first consumes the positioned tree.

Status: `dependency_semantics_unresolved_by_this_subset`.

The dependency constructor and StreamingIterator contract must settle whether a tree is skipped. No off-by-one outcome is asserted here, and no dependency or author implementation was executed. Cargo.lock identifies tskit 0.15.0-alpha.3; the current subset did not inspect that crate implementation.

Source: `src/rust/ancpath/src/sample_iterator.rs` 759-774, 834-844; `src/rust/ancpath_incremental/src/engine.rs` 674-680, 953-972, 1807-1818.

## Interpretation boundary

The two incremental refresh examples are hand-worked transitions derived from the branch conditions. They were not run against a tree sequence. The source contains tests for cache updates, static nearest-reference dropping and cap-before-cast behavior, but source test presence is not a passing-test claim. The standalone positioned-iterator difference remains explicitly unresolved in this supplement until the exact dependency contract is inspected.

The incremental benchmark reports extraction throughput. Standalone and incremental inference rows carry no ancestry truth label. Shared-data supervision, checkpoint choice, architecture attribution and scientific accuracy remain responsibilities of the primary paper agreement matrix.
