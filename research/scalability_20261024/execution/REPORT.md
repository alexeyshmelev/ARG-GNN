# TreeLoom exact execution scalability: measured slice and verified prototype

**S01: measured representative slice. S02: isolated layout prototype verified. Integration remains open.** No live source or job was modified, no signal or GPU launch was issued, and no simulation, observation or scientific training data was changed.

## Actual input and bounded measurement

The first existing American TRAIN parent, argformer_american_train_000, was selected from the frozen attempt3 job refs. Its authenticated observed tree sequence has 106,762 global nodes, 599,269 edges, 77,472 marginal trees, 227,474 sites and 1,600 samples; the file is 34.7 MB. Only the first 128 contiguous native trees were expanded for this diagnostic: 317,897 active-node occurrences, 1,600–2,506 nodes per tree, and 681 original observed SNPs. The process maximum RSS was approximately 0.87 GiB. No full extra parent build or truth read was performed.

The current static preparation materializes a complete list of per-tree native arrays and then expands full feature banks. The current bounded executor rebuilds CPU height/depth and block index layouts on every parent update. These are source-established repeated operations; their share of whole-update wall time is unmeasured.

The custom dense rank/stamp workspace for this American parent is only 1.71 MB. Exact sparse local remapping preserved every native field, but took 0.0847 s versus 0.0783 s for dense remapping on the same 128 trees. **Sparse remapping is rejected as an American speed optimization.** It also retains tskit's own dense cursor arrays, so it would not eliminate all global-node-sized state. European AP has a different global-node scale and is outside this timing claim.

## Implemented exact improvement

layout_plan.py builds an immutable LayoutPlan once and exposes the same eleven-field block tuples used by the current executor: hold/node ranges, local parent, upward levels, upward children, upward parent ranks, downward levels and episode-member ranges. It preserves every original index and ascending row/reduction order. Local index arrays use int32 immutable bytes backing; the current transfer adapter's explicit torch.long conversion restores the same GPU index type. Bounds reject a block that cannot be represented.

The plan caches structural recipes only. Learned states, gates, SNP encodings, gradients and optimizer state are never cached across updates. A content digest checks a newly restored immutable bank once; a bound plan then uses the frozen bundle's tensor pointer/version/shape contract for repeated access. Fast binding requires the same trusted immutable tensor/alias discipline as the frozen bundle; mutable NumPy inputs are outside the supported fast-binding contract.

On the same 128-tree bank, CPU-only, one thread and five cold-build repetitions:

| Operation | Median |
|---|---:|
| Existing layout rebuild | 25.50 ms |
| Cold immutable plan build, including content digest | 32.91 ms |
| Repeated access to an already bound plan, 100 repetitions | 56.92 microseconds |
| Existing layout index arrays | 12.67 MB |
| Immutable plan index arrays | 6.34 MB |

Cold construction is slower and must be amortized over repeated uses. Build-plus-lookup arithmetic for 30 uses is about 34.6 ms versus 765.0 ms of repeated existing builds, **excluding one-time restored-bank binding, persistence and integration costs**. This is a layout-stage calculation, not a measured whole-training speedup. A retained or persisted plan cache is required; this prototype alone does not eliminate the expanded feature bank or repeated compact-bank restoration.

## Verified numerical and structural contract

All layout values and row order matched the current constructor on the real slice for block sizes 1, 3, 4, 17, 64, 128 and 256, including partial last blocks and nonempty member-range probes. Immutable backing, stale tensor-version rejection, changed-content rejection and exact restored-content binding passed.

The actual frozen first-sweep operator was tested on four real observed trees, 9,031 nodes and hidden width 32, with independent seeded CPU base/gate probes. Output, base/gate input-gradient and all ten connected parameter-tensor gradients were bitwise identical: maximum absolute error zero. Disconnected parameter gradients also matched. This establishes layout-only compatibility; it does not establish site-encoder, history, full-parent or GPU parity.

## Subtree/state reuse assessment

Topology-only ordered subtree shapes had 2,199 unique signatures among 317,897 node occurrences. That count ignores sample and feature dependencies and is not a valid state-cache key.

Current first-sweep leaf state depends on the observed SNP source slice, donor/panel features, pooling reduction chunks, hold span, persistent sample column, reference label and age. Internal base depends on node age/root/sample status; edge gates depend on four geometry and five branch-support fields. Outside state additionally depends on parent/sibling context; the second sweep adds clade/episode and stable-leaf history. Parameter updates also invalidate learned states.

Under the documented strict_source_identity_lower_bound policy, the measured slice had 317,897 distinct upward expression signatures: **zero guaranteed first-sweep upward-state reuse was found**. This is a conservative guaranteed-sharing lower bound, not proof that no other exact sharing is possible. Algebraic factorization, common parameter projections or broader equivalent-input analysis may expose other reuse. Topology-only deduplication would not preserve the frozen model.

The observed-only NPZ/JSON packet was supplied to the model agent for a separately scoped cavity-DAG proof. It preserves native IDs, node offsets, all current bank fields, original level/child order, raw feature dependencies and global 256-site source chunk metadata. Boundary notes explicitly limit site-encoder equivalence claims.

## Open integration work

Keep the frozen run unchanged. A future reviewed successor can attach the reusable plan cache to compact-parent storage and measure the full preparation/restore/layout/transfer/forward/backward timeline. A compact sequential topology/feature pipeline would need to preserve the global clade episodes, cuts, complete genomic history and exact site reduction boundaries; it is a design task here, not a verified implementation. New scientific architecture or approximation should be justified by those measurements, without changing the simulation core or observed datasets.

Evidence: measurement.json, test_results.json, representative_slice.json/npz. Source and input hashes are recorded in those receipts. Reproduction commands use PYTHONPATH=src:. and the existing .runtime/env/bin/python3.11 runtime. All outputs are confined to this execution directory.
