# H060 isolated-hold compatibility diagnosis

H060v5 failed before any new update in either arm. ROOT's recorded topology diagnosis confirms that TRAIN000's first interval, [0, 19), retains all 1,600 samples as 1,600 isolated native roots with no edges or sites. The only failing original _layout predicate is root_count_not_one. ROOT's SHA-authenticated inventory of all 20 TRAIN files checked 98,516 native intervals and found 40 multiple-root holds, all exact sampled-isolate zero-site forests, with no broader unsupported case.

This is an observation-topology compatibility defect. It supplies no model-performance or ancestry-accuracy result. This reviewer read source and ROOT's saved metadata only. ROOT read the sanitized topology files during its bounded diagnostic; this reviewer did not open those scientific payloads.

## Source path

The new H060 warmstart runner imports the original dataset_v1.load_observation and stream_train_v1.SealedObservationProviderV1. It seals all providers before full-parent global preparation. That provider's seal enumerates every block, and its Dataset invokes observations_v1.encode_block, which invokes the single-root _layout. The actual first TRAIN000 interval is therefore incompatible already with initial sealing. The saved failure JSON has no traceback naming the precise call stage, but source ordering reaches this incompatible path first.

Global preparation independently uses the original observation encoder over all native trees. After observation encoding, episode preparation calls _forest_index and _tree_clades, whose contracts require one complete root per hold. Cut preparation separately requires one root per hold and one root component for each full sample row. Switching only the global encoder or only the provider leaves later incompatibilities.

If used, the generic whole-tree validate_inputs_v1 also assumes one root per interval, one sample-row root, and root support sized to T. Preserve this validator unchanged and provide a separate local compatibility validation. The actual full-parent bundle.validate does not call that generic validator. No additional scalar-root assumption was found in the actual neural forward route: node construction marks every actual parent=-1 vertex as a root, sweeps follow the prepared edge schedules, and empty child/downward schedules are structurally supported.

## Saved600 effective sources

The accepted saved600 producer used the isolated-compatible provider, Dataset and observation encoder. Its original source-bound input includes their exact hashes. The efficient continuation invokes the private GPU0 adapter, which replaces the worker factory import with the private-owned cache factory. That delegates through the original RAM cache to the persistent isolated provider. Its reader constructs american_c3_gnn_stream_train_isolated_v2.SealedObservationProviderV1; the accepted trainer also imports that isolated helper.

The exact refs and runtime source chain are in saved600_effective_route.v1.json. The relevant paths are:

- src/arggnn/american_c3_gnn_stream_train_isolated_v2.py
- src/arggnn/american_c3_gnn_dataset_isolated_v2.py
- src/arggnn/american_c3_gnn_observations_isolated_v2.py

The provider seals all blocks, retaining empty-focal and zero-site intervals. The hypothesis that saved600 avoided these intervals by processing only focal or SNP blocks is unsupported. H060 regressed to the older incompatible helper imports.

The existing isolated observation helper accepts all-sampled isolated forests without checking zero sites. The proposed H060 successor is narrower: only the exact all-isolated and zero-site condition proven by ROOT may use its new fallback.

## Minimal successor

Restore the effective saved600 isolated loader/sealer for initial preparation. Use a local per-hold index that preserves a scalar actual root for a connected hold and a tuple of actual sampled roots for an eligible isolated hold. Validate exact node ownership, full samples and complete coverage. Do not invent a virtual root or parent edge.

For each isolated hold, preserve native parent=-1 rows, original node order, sample mapping, times and physical interval. Set depth, height and duration to zero, descendant count to one, edge geometry to zero and branch support to [0,0,0,0,1]. Every node remains in the upward schedule, with no child edges or downward nodes.

An isolated hold has an empty proper-clade set. Process that hold in sequence with all vertices counted as visited. A proper clade on each side of the empty interval forms two episodes, with real loss/gain boundaries. Skipping the interval would change the structural mechanism.

For cuts, traverse every actual root under its original hold owner. Each isolated sample is its own actual representative and singleton group. Retain the original sample-age threshold, root_saturated condition, cut-group episode compression and both directional layouts. Reuse original bundle, episode and CutPlan classes and every neural equation.

A zero-site hold does not imply zero neural output. Empty-site pooling contains empty/span features, and sampled nodes retain age, observed reference mask and root/sample features. Preserve ordinary pooling, two shared sweeps, both full-parent GRU directions, memory processing and exact focal joins on all intervals. Do not prune boundary spans, fill topology forward or reset recurrence.

Both arms must use the same complete preparation. Preserve all 20 TRAIN parents, full 1,600/1,000/10,000 support, 600 additional updates per arm and 1,200 total, optimizer and schedule, accepted backbone, truth barriers, source/cache/resource/thread/zero-CUDA/output/deadline contracts. Frozen baseline and previous H060 sources remain unchanged.

## Review boundary and next step

The exact successor API and tiny structural fixture specification are in repair_design.v1.json. Structural checks should test connected delegation, exact isolates, interrupted episodes, real singleton cuts and rejection of mixed forests, hidden nodes, missing/duplicate samples, cycles, cross-hold edges and isolated site-bearing intervals. No neural numerical, FP64, optimizer-update or training-canary check is required or authorized by this diagnosis.

ROOT separately reported genuine outer tool33950 exit1, direct worker1 and consumed owned drain, with wait SHA143a003853eab76ff297a614a8cd94d4b85eca58cc79f031ba81e3b90a9ba27a. This reviewer made no process or terminal observation. Preserve the failed files, full charge and original clock.

Only this new diagnosis directory was written. No scientific library or guard was imported, no payload read by this reviewer, no numerical work or runtime probe performed, and no production source, process, policy, journal, package or Git state changed. Independently review the author's exact successor and effective source routes before ROOT considers separate fresh operational authority.
