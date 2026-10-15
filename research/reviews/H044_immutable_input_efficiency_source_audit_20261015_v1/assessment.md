# H044 immutable-input RAM reuse: source-only decision

2026-10-15. No implementation, running-worker changes, execution, target access, process inspection, admission or timing measurement.

**The source exposes substantial repeated invariant CPU work that can be reused exactly.** RAM-only caching of the original localized and expanded tree-pooled FP32 feature blocks plus original FP32 TRAIN target masses is a credible future execution improvement. It does not change the learned model or require detaching its gradients. This is a source opportunity, not a diagnosis of the observed eight-minute passes or a speedup claim. A single zero-utilization sample cannot establish the bottleneck. Keep the active H044 worker and frozen run unchanged. A new accuracy run is not recommended on this audit alone.

## Where the work repeats

- `marker_resolution_readout_v1.py:80-111` computes query genotypes, observed categorical evidence, FP64 evidence prefixes, FP32 count prefixes, physical box differences, averages/log-counts and final FP32 features. `forward:320-330` calls this for each query block in each arm's forward. The tree-pooled arm recomputes the raw features and then applies the original physical-width-weighted native-hold pooling. The MarkerCache retains evidence/bindings, not these final features.
- With the fixed query_block=64 and Q=686 there are 11 blocks, ending with 46 queries. Thirty passes, 32 parents and two arms imply 21,120 raw block constructions and 10,560 pooling calls in TRAIN. Computing raw and pooled blocks once per parent would require 352 calls of each kind. These are source-derived invocation counts. Head checkpointing repeats only `_emit`; it does not multiply these feature counts.
- `train_marker_resolution_readout_v1.py:270` calls `original_mass_v1:197-209` once per parent per pass, before the two-arm loop. Thus targets are already shared between arms, but rebuilt 960 times across passes. `FullPanelExternalTargetV1:181-198` rereads/decompresses the authorized same-parent truth archive, verifies query identity, runs physical interval intersections, allocates a complete FP64 [P,686,4] mass and scatters spans with `np.add.at`. H044 then performs its original FP32 C-order conversion. Retaining that exact final tensor reduces constructions to 32, with reuse across both arms and all passes.
- `fit.metric_helpers_v1()` also dynamically loads the same helper module on every mass construction. A single phase-local helper instance would avoid that repeated source-module construction. The substantive saving candidate is the mass construction, not a demonstrated cost of the import.

Marker evidence/binding reconstruction can additionally recur on LRU eviction, but cache capacity and actual misses were not read. This audit does not assert that those misses occur. Bank/flag/mass CPU-to-GPU transfers and every learned backbone/head computation remain unchanged by the proposed CPU caches.

## Named memory forecast

Worst requested P=20,000, Q=686, C=4, F=6, FP32=4 bytes. GB below means decimal 10^9 bytes.

| Retained item | GB |
| --- | ---: |
| One parent's localized features, P*Q*C*F*4 | 1.31712 |
| One parent's localized plus expanded pooled features | 2.63424 |
| All 32 TRAIN feature pairs | 84.29568 |
| All 32 complete original-grid FP32 masses | 7.02464 |
| Above caches plus a <=34 GB CPU bank allowance | 125.32032 |
| All 8 VAL feature pairs, if retained together | 21.07392 |
| Existing all-96 VAL probability matrices | 21.07392 |

**Yes, the TRAIN named tensors fit comfortably within a fresh 256 GB planning envelope.** Even adding a conservative 4.913 GB marker-payload allowance outside the bank, about 1.3 GB for all-40 coordinate sidecars, the existing worst-S=2,000,000/query-block construction allowance of 21.135 GB, and one simultaneous FP64-plus-FP32 mass allowance of 0.659 GB gives approximately 153.4 GB. Some borrowed storage is double-counted in that conservative sum. It leaves about 102 GB for remaining host allocations. The original resource erratum correctly records 96 exported H042 state channels, not 64.

This is not a proof of actual peak RSS or available host RAM. Truth-archive decode buffers/intersection cardinality, native libraries, allocator retention/fragmentation and concurrent work are not fully bounded by these named tensors. In particular, if the supplied statement 'worker RSS 256 GB' describes an actual current measurement rather than a limit, adding 91.3 GB of retained caches cannot be claimed to keep that process below 256 GB. An existing 256 GB RSS would need an allocation explanation or a larger operational envelope. No current-state measurements were made here.

There is no reason to retain TRAIN caches in an evaluation-only phase. VAL predictions can be generated one parent at a time with just that parent's 2.634 GB feature pair retained across its 12 arm/checkpoint forwards. All eight observation banks and all 96 sealed probabilities follow the existing protocol. If all eight feature pairs are retained instead, their cost is the table's 21.074 GB. No VAL target is needed to compute either feature cache.

## Exact semantics and smallest safe later change

1. Add a phase-local, explicitly bound derived-feature cache keyed by parent, original observation/coordinate/query recipe and fixed query range. Store the eleven original **contiguous [P,B,4,6] blocks**, separately for localized and tree-pooled arms. For a cache miss call the unchanged original `features_for_queries(start,end)` once, then the unchanged `pool_within_trees(raw)` once. Retain their exact FP32 outputs. Concatenating a new [P,686,4,6] tensor and later slicing queries would change strides/layout, so keeping the original blocks is the smaller numerical change. Preserve original query/center chunk order, devices, casts, pooling arithmetic and decoder calls. Build observation caches after their original identity validation, preferably before the TRAIN truth gate. Cache immutable derived observations only.
2. Add a distinct trainer-local `TRAIN_mass_by_parent` dictionary. The first miss must still occur after the existing complete observation barrier and `reader.truth_gate=True` at line 262, using the same phase-restricted reader, TRAIN binding and unchanged `original_mass_v1`. Keep only its returned contiguous FP32 CPU mass; release FP64 construction and truth object as now. Future hits return that same bound tensor. Do not put targets into marker/model bundles or share this dictionary with VAL scoring. The current per-construction target cap should remain, with a separate explicit retained-cache byte account in a later root-owned source/config version.
3. Change only forward's feature retrieval site to obtain the matching cached block; leave `backbone_state_v1`, `_emit`, loss, optimizers, arm order, RNG/update schedule, head checkpointing and all source model parameters untouched. Use a typed immutable cache with stamps covering **every cached block**, plus binding checks and retained-byte accounting. Existing `_tensor_stamps_v1` does not automatically traverse a tuple of cached blocks in MarkerEvidenceBankV1, so merely appending a tuple field would not protect it. Do not monkeypatch the active worker or silently reuse old source hashes/caps for a changed program.

No learned tensor enters `features_for_queries` or pooling. Ordinary CPU tensors built in normal mode or `torch.no_grad()` can be saved by downstream autograd for decoder-weight gradients. **Do not create the cache under `torch.inference_mode()`** and feed those inference tensors directly into a training decoder, which may need to save inputs for backward. Do not cache or detach encoded tree states, GRU outputs, logits, loss, or any learned parameter result. Every learned path remains recomputed for each arm/update as now. Target masses already have no gradient.

Reusing the original output bits avoids changing feature formulas, FP64 accumulation or FP32 conversion. Full bitwise training-trajectory equivalence is still unverified: CPU threading/reduction determinism, device/kernel behavior and alias discipline remain conditions. Mutation stamps cannot catch unauthorized NumPy/.data alias writes, so those remain prohibited. Caching also increases memory traffic and leaves PCIe transfers and GPU work in place, so call-count reductions are not throughput forecasts.

The smallest safe implementation is therefore a later explicitly versioned immutable-input cache at these two retrieval sites, with the original arithmetic as the cache-miss path. It should be considered only when H044's completed result motivates continued use. No broad new validation campaign or new accuracy experiment follows from this source audit.
