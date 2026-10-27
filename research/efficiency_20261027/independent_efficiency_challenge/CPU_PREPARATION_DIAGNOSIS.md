# Targeted diagnosis of preparation after encoding

**Compact packing is not the dominant tail. Explicit pool-lock serialization is not present around preparation. Python episode construction is the strongest source-supported serialization candidate, but existing telemetry does not isolate its duration from global materialization and cut preparation.** No implementation change is justified solely by the current main-process CPU percentage.

This follow-up read only source, configuration, timing JSON and existing process-family telemetry. No scientific arrays, process attachment, profiling, tests, GPU/training launches, Git operations or live-source changes were performed. Only this report was written.

## Newly available measured result

The adaptive arm's `timed_preparation.json` appeared during this audit. The complete prepare-and-pack pair took **1238.356141s**, versus **1560.893055s** serially: **322.537s saved, 20.6636% reduction, 1.26046x throughput**. These are one fixed-order pair's timings; final signature/completion acceptance was not present at the initial read and remains a separate requirement.

| Parent | Serial preparation after load | Adaptive preparation after load | Serial prepared-to-ready | Adaptive prepared-to-ready |
|---|---:|---:|---:|---:|
| 001 | 758.516s | 1206.412s | 22.611s | 25.509s |
| 007 | 754.869s | 1211.633s | 23.355s | 25.547s |

Adaptive parent001 was prepared at1207.871s and ready at1233.380s; parent007 was prepared at1212.809s and ready at1238.356s. The approximately20.57s overlap between these prepared-to-ready intervals directly contradicts a shared mutex serializing the entire packing operation. This interval includes packing, cleanup/GC and readiness bookkeeping, so25.5s is an upper bound for packing alone. Even eliminating the final approximately30.5s from earliest prepared to last ready would save at most2.46% of pair makespan.

The existing adaptive `family_samples.jsonl` records8 encoder children at90.678s,4 at261.665s and0 at265.369s on the sampler's clock. The sampler begins before the preparation timing clock, so these timestamps must not be subtracted from preparation timestamps as exact stage durations. They nevertheless establish that the lengthy remaining work occurs after child exit. The serial arm similarly loses its children long before each parent is prepared. Family samples contain memory and process identities, **not function labels, thread CPU times or GIL ownership**.

## What the actual call chain establishes

1. `process_observations.encode_block` joins encoder children at the end of its generator/ProcessPoolExecutor, then still concatenates complete-parent arrays, constructs geometry, builds stable level buckets with sorts, computes donor features and materializes tensors (lines97-161). Therefore “no encoder children” does not mean `encode_block` has returned. This is parent-thread global materialization, not compact packing.
2. `static_preparation_v2.py:87-112` next builds global clade episodes, builds the cut plan, and validates the complete bundle. Compact packing starts only after all of these return.
3. The two parent producers are threads (`adaptive_prefetch_v2.py:270`). `_produce` releases the pool lock before `prepare` and `pack` (lines295-321); `EncoderCredits.owner` also releases its ledger lock before yielding (lines90-99). Neither lock wraps the lengthy work. Readiness commits are brief locked operations after packing.
4. `episodes.py:51-76` processes every hold and proper clade through Python dictionaries, Python integer keys, membership lookups, and scalar NumPy writes/increments. `kernels.py:129` converts each compiled clade mask into a Python big integer and dictionary entry. These operations require the CPython GIL and two producer threads cannot execute their Python bytecode simultaneously. This is a concrete serialization opportunity in source, not proof that it dominates elapsed time.
5. Structural traversal itself already uses `ctypes.CDLL` native routines, which release the GIL (`kernels.py:1-25,117-129`). Claiming that the complete structural stage is GIL-serialized would be wrong. Full-array sorts/copies, span reductions and hashing also remain; low aggregate CPU does not establish memory-bandwidth saturation. The cut path contains additional Python root classification and order-two metadata loops (`cut.py:61,72-82,97-108`).
6. `compare_two_parent_preparation_v2.py:191-208,256-267` records only loaded/prepared/ready boundaries. Its exact-signature passes occur **after** the timed preparation interval. Do not blame those post-timing passes for the1238s result. In-preparation fingerprints and validation hashes are different work and are included.

## Minimum observation that distinguishes the causes

No new profiling run is needed to decide whether this20.66% cold-start improvement is useful. Proceed with the existing reviewed integration once exact comparison acceptance is complete.

If further preparation optimization is necessary during an ordinary future cold-cache preparation, record wall and `thread_time` boundaries for just five existing phases: encoder global assembly after child join, episodes, cut plan, final validation, and compact packing. Existing resource-check labels already distinguish episode/cut/validation entry; they are currently checked rather than journaled. Include parent/thread identity and one completion boundary per phase, preserving every existing resource check.

- Long global-assembly time identifies materialization/sorting rather than episode matching.
- Long episode time plus substantial Python-loop time identifies the only narrowly targeted optimization below. Thread CPU near one aggregate core while both parent episode intervals overlap is consistent with GIL serialization, but exact attribution requires separating the existing native calls from the Python episode loop; a single aggregate138% figure is insufficient.
- A lock explanation requires measured acquisition wait or a lock held across the phase. The inspected pool/credit scopes do not provide that explanation.
- A bandwidth explanation requires memory traffic/bandwidth evidence during the identified phase. RSS growth or CPU percentage alone cannot prove it. Do not infer a NUMA or bandwidth fix from this audit.

## One conditional exact optimization

If the episode loop is confirmed to dominate, batch its per-hold integer array writes while preserving the existing dictionary iteration order and episode-ID assignment: collect that hold's node/episode vectors once, write contiguous membership slices, and update integer episode counts with exact indexed accumulation. This removes repeated Python-to-NumPy scalar assignments/increments without changing clade matching, hold order, any FP32 span reduction, hashes or metadata. Retain bounds before the batch and compare the existing complete preparation signatures. Do not change parent producers into processes, move expanded banks over IPC, add new compiled kernels, or vectorize the FP32 span reduction on the present evidence. This is a scoped candidate, not a measured speedup or authorization to patch the active run.

## Exact sources

All paths are relative to `/workspace-SR003.nfs2/ARG-GNN`:

- `research/efficiency_20261027/root/runs/two_parent_preparation_v2/data/{current_serial_pair,adaptive_concurrent_pair}/timed_preparation.json` and `family_samples.jsonl`.
- `research/efficiency_20261027/execution/compare_two_parent_preparation_v2.py`, `adaptive_prefetch_v2.py`, `adaptive_encoder_hooks_v2.py`.
- `research/benchmark_rebuild_20261024/american/treeloom_speed/static_preparation_v2.py`.
- `research/preprocessing_optimization_20261026/tree/process_observations.py`, `episodes.py`, `cut.py`, `kernels.py`, `kernels.cpp`.
- `research/reviews/workflow_progress_diagnosis_20261018/treeloom/resident_pool_v2.py:45-91`.
