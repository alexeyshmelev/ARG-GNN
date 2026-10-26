# Exact TreeLoom preprocessing candidate

The candidate accelerates TreeLoom's observed-tree preparation. It does not change simulations, inferred trees, sample order, model layers, training budgets, losses, scoring or author baselines. Original sources remain unchanged.

## Implementation

- `kernels.cpp` and `kernels.py` compile integer topology traversal, exact descendant bitsets and cut indexing. Tree order, traversal order, native node identifiers, unary representatives and consecutive clade episodes remain the same.
- `observations.py` retains the existing NumPy geometry and Torch/PyG branch-support calculations. Floating operation order within each tree is unchanged.
- `episodes.py` and `cut.py` retain the existing episode aggregation, FP32 span accumulation, conditional-information calculation and verified object loaders. Current downward-level validation guards remain enforced.
- `process_observations.py` optionally prepares independent native trees in spawned CPU workers. Results are assembled in original tree order before the existing global sequence operations.
- `parallel_observations.py` records a rejected thread-pool alternative. It was slower on actual observed data and is not recommended for production.

Build the readable C++ source with `bash research/preprocessing_optimization_20261026/tree/build.sh`. Compilation uses `-O3 -fno-fast-math`. The script publishes the shared library atomically. `serial_source_manifest.json` records reviewed sources and the binary hash. `staging_history/` preserves development scripts and is not the current build procedure.

## Verification

`parity.json` compares against both the old H060 functions and the CURRENT benchmark recovery route. It passes 782 complete native layouts, 210 complete episode sequences, 210 cut plans and 25 full encoded banks. Every integer and floating array is byte identical. Metadata is equal. Cases include one sampled node, isolated samples, true unary chains, multifurcation, permuted native indexes, missing and multiallelic calls, repeated and interrupted clade episodes, bitset boundaries at 63/64/65 and 127/128/129 leaves, and 1600 sampled leaves.

`private_safety.json` adds 13 malformed ABI boundary cases. Invalid roots, links, samples, cycles, shapes and group indexes fail without a hang. Independent review additionally compares 30 valid sequences and six current downward-level rejection cases.

`process_parity.json` verifies 1, 2, 4 and 8 processes, uneven transport chunks, byte-based chunk admission, oversized-tree serial fallback, worker-error cleanup, admission-error cleanup and unchanged parent Python/NumPy/Torch random states. Full serialized inputs and outputs fit the charged transport bound in the exercised fixtures. No test opens ancestry truth.

These are extensive bounded tests, not a mathematical claim that every possible malformed machine-level input has been enumerated. The complete observed-parent bundle now also passes exact qualification. Production binding remains owned by the orchestrator and its separate integration review.

## Measured native input timing

The 4096-hold timing fixture contains 10,729,279 native nodes, 12,488 observed sites and all 1600 sampled leaves. Every output bank field and metadata entry matches the current implementation exactly.

| Encoder | Seconds | Speedup over current |
|---|---:|---:|
| Current benchmark code | 40.86 | 1.00 |
| Compiled serial | 21.61 | 1.89 |
| Compiled with 2 spawned workers | 20.64 | 1.98 |
| Compiled with 4 spawned workers | 15.57 | 2.62 |
| Compiled with 8 spawned workers | 14.61 | 2.80 |

These timings include process startup and serialization. They were recorded before adding the final process memory guards. Final full-parent timing is recorded separately by `full_process_parent.py`. The 4-worker setting offers nearly the 8-worker throughput at lower resource cost. The small 256-hold trial found threads slower than serial, so more workers are not assumed to be universally better.

## Resource and correctness controls

The proposed 4-worker configuration requires a caller reservation of 8 GiB above the existing complete-parent reserve. It admits 6.625 GiB of worker and transport allowances. Each worker has a 1.5 GiB current RSS ceiling checked at tree boundaries. Native thread pools are actively limited through `threadpoolctl` and verified at one thread. Workers remain CPU only and identify themselves as `AI Auto Research Pipeline Test`.

A chunk contains at most 128 complete trees by default and at most 64 MiB of charged input plus output. Chunk size shrinks when byte limits require it. An unusually large single tree is processed through the parent's unchanged serial path after pending chunks drain. It is never pruned or split into a different scientific example. At most one chunk per worker is pending, with an extra current-result allowance and serialization copies included in admission.

The first full-parent qualification exposed an incorrect memory guard: Linux process spawning can retain the parent's pre-exec high-water `ru_maxrss`. The guard now uses current `/proc/self/statm` RSS and records a worker-local observed peak. The failure is retained in `full_process_parent_spawn_highwater_failure.log`. The corrected full run starts workers at approximately 0.6 GiB each and monitors actual aggregate descendant RSS in addition to worker guards.

`full_process_parent.py` prepares the complete TRAIN000 observation, hashes the whole bundle and metadata with the same signature as the original integration test, and compares with `integration/full_reference.json`. It admits 128 GiB for the parent and 8 GiB for its worker subtree. It performs no GPU work, training or ancestry scoring. Root owns production integration, runtime admission and Git commits.

## Complete observed-parent qualification

The final bounded four-process source passed the complete `argformer_american_train_000` observation in `full_process_parent.json`. Every bundle array, metadata value and model input matches the current benchmark reference by complete signatures. No ancestry truth or model execution was involved.

| Full observed preparation | Seconds | Speedup over current |
|---|---:|---:|
| Current benchmark code | 1521.36 | 1.00 |
| Compiled serial | 1007.66 | 1.51 |
| Compiled with 4 spawned workers | 769.01 | 1.98 |

This is a 49.5% reduction in complete observed preparation time on this parent. Preparation timing includes the final process guards and worker startup. It excludes the separate immutable-hash qualification pass and is not an end-to-end training-speed measurement. Episode continuity and cut metadata still contain serial work, so constant GPU saturation and universal optimality are not claimed.

The run processed 606 bounded chunks using exactly four workers. Every reported native thread pool and Torch pool used one thread, CUDA was not initialized in workers, and all four workers exited after preparation. Sampled peak RSS was 58,822,332,416 bytes (54.78 GiB) for the parent and 2,677,768,192 bytes (2.49 GiB) for all descendants together. The observed simultaneous parent-plus-descendant peak was 58,835,169,280 bytes (54.79 GiB). Independent parent and descendant maxima must not be added and reported as the simultaneous peak. These are summed resident pages sampled every second, not a unique physical-memory measurement. Maximum worker-local observed RSS was 679,161,856 bytes, below its 1.5 GiB allowance.

All serial and process numerical source hashes match the reviewed manifests. A prior attempt completed the comparison but failed while serializing a NumPy integer diagnostic. Its partial file and log are preserved as `full_process_parent_serialization_failure.json.partial` and `full_process_parent_serialization_failure.log`. The harness now tests numeric scalar serialization, rejects nonfinite or unsupported values, preserves existing evidence, and publishes reports atomically. The complete rerun above passed without changing numerical sources. `test_report_serialization.py` and the independent `observed/harness_serialization_review.json` record this reporting repair.
