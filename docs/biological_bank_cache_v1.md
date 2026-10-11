# Biological bank cache v1

This is an inactive execution wrapper for the unchanged seven H020 mechanisms. It replaces only the TRAIN `biological_mechanism_controls_v1.load_bank` alias and the VAL `biological_candidate.load_bank` global inside one scoped context, then restores both on success or exception. It invokes the unchanged original `controls.main()` with the unchanged original scientific arguments. The model factories, optimizer groups and parameters, seed, data/order, validation sealing, scorer, checkpoint selection, epoch and patience budgets remain owned by the original reviewed protocol.

## Exact cached object and change detection

A cold entry comes from the unchanged original loader on CPU. The cache retains its complete returned `fixed_prepared` object, including all nested prepared fields and the original float64 physical coordinates. It performs no compact operator reformulation, sampling, precision change, mathematical change, field projection or new tolerance. The serialized full age archive remains unchanged and byte-checked. Unused raw archive objects that the original loader discards are not retained in the cache. This is complete caching of the original loader output, not retention of every raw serialized archive object.

Every request runs the original full bank SHA256 check, including both CPU and GPU hits. Cold requests consequently hash again in the original loader. Admission fully hashes pinned sources/configuration/manifests. Each request checks their immutable device/inode/size/mtime/ctime identity and fails on any change, with a full hash check for diagnosis. The original runner's full epoch source and manifest hashes remain unchanged. Parent/tile identity, bank path/hash, full tile fingerprint, actual node and query/reference/dynamics roles, split and float64 physical grid are bound to the manifest and parent join. Configuration, dtype or logical device changes fail closed.

The cache checks complete nested structure, scalars, tensor device/dtype/shape/stride/storage and PyTorch mutation versions before reuse or eviction and during close. The pinned reviewed consumer models treat these banks as read-only, and handmade tests verify unchanged full field bytes after all seven forward/backward/optimizer paths. Arbitrary `.data` or external-storage writes bypass PyTorch version tracking and are outside this reviewed read-only consumer interface. The wrapper does not expose a general mutable tensor cache.

## Memory and eviction

The shipped caps are decimal **64,000,000,000 CPU bytes** and **32,000,000,000 GPU bytes**. CPU entries store exact original prepared tensor payload with shared storages counted once. GPU entries use a bounded LRU on the single exposed logical `cuda:0`; only CPU-resident canonical entries are eligible. A conservative transfer upper bound counts each tensor occurrence and the larger of storage and logical dense payload to cover expanded views. Whole GPU entries are evicted before a new eligible transfer. CPU eviction removes matching GPU entries. Oversized entries use the original uncached transfer path. The wrapper never reduces the bank or changes the computation to fit a cap.

These caps cover resident cache tensor payload only. Root must separately preserve headroom for the current uncached bank, the original loader's transient decoded full archive, Python metadata, caller-retained references, model, optimizer, autograd and runtime. PyTorch allocator reserved memory can remain after entry eviction and is not part of resident tensor accounting. For the profile, root must also account for baseline and cached bank references held together and the two original model/optimizer copies used for parity. Serialized bank size does not establish tensor, backward or allocator peaks. Existing shared GPU workers keep their reservations. An A100's nominal 80 GB capacity is not an admission decision.

The 64/32 GB caps were fixed before sealing at root's request after a read-only production sample reported a roughly 38 GB serialized 32-bank set and 2.07 TB available host RAM. That sample motivates an execution investigation only. It does not establish a causal bottleneck, a cache speedup or sufficient profile headroom. Full SHA work remains on every hit, and a cyclic full working set may exceed the GPU cache and show low hit rates.

## Inactive root admission and profile

`research/reviews/biological_bank_cache_v1_author/profile_admission.template.json` contains the exact profile schema and source map. `memory_reservation.template.json` records the exact 64/32 GB caps and separate headroom requirements. `profile_gate.template.json` and `fit_admission.template.json` define the later fit gate. All templates have inactive status, placeholder root paths/hashes/times/job IDs, and false qualification flags. They are not actual capabilities.

An actual admission must have schema `biological_bank_cache_admission_v1`, status `active`, phase `profile` or `fit`, a distinct nonempty root job ID and current timezone-aware creation/expiry with admission duration at most 24 hours. Reference roles are exactly `cache_config`, `review`, `original_config`, `observed_manifest`, `prepared_manifest`, `memory_reservation`, plus `profile_gate` for fit. All references are path/SHA256 objects. The source map is the exact original configuration map plus the wrapper's own source hash. A fresh independent receipt must start with `qualified`, have `actual_admission:false`, not be fixture-only, and bind the exact wrapper/config artifact hashes. This author receipt cannot satisfy that review gate.

The memory reservation must have schema `biological_bank_cache_memory_reservation_v1`, status `active`, the same job ID, valid current UTC dates, exactly the shipped CPU/GPU cache byte caps and `original_model_optimizer_runtime_headroom_preserved:true`. Both admission and memory expiry are checked before and after every cache request. The manifest must contain exactly the original 32 parent/tile cells. Root still uses the unchanged audited launcher, lease, process-title, UUID/NVTOP and watchdog guards and owns all scheduling, reservations, registration and termination.

The actual profile uses the TRAIN tile with maximum original tree count, breaking ties by parent ID and tile ID, selected from metadata before outcomes. It opens original observed and prepared data only. It performs four original loads and four cache loads including required bank SHA/guard costs, with cold and warm times exposed separately. It checks exact full field fingerprint, logits, all active parameter VJPs, clipped AdamW disposable unit-step model/optimizer states and Python/NumPy/Torch CPU/CUDA RNG for all seven unchanged default models. The fixed coefficient mean-logit objective uses no query ancestry targets. Profile seed 41 is separate from original fit seeds. No scientific fit occurs.

The profile writes a terminal `one_tile_profile.json` under a fresh exclusive output. A successful source profile does not automatically authorize fitting. Root must independently accept the measured loading and memory and create `biological_bank_cache_profile_gate_v1` with status `qualified_execution_only`, exact source/config/scientific references, exact parity true, `measured_profile_acceptable:true` and a SHA-bound terminal evidence reference. This is one-tile loading/parity evidence, not sustained full-loop utilization or full-32-bank GPU residency evidence. Cache hit/miss/eviction and peak resident counts must accompany any later interpretation. No performance claim or numerical tolerance is introduced.

Root command contract, under the unchanged audited shared launcher and root's current lease/title:

```sh
python -m arggnn.biological_bank_cache_v1 \
  --cache-profile-only --cache-admission ROOT_NEW_PROFILE_ADMISSION \
  --expected-cache-admission-sha256 ROOT_INDEPENDENT_ADMISSION_SHA256 --device cuda
```

After root's bound profile gate, the fit command contract is:

```sh
python -m arggnn.biological_bank_cache_v1 \
  --cache-admission ROOT_NEW_FIT_ADMISSION \
  --expected-cache-admission-sha256 ROOT_INDEPENDENT_ADMISSION_SHA256 \
  ORIGINAL_UNCHANGED_MECHANISM_TRAIN_ARGUMENTS
```

The wrapper validates the original config/observed/prepared path and hash arguments, `--train`, seed, device and fresh root output against the admission. It writes an additional exclusive `bank_cache_execution_v1.json`; the original scientific outputs remain produced by the original runner. The module checks CUDA lease/title before Torch import on a training/profile CLI entry. These command lines are placeholders and do not launch a job.

## Author verification and limits

Only remote CPU handmade work was performed, with CUDA hidden and one CPU thread. Fixtures use full 280 sampled leaves, 47 original prepared fields, the actual 120 query-slot arrangement, full mixed/pure role populations, four handmade star marginal trees over two physical tiles and the unchanged full age/anchor preparation. No stochastic simulation, actual scientific arrays, GPU/CUDA context, actual profile or fit was used. Fake GPU bookkeeping is explicitly CPU-only and does not establish CUDA allocation or parity.

There are 27 final distinct test cases. Across preserved attempts, 31 case executions passed. Attempt 01 had 22 fixture setup errors and executed no cache cases because a handmade manifest's NumPy index array was not yet a JSON list. Attempt 02 had 21 passes and one test-only failure from incorrectly expecting an expanded zero stride to survive the original loader's `.to(cpu,float32)`. The corrected transfer-accounting test uses an explicit expanded standalone tensor. Attempt 03 ran the corrected/new/changed cases and had 10 passes, with 17 unchanged cases deselected. The all-seven default field/logit/VJP/clipped-AdamW/RNG checks passed in attempt 02, and the target-free callable profile integration passed in attempt 03. No source numerical change or tolerance relaxation was used to resolve those fixture assertions.

The bundle is pending independent source review, root H020 execution-condition registration and an actual root-admitted target-free device profile. There is no new method, predictive/accuracy gain, causal bottleneck, runtime speedup, whole-loop utilization or actual-cache-fit claim. The prior Census fitting bundle and all original pinned dependencies are preserved.
