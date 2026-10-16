# C3 post-warmup raw-cache diagnosis

Source-only, 2026-10-16. Current C3 training remains uninterrupted. No live source or policy was edited. No scientific-library imports, numerical runs, jobs, Git operations or payload reads occurred, except the one expressly authorized compact progress receipt105. The two staged files are an unexecuted future proposal with no integration or launch authority.

## What the timer measures

Receipt experiments/results/C3_GNN_efficient_resume_GPU0_train_v1/progress_update_000105.json records a completed AdamW update105 for american_c3_short_train_018: complete-parent step90.487s, raw cache30.759s, encoder values22.117s, encoder VJP34.400s and bidirectional query GRU/head backward2.570s. These are stage wall timers, not ancestry accuracy or pure CUDA kernel measurements.

raw_feature_cache_seconds encloses SingleParentDeviceFeatureCacheV1.prepare(), recreated at every complete-parent update. Each block incurs the resource check, provider.get, CPU metadata clones, size/identity traversal, cpu.clone().to(CUDA), and device signatures. The device cache is cleared at gradient-engine exit inside each step, before the optimizer update. Thus immutable raw observations are repeatedly authenticated and transferred when the same parent recurs, while learned contexts never survive an optimizer update.

**It does not rebuild branch concordance or donor frequencies.** The actual provider chain is the efficient hook -> original GPU0 wrapper -> private-owned tracking factory -> PrivateOwnedRAMObservationProviderV1 -> ImmutableRAMObservationProviderV1. The latter forbids _read outside sealing, retains every complete parent bank and has no post-seal rebuild/eviction fallback. Its get hashes all canonical CPU tensor bytes again, checks original metadata, and binds the persistent worker context. Worker binding constructs metadata/provider state without calling the feature Dataset. Additional branch-support vectorization cannot shorten this already-warmed step's feature construction because that construction is absent here.

At cache.prepare line40, PyG cpu.clone() first allocates/copies every CPU tensor, and .to(CUDA) then allocates/transfers CUDA tensors. The first copy is redundant for this GPU transfer. The five small CPU metadata clones are intentional and remain.

The source already records cache.hash_get_seconds and cache.copy_seconds in result.cost_counts, but the progress hook emits result.timings only. The105 receipt cannot separate hashing/provider work, clone/transfer work and checks. Neither a hash-dominance nor copy-dominance attribution is supported. Resource callbacks also fall inside these stage timers. The fast callback has no storage scan/journal call, but still checks the live policy/lease/title and caps.

The measured cache stage is about34% of the step. Encoder value plus VJP stages total56.517s and remain distinct work. Complete-parent timing excludes the trainer's subsequent RAM snapshot/progress write and provider-finally cleanup. Receipt105 reports the PREVIOUS progress write at8.437s, so its90.487s must not be treated as the full inter-receipt cadence.

## Minimal future patch supplied

[Read the compact diff](minimal_future.patch). The staged [cache](single_parent_device_cache_v2.staged.py) and [step](efficient_step_v2.staged.py) make only these changes:

1. On an explicit CUDA device, replace cpu.clone().to(device) with copy.copy(cpu).to(device). Require a distinct PyG Data container, store and store mapping before conversion. Preserve clone().to(device) for a CPU fallback, because CPU-to-CPU to() may alias storage.
2. Export the existing provider-get and copy-call elapsed values into timings, so the existing capped progress hook reports them. Provider-get time includes canonical hashing and context/identity work. Copy-call time remains host elapsed, not a separate CUDA-event measurement.

Installed PyG2.7.0 source was read directly: Data.clone is copy.copy(self).apply(tensor.clone), Data.__copy__ duplicates its store, BaseStorage.__copy__ duplicates the mapping, and Data.to applies tensor.to while replacing entries in that copied mapping. CPU-to-CUDA conversion allocates destination tensor storage, so removing the intermediate CPU tensor clone does not make the CUDA model bank alias the CPU source. Original size, device, no-gradient, tensor-version and memory-cap checks remain. Metadata cloning, complete graphs, SNPs, focal loss, two-phase VJP, recurrence, AdamW, RNG checks and per-update cache clearing remain unchanged.

**Full CPU digest-on-every-get is deliberately retained.** The existing private-storage clone at sealing and its full canonical digest remain unchanged. This patch does not replace content assurance with a concealed alias assumption. A later version-stamp-only getter could check tensor/storage identity, version, dtype, shape, stride, offset, source metadata and private ownership after the initial hash, but it would not detect all NumPy/.data/raw-pointer writes. It would require an explicit no-escaping-mutable-alias contract and a separately reviewed integrity change. Do not describe it as identical cryptographic coverage or fold it into this minimal transfer patch.

## Limits and integration

Only standard-library AST parsing was performed on the staged text. No Torch/PyG import, parity/speed/RSS check, optimizer update or current-run mutation occurred. No measured speedup is claimed. The extra timing fields remain subject to the original2048-byte stdout and8192-byte durable-record limits, which are not raised.

The current source hashes/admission explicitly reject these new paths. Root would need a separately bound successor and an actual continuation boundary before any use. Do not apply the diff to frozen live files or alter the current run. A later ordinary scheduled update can report the measured two component timers and actual memory under unchanged caps. This proposal does not request a discarded qualification or broad proof campaign.

## Exact source anchors

- efficient_step_v1.py:102–115: per-update cache creation/timer. Its finally clears the device cache.
- single_parent_device_cache_v1.py:25–50: block checks, provider.get, metadata clone, CPU clone plus CUDA transfer. Lines59–64 return the currently unexported timing split.
- C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/ram_cache_provider_v1.py:59–93: no post-seal rebuild and full canonical hash on get.
- C3_private_owned_RAM_cache_source_20261016_v1/private_owned_cache_v1.py and private_owned_tensors_v1.py: private CPU storage, preserved view layout and canonical digest after clone.
- integration_api_v1.py:94–107: progress exports timings, not cost_counts, and records bounded write timing.
- .runtime/env/lib/python3.11/site-packages/torch_geometric/data/data.py:346–365,587–594 and storage.py:135–141,196–199,903 onward: exact copy/conversion behavior read without importing it.
