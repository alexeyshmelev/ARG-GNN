# C3 live-startup CPU feature diagnosis

Prepared for the root agent on 2026-10-16. Read-only diagnosis of the running C3 continuation. The two observations below were collected earlier in this worker turn and are transcribed from the resumed context. No new live process, GPU or disk observation was performed to prepare this report. The original observation tool-cell identifier is not retained in the resumed context.

## Conclusion

The source-defined pre-optimizer stage is construction, validation and retention of complete CPU observation banks for all 20 TRAIN parents. The effective producer count is two. Both producers were active at close to one CPU core each over the bounded observation interval, while the main process retained another 160 MiB. This is evidence of ongoing CPU work and cache growth. It does not establish exact feature-construction, NFS, transport or guard timing.

The trainer does not restore its checkpoint or make a new optimizer update until every parent is sealed and the external TRAIN target joins complete. The continuation repeats this startup construction because the retained cache belongs to the new process. It resumes from the intact scheduled checkpoint at global step 100, with 500 scheduled updates remaining. The absence of a new update receipt during this stage is not evidence that an update completed.

## Parent job and source bindings

All paths below are relative to /workspace-SR003.nfs2/ARG-GNN.

- Controller tool 74136. Controller PID 3349405, start ticks 481281638, pool research0-routefix-oct16-v1.
- Registry: experiments/results/GPU0_routefixed_C3_native_controller_v1/registry.json. This is mutable operational state, not a frozen source receipt.
- Capability: research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json
  SHA256: 4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870
- Job: C3-GNN-efficient-resume-GPU0-TRAIN-v1.
- Main C3 worker: PID 3352021, start ticks 481303226.
- Scientific input: research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/input.json
  SHA256: b420377394b1246a55dd2ce2a731f523692ef557d8ea074d2fd3137231008a25
- Operational config: research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/config.json
  SHA256: 39a75fecdebe23b37165d9a18678511cdf93c2e428ebf5d926e190c311aa65a9
- Envelope: research/reviews/C3_module_route_repair_root_20261016_v1/C3_RESUME_TRAIN/envelope.json
  SHA256: e8561d14f8e25094fca3eef30714e88854ca1d31827ddaeeabd0053bd5bcc49d
- Output: experiments/results/C3_GNN_efficient_resume_GPU0_train_v1.
- Log: experiments/results/GPU0_routefixed_C3_native_controller_v1/jobs/C3-GNN-efficient-resume-GPU0-TRAIN-v1.log.

The earlier output observation found started.json and cuda_qualification/, a zero-byte job log, and no observation-barrier or new optimizer-update receipt. started.json declared the train stage, physical GPU0, the correct UUID, the title AI Auto Research Pipeline Test, explicit spawn and abstract Unix IPC. Qualification was not rerun.

## Exact two bounded observations

Wall-clock Unix times: 1792139858.5298066 and 1792139868.546711.
Monotonic interval: 10.0169053944 seconds.
Clock ticks: 100 per second.

Every observed PID was in the exact controller-owned roster. Both producer children had PPID and PGID 3352021. No unrelated process inventory was taken.

| PID | Start ticks | CPU seconds gained | RSS first, bytes | RSS second, bytes | wchan first | wchan second | Threads |
|---:|---:|---:|---:|---:|---|---|---:|
| 3349405 | 481281638 | 2.12 | 41332736 | 39215104 | ep_poll | ep_poll | 1 |
| 3352021 | 481303226 | 1.87 | 13643718656 | 13811490816 | do_poll.constprop.0 | 0 | 73 |
| 3352975 | 481310728 | 0.00 | 12582912 | 12582912 | pipe_read | pipe_read | 1 |
| 3352976 | 481310728 | 9.58 | 726106112 | 729206784 | 0 | unix_stream_data_wait | 66 |
| 3352977 | 481310728 | 9.50 | 753238016 | 751136768 | 0 | 0 | 66 |

PID 3352975 is consistent with the spawn resource-tracker role. Its exact command identity was not established by these measurements, so that label remains tentative.

| PID | rchar delta | wchar delta | syscr delta | syscw delta | read_bytes delta | write_bytes delta |
|---:|---:|---:|---:|---:|---:|---:|
| 3349405 | 28393332 | 101787 | 9255 | 612 | 0 | 98304 |
| 3352021 | 1258363 | 183480 | 13381 | 7441 | 0 | 0 |
| 3352975 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3352976 | 91392 | 146388 | 7424 | 3713 | 0 | 0 |
| 3352977 | 92088 | 146939 | 7458 | 3729 | 0 | 0 |

Main RSS increased by 167772160 bytes, exactly 160 MiB. The producer CPU deltas are approximately 95.6% and 94.8% of one core during this interval. Zero read_bytes does not exclude NFS traffic. Frequent small I/O calls do not measure guard overhead, shared-tensor transport cost or useful feature work separately.

## Source interpretation

The pinned trainer at research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py:215 constructs 20 providers and line 218 seals all 20 before opening TRAIN truth. The observation-barrier receipt is written at line 225. Target joins follow. Model initialization and checkpoint restore occur at lines 263-298. The new update loop begins at line 299.

Each worker uses the unchanged Dataset __getitem__, which extracts all local-tree nodes and observed genotypes, then calls encode_block. branch_concordance_support_v1 sweeps each site and each postorder node in Python to build descendant allele counts. It computes five summaries over every observed proper allele split. These loops are a concrete source-level CPU cost. The observations do not apportion their time against loading, layout, hashing or transport.

The persistent worker pool dispatches independent indexed blocks to separate child processes. It has no global lock around feature construction. The main process waits for ordered replies, checks transport hashes and clones incoming tensor storage into the private retained cache. Parent order is serial at the all20 sealing loop. Each child sets torch.set_num_threads(1), so its scientific Torch work has one intra-op thread despite a larger OS thread count.

The main-process fast callback checks expiry, elapsed deadline, RSS, the inherited GPU lease, logical job identity and CUDA memory. The lease validation uses local descriptor and path checks, including NFS metadata operations. It is not a controller RPC. No storage scan or accounting journal call is performed by this fast callback. A guard or NFS wall-time bottleneck is not established by the two observations.

## Effective worker limit and conservative memory

The actual config selects num_workers=2, max_block_bytes=128 MiB, worker_rss_cap_bytes=8 GiB, aggregate_rss_cap_bytes=120 GiB, task_timeout_seconds=900 and drain_timeout_seconds=30.

The immutable factory and root contract additionally require exactly two workers. The persistent pool accepts only 2 or 4. Eight therefore needs a fresh additive source/config successor, even though its independent block computations can run in parallel.

The current bound is main RSS 88 GiB, worker RSS 8 GiB each, 16 GiB spare inside the aggregate 120 GiB envelope, and another 8 GiB between aggregate and root 128 GiB. The main allowance contains 40 GiB logical cached tensors and 48 GiB non-cache allowance.

| CPU producers | Main plus worker caps | Aggregate cap retaining 16 GiB margin | Root cap retaining another 8 GiB |
|---:|---:|---:|---:|
| 2 | 104 GiB | 120 GiB | 128 GiB |
| 4 | 120 GiB | 136 GiB | 144 GiB |
| 8 | 152 GiB | 168 GiB | 176 GiB |

Four fits within 128 GiB only if the current 16 GiB aggregate margin is changed. Eight does not fit the unchanged worst-case caps. Current approximately 0.7 GiB producer RSS cannot prove a lower all-parent worker cap. A future 8-worker, 176 GiB-root operational successor can preserve these planning margins, subject to available host memory. This is a bound, not a claim of measured peak allocation or speedup.

## Narrow practical direction

Keep the current warming and training process uninterrupted. For a later successor, increase independent block producers with explicit memory caps and retain exact block order, observation-only inputs, full-tree coverage, hashing, cache ownership and runtime checks. The existing map-style Torch Dataset emits PyG Data blocks and can support a bounded persistent-worker DataLoader path. Custom PyG collation must preserve the original complete-tree bank and its metadata.

A smaller algorithmic improvement is to batch integer descendant-allele propagation over bounded site chunks, leaving the original final site/allele accumulation order intact. Its design and risks are in VECTORIZATION_DESIGN.md. No code, numerical qualification or experimental run was performed for that design.

## Pinned source references

| Path | SHA256 |
|---|---|
| research/reviews/C3_efficient_resume_migration_source_20261016_v1/root_resume_runtime_v1.py | 41367588489a2b2451156f881b2b824824955493b8172f85817d9a0b79387cb6 |
| research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py | b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c |
| research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/persistent_provider_v1.py | 337b01155dd8f0810a13c048bdd3f13b7afaf669b6ddeb807dbd9b263f5bc5cf |
| research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/persistent_workers_v1.py | c5fa5c28151d4c2891f96d8fcc0d8442c734d2394de2718ab37f3a4b776fcb68 |
| research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/prefetch_provider_isolated_v2.py | 64fdb94c4cdf16163d2443b1aea8d2afde8f389d6f6690628ec622e13c98c057 |
| research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/private_owned_cache_v1.py | f2c1ba189436d5280b461d93822f4e75e54eb5205f1b95f490e4388d5b946dbf |
| research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/ram_cache_provider_v1.py | 7de3ce76df415f20056bdba0ef2b4a2fe48ea1938b270fa0fa4384719ac23936 |
| research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_contract_v1.py | 8143dcd241c23f9bcf3379644c275f66febc8cf8dea01727e542d88651d734a0 |
| research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/worker_entry_v1.py | 8901ac783449859db3f1a872932806a30768fe9c4e5e7f975e78bfd34e0f624d |
| research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_native_guard_v1.py | 3a08d3bd84adc95aafd332bc224fdb7aa38730d6c2f48dee8108944251b7ee69 |
| src/arggnn/american_c3_gnn_dataset_isolated_v2.py | 6e4f32fcec9e6af0c6aa2c81cdbcc4ef1bf7ff83da04b233f0064b6a65c33cb2 |
| src/arggnn/american_c3_gnn_observations_isolated_v2.py | adf1be0aab14e54b0d48c8cb577b3df541bed2ee88ac846f16bab8a75d56f781 |
| src/arggnn/mutation_routed_tree_observations_v1.py | 922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068 |

No process was signalled or restarted. No checkpoint payload was read. No experimental, accounting, ledger, Git or installation action was performed.
