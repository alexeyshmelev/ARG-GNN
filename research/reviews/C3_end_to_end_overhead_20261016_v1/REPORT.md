# C3 end-to-end overhead audit

2026-10-16. Source, existing diagnostic log and small metadata inspection only. No project-module imports, numerical payload reads, probes or execution.

## Measured facts

- The original C3 controller records 26,560.374 seconds from start to its genuine stopped/drained terminal. Its only verified durable checkpoint has 100 updates and 38 AdamW states at step 100. Later live updates remain unknown. Dividing the whole elapsed time by 100 does not establish the true update latency.
- Its diagnostic job log is empty. Existing inspected metadata provides no per-stage training timing split. There is no measured basis for assigning a percentage of the original elapsed time to guards, hashing or model compute.
- Migration actual_launch.json records 140.908 seconds of admission/setup before worker handshake release, before any migration callback.
- The launch record is timestamped 07:15:13.261 UTC. worker_started.json is 07:16:01.002, another 47.741 seconds later. The worker source writes that marker after its initial configured-guard boundary and before importing the migration callback.
- The genuine child wait is zero at 07:17:39.794. outer_wait.json reports 317.839 seconds since root execution began, error null and returncode zero. That is a pre-completion accounting record, not total final tool lifetime. The nominal 300-second lifetime was already exceeded and cannot be reported as satisfied.
- The original monitor checks wall time while polling and callback resource checks do so before their next action. Final boundary calls at ordinary_root_cpu_v1.py:313,330,336 have no subsequent mandatory wall predicate before success. Blocking accounting can therefore cross the lifetime after the last checked callback. Root owns any operational correction.

Timing evidence: experiments/results/C3_full_state_migration_CPU_root_v1_aux/{actual_launch.json,worker_started.json,actual_wait_minimal.json,outer_wait.json}. Original elapsed and durable progress: research/reviews/C3_actual_full_state_handover_root_20261016_v1/final_registry.frozen_v1.json and research/reviews/C3_actual_checkpoint100_root_20261016_v1/counters.root_v1.json.

## Source-demonstrated costs

| Path | Source finding |
| --- | --- |
| Original C3 resource callback | worker_entry_v1.py:51-57 performs expiry, RSS, lease/logical-job and CUDA checks. It explicitly performs no storage or journal call. Do not attribute every training callback to global accounting. |
| Original lease callback | biological_candidate.py:181-186 invokes GPU0 validate_lease and sets/verifies the process title. gpu0_native_guard_v1.py:35-81,92-115 rereads/hashes policy metadata, resolves paths, checks inherited FD/inode, reads/parses lease bytes and opens/tries the NFS flock each time. Logical-job validation itself is only environment/string comparison. Individual costs were not timed. |
| Original learning path | For B blocks, stream_train_isolated_v2.py:191-225 requests B row-metadata blocks in each direction, plus B blocks inside each forward encoder closure. Ordinary backward checkpoint replay repeats the latter. This gives nominal 6B provider gets and 4B encoder calls per parent, rather than one graph encoding reused in both directions. Every encoder use clones/transfers the block. |
| Host feature cache | ram_cache_provider_v1.py:76-93 hashes the complete cached tensors on every get, then binds/checks the persistent manager. Cache reuse avoids graph reconstruction but does not avoid feature SHA work or all parent worker-binding work. |
| Existing efficient successor | single_parent_device_cache_v1.py:25-51 fetches/hashes and transfers each block once. efficient_step_v1.py explicitly requires exactly 2B full-graph encoder calls. It retains all 1,600 graph leaves while recurrent states cover the 1,000 query columns. It avoids repeated raw GPU content hashes using owned tensor/version checks. This reduces source-demonstrated work but has no measured production speedup yet. |
| Successor resource callbacks | efficient_step_v1.py retains the original callback through bounded_resource_check_v1. Cache, value and VJP loops add 3B checks. Two checkpointed directional GRUs nominally add 4B initial/replay checks, and ten classifier chunks add approximately 20. Thus the optimized numerical path still has roughly 7B+20 original lease callbacks, plus constant checks. |
| Migration accounting | ordinary_root_cpu_v1.py:93-108 does not install memo2. Admission, worker startup, writer initialization, copy writes and terminal/completion writes invoke the original global replay guard repeatedly. prospective_980gb_v1.py:269-326 replays its journal and frozen v3 overlay metadata, and :392-418 authenticates consumer bindings and measures the owned scope at each boundary. Small checkpoint/output size does not make these checks small. |
| Existing caches | The legacy base already has fingerprint-invalidated state caching, failed_scope_capacity_v1.py:221-233. It still resolves/stats watched dependencies. Memo2 deduplicates bounded raw/source reads within one locked operation and validates original identities before return. It deliberately clears state between operations. The existing consumer_namespace_memo_v1 adds two exact namespace callers. Neither memo accelerates the separate GPU lease callback. |

Old versus successor call counts are source-derived, conditional on the ordinary checkpoint replay path, not observations of executed CUDA kernels. Host-call timing fields in the successor do not independently measure asynchronous CUDA kernel time.

## Narrow follow-through

1. Continue the existing qualified efficient successor. Use its already implemented cache, encoder/VJP and complete-step counters and timings in actual full learning. Do not repeat another qualification campaign. Preserve data, full-tree equations, target count, optimizer state and the 600-update retained trajectory.
2. Reuse existing memo2/namespace support only in separately admitted read-only consumer processes where installation is already permitted. It is not installed in the current ordinary migration route. Keep root reserve/close mutations and every original write boundary unchanged.
3. A future narrowly scoped optimization could retain immutable policy bytes and validated read-only lineage in one process while preserving fresh original FD/inode/ownership checks and all required invalidation. Source equivalence must be proved before adoption. No such change or gain is claimed here.
4. No source knob currently throttles efficient_step resource checks independently of learning. Do not silently skip existing checks, remove write guards, change deterministic precision/runtime conditions or build another guard framework.

Conclusion: The migration has direct evidence of substantial overhead before scientific work. Original C3 has clear redundant hash/transfer/encoder/recurrent work and frequent NFS lease checks, but no measured attribution of its seven-hour elapsed time. The existing successor addresses the numerical/cache costs. Its end-to-end gain remains unmeasured.
