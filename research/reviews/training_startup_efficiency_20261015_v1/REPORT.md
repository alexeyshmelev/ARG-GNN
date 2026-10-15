**H048/H046 startup and throughput: measured no-source-fix verdict**

The concrete sampled bottleneck was an advisory-lock wait during H048 startup.
It had ended by the second interval, when H048 was actively training. The
inspected source already uses memo2 for storage transactions and lightweight
lease/RSS/CUDA checks in the optimizer loop. A further training-source cache or
guard change is not justified by these measurements, so no optimization module
was authored and no running/frozen source was changed.

**Read-only measurements.** Known H048 PID2912011 came from root’s existing
NVTOP-title receipt. No GPU query was issued. Its PID/start identity remained
constant.

| Interval (UTC, 15 October) | Wall | Process CPU | Read progress | State |
|---|---:|---:|---:|---|
| 15:37:00.547–15:37:10.549 | 10.0024 s | 0.01 s | 0 rchar bytes / 0 read syscalls | `locks_lock_inode_wait` in all three samples |
| 15:40:27.065–15:40:37.117 | 10.0515 s | 13.38 s | 394,783 rchar bytes / 35 read syscalls | running in all three samples, no scheduler descriptor open |

[First samples](CPU_samples_v1.json), [second samples](CPU_samples_v2.json).
CPU time is process aggregate, not GPU utilization. Linux rchar/read_bytes do
not provide an NFS-bandwidth or physical-disk benchmark.

The scheduler-lock inode was subsequently observed held by PID2911805, whose
project source argument was `American_C3_native_MLM_amendment_v12/root_namespace_v12.py`.
At that later 15:37:51 observation, H048 had already left the sampled wait.
The exact descriptor blocked during the earlier ten seconds was not captured
(`/proc/PID/syscall` was denied). Therefore scheduler contention is consistent
with the source and timing, but the later holder observation is not a stack
trace proving which lock caused the earlier wait. [Lock observation](lock_observation_v1.json),
[follow-up](lock_followup_v1.json).

H048’s actual output timestamps place the observation seal at15:37:21.504,
initialization at15:37:39.044, and first scheduled step qualification at15:37:51.527.
The gap from `started.json` at15:25:07.613 to the observation seal was733.891s.
That gap includes all-parent loading/preparation and any waits. The two short
samples cannot allocate the entire gap between those causes.

Actual first-step producer metadata records one step of960 per arm, with no
extra qualification update: graph-to-transition3.79727s and confidence3.13292s.
Both recorded peak reserved memory19,893,583,872bytes, below the original50GB
CUDA cap. These are existing runtime receipts, not measurements from a new
model run and not ancestry accuracy. [Artifact/timing receipt](H048_startup_artifact_metadata.json).

H046’s compact pass receipts available at inspection cover passes1–28 (896
updates/arm). Pass1 took676.763s. Passes2–28 have median412.211s,
minimum401.206s and maximum680.209s. This variation alone cannot distinguish
shared-GPU scheduling, CPU work or NFS/lease delays. [Actual pass metadata](H046_runtime_metadata.json).

**Why adding another memo would miss the observed issue.**

- H048’s context explicitly calls `load_guard_v1(consumer=True)`, installing the
  existing transaction-local memo2. Reserve/close operations retain the original
  unmemoized functions. [Context](../../../research/reviews/H048_GPU45_grid_root_source_v1/context_factory_v1.py#L91),
  [existing installation](../../../research/reviews/H048_GPU45_grid_root_source_v1/root_common_v1.py#L47).
- H048 and H046 use `original.lightweight_resources_v1` around cache/prebuild and
  optimizer steps. It checks active lifetime, actual inherited lease/title,
  logical job, RSS and CUDA cap. It contains no storage-accounting call. Its
  independent lease flock is nonblocking and must remain intact.
  [Lightweight guard](../../../src/arggnn/train_mutation_routed_whole_tree_gnn_v1.py#L107),
  [lease proof](../../../src/arggnn/native_argformer_exec.py#L49).
- The real global scheduler mutex is acquired by strict output/launch/storage
  boundaries. `PilotOutputV9.save` checks planned bytes before every publication.
  H048’s all-observation recipe seal is such a boundary. Memo2 deduplicates file
  reads *inside* a transaction, preserving the actual flock and fresh checks on
  the next transaction. It cannot eliminate waiting for another holder.
  [Writer](../../../research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py#L430),
  [memo2](../../../research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py#L134).
- Complete parents are loaded once and retained in RAM. Marker/grid preparation
  uses existing bounded RAM LRUs, while the original fast reader already caches
  validated source identities. There is no evidenced per-update NPZ reload to
  fix. Future source changes would need a timed loader/preparation breakdown,
  not a payload scan or a new model benchmark.

**One actionable future-run recommendation.** When root already has a required
close/reserve bookkeeping burst pending, finish that burst before launching the
next GPU worker whose startup will immediately publish guarded metadata. This
can avoid spending an acquired GPU lease waiting on the same scheduler mutex.
Keep independent scientific work parallel and preserve every mandatory
admission, revocation, lease, resource and planned-write check. Do not interrupt
or change the currently progressing H048/H046 runs. No speedup magnitude is
claimed, and no guard-throttling or cross-operation accounting cache is proposed.

**Boundary.** Only targeted known-process CPU/IO/wchan/fd and exact project-lock
metadata, compact runtime/launch receipts, and source files were read. No process
signals/attachments, GPU calls, payload/checkpoint/truth scan, model run, guard
configuration/accounting mutation, installation or Git operation occurred. Only
this folder was written. [Summary](summary.json), [source pins](source_pins.json).
