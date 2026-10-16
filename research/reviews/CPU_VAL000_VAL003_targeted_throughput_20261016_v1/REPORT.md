# Targeted CPU throughput and parallel readiness

Observed 2026-10-16, approximately19:12–19:17UTC. This report records read-only exact-process observations and source/metadata inspection. No process was signalled, restarted or attached to a debugger. No scientific payload was opened, source/runtime/accounting changed, storage-wide scan performed or Git operation issued.

## Current work is real, with one concrete lock delay

| Stage | Exact worker identity | Observed activity | Completion |
|---|---|---|---|
| VAL000 GNN prediction, outer tool75264 | PID3729481, start485131414, supervisor3728889/start485119768 | Running, about4.4–4.9CPU cores in separate3-second samples,~1.0–1.2GiB RSS. Parent supervisor waits in ep_poll at~0.02core, as expected while supervising. | started.json exists, no prediction seal or terminal wait yet |
| VAL003 native paths, outer tool48254 | PID3726755, start485083959, supervisor3726625/start485081333 | Initially~0.96CPU core and~37GiB RSS, producing native files. Later CPU stopped while waiting on the project scheduler mutex. | Partial products exist, no completion or direct-wait receipt |
| Newly launched VAL001 prediction, outer tool28921 | PID3734479, start485218657, supervisor3733598 | Initial validation/NFS reads, no started.json yet at19:17UTC. Holds the exact scheduler mutex while other project work queues. | Launched, scientific prediction phase not yet confirmed |

The VAL003 worker's start identity exactly matches its genuine launch.json. VAL000 and VAL001 identities were located using anchored exact worker command/input matches and verified through current /proc stat/cmdline. Their supervisor records deliberately save launch.json and buffered logs only after terminal wait and owned drain. Lack of these live receipts does not mean the workers never launched. These live identity observations are not substitutes for eventual actual wait/terminal receipts.

VAL000 started.json was written at19:03:43UTC, after runtime, storage and saved-artifact acceptance checks. Its worker is actively using CPU. The model's exact current tree/block index is unavailable because the source emits no per-block progress. We cannot infer an ETA or completed fraction from process CPU time.

VAL003 produced focals.npz, a1,346,294,886-byte native MLM TRAIN pickle and a187,545,298-byte compressed row-order metadata file. The latter grew from76,218,195bytes to187,545,298bytes during this inspection. These files prove real preparation, without certifying complete or valid outputs before the producer finishes. Source loops build full native paths and write compressed arrays serially. The observed one-core activity is consistent with that work.

## Exact lock contention

At19:17UTC, /proc/locks for the scheduler.lock inode9257095304247570939 contained:

```
FLOCK ADVISORY WRITE 3734479 00:43:9257095304247570939 0 EOF
-> FLOCK ADVISORY WRITE 3726755 00:43:9257095304247570939 0 EOF
-> FLOCK ADVISORY WRITE 3349405 00:43:9257095304247570939 0 EOF
```

PID3734479 is the verified VAL001 worker. VAL003 was in locks_lock_inode_wait with FD4 pointing to this exact project scheduler.lock. VAL001 had read~3.75GB through read calls and was observed in rpc_wait_bit_killable. This establishes current serialized mutex/NFS validation delay. PID3349405's role was not inspected or inferred.

The guard holds the original mutex across current-state authentication and product-boundary checks. This is a concrete reason VAL003 can stop making CPU progress even though no scientific failure or terminal receipt exists. It does not establish the cause of the earlier native ARGformer controller failure.

Starting another predictor's preflight immediately can add to this same lock queue. Let the current startup validation clear before initiating another full preflight. A reviewed immutable validation cache or shorter authenticated lock transaction could reduce repeated operational delay, but no guard change or skipped validation is proposed for these live jobs.

## Third prediction: RAM fits, CPU quota matters

Resource snapshot19:15:44UTC:

- Host MemAvailable:1,687,427,026,944bytes, approximately1,571.5GiB.
- Actual cgroup memory.max:1,826,434,842,624bytes,1701GiB.
- Actual cgroup memory.current:1,072,469,168,128bytes, approximately998.8GiB.
- Conservative cgroup headroom from these actual values:753,965,674,496bytes, approximately702.2GiB.
-256online/allowed CPUs, but cpu.max=5600000 100000 limits the cgroup to56CPU cores of quota.
- Across a3-second sample, total cgroup usage averaged49.8cores. Six of30quota periods were throttled. The aggregate throttled_usec increased2,922,371. This is an aggregate counter, not2.92seconds of wall-time delay for either project job.
- Global CPU idle+iowait was79.2%, which must not be interpreted as256unrestricted CPUs available to these workers.

A third128GiB-capped prediction fits current measured memory headroom, including substantial room for the two existing small prediction workers to grow. This is a point-in-time admission assessment, not a proof of future peak-memory fit or protection against unrelated future growth. The native-path worker currently occupies~37GiB. Preserve all actual failstop guards.

CPU throughput is the practical limit: three32-thread settings permit96Torch compute threads against56cgroup CPU cores shared with other work. Current observed VAL000 demand is only~5cores, so concurrency remains plausible, but a third job has no guaranteed speedup and can increase throttling. Stagger startup validation, then measure actual concurrent compute. There is no current RAM-based reason to postpone the third prediction until earlier predictions finish.

## Source-supported efficiency opportunities

The exact VAL000 plan has76,527trees,228,477sites,2,392blocks,1600haplotypes and10,000focal records. Each checkpoint executes forward and reverse full scans. Six checkpoints therefore require28,704block visits and918,324tree visits. Current loaders use num_workers=0. Each candidate constructs fresh directional datasets and rebuilds checkpoint-independent observed features, followed by checkpoint-dependent PyG encoding and recurrence.

This repeated preprocessing and lack of CPU preprocessing overlap are concrete avoidable work in the source. A separately reviewed successor could reuse bounded checkpoint-independent features or use already-supported loader workers, while preserving every tree/site/sample, both full recurrent scans, all six candidates and source numerical behavior. The timing share and memory/storage cost have not been measured. Do not claim a measured speedup or change the current live method based on this inspection.

[VAL003 genuine launch](../American_C3_VAL003004_downward8GB_source_20261016_v1/admitted_val003_native_paths_down8_v1/worker_aux/launch.json) · [VAL000 actual start](../../../experiments/results/C3_full30Mb_eval_predict_val_000_root01_oct16/started.json) · [Exact VAL000 plan](../../../experiments/results/C3_GNN_current980_plan_val_000_v1/american_c3_val_000_block_index.json) · [Prediction worker source](../C3_saved600_artifact_acceptance_source_20261016_v1/worker_entry_v1.py) · [Dataset source](../../../src/arggnn/american_c3_gnn_dataset_isolated_v2.py) · [Two-pass inference](../../../src/arggnn/american_c3_gnn_run_isolated_v2.py) · [Native preparation](../../../src/arggnn/american_c3_paths_v1.py).
