# Independent efficiency challenge

The guarded resident/explicit route with parallel preparation is justified and should proceed through its existing review. It has not demonstrated sustained useful GPU activity of 80%. Do not delay deployment for a new kernel, topology-only learned-state reuse, or another broad benchmark campaign.

This was a read-only source/evidence audit on anogena. Only this report was written. No scientific arrays were opened, no test/model/GPU work was launched, no Git operation was performed, and no running worker was changed.

## What is measured

The full original parent has 205,369,272 nodes, 77,472 holds, 1,600 haplotypes and 10,000 targets. The single fixed-order full comparison improved 318.315s to 237.006s, a 25.54% reduction. The accepted review confirms strict logits/loss checks, all 49 gradients and parameters, all 147 Adam tensors, zero argmax changes, and bounded memory. This establishes that mechanism's numerical/resource prerequisite, not future accuracy or repeated performance.

Existing GPU5 samples were joined by `observed_unix` to the recorded half-open top-level stage intervals. These are device-wide shared-device samples, not process attribution or GPU occupancy. Nested timers must not be added to their parents.

| Resident stage | Wall seconds | Update share | Shared-device mean | Samples |
|---|---:|---:|---:|---:|
| Cold CPU layout | 30.414 | 12.8% | 17.10% | 30 |
| Context forward | 51.465 | 21.7% | 61.25% | 52 |
| Decoder forward/backward | 29.200 | 12.3% | 69.24% | 29 |
| Trunk backward | 125.877 | 53.1% | 84.67% | 126 |

The three compute-stage samples together average **76.623%**; only 64.73% of those samples reach 80%. Thus simply removing the cold layout interval does not turn the measured evidence into an 80% result. This is a diagnostic calculation, not a warm production prediction. The total comparison mean is 69.09%; attempt7's 18.35% includes cache preparation and is not an equivalent steady-state comparator.

Trunk backward is the largest remaining runtime: second-block backward 61.529s, forcing/first backward 50.666s, complete-history backward 13.341s. Context forward comprises first/forcing 17.285s, complete history 5.728s and second blocks 28.446s. The largest runtime stage and the weakest activity stage are different. Full-stage wall timings do not determine how much is arithmetic, transfer, launch latency, host preparation, synchronization, or peer contention.

The completed serial arm of the ongoing preparation comparison supplies two parents in 1,560.893s, or 780.447s/parent. Parent001 loads in 0.861s, prepares for 758.516s, and packs for 22.611s; parent007 is similar. CPU preparation is the measured cold bottleneck, not an established NFS read bottleneck. Compact payloads are 14.012GB and 13.895GB. The adaptive arm was still active during this audit; no result is inferred.

ROOT supplied a later partial observation: at about876s the adaptive pair was still running, its main PID1220832 averaged138% CPU, and encoder children had disappeared after about380s. The configuration metadata independently confirms4-process encode hooks plus separate episode/cut preparation hooks; no final adaptive timing file existed at this audit. If that elapsed boundary is measured from the same preparation start, pair completion already exceeds876s and average supply exceeds438s/parent, so this specific pair cannot demonstrate supply fast enough for237s updates. That does not rule out a worthwhile cold-start improvement over1560.893s. Children having exited narrows the remaining work beyond parallel encoding; it does not establish GIL contention, memory bandwidth saturation, I/O waiting, or which later function dominates. Mean main-process CPU alone cannot distinguish those causes.

## Can the planned change reach 80%?

It can plausibly improve overall throughput substantially. Parallel preparation removes cold starvation; capacity routing uses the faster qualified resident executor where it fits. All20-parent compact caching avoids repeated preparation after warmup, while the larger layout cache can remove repeated layout building if actual cache receipts confirm retention. One parent retains 4.113GB of layout; multiplying that by20 is only a sizing heuristic, not proof that the complete heterogeneous set fits its80GiB budget.

During a cold supply-limited stream, the preparation system must improve the observed serial supply by about3.29x to feed one237s update, or3.78x to feed a hypothetical206.6s layout-free update. Two producers alone, without sufficient within-parent speedup, would not meet that rate. Eight encoder children do not imply eightfold speedup because preparation also contains global and packing work. Once all20 compact parents are retained, this supply condition ceases to govern steady state.

For warm useful activity `u`, compute interval `C` and uncovered wait `W`, overall activity is `u*C/(C+W)`. Even if warm activity reaches85%, a206.6s compute interval permits only12.9s uncovered waiting before the average falls below80%. The measured76.6% shared-device compute-stage average leaves no such margin. Production may differ; the present evidence provides neither a guarantee nor a proof of impossibility. Explicit fallback frequency and its elapsed-time share also matter; a mean over resident parents alone is insufficient.

## Exact next observation

Use the ordinary successor training run after its compact cache is fully populated. Predeclare one complete20-parent schedule cycle as the measurement window, retaining every update boundary, layout/cache miss, preparation wait, checkpoint, route choice and inter-update gap. Report cold initialization separately rather than silently discarding it. Confirm completed original optimizer updates and useful targets per wall second alongside activity.

Establish attribution either with a genuinely exclusive admitted GPU interval, without removing peers, or process-attributed physical activity records. A shared NVML mean cannot establish TreeLoom's contribution. Report learned-kernel activity separately from necessary copies/memsets and resource-monitor activity; an inclusive CUDA-event union is not a learned-compute utilization measure. For a concrete sustained criterion, require at least80% time-weighted attributed useful kernel activity over the whole cycle and every consecutive five-update window. Record the exact chosen criterion before observation. This criterion measures activity, not SM occupancy or arithmetic efficiency. Faster verified updates remain essential; artificially inflating busy time is not success.

## At most two follow-on mechanisms

1. **Remove any measured remaining cache/layout wait.** Complete the already planned immutable-parent caching and retain/reuse exact CPU layouts across updates. If production receipts show layout misses or restore work on the critical path, move that immutable work into the bounded preparation pipeline, with existing memory admission. This changes no learned equations. If warm receipts already show negligible waits, do not spend more work on producers.
2. **Conditionally overlap the remaining CPU-context boundary copies.** The qualified CPU-context path explicitly copies each second-block context to CPU, then copies complete-sequence decoder tiles to GPU and their gradients back. Bounded pinned staging with copy completion events could overlap those transfers with adjacent independent block/tile work while preserving CPU context layout, complete history, original tile and parameter-VJP order, and memory bounds. Pursue only if one naturally occurring warm production interval attributes consequential gaps to these boundaries. Full-run timers alone do not establish that cause. Reuse the existing strict numerical contract for a narrowly scoped change; no new kernel or broad test matrix is warranted.

Do not revive topology-only learned-state caching: hold-specific sites/span/support, outside context, history and derivative accumulation invalidate that shortcut. Do not promote the optional GPU-context candidate: its original strict context comparison failed, and the existing decision correctly defers it. The prior hotpack result saved0.094s median on a1153-hold transfer-only prefix; it is not evidence of an end-to-end full-parent win.

## Evidence and source anchors

- `root/runs/upward_residency_full_gpu5_v2/data/completed.json`, SHA256 `8529b65d7eaa591ff12bc244a742a92a94b4ae5f59e4ccd38e644f1c4029701c`.
- Same run: `device_telemetry_v2.csv`, SHA256 `3683b0f91ba8e427cd4a00e8ad161f78e6ac88d272f1d11259de40ebf3d2ba83`; `data/stage_clock.jsonl`, SHA256 `eafb61349d4161fb01bd3e25e782005882d847c31d1e4c5653598b2702a41109`; `device_interval_summary_v1.json`.
- `fidelity/UPWARD_RESIDENCY_FULL_V2_COMPLETION_REVIEW.json`; `root/production_attempt7/completed.json`; `root/runs/two_parent_preparation_v2/data/current_serial_pair/timed_preparation.json` and `completed.json`.
- Original model: `research/benchmark_rebuild_20261024/american/treeloom_speed/static_model_v2.py:275-327` (complete two-sweep model); `staged_execution_v2.py:74-115,168-187` (input transfers and CPU context); `decoder_v2.py:50-87` (full-sequence tile transfers and VJP return).
- `architecture/exact_upward_residency/upward_residency.py:142-220` (ordered second/history/first VJPs); `architecture/qualified_residency_router/router_v1.py`; `integration_candidate/ASSESSMENT_COMBINED_v5.json`.
- `architecture/SUBTREE_REUSE_SOURCE_ASSESSMENT_v2.json`; `root/context_prefix_strict_failure_v2.json`; `root/context_diagnostic_decision_v3.json`; `root/runs/hot_block_transfer_gpu5_v1/data/completed.json`.

Paths without the project prefix are relative to `research/efficiency_20261027/`. Source references are explanatory, not executable instructions.
