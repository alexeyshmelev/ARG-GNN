<!-- owner-stop-current:start -->
## Benchmarks resumed with owner authorization, 26 October

The owner authorized continuing the benchmarks. The new queue reuses **32 completed stages** and has **220 remaining stages**, including an additional European interim comparison. Four workers have started on physical GPUs **3, 4 and 5**, with task titles `AI Auto Research Pipeline Test`. Other users' processes are untouched.

| Workstream | Verified continuation |
|---|---|
| American TreeLoom, GPU 3 | Restored **238/600** complete-parent updates and optimizer state. Actively encoding the next observed parent for update 239. No new completed update is claimed yet. |
| Native ARGformer, GPU 4 | Restored **119,500** MLM updates with optimizer, schedule and data cursor. Fresh training has advanced to **120,599/224,170** logged updates. The original five-epoch budget is retained. |
| European TreeLoom, GPU 5 | Reuses the completed **90/90** fit for repaired held-out prediction. |
| GST-LAI-8, GPU 5 | Repaired loader started. Original full fit restarts because no prior checkpoint survived. |
| ARGMix | **11 completed stages reused.** The next Relate stage awaits host RAM admission alongside the four GPU workers. |
| Matched ancestry scores | Pending predictions and common scoring. An interim CNN + Dense + TreeLoom European comparison is queued without waiting for GST. |

Verified 2026-10-26T20:48:01.789032+03:00. **4 running, 32 succeeded, 216 pending, no failed or blocked stages.** GPU 4 was actively training at the check. GPUs 3 and 5 were awaiting their workers' active CPU input preparation. No new matched ancestry score is available.

The model, datasets and evaluation supports are unchanged. Global storage admission is **995 GB**, within the **1 TB** owner limit. The reviewed queue preserves interrupted evidence and checkpoints. ARGformer restores its training state, but persistent worker masking RNG was not saved, so an identical uninterrupted random mask stream is not claimed.

[Continuation and repairs](research/operations_resume_20261026/README.md) · [Latest verified status](research/operations_resume_20261026/latest_status.json) · [Queue plan](configs/benchmarks/paper_campaign_resume_20261026_v1.json) · [Historical stop record](research/operations_stop_20261025/stopped.json)
<!-- owner-stop-current:end -->
