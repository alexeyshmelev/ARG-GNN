# Benchmark continuation, 26 October 2026

The owner explicitly authorized resuming the benchmarks. The October25 stop record remains historical evidence. Only physical GPUs3,4,5 are authorized, with no per-process GPU RAM cap. All execution, caches and results remain inside the dedicated anogena repository. Unrelated jobs are not signalled.

The reviewed continuation has252 tasks, preserving32 succeeded stages and leaving220 pending at launch. The frozen model, datasets, full training budgets and evaluation supports remain unchanged. American TreeLoom resumes after238 of600 parent updates. Native ARGformer validation001 resumes after119500 of224170 MLM updates. European TreeLoom reuses its completed90-update fit for repaired prediction. GST-LAI-8 restarts its original full90-update fit because the failed attempt had no checkpoint. ARGMix continues its shared1611-person recipe. A distinct European three-method score can finish before GST, while the original four-method comparison remains queued.

## Restart repairs

- Reconstruct the effective revision6 plan from all six applied repair receipts.
- Seed a fresh controller with32 validated successes. Reject live/unknown workers, undeclared task changes and replacement of successful tasks.
- Preserve old controller state/logs and interrupted root metadata. Write resumed native outputs in fresh nested directories, retaining original checkpoints.
- Restore model/optimizer/schedule and data cursor. ARGformer's persistent masking-worker RNG was not serialized, so exact uninterrupted random masks cannot be reproduced. The original stochastic protocol and full budget are retained.
- Use the qualified European predictor and GST cursor/checkpoint repair. Keep original failed attempts and scientific controls.
- Retain the original interrupted ARGformer checkpoint during later authorized scratch retirement. The scoped fixture verified that only new generated checkpoints enter its deletion inventory.

## Resources and limits

Global admission uses995GB, leaving5GB below the owner's1TB ceiling. The old controller's14,422,151 bytes and all668,062 archived-attempt bytes remain charged. The secondary ARGMix scope ceiling is128GiB, replacing an obsolete40GiB forecast. Neither cap allocates disk or invents free space. Stage reservations and the protected native-pipeline allowance remain enforced. Cumulative inferred-tree storage may still require a later scheduling adjustment. Whole-campaign storage fit is not claimed.

## Evidence

[Sealed plan](../../configs/benchmarks/paper_campaign_resume_20261026_v1.json) · [Preflight](preflight.json) · [American restore checks](american/README.md) · [European/GST review](european_gst/RESUME_REVIEW.md) · [ARGMix and interim score](argmix/REPORT.md)

`launch.py` dispatches the reviewed plan once and records its PID/start identity. `build_resume.py` is preparation only and refuses a launched state. `check_continuation.py` covers success preservation, explicit retry accounting, live-worker rejection and additive scoring. Launch and later live-status receipts distinguish active preparation, actual training counters, predictions and final scores.
