# Queue reliability audit, 30 October 2026

The audit concerns execution and reporting. Scientific data, source hashes, model settings, completed fits and historical failures remain unchanged. Root owns controller handoffs and publication. Workers on physical GPUs 4 and 5 and CPU Relate remain active.

## Findings and resolution

| Ticket | Failure mechanism | Resolution | Deployment |
|---|---|---|---|
| Q301 | An immutable completion filename was visible before its JSON write finished. A downstream reader could permanently block the task. | Publish complete, flushed bytes with an atomic no-clobber link. Mutable state uses atomic replacement. | Active; independent live acceptance passed. |
| Q302 | A transient GPU or process probe exception could unwind a scheduler and terminate healthy workers. | Record unavailable observations, pause admissions and retain active workers; resume after fresh measurements. Preserve actual waits and identity failures. | Active in native continuation. Busy original controllers remain unchanged. |
| Q303 | A normally exiting process could vanish between /proc existence and identity reads, aborting admission. | Treat only disappearance as an exited process; still reject a live identity mismatch. | Active in native continuation; completed fit reused. |
| Q304 | Publication omitted the recovered native ARGMix owner and refreshed only when score files changed. | Overlay only its 20 owned task IDs, retain the historical failed rows, and publish meaningful stage transitions. | Active; reporting follows the current owner registry. |
| Q305 | Future native prediction commands did not set the required process title before CUDA. | A title-only CLI wrapper for eight never-started predictions; unchanged predictions and output paths. | Active in native continuation; completed fit reused. |
| Q306 | Generic ARGMix labels hid the registered attention variant. | Authenticate and display equation_corrected in reporting and final audit. | Reporting and final-audit continuation active. |

## Coverage

The independent merged-graph audit checked 282 tasks and 479 dependency edges: no missing dependencies, cycles, duplicate effective owners or duplicate output directories. It verified the current paired prediction barrier, cross-controller completion paths and preservation of the 20 original failed/blocked rows. See [dependency audit](dependencies/audit_v1.json).

The atomic publisher passed six deterministic CPU tests, including successful stage completion and both receipt destinations. Reporting passed seven tests, including no double counting, historical failure retention, meaningful-change detection, condition disclosure and active-owner binding. Independent review also compared the original numeric rendering and scientific plan bindings.

## Remaining limits

The running original Relate and ARGformer controllers cannot be replaced safely while they own workers. Monitoring hardening is not claimed active there. No training has been restarted to install operational changes. GPU 3 waits on the shared Relate test inputs; GPU activity is not fabricated with unnecessary jobs.

ARGMIX-AUTHOR-01 remains a separate scientific issue: the registered run uses equation_corrected, not the untouched author_key_i condition. This audit preserves and discloses that condition; it does not turn it into an unmodified-author comparison or exact published-table reproduction.

## Verified deployment

At 2026-10-30T14:56:31.736571+03:00, all four replacement controllers were alive with their reviewed plans. Only frozen, verified childless controllers were replaced. No scientific worker received a signal. The original Relate and two ARGformer workers retained their PID/start identities. Native monitoring was healthy, with the completed fit reused and 19 owned stages pending. Active campaign counts were {"pending": 91, "running": 3, "succeeded": 130, "superseded": 11}. Retained historical failures are separate.

Nineteen additional native runtime/helper fault tests passed, covering failed GPU/PSS observations, paused and recovered admissions, incomplete drain observations, retained GPU leases, real resource violations, source/identity violations and owner-lock exclusion. The final review retained 23 passing guard tests and six condition checks. [Live snapshot](live_status_v1.json) and [independent live acceptance](dependencies/live_final_acceptance_v1.json) record the deployment.

GPU 3 has no ready native training or prediction input while Relate prepares the first TEST parent. The last ARGformer parent, TEST009, waits for TEST007 scratch retirement under the registered lane-storage ordering. This is an operational storage dependency, not a scientific dependence between chromosomes. No ready training task is waiting solely for an idle GPU under the present plan.
