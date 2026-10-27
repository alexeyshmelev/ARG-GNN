# Recovery without arbitrary job deadlines

The owner requested removing arbitrary elapsed-time limits and continuing the benchmark. This changes execution policy, not simulation parameters, training budgets or scoring.

| Item | Status | Evidence or next action |
|---|---|---|
| GST durable recovery | Running, attempt 4 on GPU 5 | Actual checkpoint contains 79 of 90 updates. Original attempt is archived unchanged. The real-loop interrupted/resumed CPU test matched model, AdamW, RNG and schedule exactly. The production worker is loading the original observations before restoration. |
| GST fit, prediction and scoring wall limits | Applied through existing controller | The live controller requires a positive integer. All three tasks use its effectively unlimited compatibility value, 9223372036854775807 seconds. This is not literal null support. |
| European full-size validation | Running on CPU | The successor started with the original 200 Mb parent and seeds. Status-file and queue-worker handover races are fixed. All total/stage elapsed limits are removed. 22 checks and independent review passed. Guardian PID 1589272 owns worker PID 1589349. |
| Healthy TreeLoom and ARGformer | Preserved | Keep existing workers, optimizer state and warm TreeLoom caches. Do not restart them to reload administrative settings. |
| Global timeout policy | Not yet active | Isolated optional-deadline support passes 8 tests. Nested limits remain in 150 ARGMix task wrappers. The future plan is explicitly not launch-ready, see campaign_future/REPORT.md. Current loaded jobs retain their legacy settings outside the repaired GST branch. |
| Campaign storage accounting | Open | Current controller still uses the pre-cleanup baseline and 995 GB admission ceiling. One ready ARGMix Relate task waits for storage reservation. Apply the authorized 1.2 TB policy and verified cleanup credit at a safe controller transition, without inventing free capacity or interrupting healthy jobs. |
| European cohort qualification | Failed, preserved | 180 of 3,899,392 context values exceed the original numerical tolerance. Do not silently relax it or launch the corrected fit as if qualification passed. |

Execution remains inside this repository on anogena. Physical GPUs 3, 4 and 5 remain the only authorized GPUs. Retain process identity checks, real I/O errors, resource limits, heartbeat freshness and stall detection. They are distinct from total elapsed experiment limits.

No new accuracy result is claimed by this recovery. Model checkpoints, simulation receipts and metrics remain separate from operational stage counts.
