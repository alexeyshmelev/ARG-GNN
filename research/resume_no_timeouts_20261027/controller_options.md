# Controller walltime options

Audit of the live campaign completed at 2026-10-27T20:36:16+00:00. Inspection only; no production code, process, plan, or job was changed.

**Recommendation: retain the current controller and its healthy workers. There is no supported live update or controller handover that removes all currently loaded walltime limits while preserving both workers and genuine parent-process waits.** The failed GST branch can be resumed through the existing repair path without touching these workers. A large positive compatibility limit can prevent practical GST expiry under the current schema; it must be described as finite compatibility behavior, not a newly supported unlimited policy.

## Verified live identities

| Process | PID / start ticks | State and effective job budget |
|---|---|---|
| Controller | 1005327 / 571023980 | Parent PID 1; plan `configs/benchmarks/paper_campaign_resume_20261026_v1.json`; revision 4 |
| TreeLoom attempt 8 | 1230526 / 575565341 | Child of controller; own SID/PGID; GPU 3; 2,592,000 seconds, approximately 29.40 days remaining |
| ARGformer validation 003 MLM | 1557159 / 578928094 | Child of controller; own SID/PGID; GPU 4; 604,800 seconds, approximately 6.79 days remaining |

State reported 2 running, 53 succeeded, 194 pending, 1 failed, and 2 blocked. TreeLoom's actual attempt-8 job permits 824,633,720,832 bytes PSS and 102,320,936 output bytes; ARGformer permits 137,438,953,472 bytes PSS and 8,589,934,592 output bytes. These values came from actual launched job files, not superseded original task definitions.

## Why ordinary changes do not work live

- `scheduler.py:185` reads and validates the plan once. At line 242 it enforces elapsed time against that in-memory task. Editing the plan or state does not update it.
- `runtime.py:107-117` reads each job once and constructs its `ResourceCheck`; line 87 checks that in-memory `max_seconds`. TreeLoom's `root/async_resource_checks/check_v2.py:52,113` also checks the captured task. Editing job JSON or source does not update these running objects.
- `repairs.py:92` accepts only failed or blocked tasks. It cannot update running or pending tasks. Its candidate passes the existing scheduler validator, which requires positive integer `max_seconds` at `scheduler.py:63-65`. `null`, zero, or infinity is unsupported.
- SIGTERM and SIGINT raise into the scheduler's `finally` block, which drains every running worker (`scheduler.py:216-219,392-400`). They are not detach signals. Unhandled SIGHUP is not a supported handover.
- Killing only the controller would leave separate worker sessions running but remove their resource supervisor and actual parent wait. A new non-parent controller cannot honestly synthesize `Popen.wait()` results. The global lock prevents a second normal controller; continuation explicitly rejects a living prior controller and any live/unknown task (`continuation.py:29-49`).

## Practical current path

1. Keep both healthy workers and their controller running. Their present deadlines are distant; do not restart them solely to change walltime policy.
2. Resume the failed GST branch through the existing exact repair mechanism. Preserve its CPU, PSS, GPU, output, auxiliary-scope, and storage reservations. If immediate unlimited behavior must fit this legacy schema, an explicitly documented positive integer sentinel such as `9223372036854775807` gives no practical expiry for this replacement branch. It does not alter the two healthy workers or other pending tasks.
3. For the next controller/runtime version, support explicit `max_seconds: null` in validation and only the elapsed-time checks. Update the TreeLoom async wrapper consistently. Keep process-inspection timeouts, bounded termination/join behavior, and resource-scan freshness checks: these protect supervision, not campaign duration. Refresh source pins where changed files are referenced.
4. Activate that version at a genuine idle/completed-controller boundary using preserved state and the existing continuation rules. Current code has no admission-pause or graceful-detach mechanism; do not describe source editing as live activation. There is no supported immediate all-task solution satisfying every requested constraint in this already-running version.

The current admission ceiling is 995,000,000,000 bytes, with a 916,096,586,310-byte historical baseline; do not confuse either with the owner's newer 1.2 TB ceiling or silently change this accounting during a walltime-only operation. Keep 6 maximum parallel tasks, 2 jobs per GPU, 2 CPU preparation lanes, GPU UUID pinning, RAM headroom, output/scope limits, and actual descendant draining intact.

## Verified GST compatibility route

The existing `scheduler.validate()` passed an in-memory plan assigning `max_seconds: 9223372036854775807` to GST fit, predict, and score. No plan file was changed and no task launched. Python JSON round-trips this exact positive integer; both existing elapsed-time comparisons accept it. It represents approximately 292 billion years and is a compatibility encoding of the owner's no-arbitrary-cutoff policy for this replacement branch.

Use the sentinel on all three failed/blocked GST replacement tasks so a downstream task does not reintroduce the same cutoff. Attempt 3 had a 24-hour limit, 320 GiB PSS, 64 MiB output, and 8 CPU threads; preserve those non-time limits and the original execution environment. Its 71 admitted source references contain no `paper_campaign` runtime file. The GST factory computes scientific source references independently, and checkpoint validation compares those references, input references, settings, model/optimizer state, and progress. Changing only outer `max_seconds` does not change these scientific pins; the exact resume wrapper must still preserve their checks. Source maintenance for future null policy need not modify GST scientific files.

This is immediately usable through the already-running controller's repair interface. It does not change either healthy worker's loaded timer or the other 194 pending task definitions. Defer any global controller transition to a safe boundary rather than discarding healthy workers or caches.

## Source fingerprints

- scheduler.py: `494a1347d7dc5f6a06824d712736e2e0663da8a39c70574ca113eeb4dc8ec456`
- processes.py: `9860e688fe204196b72d5a868edd4a0df9287f349dce1beb39b48d6a94604f84`
- runtime.py: `1412abf76767545b9addbc6f27b1dcd449837b2b963bf277f5726171fa0defe2`
- continuation.py: `306638ec7b3c95b5aa8536cfa3f34afcefebce8165459d114e5e09d34f2cf82a`
- repairs.py: `e8e407b7145d7dae20f833c61e7c040961de4aa16b1bfcb7f3f895a7ca986d23`
