# Root CPU focal/plan fanout: exact TRAIN parents 001–013

Ready for root execution after the current FLARE seal and TRAIN014 recovery
admissions finish. This source calls the unchanged
`C3_next_plan_root_handoff_20261015_v1/focal_plan_bridge_v1.py`.
It changes no scientific producer, guard, configuration or GPU path.

Fanout SHA256:
`0bfe9c67c6a72704d61d15a53a1bdcd83475ba37f856dcb3e50aba1a69982d8a`.

## Root command

On anogena, from `/workspace-SR003.nfs2/ARG-GNN`:

```bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B research/reviews/C3_focal_plan_fanout_root_source_20261015_v1/fanout_v1.py --execute
```

A call without `--execute` prints source-only metadata and performs no guard,
admission, execution or process probe. The explicit run creates the fresh
`run_v1/` coordinator packet under this source directory. A different fresh
`--run-name` changes only that coordinator packet and does not permit parent
packet/output replay.

## Scheduling and failure behavior

Only `american_c3_short_train_001` through `013` are eligible. All thirteen
original bridge admission/output paths must be absent before the fanout starts.
The successful parent000 is never replayed.

Admissions run **one at a time**, holding the existing wave
`scheduler/admission.lock` around each actual admission subprocess. After an
actual admission wait0 and its exact returned path/SHA validate, that parent's
unchanged bridge `--execute` starts in a new owned process group. At most
**four scientific preparations** run concurrently. Admission remains sequential
while earlier preparations run.

All subprocesses use the project interpreter, repository Python paths and
`CUDA_VISIBLE_DEVICES=''`. The coordinator imports only the standard library.
Each independent monitor captures at most 65,536 bytes, drains excess stdout,
records the actual PID/start ticks/PGID/session and retains the actual direct
wait immediately before postwait logs, group checks or completion inspection.
Monitor threads keep doing this while the main thread admits another parent.

Any admission/worker/metadata/group error stops additions. Already-running owned
children finish and are genuinely waited. An admission already in progress may
finish, but no new preparation starts once the stop condition is observed.
An admitted-but-unlaunched scope remains recorded and preserved. SIGINT/SIGTERM
request this same stop-and-drain behavior; the coordinator sends no termination
signal to scientific children or unrelated processes.

Every successful phase requires fresh ESRCH for its exact owned process group.
A group still present after the 30-second postwait observation window is a
recorded failure, not a drain claim. No process inventory or historical
descendant-completeness claim is made.

## Actual evidence and outputs

Coordinator records live under:
`run_v1/american_c3_short_train_###/{admit,execute}/`.

Each actual subprocess has `launch.json`, immediately retained
`actual_wait_minimal.json`, bounded `stdout.log` and final `actual_wait.json`.
Only references to files actually produced are written. Actual spawn failures
or stopped-before-spawn work get distinct failure/not-launched metadata and no
manufactured wait. Total retained stdout for all 26 possible subprocesses is
at most 1,703,936 bytes, plus compact JSON metadata.

The original bridge still owns each fresh admission in
`C3_next_plan_root_handoff_20261015_v1/admitted_american_c3_short_train_###`
and scientific output in
`experiments/results/C3_short_TRAIN_focal_plans_v1/american_c3_short_train_###`.
Its 32 MiB reservation, current980 guard, exact focal enumeration, 10,000 focals,
all-tree/sample retention and unchanged plan builder remain intact. No native
MLM fitting, ancestry scoring or future model result is implied.

The final `run_v1/fanout_result.json` binds actual waits/completions and lists
failures, unfinished parents and admitted-but-unlaunched scopes. Inspect it and
the genuine terminal command status. No automatic retry, scope close, credit,
deletion or overwrite occurs. On failure, do not rerun this fixed full roster.
Root inspects preserved attempts and explicitly schedules any remaining fresh
parents through the existing bridge.

## Source preparation evidence

All 18 original bridge source/authority pins match. At inspection all thirteen
target packets/outputs were absent, and the existing output collection parent
was present. Parent000's actual `root_retry2_actual_wait.json` has direct wait0,
owned-group ESRCH and the exact real completion SHA
`8f4db164af3cd5ec8ccbb4faa49af49ece2aa9769a4934c5fcc38615e8ab3935`.
That producer reports 90.740942 seconds, 4,897 trees, 14,937 sites, 154 blocks and
10,000 focal records. These are preparation counts, not ancestry accuracy.

Source syntax and the source-only CLI passed. This worker performed no actual
admission, execution, guard/accounting operation, PID/group probe, GPU operation,
payload read or Git operation. No synthetic lifecycle campaign or new review
framework was added. Root owns the real fanout run.
