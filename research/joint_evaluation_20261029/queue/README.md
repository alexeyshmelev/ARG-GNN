# Exact-scope parallel queue

The prepared ARGformer queue has 35 unchanged native stages for parents003–009.
Only storage-order dependencies and arbitrary wall-clock limits change. Two
lanes advance after retirement:003→005→007→009 and004→006→008. Every native
output path, scientific parameter, runtime, disk cap and RAM cap stays pinned.

The original controller PID1005327 and MLM PID1778913 remain untouched. Launch
must occur while the sealed MLM identity is still running. The launcher creates
seven exact preparation-root claims; the original scheduler's existing
no-overwrite guard then prevents duplicate execution. Partial claims are retained
for review on failure; there is no automatic retry, claim removal or replay.

The controller allows GPUs4/5 and one of its jobs per GPU.24GiB admission stages
can share with the original52GiB contrastive reservation. Sidecar52GiB stages
wait until original parent002 prediction has a real successful wait and original
parent003 preparation is blocked by our owned claim. RAM admission reserves an
additional960GiB for the original CPU/GPU chain, the separate256GiB ARGMix native
job and64GiB other reviewed writers. No old task is signalled or mutated.

The root-confirmed storage ledger is authenticated by the plan. All35 full-stage
disk caps, the entire ARGMix24GiB auxiliary cap plus native output cap and1GiB
metadata are charged before admission. Total commitment is1,117,001,926,586 bytes,
below the exact1,200,000,000,000-byte project ceiling. The original995GB ledger
and historical cleanup evidence stay unchanged.

## Review and root launch

Run on anogena in `/workspace-SR003.nfs2/ARG-GNN`. The script has no job launch
side effects until `sidecar_scheduler` is invoked without `--validate-only`.

```bash
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.sidecar_scheduler --plan research/joint_evaluation_20261029/queue/sidecar_plan.json --validate-only
mkdir -p research/joint_evaluation_20261029/queue/records_v1
nohup .runtime/env/bin/python -B -s -u -m research.joint_evaluation_20261029.queue.sidecar_scheduler --plan research/joint_evaluation_20261029/queue/sidecar_plan.json > research/joint_evaluation_20261029/queue/records_v1/controller.log 2>&1 < /dev/null &
```

Root records the returned shell PID and authenticates its `/proc/PID/stat` start
ticks, then checks `records_v1/state.json`, `claims.json` and controller.log. Do
not rerun the launch after uncertain delivery. The source and source-pin receipts
are `focused_tests.json`, `proposal_receipt.json`, `argformer_source_pin_audit.json`
and `sidecar_scheduler.diff`. Tests only create short-lived queue-local fixtures.
They do not launch workers, claim task roots, mutate live state or send signals.

## Future reconciliation

There is no live import into the unchanged original controller. Both queues must
own no running children before reconciliation. Preserve sealed original state,
sidecar state, task logs, result artifacts and every actual_wait.json. Do not stop
a controller that owns live work. The future merge helper validates real waits,
exact result scopes, empty process sessions and original plan/revision identity:

```bash
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.future_merge --original-state .runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json --sidecar-state research/joint_evaluation_20261029/queue/records_v1/state.json
```

It writes a proposal only. It never changes state or fabricates a wait. The old
claim-blocked rows are retained in the adoption audit. Only never-launched rows
blocked by those claims may return to pending; genuine failed TreeLoom history
and branches depending on it remain unchanged. Existing continuation.py resets
all failed/blocked rows and cannot implement this adoption safely. A reviewed
continuation must seed exactly the proposal rows, authenticate all sidecar waits,
verify old controllers and process sessions have drained, and refuse replay of
any nonempty pending output. Parents005–009 already execute autonomously in this
sidecar and do not require this later merge to continue.

The separate ARGMix native queue and CPU CNN readiness report are prepared
separately. Their names and exact launch instructions are added after review.

## Separate ARGMix native CPU queue

`argmix_plan.json` retains the exact original native output and auxiliary scope.
The unchanged `paper_argmix_relate.run_parallel` executes every native command
and records every real native wait. `argmix_native_wrapper.py` adapts only the
fresh-directory check for its exact owned marker and removes subprocess wall
timeouts. It preserves all scientific arguments, source/config pins, section
parallelism, random seeds, byte caps and RAM limits. Twenty-two focused tests and
149 source-pin checks passed. The native output cap is 276,824,064 bytes, native
work cap 25,769,803,776 bytes and RAM cap 274,877,906,944 bytes.

The claim helper verifies the original native task is still storage blocked,
its prepared-input result is genuinely complete, the contrastive002 identity is
still running, and the original reservation plus candidate growth exceeds the
old 995 GB ceiling. It atomically creates only native_run/SIDECAR_CLAIM.json.
It never changes original state, creates native inferred data, or sends signals.

Root launch after review uses this readable command on anogena:

```bash
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.argmix_scheduler --plan research/joint_evaluation_20261029/queue/argmix_plan.json --validate-only
mkdir -p research/joint_evaluation_20261029/queue/argmix_records_v1
nohup .runtime/env/bin/python -B -s -u -m research.joint_evaluation_20261029.queue.argmix_scheduler --plan research/joint_evaluation_20261029/queue/argmix_plan.json > research/joint_evaluation_20261029/queue/argmix_records_v1/controller.log 2>&1 < /dev/null &
```

Record the returned PID and start ticks. Inspect `argmix_records_v1/state.json`
and the claim receipt before any subsequent action. The plan charges all 35
ARGformer stage caps and metadata independently; the native queue and ARGformer
queue have the identical 1,117,001,926,586-byte combined commitment. This queue
reserves 1,216 GiB future RAM for the ARGformer lanes, original controller and
other reviewed writers. Never replay an existing output or modify the live
original controller to import this result.

The initial MLM002 guard completed genuinely while the proposal was being
reviewed. Root approved rebinding claims to contrastive002 PID 1976765, start
ticks 591731069. The final launched ARGformer plan is SHA-256
15519ffa4936674d80a6d9d5e5177148934687694416bec862b1a3b8ab284bbb.
The scheduler and ownership helpers were unchanged by that identity refresh.

## Direct four-class author CNN baseline

`cnn4_plan.json` adapts the frozen fifteen-task factory plan without changing any
native scientific spec, source pin, dependency, output path, byte cap or RAM cap.
It re-extracts 5×8 features from existing observed trees, fits fresh four-output
weights on the original 40,000 non-BAA training rows for five epochs, predicts the
five original TEST parents and scores only after predictions complete. This is
the separately trained AP four-class baseline; direct comparison to a four-class
TreeLoom requires its own matched fit. The original six-class outputs remain.

Root confirmed a 640 MiB suballocation within the existing 1 GiB joint-evaluation
allowance: 384 MiB native output caps plus 256 MiB metadata. Both evaluation
records and CNN output scopes are checked together. The CPU controller has one
lane, a 64 GiB maximum preparation reservation, an additional 1,472 GiB reserve
for other reviewed queues and the inherited 64 GiB host headroom. The native
TensorFlow 2.15.1 / Keras 2.15.0 runtime was imported successfully with CUDA hidden.
All 38 direct frozen input references were hashed successfully (2.438 GB).

Root launch after review, on anogena:

```bash
source scripts/project_env.sh
export PYTHONDONTWRITEBYTECODE=1
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.cnn4_scheduler --plan research/joint_evaluation_20261029/queue/cnn4_plan.json --validate-only
mkdir -p research/joint_evaluation_20261029/queue/cnn4_records_v1
nohup .runtime/env/bin/python -B -s -u -m research.joint_evaluation_20261029.queue.cnn4_scheduler --plan research/joint_evaluation_20261029/queue/cnn4_plan.json > research/joint_evaluation_20261029/queue/cnn4_records_v1/controller.log 2>&1 < /dev/null &
```

No source or plan used by an already running queue was changed to add this CPU
baseline. `cnn4_tests.json`, `cnn4_readiness.json`, `cnn4_proposal_receipt.json` and
`cnn4_scheduler.diff` provide the verification and exact review delta.

For later original-controller adoption of both native sidecars, the merge helper
accepts multiple states. Invoke only when all owned children have drained:

```bash
.runtime/env/bin/python -B -m research.joint_evaluation_20261029.queue.future_merge --original-state .runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json --sidecar-state research/joint_evaluation_20261029/queue/records_v1/state.json --sidecar-state research/joint_evaluation_20261029/queue/argmix_records_v1/state.json
```

The proposal includes exact effective task definitions and genuine wait refs.
External prepared-input evidence remains the original successful row, never an
invented native wait. Eight additional merge tests verify that unrelated blocked
history cannot be reset. The CNN namespace is additive and separate; it is not
silently inserted into original-controller history.
