# Root close glue: actual VAL simulation/inference and FLARE conversion

Source-ready for root execution. All three scopes genuinely completed according
to their pinned original producer completion and actual direct-wait/drain JSON.
This agent validated metadata only and performed no guard configuration/state
query, PID/group probe, owned-scope inventory, close or Git action.

close_completed_v1.py uses the existing prospective_980gb_v1 proof API directly.
It handles only these three existing exact reservations:

| Scope option | Original key | Original cap |
|---|---|---:|
| VAL_simulate | H041/American-C3-val000-full30mb-simulate-shared-980-v1 | 6500000000 |
| VAL_infer | H041/American-C3-val000-full30mb-infer-shared-980-v1 | 12500000000 |
| FLARE_convert | FLARE/TRAIN000-convert-980-root-v1 | 950000000 |

actual_refs.json pins every actual admission/storage/launch/direct-wait/minimal
wait/outer-monitor/completion and relevant IO source ref, plus the exact active
guard/policy/repair/core/base-launcher refs. All scientific payload refs inside
completion metadata stay unopened.

Source SHA:
fff7bbc9e5152573be83ccc50481bf86b40a888dc770068af81f43a64d7d82fa

## Root commands

Run from /workspace-SR003.nfs2/ARG-GNN on anogena. Each command explicitly closes
one actual scope. Default invocation without --execute is a bounded metadata
review and performs no guard or process operation.

~~~bash
.runtime/env/bin/python -B research/reviews/VAL_FLARE_completed_close_source_20261015_v1/close_completed_v1.py --execute --scope VAL_simulate
.runtime/env/bin/python -B research/reviews/VAL_FLARE_completed_close_source_20261015_v1/close_completed_v1.py --execute --scope VAL_infer
.runtime/env/bin/python -B research/reviews/VAL_FLARE_completed_close_source_20261015_v1/close_completed_v1.py --execute --scope FLARE_convert
~~~

Stop and inspect any failed invocation before continuing. Each default output
is a separate closed_SCOPE_v1 directory in this helper packet. Existing or partial
directories are never overwritten. Root can select a fresh --output-packet for
a failed pre-close attempt after investigating it. An actual already-closed key
is rejected before new evidence-directory creation and must never be replayed.

## Preserved proof semantics

For VAL, the guard direct_wait_ref is the original actual_wait.json with actual
direct scientific-worker wait0, same Popen launch/admission/PID, no operational
error and owned worker-group ESRCH. The original minimal wait is independently
bound and checked. The root IO supervisor PID and all observed worker identities
enter the complete target roster.

For FLARE, the original launch_monitor_seed.json is the actual launch identity.
The guard direct_wait_ref is actual_wait_minimal.json, genuinely persisted after
direct worker wait0. The original outer_wait.json must independently show wait0,
no error, actual seed/release lineage and all observed owned-group ESRCH.
The outer supervisor PID and every observed worker identity/group are included.

Root's execute mode requires fresh targeted ESRCH for all known source-owned
supervisor/worker PIDs and groups, writes explicit cooperative original-scope
launch/resume/retry/write revocation while retaining read-only reuse, and calls
the unchanged apply_close_completed_v1 API. The guard again checks exact original
reservation/evidence and PID absence and performs its own single owned-scope
stat inventory. Only actual unused allowance is credited. No credit amount or
future capacity is asserted by this source handoff.

Known PID/group rosters for root's future observations:

| Scope | Root plus worker PIDs | Worker groups |
|---|---|---|
| VAL_simulate | 2884705, 2884863 | 2884863 |
| VAL_infer | 2896301, 2896608 | 2896608 |
| FLARE_convert | 2894660, 2895133 | 2895133 |

The preserved simulation RSS_observer_correction.root_v1.json explicitly voids
the original zero peak field. Zero is not a measured RSS peak, and its targeted
295751680-byte observation is not a full-run peak. The source keeps this
correction in close evidence and makes no full-run RSS claim. Genuine direct
wait/deadline/private-work evidence is preserved.

The helper claims no remote outer Python Popen.wait that did not occur. Its
guard direct-wait evidence is the genuine original worker wait in each case.
Root supplied the terminal status separately and the helper checks exact
supervisor absence before close.

The existing wave admission lock serializes this accounting operation with wave
admissions. Fresh actual 980 state and the original unmemoized v4 functions are
required. Guard semantics, historical journal, live wave/H048 source/configs and
all old scientific files remain unchanged. No deletion, new reservation,
scientific execution, native/GPU launch or future completion is introduced.
Subsequent 9 GB native FLARE or 16 GB VAL native-path admissions remain root's own
fresh capacity decisions after actual close receipts.

source_static_receipt.json records 13 source/schema and actual metadata checks.
The three metadata chains all validate. No synthetic future completion or
terminal artifact was created, no payload was read, and no root operation was
executed by this agent.
