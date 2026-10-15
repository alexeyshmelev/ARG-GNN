# Root-only prospective980 close IO for short TRAIN 014–019

close_stage_v1.py is a separate inactive source helper. It closes one explicitly
selected stage per root invocation, after validating the entire two-parent pair.
This agent performed no guard import/configuration/state mutation, PID/group
probe, scope inventory, close, GPU/model execution or Git operation.

The active prospective980 guard already supports exact completed-scope request
locators. The previous wave close helper was hardcoded to 012/013, old-v3 guard
and an old remote Popen wrapper. The new helper preserves the original guard
and its request/launch/wait/completion identity requirements while adapting root
IO to the actual available exec-session evidence.

Live driver SHA remains
f55f1162e12302c712ab96c810dcee81f8063dd170fd54d3d19e98feb82c0df6.
The new close source SHA is
854799fed7c3bee12acf4c025de5cee9cd59742730f905aaca3ece6d696965c7.
No live driver, config, producer, guard, policy, old event or completed artifact
is patched. Existing read-only reuse is preserved.

## Evidence distinction

The first pair was launched through exec session 50286. There was no explicit
remote root-owned Python Popen wrapper. Root will preserve the genuine terminal
tool exit/status/stdout once it exists. SSH propagates the remote CLI result.
The helper requires that honest evidence and explicitly forbids claiming a
remote direct Popen.wait. It never creates a terminal or completion receipt.

Each stage's original actual_wait.json remains the guard direct_wait_ref.
That evidence must genuinely record the direct scientific worker wait0, same
actual PID/launch/admission, complete worker-group drain and no operational,
capture or log-materialization error. Both parents and all four stage receipts
must be complete. The entire source-owned root/worker PID roster, all explicitly
listed additional root-owned remote PIDs, the remote supervisor group and each
worker group must have fresh targeted ESRCH. Present/reused/zombie/inaccessible
identities block. The original guard repeats exact PID absence and measures the
closed scope once before crediting only unused allowance.

No inference is made from process completion alone. The SSH terminal record,
genuine per-stage waits and actual PID/group observations have separate fields.
The scope revocation certificate preserves those distinctions and revokes only
the selected original scope's old launch/resume/retry/write capabilities.

## Actual launch record

Root already preserved the following bounded metadata (read by this agent as
JSON only, with no process probe):
research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair 014_015_exec_launch_identity_v1.json
SHA 2c85c37c4211bff26b85da3e3038becd2a8041b2941388cebd50252858802c09

It records supervisor PID 2903577, start ticks 475059625, process group 2903577,
exact frozen driver and session 50286. Its terminal flag is false. The field
remote_supervisor_start_ticks accepts the actual integer. The field
additional_root_owned_remote_pids is an integer list and is currently empty.
If a future launch has its own remote shell parent, root must include its actual
PID in that list and preserve start/role provenance in additional metadata.
Every listed PID enters fresh absence checks and the guard's complete roster.

## Genuine terminal record supplied by root

Only after actual terminal tool output, root must preserve a separately pinned
JSON receipt with these required fields. This is a schema description, not an
actual or pre-filled future receipt.

- kind: actual_exec_session_terminal
- root_authorizer: root_agent
- launch_identity_ref: exact actual launch JSON path/SHA
- driver_ref: exact frozen f55f1162 source ref
- parent_ids: exact ordered pair 014/015,016/017 or 018/019
- exec_session_id: same actual launch session
- actual_exec_session_terminal_observed: true
- remote_cli_status_propagated_by_ssh: true
- remote_direct_Popen_wait_observed: false
- tool_result: the genuine final tool result, retaining exit_code 0 and output
  as the exact terminal stdout/status string

A nonzero exit, missing terminal, incomplete parent/stage, failed actual wait,
live/reused PID/group or mismatched source/reservation blocks before close.
The helper does not synthesize or repair unavailable evidence.

The launch JSON for future pairs uses kind actual_exec_session_launch_identity,
root_authorizer root_agent, exact driver_ref and parent_ids, exec_session_id,
remote_supervisor_pid, remote_supervisor_start_ticks,
remote_supervisor_process_group, additional_root_owned_remote_pids and
all_additional_root_owned_remote_processes_listed=true.

## Root commands after genuine first-pair termination

Run on anogena from /workspace-SR003.nfs2/ARG-GNN. The terminal filename below is
a proposed destination for root's actual receipt. It does not exist by virtue of
this handoff. Do not run until genuine terminal 0, four completed stages and root
PID/group absence are available.

~~~bash
wave980_launch=research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair 014_015_exec_launch_identity_v1.json
wave980_terminal=research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair 014_015_exec_terminal_v1.json
wave980_launch_sha=2c85c37c4211bff26b85da3e3038becd2a8041b2941388cebd50252858802c09
wave980_terminal_sha="$(sha256sum "$wave980_terminal" | cut -d ' ' -f 1)"

for wave980_parent in american_c3_short_train_014 american_c3_short_train_015; do
  for wave980_stage in simulate infer; do
    .runtime/env/bin/python -B research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/close_stage_v1.py       --execute --parent "$wave980_parent" --stage "$wave980_stage"       --exec-launch "$wave980_launch" --expected-exec-launch-sha256 "$wave980_launch_sha"       --exec-terminal "$wave980_terminal" --expected-exec-terminal-sha256 "$wave980_terminal_sha" || break 2
  done
done
~~~

Each explicit invocation closes one exact 1 GB simulation or 2 GB inference stage.
Default administrative outputs are separate closed_NNN_STAGE_v1 folders under
this close-helper packet. Existing/partial folders are preserved. Root can pass
a fresh --output-packet for a failed pre-close attempt after investigating the
cause. A scope already closed in the actual journal is rejected before new
evidence-directory creation; never close or manufacture it again.

The helper holds the existing wave admission lock, binds fresh actual 980 state
for each operation, uses original unmemoized v4 functions and calls the unchanged
apply_close_completed_v1 API. It never modifies the live wave driver or creates
a new reservation. The next pair's admission gate independently requires every
prior-pair stage key to be genuinely closed.

Use the same explicit source command for parents 016–019 only after their own
genuine launch/terminal receipts and pair completions. The existing live driver
and all 40 configs/20 IDs/seeds/outputs are unchanged.

source_static_receipt.json records 17 source/AST/schema checks. The guard request
field set, exact wait identity AST and completion identity AST match the previous
close helper. No future terminal or scientific completion fixture was created.
