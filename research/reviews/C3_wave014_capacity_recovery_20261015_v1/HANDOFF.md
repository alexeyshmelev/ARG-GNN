# Only014 inference: capacity recovery and truthful pair closure

Ready for root after its completed FLARE-native close actually restores the
2-GB inference reservation plus current event overhead. No ceiling, guard,
scientific seed/config/output or original driver change is proposed. This worker
has not moved files, probed processes, admitted, launched or closed anything.

## Actual failure and retained work

Root’s actual session50286 terminal has SHA256
`66566bf7350639926c99567816c3eb443cf823ac9ddb2090652d5c5801f6528b`.
Its schema is `root_actual_exec_session_terminal_v1`, actual tool/SSH exit1,
remote direct Popen.wait observed/claimed both false, and actual root terminal
time unknown. The exact failure is `admit_stage_v1 -> guard.apply_reserve_v1 ->
charge` for `american_c3_short_train_014/infer`, with
`Prospective980GB ceiling/1TB human limit exceeded`.

The pinned guard calls `charge` before `_append('reserve',...)`. The original
014 inference packet contains **only** `config.json` and `reserve_authority.json`.
There is no receipt/admission/launch/wait/worker_aux or inference output. This
is a rejected reservation draft, not a failed scientific inference.

014 simulation and both015 stages have genuine direct scientific worker wait0,
process-group drain, no operational error and real producer completions. The015
parent terminal exists. Their supervisor is original PID2903577.014 parent
terminal is absent. [Actual metadata snapshot](actual_metadata_snapshot.json).
The original 014 seeds remain ancestry22015/mutation1022015/path92215, and015
seeds remain22016/1022016/92216. Both retain1600 haplotypes,2Mb, original unary
preprocessing and eight tsinfer threads.

## Exact root recovery commands

Run on anogena from `/workspace-SR003.nfs2/ARG-GNN`. First wait for the root’s
already-running FLARE-native close to return successfully. The next original
admission performs the live capacity check. Ledger projection is not measured
physical disk usage.

```bash
source scripts/project_env.sh
"$ARGGNN_PYTHON" -B research/reviews/C3_wave014_capacity_recovery_20261015_v1/preserve_failed_packet_v1.py --execute
```

This helper validates the actual failed terminal, three retained successful
stages, the exact two failed-draft hashes and absent014 inference output. Root
execution also requires targeted ESRCH of the original supervisor/group and any
explicit additional roots. It records revocation of the rejected draft’s future
apply/retry/write capabilities, then atomically renames **only** the two-file
failed packet into `preserved_failed_infer014_v1` in this folder. Its bytes and
all prior outputs/waits remain unchanged. No guard is configured and no
accounting changes. Do not apply the archived authority.

Only after that actual preservation command succeeds:

```bash
"$ARGGNN_PYTHON" -B research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_wave_980_driver.py \
  --run-parents american_c3_short_train_014 \
  --policy research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json \
  --policy-sha eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018
```

This is the unchanged original f55f1162 driver. Its existing partial-parent path
validates/skips014’s successful simulation, admits/launches only014 inference
into the original canonical inference output/key, then writes the missing014
parent terminal. **Do not include015**, rerun simulation, reuse a failed draft,
or use the old pair command. If this new admission fails, preserve that new
partial attempt and stop. This helper authorizes no automatic retry.

Root must retain the new actual exec session and observe its genuine terminal0.
Preserve a new launch identity JSON in this folder with kind
`actual_exec_session_launch_identity`, exact original driver_ref, parent_ids
`["american_c3_short_train_014"]`, actual exec_session_id, remote supervisor
PID/start ticks/group, explicit additional root PID list, and
all_additional_root_owned_remote_processes_listed=true.

Preserve its actual terminal using the same honest schema as root’s failed
receipt: `root_actual_exec_session_terminal_v1`, root_authorizer=root_agent,
actual_exec_session_terminal=true, matching exec_session_id,
actual_exec_session_exit_code=0, actual_SSH_propagated_returncode=0,
actual_root_pid/start_ticks/process_group matching that launch,
exact_original_root_source matching f55f1162, and both remote_direct_Popen_wait
observed/claimed=false. Keep unknown terminal time unknown. No future launch,
terminal, admission or completion receipt is supplied by this packet.

## Pair close after recovery: both roots, four actual stages

The original pair-close helper cannot be used unchanged: it requires session50286
exit0 and one common supervisor. Do not relabel its genuine exit1 as success.
`close_recovered_pair_v1.py` is narrow root IO for014/015 only. It validates the
original failed root plus the actual014-only successful recovery, and maps
**only014 inference** to the recovery supervisor.014 simulation and both015
stages stay bound to PID2903577 and their unchanged waits. Both parent terminals
must bind the correct stage refs/root. All four stages still require genuine
wait0, complete group drain, no capture/log errors and unchanged completions.

The adapter reuses the original close `run` bytecode, original fresh980 guard,
request identities, global/wave locks, targeted PID/group absence, scope
measurement, revocation and `apply_close_completed_v1`. Both root rosters and all
four worker rosters enter absence checks. Both actual terminal refs are retained
in revocation/observation metadata. It adds no guard, credit rule or scientific
fallback. The original failed draft remains archived. Each invocation closes
one exact original reservation.

After genuine recovery terminal0 and all four actual successes, replace the two
filename values below with the actual new root receipts (the suggested names
are not pre-created):

```bash
wave014_recovery_launch=research/reviews/C3_wave014_capacity_recovery_20261015_v1/root014_recovery_exec_launch_identity_actual_v1.json
wave014_recovery_terminal=research/reviews/C3_wave014_capacity_recovery_20261015_v1/root014_recovery_exec_terminal_actual_v1.json
wave014_recovery_launch_sha="$(sha256sum "$wave014_recovery_launch" | cut -d ' ' -f 1)"
wave014_recovery_terminal_sha="$(sha256sum "$wave014_recovery_terminal" | cut -d ' ' -f 1)"
for wave014_parent in american_c3_short_train_014 american_c3_short_train_015; do
  for wave014_stage in simulate infer; do
    "$ARGGNN_PYTHON" -B research/reviews/C3_wave014_capacity_recovery_20261015_v1/close_recovered_pair_v1.py \
      --execute --parent "$wave014_parent" --stage "$wave014_stage" \
      --recovery-exec-launch "$wave014_recovery_launch" --expected-recovery-exec-launch-sha256 "$wave014_recovery_launch_sha" \
      --recovery-exec-terminal "$wave014_recovery_terminal" --expected-recovery-exec-terminal-sha256 "$wave014_recovery_terminal_sha" || break 2
  done
done
```

Default output folders are `closed_014_simulate_v1`, `closed_014_infer_v1`,
`closed_015_simulate_v1`, `closed_015_infer_v1` here. Preserve partial attempts.
Use a fresh `--output-packet` only after investigating a pre-close failure.
Already-closed keys reject. No future credit amount is asserted.

##016–019 restart condition

Only after all four014/015 original scope keys are genuinely in the actual980
closed set may root run the unchanged original driver for016/017. Actual stage
completion alone, session50286 exit1 or the preservation receipt is insufficient.
The original driver independently enforces successful prior stages and actual
closed keys, with at most two open parents. After016/017 both genuinely complete,
root must preserve their actual successful pair launch/terminal and use the
existing original pair-close helper to close all four scopes. Only then may018/019
start. No completed015 stage or parent is replayed to manufacture a new pair exit0.

```bash
"$ARGGNN_PYTHON" -B research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_wave_980_driver.py \
  --run-parents american_c3_short_train_016 american_c3_short_train_017 \
  --policy research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json \
  --policy-sha eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018
```

Use the same original command with018/019 only after the016/017 closes. Neither
command is authorized by a projected future capacity estimate. Root owns actual
admissions, launches, closure accounting, ledger/status updates and Git.

## Qualification and limits

[Source qualification](source_qualification.json) records compilation, inert
default CLIs, exact acceptance of root’s actual failed-terminal shape, all three
retained actual successful stages, exact partial packet/absent output, and
rejection of the failed pair as a successful recovery. Complete future two-root
close validation cannot run before actual recovery receipts exist. No synthetic
future terminal/completion was written. Three source files are confined here,
with the common helper hash pinned by both entrypoints and the original driver
and original close helper pinned by SHA. No live/frozen source was changed.
