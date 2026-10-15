# Recovered pair018/019: stage-specific completed-scope close

This is a source-only successor to close_stage_v1.py. It preserves original
pair018/019 supervisor terminal1, all3genuine original stage waits, and the
recovery SSH transport failure255/session13056. SSH255 is not a remote helper
exit. Root observed existing helper3027397 and wrapper3027396 alive after the
transport loss. No restart or synthetic success is supported.

Actual known files are pinned exactly as written by root: immutable recovery
launch root_finish_exec_launch_identity_actual_v1.json; separate session13056
attachment root_finish_exec_launch_identity_with_tool_session_actual_v1.json;
and root_finish_SSH_transport_failure_actual_v1.json. The source reads the
existing field names, including actual_tool_exit_code255, null
remote_helper_returncode, nonempty transport_error and no_restart:true.
It does not rewrite or upgrade the transport observation.

The helper cannot close while the following genuine records are missing or
fail their checks:
- root_finish_remote_direct_wait_actual_v1.json: original immutable
  launch_identity_ref, sealed driver_ref, actual_remote_direct_Popen_wait_observed
  and remote_direct_Popen_wait_observed true, actual_supervisor_pid matching the
  helper, actual_remote_direct_Popen_returncode0, root_wrapper_pid matching the
  authenticated additional owned identity. Parent IDs are checked if recorded.
- root_recovery_finish_actual_terminal_v1.json: actual helper call return0,
  matching source/PID, no error, and result exactly equal to genuine019parent
  completion. Missing records remain unknown.
- All4genuine original stage worker direct waits0, group drains and original
  completion/observation identities. The3old waits must remain byte-identical.

The first3stages belong to old supervisor3011769. Only019infer belongs to the
new helper3027397. Parent018 completion belongs to the old supervisor;019
completion belongs to the new helper, while its simulation lineage remains
old. The full union of original and recovery supervisors, wrappers and workers
is checked with targeted PID and process-group absence before every close,
including additional wrapper groups. Any present/reused/zombie PID or group
blocks close.

Selected-stage observations reference their actual root context: old terminal1
and remote direct-Popen false for the3old stages; genuine recovery direct-wait0
and true for019infer. Every close observation/revocation also binds both
contexts and the immutable transport255 observation. No tool terminal0 is
required or claimed. The original failed pre-admission authority receives no
completion or credit.

The original generic completed-scope request, operation_draft_v1 call and
apply_close_completed_v1 call ASTs are unchanged. The8bounded metadata/path/
absence/input helper functions are unchanged. Actual980 policy/repair/
unmemoized mutation verification, current fresh-prefix checks, exact
reservation identity, preserved files and cooperative write/resume revocation
remain enforced. No policy edit, budget bypass, journal rewrite or science is
provided. Each per-scope evidence folder must be fresh.

## Root execution

Default mode is inert. Do not use --execute until the actual remote wait,
helper finish and019infer stage completion exist and root has audited them.
Run from /workspace-SR003.nfs2/ARG-GNN. Close each scope separately with the
same genuine parent-pair and recovery references.

The CLI flags use existing schemas. --recovery-exec-terminal means the actual
remote direct-Popen wait JSON, not the SSH255 transport JSON; the latter is
passed separately as --recovery-transport-terminal.

~~~bash
.runtime/env/bin/python -B research/reviews/American_C3_short_train2mb_wave_980_close_source_v1/recovered_pair018_019_v1/close_recovered_stage_v1.py --execute   --parent american_c3_short_train_018 --stage simulate   --exec-launch research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair018_019_exec_launch_identity_actual_v1.json   --expected-exec-launch-sha256 310556d7706d36bc12c1619eb0cfb4a2fb951fbe5f27c11e81d04377ef6241c9   --exec-terminal research/reviews/American_C3_short_train2mb_wave_980_source_v1/root_pair018_019_exec_terminal_actual_v1.json   --expected-exec-terminal-sha256 1843fef3eec71a83e3a35cd6e049e59acf40f8f96ca0a2446cf032b1be9221ba   --recovery-exec-launch research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_exec_launch_identity_actual_v1.json   --expected-recovery-exec-launch-sha256 68a197a78e1afd683bec5ae9d12ad8365db5a704e5245842c906c30fed478f39   --recovery-exec-terminal research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_remote_direct_wait_actual_v1.json   --expected-recovery-exec-terminal-sha256 ACTUAL_DURABLE_REMOTE_WAIT_SHA   --recovery-transport-terminal research/reviews/American_C3_short_train2mb_wave_980_source_v1/train019_infer_recovery_v1/root_finish_SSH_transport_failure_actual_v1.json   --expected-recovery-transport-terminal-sha256 fa96ca9b0e4dd9dc55fb2a5186291c63a63aa6eb2e610643de1a626a13f19dd1
~~~

Repeat with018infer,019simulate and019infer after each actual successful close.
Known exact SHAs are recorded below and in source_handoff_v1.json. The future
remote-wait SHA must come from its actual immutable file after completion;
no placeholder or prospective0receipt is supplied. Root owns genuine close
waits and final charge reporting. No close execution occurred in this worker.
