# Additive TRAIN controller execution override

Source only. No guard/configure/apply, admission, GPU inventory, probe, job,
process inspection, signal, Git, credit or live-file edit was performed here.
The original pool, canonical dispatchers, launcher, C3/H050 admitted inputs,
configs, scientific sources, storage reservations and job mappings stay exact.

Actual diagnosed bug: original start holds scheduler.lock, then calls the
original storage guard, whose v4.mutex opens and locks scheduler.lock through a
different descriptor. Source override compiles exact original start and makes
only two replacements: prospective64KiB guard immediately before scheduler
flock, then an in-lock len(raw)+8192<=64KiB assertion in place of nested guard.
Every inherited launch/owned-tree/PIDFD/subreaper/drain/wait method is unchanged.
Fresh actual workers still enter ORIGINAL arggnn.gpu45_c3cache_h050_pool_v1.

A separate32MiB fully charged controller scope binds this execution source and
checker directly. Prior never-ready controller output remains charged and is
not closed or credited. Root must first consume the actual never-ready controller
tool terminal and provide genuine exact controller identity/termination evidence.

Controller metadata optimization: reviewed memo2 is installed only in the new
controller process and caches only inside each original mutex transaction. One
full original guard validates admission/accounting at startup BEFORE any
scheduler lock. Each later controller metadata check takes the original mutex,
rechecks expiry, source/policy and every direct bound consumer byte, validates the
unchanged original prefix and all root overlay event authorities, rejects any
close of its own scope, and invokes original v4.measure_scopes with the unchanged
conservative used+planned<=charged-cap inequality on its isolated output only.
It grants no credit/rebase/cross-scope growth. Later receipts explicitly state
global_projection_recomputed_this_check=false: this is not a claim that unrelated
historical scope proofs were rescanned before every metadata write. All future
global reserve/close mutations still use the original full guard; every scientific
worker guard remains unchanged. No timing or fit result is asserted.

Root commands, using PYTHONPATH=.:src:scripts and .runtime/env/bin/python -B:

1. Read/review source_handoff_v1.json and both source files.
2. Store actual never-ready controller terminal evidence with schema
   GPU45_never_ready_controller_actual_termination_v1 and exact fields in
   never_ready_termination.proposed.inactive.json (fill actuals; status is not
   required by this evidence schema). Keep actual tool exit code including
   nonzero/signal failure; do not relabel it as success.
3. Create NEW active root admission request from
   controller_admission.proposed.inactive.json with exact prior capability ref,
   actual termination evidence ref, reserved_bytes=33554432.
4. controller_execution_override_v1.py admit --execute --record REQUEST_PATH
   --sha256 REQUEST_SHA. It installs renewal through original load_cap, creates
   fresh reserve/plan/storage/capability, and checks jobs exactly equal old jobs.
5. controller_execution_override_v1.py start --execute --record RETURNED_CAP_PATH
   --sha256 RETURNED_CAP_SHA.
   This controller entry is the declared actual execution override.
6. Use SAME installed original launcher run/status/wait commands with RETURNED
   capability ref; SAME C3-GNN-cachev3-shared-TRAIN-v1 and
   H050-direct-readout-shared-TRAIN-v2 job names. Do not submit until actual ready
   status returned by this new start session.

11 pure override fixtures pass. Existing61 source-only downstream fixtures
include the same checker semantics and synthetic overflow/expiry/prefix/closure
failure cases. This packet does not assert reserve/start succeeded.
