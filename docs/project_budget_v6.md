# Supplemental closed-scope accounting v6

This successor is source-only until root obtains a separate independent review and explicitly applies a new admission. It changes no published v5 packet, v4 or legacy source, config, admission, reservation, ledger, product, runtime or GPU authority. The independently rejected v5 bytes stay unchanged and are never treated as a passing review.

Root reported a pre-V6 projection of949,527,367,424 decimal bytes after two1GB v4 admissions. The earlier947,527,367,424-byte snapshot comprised: historical baseline 167,336,640,512, external reserve 130,000,000,000, all latest legacy job/controller allowances, and every permanent supplemental allowance. The last whole-project physical diagnostic (about 289.7GB) is a different quantity. V6 does not use that diagnostic or run a whole-project scan.

## Why a closed allowance can change

V4 reserve required each supplemental output and auxiliary scope to be absent and disjoint at admission. Each supplemental allowance caps all bytes retained in those scopes, rather than just an incremental write. The baseline remains fixed. For each unique proven closed supplemental key, v6 replaces its allowance R with the stat-measured retained charge A, where 0 <= A <= R:

    baseline + unchanged external reserve + all legacy charges
    + sum(open v4 allowances) + sum(closed v4 retained charges)
    + sum(open v6 allowances) + sum(closed v6 retained charges)

Equivalently, it adds all new v6 allowances to the full v4 projection and subtracts each distinct closed key's (R - A) once. Nothing is subtracted from the baseline, external reserve, legacy allowances, other scopes or any physical diagnostic. Legacy reservations are never closed by v6. A retained charge uses max(apparent bytes, allocated bytes) for every regular file and directory in the original output plus every original auxiliary scope; this is conservative scope accounting, not a claim of exact shared physical blocks.

The bound is contractual: it depends on cooperative project writers honoring the declared write scopes and root's revocation of future writes. External NFS clients and uninspected other-user processes are outside observation. V6 does not certify their absence. This is the same class of cooperative write-scope assumption as v4; it is not an OS quota or global proof.

## Immutable successor journal and admission barrier

Only new .runtime/storage/project_budget_v6/000001.json, 000002.json, etc. are written by an authorized apply. Events are exclusively created, read-only by default, hash-chained and never replaced. Each event binds the exact reviewed v6 script/launcher, pinned v4 source/config, human authorization, root admission, independent review and previous event hash. Closure measurements are pinned in the immutable root admission, so later journal edits cannot change a credit silently. A partial append or inconsistent chain blocks further use; preserve uncertainty evidence rather than repairing or rolling back automatically. The journal is limited to4096 events and32MiB. These bounded accounting records remain covered by the unchanged external write reserve; no external credit is claimed.

Root must freeze new legacy/v4 admissions before the first v6 event. The first root admission pins both historical ledgers. Later v6 checks validate all original v4 guards and also require those exact ledger bytes. Old schedulers cannot see v6 allowances, so they cannot continue issuing new reservations after v6 begins spending closure credits. Existing already-admitted consumers may finish under their original guards. Historical mutations after activation stop v6 scheduling.

The two1GB v4 admissions already used the earlier fallback headroom; the root-reported949,527,367,424-byte projection leaves472,632,576 bytes. These prose numbers are historical context, not action authority. Every action requires fresh exact ledger pins. Existing admitted H029 continues unchanged. No new historical admission freeze takes effect merely because source exists or review begins; it begins only at the first actual independently reviewed V6 event.

## Truthful closure evidence

Root supplies a hashed completed.json at the original output root, exact completion identity (including its successful completed status), and a new terminal receipt. For detached H026/H027 launches, the accepted terminal status is exactly:

    evidence_method: root_observed_absence
    status: driver_completed_launcher_exit_unknown
    returncode: null
    exit_status_known: false

This preserves the missing OS wait status. Completion plus recorded PID absence does not become a fabricated zero exit code. A future v3 launcher with an actual successful supervisor wait may instead use evidence_method=supervisor_wait, status=driver_completed_launcher_exit_zero_known, returncode=0, and exit_status_known=true. It must add supervisor_wait_evidence referencing output/launcher_exit_v3.json. The wait record must use kind=reserved_gpu_v3_supervisor_wait, bind the exact full reservation digest and recorded worker/launcher PIDs, and declare wait_status_known=true. Contradictory supervisor/unknown-exit tuples are rejected.

The completed record must contain positive, distinct integer runtime.pid and runtime.lease.launcher_metadata.launcher_pid. Missing identities block closure; arbitrary terminal PIDs cannot fill the gap. Both original IDs must be present in the bounded unique terminal pids list. The terminal receipt binds the complete original reservation hash, completed.json hash, observed UTC timestamp, the recorded worker and launcher PIDs, original GPU lease path/device/inode, and hashed JSON process, lease and GPU observation records. Root attests recorded_pids_absent, original_lease_exclusive_acquire_release_verified, and no_recorded_gpu_context, while explicitly setting external_NFS_writers_outside_observation=true.

Every recorded PID must be absent from the local /proc inventory. A present zombie, exited-state entry or reused PID is rejected rather than labeled absent. Apply rechecks terminal PIDs and inspects local /proc for accessible writable descriptors, directory handles, writable mappings or working directories in the scope. Unreadable own-user/recorded processes, PID reuse, process-inventory drift and live writers block closure. Uninspectable other-user PIDs are reported as outside observation. Apply also acquires the original unchanged GPU mutex nonblocking under the existing scheduler mutex and rechecks that the canonical path still names the original singly linked regular inode after flock and immediately before release. A replacement at either point fails closed; the held descriptor is always closed without writing its bytes. It performs no GPU query or GPU launch; the hashed GPU observation is root evidence.

Root must revoke the closed scope's future writes and relaunch for all cooperative project jobs. All original output and auxiliary scopes must be listed exactly. Missing auxiliary scopes count zero and are pinned as absent. The closure measurement is compared twice around writer inspection and pinned as expected_scope_measurement in the root admission.

Completed.json output_accounted_bytes is diagnostic only. It excludes later final metadata or may use a different counter; it is never the closure charge. Scope scans stat entries only and do not decode scientific products. Symlinks, hardlinks, unsafe namespaces, drift or oversize stop closure. Environments containing symlinks remain fully charged.

## Root interface

Source report and help perform no live ledger, scope, process or GPU inspection:

    source scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B scripts/project_budget_v6.py

An activated root admission must have kind=project_budget_v6_root_admission, root_authorizer=root_agent, active=true, operation=close or reserve, exact implementation/launcher/v4/human references, independent_review, and expected_before containing legacy_ledger_sha256, supplemental_ledger_sha256 and journal_tail_sha256. The review must be a separately supplied project_budget_v6_independent_review with role independent_reviewer, verdict PASS, the same references and actual_apply_performed=false. The author receipt is not that review.

Required root booleans are all_budget_writers_use_scheduler_mutex=true, all_new_consumers_use_v6_guard=true, legacy_and_v4_new_admissions_frozen=true, no_historical_record_mutation_authorized=true, global_rebase_authorized=false, GPU_authorized=false and scientific_execution_authorized=false. Closure adds scope_write_authority_revoked_for_all_cooperative_project_jobs=true, closed_scope_future_writes_and_relaunch_revoked=true, and external_NFS_writers_outside_observation=true.

A reserve admission contains one reservation with the same key/output/auxiliary_scopes/reserved_bytes/consumer_bindings fields as a v4 request. The bindings must include the actual src/arggnn/project_budget_v6_consumer.py adapter, actual launched consumer source, and original runtime/scientific/source authorities. The launcher requires its launched module to be one exact source binding. New scopes must be fresh and exclusive against every legacy, historical and successor scope, including closed scopes. Keys are permanent and cannot be reused. The combined historical/successor reservation map is capped at4096. New reservations and the journal never grant scientific, GPU or runtime authority.

A close admission contains closure with key, reservation_sha256 (SHA256 of v4 canonical JSON encoding of the entire exact stored reservation), completed reference, completed_identity, terminal_receipt reference, scope_paths, and expected_scope_measurement. Root can obtain the stat-only measurement by importing scripts.project_budget_v6.measure(ROOT, original_paths) after establishing terminal evidence. A changed measurement at apply blocks the event.

Root-only bounded inspect and apply are:

    "$ARGGNN_PYTHON" -B scripts/project_budget_v6.py --inspect --expected-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA
    "$ARGGNN_PYTHON" -B scripts/project_budget_v6.py --apply --expected-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA --admission ROOT_ADMISSION --expected-admission-sha256 ADMISSION_SHA

Inspect validates the ledgers/journal and existing closed scopes. Apply preserves original files and only appends one event. Capture stdout/stderr/return status outside the journal before executing. A receipt contains before view, resulting projection, appended event hash and full value, with historical_records_mutated=false.

## New consumer and launcher

New consumers import scripts.project_budget_v6 and call check_reservation(root, full_exact_v6_reservation, expected_source_sha256=..., expected_launcher_sha256=..., boundary=True, required_write_bytes=...). The function uses the existing canonical scheduler mutex, immutable historical guards and successor journal. It rejects closed reservations. Scope boundary plus planned growth must fit the original allowance. Initial preparation and stage/epoch boundaries should use boundary=True. Numerical training steps need not repeatedly revalidate filesystem sources. A boundary-free check validates records and authority; it relies on write revocation for previously closed scopes. Launch, admissions and boundary checks stat and validate all closed scopes.

The v3 launcher accepts only a root-issued project_budget_v6_storage_record with exact job_id, active=true, consumer_uses_v6_guard=true and full exact reservation. The launched module must be a pinned consumer source. It checks budget under the same scheduler lock immediately before GPU selection, schedules only GPUs2/3 under the current root authorization while counting all existing canonical project GPU leases toward the two-job limit, preserving existing eligibility checks and lease-FD/runtime environment. It never calls resources.storage_bytes or du. The consumer still checks its scientific and runtime authorities. A boolean consumer_uses_v6_guard and a pinned adapter are necessary metadata, not proof that calls exist. Root owns the concrete adapter and H030/H031 call sites, including preparation, stage/epoch boundaries, planned growth and every output/auxiliary scope. Their concrete source integration must be reviewed and verified before model execution. Author work does not claim this integration exists or passes.

    "$ARGGNN_PYTHON" -B scripts/launch_reserved_gpu_v3.py --gpu GPU --job JOB --storage-record STORAGE --expected-storage-sha256 STORAGE_SHA --expected-budget-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA arggnn.NEW_CONSUMER [args]

The launcher records the actual child wait status on stdout and exclusively writes output/launcher_exit_v3.json after waiting, within the admitted scope budget. It makes no descendant-drain or closure claim. Missing output after an early failure leaves the allowance unchanged. Unknown legacy exit codes are never rewritten.

Any modified closed output, historical drift, inaccessible own-user process, active original lease, missing evidence, old-source mismatch or ceiling violation fails closed. Existing closure records remain immutable. A recovery that would reopen scopes or alter the historical admission barrier requires a separately reviewed successor; v6 does not force reclamation.

## Terminal schema

The exact common terminal fields are kind, root_authorizer, reservation_sha256, completed, pids, observed_utc, recorded_pids_absent, original_lease_exclusive_acquire_release_verified, no_recorded_gpu_context, external_NFS_writers_outside_observation, lease_path, original_lease_identity, process_observation, lease_observation, gpu_observation, evidence_method, status, returncode, and exit_status_known. The supervisor method adds exactly supervisor_wait_evidence; the unknown-exit method does not. Extra or missing fields fail closed. Unknown exit remains null/false, and a supervisor wait remains zero/true.

The actual v3 wait file fields are exactly kind, job_id, launcher_pid, worker_pid, returncode, wait_status_known, completed_unix, reservation_sha256 and descendant_drain_or_no_live_writer_claim. It makes no descendant-drain claim.

## Review and API handoff

The API signature is:

    check_reservation(root, reservation, *, expected_source_sha256,
                      expected_launcher_sha256, boundary=False, required_write_bytes=0)

The source pin covers scripts/project_budget_v6.py and the launcher pin scripts/launch_reserved_gpu_v3.py. The journal is .runtime/storage/project_budget_v6 and kinds/schemas are project_budget_v6_event, project_budget_v6_root_admission, project_budget_v6_independent_review, project_budget_v6_terminal_receipt and project_budget_v6_storage_record. Record, reservation and return accounting fields otherwise retain their v5 shapes with the versioned reserved-byte diagnostic renamed v6_reserved_bytes.

V6 requires a separately supplied fresh independent PASS for these exact V6 implementation/launcher bytes, the pinned v4 source/config and human authority. The V5 FAIL, a changed kind label without matching references, or an author receipt cannot qualify. Old V5 evidence remains a source failure. Root alone supplies real closure evidence, admissions and consumer integration.

## Author validation

The34 inherited tiny accounting, scope, fake-process and journal cases were rerun because versioned schema/API paths and the shared terminal gate changed. Twelve focused additional cases exercise consistent and contradictory supervisor evidence, wrong recorded wait IDs, missing identities, exited-state PID presence, and lease replacement after open/before release plus a valid lease, and V6 review policy acceptance/rejection including the old V5 FAIL and relabeled wrong bytes. A prior zombie-acceptance test now requires rejection. All46 handmade tests passed. The seven independent V5 probe fixtures and critique were read to reproduce the blockers; no independent verdict is invented for V6.

Only tiny handmade arithmetic, scope-stat, fake-/proc and journal cases were executed. The author inspected source/schema and bounded completed/launch metadata to establish feasibility, but did not stat actual consumer output trees, inspect live PIDs/GPUs, apply closures/reservations, install software, delete products, run scientific fits or change Git. Review and root activation remain separate. See research/reviews/project_budget_v6_author/test_report.json for the narrow executed test result.
