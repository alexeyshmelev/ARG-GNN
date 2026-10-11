# Supplemental closed-scope accounting v5

This successor is source-only until root obtains a separate independent review and explicitly applies a new admission. It changes no v4 or legacy source, config, admission, reservation, ledger, product, runtime or GPU authority.

V4 currently charges 947,527,367,424 decimal bytes: historical baseline 167,336,640,512, external reserve 130,000,000,000, all latest legacy job/controller allowances, and every permanent supplemental allowance. The last whole-project physical diagnostic (about 289.7GB) is a different quantity. V5 does not use that diagnostic or run a whole-project scan.

## Why a closed allowance can change

V4 reserve required each supplemental output and auxiliary scope to be absent and disjoint at admission. Each supplemental allowance caps all bytes retained in those scopes, rather than just an incremental write. The baseline remains fixed. For each unique proven closed supplemental key, v5 replaces its allowance R with the stat-measured retained charge A, where 0 <= A <= R:

    baseline + unchanged external reserve + all legacy charges
    + sum(open v4 allowances) + sum(closed v4 retained charges)
    + sum(open v5 allowances) + sum(closed v5 retained charges)

Equivalently, it adds all new v5 allowances to the full v4 projection and subtracts each distinct closed key's (R - A) once. Nothing is subtracted from the baseline, external reserve, legacy allowances, other scopes or any physical diagnostic. Legacy reservations are never closed by v5. A retained charge uses max(apparent bytes, allocated bytes) for every regular file and directory in the original output plus every original auxiliary scope; this is conservative scope accounting, not a claim of exact shared physical blocks.

The bound is contractual: it depends on cooperative project writers honoring the declared write scopes and root's revocation of future writes. External NFS clients and uninspected other-user processes are outside observation. V5 does not certify their absence. This is the same class of cooperative write-scope assumption as v4; it is not an OS quota or global proof.

## Immutable successor journal and admission barrier

Only new .runtime/storage/project_budget_v5/000001.json, 000002.json, etc. are written by an authorized apply. Events are exclusively created, read-only by default, hash-chained and never replaced. Each event binds the exact reviewed v5 script/launcher, pinned v4 source/config, human authorization, root admission, independent review and previous event hash. Closure measurements are pinned in the immutable root admission, so later journal edits cannot change a credit silently. A partial append or inconsistent chain blocks further use; preserve uncertainty evidence rather than repairing or rolling back automatically. The journal is limited to4096 events and32MiB. These bounded accounting records remain covered by the unchanged external write reserve; no external credit is claimed.

Root must freeze new legacy/v4 admissions before the first v5 event. The first root admission pins both historical ledgers. Later v5 checks validate all original v4 guards and also require those exact ledger bytes. Old schedulers cannot see v5 allowances, so they cannot continue issuing new reservations after v5 begins spending closure credits. Existing already-admitted consumers may finish under their original guards. Historical mutations after activation stop v5 scheduling.

Before activation, two new1GB v4 jobs fit within the current2,472,632,576-byte strict-ceiling headroom. Their combined projection would be949,527,367,424, leaving472,632,576 bytes. V5 is optional while source review proceeds.

## Truthful closure evidence

Root supplies a hashed completed.json at the original output root, exact completion identity (including its successful completed status), and a new terminal receipt. For detached H026/H027 launches, the accepted terminal status is exactly:

    evidence_method: root_observed_absence
    status: driver_completed_launcher_exit_unknown
    returncode: null
    exit_status_known: false

This preserves the missing OS wait status. Completion plus recorded PID absence does not become a fabricated zero exit code. A future launcher with an actual successful supervisor wait may instead use evidence_method=supervisor_wait and returncode=0.

The terminal receipt binds the complete original reservation hash, completed.json hash, observed UTC timestamp, the recorded worker and launcher PIDs, original GPU lease path/device/inode, and hashed JSON process, lease and GPU observation records. Root attests recorded_pids_absent, original_lease_exclusive_acquire_release_verified, and no_recorded_gpu_context, while explicitly setting external_NFS_writers_outside_observation=true.

Apply rechecks terminal PIDs and inspects local /proc for accessible writable descriptors, directory handles, writable mappings or working directories in the scope. Unreadable own-user/recorded processes, PID reuse, process-inventory drift and live writers block closure. Uninspectable other-user PIDs are reported as outside observation. Apply also acquires the original unchanged GPU mutex nonblocking under the existing scheduler mutex and closes it without writing its bytes. It performs no GPU query or GPU launch; the hashed GPU observation is root evidence.

Root must revoke the closed scope's future writes and relaunch for all cooperative project jobs. All original output and auxiliary scopes must be listed exactly. Missing auxiliary scopes count zero and are pinned as absent. The closure measurement is compared twice around writer inspection and pinned as expected_scope_measurement in the root admission.

Completed.json output_accounted_bytes is diagnostic only. It excludes later final metadata or may use a different counter; it is never the closure charge. Scope scans stat entries only and do not decode scientific products. Symlinks, hardlinks, unsafe namespaces, drift or oversize stop closure. Environments containing symlinks remain fully charged.

## Root interface

Source report and help perform no live ledger, scope, process or GPU inspection:

    source scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B scripts/project_budget_v5.py

An activated root admission must have kind=project_budget_v5_root_admission, root_authorizer=root_agent, active=true, operation=close or reserve, exact implementation/launcher/v4/human references, independent_review, and expected_before containing legacy_ledger_sha256, supplemental_ledger_sha256 and journal_tail_sha256. The review must be a separately supplied project_budget_v5_independent_review with role independent_reviewer, verdict PASS, the same references and actual_apply_performed=false. The author receipt is not that review.

Required root booleans are all_budget_writers_use_scheduler_mutex=true, all_new_consumers_use_v5_guard=true, legacy_and_v4_new_admissions_frozen=true, no_historical_record_mutation_authorized=true, global_rebase_authorized=false, GPU_authorized=false and scientific_execution_authorized=false. Closure adds scope_write_authority_revoked_for_all_cooperative_project_jobs=true, closed_scope_future_writes_and_relaunch_revoked=true, and external_NFS_writers_outside_observation=true.

A reserve admission contains one reservation with the same key/output/auxiliary_scopes/reserved_bytes/consumer_bindings fields as a v4 request. New scopes must be fresh and exclusive against every legacy, historical and successor scope, including closed scopes. Keys are permanent and cannot be reused. The combined historical/successor reservation map is capped at4096. New reservations and the journal never grant scientific, GPU or runtime authority.

A close admission contains closure with key, reservation_sha256 (SHA256 of v4 canonical JSON encoding of the entire exact stored reservation), completed reference, completed_identity, terminal_receipt reference, scope_paths, and expected_scope_measurement. Root can obtain the stat-only measurement by importing scripts.project_budget_v5.measure(ROOT, original_paths) after establishing terminal evidence. A changed measurement at apply blocks the event.

Root-only bounded inspect and apply are:

    "$ARGGNN_PYTHON" -B scripts/project_budget_v5.py --inspect --expected-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA
    "$ARGGNN_PYTHON" -B scripts/project_budget_v5.py --apply --expected-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA --admission ROOT_ADMISSION --expected-admission-sha256 ADMISSION_SHA

Inspect validates the ledgers/journal and existing closed scopes. Apply preserves original files and only appends one event. Capture stdout/stderr/return status outside the journal before executing. A receipt contains before view, resulting projection, appended event hash and full value, with historical_records_mutated=false.

## New consumer and launcher

New consumers import scripts.project_budget_v5 and call check_reservation(root, full_exact_v5_reservation, expected_source_sha256=..., expected_launcher_sha256=..., boundary=True, required_write_bytes=...). The function uses the existing canonical scheduler mutex, immutable historical guards and successor journal. It rejects closed reservations. Scope boundary plus planned growth must fit the original allowance. Initial preparation and stage/epoch boundaries should use boundary=True. Numerical training steps need not repeatedly revalidate filesystem sources. A boundary-free check validates records and authority; it relies on write revocation for previously closed scopes. Launch, admissions and boundary checks stat and validate all closed scopes.

The v2 launcher accepts only a root-issued project_budget_v5_storage_record with exact job_id, active=true, consumer_uses_v5_guard=true and full exact reservation. The launched module must be a pinned consumer source. It checks budget under the same scheduler lock immediately before GPU selection, preserves physical GPUs2/3/4/5, the two-job limit, existing eligibility checks and lease-FD/runtime environment. It never calls resources.storage_bytes or du. The consumer still checks its scientific and runtime authorities.

    "$ARGGNN_PYTHON" -B scripts/launch_reserved_gpu_v2.py --gpu GPU --job JOB --storage-record STORAGE --expected-storage-sha256 STORAGE_SHA --expected-budget-source-sha256 SOURCE_SHA --expected-launcher-sha256 LAUNCHER_SHA arggnn.NEW_CONSUMER [args]

The launcher records the actual child wait status on stdout and exclusively writes output/launcher_exit_v2.json after waiting, within the admitted scope budget. It makes no descendant-drain or closure claim. Missing output after an early failure leaves the allowance unchanged. Unknown legacy exit codes are never rewritten.

Any modified closed output, historical drift, inaccessible own-user process, active original lease, missing evidence, old-source mismatch or ceiling violation fails closed. Existing closure records remain immutable. A recovery that would reopen scopes or alter the historical admission barrier requires a separately reviewed successor; v5 does not force reclamation.

## Author validation

Only tiny handmade arithmetic, scope-stat, fake-/proc and journal cases were executed. The author inspected source/schema and bounded completed/launch metadata to establish feasibility, but did not stat actual consumer output trees, inspect live PIDs/GPUs, apply closures/reservations, install software, delete products, run scientific fits or change Git. Review and root activation remain separate. See research/reviews/project_budget_v5_author/test_report.json for the narrow executed test result.
