# Exact completed candidate retained storage settlement

This separate procedure adapts the narrow, qualified GST storage-v2 contract for exactly two completed candidate keys. It references the prior source, review and sealed stat inventory without copying existing source or product data. Default invocation reports source bindings only. Author work has performed source and handmade-fixture checks only.

| Existing terminal key | Original charge | Permanent retained charge | Reduction |
| --- | ---: | ---: | ---: |
| biological_gpu2_v1/biological-candidate-seed202 | 29281048576 | 15000000000 | 14281048576 |
| biological_gpu3_v1/biological-candidate-seed303 | 29281048576 | 15000000000 | 14281048576 |

Total reduction is28562097152 bytes if the exact live gate qualifies. The sealed preliminary inventory projected448490318848 bytes. The unchanged-accounting proposal projects419928221696 bytes. These are historical baseline plus reservation projections, not fresh global consumption measurements.

The source pins the stat-only inventory receipt f39cf36c362ec3a040d70a4503b881a6bb988402960eaf0739197596f7775cfd and its ledger ddfdd905b645f48fbd7fcb7cafb3fc2c86bee801cc923abe37ba81d2b1d09e26. A changed ledger fails without adopting new charges or rebasing the baseline. Root would need a new explicit source/config admission and review after such drift.

## Root inspect, independent review, root apply

Root first issues a separate activated copy of root_inspect_admission_template.json with the exact sealed source/config references, active=true and operator_freeze_confirmed=true. This expresses the frozen-output operator contract below. Author templates remain inactive. Root records its exact JSON SHA256 and keeps outer stdout/stderr and exit evidence under a project scope.

    . scripts/project_env.sh
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" research/reviews/biological_completed_candidates_storage_settlement_v1_author/audit_and_settle.py --inspect --expected-source-sha256 <sealed-source> --expected-config-sha256 <sealed-config> --admission <root-inspect-admission.json> --expected-admission-sha256 <root-admission-hash>

Inspection writes only fresh fixed root_inspection audit files in this author directory. The live ledger remains byte-identical. Exact root admissions and source/config hashes are checked before live ledger/registry/PID or product access. The procedure verifies both original slots, exact succeeded terminal jobs against sealed job metadata, no registry safety error, all saved PID/start identities, and candidate receipt metadata. The receipt must be biological_candidate_fit_v1, completed_three_conditions_one_seed, the correct seed and source/config/observed/prepared pins, TEST_parents=0, the exact three model names, and completed positive optimizer fits for every model. Scientific arrays and labels are never decoded.

Under the existing canonical .runtime/locks/scheduler.lock, each retained output plus its exact worker log receives two full opaque byte-SHA256 and lstat membership inventories. They include every regular file and directory, apparent and allocated charges, device/inode/mode/size/mtime/ctime/link-count/uid/gid fields. The two passes must agree, and their complete stat membership must match the sealed preliminary inventory. Symlinks, multiply linked files, traversal, nonregular entries, walk errors, identity changes, content/stat drift and charges at or above15GB fail closed. The mutex is opened existing-only without creation, truncation or byte writes. Terminal jobs, receipt hashes and known identities are rechecked after inventory. The live ledger and frozen authority/source/evidence references are rechecked before a result.

A fresh independent reviewer then audits the exact source/config and root_inspection/inspection.json with its four full inventories. The reviewer emits the binding shape in independent_review_schema_template.json. The required kind is biological_completed_candidates_storage_settlement_v1_independent_review, role independent_reviewer, verdict PASS, exact procedure/config/inspection/prior-inventory/ledger/key references, operator_frozen_output_contract_accepted=true and actual_apply_performed=false. A qualified source/inspection PASS promises no apply success.

Root next issues a new activated root_apply_admission_template.json copy that binds the exact inspection and fresh independent PASS JSON hashes. Root alone runs the same CLI with --apply. Apply repeats both full inventories under the mutex, requires exact agreement with the reviewed inspection, rechecks terminal/PID/receipt and ledger bytes, and builds an inverse-verified two-field proposal. It changes reserved_bytes for exactly the two existing keys. The keys and exact output identities remain permanently present. Historical167336640512-byte baseline,130000000000-byte external reserve, pinned snapshot,450GB warning ceiling,500GB outer limit, every other job/controller charge and every other ledger field remain unchanged.

The reviewed pool atomic_json supplies the sole live ledger mutation. Root apply first preserves ledger_before.json and preapply_audit.json, then verifies the exact written proposal and records settlement_receipt.json. Product files, including every epoch probability array, checkpoint, metric and log, are retained. Errors after audit ownership preserve failure_receipt.json and partial audits. An atomic-write failure or post-write uncertainty is recorded without automatic rollback. A failure before audit ownership requires preserving the root outer launch log. Existing audit directories are never overwritten.

## Explicit operator qualifications

Completed candidate outputs and retained pool logs are contractually frozen. Any new scope writes require renewed accounting before authorization. New fits use fresh versioned output roots and reservations. Legitimate ledger writers honor scheduler.lock and pinned terminal registries are trustworthy. Known saved identity absence and stable inventories do not prove arbitrary unknown writers absent. Future unrelated side effects remain covered by the unchanged external reserve.

The freeze is contractual. The procedure supplies no OS quota or exclusion of a noncooperating writer. Saved identity coverage relies on the pinned trustworthy terminal registry. A reused PID with a different start tick is distinguished from a still-present saved worker. Active pool lifecycle and unrelated jobs/controllers can remain active, while their charges stay unchanged. Any target mutation or unknown scope writer suspicion requires renewed root accounting before reduction. The retained margin is15GB minus the measured conservative scope for each seed. It is not permission to write more finished product data.

Author tests use only synthetic outputs, registries, PID-return values and ledger snapshots under this directory’s own temporary roots. They include inspect-only and atomic apply behavior on fake ledgers, provenance corruption, all model-completion gates, symlink/hardlink/traversal/walk failures, membership/content/stat drift, stale or altered ledger slots, fresh-review/admission rejection, PID reuse and preserved atomic failure evidence. No live registry/PID/ledger/product inventory, real inspect/apply, package installation, GPU or Git operation is performed by author work.
