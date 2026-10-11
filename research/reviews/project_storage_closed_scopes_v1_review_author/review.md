# Narrow closed-scope storage adapter review

Reviewed source SHA256 b168b8e1acd4407df49f9d64d58b85503416ac02e8dd5a979f81df55914f76a2 against unchanged project_storage_full_reservation_v1.py SHA256 c0fdace58e26020c8227606e9b77325bc7308e7ec6691fe3a7e7c18cd4846105. No blocking finding under the stated cooperative certified-closure contract. No journal mutation/read, real scope measurement, global disk/process/descriptor scan, data job or GPU action occurred in this review.

## Concrete difference and retained safeguards

The additive module requires its own pinned source and accounting_mode journal_closed_scopes_credit_v1. Instead of reconstructing the sum v4_projected_bytes + v9_reserved_bytes, it uses the existing V9 core's projected_bytes and reports the existing closed_allowance_credit_bytes. It retains the explicit1TB rejection and measures only successor_paths() of this active reservation for requested writes.

The unchanged V9 reconcile() subtracts each unique certified closed reservation's allowance-minus-recorded-retained-bytes once. The adapter does not synthesize credits, close a scope, modify ledgers, release a running reservation or authorize closed-scope writes. check_reservation_locked() still verifies the reviewed implementation/launcher, immutable event authorities and exact active reservation. Initial ledger pins, component membership, source/config/runtime callsite membership, output identity and original active-scope cap checks are unchanged.

AST comparison confirms _source_ref(), validate_storage_v9(), PilotOutputV9 and source_report_v9 are identical to the original definitions. The original files are untouched. The adapter-local SOURCE reference now binds the new additive consumer bytes, so root must include both this reviewed consumer and the unchanged V9 implementation/launcher in the reservation's component bindings. The V9 core additionally retains its original concrete project_budget_v9_consumer.py binding requirement.

## Meaningful fake-core check

fake_metadata_check.py patches only the budget dependency with an in-memory fake. It opens no actual journal or output scope and takes no real scheduler lock. Its single clearly labelled fixture returns v4 charge960GB plus V9 reservations80GB and certified credits140GB, giving900GB projected charge. These are **invented test metadata, not measured current project totals**.

The new adapter accepts900GB and reports140GB credit; the original full-reservation adapter correctly rejects the same fixture's1,040GB uncredited charge. A3GB measured active scope within a4GB parent cap leaves1GB. Planning1,000,000,001 additional bytes fails at exactly one byte over that unchanged cap. Missing a pinned launcher membership and changing the consumer source digest both fail. Input storage/component metadata remains unchanged, and the only simulated measured paths are the active parent scope.

Reproduction:

~~~bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$ARGGNN_ROOT/src:$ARGGNN_ROOT" \
  PYTHONDONTWRITEBYTECODE=1 .runtime/env/bin/python \
  research/reviews/project_storage_closed_scopes_v1_review_author/fake_metadata_check.py
~~~

The extra project root on PYTHONPATH is needed for the existing scripts namespace in this standalone harness; it does not alter package versions or journal authority.

## Precise limitation

Both consumers deliberately call the original core with boundary=False before their separate active-scope measurement. In the core this means verify_closed=False: previously certified closure authority/digests and write revocation are used without re-running closure-wide stat manifests or global writable-descriptor checks. This is necessary for the requested narrow consumer behavior and is preserved from the original wrapper. Thus credited charge is cooperative journal accounting under the already certified no-future-writes covenant, not a new physical filesystem inventory or proof against external NFS writers. Root remains the sole owner of closure certification, scheduling and admissions.

The inherited PilotOutputV9.save() statvfs check is a local filesystem free-block query, not a recursive global disk scan. It is unchanged. The review provides no new live closure certification, capacity claim, dataset admission or global budget redesign.
