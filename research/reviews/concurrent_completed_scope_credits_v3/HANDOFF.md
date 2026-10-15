# Concurrent completed-scope credits v3: source-only root handoff

The adapter is not activated. The implementation agent read only source and source-file metadata. It did not configure a guard, replay a journal, inspect processes/PIDs, inventory an output, close a scope or admit execution. The v3 source check uses only fabricated JSON byte strings and extracted pure parser/identity helpers. No scientific payload is written.

## Mechanism

completed_scope_credits_v3.py calls the SHA-frozen H039_failed_scope_capacity_v2 base reader directly. It creates no cloned guard stack and modifies no historical source, journal, reservation or marker. Root freezes the exact old base pins and charged projection in one immutable activation policy. Old future accounting admissions are revoked prospectively. Existing active scopes continue under their original guards, caps and journal, fully charged. Root must not credit a scope while any launcher, supervisor or worker can still write it.

All future accounting events use a separate .runtime/storage/concurrent_completed_scope_credits_v3 journal under the original budget.v4.mutex. Old active guards see no new event in their own journal, so their historical pins remain valid. New admission/closure events are sequential, source/policy-bound, hash-chained and exclusive. Closed reservations remain in the occupied-key/path set forever. No new key may reuse or overlap their output or auxiliary scopes.

Conservative projection is:

    frozen_base_projection + all_new_caps - certified_completed_credits
        + 1MiB_bootstrap_metadata + 512KiB_per_new_event

Every prefix must remain strictly below 950000000000 bytes and the unchanged 1000000000000 human limit. The base itself must still pass its original reader/ceiling. No over-limit base bypass, global scan or rebase is provided.

New consumer initial pins bind a verified prefix containing that consumer's own reserve event. An unrelated later admission is an allowed append-only descendant. Rewritten hashes, a shortened chain, a pin predating the consumer's admission or a later closure of its key reject the writer. Admission authorities themselves still bind the exact current predecessor prefix.

## Root API and bindings

1. Root obtains the actual capacity-v2 base pins/projection and fills activation_policy.inactive.json. Set active true only after auditing prospective old-admission revocation and accounting assumptions. The policy pins actual implementation/launcher, this exact source, immutable capacity-v2/v1, marker repair/helper authority and the actual V9 marker. The malformed historical15MB key remains fully charged and uncreditable.
2. Configure with configure_v2(marker_repair_authority_ref, policy_ref=exact_policy_ref). Existing one-argument producers can instead receive the exact JSON {path,sha256} record through ARGGNN_COMPLETED_CREDITS_POLICY_REF. Their storage record must also bind completed_overlay_policy_ref. A configured consumer cannot switch policies.
3. inspect_v1 returns projection, exact bindings and the new prefix. operation_draft_v1 constructs inactive reserve/close_completed authorities from a supplied request, bindings and expected_before. Root supplies genuine activated authorities with expiry. apply_reserve_v1 and apply_close_completed_v1 hold the original mutex.
4. Future storage uses accounting_mode concurrent_completed_scope_credits_v3, actual budget_components.consumer_adapter pointing to this source, the reserve receipt's exact reservation and initial_ledger_pins, and a valid storage expiry. Reserve every binding returned by inspect plus the actual producer, launcher/helper closure, config and runtime. validate_storage_v9 and budget_check_v9 preserve those exact bindings and check the owned boundary before planned writes. They reject all closed or old-only keys.
5. PilotOutputV9 supports the existing constructor and json/npz/torch save calls. v2 returns the SHA256 of the actual serialized data after validating the saved pathname still has the opened inode and expected size; scientific product references do not pass through the 32MiB metadata reader. It preserves files; discard explicitly rejects. Native C3 path preparation uses its existing streaming Writer with this adapter's boundary methods and needs no writer replacement.

Quota operations authorize no GPU execution and never allocate a CUDA context. Policy restricts eligible physical GPUs to4/5 and disallows new occupancy on2/3. Actual assignment, leases and hardware enforcement remain in root's bound GPU launcher. Migration/continuation of already active jobs retains the full old scope charge; this overlay never terminates, moves or closes them.

## One completed-scope certificate

close_request.template.json lists the exact request fields. Root binds the reservation digest, actual storage/admission JSON and locator, actual launch metadata/PID and identity, actual direct-wait receipt/PID/exit locator and identity, producer output-root completed.json and its exact completed status identity or one of the schema-only identities below, and a root observation certificate.

The direct wait must resolve to integer exit0 for the same launcher PID. A successful-looking file alone is insufficient. Root must bind the genuine wait result; if its raw receipt lacks identity fields, any normalization needs honest links to that original result and launch, audited by root.

The root observation must list every root-owned launcher, supervisor and worker for that exact scope, with ESRCH absence observations, and revoke every old/original/resume write capability for it. This is targeted evidence, not a project-wide process inventory. At apply time the adapter rechecks each listed PID absent, calls stat-only measure_scopes once for all exact owned output/auxiliary paths, rejects retained bytes above cap, and checks those PIDs absent again. Any present PID, including a zombie or reused PID, rejects the credit. It appends the original cap minus measured retained bytes and permanently closes the key.

Replay validates the bound completion/wait/revocation proof and stat certificate; it does not re-inventory closed scopes or re-observe their PIDs. Original files remain untouched. Duplicate closure rejects. Append uncertainty or a failed post-measurement observation needs root audit, not an automatic retry or scope deletion.

## Assumptions and counterexamples for root audit

This remains cooperative accounting, not an OS quota. Immutable old guards cannot themselves learn new closures. Root must actually retire all closed-scope launch/resume capabilities and route every future admission through this overlay. Relaunching an old packet outside that revocation contract, omitting a live worker from the roster, or an external NFS writer modifying a credited directory would make credit unsound. The source deliberately does no global FD/process scan to discover those cases.

New product writes must remain inside reserved scopes. New unscoped bookkeeping must stay within the charged bootstrap/per-event metadata allowances. Operation authorities, events and other evidence JSON remain limited to64KiB each. Only the original producer completed.json uses a separate SHA-bound reader limited to1MiB. Existing producer receipts stay inside their originally charged scopes; their content is not embedded in journal events. Other source/runtime refs preserve their original guards. No scientific payload decode is used as storage evidence.

At an illustrative947.5GB base, admitting16GB needs more than13.5GB completed credits plus metadata. A6.5GB allowance retaining0.5GB yields about6GB credit and is insufficient alone. Multiple genuinely completed scopes may be necessary. Live H040 or CPU scopes cannot supply that difference. Already-base-closed/credited scopes and the malformed historical key cannot be credited again.

With a job pinned at new event1, appending unrelated event2 must leave it writable; substituting another event1 hash must fail. Closing its own key at event3 must then revoke it despite a valid event1 prefix. The original v1 pure checks cover these small predicates only. The preserved v2 check covers the large-output reference regression using fake I/O. The v3 parser check uses fabricated JSON, not live guard integration. Root must audit and activate before any resource effect.

## v2 source correction

The independent v1 audit found that save() wrote outputs above 32MiB and then failed while calling the metadata-bounded reference helper. v2 hashes the existing serialized bytes directly and restores post-write inode/presence validation. Its source path, accounting mode, fresh journal path and inactive templates use v2. Public API names and v1 data schemas remain compatible. The audited v1 files and every historical guard/journal are preserved. See ../concurrent_completed_scope_credits_independent_v1/AUDIT.md. This source is inactive; no credit or storage capacity has been established.


## v3 producer completion compatibility

The producer completed.json reader accepts at most1048576 bytes. Operation authorities and journal events retain the65536-byte bound. The original status string starting with completed remains accepted exactly as before.

A schema-only identity is accepted only for these exact producer schemas:

- expanded_native_ap_driver_completed_v2
- expanded_native_argmix_preparation_v1_root_admission_completed
- expanded_pyg_observations_v5_completed

For that branch, completed_identity must bind the actual receipt's exact schema and completed_utc string. Both fields are compared against the untouched SHA-bound original receipt. completed_utc must carry a timezone and must not be in the future. The inactive schema-only templates contain null timestamps; root must copy genuine original fields and add any actual producer identity/count fields needed for its audit. No receipt is edited, normalized or given a synthetic status.

The same direct launcher wait with integer exit0 for the same PID, complete targeted root launcher/supervisor/worker roster, apply-time absence checks, actual write/resume revocation, exact owned-scope stat inventory and append-only accounting remain required. Schema/clock parsing alone does not establish successful scientific completion or available credit. Root owns the genuine wait/roster evidence, semantic completion audit and activation. This v3 source does not inspect candidate receipts or certify the root-reported candidate counts.

v1/v2 sources, templates and journals are preserved. v3 uses a new source path, accounting mode and fresh journal. It retains the v2 large-output save correction and the exact immutable capacity-v2 base.
