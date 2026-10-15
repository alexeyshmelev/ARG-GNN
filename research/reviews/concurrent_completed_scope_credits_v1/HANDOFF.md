# Concurrent completed-scope credits: source-only root handoff

The adapter is not activated. The implementation agent read only source and source-file metadata. It did not configure a guard, replay a journal, inspect processes/PIDs, inventory an output, close a scope or admit execution. Synthetic checks used only fabricated integers, hash strings and key sets.

## Mechanism

completed_scope_credits_v1.py calls the SHA-frozen H039_failed_scope_capacity_v2 base reader directly. It creates no cloned guard stack and modifies no historical source, journal, reservation or marker. Root freezes the exact old base pins and charged projection in one immutable activation policy. Old future accounting admissions are revoked prospectively. Existing active scopes continue under their original guards, caps and journal, fully charged. Root must not credit a scope while any launcher, supervisor or worker can still write it.

All future accounting events use a separate .runtime/storage/concurrent_completed_scope_credits_v1 journal under the original budget.v4.mutex. Old active guards see no new event in their own journal, so their historical pins remain valid. New admission/closure events are sequential, source/policy-bound, hash-chained and exclusive. Closed reservations remain in the occupied-key/path set forever. No new key may reuse or overlap their output or auxiliary scopes.

Conservative projection is:

    frozen_base_projection + all_new_caps - certified_completed_credits
        + 1MiB_bootstrap_metadata + 512KiB_per_new_event

Every prefix must remain strictly below 950000000000 bytes and the unchanged 1000000000000 human limit. The base itself must still pass its original reader/ceiling. No over-limit base bypass, global scan or rebase is provided.

New consumer initial pins bind a verified prefix containing that consumer's own reserve event. An unrelated later admission is an allowed append-only descendant. Rewritten hashes, a shortened chain, a pin predating the consumer's admission or a later closure of its key reject the writer. Admission authorities themselves still bind the exact current predecessor prefix.

## Root API and bindings

1. Root obtains the actual capacity-v2 base pins/projection and fills activation_policy.inactive.json. Set active true only after auditing prospective old-admission revocation and accounting assumptions. The policy pins actual implementation/launcher, this exact source, immutable capacity-v2/v1, marker repair/helper authority and the actual V9 marker. The malformed historical15MB key remains fully charged and uncreditable.
2. Configure with configure_v2(marker_repair_authority_ref, policy_ref=exact_policy_ref). Existing one-argument producers can instead receive the exact JSON {path,sha256} record through ARGGNN_COMPLETED_CREDITS_POLICY_REF. Their storage record must also bind completed_overlay_policy_ref. A configured consumer cannot switch policies.
3. inspect_v1 returns projection, exact bindings and the new prefix. operation_draft_v1 constructs inactive reserve/close_completed authorities from a supplied request, bindings and expected_before. Root supplies genuine activated authorities with expiry. apply_reserve_v1 and apply_close_completed_v1 hold the original mutex.
4. Future storage uses accounting_mode concurrent_completed_scope_credits_v1, actual budget_components.consumer_adapter pointing to this source, the reserve receipt's exact reservation and initial_ledger_pins, and a valid storage expiry. Reserve every binding returned by inspect plus the actual producer, launcher/helper closure, config and runtime. validate_storage_v9 and budget_check_v9 preserve those exact bindings and check the owned boundary before planned writes. They reject all closed or old-only keys.
5. PilotOutputV9 supports the existing constructor and json/npz/torch save calls. It preserves files; discard explicitly rejects. Native C3 path preparation uses its existing streaming Writer with this adapter's boundary methods and needs no writer replacement.

Quota operations authorize no GPU execution and never allocate a CUDA context. Policy restricts eligible physical GPUs to4/5 and disallows new occupancy on2/3. Actual assignment, leases and hardware enforcement remain in root's bound GPU launcher. Migration/continuation of already active jobs retains the full old scope charge; this overlay never terminates, moves or closes them.

## One completed-scope certificate

close_request.template.json lists the exact request fields. Root binds the reservation digest, actual storage/admission JSON and locator, actual launch metadata/PID and identity, actual direct-wait receipt/PID/exit locator and identity, producer output-root completed.json and its completed status identity, and a root observation certificate.

The direct wait must resolve to integer exit0 for the same launcher PID. A successful-looking file alone is insufficient. Root must bind the genuine wait result; if its raw receipt lacks identity fields, any normalization needs honest links to that original result and launch, audited by root.

The root observation must list every root-owned launcher, supervisor and worker for that exact scope, with ESRCH absence observations, and revoke every old/original/resume write capability for it. This is targeted evidence, not a project-wide process inventory. At apply time the adapter rechecks each listed PID absent, calls stat-only measure_scopes once for all exact owned output/auxiliary paths, rejects retained bytes above cap, and checks those PIDs absent again. Any present PID, including a zombie or reused PID, rejects the credit. It appends the original cap minus measured retained bytes and permanently closes the key.

Replay validates the bound completion/wait/revocation proof and stat certificate; it does not re-inventory closed scopes or re-observe their PIDs. Original files remain untouched. Duplicate closure rejects. Append uncertainty or a failed post-measurement observation needs root audit, not an automatic retry or scope deletion.

## Assumptions and counterexamples for root audit

This remains cooperative accounting, not an OS quota. Immutable old guards cannot themselves learn new closures. Root must actually retire all closed-scope launch/resume capabilities and route every future admission through this overlay. Relaunching an old packet outside that revocation contract, omitting a live worker from the roster, or an external NFS writer modifying a credited directory would make credit unsound. The source deliberately does no global FD/process scan to discover those cases.

New product writes must remain inside reserved scopes. New unscoped bookkeeping must stay within the charged bootstrap/per-event metadata allowances. Bound source/policy/authority/evidence JSON is limited to64KiB each; other source/runtime refs preserve their original guards. No scientific payload decode is used as storage evidence.

At an illustrative947.5GB base, admitting16GB needs more than13.5GB completed credits plus metadata. A6.5GB allowance retaining0.5GB yields about6GB credit and is insufficient alone. Multiple genuinely completed scopes may be necessary. Live H040 or CPU scopes cannot supply that difference. Already-base-closed/credited scopes and the malformed historical key cannot be credited again.

With a job pinned at new event1, appending unrelated event2 must leave it writable; substituting another event1 hash must fail. Closing its own key at event3 must then revoke it despite a valid event1 prefix. Pure arithmetic/prefix/closed-key checks cover these small predicates only, not live guard integration. Root must audit and activate before any resource effect.
