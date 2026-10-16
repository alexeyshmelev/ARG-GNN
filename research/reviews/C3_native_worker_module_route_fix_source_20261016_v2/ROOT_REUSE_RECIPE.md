# Root recipe: reuse both admitted scientific inputs

This routing packet fixes the generated child module. Its immutable handoff is routing_handoff.source_only.v1.json. The closed bootstrap-failed-controller handover must be integrated and independently reviewed before actual admission. Use that final integrated namespace for the APIs below.

Preserve every old failed outer/controller scope and charge. No old reservation, journal, capability, source, input or checkpoint is rewritten.

## Exact scientific inputs to reuse

- C3: research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/input.json, SHA256 b420377394b1246a55dd2ce2a731f523692ef557d8ea074d2fd3137231008a25.
- Native MLM: research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/input.json, SHA256 52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2.
- Original transition: research/reviews/C3_actual_full_state_handover_root_20261016_v1/prior_controller_closed_transport_unknown.proof_v1.json, SHA256 e545b8c331237fcb9da085125d7fd45b76826203c836799fe2af076d4eff54a6.

C3 was never submitted to the failed controller. Native stopped in argparse before the finite worker/method entry. The new handover must authenticate these facts from root's frozen actual records. Both scientific expiries remain unchanged. Do not call prepare_resume_v1, finish_scientific_input_v1 or reserve scientific storage again.

## Existing metadata builders and fresh small scopes

The root process supplies the new handover proof, the final reviewed e and p modules, its original exclusive JSON save function and original reserve/storage helper. This is deferred pseudocode using exact existing API names, not execution authority:

    old_C3_input = dict(
        path="research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/root_resume_actual_v1/input.json",
        sha256="b420377394b1246a55dd2ce2a731f523692ef557d8ea074d2fd3137231008a25",
    )
    C3_method = p.c3_method_v1(old_C3_input)
    native_method = p.native_method_unchanged_v1()
    native_reuse_ref = save_JSON("native_reuse.json",
        e.native_reuse_proof_v1(actual_new_handover_ref))

    envelope_refs = {}
    for slot, method, reuse in (
        ("C3_RESUME_TRAIN", C3_method, None),
        ("NATIVE_MLM", native_method, native_reuse_ref),
    ):
        request_ref = save_JSON(slot + ".request.json",
            e.operational_request_v1(slot, method, actual_new_handover_ref,
                actual_native_reuse_ref=reuse))
        reserve_request = e.outer_reserve_request_v1(
            request_ref,
            job_id=fresh_distinct_outer_job_ids[slot],
            output_directory=fresh_outer_output_directories[slot],
            actual_guard_bindings=actual_current_guard_bindings,
        )
        # Existing unmodified root reserve helper. Guard checks actual fresh pins.
        storage_ref = original_reserve_and_save_storage(
            reserve_request,
            job_id=fresh_distinct_outer_job_ids[slot],
            expires_at_utc=p.read(method["input_ref"])["expires_at_utc"],
        )
        envelope_refs[slot] = save_JSON(slot + ".envelope.json",
            e.operational_envelope_v1(request_ref, storage_ref,
                actual_guard_bindings=actual_current_guard_bindings))

    job_map_ref = save_JSON("job_map.json",
        e.finite_job_map_v1(envelope_refs["C3_RESUME_TRAIN"],
            envelope_refs["NATIVE_MLM"]))
    plan_ref = save_JSON("controller_plan.json",
        e.controller_plan_v1(job_map_ref, actual_new_handover_ref,
            pool=fresh_pool_name,
            controller_job_id=fresh_controller_job_id,
            expires_at_utc=pool_expiry_covering_both_scientific_expiries,
            drain_expires_at_utc=drain_at_least_35s_later_before_renewal_expiry,
            actual_guard_bindings=actual_current_guard_bindings))
    reserve_request = e.controller_reserve_request_v1(plan_ref,
        output_directory=fresh_controller_output_directory,
        actual_guard_bindings=actual_current_guard_bindings)
    controller_storage_ref = original_reserve_and_save_storage(
        reserve_request,
        job_id=fresh_controller_job_id,
        expires_at_utc=drain_at_least_35s_later_before_renewal_expiry,
    )
    capability_ref = save_JSON("capability.json",
        e.controller_capability_v1(plan_ref, controller_storage_ref,
            actual_guard_bindings=actual_current_guard_bindings))

Two outer scopes cost4MiB each. The new controller costs32MiB and has no auxiliary scope. No repeated7GiB native or32MiB C3 science charge is needed. Actual complete direct unions must fit64 after the final handover integration. An old count is not evidence for a changed union.

## Launch and clients

For the routing-only packet, the exact combined launcher is:

    research/reviews/C3_native_worker_module_route_fix_source_20261016_v2/launch_declared_gpu0_v1.py

and its module is:

    research.reviews.C3_native_worker_module_route_fix_source_20261016_v2.gpu0_declared_pool_v1

The final integrated source family may have a fresh namespace. Its generated worker module, launcher MODULE, launcher import and every actual route ref must all use the same final family. Root starts the controller with the existing renewal bootstrap arguments, consumes ready, submits the native logical job and C3 logical job separately, and consumes their genuine waits. Keep physicalGPU0, exact UUID/process title, original scientific inputs, actual memory/headroom gates, owned waits/drains, original write guards and both absolute expiries.

No import, resource admission, lease probe, launch, training, scoring or Git action was performed by the source author.
