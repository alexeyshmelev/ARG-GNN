# Root execution guide and canonical module correction

This additive guide corrects README.md, “Exact deferred root recipe”, step 7. For EVERY NEW controller admission/start/worker operation, use the canonical module:

    research.reviews.C3_native_H050_failed_CLI_transport_admission_source_20261016_v1.gpu0_declared_pool_v1

Its combined launcher/bootstrap is:

    research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/launch_declared_gpu0_v1.py

The old success-only transport-v2 module is not a new-controller target. Original-pool retirement controls intentionally target the original controller instead. No sealed source or README was edited. This is documentation only, with no stop, probe, checkpoint read, reserve or launch executed by its author.

## Original controls

Run each root command separately and consume its actual terminal. Cancel native before releasing C3 memory, so a queued native job cannot start in the gap. Do not repeat controls already genuinely completed.

~~~sh
old_control() {
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/env/bin/python -B \
  research/reviews/GPU0_shared_execution_source_20261016_v1/gpu0_bootstrap_renewal_v1.py \
  --execute \
  --renewal-certificate research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json \
  --renewal-sha256 66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7 \
  --renewal-source-sha256 491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468 \
  --module research.reviews.GPU0_shared_execution_source_20261016_v1.gpu0_shared_pool_v1 \
  -- "$@" --execute \
  --capability research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_controller_v1/pool_capability.root_active.json \
  --sha256 c88db41d22769af8cf7a02ce09134a0c0ac8ddc95006875d60b6d61a4c10e852
}
old_control cancel --job American-C3-native-val000-MLM-prediction-route-sharedGPU0-mlm-v1
old_control wait --job American-C3-native-val000-MLM-prediction-route-sharedGPU0-mlm-v1
old_control cancel --job C3-GNN-cachev3-GPU0-TRAIN-v1
old_control wait --job C3-GNN-cachev3-GPU0-TRAIN-v1
old_control shutdown
~~~

Native wait returns CLI 2 for honest cancelled/never-started. PID/start/started time must be null and identities empty. The actual stopped C3 leader code is a distinct fact from the wait client's shell code. Root now reports C3 leader -15, wait client 241, tool24047/chunk92dfd6. Retain both values. Shutdown request0 acknowledges a request. It does not recover original controller tool23022. Use the genuine frozen lifecycle-closed registry and separate root-owned absence/lease observation. Original23022 stays UNKNOWN/null.

## Fixed predecessor refs

| Role | Path | SHA256 |
|---|---|---|
| Original capability | research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_controller_v1/pool_capability.root_active.json | c88db41d22769af8cf7a02ce09134a0c0ac8ddc95006875d60b6d61a4c10e852 |
| Original ready | research/reviews/GPU0_shared_root_integration_20261016_v1/controller_actual_ready.root_v1.json | f62455a45fe5a79d3df41935df2ff23db3e90bcfa25b5e717a7b6fdd7e324ca0 |
| Original C3 input | research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_C3_GPU0_train_v1/input.json | ed223e3bc46a47a07e999a0c4599ee8a150c3fc237c9ac8880b131bfc2dae0dc |
| Actual H050 CLI1 cert | research/reviews/H050_full_budget_failed_CLI_root_20261016_v1/typed_certificate_v1/failed_CLI_scientific_completion_certificate.json | 91e44c61cfd6fb9c4cc4e2ff29efd4e0517128f7f9b13b683bdf7d59ea9991a7 |
| Original native input | research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/admitted_mlm_v1/input.json | 52242fe04f8f944dccbc81bf12940ad311d0d80f300e3cdd099be5e1748bc5d2 |
| Existing FP32 actual tool0 | research/reviews/C3_one_new_FP32_CPU_root_source_20261016_v2/actual_tool_terminal.root_v1.json | cc82827f8a0f6b92fb4f2b392ea91d6844bdc679a3bbc3c76ad3ae2b48622af3 |
| Existing FP32 result | experiments/results/C3_one_new_H8Q2_FP32_CPU_root_v2/result.json | 65b30e85061302fe5914b0dd7a67362aaaff4b532b5cfc4cb2952b7a01d6178a |
| Existing FP32 completion | experiments/results/C3_one_new_H8Q2_FP32_CPU_root_v2/completed.json | 0b7e4f15f201991adf5c8c2783fbe859c11c316c60f878465dc2de2011e195da |

Native expiry remains 2026-10-18T23:41:32.476768Z and its scientific 7 GiB charge is not repeated. Root's new transition folder is research/reviews/C3_actual_full_state_handover_root_20261016_v1. Derive every NEW ref from an actually saved immutable file. This guide invents no migration, final-registry, probe or reservation SHA.

## Migration schema and existing API

Root has already constructed the actual rollback100 request and launched genuine CPU migration tool30001. Consume that actual attempt rather than repeat it. The source validator is research/reviews/C3_efficient_resume_migration_source_20261016_v1/checkpoint_migration_validator_v1.py, SHA31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd. Its API is:

~~~python
result = migration_validator.migrate_complete_boundary_v1(
    actual_intent_ref,
    resource_check=actual_root_CPU_resource_callback,
    persist_checkpoint=existing_root_guarded_checkpoint_writer,
)
~~~

The root callback receives an exact fresh project-relative target and checkpoint value. The original guard exposes PilotOutputV9.save(name, value, kind="torch"). A callback can validate target.parent equals the charged migration writer's path, then use its exclusive original save. No new writer/guard implementation is needed.

The migration intent schema C3_actual_complete_boundary_migration_intent_v1 has exactly19keys: schema, active, root_authorizer, old_boundary_ref, old_protocol_sha256, old_job_id, old_input_ref, old_terminal_observation_ref, old_scheduled_checkpoints, new_integration_ref, new_runtime_ref, bounds_ref, qualification_ref, migration_output_directory, max_checkpoint_bytes, expected_actual_global_step, continuation_mode and rollback_authorization_ref.

Root binds actual epoch005 complete state and actual protocol SHA. The expected step is100 and scheduled list is the actual eligible epoch5/step100 candidate. Continuation mode is root_authorized_durable_scheduled_checkpoint_rollback. The separate root authorization must explicitly allow discarded later work, preserve all old files, forbid weights/optimizer reset, report no_work_lost_claimed=false and later_completed_update_count=null when unknown. The original elapsed26560.373677seconds is retained with actual provenance. Terminal evidence records the genuine nonzero C3 leader exit, consumed terminal/drain, old writer revocation, exact PID/start and actual boundary.

Reuse the genuine passed FP32 qualification. The active bounds keep the original exact field set, precision, memory caps and resume integration. The root qualification schema is C3_efficient_query_cache_root_qualification_v1 with passed=true, actual_tool_terminal_consumed=true, actual_tool_exit_code0, precisionfp32, optimizer_steps1, full_training_fit_or_accuracy_claimedfalse and exact step/cache refs.

The validator derives the new full protocol/signature and performs zero optimizer updates. Save its returned unconsumed manifest. Only after genuine tool0 save a distinct sealed manifest with actual_migration_tool_terminal_consumed=true and actual_migration_tool_exit_code=0. Preserve the returned false/unconsumed record. resume_integration_v1.read_resume_manifest_v1(actual_sealed_manifest_ref,actual_bounds_ref) checks eligibility. No fresh-weight fallback or H050 replay exists.

## Next exact transport request and proof

In an existing root CPU-only monitored/charged process, import THIS packet's finite_jobs_v1.py as fj. Save a permanent revocation JSON with schema GPU0_old_controller_permanent_capability_revocation_v1, root_authorizer=root_agent, active=true, permanent=true, exact prior_capability_ref/prior_ready_ref/controller_identity, all_old_controller_start_submit_requeue_worker_write_paths_revoked=true and new_scientific_or_restart_authority_granted=false.

After the actual migration terminal and sealed manifest, construct this request. All UPPERCASE values below mean existing actual references, not future refs:

~~~python
cat = fj.catalogue_v1()
request = dict(
    schema="GPU0_transport_unknown_H050_failed_CLI_closed_handover_request_v1",
    status="active",
    root_authorizer="root_agent",
    prior_capability_ref=cat["known_prior_controller_capability_ref"],
    prior_ready_ref=cat["known_prior_controller_ready_ref"],
    final_registry_ref=ACTUAL_FROZEN_CLOSED_REGISTRY_REF,
    C3_resume_manifest_ref=ACTUAL_ROOT_SEALED_MIGRATION_REF,
    explicit_old_capability_revocation_ref=ACTUAL_PERMANENT_REVOCATION_REF,
    old_controller_future_writes_revoked=True,
    observation_handle_loss_ref=fj.HANDLE_LOSS_REF,
    original_external_tool_session_id=23022,
    original_external_tool_exit_status="UNKNOWN",
    original_external_tool_exit_code=None,
    original_external_tool_terminal_consumed=False,
    original_controller_outer_wait0_claimed=False,
    actual_controller_terminal_ref=None,
    transport_unknown_closed_transition_explicitly_authorized=True,
    H050_failed_CLI_certificate_ref=ACTUAL_H050_CLI1_CERT_REF,
    H050_full_science_actual_CLI_exit1_explicitly_authorized=True,
    H050_training_replay_authorized=False,
    H050_success0_or_completed_scope_credit_claimed=False,
)
request_ref = original_exclusive_JSON_save("transport_request.json", request)
fj.validate_transport_unknown_request_v1(request_ref)
observation = fj.observe_transport_unknown_closed_v1(
    request_ref, resource_check=actual_root_CPU_resource_callback,
)
observation_ref = original_exclusive_JSON_save("transport_observation.json", observation)
~~~

The observer validates metadata before any exact PID/start/UID and original-inode probe. It returns actual evidence and writes nothing. Persist it via the existing guarded exclusive writer. Consume this NEW probe process's real terminal0 and save:

~~~python
probe_terminal = dict(
    schema="GPU0_transport_unknown_probe_actual_tool_terminal_v1",
    root_authorizer="root_agent",
    actual_terminal_consumed=True,
    actual_exit_code=0,
    exec_session_id=ACTUAL_NEW_PROBE_TOOL_SESSION,
    terminal_chunk_id=ACTUAL_NEW_PROBE_TERMINAL_CHUNK,
    observation_ref=observation_ref,
    request_ref=request_ref,
    probe_source_ref=fj.TRANSPORT_HELPER_REF,
    metadata_observation_not_original_controller_exit=True,
    original_external_tool_exit_status="UNKNOWN",
    original_external_tool_exit_code=None,
)
probe_terminal_ref = original_exclusive_JSON_save("probe_actual_terminal.json", probe_terminal)
proof = dict(request)
proof.update(
    schema=fj.UNKNOWN_SCHEMA,
    status="closed_lifecycle_transport_unknown",
    transport_unknown_request_ref=request_ref,
    transport_unknown_observation_ref=observation_ref,
    probe_actual_tool_terminal_ref=probe_terminal_ref,
)
proof_ref = original_exclusive_JSON_save("prior_controller_completion.json", proof)
fj.validate_previous_controller_v1(proof_ref)
~~~

The new proof schema is GPU0_prior_controller_transport_unknown_H050_failed_CLI_handover_v1. Old success-only schemas are rejected. Final H050 row must exactly equal its certified failed row. C3 keeps its genuine stopped state. Original23022remainsUNKNOWN. Probe0doesnotbecomeoriginalcontroller0.

## Exact fresh admission APIs and original reserve helper

Import this packet's prepare_resume_v1.py as p and finite_envelopes_v1.py as e. Root's existing c=p.original_contract_v1() exposes c.save(directory,filename,value), the original exclusive JSON writer. Its per-file bound is64KiB. Keep oversized metadata in the existing exact-ref/manifest structure.

For every new reservation, use a fresh original guard state and complete current bindings. This root-side sequence uses unchanged APIs, and the build_request callback must return the actual full request. It is documentation, not an executed or newly installed driver:

~~~python
guard = c.original_guard()
with guard.budget.v4.mutex(ROOT):
    state = guard._state_locked(c.CORE, c.BASE, fresh=True)
    view, bindings = state[2], state[3]
request = build_request(list(bindings.values()))
assert len(request["consumer_bindings"]) <= 64
authority = guard.operation_draft_v1(
    "reserve", request, expected_before=guard._pins(view), bindings=bindings,
)
authority.update(active=True, expires_at_utc=ACTUAL_SCOPE_EXPIRY)
authority_ref = c.save(ACTUAL_PACKET, "reserve_authority.json", authority)
receipt = guard.apply_reserve_v1(
    authority_ref,
    expected_source_sha256=c.CORE,
    expected_launcher_sha256=c.BASE,
)
receipt_ref = c.save(ACTUAL_PACKET, "reserve_receipt.json", receipt)
storage = e.storage_from_actual_reserve_v1(
    request, receipt_ref,
    job_id=ACTUAL_SCOPE_JOB_ID,
    expires_at_utc=ACTUAL_SCOPE_EXPIRY,
    actual_budget_components=ACTUAL_COMPONENT_REFS,
    actual_policy_ref=c.POLICY_REF,
)
storage_ref = c.save(ACTUAL_PACKET, "storage.json", storage)
~~~

ACTUAL_COMPONENT_REFS has exactly implementation/launcher/consumer_adapter and the actual unchanged guard refs. No guard/source/old receipt is cloned or rewritten. If another admission advances the prefix, preserve the rejection and create a fresh authority against actual current state.

Use the following exact builders in order:

1. materials=p.prepare_resume_v1(p.OLD_INPUT_REF,actual_sealed_manifest_ref,names). Names has job_id,packet_directory inside this source packet, fresh experiments/results output_directory and fresh .runtime/tmp/gpu0_shared short_tmp_directory.
2. Save materials["caps"], p.worker_request_v1(materials,caps_ref), p.operational_config_v1(materials,request_ref), an exact JSON runtime copy from the old input's refs.runtime, p.admission_authorization_v1(materials,created_at_utc=...,expires_at_utc=...), and p.source_closure_v1(materials,request_ref,authorization_ref). Fresh C3 lifetime is positive and at most24hours.
3. refs has exactly caps/request/config/runtime/closure/admission_authorization. Scientific consumers=p.direct_scientific_consumers_v1(materials,refs,actual_guard_bindings=...). Reserve32MiB with keyH041/newjob, fresh output and exactly packet/worker_aux plus fresh TMP auxiliary scopes. Use the standard reserve/storage sequence above. Save p.finish_scientific_input_v1(materials,refs,storage_ref,actual_guard_bindings=...).
4. Save e.native_reuse_proof_v1(proof_ref). C3 method=p.c3_method_v1(actual_new_scientific_input_ref). Native method=p.native_method_unchanged_v1(), an exact copy of the old nested method. Never reserve native science again.
5. For each of exactly C3_RESUME_TRAIN and NATIVE_MLM, save e.operational_request_v1(slot,method,proof_ref,actual_native_reuse_ref=...nativeonly...). Its 4MiB request comes from e.outer_reserve_request_v1(request_ref,job_id=distinct_outer_job,output_directory=fresh_outer_results,actual_guard_bindings=...). Reserve it with the original sequence, then save e.operational_envelope_v1(request_ref,actual_outer_storage_ref,actual_guard_bindings=...). Native outer expiry cannot exceed the original native expiry.
6. Save e.finite_job_map_v1(actual_C3_envelope_ref,actual_native_envelope_ref).
7. Save e.controller_plan_v1(map_ref,proof_ref,pool=...,controller_job_id=...,expires_at_utc=...,drain_expires_at_utc=...,actual_guard_bindings=...). Pool expiry covers both scientific expiries. Drain is at least35seconds later and strictly before the genuine renewal expires.
8. New controller32MiB/noaux request=e.controller_reserve_request_v1(plan_ref,output_directory=fresh_controller_results,actual_guard_bindings=...). Reserve through the same original guard and save standard storage, whose expiry covers drain. Save e.controller_capability_v1(plan_ref,actual_controller_storage_ref,actual_guard_bindings=...).
9. Actual complete unions must remain at most64. Reviewed worst counts are science60/C3outer62/nativeouter64/controller61. Native has zero spare slots. The guide is documentation only and is not a runtime dependency. Do not add it as a consumer or drop existing required refs.

## Correct new controller and clients

The combined launcher SHA is f59cfac77c9a3303cc57446d34557277f0a3c8f11ae54027ecf43cbb3606360b. Pool SHA is38b7fb1122983d99f786023f51a01a5db17e2192466aab4505ac486c3e8700e2.

~~~sh
new_control() {
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  .runtime/env/bin/python -B \
  research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/launch_declared_gpu0_v1.py \
  --execute \
  --renewal-certificate research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json \
  --renewal-sha256 66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7 \
  --renewal-source-sha256 491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468 \
  --module research.reviews.C3_native_H050_failed_CLI_transport_admission_source_20261016_v1.gpu0_declared_pool_v1 \
  -- "$@" --execute --capability ACTUAL_NEW_CAPABILITY_PATH --sha256 ACTUAL_NEW_CAPABILITY_SHA
}
new_control start
~~~

Start is a genuine long-running controller and needs its own actual tool/session. From separate root clients use new_control submit --job ACTUAL_DECLARED_JOB, then new_control wait --job ACTUAL_DECLARED_JOB. Startup still repeats exact original known-identity absence and original free-inode test, then verifies the actually acquired original FD before metadata overwrite. All original GPU0 UUID, title, actual memory gates, per-job expiry, integer exit, ownership, wait/drain and storage checks remain.
