# C3 full-state migration binding plan

Source/small-metadata inspection only. The checkpoint path's existence at
04:36 UTC was supplied by root. This worker did not stat, open, read or hash
a checkpoint, import Torch, call a guard, admit, stop, launch, probe processes
or GPUs, or run Git.

## Exact current lineage

- Current scientific job: C3-GNN-cachev3-GPU0-TRAIN-v1.
- Current input: research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_C3_GPU0_train_v1/input.json
  SHA ed223e3bc46a47a07e999a0c4599ee8a150c3fc237c9ac8880b131bfc2dae0dc.
- Original output: experiments/results/C3_GNN_cachev3_GPU0_train_v1.
- Original storage: research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_C3_GPU0_train_v1/storage.json
  SHA a30cc34301ac894ffd775cf7d5158c8263654c0be3dd98d42401fb2b6c3b68b7,
  original2GiB charge, expiry2026-10-16T23:40:08.222700+00:00.
- Pool research0-oct16-v1, physicalGPU0 UUID GPU-ff06ca1f-403a-f2d0-b584-71e177452667.
- Controller capability: research/reviews/GPU0_shared_root_integration_20261016_v1/admitted_controller_v1/pool_capability.root_active.json
  SHA c88db41d22769af8cf7a02ce09134a0c0ac8ddc95006875d60b6d61a4c10e852.
- Controller plan at same directory/pool_plan.root_active.json
  SHA867def824e1fd7cbe817383994a2c4eed645377dd8431ba8eccc062328f6f5d8.
- Reviewed migration handoff:
  research/reviews/C3_efficient_resume_migration_source_20261016_v1/source_handoff.v1.json.
- Independent migration review:
  research/reviews/C3_efficient_resume_migration_independent_review_20261016_v1/review.v1.json
  SHA0747829a320ed60ae19ec3608ae4f3d1438344dcf5170826290e16f06d07fada.

## State schema and boundary selection

Original trainer_prefetch_v1.py125-141 makes a complete model/AdamW snapshot,
next cursor/order, full history, constant scheduler, Python/NumPy/Torch CPU/CUDA
RNG, observation/target seals and prior candidate refs. Lines312-321 save a
scheduled epoch candidate after each100 updates. Epoch005 is therefore boundary100
by source contract, subject to actual payload validation. The checkpoint excludes
its own candidate ref from trainer_state.scheduled_checkpoints to avoid a hash
cycle. On resumption the original trainer appends that own ref.

If epoch005 is still the latest genuine durable scheduled checkpoint at the
later transition, intent.old_scheduled_checkpoints must therefore be exactly
[{epoch:5,global_step:100,checkpoint_ref:ACTUAL_ROOT_HASHED_EPOCH005_REF}].
It must not be empty. If newer scheduled checkpoints exist then select the
latest genuine durable candidate and include all earlier candidates chronologically.
Do not freeze epoch005 now while the live job continues.

The path is experiments/results/C3_GNN_cachev3_GPU0_train_v1/checkpoint_epoch_005.pt.
Its SHA, actual embedded cursor/protocol and full optimizer validity remain
unknown to this worker. A filename/existence observation is not payload validation
and does not prove that no later parent update completed.

## Root sequence

1. Leave C3/H050 and their current pool untouched while completing the full
   source route and one ordinaryFP32 efficient step/cache qualification.
   No FP64 campaign, repeated training or source alias is needed.
2. Only after a separately root-authorized actual transition, consume the
   genuine C3 logical terminal, capture exact worker PID/start, original elapsed
   time/provenance, owned drain and revocation of all old C3 write/relaunch paths.
   H050 must genuinely complete before the old pool can retire. CPU migration
   can occur after C3's terminal while that old pool still supervises H050.
3. Obtain actual immutable checkpoint SHA/size and embedded old protocol SHA
   through a separately guarded root CPU inspection. These are required intent
   fields and are not available from the filename. No placeholder hash or
   invented cursor is usable.
4. Choose exact_latest_boundary only with actual original count equal toB and
   genuine no-completed-update-afterB evidence. Otherwise use the reviewed
   root_authorized_durable_scheduled_checkpoint_rollback mode with a separate
   explicit root authorization. Record later uncheckpointed updates as unknown
   if unknown and disclose discarded/repeated work. Retain old elapsed time.
   A normal SIGTERM is not proof that failure_last_complete_parent.pt was saved.
5. Create fresh active bounds from bounds.inactive.v1.json. Keep integration,
   step and cache exact refs, FP32/autocastfalse/TF32false, per-update telemetry,
   strict24GiB additional CUDA abort cap and actual ordinaryFP32 qualification.
   The scientific worker's original40GiB ceiling stays intact.
6. Create separate actual terminal evidence, optional rollback authorization
   and intent using the reviewed templates. Set old_job_id/input above, actual
   selected boundary/protocol SHA/cursorB, exact chronological candidate list,
   source/runtime/bounds/qualification refs and a new charged migration output
   directory disjoint from old output and future training output.
   Explicit max_checkpoint_bytes must be positive and at most2GiB.
7. Use original root CPU monitoring/resource and exclusive guarded writer.
   The exact deferred call is:
   validator.migrate_complete_boundary_v1(
       ACTUAL_INTENT_REF,
       resource_check=ACTUAL_ROOT_RESOURCE_CALLBACK,
       persist_checkpoint=ACTUAL_EXCLUSIVE_GUARDED_PERSIST_CALLBACK)
   The persistence callback permits only fresh basenames under the newly charged
   migration writer.path and uses original writer.save(name,value,kind="torch").
   Validator source SHA31414598d0f69533864cf600a9a6f0c24ef5f0d81f9ca131fb5f9612f761fafd.
   It runs CPU-only, validates complete state and all candidates, writes copies
   with changed execution protocol/reference provenance only, checks protected
   tensor/state fingerprints and serialized readback, and performs zero updates.
8. Preserve the returned unconsumed manifest, consume the actual tool terminal0,
   then root seals a separate manifest with actual_migration_tool_terminal_consumed
   true and actual_migration_tool_exit_code0. Never prefill these in source or
   overwrite the original returned false/unconsumed result.
9. Bind a fresh C3 scientific input/output/TMP/admission to that sealed manifest,
   actual bounds and exact new runtime/integration/validator/caps/qualification/
   intent/rollback refs, retaining original science/plans/providers/target seals.
   Include every required new ref in source closure and original<=64 direct
   bindings. Original _resume restores full state, and range(B,600) executes
   only the suffix. AtB100 that is500 additional effective updates.
10. Only after actual old-controller outer wait0, final closed registry and
    drains may a new GPU0 controller acquire the canonical singleton lock.
    Report retained trajectory, old-prefix implementation, efficient suffix and
    any discarded/repeated physical work truthfully.

## Native compatibility caveat

The reviewed migration pool itself requires every native spec to declare that
new pool as its operational_pool_source_ref and source closure. The unchanged
retained original native input pins the old pool, so it cannot be inserted in
that pool without invalid mutation.

The existing finite-declared-v2 envelope route preserves the retained native
input, original7GiB charge and expiry2026-10-18T23:41:32.476768+00:00. Its catalogue
currently has only C3 EVALUATE slots, and its old-controller validator requires
old H050/C3 success. Consequently it cannot yet host an unfinished C3 resumed
TRAIN job after a genuine nonzero C3 transition.

The correct combined route is a separately reviewed narrow finite-v2 extension:
one C3 full-state resume slot invoking the existing exact resume runtime,
strict24GiB logical reservation with original40GiB worker guard, and an explicit
old-C3 terminal/migration exception. H050 must still have genuine success and
the old whole controller must still have actual outer wait0/closed/drain.
The original C3 row remains failed/interrupted if so, never relabelled success.
Native method input/storage/caps/name/expiry and7GiB charge remain exact through
the original unchanged envelope/reuse proof.

Efficient C3 plus native needs72GiB including two2GiB margins and4GiB pool
headroom, leaving at most8GiB for actual external/unattributed usage on80GiB.
The original actual-free gate still applies. No observed fit, runtime speedup,
completion or ancestry score is implied by this plan.
