# H040 GPU4/5 resume handoff

Frozen source-only continuation from the last complete saved H040 checkpoints. Root owns admissions, physical GPU4/5 leases, launcher process titles and execution. Original worker, mechanism, configuration, checkpoint and data sources were not edited. This task ran no GPU, biological training, live guard configuration, admission, installation or Git command.

## Resume boundaries

| Arm | Saved checkpoint | Saved AdamW steps | Remaining passes | Remaining updates |
| --- | --- | ---: | --- | ---: |
| tensor | tensor.pass_15.pt | 480 | 16–30 | 480 |
| event_mlp | event_mlp.pass_05.pt | 160 | 6–30 | 800 |

Both finish the original 30 passes / 960 updates. The original deterministic parent order remains `random.Random(101+pass_index).shuffle(TRAIN)`. All 32 TRAIN parents, target construction, observation-only witnesses, model equations, FP32 losses, gradient clipping, AdamW settings, checkpoint passes, VAL policy and selection rule are unchanged.

The old tensor pass16 and event MLP pass6–8 history receipts are preserved separately as uncheckpointed progress. Their optimizer states were not durable, so continuation begins from the actual saved pass15/pass5 states. They are not counted twice in the resumed optimization lineage. All old checkpoint groups and history through the saved checkpoint remain referenced unchanged. Fresh output paths are mandatory.

## Root capability contract

The new module is `arggnn.H040_GPU45_resume_v1`; its CLI is:

    .runtime/env/bin/python -B -m arggnn.H040_GPU45_resume_v1 --execute --input ROOT_INPUT_JSON --expected-input-sha256 EXACT_SHA256

It must be invoked by root's `scripts/launch_reserved_gpu_authorized_GPU45_v1.py` launcher. No flags print an inert source report. Context factory is `research/reviews/H040_GPU45_resume_source_v1/context_factory_v2.py`, exporting the original `build_context_v1(record,spec)` interface.

Root input schema is `H040_GPU45_resume_input_v1`. Preserve original root active input fields, TRAIN data/target bindings and observation sources, and supply:

- `main_source_ref`: exact new worker source.
- `root_context_module_ref`: exact new context source.
- `resume_manifest_ref`: `resume_tensor_inactive.json` or `resume_event_mlp_inactive.json` here.
- `GPU45_authorization_ref`: exact root GPU45 authorization JSON.
- `source_sha256`: every original source pin plus the new worker/context.
- Fresh config/runtime/storage/job/expiry/output fields under root's current resource authority.

Resume manifests are immutable provenance, not execution capabilities. Their execution_authorized false field records that this source task granted no execution. Root activates the new input, not the provenance manifest.

The new config must equal the actual original checkpoint's config except `helper_refs.actual_full_storage_launcher`, which changes to the new GPU45 launcher ref. Config identity changes are recorded explicitly per checkpoint. Every registration, generation config, cache, observed entry and phase target-binding ref remains the original one.

The context verifies its new main/source closure and original worker SHA, exact root GPU authorization and physical lease4/5, process title, torch/PyG versions, storage guard, original target provenance and unchanged science protocol. Checkpoint binaries and histories are transitively pinned through the directly reserved resume JSON; do not directly reserve .pt consumers under frozen v4.

## Restore and checkpoint provenance

The worker loads saved learned tensors strictly and calls `optimizer.load_state_dict`, preserving all42 AdamW states, moments, per-parameter steps and original optimizer options. Moment tensors follow the restored parameter device. It never invokes the original warm-start initializer or resets optimizer state on resume.

Original observations and all 32 witness recipes are reconstructed and compared to the original sealed recipes before TRAIN truth access. No witness sidecar arrays are saved. Original run_context, train_one, targets, losses and update order remain the execution core; only the TRAIN loop's saved state/start boundary and provenance are adapted.

Completion retains the original status `completed_fixed_30_pass_single_arm_TRAIN`, six checkpoint_groups, 30 history rows and steps {arm:960}. New fields are:

- `checkpoint_provenance[str(pass_index)][arm] = {checkpoint_ref,input_ref,config_ref}`, covering old and resumed checkpoints.
- `history_record_refs`, pointing to all retained original/new pass receipts.
- `resume_manifest_ref`, `resumed_from_checkpoint_ref`, `resumed_from_pass`, original_input_ref/config_ref and resumed_optimizer_steps_performed.
- `optimizer_state_restored=true`, `optimizer_state_reset_on_resume=false`.
- `preserved_uncheckpointed_history_refs`, retaining interrupted progress evidence separately.

New checkpoint payloads preserve schema H040_frozen_checkpoint_v1 and original scientific fields, use their actual new input_ref/config_ref, set optimizer_state_reset false and record restore provenance. Root's evaluator successor must validate each checkpoint against its own checkpoint_provenance entry. The original evaluator assumes one input/config for all checkpoints and must not silently be used for interrupted provenance.

## Verification performed

Both sources compiled in memory. CPU reads of the actual saved checkpoints confirmed exact input/config refs, all42 AdamW state steps480/160, finite FP32 learned state and complete retained history. Source/config contracts passed for both per-arm manifests. One three-parameter CPU model restored weights/moments/steps exactly and produced an identical next AdamW update. No full H040 model was instantiated or tested. The actual CPU check command exited0; see source_CPU_restore_check.json.

Root reviewed and froze worker/context/manifests at the handed-off hashes while starting coordinated admissions. Subsequent work here writes documentation/receipts only. No launch or admission by this agent is authorized.
