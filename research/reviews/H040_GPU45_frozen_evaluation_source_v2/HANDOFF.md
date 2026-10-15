# H040 GPU45 frozen evaluator v2 source handoff

Status: source ready, inactive. Root owns admission, GPU5 scheduling, actual direct waits and execution. This task compiled source in memory and compared source bytes only. It imported no guard or scientific module, read no training completion/wait/checkpoint/target data, replayed no journal, admitted nothing and created no CUDA context.

## Exact execution interface

Main module: arggnn.H040_GPU45_frozen_evaluation_v2.

Context: research/reviews/H040_GPU45_frozen_evaluation_source_v2/context_factory_v2.py, exporting build_context_v2(record,spec).

Deferred root driver: research/reviews/H040_GPU45_frozen_evaluation_source_v2/root_evaluation_driver_v2.py. Default invocation prints an inert report. Root may run the following only after both resumed TRAIN jobs actually finish all30 passes/960 updates and their root launcher direct waits return integer0:

    .runtime/env/bin/python -B research/reviews/H040_GPU45_frozen_evaluation_source_v2/root_evaluation_driver_v2.py --admit
    .runtime/env/bin/python -B research/reviews/H040_GPU45_frozen_evaluation_source_v2/root_evaluation_driver_v2.py --launch --gpu 5

The driver uses the original resume packet actual_wait.json and launch.json for each arm. It binds both untouched resumed completed.json files. Missing/pending/failed waits reject admission before any packet or guard is created. At launch it rechecks exact completion/wait refs and metadata. GPU5 is mandatory; root must leave H042 on GPU4 and preserve external processes. The bound launcher still independently checks the lease and available memory. This source performs no polling or scheduling.

The fresh packet is created under this source folder's admitted_evaluate directory. The fresh evaluation output is experiments/results/H040_GPU45_frozen_evaluation_seed101_v2. Original output cap remains20000000 bytes. The reservation includes the fresh launcher_artifacts auxiliary directory for the actual launcher log, so that log shares the same fully charged cap. Original RSS512000000000 and CUDA50000000000 admission caps, process title, FP32 torch2.6.0+cu124/PyG2.7.0, lease and qualification guards are retained. Original scientific config is copied from the actual tensor TRAIN config, not reconstructed from a proposal.

## Main/context input fields

Input schema: H040_GPU45_frozen_evaluation_input_v2. Retain the standard active H040 authority fields, phase evaluate, arm joint_pair, seed101, device cuda, fit/PyG authorization, fresh job/output/created/expiry, exact source_sha256, refs to config/runtime/storage/registration/generation_config/observation_cache/target_bindings, all original40 observation_entry_refs and original observation_cache_source_refs. Required successor fields are:

- main_source_ref: exact new evaluator source.
- root_context_module_ref: exact context_factory_v2.py here.
- GPU45_authorization_ref: exact existing H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json.
- physical_gpu: integer5.
- completed_overlay_policy_ref: exact policy ref below, also equal to storage.completed_overlay_policy_ref.
- original_H038_input_ref: original SHA-bound research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json metadata.
- transitive_source_closure_ref: the admitted JSON with schema H040_GPU45_evaluation_transitive_source_closure_v2.
- train_completion_refs: exactly tensor and event_mlp, pointing to the actual resumed completed.json files.
- train_actual_wait_refs: exactly tensor and event_mlp, pointing to each original resume packet's actual_wait.json.

The transitive JSON contains the exact source_sha256 map, identical train_completion_refs/train_actual_wait_refs, both train_launch_refs pointing to the original resume packet launch.json files and original_H038_input_ref. Each wait must be actual_direct_launcher_wait true, integer returncode0, the same integer launcher PID as its launch receipt, the launch's exact input_ref from the supplied resumed completion, and the original physical GPU4 for tensor/GPU5 for event_mlp. The worker revalidates full 30-pass/960-step lineage before CUDA setup and again at evaluation.

The driver creates fresh VAL_bindings.json by reading only original H038 input/target metadata and copying exactly its eight original VAL parent entries. It sets phase evaluate, TRAIN_included false, VAL_included true and outer_included false; all other metadata is retained. The worker recomputes that exact expected object and requires full equality. The context checks same-parent VAL generation-receipt metadata; private target arrays are opened only by the unchanged evaluation truth gate. No outer data is opened.

## Current fixed resource identities

Guard: research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py, SHA256 ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef.

Policy: research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json, SHA256 a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85.

Actual GPU launcher: scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py, SHA256 d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d.

Only helper_refs.full_reservation_adapter and helper_refs.actual_full_storage_launcher are replaced with these exact fixed refs when comparing evaluation config against each original and resumed config. Every other config field must be equal. Actual resumed input/config refs and per-checkpoint tuples are never changed. The frozen resume worker still validates each actual resumed config against its original contract.

The context explicitly configures the completed-credit guard with the fixed policy_ref and original marker-repair authority. The driver supplies the same exact policy through ARGGNN_COMPLETED_CREDITS_POLICY_REF for the one-argument completed GPU launcher. Every guard/policy/actual launcher/helper pin is verified and bound. Runtime callbacks retain original logical-job, lease, RSS and CUDA checks and also check the current storage authority.

The deferred driver uses the guard's inspect/operation_draft/apply_reserve API, never old capacity-v2 admission. It serializes no old journal changes. Root must serialize its accounting actions; a changed predecessor causes rejection rather than an automatic retry. Append uncertainty or partially written root packets require root audit, not a blind rerun.

A JSON transitive closure carries all original/new source and TRAIN evidence provenance to stay within the immutable64-consumer/.py/.sh/.json interface. Main/context/driver/config/runtime/VAL bindings, the transitive JSON and actual required guard/launcher/policy bindings remain direct consumers. Every original actual source is still SHA-checked by the factory; checkpoint arrays and truth arrays remain transitive data refs.

## Scientific byte preservation

The entire run_evaluate function is byte-identical to the v1 frozen evaluator. expected_checkpoint_provenance_v1 and validate_checkpoint_provenance_v1 are also byte-identical. The original RAMMatrix class is imported without modification. The original code from cache=WitnessCache through final return is byte-identical to H040_latent_clade_execution_v1.

This preserves all96 native-hold probability cells in RAM before any VAL truth, checkpoint passes5/10/15/20/25/30 for both arms, all eight original VAL parents, original FP32 model/math/geometry checks, original equal-parent scoring, accuracy/NLL/earliest-pass selection, selected diagnostics and no outer truth. No optimization or probability/selection convention changes were introduced.

Each checkpoint still has one exact checkpoint_ref/input_ref/config_ref tuple derived from its own original or actual resumed provenance. Retained original checkpoints require original reset provenance; resumed checkpoints require precise optimizer restoration and optimizer_state_reset false. No membership-set relaxation was added.

source_qualification_receipt.json records in-memory compilation, exact full-function byte equality and source hash spans. No scientific, training, checkpoint-load, prediction or live resource integration test ran. Operational success and evaluation results remain pending root execution.
