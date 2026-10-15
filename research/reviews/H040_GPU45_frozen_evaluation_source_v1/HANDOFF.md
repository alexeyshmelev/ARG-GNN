# H040 GPU45 frozen evaluator successor

Source-only successor for resumed H040 checkpoints. Original H040 worker, resume worker/context, model, data, targets and training records remain unchanged. Root owns admission, GPU4/5 scheduling, process title, execution, completion binding and Git. This task created no CUDA context, predictions, admission, live guard configuration or model test.

## Preserved scientific evaluation

The original `RAMMatrix` class is reused. The original `run_evaluate` body is copied with only one training-lineage metadata validation call and replacement of its checkpoint input/config comparison by an exact per-pass provenance check. All code from `cache=WitnessCache(cfg,guard)` through the return is byte-identical to the original, SHA256 `7e26bcc1d421faf59b0e04c4ba898761c81c3c3a3a83611e4494d61b654afaf3`.

The evaluator still loads all six checkpoints5/10/15/20/25/30 for both tensor/event_mlp arms, predicts all eight original VAL parents and seals all96 native-hold probability arrays in RAM before any VAL truth. Original array geometry/identity checks, FP32 probabilities, model math, equal-parent metrics, earliest accuracy/NLL/pass selection, selected full diagnostics and no outer-truth rule are unchanged. It performs no optimization.

## Exact root interface

Main module: `arggnn.H040_GPU45_frozen_evaluation_v1`.

Context: `research/reviews/H040_GPU45_frozen_evaluation_source_v1/context_factory_v1.py`, exporting `build_context_v1(record,spec)`.

    .runtime/env/bin/python -B -m arggnn.H040_GPU45_frozen_evaluation_v1 --execute --input ROOT_INPUT_JSON --expected-input-sha256 EXACT_SHA256

Only root's `scripts/launch_reserved_gpu_authorized_GPU45_v1.py` may launch it. Both the context and root authorization restrict physical GPUs to4/5. No flags print an inert source-only report.

Input schema is `H040_GPU45_frozen_evaluation_input_v1`. Keep the standard active H040 root spec with root_authorizer root_agent, status active, hypothesis_id H040, phase evaluate, arm joint_pair, seed101, CUDA fit/PyG authority, fresh job/output/expiry, runtime/storage refs, the original registration/generation/cache refs, all40 observation entry refs, original observation-cache source refs and exact eight-parent VAL target bindings. Add:

- `main_source_ref`: exact successor main source.
- `root_context_module_ref`: exact evaluation context here.
- `GPU45_authorization_ref`: exact `H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json`.
- `train_completion_refs`: object with exactly tensor and event_mlp keys, each SHA-bound to its actual resumed completed.json.
- `train_actual_wait_refs`: object with exactly tensor and event_mlp keys, each SHA-bound to its actual successful root launcher wait.
- `source_sha256`: every original training source, frozen resume worker/context and new evaluator/context sources.

The evaluation config can directly reuse the resumed training config ref. It must equal each actual original checkpoint config after replacing only helper_refs.actual_full_storage_launcher with the current GPU45 launcher ref. No scientific/resource config membership exceptions are introduced. Root should retain the original20MB output cap/config identity unless it versions and reviews a future changed condition separately.

The existing root storage/lease factory is retained with the new main source used for validate_storage_v9, GPU45 restrictions and strict capability/source validation. Fresh storage output is mandatory. Root must serialize admissions/launches and refresh initial pins with its current guard.

## Strict checkpoint provenance

Before predictions, both actual training completions must declare 30 complete pass histories,960 updates, six checkpoint groups and exact optimizer-restore lineage. Their actual launcher waits must be direct exit0 waits with completed_ref equal to the supplied resumed completion. Each completion path must be the completed.json under its actual bound resume input's fresh output.

Both frozen per-arm resume manifests are hard-pinned to the existing handed-off hashes. Tensor resumes at15/480 and event_mlp at5/160. Original initializer identities, observation/generation/data refs, complete original source closure, actual resumed input/schema/config and GPU authority are revalidated.

For each pass, `checkpoint_provenance[str(pass)][arm]` must exactly equal the unique expected object `{checkpoint_ref,input_ref,config_ref}`, with no extra keys:

- At/before that arm's saved resume pass, all three refs are derived from the frozen original resume manifest and retained checkpoint map.
- After the saved pass, input/config must be the actual resumed completion input/config and checkpoint path must be exactly that input output_directory/ARM.pass_PP.pt. The checkpoint hash equals both the checkpoint_groups ref and provenance ref.

Each loaded checkpoint payload must match that exact input/config tuple as well as all original schema, model, parameter, pass, update-count, teacher, mechanism and warm-start checks. Resumed payloads additionally require the precise restored-checkpoint/manifest/pass lineage and optimizer_state_reset false. Original payloads retain their original optimizer_state_reset true marker. No accepted-list membership substitutes for equality.

Checkpoint arrays and private truth remain transitive data refs. Reserve only allowed .py/.sh/.json consumers: new main/context, root launcher/driver/config/runtime/VAL-bindings/completion/wait metadata and existing guard/repair/marker closure. The frozen source_closure.json here may carry transitive provenance metadata when root's64-consumer ceiling requires it; actual sources remain SHA-checked by the factory.

## Verification and remaining prerequisites

source_qualification_receipt.json records successful in-memory compilation of both new sources, reuse of original RAMMatrix and exact byte identity of the complete scientific evaluation core. No actual completion validation was run because training is still running. No predictions or handmade/full model tests ran.

Root must wait for both actual resumed960-update completions and their successful launcher waits, then bind those refs and create its fresh admission/launcher. No evaluation result or operational success is claimed by this source handoff.
