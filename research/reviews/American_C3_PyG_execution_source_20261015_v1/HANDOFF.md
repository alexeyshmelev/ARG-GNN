# American C3 executable phase hooks — source handoff

This packet wraps the existing C3 whole-tree PyG engines. It is source-ready, not admitted or scientifically executed. Root owns integration, current writer/runtime capabilities, process title, leases, storage, devices and separate TRAIN/VAL truth resolvers. Do not use the older capacity-v2 CLI.

## Frozen supported experiment

Fresh C3 weights, 29,571 parameters, 20 independent 2 Mb TRAIN parents, 30 epochs and exactly 600 complete-parent FP32 updates expose all 10,000 original focal records per parent per epoch. Six checkpoints are epochs 5/10/15/20/25/30. The original five independent 30 Mb VAL parents are the held-out selection cohort: this is explicitly 2 Mb to 30 Mb length transfer. The original full30Mb TRAIN000 remains TRAIN. The separate original20×30Mb training draft is inactive and outside this production closure.

Every parent has 1,600 haplotypes: AFR columns0–199, EUR200–399, ASIA400–599, unknown queries600–1599. Output order is EUR/EAS/AFR, hence reference labels [2]×200+[0]×200+[1]×200 and query labels -1. Full inferred observations and the original native reference/query/focal split are shared across methods. No node-ID vocabulary or checkpoint is transferred between ARGs; no C4 weights are loaded or sliced. This is an existing architecture, with no new architecture claim.

Existing checkpointed whole-tree encoder and forward/reverse recurrent passes retain complete-parent autograd and genomic state. CPU boundary tensors are not detached. The inferred tsinfer/tsdate reconstruction explicitly collapses unary nodes, then every resulting node/site/sample is retained; simulated true ORACLE topology is rejected as GNN input.

## Narrow integration needed

The frozen short-parent registry records wave_v1 simulation output directories. Actual completed simulations/inferences are under American_C3_short_train2mb_wave_v2. The old target joiner therefore rejects genuine actual truth refs.

Review and integrate minimal_train_target_factory.patch into src/arggnn/american_c3_gnn_train_v1.py. The patch is deliberately unapplied. It adds an explicit target-factory argument and binds the factory's actual metadata/source into the checkpoint protocol. The supplied factory validates all20 original scientific rows/seeds against actual simulation config/completion/inferred metadata, and changes only copied runtime simulation_output_directory values. Original NPZ parsing, focal target joins, training schedule and numerical equations are inherited unchanged. The resolver must return the exact same-parent truth ref in genuine completion metadata. No truth copying or aliasing is needed. VAL/TEST are rejected before resolver invocation.

The patch dry-run passed against the current trainer. mock_source_check_receipt_v1.json records the passing metadata/mock check: no model, actual observations, checkpoint, truth, optimizer or GPU was used. _patched_train_for_source_check.py is an AST-only preview, not a production import.

## Root integration API

Import execution_hooks_v1.py by its actual bound path inside the current admitted root context. The following names are concrete callables, not a self-authorizing launcher:

1. actual_train_metadata_v1(bindings) validates public JSON bindings only.
2. build_plan_stage_v1(cfg, parent_id=..., writer=..., runtime_guard=...) performs the future admitted observation-only plan pass. Bind actual original native preparation/focal refs first.
3. run_train_stage_v1(cfg, parent_plan_refs=..., actual_train_bindings=..., writer=..., device=..., worker_tmp_directory=..., resolve_train_truth=..., runtime_guard=..., resume_ref=None) delegates the fixed600-update trainer after the narrow patch is integrated.
4. predict_one_original_VAL_v1(plan_ref, parent_id=..., training_completion_ref=..., writer=..., device=..., worker_tmp_directory=..., runtime_guard=...) seals six checkpoints for one original VAL parent, without a truth-resolver parameter.
5. seal_original_VAL_matrix_stage_v1(all30_prediction_seal_refs, training_completion_ref=..., writer=..., runtime_guard=...) uses the unchanged all30-cell barrier.
6. score_and_select_stage_v1(matrix_ref, writer=..., resolve_val_truth=..., runtime_guard=...) opens external VAL targets only after that barrier and delegates the existing scorer/selector.

Writer implements save(name,value,kind='json'|'torch'|'npz'). runtime_guard is the parent's current admitted callback; device and temporary paths are explicitly root supplied. Current sources and configs must be authenticated before activation. TRAIN and VAL resolvers are separate capabilities. TEST is deferred.

Root can review metadata-only description with:

    source scripts/project_env.sh
    PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" research/reviews/American_C3_PyG_execution_source_20261015_v1/execution_hooks_v1.py --describe

The default CLI is also description only. No scientific launch CLI or admission authority is included.

## Readiness and evaluation

metadata_readiness_inventory_v1.json is a timestamped metadata snapshot:14 completed short simulations and14 completed inferences, not all20. actual_train_public_bindings.inactive.json prebinds available000–013 public metadata;014–019 require completion and exact refs. All20 actual TRAIN plans and each native preparation/focal ref must still be supplied and validated; this packet did not read, assume or bind those scientific payloads. The five original30Mb VAL preparation/observation/plan refs also require root binding. Config/request files are explicitly inactive and contain unresolved refs.

Native ARGformer training/MLM/checkpoint completion is not required to execute the GNN. The shared original focal registry/preparation remains an input dependency. Each original VAL parent can be prepared and six predictions sealed independently; external truth scoring waits for all30 seals across5 parents. Metrics are original unweighted focal-record metrics, not European dense query-bp accuracy. Selection is greatest mean per-parent macro-F1, with earlier epoch on an exact tie.

The32,000,000-byte prospective TRAIN truth-read cap follows the frozen maximum20,000 truth intervals and1,000 query uint8 labels plus axes, not a private-payload inspection. Bounded plan blocks default to32 trees/250,000 nodes/4,096 sites, with512MiB CPU observation cache and focal chunks1024. Neither these caps nor metadata/mock checks certify full-parent GPU runtime fit or scientific performance.

Only this review packet was written. No existing production source was edited, no scientific payload/checkpoint/truth was opened, no model or optimizer was run, no GPU or guard/admission mutation occurred, and no install or Git operation occurred.
