# H050 shared admission: metadata-only direct binding fix

Source only. No admission/guard/job/probe/Git action was executed. No C3-bound
pool, canonical, adapter, route, config, input or source file was changed.
The failed partial H050 shared TRAIN v1 packet remains preserved.

Actual failure: the original pending60direct bindings contained exactly one
disallowed scientific payload: H045 own_state.pass_30.pt SHA11dc132ac1958c452fa1d2651bd261258aac6c036ebbdbe7175f235148905331.
Original v4 accepts only .py/.sh/.json direct source/runtime authority metadata.
Original H050 validate_spec repeated that same direct_required list at runtime,
so admission-only filtering would not work.

The new private context loads exact original bridge_common bytes, changes its
CONTEXT to this actual declared context adapter, and narrows only its private
direct_required_refs function. The ONE exact original checkpoint ref moves behind
a direct-bound JSON scientific-input proxy. Arbitrary other data bindings fail;
no blanket suffix filtering is allowed. All original producer/launcher/guard/
repair/source/authority refs remain direct, including original common/context
source bodies. The original producer still authenticates the exact checkpoint
payload bytes before loading it. The original context function BODY is compiled
unchanged into a private metadata namespace; no guard/Torch/model/science patch.

Bind creates separate config/manifest/request/proxy files. Original scientific
config/ref values are preserved; only declared context/source/operational metadata
fields change. Original source closure remains and gains exact new source refs.
Fresh H050 uses jobH050-direct-readout-shared-TRAIN-v2, output
experiments/results/H050_shared_direct_readout_train_seed101_v2,
and this folder/admitted_H050_shared_train_v2. No failed namespace is reused.
The assembly wrapper takes actual admitted H050 job_id, keeps C3's existing
job/input/storage/direct bindings unchanged, and adds successor source/context/
proxy only to H050 and the fresh controller reservation. It does not extend C3's
immutable reservation. Pool/launcher/bootstrap/certificate stay unchanged.

Root command sequence (PYTHONPATH=.:src:scripts, python=.runtime/env/bin/python -B):

1. Driver=root_h050_shared_admission_fix_v1.py in this folder.
   Driver bind --execute --record
   research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/H050_worker_admission.root_active_v2.json
   --sha256 b7c9d9272f4b44c6940f121e49a45aafe5c6e78f5dcd1210cd386c45b4e40515

   This is metadata binding only. Returned binding_ref points to
   this folder/metadata_v1/binding.json. Preserve its exact returned SHA.

2. Driver admit --execute --record BINDING_PATH --sha256 BINDING_SHA

   Returns genuine H050 admitted_ref. Do not overwrite the failed v1 packet.

3. Create a NEW active root controller request with schema
   GPU45_C3cache_H050_controller_admission_v1, status=active,
   root_authorizer=root_agent, shared_route_ref=the EXISTING exact route,
   C3_admitted_ref=the already successful actual C3 admission (unchanged),
   H050_admitted_ref=the returned new H050 admitted_ref, reserved_bytes=33554432.
   Driver assemble --execute --record CONTROLLER_REQUEST --sha256 REQUEST_SHA.

   Controller packet=this folder/admitted_controller_v1;
   controller job=GPU4-C3cache-H050-shared-controller-v2.
   This produces/validates the actual final pool capability.

4. Same installed launcher start/run commands and returned capability ref.
   Run C3-GNN-cachev3-shared-TRAIN-v1 and H050-direct-readout-shared-TRAIN-v2.
   NativeGPU5 stays untouched; actual resource checks still govern launch.

11 pure fixtures pass against the actual failed binding list;63direct metadata
bindings result from the classification fixture, within the unchanged64limit.
Compilation checks the exact frozen context/admission/assembly adaptation
boundaries. This is not a runtime qualification or a claim that reserve succeeded.
