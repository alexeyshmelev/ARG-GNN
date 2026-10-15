"""Minimal C3 execution hooks around existing scientific engines.

No launcher, storage/lease implementation or execution authority. No Torch or
scientific payload import occurs for --describe. Root supplies current admitted
writer/device/resource capabilities and separate TRAIN/VAL private resolvers.
"""
from __future__ import annotations
from pathlib import Path
import argparse,copy,dataclasses,inspect,json,sys
from arggnn.american_c3_common_v1 import CLASSES,require,read,ref

SOURCE="research/reviews/American_C3_PyG_execution_source_20261015_v1/execution_hooks_v1.py"
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
BINDING_SCHEMA="C3_actual_TRAIN_simulation_binding_operational_v1"


def describe_v1():
    return dict(status="source_only_no_execution_authority",TRAIN_parent_ids=list(TRAIN),
        TRAIN_length_bp=2000000,epochs=30,complete_parent_updates=600,focals_per_parent=10000,
        VAL_parent_ids=list(VAL),VAL_length_bp=30000000,checkpoint_epochs=[5,10,15,20,25,30],
        parameters=29571,haplotypes=1600,reference_haplotypes=600,query_haplotypes=1000,
        class_order=list(CLASSES),reference_counts=[200,200,200],
        source_ref=ref(SOURCE),scientific_payload_opened=False,Torch_imported=("torch" in sys.modules),
        required_train_patch="minimal_train_target_factory.patch",
        current_GPU_admission_and_writer_capabilities="root_owned_not_implemented_here")


def actual_train_metadata_v1(bindings):
    """Only JSON metadata reads. Never bind/hash/open a private/scientific payload."""
    from arggnn.american_c3_gnn_targets_v1 import authoritative_registry_v1,REGISTRY_REF
    rows=authoritative_registry_v1()
    require(isinstance(bindings,dict) and set(bindings)==set(TRAIN),"All20 original short TRAIN identities")
    result={}
    science=("schema","species","model","parent_id","split","sample_unit","sample_counts",
        "sequence_length_bp","mutation_rate","recombination_rate","admixture_generation",
        "ancestry_seed","mutation_seed","class_order")
    for parent in TRAIN:
        binding=bindings[parent]
        require(set(binding)=={"simulation_completion_ref","simulation_config_ref","inferred_observation_ref"},
            "Only actual public simulation/config/inferred-observation metadata refs")
        completion=read(binding["simulation_completion_ref"])
        cfg=read(binding["simulation_config_ref"])
        inferred=read(binding["inferred_observation_ref"])
        template=read(rows[parent]["simulation_config_ref"])
        require(all(cfg[k]==template[k] for k in science) and cfg["execution_enabled"] is True,
            "Actual completed simulation must preserve all frozen TRAIN science/seeds")
        require(completion["status"]=="completed_C3_simulation_reconstruction"
            and completion["parent_id"]==parent and completion["class_order"]==list(CLASSES)
            and completion["sample_unit"]=="haploid", "Actual same-parent C3 simulation completion")
        require(inferred["schema"]=="american_c3_observed_v1" and inferred["parent_id"]==parent
            and inferred["split"]=="train" and inferred["sequence_length_bp"]==2000000
            and inferred["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and inferred["class_order"]==list(CLASSES) and inferred["sample_unit"]=="haploid"
            and inferred["source_observation_ref"]==completion["observation_ref"]
            and inferred["query_ancestry_truth_in_inputs"] is False
            and inferred["genealogy_kind"]==
                "inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
            "Actual inferred observations must descend from that exact simulation, never the ORACLE manifest")
        simulation_dir=str(Path(binding["simulation_completion_ref"]["path"]).parent)
        expected=simulation_dir+"/truth/c3_pulse_ancestry.npz"
        require(completion["private_truth_ref"]["path"]==expected,
            "Actual same-stage private truth handle; payload stays closed")
        result[parent]=dict(binding=binding,simulation_directory=simulation_dir,
            private_truth_ref=completion["private_truth_ref"],
            original_scientific_parent_row=rows[parent],registry_ref=REGISTRY_REF)
    return result


def make_actual_target_joiner_factory_v1(bindings):
    """Reuse original truth parsing/joins, changing only validated physical path.

    The original target source pins wave_v1 output paths. Actual completed wave_v2
    paths are bound by real simulation/inferred metadata instead of copied or
    aliased truth. The factory has no VAL/TEST membership or resolver capability.
    """
    from arggnn.american_c3_gnn_targets_v1 import TrainTargetJoinV1,REGISTRY_REF
    metadata=actual_train_metadata_v1(bindings)
    class ActualTrainTargetJoinV1(TrainTargetJoinV1):
        def __init__(self,resolve_train_truth,*,max_truth_uncompressed_bytes):
            def actual_resolve(parent):
                require(parent in TRAIN,"Only canonical short TRAIN before resolver invocation")
                value=resolve_train_truth(parent)
                require(value==metadata[parent]["private_truth_ref"],
                    "Resolver must return the genuine metadata-bound same-parent truth ref")
                return value
            super().__init__(actual_resolve,max_truth_uncompressed_bytes=max_truth_uncompressed_bytes)
            self.rows={parent:dict(row,simulation_output_directory=metadata[parent]["simulation_directory"])
                       for parent,row in self.rows.items()}
        def join(self,parent_id,*,split,plan,observation_seal):
            require(split=="train" and parent_id in TRAIN,"No held-out target join")
            require(plan["observation_ref"]==metadata[parent_id]["binding"]["inferred_observation_ref"],
                "Target join binds the same actual inferred observations")
            value=super().join(parent_id,split=split,plan=plan,observation_seal=observation_seal)
            receipt=dict(value.receipt,actual_simulation_binding_schema=BINDING_SCHEMA,
                actual_simulation_completion_ref=metadata[parent_id]["binding"]["simulation_completion_ref"],
                actual_simulation_config_ref=metadata[parent_id]["binding"]["simulation_config_ref"],
                original_scientific_registry_unchanged=True,target_equations_unchanged=True)
            return dataclasses.replace(value,receipt=receipt)
    ActualTrainTargetJoinV1.execution_binding=dict(schema=BINDING_SCHEMA,registry_ref=REGISTRY_REF,
        source_refs=[ref(SOURCE)],parents=copy.deepcopy(bindings),target_equations_unchanged=True,
        change="validated_actual_wave_output_paths_only_no_truth_copy_or_node_join_change")
    return ActualTrainTargetJoinV1


def run_train_stage_v1(cfg,*,parent_plan_refs,actual_train_bindings,writer,device,
        worker_tmp_directory,resolve_train_truth,runtime_guard,resume_ref=None,
        observation_provider_factory=None):
    """Canonical20x2Mb/600-update training; requires the supplied narrow patch."""
    require(callable(runtime_guard),"Root-owned current runtime boundary")
    runtime_guard("before_C3_TRAIN_scientific_import")
    from arggnn import american_c3_gnn_train_v1 as train
    require("train_target_joiner_factory" in inspect.signature(train.run_train_v1).parameters,
        "Root must first review/integrate minimal_train_target_factory.patch; no monkeypatch or old-path fallback")
    if observation_provider_factory is not None:
        require("observation_provider_factory" in inspect.signature(train.run_train_v1).parameters,
            "Root must review optional provider-factory successor; no monkeypatch")
    factory=make_actual_target_joiner_factory_v1(actual_train_bindings)
    return train.run_train_v1(cfg,parent_plan_refs=parent_plan_refs,writer=writer,device=device,
        worker_tmp_directory=worker_tmp_directory,resolve_train_truth=resolve_train_truth,
        runtime_guard=runtime_guard,resume_ref=resume_ref,train_target_joiner_factory=factory,
        **({"observation_provider_factory":observation_provider_factory}
            if observation_provider_factory is not None else {}))


def build_plan_stage_v1(cfg,*,parent_id,writer,runtime_guard):
    """Future admitted observation-only pass; never invoked by --describe."""
    require(parent_id in TRAIN+VAL and cfg["execution_enabled"] is True
        and callable(runtime_guard) and callable(getattr(writer,"save",None)),"Activated root plan stage")
    runtime_guard("before_C3_observed_plan_payload")
    from arggnn.american_c3_gnn_dataset_v1 import build_plan
    from arggnn.american_c3_gnn_stream_train_v1 import validate_plan_v1
    plan=build_plan(cfg)
    validate_plan_v1(plan,parent_id=parent_id,length_bp=2000000 if parent_id in TRAIN else 30000000)
    if parent_id in VAL:
        from arggnn.american_c3_gnn_shared30mb_v1 import validate_shared_plan_v1,validate_shared_metadata_v1
        validate_shared_plan_v1(plan,parent_id=parent_id)
        validate_shared_metadata_v1(cfg,parent_id=parent_id)
    runtime_guard("before_C3_observed_plan_write")
    plan_ref=writer.save(parent_id+"_block_index.json",plan)
    return writer.save(parent_id+"_plan_completed.json",dict(status="completed_C3_observation_only_block_plan",
        parent_id=parent_id,block_index_ref=plan_ref,trees=plan["trees"],sites=plan["sites"],
        blocks=len(plan["blocks"]),focal_records=10000,targets_opened=False))


def predict_one_original_VAL_v1(plan_ref,*,parent_id,training_completion_ref,writer,device,
        worker_tmp_directory,runtime_guard):
    """Six candidates for one independent original30Mb parent, without truth.

    Different original VAL parents may be prepared/executed separately. Their
   30 ordinary seals must all exist before score_and_select_stage_v1 is reached.
    Native ARGformer completion/MLM/checkpoints are not prerequisites.
    """
    require(parent_id in VAL and callable(runtime_guard),"Original VAL only; TEST deferred")
    runtime_guard("before_C3_VAL_scientific_import")
    import torch
    from arggnn.american_c3_gnn_shared30mb_v1 import (
        MODEL_KWARGS,validate_shared_plan_v1,validate_shared_metadata_v1,seal_shared_val_prediction_v1)
    from arggnn.american_c3_gnn_predict_v1 import completed_training_v1
    from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
    from arggnn.american_c3_gnn_dataset_v1 import AmericanC3WholeTreeDatasetV1,make_dataloader
    plan=read(plan_ref);validate_shared_plan_v1(plan,parent_id=parent_id)
    validate_shared_metadata_v1(plan["config"],parent_id=parent_id)
    training=completed_training_v1(training_completion_ref)
    actual=torch.device(device)
    require(actual.type in ("cpu","cuda") and (actual.type!="cuda" or actual.index is not None),
        "Root supplies one explicit admitted device")
    seals=[]
    with torch.device("cpu"):
        model=AmericanC3WholeTreeGNNV1(**MODEL_KWARGS)
    model=model.to(actual)
    for candidate in training["checkpoint_candidates"]:
        runtime_guard("before_C3_VAL_checkpoint_cell")
        forward=AmericanC3WholeTreeDatasetV1(plan_ref,direction="forward",worker_tmp_directory=worker_tmp_directory)
        reverse=AmericanC3WholeTreeDatasetV1(plan_ref,direction="reverse",worker_tmp_directory=worker_tmp_directory)
        # Root-owned process only. Avoid worker/prefetch/IPC multiplication.
        fl=make_dataloader(forward,num_workers=0,pin_memory=False)
        rl=make_dataloader(reverse,num_workers=0,pin_memory=False)
        try:
            seals.append(seal_shared_val_prediction_v1(model,fl,rl,parent_id=parent_id,
                checkpoint_ref=candidate["checkpoint_ref"],training_completion_ref=training_completion_ref,
                writer=writer,device=actual,runtime_guard=runtime_guard))
        finally:
            # Existing num_workers0 drain has no iterator workers to release.
            # Release retained native tree/mmap state on success and failure.
            forward._state=reverse._state=None
            forward._pid=reverse._pid=None
    del model
    return writer.save(parent_id+"_all6_prediction_seals.json",dict(
        status="sealed_C3_one_original_VAL_all6_candidates",parent_id=parent_id,
        training_completion_ref=training_completion_ref,prediction_seal_refs=seals,
        common_focal_records=10000,private_truth_opened=False,TEST_deferred=True,
        complete_original5VAL_barrier_still_required=True))


def seal_original_VAL_matrix_stage_v1(prediction_seal_refs,*,training_completion_ref,writer,runtime_guard):
    require(callable(runtime_guard),"Root current boundary")
    runtime_guard("before_C3_all30_VAL_matrix")
    from arggnn.american_c3_gnn_predict_v1 import seal_validation_matrix_v1
    return seal_validation_matrix_v1(prediction_seal_refs,
        training_completion_ref=training_completion_ref,writer=writer)


def score_and_select_stage_v1(matrix_ref,*,writer,resolve_val_truth,runtime_guard):
    """Separate held-out resolver capability, reachable after the existing30-cell barrier."""
    require(callable(runtime_guard),"Root current boundary")
    runtime_guard("before_C3_external_VAL_scoring_import")
    from arggnn.american_c3_gnn_selection_v1 import (
        score_validation_matrix_v1,select_original_val_checkpoint_v1)
    score_ref=score_validation_matrix_v1(matrix_ref,writer=writer,
        resolve_val_truth=resolve_val_truth,runtime_guard=runtime_guard)
    return select_original_val_checkpoint_v1(score_ref,writer=writer)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--describe",action="store_true")
    p.parse_args()
    print(json.dumps(describe_v1(),indent=2))
