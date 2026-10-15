"""Minimal shared-observation C3 binding for the existing whole-tree PyG GNN.

Root owns execution, resources, writer, device and process lifecycle. No CLI,
admission, private-label capability or scientific payload is read on import.
"""
from __future__ import annotations
import copy
import torch
from .american_c3_common_v1 import CLASSES,require,read
from .american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from .american_c3_gnn_dataset_isolated_v2 import DEFAULTS as PLAN_DEFAULTS,build_plan
from .american_c3_gnn_stream_train_isolated_v2 import validate_plan_v1
from .american_c3_gnn_train_v1 import (
    DEFAULTS as DEVELOPMENT_TRAIN_DEFAULTS,run_train_v1 as run_development_train_v1)
from .american_c3_gnn_predict_isolated_v2 import eligible_checkpoint_v1,seal_prediction_v1
SOURCE="src/arggnn/american_c3_gnn_shared30mb_isolated_v2.py"
REGISTRY_REF={"path":"research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json",
 "sha256":"608da56508757646a0584cb653f7f6af4c910f05d560d6da1b99c0c1ea691f9a"}
TRAIN=tuple("american_c3_train_%03d"%i for i in range(20))
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
TEST=tuple("american_c3_test_%03d"%i for i in range(10))
LENGTH=30000000
MODEL_KWARGS=dict(hidden_dim=32,class_order=list(CLASSES),expected_haplotypes=1600,
 donor_cohorts=2,site_chunk_size=256,checkpoint_sites=True)
DEVELOPMENT_BRANCH="independent20x2Mb_development_training_original30Mb_evaluation"
FUTURE_ORIGINAL_BRANCH="inactive_original20x30Mb_matched_training_source_draft"
KNOWN_TRAIN000={
 "observation_ref":{"path":"experiments/results/American_C3_observed_full30mb_supervisor_repair_20261015_v1/observed/manifest.json",
  "sha256":"c3d58a8d10d901a162f406e2e218a10e17e5f1617b3c626503d72bcbd1b4ccee"},
 "focal_registry_ref":{"path":"experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/focals.npz",
  "sha256":"853b28cf2383d4e7fdf5f2c24526c64ef8071ce81ec04a07d1d69cd7350c8da5"},
 "native_preparation_ref":{"path":"experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json",
  "sha256":"3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa"}}

def original_registry_v1():
    value=read(REGISTRY_REF);rows=value["parents"]
    expected=[*(("train",p,21001+i,1021001+i,92001+i) for i,p in enumerate(TRAIN)),
      *(("val",p,31001+i,1031001+i,93001+i) for i,p in enumerate(VAL)),
      *(("test",p,41001+i,1041001+i,94001+i) for i,p in enumerate(TEST))]
    require(value["schema"]=="american_c3_parent_split_seeds_v1" and len(rows)==35
      and [(r["split"],r["parent_id"],r["ancestry_seed"],r["mutation_seed"],r["path_sampling_seed"])
       for r in rows]==expected,"Exact original split/seeds before any observed payload")
    return {r["parent_id"]:r for r in rows}

def dimensions_v1():
    """Exact class-dependent widths, without creating a model or loading weights."""
    return dict(class_order=list(CLASSES),haplotypes=1600,reference_haplotypes=600,query_haplotypes=1000,
      reference_counts_by_class=[200,200,200],donor_age_cohorts=2,donor_channels=6,
      site_input_width=15,observed_leaf_width=5,hidden_dim=32,leaf_projection_input_width=37,
      directional_GRU_input_width=35,classifier_input_width=96,output_classes=3,
      dummy_fourth_class=False,C4_weight_transfer=False,posterior_floor_or_renormalization=False)

def shared_plan_config_v1(*,observation_ref,focal_registry_ref,native_preparation_ref):
    """Root fills exact common refs, then separately authorizes an observed-only count pass."""
    cfg=copy.deepcopy(PLAN_DEFAULTS)
    cfg.update(observation_ref=observation_ref,focal_registry_ref=focal_registry_ref,
      native_preparation_ref=native_preparation_ref,class_order=list(CLASSES),expected_haplotypes=1600,
      reference_counts=[200,200,200],input_reference_class_mask=None,input_reference_sample_indices=None)
    return cfg

def validate_shared_plan_v1(plan,*,parent_id):
    require(parent_id in TRAIN+VAL,"Current source scope is original TRAIN/VAL; TEST stays deferred")
    validate_plan_v1(plan,parent_id=parent_id,length_bp=LENGTH)
    require(plan["config"]["reference_counts"]==[200,200,200]
      and plan["config"]["native_preparation_ref"] is not None,"Same native preparation and complete panel")
    if parent_id==TRAIN[0]:
        require(all((plan["observation_ref"] if k=="observation_ref" else
          plan["focal_registry_ref"] if k=="focal_registry_ref" else plan["config"][k])==v
          for k,v in KNOWN_TRAIN000.items()),"Exact current original30Mb TRAIN000 common observation/focals")

def validate_shared_metadata_v1(cfg,*,parent_id):
    """Metadata only. Never hashes/opens genotypes, trees, focals or private truth."""
    rows=original_registry_v1()
    require(parent_id in TRAIN+VAL,"Current original TRAIN/VAL scope")
    m=read(cfg["observation_ref"]);prep=read(cfg["native_preparation_ref"])
    require(m["schema"]=="american_c3_observed_v1" and m["parent_id"]==parent_id
      and m["split"]==rows[parent_id]["split"] and m["sequence_length_bp"]==LENGTH
      and m["class_order"]==list(CLASSES) and m["sample_unit"]=="haploid"
      and m["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
      and m["query_ancestry_truth_in_inputs"] is False
      and m["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
      and prep["status"]=="completed_native_C3_path_preparation" and prep["parent_id"]==parent_id
      and prep["observation_ref"]==cfg["observation_ref"] and prep["focals_ref"]==cfg["focal_registry_ref"]
      and prep["class_order"]==list(CLASSES) and prep["genealogy_kind"]==m["genealogy_kind"]
      and prep["query_ancestry_truth_opened"] is False and prep["query_focal_count"]==10000,
      "Same inferred observed genotypes/native focal registry, with no query truth input")
    return dict(parent_id=parent_id,split=rows[parent_id]["split"],sequence_length_bp=LENGTH,
      class_order=list(CLASSES),haplotypes=1600,focal_records=10000,
      observation_ref=cfg["observation_ref"],focal_registry_ref=cfg["focal_registry_ref"],
      native_preparation_ref=cfg["native_preparation_ref"],scientific_payload_opened=False)

def count_shared_plan_v1(cfg,*,parent_id):
    """Future admitted observed-only count/plan pass. No private-truth capability."""
    validate_shared_metadata_v1(cfg,parent_id=parent_id)
    require(cfg["execution_enabled"] is True,"Root must explicitly activate observed-only count pass")
    plan=build_plan(cfg);validate_shared_plan_v1(plan,parent_id=parent_id)
    return plan,resource_plan_v1(plan)

def resource_plan_v1(plan):
    """Count-derived bounds, never a measured memory/time/fit claim."""
    validate_shared_plan_v1(plan,parent_id=plan["parent_id"])
    blocks=plan["blocks"];d=32;h=1600;n=10000
    feature_bound=max(256*b["nodes"]+32*(b["stop_tree"]-b["start_tree"])*h
      +(h+256)*b["sites"] for b in blocks)
    max_trees=max(b["stop_tree"]-b["start_tree"] for b in blocks)
    return dict(schema="C3_shared_complete_history_inference_count_plan_v1",
      parent_id=plan["parent_id"],observation_ref=plan["observation_ref"],focal_registry_ref=plan["focal_registry_ref"],
      trees=plan["trees"],sites=plan["sites"],blocks=len(blocks),
      total_active_marginal_node_occurrences=sum(b["nodes"] for b in blocks),
      max_block_nodes=max(b["nodes"] for b in blocks),max_block_sites=max(b["sites"] for b in blocks),
      max_block_trees=max_trees,feature_bank_byte_bound_per_block=feature_bound,
      one_block_context_or_history_FP32_bytes=max_trees*h*d*4,
      CPU_focal_forward_state_bytes=n*d*4,one_direction_recurrent_state_bytes=h*d*4,
      FP32_posterior_bytes=n*3*4,prediction_NPZ_planning_cap_bytes=1048576,
      prediction_receipt_planning_cap_bytes=1048576,inference_tree_encoding_passes=2,
      full_chromosome_state_carry=True,history_reset_only_at_chromosome_endpoints=True,
      every_tree_node_site_sample_retained=True,finite_halo_or_tree_pruning=False,
      dataloader_workers_and_prefetch_require_root_count_bound=True,
      GPU_activations_workspace_RSS_time_and_fit_unmeasured=True,
      count_plan_is_not_admission_or_fit_readiness=True)

def seal_shared_val_prediction_v1(model,forward_loader,reverse_loader,*,parent_id,
    checkpoint_ref,training_completion_ref,writer,device,runtime_guard):
    """Use disclosed short-trained checkpoint on exact original independent VAL observation.

    Missing original30Mb TRAIN parents do not block this branch. The existing
    strict development checkpoint/completion validator remains authoritative.
    No VAL/TEST truth path/resolver is accepted.
    """
    require(parent_id in VAL,"Only original independent VAL; TRAIN000 remains TRAIN, TEST deferred")
    require(type(model) is AmericanC3WholeTreeGNNV1 and model.class_order==CLASSES
      and model.hidden_dim==32 and model.expected_haplotypes==1600
      and model.donor_cohorts==2 and model.donor_channels==6
      and model.site_chunk_size==256 and model.checkpoint_sites is True
      and model.site_mlp[0].in_features==15 and model.leaf_projection[0].in_features==37
      and model.forward_gru.input_size==model.reverse_gru.input_size==35
      and model.classifier[-1].out_features==3,
      "Exact fresh C3 architecture/dimensions; no sliced/relabelled C4 head")
    require(callable(runtime_guard) and callable(getattr(writer,"save",None)),"Root writer/resource capabilities")
    runtime_guard("before_shared_original_VAL_prediction")
    training,metadata=eligible_checkpoint_v1(checkpoint_ref,training_completion_ref)
    require(parent_id not in training["trained_parent_ids"]
      and training["train_sequence_lengths_bp"]==[2000000]*20,
      "Explicit independent short20 development-training provenance")
    fd,rd=forward_loader.dataset,reverse_loader.dataset
    require(fd.plan_ref==rd.plan_ref and fd.direction=="forward" and rd.direction=="reverse"
      and fd.plan["parent_id"]==parent_id,"Same complete original VAL plan in opposite orders")
    validate_shared_plan_v1(fd.plan,parent_id=parent_id)
    validate_shared_metadata_v1(fd.plan["config"],parent_id=parent_id)
    class DisclosedWriter:
        def save(self,name,value,kind="json"):
            if kind=="json" and value.get("status")=="sealed_C3_original_heldout_predictions":
                value=dict(value,training_branch=DEVELOPMENT_BRANCH,
                    trained_parent_ids=list(training["trained_parent_ids"]),
                    train_sequence_lengths_bp=[2000000]*20,evaluation_length_bp=LENGTH,
                    training_2Mb_to_30Mb_length_transfer=True,
                    original_full30Mb_matched_training_claimed=False,
                    same_observation_and_native_focals=True,
                    original_C3_scorer="src/arggnn/american_c3_evaluate_v1.py",
                    complete_original_5VAL_selection_required_before_TEST=True)
            return writer.save(name,value,kind=kind)
    # Retain the original seal status/fields and worker lifecycle so its existing
    # all30-cell VAL barrier/selection/scorer accepts these ordinary C3 seals.
    return seal_prediction_v1(model,forward_loader,reverse_loader,split="val",parent_id=parent_id,
        checkpoint_ref=checkpoint_ref,training_completion_ref=training_completion_ref,
        writer=DisclosedWriter(),device=device,runtime_guard=runtime_guard)
