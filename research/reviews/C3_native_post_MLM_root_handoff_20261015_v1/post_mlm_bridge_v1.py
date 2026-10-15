"""Truthful v12 MLM adoption and contrastive-only fit-envelope bridge.

No CLI, launcher, model import, storage guard, accounting or admission is added.
Root must bind this and an actual qualified run_child in a fresh current-policy
consumer. Adoption/assembly opens bounded JSON metadata only. The run hook
delegates normal scientific input checks and execution to existing code.
"""
from __future__ import annotations
import inspect,re
from pathlib import Path
from arggnn.american_c3_common_v1 import ROOT,CLASSES,require,ref
from arggnn.american_c3_native_contract_v1 import AUTHOR_MANIFEST_REF,metadata
SOURCE="research/reviews/C3_native_post_MLM_root_handoff_20261015_v1/post_mlm_bridge_v1.py"
PREP_REF=dict(path="experiments/results/American_C3_native_paths_train000_full30mb_seed92001_v3/completed.json",
    sha256="3cc49fbf55bd4b4b3859a93ce1123e9218f407f7c7299ad5029be81f59c7c6fa")
V12="research/reviews/American_C3_native_MLM_amendment_v12"
MLM_PARENT="experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8"
MLM_NAMESPACE=MLM_PARENT+"/amended_v12_attempt1"

def _counters(value):
    require(isinstance(value,dict) and value.get("optimizer_names")==["DecoupledAdamW"]
        and type(value.get("parameter_state_count")) is int and value["parameter_state_count"]>0
        and value.get("unique_step_values")==[224775]
        and value.get("minimum_step")==value.get("maximum_step")==224775
        and value.get("source")=="actual_author_optimizer_state_dict",
        "Genuine v12 live/saved Adam counters at update224775")

def _checkpoint_reference(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"}
        and isinstance(record["path"],str) and isinstance(record["sha256"],str)
        and re.fullmatch(r"[0-9a-f]{64}",record["sha256"]),"Actual checkpoint path/SHA reference")
    path=Path(record["path"])
    require(path.parent==Path(MLM_NAMESPACE)/"mlm/checkpoints" and path.suffix==".pt"
        and not path.name.startswith("latest"),"Actual own final regular checkpoint reference, never an alias")

def v12_mlm_phase_v1(mlm_completion_ref,mlm_root_wait_ref,spec):
    """Adopt actual published v12 completion after genuine root/native waits0.
    Does not load/hash checkpoint/corpus. The existing child authenticates
    initializer bytes at the actual admitted execution boundary.
    """
    require(spec["parent_id"]=="american_c3_train_000" and spec["refs"]["preparation"]==PREP_REF,
        "Only original full30Mb TRAIN000 v3 corpus, no other-parent adoption")
    require(mlm_completion_ref["path"]==MLM_PARENT+"/completed.json","Actual published v12 parent completion")
    value=metadata(mlm_completion_ref);outer=metadata(mlm_root_wait_ref)
    require(value.get("status")=="completed_C3_native_full_MLM_v1"
        and value.get("execution_namespace_version")==12
        and value.get("original_parent_scopes_fully_charged") is True
        and value.get("actual_producer_ref")==ref(V12+"/entry_v12.py")
        and value.get("parent_id")==spec["parent_id"] and value.get("preparation_ref")==PREP_REF
        and value.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and value.get("query_ancestry_truth_opened") is False
        and value.get("model_reused_from_other_parent") is False and value.get("contrastive_executed") is False,
        "Authentic same-parent source-bound v12 MLM completion")
    require(outer.get("returncode")==0 and outer.get("actual_direct_launcher_wait") is True
        and outer.get("all_owned_processes_and_groups_drained") is True and outer.get("operational_failure") is None
        and outer.get("phase")=="mlm" and outer.get("parent_id")==spec["parent_id"]
        and outer.get("completed_ref")==mlm_completion_ref
        and outer.get("driver_ref")==ref(V12+"/root_native_driver_v1.py")
        and outer.get("execution_amendment_ref")==value.get("execution_amendment_ref")
        and value.get("actual_wait_ref") in outer.get("native_actual_wait_refs",[]),
        "Actual root launcher wait0/drain bound to published completion")
    namespace_ref=value["namespace_child_completion_ref"]
    require(namespace_ref["path"]==MLM_NAMESPACE+"/completed.json","Actual isolated v12 namespace")
    namespace=metadata(namespace_ref)
    require(value==dict(namespace,namespace_child_completion_ref=namespace_ref,execution_namespace_version=12,
        original_parent_scopes_fully_charged=True),"Published completion preserves v12 namespace fields")
    child=metadata(value["child_completion_ref"]);wait=metadata(value["actual_wait_ref"])
    require(value["child_completion_ref"]["path"]==MLM_NAMESPACE+"/mlm/completed.json"
        and child.get("schema")=="american_c3_native_mlm_child_completion_v1"
        and child.get("status")=="completed_C3_native_MLM_child" and child.get("stage")=="mlm"
        and child.get("parent_id")==spec["parent_id"] and child.get("preparation_ref")==PREP_REF
        and child.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and child.get("runtime_ref")==value["runtime_ref"] and child.get("initializer_checkpoint_ref") is None
        and child.get("query_ancestry_truth_opened") is False and child.get("author_code_modified") is False,
        "Actual fresh author MLM child")
    require(wait.get("schema")=="american_c3_native_direct_wait_and_drain_v1"
        and wait.get("returncode")==0 and wait.get("actual_direct_child_wait") is True
        and wait.get("process_group_ESRCH") is True and wait.get("owned_identities_drained") is True
        and wait.get("no_adopted_live_writer") is True and wait.get("adopted_wait_records_may_be_truncated") is False
        and wait.get("guard_failure_type") is None and wait.get("interruption_signal") is None
        and wait.get("child_completion_ref")==value["child_completion_ref"]
        and wait.get("child_input_ref")==child["child_input_ref"],"Actual complete native wait0/drain")
    for receipt in (value,child):
        require(receipt.get("completed_epochs")==5 and receipt.get("actual_optimizer_updates")==224775
            and receipt.get("observed_counters_verified") is True
            and receipt["trainer_timestamp"]["epoch"]==5 and receipt["trainer_timestamp"]["batch"]==224775,
            "Actual five epochs and224775 updates")
        _counters(receipt["trainer_optimizer_counters"]);_counters(receipt["checkpoint_optimizer_counters"])
    require(all(child[key]==value[key] for key in ("checkpoint_ref","corpus_authentication_ref",
        "trainer_timestamp","trainer_optimizer_counters","checkpoint_optimizer_counters")),
        "Parent forwards genuine child checkpoint/counters unchanged")
    _checkpoint_reference(child["checkpoint_ref"])
    prep=metadata(PREP_REF);authentication=metadata(child["corpus_authentication_ref"])
    expected={"mlm/train":prep["native_dataset_refs"]["mlm/train"],
        "mlm/val":prep["native_dataset_refs"]["mlm/val"],"vocab":prep["vocab_ref"]}
    require(authentication.get("status")=="authenticated_full_MLM_inputs"
        and authentication.get("preparation_ref")==PREP_REF
        and {k:v["ref"] for k,v in authentication["inputs"].items()}==expected
        and all(v["sha256_verified"] and v["stable_read_verified"] for v in authentication["inputs"].values()),
        "Original v12 complete MLM corpus byte-authentication evidence")
    return dict(child_completion_ref=value["child_completion_ref"],actual_wait_ref=value["actual_wait_ref"],
        checkpoint_ref=child["checkpoint_ref"])

def contrastive_packet_v1(mlm_phase):
    return dict(name="contrastive",stage="contrastive",initializer_checkpoint_ref=mlm_phase["checkpoint_ref"])

def complete_fit_v1(*,spec,mlm_completion_ref,mlm_root_wait_ref,contrastive_completion_ref,contrastive_wait_ref,writer):
    """Write the generic fit envelope only after both genuine native phases."""
    mlm_phase=v12_mlm_phase_v1(mlm_completion_ref,mlm_root_wait_ref,spec)
    child=metadata(contrastive_completion_ref);wait=metadata(contrastive_wait_ref);counts=metadata(spec["refs"]["counts"])
    require(child.get("schema")=="american_c3_native_child_completion_v1"
        and child.get("status")=="completed_C3_native_child" and child.get("stage")=="contrastive"
        and child.get("parent_id")==spec["parent_id"] and child.get("preparation_ref")==PREP_REF
        and child.get("counts_ref")==spec["refs"]["counts"] and child.get("runtime_ref")==spec["refs"]["runtime"]
        and child.get("author_source_manifest_ref")==AUTHOR_MANIFEST_REF
        and child.get("initializer_checkpoint_ref")==mlm_phase["checkpoint_ref"]
        and child.get("completed_epochs")==1
        and child.get("actual_optimizer_updates")==counts["contrastive_optimizer_updates"]>0
        and child.get("uniform_class_weight_fallback_preserved") is True
        and child.get("training_config_plumbing_correction") is False
        and child.get("query_ancestry_truth_opened") is False and child.get("author_code_modified") is False,
        "Actual native contrastive completion initialized by genuine v12 MLM")
    require(counts.get("schema")=="american_c3_native_actual_counts_v1"
        and counts.get("status")=="completed_C3_native_observation_only_counts"
        and counts.get("source_ref")==ref("src/arggnn/american_c3_native_concat_v1.py")
        and counts.get("parent_id")==spec["parent_id"] and counts.get("preparation_ref")==PREP_REF
        and counts.get("MLM_optimizer_updates")==224775
        and counts["contrastive_optimizer_updates"]==counts["datasets"]["contrastive/train"]["rows"]//4096,
        "Actual full same-corpus count receipt and exact release drop_last budget")
    require(wait.get("returncode")==0 and wait.get("actual_direct_child_wait") is True
        and wait.get("process_group_ESRCH") is True and wait.get("no_adopted_live_writer") is True
        and wait.get("adopted_wait_records_may_be_truncated") is False
        and wait.get("guard_failure_type") is None and wait.get("interruption_signal") is None
        and wait.get("child_completion_ref")==contrastive_completion_ref
        and wait.get("child_input_ref")==child["child_input_ref"],"Actual successful contrastive wait0/drain")
    phases=dict(mlm=mlm_phase,contrastive=dict(child_completion_ref=contrastive_completion_ref,
        actual_wait_ref=contrastive_wait_ref,checkpoint_ref=child["checkpoint_ref"]))
    return writer.save("completed.json",dict(status="completed_C3_fresh_native_fit",parent_id=spec["parent_id"],
        preparation_ref=PREP_REF,counts_ref=spec["refs"]["counts"],class_order=list(CLASSES),
        author_source_manifest_ref=AUTHOR_MANIFEST_REF,phases=phases,uniform_class_weight_fallback_preserved=True,
        training_config_plumbing_correction=False,query_ancestry_truth_opened=False,
        model_reused_from_other_parent=False,author_Table1_exact_reproduction=False,
        mlm_completion_ref=mlm_completion_ref,mlm_root_actual_wait_ref=mlm_root_wait_ref,
        post_MLM_bridge_source_ref=ref(SOURCE),MLM_reexecuted=False))

def run_contrastive_after_v12_v1(record,spec,cfg,ctx,lease,*,run_child,mlm_completion_ref,mlm_root_wait_ref):
    """Root integration hook using the existing supervisor API.
    The caller must already admit actual counts/config/runtime/qualified child,
    lifecycle and IPC sources under prospective980. This is not an entrypoint.
    """
    from arggnn.american_c3_native_contract_v1 import preparation_v1,counts_v1
    require(spec["phase"]==cfg["phase"]=="fit","Existing generic fit child contract, contrastive only")
    require(spec["source_sha256"].get(SOURCE)==ref(SOURCE)["sha256"],"Bridge in actual source closure")
    run_source=Path(inspect.getsourcefile(run_child)).resolve();run_ref=ref(run_source)
    require(spec["source_sha256"].get(run_ref["path"])==run_ref["sha256"],"Qualified child supervisor source bound")
    prep,_,_=preparation_v1(spec);counts=counts_v1(spec,prep)
    require(counts["MLM_optimizer_updates"]==224775,"Same exact full-corpus MLM count budget")
    mlm_phase=v12_mlm_phase_v1(mlm_completion_ref,mlm_root_wait_ref,spec)
    ctx["boundary_guard"]("before_complete_native_contrastive_after_v12",required_write_bytes=cfg["phase_write_budget_bytes"])
    _,completion_ref,wait_ref=run_child(contrastive_packet_v1(mlm_phase),record=record,spec=spec,cfg=cfg,ctx=ctx,lease=lease)
    return complete_fit_v1(spec=spec,mlm_completion_ref=mlm_completion_ref,mlm_root_wait_ref=mlm_root_wait_ref,
        contrastive_completion_ref=completion_ref,contrastive_wait_ref=wait_ref,writer=ctx["writer"])
