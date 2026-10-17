"""Concrete later H059 stage builder and dispatch to exact frozen v3 native scientific calls.
No CLI resources, fits, predictions or truth reads occur at import or in request construction.
A real reviewed current-policy driver owns every separately admitted execution.
"""
from pathlib import Path
import copy,importlib,json,re
from datetime import datetime,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H059_saved_outer000_acceptance_and_continuation_source_20261017_v2"
SCIENCE="research/reviews/European_outer8_preparation_source_20261016_v3"
PARENTS=tuple("ap_expanded_outer_%03d"%i for i in range(8))
FREEZE_REF={"path":"research/reviews/European_outer8_ROOT_freeze_20261017_v1/method_freeze.root_frozen_v1.json","sha256":"6d6786c92158b2e7e36137fb9b51f709f814704f7e8e8cdc29b75a651f245bea"}
METHODS=("H050_stable_leaf_memory","H050_h045_continued_training","AncestralPaths_native_Dense5","ARGMix_current_native","ARGformer_native_within_ARG")
def need(ok,message):
 if not ok:raise ValueError(message)
def stamp(text):
 t=datetime.fromisoformat(text);need(t.tzinfo and t<=datetime.now(timezone.utc),"Nonfuture aware metadata timestamp");return t
def stage_plan_v1(run_id):
 need(isinstance(run_id,str)and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,63}",run_id)is not None,"Flat immutable run ID")
 result=[]
 for parent in PARENTS:
  for method in METHODS:
   for stage in (("observe","prepare","fit","predict")if method==METHODS[-1]else("predict",)):
    result.append(dict(parent_id=parent,method_id=method,stage=stage,
     output_directory="experiments/results/European_outer8_predictions_"+run_id+"/cells/"+method+"/"+parent,
     primary_product_count=1 if stage=="predict"else 0,raw_top5_product_count=1 if stage=="predict"and method==METHODS[-1]else 0))
 need(len(result)==64 and sum(r["primary_product_count"]for r in result)==40 and sum(r["raw_top5_product_count"]for r in result)==8,"Exact eight-parent40primary+8rawtop5 frozen inventory")
 return result
def genuine_stage_v1(evidence,read,ref,*,parent=None,method=None,stage=None):
 need(isinstance(evidence,dict),"Explicit completed stage evidence")
 wait=read(evidence["actual_wait_ref"]);terminal=read(evidence["ROOT_terminal_ref"]);admission=read(evidence["input_ref"])
 source=ref(evidence["driver_source_ref"]["path"]);review=read(evidence["driver_review_ref"])
 need(source==evidence["driver_source_ref"]and review["independent"]is True and review["verdict"]in("SOURCE_PASS","PASS")
  and review["blockers"]==[] and (review.get("source_ref")==source or source in review.get("generation_source_refs",[])),"Genuine independently reviewed actual stage driver")
 need(type(wait["returncode"])is int and wait["returncode"]==0 and wait["actual_direct_worker_wait"]is True
  and wait["actual_OS_wait_status"]==0 and wait["all_owned_processes_and_groups_drained"]is True
  and wait["operational_failure"]is None and wait["input_ref"]==evidence["input_ref"]
  and wait["completion_ref"]==evidence["completion_ref"]and wait["driver_source_ref"]==source
  and terminal["root_authorizer"]=="root_agent"and terminal["actual_tool_terminal_consumed"]is True
  and type(terminal["actual_tool_exit_code"])is int and terminal["actual_tool_exit_code"]==0
  and terminal["actual_wait_ref"]==evidence["actual_wait_ref"]and terminal["input_ref"]==evidence["input_ref"]
  and terminal["completion_ref"]==evidence["completion_ref"],"Genuine new stage wait0/drain/consumed outer0; no old inferred wait")
 completion=read(evidence["completion_ref"])
 need(admission.get("operational_source_ref",admission.get("separately_admitted_driver_source_ref"))==source,
  "Exact actual stage admission binds reviewed driver")
 if parent is not None:need(admission["parent_id"]==completion["parent_id"]==parent,"Same requested biological parent")
 if method is not None:need(admission["method_id"]==method,"Same frozen method stage evidence")
 if stage is not None:need(admission["stage"]==stage,"Same concrete original stage evidence")
 return completion
def data_ready_v1(parent,evidence,read,ref):
 need(parent in PARENTS,"Original outer parent only")
 if parent==PARENTS[0]:
  saved=importlib.import_module(HERE.replace("/",".")+".saved_outer000_v1")
  accepted=saved.acceptance_values_v1(evidence["saved_observed_acceptance_ref"],read,ref)
  return dict(parent_id=parent,completion_ref=saved.COMPLETION,
   observation_entry_ref=read(saved.COMPLETION)["observation_entry_ref"],saved_observed_acceptance_ref=evidence["saved_observed_acceptance_ref"],
   original_failure_and_monitoring_gap_preserved=True)
 need(evidence["completion_ref"]["path"]=="data/expanded_full_panel_v1/"+parent+"/expanded_parent_completed.json"
  and evidence["driver_source_ref"]==ref(HERE+"/prepare_outer"+parent[-3:]+"_v1.py"),"Exact one remaining-parent generation source/output")
 completed=genuine_stage_v1(evidence,read,ref,parent=parent)
 need(completed["schema"]=="expanded_full_panel_v1_parent_completion"and completed["parent_id"]==parent
  and completed["split"]=="outer"and completed["generation_completed"]is True and completed["Relate_completed"]is True
  and completed["outer_truth_evaluated"]is False and completed["models_prepared"]is False,"Complete original remaining-parent generation, no model or truth shortcut")
 return dict(parent_id=parent,completion_ref=evidence["completion_ref"],observation_entry_ref=completed["observation_entry_ref"])
def all_data_ready_v1(evidence_by_parent,read,ref):
 need(set(evidence_by_parent)==set(PARENTS),"All eight original biological parents, no replacement or successful-subset intersection")
 return {p:data_ready_v1(p,evidence_by_parent[p],read,ref)for p in PARENTS}
def build_stage_v1(intent_ref,parent,method,stage,read,ref):
 intent=read(intent_ref)
 need(intent["schema"]=="H059_later_prediction_ROOT_intent_v1"and intent["root_authorizer"]=="root_agent"
  and intent["active"]is True and intent["outer_truth_access_authorized"]is False
  and intent["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")and intent["method_freeze_ref"]==FREEZE_REF,"Separate exact later ROOT intent")
 need(datetime.fromisoformat(intent["expires_at_utc"])>datetime.now(timezone.utc),"Finite current prediction intent")
 review=read(intent["source_review_ref"])
 need(review["independent"]is True and review["verdict"]in("SOURCE_PASS","PASS")and review["blockers"]==[]
  and review["prediction_stage_source_ref"]==ref(HERE+"/prediction_stage_v1.py")
  and review["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json"),"Whole downstream builder/dispatch integration source review")
 selected=importlib.import_module(HERE.replace("/",".")+".prepare_outer001_v1")
 selected.selection(intent["policy_selector_ref"])
 rows=[x for x in stage_plan_v1(intent["run_id"])if(x["parent_id"],x["method_id"],x["stage"])==(parent,method,stage)]
 need(len(rows)==1,"One exact original method/parent/stage, no extra fit or H058")
 data=all_data_ready_v1(intent["data_evidence_by_parent"],read,ref)[parent]
 inventory=read(ref(SCIENCE+"/frozen_method_ids.v1.json"));fixed=inventory["methods"][method]
 freeze=read(intent["method_freeze_ref"])
 need(freeze["active"]is True and freeze["root_authorizer"]=="root_agent"and freeze["all_methods_fixed_before_first_outer_generation"]is True
  and freeze["outer_data_or_truth_used_for_selection"]is False and freeze["method_ids"]==list(METHODS)and freeze["H058_automatically_included"]is False
  and freeze["identity_inventory_ref"]==ref(SCIENCE+"/frozen_method_ids.v1.json"),"Same five methods fixed before first outer generation")
 qualified_ref=intent["actual_stage_qualification_refs"][method][stage];qualified=read(qualified_ref)
 need(qualified["root_authorizer"]=="root_agent"and qualified["status"]=="qualified"and qualified["unresolved_items"]==[]
  and qualified["method_id"]==method and qualified["stage"]==stage
  and qualified["source_manifest_ref"]==ref(SCIENCE+"/source_manifest.v1.json")
  and qualified["operational_integration_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")
  and qualified["exact_scientific_adapter_ref"]==ref(SCIENCE+"/outer_adapters_v2.py")
  and qualified["full952_observed_samples"]==952 and qualified["query_haplotypes"]==686
  and qualified["private_truth_used"]is False,"Genuine full952 runtime/driver/recipe qualification remains mandatory")
 driver_review=read(qualified["actual_driver_review_ref"])
 need(driver_review["independent"]is True and driver_review["verdict"]in("SOURCE_PASS","PASS")and driver_review["blockers"]==[]
  and driver_review["source_ref"]==qualified["actual_driver_source_ref"]
  and ref(qualified["actual_driver_source_ref"]["path"])==qualified["actual_driver_source_ref"],"Actual later resource driver independently reviewed, not invented by this builder")
 if method in METHODS[:2]:
  need(qualified["full952_observed_H050_bank_and_cut_recipes_checked"]is True,"Complete observed H050 bank/episode/cut recipes")
 if method!=METHODS[-1]:
  need(qualified["fixed_checkpoint_ref"]==fixed["checkpoint_ref"],"Exact preselected frozen transfer checkpoint, no outer retraining/reselection")
 previous=intent.get("stage_evidence_by_parent",{}).get(parent,{}).get(method,{})
 if method==METHODS[-1]:
  need(qualified["full952_native_tokenizer_and_transfer_qualified"]is True
   and qualified["MLM_epochs"]==5 and qualified["MLM_logical_batch"]==512 and qualified["MLM_seed"]==17
   and qualified["MLM_precision"]=="amp_bf16"and qualified["contrastive_precision"]=="fp32"
   and qualified["contrastive_epochs"]==1 and qualified["contrastive_logical_batch"]==4096 and qualified["contrastive_seed"]==36
   and qualified["reduced_batch_or_gradient_cache"]is False,"Original same-ARG full952 five-epochMLM/one-epochcontrastive science")
  if stage in("prepare","fit","predict"):genuine_stage_v1(previous["observe"],read,ref,parent=parent,method=method,stage="observe")
  if stage in("fit","predict"):
   prepared=genuine_stage_v1(previous["prepare"],read,ref,parent=parent,method=method,stage="prepare")
   need(prepared["input_samples"]==952 and prepared["query_haplotypes"]==686 and prepared["reference_haplotypes"]==266
    and prepared["parent_id"]==parent and prepared["true_query_ancestry_read"]is False
    and type(prepared["MLM_optimizer_updates"])is int and prepared["MLM_optimizer_updates"]>0
    and type(prepared["contrastive_optimizer_updates"])is int and prepared["contrastive_optimizer_updates"]>0,"Genuine complete same-parent native corpus before fit")
  if stage=="predict":
   fit=genuine_stage_v1(previous["fit"],read,ref,parent=parent,method=method,stage="fit")
   need(fit["fresh_same_ARG_fit"]is True and fit["old_checkpoint_reused"]is False and fit["query_truth_used"]is False,"Genuine new same-parent native fit")
   for phase,epochs,batch,seed,key in [("MLM",5,512,17,"MLM_optimizer_updates"),("contrastive",1,4096,36,"contrastive_optimizer_updates")]:
    f=fit[phase];need(f["epochs"]==epochs and f["logical_batch"]==batch and f["seed"]==seed and f["precision"]==("amp_bf16"if phase=="MLM"else"fp32")
     and type(f["optimizer_updates"])is int and f["optimizer_updates"]==prepared[key]
     and f["same_ARG_only"]is True and f["query_truth_used"]is False and f["source_batch_reduced"]is False
     and f["gradient_cache_used"]is False,"Complete native original budget and counters, no checkpoint eligibility by flags")
    need(parent in Path(f["checkpoint_ref"]["path"]).parts,"Distinct same-parent native checkpoint namespace")
 q=dict(schema="European_outer_native_adapter_ROOT_request_v2",root_authorizer="root_agent",status="active",
  ROOT_intent_ref=intent_ref,run_id=intent["run_id"],parent_id=parent,method_id=method,stage=stage,output_directory=rows[0]["output_directory"],
  adapter_ref=ref(SCIENCE+"/outer_adapters_v2.py"),source_manifest_ref=ref(SCIENCE+"/source_manifest.v1.json"),
  source_review_ref=intent["scientific_adapter_review_ref"],scientific_execution_authorized=True,
  outer_truth_access_authorized=False,expires_at_utc=intent["expires_at_utc"],
  actual_runtime_and_driver_qualification_ref=qualified_ref,operational_integration_manifest_ref=ref(HERE+"/source_manifest.v1.json"),
  operational_source_review_ref=intent["source_review_ref"],current_storage_policy_selector_ref=intent["policy_selector_ref"],
  observed_data_evidence=data,method_freeze_ref=intent["method_freeze_ref"],frozen_method_identity=copy.deepcopy(fixed),
  separately_admitted_driver_source_ref=qualified["actual_driver_source_ref"],no_resource_admission_by_builder=True)
 return q
def dispatch_v1(request_ref,ctx,*,guard,resource_caps,gpu_authorizer=None):
 """Called only inside actual qualified separately admitted driver; uses exact existing v3 native APIs."""
 need(callable(guard),"Actual guarded admitted caller");guard()
 api=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_adapters_v2")
 reader=api.ObservedReaderV2(guard);q=reader.decode_json(request_ref)
 method,stage=q["method_id"],q["stage"];api._authority(request_ref,method,q["parent_id"],guard)
 need(q["operational_integration_manifest_ref"]==api.ref(HERE+"/source_manifest.v1.json")
  and q["outer_truth_access_authorized"]is False,"Current operational integration/no truth")
 need(q==build_stage_v1(q["ROOT_intent_ref"],q["parent_id"],method,stage,reader.decode_json,api.ref),"Exact rebuilt source-bound later ROOT request before scientific calls")
 if method in METHODS[:2]:return api.predict_h050_outer_v2(request_ref,ctx,guard=guard)
 if method==METHODS[2]:return api.predict_ap_outer_v2(request_ref,q["observed_data_evidence"]["completion_ref"],guard=guard)
 if method==METHODS[3]:return api.predict_argmix_outer_v2(request_ref,q["observed_data_evidence"]["observation_entry_ref"],ctx["config"],ctx["modules"],guard=guard)
 need(method==METHODS[-1],"Only frozen five method calls")
 return api.run_argformer_outer_stage_v2(request_ref,stage,ctx["args"],guard=guard,resource_caps=resource_caps,gpu_authorizer=gpu_authorizer)
def build_matrix_bundle_v1(intent_ref,cells,read,ref):
 intent=read(intent_ref)
 need(intent["schema"]=="H059_later_prediction_ROOT_intent_v1"and intent["active"]is True
  and intent["root_authorizer"]=="root_agent"and intent["outer_truth_access_authorized"]is False
  and intent["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")and intent["method_freeze_ref"]==FREEZE_REF,"Separate complete-matrix ROOT intent")
 all_data_ready_v1(intent["data_evidence_by_parent"],read,ref)
 need(len(cells)==40 and {(x["method_id"],x["parent_id"])for x in cells}=={(m,p)for m in METHODS for p in PARENTS},
  "All40 unique frozen method/parent cells, never success-only subset")
 need(all(x["run_id"]==intent["run_id"]for x in cells)
  and all("top5_predictions_ref"in x for x in cells if x["method_id"]==METHODS[-1]),"Eight independent raw-top5 products accompany forty primaries")
 return dict(schema="European_outer8_native_prediction_bundle_v2",run_id=intent["run_id"],
  source_manifest_ref=ref(SCIENCE+"/source_manifest.v1.json"),method_freeze_ref=intent["method_freeze_ref"],
  method_ids=list(METHODS),parent_ids=list(PARENTS),cells=copy.deepcopy(cells),
  scorer_source_ref=ref(SCIENCE+"/score_outer_v2.py"),scorer_review_ref=intent["actual_scorer_review_ref"],
  validator_review_ref=intent["actual_matrix_validator_review_ref"],outer_truth_access_authorized=False)
def full_prediction_barrier_v1(matrix_request_ref,*,guard):
 """Exact original forty primary + eight raw-top5 validator; no target resolver exists."""
 need(callable(guard),"Actual separately admitted observed-only matrix guard")
 validator=importlib.import_module("research.reviews.European_outer8_preparation_source_20261016_v3.outer_prediction_validator_v2")
 return validator.validate_matrix(matrix_request_ref,validator.ObservedReaderV2(guard))
