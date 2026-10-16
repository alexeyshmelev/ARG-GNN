"""Inactive CPU full-budget metadata contract. ROOT owns all real operations."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL=HERE="research/reviews/H061_postselection_execution_source_20261017_v1"
SOURCE=HERE+"/root_contract_v1.py";DRIVER=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
PHASE="postselection";PARENT="Chickenpox_current_TreeLoom_core"
SEEDS=(101,202,303);ARMS=("full","no_memory")
STATUSES={PHASE:"completed_H061_postselection_CPU_stage"}
HELPER=dict(path="research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py",sha256="5bc07b0021b1b35b4ddee9ae6dce8680b61773413593ab829869125a7b7a3c3d")
assert hashlib.sha256((ROOT/HELPER["path"]).read_bytes()).hexdigest()==HELPER["sha256"]
_sp=importlib.util.spec_from_file_location("_TreeLoom_CPU_source_metadata_helpers",ROOT/HELPER["path"])
_h=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(_h)
require=_h.need;local=_h.local;ref=_h.ref;raw=_h.raw;read=_h.read;load=_h.load;put=_h.put;current=_h.current
DEPS=json.loads((ROOT/HERE/"dependencies.v1.json").read_text())
for _name in ("PIDFD","OWNED","SUPERVISOR","SELECTOR","GUARD_REF","POLICY_REF","NAMESPACE","V4","BASE_RUNTIME"):globals()[_name]=DEPS[_name]
CAPS=json.loads((ROOT/HERE/"caps.inactive.v1.json").read_text())
SELECTOR_REF=None;_CHOSEN=None;_ACTIVE=None
def binding():return read(ref(HERE+"/scientific_binding.v1.json"))

def sources():
 m=read(ref(MANIFEST))
 for path,sha in m["source_sha256"].items():raw(dict(path=path,sha256=sha))
 return m

def selected(record):
 # Read only the explicit ROOT record. No guard is imported/configured here.
 profiles=read(ref(HERE+"/storage_profiles.v1.json"));v=read(record);profile=profiles.get(v.get("accounting_mode"))
 require(profile is not None and v.get("schema")=="explicit_ROOT_storage_policy_selector_v1"
  and v.get("active")is True and v.get("root_authorizer")=="root_agent"
  and all(v.get(k)==profile[k]for k in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref",
   "measurement_source_ref","repair_authority_ref"))
  and v.get("scheduling_ceiling_bytes")==980000000000 and v.get("human_limit_bytes")==1000000000000
  and v.get("warning_threshold_bytes")==950000000000
  and v.get("no_fallback_discovery_upgrade_or_in_run_swap")is True
  and v.get("future_ROOT_adapters_only")is True
  and v.get("existing_adapters_runs_and_storage_records_unchanged")is True,
  "Exactly one of two source-pinned ROOT selector interfaces; no discovery/fallback")
 if profile.get("selected_record_ref")is not None:
  require(record==profile["selected_record_ref"]and v["policy_ref"]==profile["policy_ref"],"Exact known native7 active selector/policy")
 else:
  require(v.get("one_new_direct_guard_selector_capsule")is True
   and v.get("journal")==".runtime/storage/Relate_failed64GiB_retention_v1",
   "Only reviewed Relate64 single-capsule successor")
  policy=read(v["policy_ref"]);activation=read(policy["root_activation_ref"])
  require(policy.get("active")is True and policy.get("root_authorizer")=="root_agent"
   and all(policy.get(k)==v[k]for k in("source_ref","source_manifest_ref","source_review_ref","measurement_source_ref"))
   and policy.get("selector_source_ref")==profile["selector_source_ref"]
   and policy.get("scheduling_ceiling_bytes")==980000000000 and policy.get("human_limit_bytes")==1000000000000
   and policy.get("warning_threshold_bytes")==950000000000
   and activation.get("active")is True and activation.get("root_authorizer")=="root_agent"
   and activation.get("source_ref")==profile["source_ref"]and activation.get("source_manifest_ref")==profile["source_manifest_ref"],
   "Actual separately ROOT-activated exact successor, never inactive/planning authority")
 for key in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref","measurement_source_ref","repair_authority_ref"):
  raw(v[key])
 return v
def bind_selector(record):
 global SELECTOR_REF,_CHOSEN,SELECTOR,GUARD_REF,POLICY_REF
 v=selected(record)
 require(SELECTOR_REF is None or(SELECTOR_REF==record and _CHOSEN==v),"Immutable lifetime selector/source/policy; never swap")
 SELECTOR_REF=copy.deepcopy(record);_CHOSEN=copy.deepcopy(v)
 SELECTOR=copy.deepcopy(v["selector_source_ref"]);GUARD_REF=copy.deepcopy(v["source_ref"]);POLICY_REF=copy.deepcopy(v["policy_ref"])
 return v
def guard(consumer):
 require(SELECTOR_REF is not None and _CHOSEN is not None,"Explicit ROOT selection before guard/memo")
 require(SELECTOR==_CHOSEN["selector_source_ref"]and GUARD_REF==_CHOSEN["source_ref"]
  and POLICY_REF==_CHOSEN["policy_ref"],"Same lifetime selected source/policy identity")
 # Only the explicit module choice changes. Original cache-before-memo ordering,
 # returned singleton, consumer memo and every direct guard role remain unchanged.
 _h.selection=selected;_h.SELECTOR=copy.deepcopy(SELECTOR)
 return _h.guard(SELECTOR_REF,consumer)


def scientific_review(record):
 b=binding();require(record==b["scientific_review_ref"],"Exact independent current-core scientific review")
 v=read(record);raw(b["scientific_review_manifest_ref"])
 require(v.get("status")=="SOURCE_PASS"and v.get("source_only")is True
  and v.get("blocking_source_findings")==[] and v.get("reviewer")=="treeloom_transfer_independent_source_review"
  and v.get("completion_ref")==b["completion_ref"]and v.get("generic_core_ref")==b["generic_core_ref"]
  and v.get("source_manifest_ref")==b["source_manifest_ref"]
  and v.get("source_handoff_ref")==b["source_handoff_ref"],"Exact reviewed current-core sources and declared limits")

def context(record):
 b=binding();require(record==b["data_source_manifest_ref"],"Only existing pinned Chickenpox source metadata")
 v=read(record)
 require(v.get("archive_sha256")==b["archive_ref"]["sha256"]and v.get("archive_bytes")==14765
  and v.get("observations")==522 and v.get("counties")==b["counties"]
  and v.get("license")=="CC BY 4.0","Exact existing source/license/counties/rows, no download or fabricated data")
 return v

def registration(q):
 b=binding()
 require(q["registration_ref"]==b["registration_ref"]and q["registration_decisions_ref"]==b["registration_decisions_ref"],
  "Exact immutable prospective H061 registration and decision snapshot; no stale Markdown ledger proof")
 r=read(q["registration_ref"]);decisions=read(q["registration_decisions_ref"])
 require(r.get("schema")=="H061_Chickenpox_current_core_scientific_registration_v1"
  and r.get("root_authorizer")=="root_agent"and r.get("hypothesis_id")=="H061"
  and r.get("status")=="registered_before_current_core_fit"
  and r.get("source_protocol_ref")==b["protocol_ref"]
  and r.get("independent_source_review_ref")==b["scientific_review_ref"]
  and r.get("source_manifest_ref")==b["source_manifest_ref"]and r.get("source_handoff_ref")==b["source_handoff_ref"]
  and r.get("seeds")==list(SEEDS)and r.get("arms")==list(ARMS)
  and r.get("epochs_each")==40 and r.get("updates_each")==800
  and r.get("new_fits")==6 and r.get("new_optimizer_updates")==4800
  and r.get("history_weeks")==52 and r.get("county_count")==20
  and r.get("TRAIN_target_range")==[52,365]and r.get("VAL_target_range")==[365,443]
  and r.get("TRAIN_only_scaler_range")==[0,365]
  and r.get("TEST_access_authorized")is False and r.get("scoring_authorized")is False
  and r.get("scientific_recipe_unchanged")is True
  and r.get("resource_admission_or_execution_granted")is False,
  "Genuine prospective H061 science only, never TEST or resource execution authority")
 rows=[d for d in decisions["decisions"]if d.get("id")=="H061"]
 require(len(rows)==1 and rows[0].get("protocol_ref")==q["registration_ref"]
  and rows[0].get("registered_utc")==r["registered_utc"],"Immutable H061 decision names exact frozen protocol")
 stamp=datetime.fromisoformat(r["registered_utc"])
 require(stamp.tzinfo and stamp<=datetime.fromisoformat(q["created_at_utc"]),"Registration precedes request")


# Postselection metadata gates. This module never imports numerical libraries.
import math
STAGES=("freeze","predict","score")
COMPARATORS=("last_value","seasonal_lag52","ridge_alpha1000_from_prior_pilot")
DIRECTIVE="ROOT authorizes one separately admitted H061 postselection stage with exact barriers, no core fitting or automatic continuation"
def validate_config(cfg):
 defaults=read(ref(HERE+"/scientific_config.INACTIVE.template.v1.json"))
 require(set(cfg)==set(defaults)and cfg.get("execution_enabled")is True
  and all(cfg[k]==defaults[k]for k in defaults if k!="execution_enabled"),"Exact registered postselection semantics")
def reviewed(record):
 v=read(record);b=binding()
 require(v.get("independent")is True and v.get("verdict")=="SOURCE_PASS"
  and v.get("source_ref")==ref(DRIVER)and v.get("source_manifest_ref")==ref(MANIFEST)
  and v.get("source_handoff_ref")==ref(HANDOFF)and v.get("runner_source_ref")==b["runner_source_ref"]
  and v.get("protocol_ref")==b["protocol_ref"]and v.get("blockers")==[],
  "Actual independent whole postselection source review")
def _in_output(record,directory,name):
 require(record["path"]==directory+"/"+name,"Exact retained flat stage product")
 return read(record)
def train_selections(q,*,authenticate_payloads=False):
 b=binding();a=read(q["train_acceptance_ref"])
 require(a.get("schema")=="H061_TRAIN_outer_acceptance_v1"and a.get("root_authorizer")=="root_agent"
  and a.get("accepted")is True and a.get("outer_terminal_consumed")is True
  and a.get("outer_terminal_exit_code")==0 and a.get("source_manifest_ref")==b["TRAIN_source_manifest_ref"]
  and a.get("source_review_ref")==b["TRAIN_wrapper_review_ref"],"ROOT accepted actual complete TRAIN outer0, never launch/admission evidence")
 w=read(a["actual_wait_ref"]);directory=a["output_directory"]
 require(w.get("returncode")==0 and w.get("actual_direct_worker_wait")is True
  and w.get("all_owned_processes_and_groups_drained")is True and w.get("operational_failure")is None
  and w.get("completed_ref")==a["completed_ref"],"Genuine successful TRAIN OS wait and complete owned drain")
 envelope=_in_output(a["completed_ref"],directory,"completed.json")
 require(envelope.get("status")=="completed_TreeLoom_Chickenpox_CPU_TRAIN_selection"
  and envelope.get("root_driver_ref")==b["lifecycle_driver_ref"]
  and envelope.get("input_ref")==w["input_ref"]and envelope.get("TEST_deferred")is True
  and envelope.get("scoring_authorized")is False,"Actual reviewed TRAIN completion envelope")
 result=_in_output(envelope["result_ref"],directory,"completed_TRAIN.json")
 require(result.get("status")=="completed_TreeLoom_Chickenpox_TRAIN_six_selections"
  and result.get("seeds")==list(SEEDS)and result.get("arms")==list(ARMS)
  and result.get("total_optimizer_updates")==4800 and result.get("selected_count")==6
  and result.get("validation_prediction_batches")==1200 and result.get("county_count")==20
  and result.get("history_weeks")==52 and result.get("separate_ROOT_TEST_release_required")is True
  and result.get("TEST_accessed")is False
  and result.get("TEST_scored")is False and result.get("test_predictions_sealed")==0
  and result.get("first_scheduled_update_timed_inline")is True and result.get("standalone_numerical_probe")is False,
  "All six complete actual fits, original 4800 TRAIN updates and1200 validation batches")
 barrier=_in_output(result["selection_barrier_ref"],directory,"six_selection_barrier.json")
 require(a["selection_barrier_ref"]==result["selection_barrier_ref"]
  and barrier.get("status")=="six_TRAIN_selected_states_and_configurations_frozen"
  and barrier.get("source_protocol_ref")==b["protocol_ref"]and barrier.get("registration_ref")==b["registration_ref"]
  and barrier.get("completion_source_ref")==b["completion_ref"]and barrier.get("generic_core_ref")==b["generic_core_ref"]
  and barrier.get("archive_ref")==b["archive_ref"]and barrier.get("data_source_manifest_ref")==b["data_source_manifest_ref"]
  and barrier.get("county_order")==b["counties"]
  and barrier.get("TEST_accessed")is False and barrier.get("TEST_release_granted")is False
  and barrier.get("scoring_authorized")is False,"Actual six-selection source/config/scaler barrier")
 cfg=barrier["scientific_config"];defaults=read(b["TRAIN_scientific_config_ref"])
 require(set(cfg)==set(defaults)and cfg["execution_enabled"]is True
  and all(cfg[k]==defaults[k]for k in defaults if k!="execution_enabled"),"Original unmodified scientific TRAIN configuration")
 fits=result["fits"];required={(s,a)for s in SEEDS for a in ARMS}
 require(len(fits)==6 and {(x["seed"],x["arm"])for x in fits}==required,"All six unique actual selected identities")
 products=[]
 for row in fits:
  seed,arm=row["seed"],row["arm"];checkpoint=row["selected_checkpoint_ref"]
  require(checkpoint["path"]==directory+"/selected_seed%d_%s.pt"%(seed,arm),"Actual retained selected state path")
  meta=_in_output(row["selected_metadata_ref"],directory,"selection_seed%d_%s.json"%(seed,arm))
  history=meta["history"];scores=[x["mae_cases"]for x in history]
  require(meta["seed"]==seed and meta["arm"]==arm and meta["selected_checkpoint_ref"]==checkpoint
   and meta["scientific_config"]==cfg and meta["source_protocol_ref"]==b["protocol_ref"]
   and meta["registration_ref"]==b["registration_ref"]and meta["state_kind"]=="selected_cpu_model_state_only"
   and meta["optimizer_state_saved"]is False and meta["TEST_release_granted"]is False
   and meta["epochs_completed"]==40 and meta["optimizer_updates"]==800 and meta["test_accessed"]is False
   and meta["training_targets"]==list(range(52,365))and meta["validation_targets"]==list(range(365,443))
   and len(history)==40 and [x["epoch"]for x in history]==list(range(1,41))
   and [x["updates"]for x in history]==list(range(20,801,20))
   and all(x["forecast_weeks"]==78 and len(x["per_county_mae"])==20 and len(x["per_week_mae"])==78 for x in history)
   and all(math.isfinite(v)for v in scores)and meta["selected_epoch"]==scores.index(min(scores))+1
   and meta["selected_validation_mae_cases"]==min(scores)
   and meta["parameters"]=={"full":25378,"no_memory":24865}[arm]
   and all(row[k]==meta[k]for k in("seed","arm","selected_epoch","selected_validation_mae_cases","epochs_completed",
    "optimizer_updates","parameters","test_accessed")),"Complete exact original configuration and40epoch history; earliest minimum VAL checkpoint")
  if authenticate_payloads:
   require(len(raw(checkpoint))<=CAPS["checkpoint_serialized_bytes"],"Actual selected state hash/byte bound, post gate only")
  products.append(dict(seed=seed,arm=arm,kind="selected_checkpoint",**checkpoint,selected_metadata_ref=row["selected_metadata_ref"]))
 require(barrier["selected_products"]==products,"Exact six actual selected product hashes")
 scaler=barrier["TRAIN_only_scaler"]
 require(scaler["fit_rows"]==[0,365]and len(scaler["mean"])==len(scaler["scale"])==20
  and all(math.isfinite(v)for v in scaler["mean"]+scaler["scale"])and all(v>0 for v in scaler["scale"]),
  "Complete TRAIN-only scaler frozen with selections")
 return dict(acceptance_ref=q["train_acceptance_ref"],barrier_ref=result["selection_barrier_ref"],
  products=products,scientific_config=cfg,scaler=scaler)
def upstream(q):
 return {k:q[k]for k in("train_acceptance_ref","freeze_acceptance_ref","prediction_acceptance_ref")}
def accepted_stage(q,stage):
 key={"freeze":"freeze_acceptance_ref","predict":"prediction_acceptance_ref"}[stage]
 a=read(q[key])
 require(a.get("schema")=="H061_postselection_outer_acceptance_v1"and a.get("root_authorizer")=="root_agent"
  and a.get("accepted")is True and a.get("stage")==stage and a.get("source_manifest_ref")==ref(MANIFEST)
  and a.get("outer_terminal_consumed")is True and a.get("outer_terminal_exit_code")==0,
  "Separate ROOT acceptance of genuine prior-stage outer0")
 w=read(a["actual_wait_ref"])
 require(w.get("returncode")==0 and w.get("actual_direct_worker_wait")is True
  and w.get("all_owned_processes_and_groups_drained")is True and w.get("operational_failure")is None
  and w.get("completed_ref")==a["completed_ref"],"Actual prior-stage successful wait/drain")
 envelope=_in_output(a["completed_ref"],a["output_directory"],"completed.json")
 require(envelope.get("status")==STATUSES[PHASE]and envelope.get("stage")==stage
  and envelope.get("root_driver_ref")==ref(DRIVER)and envelope.get("input_ref")==w["input_ref"]
  and envelope.get("upstream",{}).get("train_acceptance_ref")==q["train_acceptance_ref"]
  and envelope.get("scientific_config")==q["scientific_config"],"Exact accepted stage/science/upstream")
 if stage=="predict":
  require(envelope["upstream"]["freeze_acceptance_ref"]==q["freeze_acceptance_ref"],"Prediction belongs to same comparator freeze")
 result=_in_output(envelope["result_ref"],a["output_directory"],"completed_"+stage+".json")
 require(result.get("stage")==stage and result.get("status")=="completed_H061_"+stage
  and result.get("new_core_fits")==0 and result.get("new_optimizer_updates")==0,"No new fit/resume semantics")
 return result
def frozen_barrier(q,selections):
 result=accepted_stage(q,"freeze");v=read(result["barrier_ref"]);b=binding()
 require(v.get("schema")=="H061_complete_selection_comparator_barrier_v1"
  and v.get("selection_barrier_ref")==selections["barrier_ref"]and v.get("selected_products")==selections["products"]
  and v.get("TRAIN_only_scaler")==selections["scaler"]and v.get("comparators")==list(COMPARATORS)
  and v.get("legacy_GRU_status")=="unavailable_payload_not_accepted"
  and v.get("TEST_accessed")is False and v.get("TEST_scored")is False
  and v.get("source_protocol_ref")==b["protocol_ref"]and v.get("registration_ref")==b["registration_ref"]
  and result.get("TEST_accessed")is False and result.get("TEST_scored")is False,
  "All six actual states/config/histories plus all mandatory comparator states frozen before TEST")
 return result["barrier_ref"],v
def prediction_barrier(q,selections,freeze_ref):
 result=accepted_stage(q,"predict");v=read(result["barrier_ref"])
 required={(s,a)for s in SEEDS for a in ARMS}
 require(v.get("schema")=="H061_complete_prediction_barrier_v1"and v.get("freeze_barrier_ref")==freeze_ref
  and v.get("selection_barrier_ref")==selections["barrier_ref"]and v.get("TEST_scored")is False
  and v.get("target_indices")==list(range(443,522))and v.get("county_order")==binding()["counties"]
  and len(v["predictions"])==6 and {(r["seed"],r["arm"])for r in v["predictions"]}==required
  and len(v["chronology_predictions"])==6 and {(r["seed"],r["arm"])for r in v["chronology_predictions"]}==required
  and len(v["comparator_predictions"])==3 and {r["comparator"]for r in v["comparator_predictions"]}==set(COMPARATORS)
  and all(r["kind"]=="test_prediction"for r in v["predictions"])
  and all(r["kind"]=="chronology_prediction"for r in v["chronology_predictions"])
  and all(r["kind"]=="comparator_prediction"for r in v["comparator_predictions"])
  and result.get("TEST_scored")is False,"All six complete chronological predictions, six diagnostics and three controls sealed before any scoring")
 return result["barrier_ref"],v
def validate_request(record):
 q=read(record);current(q,CAPS["wall_seconds"]+600);b=binding();sources();bind_selector(q["storage_policy_selector_ref"])
 stage=q.get("stage");test=stage in("predict","score");score=stage=="score"
 require(q.get("schema")=="H061_postselection_ROOT_request_v1"and q.get("active")is True
  and q.get("root_authorizer")=="root_agent"and q.get("execution_authorized")is True
  and stage in STAGES and q.get("phase")==PHASE and q.get("parent_id")==PARENT
  and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("caps")==CAPS
  and q.get("TEST_access_authorized")is test and q.get("scoring_authorized")is score
  and q.get("GPU_authorized")is False and q.get("physical_gpu")is None and q.get("CUDA_authorized")is False
  and q.get("legacy_GRU_authorized")is False and q.get("automatic_retry_resume_or_fallback")is False
  and q.get("standalone_token_update_qualification")is False and q.get("new_core_fits")==0
  and q.get("new_optimizer_updates")==0 and q.get("cpu_ids")==[223],"Exact separately ROOT released stage; no fit, probe, optional substitution or automatic continuation")
 preparation=local(q["preparation_directory"]);packet=local(q["packet_directory"]);output=local(q["output_directory"])
 require(preparation==local(record["path"]).parent and preparation.is_dir()
  and q["preparation_directory"].startswith("research/reviews/H061_postselection_ROOT_preparation_")
  and q["packet_directory"].startswith("research/reviews/H061_postselection_ROOT_actual_")
  and q["output_directory"].startswith("experiments/results/H061_postselection_")
  and not any(preparation==p or preparation in p.parents or p in preparation.parents for p in(packet,output)),
  "Separate fresh preparation/charged packet/output")
 reviewed(q["source_review_ref"]);scientific_review(q["scientific_review_ref"]);registration(q);context(q["data_context_ref"]);validate_config(q["scientific_config"])
 selections=train_selections(q)
 if stage=="freeze":
  require(q["freeze_acceptance_ref"]is None and q["prediction_acceptance_ref"]is None,"Freeze has no later-stage input")
 else:
  freeze_ref,_=frozen_barrier(q,selections)
  if stage=="predict":require(q["prediction_acceptance_ref"]is None,"Predict cannot depend on later score")
  else:prediction_barrier(q,selections,freeze_ref)
 runtime=read(q["runtime_ref"]);qualified=read(q["runtime_qualification_ref"]);directive=read(q["root_directive_ref"])
 require(runtime.get("device")=="cpu"and runtime.get("dtype")=="float32"and runtime.get("CPU_threads")==1
  and runtime.get("cpu_ids")==[223]and runtime.get("prefix")==str(ROOT/".runtime/env")
  and runtime.get("package_report_ref")==BASE_RUNTIME and runtime.get("actual_runtime_qualified")is True,"Same qualified shared CPU223 FP32 runtime")
 require(qualified.get("root_authorizer")=="root_agent"and qualified.get("status")=="qualified"
  and qualified.get("runtime_ref")==q["runtime_ref"]and qualified.get("source_manifest_ref")==ref(MANIFEST)
  and qualified.get("runner_source_ref")==b["runner_source_ref"]and qualified.get("unresolved_items")==[]
  and qualified.get("standalone_token_update_probe")is False,"Actual ROOT metadata/source qualification")
 require(directive.get("root_authorizer")=="root_agent"and directive.get("active")is True
  and directive.get("directive")==DIRECTIVE
  and all(directive.get(k)==q[k]for k in("stage","phase","preparation_directory","source_manifest_ref","source_review_ref",
   "scientific_review_ref","registration_ref","registration_decisions_ref","data_context_ref","runtime_ref",
   "runtime_qualification_ref","scientific_config","caps","cpu_ids","created_at_utc","expires_at_utc",
   "train_acceptance_ref","freeze_acceptance_ref","prediction_acceptance_ref","TEST_access_authorized","scoring_authorized")),
  "Separate exact ROOT directive for this one stage")
 return q,CAPS,runtime
def direct_roles(request_ref,q,cfgref,closure):
 b=binding()
 roles=[ref(MANIFEST),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(HANDOFF),b["runner_source_ref"],b["protocol_ref"],
  q["source_review_ref"],q["scientific_review_ref"],q["runtime_ref"],q["runtime_qualification_ref"],cfgref,q["root_directive_ref"],
  request_ref,q["registration_ref"],q["registration_decisions_ref"],q["data_context_ref"],closure,DEPS["SELECTOR"],
  q["storage_policy_selector_ref"],NAMESPACE,PIDFD,OWNED,SUPERVISOR,q["train_acceptance_ref"]]
 roles.extend(q[k]for k in("freeze_acceptance_ref","prediction_acceptance_ref")if q[k]is not None)
 return roles
def make_config(q,caps):
 return dict(schema="H061_postselection_operational_config_v1",stage=q["stage"],scientific_config=q["scientific_config"],caps=caps,
  short_tmp_directory=q["packet_directory"]+"/worker_aux/tmp",data_context_ref=q["data_context_ref"],upstream=upstream(q))
def input_spec(record):
 global _ACTIVE
 s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);current(s,300);cfg=read(s["refs"]["config"])
 require(record["path"]==q["packet_directory"]+"/input.json"and s.get("schema")=="H061_postselection_admitted_input_v1"
  and s.get("active")is True and s.get("root_authorizer")=="root_agent"and s.get("execution_authorized")is True
  and s.get("GPU_authorized")is False and s.get("physical_gpu")is None
  and s["root_driver_ref"]==ref(DRIVER)and s["main_source_ref"]==ref(ENTRY)and s["parent_id"]==PARENT
  and s["phase"]==PHASE and s["stage"]==q["stage"]and s["output_directory"]==q["output_directory"]
  and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
  and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"]and s["runtime_ref"]==q["runtime_ref"]
  and s["storage"]==read(s["refs"]["storage"])and cfg==make_config(q,caps)and s["source_sha256"]==sources()["source_sha256"],
  "Exact admitted stage/config/runtime/source, never hand-built worker input")
 closure=read(s["transitive_source_closure_ref"]);storage=s["storage"]
 require(closure["source_sha256"]==s["source_sha256"]and closure["root_stage_request_ref"]==s["root_stage_request_ref"]
  and storage["reserved_new_bytes"]==CAPS["reservation_bytes"]and storage["reservation"]["reserved_bytes"]==CAPS["reservation_bytes"]
  and storage["output_directory"]==q["output_directory"]and storage["completed_overlay_policy_ref"]==POLICY_REF
  and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["job_id"]==s["job_id"]
  and storage["reservation"]["output"]==str(local(q["output_directory"]))
  and storage["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))],"Exact source/input closure and whole32MiB selected-policy charge")
 required=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"])
 require(len(storage["reservation"]["consumer_bindings"])<=64
  and all(r in storage["reservation"]["consumer_bindings"]for r in required),"All direct roles preserved under64; never trim")
 _ACTIVE=s;return s,q,cfg
def verify_actual_roles(s,q,g):
 impl=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=False)
 complete=g.consumer_bindings_v1(list(state[3].values()),direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"]))
 require(all(r in s["storage"]["reservation"]["consumer_bindings"]for r in complete),"Every current/returned guard directly retained")
 return complete
def phase_paths(phase):
 require(phase==PHASE and _ACTIVE is not None,"Actual admitted postselection stage")
 return local(_ACTIVE["worker_auxiliary_scope"]).parent,_ACTIVE["output_directory"]
