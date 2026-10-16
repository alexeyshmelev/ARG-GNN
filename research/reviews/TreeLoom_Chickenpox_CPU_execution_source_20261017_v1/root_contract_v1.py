"""Inactive CPU full-budget metadata contract. ROOT owns all real operations."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL=HERE="research/reviews/TreeLoom_Chickenpox_CPU_execution_source_20261017_v1"
SOURCE=HERE+"/root_contract_v1.py";DRIVER=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
PHASE="train_development";PARENT="Chickenpox_current_TreeLoom_core"
SEEDS=(101,202,303);ARMS=("full","no_memory")
STATUSES={PHASE:"completed_TreeLoom_Chickenpox_CPU_TRAIN_selection"}
DIRECTIVE="ROOT authorizes one fresh H061 TRAIN/VALID CPU run, six fits/4800 updates, TEST and scoring deferred to separate ROOT release after six selections, first scheduled step timed inline"
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
 v=read(record)
 require(v.get("active")is True and v.get("root_authorizer")=="root_agent"and v.get("selector_source_ref")==SELECTOR
  and v.get("source_ref")==GUARD_REF and v.get("policy_ref")==POLICY_REF
  and v.get("accounting_mode")=="native7_H050_stopped_retention_v1"
  and v.get("scheduling_ceiling_bytes")==980000000000 and v.get("human_limit_bytes")==1000000000000
  and v.get("warning_threshold_bytes")==950000000000
  and v.get("no_fallback_discovery_upgrade_or_in_run_swap")is True and v.get("future_ROOT_adapters_only")is True,"Exact explicitly active reviewed child; no fallback")
 return v
def bind_selector(record):
 global SELECTOR_REF,_CHOSEN
 v=selected(record);require(SELECTOR_REF is None or(SELECTOR_REF==record and _CHOSEN==v),"Immutable lifetime ROOT selector")
 SELECTOR_REF=copy.deepcopy(record);_CHOSEN=copy.deepcopy(v);return v
def guard(consumer):
 require(SELECTOR_REF is not None,"ROOT selector before guard/memo")
 # Exact reviewed cache-before-memo implementation; select only our child.
 _h.selection=selected
 return _h.guard(SELECTOR_REF,consumer)
def validate_config(cfg):
 b=binding();defaults=read(ref(HERE+"/scientific_config.INACTIVE.template.v1.json"))
 require(set(cfg)==set(defaults) and cfg.get("execution_enabled")is True
  and all(cfg[k]==defaults[k]for k in defaults if k!="execution_enabled"),
  "Exact reviewed six fits/4800 updates, no optional legacy substitute or reduced run")

def reviewed(record):
 v=read(record);b=binding()
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS")and v.get("source_ref")==ref(DRIVER)
  and v.get("source_manifest_ref")==ref(MANIFEST)and v.get("source_handoff_ref")==ref(HANDOFF)
  and v.get("runner_source_ref")==b["runner_source_ref"]and v.get("protocol_ref")==b["protocol_ref"]
  and v.get("blockers")==[],"Actual complete independent CPU source review")
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

def validate_request(record):
 q=read(record);current(q,CAPS["wall_seconds"]+600);b=binding();sources();bind_selector(q["storage_policy_selector_ref"])
 require(q.get("schema")=="TreeLoom_Chickenpox_CPU_ROOT_request_v1"and q.get("active")is True
  and q.get("root_authorizer")=="root_agent"and q.get("execution_authorized")is True
  and q.get("phase")==PHASE and q.get("stage")=="train"and q.get("parent_id")==PARENT
  and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("caps")==CAPS
  and q.get("GPU_authorized")is False and q.get("physical_gpu")is None and q.get("CUDA_authorized")is False
  and q.get("TRAIN_and_VAL_selection_only")is True
  and q.get("TEST_access_authorized")is False and q.get("scoring_authorized")is False
  and q.get("separate_ROOT_TEST_release_required")is True
  and q.get("legacy_GRU_authorized")is False and q.get("automatic_retry_resume_or_fallback")is False
  and q.get("standalone_token_update_qualification")is False,"Exact fixed CPU-only H061 TRAIN/VALID selection request")
 preparation=local(q["preparation_directory"]);packet=local(q["packet_directory"]);output=local(q["output_directory"])
 require(preparation==local(record["path"]).parent and preparation.is_dir()
  and q["preparation_directory"].startswith("research/reviews/TreeLoom_Chickenpox_CPU_ROOT_preparation_")
  and not any(preparation==p or preparation in p.parents or p in preparation.parents for p in(packet,output)),
  "Separate existing immutable ROOT preparation; no future charged scope created early")
 require(q["packet_directory"].startswith("research/reviews/TreeLoom_Chickenpox_CPU_ROOT_actual_")
  and q["output_directory"].startswith("experiments/results/TreeLoom_Chickenpox_CPU_")
  and q.get("cpu_ids")==[223],"Fresh exclusive output/packet and exact CPU223")
 reviewed(q["source_review_ref"]);scientific_review(q["scientific_review_ref"]);registration(q);context(q["data_context_ref"]);validate_config(q["scientific_config"])
 runtime=read(q["runtime_ref"]);qualified=read(q["runtime_qualification_ref"]);directive=read(q["root_directive_ref"])
 require(runtime.get("device")=="cpu"and runtime.get("dtype")=="float32"and runtime.get("CPU_threads")==1
  and runtime.get("cpu_ids")==[223]and runtime.get("prefix")==str(ROOT/".runtime/env")
  and runtime.get("package_report_ref")==BASE_RUNTIME and runtime.get("actual_runtime_qualified")is True,"Exact existing qualified shared CPU runtime/FP32/CPU223")
 require(qualified.get("root_authorizer")=="root_agent"and qualified.get("status")=="qualified"
  and qualified.get("runtime_ref")==q["runtime_ref"]and qualified.get("source_manifest_ref")==ref(MANIFEST)
  and qualified.get("runner_source_ref")==b["runner_source_ref"]and qualified.get("unresolved_items")==[]
  and qualified.get("standalone_token_update_probe")is False,"Actual bounded ROOT runtime/source qualification")
 require(directive.get("root_authorizer")=="root_agent"and directive.get("active")is True
  and directive.get("directive")==DIRECTIVE
  and all(directive.get(k)==q[k]for k in("preparation_directory","source_manifest_ref","source_review_ref","scientific_review_ref",
   "registration_ref","registration_decisions_ref","data_context_ref","runtime_ref",
   "runtime_qualification_ref","scientific_config","caps","cpu_ids","created_at_utc","expires_at_utc"))
  and directive.get("TRAIN_and_VAL_selection_only")is True
  and directive.get("TEST_access_authorized")is False and directive.get("scoring_authorized")is False
  and directive.get("separate_ROOT_TEST_release_required")is True
  and directive.get("first_scheduled_update_timed_inside_full_budget")is True,
  "Separate exact ROOT directive, full representative work and inline first-step timing")
 return q,CAPS,runtime

def direct_roles(request_ref,q,cfgref,closure):
 # Immutable JSON H061 registration and decision snapshot replace stale Markdown ledger proof.
 b=binding()
 return [ref(MANIFEST),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(HANDOFF),b["runner_source_ref"],b["protocol_ref"],
  q["source_review_ref"],q["scientific_review_ref"],q["runtime_ref"],q["runtime_qualification_ref"],cfgref,q["root_directive_ref"],
  request_ref,q["registration_ref"],q["registration_decisions_ref"],q["data_context_ref"],closure,SELECTOR,q["storage_policy_selector_ref"],NAMESPACE,PIDFD,OWNED,SUPERVISOR]
def make_config(q,caps):
 return dict(schema="TreeLoom_Chickenpox_CPU_operational_config_v1",scientific_config=q["scientific_config"],caps=caps,
  short_tmp_directory=q["packet_directory"]+"/worker_aux/tmp",data_context_ref=q["data_context_ref"])

def input_spec(record):
 global _ACTIVE
 s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);current(s,300);cfg=read(s["refs"]["config"])
 require(record["path"]==q["packet_directory"]+"/input.json"and s["schema"]=="TreeLoom_Chickenpox_CPU_admitted_input_v1"
  and s.get("active")is True and s.get("root_authorizer")=="root_agent"and s.get("execution_authorized")is True
  and s.get("GPU_authorized")is False and s.get("physical_gpu")is None
  and s["root_driver_ref"]==ref(DRIVER)and s["main_source_ref"]==ref(ENTRY)and s["parent_id"]==PARENT and s["phase"]==PHASE
  and s["output_directory"]==q["output_directory"]and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
  and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"]and s["runtime_ref"]==q["runtime_ref"]
  and s["storage"]==read(s["refs"]["storage"])and cfg==make_config(q,caps)and s["source_sha256"]==sources()["source_sha256"],"Exact admitted input/config/runtime/source")
 closure=read(s["transitive_source_closure_ref"])
 require(closure["source_sha256"]==s["source_sha256"]and closure["root_stage_request_ref"]==s["root_stage_request_ref"],"Exact actual source/input closure")
 storage=s["storage"]
 require(storage["reserved_new_bytes"]==CAPS["reservation_bytes"]and storage["reservation"]["reserved_bytes"]==CAPS["reservation_bytes"]
  and storage["output_directory"]==q["output_directory"]and storage["completed_overlay_policy_ref"]==POLICY_REF
  and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["job_id"]==s["job_id"]
  and storage["reservation"]["output"]==str(local(q["output_directory"]))
  and storage["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))],"Whole32MiB exact output+packet selected-policy charge")
 required=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"])
 require(len(storage["reservation"]["consumer_bindings"])<=64 and all(r in storage["reservation"]["consumer_bindings"]for r in required),"Every required role direct under64")
 _ACTIVE=s;return s,q,cfg
def verify_actual_roles(s,q,g):
 impl=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=False)
 complete=g.consumer_bindings_v1(list(state[3].values()),direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"]))
 require(all(r in s["storage"]["reservation"]["consumer_bindings"]for r in complete),"Every returned/current guard remains directly bound")
 return complete
def phase_paths(phase):
 require(phase==PHASE and _ACTIVE is not None,"Actual admitted matched stage")
 return local(_ACTIVE["worker_auxiliary_scope"]).parent,_ACTIVE["output_directory"]
