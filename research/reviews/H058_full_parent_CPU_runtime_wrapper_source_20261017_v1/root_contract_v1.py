"""Inactive CPU full-budget metadata contract. ROOT owns all real operations."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL=HERE="research/reviews/H058_full_parent_CPU_runtime_wrapper_source_20261017_v1"
SOURCE=HERE+"/root_contract_v1.py";DRIVER=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
PHASE="train";PARENT="H058_European32_full_TRAIN_CPU"
TRAIN=tuple("ap_expanded_train_%03d"%i for i in range(32))
STATUSES={PHASE:"completed_H058_full960_each_arm_CPU_training"}
DIRECTIVE="ROOT authorizes one fresh H058 full original32parent CPU run with960updates EACHarm and1920total, firstscheduled updates checked and timed inline"
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
 b=binding();defaults=b["config_defaults"];op=set(b["operational_ref_keys"])
 require(set(cfg)==set(defaults)and all(cfg[k]==defaults[k]for k in defaults if k not in op),
  "Exact original32TRAIN/30passes/960EACH/1920total; no changed math or support")
 for key in op:
  require(isinstance(cfg[key],dict)and set(cfg[key])=={"path","sha256"},"Actual source-bound ROOT CPU metadata refs")
  require(read(cfg[key]).get("root_authorizer")=="root_agent","ROOT CPU declarations, never GPU proof")
 require(cfg["checkpoint_file_cap_bytes"]==CAPS["checkpoint_serialized_bytes"]and cfg["output_cap_bytes"]==CAPS["output_bytes"]
  and cfg["max_cpu_rss_bytes"]==CAPS["rss_bytes"]and cfg["CPU_prepared_cache_cap_bytes"]==CAPS["max_cached_parent_bytes"],
  "Actual capped CPU and output envelopes")
def reviewed(record):
 v=read(record);b=binding()
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS")and v.get("source_ref")==ref(DRIVER)
  and v.get("source_manifest_ref")==ref(MANIFEST)and v.get("source_handoff_ref")==ref(HANDOFF)
  and v.get("runner_source_ref")==b["runner_source_ref"]and v.get("protocol_ref")==b["protocol_ref"]
  and v.get("blockers")==[],"Actual complete independent CPU source review")
def scientific_review(record):
 b=binding();v=read(record);producer=v.get("producer_review",v)
 require(v.get("independent")is True and producer.get("verdict")in("SOURCE_PASS","PASS")
  and producer.get("source_manifest_ref")==b["source_manifest_ref"]and producer.get("source_handoff_ref")==b["source_handoff_ref"]
  and producer.get("producer_ref")==b["runner_source_ref"]and producer.get("blockers")==[],
  "Actual independent review of sealed CPU producer/cache/context, not wrapper self-review")
def context(record):
 v=read(record);b=binding();original=read(b["original_H050_input_ref"]);targets=read(b["original_TRAIN_bindings_ref"])["parents"]
 require(v.get("schema")=="H058_original32_TRAIN_CPU_context_v1"and v.get("root_authorizer")=="root_agent"
  and v.get("original_H050_input_ref")==b["original_H050_input_ref"]and v.get("original_TRAIN_bindings_ref")==b["original_TRAIN_bindings_ref"]
  and v.get("warmstart_checkpoint_ref")==b["warmstart_checkpoint_ref"]and v.get("warmstart_completion_ref")==b["warmstart_completion_ref"]
  and v.get("TRAIN_parent_ids")==list(TRAIN)and set(targets)==set(TRAIN)
  and set(original["observation_entry_refs"])==set(TRAIN)|{"ap_expanded_val_%03d"%i for i in range(8)},
  "Original32EuropeanTRAIN inputs/target authority and exact accepted H045 initializer")
 return v
def registration(q):
 b=binding();require(q["registration_ref"]==b["registration_ref"],"Preserve exact existing H058registration")
 r=read(q["registration_ref"]);decisions=read(q["registration_decisions_ref"]);ledger=raw(q["registration_ledger_ref"]).decode()
 require(r.get("id")=="H058"and r["protocol"]["passes"]==30and r["protocol"]["aligned_and_null_new_updates_each"]==960
  and r["protocol"]["untouched_outer_or_TEST_access_authorized"]is False,"Registered original H058condition; no new hypothesis")
 require(len([d for d in decisions["decisions"]if d.get("id")=="H058"])==1and "H058"in ledger,"Actual immutable H058decision/ledger snapshots")
 stamp=datetime.fromisoformat(r["registered_utc"])
 require(stamp.tzinfo and stamp<=datetime.fromisoformat(q["created_at_utc"]),"Registration precedes fresh CPU request")
def validate_request(record):
 q=read(record);current(q,CAPS["wall_seconds"]+600);b=binding();sources();bind_selector(q["storage_policy_selector_ref"])
 require(q.get("schema")=="H058_full_parent_CPU_ROOT_request_v1"and q.get("active")is True and q.get("root_authorizer")=="root_agent"
  and q.get("execution_authorized")is True and q.get("phase")==PHASE and q.get("stage")=="train"and q.get("parent_id")==PARENT
  and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("caps")==CAPS and q.get("GPU_authorized")is False
  and q.get("physical_gpu")is None and q.get("CUDA_authorized")is False and q.get("TRAIN_truth_only")is True
  and q.get("VAL_truth_authorized")is False and q.get("TEST_authorized")is False
  and q.get("automatic_retry_resume_or_fallback")is False and q.get("standalone_token_update_qualification")is False,"Exact fresh CPU-only full-budget ROOT request")
 preparation=local(q["preparation_directory"]);packet=local(q["packet_directory"]);output=local(q["output_directory"])
 require(preparation==local(record["path"]).parent and preparation.is_dir()
  and q["preparation_directory"].startswith("research/reviews/H058_CPU_ROOT_preparation_")
  and not any(preparation==p or preparation in p.parents or p in preparation.parents for p in (packet,output)),"Separate existing ROOT request/preparation directory; never a reserved target scope")
 require(q["packet_directory"].startswith("research/reviews/H058_CPU_ROOT_")
  and q["output_directory"].startswith("experiments/results/H058_full_parent_CPU_")
  and type(q["cpu_ids"])is list and len(q["cpu_ids"])==len(set(q["cpu_ids"]))==32
  and all(type(x)is int and x>=0 for x in q["cpu_ids"]),"Fresh canonical scopes and32CPU identities")
 reviewed(q["source_review_ref"]);scientific_review(q["scientific_review_ref"]);registration(q);context(q["training_context_ref"]);validate_config(q["scientific_config"])
 runtime=read(q["runtime_ref"]);qualified=read(q["runtime_qualification_ref"]);directive=read(q["root_directive_ref"])
 require(runtime.get("device")=="cpu"and runtime.get("dtype")=="float32"and runtime.get("CPU_threads")==32
  and runtime.get("cpu_ids")==q["cpu_ids"]and runtime.get("prefix")==str(ROOT/".runtime/env")and runtime.get("package_report_ref")==BASE_RUNTIME,"Actual qualified CPU/FP32/32thread runtime")
 require(qualified.get("root_authorizer")=="root_agent"and qualified.get("status")=="qualified"
  and qualified.get("runtime_ref")==q["runtime_ref"]and qualified.get("source_manifest_ref")==ref(MANIFEST)
  and qualified.get("runner_source_ref")==b["runner_source_ref"]and qualified.get("unresolved_items")==[]
  and qualified.get("standalone_token_update_probe")is False
   and qualified.get("one_hold_summary_ref")==b["one_hold_summary_ref"]
   and qualified.get("full_training_fit_measured")is False
   and qualified.get("scientific_probe_updates")==0,"Existing genuine CPU arithmetic/runtime evidence only; no new probes or measured960fit claim")
 require(directive.get("root_authorizer")=="root_agent"and directive.get("active")is True and directive.get("directive")==DIRECTIVE
  and all(directive.get(k)==q[k]for k in ("preparation_directory","source_manifest_ref","source_review_ref","scientific_review_ref","registration_ref","registration_decisions_ref","registration_ledger_ref","training_context_ref",
   "runtime_ref","runtime_qualification_ref","scientific_config","caps","cpu_ids","created_at_utc","expires_at_utc"))
  and directive.get("first_scheduled_update_timed_inside_full_budget")is True,"Separate real ROOT directive binds complete representative budget")
 return q,CAPS,runtime
def direct_roles(request_ref,q,cfgref,closure):
 # The exact Markdown ledger stays SHA-bound through request/directive and registration(q).
 # V4 direct consumer metadata permits only .py/.sh/.json, so never list that .md directly.
 b=binding()
 return [ref(MANIFEST),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(HANDOFF),b["runner_source_ref"],b["protocol_ref"],
  q["source_review_ref"],q["scientific_review_ref"],q["runtime_ref"],q["runtime_qualification_ref"],cfgref,q["root_directive_ref"],
  make_config(q,CAPS)["scientific_config_ref"],request_ref,q["registration_ref"],q["registration_decisions_ref"],q["training_context_ref"],closure,SELECTOR,q["storage_policy_selector_ref"],NAMESPACE,PIDFD,OWNED,SUPERVISOR]
def make_config(q,caps):
 science=ref(q["preparation_directory"]+"/"+local(q["packet_directory"]).name+".scientific_config.json")
 require(read(science)==q["scientific_config"],"Exact exclusive staged H058scientific config")
 return dict(schema="H058_full_parent_CPU_operational_config_v1",scientific_config_ref=science,caps=caps,
  short_tmp_directory=q["packet_directory"]+"/worker_aux/tmp",training_context_ref=q["training_context_ref"])
def input_spec(record):
 global _ACTIVE
 s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);current(s,300);cfg=read(s["refs"]["config"])
 require(record["path"]==q["packet_directory"]+"/input.json"and s["schema"]=="H058_full_parent_CPU_admitted_input_v1"
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
