"""Inactive integration of exact FLARE stages with existing owned CPU framework."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1"
SOURCE=HERE+"/root_contract_v1.py";DRIVER=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
DEPS=json.loads((ROOT/HERE/"dependencies.v1.json").read_text())
HELPER=DEPS["HELPER"]
assert hashlib.sha256((ROOT/HELPER["path"]).read_bytes()).hexdigest()==HELPER["sha256"]
sp=importlib.util.spec_from_file_location("_FLARE_remaining_metadata",ROOT/HELPER["path"])
_h=importlib.util.module_from_spec(sp);sp.loader.exec_module(_h)
require=_h.need;local=_h.local;ref=_h.ref;raw=_h.raw;read=_h.read;load=_h.load;put=_h.put;current=_h.current
for key in ("SELECTOR","GUARD_REF","POLICY_REF","NAMESPACE","PIDFD","OWNED","BASE_RUNTIME","FRAMEWORK","WAIT_WRITER","COMPLETION","COMPLETION_REVIEW"):
 globals()[key]=DEPS[key]
VAL=tuple("american_c3_val_%03d"%i for i in range(1,5))
STAGES=("convert","native","seal")
MODULES=dict(convert="arggnn.american_c3_flare_convert_v1",native="arggnn.american_c3_flare_run_v2",seal="arggnn.american_c3_flare_evaluate_v1")
PRODUCERS=dict(convert="src/arggnn/american_c3_flare_convert_v1.py",native="src/arggnn/american_c3_flare_run_v2.py",seal="src/arggnn/american_c3_flare_evaluate_v1.py")
STATUS=dict(convert="completed_C3_FLARE_observation_conversion",native="completed_C3_native_FLARE",seal="sealed_C3_native_FLARE_calls_and_raw_posteriors")
CAPS={};PHASE=PARENT=None;SELECTOR_REF=None;_CHOSEN=_ACTIVE=None
def completion():
 raw(COMPLETION)
 import importlib
 return importlib.import_module("research.reviews.FLARE_original_VAL001004_completion_source_20261017_v1.completion_v1")
def sources():
 m=read(ref(MANIFEST))
 for path,sha in m["source_sha256"].items():raw(dict(path=path,sha256=sha))
 for r in m["metadata_refs"]:raw(r)
 return m
def reviewed(record):
 v=read(record)
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS")
  and v.get("source_ref")==ref(DRIVER)and v.get("source_manifest_ref")==ref(MANIFEST)
  and v.get("source_handoff_ref")==ref(HANDOFF)and v.get("blockers")==[],
  "Actual complete independent FLARE runtime integration source review")
def scientific_review():
 v=read(COMPLETION_REVIEW)
 require(v.get("status")=="SOURCE_PASS"and v.get("source_ref")==COMPLETION
  and v.get("source_manifest_ref")==DEPS["COMPLETION_MANIFEST"]and v.get("blocking_findings")==[]
  and v.get("execution_admission_or_truth_authority_granted")is False,
  "Exact independent original completion source pass; no execution/truth authority inferred")
def selected(record):
 v=read(record)
 require(v.get("active")is True and v.get("root_authorizer")=="root_agent"
  and v.get("selector_source_ref")==SELECTOR and v.get("source_ref")==GUARD_REF and v.get("policy_ref")==POLICY_REF
  and v.get("accounting_mode")=="native7_H050_stopped_retention_v1"
  and v.get("scheduling_ceiling_bytes")==980000000000 and v.get("human_limit_bytes")==1000000000000
  and v.get("warning_threshold_bytes")==950000000000
  and v.get("no_fallback_discovery_upgrade_or_in_run_swap")is True and v.get("future_ROOT_adapters_only")is True,
  "Exact active native7/H050 selection; never use old FLARE guard")
 return v
def bind_selector(record):
 global SELECTOR_REF,_CHOSEN
 v=selected(record);require(SELECTOR_REF is None or(SELECTOR_REF==record and _CHOSEN==v),"One immutable lifetime selection")
 SELECTOR_REF=copy.deepcopy(record);_CHOSEN=copy.deepcopy(v)
def guard(consumer):
 require(SELECTOR_REF is not None,"Explicit selected ROOT policy before guard")
 _h.selection=selected
 return _h.guard(SELECTOR_REF,consumer)
def stage_caps(stage,cfg):
 require(stage in STAGES and type(cfg["output_cap_bytes"])is int and cfg["output_cap_bytes"]>0,"Exact original stage output bound")
 seconds=14400 if stage=="native"else 3600
 return dict(reservation_bytes=cfg["output_cap_bytes"]+8*2**20,output_bytes=cfg["output_cap_bytes"],
  aux_bytes=4*2**20,root_packet_bytes=2*2**20,private_work_bytes=2**20,log_bytes=256*2**10,
  wall_seconds=seconds+600,scientific_seconds=seconds,cleanup_seconds=30,cpu_threads=4,
  rss_bytes=(24 if stage=="native"else 4)*2**30)
def stage_evidence(record,*,stage,parent,completion_ref):
 e=read(record)
 require(e.get("schema")=="FLARE_remaining_stage_ROOT_terminal_evidence_v1"and e.get("root_authorizer")=="root_agent"
  and e.get("stage")==stage and e.get("parent_id")==parent and e.get("completion_ref")==completion_ref,
  "Explicit ROOT evidence for this exact completed original cell/stage")
 wait=read(e["actual_wait_ref"]);outer=read(e["outer_terminal_ref"]);inp=read(e["input_ref"])
 require(wait.get("actual_direct_worker_wait")is True and wait.get("returncode")==0
  and wait.get("all_owned_processes_and_groups_drained")is True and wait.get("operational_failure")is None
  and wait.get("input_ref")==e["input_ref"]and wait.get("completed_ref")==completion_ref
  and outer.get("actual_terminal_consumed")is True
  and(outer.get("actual_exit_code")==0 or outer.get("actual_tool_exit_code")==0)
  and outer.get("completed_ref")==completion_ref
  and(outer.get("direct_wait_ref")==e["actual_wait_ref"]or outer.get("actual_direct_wait_ref")==e["actual_wait_ref"]),
  "Genuine direct worker0, complete owned drain, consumed outer0, exact completion")
 require(inp.get("schema")=="FLARE_remaining_CPU_admitted_input_v1"and inp.get("parent_id")==parent
  and inp.get("stage")==stage and inp.get("root_driver_ref")==ref(DRIVER)
  and inp.get("main_source_ref")==ref(ENTRY) and inp.get("source_sha256")==sources()["source_sha256"]
  and inp.get("output_directory")==str(local(completion_ref["path"]).parent.relative_to(ROOT)),
  "Completion belongs to exact admitted source/output, no transplanted success")
 closed=read(e["ordinary_close_receipt_ref"]);reserve=inp["storage"]["reservation"]
 require(closed.get("kind")=="concurrent_completed_scope_close_receipt_v1"
  and closed.get("key")==reserve["key"]and closed.get("artifacts_deleted")==0
  and closed.get("value",{}).get("future_writes_and_relaunch_revoked")is True
  and closed["value"].get("request",{}).get("completed_ref")==completion_ref
  and closed["value"]["request"].get("direct_wait_ref")==e["actual_wait_ref"]
  and closed["value"].get("reservation_sha256")==hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),
  "Actual ordinary completed close of exact preceding charged reservation; no fabricated credit")
 return e
def make_config(q,caps=None):
 stage=q["stage"];parent=q["parent_id"];core=completion()
 templates=core.inactive_stage_configs_v1(q["context_ref"],parent,
  conversion_ref=q.get("conversion_ref"),native_run_ref=q.get("native_run_ref"))
 cfg=templates[{"convert":"conversion","native":"native","seal":"seal"}[stage]]
 cfg["execution_enabled"]=True
 # ROOT registration is staged beside config before reserve, never in a reserved target.
 cfg["registration_ref"]=q["registration_ref"]
 return cfg
def validate_request(record):
 global CAPS,PHASE,PARENT
 q=read(record);sources();scientific_review()
 require(q.get("schema")=="FLARE_remaining_CPU_ROOT_request_v1"and q.get("active")is True
  and q.get("root_authorizer")=="root_agent"and q.get("execution_authorized")is True
  and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("stage")in STAGES and q.get("parent_id")in VAL
  and q.get("phase")==q["stage"]+"_"+q["parent_id"]
  and q.get("GPU_authorized")is False and q.get("CUDA_authorized")is False and q.get("physical_gpu")is None
  and q.get("remaining_VAL_truth_authorized")is False and q.get("TEST_authorized")is False
  and q.get("automatic_retry_resume_or_fallback")is False and q.get("VAL000_rerun")is False,
  "Exact one-shot ROOT original missing-cell stage; no scoring, TEST, retry or VAL000")
 reviewed(q["source_review_ref"]);bind_selector(q["storage_policy_selector_ref"])
 prep=local(q["preparation_directory"]);packet=local(q["packet_directory"]);out=local(q["output_directory"])
 require(prep==local(record["path"]).parent and prep.is_dir()
  and q["preparation_directory"].startswith("research/reviews/FLARE_remaining_ROOT_preparation_")
  and q["packet_directory"].startswith("research/reviews/FLARE_remaining_ROOT_actual_")
  and q["output_directory"].startswith("data/FLARE_original_remaining_")
  and not any(a==b or a in b.parents or b in a.parents for a,b in ((prep,packet),(prep,out),(packet,out))),
  "Separate existing ROOT preparation and fresh disjoint reserved targets")
 require(type(q["cpu_ids"])is list and len(q["cpu_ids"])==len(set(q["cpu_ids"]))==4
  and all(type(x)is int and x>=0 for x in q["cpu_ids"]),"Four explicit ROOT CPU identities")
 require(q["registration_ref"]["path"].startswith(q["preparation_directory"]+"/"),"Registration only in separate preparation")
 ctx=completion()._context(q["context_ref"],parent_id=q["parent_id"]);row=ctx["targets"][q["parent_id"]]
 count=read(row["count_manifest_ref"]);plan=read(row["plan_ref"])
 require(count.get("schema")=="American_C3_full30Mb_observation_count_root_v1"and count.get("truth_opened")is False
  and count.get("source_observation_ref")==row["observation_ref"]and count.get("num_sites")==row["input_sites"]
  and count.get("num_samples")==1600 and count.get("sequence_length")==30000000
  and count.get("count_source_ref")==row["plan_ref"]and plan.get("sites")==row["input_sites"]
  and plan.get("parent_id")==q["parent_id"]and plan.get("observation_ref")==row["observation_ref"]
  and plan.get("focal_registry_ref")==row["focal_registry_ref"]and plan.get("targets_opened")is False,
  "Exact original plan JSON count source, full observations and same focals; no scientific array read in admission")
 reg=read(q["registration_ref"])
 require(reg.get("schema")=="FLARE_root_target_registration_v1"and reg.get("root_authorizer")=="root_agent"
  and reg.get("parent_id")==q["parent_id"]and reg.get("observation_ref")==row["observation_ref"]
  and reg.get("focal_registry_ref")==row["focal_registry_ref"]and reg.get("count_manifest_ref")==row["count_manifest_ref"]
  and reg.get("scientific_role")=="registered_heldout_full30Mb_comparison"
  and reg.get("query_ancestry_truth_opened")is False and reg.get("score_authorized")is False,
  "Actual H041 unchanged comparison target registration")
 if q["stage"]=="convert":
  require(q.get("conversion_ref")is None and q.get("native_run_ref")is None and q.get("previous_stage_evidence_ref")is None,
   "Conversion first; no old outputs/terminal substitution")
 else:
  converted=read(q["conversion_ref"])
  require(converted.get("parent_id")==q["parent_id"]and converted.get("focal_registry_ref")==row["focal_registry_ref"],
   "Same actual completed conversion")
  previous=q["conversion_ref"]if q["stage"]=="native"else q["native_run_ref"]
  stage_evidence(q["previous_stage_evidence_ref"],stage="convert"if q["stage"]=="native"else"native",
   parent=q["parent_id"],completion_ref=previous)
  if q["stage"]=="native":require(q.get("native_run_ref")is None,"Native has no prior native rerun")
 cfg=make_config(q);caps=stage_caps(q["stage"],cfg);require(q.get("caps")==caps,"Whole original output+8MiB AUX/admin charge; not a fit assertion")
 current(q,caps["wall_seconds"]+600)
 runtime=read(q["runtime_ref"])
 require(runtime.get("prefix")==str(ROOT/".runtime/env")and runtime.get("CPU_threads")==4
  and runtime.get("cpu_ids")==q["cpu_ids"]and runtime.get("CUDA_VISIBLE_DEVICES")==""
  and runtime.get("package_report_ref")==BASE_RUNTIME,"Existing qualified CPU interpreter/packages and4threads")
 directive=read(q["root_directive_ref"])
 require(directive.get("schema")=="FLARE_remaining_CPU_ROOT_directive_v1"and directive.get("root_authorizer")=="root_agent"
  and directive.get("active")is True and directive.get("directive")=="Execute one exact missing original FLARE stage after genuine fresh admission0"
  and all(directive.get(k)==q[k]for k in ("stage","parent_id","phase","context_ref","registration_ref","preparation_directory",
    "packet_directory","output_directory","source_manifest_ref","source_review_ref","runtime_ref","storage_policy_selector_ref",
    "cpu_ids","caps","created_at_utc","expires_at_utc","conversion_ref","native_run_ref","previous_stage_evidence_ref")),
  "Actual separate ROOT directive for complete reviewed stage, no automatic chain")
 CAPS=caps;PARENT=q["parent_id"];PHASE=q["phase"]
 return q,caps,runtime
def direct_roles(request_ref,q,cfgref,closure):
 row=completion()._context(q["context_ref"],parent_id=q["parent_id"])["targets"][q["parent_id"]]
 roles=[ref(MANIFEST),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(HANDOFF),ref(PRODUCERS[q["stage"]]),
  COMPLETION_REVIEW,q["source_review_ref"],q["runtime_ref"],cfgref,q["root_directive_ref"],request_ref,q["context_ref"],
  row["count_manifest_ref"],q["registration_ref"],closure,SELECTOR,q["storage_policy_selector_ref"],NAMESPACE,PIDFD,OWNED,FRAMEWORK]
 if q["stage"]!="convert":roles += [q["conversion_ref"],q["previous_stage_evidence_ref"],make_config(q)["binary_provenance_ref"]if q["stage"]=="native"else read(q["native_run_ref"])["binary_provenance_ref"]]
 if q["stage"]=="seal":roles.append(q["native_run_ref"])
 return roles
def input_spec(record):
 global _ACTIVE
 s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);current(s,300);cfg=read(s["refs"]["config"])
 require(record["path"]==q["packet_directory"]+"/input.json"and s.get("schema")=="FLARE_remaining_CPU_admitted_input_v1"
  and s.get("active")is True and s.get("root_authorizer")=="root_agent"and s.get("execution_authorized")is True
  and s.get("GPU_authorized")is False and s.get("physical_gpu")is None
  and s["root_driver_ref"]==ref(DRIVER)and s["main_source_ref"]==ref(ENTRY)
  and all(s.get(k)==q[k]for k in ("stage","phase","parent_id","output_directory","created_at_utc","expires_at_utc","runtime_ref"))
  and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
  and s["storage"]==read(s["refs"]["storage"])and cfg==make_config(q)and s["source_sha256"]==sources()["source_sha256"],
  "Exact admitted one-shot FLARE input/config/source/runtime")
 closure=read(s["transitive_source_closure_ref"])
 require(closure["source_sha256"]==s["source_sha256"]and closure["root_stage_request_ref"]==s["root_stage_request_ref"],"Exact staged source/input closure")
 storage=s["storage"]
 require(storage["reserved_new_bytes"]==caps["reservation_bytes"]and storage["reservation"]["reserved_bytes"]==caps["reservation_bytes"]
  and storage["output_directory"]==q["output_directory"]and storage["completed_overlay_policy_ref"]==POLICY_REF
  and storage["storage_policy_selector_ref"]==SELECTOR_REF and storage["job_id"]==s["job_id"]
  and storage["reservation"]["output"]==str(local(q["output_directory"]))
  and storage["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))],"Whole output+entire packet/AUX/admin selected-child charge")
 required=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"])
 require(len(storage["reservation"]["consumer_bindings"])<=64
  and all(r in storage["reservation"]["consumer_bindings"]for r in required),"All actual producer/runtime/control roles direct under64")
 _ACTIVE=s;return s,q,cfg
def verify_actual_roles(s,q,g):
 impl=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=False)
 complete=g.consumer_bindings_v1(list(state[3].values()),direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"]))
 require(all(r in s["storage"]["reservation"]["consumer_bindings"]for r in complete),"Every returned/current guard role retained directly;64cap unchanged")
 return complete
