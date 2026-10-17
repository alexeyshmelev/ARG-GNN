"""Inactive exact original Relate ALL contract. ROOT alone supplies real authority."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,importlib.util,json,re,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/Relate_exactALL_native_only_source_20261017_v1"
SOURCE=HERE+"/root_contract_v1.py";DRIVER=HERE+"/root_driver_v1.py";ENTRY=HERE+"/worker_entry_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json";HANDOFF=HERE+"/source_handoff.v1.json"
PARENT="ARGMix_prospective1611_TRAIN000";PHASE="native_ALL_only"
DIRECTIVE="ROOT authorizes one fresh exact original Relate ALL only from five authenticated readonly original preprocessing inputs, no internal resume, no conversion, no scoring, no automatic retry"
HELPER=dict(path="research/reviews/European_outer8_current_policy_successor_source_20261016_v1/prepare_outer000_v1.py",sha256="5bc07b0021b1b35b4ddee9ae6dce8680b61773413593ab829869125a7b7a3c3d")
assert hashlib.sha256((ROOT/HELPER["path"]).read_bytes()).hexdigest()==HELPER["sha256"]
_sp=importlib.util.spec_from_file_location("_Relate_exactALL_metadata_helpers",ROOT/HELPER["path"])
_h=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(_h)
require=_h.need;local=_h.local;ref=_h.ref;raw=_h.raw;read=_h.read;load=_h.load;put=_h.put;current=_h.current
DEPS=json.loads((ROOT/HERE/"dependencies.v1.json").read_text())
for _name,_value in DEPS.items():globals()[_name]=_value
CAPS=json.loads((ROOT/HERE/"caps.inactive.v1.json").read_text())
SELECTOR_REF=POLICY_REF=_CHOSEN=None
def binding():return read(ref(HERE+"/scientific_binding.v1.json"))
def sources():
 m=read(ref(MANIFEST))
 for path,sha in m["source_sha256"].items():raw(dict(path=path,sha256=sha))
 return m
def selected(record):
 v=read(record)
 require(v.get("schema")=="explicit_ROOT_storage_policy_selector_v1"
  and v.get("active")is True and v.get("root_authorizer")=="root_agent"
  and v.get("selector_source_ref")==SELECTOR and v.get("source_ref")==GUARD_REF
  and v.get("source_manifest_ref")==GUARD_MANIFEST and v.get("source_review_ref")==GUARD_REVIEW
  and v.get("accounting_mode")=="Relate_failed64GiB_retention_v1"
  and v.get("scheduling_ceiling_bytes")==980000000000 and v.get("human_limit_bytes")==1000000000000
  and v.get("warning_threshold_bytes")==950000000000
  and v.get("no_fallback_discovery_upgrade_or_in_run_swap")is True
  and v.get("future_ROOT_adapters_only")is True,"Only actual ROOT-activated exact reviewed Relate child")
 for k in("selector_source_ref","source_ref","source_manifest_ref","source_review_ref","policy_ref","repair_authority_ref","measurement_source_ref"):raw(v[k])
 return v
def bind_selector(record):
 global SELECTOR_REF,POLICY_REF,_CHOSEN
 v=selected(record)
 require(SELECTOR_REF is None or(SELECTOR_REF==record and _CHOSEN==v),"Immutable lifetime selected child; no swap")
 SELECTOR_REF=copy.deepcopy(record);POLICY_REF=copy.deepcopy(v["policy_ref"]);_CHOSEN=copy.deepcopy(v)
 return v
def guard(consumer):
 require(SELECTOR_REF is not None,"Actual ROOT selector before guard/memo")
 _h.SELECTOR=SELECTOR;_h.selection=selected
 return _h.guard(SELECTOR_REF,consumer)
def reviewed(record):
 v=read(record)
 require(v.get("independent")is True and v.get("verdict")in("SOURCE_PASS","PASS")
  and v.get("source_ref")==ref(DRIVER)and v.get("source_manifest_ref")==ref(MANIFEST)
  and v.get("source_handoff_ref")==ref(HANDOFF)and v.get("blockers")==[],
  "Actual independent review of this complete exact ALL source seal")
def authenticate_closure(record):
 """Validate only ROOT's metadata here. Payload rehash is after the actual owned gate."""
 v=read(record);b=binding()
 require(v.get("schema")=="ROOT_Relate_five_original_inputs_readonly_authentication_v1"
  and v.get("root_authorizer")=="root_agent"and v.get("status")=="authenticated_readonly"
  and v.get("original_key")==b["original_key"]and v.get("scientific_binding_ref")==ref(HERE+"/scientific_binding.v1.json")
  and v.get("source_evidence")==b["source_evidence"]and v.get("support")==b["support"]
  and v.get("original_inputs_unchanged")is True and v.get("no_payload_rewrite_copy_or_normalization")is True,
  "Genuine ROOT authentication of original full support/source/preprocessing lineage")
 rows=v.get("files")
 require(isinstance(rows,list)and len(rows)==5 and [x.get("path")for x in rows]==b["readonly_input_paths"],
  "Exactly five original paths in exact ALL argument order")
 for x in rows:
  require(set(x)=={"path","sha256","bytes"}and isinstance(x["sha256"],str)
   and re.fullmatch("[0-9a-f]{64}",x["sha256"])and type(x["bytes"])is int and x["bytes"]>0,
   "Actual ROOT SHA256 and byte size for every original input; no placeholders")
 stamp=datetime.fromisoformat(v["authenticated_at_utc"])
 require(stamp.tzinfo and stamp<=datetime.now(timezone.utc),"Genuine nonfuture readonly authentication")
 return v
def validate_request(record):
 q=read(record);current(q,CAPS["wall_seconds"]+600);sources();bind_selector(q["storage_policy_selector_ref"])
 b=binding()
 require(q.get("schema")=="Relate_exactALL_native_only_ROOT_request_v1"and q.get("active")is True
  and q.get("root_authorizer")=="root_agent"and q.get("execution_authorized")is True
  and q.get("phase")==PHASE and q.get("parent_id")==PARENT and q.get("stage")=="native_all"
  and q.get("source_manifest_ref")==ref(MANIFEST)and q.get("caps")==CAPS
  and q.get("scientific_binding_ref")==ref(HERE+"/scientific_binding.v1.json")
  and q.get("exact_argv")==b["exact_argv"]and q.get("support")==b["support"]
  and q.get("GPU_authorized")is False and q.get("physical_gpu")is None
  and q.get("internal_resume")is False and q.get("conversion_or_export_authorized")is False
  and q.get("normalization_or_scoring_authorized")is False
  and q.get("automatic_retry_resume_or_fallback")is False
  and q.get("old_key_and_all_old_paths_write_revoked")is True,
  "One fresh exact full ALL only; old failed source/scopes remain immutable")
 preparation=local(q["preparation_directory"]);packet=local(q["packet_directory"]);output=local(q["output_directory"])
 require(preparation==local(record["path"]).parent and preparation.is_dir()
  and q["preparation_directory"].startswith("research/reviews/Relate_exactALL_native_only_ROOT_preparation_")
  and q["packet_directory"].startswith("research/reviews/Relate_exactALL_native_only_ROOT_actual_")
  and q["output_directory"].startswith("data/Relate_exactALL_native_only_")
  and not any(a==z or a in z.parents or z in a.parents for a,z in((preparation,packet),(preparation,output),(packet,output))),
  "Separate existing ROOT preparation and fresh exclusive disjoint charged namespaces")
 require(re.fullmatch("ARGMix/Relate-exactALL-native-only-[A-Za-z0-9_-]+",q.get("new_key",""))is not None
  and q["new_key"]!=b["original_key"],"Fresh never-reserved key, no rebinding")
 cpu=q.get("cpu_ids")
 require(isinstance(cpu,list)and len(cpu)==1 and type(cpu[0])is int and cpu[0]>=0,
  "ROOT separately authenticates one actual CPU; original one-thread author algorithm")
 reviewed(q["source_review_ref"]);authenticate_closure(q["readonly_input_authentication_ref"])
 runtime=read(q["runtime_ref"])
 require(runtime.get("schema")=="ROOT_Relate_exactALL_CPU_runtime_v1"and runtime.get("root_authorizer")=="root_agent"
  and runtime.get("actual_runtime_qualified")is True and runtime.get("prefix")==str(ROOT/".runtime/env")
  and runtime.get("native_binary_ref")==b["binary_ref"]and runtime.get("native_threads")==1
  and runtime.get("CPU_threads")==1 and runtime.get("cpu_ids")==cpu
  and runtime.get("CUDA_VISIBLE_DEVICES")=="" and runtime.get("standalone_numerical_probe")is False,
  "Actual ROOT qualified original binary/shared interpreter/one CPU; no numerical qualification")
 reg=read(q["registration_ref"])
 require(reg.get("schema")=="ROOT_Relate_exactALL_repeat_registration_v1"and reg.get("root_authorizer")=="root_agent"
  and reg.get("registered_before_execution")is True and reg.get("original_key")==b["original_key"]
  and reg.get("condition")=="fresh_exact_ALL_after_unfinished_original_inference"
  and isinstance(reg.get("repeat_reason"),str)and reg["repeat_reason"].strip()
  and reg.get("scientific_binding_ref")==q["scientific_binding_ref"]
  and reg.get("resource_only_changes")==dict(scientific_seconds=43200,rss_bytes=CAPS["rss_bytes"],reservation_bytes=CAPS["reservation_bytes"]),
  "Prospective registered unfinished-cell repeat; unchanged science and explicit operational changes")
 raw(reg["decisions_snapshot_ref"]);raw(reg["ledger_snapshot_ref"])
 require(datetime.fromisoformat(reg["registered_at_utc"])<=datetime.fromisoformat(q["created_at_utc"]),
  "Actual registration precedes request")
 d=read(q["root_directive_ref"])
 keys=("preparation_directory","packet_directory","output_directory","new_key","source_manifest_ref","source_review_ref",
  "scientific_binding_ref","readonly_input_authentication_ref","storage_policy_selector_ref","registration_ref","runtime_ref",
  "exact_argv","support","caps","cpu_ids","created_at_utc","expires_at_utc")
 require(d.get("root_authorizer")=="root_agent"and d.get("active")is True and d.get("directive")==DIRECTIVE
  and all(d.get(k)==q[k]for k in keys)and d.get("old_key_and_all_old_paths_write_revoked")is True,
  "Separate genuine ROOT directive bound to exact new namespace/input closure/expiry; no replay")
 return q,CAPS,runtime
def direct_roles(request_ref,q,cfgref,closure):
 # Every actual returned guard remains direct; these add producer-specific refs.
 return [ref(MANIFEST),ref(SOURCE),ref(DRIVER),ref(ENTRY),ref(HANDOFF),
  q["source_review_ref"],q["runtime_ref"],q["root_directive_ref"],request_ref,
  q["readonly_input_authentication_ref"],q["registration_ref"],cfgref,closure,
  SELECTOR,q["storage_policy_selector_ref"],NAMESPACE,PIDFD,OWNED,SUPERVISOR]
def make_config(q,caps):
 return dict(schema="Relate_exactALL_native_only_operational_config_v1",caps=caps,exact_argv=q["exact_argv"],
  readonly_input_authentication_ref=q["readonly_input_authentication_ref"],
  short_tmp_directory=q["packet_directory"]+"/worker_aux/tmp",native_cwd=q["output_directory"])
def input_spec(record):
 s=read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);current(s,300);cfg=read(s["refs"]["config"])
 require(record["path"]==q["packet_directory"]+"/input.json"and s.get("schema")=="Relate_exactALL_native_only_admitted_input_v1"
  and s.get("active")is True and s.get("root_authorizer")=="root_agent"and s.get("execution_authorized")is True
  and s.get("GPU_authorized")is False and s.get("physical_gpu")is None and s["stage"]=="native_all"
  and s["root_driver_ref"]==ref(DRIVER)and s["main_source_ref"]==ref(ENTRY)and s["parent_id"]==PARENT and s["phase"]==PHASE
  and s["output_directory"]==q["output_directory"]and s["worker_auxiliary_scope"]==q["packet_directory"]+"/worker_aux"
  and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"]and s["runtime_ref"]==q["runtime_ref"]
  and s["storage"]==read(s["refs"]["storage"])and cfg==make_config(q,caps)
  and s["source_sha256"]==sources()["source_sha256"],"Exact admitted producer/config/runtime/source and lifetime")
 closure=read(s["transitive_source_closure_ref"])
 require(closure==dict(source_sha256=s["source_sha256"],root_stage_request_ref=s["root_stage_request_ref"]),
  "Immutable complete source/input request closure")
 z=s["storage"]
 require(z["reserved_new_bytes"]==CAPS["reservation_bytes"]and z["reservation"]["reserved_bytes"]==CAPS["reservation_bytes"]
  and z["output_directory"]==q["output_directory"]and z["completed_overlay_policy_ref"]==POLICY_REF
  and z["storage_policy_selector_ref"]==SELECTOR_REF and z["job_id"]==s["job_id"]
  and z["reservation"]["key"]==q["new_key"]
  and z["reservation"]["output"]==str(local(q["output_directory"]))
  and z["reservation"]["auxiliary_scopes"]==[str(local(q["packet_directory"]))],
  "32GiB whole charge:31GiB native plus1GiB all packet/AUX, original guards")
 roles=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"])
 require(len(z["reservation"]["consumer_bindings"])<=64
  and all(x in z["reservation"]["consumer_bindings"]for x in roles),"All required direct roles under unchanged64 ceiling")
 return s,q,cfg
def verify_actual_roles(s,q,g):
 impl=ref(g.budget.SOURCE_REL);launcher=ref(g.budget.LAUNCHER_REL)
 with g.budget.v4.mutex(ROOT):state=g._state_locked(impl["sha256"],launcher["sha256"],fresh=False)
 full=g.consumer_bindings_v1(list(state[3].values()),direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"]))
 require(all(x in s["storage"]["reservation"]["consumer_bindings"]for x in full),"Every actual returned guard remains directly bound")
 return full
