"""Finite metadata-only ROOT builder for executable compact focal admission.

Default is inert. No guard import/probe/reserve/launch; ROOT consumes the actual
builder0, runs returned admission once, consumes its actual0, then runs controller.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,re,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/TreeLoom_compact_focal_CPU_continuation_source_20261017_v1"
MANIFEST=REL+"/source_manifest.v1.json"
BATCH=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json",sha256="dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092")
INFERENCE_MANIFEST=dict(path="research/reviews/frozen_TRAIN002_019_inference_only_source_20261017_v1/source_manifest.v1.json",sha256="4a96cba030ea11b302bbca0bd0eed2c5f8da7f716f5b5e448cec242bb06a8a95")
TRAIN001_INPUT=dict(path="research/reviews/frozen_TRAIN001_observe_ready_20261017_v2/ROOT_actual_TRAIN001_root01_oct17_v2/input.json",sha256="3f7399e8b26d1c8177b23e1cf6008c77c6e66d82256b54f304fa766312068776")
TRAIN001_SUPERVISOR=dict(path="research/reviews/research_execution_ROOT_20261017_v34/TRAIN001_observation_v2_execution_actual_wait.root_v1.json",sha256="88af5a2c743e6dbc90a97b431929ae93f53f4e6eed73c414e37e0ded0618e6b8")
def need(ok,message):
 if not ok:raise ValueError(message)
def local(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained ROOT metadata")
 return p
def ref(path):
 p=local(path);s=p.lstat();need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=262144,"Bounded singly linked metadata/source")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 need(type(r)is dict and set(r)=={"path","sha256"}and ref(r["path"])==r,"Exact immutable metadata ref")
 p=local(r["path"]);need(p.suffix==".json"and p.stat().st_size<=262144,"JSON metadata only, no scientific payload")
 return json.loads(p.read_text())
def save(directory,name,value):
 raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
 need(len(raw)<=65536 and len(list(directory.iterdir()))<4,"Bounded flat ROOT builder output")
 fd=os.open(directory/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
 with os.fdopen(fd,"wb")as stream:need(stream.write(raw)==len(raw),"Complete exclusive ROOT metadata");stream.flush();os.fsync(stream.fileno())
 return ref(directory/name)
def train001_ready(p):
 need(p["TRAIN001_inference_input_ref"]==TRAIN001_INPUT,"Exact existing closed TRAIN001v2 inference input")
 inp=read(TRAIN001_INPUT);request=read(inp["root_stage_request_ref"])
 done_ref=ref(inp["output_directory"]+"/completed.json");done=read(done_ref)
 wait_ref=ref(inp["worker_auxiliary_scope"]+"/actual_wait.json");wait=read(wait_ref);sw=read(TRAIN001_SUPERVISOR)
 outer=read(p["TRAIN001_inference_outer_terminal_ref"]);rev=read(p["TRAIN001_inference_permanent_revocation_ref"])
 accepted=read(p["TRAIN001_inference_close_accepted_ref"]);receipt=read(accepted["receipt_ref"])
 need(inp["parent_id"]==wait["parent_id"]=="american_c3_train_001"and inp["stage"]=="infer"
  and done["status"]=="completed_current_C3_inference"and wait["input_ref"]==TRAIN001_INPUT and wait["completed_ref"]==done_ref
  and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
  and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
  and type(wait["returncode"])is int and wait["returncode"]==0 and wait["all_owned_processes_and_groups_drained"]is True
  and wait["operational_failure"]is None and wait["cleanup_errors"]==[]and all(x["returncode"]==0 for x in wait["adopted_waits"])
  and sw["actual_direct_wait"]is True and type(sw["actual_returncode"])is int and sw["actual_returncode"]==0
  and sw["pid"]==wait["supervisor_pid"]and sw["input_ref"]==TRAIN001_INPUT,"Genuine original TRAIN001 worker/supervisor0 and drain")
 need(outer["root_authorizer"]=="root_agent"and outer["actual_terminal_consumed"]is True
  and type(outer["actual_exit_code"])is int and outer["actual_exit_code"]==0
  and outer["input_ref"]==TRAIN001_INPUT and outer["completed_ref"]==done_ref
  and outer["actual_worker_wait_ref"]==wait_ref and outer["actual_supervisor_wait_ref"]==TRAIN001_SUPERVISOR
  and outer["all_owned_processes_and_groups_drained"]is True,"Actual ROOT TRAIN001 observation terminal, no new UIclaim")
 need(accepted["parent_id"]=="american_c3_train_001"and accepted["actual_direct_OS_wait0"]is True
  and accepted["all_owned_processes_and_groups_drained"]is True and receipt["key"]==inp["storage"]["reservation"]["key"]
  and rev["input_ref"]==TRAIN001_INPUT and rev["direct_wait_ref"]==wait_ref and rev["completed_ref"]==done_ref
  and rev["actual_outer_terminal_ref"]==p["TRAIN001_inference_outer_terminal_ref"]
  and rev["all_original_launch_resume_retry_write_routes_permanently_revoked"]is True,
  "Actual accepted ordinary inference close and separate permanent revocation; no new close/credit")
 obs=read(done["observation_ref"]);sim_ref=request["prior"]["completion_ref"];sim=read(sim_ref)
 need(obs["parent_id"]==sim["parent_id"]=="american_c3_train_001"and obs["sequence_length_bp"]==30000000
  and obs["query_ancestry_truth_in_inputs"]is False and sim["status"]=="completed_C3_simulation_reconstruction",
  "Original same-parent full30Mb inference and simulation metadata only")
 return dict(schema="TreeLoom_inference_only_parent_ready_v1",root_authorizer="root_agent",parent_id="american_c3_train_001",
  input_ref=TRAIN001_INPUT,config_ref=inp["refs"]["config"],completed_ref=done_ref,direct_wait_ref=wait_ref,
  actual_supervisor_wait_ref=TRAIN001_SUPERVISOR,actual_outer_terminal_ref=p["TRAIN001_inference_outer_terminal_ref"],
  observation_ref=done["observation_ref"],roster_ref=obs["roster_ref"],inference_close_accepted_ref=p["TRAIN001_inference_close_accepted_ref"],
  permanent_revocation_ref=p["TRAIN001_inference_permanent_revocation_ref"],simulation_completion_ref=sim_ref,
  source_manifest_ref=request["source_manifest_ref"],inference_source_ref=inp["root_driver_ref"],class_order=["EUR","EAS","AFR"],
  sequence_length_bp=30000000,query_truth_opened=False,output_frozen=True,actual_original_worker_and_supervisor_OS_wait0=True,
  all_owned_processes_and_groups_drained=True,UI_tool_terminal_claimed=False)
def build_values(parameters_ref):
 p=read(parameters_ref);now=datetime.now(timezone.utc);start=datetime.fromisoformat(p["created_at_utc"]);end=datetime.fromisoformat(p["expires_at_utc"])
 need(start.tzinfo and end.tzinfo and start<=now<end and 0<(end-start).total_seconds()<=900
  and p["schema"]=="ROOT_compact_focal_CPU_explicit_build_parameters_v1"and p["root_authorizer"]=="root_agent"
  and p["active"]is True and p["build_only"]is True and p["admission_or_execution_authorized"]is False,
  "Finite<=15minute ROOT metadata-only builder authority")
 m=read(ref(MANIFEST));review=read(p["operational_source_review_ref"])
 need(review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]
  and review["source_manifest_ref"]==ref(MANIFEST)
  and all(ref(REL+"/"+name)in review["reviewed_sources"]for name in("worker_entry_v1.py","batch_ROOT_v1.py","close_unit_ROOT_v1.py","build_ROOT_authority_v1.py")),
  "Exact independent operational four-source pass")
 need(p["simulation_batch_control_input_ref"]==BATCH,"Genuine original simulation coordinator control")
 control=read(BATCH);simulation_authority=read(control["authority_ref"])
 infer_admin=read(p["inference_only_admin_input_ref"]);ia=read(infer_admin["authority_ref"])
 need(infer_admin["source_manifest_ref"]==INFERENCE_MANIFEST and ia["source_manifest_ref"]==INFERENCE_MANIFEST
  and ia["simulation_batch_control_input_ref"]==BATCH and ia["stages"]==["infer"]and ia["inference_only"]is True,
  "Genuine independently admitted inference-only source frontier namespace")
 lanes=p["CPU_lanes"]
 need(type(lanes)is list and len(lanes)==2 and all(type(v)is list and len(v)==8 and all(type(i)is int and i>=0 for i in v)for v in lanes)
  and len(set(lanes[0]+lanes[1]))==16,"Explicit two distinct compact8CPU lanes; actual availability checked at admission")
 cap=p["actual_initial_capacity"];stamp=datetime.fromisoformat(cap["created_at_utc"])
 need(stamp.tzinfo and 0<=(now-stamp).total_seconds()<=240 and cap["root_authorizer"]=="root_agent"
  and cap["actual_current_observation"]is True and cap["source_only_estimate"]is False and cap["admissible_new_bytes"]>=33554432,
  "ROOT genuine fresh32MiB admin capacity; never estimated by builder")
 scope=p["scope_id"];need(type(scope)is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",scope)is not None,"Fresh bounded compact scope")
 expires=datetime.fromisoformat(p["controller_expires_at_utc"]);sim_end=datetime.fromisoformat(simulation_authority["expires_at_utc"])
 need(now+__import__("datetime").timedelta(seconds=13350)<expires<=sim_end
  and(expires-now).total_seconds()<=172800,"Finite compact controller within actual original coordinator authority")
 a=read(ref(REL+"/ROOT_authority.template.INACTIVE.v1.json"))
 run=REL+"/ROOT_actual_pipeline_"+scope
 a.update(active=True,source_only=False,root_authorizer="root_agent",created_at_utc=now.isoformat(),expires_at_utc=expires.isoformat(),
  scope_id=scope,source_manifest_ref=ref(MANIFEST),source_review_ref=p["operational_source_review_ref"],
  simulation_batch_control_input_ref=BATCH,CPU_lanes=lanes,actual_initial_capacity=cap,
  run_directory=run,admission_directory=REL+"/ROOT_pipeline_admission_"+scope,slot_directory=run+"/producer_slots",
  all20_adapter_output_directory=run+"/all20_ready_control",all20_adapter_source_ref=p.get("all20_adapter_source_ref"),
  all20_adapter_manifest_ref=p.get("all20_adapter_manifest_ref"),inference_only_admin_input_ref=p["inference_only_admin_input_ref"],
  inference_only_run_directory=ia["run_directory"],TRAIN001_inference_ready_ref=None)
 for key in("all20_adapter_source_ref","all20_adapter_manifest_ref"):
  if a[key]is not None:need(ref(a[key]["path"])==a[key],"Actual sealed later all20 metadata adapter source/manifest pin")
 need(p["all_other_compact_controller_routes_revoked"]is True,"One active compact controller, no extra producer pool")
 a["all_other_compact_controller_routes_revoked"]=True
 return p,REL+"/ROOT_builder_"+scope,train001_ready(p),a
def build(parameters_ref):
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
  and not any(n in sys.modules for n in("torch","torch_geometric")),"Fresh ROOT stdlib builder interpreter")
 p,directory,ready,a=build_values(parameters_ref);d=local(directory);need(not d.exists(),"One-shot exclusive metadata builder namespace")
 d.mkdir();ready_ref=save(d,"TRAIN001_inference_ready.json",ready);a["TRAIN001_inference_ready_ref"]=ready_ref
 ar=save(d,"authority.json",a)
 argv=[".runtime/env/bin/python","-B","-s","-m",REL.replace("/",".")+".batch_ROOT_v1","--admit-batch-ref",json.dumps(ar,sort_keys=True)]
 result=dict(status="ROOT_COMPACT_FOCAL_METADATA_BUILT_NO_ADMISSION",authority_ref=ar,TRAIN001_inference_ready_ref=ready_ref,
  admit_argv=argv,environment={"PYTHONPATH":".:src:scripts"},admission_started=False,scientific_execution_started=False,
  automatic_launch=False,ROOT_must_consume_actual_builder0_then_admission0_before_single_controller_start=True)
 save(d,"built.json",result);return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parameters-ref");a=p.parse_args(argv)
 if not a.parameters_ref:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
 print(json.dumps(build(json.loads(a.parameters_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
