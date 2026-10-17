"""Build-only ROOT metadata for sealed original CPU preparation v2. Default is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN001_019_preparation_ROOT_builder_source_20261017_v1"
PIPE="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
SOURCE=REL+"/build_ROOT_authority_v1.py"
MANIFEST_REF=dict(path=PIPE+"/source_manifest.v1.json",sha256="2c9a08835aa60dafa0a25ccf51ac1933655797e89f9e1aaa49a25ff97d8d9e8b")
WORKER_REF=dict(path=PIPE+"/worker_entry_v1.py",sha256="4f8acf67fc0da781c79f738831577b9c258ecb0d92946a953b03353985f2980e")
CONTROL=dict(path="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/ROOT_batch_admission_root01_oct17/batch_input.json",
 sha256="dfd6ced48bf7ccfe8ebf9d58e3ffb2ab5193e8a586882d873e3bd4eb9279c092")
def need(ok,message):
 if not ok:raise ValueError(message)
def local(value):
 p=Path(value);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained source/metadata path")
 return p
def ref(value):
 p=local(value);s=p.lstat()
 need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=4*2**20 and p.suffix in(".py",".json"),"Small regular source/JSON only")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
 need(type(record)is dict and set(record)=={"path","sha256"}and ref(record["path"])==record,"Exact immutable metadata")
 return json.loads(local(record["path"]).read_text())
def put(directory,name,value):
 need(Path(name).name==name,"Flat exclusive ROOT metadata")
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();need(len(data)<=2**20,"Bounded build-only metadata")
 with(directory/name).open("xb")as f:need(f.write(data)==len(data),"Complete write")
 return ref(directory/name)
def compact_close(record,parent):
 close=read(record);receipt=read(close["receipt_ref"])
 need(close["parent_id"]==parent and close["actual_direct_OS_wait0"]is True
  and close["all_owned_processes_and_groups_drained"]is True and type(receipt["key"])is str,
  "ROOT genuine successful compact close acceptance, never inferred from completed.json")
 return close
def build_values(parameters_ref):
 p=read(parameters_ref);now=datetime.now(timezone.utc)
 start=datetime.fromisoformat(p["created_at_utc"]);end=datetime.fromisoformat(p["expires_at_utc"])
 need(start.tzinfo and end.tzinfo and start<=now<end and 0<(end-start).total_seconds()<=900,
  "Fresh finite explicit ROOT build-only directive")
 need(p.get("schema")=="TreeLoom_preparation_ROOT_explicit_build_parameters_v1"and p.get("active")is True
  and p.get("root_authorizer")=="root_agent"and p.get("build_only")is True
  and p.get("admission_or_execution_authorized")is False and p.get("source_only_estimate")is False
  and p.get("simulation_batch_control_input_ref")==CONTROL,"Exact build-only metadata authority, no runtime operation")
 m=read(MANIFEST_REF);review=read(p["pipeline_source_review_ref"])
 expected_sources=[r for r in m["refs"]if r["path"].startswith(PIPE+"/")]
 need(review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
  and review.get("source_manifest_ref")==MANIFEST_REF and all(r in review.get("reviewed_sources",[])for r in expected_sources),
  "Exact complete independent pipeline-v2 source pass")
 control=read(CONTROL);sim=read(control["authority_ref"])
 need(sim["schema"]=="ROOT_TRAIN002_019_finite_batch_authority_v1"and sim["active"]is True
  and sim["root_authorizer"]=="root_agent","Existing genuine simulation authority supplies shared coordinator")
 lanes=p["CPU_lanes"];need(type(lanes)is list and len(lanes)==2
  and all(type(v)is list and len(v)==8 and all(type(i)is int and i>=0 for i in v)for v in lanes)
  and len(set(lanes[0]+lanes[1]))==16,"Explicit two distinct8CPU lanes, never a resource probe")
 capacity=p["actual_initial_capacity"];stamp=datetime.fromisoformat(capacity["created_at_utc"])
 need(stamp.tzinfo and 0<=(now-stamp).total_seconds()<=240 and capacity.get("root_authorizer")=="root_agent"
  and capacity.get("actual_current_observation")is True and capacity.get("source_only_estimate")is False
  and capacity.get("admissible_new_bytes",0)>=33554432,"ROOT supplied actual fresh32MiB capacity, never invented")
 scope=p["scope_id"];need(type(scope)is str and scope and all(ch.isalnum()or ch in"_-"for ch in scope),"Explicit ROOT scope")
 metadata=read(p["TRAIN000_preparation_metadata_ref"]);obs=read(metadata["observation_ref"]);native=read(metadata["native_preparation_ref"])
 need(metadata["parent_id"]==obs["parent_id"]==native["parent_id"]=="american_c3_train_000"
  and metadata["observation_only"]is True and metadata["all_native_trees_nodes_sites_samples_retained"]is True
  and len(metadata["focal_identity_sha256"])==64 and native["status"]=="completed_native_C3_path_preparation"
  and native["observation_ref"]==metadata["observation_ref"]and native["focals_ref"]==metadata["focal_registry_ref"]
  and native["query_focal_count"]==10000 and native["query_ancestry_truth_opened"]is False
  and obs["sequence_length_bp"]==30000000 and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
  "Original full30Mb TRAIN000 observed/focal metadata reused, no payloads")
 row={k:metadata[k]for k in("parent_id","observation_ref","native_preparation_ref","focal_registry_ref","focal_identity_sha256",
  "observation_only","all_native_trees_nodes_sites_samples_retained")}
 for stage,key in[("inference","inference_close_accepted_ref"),("native","native_close_accepted_ref")]:
  rr=p["TRAIN000_"+stage+"_close_accepted_ref"];compact_close(rr,"american_c3_train_000");row[key]=rr
 inp=read(p["TRAIN001_inference_input_ref"]);done_ref=ref(inp["output_directory"]+"/completed.json");done=read(done_ref)
 observation=read(done["observation_ref"])
 need(inp["parent_id"]==observation["parent_id"]=="american_c3_train_001"and inp["stage"]=="infer"
  and inp["root_driver_ref"]==inp["main_source_ref"]==m["condition"]["original_TRAIN001_inference_source_ref"]
  and done["status"]=="completed_current_C3_inference"
  and observation["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
  "Actual original completed TRAIN001 inference lineage, no substitute science")
 previous=dict(input_ref=p["TRAIN001_inference_input_ref"],config_ref=inp["refs"]["config"],completion_ref=done_ref,
  wait_ref=ref(inp["worker_auxiliary_scope"]+"/actual_wait.json"),outer_ref=p["TRAIN001_inference_outer_terminal_ref"],
  observation_ref=done["observation_ref"],roster_ref=observation["roster_ref"])
 compact_close(p["TRAIN001_inference_close_accepted_ref"],"american_c3_train_001")
 capsule=dict(schema="TreeLoom_preparation_predecessor_capsule_v1",parent_id="american_c3_train_001",next_stage="native_paths",
  refs=previous,metadata={k:read(v)for k,v in previous.items()},close_accepted_ref=p["TRAIN001_inference_close_accepted_ref"],
  permanent_revocation_ref=p["TRAIN001_inference_permanent_revocation_ref"],query_truth_opened=False)
 for parent in("000","001"):
  done=read(p["TRAIN"+parent+"_simulation_completion_ref"])
  need(done["parent_id"]=="american_c3_train_"+parent and done["status"]=="completed_C3_simulation_reconstruction",
   "Actual original simulator completion metadata, truth payloads unopened")
 authority=read(ref(PIPE+"/ROOT_authority.template.INACTIVE.v1.json"))
 expires=datetime.fromisoformat(p["pipeline_expires_at_utc"]);sim_end=datetime.fromisoformat(sim["expires_at_utc"])
 need(now+timedelta(seconds=13350)<expires<=sim_end and(expires-now).total_seconds()<=172800,
  "Explicit finite original work/close window contained within original coordinator authority")
 authority.update(active=True,source_only=False,root_authorizer="root_agent",created_at_utc=now.isoformat(),expires_at_utc=expires.isoformat(),
  source_manifest_ref=MANIFEST_REF,source_review_ref=p["pipeline_source_review_ref"],simulation_batch_control_input_ref=CONTROL,
  scope_id=scope,CPU_lanes=lanes,actual_initial_capacity=capacity,TRAIN000_readiness_reuse=row,
  TRAIN000_simulation_completion_ref=p["TRAIN000_simulation_completion_ref"],TRAIN001_simulation_completion_ref=p["TRAIN001_simulation_completion_ref"],
  run_directory=PIPE+"/ROOT_actual_pipeline_"+scope,admission_directory=PIPE+"/ROOT_pipeline_admission_"+scope,
  TRAIN001_inference_predecessor_capsule_ref=None)
 directory=local(p["run_directory"])
 need(str(directory.relative_to(ROOT))==REL+"/ROOT_once_"+scope,"Exclusive dedicated build-only ROOT namespace")
 return p,directory,capsule,authority
def build(parameters_ref):
 p,directory,capsule,authority=build_values(parameters_ref)
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact ROOT source/metadata interpreter")
 need(not directory.exists(),"One-shot fresh metadata build, no overwrite")
 directory.mkdir();put(directory,"parameters.binding.json",dict(parameters_ref=parameters_ref,build_only=True,operation=False))
 cr=put(directory,"TRAIN001_inference_predecessor_capsule.json",capsule);authority["TRAIN001_inference_predecessor_capsule_ref"]=cr
 need(ref(WORKER_REF["path"])==WORKER_REF,"Exact reviewed pipeline worker source")
 sp=importlib.util.spec_from_file_location("_ROOT_build_only_preparation_worker_v2",local(WORKER_REF["path"]))
 w=importlib.util.module_from_spec(sp);sys.modules[sp.name]=w;sp.loader.exec_module(w)
 w.PARENT="american_c3_train_001";w.STAGE="native_paths"
 w.predecessor_capsule(cr,read(MANIFEST_REF)["condition"]) # Real metadata route only, no initialize or guard.
 ar=put(directory,"pipeline_authority.json",authority)
 w.validate_pipeline_authority(ar) # Reads original finite authority metadata, no lock/guard/probe.
 result=dict(status="ROOT_PREPARATION_AUTHORITY_BUILT_ONLY",authority_ref=ar,predecessor_capsule_ref=cr,
  source_manifest_ref=MANIFEST_REF,independent_source_review_ref=p["pipeline_source_review_ref"],
  admission=False,execution=False,guard_imported=False,capacity_or_journal_measured=False,truth_payloads_read=False)
 put(directory,"built_only.json",result);return result
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--parameters-ref");a=p.parse_args(argv)
 if a.parameters_ref is None:
  print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
 print(json.dumps(build(json.loads(a.parameters_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
