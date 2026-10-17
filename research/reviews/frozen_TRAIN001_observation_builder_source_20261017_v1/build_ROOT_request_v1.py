"""ROOT-only metadata builder for the independently reviewed TRAIN001 observation.
Default invocation is inert. This script never admits or launches a process.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/frozen_TRAIN001_observation_builder_source_20261017_v1"
OBSERVE="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1"
SOURCE=dict(path=OBSERVE+"/worker_entry_v1.py",sha256="d672b38e5d317ad9844104e1e0f741c6e338ac0f33b39e7d0f897f124a01a912")
MANIFEST=dict(path=OBSERVE+"/source_manifest.v1.json",sha256="ff208ebdf593811aacd876257aaa5c329d904b18522fec97bd306b5f58bea1a5")
REVIEW=dict(path="research/reviews/frozen_TRAIN001_observation_independent_20261017_v1/review.observation.v1.json",
 sha256="a39a3cfe7403d0c6e6809490ee9355d3f1fb57efdf8b8ed7e02542db88f31b5d")
ACTUAL_SIM_INPUT=dict(path="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/ROOT_actual_TRAIN001_root02_oct17_v5/input.json",
 sha256="91743348259ade65ccf83017ece27a01ba61df0147c8f5ee192e9f05b289c491")
def need(ok,msg):
 if not ok:raise ValueError(msg)
def local(value):
 p=Path(value);p=p if p.is_absolute()else ROOT/p
 need(p!=ROOT and p.is_relative_to(ROOT)and".."not in p.parts and p.resolve()==p,"Canonical contained metadata path");return p
def ref(value):
 p=local(value);st=p.lstat()
 need(stat.S_ISREG(st.st_mode)and st.st_nlink==1 and st.st_size<=4*2**20,"Small singly linked source or JSON")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(rr):
 need(type(rr)is dict and set(rr)=={"path","sha256"}and ref(rr["path"])==rr,"Exact existing immutable metadata ref")
 return json.loads(local(rr["path"]).read_text())
def actual_prior():
 inp=read(ACTUAL_SIM_INPUT);packet=Path(ACTUAL_SIM_INPUT["path"]).parent
 need(inp["parent_id"]=="american_c3_train_001"and inp["phase"]=="simulate_train_001"and inp["stage"]=="simulate","Actual v5 TRAIN001 predecessor")
 completed=ref(inp["output_directory"]+"/completed.json");done=read(completed)
 observation=done["observation_ref"];observed=read(observation)
 return dict(input_ref=ACTUAL_SIM_INPUT,config_ref=inp["refs"]["config"],completion_ref=completed,
  wait_ref=ref(inp["worker_auxiliary_scope"]+"/actual_wait.json"),
  outer_ref=ref(packet/"outer_terminal.root_actual_v1.json"),
  observation_ref=observation,roster_ref=observed["roster_ref"])
def compose_request(*,capacity,scope_id,prior,manifest,expires_at,now):
 need(isinstance(scope_id,str)and scope_id and all(c.isalnum()or c in"_-"for c in scope_id),"Fresh explicit scope ID")
 need(now.tzinfo and expires_at.tzinfo and expires_at>now+timedelta(seconds=10950),"New finite observation stage clock with full wall/evidence headroom")
 need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
  and capacity.get("source_only_estimate")is False and capacity.get("parent_id")=="american_c3_train_001","Genuine fresh ROOT capacity, never an authored proposal")
 need(datetime.fromisoformat(capacity["created_at_utc"])<=now<datetime.fromisoformat(capacity["expires_at_utc"])
  and expires_at<=datetime.fromisoformat(capacity["expires_at_utc"]),"Observation remains inside genuine capacity lifetime")
 caps=manifest["condition"]["caps_proposed"]
 ids=capacity["cpu_ids"]
 need(type(ids)is list and len(ids)==len(set(ids))==8 and all(type(i)is int and i>=0 for i in ids)
  and capacity["admissible_new_bytes"]>=13000000000 and capacity["available_owned_RSS_bytes"]>=68719476736,"Actual8CPU/13GBstorage/64GiBRSS capacity")
 cfg=dict(manifest["condition"]["inference_template_inactive"],execution_enabled=True,observation_ref=prior["observation_ref"])
 return dict(schema="frozen_TRAIN001_full30Mb_observation_ROOT_request_v1",root_authorizer="root_agent",active=True,
  execution_authorized=True,GPU_authorized=False,parent_id="american_c3_train_001",stage="infer",phase="infer_train_001",
  fresh_unfinished_stage=True,automatic_retry=False,source_manifest_ref=MANIFEST,
  storage_policy_selector_ref=next(r for r in manifest["refs"]if r["path"].endswith("explicit_child_selector.root_active_v1.json")),
  query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
  current_Census_original_reserve_execute_authorized=True,source_review_ref=REVIEW,caps=caps,actual_capacity=capacity,
  cpu_ids=ids,scope_id=scope_id,runtime_ref=manifest["condition"]["inference_runtime_ref"],prior=prior,scientific_config=cfg,
  bindings=dict(parent_id="american_c3_train_001",scientific_config=cfg,simulation_prerequisites=prior),
  native_MLM_dependency=False,source_sha256={r["path"]:r["sha256"]for r in manifest["refs"]},
  packet_directory=OBSERVE+"/ROOT_actual_TRAIN001_"+scope_id,
  output_directory="experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_"+scope_id,
  created_at_utc=now.isoformat(),expires_at_utc=expires_at.isoformat())
def build_only(*,capacity_ref,scope_id,expires_at):
 need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,"ROOT -B -s nonoptimized metadata builder")
 manifest=read(MANIFEST);review=read(REVIEW)
 need(review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]and review["source_manifest_ref"]==MANIFEST,"Exact actual independent source pass")
 prior=actual_prior();capacity=read(capacity_ref);now=datetime.now(timezone.utc)
 q=compose_request(capacity=capacity,scope_id=scope_id,prior=prior,manifest=manifest,expires_at=expires_at,now=now)
 need(not local(q["packet_directory"]).exists()and not local(q["output_directory"]).exists(),"Fresh actual observation namespaces")
 need(ref(SOURCE["path"])==SOURCE,"Exact independently reviewed actual worker")
 sp=importlib.util.spec_from_file_location("_ROOT_observe_build_metadata_only",local(SOURCE["path"]))
 module=importlib.util.module_from_spec(sp);sys.modules[sp.name]=module;sp.loader.exec_module(module)
 # initialize/metadata validation never calls guard(), admit(), worker() or execute().
 module.initialize();module.simulation_prerequisites(prior,manifest["condition"])
 path=local(HERE)/("ROOT_actual_infer_request_"+scope_id+".json")
 data=(json.dumps(q,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();need(len(data)<=2**20,"Compact complete immutable request")
 with path.open("xb")as stream:need(stream.write(data)==len(data),"Complete exclusive request");stream.flush();os.fsync(stream.fileno())
 rr=ref(path);module.validate_request(rr)
 return dict(status="REQUEST_METADATA_VALIDATED_NOT_ADMITTED",request_ref=rr,capacity_ref=capacity_ref,
  source_manifest_ref=MANIFEST,source_review_ref=REVIEW,guard_loaded=False,admission=False,execution=False)
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--build-only",action="store_true")
 p.add_argument("--ROOT-authorizes-new-observation-request",action="store_true",dest="authorized")
 p.add_argument("--capacity-ref");p.add_argument("--scope-id");p.add_argument("--expires-at")
 a=p.parse_args(argv)
 if not a.build_only:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
 need(a.authorized and a.capacity_ref and a.scope_id and a.expires_at,"Explicit ROOT new-stage metadata authorization and genuine capacity required")
 print(json.dumps(build_only(capacity_ref=json.loads(a.capacity_ref),scope_id=a.scope_id,
  expires_at=datetime.fromisoformat(a.expires_at)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
