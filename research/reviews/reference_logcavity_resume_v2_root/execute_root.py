"""Root integration for H032. Scientific sources and earlier admissions stay immutable."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util,json,os,sys,subprocess,traceback,argparse
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
s=importlib.util.spec_from_file_location("reviewed_h30root",ROOT/"research/reviews/novelty_discovery_execution_v1_root/execute_root_v4.py")
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
b=m.b
ref,save,read,observe_retry,reviewed,authority=m.ref,m.save,m.read,m.observe_retry,m.reviewed,m.authority
PACKET=ROOT/"research/reviews/reference_logcavity_resume_v2_root"
SOURCE="src/arggnn/train_reference_logcavity_resume_v2.py"
MODEL="src/arggnn/reference_logcavity_gnn_v1.py"
CONFIG="configs/reference_logcavity_resume_v2.json"
DOC="docs/reference_logcavity_resume_v1.md"
EXPECTED={
 MODEL:"8970aa2e82a43b45d1ec10b2eaae6e2203dab12356a4b968bea3c65b2e21fc0b",
 SOURCE:"62c101afec53a540dfb987efdd5c06ffa53dc80517af5b119d58b24087bfc5cb",
 CONFIG:"9fb87a887fb4bafc66381b47808a72f6893709d89be3d350ba44c6a74f8cfb97",
 "src/arggnn/project_budget_v9_consumer_retry_v3.py":"dc735ee914dcddca9cd6078948225fa9652b10c4decef6f42df508798649de2b"}
SCIENCE_REVIEW="research/reviews/process_snapshot_retry_v3_independent/H030_receipt.json"
import time
from arggnn.project_budget_v9_consumer_retry_v3 import transient_process_observation
def observe_retry(function,*args,**kwargs):
 for attempt in range(16):
  try:return function(*args,**kwargs)
  except ValueError as error:
   if not transient_process_observation(error) or attempt==15:raise
   time.sleep(min(0.25*(2**attempt),2.0))

def frozen():
 for path,digest in EXPECTED.items():
  assert ref(path)["sha256"]==digest
 review=read(SCIENCE_REVIEW)
 assert review["verdict"]=="PASS"
 assert review.get("blockers",[])==[]

 # Root independently binds exact final source hashes to the final receipt.
 refs=[ref(p) for p in EXPECTED]
 encoded=json.dumps(review,sort_keys=True)
 assert all(r["path"] in encoded and r["sha256"] in encoded for r in refs)
 return ref(SCIENCE_REVIEW)
def prepare():
 review_ref=frozen()
 core,launcher,bindings,budget_review=reviewed()
 cfg=read(CONFIG)
 assert cfg["execution_guard_ready"] is True
 old=read(cfg["original_input_ref"]["path"])
 now=datetime.now(timezone.utc);expires=(now+timedelta(hours=23)).isoformat()
 spec=dict(old)
 spec.update(schema="reference_logcavity_resume_input_v1",job_id="reference-logcavity-resume-v2",
     output_directory="experiments/results/reference_logcavity_resume_v2_seed101_GPU3",
     created_at_utc=now.isoformat(),expires_at_utc=expires,
     max_cpu_rss_bytes=128000000000,max_cuda_reserved_bytes=34359738368)
 spec["refs"]=dict(old["refs"])
 spec["source_sha256"]={path:digest for path,digest in EXPECTED.items() if path in (SOURCE,MODEL)}
 adapter=ref("src/arggnn/project_budget_v9_consumer_retry_v3.py")
 registration=next(x for x in read("research/experiment_decisions.json")["decisions"] if x["id"]=="H030")
 registration_ref=ref(PACKET/"registration.json") if (PACKET/"registration.json").exists() else save(PACKET/"registration.json",registration)
 runtime=dict(schema="H030_resume_runtime_v1",status="active",job_id=spec["job_id"],device="cuda",
    created_at_utc=now.isoformat(),expires_at_utc=expires,output_directory=spec["output_directory"],
    training_dtype="float32",physical_gpu=3,maximum_project_physical_GPUs=2,
    process_title="AI Auto Research Pipeline Test",max_cpu_rss_bytes=spec["max_cpu_rss_bytes"],
    max_cuda_reserved_bytes=spec["max_cuda_reserved_bytes"],
    launcher=bindings["launcher"],scientific_source=ref(SOURCE),scientific_model=ref(MODEL),
    config=ref(CONFIG),independent_scientific_source_review=review_ref,
    first_tile_qualification="actual_first_TRAIN_FP32_update_included",
    H030_static_comparison_pending=True,H030_completion_dependency_for_fit=False,
    no_H030_live_checkpoint_reads=True)
 runtime_ref=ref(PACKET/"runtime.json") if (PACKET/"runtime.json").exists() else save(PACKET/"runtime.json",runtime)
 existing=b.v4.bound(ROOT,runtime_ref,True)
 assert existing["scientific_source"]==ref(SOURCE) and existing["config"]==ref(CONFIG) and existing["job_id"]==spec["job_id"]
 expires=existing["expires_at_utc"]
 spec.update(created_at_utc=existing["created_at_utc"],expires_at_utc=expires)
 view=observe_retry(b.inspect,ROOT,expected_source_sha256=core,expected_launcher_sha256=launcher)
 a=authority("reserve",view,bindings,budget_review)
 a["reservation"]=dict(key="H030/"+Path(spec["output_directory"]).name,
     output=str(ROOT/spec["output_directory"]),auxiliary_scopes=[],reserved_bytes=2000000000,
     consumer_bindings=[ref(SOURCE),ref(MODEL),ref(CONFIG),runtime_ref,adapter,
                       bindings["implementation"],bindings["launcher"],ref("src/arggnn/project_budget_v9_consumer.py")])
 admission=save(PACKET/"budget_admission_v2.json",a)
 result=observe_retry(b.apply,ROOT,admission,expected_source_sha256=core,expected_launcher_sha256=launcher)
 save(PACKET/"budget_receipt.json",result)
 view=observe_retry(b.inspect,ROOT,expected_source_sha256=core,expected_launcher_sha256=launcher)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
    job_id=spec["job_id"],output_directory=spec["output_directory"],reserved_new_bytes=2000000000,
    expires_at_utc=expires,reservation=result["value"],initial_ledger_pins=b.pins(view),
    budget_components=dict(implementation=bindings["implementation"],launcher=bindings["launcher"],consumer_adapter=adapter))
 storage_ref=save(PACKET/"storage.json",storage)
 spec["refs"].update(config=ref(CONFIG),runtime=runtime_ref,storage=storage_ref)
 spec["source_sha256"].update({r["path"]:r["sha256"] for r in [adapter,bindings["implementation"],bindings["launcher"]]})
 input_ref=save(PACKET/"input.json",spec)
 print(json.dumps(dict(stage="prepared_H030_resume",input_ref=input_ref,storage_ref=storage_ref,
    projected_bytes=view["projected_bytes"],H030_static_comparison_pending=True)),flush=True)
def launch():
 frozen()
 core,launcher,bindings,budget_review=reviewed()
 inp=ref(PACKET/"input.json");spec=b.v4.bound(ROOT,inp,True);sr=spec["refs"]["storage"]
 cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/"scripts/launch_reserved_gpu_v6.py"),
   "--gpu","3","--job",spec["job_id"],"--storage-record",sr["path"],
   "--expected-storage-sha256",sr["sha256"],"--expected-budget-source-sha256",core,
   "--expected-launcher-sha256",launcher,"arggnn.train_reference_logcavity_resume_v2",
   "--execute","--input",inp["path"],"--expected-input-sha256",inp["sha256"]]
 env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 logfile=PACKET/"launch.log"
 with logfile.open("xb") as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(PACKET/"launch.json",dict(root_authorizer="root_agent",hypothesis_id="H030",
    input_ref=inp,storage_ref=sr,launched_utc=datetime.now(timezone.utc).isoformat(),
    pid=proc.pid,physical_gpu=3,command=cmd,log_path=str(logfile.relative_to(ROOT)),
    exit_status_known=False,H030_static_comparison_pending=True))
 print(json.dumps(dict(stage="launched_H030_resume",launcher_pid=proc.pid,physical_gpu=3)),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=["prepare","launch"]);args=parser.parse_args()
 try:
  if args.action=="prepare":prepare()
  else:launch()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
