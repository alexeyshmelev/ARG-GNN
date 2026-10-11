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
PACKET=ROOT/"research/reviews/reference_dependence_gnn_v1_root"
SOURCE="src/arggnn/train_reference_dependence_gnn_v1.py"
MODEL="src/arggnn/reference_dependence_gnn_v1.py"
CONFIG="configs/reference_dependence_gnn_v1.json"
DOC="docs/reference_dependence_gnn_v1.md"
EXPECTED={
 MODEL:"53e0f3f23c0c63b104303131cf63fe9c6452729b7f8568f4419e37b1dc6341fd",
 SOURCE:"fcd7be44f02ddd36a0300d1933441dfcec5b3bc7d8162b39785dd52c16623c73",
 CONFIG:"aa1c20a2be7451fdfc96d6cb44b8264449d0bb7a6d50133a749e916d28b1e9af",
 DOC:"154c8ef6e295252c66a83757ad431a19275bca0ea01416f543258079bb9d2dff"}
SCIENCE_REVIEW="research/reviews/reference_dependence_gnn_v1_independent/receipt.json"
def frozen():
 for path,digest in EXPECTED.items():
  assert ref(path)["sha256"]==digest
 review=read(SCIENCE_REVIEW)
 assert review["verdict"]["mathematical_implementation"]=="PASS"
 assert review["verdict"]["source_execution_route"]=="PASS_source_only"
 assert review["blockers"]==[] and review["final_source_sha256"]==EXPECTED
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
 old=read(cfg["first_moment_reference_input"]["path"])
 now=datetime.now(timezone.utc);expires=(now+timedelta(hours=23)).isoformat()
 spec=dict(old)
 spec.update(schema="reference_dependence_gnn_input_v1",job_id="reference-dependence-gnn-v1",
     output_directory="experiments/results/reference_dependence_gnn_v1_seed101_GPU2",
     created_at_utc=now.isoformat(),expires_at_utc=expires,
     max_cpu_rss_bytes=128000000000,max_cuda_reserved_bytes=34359738368)
 spec["refs"]=dict(old["refs"])
 spec["refs"].update(first_moment_reference_input=cfg["first_moment_reference_input"])
 spec["refs"].pop("registration",None)
 spec["source_sha256"]={path:digest for path,digest in EXPECTED.items() if path in (SOURCE,MODEL)}
 adapter=ref("src/arggnn/project_budget_v9_consumer.py")
 registration=next(x for x in read("research/experiment_decisions.json")["decisions"] if x["id"]=="H032")
 registration_ref=save(PACKET/"registration.json",registration)
 runtime=dict(schema="H032_runtime_v1",status="active",job_id=spec["job_id"],device="cuda",
    created_at_utc=now.isoformat(),expires_at_utc=expires,output_directory=spec["output_directory"],
    training_dtype="float32",physical_gpu=2,maximum_project_physical_GPUs=2,
    process_title="AI Auto Research Pipeline Test",max_cpu_rss_bytes=spec["max_cpu_rss_bytes"],
    max_cuda_reserved_bytes=spec["max_cuda_reserved_bytes"],
    launcher=bindings["launcher"],scientific_source=ref(SOURCE),scientific_model=ref(MODEL),
    config=ref(CONFIG),independent_scientific_source_review=review_ref,
    first_tile_qualification="actual_first_TRAIN_FP32_update_included",
    H030_static_comparison_pending=True,H030_completion_dependency_for_fit=False,
    no_H030_live_checkpoint_reads=True)
 runtime_ref=save(PACKET/"runtime.json",runtime)
 view=observe_retry(b.inspect,ROOT,expected_source_sha256=core,expected_launcher_sha256=launcher)
 a=authority("reserve",view,bindings,budget_review)
 a["reservation"]=dict(key="H032/"+Path(spec["output_directory"]).name,
     output=str(ROOT/spec["output_directory"]),auxiliary_scopes=[],reserved_bytes=2000000000,
     consumer_bindings=[ref(SOURCE),ref(MODEL),ref(CONFIG),runtime_ref,adapter,
                       bindings["implementation"],bindings["launcher"]])
 admission=save(PACKET/"budget_admission.json",a)
 result=observe_retry(b.apply,ROOT,admission,expected_source_sha256=core,expected_launcher_sha256=launcher)
 save(PACKET/"budget_receipt.json",result)
 view=observe_retry(b.inspect,ROOT,expected_source_sha256=core,expected_launcher_sha256=launcher)
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
    job_id=spec["job_id"],output_directory=spec["output_directory"],reserved_new_bytes=2000000000,
    expires_at_utc=expires,reservation=result["value"],initial_ledger_pins=b.pins(view),
    budget_components=dict(implementation=bindings["implementation"],launcher=bindings["launcher"],consumer_adapter=adapter))
 storage_ref=save(PACKET/"storage.json",storage)
 spec["refs"].update(config=ref(CONFIG),runtime=runtime_ref,storage=storage_ref,registration=registration_ref)
 spec["source_sha256"].update({r["path"]:r["sha256"] for r in [adapter,bindings["implementation"],bindings["launcher"]]})
 input_ref=save(PACKET/"input.json",spec)
 print(json.dumps(dict(stage="prepared_H032",input_ref=input_ref,storage_ref=storage_ref,
    projected_bytes=view["projected_bytes"],H030_static_comparison_pending=True)),flush=True)
def launch():
 frozen()
 core,launcher,bindings,budget_review=reviewed()
 inp=ref(PACKET/"input.json");spec=b.v4.bound(ROOT,inp,True);sr=spec["refs"]["storage"]
 cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/"scripts/launch_reserved_gpu_v6.py"),
   "--gpu","2","--job",spec["job_id"],"--storage-record",sr["path"],
   "--expected-storage-sha256",sr["sha256"],"--expected-budget-source-sha256",core,
   "--expected-launcher-sha256",launcher,"arggnn.train_reference_dependence_gnn_v1",
   "--execute","--input",inp["path"],"--expected-input-sha256",inp["sha256"]]
 env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 logfile=PACKET/"launch.log"
 with logfile.open("xb") as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(PACKET/"launch.json",dict(root_authorizer="root_agent",hypothesis_id="H032",
    input_ref=inp,storage_ref=sr,launched_utc=datetime.now(timezone.utc).isoformat(),
    pid=proc.pid,physical_gpu=2,command=cmd,log_path=str(logfile.relative_to(ROOT)),
    exit_status_known=False,H030_static_comparison_pending=True))
 print(json.dumps(dict(stage="launched_H032",launcher_pid=proc.pid,physical_gpu=2)),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=["prepare","launch"]);args=parser.parse_args()
 try:
  if args.action=="prepare":prepare()
  else:launch()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
