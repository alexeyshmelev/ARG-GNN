"""Root H033 scheduling, preserving frozen model/data/scorer and existing leases."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,importlib.util,json,os,subprocess,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v9 as b
PACKET=ROOT/"research/reviews/interleaved_tree_tv_gnn_v1_root"
SOURCE="src/arggnn/train_interleaved_tree_tv_gnn_v1.py"
MODEL="src/arggnn/interleaved_tree_tv_gnn_v1.py"
CONFIG="configs/interleaved_tree_tv_gnn_v1.json"
LAUNCHER="scripts/launch_reserved_gpu_full_storage_v1.py"
ADAPTER="src/arggnn/project_storage_full_reservation_v1.py"
HELPER="research/reviews/full_reservation_admission_v1_root/admission.py"
HELPER_REVIEW="research/reviews/full_reservation_admission_v1_independent/receipt.json"
SCIENCE_REVIEW="research/reviews/interleaved_tree_tv_gnn_v1_independent/receipt.json"
EXPECTED={MODEL:"c173280d1206bb89e99997079234920bc93bc51b2db0a6b674d0b9c173552d48",
 SOURCE:"ba7964aeec1327f9a3adee8eeb362d29af09230e0794f6a64b8f3421f92c788b",
 CONFIG:"a37836ba5768ed62b61378d8e184e3d4d624dd56e8e75bec73fe838dd2daf32a",
 LAUNCHER:"d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff",
 ADAPTER:"c0fdace58e26020c8227606e9b77325bc7308e7ec6691fe3a7e7c18cd4846105"}
CORE_SHA="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE_LAUNCHER_SHA="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
def ref(p):return b.v4.reference(ROOT,p)
def read(p):return json.loads((ROOT/p).read_text())
def save(p,v):
 p=Path(p)
 if not p.is_absolute():p=ROOT/p
 with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
 return ref(p)
def frozen():
 for path,digest in EXPECTED.items():assert ref(path)["sha256"]==digest,path
 review=read(SCIENCE_REVIEW)
 assert review["verdict"]=="PASS" and not review.get("blockers",[])
 encoded=json.dumps(review,sort_keys=True)
 for p in (SOURCE,MODEL,CONFIG):
  assert p in encoded and EXPECTED[p] in encoded,p
 return ref(SCIENCE_REVIEW)
def view():
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE_SHA,BASE_LAUNCHER_SHA)
  return b.state(ROOT,c,bindings,verify_closed=False)[4],bindings
def prepare():
 science=frozen()
 v,bindings=view()
 prior=read("research/reviews/reference_dependence_full_storage_v1_root/input.json")
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat()
 spec=dict(prior,schema="interleaved_tree_tv_gnn_input_v1",
  job_id="interleaved-tree-tv-gnn-v1",output_directory="experiments/results/interleaved_tree_tv_gnn_v1_seed101_GPU2",
  created_at_utc=now.isoformat(),expires_at_utc=expiry,max_cpu_rss_bytes=128000000000,max_cuda_reserved_bytes=34359738368)
 spec["refs"]=dict(prior["refs"]);spec["refs"].pop("first_moment_reference_input",None)
 spec["refs"]["registration"]=ref(PACKET/"registration.json")
 spec["source_sha256"]={p:EXPECTED[p] for p in (SOURCE,MODEL,ADAPTER,LAUNCHER)}
 spec["source_sha256"].update({"scripts/project_budget_v9.py":CORE_SHA,"scripts/launch_reserved_gpu_v6.py":BASE_LAUNCHER_SHA})
 runtime=save(PACKET/"runtime.json",dict(schema="H033_runtime_v1",status="active",
  job_id=spec["job_id"],device="cuda",physical_gpu=2,maximum_project_physical_GPUs=2,
  process_title="AI Auto Research Pipeline Test",training_dtype="float32",
  created_at_utc=now.isoformat(),expires_at_utc=expiry,output_directory=spec["output_directory"],
  scientific_source=ref(SOURCE),scientific_model=ref(MODEL),config=ref(CONFIG),
  launcher=ref(LAUNCHER),base_budget_launcher=bindings["launcher"],
  independent_scientific_source_review=science,first_actual_optimizer_update_included_in160=True,
  max_cpu_rss_bytes=spec["max_cpu_rss_bytes"],max_cuda_reserved_bytes=spec["max_cuda_reserved_bytes"]))
 authority=read("research/reviews/reference_logcavity_resume_full_storage_v1_root/budget_admission_full_charge_v3.json")
 authority.update(expected_before=b.pins(v),accounting_mode="full_reserved_no_closed_scope_credit",
  actual_reserve_helper=ref(HELPER),full_reservation_helper_review=ref(HELPER_REVIEW))
 authority["reservation"]=dict(key="H033/"+Path(spec["output_directory"]).name,output=str(ROOT/spec["output_directory"]),
  auxiliary_scopes=[],reserved_bytes=2000000000,
  consumer_bindings=[ref(SOURCE),ref(MODEL),ref(CONFIG),runtime,ref(ADAPTER),ref(LAUNCHER),
   bindings["implementation"],bindings["launcher"],ref("src/arggnn/project_budget_v9_consumer.py")])
 admission=save(PACKET/"budget_admission.json",authority)
 loader=importlib.util.spec_from_file_location("root_reserve_helper",ROOT/HELPER)
 helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
 result=helper.apply(ROOT,admission,expected_source_sha256=CORE_SHA,expected_launcher_sha256=BASE_LAUNCHER_SHA)
 save(PACKET/"budget_receipt.json",result)
 v,_=view()
 storage=save(PACKET/"storage.json",dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
  accounting_mode="full_reserved_no_closed_scope_credit",actual_launcher=ref(LAUNCHER),job_id=spec["job_id"],
  output_directory=spec["output_directory"],reserved_new_bytes=2000000000,expires_at_utc=expiry,
  reservation=result["value"],initial_ledger_pins=b.pins(v),
  budget_components=dict(implementation=bindings["implementation"],launcher=bindings["launcher"],consumer_adapter=ref(ADAPTER))))
 spec["refs"].update(config=ref(CONFIG),runtime=runtime,storage=storage)
 inp=save(PACKET/"input.json",spec)
 print(json.dumps(dict(stage="H033_prepared",input_ref=inp,storage_ref=storage,full_charge=result["full_reserved_bytes"])),flush=True)
def launch():
 frozen()
 inp=ref(PACKET/"input.json");spec=read(inp["path"]);storage=spec["refs"]["storage"]
 cmd=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/LAUNCHER),"--gpu","2","--job",spec["job_id"],
  "--storage-record",storage["path"],"--expected-storage-sha256",storage["sha256"],
  "--expected-budget-source-sha256",CORE_SHA,"--expected-launcher-sha256",BASE_LAUNCHER_SHA,
  "arggnn.train_interleaved_tree_tv_gnn_v1","--execute","--input",inp["path"],"--expected-input-sha256",inp["sha256"]]
 env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 with (PACKET/"launch.log").open("xb") as f:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save(PACKET/"launch.json",dict(hypothesis_id="H033",input_ref=inp,storage_ref=storage,pid=child.pid,physical_gpu=2,
  launched_utc=datetime.now(timezone.utc).isoformat(),command=cmd,exit_status_known=False,
  log_path=str((PACKET/"launch.log").relative_to(ROOT))))
 print(json.dumps(dict(stage="H033_launched",launcher_pid=child.pid,physical_gpu=2)),flush=True)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=("prepare","launch"));args=parser.parse_args()
 try:prepare() if args.action=="prepare" else launch()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
