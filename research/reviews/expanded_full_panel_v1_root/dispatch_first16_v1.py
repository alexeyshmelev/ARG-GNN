"""Prepare the first16 of the prespecified full-author TRAIN parents, eight CPU jobs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,subprocess,sys,time,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v9 as b
P=ROOT/"research/reviews/expanded_full_panel_v1_root"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
SOURCE="src/arggnn/prepare_expanded_full_panel_v1.py"
CONFIG="configs/expanded_full_panel_v1.json"
ADAPTER="src/arggnn/project_storage_closed_scopes_v1.py"
HELPER="research/reviews/closed_scopes_admission_v1_root/admission.py"
REVIEW="research/reviews/closed_scopes_admission_v1_review_author/receipt.json"
def read(p):return json.loads((ROOT/p).read_text())
def ref(p):return b.v4.reference(ROOT,p)
def save(packet,name,value):
 q=packet/name
 with q.open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write(chr(10))
 return ref(q)
def view():
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE,BASE)
  return b.state(ROOT,c,bindings,verify_closed=False)[4],bindings
def admit(parent):
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat()
 packet=P/parent;packet.mkdir(exist_ok=False)
 cfg=read(CONFIG);output=cfg["output_root"]+"/"+parent;job="expanded-full-panel-"+parent.replace("_","-")
 rt=save(packet,"runtime.json",dict(status="active",device="cpu",CUDA_VISIBLE_DEVICES="",
  parent_id=parent,job_id=job,output_directory=output,source_ref=ref(SOURCE),config_ref=ref(CONFIG),
  created_at_utc=now.isoformat(),expires_at_utc=expiry,max_project_GPUs=2,
  no_GPU_used=True,native_threads_per_parent=8,maximum_concurrent_CPU_parents=8))
 consumers=[ref(SOURCE),ref(CONFIG),rt,ref(ADAPTER),
  ref("src/arggnn/project_budget_v9_consumer.py"),ref("scripts/project_budget_v9.py"),
  ref("scripts/launch_reserved_gpu_v6.py"),ref("src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py")]
 consumers.extend(cfg["dependencies"].values())
 consumers=[x for x in consumers if isinstance(x,dict) and set(x)=={"path","sha256"}]
 template=read("research/reviews/ancestralpaths_full_panel_discovery_v1_root/ap_full_train_000/budget_admission.json")
 before,bindings=view()
 template.update(expected_before=b.pins(before),accounting_mode="journal_closed_scopes_credit_v1",
  actual_reserve_helper=ref(HELPER),full_reservation_helper_review=ref(REVIEW),
  reservation=dict(key="H038/"+parent,output=str(ROOT/output),auxiliary_scopes=[],
   reserved_bytes=4000000000,consumer_bindings=consumers))
 ar=save(packet,"budget_admission.json",template)
 sp=importlib.util.spec_from_file_location("root_credited_admit",ROOT/HELPER)
 m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 result=m.apply(ROOT,ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
 save(packet,"budget_receipt.json",result)
 return dict(parent=parent,packet=packet,runtime=rt,reservation=result["value"],job=job,output=output,expiry=expiry)
def finish_input(item,pins,bindings):
 cfg=read(CONFIG);packet=item["packet"]
 storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
  accounting_mode="journal_closed_scopes_credit_v1",job_id=item["job"],output_directory=item["output"],
  reserved_new_bytes=4000000000,reservation=item["reservation"],expires_at_utc=item["expiry"],
  initial_ledger_pins=pins,budget_components=dict(implementation=bindings["implementation"],
   launcher=bindings["launcher"],consumer_adapter=ref(ADAPTER)))
 st=save(packet,"storage.json",storage)
 from arggnn.prepare_ancestralpaths_full_panel_discovery_v1 import digest
 inp=dict(schema="expanded_full_panel_v1_root_admission",root_authorizer="root_agent",
  execution_authorized=True,status="active",parent_id=item["parent"],allowed_stages=["all"],
  job_id=item["job"],output_directory=item["output"],source_ref=ref(SOURCE),config_ref=ref(CONFIG),
  config_content_sha256=digest(cfg),parent_registry_sha256=digest(cfg["parent_registry"]),
  runtime_ref=item["runtime"],expires_at_utc=item["expiry"],storage=storage,
  storage_adapter_ref=ref(ADAPTER),outer_evaluation_authorized=False,
  registration_ref=ref(P/"registration.json"),no_scientific_modifications_to_original_generation=True)
 inp_ref=save(packet,"input.json",inp)
 item["command"]=[str(ROOT/".runtime/env/bin/python"),"-B","-m","arggnn.prepare_expanded_full_panel_v1",
  "--execute","--config",CONFIG,"--expected-config-sha256",ref(CONFIG)["sha256"],
  "--expected-source-sha256",ref(SOURCE)["sha256"],"--parent",item["parent"],"--stage","all",
  "--admission",inp_ref["path"]]
 item["input_ref"]=inp_ref
def launch(item):
 env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",PYTHONDONTWRITEBYTECODE="1")
 log=(item["packet"]/"launch.log").open("xb")
 child=subprocess.Popen(item["command"],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
  stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 log.close()
 save(item["packet"],"launch.json",dict(pid=child.pid,parent_id=item["parent"],command=item["command"],
  input_ref=item["input_ref"],launched_utc=datetime.now(timezone.utc).isoformat(),
  device="cpu",CUDA_VISIBLE_DEVICES="",returncode=None))
 print(json.dumps(dict(stage="CPU_parent_launched",parent=item["parent"],pid=child.pid)),flush=True)
 return child
def run():
 cfg=read(CONFIG)
 registry=read(str((P/"registration.json").relative_to(ROOT)))["first_generation_wave"]
 assert len(registry)==16 and all("train" in x for x in registry)
 assert ref(SOURCE)["sha256"]=="2ff091677986d5cfec2936c2acf53eeed71c957192f9f048cb47f59d3351913e"
 assert read(REVIEW)["verdict"]=="PASS" and read(REVIEW)["source"]==ref(HELPER)
 items=[admit(p) for p in registry]
 v,bindings=view()
 for item in items:finish_input(item,b.pins(v),bindings)
 queue=list(items);live={};complete=[];failed=[]
 while queue or live:
  while queue and len(live)<8:
   item=queue.pop(0);live[item["parent"]]=(item,launch(item))
  for parent,(item,child) in list(live.items()):
   code=child.poll()
   if code is None:continue
   completed=ROOT/item["output"]/"all_completion.json"
   event=dict(parent_id=parent,worker_pid=child.pid,returncode=code,
    completed_utc=datetime.now(timezone.utc).isoformat(),completed_ref=ref(completed) if completed.exists() else None,
    native_children_waited_by_original_inference_code=True,CPU_only=True)
   save(item["packet"],"launcher_exit.json",event)
   (complete if code==0 and completed.exists() else failed).append(event)
   del live[parent]
   print(json.dumps(dict(stage="CPU_parent_completed",parent=parent,returncode=code)),flush=True)
  if failed and queue:
   save(P,"first_wave_queued_jobs_not_started_after_failure.json",dict(queued_parents=[x["parent"] for x in queue],failures=failed))
   queue=[]
  if live:time.sleep(1)
 save(P,"first_wave_completed.json",dict(status="completed_first_wave" if not failed else "partial_first_wave_preserved",
  parents=complete,failures=failed,registered_total_TRAIN=32,TRAIN_completed_this_wave=len(complete),
  total_registered_VAL=8,outer_truth_scored=False,models_trained=False,
  maximum_concurrent_CPU_parents=8,GPUs_used=False))
if __name__=="__main__":
 try:run()
 except Exception:
  traceback.print_exc(file=sys.stdout);raise SystemExit(1)

