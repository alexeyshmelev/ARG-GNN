"""Root admission and real wait of final two H038 parents and observed cache."""
from pathlib import Path
import sys
sys.path.insert(0,"/workspace-SR003.nfs2/ARG-GNN")
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,subprocess,time,traceback,sys
from arggnn import project_storage_terminal_scopes_v1 as m
from arggnn import prepare_expanded_pyg_observations_v4 as prep
R=m.ROOT;P=R/"research/reviews/expanded_full_panel_v1_root"
sp=importlib.util.spec_from_file_location("root_terminal_authority",P/"close_failed_preparation_scopes_v1.py");root=importlib.util.module_from_spec(sp);sp.loader.exec_module(root)
save=root.save
def pins(v):return m._pins(v)
def read(p):return json.loads((R/p).read_text())
def reserve(packet,key,output,cap,consumers):
 a=root.authority("reserve");a["reservation"]=dict(key=key,output=str(R/output),auxiliary_scopes=[],reserved_bytes=cap,consumer_bindings=consumers)
 ar=save(packet/"reserve_authority.json",a);receipt=m.apply_reserve_v1(ar,expected_source_sha256=m._ref(m.budget.SOURCE_REL)["sha256"],expected_launcher_sha256=m._ref(m.budget.LAUNCHER_REL)["sha256"])
 save(packet/"reserve_receipt.json",receipt);return receipt["value"]
def storage(reservation,job,output,cap,expiry):
 return dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=job,output_directory=output,reserved_new_bytes=cap,reservation=reservation,expires_at_utc=expiry,
 initial_ledger_pins=pins(m.inspect_v1(expected_source_sha256=m._ref(m.budget.SOURCE_REL)["sha256"],expected_launcher_sha256=m._ref(m.budget.LAUNCHER_REL)["sha256"])),
 budget_components=dict(implementation=m._ref(m.budget.SOURCE_REL),launcher=m._ref(m.budget.LAUNCHER_REL),consumer_adapter=m._ref(m.SOURCE)))
def launch(packet,cmd,detail):
 with (packet/"launch.log").open("xb") as log:child=subprocess.Popen(cmd,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8"),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(packet/"launch.json",dict(pid=child.pid,command=cmd,device="cpu",CUDA_VISIBLE_DEVICES="",returncode=None,**detail))
 print(json.dumps(dict(stage="CPU_launched",pid=child.pid,**detail)),flush=True);return child
def run():
 cfgref=m._ref("configs/expanded_full_panel_v1.json");cfg=read(cfgref["path"]);registration=m._ref("research/reviews/expanded_full_panel_v1_root/registration.json")
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat();live=[]
 common=[m._ref(m.budget.SOURCE_REL),m._ref(m.budget.LAUNCHER_REL),m._ref(m.CPU_SOURCE),m._ref(m.SOURCE),m._ref("src/arggnn/project_budget_v9_consumer.py")]
 producer=m._ref("src/arggnn/prepare_expanded_full_panel_v1.py");original=m._ref("src/arggnn/prepare_ancestralpaths_full_panel_discovery_v1.py")
 for i in (6,7):
  parent=f"ap_expanded_val_{i:03d}";output=cfg["output_root"]+"/"+parent;packet=P/parent/"terminal_launch_v1";job="expanded-full-panel-"+parent.replace("_","-")
  rt=save(packet/"runtime.json",dict(status="active",device="cpu",CUDA_VISIBLE_DEVICES="",parent_id=parent,job_id=job,output_directory=output,source_ref=producer,config_ref=cfgref,created_at_utc=now.isoformat(),expires_at_utc=expiry,no_GPU_used=True,native_threads_per_parent=8))
  deps=cfg["dependencies"].values() if isinstance(cfg["dependencies"],dict) else cfg["dependencies"]
  consumers=common+[producer,original,cfgref,registration,rt,m._ref("src/arggnn/project_storage_closed_scopes_v1.py"),m._ref("research/reviews/expanded_full_panel_v1_root/prepare_expanded_terminal_root_v1.py")]+[dict(path=x["path"],sha256=x["sha256"]) for x in deps]
  res=reserve(packet,"H038/"+parent,output,3000000000,consumers);st=storage(res,job,output,3000000000,expiry);save(packet/"storage.json",st)
  spec=dict(schema="expanded_full_panel_v1_root_admission",root_authorizer="root_agent",execution_authorized=True,status="active",parent_id=parent,allowed_stages=["all"],job_id=job,output_directory=output,source_ref=producer,config_ref=cfgref,
   config_content_sha256=m.budget.digest(cfg),parent_registry_sha256=m.budget.digest(cfg["parent_registry"]),runtime_ref=rt,expires_at_utc=expiry,storage=st,storage_adapter_ref=m._ref(m.SOURCE),registration_ref=registration,outer_evaluation_authorized=False,no_scientific_modifications_to_original_generation=True)
  ir=save(packet/"input.json",spec)
  cmd=[str(R/".runtime/env/bin/python"),"-B","research/reviews/expanded_full_panel_v1_root/prepare_expanded_terminal_root_v1.py","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
  child=launch(packet,cmd,dict(parent_id=parent,input_ref=ir));live.append((packet,output,parent,child))
 packet=P/"pyg_observation_preparation_v4";output="data/expanded_full_panel_pyg_cache_v4";job="expanded-full-panel-pyg-preparation-v4";cap=2000000000
 report=prep.source_report_v1();rt=save(packet/"runtime.json",dict(status="active",device="cpu",CUDA_VISIBLE_DEVICES="",CPU_workers=8,job_id=job,output_directory=output,source_ref=report["source_ref"],created_at_utc=now.isoformat(),expires_at_utc=expiry))
 refs=[dict(path=p,sha256=h) for p,h in report["source_sha256"].items()]
 consumers=common+refs+[cfgref,registration,producer,original,rt]
 res=reserve(packet,"H038/expanded_observed_PyG_cache_v4",output,cap,consumers);st=storage(res,job,output,cap,expiry);save(packet/"storage.json",st)
 spec=dict(report,schema=prep.SCHEMA,root_authorizer="root_agent",execution_authorized=True,status="active",job_id=job,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,CPU_workers=8,max_cpu_rss_bytes=512000000000,
  output_cap_bytes=cap,readiness_timeout_seconds=72000,poll_seconds=2.0,GPU_authorized=False,outer_included=False,targets_read=False,runtime_ref=rt,storage=st,
  repeat_reason="Repair import temp-file race and exact original same-simulation recovery readiness. No regeneration or scientific change. Terminal unused allowance credits.")
 ir=save(packet/"input.json",spec)
 cmd=[str(R/".runtime/env/bin/python"),"-B","-m","arggnn.prepare_expanded_pyg_observations_v4","--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 child=launch(packet,cmd,dict(input_ref=ir,source_ref=report["source_ref"],job_id=job,CPU_workers=8,GPU_used=False,outer_included=False));live.append((packet,output,None,child))
 while live:
  for packet,output,parent,child in list(live):
   rc=child.poll()
   if rc is None:continue
   cp=R/output/("expanded_parent_completed.json" if parent else "completed.json")
   receipt=dict(parent_id=parent,worker_pid=child.pid,returncode=rc,completed_ref=m._ref(cp) if cp.exists() else None,native_children_waited_by_original_inference_code=True,CPU_only=True,completed_utc=datetime.now(timezone.utc).isoformat())
   save(packet/"launcher_exit.json",receipt);live.remove((packet,output,parent,child));print(json.dumps(dict(stage="CPU_exited",**receipt)),flush=True)
  if live:time.sleep(5)
if __name__=="__main__":
 try:run()
 except Exception:print(traceback.format_exc(),flush=True);raise SystemExit(1)
