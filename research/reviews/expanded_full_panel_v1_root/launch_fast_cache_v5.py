"""Root cache v5 admission and actual direct CPU-worker wait."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,subprocess,sys,traceback
R=Path("/workspace-SR003.nfs2/ARG-GNN");P=R/"research/reviews/expanded_full_panel_v1_root"
sp=importlib.util.spec_from_file_location("root_storage",P/"retire_inactive_scopes_before_fast_cache_v1.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
def run():
 from arggnn import prepare_expanded_pyg_observations_v5 as prep
 report=prep.source_report_v1();packet=P/"pyg_observation_preparation_v5"
 packet.mkdir(exist_ok=False)
 spec=m.read("research/reviews/expanded_full_panel_v1_root/pyg_observation_preparation_v4/input.json")
 now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat();job="expanded-full-panel-pyg-preparation-v5"
 output="data/expanded_full_panel_pyg_cache_v5";cap=4000000000
 runtime=m.save(packet,"runtime.json",dict(status="active",device="cpu",GPU_authorized=False,CPU_workers=8,source_ref=report["source_ref"],created_at_utc=now.isoformat(),expires_at_utc=expiry,job_id=job,output_directory=output))
 spec.update(schema=prep.SCHEMA,job_id=job,created_at_utc=now.isoformat(),expires_at_utc=expiry,output_directory=output,output_cap_bytes=cap,runtime_ref=runtime,
  repeat_reason="Complete interrupted cache with equivalent vectorized validation and parallel RAM compression, unchanged observations/math/splits")
 for k in ["source_ref","source_sha256","config_ref","registration_ref","producer_ref","original_producer_ref","storage_adapter_ref"]:spec[k]=report[k]
 a=m.authority("reserve");bindings=list(a["bindings"].values())+[m.ref("src/arggnn/project_budget_v9_consumer.py"),runtime]
 bindings += [report[k] for k in ["source_ref","config_ref","registration_ref","producer_ref","original_producer_ref","storage_adapter_ref"]]
 bindings += [dict(path=p,sha256=h) for p,h in report["source_sha256"].items()]
 unique=[]
 for b in bindings:
  if b not in unique:unique.append(b)
 a["reservation"]=dict(key="H038/"+job,output=str(R/output),auxiliary_scopes=[],reserved_bytes=cap,consumer_bindings=unique)
 ar=m.save(packet,"reserve_authority.json",a);res=m.m.apply_reserve_v1(ar,expected_source_sha256=m.CORE,expected_launcher_sha256=m.BASE)
 m.save(packet,"reserve_receipt.json",res)
 spec["storage"]=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.m.MODE,job_id=job,output_directory=output,reserved_new_bytes=cap,reservation=res["value"],expires_at_utc=expiry,initial_ledger_pins=m.m._pins(m.inspect()),budget_components=dict(implementation=m.ref(m.m.budget.SOURCE_REL),launcher=m.ref(m.m.budget.LAUNCHER_REL),consumer_adapter=m.ref(m.m.SOURCE)))
 ir=m.save(packet,"input.json",spec)
 command=[str(R/".runtime/env/bin/python"),"-B","-m","arggnn.prepare_expanded_pyg_observations_v5","--execute","--input",ir["path"],"--expected-input-sha256",ir["sha256"]]
 with (packet/"launch.log").open("xb") as log:child=subprocess.Popen(command,cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 m.save(packet,"launch.json",dict(pid=child.pid,command=command,input_ref=ir,CPU_workers=8,GPU_used=False))
 print(json.dumps(dict(stage="fast_cache_v5_launched",pid=child.pid,input_ref=ir)),flush=True)
 rc=child.wait();done=R/output/"completed.json"
 wait=m.save(packet,"launcher_exit.json",dict(pid=child.pid,returncode=rc,completed_ref=m.ref(done) if done.exists() else None))
 print(json.dumps(dict(stage="fast_cache_v5_exited",returncode=rc,wait_ref=wait)),flush=True)
 if rc or not done.exists():raise RuntimeError("Cache preparation failed, preserved without automatic replay")
if __name__=="__main__":
 try:run()
 except Exception:print(traceback.format_exc(),flush=True);raise SystemExit(1)
