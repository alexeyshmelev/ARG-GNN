"""Root completed-output credits after every old terminal writer finishes."""
from pathlib import Path
import sys
sys.path.insert(0,"/workspace-SR003.nfs2/ARG-GNN")
from datetime import datetime,timezone,timedelta
import errno,json,os,time,traceback
from arggnn import project_storage_terminal_success_scopes_v1 as m
R=m.ROOT;P=R/"research/reviews/expanded_full_panel_v1_root"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e";BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
def ref(p):return m._ref(p)
def read(p):return json.loads((R/p).read_text())
def save(packet,name,v):
 packet.mkdir(parents=True,exist_ok=True);p=packet/name
 with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True);f.write("\n")
 return ref(p)
def inspect():return m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
def pins(v):return m._pins(v)
def absent(pid):
 try:os.kill(pid,0)
 except OSError as e:
  if e.errno!=errno.ESRCH:raise
 else:return False
 return True
def authority(op):
 return dict(kind="project_budget_terminal_success_root_admission_v1",root_authorizer="root_agent",active=True,operation=op,accounting_mode=m.MODE,
  bindings=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),cpu_overlay=ref(m.old.CPU_SOURCE),terminal_overlay=ref(m.old.SOURCE),terminal_success_overlay=ref(m.SOURCE)),
  expected_before=pins(inspect()),all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_terminal_success_guard=True,
  all_terminal_v1_writers_finished_before_successor_activation=True,historical_records_mutation_authorized=False,global_rebase_authorized=False,
  scientific_execution_authorized=False,GPU_authorized=False,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=23)).isoformat())
def writer_barrier():
 packets=[P/"ap_expanded_val_006/terminal_launch_v1",P/"ap_expanded_val_007/terminal_launch_v1",P/"pyg_observation_preparation_v4"]
 for packet in packets[:2]:
  exit=packet/"launcher_exit.json"
  if not exit.exists():return None
  v=json.loads(exit.read_text())
  if v["returncode"]!=0 or v["completed_ref"] is None:raise RuntimeError("VAL producer failed")
  if not absent(v["worker_pid"]):return None
 packet=packets[2];exit=packet/"launcher_exit.json"
 if not exit.exists():return None
 v=json.loads(exit.read_text());failed=R/"data/expanded_full_panel_pyg_cache_v4/failed.json"
 stop=P/"cache_v4_superseded_for_efficiency_v1.json"
 if not failed.exists() or not stop.exists():raise RuntimeError("Missing real source failure/root optimization stop")
 f=json.loads(failed.read_text());s=json.loads(stop.read_text());launch=json.loads((packet/"launch.json").read_text())
 if (v["returncode"]!=-2 or v["completed_ref"] is not None or f["error_type"]!="KeyboardInterrupt"
  or f["input_ref"]!=launch["input_ref"] or s["pid"]!=v["worker_pid"] or s["signal"]!="SIGINT"
  or s["old_SOURCE_changed"] is not False or s["scientific_model_updates"]!=0):raise RuntimeError("Unbound/incorrect cache stop proof")
 for pid in [v["worker_pid"],*s["recorded_children"],251730,300699,322742]:
  if not absent(pid):return None
 if (R/"data/expanded_full_panel_pyg_cache_v4/worker_tmp").exists():raise RuntimeError("Owned pool scratch cleanup did not complete")
 return [ref(p/"launcher_exit.json") for p in packets]

def run():
 deadline=time.monotonic()+72000
 while True:
  barrier=writer_barrier()
  if barrier:break
  if time.monotonic()>deadline:raise TimeoutError("Real old-writer completion barrier not reached")
  time.sleep(5)
 packet=P/"terminal_completed_VAL_credits_v1"
 save(packet,"writer_barrier.json",dict(kind="terminal_v1_writer_completion_barrier_v1",completed_waits=barrier,supervisor_pid=251730,supervisor_observed_absent=True,checked_utc=datetime.now(timezone.utc).isoformat()))
 for i in range(6):
  parent=f"ap_expanded_val_{i:03d}";proof=packet/parent
  if (proof/"receipt.json").exists():continue
  folder=P/parent;launch=ref(folder/"launch.json");completion=ref(R/"data/expanded_full_panel_v1"/parent/"expanded_parent_completed.json");pid=json.loads((folder/"launch.json").read_text())["pid"]
  if not absent(pid):raise RuntimeError("Original orphan producer still present")
  obs=dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat())
  terminal=dict(kind="project_budget_CPU_completed_observed_absent_v1",root_authorizer="root_agent",status="completed_CPU_scope_observed_absent_launcher_exit_unknown",
   parent_id=parent,launch_ref=launch,completion_ref=completion,worker_pid=pid,observation=obs,launcher_exit_status_known=False,launcher_returncode=None)
  obref=save(proof,"terminal_observation.json",terminal)
  a=authority("close_completed_cpu_observed_absent");a.update(scope_write_authority_revoked_for_all_cooperative_project_jobs=True,closed_scope_future_writes_and_relaunch_revoked=True,external_NFS_writers_outside_observation=True,
   closure=dict(key="H038/"+parent,parent_id=parent,storage_ref=ref(folder/"storage.json"),completion_ref=completion,input_ref=ref(folder/"input.json"),launch_ref=launch,observation_ref=obref))
  ar=save(proof,"authority.json",a);receipt=m.apply_close_completed_cpu_observed_absent_v1(ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE);save(proof,"receipt.json",receipt)
  print(json.dumps(dict(stage="completed_VAL_allowance_closed",parent_id=parent,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],launcher_exit_status_known=False)),flush=True)
 save(packet,"completed.json",dict(completed_CPU_credits=6,writer_barrier_ref=ref(packet/"writer_barrier.json"),all_original_records_preserved=True,view=inspect()))
 cfg=read("configs/expanded_full_panel_v1.json");refs={r["parent_id"]:ref(cfg["output_root"]+"/"+r["parent_id"]+"/expanded_parent_completed.json") for r in cfg["parent_registry"] if r["split"]!="outer"}
 save(P,"all40_development_completed_v1.json",dict(completed_development_parents=40,TRAIN=32,VAL=8,completion_refs=refs,outer_generated=0,models_trained=False,top1_claim=False))
 print(json.dumps(dict(stage="successor_ready_and_all40_complete")),flush=True)
if __name__=="__main__":
 try:run()
 except Exception:print(traceback.format_exc(),flush=True);raise SystemExit(1)
