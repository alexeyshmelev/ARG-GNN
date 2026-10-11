from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,os,errno,traceback
from arggnn import project_storage_terminal_scopes_v1 as m
R=m.ROOT
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("x") as f:json.dump(v,f,indent=2,sort_keys=True);f.write("\n")
 return m._ref(p)
def authority(op):
 view=m.inspect_v1(expected_source_sha256=m._ref(m.budget.SOURCE_REL)["sha256"],expected_launcher_sha256=m._ref(m.budget.LAUNCHER_REL)["sha256"])
 keys=("legacy_ledger_sha256","supplemental_ledger_sha256","journal_tail_sha256","cpu_overlay_tail_sha256","terminal_overlay_tail_sha256")
 return dict(kind="project_budget_terminal_overlay_root_admission_v1",root_authorizer="root_agent",active=True,operation=op,accounting_mode=m.MODE,
 bindings=dict(implementation=m._ref(m.budget.SOURCE_REL),launcher=m._ref(m.budget.LAUNCHER_REL),cpu_overlay=m._ref(m.CPU_SOURCE),terminal_overlay=m._ref(m.SOURCE)),
 expected_before={k:view[k] for k in keys},all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_terminal_overlay_guard=True,
 historical_records_mutation_authorized=False,global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,
 expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=23)).isoformat())
def run():
 for version in (1,3):
  folder=f"research/reviews/expanded_full_panel_v1_root/pyg_observation_preparation_v{version}"
  output=f"data/expanded_full_panel_pyg_cache_v{version}"
  packet=R/f"research/reviews/expanded_full_panel_v1_root/terminal_failed_cache_credits_v1/v{version}"
  if (packet/"receipt.json").exists():continue
  storage=m._bound(m._ref(folder+"/storage.json"));launch=m._bound(m._ref(folder+"/launch.json"));admission=m._bound(m._ref(folder+"/input.json"))
  events_ref=m._ref(output+"/events.jsonl")
  events=[m.budget.v4.parse(row) for row in m.budget.v4.bound(R,events_ref).splitlines()]
  pids={launch["pid"]}
  for row in events:
   pids.update(row[field] for field in ("worker_pid","prepare_worker_pid") if field in row)
  observations=[]
  for pid in sorted(pids):
   try:os.kill(pid,0)
   except OSError as e:
    if e.errno!=errno.ESRCH:raise
   else:raise RuntimeError("Owned PID still present: "+str(pid))
   observations.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()))
  req=dict(key=storage["reservation"]["key"],storage_ref=m._ref(folder+"/storage.json"),failure_ref=m._ref(output+"/failed.json"),started_ref=m._ref(output+"/started.json"),
   input_ref=m._ref(folder+"/input.json"),launch_ref=m._ref(folder+"/launch.json"))
  drain=dict(kind="project_budget_failed_cpu_drain_receipt_v1",root_authorizer="root_agent",status="failed_CPU_preparation_workers_drained",
   evidence_method="pinned_finally_after_owned_ProcessPool_drain",input_ref=req["input_ref"],source_ref=admission["source_ref"],failure_ref=req["failure_ref"],launch_ref=req["launch_ref"],
   events_ref=events_ref,pids=sorted(pids),observations=observations,owned_process_pool_drained=True,GPU_used=False)
  req["drain_ref"]=m._ref(packet/"drain.json") if (packet/"drain.json").exists() else save(packet/"drain.json",drain)
  a=authority("close_failed_cpu");a.update(closure=req,scope_write_authority_revoked_for_all_cooperative_project_jobs=True,
   closed_scope_future_writes_and_relaunch_revoked=True,external_NFS_writers_outside_observation=True)
  ar=save(packet/"authority.json",a)
  result=m.apply_close_failed_cpu_v1(ar,expected_source_sha256=m._ref(m.budget.SOURCE_REL)["sha256"],expected_launcher_sha256=m._ref(m.budget.LAUNCHER_REL)["sha256"])
  save(packet/"receipt.json",result)
  print(json.dumps(dict(version=version,retained_bytes=result["retained_bytes"],credit_bytes=result["credit_bytes"],status=result["status"])),flush=True)
 print(json.dumps(m.inspect_v1(expected_source_sha256=m._ref(m.budget.SOURCE_REL)["sha256"],expected_launcher_sha256=m._ref(m.budget.LAUNCHER_REL)["sha256"])),flush=True)
if __name__=="__main__":
 try:run()
 except Exception:traceback.print_exc();raise
