"""Finalize only our completed job after its recorded processes released the GPU lease.
This preserves old outputs and pinned budget sources. It makes no claim about
other users' descriptors. Cooperative project ownership revokes future writes.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, fcntl, json, os, subprocess, sys, traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0,str(ROOT))
from scripts import project_budget_v9 as b
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
def save(packet,name,value):
 p=packet/name
 with p.open("x") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n")
 return b.v4.reference(ROOT,p)
def run(key):
 packet=ROOT/"research/reviews/completed_job_reservation_finalization_v1"/key.split("/")[0]
 packet.mkdir(parents=True,exist_ok=False)
 with b.v4.mutex(ROOT):
  c,bindings=b.reviewed(ROOT,CORE,BASE)
  _,old,new,closed,before,_=b.state(ROOT,c,bindings,verify_closed=False)
  own=old["reservations"].get(key) or new.get(key)
  if own is None or key in closed: raise ValueError("Exact open project reservation required")
  paths=b.successor_paths(ROOT,own,True) if key in new else b.v4.scope_spec(ROOT,own,True)
  completed=b.v4.reference(ROOT,paths[0]/"completed.json")
  obj=b.v4.bound(ROOT,completed,True)
  if not str(obj.get("status","")).startswith("completed"):raise ValueError("Completed job required")
  rt=obj["runtime"];lease=rt["lease"]
  pids=[rt["pid"],lease["launcher_metadata"]["launcher_pid"]]
  if any(Path("/proc",str(pid)).exists() for pid in pids):raise ValueError("Recorded project process remains")
  process=save(packet,"process_observation.json",dict(recorded_pids=pids,all_recorded_pids_absent=True,global_process_inventory=False))
  lease_path=Path(lease["lease_path"])
  fd=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   st=os.fstat(fd)
   if (st.st_dev,st.st_ino)!=(lease["lease_device"],lease["lease_inode"]):raise ValueError("Original lease identity changed")
   lease_obs=save(packet,"lease_observation.json",dict(path=str(lease_path),device=st.st_dev,inode=st.st_ino,exclusive_lock_acquired=True))
   gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid","--format=csv,noheader,nounits"],text=True)
   active=[int(line.split(",")[0]) for line in gpu.splitlines() if line.strip()]
   if any(pid in active for pid in pids):raise ValueError("Recorded GPU context remains")
   gpu_obs=save(packet,"gpu_observation.json",dict(recorded_pids=pids,no_recorded_gpu_context=True,query=gpu))
   terminal=save(packet,"terminal_receipt.json",dict(kind="project_budget_v9_terminal_receipt",root_authorizer="root_agent",
    reservation_sha256=b.digest(own),completed=completed,pids=pids,observed_utc=datetime.now(timezone.utc).isoformat(),
    recorded_pids_absent=True,original_lease_exclusive_acquire_release_verified=True,no_recorded_gpu_context=True,
    external_NFS_writers_outside_observation=True,lease_path=str(lease_path),
    original_lease_identity=dict(device=st.st_dev,inode=st.st_ino),process_observation=process,lease_observation=lease_obs,gpu_observation=gpu_obs,
    evidence_method="root_observed_absence",status="driver_completed_launcher_exit_unknown",returncode=None,exit_status_known=False))
   measurement=b.measure(ROOT,paths)
   closure=dict(key=key,reservation_sha256=b.digest(own),completed=completed,
    completed_identity={k:obj[k] for k in ["status","hypothesis_id","input_ref"]},
    terminal_receipt=terminal,scope_paths=[str(p) for p in paths],expected_scope_measurement=measurement)
   authority=json.loads((ROOT/"research/reviews/project_budget_v9_H025_close_root/budget_admission.json").read_text())
   authority.update(expected_before=b.pins(before),closure=closure)
   authority_ref=save(packet,"budget_admission.json",authority)
   b.check_authority(ROOT,authority_ref,"close",bindings)
   b.terminal_binding(ROOT,closure,own,paths)
   if b.measure(ROOT,paths)!=measurement:raise ValueError("Own completed output changed")
   writer_check=dict(method="root_completed_project_job_ownership",terminal_pids=pids,recorded_pids_absent=True,
    scope_write_authority_revoked_for_all_cooperative_project_jobs=True,
    closed_scope_future_writes_and_relaunch_revoked=True,original_lease_exclusive=True,
    global_process_inspection_performed=False,external_NFS_writers_outside_observation=True,
    unrelated_process_descriptors_inspected=False)
   value=dict(request=closure,reservation_sha256=b.digest(own),measurement=measurement,writer_check=writer_check)
   after=b.reconcile({**before,"projected_bytes":before["v4_projected_bytes"]},old["reservations"],new,{**closed,key:value})
   event=dict(schema="project_budget_v9_event",root=str(ROOT),sequence=before["journal_events"]+1,
    previous_sha256=before["journal_tail_sha256"],bindings=bindings,operation="close",authority=authority_ref,value=value)
   sha=b.append_event(ROOT,event,before,c,bindings)
   result=dict(key=key,retained_output_bytes=measurement["conservative_scope_bytes"],
    reservation_released_bytes=own["reserved_bytes"]-measurement["conservative_scope_bytes"],
    outputs_preserved=True,scientific_sources_changed=False,global_process_inspection_performed=False,
    projected_bytes=after["projected_bytes"],journal_tail_sha256=sha,
    full_charge_consumers_still_charge_original_reservation=True)
   save(packet,"receipt.json",result);print(json.dumps(result),flush=True)
  finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("key")
 try:run(parser.parse_args().key)
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
