from pathlib import Path
import importlib.util,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
s=importlib.util.spec_from_file_location("reviewed_h30root",ROOT/"research/reviews/novelty_discovery_execution_v1_root/execute_root_v4.py")
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
globals().update({k:v for k,v in vars(m).items() if not k.startswith("__")})
CLOSE=ROOT/"research/reviews/project_budget_v9_H025_close_root"
import time
def observe_retry(function,*args,**kwargs):
 for attempt in range(16):
  try:return function(*args,**kwargs)
  except ValueError as error:
   if str(error) not in {"Process inventory changed during writer inspection","Process descriptors changed during writer inspection"} or attempt==15:raise
   time.sleep(min(0.25*(2**attempt),2.0))

def close_h025():
 source,launcher,bindings,review_ref=reviewed()
 assert not (ROOT/".runtime/storage/project_budget_v6").exists()
 assert not (ROOT/".runtime/storage/project_budget_v7").exists()
 assert not (ROOT/".runtime/storage/project_budget_v8").exists()
 view=observe_retry(b.inspect,ROOT,expected_source_sha256=source,expected_launcher_sha256=launcher)
 old=read(str(b.v4.SUPPLEMENTAL))["reservations"]
 own=old["H025/persistence_trace_pilot_v1_seed101_GPU3"]
 paths=b.v4.scope_spec(ROOT,own,True)
 completed_ref=ref(paths[0]/"completed.json")
 completed=b.v4.bound(ROOT,completed_ref,True);runtime=completed["runtime"];lease=runtime["lease"]
 terminal_pids=[runtime["pid"],lease["launcher_metadata"]["launcher_pid"]]
 assert len(set(terminal_pids))==2 and all(not Path("/proc",str(x)).exists() for x in terminal_pids)
 now=datetime.now(timezone.utc).isoformat()
 process=save(CLOSE/"process_observation.json",dict(observed_utc=now,recorded_pids=terminal_pids,
       recorded_pids_absent=True,launcher_exit_status_known=False))
 lease_path=Path(lease["lease_path"]);s=lease_path.lstat()
 expected=dict(device=lease["lease_device"],inode=lease["lease_inode"])
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and expected==dict(device=s.st_dev,inode=s.st_ino)
 fd=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  current=lease_path.lstat();opened=os.fstat(fd)
  assert (s.st_dev,s.st_ino)==(current.st_dev,current.st_ino)==(opened.st_dev,opened.st_ino)
 finally:os.close(fd)
 lo=save(CLOSE/"lease_observation.json",dict(observed_utc=now,path=str(lease_path),
       original_lease_identity=expected,exclusive_nonblocking_acquire_release_verified=True,lease_bytes_modified=False))
 proc=subprocess.run(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
       "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
 rows=[]
 for line in proc.stdout.splitlines():
  x=[z.strip() for z in line.split(",")]
  rows.append(dict(pid=int(x[0]),gpu_uuid=x[1],process_name=x[2],used_gpu_memory_mib=x[3]))
 assert all(row["pid"] not in terminal_pids for row in rows)
 go=save(CLOSE/"gpu_observation.json",dict(observed_utc=now,recorded_pids=terminal_pids,
       no_recorded_gpu_context=True,compute_apps=rows,nvidia_smi_returncode=proc.returncode))
 terminal=save(CLOSE/"terminal_receipt.json",dict(kind="project_budget_v9_terminal_receipt",
    root_authorizer="root_agent",reservation_sha256=b.digest(own),completed=completed_ref,pids=terminal_pids,
    observed_utc=now,recorded_pids_absent=True,original_lease_exclusive_acquire_release_verified=True,
    no_recorded_gpu_context=True,external_NFS_writers_outside_observation=True,lease_path=str(lease_path),
    original_lease_identity=expected,process_observation=process,lease_observation=lo,gpu_observation=go,
    evidence_method="root_observed_absence",status="driver_completed_launcher_exit_unknown",returncode=None,
    exit_status_known=False))
 measurement=b.measure(ROOT,paths)
 a=authority("close",view,bindings,review_ref)
 a.update(scope_write_authority_revoked_for_all_cooperative_project_jobs=True,
      closed_scope_future_writes_and_relaunch_revoked=True,external_NFS_writers_outside_observation=True,
      opaque_credential_daemons=ref(CLOSE/"opaque_credential_daemons.json"),
      closure=dict(key=own["key"],reservation_sha256=b.digest(own),completed=completed_ref,
      completed_identity={k:completed[k] for k in ["status","hypothesis_id","seed","config_ref","input_ref"]},
      terminal_receipt=terminal,scope_paths=[str(p) for p in paths],expected_scope_measurement=measurement))
 ar=save(CLOSE/"budget_admission.json",a)
 result=observe_retry(b.apply,ROOT,ar,expected_source_sha256=source,expected_launcher_sha256=launcher)
 save(CLOSE/"apply_receipt.json",result)
 print(json.dumps(dict(stage="H025_retained_readonly",retained_bytes=measurement["conservative_scope_bytes"],
       projected_bytes=result["projected_bytes"],journal_tail_sha256=result["journal_tail_sha256"])),flush=True)

if __name__=="__main__":
 try:close_h025()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
