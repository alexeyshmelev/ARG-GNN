"""Root-only CPU execution of already-admitted original30Mb VAL stages."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,selectors,subprocess,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.American_C3_VAL003_simulate_H020v3_source_20261016_v1 import root_packet_v1 as packet
packet.install_renewal_v1()
from research.reviews.American_C3_native_paths_root_v1 import root_native_paths_driver as old
LOG_CAP=16*1048576
def ref(p):return packet.ref(p)
def save(folder,name,value):
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
 assert len(raw)<65536
 with (folder/name).open("xb") as f:
  f.write(raw)
  f.flush()
  os.fsync(f.fileno())
 return ref(folder/name)
def main(stage,parent):
 packet.select_parent(parent)
 folder,output,job=packet.identity(stage)
 admitted=packet.read(ref(folder/"admitted.json"))
 admission_ref=admitted["admission_ref"]
 spec=packet.read(admission_ref)
 plan=packet.command(stage,admission_ref)
 assert not (ROOT/output).exists() and not (folder/"launch.json").exists()
 m=packet.load_guard_v1(consumer=True)
 assert spec["IO_supervisor_ref"]==ref(__file__)
 storage=spec["storage"]
 m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+65536)
 # Record the separately pinned bootstrap and supervisor already bound at admission.
 save(folder,"root_execution_authority.json",dict(root_authorizer="root_agent",active=True,
  actual_admission_ref=admission_ref,actual_IO_supervisor_ref=ref(__file__),
  existing_owned_RSS_wait_helper_ref=ref(old.__file__),producer_ref=spec["source_ref"],
  storage_unchanged=True,scientific_module_arguments_and_config_unchanged=True,
   operational_memo_bootstrap_added=True,original_scientific_argv=plan["original_scientific_argv"],
   actual_wrapper_argv=plan["argv"],
  new_reservation_or_cap_change=False,CPU_only=True,private_work_cap_bytes=134217728,
  log_RAM_cap_bytes=LOG_CAP,wall_deadline_seconds=plan["deadlines_seconds"],
  created_utc=datetime.now(timezone.utc).isoformat()))
 for d in plan["create_before_launch"]:
  p=Path(d)
  assert p.resolve()==p and p.is_relative_to(ROOT)
  p.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy()
 env.update(plan["environment"])
 started=time.monotonic()
 proc=subprocess.Popen(plan["argv"],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
 launch_ref=save(folder,"launch.json",dict(pid=proc.pid,worker_pid=proc.pid,process_group=proc.pid,
  command=plan["argv"],admission_ref=admission_ref,supervisor_pid=os.getpid(),actual_wait_pending=True,
  IO_supervisor_ref=ref(__file__)))
 print(json.dumps(dict(event="original_full30Mb_VAL_CPU_worker_started",stage=stage,pid=proc.pid,launch_ref=launch_ref)),flush=True)
 known={proc.pid:None}
 selector=selectors.DefaultSelector()
 selector.register(proc.stdout,selectors.EVENT_READ)
 log=bytearray()
 received=0
 peak=0
 last_resource=0.
 failure=None
 term_at=None
 while proc.poll() is None:
  now=time.monotonic()
  if now-last_resource>=1:
   try:
    rss=old.owned_group_RSS_v1(proc.pid,known)
    if isinstance(rss,tuple):rss=rss[0]
    peak=max(peak,rss)
    assert rss<=68719476736,"Owned CPU RSS exceeded64GiB"
    assert now-started<=plan["deadlines_seconds"],"Owned CPU wall deadline exceeded"
    assert datetime.now(timezone.utc)<datetime.fromisoformat(spec["expires_at_utc"]),"Admission expired"
    scratch=m.budget.v4.measure_scopes(ROOT,[Path(plan["create_before_launch"][-1])])["conservative_scope_bytes"]
    assert scratch<=134217728,"Private-work cap exceeded"
   except Exception as exc:
    failure=failure or type(exc).__name__+": "+str(exc)
   last_resource=now
  if failure is not None and term_at is None:
   proc.terminate()
   term_at=now
  if term_at is not None and now-term_at>30:
   os.killpg(proc.pid,9)
  for key,_ in selector.select(.25):
   chunk=os.read(key.fd,65536)
   if chunk:
    received+=len(chunk)
    log.extend(chunk[:max(0,LOG_CAP-len(log))])
   else:selector.unregister(key.fileobj)
 rc=proc.wait()
 # First terminal evidence precedes all guard, namespace and log postprocessing.
 minimal=save(folder,"actual_wait_minimal.json",dict(actual_direct_worker_wait=True,pid=proc.pid,
  returncode=rc,process_group=proc.pid,stage=stage,admission_ref=admission_ref,waited_utc=datetime.now(timezone.utc).isoformat()))
 print(json.dumps(dict(event="original_full30Mb_VAL_actual_direct_wait",pid=proc.pid,returncode=rc,minimal_ref=minimal)),flush=True)
 rc2,drained,cleanup=old.c.previous.wait_owned_worker_v1(proc,failure is not None)
 assert rc2==rc
 for _ in range(32):
  ready=selector.select(0)
  if not ready:break
  for key,_ in ready:
   chunk=os.read(key.fd,65536)
   if chunk:
    received+=len(chunk)
    log.extend(chunk[:max(0,LOG_CAP-len(log))])
   else:selector.unregister(key.fileobj)
 selector.close()
 proc.stdout.close()
 if not drained or cleanup:failure=failure or "Owned group drain unresolved: "+str(cleanup)
 with (folder/"worker_aux"/"worker_stdout.log").open("xb") as f:
  f.write(log)
  f.flush()
  os.fsync(f.fileno())
 completed=ROOT/output/"completed.json"
 try:m.budget_check_v9(storage,boundary=True)
 except Exception as exc:failure=failure or type(exc).__name__+": "+str(exc)
 receipt=dict(actual_direct_worker_wait=True,pid=proc.pid,returncode=rc,stage=stage,parent_id=packet.PARENT,
  admission_ref=admission_ref,launch_ref=launch_ref,minimal_wait_ref=minimal,
  completed_ref=ref(completed) if completed.exists() else None,owned_worker_group_ESRCH=drained,
  operational_failure=failure,observed_owned_pid_identities=known,
  observed_peak_worker_group_RSS_bytes=peak,log_ref=ref(folder/"worker_aux"/"worker_stdout.log"),
  stdout_bytes_received=received,stdout_bytes_discarded=received-len(log),
  elapsed_seconds=time.monotonic()-started,root_IO_source_ref=ref(__file__))
 save(folder,"actual_wait.json",receipt)
 print(json.dumps(dict(event="original_full30Mb_VAL_CPU_worker_terminal",stage=stage,returncode=rc,
  operational_failure=failure,completed_ref=receipt["completed_ref"])),flush=True)
 return 0 if rc==0 and failure is None else 1
if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--launch",choices=packet.STAGES,required=True)
 p.add_argument("--parent",choices=packet.PARENTS,required=True)
 args=p.parse_args()
 raise SystemExit(main(args.launch,args.parent))
