"""Root closes original012/013 scopes after the recorded actual supervisor wait."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,fcntl,hashlib,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_root_wave_wait_012_013_20261015_v1"
BASE=ROOT/"research/reviews/American_C3_short_train2mb_wave_supervision_v2/admitted"
PARENTS=["american_c3_short_train_012","american_c3_short_train_013"]
def require(ok,message):
 if not ok:raise ValueError(message)
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(rec):
 require(ref(rec["path"])==rec,"Exact original evidence SHA required")
 return json.loads((ROOT/rec["path"]).read_bytes())
def save(folder,name,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
 require(len(raw)<=65536,"Compact root certificate required")
 p=folder/name
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
def absence(pid,group=False):
 try:
  if group:os.killpg(pid,0)
  else:os.kill(pid,0)
 except ProcessLookupError:
  return dict(**({"process_group":pid} if group else {"pid":pid}),absent=True,
      method="targeted os.killpg(group,0) ESRCH" if group else "targeted os.kill(pid,0) ESRCH",
      checked_utc=datetime.now(timezone.utc).isoformat())
 raise ValueError("Original PID/group is present, reused or zombie")
def run():
 from research.reviews.H044_marker_resolution_root_v1.root_driver_v4 import context,load_guard,CORE,BASE as LAUNCHER_SHA
 outer_ref=ref(HERE/"actual_supervisor_wait.json");outer=read(outer_ref)
 launch_ref=ref(HERE/"launch_identity.json");outer_launch=read(launch_ref)
 require(outer["actual_direct_supervisor_wait"] is True and outer["returncode"]==0
   and outer["parent_preparations_completed"]==PARENTS and len(outer["stage_wait_refs"])==4,
   "Actual successful full original two-parent supervisor wait required")
 require(outer["direct_supervisor_pid"]==outer_launch["direct_supervisor_pid"]
   and outer["wrapper_pid"]==outer_launch["wrapper_pid"]
   and outer_launch["wrapper_source_ref"]==ref(HERE/"root_direct_wait.py"),
   "Original wrapper/supervisor source and identities required")
 require(outer_launch["pins"]["driver"]==ref("research/reviews/American_C3_short_train2mb_wave_supervision_v2/root_wave_driver.py"),
   "Unchanged original wave driver required")
 pids={outer["wrapper_pid"],outer["direct_supervisor_pid"]};groups=set();rows=[]
 for parent in PARENTS:
  for stage in ("simulate","infer"):
   folder=BASE/parent/stage
   ar,lr,wr=ref(folder/"admission.json"),ref(folder/"launch.json"),ref(folder/"actual_wait.json")
   admission,launch,wait=read(ar),read(lr),read(wr)
   own=admission["storage"]["reservation"];cr=wait["completed_ref"];completed=read(cr)
   require(launch["pid"]==wait["pid"]==wait["worker_pid"] and launch["supervisor_pid"]==outer["direct_supervisor_pid"]
     and wait["supervisor_pid"]==outer["direct_supervisor_pid"] and wait["returncode"]==0
     and wait["actual_direct_worker_wait"] is True and wait["worker_process_group_drained"] is True
     and wait["operational_failure"] is None and wait["supervisor_capture_error"] is None
     and wait["terminal_log_materialization_error"] is None,"Exact original successful stage direct wait/drain required")
   require(dict(parent_id=parent,stage=stage,wait_ref=wr) in outer["stage_wait_refs"],
     "Actual supervisor wait must bind all original stage waits")
   require(read(ref(BASE/parent/"parent_preparation_completed.json"))["parent_id"]==parent,
     "Both complete original parents required")
   pids.add(wait["pid"]);groups.add(wait["worker_process_group"])
   rows.append((parent,stage,ar,lr,wr,cr,admission,launch,wait,completed,own))
 pids=sorted(pids);groups=sorted(groups)
 observed=[absence(x) for x in pids];group_observed=[absence(x,True) for x in groups]
 directory=HERE/"stage_close_execution_v1";directory.mkdir()
 revoke=save(directory,"all_four_scope_write_resume_revocation.json",dict(
   schema="C3_completed_012_013_root_revocation_v1",root_authorizer="root_agent",
   keys=[row[-1]["key"] for row in rows],old_input_config_storage_source_launch_resume_retry_write_capabilities_revoked=True,
   read_only_reuse_preserved=True,preserve_all_files=True,actual_supervisor_wait_ref=outer_ref,
   complete_original_roster_from_pinned_source=True,pids=pids,observations=observed,worker_group_observations=group_observed))
 m=load_guard(context())
 lock=ROOT/"research/reviews/American_C3_short_train2mb_wave_source_v1/scheduler/admission.lock"
 with lock.open("a+") as stream:
  fcntl.flock(stream,fcntl.LOCK_EX)
  for parent,stage,ar,lr,wr,cr,admission,launch,wait,completed,own in rows:
   folder=directory/(parent[-3:]+"-"+stage);folder.mkdir()
   observation=save(folder,"root_observation.json",dict(
      schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
      key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,launch_ref=lr,
      direct_wait_ref=wr,completed_ref=cr,launcher_pid=launch["pid"],
      all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
      completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
      external_NFS_writers_outside_observation=True,pids=pids,observations=[absence(x) for x in pids],
      worker_group_observations=[absence(x,True) for x in groups],scope_revocation_ref=revoke,
      actual_supervisor_wait_ref=outer_ref))
   launch_identity={k:launch[k] for k in ("parent_id","stage","pid","worker_pid","worker_process_group","supervisor_pid","owned_process_roster","admission_ref","no_helper_launcher")}
   wait_identity={k:wait[k] for k in ("parent_id","stage","pid","worker_pid","worker_process_group","supervisor_pid",
      "owned_process_roster","admission_ref","driver_ref","returncode","operational_failure","worker_process_group_drained",
      "group_drain_check","actual_direct_worker_wait","supervisor_capture_error","terminal_log_materialization_error",
      "root_must_observe_supervisor_exit_before_credit")}
   ci={k:completed[k] for k in (("status","parent_id","sample_unit") if stage=="simulate" else
      ("status","tsinfer_num_threads","unary_preprocessing_policy","observation_ref"))}
   request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=ar,storage_locator=["storage"],
      launch_ref=lr,launch_pid_locator=["pid"],direct_wait_ref=wr,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
      launch_identity=launch_identity,wait_identity=wait_identity,completed_ref=cr,completed_identity=ci,observation_ref=observation)
   with m.budget.v4.mutex(ROOT):
    state=m._state_locked(CORE,LAUNCHER_SHA,fresh=True);m.assert_open(own["key"],state[0],state[1])
    require(state[0][own["key"]]==own,"Original fully charged reservation unchanged")
    authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
   authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
   receipt=m.apply_close_completed_v1(save(folder,"close_authority.json",authority),expected_source_sha256=CORE,expected_launcher_sha256=LAUNCHER_SHA)
   saved=save(folder,"close_receipt.json",receipt)
   print(json.dumps(dict(receipt_ref=saved,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"])),flush=True)
 return 0
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()
 if not a.execute:print('{"status":"inactive_root_only_actual_wait_required"}')
 else:raise SystemExit(run())
