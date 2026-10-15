"""Close four original completed stage scopes, preserving unknown outer transport exit."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,hashlib,json,os
from research.reviews.H044_marker_resolution_root_v1.root_driver_v4 import context,load_guard,ROOT,CORE,BASE
HERE=ROOT/"research/reviews/American_C3_stage_wait_credit_audit_20261015_v1"
PIDS=[2598798,2599474,2600696,2600847,2603505]
GROUPS=[2599474,2600696,2600847,2603505]
ORDER=["010-simulate","010-infer","011-simulate","011-infer"]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 require(ref(r["path"])==r,"Pinned original metadata drift")
 return json.loads((ROOT/r["path"]).read_bytes())
def save(folder,name,v):
 p=folder/name
 with p.open("x") as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 return ref(p)
def absent_pid(pid):
 try:os.kill(pid,0)
 except ProcessLookupError:return dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat())
 raise ValueError("Present, reused or zombie PID cannot be certified")
def absent_groups():
 rows=[]
 for gid in GROUPS:
  try:os.killpg(gid,0)
  except ProcessLookupError:
   rows.append(dict(process_group=gid,absent=True,method="targeted os.killpg(group,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat()));continue
  raise ValueError("An original worker group is still present")
 return rows
def run():
 audit=read(dict(path=str((HERE/"stage_evidence_audit.json").relative_to(ROOT)),sha256="75cf5f3f374348c557a382c26af54017a9a4602c574b62fa925e6646c550484f"))
 outer=read(dict(path=str((HERE/"original_transport_command_root_OS_snapshot.json").relative_to(ROOT)),sha256="eb35c3a0ddc761af2c122dabbc64149a75c965d75b2468ea3ae2cf8695558d3e"))
 require(outer["outer_transport_exit_status"]=="unknown" and outer["no_new_launch_or_retry_performed"] is True,"Do not invent an outer exit0")
 require(audit["named_minimum_possible_server_writer_pids"]==PIDS and audit["named_worker_process_groups"]==GROUPS,"Exact independently reviewed original roster")
 directory=HERE/"root_stage_close_execution_v1";directory.mkdir()
 groups=absent_groups();observations=[absent_pid(pid) for pid in PIDS]
 revocation=save(directory,"four_scope_write_resume_revocation.json",dict(schema="C3_original_four_completed_scope_root_revocation_v1",root_authorizer="root_agent",keys=[audit["stages"][s]["key"] for s in ORDER],scope="Only original TRAIN010/011 simulate/infer output and auxiliary scopes",old_input_config_storage_source_launch_resume_retry_write_capabilities_revoked=True,read_only_reuse_preserved=True,preserve_all_files_and_charges=True,outer_transport_exit_status="unknown",original_two_parent_adapter_not_executed=True,source_roster_addendum_ref=ref(HERE/"transport_roster_addendum.md"),original_transport_command_ref=ref(HERE/"original_transport_command_root_OS_snapshot.json"),observations=observations,groups=groups,created_utc=datetime.now(timezone.utc).isoformat()))
 m=load_guard(context())
 for stage in ORDER:
  row=audit["stages"][stage];loc=row["primitive_locator_guidance"];original=row["original_evidence_refs"]
  admission=read(original["admission_ref"]);own=admission["storage"]["reservation"]
  require(own["key"]==row["key"] and own["output"]==row["output"] and own["reserved_bytes"]==row["reserved_bytes"],"Actual original reservation identity")
  launch=read(original["launch_ref"]);wait=read(original["wait_ref"]);read(original["completed_ref"])
  require(type(wait["returncode"]) is int and wait["returncode"]==0 and wait["pid"]==launch["pid"] and wait["actual_direct_worker_wait"] is True and wait["worker_process_group_drained"] is True,"Actual original direct stage wait0, not parent/transport exit")
  folder=directory/stage;folder.mkdir();groups=absent_groups();observations=[absent_pid(pid) for pid in PIDS]
  observation=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",key=own["key"],reservation_sha256=m.budget.digest(own),storage_ref=loc["storage_ref"],launch_ref=loc["launch_ref"],direct_wait_ref=loc["direct_wait_ref"],completed_ref=original["completed_ref"],launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,external_NFS_writers_outside_observation=True,pids=PIDS,observations=observations,worker_group_observations=groups,scope_revocation_ref=revocation,outer_control_result="unknown",original_two_parent_adapter_not_executed=True)
  observation_ref=save(folder,"root_observation.json",observation)
  request=dict(key=own["key"],reservation_sha256=m.budget.digest(own),**loc,launch_identity=row["launch_identity_guidance"],wait_identity=row["wait_identity_guidance"],completed_ref=original["completed_ref"],completed_identity=row["completed_identity_guidance"],observation_ref=observation_ref)
  with m.budget.v4.mutex(ROOT):
   state=m._state_locked(CORE,BASE,fresh=True);m.assert_open(own["key"],state[0],state[1]);require(state[0][own["key"]]==own,"Original open reservation unchanged")
   authority=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
  authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
  authority_ref=save(folder,"close_authority.json",authority)
  receipt=m.apply_close_completed_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
  saved=save(folder,"close_receipt.json",receipt)
  print(json.dumps(dict(stage=stage,receipt_ref=saved,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],artifacts_deleted=0,outer_exit_claim="unknown")),flush=True)
 return 0
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");a=p.parse_args()
 if not a.execute:print(json.dumps(dict(status="inactive_source_only_root_owns_execution")))
 else:raise SystemExit(run())
