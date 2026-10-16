"""Root-only close of the genuinely completed VAL004 simulation scope."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
B="research/reviews/American_C3_VAL004_simulate_H020v3_source_20261016_v1"
OUT=ROOT/"research/reviews/American_C3_VAL004_simulate_H020v3_completed_root_20261016_v1"
PINS={'research/reviews/American_C3_VAL004_simulate_H020v3_source_20261016_v1/admitted_simulate_H020v3_v1/admission.json': '4da7939453af10cb3231f24006b5a4086a3374d75e2a0eb1896c1e89fce3c15e', 'research/reviews/American_C3_VAL004_simulate_H020v3_source_20261016_v1/admitted_simulate_H020v3_v1/launch.json': 'ce375fe8f5cf2b55868e91010f395852e7a092ba74d43305a06d26e121ba4849', 'research/reviews/American_C3_VAL004_simulate_H020v3_source_20261016_v1/admitted_simulate_H020v3_v1/actual_wait.json': 'e863bbba41b4e4b2318f39f7390ff9aea529a7ebbea9860de00a73c3cedfbaa4', 'research/reviews/American_C3_VAL004_simulate_H020v3_source_20261016_v1/actual_tool_terminal.root_v1.json': 'a375d10ce1a60dd751e4ac135177d34cb20b18236c168077bda032e3bd9b0445', 'experiments/results/American_C3_val004_full30mb_simulate_H020v3_v1/completed.json': '752734825d20266095c65d73ac2753450453e8b4c95be226d0e7fe6b4eacf87a'}
def ref(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(record,name):
 assert ref(record["path"])==record
 sp=importlib.util.spec_from_file_location(name,ROOT/record["path"])
 m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def save(name,value):
 with (OUT/name).open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
 return ref(OUT/name)
def absent(pid,group=False):
 try:
  if group:os.killpg(pid,0)
  else:os.kill(pid,0)
 except ProcessLookupError:return dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat())
 raise RuntimeError("Exact old owned identity/group is not absent")
def main():
 assert Path.cwd()==ROOT
 for path,sha in PINS.items():assert ref(path)["sha256"]==sha
 spec=json.loads((ROOT/(B+"/admitted_simulate_H020v3_v1/admission.json")).read_text())
 launch=json.loads((ROOT/(B+"/admitted_simulate_H020v3_v1/launch.json")).read_text())
 wait=json.loads((ROOT/(B+"/admitted_simulate_H020v3_v1/actual_wait.json")).read_text())
 terminal=json.loads((ROOT/(B+"/actual_tool_terminal.root_v1.json")).read_text())
 assert terminal["actual_terminal_consumed"] is True and terminal["actual_exit_code"]==0 and terminal["actual_tool_session_id"]==44752
 assert wait["actual_direct_worker_wait"] is True and wait["returncode"]==0 and wait["owned_worker_group_ESRCH"] is True and wait["operational_failure"] is None
 storage=spec["storage"];own=storage["reservation"]
 assert own["key"]=="H041/American-C3-val004-full30mb-simulate-H020v3-v1" and own["reserved_bytes"]==6500000000
 assert wait["admission_ref"]==ref(B+"/admitted_simulate_H020v3_v1/admission.json") and wait["completed_ref"]==terminal["completed_ref"]
 assert launch["pid"]==wait["pid"]==3472153 and launch["supervisor_pid"]==3472056
 assert wait["observed_owned_pid_identities"]=={"3472153":482257854}
 pids=[3472056,3472153]
 observations=[absent(pid) for pid in pids];absent(3472153,True)
 renewal=load(dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py",sha256="491236877e8d71cae8bd7619880858ac501c57dd93e59479ca32dacacab20468"),"_VAL004_close_renewal")
 renewal.install_renewal_v1(dict(path="research/reviews/H039_marker_repair_renewal_source_20261016_v1/renewal_certificate.root_active_v1.json",sha256="66006202094d7a3ef2b29cbec28eb93f334505b72589f09d86e4b57498162db7"))
 guard_ref=dict(path="research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py",sha256="ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635")
 policy_ref=dict(path="research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json",sha256="7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582")
 policy=json.loads((ROOT/policy_ref["path"]).read_text());assert ref(policy_ref["path"])==policy_ref
 g=load(guard_ref,"_VAL004_actual_completed_close");g.configure_v2(policy["marker_repair_authority_ref"],policy_ref=policy_ref)
 assert g.MODE=="prospective_980gb_H020_v3"
 assert not OUT.exists();OUT.mkdir()
 sr=ref(B+"/admitted_simulate_H020v3_v1/admission.json");lr=ref(B+"/admitted_simulate_H020v3_v1/launch.json")
 wr=ref(B+"/admitted_simulate_H020v3_v1/actual_wait.json");cr=terminal["completed_ref"]
 observation=save("root_observation.root_active_v1.json",dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
  root_authorizer="root_agent",key=own["key"],reservation_sha256=g.budget.digest(own),storage_ref=sr,
  launch_ref=lr,direct_wait_ref=wr,completed_ref=cr,launcher_pid=3472153,
  all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
  completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
  external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
  read_only_reuse_preserved=True,actual_outer_terminal_ref=ref(B+"/actual_tool_terminal.root_v1.json"),close_source_ref=ref(__file__)))
 request=dict(key=own["key"],reservation_sha256=g.budget.digest(own),storage_ref=sr,storage_locator=["storage"],
  launch_ref=lr,launch_pid_locator=["pid"],launch_identity={"pid":3472153,"supervisor_pid":3472056,"admission_ref":sr},
  direct_wait_ref=wr,wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
  wait_identity={"pid":3472153,"actual_direct_worker_wait":True,"returncode":0,"parent_id":"american_c3_val_004","stage":"simulate","owned_worker_group_ESRCH":True,"operational_failure":None,"completed_ref":cr},
  completed_ref=cr,completed_identity={'status': 'completed_C3_simulation_reconstruction', 'parent_id': 'american_c3_val_004', 'sample_unit': 'haploid', 'sites': 229643, 'truth_intervals': 2122},
  observation_ref=observation)
 with g.budget.v4.mutex(ROOT):
  state=g._state_locked(policy["implementation_ref"]["sha256"],policy["launcher_ref"]["sha256"],fresh=True)
  g.assert_open(own["key"],state[0],state[1]);assert state[0][own["key"]]==own
  authority=g.operation_draft_v1("close_completed",request,expected_before=g._pins(state[2]),bindings=state[3])
 authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
 ar=save("close_authority.root_active_v1.json",authority)
 receipt=g.apply_close_completed_v1(ar,expected_source_sha256=policy["implementation_ref"]["sha256"],expected_launcher_sha256=policy["launcher_ref"]["sha256"])
 rr=save("close_receipt.root_actual_v1.json",receipt)
 print(json.dumps(dict(receipt_ref=rr,credit_bytes=receipt["credit_bytes"],retained_bytes=receipt["retained_bytes"],projected_bytes=receipt["projected_bytes"],files_deleted=0)),flush=True)
if __name__=="__main__":main()
