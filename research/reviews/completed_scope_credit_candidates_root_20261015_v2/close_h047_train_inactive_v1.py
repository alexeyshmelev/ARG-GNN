"""Inactive H047 TRAIN-only close capability. Root explicitly executes --close.

No process probe/guard import/write occurs at import or default mode. Root must
first consume genuine supervisor wait0 and supply exact active approval. This
helper never launches/admit/evaluates a model or scans global processes.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/completed_scope_credit_candidates_root_20261015_v2"
SOURCE=str((HERE/"close_h047_train_inactive_v1.py").relative_to(ROOT))
PACKET=ROOT/"research/reviews/H047_GPU45_cohort_risk_root_source_v1/admitted_train"
OUTPUT=ROOT/"experiments/results/H047_GPU4_cohort_risk_train_seed101_v1"
KEY="H047/H047-cohort-risk-train-seed101-GPU4"
PIDS=[2851242,2851287,2851520]
GROUPS=[2851242,2851520]
GROUP_REF={"path":"research/reviews/H047_evaluation_root_integration_v1/actual_train_root_process_groups_v1.json","sha256":"414a66b89bcfe9a906bedf23a06dd3070783ba7614d982676efb2bd34a657894"}
GUARD="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py"
GUARD_SHA="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"
POLICY="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json"
REPAIR="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT),"Canonical own metadata")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact metadata SHA")
    raw=(ROOT/record["path"]).read_bytes();require(len(raw)<=1048576,"Bounded completion/source metadata")
    return json.loads(raw)
def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Compact root evidence")
    with (folder/name).open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return ref(folder/name)
def evidence():
    groups=read(GROUP_REF)
    require(groups["root_authorizer"]=="root_agent" and [(r["pid"],r["pgid"],r["start_ticks"]) for r in groups["processes"]]
      ==[(2851242,2851242,474211099),(2851287,2851242,474211805),(2851520,2851520,474214096)],"Actual root-observed identities/groups")
    launch_ref=ref(PACKET/"launch.json");launch=read(launch_ref)
    wait_ref=ref(PACKET/"actual_wait.json");wait=read(wait_ref)
    completed_ref=ref(OUTPUT/"completed.json");done=read(completed_ref)
    storage_ref=ref(PACKET/"storage.json");storage=read(storage_ref);own=storage["reservation"]
    require(own["key"]==KEY and own["reserved_bytes"]==100000000 and own["output"]==str(OUTPUT)
      and own["auxiliary_scopes"]==[str(PACKET/"worker_aux")],"Exact original100MB complete scopes")
    require(launch["launcher_pid"]==wait["launcher_pid"]==2851287 and launch["phase"]==wait["phase"]=="train"
      and launch["physical_gpu"]==wait["physical_gpu"]==4 and wait["returncode"]==0
      and wait["actual_direct_launcher_wait"] is True and wait["completed_ref"]==completed_ref
      and wait.get("logging_error") is None and wait.get("supervisor_error") is None
      and wait.get("log_truncated") is False,"Genuine exact original direct launcher wait0 required")
    require(done["schema"]=="H047_paired_train_completion_v1" and done["status"]=="completed_fixed30pass_paired_TRAIN"
      and done["hypothesis_id"]=="H047" and done["arms"]==["original_risk","balanced_risk"]
      and done["steps"]=={"original_risk":960,"balanced_risk":960} and len(done["history"])==30
      and set(done["checkpoint_groups"])=={"5","10","15","20","25","30"}
      and all(set(v)=={"original_risk","balanced_risk"} for v in done["checkpoint_groups"].values())
      and done["VAL_truth_read"] is False and done["outer_truth_read"] is False,"Complete paired science, no partial training")
    return storage_ref,own,launch_ref,wait_ref,completed_ref,done
def close(approval_ref):
    # Required evidence is read before any signal0 probe or guard import.
    sr,own,lr,wr,cr,done=evidence();approval=read(approval_ref)
    require(approval["schema"]=="H047_TRAIN_completed_close_root_approval_v1"
      and approval["root_authorizer"]=="root_agent" and approval["active"] is True
      and approval["close_helper_ref"]==ref(SOURCE) and approval["key"]==KEY
      and approval["actual_wait_ref"]==wr and approval["completed_ref"]==cr
      and approval["actual_original_group_ref"]==GROUP_REF and approval["known_pids"]==PIDS
      and approval["known_process_group_ids"]==GROUPS
      and approval["root_consumed_actual_supervisor_wait0"] is True
      and approval["all_owned_processes_and_groups_listed"] is True
      and approval["all_original_TRAIN_scope_launch_resume_write_capabilities_revoked"] is True
      and approval["preserve_all_files_and_read_only_checkpoint_reuse"] is True,
      "Explicit actual root wait/complete roster/revocation approval")
    observations=[];group_obs=[];stamp=lambda:datetime.now(timezone.utc).isoformat()
    for pid in PIDS:
        try:os.kill(pid,0)
        except ProcessLookupError:observations.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=stamp()))
        else:raise ValueError("Known H047 process remains live/reused; no close")
    for pgid in GROUPS:
        try:os.killpg(pgid,0)
        except ProcessLookupError:group_obs.append(dict(process_group=pgid,absent=True,method="targeted os.killpg(group,0) ESRCH",checked_utc=stamp()))
        else:raise ValueError("Known H047 group remains live/reused; no close")
    require(ref(GUARD)["sha256"]==GUARD_SHA,"Frozen current-v3 primitive")
    loader=importlib.util.spec_from_file_location("_root_H047_train_close_current_v3",ROOT/GUARD)
    m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m);m.configure_v2(ref(REPAIR),policy_ref=ref(POLICY))
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    require(state[0].get(KEY)==own and KEY not in state[1],"Exact open original reservation; no replay")
    folder=HERE/"H047_TRAIN_close_root_v1";require(not folder.exists(),"Fresh root close packet; do not replay after partial/unknown apply");folder.mkdir()
    obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
      key=KEY,reservation_sha256=m.budget.digest(own),storage_ref=sr,launch_ref=lr,direct_wait_ref=wr,completed_ref=cr,
      launcher_pid=2851287,all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
      completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
      external_NFS_writers_outside_observation=True,pids=PIDS,observations=observations,
      root_owned_process_groups=GROUPS,process_group_observations=group_obs,actual_original_group_ref=GROUP_REF,
      root_approval_ref=approval_ref,read_only_checkpoint_reuse_preserved=True)
    obs_ref=save(folder,"observation.json",obs)
    request=dict(key=KEY,reservation_sha256=m.budget.digest(own),storage_ref=sr,storage_locator=[],
      launch_ref=lr,launch_pid_locator=["launcher_pid"],launch_identity=dict(launcher_pid=2851287,phase="train",physical_gpu=4),
      direct_wait_ref=wr,wait_pid_locator=["launcher_pid"],wait_exit_locator=["returncode"],
      wait_identity=dict(launcher_pid=2851287,phase="train",returncode=0,actual_direct_launcher_wait=True,completed_ref=cr),
      completed_ref=cr,completed_identity=dict(schema=done["schema"],status=done["status"],hypothesis_id="H047",steps=done["steps"]),
      observation_ref=obs_ref)
    auth=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
    auth.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(minutes=10)).isoformat())
    ar=save(folder,"authority.json",auth)
    for pgid in GROUPS:
        try:os.killpg(pgid,0)
        except ProcessLookupError:pass
        else:raise ValueError("H047 group reappeared before apply")
    receipt=m.apply_close_completed_v1(ar,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    rr=save(folder,"receipt.json",receipt)
    print(json.dumps(dict(receipt_ref=rr,credit_bytes=receipt["credit_bytes"],retained_bytes=receipt["retained_bytes"],projected_bytes=receipt["projected_bytes"])),flush=True)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--close",action="store_true")
    p.add_argument("--approval");p.add_argument("--approval-sha256");a=p.parse_args()
    if a.close:
        require(a.approval and a.approval_sha256,"Exact active root approval")
        close(dict(path=a.approval,sha256=a.approval_sha256))
    else:print(json.dumps(dict(status="inactive_source_only",key=KEY,known_pids=PIDS,known_groups=GROUPS,process_probes=0,guard_imports=0)))
if __name__=="__main__":main()
