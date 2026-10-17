"""Narrow original ordinary completed close of the separate native v12 allocation.

Default is inactive. Historical supervisor outer70445=1 and its operational
failure remain true. This never closes original never-credit H041/native MLM or
the stopped native32 scope, never asserts full MLM wrapper fit, never normalizes
or deletes files, and never loads checkpoint/scientific payloads.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,sys
from . import contract_v1 as c
HERE=c.HERE
SOURCE=HERE+"/native_MLM_v12_close_ROOT_v1.py"
MANIFEST=HERE+"/native_MLM_v12_close_manifest.v1.json"
PREFIX="research/reviews/native_VAL000_fresh_continuation_ROOT_v12_root01_oct16/"
KEY="H041/native-VAL000-verified22000-continuation-v12_root01_oct16"
OUTPUT="experiments/results/native_VAL000_verified22000_continuation_v12_root01_oct16"
AUX=PREFIX+"worker_aux"
CAP=4563402752
def R(path,sha):return dict(path=path,sha256=sha)
P=dict(
 storage=R(PREFIX+"storage.json","45d597341d3e05b51ff651b2d102c9bcfd740a4a2b84bea12c0f060757580b9f"),
 launch=R(AUX+"/launch.json","49bdb1b7a091ea299a07a4e3dd71ebb2d2e05b1bc947f7b97a493726296e48cd"),
 wait=R(AUX+"/actual_wait.json","fe35274be6a945a134535634f92647d36e6f2600824c05d39233d617c226565e"),
 minimal=R(AUX+"/actual_wait_minimal.json","1f3f56a26bcda5a3efac5d41c10aeb6769848104624de17fb8d7f5f1c9430a28"),
 completed=R(OUTPUT+"/completed.json","99f575d18d4c0728339c1fb7778f784fa6ce81e0e0b8492fc897a8251e782bc4"),
 outer=R("research/reviews/research_execution_ROOT_20261017_v19/native70445_actual_outer_terminal_consumed.root_v1.json","1eaa378c5650a7c72423f9fba4b4b2603957c2a4e30f5b21b3db47bc24fa6371"),
 acceptance=R("research/reviews/research_execution_ROOT_20261017_v34/native_saved_MLM_artifact_acceptance.root_active_v1.json","a74bcce6bab94e01d367a7e32795e45942d9639bfd56bf637fb3ac634d0dec8e"),
 input=R("research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/input.json","9550f5b0a33b82b329e483796683d400e7053cbd9bb84319c9b6e00d883436e5"))
GUARD=c.GUARD
POLICY=R("research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json","5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
SELECTOR=R("research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json","fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
COORDINATOR=R("research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py","2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73")
EXPECTED_FAILURE="ValueError: Actual native worker current identity"
ADMIN_CAP=384*1024

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def _save(directory,name,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    rows=list(directory.iterdir())
    c.require(len(data)<=65536 and len(rows)<12 and all(p.is_file()and not p.is_symlink()for p in rows)
        and 4096+sum(p.stat().st_size+4096 for p in rows)+len(data)+4096<=ADMIN_CAP
        and ADMIN_CAP+65536+4096<=524288,"Bounded own close metadata within unchanged original event allowance")
    with(directory/name).open("xb")as stream:
        c.require(stream.write(data)==len(data),"Complete exclusive close record");stream.flush();os.fsync(stream.fileno())
    return c.ref(str((directory/name).relative_to(c.ROOT)))

def metadata_v1():
    storage=c.read(P["storage"]);own=storage["reservation"];launch=c.read(P["launch"]);wait=c.read(P["wait"])
    minimal=c.read(P["minimal"]);done=c.read(P["completed"]);outer=c.read(P["outer"]);accepted=c.read(P["acceptance"])
    c.raw(P["input"])
    c.require(own["key"]==KEY and own["reserved_bytes"]==storage["reserved_new_bytes"]==CAP
        and own["output"]==str(c.local(OUTPUT))and own["auxiliary_scopes"]==[str(c.local(AUX))]
        and storage["output_directory"]==OUTPUT and 1<=len(own["consumer_bindings"])<=64
        and c.local(P["completed"]["path"])==c.local(OUTPUT)/"completed.json",
        "Exact separate original v12 reservation/output/AUX; never substitute for H041 full-fit or native32")
    c.require(launch["schema"]=="native32_fixed_full_parent_actual_child_launch_v1"
        and launch["input_ref"]==wait["input_ref"]==minimal["input_ref"]==P["input"]
        and launch["pid"]==wait["pid"]==minimal["pid"]==3752065 and launch["parent_pid"]==3750912
        and wait["original_storage_ref"]==minimal["original_storage_ref"]==P["storage"]
        and wait["completed_ref"]==P["completed"]and wait["actual_wait_minimal_ref"]==P["minimal"]
        and wait["schema"]=="native32_fixed_full_parent_actual_wait_v2"
        and type(wait["returncode"])is int and wait["returncode"]==0
        and type(wait["actual_OS_wait_status"])is int and wait["actual_OS_wait_status"]==0
        and wait["actual_direct_worker_wait"]is True
        and wait["all_owned_processes_and_groups_drained"]is True and wait["pipe_EOF_observed"]is True
        and wait["owned_identity_absence"]is True and wait["cleanup_errors"]==[]
        and wait["complete_owned_roster_known"]is True and wait["adopted_waits_truncated"]is False
        and wait["operational_failure"]==minimal["operational_failure"]==EXPECTED_FAILURE
        and all(type(r["returncode"])is int and r["returncode"]==0 for r in wait["adopted_waits"]),
        "Genuine exact direct worker raw0/return0/full drain; historical supervisor operational failure preserved")
    c.require(outer["terminal_consumed"]is True and outer["tool_session_id"]==70445
        and type(outer["actual_tool_terminal_exit_code"])is int and outer["actual_tool_terminal_exit_code"]==1
        and outer["actual_direct_worker_returncode_in_tool_output"]==0
        and outer["all_owned_processes_and_groups_drained_in_tool_output"]is True
        and outer["outer_failure_preserved"]is True and outer["successful_outer_terminal_claimed"]is False,
        "Historical actual supervisor outer1 is required, never relabelled0")
    c.require(done["status"]=="completed_C3_native_MLM_child"and done["parent_id"]=="american_c3_val_000"
        and done["completed_epochs"]==5 and done["actual_optimizer_updates"]==227990
        and done["observed_counters_verified"]is True and done["query_ancestry_truth_opened"]is False
        and done["active_input_ref"]==P["input"]and done["checkpoint_ref"]==accepted["saved_MLM_checkpoint_ref"]
        and accepted["root_authorizer"]=="root_agent"and accepted["contrastive_only_artifact_use_accepted"]is True
        and accepted["historical_MLM_outer_exit_code"]==1
        and accepted["original_successful_worker_and_failed_supervisor_preserved"]is True
        and accepted["saved_MLM_eligibility"]["full_fit_completion_claimed"]is False
        and accepted["saved_MLM_eligibility"]["evidence_refs"]["completed"]==P["completed"]
        and accepted["saved_MLM_eligibility"]["evidence_refs"]["wait"]==P["wait"]
        and accepted["saved_MLM_eligibility"]["evidence_refs"]["outer"]==P["outer"],
        "Genuine completed five-epoch worker and contrastive-only artifact acceptance, no full MLM fit claim")
    roster=wait["ownership"]
    c.require(roster["complete_owned_roster_known"]is True and roster["owned_roster_truncated"]is False
        and roster["known_identity_count"]==len(roster["owned_identities"])==50
        and roster["reused_numeric_PIDs_excluded"]==[],"Original complete retained50identity roster, no unknown/reused omission")
    pids=sorted({launch["parent_pid"],launch["pid"],*[r["pid"]for r in roster["owned_identities"]]})
    groups=sorted({r["pgid"]for r in wait["group_observations"]})
    c.require(len(pids)==51 and all(type(pid)is int and pid>=2 for pid in pids)
        and groups==[3752065]and all(r["absent"]is True for r in wait["group_observations"]),
        "All51 actual supervisor/worker/helper PIDs and only actually recorded process group")
    policy=c.read(POLICY)
    c.require(KEY not in policy["never_credit_keys"]and KEY not in policy["ordinary_completed_close_excluded_keys"]
        and "H041/American-C3-native-MLM-full30mb-mlm-fit-v8"in policy["never_credit_keys"]
        and "native32/native-VAL000-full227990-borrowed32-GPU0-v1"in policy["ordinary_completed_close_excluded_keys"],
        "Current original policy permits this key only and preserves both unrelated historical blockers")
    return dict(storage=storage,own=own,launch=launch,wait=wait,done=done,pids=pids,groups=groups,reservation_sha256=digest(own))

def request_v1(meta,observation_ref):
    return dict(key=KEY,reservation_sha256=meta["reservation_sha256"],storage_ref=P["storage"],storage_locator=[],
        launch_ref=P["launch"],launch_pid_locator=["pid"],launch_identity=dict(pid=3752065,parent_pid=3750912,input_ref=P["input"]),
        direct_wait_ref=P["wait"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
        wait_identity=dict(pid=3752065,input_ref=P["input"],returncode=0,actual_OS_wait_status=0,
            actual_direct_worker_wait=True,all_owned_processes_and_groups_drained=True,
            completed_ref=P["completed"],operational_failure=EXPECTED_FAILURE),
        completed_ref=P["completed"],completed_identity=dict(status="completed_C3_native_MLM_child",
            parent_id="american_c3_val_000",completed_epochs=5,actual_optimizer_updates=227990,
            observed_counters_verified=True,query_ancestry_truth_opened=False),observation_ref=observation_ref)

def close_once_v1(authorization_ref):
    q=c.read(authorization_ref);c.active(q)
    c.require(q["schema"]=="ROOT_native_MLM_v12_ordinary_close_once_authorization_v1"
        and q["source_manifest_ref"]==c.ref(MANIFEST)and q["operation"]=="close_completed"
        and q["reservation_key"]==KEY and q["storage_ref"]==P["storage"]
        and q["storage_policy_selector_ref"]==SELECTOR and q["accounting_coordinator_source_ref"]==COORDINATOR
        and q["all_original_v12_launch_resume_retry_write_routes_permanently_revoked"]is True
        and q["preserve_all_files_and_historical_outer1"]is True and q["automatic_retry"]is False
        and q["full_MLM_fit_completion_claimed"]is False and q["old_H041_or_native32_close_authorized"]is False
        and type(q["instance"])is str and re.fullmatch("[a-z][a-z0-9_]{0,39}",q["instance"])is not None,
        "Finite explicit v12-only original ordinary-close ROOT authority")
    a=datetime.fromisoformat(q["created_at_utc"]);e=datetime.fromisoformat(q["expires_at_utc"])
    c.require(0<(e-a).total_seconds()<=3600,"Fresh finite ROOT close authority <=1h")
    manifest=c.read(q["source_manifest_ref"])
    for r in manifest["source_refs"]:c.raw(r)
    review=c.read(q["source_review_ref"])
    c.require(review["independent"]is True and review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]
        and review["source_ref"]==c.ref(SOURCE)and review["source_manifest_ref"]==q["source_manifest_ref"],
        "Independent narrow exact source-only close pass")
    c.raw(q["accounting_batch_input_ref"]);meta=metadata_v1()
    directory=c.local(HERE+"/ROOT_native_MLM_v12_close_"+q["instance"])
    c.require(not directory.exists(),"Exclusive close attempt, never retry uncertainty")
    directory.mkdir(mode=0o700)
    _save(directory,"attempt.json",dict(authorization_ref=authorization_ref,source_manifest_ref=q["source_manifest_ref"],
        reservation_key=KEY,historical_supervisor_outer_exit_code=1,automatic_retry=False))
    coordinator=c.load(COORDINATOR,"_native_v12_close_original_v3_coordinator")
    entered=False
    try:
        with coordinator.accounting_transaction(q["accounting_batch_input_ref"]):
            entered=True;c.active(q)
            g=c.load(GUARD,"_native_v12_close_exact_current_Census")
            c.require(g.load_selected_guard_v1(SELECTOR)is g and g.MODE=="Census_observation_two_scope_retention_v1",
                "Exact current original Census guard, no old-overlay mutation")
            c.require(all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in("canonical","file_bytes","mutex")),
                "Fresh unwrapped ROOT mutation interpreter, no consumer memo")
            c.require(g.budget.digest(meta["own"])==meta["reservation_sha256"],"Original reservation digest")
            revocation_ref=_save(directory,"permanent_v12_route_revocation.json",dict(
                schema="ROOT_native_MLM_v12_permanent_original_route_revocation_v1",root_authorizer="root_agent",active=True,
                revoked_at_utc=datetime.now(timezone.utc).isoformat(),authorization_ref=authorization_ref,
                reservation_key=KEY,reservation_sha256=meta["reservation_sha256"],storage_ref=P["storage"],
                output_directory=OUTPUT,worker_auxiliary_scope=AUX,
                all_original_v12_launch_resume_retry_write_routes_permanently_revoked=True,
                saved_artifact_read_only_use_permitted=True,preserve_all_files=True,historical_outer_terminal_ref=P["outer"],
                historical_supervisor_outer_exit_code=1,full_MLM_fit_completion_claimed=False))
            rows=[g._absent(pid)for pid in meta["pids"]];group_rows=[]
            for group in meta["groups"]:
                try:os.killpg(group,0)
                except ProcessLookupError:group_rows.append(dict(pgid=group,absent=True))
                else:raise ValueError("Recorded original group present/reused/unknown, no close")
            observation=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",
                root_authorizer="root_agent",key=KEY,reservation_sha256=meta["reservation_sha256"],storage_ref=P["storage"],
                launch_ref=P["launch"],direct_wait_ref=P["wait"],completed_ref=P["completed"],launcher_pid=3752065,
                all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
                completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
                external_NFS_writers_outside_observation=True,pids=meta["pids"],observations=rows,
                groups=meta["groups"],group_observations=group_rows,permanent_route_revocation_ref=revocation_ref,
                historical_outer_terminal_ref=P["outer"],historical_supervisor_outer_exit_code=1,
                saved_artifact_acceptance_ref=P["acceptance"],full_MLM_fit_completion_claimed=False)
            observation_ref=_save(directory,"targeted_completion_observation.json",observation)
            request=request_v1(meta,observation_ref);policy=c.read(POLICY)
            implementation=policy["implementation_ref"];launcher=policy["launcher_ref"]
            with g.budget.v4.mutex(c.ROOT):
                state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
                c.require(len(state[3])==47 and state[0].get(KEY)==meta["own"]and KEY not in state[1],
                    "Exact currently open unchanged original v12 reservation and all47 current role bindings")
                c.require(KEY not in g._policy()["never_credit_keys"]and KEY not in g.ORDINARY_EXCLUSIONS,
                    "Original policy/exclusions checked again, no exception")
                c.active(q);g._proof(request,meta["own"]) # Metadata only; no observe=True or measurement here.
                authority=g.operation_draft_v1("close_completed",request,expected_before=g._pins(state[2]),bindings=state[3])
                authority.update(active=True,expires_at_utc=q["expires_at_utc"])
                authority_ref=_save(directory,"ordinary_close_authority.json",authority)
            c.active(q)
            receipt=g.apply_close_completed_v1(authority_ref,expected_source_sha256=implementation["sha256"],
                expected_launcher_sha256=launcher["sha256"])
            receipt_ref=_save(directory,"ordinary_close_receipt.json",receipt)
    except ValueError as exc:
        if not entered and str(exc)=="Finite shared ROOT accounting coordinator wait":
            ready=_save(directory,"ready_no_close.json",dict(status="READY_NO_CLOSE_COORDINATOR_TIMEOUT",
                actual_close_started=False,measurement_performed=False,credit_claimed=False,automatic_retry=False))
            return dict(status="READY_NO_CLOSE_COORDINATOR_TIMEOUT",ready_ref=ready,actual_close_started=False)
        raise # If apply began or returned uncertainty, preserve attempt; never report no mutation.
    result=dict(status="ORDINARY_V12_CLOSE_APPLIED",receipt_ref=receipt_ref,authority_ref=authority_ref,
        reservation_key=KEY,retained_bytes=receipt["retained_bytes"],credit_bytes=receipt["credit_bytes"],
        projected_bytes=receipt["projected_bytes"],all_files_preserved=True,historical_supervisor_outer_exit_code=1,
        full_MLM_fit_completion_claimed=False,old_H041_or_native32_closed=False,automatic_retry=False,
        ROOT_actual_close_outer0_and_journal_acceptance_still_required=True)
    _save(directory,"result.json",result);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--authorization-ref");a=p.parse_args(argv)
    if not a.authorization_ref:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",guard_imported=False,operation=False)));return 0
    c.require(Path.cwd()==c.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
        and not any(n in sys.modules for n in("torch","torch_geometric")),"Fresh ROOT -B -s stdlib mutation interpreter")
    print(json.dumps(close_once_v1(json.loads(a.authorization_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
