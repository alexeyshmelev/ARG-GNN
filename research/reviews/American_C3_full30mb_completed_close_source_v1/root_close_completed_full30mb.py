"""Root-only prospective exact completed full30Mb C3 close. Default is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/American_C3_full30mb_completed_close_source_v1"
SOURCE="research/reviews/American_C3_full30mb_completed_close_source_v1/root_close_completed_full30mb.py"
PREVIOUS_REF=dict(path="research/reviews/American_C3_short2mb_completed_close_source_v1/root_close_completed_stages.py",
                  sha256="d85ef0b3811545eff5dc149dd4de144d7873bb7914e5d1c847feba654e8f6259")
EVIDENCE_REF=dict(path="research/reviews/American_C3_full30mb_completed_close_source_v1/stage_evidence_refs.json",
                  sha256="ab66843e1677acd2305539e72f7b2b79ddf38ee6df48489024bc62776e276a3b")
if hashlib.sha256((ROOT/PREVIOUS_REF["path"]).read_bytes()).hexdigest()!=PREVIOUS_REF["sha256"]:
    raise ValueError("Exact frozen original four-stage close helper required")
_ms=importlib.util.spec_from_file_location("_C3_original_completed_close",ROOT/PREVIOUS_REF["path"])
previous=importlib.util.module_from_spec(_ms);_ms.loader.exec_module(previous)
require,ref,read,save,utc,overlay,absent,group_absent=(
    previous.require,previous.ref,previous.read,previous.save,previous.utc,
    previous.overlay,previous.absent,previous.group_absent)
CORE,BASE,POLICY_REF=previous.CORE,previous.BASE,previous.POLICY_REF

def evidence_for():
    evidence=read(EVIDENCE_REF);row=evidence["stage"]
    ad,admitted,launch,wait,completed=(read(row[name]) for name in
        ("admission_ref","admitted_ref","launch_ref","wait_ref","completed_ref"))
    own=ad["storage"]["reservation"]
    require(own["key"]==row["key"] and own["output"]==row["output"]
        and own["auxiliary_scopes"]==row["auxiliary_scopes"]
        and own["reserved_bytes"]==row["reserved_bytes"]==12500000000,
        "Exact original repaired full30Mb reservation and all owned scopes/cap")
    require(ad["job_id"]==row["job_id"]
        and ad["output_directory"]==str(Path(row["output"]).relative_to(ROOT))
        and ad["driver_ref"]==launch["driver_ref"]==wait["driver_ref"]==row["driver_ref"]
        and ad["source_ref"]==wait["scientific_worker_ref"]==row["scientific_worker_ref"]
        and admitted["admission_ref"]==launch["admission_ref"]==wait["admission_ref"]==row["admission_ref"]
        and admitted["stage"]==launch["stage"]==wait["stage"]==row["stage"]=="full30mb",
        "Original admitted stage/job/producer/supervisor source and admission provenance")
    require(launch["pid"]==wait["pid"]==row["known_worker_pid"]==2548964
        and launch["process_group"]==wait["process_group"]==row["known_worker_process_group"]==2548964
        and launch["worker_auxiliary_scope"]==wait["worker_auxiliary_scope"]==row["auxiliary_scopes"][0]
        and launch["policy_ref"]==wait["policy_ref"]==ad["completed_overlay_policy_ref"]==POLICY_REF,
        "Original exact worker PID/group/auxiliary/policy identities")
    require(type(wait["returncode"]) is int and wait["returncode"]==0
        and wait["actual_direct_worker_wait"] is True and wait["owned_worker_group_ESRCH"] is True
        and wait["operational_failure"] is None and wait["cleanup_errors"]==[]
        and wait["supervisor_errors"]==[] and wait["post_wait_errors"]==[]
        and wait["failed_v5_artifacts_preserved"] is True
        and wait["completed_ref"]==row["completed_ref"],
        "Original clean actual worker wait0/groupESRCH and exact completion")
    require(completed==row["completed_identity"]
        and completed["status"]=="completed_current_C3_inference"
        and completed["tsinfer_num_threads"]==8 and completed["author_Table1_exact_reproduction"] is False,
        "Original completed producer metadata; no exact-paper reproduction claim")
    return evidence,row,own,launch,wait

def validate_approval_v1(cert,evidence):
    row=evidence["stage"]
    require(cert.get("schema")=="American_C3_full30mb_root_close_approval_v1"
        and cert.get("root_authorizer")=="root_agent" and cert.get("active") is True
        and cert.get("close_helper_ref")==ref(SOURCE) and cert.get("stage_evidence_ref")==EVIDENCE_REF
        and cert.get("stage_evidence")==row
        and cert.get("all_genuinely_owned_processes_and_groups_listed") is True
        and cert.get("all_prior_input_config_storage_source_launch_resume_write_capabilities_revoked") is True
        and cert.get("preserve_all_files_and_historical_journals") is True
        and cert.get("all_other_failed_v5_scopes_remain_charged") is True,
        "Explicit root complete-roster/revocation certificate required for this one repaired scope")
    terminal=cert.get("supervisor_terminal_wait")
    require(terminal==dict(root_actually_consumed_terminal_wait=True,session_id=98609,
        supervisor_pid=2548839,returncode=0),
        "Root must explicitly attest its actual terminalsession98609 supervisor wait0")
    pids,groups=cert.get("root_owned_pids"),cert.get("root_owned_process_groups")
    require(isinstance(pids,list) and 1<=len(pids)<=256 and len(set(pids))==len(pids)
        and all(type(p) is int and p>=2 for p in pids) and set(row["required_owned_pids"])<=set(pids),
        "Root must expand the complete supervisor/worker/descendant PID roster")
    require(isinstance(groups,list) and 1<=len(groups)<=256 and len(set(groups))==len(groups)
        and all(type(g) is int and g>=2 for g in groups) and set(row["required_owned_worker_groups"])<=set(groups),
        "Root must expand the complete genuinely owned process-group roster")
    return pids,groups

def close(approval_ref):
    evidence,row,own,launch,wait=evidence_for()
    pids,groups=validate_approval_v1(read(approval_ref),evidence)
    observations=[absent(pid) for pid in pids]
    group_observations=[group_absent(g) for g in groups]
    m=overlay()  # Original credit-v3 loader rejects consumer memo wrappers.
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    require(state[0].get(row["key"])==own and row["key"] not in state[1],
            "Exact original reservation must still be open and unchanged")
    packet=HERE/"closed_full30mb";require(not packet.exists(),"No close replay or overwrite");packet.mkdir()
    obs=dict(schema="concurrent_completed_scope_targeted_root_observation_v1",root_authorizer="root_agent",
        key=row["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["admission_ref"],
        launch_ref=row["launch_ref"],direct_wait_ref=row["wait_ref"],completed_ref=row["completed_ref"],
        launcher_pid=launch["pid"],all_root_owned_launcher_supervisor_worker_pids_listed=True,direct_wait_observed=True,
        completed_scope_all_old_original_resume_write_capabilities_revoked=True,preserve_all_files=True,
        external_NFS_writers_outside_observation=True,pids=pids,observations=observations,
        root_owned_process_groups=groups,process_group_observations=group_observations,
        root_approval_ref=approval_ref,source_ref=ref(SOURCE),recorded_utc=utc(),
        supervisor_pid=row["known_supervisor_pid"],root_terminal_session_id=98609,
        revocation_applies_to_all_prior_input_config_storage_source_launch_and_resume_capabilities=True)
    observation_ref=save(packet,"root_observation.json",obs)
    request=dict(key=row["key"],reservation_sha256=m.budget.digest(own),storage_ref=row["admission_ref"],
        storage_locator=["storage"],launch_ref=row["launch_ref"],launch_pid_locator=["pid"],
        launch_identity=dict(pid=launch["pid"],process_group=launch["process_group"],stage="full30mb",
            admission_ref=row["admission_ref"],driver_ref=row["driver_ref"]),
        direct_wait_ref=row["wait_ref"],wait_pid_locator=["pid"],wait_exit_locator=["returncode"],
        wait_identity=dict(returncode=0,actual_direct_worker_wait=True,owned_worker_group_ESRCH=True,
            pid=wait["pid"],process_group=wait["process_group"],stage="full30mb",
            admission_ref=row["admission_ref"],completed_ref=row["completed_ref"]),
        completed_ref=row["completed_ref"],completed_identity=row["completed_identity"],observation_ref=observation_ref)
    operation=m.operation_draft_v1("close_completed",request,expected_before=m._pins(state[2]),bindings=state[3])
    operation.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    authority_ref=save(packet,"close_authority.json",operation)
    for group in groups:group_absent(group)
    # The unchanged primitive alone inventories exact original output+auxiliary scopes.
    receipt=m.apply_close_completed_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"close_receipt.json",receipt)
    print(json.dumps(dict(key=row["key"],retained_bytes=receipt["retained_bytes"],
        credit_bytes=receipt["credit_bytes"],projected_bytes=receipt["projected_bytes"],
        scope_measurements=receipt["scope_measurements"],files_deleted=0)),flush=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--close",action="store_true")
    parser.add_argument("--root-approval-ref");parser.add_argument("--root-approval-sha256")
    args=parser.parse_args(argv)
    if args.close:
        require(args.root_approval_ref and args.root_approval_sha256,"Exact explicit root certificate path/SHA")
        close(dict(path=args.root_approval_ref,sha256=args.root_approval_sha256));return 0
    print(json.dumps(dict(status="source_only_no_guard_accounting_or_process_checks",
        explicit_root_certificate_required=True,scope_inventories=0,closes_applied=0,
        files_deleted=0,other_failed_v5_scopes_remain_charged=True)));return 0
if __name__=="__main__":raise SystemExit(main())
