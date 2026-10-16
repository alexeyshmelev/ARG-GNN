"""Known-controller transport-UNKNOWN handover probes. Import is stdlib only.
No probe or write occurs on import. Explicit calls require closed metadata,
a root resource callback and CPU-only project context. No signaling exists.
"""
from pathlib import Path
import errno,fcntl,hashlib,json,os,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/C3_bootstrap_failed_controller_handover_source_20261016_v1/transport_handover_v1.py"
LOCK=ROOT/".runtime/locks/gpu_0.lock"
MAX_IDENTITIES=4096

def require(ok,message):
    if not ok:raise ValueError(message)

def known_roster_v1(final):
    items=[final["controller"]]
    for row in final["jobs"]:
        items.extend(row["process_identities"])
        if row["pid"] is not None:
            require(any(p["pid"]==row["pid"] and p["start_ticks"]==row["leader_start_ticks"]
                for p in row["process_identities"]),"Every started leader in immutable known roster")
    values={}
    for p in items:
        require(all(type(p.get(k)) is int and p[k]>=minimum for k,minimum in
            (("pid",1),("start_ticks",1),("uid",0))),"Exact saved known PID/start/UID")
        key=(p["pid"],p["start_ticks"]);value={k:p[k] for k in ("pid","start_ticks","uid")}
        require(key not in values or values[key]==value,"No conflicting known identity")
        values[key]=value
    require(0<len(values)<=MAX_IDENTITIES,"Finite exact old controller/worker roster")
    return sorted(values.values(),key=lambda p:(p["pid"],p["start_ticks"]))

def roster_sha_v1(roster):
    return hashlib.sha256(json.dumps(roster,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def old_identity_absence_v1(expected,resource_check):
    resource_check("before_exact_old_PID_start_absence")
    p=Path("/proc")/str(expected["pid"])
    def one():
        text=(p/"stat").read_text()
        fields=text[text.rfind(")")+2:].split()
        require(len(fields)>19,"Complete exact known Linux PID stat")
        return dict(pid=expected["pid"],start_ticks=int(fields[19]),uid=p.stat().st_uid,state=fields[0])
    try:
        first=one();second=one()
    except OSError as error:
        if error.errno not in (errno.ENOENT,errno.ESRCH):raise
        resource_check("after_exact_old_PID_absence")
        return dict(expected=expected,old_start_identity_absent=True,observation="ENOENT_or_ESRCH")
    require(all(first[k]==second[k] for k in ("pid","start_ticks","uid")),"Stable exact known PID/start/UID observation, no churn or unreadable skip")
    require(first["start_ticks"]!=expected["start_ticks"],
        "Exact old process identity still present, including zombie identity")
    resource_check("after_exact_old_PID_reused_absence")
    return dict(expected=expected,old_start_identity_absent=True,observation="PID_reused",
        current_identity=first,unrelated_reused_process_untouched=True)

def validate_observation_v1(observation,request_ref,roster,lease_expected,helper_ref):
    require(observation.get("schema")=="GPU0_transport_unknown_closed_handover_observation_v1"
        and observation.get("root_authorizer")=="root_agent"
        and observation.get("actual_probe_execution") is True
        and observation.get("request_ref")==request_ref and observation.get("probe_source_ref")==helper_ref
        and observation.get("original_external_tool_exit_status")=="UNKNOWN"
        and observation.get("original_external_tool_exit_code") is None
        and observation.get("original_controller_outer_wait0_claimed") is False
        and observation.get("known_identity_roster_sha256")==roster_sha_v1(roster)
        and [p.get("expected") for p in observation.get("known_identity_observations",[])]==roster
        and all(p.get("old_start_identity_absent") is True
            and p.get("observation") in ("ENOENT_or_ESRCH","PID_reused")
            for p in observation["known_identity_observations"]),
        "Genuine exact-source known-identity observation, original external exit remains UNKNOWN")
    lock=observation.get("old_lease_free_probe",{})
    require(lock.get("path")==str(LOCK) and lock.get("device")==lease_expected["device"]
        and lock.get("inode")==lease_expected["inode"]
        and lock.get("metadata_sha256")==lease_expected["metadata_sha256"]
        and lock.get("exclusive_nonblocking_acquired") is True
        and lock.get("own_probe_FD_closed") is True
        and lock.get("old_inode_or_metadata_modified") is False,
        "Actual original canonical inode verified free without mutation or borrowed FD")
    require(type(observation.get("observed_unix")) in (int,float)
        and 0<observation["observed_unix"]<=time.time()+1,"Actual finite past observation time")
    return observation

def observe_v1(request_ref,final,lease_expected,helper_ref,*,resource_check):
    require(callable(resource_check),"Actual root resource callback, no default bypass")
    resource_check("before_transport_unknown_known_identity_probe")
    require(sys.platform.startswith("linux") and Path.cwd().resolve()==ROOT
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="",
        "Fresh root CPU-only Linux/project observation, never GPU access")
    roster=known_roster_v1(final)
    observations=[old_identity_absence_v1(p,resource_check) for p in roster]
    require(LOCK.resolve()==LOCK and not LOCK.is_symlink(),"Canonical old GPU0 inode")
    before=LOCK.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1
        and (before.st_dev,before.st_ino)==(lease_expected["device"],lease_expected["inode"]),
        "Pinned actual old lease inode, no replaced file")
    resource_check("before_old_GPU0_lease_free_probe")
    fd=os.open(LOCK,os.O_RDWR|os.O_NOFOLLOW)
    acquired=False
    try:
        current=os.fstat(fd)
        require((current.st_dev,current.st_ino)==(before.st_dev,before.st_ino),"Same original independent lock FD")
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
        raw=os.pread(fd,65536,0)
        require(len(raw)==current.st_size and hashlib.sha256(raw).hexdigest()==lease_expected["metadata_sha256"],
            "Original lease metadata remains exact under independent exclusive probe")
        require(json.loads(raw)==lease_expected["metadata"],"Exact old launcher/pool/UUID lease metadata")
        # A second exact known roster check occurs while the original inode is
        # proven free. No global process, FD or unrelated-user inventory occurs.
        observations=[old_identity_absence_v1(p,resource_check) for p in roster]
        after=LOCK.lstat()
        require((after.st_dev,after.st_ino)==(before.st_dev,before.st_ino)
            and hashlib.sha256(os.pread(fd,65536,0)).hexdigest()==lease_expected["metadata_sha256"],
            "Original lease inode/content unchanged through probe")
    finally:
        os.close(fd)  # Close our independent OFD, never unlock an inherited lease.
    require(acquired,"Old canonical lifetime lease must actually be free")
    resource_check("after_old_GPU0_lease_free_probe")
    observation=dict(schema="GPU0_transport_unknown_closed_handover_observation_v1",
        root_authorizer="root_agent",actual_probe_execution=True,request_ref=request_ref,probe_source_ref=helper_ref,
        original_external_tool_exit_status="UNKNOWN",original_external_tool_exit_code=None,
        original_controller_outer_wait0_claimed=False,observed_unix=time.time(),
        known_identity_roster_sha256=roster_sha_v1(roster),known_identity_observations=observations,
        old_lease_free_probe=dict(path=str(LOCK),device=before.st_dev,inode=before.st_ino,
            metadata_sha256=lease_expected["metadata_sha256"],exclusive_nonblocking_acquired=True,
            own_probe_FD_closed=True,old_inode_or_metadata_modified=False),
        no_signals_or_other_process_changes=True,no_GPU_query=True)
    return validate_observation_v1(observation,request_ref,roster,lease_expected,helper_ref)

def verify_acquired_lease_v1(lease,expected):
    """Final authoritative check under the new controller's actual acquired FD."""
    disk=LOCK.lstat();held=os.fstat(lease.fileno())
    require(LOCK.resolve()==LOCK and stat.S_ISREG(disk.st_mode) and disk.st_nlink==1
        and (disk.st_dev,disk.st_ino)==(held.st_dev,held.st_ino)
        ==(expected["device"],expected["inode"]),"Actual new lease is the pinned original canonical inode")
    require(hashlib.sha256(os.pread(lease.fileno(),65536,0)).hexdigest()==expected["metadata_sha256"],
        "Original lock metadata unchanged until actual new lease acquisition")


# Exact bootstrap-only administrative successor. The original functions above
# are retained verbatim. No import-time probe, source import or live action.
HISTORICAL_TRANSPORT_HELPER_REF={'path': 'research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/transport_handover_v1.py', 'sha256': 'd0c6f150cdf4b987e67a1dd95c9918e4467e601e467c7705d4a7d6fabd80dc4e'}
BOOTSTRAP_EVIDENCE_REFS={'registry': {'path': 'research/reviews/C3_actual_full_state_handover_root_20261016_v1/module_route_failed_controller.final_registry_v1.json', 'sha256': '52c2bf6b60dcdf34a75e001505b188e9d71d8f3f966676404a88debcde3816be'}, 'terminal': {'path': 'research/reviews/C3_actual_full_state_handover_root_20261016_v1/module_route_failed_controller.actual_terminal_v1.json', 'sha256': 'fac187e17fbebdca92ca817bc07a17ebee73ec25284842eb2a7423459f561c4c'}, 'revocation': {'path': 'research/reviews/C3_actual_full_state_handover_root_20261016_v1/module_route_failed_controller.permanent_revocation_v1.json', 'sha256': 'ce1dede0412e64c1bb6907498c4985144f7f7e0697b6e7336a2a45c5d1f4cb94'}}
BOOTSTRAP_CASE_KEY="bootstrap_failed_controller_case_v1"

def bootstrap_case_metadata_v1(contract,helper_ref):
    """Root may construct JSON only after explicitly selecting this source."""
    require(helper_ref==contract.TRANSPORT_HELPER_REF,
        "Exact current consolidated helper role")
    return dict(schema="GPU0_bootstrap_failed_controller_transitive_administrative_case_v1",
        root_authorizer="root_agent",administrative_evidence_transitive=True,
        all_scientific_runtime_guard_direct_source_roles_retained=True,
        helper_ref=helper_ref,evidence={name:dict(ref=record,metadata=contract.read(record))
            for name,record in BOOTSTRAP_EVIDENCE_REFS.items()})

def validate_bootstrap_case_v1(proof_ref,proof,contract):
    """Metadata only, called after all original UNKNOWN-proof checks."""
    case=proof.get(BOOTSTRAP_CASE_KEY)
    require(isinstance(case,dict) and set(case)=={"schema","root_authorizer",
        "administrative_evidence_transitive","all_scientific_runtime_guard_direct_source_roles_retained",
        "helper_ref","evidence"}
        and case["schema"]=="GPU0_bootstrap_failed_controller_transitive_administrative_case_v1"
        and case["root_authorizer"]=="root_agent" and case["administrative_evidence_transitive"] is True
        and case["all_scientific_runtime_guard_direct_source_roles_retained"] is True
        and case["helper_ref"]==contract.TRANSPORT_HELPER_REF
        and set(case["evidence"])==set(BOOTSTRAP_EVIDENCE_REFS),
        "Explicit exact bootstrap-only transitive administrative case")
    metadata={}
    for name,record in BOOTSTRAP_EVIDENCE_REFS.items():
        item=case["evidence"][name]
        require(set(item)=={"ref","metadata"} and item["ref"]==record,
            "Exact immutable immediate administrative evidence reference")
        actual=contract.read(record)
        require(item["metadata"]==actual,"Embedded immediate metadata equals authenticated original bytes")
        metadata[name]=actual
    final,terminal,revoke=(metadata[n] for n in ("registry","terminal","revocation"))
    require(final.get("schema")=="GPU0_declared_logical_pool_registry_v1"
        and final.get("lifecycle")=="closed" and final.get("physical_gpu")==0
        and final.get("pool")=="research0-resume-oct16-v1"
        and final.get("pool_job")=="gpu0_shared_v1:research0-resume-oct16-v1:gpu0"
        and final.get("safety_error") is None
        and final.get("ambiguous_descendants_or_lease_holders")==[]
        and final.get("owned_orphan_waits")==[],
        "Exact immediately closed bootstrap controller with no ambiguous ownership")
    require(terminal.get("schema")=="root_consumed_actual_module_route_failure_terminal_v1"
        and terminal.get("final_registry_ref")==BOOTSTRAP_EVIDENCE_REFS["registry"]
        and terminal.get("scientific_entry_started") is False
        and terminal.get("C3_new_updates")==terminal.get("native_new_updates")==0
        and terminal.get("old_controller_23022_original_terminal")=="UNKNOWN"
        and terminal.get("prior_scientific_failures_preserved") is True,
        "Actual bootstrap-only failure, zero new science, original UNKNOWN retained")
    for name,session,code in (("controller",95438,0),("native_wait",85327,2),("shutdown",23812,0)):
        row=terminal[name]
        require(row.get("actual_tool_session_id")==session
            and row.get("actual_terminal_consumed") is True
            and type(row.get("actual_exit_code")) is int and row["actual_exit_code"]==code
            and isinstance(row.get("actual_chunk_id"),str) and row["actual_chunk_id"],
            "Exact genuine controller0, native-client2 and shutdown0 terminals")
    cap=contract.read(final["capability_ref"])
    require(cap.get("schema")=="GPU0_declared_logical_pool_capability_v1"
        and cap.get("pool")==final["pool"] and len(cap.get("jobs",{}))==2
        and {r["worker_kind"] for r in cap["jobs"].values()}=={"C3","native_ARGformer"},
        "Exact original two-slot immediate controller, not a successful fit")
    original_ref=cap["prior_controller_completion_ref"]
    original=contract.read(original_ref)
    require(proof_ref!=original_ref and
        {k:v for k,v in proof.items() if k!=BOOTSTRAP_CASE_KEY}==original,
        "Fresh proof is exactly preserved original UNKNOWN proof plus one administrative case")
    require(revoke.get("schema")=="GPU0_bootstrap_failed_controller_permanent_capability_revocation_v1"
        and revoke.get("root_authorizer")=="root_agent"
        and revoke.get("active") is True and revoke.get("permanent") is True
        and revoke.get("prior_capability_ref")==final["capability_ref"]
        and revoke.get("final_registry_ref")==BOOTSTRAP_EVIDENCE_REFS["registry"]
        and revoke.get("actual_tool_terminal_ref")==BOOTSTRAP_EVIDENCE_REFS["terminal"]
        and revoke.get("controller_identity")==final["controller"]
        and revoke.get("all_old_controller_start_submit_requeue_worker_write_paths_revoked") is True
        and revoke.get("new_scientific_or_restart_authority_granted") is False
        and revoke.get("scientific_native_entry_started") is False
        and revoke.get("C3_never_submitted") is True
        and revoke.get("unchanged_scientific_native_input_reuse_authorized_for_bootstrap_correction") is True,
        "Exact permanent immediate capability revocation and unchanged native-input reuse")
    require(len(final["jobs"])==1,"Only native bootstrap was submitted, C3 was never submitted")
    job=final["jobs"][0];name=job["logical_job"]
    require(name in cap["jobs"] and job["requested"]==cap["jobs"][name]
        and job["requested"]["worker_kind"]=="native_ARGformer"
        and job.get("state")=="failed" and job.get("exit_code")==2
        and job.get("pid")==3333302 and job.get("leader_start_ticks")==481133372
        and job.get("leader_wait_status_known") is True and job.get("owned_drain_observed") is True
        and job.get("descendant_waits")==[] and job.get("log_dropped_bytes")==0,
        "Actual native bootstrap leader wait2 and owned drain, no scientific descendants")
    require(final["controller"].get("pid")==3331919
        and final["controller"].get("start_ticks")==481110879,
        "Exact immediate controller PID/start identity")
    wait=contract.read(terminal["native_wait"]["saved_result_ref"])
    shutdown=contract.read(terminal["shutdown"]["saved_result_ref"])
    require(wait.get("schema")=="root_actual_native_bootstrap_failure_wait_v1"
        and wait.get("client_returncode")==2 and shutdown.get("client_returncode")==0
        and shutdown.get("schema")=="root_actual_module_route_failed_controller_shutdown_v1",
        "Authenticated actual client wait2 and shutdown0 result metadata")
    waited=json.loads(wait["stdout"]);closed=json.loads(shutdown["stdout"])
    require(all(waited.get(k)==job.get(k) for k in ("pid","leader_start_ticks","state","exit_code",
        "leader_wait_status_known","owned_drain_observed","requested","argv"))
        and closed.get("controller")==final["controller"]
        and closed.get("capability_ref")==final["capability_ref"] and closed.get("lifecycle")=="closed",
        "Actual saved wait and shutdown bind the same exact immediate controller/worker")
    log_ref=terminal["failure_log_ref"]
    log_path=ROOT/log_ref["path"]
    require(log_ref["path"]==str(Path(job["log_path"]).relative_to(ROOT))
        and contract.ref(log_ref["path"])==log_ref
        and log_path.resolve()==log_path and not log_path.is_symlink()
        and log_path.stat().st_size==job["log_bytes"]==760,
        "Exact small actual argparse log, not a science result")
    raw=log_path.read_bytes()
    require(b"argument --module: invalid choice:" in raw
        and b"research.reviews.GPU0_declared_C3_resume_transport_unknown_source_20261016_v2.gpu0_declared_pool_v1" in raw
        and job["argv"][job["argv"].index("--module")+1]
            =="research.reviews.GPU0_declared_C3_resume_transport_unknown_source_20261016_v2.gpu0_declared_pool_v1",
        "Stale route failed in launcher argparse before pool import or scientific entry")
    request=contract.read(original["transport_unknown_request_ref"])
    old_final=contract.read(request["final_registry_ref"])
    ready=contract.read(request["prior_ready_ref"])
    old_expected=contract.validate_loss_revocation_v1(request,ready,old_final)
    start=final["prior_transport_unknown_start_recheck"]
    validate_observation_v1(start,original["transport_unknown_request_ref"],
        known_roster_v1(old_final),old_expected,HISTORICAL_TRANSPORT_HELPER_REF)
    require(start.get("request_validator_ref")==contract.ref(
        "research/reviews/C3_native_H050_failed_CLI_transport_admission_source_20261016_v1/finite_jobs_v1.py"),
        "Preserved actual original startup check uses the original validator source")
    previous=start["old_lease_free_probe"]
    current=final["lease_metadata"]
    require(final["lease_path"]==str(LOCK)
        and current.get("gpu")==0 and current.get("launcher_pid")==final["controller"]["pid"]
        and current.get("job")==final["pool_job"] and current.get("logical_sharing") is True
        and current.get("uuid")=="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
        and current.get("storage_bytes")==33554432
        and previous["device"]==old_expected["device"]
        and previous["inode"]==old_expected["inode"]
        and final["lease_sha256"]!=old_expected["metadata_sha256"],
        "Same original inode, honestly changed immediate-controller lease metadata")
    expected=dict(device=previous["device"],inode=previous["inode"],
        metadata_sha256=final["lease_sha256"],metadata=current)
    # Only the finite exact original and immediate known identities are probed.
    merged=dict(controller=final["controller"],jobs=[*old_final["jobs"],*final["jobs"]])
    roster=known_roster_v1(merged)
    require(len(roster)==len(known_roster_v1(old_final))+len(known_roster_v1(final)),
        "Original and immediate exact known identity rosters do not overlap")
    return dict(original_prior_proof_ref=original_ref,merged_final=merged,expected_lease=expected,
        immediate_registry_ref=BOOTSTRAP_EVIDENCE_REFS["registry"],
        immediate_terminal_ref=BOOTSTRAP_EVIDENCE_REFS["terminal"])

def recheck_bootstrap_before_start_v1(proof_ref,proof,contract,*,resource_check):
    require(callable(resource_check),"Original finite startup resource callback")
    resource_check("before_bootstrap_closed_controller_metadata")
    material=validate_bootstrap_case_v1(proof_ref,proof,contract)
    resource_check("before_bootstrap_closed_controller_actual_probe")
    observation=observe_v1(proof_ref,material["merged_final"],material["expected_lease"],
        contract.TRANSPORT_HELPER_REF,resource_check=resource_check)
    # Give the new event its own honest type. The immutable original observation
    # remains a separate record and is never relabelled controller95438 proof.
    observation["schema"]="GPU0_bootstrap_failed_controller_start_observation_v1"
    observation["current_closed_lease_free_probe"]=observation.pop("old_lease_free_probe")
    observation.update(original_preserved_transport_proof_ref=material["original_prior_proof_ref"],
        immediate_closed_registry_ref=material["immediate_registry_ref"],
        immediate_controller_actual_terminal_ref=material["immediate_terminal_ref"],
        immediate_controller_tool_session=95438,immediate_controller_exit_code=0,
        original_metadata_bytes_continuity_claimed=False,
        current_lease_metadata_authenticated=True,bootstrap_only_no_new_science=True)
    resource_check("after_bootstrap_closed_controller_actual_probe")
    return observation,material["expected_lease"]
