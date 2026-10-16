"""Known-controller transport-UNKNOWN handover probes. Import is stdlib only.
No probe or write occurs on import. Explicit calls require closed metadata,
a root resource callback and CPU-only project context. No signaling exists.
"""
from pathlib import Path
import errno,fcntl,hashlib,json,os,stat,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/GPU0_declared_C3_resume_transport_unknown_source_20261016_v1/transport_handover_v1.py"
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
