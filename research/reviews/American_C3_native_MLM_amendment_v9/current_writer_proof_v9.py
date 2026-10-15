"""Narrow pre-spawn/post-drain same-UID current writer proof; inert on import."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,stat,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
PROC=Path("/proc")
KEY="H041/American-C3-native-MLM-full30mb-mlm-fit-v8"
JOB="American-C3-native-MLM-full30mb-mlm-fit-v8"
KNOWN_ROOTS=(2817234,2817625,2817931,2818200)
KNOWN_GROUPS=KNOWN_ROOTS
OLD_REL="research/reviews/American_C3_native_MLM_full_fit_v8"
OUTPUT=str(ROOT/"experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8")
TMP=str(ROOT/".runtime/tmp/c3full8/f")
AUX=str(ROOT/OLD_REL/"admitted_mlm_v1/worker_aux")
SOURCE_TOKENS={"arggnn.american_c3_native_mlm_full_fit_v8","src/arggnn/american_c3_native_mlm_full_fit_v8.py",
    str(ROOT/"src/arggnn/american_c3_native_mlm_full_fit_v8.py")}
for name in ("root_native_driver_v1.py","supervisor_v1.py","native_child_v1.py"):
    SOURCE_TOKENS.update({OLD_REL+"/"+name,str(ROOT/OLD_REL/name)})
SCOPES=(OUTPUT,TMP,AUX)
MAX_PROCESSES=8192
MAX_FDS=4096
MAX_SECONDS=15.
class Changed(RuntimeError):pass
def require(value,message):
    if not value:raise ValueError(message)
def _time_limit(deadline):
    require(time.monotonic()<=deadline,"Current writer scan deadline exceeded; no absence claimed")
def _read(path,cap):
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        chunks=[];total=0
        while True:
            block=os.read(fd,min(65536,cap+1-total))
            if not block:break
            chunks.append(block);total+=len(block)
            require(total<=cap,"Current process metadata limit exceeded; no truncated scan")
        return b"".join(chunks)
    finally:os.close(fd)
def _process_identity(pid):
    raw=_read(PROC/str(pid)/"stat",8192).decode()
    values=raw.rsplit(")",1)[1].split()
    return int(values[19]),values[0]
def _snapshot(uid,deadline):
    _time_limit(deadline)
    dirs=[]
    for path in PROC.iterdir():
        if not path.name.isdecimal():continue
        dirs.append(path)
        require(len(dirs)<=MAX_PROCESSES,"Bounded complete current process enumeration")
    result={}
    for path in dirs:
        _time_limit(deadline)
        try:
            if path.lstat().st_uid!=uid:continue
            start,state=_process_identity(int(path.name))
            if path.lstat().st_uid!=uid:raise Changed("Process UID changed")
            result[int(path.name)]=start
        except FileNotFoundError:continue  # This process is no longer current.
    return result
def _fd_targets(pid,deadline):
    directory=PROC/str(pid)/"fd"
    result={}
    # Keep scandir alive while reading each target. For the observer itself,
    # listdir/Path.iterdir can return their already-closed inspection FD.
    # The live inspection descriptor is included as observed metadata; no
    # disappeared descriptor is silently ignored for this or any other PID.
    with os.scandir(directory) as entries:
        for entry in entries:
            _time_limit(deadline)
            require(len(result)<MAX_FDS,"Current same-UID fd metadata limit exceeded")
            try:result[entry.name]=os.readlink(entry.path)
            except FileNotFoundError:raise Changed("Current process fd membership changed")
    return result
def _metadata(pid,deadline):
    _time_limit(deadline)
    return (_read(PROC/str(pid)/"cmdline",131072),
        _read(PROC/str(pid)/"environ",1048576),_fd_targets(pid,deadline))
def _under(value,scope):
    return value==scope or value.startswith(scope+"/")
def _path_reasons(value):
    if value.endswith(" (deleted)"):value=value[:-10]
    reasons=[]
    if value in SOURCE_TOKENS:reasons.append("exact_old_source_token")
    if any(_under(value,scope) for scope in SCOPES):reasons.append("exact_old_scope_path")
    # Relative repo paths are accepted only as complete individual argv/env/fd values.
    if any(_under(value,scope[len(str(ROOT))+1:]) for scope in SCOPES):reasons.append("exact_old_relative_scope_path")
    return reasons
def _classify(pid,start,uid,deadline):
    try:
        first,state=_process_identity(pid)
        if first!=start or (PROC/str(pid)).lstat().st_uid!=uid:raise Changed("Current process identity changed")
        if state in ("Z","X"):return []
        one=_metadata(pid,deadline);two=_metadata(pid,deadline)
        last,_=_process_identity(pid)
        if one!=two or last!=start or (PROC/str(pid)).lstat().st_uid!=uid:raise Changed("Current process metadata changed")
    except FileNotFoundError:raise Changed("Current process vanished during classification")
    argv,env,fds=one;reasons=set()
    for token in argv.split(bytes([0])):
        if not token:continue
        value=token.decode("utf-8",errors="surrogateescape")
        if value in (JOB,"--job="+JOB):reasons.add("exact_old_job_argv")
        reasons.update(_path_reasons(value))
    for raw in env.split(bytes([0])):
        if b"=" not in raw:continue
        key,value=raw.split(b"=",1)
        value=value.decode("utf-8",errors="surrogateescape")
        if key in (b"ARGGNN_JOB_NAME",b"ARGGNN_LOGICAL_JOB_NAME") and value==JOB:
            reasons.add("exact_old_job_environment")
        # Only full values equal to own source/scope paths; no broad substring match.
        reasons.update(_path_reasons(value))
    for value in fds.values():reasons.update(_path_reasons(value))
    return sorted(reasons)
def scan_current_same_uid():
    uid=os.geteuid();deadline=time.monotonic()+MAX_SECONDS
    for attempt in range(3):
        _time_limit(deadline)
        try:
            before=_snapshot(uid,deadline);matches=[]
            for pid,start in sorted(before.items()):
                reasons=_classify(pid,start,uid,deadline)
                if reasons:matches.append(dict(pid=pid,start_ticks=start,match_reasons=reasons))
            after=_snapshot(uid,deadline)
            if before!=after:raise Changed("Current same-UID membership changed")
            return dict(complete=True,stable_same_uid_membership=True,observer_effective_uid=uid,
                same_uid_processes_classified=len(before),current_matching_processes=matches,
                process_scope="proc directory owner uid equals observer effective uid",
                metadata_fields=["stat identity","cmdline tokens","environment exact job/path values","fd symlink targets"],
                unrelated_process_contents_recorded=False,attempts=attempt+1,
                bounds=dict(max_processes=MAX_PROCESSES,max_fds_per_process=MAX_FDS,
                    cmdline_bytes=131072,environ_bytes=1048576,seconds=MAX_SECONDS))
        except Changed:
            if attempt==2:raise ValueError("Current writer scan remained unstable; no absence claimed")
    raise ValueError("No complete current writer observation")
def observe_known_roots_and_groups():
    roots=[];groups=[];now=lambda:datetime.now(timezone.utc).isoformat()
    for pid in KNOWN_ROOTS:
        try:os.kill(pid,0)
        except ProcessLookupError:roots.append(dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=now()))
        else:raise ValueError("Known old root present/reused; no absence claimed")
    for group in KNOWN_GROUPS:
        try:os.killpg(group,0)
        except ProcessLookupError:groups.append(dict(group=group,absent=True,method="targeted os.killpg(group,0) ESRCH",checked_utc=now()))
        else:raise ValueError("Known original process group remains; no absence claimed")
    return roots,groups
def prove_current_absence():
    roots,groups=observe_known_roots_and_groups()
    scan=scan_current_same_uid()
    require(scan["complete"] is True and scan["current_matching_processes"]==[],"Current same-UID old job/source/scope writer detected")
    # Repeat targeted roots/groups after the complete metadata scan.
    roots,groups=observe_known_roots_and_groups()
    return dict(schema="native_MLM_honest_current_writer_observation_v9_v2",
        checked_utc=datetime.now(timezone.utc).isoformat(),historical_ids_complete=False,
        historical_descendant_ids_lost=True,native_wait_exit="UNKNOWN",actual_optimizer_updates="UNKNOWN",
        known_old_roots=roots,known_original_groups=groups,current_same_uid_scan=scan,
        external_NFS_writers_outside_observation=True,other_UID_and_remote_host_writers_outside_observation=True,
        future_relaunch_prevented_by_cooperative_root_revocation=True)
def validate_static_certificate(cert):
    require(cert["schema"]=="native_MLM_fullfit_current_writer_absence_root_v9_v2"
        and cert["root_authorizer"]=="root_agent" and cert["active"] is True and cert["key"]==KEY
        and cert["historical_ids_complete"] is False and cert["historical_descendant_ids_lost"] is True
        and cert["all_v8_future_launch_resume_write_capabilities_cooperatively_revoked"] is True
        and cert["native_wait_exit"]=="UNKNOWN" and cert["actual_optimizer_updates"]=="UNKNOWN"
        and cert["exact_old_kernel_listener_matches"]==[]
        and cert["external_NFS_writers_outside_observation"] is True,"Honest limited historical/current-writer root certificate")
    proof=cert["current_writer_observation"]
    require(proof["schema"]=="native_MLM_honest_current_writer_observation_v9_v2"
        and proof["historical_ids_complete"] is False and proof["historical_descendant_ids_lost"] is True
        and proof["native_wait_exit"]=="UNKNOWN" and proof["actual_optimizer_updates"]=="UNKNOWN"
        and [r["pid"] for r in proof["known_old_roots"]]==list(KNOWN_ROOTS)
        and [r["group"] for r in proof["known_original_groups"]]==list(KNOWN_GROUPS)
        and all(r["absent"] is True and r["method"]=="targeted os.kill(pid,0) ESRCH" for r in proof["known_old_roots"])
        and all(r["absent"] is True and r["method"]=="targeted os.killpg(group,0) ESRCH" for r in proof["known_original_groups"])
        and proof["current_same_uid_scan"]["complete"] is True
        and proof["current_same_uid_scan"]["stable_same_uid_membership"] is True
        and proof["current_same_uid_scan"]["current_matching_processes"]==[]
        and proof["current_same_uid_scan"]["unrelated_process_contents_recorded"] is False
        and proof["external_NFS_writers_outside_observation"] is True
        and proof["other_UID_and_remote_host_writers_outside_observation"] is True,
        "Exact known roots/groups and complete bounded empty current metadata observation")
    return cert
def read_certificate(record):
    require(set(record)=={"path","sha256"},"Exact root certificate ref")
    path=ROOT/record["path"]
    require(path.resolve()==path and path.is_relative_to(ROOT),"Canonical own root metadata")
    s=path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=65536,"Bounded single-link root certificate")
    raw=path.read_bytes()
    require(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==record["sha256"],"Bounded exact certificate bytes")
    return validate_static_certificate(json.loads(raw))
