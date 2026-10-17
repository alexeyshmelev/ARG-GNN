"""One H058 borrowed-inode GPU0 attachment, no dispatcher/controller replacement.

The existing controller remains the exclusive physical-lock owner. This adapter
borrows an authenticated descriptor to that inode and owns ONLY H058's fresh
logical lock/process tree. It never claims inheritance from the old controller.
"""
from pathlib import Path
import fcntl,hashlib,json,os,stat,sys,time
from research.reviews.H058_GPU0_full_training_successor_source_20261017_v4.common_v1 import *
SOURCE=DIRECT
CONTROLLER_PID=3349405
CONTROLLER_START=481281638
C3_JOB="C3-GNN-efficient-resume-GPU0-TRAIN-v1"
CURRENT_CAP={"path":"research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json","sha256":"4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870"}

def base_v1():return load(BASE_POOL,"_H058_original_pool_primitives")

def held_writable_probe_v1(path):
    # NFS flock requires a writable independently opened descriptor. EBADF,
    # EACCES, ENOLCK and every error other than EAGAIN/EWOULDBLOCK propagate.
    with path.open("r+") as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        else:
            fcntl.flock(probe,fcntl.LOCK_UN)
            raise RuntimeError("Expected physical/logical exclusion lock is unheld")

def controller_v1(q,base):
    authority=read(q["direct_attachment_authority_ref"]);active(authority)
    require(authority.get("schema")=="H058_fixed_GPU0_direct_attachment_authority_v1"
        and authority.get("root_authorizer")=="root_agent" and authority.get("status")=="active"
        and authority.get("borrowed_physical_inode_attachment_authorized") is True
        and authority.get("additional_old_controller_submissions_revoked_while_H058_attached") is True
        and authority.get("old_controller_preserved_until_H058_drained") is True
        and authority.get("controller_pid")==CONTROLLER_PID
        and authority.get("controller_start_ticks")==CONTROLLER_START
        and authority.get("controller_capability_ref")==CURRENT_CAP
        and authority.get("H058_job_id")==q["job_id"]
        and authority.get("C3_fixed_envelope_bytes")==26*2**30
        and authority.get("H058_max_cuda_reserved_bytes")==28*2**30
        and authority.get("H058_non_Torch_margin_bytes")==2*2**30
        and authority.get("headroom_bytes")==4*2**30
        and authority.get("all_old_native_launch_resume_write_routes_permanently_revoked") is True
        and authority.get("native_resource_disposition_ref")==q["native_resource_disposition_ref"],
        "Fresh exact ROOT borrowed attachment after genuine native terminal and owned drain")
    native=validate_native_resource_disposition_v1(q,fresh=False)
    for saved in native["native_process_identities"]:
        require(set(saved)=={"pid","start_ticks","uid"},"Exact known native PID/start/UID identity")
        try:now=base.proc_identity(saved["pid"])
        except (FileNotFoundError,ProcessLookupError):continue
        require(not(now["uid"]==saved["uid"] and now["start_ticks"]==saved["start_ticks"]
            and now["state"] not in ("Z","X")),"Known native identity still live; never attach H058")
    for group in native["native_process_groups"]:
        try:os.killpg(group,0)
        except ProcessLookupError:pass
        else:raise ValueError("Known prior native process group still exists; never attach H058")
    cap=read(CURRENT_CAP);row=base.proc_identity(CONTROLLER_PID)
    require(time.time()<cap["expires_unix"]
        and datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],
        "Fresh H058 interval remains within preserved controller authority")
    require(row["uid"]==os.getuid() and row["start_ticks"]==CONTROLLER_START
        and row["state"] not in ("Z","X") and cap["gpu_uuid"]==GPU0_UUID
        and cap["physical_gpu"]==0,"Current exact live preserved GPU0 controller")
    declared=cap["jobs"][C3_JOB]
    require(declared["cuda_cap_bytes"]==24*2**30 and declared["margin_mib"]==2048
        and declared["peak_mib"]==26*1024,"Retain entire C3 24+2GiB envelope")
    storage=read(cap["controller_storage_ref"])
    registry_path=local(storage["output_directory"]+"/registry.json")
    require(registry_path.is_file() and not registry_path.is_symlink()
        and registry_path.stat().st_size<=32*2**20,"Bounded unchanged controller registry")
    registry=json.loads(registry_path.read_bytes())
    terminal=read(q["controller_terminal_qualification_ref"]);active(terminal)
    require(terminal.get("schema")=="H058_ROOT_preserved_terminal_controller_v1"
        and terminal.get("root_authorizer")=="root_agent"
        and terminal.get("controller_pid")==CONTROLLER_PID
        and terminal.get("controller_start_ticks")==CONTROLLER_START
        and terminal.get("controller_capability_ref")==CURRENT_CAP
        and terminal.get("all_old_controller_submissions_revoked") is True
        and terminal.get("unknown_original_outer74136_preserved") is True
        and terminal.get("controller_preserved") is True,
        "Actual future ROOT terminal-table evidence, never controller restart or history rewrite")
    audited=read(terminal["registry_snapshot_ref"])
    require(registry["controller"]["pid"]==CONTROLLER_PID
        and registry["controller"]["start_ticks"]==CONTROLLER_START
        and registry["physical_gpu"]==0 and registry["lifecycle"]=="running"
        and registry["safety_error"] is None and not registry["ambiguous_descendants_or_lease_holders"]
        and registry.get("capability_ref")==CURRENT_CAP
        and registry["jobs"]==audited["jobs"],
        "Exact terminal job table; timestamp is historical, not a refreshed heartbeat")
    active_jobs=[j for j in registry["jobs"] if j["state"] in ("queued","launching","running","draining")]
    c3=[j for j in registry["jobs"] if j["logical_job"]==C3_JOB]
    require(not active_jobs and len(c3)==1 and c3[0]["requested"]==declared
        and c3[0]["state"]=="failed" and c3[0].get("exit_code")==1
        and c3[0].get("leader_wait_status_known") is True and c3[0].get("owned_drain_observed") is True,
        "Preserve genuine recorded C3 failed exit1/wait/drain and fixed26GiB reservation")
    return cap,registry,c3[0],authority

class GPUCapacityFailureV3(ValueError):
    def __init__(self,message,sample):
        super().__init__(message);self.capacity_sample=sample

def bounded_capacity_failure_v3(gpu,observed_unix,metrics,error):
    """Bounded actual GPU0 metadata only; capacity decisions still use the full inventory."""
    rows=[]
    for app in gpu["compute"][:8]:
        name=str(app["name"])
        rows.append(dict(pid=app["pid"],used_mib=app["used_mib"],name=name[:96],name_truncated=len(name)>96))
    result=dict(schema="H058_bounded_failing_GPU0_capacity_sample_v3",observed_unix=observed_unix,
        physical_gpu=gpu["index"],gpu_uuid=gpu["uuid"],total_mib=gpu["total_mib"],used_mib=gpu["used_mib"],
        compute_count=len(gpu["compute"]),retained_compute=rows,compute_rows_truncated=len(gpu["compute"])>len(rows),
        computed_metrics=metrics,capacity_checks_passed=False,original_failure_type=type(error).__name__,
        original_failure=str(error)[:384],record_limit_bytes=4096,
        raw_capacity_decisions_used_complete_inventory=True)
    while len(json.dumps(result,sort_keys=True,ensure_ascii=True,allow_nan=False).encode())>4096:
        if result["retained_compute"]:
            result["retained_compute"].pop();result["compute_rows_truncated"]=True
        elif len(result["original_failure"])>32:result["original_failure"]=result["original_failure"][:32]
        else:raise ValueError("Unexpected capacity metadata exceeds bounded operational record")
    return result

def capacity_v1(q,base,*,worker_pid=None):
    observed_gpu=None;observed_unix=None;metrics=dict(classification_complete=False)
    try:
        cap,registry,c3,authority=controller_v1(q,base)
        rows=base.real_inventory()
        rows=[r for r in rows if r["index"]==0]
        require(len(rows)==1 and rows[0]["uuid"]==GPU0_UUID,"ActualGPU0 inventory UUID")
        gpu=rows[0];known={}
        native=validate_native_resource_disposition_v1(q,fresh=False)
        native_pids={r["pid"] for r in native["native_process_identities"]}
        require(not any(app["pid"] in native_pids for app in gpu["compute"]),
            "Prior native roster GPU0 PID contexts must be absent; reused numeric PIDs conservatively block")
        observed_gpu=gpu;observed_unix=time.time()
        for saved in c3["process_identities"]:
            try:now=base.proc_identity(saved["pid"])
            except (FileNotFoundError,ProcessLookupError):continue
            if now["uid"]==saved["uid"]==os.getuid() and now["start_ticks"]==saved["start_ticks"] and now["state"] not in ("Z","X"):
                known[now["pid"]]="C3"
        c3_mib=h058_mib=0;external=[]
        for app in gpu["compute"]:
            if app["pid"] in known:
                require(app["name"]=="AI Auto Research Pipeline Test","KnownC3 actual CUDA title")
                c3_mib+=app["used_mib"]
            elif worker_pid is not None and app["pid"]==worker_pid:
                require(app["name"]=="AI Auto Research Pipeline Test","H058 actual CUDA title")
                h058_mib+=app["used_mib"]
            else:external.append(dict(app))
        metrics.update(classification_complete=True,C3_observed_mib=c3_mib,H058_observed_mib=h058_mib)
        require(c3_mib<=26*1024,"C3 actual usage inside original26GiB envelope")
        peak_mib=(q["caps"]["max_cuda_reserved_bytes"]+2*2**30+2**20-1)//2**20
        metrics.update(H058_cap_plus_margin_mib=peak_mib)
        require(h058_mib<=peak_mib,"H058 actual usage inside cap plus2GiB margin")
        external_mib=max(sum(a["used_mib"] for a in external),
            gpu["used_mib"]-c3_mib-h058_mib,0)
        projected=26*1024+peak_mib+external_mib+4096
        free_required=max(0,26*1024-c3_mib)+max(0,peak_mib-h058_mib)+4096
        metrics.update(external_and_driver_mib=external_mib,projected_mib=projected,actual_free_required_mib=free_required)
        require(projected<=gpu["total_mib"] and gpu["total_mib"]-gpu["used_mib"]>=free_required,
            "Actual external+all unfilled C3/H058 envelopes+4GiB headroom fitGPU0")
        return dict(schema="H058_fixed_GPU0_capacity_sample_v1",observed_unix=time.time(),
            GPU0=gpu,C3_fixed_envelope_mib=26*1024,H058_cap_plus_margin_mib=peak_mib,
            C3_observed_mib=c3_mib,H058_observed_mib=h058_mib,external_compute=external,
            external_and_driver_mib=external_mib,projected_mib=projected,
            actual_free_required_mib=free_required,headroom_mib=4096,
            registry_updated_unix=registry["updated_unix"],controller_capability_ref=CURRENT_CAP)
    except Exception as exc:
        if observed_gpu is None:raise
        raise GPUCapacityFailureV3(str(exc),bounded_capacity_failure_v3(observed_gpu,observed_unix,metrics,exc)) from exc

def attach_v1(spec,q):
    require(spec["phase"]=="train","Only fixed TRAIN uses GPU0 attachment")
    base=base_v1();physical=local(".runtime/locks/gpu_0.lock")
    logical_path=local(spec["worker_auxiliary_scope"]+"/H058_GPU0.logical.lock")
    physical_fd=logical_fd=None
    with base.lock_file(ROOT,"scheduler.lock") as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        try:
            cap,registry,c3,authority=controller_v1(q,base)
            physical_fd=os.open(physical,os.O_RDWR|os.O_NOFOLLOW)
            st=os.fstat(physical_fd);named=physical.lstat()
            require(stat.S_ISREG(st.st_mode) and st.st_nlink==1
                and (st.st_dev,st.st_ino)==(named.st_dev,named.st_ino),"Original canonical physical inode")
            raw_bytes=os.pread(physical_fd,65536,0)
            require(len(raw_bytes)==st.st_size and hashlib.sha256(raw_bytes).hexdigest()==registry["lease_sha256"],
                "Original registry-bound physical lease bytes")
            metadata=json.loads(raw_bytes)
            require(metadata==registry["lease_metadata"] and metadata["launcher_pid"]==CONTROLLER_PID
                and metadata["gpu"]==0 and metadata["uuid"]==GPU0_UUID,"Original actual physical lease identity")
            held_writable_probe_v1(physical)
            logical_fd=os.open(logical_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            fcntl.flock(logical_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            capacity=capacity_v1(q,base)
            ticket=dict(schema="H058_borrowed_GPU0_attachment_v1",job_id=spec["job_id"],
                physical_gpu=0,gpu_uuid=GPU0_UUID,borrowed_physical_inode=True,
                physical_exclusive_lock_owner="unchanged_original_controller",
                controller_pid=CONTROLLER_PID,controller_start_ticks=CONTROLLER_START,
                physical_inode=st.st_ino,physical_device=st.st_dev,
                physical_lease_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                physical_lease_metadata=metadata,logical_lock_path=str(logical_path),
                logical_inode=os.fstat(logical_fd).st_ino,logical_device=os.fstat(logical_fd).st_dev,
                H058_logical_lock_owned_by_supervisor=True,capacity_at_attachment=capacity,
                direct_attachment_authority_ref=q["direct_attachment_authority_ref"])
            return physical_fd,logical_fd,ticket
        except BaseException:
            for fd in (physical_fd,logical_fd):
                if fd is not None:os.close(fd)
            raise

def install_before_torch_v1(spec,q,ticket):
    require(not any(n in sys.modules for n in ("torch","torch_geometric")),"Direct lease/title install before science")
    g=load(GPU0_GUARD,"_H058_original_GPU0_guard");native=g.install_before_torch_v1()
    original=native.validate_lease;base=base_v1()
    logical_fd=int(os.environ["H058_LOGICAL_LOCK_FD"])
    physical_fd=int(os.environ["ARGGNN_GPU_LEASE_FD"])
    def validate_lease(environ=None):
        lease=original(environ)
        require(lease["lease_fd"]==physical_fd
            and lease["lease_inode"]==ticket["physical_inode"]
            and lease["lease_device"]==ticket["physical_device"]
            and lease["lease_metadata_sha256"]==ticket["physical_lease_sha256"]
            and lease["launcher_metadata"]==ticket["physical_lease_metadata"],
            "Stable authenticated borrowed physical inode, never controller inheritance")
        controller_v1(q,base)
        st=os.fstat(logical_fd);path=Path(ticket["logical_lock_path"]);named=path.lstat()
        require(stat.S_ISREG(st.st_mode) and (st.st_dev,st.st_ino)==(named.st_dev,named.st_ino)
            ==(ticket["logical_device"],ticket["logical_inode"]),"Actual inherited H058 logical lock descriptor")
        held_writable_probe_v1(path)
        # Reuse the existing current inventory/capacity boundary. configure_fit
        # invokes this wrapped lease function before its first CUDA call.
        capacity_v1(q,base,worker_pid=os.getpid())
        return dict(lease,borrowed_physical_inode=True,
            physical_exclusive_lock_owned_by_current_controller=True,
            H058_logical_lock_inode=ticket["logical_inode"],
            H058_direct_attachment_source_ref=ref(SOURCE))
    native.validate_lease=validate_lease
    validate_lease();native.apply_and_validate_title()
    return native

def check_attachment_v1(q,ticket,physical_fd,logical_fd):
    controller_v1(q,base_v1())
    for fd,path,device,inode in (
        (physical_fd,local(".runtime/locks/gpu_0.lock"),ticket["physical_device"],ticket["physical_inode"]),
        (logical_fd,Path(ticket["logical_lock_path"]),ticket["logical_device"],ticket["logical_inode"])):
        opened=os.fstat(fd);named=path.lstat()
        require(stat.S_ISREG(opened.st_mode) and (opened.st_dev,opened.st_ino)
            ==(named.st_dev,named.st_ino)==(device,inode),"Stable actual physical/logical inode identity")
        held_writable_probe_v1(path)
    contents=os.pread(physical_fd,65536,0)
    require(len(contents)==os.fstat(physical_fd).st_size
        and hashlib.sha256(contents).hexdigest()==ticket["physical_lease_sha256"],
        "Actual original physical lease bytes unchanged")
