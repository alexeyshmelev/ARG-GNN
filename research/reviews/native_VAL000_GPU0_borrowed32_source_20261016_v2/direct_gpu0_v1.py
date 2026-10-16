"""One native32 borrowed-inode GPU0 attachment, no dispatcher/controller replacement.

The existing controller remains the exclusive physical-lock owner. This adapter
borrows an authenticated descriptor to that inode and owns ONLY native32's fresh
logical lock/process tree. It never claims inheritance from the old controller.
"""
from pathlib import Path
import fcntl,hashlib,json,os,stat,sys,time
from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.common_v1 import *
SOURCE=DIRECT
CONTROLLER_PID=3349405
CONTROLLER_START=481281638
C3_JOB="C3-GNN-efficient-resume-GPU0-TRAIN-v1"
CURRENT_CAP={"path":"research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json","sha256":"4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870"}

def base_v1():return load(BASE_POOL,"_native32_original_pool_primitives")

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
    require(authority.get("schema")=="native32_fixed_GPU0_direct_attachment_authority_v1"
        and authority.get("root_authorizer")=="root_agent" and authority.get("status")=="active"
        and authority.get("borrowed_physical_inode_attachment_authorized") is True
        and authority.get("additional_old_controller_submissions_revoked_while_native32_attached") is True
        and authority.get("old_controller_preserved_until_native32_drained") is True
        and authority.get("controller_pid")==CONTROLLER_PID
        and authority.get("controller_start_ticks")==CONTROLLER_START
        and authority.get("controller_capability_ref")==CURRENT_CAP
        and authority.get("native32_job_id")==q["job_id"]
        and authority.get("C3_fixed_envelope_bytes")==26*2**30
        and authority.get("native32_max_cuda_reserved_bytes")==q["caps"]["max_cuda_reserved_bytes"]
        and authority.get("native32_non_Torch_margin_bytes")==2*2**30
        and authority.get("headroom_bytes")==4*2**30,
        "Exact ROOT fixed-job borrowed attachment/revocation authority")
    cap=read(CURRENT_CAP);row=base.proc_identity(CONTROLLER_PID)
    require(time.time()<cap["expires_unix"]
        and datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],
        "native32 actual authority remains inside original live-controller execution authority")
    require(row["uid"]==os.getuid() and row["start_ticks"]==CONTROLLER_START
        and row["state"] not in ("Z","X") and cap["gpu_uuid"]==GPU0_UUID
        and cap["physical_gpu"]==0,"Exact live known controller identity andGPU0")
    declared=cap["jobs"][C3_JOB]
    require(declared["cuda_cap_bytes"]==24*2**30 and declared["margin_mib"]==2048
        and declared["peak_mib"]==26*1024,"Exact unchanged C3 24+2GiB reservation")
    storage=read(cap["controller_storage_ref"])
    registry_path=local(storage["output_directory"]+"/registry.json")
    require(registry_path.is_file() and not registry_path.is_symlink()
        and registry_path.stat().st_size<=32*2**20,"Bounded current original registry")
    registry=json.loads(registry_path.read_bytes())
    require(registry["controller"]["pid"]==CONTROLLER_PID
        and registry["controller"]["start_ticks"]==CONTROLLER_START
        and registry["physical_gpu"]==0 and registry["lifecycle"]=="running"
        and registry["safety_error"] is None and not registry["ambiguous_descendants_or_lease_holders"]
        and registry.get("capability_ref")==CURRENT_CAP
        and -5<=time.time()-registry["updated_unix"]<=60,"Fresh exact original controller registry")
    active_jobs=[j for j in registry["jobs"] if j["state"] in ("queued","launching","running","draining")]
    c3_rows=[j for j in registry["jobs"] if j["logical_job"]==C3_JOB]
    require(len(c3_rows)==1 and all(j["logical_job"]==C3_JOB for j in active_jobs)
        and c3_rows[0]["requested"]==declared
        and c3_rows[0]["state"] in ("running","draining","succeeded"),
        "Only original C3 may run/drain/finish; no native/other queued or live work")
    if c3_rows[0]["state"]=="succeeded":
        require(c3_rows[0].get("owned_drain_observed") is True
            and c3_rows[0].get("exit_code")==0,"Actual successful original C3 drain")
    return cap,registry,c3_rows[0],authority

def capacity_v1(q,base,*,worker_pid=None):
    cap,registry,c3,authority=controller_v1(q,base)
    rows=base.real_inventory()
    rows=[r for r in rows if r["index"]==0]
    require(len(rows)==1 and rows[0]["uuid"]==GPU0_UUID,"ActualGPU0 inventory UUID")
    gpu=rows[0];known={}
    for saved in c3["process_identities"]:
        try:now=base.proc_identity(saved["pid"])
        except (FileNotFoundError,ProcessLookupError):continue
        if now["uid"]==saved["uid"]==os.getuid() and now["start_ticks"]==saved["start_ticks"] and now["state"] not in ("Z","X"):
            known[now["pid"]]="C3"
    c3_mib=native32_mib=0;external=[]
    for app in gpu["compute"]:
        if app["pid"] in known:
            require(app["name"]=="AI Auto Research Pipeline Test","KnownC3 actual CUDA title")
            c3_mib+=app["used_mib"]
        elif worker_pid is not None and app["pid"]==worker_pid:
            require(app["name"]=="AI Auto Research Pipeline Test","native32 actual CUDA title")
            native32_mib+=app["used_mib"]
        else:external.append(dict(app))
    require(c3_mib<=26*1024,"C3 actual usage inside original26GiB envelope")
    peak_mib=(q["caps"]["max_cuda_reserved_bytes"]+2*2**30+2**20-1)//2**20
    require(native32_mib<=peak_mib,"native32 actual usage inside cap plus2GiB margin")
    external_mib=max(sum(a["used_mib"] for a in external),
        gpu["used_mib"]-c3_mib-native32_mib,0)
    projected=26*1024+peak_mib+external_mib+4096
    free_required=max(0,26*1024-c3_mib)+max(0,peak_mib-native32_mib)+4096
    require(projected<=gpu["total_mib"] and gpu["total_mib"]-gpu["used_mib"]>=free_required,
        "Actual external+all unfilled C3/native32 envelopes+4GiB headroom fitGPU0")
    return dict(schema="native32_fixed_GPU0_capacity_sample_v1",observed_unix=time.time(),
        GPU0=gpu,C3_fixed_envelope_mib=26*1024,native32_cap_plus_margin_mib=peak_mib,
        C3_observed_mib=c3_mib,native32_observed_mib=native32_mib,external_compute=external,
        external_and_driver_mib=external_mib,projected_mib=projected,
        actual_free_required_mib=free_required,headroom_mib=4096,
        registry_updated_unix=registry["updated_unix"],controller_capability_ref=CURRENT_CAP)

def attach_v1(spec,q):
    require(spec["phase"]=="mlm","Only fixed TRAIN uses GPU0 attachment")
    require(q["caps"]["max_cuda_reserved_bytes"]==32*2**30,"Exact native32GiB allocator cap")
    base=base_v1();physical=local(".runtime/locks/gpu_0.lock")
    logical_path=local(spec["worker_auxiliary_scope"]+"/native32_GPU0.logical.lock")
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
            ticket=dict(schema="native32_borrowed_GPU0_attachment_v1",job_id=spec["job_id"],
                physical_gpu=0,gpu_uuid=GPU0_UUID,borrowed_physical_inode=True,
                physical_exclusive_lock_owner="unchanged_original_controller",
                controller_pid=CONTROLLER_PID,controller_start_ticks=CONTROLLER_START,
                physical_inode=st.st_ino,physical_device=st.st_dev,
                physical_lease_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                physical_lease_metadata=metadata,logical_lock_path=str(logical_path),
                logical_inode=os.fstat(logical_fd).st_ino,logical_device=os.fstat(logical_fd).st_dev,
                native32_logical_lock_owned_by_supervisor=True,capacity_at_attachment=capacity,
                direct_attachment_authority_ref=q["direct_attachment_authority_ref"])
            return physical_fd,logical_fd,ticket
        except BaseException:
            for fd in (physical_fd,logical_fd):
                if fd is not None:os.close(fd)
            raise

def install_before_torch_v1(spec,q,ticket):
    require(not any(n in sys.modules for n in ("torch","torch_geometric")),"Direct lease/title install before science")
    g=load(GPU0_GUARD,"_native32_original_GPU0_guard");native=g.install_before_torch_v1()
    original=native.validate_lease;base=base_v1()
    logical_fd=int(os.environ["native32_LOGICAL_LOCK_FD"])
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
            ==(ticket["logical_device"],ticket["logical_inode"]),"Actual inherited native32 logical lock descriptor")
        held_writable_probe_v1(path)
        return dict(lease,borrowed_physical_inode=True,
            physical_exclusive_lock_owned_by_current_controller=True,
            native32_logical_lock_inode=ticket["logical_inode"],
            native32_direct_attachment_source_ref=ref(SOURCE))
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
