"""Fresh downstream borrowed descriptor to the NEW post-drain pool owner only."""
from pathlib import Path
import fcntl,hashlib,json,os,stat,sys,time
from .native_common_v1 import *
from . import pool_contract_v1 as pc
SOURCE=HERE+"/downstream_direct_gpu0_v1.py"
_CANCEL_CHECK=lambda:None
def install_cancellation_check_v1(check):
    global _CANCEL_CHECK
    require(callable(check),"Finite actual cancellation callback");_CANCEL_CHECK=check

def base_v1():return load(BASE_POOL,"_downstream_postdrain_original_pool_primitives")

def controller_v1(q,base):
    _CANCEL_CHECK()
    cap,registry=pc.registry_v1(q["postdrain_pool_capability_ref"])
    authority=read(q["direct_attachment_authority_ref"]);active(authority)
    require(authority.get("schema")=="native_parent_new_pool_attachment_authority_v1"
        and authority.get("root_authorizer")=="root_agent" and authority.get("status")=="active"
        and authority.get("pool_capability_ref")==q["postdrain_pool_capability_ref"]
        and authority.get("controller_identity")==registry["controller"] and authority.get("job_id")==q["job_id"]
        and authority.get("parent_id")==q["parent_id"] and authority.get("borrowed_physical_inode_attachment_authorized") is True
        and authority.get("pool_owner_preserved_until_all_new_borrowers_drained") is True
        and authority.get("adapter_manifest_ref")==ref(MANIFEST),"Exact new parent/native borrowed pool authority")
    require(datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],"Native lifetime inside pool authority")
    rows=[j for j in registry["jobs"]if j["logical_job"]==q["job_id"]]
    require(len(rows)==1 and rows[0]["state"]in("launching","running","draining")
        and rows[0]["requested"]["worker_kind"]=="native" and rows[0]["requested"]["parent_id"]==q["parent_id"],"Actual reserved native slot before inner fork")
    return cap,registry,rows[0],authority

def capacity_v1(q,base,*,worker_pid=None):
    observed_gpu=None;observed_unix=None;metrics=dict(classification_complete=False)
    try:
        cap,registry,job,authority=controller_v1(q,base)
        rows=[r for r in base.real_inventory()if r["index"]==0]
        require(len(rows)==1 and rows[0]["uuid"]==GPU0_UUID,"Only actual new-pool GPU0 UUID")
        gpu=observed_gpu=rows[0];observed_unix=time.time();active_jobs=[j for j in registry["jobs"]if j["state"]in("launching","running","draining")]
        owners={};usage={j["logical_job"]:0 for j in active_jobs};external=[]
        for j in active_jobs:
            for saved in j["process_identities"]:
                try:now=base.proc_identity(saved["pid"])
                except(FileNotFoundError,ProcessLookupError):continue
                if now["uid"]==saved["uid"]==os.getuid() and now["start_ticks"]==saved["start_ticks"] and now["state"]not in("Z","X"):
                    require(saved["pid"]not in owners or owners[saved["pid"]]==j["logical_job"],"No ambiguous peer ownership")
                    owners[saved["pid"]]=j["logical_job"]
        if worker_pid is not None:
            current=base.proc_identity(worker_pid)
            require(current["uid"]==os.getuid()and current["state"]not in("Z","X"),"Actual owned inner worker")
            owners[worker_pid]=q["job_id"]
        for app in gpu["compute"]:
            name=owners.get(app["pid"])
            if name is None:external.append(dict(app))
            else:
                require(app["name"]=="AI Auto Research Pipeline Test","Exact owned CUDA title")
                usage[name]+=app["used_mib"]
        reserved=sum(j["requested"]["peak_mib"]for j in active_jobs)
        require(all(usage[j["logical_job"]]<=j["requested"]["peak_mib"]for j in active_jobs),"Each actual peer inside allocator+margin")
        external_mib=max(sum(a["used_mib"]for a in external),gpu["used_mib"]-sum(usage.values()),0)
        projected=reserved+external_mib+4096
        free_required=sum(max(0,j["requested"]["peak_mib"]-usage[j["logical_job"]])for j in active_jobs)+4096
        metrics.update(classification_complete=True,active_reservation_mib=reserved,owned_usage_mib=usage,external_and_driver_mib=external_mib,projected_mib=projected,actual_free_required_mib=free_required,C3_compute_envelope_mib=0)
        require(projected<=gpu["total_mib"]and gpu["total_mib"]-gpu["used_mib"]>=free_required,"All active peer envelopes+external+4GiB headroom fit")
        return dict(schema="downstream_postdrain_two_slot_capacity_sample_v1",observed_unix=time.time(),GPU0=gpu,metrics=metrics,pool_capability_ref=q["postdrain_pool_capability_ref"],headroom_mib=4096,C3_compute_envelope_mib=0)
    except Exception as exc:
        if observed_gpu is None:raise
        raise GPUCapacityFailureV3(str(exc),bounded_capacity_failure_v3(observed_gpu,observed_unix,metrics,exc))from exc


def held_writable_probe_v1(path):
    # NFS flock requires a writable independently opened descriptor. EBADF,
    # EACCES, ENOLCK and every error other than EAGAIN/EWOULDBLOCK propagate.
    with path.open("r+") as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        else:
            fcntl.flock(probe,fcntl.LOCK_UN)
            raise RuntimeError("Expected physical/logical exclusion lock is unheld")

class GPUCapacityFailureV3(ValueError):
    def __init__(self,message,sample):
        super().__init__(message);self.capacity_sample=sample

def bounded_capacity_failure_v3(gpu,observed_unix,metrics,error):
    """Bounded actual GPU0 metadata only; capacity decisions still use the full inventory."""
    rows=[]
    for app in gpu["compute"][:8]:
        name=str(app["name"])
        rows.append(dict(pid=app["pid"],used_mib=app["used_mib"],name=name[:96],name_truncated=len(name)>96))
    result=dict(schema="downstream_bounded_failing_GPU0_capacity_sample_v3",observed_unix=observed_unix,
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

def attach_v1(spec,q):
    require(spec["phase"]in("contrastive","embedding_contrastive_train","embedding_contrastive_val","embedding_query_paths_train","embedding_query_paths_val","RAM_retrieve_and_seal"),"Only fixed TRAIN uses GPU0 attachment")
    base=base_v1();physical=local(".runtime/locks/gpu_0.lock")
    logical_path=local(spec["worker_auxiliary_scope"]+"/downstream_GPU0.logical.lock")
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
            require(metadata==registry["lease_metadata"] and metadata["launcher_pid"]==registry["controller"]["pid"]
                and metadata["gpu"]==0 and metadata["uuid"]==GPU0_UUID,"Original actual physical lease identity")
            held_writable_probe_v1(physical)
            logical_fd=os.open(logical_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            fcntl.flock(logical_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            capacity=capacity_v1(q,base)
            ticket=dict(schema="downstream_borrowed_GPU0_attachment_v1",job_id=spec["job_id"],
                physical_gpu=0,gpu_uuid=GPU0_UUID,borrowed_physical_inode=True,
                physical_exclusive_lock_owner="new_postdrain_exclusive_controller",
                controller_pid=registry["controller"]["pid"],controller_start_ticks=registry["controller"]["start_ticks"],
                physical_inode=st.st_ino,physical_device=st.st_dev,
                physical_lease_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                physical_lease_metadata=metadata,logical_lock_path=str(logical_path),
                logical_inode=os.fstat(logical_fd).st_ino,logical_device=os.fstat(logical_fd).st_dev,
                downstream_logical_lock_owned_by_supervisor=True,capacity_at_attachment=capacity,
                direct_attachment_authority_ref=q["direct_attachment_authority_ref"])
            return physical_fd,logical_fd,ticket
        except BaseException:
            for fd in (physical_fd,logical_fd):
                if fd is not None:os.close(fd)
            raise

def install_before_torch_v1(spec,q,ticket):
    require(not any(n in sys.modules for n in ("torch","torch_geometric")),"Direct lease/title install before science")
    g=load(GPU0_GUARD,"_downstream_original_GPU0_guard");native=g.install_before_torch_v1()
    original=native.validate_lease;base=base_v1()
    logical_fd=int(os.environ["downstream_LOGICAL_LOCK_FD"])
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
            ==(ticket["logical_device"],ticket["logical_inode"]),"Actual inherited downstream logical lock descriptor")
        held_writable_probe_v1(path)
        return dict(lease,borrowed_physical_inode=True,
            physical_exclusive_lock_owned_by_current_controller=True,
            downstream_logical_lock_inode=ticket["logical_inode"],
            downstream_direct_attachment_source_ref=ref(SOURCE))
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
