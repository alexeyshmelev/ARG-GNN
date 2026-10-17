"""One downstream borrowed-inode GPU0 attachment, no dispatcher/controller replacement.

The existing controller remains the exclusive physical-lock owner. This adapter
borrows an authenticated descriptor to that inode and owns ONLY downstream's fresh
logical lock/process tree. It never claims inheritance from the old controller.
"""
from pathlib import Path
import errno,fcntl,hashlib,json,os,select,socket,stat,struct,sys,time
from research.reviews.native_saved_MLM_current_Census64_integration_source_20261017_v2.contract_v1 import *
from research.reviews.native_saved_MLM_current_Census64_integration_source_20261017_v2 import contract_v1 as c
SOURCE=DIRECT
CONTROLLER_PID=CONTROLLER_START=C3_JOB=CURRENT_CAP=None
C3_ENVELOPE_MIB=0
BINDING_REF=None
ENDPOINT_SOURCE_REF={'path': 'research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py', 'sha256': 'dca4230a3d79540b947fa5488c6a49a885970db367b9955f78c343b8305b398f'}
CURRENT_C3_CAP={"path":"research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json","sha256":"4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870"}

def configure_controller_binding_v1(q):
    global CONTROLLER_PID,CONTROLLER_START,C3_JOB,CURRENT_CAP,C3_ENVELOPE_MIB,BINDING_REF
    record=q["controller_binding_ref"]
    require(BINDING_REF is None or BINDING_REF==record,"One exact independently reviewed ROOT controller binding")
    value=read(record);active(value)
    require(value["schema"]=="native_VAL000_downstream_GPU0_controller_binding_v1"
        and value["source_review_passed"] is True and value["endpoint_source_ref"]==ENDPOINT_SOURCE_REF
        and value["physical_gpu"]==0 and value["GPU_UUID"]==GPU0_UUID
        and value["additional_controller_and_other_borrower_submissions_revoked"] is True
        and value["controller_preserved_until_downstream_drain"] is True,
        "Exact explicit ROOT borrowed controller/same-protocol successor authority")
    raw(value["independent_source_review_ref"]);raw(ENDPOINT_SOURCE_REF)
    require(value["route"] in ("current_C3","reviewed_same_protocol_successor"),"Only explicit reviewed routes, no guessed controller")
    if value["route"]=="current_C3":
        require(value["controller_pid"]==3349405 and value["controller_start_ticks"]==481281638
            and value["controller_capability_ref"]==CURRENT_C3_CAP
            and value["protected_job"]=="C3-GNN-efficient-resume-GPU0-TRAIN-v1"
            and value["protected_envelope_bytes"]==26*2**30,"Unchanged actual current C3 controller, no cap/source edit")
    else:
        require(value["protected_job"] is None and value["protected_envelope_bytes"]==0
            and value["successor_capability_explicitly_ROOT_authorized"] is True,
            "Successor is separately ROOT-bound and reviewed, held with no undeclared project jobs")
    require(type(value["controller_pid"]) is int and value["controller_pid"]>0
        and type(value["controller_start_ticks"]) is int and value["controller_start_ticks"]>0,"Exact future controller PID/start")
    cap=read(value["controller_capability_ref"])
    require(cap["source_refs"].get(ENDPOINT_SOURCE_REF["path"])==ENDPOINT_SOURCE_REF,
        "Actual controller capability authenticates unchanged installed endpoint implementation")
    BINDING_REF=record;CONTROLLER_PID=value["controller_pid"];CONTROLLER_START=value["controller_start_ticks"]
    CURRENT_CAP=value["controller_capability_ref"];C3_JOB=value["protected_job"];C3_ENVELOPE_MIB=value["protected_envelope_bytes"]//2**20
    return value

LAST_CONTROLLER_LIVENESS=None
CANCEL_CHECK=None
RPC_SECONDS=30.

def install_cancellation_check_v1(callback):
    global CANCEL_CHECK
    require(callable(callback),"Only an interpreter-local parent cancellation/deadline callback")
    CANCEL_CHECK=callback

def _rpc_tick_v1(deadline):
    if CANCEL_CHECK is not None:CANCEL_CHECK()
    require(time.monotonic()<deadline,"No fresh authenticated controller reply before bounded deadline")

def _fresh_status_v1(q,base,cap):
    # Exact unchanged status protocol. No submit/cancel/shutdown or registry write.
    token=hashlib.sha256(base.encoded(cap["plan_ref"])).hexdigest()[:16]
    endpoint="\0arggnn_gpu0_declared_"+cap["pool"]+"_"+token
    before=base.proc_identity(CONTROLLER_PID)
    require(before["pid"]==CONTROLLER_PID and before["uid"]==os.getuid()
        and before["start_ticks"]==CONTROLLER_START and before["state"] not in ("Z","X"),"Exact live controller before status challenge")
    started_wall=time.time();started_mono=time.monotonic()
    deadline=min(started_mono+RPC_SECONDS,work_deadline_monotonic_v1())
    payload=base.encoded({"v":1,"op":"status"})+b"\n";reply=bytearray();offset=0
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
        client.setblocking(False)
        code=client.connect_ex(endpoint)
        require(code in (0,errno.EINPROGRESS,errno.EALREADY,errno.EWOULDBLOCK),"Existing exact controller abstract endpoint reachable")
        while True:
            _rpc_tick_v1(deadline)
            _,ready,_=select.select([], [client], [], min(.2,max(0.,deadline-time.monotonic())))
            if ready:
                require(client.getsockopt(socket.SOL_SOCKET,socket.SO_ERROR)==0,"Controller connect completed")
                break
        peer=struct.unpack("3i",client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
        require(peer[0]==CONTROLLER_PID and peer[1]==os.getuid(),"Actual Unix peer PID/UID matches original controller")
        while offset<len(payload):
            _rpc_tick_v1(deadline)
            _,ready,_=select.select([], [client], [], min(.2,max(0.,deadline-time.monotonic())))
            if ready:
                try:n=client.send(payload[offset:])
                except BlockingIOError:continue
                require(n>0,"Complete exact status request");offset+=n
        while b"\n" not in reply:
            _rpc_tick_v1(deadline)
            ready,_,_=select.select([client], [], [], min(.2,max(0.,deadline-time.monotonic())))
            if ready:
                try:block=client.recv(65536)
                except BlockingIOError:continue
                require(bool(block),"Complete real controller status reply")
                reply.extend(block);require(len(reply)<=base.MAX_REPLY,"Original bounded reply size")
    _rpc_tick_v1(deadline)
    line,tail=bytes(reply).split(b"\n",1);require(not tail,"Exactly one real status response")
    packet=json.loads(line);require(packet.get("ok") is True,"Controller returned actual status success")
    answer=packet["result"];after=base.proc_identity(CONTROLLER_PID);ended=time.time()
    require(after["pid"]==before["pid"] and after["uid"]==before["uid"]
        and after["start_ticks"]==before["start_ticks"]==CONTROLLER_START and after["state"] not in ("Z","X"),
        "Same live controller process after actual response")
    require(answer.get("socket")==endpoint and started_wall-5<=answer["updated_unix"]<=ended+5
        and -5<=ended-answer["updated_unix"]<=60,"Real status timestamp generated during exchange and original60s freshness")
    return answer,dict(method="authenticated_existing_controller_status_RPC",protocol={"v":1,"op":"status"},
        peer_pid=peer[0],peer_uid=peer[1],peer_gid=peer[2],controller_start_ticks=CONTROLLER_START,
        started_unix=started_wall,received_unix=ended,response_updated_unix=answer["updated_unix"],
        elapsed_seconds=time.monotonic()-started_mono,bounded_seconds=RPC_SECONDS,
        disk_threshold_seconds_unchanged=60,response_freshness_threshold_seconds=60,
        registry_file_modified=False,current_controller_or_guard_modified=False)

def _registry_v1(q,base,cap):
    storage=read(cap["controller_storage_ref"])
    registry_path=local(storage["output_directory"]+"/registry.json")
    require(registry_path.is_file() and not registry_path.is_symlink()
        and registry_path.stat().st_size<=32*2**20,"Bounded current original registry")
    registry=json.loads(registry_path.read_bytes())
    global LAST_CONTROLLER_LIVENESS
    now=time.time();age=now-registry["updated_unix"]
    if -5<=age<=60:
        LAST_CONTROLLER_LIVENESS=dict(method="original_bounded_disk_registry",registry_age_seconds=age,
            disk_threshold_seconds=60,controller_reply_requested=False)
        return registry
    answer,proof=_fresh_status_v1(q,base,cap)
    LAST_CONTROLLER_LIVENESS=proof
    return answer

def base_v1():return load(BASE_POOL,"_downstream_original_pool_primitives")

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
    active_amendment_v1()
    binding=configure_controller_binding_v1(q)
    authority=read(q["direct_attachment_authority_ref"]);active(authority)
    require(authority.get("schema")=="native_VAL000_downstream_direct_attachment_authority_v1"
        and authority.get("root_authorizer")=="root_agent" and authority.get("status")=="active"
        and authority.get("borrowed_physical_inode_attachment_authorized") is True
        and authority.get("additional_old_controller_submissions_revoked_while_downstream_attached") is True
        and authority.get("old_controller_preserved_until_downstream_drained") is True
        and authority.get("controller_pid")==CONTROLLER_PID
        and authority.get("controller_start_ticks")==CONTROLLER_START
        and authority.get("controller_capability_ref")==CURRENT_CAP
        and authority.get("controller_binding_ref")==q["controller_binding_ref"]
        and authority.get("downstream_job_id")==q["job_id"]
        and authority.get("protected_envelope_bytes")==C3_ENVELOPE_MIB*2**20
        and authority.get("downstream_max_cuda_reserved_bytes")==q["caps"]["max_cuda_reserved_bytes"]
        and authority.get("downstream_non_Torch_margin_bytes")==2*2**30
        and authority.get("headroom_bytes")==4*2**30,
        "Exact ROOT fixed-job borrowed attachment/revocation authority")
    cap=read(CURRENT_CAP);row=base.proc_identity(CONTROLLER_PID)
    require(time.time()<cap["expires_unix"]
        and datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],
        "downstream actual authority remains inside original live-controller execution authority")
    require(row["uid"]==os.getuid() and row["start_ticks"]==CONTROLLER_START
        and row["state"] not in ("Z","X") and cap["gpu_uuid"]==GPU0_UUID
        and cap["physical_gpu"]==0,"Exact live known controller identity andGPU0")
    declared=cap["jobs"][C3_JOB] if C3_JOB is not None else None
    if declared is not None:
        require(declared["cuda_cap_bytes"]==24*2**30 and declared["margin_mib"]==2048
            and declared["peak_mib"]==26*1024,"Exact unchanged C3 24+2GiB reservation")
    registry=_registry_v1(q,base,cap)
    require(registry["controller"]["pid"]==CONTROLLER_PID
        and registry["controller"]["start_ticks"]==CONTROLLER_START
        and registry["physical_gpu"]==0 and registry["lifecycle"]=="running"
        and registry["safety_error"] is None and not registry["ambiguous_descendants_or_lease_holders"]
        and registry.get("capability_ref")==CURRENT_CAP
        and -5<=time.time()-registry["updated_unix"]<=60,"Fresh exact original controller registry")
    active_jobs=[j for j in registry["jobs"] if j["state"] in ("queued","launching","running","draining")]
    c3_rows=[j for j in registry["jobs"] if j["logical_job"]==C3_JOB]
    if C3_JOB is None:
        require(not active_jobs,"Reviewed successor has no other queued/live controller jobs")
        c3=dict(process_identities=[])
    else:
        require(len(c3_rows)==1 and all(j["logical_job"]==C3_JOB for j in active_jobs)
            and c3_rows[0]["requested"]==declared and c3_rows[0]["state"] in ("running","draining","succeeded"),
            "Only original C3 may run/drain/finish alongside downstream")
        c3=c3_rows[0]
        if c3["state"]=="succeeded":
            require(c3.get("owned_drain_observed") is True and c3.get("exit_code")==0,"Actual successful original C3 drain")
    return cap,registry,c3,authority

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
    c3_mib=downstream_mib=0;external=[]
    for app in gpu["compute"]:
        if app["pid"] in known:
            require(app["name"]=="AI Auto Research Pipeline Test","KnownC3 actual CUDA title")
            c3_mib+=app["used_mib"]
        elif worker_pid is not None and app["pid"]==worker_pid:
            require(app["name"]=="AI Auto Research Pipeline Test","downstream actual CUDA title")
            downstream_mib+=app["used_mib"]
        else:external.append(dict(app))
    require(c3_mib<=C3_ENVELOPE_MIB,"C3 actual usage inside original26GiB envelope")
    peak_mib=(q["caps"]["max_cuda_reserved_bytes"]+2*2**30+2**20-1)//2**20
    require(downstream_mib<=peak_mib,"downstream actual usage inside cap plus2GiB margin")
    external_mib=max(sum(a["used_mib"] for a in external),
        gpu["used_mib"]-c3_mib-downstream_mib,0)
    projected=C3_ENVELOPE_MIB+peak_mib+external_mib+4096
    free_required=max(0,C3_ENVELOPE_MIB-c3_mib)+max(0,peak_mib-downstream_mib)+4096
    require(projected<=gpu["total_mib"] and gpu["total_mib"]-gpu["used_mib"]>=free_required,
        "Actual external+all unfilled C3/downstream envelopes+4GiB headroom fitGPU0")
    return dict(schema="downstream_fixed_GPU0_capacity_sample_v1",observed_unix=time.time(),
        GPU0=gpu,C3_fixed_envelope_mib=C3_ENVELOPE_MIB,downstream_cap_plus_margin_mib=peak_mib,
        C3_observed_mib=c3_mib,downstream_observed_mib=downstream_mib,external_compute=external,
        external_and_driver_mib=external_mib,projected_mib=projected,
        actual_free_required_mib=free_required,headroom_mib=4096,
        registry_updated_unix=registry["updated_unix"],controller_capability_ref=CURRENT_CAP,
        actual_controller_liveness_evidence=LAST_CONTROLLER_LIVENESS)

def attach_v1(spec,q):
    require(spec["phase"] in GPU_PHASES,"Only fixed downstream scientific phases use GPU0")
    require(0<q["caps"]["max_cuda_reserved_bytes"]<=74*2**30,"Exact separately ROOT-admitted downstream allocator cap")
    base=base_v1();physical=local(".runtime/locks/gpu_0.lock")
    logical_path=local(c.STATE[0]["worker_auxiliary_scope"]+"/contrastive/downstream_master_GPU0.logical.lock")
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
            if spec["phase"]=="contrastive":
                logical_fd=os.open(logical_path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            else:
                logical_fd=os.open(logical_path,os.O_RDWR|os.O_NOFOLLOW)
                logical_stat=os.fstat(logical_fd)
                require(stat.S_ISREG(logical_stat.st_mode) and logical_stat.st_nlink==1,
                    "Existing own master logical inode, new stage never unlocks or steals it")
            fcntl.flock(logical_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            capacity=capacity_v1(q,base)
            ticket=dict(schema="downstream_borrowed_GPU0_attachment_v1",job_id=spec["job_id"],
                physical_gpu=0,gpu_uuid=GPU0_UUID,borrowed_physical_inode=True,
                physical_exclusive_lock_owner="exact_ROOT_bound_unchanged_controller",
                controller_pid=CONTROLLER_PID,controller_start_ticks=CONTROLLER_START,
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
