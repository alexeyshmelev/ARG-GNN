"""Fresh8GiB evaluator borrows the held GPU0 inode; existing H058/controller remain unchanged."""
from pathlib import Path
import fcntl,hashlib,json,os,stat,sys,time
from .contract_v1 import *
from . import contract_v1 as c
SOURCE=DIRECT;CONTROLLER_PID=3349405;CONTROLLER_START=481281638
_PEER=None;CANCEL_CHECK=None
def install_cancellation_check_v1(callback):
 global CANCEL_CHECK
 require(callable(callback),"Finite actual caller cancellation callback");CANCEL_CHECK=callback
def base_v1():return load(BASE_POOL,"_C3_remaining_original_pool")
def peer_module_v1():
 global _PEER
 if _PEER is None:_PEER=load(c.pins_v1()["pins"]["H058_DIRECT"],"_C3_remaining_unchanged_H058_guard")
 return _PEER
def controller_v1(q,base):
 if CANCEL_CHECK is not None:CANCEL_CHECK()
 e,hq=c.peer_values_v1(q["peer_launch_capsule_ref"])
 try:row=base.proc_identity(e["worker_identity"]["pid"])
 except (FileNotFoundError,ProcessLookupError):row=None
 require(e.get("terminal_peer_absence_preserves_full30GiB")is True,"Peer disappearance never releases its full reserved envelope")
 if row is not None:require(row["uid"]==os.getuid()and row["start_ticks"]==e["worker_identity"]["start"]and row["state"]not in("Z","X"),"Exact live peer PID/start/UID; positive reuse fails closed")
 a=read(q["direct_attachment_authority_ref"]);active(a)
 require(a.get("schema")=="C3_remaining_ROOT_direct_attachment_v1"and a.get("job_id")==q["job_id"]and a.get("peer_launch_capsule_ref")==q["peer_launch_capsule_ref"]and a.get("borrowed_physical_inode_attachment_authorized")is True and a.get("old_controller_and_H058_preserved")is True and a.get("old_controller_submissions_remain_revoked")is True and a.get("evaluator_cuda_bytes")==8*2**30 and a.get("evaluator_margin_bytes")==2*2**30 and a.get("C3_reserved_bytes")==26*2**30 and a.get("H058_reserved_bytes")==30*2**30 and a.get("headroom_bytes")==4*2**30,"Separate exact direct-borrow authority, never reuse/relabel C3 slot")
 cap,registry,c3,oldauth=peer_module_v1().controller_v1(hq,base)
 require(datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(hq["expires_at_utc"]),"New borrower cannot outlive peer/controller authority")
 return cap,registry,c3,a
def capacity_v1(q,base,*,worker_pid=None):
 controller_v1(q,base);e,hq=c.peer_values_v1(q["peer_launch_capsule_ref"])
 sample=peer_module_v1().capacity_v1(hq,base,worker_pid=e["worker_identity"]["pid"])
 gpu=sample["GPU0"]
 try:peer_now=base.proc_identity(e["worker_identity"]["pid"])
 except (FileNotFoundError,ProcessLookupError):peer_now=None
 if peer_now is None:require(not any(r["pid"]==e["worker_identity"]["pid"]for r in gpu["compute"]),"Absent peer PID cannot retain an unexplained CUDA context")
 else:require(peer_now["uid"]==os.getuid()and peer_now["start_ticks"]==e["worker_identity"]["start"],"Peer identity remains exact after capacity sample")
 mine=[r for r in gpu["compute"]if worker_pid is not None and r["pid"]==worker_pid]
 require(len(mine)<=1 and all(r["name"]=="AI Auto Research Pipeline Test"for r in mine),"Own exact titled evaluator CUDA identity")
 own=sum(r["used_mib"]for r in mine);limit=10*1024
 require(own<=limit,"8GiB allocator plus2GiB nonTorch evaluator envelope")
 unfilled=max(0,limit-own);projected=sample["projected_mib"]+unfilled;free_required=sample["actual_free_required_mib"]+unfilled
 require(projected<=gpu["total_mib"]and gpu["total_mib"]-gpu["used_mib"]>=free_required,"Preserve C3_26+H058_30+EVAL_10+4headroom plus actual external/driver occupancy")
 return dict(schema="C3_remaining_conservative_GPU0_capacity_v1",GPU0=gpu,original_unchanged_H058_capacity_sample=sample,evaluator_observed_mib=own,evaluator_cap_plus_margin_mib=limit,projected_mib=projected,actual_free_required_mib=free_required,observed_unix=time.time(),C3_slot_reused=False)


def held_writable_probe_v1(path):
    with path.open('r+') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError('Expected physical/logical exclusion lock is unheld')

def attach_v1(spec, q):
    require(spec['phase'] in GPU_PHASES, 'Only exact remaining-cell prediction attaches GPU0')
    base = base_v1()
    physical = local('.runtime/locks/gpu_0.lock')
    logical_path = local(spec['worker_auxiliary_scope'] + '/C3_remaining_GPU0.logical.lock')
    physical_fd = logical_fd = None
    with base.lock_file(ROOT, 'scheduler.lock') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX)
        try:
            (cap, registry, c3, authority) = controller_v1(q, base)
            physical_fd = os.open(physical, os.O_RDWR | os.O_NOFOLLOW)
            st = os.fstat(physical_fd)
            named = physical.lstat()
            require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and ((st.st_dev, st.st_ino) == (named.st_dev, named.st_ino)), 'Original canonical physical inode')
            raw_bytes = os.pread(physical_fd, 65536, 0)
            require(len(raw_bytes) == st.st_size and hashlib.sha256(raw_bytes).hexdigest() == registry['lease_sha256'], 'Original registry-bound physical lease bytes')
            metadata = json.loads(raw_bytes)
            require(metadata == registry['lease_metadata'] and metadata['launcher_pid'] == CONTROLLER_PID and (metadata['gpu'] == 0) and (metadata['uuid'] == GPU0_UUID), 'Original actual physical lease identity')
            held_writable_probe_v1(physical)
            logical_fd = os.open(logical_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
            fcntl.flock(logical_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            capacity = capacity_v1(q, base)
            ticket = dict(schema='C3_remaining_borrowed_GPU0_attachment_v1', job_id=spec['job_id'], physical_gpu=0, gpu_uuid=GPU0_UUID, borrowed_physical_inode=True, physical_exclusive_lock_owner='unchanged_original_controller', controller_pid=CONTROLLER_PID, controller_start_ticks=CONTROLLER_START, physical_inode=st.st_ino, physical_device=st.st_dev, physical_lease_sha256=hashlib.sha256(raw_bytes).hexdigest(), physical_lease_metadata=metadata, logical_lock_path=str(logical_path), logical_inode=os.fstat(logical_fd).st_ino, logical_device=os.fstat(logical_fd).st_dev, C3_remaining_logical_lock_owned_by_supervisor=True, capacity_at_attachment=capacity, direct_attachment_authority_ref=q['direct_attachment_authority_ref'])
            return (physical_fd, logical_fd, ticket)
        except BaseException:
            for fd in (physical_fd, logical_fd):
                if fd is not None:
                    os.close(fd)
            raise

def install_before_torch_v1(spec, q, ticket):
    require(not any((n in sys.modules for n in ('torch', 'torch_geometric'))), 'Direct lease/title install before science')
    g = load(GPU0_GUARD, '_H058_original_GPU0_guard')
    native = g.install_before_torch_v1()
    original = native.validate_lease
    base = base_v1()
    logical_fd = int(os.environ['downstream_LOGICAL_LOCK_FD'])
    physical_fd = int(os.environ['ARGGNN_GPU_LEASE_FD'])

    def validate_lease(environ=None):
        lease = original(environ)
        require(lease['lease_fd'] == physical_fd and lease['lease_inode'] == ticket['physical_inode'] and (lease['lease_device'] == ticket['physical_device']) and (lease['lease_metadata_sha256'] == ticket['physical_lease_sha256']) and (lease['launcher_metadata'] == ticket['physical_lease_metadata']), 'Stable authenticated borrowed physical inode, never controller inheritance')
        controller_v1(q, base)
        st = os.fstat(logical_fd)
        path = Path(ticket['logical_lock_path'])
        named = path.lstat()
        require(stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == (named.st_dev, named.st_ino) == (ticket['logical_device'], ticket['logical_inode']), 'Actual inherited H058 logical lock descriptor')
        held_writable_probe_v1(path)
        capacity_v1(q, base, worker_pid=os.getpid())
        return dict(lease, borrowed_physical_inode=True, physical_exclusive_lock_owned_by_current_controller=True, C3_remaining_logical_lock_inode=ticket['logical_inode'], C3_remaining_direct_attachment_source_ref=ref(SOURCE))
    native.validate_lease = validate_lease
    validate_lease()
    native.apply_and_validate_title()
    return native

def check_attachment_v1(q, ticket, physical_fd, logical_fd):
    controller_v1(q, base_v1())
    for (fd, path, device, inode) in ((physical_fd, local('.runtime/locks/gpu_0.lock'), ticket['physical_device'], ticket['physical_inode']), (logical_fd, Path(ticket['logical_lock_path']), ticket['logical_device'], ticket['logical_inode'])):
        opened = os.fstat(fd)
        named = path.lstat()
        require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino) == (device, inode), 'Stable actual physical/logical inode identity')
        held_writable_probe_v1(path)
    contents = os.pread(physical_fd, 65536, 0)
    require(len(contents) == os.fstat(physical_fd).st_size and hashlib.sha256(contents).hexdigest() == ticket['physical_lease_sha256'], 'Actual original physical lease bytes unchanged')
