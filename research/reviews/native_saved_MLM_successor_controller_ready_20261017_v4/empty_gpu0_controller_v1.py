"""Reviewed owner4 status-service adapter over the unchanged v3 wire protocol.

No model/science/compute/submit path. Default performs no operations. Full source,
finite capability, handover and storage authentication occurs at startup. The
steady loop uses immutable authenticated snapshots plus fresh identity/lease/OFD
checks and fresh status timestamps; it performs no registry writes or accounting.
Initial and final/error registry writes use the unchanged original guarded writer.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,fcntl,hashlib,importlib.util,json,os,resource,select,signal,socket,stat,struct,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v4"
SOURCE=HERE+"/empty_gpu0_controller_v1.py"
MANIFEST=HERE+"/source_manifest.v1.json"
PROTOCOL_REF=dict(path="research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3/empty_gpu0_controller_v1.py",
 sha256="dca4230a3d79540b947fa5488c6a49a885970db367b9955f78c343b8305b398f")
PREDECESSOR_CAP=dict(path="research/reviews/research_execution_ROOT_20261017_v34/native_owner_v3_capability.root_active_v1.json",
 sha256="104d4f99d16327a73c96959cf7356f38fed8190e8b24145548e0fa6b14cb3a71")
PREDECESSOR_ID=dict(pid=239065,start_ticks=493076932,uid=1000)
FAILED_QUALIFICATION=dict(path="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/ROOT_qualification_records_root01_oct17/input.json",
 sha256="195799008a160489cd82930368b03e88c11d521f56fd003baef2ffc7f3cad538")
UUID="GPU-ff06ca1f-403a-f2d0-b584-71e177452667"
MAX_JSON=65536
_PROTOCOL=None

def _protocol_v4():
    global _PROTOCOL
    if _PROTOCOL is None:
        path=ROOT/PROTOCOL_REF["path"];before=path.lstat()
        if not(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=32*2**20 and path.resolve()==path):
            raise ValueError("Canonical singly linked bounded original v3 source")
        data=path.read_bytes();after=path.lstat()
        if(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ValueError("Original v3 source changed during authenticated import")
        if hashlib.sha256(data).hexdigest()!=PROTOCOL_REF["sha256"]:raise ValueError("Unchanged executed v3 protocol source pin")
        name="_empty_owner_v4_original_v3_protocol"
        spec=importlib.util.spec_from_loader(name,loader=None);p=importlib.util.module_from_spec(spec)
        p.__file__=str(path);sys.modules[name]=p;exec(compile(data,str(path),"exec",dont_inherit=True,optimize=0),p.__dict__)
        p.HERE=HERE;p.SOURCE=SOURCE;p.MANIFEST=MANIFEST
        p.OLD_CAP=PREDECESSOR_CAP;p.OLD_ID=PREDECESSOR_ID
        p._original_validate_cap_v3=p.validate_cap_v1
        p.validate_cap_v1=validate_cap_v1;p.serve_v1=serve_v1
        _PROTOCOL=p
        for n in("need","local","raw","read","ref","encoded","stamp","active","proc_identity","absent",
            "socket_name","guard_v1","atomic_registry","inventory","dispatch_v1"):
            globals()[n]=getattr(p,n)
    return _PROTOCOL

def __getattr__(name):
    if name.startswith("__"):raise AttributeError(name)
    return getattr(_protocol_v4(),name)

def validate_cap_v1(record,*,allow_drain=False):
    p=_protocol_v4()
    cap,h,cert,storage=p._original_validate_cap_v3(record,allow_drain=allow_drain)
    p.need(cap["source_refs"].get(PROTOCOL_REF["path"])==PROTOCOL_REF,
        "Actual unchanged v3 protocol and reviewed owner4 service adapter both authenticated")
    p.need(h["predecessor_owner_capability_ref"]==PREDECESSOR_CAP
        and h["predecessor_owner_identity"]==PREDECESSOR_ID
        and h["predecessor_termination_outcome_preserved"]is True and h["predecessor_actual_outer_exit_code"]==1
        and h["failed_TreeLoom_qualification_input_ref"]==FAILED_QUALIFICATION
        and h["failed_TreeLoom_qualification_worker_started"]is False
        and h["failed_TreeLoom_qualification_CUDA_context_created"]is False
        and h["failed_TreeLoom_qualification_namespace_replay_authorized"]is False,
        "Genuine controlled v3 turnover and failed qualification preserved; no graceful0 or replay invented")
    for name in("predecessor_owner_actual_wait_ref","predecessor_owner_actual_outer_ref","failed_TreeLoom_qualification_actual_outer_ref"):
        p.raw(h[name])
    stopped=p.read(h["predecessor_owner_actual_wait_ref"])
    p.need(stopped["controller_capability_ref"]==PREDECESSOR_CAP
        and stopped["controller_pid"]==PREDECESSOR_ID["pid"]
        and stopped["controller_start_ticks"]==PREDECESSOR_ID["start_ticks"]
        and type(stopped["actual_raw_wait"])is int and stopped["actual_raw_wait"]==9
        and type(stopped["actual_direct_returncode"])is int and stopped["actual_direct_returncode"]==-9
        and stopped["all_owned_processes_drained"]is True and stopped["pipe_eof"]is True
        and stopped["cleanup_errors"]==[],"Genuine exact predecessor SIGKILL wait/drain, never graceful0")
    previous=p.read(PREDECESSOR_CAP);old_h=p.read(previous["handover_ref"]);old_cert=p.read(old_h["known_root_absence_ref"])
    required=old_cert["known_identities"]+[PREDECESSOR_ID]
    p.need(all(row in cert["known_identities"]for row in required),
        "Every previously known root plus exact actual v3 predecessor remains in targeted startup roster")
    return cap,h,cert,storage

def serve_v1(record,gate_fd):
    _protocol_v4()
    need(os.read(gate_fd,1)==b'G','Parent actual identity receipt precedes owner start');os.close(gate_fd)
    cap,h,cert,storage=validate_cap_v1(record);g=guard_v1(cap,storage)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(n in sys.modules for n in('torch','torch_geometric')),'CPU-only owner, no CUDA import')
    need(max(stamp(h['created_at_utc']),stamp(cap['created_at_utc']))<=stamp(cert['observed_at_utc'])<=time.time()<min(stamp(h['expires_at_utc']),cap['expires_unix']),'Finite ROOT-authorized known-root roster inside handover/plan validity; actual absence follows')
    absent(cert['known_identities'])
    identity=proc_identity(os.getpid());need(identity is not None and identity['uid']==os.getuid()and identity['pid']==os.getpgrp()==os.getsid(0),'Own fresh direct session')
    out=local(cap['output_directory']);need(not out.exists()and out.parent.is_dir(),'Fresh admitted controller output')
    g.budget_check_v9(storage,boundary=True,required_write_bytes=131072);out.mkdir(mode=0o700)
    lease_path=local('.runtime/locks/gpu_0.lock');scheduler_path=local('.runtime/locks/scheduler.lock')
    listener=lease=scheduler=None;lease_meta=None;lease_sha=None;stopped=[False];lifecycle='new';safety=None
    old_handlers={s:signal.signal(s,lambda signum,frame:stopped.__setitem__(0,True))for s in(signal.SIGTERM,signal.SIGINT)}
    # Decoded immutable cap/plan/handover clocks were fully source-authenticated above.
    # No steady-state source walk, JSON read, registry write or accounting operation.
    authority_start=max(stamp(cap['created_at_utc']),stamp(h['created_at_utc']))
    authority_end=min(cap['expires_unix'],stamp(cap['expires_at_utc']),stamp(h['expires_at_utc']))
    def hot_check():
        now=time.time()
        need(authority_start<=now<authority_end,'Original finite capability/handover clocks')
        current=proc_identity(os.getpid())
        need(current is not None and current['pid']==identity['pid']
            and current['uid']==identity['uid']==os.getuid()
            and current['start_ticks']==identity['start_ticks']and current['state']not in('Z','X')
            and os.getpid()==os.getpgrp()==os.getsid(0),'Fresh same owner PID/start/UID/session')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*2**30,'Own2GiB RSS cap')
        need(Path('/proc/self/task/%d/children'%os.getpid()).read_text().strip()=='','Empty owner never has scientific children')
        opened=os.fstat(lease);named=lease_path.lstat()
        need(stat.S_ISREG(opened.st_mode)and stat.S_ISREG(named.st_mode)
            and opened.st_nlink==named.st_nlink==1
            and(opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),
            'Stable same canonical exclusive physical lease inode')
        data=os.pread(lease,65536,0)
        need(len(data)==opened.st_size<=8192 and hashlib.sha256(data).hexdigest()==lease_sha,
            'Exact unchanged complete physical lease generation bytes')
        probe=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
        try:
            checked=os.fstat(probe)
            need((checked.st_dev,checked.st_ino,checked.st_nlink)==(opened.st_dev,opened.st_ino,1),
                'Independent no-follow probe references same original inode')
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise RuntimeError('Independent writable exclusion proof failed')
        finally:os.close(probe)
    def status():
        if lifecycle=='running':hot_check()
        return dict(schema='GPU0_declared_logical_pool_registry_v1',pool=cap['pool'],pool_job='gpu0_shared_v1:'+cap['pool']+':gpu0',lifecycle=lifecycle,controller=identity,socket=socket_name(cap),physical_gpu=0,capability_ref=record,lease_path=str(lease_path),lease_sha256=lease_sha,lease_metadata=lease_meta,headroom_mib=4096,safety_error=safety,ambiguous_descendants_or_lease_holders=[],updated_unix=time.time(),jobs=[],empty_owner_only=True,borrowed_workers_supervised_by_their_separate_pidfd_driver=True,owner_service_revision=4,periodic_registry_writes=False,periodic_accounting_transactions=False)
    try:
        scheduler=os.open(scheduler_path,os.O_RDWR|os.O_NOFOLLOW);fcntl.flock(scheduler,fcntl.LOCK_EX)
        try:
            absent(cert['known_identities']);lease=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
            before=os.fstat(lease);named=lease_path.lstat();expected=h['old_physical_lease']
            need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and(before.st_dev,before.st_ino)==(named.st_dev,named.st_ino)==(expected['device'],expected['inode']),'Same canonical old physical inode')
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=os.pread(lease,65536,0)
            need(len(old)==before.st_size and hashlib.sha256(old).hexdigest()==expected['sha256'],'Old bytes preserved until exclusive acquisition')
            probe=os.open(lease_path,os.O_RDWR|os.O_NOFOLLOW)
            try:
                try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:pass
                else:raise RuntimeError('Independent writable exclusion proof failed')
            finally:os.close(probe)
            inventory(cap)
            lease_meta=dict(launcher_pid=os.getpid(),gpu=0,uuid=UUID,job='gpu0_shared_v1:'+cap['pool']+':gpu0',started_unix=time.time(),storage_bytes=32*2**20,logical_sharing=True)
            data=encoded(lease_meta);need(len(data)<8192,'Compact lease')
            os.ftruncate(lease,len(data));need(os.pwrite(lease,data,0)==len(data),'Complete own new lease generation');os.fsync(lease)
            lease_sha=hashlib.sha256(data).hexdigest()
        finally:os.close(scheduler);scheduler=None
        listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.bind(socket_name(cap));listener.listen(8);listener.setblocking(False)
        lifecycle='running';atomic_registry(out/'registry.json',status(),g,storage)
        while not stopped[0] and time.time()<authority_end:
            hot_check()
            ready,_,_=select.select([listener],[],[],.2)
            if not ready:continue
            client,_=listener.accept()
            with client:
                deadline=min(time.monotonic()+2.,time.monotonic()+authority_end-time.time())
                client.settimeout(min(2.,max(.001,deadline-time.monotonic())))
                try:
                    peer=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    need(peer[1]==os.getuid(),'Original same-UID authenticated protocol')
                    peer_before=proc_identity(peer[0])
                    need(peer_before is not None and peer_before['pid']==peer[0]
                        and peer_before['uid']==peer[1]and peer_before['state']not in('Z','X'),
                        'Fresh actual peer PID/UID before bounded request')
                    request=bytearray()
                    while b'\n'not in request:
                        need(time.monotonic()<deadline,'Aggregate2s status exchange, no unbounded trickle')
                        client.settimeout(max(.001,deadline-time.monotonic()))
                        block=client.recv(4096);need(bool(block),'Complete request');request.extend(block);need(len(request)<=8192,'Bounded status request')
                    line,tail=bytes(request).split(b'\n',1);need(not tail,'One status request only')
                    peer_after=proc_identity(peer[0])
                    need(peer_after is not None and peer_after['pid']==peer_before['pid']
                        and peer_after['uid']==peer_before['uid']and peer_after['start_ticks']==peer_before['start_ticks']
                        and peer_after['state']not in('Z','X'),'Same fresh live peer PID/start/UID before response')
                    need(time.monotonic()<deadline,'Aggregate bounded status deadline before dispatch')
                    answer,shutdown=dispatch_v1(json.loads(line),record,cap,status,lambda path:read(ref(path)))
                    if shutdown:stopped[0]=True
                    packet=dict(ok=True,result=answer)
                except(ValueError,KeyError,TypeError,OSError)as error:packet=dict(ok=False,error=str(error))
                try:
                    client.settimeout(max(.001,deadline-time.monotonic()))
                    if time.monotonic()<deadline:client.sendall(encoded(packet)+b'\n')
                except OSError:pass
        lifecycle='closed';atomic_registry(out/'registry.json',status(),g,storage)
        return 0
    except BaseException as error:
        safety=type(error).__name__+': '+str(error);lifecycle='failed'
        if out.exists():
            try:atomic_registry(out/'registry.json',status(),g,storage)
            except BaseException:pass
        raise
    finally:
        if listener is not None:listener.close()
        if scheduler is not None:os.close(scheduler)
        if lease is not None:os.close(lease)
        for s,v in old_handlers.items():signal.signal(s,v)


def main(argv=None):
    argv=sys.argv[1:]if argv is None else argv
    if '--execute'not in argv:
        print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',operations=0)));return 0
    return _protocol_v4().main(argv)
if __name__=='__main__':raise SystemExit(main())
