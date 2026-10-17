"""Inactive empty GPU0 lease owner. It cannot launch or submit scientific jobs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,fcntl,hashlib,importlib.util,json,os,resource,select,signal,socket,stat,struct,sys,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_saved_MLM_successor_controller_ready_20261017_v2'
SOURCE=HERE+'/empty_gpu0_controller_v1.py'
MANIFEST=HERE+'/source_manifest.v1.json'
C64='research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2'
C64_CONTRACT=C64+'/contract_v1.py'
UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667'
OLD_CAP={'path':'research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json','sha256':'4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870'}
OLD_ID={'pid':3349405,'start_ticks':481281638,'uid':1000}
BASE_POOL={'path':'src/arggnn/gpu_pool_v1.py','sha256':'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1'}
MAX_JSON=65536
def need(ok,message):
    if not ok:raise ValueError(message)
def local(name):
    p=Path(name)
    need(isinstance(name,str) and not p.is_absolute() and '..' not in p.parts and p.as_posix()==name,'Exact repository-relative path')
    value=ROOT/p
    need(value.resolve()==value,'No symlink path')
    return value
def raw(record):
    need(isinstance(record,dict) and set(record)=={'path','sha256'},'Exact immutable ref')
    p=local(record['path']);before=p.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=32*2**20,'Bounded regular source/metadata')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb')as f:
        opened=os.fstat(f.fileno());data=f.read(32*2**20+1);after=os.fstat(f.fileno())
    identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    need(identity(before)==identity(opened)==identity(after)==identity(p.lstat()) and len(data)==before.st_size and hashlib.sha256(data).hexdigest()==record['sha256'],'Stable exact source/metadata')
    return data
def read(record):return json.loads(raw(record))
def ref(path):return dict(path=path,sha256=hashlib.sha256(local(path).read_bytes()).hexdigest())
def load(record,name):
    data=raw(record);spec=importlib.util.spec_from_loader(name,loader=None);m=importlib.util.module_from_spec(spec)
    m.__file__=str(local(record['path']));sys.modules[name]=m
    exec(compile(data,m.__file__,'exec',dont_inherit=True,optimize=0),m.__dict__);return m
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def stamp(value):
    v=datetime.fromisoformat(value.replace('Z','+00:00'));need(v.tzinfo is not None,'Absolute timezone-aware clock');return v.timestamp()
def active(value):
    need(value.get('status')=='active' and value.get('root_authorizer')=='root_agent' and value.get('execution_authorized')is True and value.get('template_only')is not True and stamp(value['created_at_utc'])<=time.time()<stamp(value['expires_at_utc']),'Finite explicit ROOT authority')
def proc_identity(pid):
    try:text=Path('/proc/%d/stat'%pid).read_text()
    except(FileNotFoundError,ProcessLookupError):return None
    i=text.rfind(')');parts=text[i+2:].split()
    return dict(pid=pid,start_ticks=int(parts[19]),uid=Path('/proc/%d'%pid).stat().st_uid,state=parts[0])
def absent(rows):
    for saved in rows:
        now=proc_identity(saved['pid'])
        need(now is None or now['start_ticks']!=saved['start_ticks'] or now['state']in('Z','X'),'Known old process remains live')
def socket_name(cap):
    token=hashlib.sha256(encoded(cap['plan_ref'])).hexdigest()[:16]
    value='\0arggnn_gpu0_declared_'+cap['pool']+'_'+token
    need(len(os.fsencode(value))<=107,'Bounded original abstract endpoint')
    return value
def validate_cap_v1(record,*,allow_drain=False):
    cap=read(record)
    need(cap.get('schema')=='ROOT_empty_GPU0_owner_capability_v1' and cap.get('status')=='active' and cap.get('root_authorizer')=='root_agent' and cap.get('execution_authorized')is True and cap.get('template_only')is not True,'Explicit empty-owner capability')
    need(cap.get('jobs')=={} and cap.get('submit_authorized')is False and cap.get('physical_gpu')==0 and cap.get('gpu_uuid')==UUID and cap.get('maximum_distinct_project_GPUs')==1 and cap.get('maximum_concurrent_jobs')==0 and cap.get('protected_envelope_bytes')==0 and cap.get('headroom_mib')==4096 and cap.get('controller_rss_cap_bytes')==2*2**30,'Empty GPU0 owner only, no compute envelope')
    need(cap.get('base_pool_ref')==BASE_POOL,'Original pinned inventory implementation')
    need(cap.get('external_job_sharing_authorized')is True and cap.get('preserve_external_processes')is True and cap.get('borrowed_workers_use_separate_logical_gate')is True and cap.get('preserve_owner_until_explicit_downstream_drain')is True,'Explicit borrowed-worker ownership boundaries')
    need(isinstance(cap.get('pool'),str) and 1<=len(cap['pool'])<=32 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-'for c in cap['pool']),'Safe distinct pool identity')
    now=time.time()
    need(stamp(cap['created_at_utc'])<=now<(cap['drain_expires_unix']if allow_drain else cap['expires_unix']) and cap['expires_unix']+35<=cap['drain_expires_unix']<=stamp(cap['expires_at_utc']),'Finite original work and owned cleanup interval')
    renewal=read(cap['marker_repair_renewal_ref']);need(cap['drain_expires_unix']<stamp(renewal['renewed_expires_at_utc']),'Owner drain remains inside actual unchanged marker renewal')
    manifest=read(cap['source_manifest_ref'])
    need(cap['source_manifest_ref']==ref(MANIFEST) and manifest['source_sha256'][SOURCE]==ref(SOURCE)['sha256'],'Exact new complete source manifest')
    for p,s in manifest['source_sha256'].items():raw(dict(path=p,sha256=s))
    for p,r in cap['source_refs'].items():need(p==r['path'],'Exact capability source key');raw(r)
    need(cap['source_refs'].get(SOURCE)==ref(SOURCE),'New endpoint implementation explicitly authenticated')
    review=read(cap['source_review_ref'])
    need(review.get('independent')is True and review.get('verdict')=='SOURCE_PASS' and review.get('blockers')==[] and review.get('candidate_manifest_ref')==cap['source_manifest_ref'],'Fresh independent complete source review')
    plan=read(cap['plan_ref'])
    need(plan.get('schema')=='ROOT_empty_GPU0_owner_plan_v1' and {k:v for k,v in cap.items()if k not in('schema','plan_ref','controller_storage_ref')}=={k:v for k,v in plan.items()if k!='schema'},'Exact ROOT plan/capability join')
    active(plan);h=read(cap['handover_ref']);active(h)
    need(h.get('schema')=='ROOT_empty_GPU0_successor_handover_v1' and h.get('old_controller_capability_ref')==OLD_CAP and h.get('old_controller_identity')==OLD_ID and h.get('future_old_controller_and_C3_routes_revoked')is True and h.get('all_current_borrower_routes_revoked')is True and h.get('historical_old_drain_proven')is False and h.get('old_files_charges_failures_unknowns_preserved')is True and h.get('saved_MLM_original_outer_exit_code')==1 and h.get('saved_MLM_full_fit_success_claimed')is False,'Honest known-root handover preserves old unknowns and native outer1')
    for key in('old_controller_revocation_ref','old_controller_signal_ref','H058_actual_terminal_ref','known_root_absence_ref'):raw(h[key])
    cert=read(h['known_root_absence_ref'])
    need(cert.get('root_authorizer')=='root_agent' and cert.get('all_listed_known_roots_absent')is True and cert.get('historical_drain_proven')is False and cert.get('includes_old_controller')is True and OLD_ID in cert['known_identities'] and isinstance(cert['known_identities'],list),'Explicit known identities, no global historical drain claim')
    storage=read(cap['controller_storage_ref'])
    need(storage['job_id']==cap['controller_job_id'] and storage['reserved_new_bytes']==32*2**20 and storage['reservation']['auxiliary_scopes']==[] and storage['output_directory']==cap['output_directory'] and storage['storage_policy_selector_ref']==cap['storage_policy_selector_ref'],'Actual dedicated32MiB current Census storage')
    needed=[ref(SOURCE),cap['source_manifest_ref'],cap['config_ref'],cap['runtime_ref']]
    need(all(r in storage['reservation']['consumer_bindings']for r in needed)and len(storage['reservation']['consumer_bindings'])<=64,'Original direct64-ref ceiling and actual source/config/runtime')
    cfg=read(cap['config_ref']);runtime=read(cap['runtime_ref'])
    need(cfg==dict(schema='ROOT_empty_GPU0_owner_config_v1',output_cap_bytes=32*2**20,physical_gpu=0,jobs={},submit_authorized=False,headroom_mib=4096,max_cpu_rss_bytes=2*2**30)and runtime==dict(schema='ROOT_empty_GPU0_owner_runtime_v1',python='.runtime/env/bin/python',source_ref=ref(SOURCE),CUDA_VISIBLE_DEVICES=''),'Frozen CPU-only owner config/runtime')
    return cap,h,cert,storage
def guard_v1(cap,storage):
    m=load(dict(path=C64_CONTRACT,sha256=read(cap['source_manifest_ref'])['source_sha256'][C64_CONTRACT]),'_empty_owner_selected_Census_metadata')
    m.configure_selector_v1(cap['storage_policy_selector_ref']);g=m.selected_guard_v1(consumer=True)
    gpu_policy=load(m.GPU0_GUARD,'_empty_owner_original_GPU0_policy');gpu_policy.validate_policy_v1();need(cap['human_GPU0_policy_ref']==gpu_policy.POLICY_REF,'Exact original human GPU0 policy')
    need(cap['capacity_guard_ref']==m.GUARD and cap['capacity_policy_ref']==m.POLICY,'Exact selected current Census child')
    spec=dict(job_id=cap['controller_job_id'],output_directory=cap['output_directory'])
    g.validate_storage_v9(spec,dict(output_cap_bytes=32*2**20),storage,source_ref=ref(SOURCE),config_ref=cap['config_ref'],runtime_ref=cap['runtime_ref'])
    return g
def atomic_registry(path,value,g,storage):
    data=encoded(value)+b'\n';need(len(data)<=MAX_JSON,'Compact registry')
    g.budget_check_v9(storage,boundary=True,required_write_bytes=2*len(data)+16384)
    temporary=path.with_name('registry.new.json');fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:need(f.write(data)==len(data),'Complete registry');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
def inventory(cap):
    base=load(cap['base_pool_ref'],'_empty_owner_original_inventory')
    rows=[r for r in base.real_inventory()if r['index']==0]
    need(len(rows)==1 and rows[0]['uuid']==UUID and rows[0]['used_mib']+4096<rows[0]['total_mib'],'Actual GPU0 UUID and4GiB free headroom')
    return rows[0]
def dispatch_v1(value,record,cap,status,release_reader):
    need(isinstance(value,dict)and type(value.get('v'))is int and value['v']==1,'Protocol1 object')
    if value==dict(v=1,op='status'):return status(),False
    need(value==dict(v=1,op='shutdown',terminate=False),'Empty owner forbids submit, worker, cancel and termination')
    release=release_reader(cap['ROOT_downstream_release_path']);active(release)
    need(release.get('schema')=='ROOT_empty_GPU0_owner_downstream_release_v1'and release.get('controller_capability_ref')==record and release.get('all_authorized_borrowers_actually_drained')is True and release.get('all_future_borrower_routes_revoked')is True and isinstance(release.get('genuine_wait_and_outer_refs'),list)and len(release['genuine_wait_and_outer_refs'])>0,'ROOT genuine downstream drain before explicit shutdown')
    for proof in release['genuine_wait_and_outer_refs']:raw(proof)
    return status(),True
def serve_v1(record,gate_fd):
    need(os.read(gate_fd,1)==b'G','Parent actual identity receipt precedes owner start');os.close(gate_fd)
    cap,h,cert,storage=validate_cap_v1(record);g=guard_v1(cap,storage)
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(n in sys.modules for n in('torch','torch_geometric')),'CPU-only owner, no CUDA import')
    need(0<=time.time()-stamp(cert['observed_at_utc'])<=120,'Fresh known-root certificate120s')
    absent(cert['known_identities'])
    identity=proc_identity(os.getpid());need(identity is not None and identity['uid']==os.getuid()and identity['pid']==os.getpgrp()==os.getsid(0),'Own fresh direct session')
    out=local(cap['output_directory']);need(not out.exists()and out.parent.is_dir(),'Fresh admitted controller output')
    g.budget_check_v9(storage,boundary=True,required_write_bytes=131072);out.mkdir(mode=0o700)
    lease_path=local('.runtime/locks/gpu_0.lock');scheduler_path=local('.runtime/locks/scheduler.lock')
    listener=lease=scheduler=None;lease_meta=None;lease_sha=None;stopped=[False];lifecycle='new';safety=None
    old_handlers={s:signal.signal(s,lambda signum,frame:stopped.__setitem__(0,True))for s in(signal.SIGTERM,signal.SIGINT)}
    def status():
        return dict(schema='GPU0_declared_logical_pool_registry_v1',pool=cap['pool'],pool_job='gpu0_shared_v1:'+cap['pool']+':gpu0',lifecycle=lifecycle,controller=identity,socket=socket_name(cap),physical_gpu=0,capability_ref=record,lease_path=str(lease_path),lease_sha256=lease_sha,lease_metadata=lease_meta,headroom_mib=4096,safety_error=safety,ambiguous_descendants_or_lease_holders=[],updated_unix=time.time(),jobs=[],empty_owner_only=True,borrowed_workers_supervised_by_their_separate_pidfd_driver=True)
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
        lifecycle='running';atomic_registry(out/'registry.json',status(),g,storage);next_check=0.
        while not stopped[0] and time.time()<cap['expires_unix']:
            need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2*2**30,'Own2GiB RSS cap')
            need(Path('/proc/self/task/%d/children'%os.getpid()).read_text().strip()=='','Empty owner never has scientific children')
            opened=os.fstat(lease);named=lease_path.lstat()
            need(opened.st_nlink==named.st_nlink==1 and(opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino)and hashlib.sha256(os.pread(lease,65536,0)).hexdigest()==lease_sha,'Stable exclusive lease generation')
            if time.monotonic()>=next_check:
                validate_cap_v1(record);atomic_registry(out/'registry.json',status(),g,storage);next_check=time.monotonic()+5
            ready,_,_=select.select([listener],[],[],.2)
            if not ready:continue
            client,_=listener.accept()
            with client:
                client.settimeout(min(2.,max(.001,cap['expires_unix']-time.time())))
                try:
                    peer=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                    need(peer[1]==os.getuid(),'Original same-UID authenticated protocol')
                    request=bytearray()
                    while b'\n'not in request:
                        block=client.recv(4096);need(bool(block),'Complete request');request.extend(block);need(len(request)<=8192,'Bounded status request')
                    line,tail=bytes(request).split(b'\n',1);need(not tail,'One status request only')
                    answer,shutdown=dispatch_v1(json.loads(line),record,cap,status,lambda path:read(ref(path)))
                    if shutdown:stopped[0]=True
                    packet=dict(ok=True,result=answer)
                except(ValueError,KeyError,TypeError,OSError)as error:packet=dict(ok=False,error=str(error))
                try:client.sendall(encoded(packet)+b'\n')
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
def request_v1(record,operation,ready_ref):
    cap,_,_,_=validate_cap_v1(record,allow_drain=True);expected=read(ready_ref)
    need(expected.get('controller_capability_ref')==record,'Exact actual ready identity')
    value=dict(v=1,op='status')if operation=='status'else dict(v=1,op='shutdown',terminate=False)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)as client:
        client.settimeout(30);client.connect(socket_name(cap))
        peer=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
        need(peer[0]==expected['controller_pid']and peer[1]==os.getuid(),'Exact live owner Unix peer')
        row=proc_identity(peer[0]);need(row is not None and row['start_ticks']==expected['controller_start_ticks'],'Exact owner start ticks')
        client.sendall(encoded(value)+b'\n');data=bytearray()
        while b'\n'not in data:
            block=client.recv(65536);need(bool(block),'Complete owner response');data.extend(block);need(len(data)<=MAX_JSON,'Bounded response')
    answer=json.loads(bytes(data).split(b'\n',1)[0]);need(answer.get('ok')is True,answer.get('error','Rejected'));return answer['result']
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',nargs='?',choices=('worker','status','shutdown'));p.add_argument('--execute',action='store_true');p.add_argument('--capability');p.add_argument('--sha256');p.add_argument('--gate-fd',type=int);p.add_argument('--ready-ref');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',operations=0)));return 0
    need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0,'Fresh exact ROOT -B -s interpreter')
    need(a.capability and a.sha256 and a.command,'Explicit actual capability')
    record=dict(path=a.capability,sha256=a.sha256)
    if a.command=='worker':need(a.gate_fd is not None,'Parent gate required');return serve_v1(record,a.gate_fd)
    need(a.ready_ref,'Exact actual ready identity ref');print(json.dumps(request_v1(record,a.command,json.loads(a.ready_ref)),sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
