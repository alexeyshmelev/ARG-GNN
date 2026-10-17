"""Independent source/AST/RAM checks, with no operational/guard/science imports."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
from types import SimpleNamespace,ModuleType
from contextlib import redirect_stdout
import argparse,ast,copy,hashlib,io,json,os,stat,struct,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE='research/reviews/native_saved_MLM_successor_controller_ready_20261017_v1'
C64='research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2'
OWNED='research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py'
BUDGET='src/arggnn/project_budget_v4.py'
CHECKS=[]
def need(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path):return json.loads((ROOT/path).read_text())
def function(path,name,scope):
    tree=ast.parse((ROOT/path).read_bytes())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path+'#independent_RAM_AST','exec'),scope)
    return scope[name]
def check(name,fn):fn();CHECKS.append(name)
def rejects(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('Expected rejection')
def baseline_roles():
    rows=doc('research/reviews/FLARE_VAL004_native_ROOT_close_Census_root01_oct17/ordinary_close_authority.json')['bindings']
    assert len(rows)==47 and len({json.dumps(v,sort_keys=True)for v in rows.values()})==40
    for rr in rows.values():assert ref(rr['path'])==rr
    return rows
def schema_RAM(request,blobs):
    handles={};counter=[10]
    root=None
    def st(path):
        key=str(path.relative_to(root));data=blobs[key]
        return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=len(data),st_dev=1,
            st_ino=int(hashlib.sha256(key.encode()).hexdigest()[:12],16),st_mtime_ns=1,st_ctime_ns=1)
    class RAMPath(PurePosixPath):
        def resolve(self):return self
        def lstat(self):return st(self)
    root=RAMPath(str(ROOT))
    class Memory(io.BytesIO):
        def fileno(self):return self.fd
    def open_(path,flags):
        assert flags==os.O_RDONLY|os.O_NOFOLLOW
        path=RAMPath(path);st(path);fd=counter[0];counter[0]+=1;handles[fd]=path;return fd
    def fdopen(fd,mode):
        assert mode=='rb';f=Memory(blobs[str(handles[fd].relative_to(root))]);f.fd=fd;return f
    scope=dict(Path=RAMPath,LEGACY='.runtime/gpup/storage_reservations.json',LOCK='.runtime/locks/scheduler.lock',
        MAX_JSON=32*2**20,json=json,hashlib=hashlib,stat=stat,os=SimpleNamespace(open=open_,fdopen=fdopen,
        fstat=lambda fd:st(handles[fd]),O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW))
    for name in ('encoded','integer','canonical','file_bytes','parse','paths_overlap','accounting_paths','bound','scope_spec'):
        function(BUDGET,name,scope)
    return scope['scope_spec'](root,copy.deepcopy(request))

def old_proofs():
    manifest=doc(BASE+'/source_manifest.v1.json')
    for p,sha in manifest['source_sha256'].items():
        assert not p.startswith('.runtime/storage/')and Path(p).suffix not in ('.pt','.pth','.npy','.npz','.pkl','.trees')
        assert ref(p)['sha256']==sha
    CHECKS.append('v1_all238_source_and_compact_metadata_refs_authenticated')
    old=ast.parse((ROOT/C64/'direct_gpu0_v1.py').read_bytes())
    new=ast.parse((ROOT/BASE/'native_direct_gpu0.NEW_ENDPOINT.source.py').read_bytes())
    def strip(tree):
        tree=copy.deepcopy(tree)
        tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ENDPOINT_SOURCE_REF'for t in n.targets))]
        return ast.dump(tree,include_attributes=False)
    assert strip(old)==strip(new);CHECKS.append('v1_staged_native_adapter_AST_changes_only_endpoint_source_ref')
    # Exact pinned lifecycle return schema, using only a RAM /proc text provider.
    class Text:
        def read_text(self):
            row=['S','1','2','3']+['0']*15+['777']
            return '123 (RAM) '+' '.join(row)
    fn=function(OWNED,'_stat',dict(Path=lambda value:Text()))
    value=fn(123);assert value['start']==777 and 'start_ticks'not in value
    CHECKS.append('original_owned_stat_RAM_proves_start_field_not_start_ticks')
    check('v1_parent_public_start_ticks_lookup_fails_before_gate_RAM',lambda:rejects(lambda:value['start_ticks']))
    roles=baseline_roles();assert len(roles)==47
    CHECKS.append('all47_roles_40_unique_source_refs_authenticated_from_immutable_metadata')
    # Whole original native master formula, with distinct uncreated path descriptors.
    constants={}
    for n in ast.parse((ROOT/C64/'contract_v1.py').read_text()).body:
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    c=SimpleNamespace(ref=ref,MANIFEST=C64+'/source_manifest.v1.json',**{k:constants[k]for k in ('GUARD','SAVED_INTEGRATION_REF','SAVED_SCIENTIFIC_ENTRY_REF','SAVED_SCIENCE_REVIEW_REF')})
    descriptor=lambda name:dict(path='UNCREATED_DESCRIPTOR_ONLY/'+name)
    q=dict(source_review_ref=descriptor('complete_review.json'),ROOT_saved_MLM_artifact_acceptance_ref=descriptor('actual_acceptance.json'))
    scope=dict(c=c,config_ref=ref(C64+'/scientific_config.FROZEN.v1.json'),
        master=dict(runtime_ref=ref('research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/runtime_mlm.root_active_v1.json')),
        request_ref=descriptor('prepare_master.json'),q=q,manifest=doc(C64+'/source_manifest.v1.json'))
    fn=next(n for n in ast.parse((ROOT/C64/'root_builder_v1.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_master_v1')
    nodes=[n for n in fn.body if isinstance(n,(ast.Assign,ast.AugAssign))and getattr(n,'lineno',0)in(115,117)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_exact_native_master_direct_formula','exec'),scope)
    union=[]
    for rr in list(roles.values())+scope['direct']:
        if rr not in union:union.append(rr)
    assert len(scope['direct'])==17 and len(union)==57
    CHECKS.append('original_native_master_formula_full47role_union_is57_not56_with_distinct_future_descriptors')
    return dict(v1_native_master_shape=57,all_actual_guard_roles=47,guard_unique_refs=40)
V2='research/reviews/native_saved_MLM_successor_controller_ready_20261017_v3'
def parent_RAM(case):
    import signal as real_signal
    events=[];written={};clock=[0.0];helper_live=[case in ('orphan','nonzero_adopted','persistent_refresh','transient_refresh','unresolved')]
    owner_waited=[False];adopted_done=[False];refreshes=[0];handlers={}
    def mono():clock[0]+=.1;return clock[0]
    record=dict(path='RAM_ONLY/capability.json',sha256='0'*64)
    cap=dict(drain_expires_unix=10000,base_pool_ref=dict(path='src/arggnn/gpu_pool_v1.py',sha256='0'*64))
    owner=dict(pid=123,ppid=1,group=123,session=123,start=777,state='S')
    helper=dict(pid=456,ppid=123,group=123,session=123,start=888,state='S')
    class Tree:
        def __init__(self,pid):assert pid==123;self.known={123:dict(owner)}
        def evidence(self):return dict(owned_identities=[dict(pid=r['pid'],start=r['start'])for r in self.known.values()])
        def refresh(self):
            refreshes[0]+=1
            if case!='cancel':self.known[456]=dict(helper)
            if case=='persistent_refresh' or (case=='transient_refresh'and refreshes[0]==1):raise ValueError('RAM refresh failure')
            return self
        def identities_absent(self):
            if case=='persistent_refresh':raise ValueError('RAM final absence failure')
            if case=='unresolved':return False
            return owner_waited[0]and not helper_live[0]
    def signal_identity(row,sig):
        events.append(('pidfd_attempt',row['pid'],sig))
        if row['pid']==456:
            if case=='reused_helper':events.append('reused_helper_excluded');return False
            events.append(('pidfd_helper',owner_waited[0]))
            if case!='unresolved':helper_live[0]=False
        return True
    owned=SimpleNamespace(_children=lambda pid:([456]if owner_waited[0]and helper_live[0]else[]),
        _stat=lambda pid:dict(owner),OwnedTree=Tree,signal_identity=signal_identity)
    # Initial unrelated-child check must be empty before child exists.
    child_spawned=[False]
    owned._children=lambda pid:([456]if child_spawned[0]and owner_waited[0]and helper_live[0]else[])
    class Stream:
        def fileno(self):return 9
        def close(self):events.append('stdout_closed')
    proc=SimpleNamespace(pid=123,stdout=Stream(),returncode=None)
    def popen(argv,**kw):
        assert argv[:5]==[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',V2.replace('/','.')+'.empty_gpu0_controller_v1']
        assert kw['start_new_session']is True and kw['pass_fds']==(10,)
        assert kw['env']['CUDA_VISIBLE_DEVICES']==''and kw['env']['PYTHONNOUSERSITE']=='1'
        assert 'subreaper_enabled'in events
        events.append('popen_RAM');child_spawned[0]=True
        if case=='cancel':handlers[15](15,None)
        return proc
    def waitpid(pid,flags):
        if pid==123:
            owner_waited[0]=True
            return 123,256 if case=='owner1'else 0
        if pid==-1 and not adopted_done[0]and case in ('orphan','nonzero_adopted','persistent_refresh','transient_refresh','unresolved'):
            if helper_live[0]:return 0,0
            adopted_done[0]=True
            return 456,256 if case=='nonzero_adopted'else 0
        raise ChildProcessError()
    def signal_fn(number,handler):
        previous=handlers.get(number,'RAM_OLD');handlers[number]=handler;return previous
    osfake=SimpleNamespace(environ={},getpid=lambda:1,pipe=lambda:(10,11),close=lambda fd:events.append(('close_fd',fd)),
        set_blocking=lambda *args:None,write=lambda fd,data:events.append(('gate_write',data)),
        waitid=lambda *args:object(),waitpid=waitpid,read=lambda fd,n:b'',waitstatus_to_exitcode=lambda status:status>>8,
        P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8)
    fake_sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1,optimize=0),modules={})
    backend=SimpleNamespace(self_probe_v1=lambda:events.append('pidfd_mock'))
    fake_sys.modules['pidfd_backend_v1']=backend
    def loader(rr,name):
        if name=='_empty_owner_original_owned_lifecycle':return owned
        if name=='_empty_owner_original_subreaper':return SimpleNamespace(enable_child_subreaper=lambda:events.append('subreaper_enabled'))
        return backend
    class P:
        @staticmethod
        def cwd():return ROOT
    c=SimpleNamespace(ROOT=ROOT,HERE=V2,need=need,validate_cap_v1=lambda rr:(cap,None,None,{}),
        guard_v1=lambda *args:SimpleNamespace(),load=loader)
    def put(cap,name,value):
        written[name]=copy.deepcopy(value);events.append(('put',name));return dict(path='RAM_ONLY/'+name,sha256='0'*64)
    scope=dict(c=c,Path=P,sys=fake_sys,os=osfake,time=SimpleNamespace(time=lambda:100.,monotonic=mono,sleep=lambda seconds:None),
        signal=SimpleNamespace(SIGTERM=15,SIGINT=2,SIGKILL=9,signal=signal_fn),subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),
        select=SimpleNamespace(select=lambda *args:([proc.stdout],[],[])),json=json,PIDFD={},OWNED={},OWNED_ROSTER_CAP=64,put=put)
    with redirect_stdout(io.StringIO()):rc=function(V2+'/owner_parent_v1.py','execute_v1',scope)(record)
    out=written['actual_wait.json']
    assert out['controller_start_ticks']==777
    assert out['actual_tool_outer_terminal_consumed']is False and out['storage_credit_claimed']is False
    assert out['historical_old_drain_proven']is False and out['owner_subreaper_enabled_before_spawn']is True
    assert handlers[15]==handlers[2]=='RAM_OLD'
    if case=='cancel':
        assert ('gate_write',b'G')not in events and rc==1
    else:assert events.index(('put','actual_launch.json'))<events.index(('gate_write',b'G'))
    if case in ('healthy','orphan','reused_helper'):
        assert rc==0 and out['all_owned_processes_drained']is True and out['actual_direct_returncode']==0
    else:assert rc==1
    if case in ('orphan','nonzero_adopted','persistent_refresh','transient_refresh','unresolved'):
        assert ('pidfd_helper',True)in events
    if case=='persistent_refresh':
        assert out['actual_raw_wait']==0 and out['all_owned_processes_drained']is False and out['failure']is not None
        assert out['owned_evidence']['final_observation_uncertain']is True
    if case=='nonzero_adopted':assert out['adopted_waits'][0]['returncode']==1 and out['failure']is not None
    if case=='owner1':assert out['actual_direct_returncode']==1
    if case=='unresolved':assert out['all_owned_processes_drained']is False
    return None

def lease_transaction_RAM(case):
    import os,fcntl
    source=V2+'/empty_gpu0_controller_v1.py'
    old=b'{"old":true}';disk=[old];events=[];fds={11:'scheduler',12:'lease',13:'probe'};next_open=[11]
    class P:
        def lstat(self):return st()
    def st():
        return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=2,st_size=len(disk[0]))
    expected=dict(device=1,inode=999 if case=='wrong_inode'else 2,sha256='0'*64 if case=='wrong_old_SHA'else hashlib.sha256(old).hexdigest())
    def open_(path,flags):fd=next_open[0];next_open[0]+=1;events.append(('open',fd));return fd
    def flock(fd,flags):
        events.append(('flock',fd,flags))
        if fd==13 and case!='probe_unheld':raise BlockingIOError()
    def truncate(fd,size):events.append('truncate_after_exclusion');disk[0]=disk[0][:size].ljust(size,b'\0')
    def pwrite(fd,data,offset):assert fd==12 and offset==0;disk[0]=data;return len(data)
    def inventory(cap):assert disk[0]==old;events.append('inventory_after_exclusion')
    scope=dict(os=SimpleNamespace(open=open_,fstat=lambda fd:st(),pread=lambda fd,n,o:disk[0],
        close=lambda fd:events.append(('close',fd)),ftruncate=truncate,pwrite=pwrite,fsync=lambda fd:events.append(('fsync',fd)),getpid=lambda:123,
        O_RDWR=os.O_RDWR,O_NOFOLLOW=os.O_NOFOLLOW),fcntl=SimpleNamespace(flock=flock,LOCK_EX=fcntl.LOCK_EX,LOCK_NB=fcntl.LOCK_NB),
        lease_path=P(),scheduler_path=P(),h=dict(old_physical_lease=expected),absent=lambda rows:events.append('known_absence_recheck'),
        cert=dict(known_identities=[]),stat=stat,hashlib=hashlib,need=need,inventory=inventory,cap=dict(pool='RAM_EMPTY'),
        UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667',time=SimpleNamespace(time=lambda:100.),
        encoded=lambda value:json.dumps(value,sort_keys=True,separators=(',',':')).encode())
    fn=next(n for n in ast.parse((ROOT/source).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='serve_v1')
    outer=next(n for n in fn.body if isinstance(n,ast.Try));nodes=outer.body[:3]
    error=None
    try:exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_exact_physical_lease_transaction','exec'),scope)
    except (ValueError,RuntimeError)as exc:error=exc
    if case=='valid':
        assert error is None
        value=json.loads(disk[0]);assert value['launcher_pid']==123 and value['gpu']==0 and value['logical_sharing']is True
        assert value['job']=='gpu0_shared_v1:RAM_EMPTY:gpu0'
        assert scope['lease_sha']==hashlib.sha256(disk[0]).hexdigest()
        assert events.index(('flock',13,fcntl.LOCK_EX|fcntl.LOCK_NB))<events.index('inventory_after_exclusion')<events.index('truncate_after_exclusion')
    else:assert error is not None and disk[0]==old
    assert ('close',11)in events
    return None
def final_source_checks():
    owner=doc(V2+'/source_manifest.v1.json')
    assert ref(V2+'/source_manifest.v1.json')['sha256']=='593676228c7100c73dede763fe190086e4bf40b3fd8d678a5b399dcd53863d48'
    for p,sha in owner['source_sha256'].items():assert ref(p)['sha256']==sha
    CHECKS.append('v2_all238_owner_source_refs_authenticated')
    native=V2+'/native64';manifest=doc(native+'/source_manifest.v1.json')
    assert ref(native+'/source_manifest.v1.json')['sha256']=='cc6884cb898545c37bab9313b910d6582e6c078427ca91a2e05b6a0451dbd97f'
    for rr in manifest['executed_source_refs']+manifest['source_authentication_refs']:assert ref(rr['path'])==rr
    CHECKS.append('v2_complete_native_manifest_executed_and_authenticated_refs_verified')
    for name in ('__init__.py','contract_v1.py','direct_gpu0_v1.py','execution_v1.py','worker_context_v1.py','science_binding_v1.py','root_builder_v1.py'):
        before=(ROOT/C64/name).read_text();after=(ROOT/native/name).read_text()
        assert C64 not in after and C64.replace('/','.')not in after,name
        after=after.replace(native,C64).replace(native.replace('/','.'),C64.replace('/','.'))
        a=ast.parse(before);b=ast.parse(after)
        def strip(tree):
            tree=copy.deepcopy(tree)
            if name=='direct_gpu0_v1.py':tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='ENDPOINT_SOURCE_REF'for t in n.targets))]
            return ast.dump(tree,include_attributes=False)
        assert strip(a)==strip(b),name
        CHECKS.append('v2_native_complete_'+name+'_AST_only_namespace_and_endpoint_delta')
    assert doc(native+'/scientific_config.FROZEN.v1.json')==doc(C64+'/scientific_config.FROZEN.v1.json')
    CHECKS.append('v2_all_seven_native_scientific_phase_configuration_and_outer1_bridge_unchanged')
    for name in ('empty_gpu0_controller_v1.py','owner_parent_v1.py'):
        tree=ast.parse((ROOT/V2/name).read_bytes())
        assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('kill','killpg')for n in ast.walk(tree))
    CHECKS.append('v2_owner_parent_no_numeric_signal_fallback_AST')
    return dict(v2_owner_manifest_ref=ref(V2+'/source_manifest.v1.json'),v2_native_manifest_ref=ref(native+'/source_manifest.v1.json'))

def cap_RAM_fixture():
    path=V2+'/empty_gpu0_controller_v1.py';scope=dict(ROOT=ROOT,HERE=V2,SOURCE=path,MANIFEST=V2+'/source_manifest.v1.json',
        C64_CONTRACT=C64+'/contract_v1.py',UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667',need=need,
        OLD_CAP=dict(path='research/reviews/C3_module_route_repair_root_20261016_v1/controller/capability.root_active_v1.json',sha256='4b781d0d88814508a02037c5935bb1dc6406a203cfa99939db4dabb5c04c9870'),
        OLD_ID=dict(pid=3349405,start_ticks=481281638,uid=1000),BASE_POOL=ref('src/arggnn/gpu_pool_v1.py'))
    now=datetime.now(timezone.utc).timestamp();scope.update(datetime=datetime,time=SimpleNamespace(time=lambda:now),json=json,hashlib=hashlib)
    rows={};refs={};calls=[]
    def put(label,value):
        data=json.dumps(value,sort_keys=True).encode();rr=dict(path=V2+'/RAM_ONLY/'+label+'.json',sha256=hashlib.sha256(data).hexdigest())
        rows[rr['path']]=copy.deepcopy(value);refs[rr['path']]=rr;return rr
    def read(rr):
        if rr['path']in rows:return copy.deepcopy(rows[rr['path']])
        return doc(rr['path'])
    def raw(rr):
        calls.append(rr['path'])
        if rr['path']in refs:need(refs[rr['path']]==rr,'Exact RAM ref')
        else:need(ref(rr['path'])==rr,'Exact source ref')
        return b'RAM boundary'
    def reference(name):return refs[name]if name in refs else ref(name)
    clock=dict(status='active',root_authorizer='root_agent',execution_authorized=True,template_only=False,
        created_at_utc=datetime.fromtimestamp(now-30,timezone.utc).isoformat(),expires_at_utc=datetime.fromtimestamp(now+3600,timezone.utc).isoformat())
    manifest=ref(scope['MANIFEST']);source=ref(path)
    review=put('review',dict(independent=True,verdict='SOURCE_PASS',blockers=[],candidate_manifest_ref=manifest))
    proof=put('immutable_proof',dict(RAM_ONLY=True))
    cert=put('known_absence',dict(root_authorizer='root_agent',all_listed_known_roots_absent=True,historical_drain_proven=False,
        includes_old_controller=True,known_identities=[scope['OLD_ID']],observed_at_utc=clock['created_at_utc']))
    handover=put('handover',dict(clock,schema='ROOT_empty_GPU0_successor_handover_v1',old_controller_capability_ref=scope['OLD_CAP'],
        old_controller_identity=scope['OLD_ID'],future_old_controller_and_C3_routes_revoked=True,all_current_borrower_routes_revoked=True,
        historical_old_drain_proven=False,old_files_charges_failures_unknowns_preserved=True,saved_MLM_original_outer_exit_code=1,
        saved_MLM_full_fit_success_claimed=False,old_controller_revocation_ref=proof,old_controller_signal_ref=proof,
        H058_actual_terminal_ref=proof,known_root_absence_ref=cert))
    renewal=put('renewal',dict(renewed_expires_at_utc=datetime.fromtimestamp(now+7200,timezone.utc).isoformat()))
    config=put('config',dict(schema='ROOT_empty_GPU0_owner_config_v1',output_cap_bytes=32*2**20,physical_gpu=0,jobs={},
        submit_authorized=False,headroom_mib=4096,max_cpu_rss_bytes=2*2**30))
    runtime=put('runtime',dict(schema='ROOT_empty_GPU0_owner_runtime_v1',python='.runtime/env/bin/python',source_ref=source,CUDA_VISIBLE_DEVICES=''))
    selector=ref('research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json')
    storage=put('storage',dict(job_id='RAM-empty-owner',reserved_new_bytes=32*2**20,reservation=dict(auxiliary_scopes=[],
        consumer_bindings=[source,manifest,config,runtime]),output_directory='RAM_ONLY/output',storage_policy_selector_ref=selector))
    plan=dict(clock,schema='ROOT_empty_GPU0_owner_plan_v1',jobs={},submit_authorized=False,physical_gpu=0,gpu_uuid=scope['UUID'],
        maximum_distinct_project_GPUs=1,maximum_concurrent_jobs=0,protected_envelope_bytes=0,headroom_mib=4096,
        controller_rss_cap_bytes=2*2**30,external_job_sharing_authorized=True,preserve_external_processes=True,
        borrowed_workers_use_separate_logical_gate=True,preserve_owner_until_explicit_downstream_drain=True,pool='ram_empty',
        expires_unix=now+3000,drain_expires_unix=now+3035,source_manifest_ref=manifest,source_refs={path:source},
        source_review_ref=review,handover_ref=handover,controller_job_id='RAM-empty-owner',output_directory='RAM_ONLY/output',
        storage_policy_selector_ref=selector,config_ref=config,runtime_ref=runtime,marker_repair_renewal_ref=renewal,
        base_pool_ref=scope['BASE_POOL'],ROOT_downstream_release_path=V2+'/RAM_ONLY/release.json')
    pr=put('plan',plan);cr=put('capability',dict(plan,schema='ROOT_empty_GPU0_owner_capability_v1',plan_ref=pr,controller_storage_ref=storage))
    scope.update(read=read,raw=raw,ref=reference)
    for name in ('encoded','stamp','active','socket_name','validate_cap_v1','dispatch_v1'):function(path,name,scope)
    return SimpleNamespace(scope=scope,rows=rows,refs=refs,put=put,cr=cr,pr=pr,review=review,cert=cert,handover=handover,storage=storage,
        config=config,runtime=runtime,renewal=renewal,calls=calls,clock=clock)
def capability_protocol_checks():
    f=cap_RAM_fixture();cap=f.scope['validate_cap_v1'](f.cr)[0]
    assert cap['jobs']=={}and cap['maximum_concurrent_jobs']==0
    CHECKS.append('v2_actual_empty_cap_validator_accepts_zero_jobs_zero_envelopes_RAM')
    for title,target,key,value in [
        ('nonempty_jobs','cap','jobs',{'RAM_HIDDEN':{}}),('submit','cap','submit_authorized',True),
        ('wrong_GPU','cap','physical_gpu',1),('old_envelope','cap','protected_envelope_bytes',26*2**30),
        ('nonzero_compute_slots','cap','maximum_concurrent_jobs',1),('expired','cap','expires_unix',0),
        ('not_independent','review','independent',False),('wrong_manifest_review','review','candidate_manifest_ref',{}),
        ('invented_historical_drain','handover','historical_old_drain_proven',True),
        ('MLM_outer0_relabel','handover','saved_MLM_original_outer_exit_code',0),
        ('MLM_replay_success','handover','saved_MLM_full_fit_success_claimed',True),
        ('unrevoked_old_routes','handover','future_old_controller_and_C3_routes_revoked',False),
        ('missing_known_owner','cert','known_identities',[]),('wrong_scope','storage','reserved_new_bytes',64*2**20),
        ('CUDA_visible','runtime','CUDA_VISIBLE_DEVICES','0'),('hidden_config_job','config','jobs',{'RAM':{}})
    ]:
        f=cap_RAM_fixture()
        if target=='cap':
            f.rows[f.cr['path']][key]=value;f.rows[f.pr['path']][key]=value
        else:f.rows[getattr(f,target)['path']][key]=value
        check('v2_cap_rejects_'+title,lambda f=f:rejects(lambda:f.scope['validate_cap_v1'](f.cr)))
    f=cap_RAM_fixture();cap=f.scope['validate_cap_v1'](f.cr)[0]
    status=lambda:dict(jobs=[],physical_gpu=0)
    answer,stop=f.scope['dispatch_v1'](dict(v=1,op='status'),f.cr,cap,status,lambda p:None)
    assert answer==status()and stop is False
    CHECKS.append('v2_original_status_wire_shape_RAM')
    release=dict(f.clock,schema='ROOT_empty_GPU0_owner_downstream_release_v1',controller_capability_ref=f.cr,
        all_authorized_borrowers_actually_drained=True,all_future_borrower_routes_revoked=True,genuine_wait_and_outer_refs=[f.cert])
    answer,stop=f.scope['dispatch_v1'](dict(v=1,op='shutdown',terminate=False),f.cr,cap,status,lambda p:release)
    assert stop is True
    CHECKS.append('v2_shutdown_requires_separate_plan_bound_genuine_borrower_release_RAM')
    for value in ({'v':1,'op':'submit'},{'v':1,'op':'worker'},{'v':1,'op':'cancel'},{'v':1,'op':'shutdown','terminate':True},{'v':True,'op':'status'},{'v':1,'op':'status','extra':True}):
        check('v2_wire_rejects_'+repr(value),lambda value=value:rejects(lambda:f.scope['dispatch_v1'](value,f.cr,cap,status,lambda p:release)))
    for field in ('all_authorized_borrowers_actually_drained','all_future_borrower_routes_revoked'):
        bad=dict(release);bad[field]=False
        check('v2_shutdown_rejects_'+field,lambda bad=bad:rejects(lambda:f.scope['dispatch_v1'](dict(v=1,op='shutdown',terminate=False),f.cr,cap,status,lambda p:bad)))

def whole_scope_checks():
    native=V2+'/native64';roles=baseline_roles();blobs={}
    def metadata(name):
        path=V2+'/RAM_RESERVE_ONLY/'+name+'.json';data=json.dumps(dict(RAM_ONLY=name),sort_keys=True).encode()
        blobs[path]=data;return dict(path=path,sha256=hashlib.sha256(data).hexdigest())
    constants={}
    for n in ast.parse((ROOT/native/'contract_v1.py').read_bytes()).body:
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:constants[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    c=SimpleNamespace(ref=ref,MANIFEST=native+'/source_manifest.v1.json',**{k:constants[k]for k in ('GUARD','SAVED_INTEGRATION_REF','SAVED_SCIENTIFIC_ENTRY_REF','SAVED_SCIENCE_REVIEW_REF')})
    scope=dict(c=c,config_ref=ref(native+'/scientific_config.FROZEN.v1.json'),
        master=dict(runtime_ref=ref('research/reviews/native_ARGformer_VAL000_GPU0_shared_execution_20261016_v1/mlm_v1/runtime_mlm.root_active_v1.json')),
        request_ref=metadata('native_prepare_master'),q=dict(source_review_ref=metadata('native_review'),
        ROOT_saved_MLM_artifact_acceptance_ref=metadata('native_saved_acceptance')),manifest=doc(native+'/source_manifest.v1.json'))
    fn=next(n for n in ast.parse((ROOT/native/'root_builder_v1.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_master_v1')
    nodes=[n for n in fn.body if isinstance(n,(ast.Assign,ast.AugAssign))and any(isinstance(t,ast.Name)and t.id=='direct'for t in ([n.target]if isinstance(n,ast.AugAssign)else n.targets))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'RAM_actual_fresh_native_direct_formula','exec'),scope)
    def unique(rows):
        out=[]
        for rr in rows:
            if rr not in out:out.append(rr)
        return out
    native_bindings=unique(list(roles.values())+scope['direct'])
    controller_direct=[ref(V2+'/empty_gpu0_controller_v1.py'),ref(V2+'/owner_parent_v1.py'),ref(V2+'/source_manifest.v1.json'),c.GUARD]
    controller_direct += [metadata('owner_'+name)for name in ('plan','config','runtime','request','review','handover')]
    owner_bindings=unique(list(roles.values())+controller_direct)
    assert len(native_bindings)==57 and len(owner_bindings)==50
    for rr in native_bindings+owner_bindings:
        if rr['path']not in blobs:blobs[rr['path']]=(ROOT/rr['path']).read_bytes()
        assert hashlib.sha256(blobs[rr['path']]).hexdigest()==rr['sha256']
    events=[]
    for name,bindings,cap,aux in (
        ('native',native_bindings,64*2**30,[str(ROOT/'.runtime/tmp/native_saved_MLM_Census64_RAM_ONLY')]),
        ('owner',owner_bindings,32*2**20,[])):
        request=dict(key='H056/RAM_ONLY_'+name,output=str(ROOT/('experiments/results/RAM_ONLY_'+name)),auxiliary_scopes=aux,
            reserved_bytes=cap,consumer_bindings=bindings)
        assert all(rr in bindings for rr in roles.values())
        schema_RAM(request,blobs);CHECKS.append('v2_'+name+'_full_original_scope_spec_and_all_file_bytes_helpers_RAM')
        changed=copy.deepcopy(request);changed['consumer_bindings'].append(dict(path='RAM_ONLY/disallowed.lock',sha256='0'*64))
        check('v2_'+name+'_full_scope_rejects_lock_binding',lambda changed=changed:rejects(lambda:schema_RAM(changed,blobs)))
        tree=ast.parse((ROOT/'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py').read_bytes())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_append')
        node=next(n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='event'for t in n.targets))
        authority=metadata(name+'_reserve_authority')
        env=dict(state=(None,None,None,roles),events=[None]*4095,hashes=['0'*64],operation='reserve',
            authority_ref=authority,value=dict(copy.deepcopy(request),root_admission=authority))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'RAM_actual_original_Census_event','exec'),env)
        data=json.dumps(env['event'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        assert len(data)<=65536;events.append((name,len(data)));CHECKS.append('v2_'+name+'_actual_original_Census_event_under64KiB_RAM')
    return dict(v2_native_union=57,v2_owner_union=50,all47_roles_retained=True,event_bytes=dict(events))

def request_client_RAM(case):
    f=cap_RAM_fixture();scope=f.scope;cap=scope['validate_cap_v1'](f.cr)[0];sent=[]
    ready=f.put('ready',dict(controller_capability_ref=f.cr,controller_pid=123,controller_start_ticks=777))
    answer=dict(ok=True,result=dict(physical_gpu=0,jobs=[],lifecycle='running'))
    reply=[json.dumps(answer).encode()+b'\n']
    class Client:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def settimeout(self,value):assert value==30
        def connect(self,value):assert value==scope['socket_name'](cap)
        def getsockopt(self,*args):
            return struct.pack('3i',999 if case=='wrong_peer_PID'else 123,999 if case=='wrong_peer_UID'else 1000,1000)
        def sendall(self,value):sent.append(value)
        def recv(self,n):return reply.pop(0)if reply else b''
    scope.update(socket=SimpleNamespace(socket=lambda *args:Client(),AF_UNIX=1,SOCK_STREAM=1,SOL_SOCKET=1,SO_PEERCRED=1),
        struct=struct,os=SimpleNamespace(getuid=lambda:1000,fsencode=lambda value:value.encode()),
        MAX_JSON=65536,proc_identity=lambda pid:dict(pid=pid,start_ticks=999 if case=='wrong_start'else 777))
    fn=function(V2+'/empty_gpu0_controller_v1.py','request_v1',scope)
    if case=='valid':
        assert fn(f.cr,'status',ready)==answer['result']and sent==[scope['encoded'](dict(v=1,op='status'))+b'\n']
    else:rejects(lambda:fn(f.cr,'status',ready));assert not sent
def native_controller_RAM(case):
    scope={};native=V2+'/native64'
    now=datetime.now(timezone.utc);end=datetime.fromtimestamp(now.timestamp()+500,timezone.utc).isoformat()
    clock=dict(root_authorizer='root_agent',status='active',execution_authorized=True,template_only=False,
        created_at_utc=datetime.fromtimestamp(now.timestamp()-10,timezone.utc).isoformat(),expires_at_utc=end)
    endpoint=ref(V2+'/empty_gpu0_controller_v1.py');fake=lambda name:dict(path='RAM_ONLY/'+name+'.json',sha256='0'*64)
    binding_ref=fake('binding');cap_ref=fake('cap');authority_ref=fake('attachment')
    binding=dict(clock,schema='native_VAL000_downstream_GPU0_controller_binding_v1',source_review_passed=True,
        endpoint_source_ref=endpoint,physical_gpu=0,GPU_UUID='GPU-ff06ca1f-403a-f2d0-b584-71e177452667',
        additional_controller_and_other_borrower_submissions_revoked=True,controller_preserved_until_downstream_drain=True,
        independent_source_review_ref=fake('complete_review'),route='reviewed_same_protocol_successor',
        protected_job=None,protected_envelope_bytes=0,successor_capability_explicitly_ROOT_authorized=True,
        controller_pid=123,controller_start_ticks=777,controller_capability_ref=cap_ref)
    q=dict(controller_binding_ref=binding_ref,direct_attachment_authority_ref=authority_ref,
        job_id='RAM_NATIVE_CONTRASTIVE',caps=dict(max_cuda_reserved_bytes=32*2**30),expires_at_utc=end)
    authority=dict(clock,schema='native_VAL000_downstream_direct_attachment_authority_v1',
        borrowed_physical_inode_attachment_authorized=True,additional_old_controller_submissions_revoked_while_downstream_attached=True,
        old_controller_preserved_until_downstream_drained=True,controller_pid=123,controller_start_ticks=777,
        controller_capability_ref=cap_ref,controller_binding_ref=binding_ref,downstream_job_id=q['job_id'],
        protected_envelope_bytes=0,downstream_max_cuda_reserved_bytes=32*2**30,downstream_non_Torch_margin_bytes=2*2**30,headroom_bytes=4*2**30)
    cap=dict(source_refs={endpoint['path']:endpoint},expires_unix=now.timestamp()+1000,physical_gpu=0,gpu_uuid=binding['GPU_UUID'],jobs={})
    registry=dict(controller=dict(pid=123,start_ticks=777),physical_gpu=0,lifecycle='running',safety_error=None,
        ambiguous_descendants_or_lease_holders=[],capability_ref=cap_ref,updated_unix=now.timestamp(),jobs=[])
    if case=='wrong_endpoint':binding['endpoint_source_ref']=ref(BASE+'/empty_gpu0_controller_v1.py')
    if case=='hidden_active_job':registry['jobs']=[dict(logical_job='RAM_HIDDEN',state='running')]
    if case=='phase_after_owner_expiry':q['expires_at_utc']=datetime.fromtimestamp(now.timestamp()+2000,timezone.utc).isoformat()
    rows={binding_ref['path']:binding,cap_ref['path']:cap,authority_ref['path']:authority}
    scope.update(require=need,read=lambda rr:copy.deepcopy(rows[rr['path']]),raw=lambda rr:b'RAM_ONLY',
        active=function(native+'/contract_v1.py','active',dict(datetime=datetime,timezone=timezone,require=need)),
        ENDPOINT_SOURCE_REF=endpoint,GPU0_UUID=binding['GPU_UUID'],BINDING_REF=None,
        CURRENT_C3_CAP={},CONTROLLER_PID=None,CONTROLLER_START=None,C3_JOB=None,CURRENT_CAP=None,C3_ENVELOPE_MIB=0,
        active_amendment_v1=lambda:None,time=SimpleNamespace(time=lambda:now.timestamp()),datetime=datetime,
        os=SimpleNamespace(getuid=lambda:1000),_registry_v1=lambda *args:copy.deepcopy(registry))
    function(native+'/direct_gpu0_v1.py','configure_controller_binding_v1',scope)
    fn=function(native+'/direct_gpu0_v1.py','controller_v1',scope)
    base=SimpleNamespace(proc_identity=lambda pid:dict(uid=999 if case=='wrong_UID'else 1000,start_ticks=777,state='S'))
    if case=='valid':
        value=fn(q,base);assert value[2]==dict(process_identities=[])and scope['C3_JOB']is None and scope['C3_ENVELOPE_MIB']==0
    else:rejects(lambda:fn(q,base))


def independent_v3_timing():
    old='research/reviews/native_saved_MLM_successor_controller_ready_20261017_v2'
    a=ast.parse((ROOT/old/'empty_gpu0_controller_v1.py').read_text())
    b=ast.parse((ROOT/V2/'empty_gpu0_controller_v1.py').read_text().replace(V2,old))
    def age_node(tree):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='serve_v1')
        return fn,next(n for n in fn.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(a,ast.Constant)and isinstance(a.value,str)and ('known-root certificate120s' in a.value or 'Finite ROOT-authorized known-root roster' in a.value)for a in n.value.args))
    old_fn,old_age=age_node(a);new_fn,new_age=age_node(b)
    assert '120' in ast.unparse(old_age)and 'Finite ROOT-authorized' in ast.unparse(new_age)
    old_fn.body[old_fn.body.index(old_age)]=new_age
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    CHECKS.append('v3_complete_owner_AST_only_namespace_and_exact_timing_predicate_delta')
    assert (ROOT/old/'owner_parent_v1.py').read_bytes()==(ROOT/V2/'owner_parent_v1.py').read_bytes()
    CHECKS.append('v3_parent_byte_identical_to_v2')
    scope=dict(datetime=datetime,need=need)
    stamp=function(V2+'/empty_gpu0_controller_v1.py','stamp',scope)
    now=1000.
    def iso(value):return datetime.fromtimestamp(value,timezone.utc).isoformat()
    h=dict(created_at_utc=iso(100),expires_at_utc=iso(2000))
    cap=dict(created_at_utc=iso(110),expires_unix=1800)
    expr=compile(ast.Expression(new_age.value.args[0]),'RAM_exact_v3_timing','eval')
    for label,value,expected in [('300s_guard_delay',700,True),('850s_delay',150,True),('at_window_start',110,True),('now',1000,True),('future',1001,False),('before_handover',99,False),('before_plan',109,False)]:
        env=dict(stamp=stamp,time=SimpleNamespace(time=lambda:now),h=h,cap=cap,cert=dict(observed_at_utc=iso(value)))
        assert eval(expr,env)is expected;CHECKS.append('v3_exact_timing_'+label+'_RAM')
    for label,changes in [('handover_expired',dict(h=dict(h,expires_at_utc=iso(1000)))),('plan_expired',dict(cap=dict(cap,expires_unix=1000)))]:
        env=dict(stamp=stamp,time=SimpleNamespace(time=lambda:now),h=h,cap=cap,cert=dict(observed_at_utc=iso(700)));env.update(changes)
        assert eval(expr,env)is False;CHECKS.append('v3_exact_timing_'+label+'_RAM')
    check('v3_naive_evidence_timestamp_rejects_RAM',lambda:rejects(lambda:stamp('1970-01-01T00:11:40')))
    calls=[(n.lineno,n.func.id)for n in ast.walk(new_fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='absent']
    assert len(calls)==2 and calls[0][0]<calls[1][0]
    CHECKS.append('v3_both_live_absent_rechecks_preserved_in_complete_AST_delta')

if __name__=='__main__':
    independent_v3_timing()
    summary=old_proofs()
    summary.update(final_source_checks())
    summary.update(whole_scope_checks())
    capability_protocol_checks()
    for case in ('valid','wrong_peer_PID','wrong_peer_UID','wrong_start'):
        check('v2_authenticated_status_client_'+case+'_RAM',lambda case=case:request_client_RAM(case))
    for case in ('valid','wrong_endpoint','hidden_active_job','phase_after_owner_expiry','wrong_UID'):
        check('v2_native_empty_controller_protocol_'+case+'_RAM',lambda case=case:native_controller_RAM(case))
    for case in ('valid','wrong_inode','wrong_old_SHA','probe_unheld'):
        check('v2_original_physical_inode_generation_'+case+'_RAM',lambda case=case:lease_transaction_RAM(case))
    for case in ('healthy','orphan','owner1','nonzero_adopted','persistent_refresh','transient_refresh','reused_helper','unresolved','cancel'):
        check('v2_full_parent_'+case+'_original_start_schema_and_pidfd_cleanup_RAM',lambda case=case:parent_RAM(case))
    print(json.dumps(dict(schema='independent_empty_GPU0_controller_AST_RAM_checks_v1',status='INDEPENDENT_SOURCE_AST_RAM_PASS',
        checks=CHECKS,count=len(CHECKS),summary=summary,no_actual_guard_operational_scientific_imports=True,
        process_socket_lock_GPU_state_probes=0,admissions_launches_or_Git=False),sort_keys=True,indent=2))
