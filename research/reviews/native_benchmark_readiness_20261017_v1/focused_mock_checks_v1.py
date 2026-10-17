"""Author RAM-only checks. Never imports the real guard, NumPy or scientific module."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from contextlib import contextmanager
from types import SimpleNamespace,ModuleType
import copy,hashlib,io,json,os,re,stat,sys
from contextlib import redirect_stdout
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_benchmark_readiness_20261017_v1'
SOURCE=HERE+'/aggregate_cli_v1.py'
CHECKS=[]

def module():
    value=ModuleType('_aggregate_author_ram_fixture')
    value.__file__=str(ROOT/SOURCE)
    exec(compile((ROOT/SOURCE).read_bytes(),value.__file__,'exec'),value.__dict__)
    return value

def check(name,fn):
    fn();CHECKS.append(name)

def rejects(fn):
    try:fn()
    except (ValueError,KeyError,FileExistsError):return
    raise AssertionError('Expected fail-closed rejection')

def fixture(stage='freeze'):
    m=module();rows={};events=[]
    def put(path,value,record=None):
        data=value if type(value)is bytes else (json.dumps(value,sort_keys=True)+'\n').encode()
        rr=record or dict(path=path,sha256=hashlib.sha256(data).hexdigest());rows[rr['path']]=(rr,data);return rr
    def get_raw(rr):
        saved,data=rows[rr['path']]
        if saved!=rr:raise ValueError('Immutable synthetic ref changed')
        return data
    def get_ref(path):return rows[str(path)][0]
    m.raw=get_raw;m.reference=get_ref
    src=put(m.SOURCE,b'synthetic source')
    cfg=put(m.CONFIG,m.CFG)
    for rr in (m.GUARD,m.SELECTOR,m.POLICY,m.SCIENCE,m.SCIENCE_MANIFEST):put(rr['path'],b'synthetic pinned dependency',rr)
    put(m.SCIENCE_REVIEW['path'],dict(status='SOURCE_PASS',blocking_findings=[],source_ref=m.SCIENCE,source_manifest_ref=m.SCIENCE_MANIFEST),m.SCIENCE_REVIEW)
    manifest=put(m.MANIFEST,dict(source_ref=src,scientific_source_ref=m.SCIENCE,scientific_manifest_ref=m.SCIENCE_MANIFEST,
        scientific_review_ref=m.SCIENCE_REVIEW,source_refs=[src,cfg]))
    review=put(HERE+'/SYNTHETIC_review.json',dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_ref=src,source_manifest_ref=manifest))
    runtime=put(HERE+'/SYNTHETIC_runtime.json',dict(schema='ROOT_FLARE_original5VAL_actual_CPU_runtime_v1',
        root_authorizer='root_agent',python_version='SYNTHETIC',numpy_version='SYNTHETIC',CPU_threads=1,CUDA_devices=0))
    context=put(HERE+'/SYNTHETIC_context.json',dict(synthetic=True))
    seals={parent:put(HERE+'/SYNTHETIC_'+parent+'.json',dict(synthetic=True,parent=parent))for parent in m.MISSING}
    now=datetime.now(timezone.utc);a=(now-timedelta(seconds=30)).isoformat();b=(now+timedelta(minutes=30)).isoformat()
    master=dict(schema='ROOT_FLARE_original5VAL_aggregate_master_contract_v1',root_authorizer='root_agent',active=True,
        created_at_utc=a,expires_at_utc=b,source_ref=src,source_manifest_ref=manifest,config_ref=cfg,
        guard_ref=m.GUARD,selector_ref=m.SELECTOR,policy_ref=m.POLICY,source_review_ref=review,
        runtime_ref=runtime,context_ref=context,remaining_seal_refs=seals,cpu_id=0,instance='SYNTHETIC',
        job_id='FLARE-original5VAL-aggregate-SYNTHETIC',
        output_directory='experiments/results/FLARE_original5VAL_aggregate_SYNTHETIC',
        packet_directory='research/reviews/FLARE_original5VAL_ROOT_SYNTHETIC')
    mr=put(HERE+'/SYNTHETIC_master.json',master)
    roles={str(i):put(HERE+'/SYNTHETIC_role_%02d.json'%i,dict(role=i))for i in range(47)}
    bindings=list(roles.values())+m.direct_roles(dict(master_contract_ref=mr),master)
    storage=dict(job_id=master['job_id'],output_directory=master['output_directory'],reserved_new_bytes=m.CFG['output_cap_bytes'],
        reservation=dict(reserved_bytes=m.CFG['output_cap_bytes'],key='H041/'+master['job_id'],
        output=str(ROOT/master['output_directory']),auxiliary_scopes=[str(ROOT/master['packet_directory'])],consumer_bindings=bindings),
        accounting_mode='Census_observation_two_scope_retention_v1',storage_policy_selector_ref=m.SELECTOR,completed_overlay_policy_ref=m.POLICY,
        budget_components=dict(consumer_adapter=m.GUARD,implementation=roles['0'],launcher=roles['1']),
        created_at_utc=a,expires_at_utc=b)
    sr=put(HERE+'/SYNTHETIC_storage.json',storage)
    q=dict(schema='ROOT_FLARE_original5VAL_aggregate_stage_request_v1',root_authorizer='root_agent',active=True,
        created_at_utc=a,expires_at_utc=b,stage='freeze',ROOT_owns_direct_wait_and_group_monitor=True,
        master_contract_ref=mr,storage_ref=sr,truth_access_authorized=False,score_authority_ref=None,truth_map_ref=None)
    fr=put(HERE+'/SYNTHETIC_freeze_request.json',q)
    if stage=='score':
        barrier=put(HERE+'/SYNTHETIC_barrier.json',dict(synthetic=True))
        completed=put(HERE+'/SYNTHETIC_freeze_completed.json',dict(status='completed_FLARE_original5VAL_freeze_v1',
            master_contract_ref=mr,result_ref=barrier,input_ref=fr,pid=123))
        wait=put(HERE+'/SYNTHETIC_freeze_wait.json',dict(actual_direct_wait=True,actual_returncode=0,pid=123))
        terminal=put(HERE+'/SYNTHETIC_freeze_terminal.json',dict(schema='ROOT_FLARE_original5VAL_freeze_terminal_consumed_v1',
            root_authorizer='root_agent',actual_terminal_consumed=True,actual_exit_code=0,
            all_owned_processes_and_groups_drained=True,actual_tool_session_id=42,
            completed_ref=completed,direct_wait_ref=wait,input_ref=fr))
        authority=put(HERE+'/SYNTHETIC_authority.json',dict(synthetic=True))
        truth=put(HERE+'/SYNTHETIC_handles.json',dict(schema='ROOT_FLARE_original5VAL_VAL_only_truth_handles_v1',
            root_authorizer='root_agent',VAL_truth_refs={parent:dict(path='SYNTHETIC_DO_NOT_OPEN_'+parent,sha256='0'*64)for parent in m.VAL},TEST_authorized=False))
        q=dict(q,stage='score',truth_access_authorized=True,score_authority_ref=authority,truth_map_ref=truth,
            freeze_terminal_ref=terminal,barrier_ref=barrier)
    qr=put(HERE+'/SYNTHETIC_'+stage+'_request_current.json',q)
    return SimpleNamespace(m=m,rows=rows,put=put,q=q,master=master,storage=storage,qr=qr,roles=roles,events=events)

def mutate(f,rr,operation):
    value=json.loads(f.m.raw(rr));operation(value)
    f.rows[rr['path']]=(rr,json.dumps(value).encode()) # same synthetic ref: RAM mutation rejected by real controls content gates

def guard_for(f,roles=None):
    roles=f.roles if roles is None else roles
    @contextmanager
    def mutex(root):
        assert root==ROOT;f.events.append('mutex_entry')
        try:yield
        finally:f.events.append('mutex_exit')
    def state(a,b,**kw):
        assert kw==dict(fresh=True);assert f.events[-1]=='mutex_entry';f.events.append('fresh_state')
        return (None,None,None,roles,None,None,None)
    def union(*groups):
        value=[]
        for group in groups:
            for rr in group:
                if rr not in value:value.append(rr)
        if not 1<=len(value)<=64:raise ValueError('Original64cap')
        return value
    def locked(*args,**kw):
        assert kw==dict(first=True,boundary=True);f.events.append('original_locked_check')
        return dict(bindings=roles)
    def budget(*args,**kw):f.events.append('original_boundary_guard');return dict(synthetic=True)
    return SimpleNamespace(budget=SimpleNamespace(v4=SimpleNamespace(mutex=mutex)),_state_locked=state,
        consumer_bindings_v1=union,check_storage_locked_v1=locked,budget_check_v9=budget,
        MODE='Census_observation_two_scope_retention_v1',_POLICY_REF=f.m.POLICY)
# Returned loader singleton installed below because it needs the final g variable.
def install_execute_fakes(f):
    m=f.m;g=guard_for(f);g.load_selected_guard_v1=lambda rr:g
    m.sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1,optimize=0),
        prefix=str(ROOT/'.runtime/env'),modules={},version='SYNTHETIC RAM fixture',path=[])
    m.threading=SimpleNamespace(active_count=lambda:1)
    m.os=SimpleNamespace(environ={},sched_setaffinity=lambda pid,cpus:f.events.append('mock_affinity'),
        sched_getaffinity=lambda pid:{0},getpid=lambda:123,getpgrp=lambda:123,getsid=lambda pid:123)
    m.resource=SimpleNamespace(RLIMIT_AS=1,RLIMIT_CPU=2,RLIMIT_FSIZE=3,RUSAGE_SELF=0,
        setrlimit=lambda *args:f.events.append('mock_setrlimit'),getrusage=lambda _:SimpleNamespace(ru_maxrss=1))
    m.local=lambda _:SimpleNamespace(is_dir=lambda:True)
    writes=[]
    class Writer:
        def __init__(self,directory,boundary):self.boundary=boundary
        def json(self,name,value):
            self.boundary('mock_before_write',128);writes.append((name,value))
            rr=f.put(HERE+'/SYNTHETIC_memory_output_'+name,value);self.boundary('mock_after_write')
            return rr
    m.JSONWriter=Writer
    class Science:
        def freeze_remaining_predictions_v1(self,context,seals,**kw):
            assert context==f.master['context_ref']and seals==f.master['remaining_seal_refs']
            assert set(kw)=={'max_npz_uncompressed_bytes','writer','runtime_guard'}
            assert kw['max_npz_uncompressed_bytes']==m.CFG['max_npz_uncompressed_bytes']
            for parent in m.VAL:
                kw['runtime_guard']('mock_pretruth_payload');f.events.append('payload_'+parent)
            return kw['writer'].json('barrier.json',dict(synthetic=True))
        def score_five_original_VAL_v1(self,barrier,**kw):
            assert barrier==f.q['barrier_ref']and kw['authority_ref']==f.q['score_authority_ref']
            assert kw['max_npz_uncompressed_bytes']==m.CFG['max_npz_uncompressed_bytes']
            assert kw['max_truth_uncompressed_bytes']==m.CFG['max_truth_uncompressed_bytes']
            for parent in m.VAL:
                kw['runtime_guard']('mock_all5_pretruth_payload');f.events.append('payload_'+parent)
            for parent in m.VAL:
                assert len([x for x in f.events if x.startswith('payload_')])==5
                assert kw['resolve_val_truth'](parent)['path']=='SYNTHETIC_DO_NOT_OPEN_'+parent
                f.events.append('resolve_'+parent)
            rejects(lambda:kw['resolve_val_truth']('test'))
            return kw['writer'].json('aggregate.json',dict(synthetic=True))
    science=Science()
    m.load=lambda rr,name:g if rr==m.GUARD else science if rr==m.SCIENCE else (_ for _ in ()).throw(AssertionError('Unmocked load'))
    return writes

def check_writer_RAM():
    m=module();events=[];files={};handles={};next_fd=[10]
    class P:
        def __init__(self,name):self.name=name
        def __truediv__(self,name):return P(self.name+'/'+name)
        def __str__(self):return self.name
        @property
        def parent(self):return P(self.name.rsplit('/',1)[0])
        def lstat(self):return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_dev=1,st_ino=2)
        def relative_to(self,root):return self.name
    class B(io.BytesIO):
        def fileno(self):return self.fd
        def close(self):files[self.name]=self.getvalue()
    def open_(path,flags,*args):
        path=str(path);events.append(('open',path,flags))
        if flags&os.O_CREAT:
            assert flags&(os.O_EXCL|os.O_NOFOLLOW)==os.O_EXCL|os.O_NOFOLLOW
            if path in files:raise FileExistsError(path)
        else:assert flags&(os.O_DIRECTORY|os.O_NOFOLLOW)==os.O_DIRECTORY|os.O_NOFOLLOW
        fd=next_fd[0];next_fd[0]+=1;handles[fd]=path;return fd
    def fdopen(fd,mode):
        b=B();b.fd=fd;b.name=handles[fd];return b
    m.os=SimpleNamespace(open=open_,fdopen=fdopen,fsync=lambda fd:events.append(('fsync',fd)),
        close=lambda fd:events.append(('close',fd)),fstat=lambda fd:P(handles[fd]).lstat(),
        O_WRONLY=os.O_WRONLY,O_CREAT=os.O_CREAT,O_EXCL=os.O_EXCL,O_NOFOLLOW=os.O_NOFOLLOW,
        O_RDONLY=os.O_RDONLY,O_DIRECTORY=os.O_DIRECTORY)
    m.reference=lambda path:dict(path=path,sha256=hashlib.sha256(files[path]).hexdigest())
    boundaries=[];writer=m.JSONWriter(P('SYNTHETIC_RAM_DIRECTORY'),lambda *args:boundaries.append(args))
    rr=writer.json('complete.json',dict(full='value'))
    assert json.loads(files[rr['path']])==dict(full='value')
    assert [x[0]for x in boundaries]==['before_aggregate_JSON_write','after_aggregate_JSON_write']
    assert len([x for x in events if x[0]=='fsync'])==2
    rejects(lambda:writer.json('complete.json',dict(full='second')))
    return None

if __name__=='__main__':
    # Every fixture is synthetic and stays in RAM; no original guard or science is imported.
    f=fixture();check('freeze_controls_accept_original_review_status_schema',lambda:f.m.controls(f.qr,'freeze'))
    f=fixture('score');check('score_controls_accept_distinct_consumed_freeze0',lambda:f.m.controls(f.qr,'score'))
    for title,stage,target,op in [
        ('freeze_truth_capability_rejected','freeze','q',lambda v:v.update(truth_access_authorized=True)),
        ('phase_expiry_extension_rejected','freeze','q',lambda v:v.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat())),
        ('missing_fourth_seal_rejected','freeze','master',lambda v:v['remaining_seal_refs'].pop('american_c3_val_004')),
        ('duplicate_original_bindings_rejected','freeze','storage',lambda v:v['reservation']['consumer_bindings'].append(v['reservation']['consumer_bindings'][0])),
        ('wrong_two_scope_cap_rejected','freeze','storage',lambda v:v.update(reserved_new_bytes=1)),
        ('wrong_current_selector_rejected','freeze','storage',lambda v:v.update(storage_policy_selector_ref=v['budget_components']['implementation'])),
        ('freeze_outer1_rejected_before_score','score','terminal',lambda v:v.update(actual_exit_code=1)),
        ('freeze_not_drained_rejected_before_score','score','terminal',lambda v:v.update(all_owned_processes_and_groups_drained=False)),
        ('freeze_worker_wait1_rejected_before_score','score','wait',lambda v:v.update(actual_returncode=1)),
        ('different_barrier_rejected_before_score','score','done',lambda v:v.update(result_ref=dict(path='SYNTHETIC_WRONG',sha256='0'*64))),
        ('original_review_invented_verdict_rejected','freeze','review',lambda v:(v.pop('status'),v.update(verdict='SOURCE_PASS'))),
        ('new_review_not_independent_rejected','freeze','integration_review',lambda v:v.update(independent=False)),
    ]:
        f=fixture(stage)
        rr={'q':f.qr,'master':f.q['master_contract_ref'],'storage':f.q['storage_ref'],'review':f.m.SCIENCE_REVIEW,
            'integration_review':f.master['source_review_ref']}.get(target)
        if target in ('terminal','wait','done'):
            rr=f.q['freeze_terminal_ref']
            if target!='terminal':
                t=json.loads(f.m.raw(rr));rr=t['direct_wait_ref'if target=='wait'else'completed_ref']
        mutate(f,rr,op);check(title,lambda f=f:rejects(lambda:f.m.controls(f.qr,stage)))
    f=fixture();g=guard_for(f);check('original_mutex_fresh_state47_and_original_locked_checker',lambda:f.m.initial_storage_transaction(f.q,f.master,f.storage,g))
    assert f.events==['mutex_entry','fresh_state','original_locked_check','mutex_exit']
    for count in (46,48):
        f=fixture();roles=dict(list(f.roles.items())[:count])
        if count==48:roles['extra']=dict(path='SYNTHETIC_EXTRA',sha256='0'*64)
        g=guard_for(f,roles);check('current_role_count_%d_rejected'%count,lambda:rejects(lambda:f.m.initial_storage_transaction(f.q,f.master,f.storage,g)))
    f=fixture();f.storage['reservation']['consumer_bindings'].remove(f.roles['46']);g=guard_for(f)
    check('omitted_actual_guard_role_rejected',lambda:rejects(lambda:f.m.initial_storage_transaction(f.q,f.master,f.storage,g)))
    f=fixture();f.storage['reservation']['consumer_bindings'].extend([dict(path='SYNTHETIC_%d'%i,sha256='0'*64)for i in range(11)]);g=guard_for(f)
    check('original64_binding_cap_preserved',lambda:rejects(lambda:f.m.initial_storage_transaction(f.q,f.master,f.storage,g)))
    m=module()
    capture=io.StringIO()
    with redirect_stdout(capture):code=m.main([])
    assert code==0 and json.loads(capture.getvalue())==dict(status='SOURCE_ONLY_INACTIVE',operations=0)
    CHECKS.append('no_argument_CLI_is_inactive_without_control_guard_or_science')
    class NameStop(Exception):pass
    def name_boundary(*args):raise NameStop()
    writer=m.JSONWriter(None,name_boundary)
    for name in ('freeze.invocation.json','FLARE_original_remaining4_pretruth_barrier.json','FLARE_original5VAL_descriptive_aggregate.json'):
        def valid(name=name):
            try:writer.json(name,{})
            except NameStop:return
            raise AssertionError('Valid flat JSON did not reach boundary')
        check('writer_accepts_'+name,valid)
    for name in ('../escape.json','x.txt','x/json','a.json/evil.json'):
        check('writer_rejects_'+name,lambda name=name:rejects(lambda:writer.json(name,{})))
    check('writer_rejects_NaN',lambda:rejects(lambda:writer.json('a.json',dict(v=float('nan')))))
    check('writer_rejects_oversized_JSON',lambda:rejects(lambda:writer.json('a.json',dict(v='x'*m.CFG['max_JSON_bytes']))))
    for stage in ('freeze','score'):
        f=fixture(stage);writes=install_execute_fakes(f)
        old_numpy=sys.modules.get('numpy');sys.modules['numpy']=SimpleNamespace(__version__='SYNTHETIC')
        try:result=f.m.execute(f.qr,stage)
        finally:
            if old_numpy is None:sys.modules.pop('numpy',None)
            else:sys.modules['numpy']=old_numpy
        assert len(writes)==3 and writes[0][0]==stage+'.invocation.json'and writes[-1][0]==stage+'.completed.json'
        assert writes[-1][1]['actual_direct_wait_and_ROOT_outer_consumption_pending']is True
        assert writes[-1][1]['storage_close_or_credit_authorized']is False
        assert result['automatic_retry']is False
        if stage=='freeze':assert not any(x.startswith('resolve_')for x in f.events)
        else:assert f.events.index('resolve_'+f.m.VAL[0])>f.events.index('payload_'+f.m.VAL[-1])
        CHECKS.append('full_'+stage+'_dispatch_with_only_RAM_guard_runtime_filesystem_science_fakes')
    check('writer_exclusive_no_follow_complete_fsync_and_duplicate_rejection_RAM',check_writer_RAM)
    f=fixture();install_execute_fakes(f);real_controls=f.m.controls;calls=[0]
    def drifting(rr,stage):
        value=real_controls(rr,stage);calls[0]+=1
        if calls[0]>1:value[0]['ROOT_owns_direct_wait_and_group_monitor']=False
        return value
    f.m.controls=drifting;old_numpy=sys.modules.get('numpy');sys.modules['numpy']=SimpleNamespace(__version__='SYNTHETIC')
    try:check('boundary_uncached_control_substitution_rejected',lambda:rejects(lambda:f.m.execute(f.qr,'freeze')))
    finally:
        if old_numpy is None:sys.modules.pop('numpy',None)
        else:sys.modules['numpy']=old_numpy
    print(json.dumps(dict(schema='FLARE_aggregate_author_mock_checks_v1',status='AUTHOR_CHECKS_PASS',
        count=len(CHECKS),checks=CHECKS,real_guard_imported=False,real_numpy_or_science_imported=False,
        process_or_resource_probes=0,scientific_payloads_or_truth_reads=0,admissions_or_actual_execution=0,
        independent_review=False),sort_keys=True,indent=2))
