"""Independent AST and RAM checks; never imports guard, science or operational modules."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE='research/reviews/frozen_benchmark_training_ready_20261017_v1'
OLD='research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1'
CHILD='research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py'
CAPTURE='research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json'
SELECTOR='research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json'
CHECKS=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path):return json.loads((ROOT/path).read_text())
def function(path,name,scope):
    tree=ast.parse((ROOT/path).read_bytes())
    fn=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),path+'#independent_RAM_AST','exec'),scope)
    return scope[name]
def rejects(fn):
    try:fn()
    except (ValueError,KeyError):return
    raise AssertionError('Expected rejection')
def check(name,fn):fn();CHECKS.append(name)
def unique(rows):
    scope=dict(raw=lambda rr:None,require=require)
    return function(OLD+'/root_contract_v1.py','unique',scope)(rows)
def lineage():
    capture=doc(CAPTURE)
    # Exact child _bindings copies all parent roles, then adds this selector role.
    roles=dict(capture['all_actual_parent_bindings'])
    assert len(roles)==46
    roles['Census_observation_explicit_selector_capsule']=ref(SELECTOR)
    assert len(roles)==47 and len({json.dumps(v,sort_keys=True)for v in roles.values()})==40
    for rr in roles.values():assert ref(rr['path'])==rr
    return roles
def original_refs():
    tree=ast.parse((ROOT/OLD/'root_contract_v1.py').read_text());result={}
    for n in tree.body:
        if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='dict':
            try:result[n.targets[0].id]={k.arg:ast.literal_eval(k.value)for k in n.value.keywords}
            except (ValueError,TypeError):pass
    return result
def direct_roles(path,scope):
    refs=original_refs()
    scope=dict(scope)
    scope['c']=SimpleNamespace(ref=ref,**{k:refs[k]for k in ('PIDFD','OWNED','NAMESPACE_REF','V4_REF')},
        POLICY_REF=ref('research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json'))
    scope['SELECTOR']=ref(SELECTOR);scope['CHILD']=ref(CHILD)
    fake=lambda name:dict(path='SYNTHETIC_ONLY_'+name+'.json',sha256='0'*64)
    q={k:fake(k)for k in ('caps_ref','runtime_ref','root_directive_ref','actual_capacity_ref','source_review_ref')}
    fn=function(path,'direct_roles',scope)
    return fn(fake('request'),q,fake('config'),fake('closure'),q['runtime_ref'])
def foundation():
    manifest=doc(BASE+'/train001_simulate_source_manifest.v1.json')
    for rr in manifest['refs']:assert ref(rr['path'])==rr
    CHECKS.append('v1_all18_source_and_compact_metadata_refs_authenticated')
    cfg=doc(BASE+'/american_c3_train_001.simulation.inactive.json')
    bound=function('src/arggnn/american_c3_simulate_v1.py','storage_bound',{})(cfg,cfg['sample_counts'])
    assert sum(cfg['sample_counts'].values())==1600 and bound==5896320256 and bound<=cfg['output_cap_bytes']==6500000000
    assert cfg['sequence_length_bp']==30000000 and cfg['sample_unit']=='haploid'
    assert cfg['ancestry_seed']==21002 and cfg['mutation_seed']==1021002
    CHECKS.append('original_full30Mb_1600hap_registered_seeds_and_scalar_count_bound')
    scope=dict(REL=BASE,OLD=OLD,SOURCE=BASE+'/train001_simulate_context_v1.py',
        DRIVER=BASE+'/train001_simulate_driver_v1.py',WORKER=BASE+'/worker_entry_v1.py',
        BRIDGE=BASE+'/american_full30mb_simulator_bridge_v1.py',MANIFEST=BASE+'/train001_simulate_source_manifest.v1.json')
    direct=direct_roles(BASE+'/train001_simulate_context_v1.py',scope);roles=lineage()
    CHECKS.append('all40_unique_guard_lineage_source_and_metadata_refs_authenticated')
    assert len(direct)==29 and len({r['path']for r in list(roles.values())+direct})==68
    check('v1_actual_role_union68_rejected_by_original_unique64_AST',lambda:rejects(lambda:unique(list(roles.values())+direct)))
    events=[]
    scope=dict(require=require,_raw=lambda rr:events.append(('raw',rr)),
        budget_check_v9=lambda storage,**kw:events.append(('budget',kw)))
    validate=function(CHILD,'validate_storage_v9',scope)
    src=dict(path='SYNTHETIC_source.py',sha256='0'*64);conf=dict(path='SYNTHETIC_config.json',sha256='0'*64);runtime=dict(path='SYNTHETIC_runtime.json',sha256='0'*64)
    spec=dict(job_id='SYNTHETIC_JOB',output_directory='SYNTHETIC_OUTPUT')
    storage=dict(spec,reserved_new_bytes=7000000000,reservation=dict(consumer_bindings=[src,conf,runtime]))
    args=dict(source_ref=src,config_ref=conf,runtime_ref=runtime)
    check('v1_original_guard_rejects7GB_whole_under6p5GB_scientific_config_AST',lambda:rejects(lambda:validate(spec,cfg,storage,**args)))
    assert events==[]
    validate(spec,dict(cfg,output_cap_bytes=7000000000),storage,**args)
    assert events[-1]==('budget',dict(first=True,boundary=True))
    CHECKS.append('original_guard_accepts_whole_scope_cap_join_in_RAM')
    return dict(scientific_bound_bytes=bound,v1_union_count=68,current_roles=47,current_unique_refs=40)
def supervisor_RAM(case='success',here=None,source=None):
    events=[];clock=[0.0];stdout=[]
    def monotonic():clock[0]+=.05;return clock[0]
    now=datetime.now(timezone.utc)
    caps=dict(reservation_bytes=7000000000,wall_seconds=10800,cleanup_seconds=30,
        rss_bytes=64*2**30,log_bytes=16*2**20,aux_bytes=160*2**20,private_work_bytes=128*2**20)
    class P:
        def __init__(self,name):self.name=str(name)
        def __truediv__(self,name):return P(self.name+'/'+name)
        def __str__(self):return self.name
        def exists(self):return False
        def mkdir(self):events.append(('mkdir',self.name))
    parent='american_c3_train_001';out='SYNTHETIC_OUTPUT';aux='SYNTHETIC_AUX'
    record=dict(path='SYNTHETIC_INPUT.json',sha256='0'*64)
    spec=dict(output_directory=out,worker_auxiliary_scope=aux,storage=dict(synthetic=True),
        runtime_ref=dict(path='SYNTHETIC_RUNTIME.json',sha256='0'*64),expires_at_utc=(now+timedelta(hours=4)).isoformat())
    q=dict(phase='simulate_train_001',parent_id=parent,stage='simulate')
    row=dict(pid=123,group=123,start_ticks=1)
    class Tree:
        known={123:row}
        def __init__(self,pid):assert pid==123
        def refresh(self):events.append('tree_refresh')
        def evidence(self):return dict(synthetic=True)
        def identities_absent(self):return case!='unresolved'
    owned=SimpleNamespace(_children=lambda pid:[],_stat=lambda pid:dict(row),_same=lambda a,b:False,
        OwnedTree=Tree,signal_identity=lambda row,sig:events.append(('signal_owned',row['pid'],sig)))
    class Stream:
        def fileno(self):return 9
        def close(self):events.append('stdout_close')
    proc=SimpleNamespace(pid=123,stdout=Stream(),returncode=None)
    def popen(argv,**kw):
        assert argv==['SYNTHETIC_PYTHON','-B','-m',(here or BASE).replace('/','.')+'.worker_entry_v1',
            '--worker-ref',json.dumps(record),'--gate-fd','10']
        assert kw['start_new_session']is True and kw['pass_fds']==(10,8)
        assert kw['env']['CUDA_VISIBLE_DEVICES']==''
        assert kw['env']['OMP_NUM_THREADS']=='8'and kw['env']['MKL_NUM_THREADS']=='1'and kw['env']['OPENBLAS_NUM_THREADS']=='1'
        assert kw['env']['PYTHONNOUSERSITE']=='1'and kw['env']['ARGGNN_C3_PREPARATION_SLOT_FD']=='8'
        assert kw['env']['TMPDIR']=='SYNTHETIC_AUX/tmp'
        events.append('popen_RAM');return proc
    def waitpid(pid,flags):
        if pid==-1:raise ChildProcessError()
        return 123,256 if case=='worker1'else 0
    def killpg(pgid,sig):raise ProcessLookupError()
    osfake=SimpleNamespace(getpid=lambda:1,getpgrp=lambda:1,getsid=lambda pid:1,environ={},
        pathsep=':',pipe=lambda:(10,11),close=lambda fd:events.append(('fd_close',fd)),
        set_blocking=lambda *args:None,write=lambda fd,data:events.append(('gate_write',data)),
        read=lambda fd,n:b'',waitid=lambda *args:object(),waitpid=waitpid,killpg=killpg,
        waitstatus_to_exitcode=lambda value:value>>8,P_PID=1,WEXITED=2,WNOHANG=4,WNOWAIT=8)
    oldhandlers={}
    def signal_fn(number,handler):
        previous=oldhandlers.get(number,'SYNTHETIC_OLD');oldhandlers[number]=handler;return previous
    sigfake=SimpleNamespace(SIGINT=2,SIGTERM=15,SIGKILL=9,signal=signal_fn)
    class PathShim:
        def __new__(cls,p):return Path(p)
        @staticmethod
        def cwd():return ROOT
    fake_sys=SimpleNamespace(executable=str(ROOT/'.runtime/env/bin/python'),modules={})
    fake_sys.modules['pidfd_backend_v1']=SimpleNamespace(self_probe_v1=lambda:events.append('pidfd_mock'))
    def loader(rr,name):
        return owned if name=='_European_outer_owned'else fake_sys.modules['pidfd_backend_v1']
    def put(path,value,**kw):events.append(('put',str(path)));return dict(path=str(path),sha256='0'*64)
    guard=SimpleNamespace(validate_storage_v9=lambda *args,**kw:events.append(('validate_storage',args[1])),
        budget_check_v9=lambda *args,**kw:events.append('guard_RAM'))
    def completion(value,request):
        assert value['parent_id']==parent;events.append('completion_after_drain')
    scope=dict(ROOT=ROOT,HERE=here or BASE,SOURCE=source or BASE+'/train001_simulate_driver_v1.py',
        PIDFD={},OWNED={},need=require,local=lambda value:P(value),ref=lambda value:dict(path=str(value),sha256='0'*64),
        read=lambda rr:dict(status='completed_C3_simulation_reconstruction',parent_id=parent),
        load=loader,put=put,current=lambda *args:None,guard=lambda consumer:guard,
        SLOT_FD=8,CAPS=caps,OUTPUT=out,PARENT=parent,CONFIG=dict(path='SYNTHETIC_CONFIG.json',sha256='0'*64),
        input_spec=lambda rr:(spec,q,{}),Path=PathShim,sys=fake_sys,os=osfake,time=SimpleNamespace(
            monotonic=monotonic,time=lambda:now.timestamp(),sleep=lambda seconds:None),
        datetime=datetime,timezone=timezone,ctypes=SimpleNamespace(CDLL=lambda *args,**kwargs:SimpleNamespace(prctl=lambda *args:0)),
        signal=sigfake,subprocess=SimpleNamespace(Popen=popen,PIPE=1,STDOUT=2),json=json,
        c=SimpleNamespace(POLICY_REF={}),worker_python=lambda q:'SYNTHETIC_PYTHON',
        sample_scope=lambda s:events.append('scope_RAM'),boundary=lambda *args,**kw:events.append('post_drain_boundary'),
        validate_completion=completion)
    fn=function(OLD+'/root_driver_v1.py','execute',scope)
    from contextlib import redirect_stdout
    import io
    with redirect_stdout(io.StringIO()):result=fn(record)
    assert events.index('popen_RAM')<events.index(('gate_write',b'G'))
    if case=='success':
        assert result['status']=='PREPARED'and result['returncode']==0
        assert result['outer_tool_terminal_consumed']is False and result['outer_truth_scored']is False
        assert 'completion_after_drain'in events
    elif case=='worker1':assert result['status']=='FAILED'and result['returncode']==1
    else:
        assert result['status']=='FAILED_UNRESOLVED_OR_EVIDENCE_MARGIN'and result['returncode']==1
        assert 'completion_after_drain'not in events
    assert oldhandlers[2]==oldhandlers[15]=='SYNTHETIC_OLD'
    return events

V4=BASE+'/train001_simulate_v4'
V4_SOURCE=V4+'/worker_entry_v1.py'
V4_MANIFEST=V4+'/source_manifest.v4.json'
V4_SCIENCE=V4+'/scientific_condition.v4.json'
def v4_scope():
    return dict(ROOT=ROOT,REL=V4,OLD=OLD,SOURCE=V4_SOURCE,DRIVER=V4_SOURCE,WORKER=V4_SOURCE,
        BRIDGE=V4_SOURCE,MANIFEST=V4_MANIFEST,SCIENCE=V4_SCIENCE,PARENT='american_c3_train_001',
        PHASE='simulate_train_001',STAGE='simulate',SELECTOR=ref(SELECTOR),CHILD=ref(CHILD),
        datetime=datetime,timezone=timezone,need=require)
def request_fixture():
    scope=v4_scope();rows={};events=[]
    fake=lambda key:dict(path='SYNTHETIC_ONLY_'+key+'.json',sha256='0'*64)
    def put(key,value):rr=fake(key);rows[rr['path']]=value;return rr
    def read(rr):
        if rr['path']in rows:return copy.deepcopy(rows[rr['path']])
        return doc(rr['path'])
    cfg=dict(doc(V4_SCIENCE)['train001_inactive'],execution_enabled=True)
    caps=doc(BASE+'/train001_simulate_caps.proposed.inactive.json')
    now=datetime.now(timezone.utc);created=(now-timedelta(seconds=5)).isoformat();expires=(now+timedelta(seconds=caps['wall_seconds']+600)).isoformat()
    clock=dict(created_at_utc=created,expires_at_utc=expires)
    review=put('review',dict(independent=True,verdict='SOURCE_PASS',blockers=[],source_manifest_ref=ref(V4_MANIFEST)))
    directive=put('directive',dict(clock,root_authorizer='root_agent',active=True,parent_id=scope['PARENT'],phase=scope['PHASE'],
        source_manifest_ref=ref(V4_MANIFEST),current_Census_original_reserve_execute_authorized=True))
    capacity=put('capacity',dict(clock,root_authorizer='root_agent',actual_current_observation=True,source_only_estimate=False,
        parent_id=scope['PARENT'],cpu_ids=list(range(8)),admissible_new_bytes=caps['reservation_bytes'],available_owned_RSS_bytes=caps['rss_bytes']))
    cr=put('caps',caps);runtime=put('runtime',doc('research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json'))
    q=dict(clock,schema='frozen_TRAIN001_full30Mb_simulator_ROOT_request_v1',root_authorizer='root_agent',active=True,
        execution_authorized=True,GPU_authorized=False,parent_id=scope['PARENT'],stage='simulate',phase=scope['PHASE'],
        fresh_unfinished_parent=True,automatic_retry=False,source_manifest_ref=ref(V4_MANIFEST),storage_policy_selector_ref=ref(SELECTOR),
        query_truth_access_authorized=False,TRAIN_truth_creation_only=True,TEST_access_authorized=False,
        source_review_ref=review,root_directive_ref=directive,actual_capacity_ref=capacity,caps_ref=cr,runtime_ref=runtime,cpu_ids=list(range(8)),
        scope_id='SYNTHETIC',scientific_config=cfg,bindings=dict(parent_id=scope['PARENT'],scientific_config=cfg),
        packet_directory=V4+'/ROOT_actual_TRAIN001_SYNTHETIC',
        output_directory='experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_SYNTHETIC')
    qr=put('request',q)
    actscope=dict(datetime=datetime,timezone=timezone,timedelta=timedelta,require=require)
    active=function(OLD+'/root_contract_v1.py','active',actscope)
    def local(path):
        path=Path(path);path=path if path.is_absolute()else ROOT/path
        require('..'not in path.parts and path.is_relative_to(ROOT),'Synthetic canonical contained path')
        return path
    c=SimpleNamespace(read=read,ref=ref,active=active,local=local)
    scope.update(c=c,initialize=lambda:(c,None),source_paths=lambda:events.append('authenticated_source_paths'),_active=None)
    validate=function(V4_SOURCE,'validate_request',scope)
    return SimpleNamespace(scope=scope,rows=rows,q=q,qr=qr,validate=validate,caps=caps,c=c,events=events)
def v4_checks():
    m=doc(V4_MANIFEST)
    assert ref(V4_MANIFEST)['sha256']=='f4e79ab210f45ff30dbe6b891f00e78f395f873f9417e48e4d0efda5aac49af6'
    assert ref(V4_SOURCE)['sha256']=='dd6aa814104979ab13467ba868b4ed2b3050c48e7c1f517556c6ac076649ce31'
    for rr in m['refs']:assert ref(rr['path'])==rr
    CHECKS.append('v4_all13_sealed_source_and_condition_refs_authenticated')
    condition=doc(V4_SCIENCE)
    assert condition['train001_inactive']==doc(BASE+'/american_c3_train_001.simulation.inactive.json')
    assert condition['original_registry']==doc('research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json')
    assert condition['original_full30mb_template']==doc('research/reviews/ARGformer_American_C3_implementation_20261015_v1/simulation_full30mb_inactive.json')
    CHECKS.append('v4_condition_values_equal_all_three_original_objects')
    direct=direct_roles(V4_SOURCE,v4_scope());roles=lineage();joined=unique(list(roles.values())+direct)
    assert len(direct)==24 and len(joined)==63 and all(r in joined for r in roles.values())
    assert all(rr in joined for rr in direct)
    CHECKS.append('v4_original_unique_AST_retains47roles_and24directroles_union63')
    f=request_fixture();f.validate(f.qr);CHECKS.append('v4_actual_request_schema_positive_RAM')
    negatives=[
        ('inactive_request','q',lambda v:v.update(active=False)),
        ('GPU_authority','q',lambda v:v.update(GPU_authorized=True)),
        ('query_truth_authority','q',lambda v:v.update(query_truth_access_authorized=True)),
        ('TEST_authority','q',lambda v:v.update(TEST_access_authorized=True)),
        ('automatic_retry','q',lambda v:v.update(automatic_retry=True)),
        ('TRAIN000_rerun','q',lambda v:v.update(parent_id='american_c3_train_000')),
        ('full30Mb_crop','q',lambda v:v['scientific_config'].update(sequence_length_bp=2000000)),
        ('sample_reduction','q',lambda v:v['scientific_config']['sample_counts'].update(ADMIX=100)),
        ('seed_change','q',lambda v:v['scientific_config'].update(ancestry_seed=999)),
        ('scientific_cap_change','q',lambda v:v['scientific_config'].update(output_cap_bytes=7000000000)),
        ('duplicate_CPU','q',lambda v:v.update(cpu_ids=[0]*8)),
        ('wrong_selector','q',lambda v:v.update(storage_policy_selector_ref=dict(path='SYNTHETIC_wrong',sha256='0'*64))),
        ('scope_escape','q',lambda v:v.update(packet_directory=V4+'/ROOT_actual_TRAIN001_X/../../outside')),
        ('short_wall_authority','q',lambda v:v.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=1000)).isoformat())),
        ('not_independent','source_review_ref',lambda v:v.update(independent=False)),
        ('review_wrong_manifest','source_review_ref',lambda v:v.update(source_manifest_ref=dict(path='SYNTHETIC_wrong',sha256='0'*64))),
        ('missing_directive','root_directive_ref',lambda v:v.update(current_Census_original_reserve_execute_authorized=False)),
        ('capacity_source_estimate','actual_capacity_ref',lambda v:v.update(source_only_estimate=True)),
        ('insufficient_storage_capacity','actual_capacity_ref',lambda v:v.update(admissible_new_bytes=1)),
        ('insufficient_RSS_capacity','actual_capacity_ref',lambda v:v.update(available_owned_RSS_bytes=1)),
        ('wrong_caps_CPU','caps_ref',lambda v:v.update(CPU_threads=1)),
        ('omitted_positive_AUX_charge','caps_ref',lambda v:v.update(reservation_bytes=6500000000)),
        ('wrong_cleanup','caps_ref',lambda v:v.update(cleanup_seconds=1)),
        ('wrong_runtime','runtime_ref',lambda v:v.update(prefix='/SYNTHETIC_wrong')),
    ]
    for title,target,mutate in negatives:
        f=request_fixture()
        key=f.qr['path']if target=='q'else f.q[target]['path']
        mutate(f.rows[key])
        check('v4_rejects_'+title,lambda f=f:rejects(lambda:f.validate(f.qr)))
    f=request_fixture();f.validate(f.qr);f.rows[f.qr['path']]['scope_id']='SYNTHETIC_SWAP'
    check('v4_rejects_lifetime_request_swap',lambda:rejects(lambda:f.validate(f.qr)))
    return dict(v4_current_roles=47,v4_direct_roles=24,v4_union_unique_refs=63)

def bridge_RAM():
    import sys,types
    events=[];scope=v4_scope();cfg=dict(doc(V4_SCIENCE)['train001_inactive'],execution_enabled=True)
    config_ref=dict(path='SYNTHETIC_config.json',sha256='0'*64);runtime_ref=dict(path='SYNTHETIC_runtime.json',sha256='0'*64)
    runtime=doc('research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json')
    condition=doc(V4_SCIENCE);original_cfg=copy.deepcopy(cfg)
    def read(rr):
        if rr==config_ref:return cfg
        if rr==runtime_ref:return runtime
        if rr==ref(V4_SCIENCE):return condition
        raise ValueError('Unexpected RAM metadata ref')
    storage=dict(job_id='SYNTHETIC',output_directory='SYNTHETIC_OUTPUT',reserved_new_bytes=7000000000,
        reservation=dict(reserved_bytes=7000000000,consumer_bindings=[ref(p)for p in (V4_SOURCE,'src/arggnn/american_c3_simulate_v1.py','src/arggnn/american_c3_common_v1.py',CHILD)]+[config_ref,runtime_ref]))
    spec=dict(root_authorizer='root_agent',execution_authorized=True,GPU_authorized=False,source_ref=ref(V4_SOURCE),
        config_ref=config_ref,runtime_ref=runtime_ref,storage=storage,job_id='SYNTHETIC',output_directory='SYNTHETIC_OUTPUT')
    prelude_scope=dict(require=require,_raw=lambda rr:events.append(('raw_RAM',rr['path'])),
        budget_check_v9=lambda storage,**kw:events.append(('original_guard_RAM',kw)))
    prelude=function(CHILD,'validate_storage_v9',prelude_scope)
    def validate(actual_spec,storage_cfg,actual_storage,**kw):
        assert storage_cfg is not cfg and storage_cfg==dict(cfg,output_cap_bytes=7000000000)
        events.append('whole_scope_storage_config_only');return prelude(actual_spec,storage_cfg,actual_storage,**kw)
    guard=SimpleNamespace(validate_storage_v9=validate,budget_check_v9=lambda *args,**kw:events.append('writer_guard_RAM'))
    def resource(event):events.append(event)
    class PathShim:
        def __init__(self,value):self.value=str(value)
        def resolve(self):return self
        def __str__(self):return self.value
        @staticmethod
        def cwd():return ROOT
    class Writer:
        def __init__(self,actual_spec,adapter):
            assert actual_spec is spec
            adapter.budget_check_v9(storage,boundary=True,required_write_bytes=8192)
            events.append('original_Writer_constructor_RAM')
    result=dict(path='SYNTHETIC_completed.json',sha256='0'*64)
    def execute(actual_cfg,writer):
        assert actual_cfg is cfg and actual_cfg==original_cfg and actual_cfg['output_cap_bytes']==6500000000
        events.append('verbatim_original_science_cfg_RAM');return result
    pkg=types.ModuleType('arggnn');sim=types.ModuleType('arggnn.american_c3_simulate_v1');common=types.ModuleType('arggnn.american_c3_common_v1')
    sim.execute=execute;pkg.american_c3_simulate_v1=sim
    common.Writer=Writer;common.runtime_report=lambda:dict(packages=runtime['packages'])
    replaced={key:sys.modules.get(key)for key in ('arggnn','arggnn.american_c3_simulate_v1','arggnn.american_c3_common_v1')}
    scope.update(Path=PathShim,sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1),prefix=str(ROOT/'.runtime/env')),
        os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':''}),read=read,metadata_ref=ref,CENSUS_CHILD=ref(CHILD),
        GENERATOR='src/arggnn/american_c3_simulate_v1.py',COMMON='src/arggnn/american_c3_common_v1.py')
    tree=ast.parse((ROOT/V4_SOURCE).read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='_OriginalWriterCurrentGuard')
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'RAM_original_writer_adapter_AST','exec'),scope)
    fn=function(V4_SOURCE,'execute_in_admitted_context_v1',scope)
    try:
        sys.modules.update({'arggnn':pkg,'arggnn.american_c3_simulate_v1':sim,'arggnn.american_c3_common_v1':common})
        assert fn(config_ref,spec=spec,current_guard=guard,current_guard_source_ref=ref(CHILD),resource_check=resource)==result
    finally:
        for key,value in replaced.items():
            if value is None:sys.modules.pop(key,None)
            else:sys.modules[key]=value
    assert cfg==original_cfg
    assert events.index('whole_scope_storage_config_only')<events.index('verbatim_original_science_cfg_RAM')
    assert 'before_original_Writer_current_Census_boundary'in events and 'after_original_Writer_current_Census_boundary'in events
    return None
def CLI_RAM(mode):
    import argparse,io
    from contextlib import redirect_stdout,contextmanager
    events=[];scope=v4_scope();record=dict(path='SYNTHETIC_input.json',sha256='0'*64)
    @contextmanager
    def slot():
        events.append('slot_enter')
        try:yield 8
        finally:events.append('slot_exit')
    c=SimpleNamespace(require=require,cpu_slot=slot)
    d=SimpleNamespace(input_spec=lambda rr:({},dict(cpu_ids=list(range(8))),{}))
    def execute(rr):
        assert d.SLOT_FD==8 and events[-1]=='slot_enter';events.append('execute_RAM')
        return dict(returncode=0)
    d.execute=execute
    def initialize():events.append('initialize_RAM');return c,d
    def worker(rr,gate):assert rr==record and gate==10;events.append('worker_RAM');return record
    def admit(rr):assert rr==record;events.append('admit_RAM');return dict(execution_started=False)
    osfake=SimpleNamespace(getcwd=lambda:str(ROOT),sched_getaffinity=lambda pid:set(range(16)),
        sched_setaffinity=lambda pid,cpus:events.append(('affinity_RAM',cpus)))
    scope.update(__doc__='RAM only',argparse=argparse,json=json,initialize=initialize,worker=worker,admit=admit,
        sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1)),os=osfake)
    main=function(V4_SOURCE,'main',scope)
    argv=[]if mode=='inactive'else ['--'+mode+'-ref',json.dumps(record)]+(['--gate-fd','10']if mode=='worker'else [])
    output=io.StringIO()
    with redirect_stdout(output):code=main(argv)
    assert code==0
    if mode=='inactive':assert events==[] and json.loads(output.getvalue())['status']=='SOURCE_ONLY_INACTIVE'
    elif mode=='execute':
        assert events==['initialize_RAM',('affinity_RAM',set(range(8))),'slot_enter','execute_RAM','slot_exit']
    elif mode=='worker':assert events==['initialize_RAM','worker_RAM']
    else:assert events==['initialize_RAM','admit_RAM']
    return None

if __name__=='__main__':
    summary=foundation()
    summary.update(v4_checks())
    check('v4_bridge_full_AST_RAM_storage_join_verbatim_science_and_writer_callbacks',bridge_RAM)
    for mode in ('inactive','admit','execute','worker'):
        check('v4_CLI_'+mode+'_dispatch_RAM',lambda mode=mode:CLI_RAM(mode))
    for case in ('success','worker1','unresolved'):
        check('unchanged_supervisor_full_'+case+'_RAM_launch_wait_gate_cleanup',lambda case=case:supervisor_RAM(case,here=V4,source=V4_SOURCE))
    print(json.dumps(dict(schema='independent_TRAIN001_AST_RAM_checks_v1',status='INDEPENDENT_AST_RAM_CHECKS_PASS',
        checks=CHECKS,count=len(CHECKS),summary=summary,
        actual_guard_science_or_operational_imports=False,actual_process_resource_GPU_state_probes=0,
        admission_launch_Git_or_scientific_payload_truth_reads=False),sort_keys=True,indent=2))
