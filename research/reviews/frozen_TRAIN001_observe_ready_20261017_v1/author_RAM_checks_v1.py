"""Author source/AST/RAM checks. No production imports, probes or payload IO."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,io,json,os,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE=REL="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
CHILD="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
CAPTURE="research/reviews/Census_observation_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json"
SELECTOR="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json"
MANIFEST=REL+"/source_manifest.v1.json";SOURCE=V5_SOURCE=REL+"/worker_entry_v1.py"
CHECKS=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(path):
    p=Path(path);p=p if p.is_absolute()else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(path):return json.loads((ROOT/path).read_text())
def v5_scope():
    return dict(ROOT=ROOT,REL=REL,OLD=OLD,SOURCE=SOURCE,DRIVER=SOURCE,WORKER=SOURCE,BRIDGE=SOURCE,
        MANIFEST=MANIFEST,SCIENCE=MANIFEST,PARENT="american_c3_train_001",PHASE="infer_train_001",STAGE="infer",
        SELECTOR=ref(SELECTOR),CHILD=ref(CHILD),datetime=datetime,timezone=timezone,need=require,Path=Path)
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
    q=dict(phase='infer_train_001',parent_id=parent,stage='infer')
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
        read=lambda rr:dict(status='completed_current_C3_inference',parent_id=parent),
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

def CLI_RAM(mode):
    import argparse,io
    from contextlib import redirect_stdout,contextmanager
    events=[];scope=v5_scope();record=dict(path='SYNTHETIC_input.json',sha256='0'*64)
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
    main=function(V5_SOURCE,'main',scope)
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

def fixture():
    rows={};by_path={};scope=v5_scope();events=[]
    def put(path,value):
        rows[path]=copy.deepcopy(value)
        rr=dict(path=path,sha256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())
        by_path[path]=rr
        return rr
    def read(rr):
        if rr["path"]in rows:return copy.deepcopy(rows[rr["path"]])
        return doc(rr["path"])
    def rref(path):
        p=str(path)
        if p.startswith(str(ROOT)+"/"):p=p[len(str(ROOT))+1:]
        return by_path[p]if p in by_path else ref(p)
    m=doc(MANIFEST);condition=m["condition"]
    packet=REL+"/RAM_ONLY_simulation_packet";out="experiments/results/RAM_ONLY_frozen_TRAIN001_simulation"
    cfg=dict(condition["simulation_config_inactive"],execution_enabled=True)
    cr=put(packet+"/config.json",cfg)
    s=dict(schema="frozen_TRAIN001_current_Census_simulator_input_v1",root_authorizer="root_agent",execution_authorized=True,
        GPU_authorized=False,parent_id="american_c3_train_001",stage="simulate",phase="simulate_train_001",
        root_driver_ref=condition["simulation_stage"]["worker_source_ref"],main_source_ref=condition["simulation_stage"]["worker_source_ref"],
        source_sha256=condition["simulation_stage"]["source_sha256"],refs=dict(config=cr),output_directory=out,
        worker_auxiliary_scope=packet+"/worker_aux")
    ir=put(packet+"/input.json",s)
    roster=[];offset=0
    for pop,count,cls in [("AFR",200,2),("EUR",200,0),("ASIA",200,1),("ADMIX",1000,None)]:
        for j in range(count):
            roster.append(dict(sample_index=offset+j,observed_node_id=offset+j,sample_id=f"{pop}_{j//2:05d}",haplotype=j%2,
                age_generations=0,role="query"if pop=="ADMIX"else"pure_source_reference",observed_reference_class=cls))
        offset+=count
    rr=put(out+"/observed/roster.json",dict(parent_id="american_c3_train_001",class_order=["EUR","EAS","AFR"],samples=roster))
    obs=dict(schema="american_c3_observed_v1",parent_id="american_c3_train_001",split="train",sequence_length_bp=30000000,
        sample_unit="haploid",class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
        query_ancestry_truth_in_inputs=False,genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG",
        migrations_and_internal_population_labels_removed=True,roster_ref=rr)
    for key,name in [("genotypes_ref","genotypes_int8.npy"),("positions_ref","positions_bp0.npy"),("alleles_ref","site_alleles.jsonl")]:
        obs[key]=dict(path=out+"/observed/"+name,sha256="0"*64)
    obsref=put(out+"/observed/manifest.json",obs)
    cp=put(out+"/completed.json",dict(status="completed_C3_simulation_reconstruction",parent_id="american_c3_train_001",
        sample_unit="haploid",class_order=["EUR","EAS","AFR"],observation_ref=obsref,sites=12345))
    wr=put(packet+"/worker_aux/actual_wait.json",dict(schema="American_C3_full_original_CPU_actual_wait_v1",input_ref=ir,
        completed_ref=cp,parent_id="american_c3_train_001",phase="simulate_train_001",actual_direct_worker_wait=True,
        actual_direct_launcher_wait=True,actual_OS_wait_status=0,returncode=0,all_owned_processes_and_groups_drained=True,
        operational_failure=None,cleanup_errors=[],adopted_waits=[],group_observations=[dict(pgid=123,absent=True)]))
    tr=put(packet+"/outer_terminal.root_actual_v1.json",dict(root_authorizer="root_agent",actual_terminal_consumed=True,
        actual_exit_code=0,input_ref=ir,completed_ref=cp,direct_wait_ref=wr,
        all_prior_launch_resume_write_capabilities_revoked=True,simulation_outputs_frozen=True))
    prior=dict(input_ref=ir,config_ref=cr,completion_ref=cp,wait_ref=wr,outer_ref=tr,observation_ref=obsref,roster_ref=rr)
    now=datetime.now(timezone.utc);clock=dict(created_at_utc=(now-timedelta(seconds=1)).isoformat(),
        expires_at_utc=(now+timedelta(seconds=12000)).isoformat())
    review=put(REL+"/RAM_ONLY_review.json",dict(verdict="SOURCE_PASS",independent=True,blockers=[],source_manifest_ref=ref(MANIFEST)))
    caps=condition["caps_proposed"]
    q=dict(clock,schema="frozen_TRAIN001_full30Mb_observation_ROOT_request_v1",root_authorizer="root_agent",active=True,
        execution_authorized=True,GPU_authorized=False,parent_id="american_c3_train_001",stage="infer",phase="infer_train_001",
        fresh_unfinished_stage=True,automatic_retry=False,source_manifest_ref=ref(MANIFEST),storage_policy_selector_ref=ref(SELECTOR),
        query_truth_access_authorized=False,GNN_optimization_authorized=False,TEST_access_authorized=False,
        current_Census_original_reserve_execute_authorized=True,source_review_ref=review,caps=caps,
        actual_capacity=dict(clock,root_authorizer="root_agent",actual_current_observation=True,source_only_estimate=False,
            parent_id="american_c3_train_001",cpu_ids=list(range(8)),admissible_new_bytes=caps["reservation_bytes"],
            available_owned_RSS_bytes=caps["rss_bytes"]),cpu_ids=list(range(8)),scope_id="RAM_ONLY",runtime_ref=condition["inference_runtime_ref"],
        prior=prior,scientific_config=dict(condition["inference_template_inactive"],execution_enabled=True,observation_ref=obsref),
        source_sha256={r["path"]:r["sha256"]for r in m["refs"]},native_MLM_dependency=False,
        packet_directory=REL+"/ROOT_actual_TRAIN001_RAM_ONLY",output_directory="experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_RAM_ONLY")
    q["bindings"]=dict(parent_id=q["parent_id"],scientific_config=q["scientific_config"],simulation_prerequisites=prior)
    qr=put(REL+"/RAM_ONLY_request.json",q)
    active=function(OLD+"/root_contract_v1.py","active",dict(datetime=datetime,timezone=timezone,timedelta=timedelta,require=require))
    def local(path):
        p=Path(path);p=p if p.is_absolute()else ROOT/p
        require(".."not in p.parts and p.is_relative_to(ROOT),"Canonical contained RAM path")
        return p
    c=SimpleNamespace(read=read,ref=rref,active=active,local=local,raw=lambda rr:require(ref(rr["path"])==rr,"Actual source ref"),
        **{k:original_refs()[k]for k in ("PIDFD","OWNED","NAMESPACE_REF","V4_REF")},
        POLICY_REF=ref("research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json"))
    scope.update(c=c,_active=None,initialize=lambda:(c,None))
    for name in ("source_paths","observation_metadata","simulation_prerequisites","direct_roles","validate_request","phase_paths","validate_completion","input_spec"):
        function(SOURCE,name,scope)
    return SimpleNamespace(rows=rows,by_path=by_path,scope=scope,c=c,q=q,qr=qr,prior=prior,condition=condition,read=read,rref=rref,events=events)

def source_and_prerequisite_checks():
    m=doc(MANIFEST)
    for rr in m["refs"]:assert ref(rr["path"])==rr
    CHECKS.append("all13_actual_source_and_guard_metadata_refs_authenticated")
    original=doc(m["condition"]["inference_config_provenance_ref"]["path"])
    assert m["condition"]["inference_template_inactive"]==dict(original,execution_enabled=False,observation_ref=None)
    simdir="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5"
    simm=doc(simdir+"/source_manifest.v5.json")
    assert m["condition"]["simulation_stage"]["source_sha256"]=={r["path"]:r["sha256"]for r in simm["refs"]}
    assert m["condition"]["simulation_config_inactive"]==doc(simdir+"/scientific_condition.v5.json")["train001_inactive"]
    CHECKS.append("exact_original_inference_config_and_v5_simulator_source_map_science_consolidated")
    f=fixture();assert f.scope["simulation_prerequisites"](f.prior,f.condition)==f.prior
    f.scope["validate_request"](f.qr);CHECKS.append("full_genuine_simulator0_drain_outer0_and_active_request_positive_RAM")
    muts=[
        ("input_other_parent","input_ref",lambda v:v.update(parent_id="american_c3_train_002")),
        ("input_old_source","input_ref",lambda v:v["root_driver_ref"].update(sha256="0"*64)),
        ("input_drifted_source_map","input_ref",lambda v:v["source_sha256"].update(unbound="0"*64)),
        ("input_nonzero_GPU","input_ref",lambda v:v.update(GPU_authorized=True)),
        ("config_seed_change","config_ref",lambda v:v.update(ancestry_seed=1)),
        ("config_crop","config_ref",lambda v:v.update(sequence_length_bp=2000000)),
        ("config_fewer_queries","config_ref",lambda v:v["sample_counts"].update(ADMIX=100)),
        ("completion_not_success","completion_ref",lambda v:v.update(status="FAILED")),
        ("completion_other_parent","completion_ref",lambda v:v.update(parent_id="american_c3_train_002")),
        ("completion_zero_sites","completion_ref",lambda v:v.update(sites=0)),
        ("wait_nonzero","wait_ref",lambda v:v.update(returncode=1)),
        ("wait_unknown","wait_ref",lambda v:v.update(actual_direct_worker_wait=False)),
        ("wait_status_notzero","wait_ref",lambda v:v.update(actual_OS_wait_status=None)),
        ("wait_not_drained","wait_ref",lambda v:v.update(all_owned_processes_and_groups_drained=False)),
        ("wait_other_input","wait_ref",lambda v:v["input_ref"].update(sha256="0"*64)),
        ("wait_cleanup_error","wait_ref",lambda v:v.update(cleanup_errors=["UNKNOWN"])),
        ("wait_adopted_nonzero","wait_ref",lambda v:v.update(adopted_waits=[dict(returncode=1)])),
        ("wait_group_alive","wait_ref",lambda v:v["group_observations"][0].update(absent=False)),
        ("outer_not_consumed","outer_ref",lambda v:v.update(actual_terminal_consumed=False)),
        ("outer_nonzero","outer_ref",lambda v:v.update(actual_exit_code=1)),
        ("outer_wrong_wait","outer_ref",lambda v:v["direct_wait_ref"].update(sha256="0"*64)),
        ("outer_unrevoked","outer_ref",lambda v:v.update(all_prior_launch_resume_write_capabilities_revoked=False)),
        ("outer_unfrozen","outer_ref",lambda v:v.update(simulation_outputs_frozen=False)),
        ("observation_query_truth","observation_ref",lambda v:v.update(query_ancestry_truth_in_inputs=True)),
        ("observation_2Mb","observation_ref",lambda v:v.update(sequence_length_bp=2000000)),
        ("observation_wrong_payload_path","observation_ref",lambda v:v["genotypes_ref"].update(path="truth/bad.npy")),
        ("roster_fewer_samples","roster_ref",lambda v:v["samples"].pop()),
        ("roster_query_labeled","roster_ref",lambda v:v["samples"][-1].update(observed_reference_class=0)),
        ("roster_swapped_diploid","roster_ref",lambda v:v["samples"][0].update(haplotype=1)),
        ("roster_duplicate_node","roster_ref",lambda v:v["samples"][1].update(observed_node_id=0)),
    ]
    for title,key,mut in muts:
        f=fixture();mut(f.rows[f.prior[key]["path"]])
        check("rejects_prerequisite_"+title,lambda f=f:rejects(lambda:f.scope["simulation_prerequisites"](f.prior,f.condition)))
    for title,mut in [
        ("inactive",lambda q:q.update(active=False)),("retry",lambda q:q.update(automatic_retry=True)),
        ("truth",lambda q:q.update(query_truth_access_authorized=True)),("TEST",lambda q:q.update(TEST_access_authorized=True)),
        ("GPU",lambda q:q.update(GPU_authorized=True)),("optimization",lambda q:q.update(GNN_optimization_authorized=True)),
        ("TRAIN000",lambda q:q.update(parent_id="american_c3_train_000")),
        ("directive",lambda q:q.update(current_Census_original_reserve_execute_authorized=False)),
        ("cap_change",lambda q:q["caps"].update(reservation_bytes=12500000000)),
        ("capacity_estimate",lambda q:q["actual_capacity"].update(source_only_estimate=True)),
        ("capacity_small",lambda q:q["actual_capacity"].update(admissible_new_bytes=1)),
        ("RSS_small",lambda q:q["actual_capacity"].update(available_owned_RSS_bytes=1)),
        ("duplicate_CPU",lambda q:q.update(cpu_ids=[0]*8)),
        ("wrong_runtime",lambda q:q["runtime_ref"].update(sha256="0"*64)),
        ("science_change",lambda q:q["scientific_config"].update(num_threads=1)),
        ("closure_omission",lambda q:q["source_sha256"].pop(next(iter(q["source_sha256"])))),
        ("scope_escape",lambda q:q.update(packet_directory=REL+"/ROOT_actual_TRAIN001_X/../../outside")),
        ("short_lifetime",lambda q:q.update(expires_at_utc=(datetime.now(timezone.utc)+timedelta(seconds=600)).isoformat())),
    ]:
        f=fixture();mut(f.rows[f.qr["path"]])
        check("rejects_request_"+title,lambda f=f:rejects(lambda:f.scope["validate_request"](f.qr)))
    f=fixture();f.rows[f.q["source_review_ref"]["path"]]["independent"]=False
    check("rejects_nonindependent_review",lambda:rejects(lambda:f.scope["validate_request"](f.qr)))
    return None

def full_reserve_checks():
    f=fixture();direct=f.scope["direct_roles"](f.qr,f.q,f.rref(f.prior["config_ref"]["path"]),f.qr,f.q["runtime_ref"])
    # The current stage config must be a distinct real metadata role.
    currentcfg=dict(path=REL+"/RAM_ONLY_current_config.json",sha256="0"*64)
    direct=f.scope["direct_roles"](f.qr,f.q,currentcfg,f.qr,f.q["runtime_ref"])
    roles=lineage();bindings=unique(list(roles.values())+direct)
    assert len(roles)==47 and len(direct)==25 and len(bindings)==64 and all(r in bindings for r in roles.values())
    CHECKS.append("complete47_guard_roles_plus25_stage_roles_union64")
    blobs={}
    for rr in bindings:
        if rr["path"]in f.rows:blobs[rr["path"]]=json.dumps(f.rows[rr["path"]],sort_keys=True).encode()
        elif rr==currentcfg:
            blobs[rr["path"]]=b'{"RAM_ONLY":true}';rr["sha256"]=hashlib.sha256(blobs[rr["path"]]).hexdigest()
        else:blobs[rr["path"]]=(ROOT/rr["path"]).read_bytes()
        assert hashlib.sha256(blobs[rr["path"]]).hexdigest()==rr["sha256"]
    root=None;fds={};counter=[10]
    class RAMPath(PurePosixPath):
        def resolve(self):return self
        def lstat(self):
            data=blobs[str(self.relative_to(root))]
            return SimpleNamespace(st_mode=stat.S_IFREG|0o600,st_nlink=1,st_size=len(data),st_dev=1,
                st_ino=int(hashlib.sha256(str(self).encode()).hexdigest()[:12],16),st_mtime_ns=1,st_ctime_ns=1)
    root=RAMPath(str(ROOT))
    class MemoryFile(io.BytesIO):
        def fileno(self):return self.fd
    def open_(path,flags):
        assert flags==os.O_RDONLY|os.O_NOFOLLOW
        path=RAMPath(path);path.lstat();fd=counter[0];counter[0]+=1;fds[fd]=path;return fd
    def fdopen(fd,mode):
        assert mode=="rb";f=MemoryFile(blobs[str(fds[fd].relative_to(root))]);f.fd=fd;return f
    scope=dict(Path=RAMPath,LEGACY=".runtime/gpup/storage_reservations.json",LOCK=".runtime/locks/scheduler.lock",
        MAX_JSON=32*2**20,json=json,hashlib=hashlib,stat=stat,
        os=SimpleNamespace(open=open_,fdopen=fdopen,fstat=lambda fd:fds[fd].lstat(),O_RDONLY=os.O_RDONLY,O_NOFOLLOW=os.O_NOFOLLOW))
    for name in ("encoded","integer","canonical","file_bytes","parse","paths_overlap","accounting_paths","bound","scope_spec"):
        function("src/arggnn/project_budget_v4.py",name,scope)
    reserve=dict(key="H041/TreeLoom-frozen-full30Mb-TRAIN001-infer-RAM_ONLY",
        output=str(root/"experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_RAM_ONLY"),
        auxiliary_scopes=[str(root/REL/"ROOT_actual_TRAIN001_RAM_ONLY/worker_aux")],
        reserved_bytes=13000000000,consumer_bindings=bindings)
    assert len(scope["scope_spec"](root,reserve))==2
    CHECKS.append("full_original_scope_spec_all_helpers_and_file_bytes_accept_exact64_RAM_filesystem")
    for title,mut in [
        ("65_refs",lambda v:v["consumer_bindings"].append(v["consumer_bindings"][0])),
        ("lock_suffix",lambda v:v["consumer_bindings"][0].update(path=OLD+"/cpu_slot_0.lock")),
        ("npz_suffix",lambda v:v["consumer_bindings"][0].update(path="RAM_FORBIDDEN.npz")),
        ("wrong_hash",lambda v:v["consumer_bindings"][0].update(sha256="0"*64)),
        ("extra_field",lambda v:v.update(extra=True)),("missing_charge",lambda v:v.pop("reserved_bytes")),
        ("bool_charge",lambda v:v.update(reserved_bytes=True)),
        ("scope_overlap",lambda v:v.update(auxiliary_scopes=[v["output"]+"/child"])),
        ("accounting_overlap",lambda v:v.update(output=str(root/".runtime/storage/RAM_FORBIDDEN"))),
        ("relative_output",lambda v:v.update(output="experiments/results/RAM")),
    ]:
        value=copy.deepcopy(reserve);mut(value)
        check("full_reserve_rejects_"+title,lambda value=value:rejects(lambda:scope["scope_spec"](root,value)))
    child=ast.parse((ROOT/CHILD).read_text())
    append=next(n for n in child.body if isinstance(n,ast.FunctionDef)and n.name=="_append")
    event=next(n for n in append.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="event"for t in n.targets))
    authority=dict(path=REL+"/ROOT_actual_TRAIN001_RAM_ONLY/reserve_authority.json",sha256="0"*64)
    env=dict(state=(None,None,None,roles),events=[],hashes=[],operation="reserve",authority_ref=authority,
        value=dict(copy.deepcopy(reserve),root_admission=authority))
    exec(compile(ast.Module(body=[event],type_ignores=[]),"RAM_exact_Census_event","exec"),env)
    eventbytes=len(scope["encoded"](env["event"])+b"\n")
    assert eventbytes<=65536
    CHECKS.append("original_Census_actual_reserve_event_shape_fits64KiB")
    return dict(current_guard_roles=47,current_unique_refs=40,direct_stage_roles=len(direct),union_unique_refs=64,reserve_event_bytes=eventbytes)


def lifecycle_and_source_checks():
    text=(ROOT/SOURCE).read_text();tree=ast.parse(text)
    init=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="initialize")
    assert 'd.validate_completion=validate_completion'in ast.get_source_segment(text,init)
    assert 'd.execute='not in text and 'snippet=snippet.replace'in text
    assert 'os.getpid()==os.getpgrp()==os.getsid(0)'in text
    assert 'Original shared slot is not genuinely held'in text
    assert 'biological_tsinfer_v1'in text
    assert 'inference.execute(cfg,Writer('in text
    assert 'simulator.execute('not in text
    CHECKS.append("actual_original_supervisor_execute_unchanged_dynamic_context_and_original_shared_pool")
    original=(ROOT/"src/arggnn/american_c3_observe_v3.py").read_text()
    assert ref("src/arggnn/american_c3_observe_v3.py")["sha256"]=="4f381c24b75f6ab94d80cf457605375ad01adaed9dcc46dadc27f3dea4f3e630"
    assert "tsinfer.infer(samples,num_threads=cfg['num_threads'],record_provenance=True)"in original
    assert "tsdate.date(collapsed,mutation_rate=cfg['mutation_rate'],record_provenance=True)"in original
    assert "filter_sites=False,filter_individuals=False,filter_populations=False,map_nodes=True"in original
    CHECKS.append("pinned_original_tsinfer_tsdate_all_sites_all_samples_algorithm_unchanged")
    runtime=doc(doc(MANIFEST)["condition"]["inference_runtime_ref"]["path"])
    python=function(OLD+"/root_driver_v1.py","worker_python",dict(ROOT=ROOT,read=lambda rr:runtime,need=require))
    assert python(dict(stage="infer",runtime_ref={}))==str(ROOT/".runtime/envs/biological_tsinfer_v1/bin/python")
    CHECKS.append("unchanged_original_worker_python_selects_exact_existing_inference_runtime")
    for mode in ("inactive","admit","execute","worker"):
        check("concrete_CLI_"+mode+"_dispatch_RAM",lambda mode=mode:CLI_RAM(mode))
    for case in ("success","worker1","unresolved"):
        check("unchanged_full_supervisor_"+case+"_RAM_owned_gate_wait_cleanup",lambda case=case:supervisor_RAM(case,here=REL,source=SOURCE))


def admitted_input_RAM():
    f=fixture();f.scope["validate_request"](f.qr)
    sources=f.q["source_sha256"];cfgref=dict(path=f.q["packet_directory"]+"/config.json",sha256="1"*64)
    f.rows[cfgref["path"]]=f.q["scientific_config"];f.by_path[cfgref["path"]]=cfgref
    needed=f.scope["direct_roles"](f.qr,f.q,cfgref,f.qr,f.q["runtime_ref"])
    storage=dict(reservation=dict(reserved_bytes=f.q["caps"]["reservation_bytes"],consumer_bindings=unique(list(lineage().values())+needed)),
        completed_overlay_policy_ref=f.c.POLICY_REF,budget_components=dict(consumer_adapter=ref(CHILD)))
    sr=dict(path=f.q["packet_directory"]+"/storage.json",sha256="2"*64);f.rows[sr["path"]]=storage;f.by_path[sr["path"]]=sr
    value=dict(root_stage_request_ref=f.qr,root_driver_ref=ref(SOURCE),parent_id="american_c3_train_001",phase="infer_train_001",
        stage="infer",GPU_authorized=False,storage=storage,output_directory=f.q["output_directory"],runtime_ref=f.q["runtime_ref"],
        created_at_utc=f.q["created_at_utc"],expires_at_utc=f.q["expires_at_utc"],caps_ref=f.qr,caps_locator=["caps"],
        source_sha256=sources,transitive_source_closure_ref=f.qr,refs=dict(storage=sr,config=cfgref))
    ir=dict(path=f.q["packet_directory"]+"/input.json",sha256="3"*64);f.rows[ir["path"]]=value;f.by_path[ir["path"]]=ir
    assert f.scope["input_spec"](ir)[2]==f.q["scientific_config"]
    CHECKS.append("complete_actual_input_spec_RAM_with64_union_and_real_request_closure")
    for title,mut in [
        ("closure_swap",lambda v:v.update(transitive_source_closure_ref=cfgref)),
        ("unbound_predecessor",lambda v:v["storage"]["reservation"]["consumer_bindings"].remove(f.prior["outer_ref"])),
        ("wrong_whole_cap",lambda v:v["storage"]["reservation"].update(reserved_bytes=1)),
        ("wrong_caps_locator",lambda v:v.update(caps_locator=[])),
    ]:
        original=copy.deepcopy(value);mut(value)
        # Storage JSON is an independent bound metadata object in production.
        f.rows[sr["path"]]=copy.deepcopy(value["storage"])
        check("actual_input_spec_rejects_"+title,lambda:rejects(lambda:f.scope["input_spec"](ir)))
        value.clear();value.update(original);f.rows[sr["path"]]=copy.deepcopy(value["storage"])

def original_bridge_RAM():
    import sys,types
    f=fixture();events=[];scope=dict(f.scope)
    cfg=copy.deepcopy(f.q["scientific_config"]);before=copy.deepcopy(cfg)
    cr=dict(path=REL+"/RAM_ONLY_bridge_config.json",sha256="1"*64);f.rows[cr["path"]]=cfg
    runtime=f.read(f.q["runtime_ref"]);storage=dict(job_id="RAM_ONLY",output_directory="RAM_ONLY_OUTPUT",reserved_new_bytes=13000000000,
        reservation=dict(reserved_bytes=13000000000,consumer_bindings=[ref(p)for p in (SOURCE,"src/arggnn/american_c3_observe_v3.py",
            "src/arggnn/american_c3_simulate_v1.py","src/arggnn/american_c3_common_v1.py",CHILD)]+[cr,f.q["runtime_ref"]]))
    spec=dict(root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,source_ref=ref(SOURCE),config_ref=cr,
        runtime_ref=f.q["runtime_ref"],root_stage_request_ref=f.qr,storage=storage,job_id="RAM_ONLY",output_directory="RAM_ONLY_OUTPUT")
    prelude=function(CHILD,"validate_storage_v9",dict(require=require,_raw=lambda r:events.append(("raw",r)),
        budget_check_v9=lambda *a,**k:events.append(("guard",k))))
    def validate(actual,storagecfg,actualstorage,**kw):
        assert storagecfg==dict(cfg,output_cap_bytes=13000000000)and storagecfg is not cfg
        events.append("whole_scope_only");return prelude(actual,storagecfg,actualstorage,**kw)
    guard=SimpleNamespace(validate_storage_v9=validate,budget_check_v9=lambda *a,**k:events.append("Writer_guard"))
    class Writer:
        def __init__(self,actual,adapter):
            assert actual is spec
            adapter.budget_check_v9(storage,boundary=True,required_write_bytes=8192);events.append("Writer")
    result=dict(path="RAM_ONLY_completed.json",sha256="0"*64)
    def execute(actual,w):
        assert actual==before and actual["output_cap_bytes"]==12500000000
        events.append("original_inference_verbatim");return result
    pkg=types.ModuleType("arggnn");inference=types.ModuleType("arggnn.american_c3_observe_v3")
    common=types.ModuleType("arggnn.american_c3_common_v1");inference.execute=execute
    inference.__file__=str(ROOT/"src/arggnn/american_c3_observe_v3.py");pkg.american_c3_observe_v3=inference
    common.Writer=Writer;common.runtime_report=lambda:dict(packages=runtime["packages"])
    modules={"arggnn":pkg,"arggnn.american_c3_observe_v3":inference,"arggnn.american_c3_common_v1":common}
    previous={key:sys.modules.get(key)for key in modules}
    def read(rr):
        if rr==cr:return cfg
        return f.read(rr)
    class PathShim:
        def __new__(cls,value):return Path(value)
        @staticmethod
        def cwd():return ROOT
    scope.update(Path=PathShim,sys=SimpleNamespace(dont_write_bytecode=True,flags=SimpleNamespace(no_user_site=1),
        prefix=str(ROOT/".runtime/envs/biological_tsinfer_v1")),os=SimpleNamespace(environ={"CUDA_VISIBLE_DEVICES":""}),
        read=read,metadata_ref=ref,CENSUS_CHILD=ref(CHILD),local=lambda p:ROOT/p,
        INFERENCE="src/arggnn/american_c3_observe_v3.py",GENERATOR="src/arggnn/american_c3_simulate_v1.py",
        COMMON="src/arggnn/american_c3_common_v1.py")
    tree=ast.parse((ROOT/SOURCE).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="_OriginalWriterCurrentGuard")
    exec(compile(ast.Module(body=[cls],type_ignores=[]),"RAM_Writer_bridge","exec"),scope)
    fn=function(SOURCE,"execute_in_admitted_context_v1",scope)
    try:
        sys.modules.update(modules)
        assert fn(cr,spec=spec,current_guard=guard,current_guard_source_ref=ref(CHILD),resource_check=events.append)==result
    finally:
        for key,value in previous.items():
            if value is None:sys.modules.pop(key,None)
            else:sys.modules[key]=value
    assert cfg==before and events.index("whole_scope_only")<events.index("original_inference_verbatim")
    assert "before_original_Writer_current_Census_boundary"in events and "after_original_Writer_current_Census_boundary"in events
    CHECKS.append("bridge_full_AST_RAM_original_guard_verbatim12p5GB_config_Writer_callbacks")

if __name__=="__main__":
    source_and_prerequisite_checks();summary=full_reserve_checks();admitted_input_RAM();original_bridge_RAM();lifecycle_and_source_checks()
    print(json.dumps(dict(schema="frozen_TRAIN001_observe_author_AST_RAM_checks_v1",status="AUTHOR_AST_RAM_CHECKS_PASS",
        count=len(CHECKS),checks=CHECKS,summary=summary,source_ref=ref(SOURCE),source_manifest_ref=ref(MANIFEST),checks_source_ref=ref(REL+"/author_RAM_checks_v1.py"),production_guard_science_or_operational_imports=False,
        actual_process_resource_or_device_probes=0,actual_admission_launch=False,payload_truth_checkpoint_reads=False),sort_keys=True,indent=2))
