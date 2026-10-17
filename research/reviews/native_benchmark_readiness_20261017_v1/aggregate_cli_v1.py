"""Inactive CPU entry for the exact passed original five-parent FLARE scorer.

ROOT reserves once, monitors each fresh setsid process, consumes genuine waits
and supplies a separate score authority after the genuine truth-free barrier.
No admission, launcher, controller, cleanup, close or credit is created here.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, resource, stat, sys, threading, time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE='research/reviews/native_benchmark_readiness_20261017_v1'
SOURCE=HERE+'/aggregate_cli_v1.py';MANIFEST=HERE+'/aggregate_manifest.v1.json';CONFIG=HERE+'/aggregate_config.FROZEN.v1.json'
SCIENCE={'path':'research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py','sha256':'522cacd5e580d72c2723beee12022de5efff06b14003b6c59e8be57d5027bd38'}
SCIENCE_MANIFEST={'path':'research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/source_manifest.v1.json','sha256':'3592b0fae199d43b01bd1c3cb87480423bb2045e6b87cdbed438009ad09bed0c'}
SCIENCE_REVIEW={'path':'research/reviews/FLARE_original_VAL001004_completion_independent_review_20261017_v1/review.v1.json','sha256':'073802509c5f3e74406f5a00c56d8df8969add05aec40d495643ca02ccf64e4f'}
GUARD={'path':'research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py','sha256':'72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb'}
SELECTOR={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json','sha256':'fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4'}
POLICY={'path':'research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json','sha256':'5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5'}
VAL=tuple('american_c3_val_%03d'%i for i in range(5));MISSING=VAL[1:]
CFG=dict(schema='FLARE_original5VAL_aggregate_CPU_config_v1',output_cap_bytes=32*2**20,cpu_threads=1,
    max_RSS_bytes=4*2**30,max_stage_seconds=3600,max_JSON_bytes=65536,
    max_npz_uncompressed_bytes=128*2**20,max_truth_uncompressed_bytes=128*2**20,
    CUDA_devices=0,external_ROOT_monitor_required=True,scientific_callables_unchanged=True)

def need(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute()else ROOT/p
    need(p!=ROOT and ROOT in p.parents and '..'not in p.parts and p.resolve()==p,'Canonical contained path')
    return p
def raw(record):
    need(type(record)is dict and set(record)=={'path','sha256'},'Exact immutable ref')
    p=local(record['path']);before=p.lstat()
    need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=32*2**20,'Bounded regular immutable source/metadata')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb')as f:opened=os.fstat(f.fileno());data=f.read(32*2**20+1);after=os.fstat(f.fileno())
    identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    need(identity(before)==identity(opened)==identity(after)==identity(p.lstat())and len(data)==before.st_size
        and hashlib.sha256(data).hexdigest()==record['sha256'],'Stable exact source/metadata bytes')
    return data
def reference(value):
    p=local(value);record=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest());raw(record);return record
def read(record):return json.loads(raw(record))
def finite(value):
    a=datetime.fromisoformat(value['created_at_utc']);b=datetime.fromisoformat(value['expires_at_utc']);now=datetime.now(timezone.utc)
    need(a.tzinfo is not None and b.tzinfo is not None and a<=now<b and 0<(b-a).total_seconds()<=3600,'Fresh at-most-one-hour ROOT interval')
def load(record,name):
    need(name not in sys.modules,'Fresh pinned module only')
    data=raw(record);module=type(sys)(name);module.__file__=str(ROOT/record['path']);sys.modules[name]=module
    exec(compile(data,module.__file__,'exec',dont_inherit=True,optimize=0),module.__dict__);return module

def controls(request_ref,stage):
    need(stage in ('freeze','score'),'Exact aggregate stage')
    q=read(request_ref);finite(q)
    need(q.get('schema')=='ROOT_FLARE_original5VAL_aggregate_stage_request_v1'and q.get('root_authorizer')=='root_agent'
        and q.get('active')is True and q.get('stage')==stage and stage in ('freeze','score')
        and q.get('ROOT_owns_direct_wait_and_group_monitor')is True,'One explicit ROOT monitored CPU phase')
    master=read(q['master_contract_ref']);finite(master)
    need(master.get('schema')=='ROOT_FLARE_original5VAL_aggregate_master_contract_v1'and master.get('root_authorizer')=='root_agent'
        and master.get('active')is True and master['source_ref']==reference(SOURCE)
        and master['source_manifest_ref']==reference(MANIFEST)and master['config_ref']==reference(CONFIG)
        and read(master['config_ref'])==CFG and master['guard_ref']==GUARD and master['selector_ref']==SELECTOR and master['policy_ref']==POLICY,
        'Exact current Census, source and immutable aggregate configuration')
    need(datetime.fromisoformat(master['created_at_utc'])<=datetime.fromisoformat(q['created_at_utc'])
        and datetime.fromisoformat(q['expires_at_utc'])<=datetime.fromisoformat(master['expires_at_utc']), 'Phase never resets master expiry')
    review=read(master['source_review_ref'])
    need(review.get('independent')is True and review.get('verdict')=='SOURCE_PASS'and review.get('blockers')==[]
        and review.get('source_ref')==reference(SOURCE)and review.get('source_manifest_ref')==reference(MANIFEST),'Independent complete aggregate integration review')
    original=read(SCIENCE_REVIEW)
    need(original.get('status')=='SOURCE_PASS'and original.get('blocking_findings')==[]and original.get('source_ref')==SCIENCE
        and original.get('source_manifest_ref')==SCIENCE_MANIFEST,'Exact original scientific review schema, no invented verdict fields')
    manifest=read(master['source_manifest_ref'])
    need(manifest['source_ref']==reference(SOURCE)and manifest['scientific_source_ref']==SCIENCE
        and manifest['scientific_manifest_ref']==SCIENCE_MANIFEST and manifest['scientific_review_ref']==SCIENCE_REVIEW,'Exact scientific integration closure')
    for record in manifest['source_refs']:raw(record)
    need(type(master['remaining_seal_refs'])is dict and set(master['remaining_seal_refs'])==set(MISSING)
        and type(master['cpu_id'])is int and master['cpu_id']>=0,'All four genuine remaining seals and one explicit CPU')
    raw(master['context_ref'])
    for record in master['remaining_seal_refs'].values():raw(record)
    raw(master['runtime_ref'])
    instance=master['instance'];need(type(instance)is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',instance),'Explicit new aggregate instance')
    need(master['job_id']=='FLARE-original5VAL-aggregate-'+instance
        and master['output_directory']=='experiments/results/FLARE_original5VAL_aggregate_'+instance
        and master['packet_directory']=='research/reviews/FLARE_original5VAL_ROOT_'+instance,'Exact charged output/packet namespace')
    storage=read(q['storage_ref']);own=storage['reservation']
    need(storage['job_id']==master['job_id']and storage['output_directory']==master['output_directory']
        and storage['reserved_new_bytes']==own['reserved_bytes']==CFG['output_cap_bytes']
        and own['key']=='H041/'+master['job_id']and own['output']==str(ROOT/master['output_directory'])
        and own['auxiliary_scopes']==[str(ROOT/master['packet_directory'])]
        and storage['accounting_mode']=='Census_observation_two_scope_retention_v1'
        and storage['storage_policy_selector_ref']==SELECTOR and storage['completed_overlay_policy_ref']==POLICY
        and storage['budget_components']['consumer_adapter']==GUARD,'Genuine original32MiB current-Census whole master reservation')
    need(storage['created_at_utc']==master['created_at_utc']and storage['expires_at_utc']==master['expires_at_utc'],'Original storage/master clock unchanged')
    required=direct_roles(q,master)
    need(all(r in own['consumer_bindings']for r in required)and len(own['consumer_bindings'])<=64
        and len({json.dumps(r,sort_keys=True) for r in own['consumer_bindings']})==len(own['consumer_bindings']),
        'Original unique direct source/config/runtime/master/review/guard bindings under64')
    if stage=='freeze':
        need(q.get('truth_access_authorized')is False and q.get('score_authority_ref')is None and q.get('truth_map_ref')is None,'Freeze has no truth capability')
    else:
        need(q.get('truth_access_authorized')is True,'Separate explicit five-parent VAL-only scoring capability')
        terminal=read(q['freeze_terminal_ref']);done=read(terminal['completed_ref']);wait=read(terminal['direct_wait_ref']);freeze=read(terminal['input_ref'])
        need(terminal.get('schema')=='ROOT_FLARE_original5VAL_freeze_terminal_consumed_v1'and terminal.get('root_authorizer')=='root_agent'
            and terminal.get('actual_terminal_consumed')is True and type(terminal.get('actual_exit_code'))is int and terminal['actual_exit_code']==0
            and terminal.get('all_owned_processes_and_groups_drained')is True and type(terminal.get('actual_tool_session_id'))is int and terminal['actual_tool_session_id']>0
            and wait.get('actual_direct_wait')is True and type(wait.get('actual_returncode'))is int and wait['actual_returncode']==0
            and done['status']=='completed_FLARE_original5VAL_freeze_v1'and done['master_contract_ref']==q['master_contract_ref']
            and done['result_ref']==q['barrier_ref']and done['input_ref']==terminal['input_ref']
            and done['pid']==wait['pid']and type(done['pid'])is int and done['pid']>0
            and freeze.get('schema')=='ROOT_FLARE_original5VAL_aggregate_stage_request_v1' and freeze.get('stage')=='freeze'
            and freeze.get('root_authorizer')=='root_agent' and freeze.get('active')is True
            and freeze['master_contract_ref']==q['master_contract_ref'] and freeze['storage_ref']==q['storage_ref']
            and freeze.get('truth_access_authorized')is False and freeze.get('score_authority_ref')is None
            and freeze.get('truth_map_ref')is None,'Genuine separately consumed freeze0/drain and exact barrier before score')
        for record in (q['barrier_ref'],q['score_authority_ref'],q['truth_map_ref']):raw(record)
    return q,master,storage

def direct_roles(q,master):
    return [master['source_ref'],master['source_manifest_ref'],master['config_ref'],master['runtime_ref'],
        q['master_contract_ref'],master['source_review_ref'],GUARD]

def initial_storage_transaction(q,master,storage,guard):
    """Existing guard, original mutex and all47returned roles; no accounting replacement."""
    refs=storage['budget_components']
    need(len(storage['reservation']['consumer_bindings'])<=64,'Original full reservation64ref cap')
    with guard.budget.v4.mutex(ROOT):
        state=guard._state_locked(refs['implementation']['sha256'],refs['launcher']['sha256'],fresh=True)
        need(type(state[3])is dict and len(state[3])==47,'Exact current47guard roles; never cache or drop ancestors')
        complete=guard.consumer_bindings_v1(list(state[3].values()),direct_roles(q,master))
        need(all(r in storage['reservation']['consumer_bindings']for r in complete),'Entire actual current-role/direct union fits original reserved64cap')
        # Exact public preconditions retained before the existing locked checker.
        need(storage['job_id']==master['job_id'] and storage['output_directory']==master['output_directory']
            and 0<storage['reserved_new_bytes']<=CFG['output_cap_bytes']
            and all(r in storage['reservation']['consumer_bindings']for r in (master['source_ref'],master['config_ref'],master['runtime_ref'])),
            'Original validate_storage_v9 source/config/runtime/output/cap preconditions')
        for record in (master['source_ref'],master['config_ref'],master['runtime_ref']):raw(record)
        view=guard.check_storage_locked_v1(storage,first=True,boundary=True)
        need(view.get('bindings')==state[3],'Same current role view under original mutex')
        return view

class JSONWriter:
    def __init__(self,directory,boundary):self.directory=directory;self.boundary=boundary
    def json(self,name,value):
        need(type(name)is str and re.fullmatch(r'[A-Za-z0-9_.-]{1,160}\.json',name),'Flat immutable JSON output name')
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        need(len(data)<=CFG['max_JSON_bytes'],'Complete bounded JSON result; never truncate')
        self.boundary('before_aggregate_JSON_write',len(data)+8192)
        path=self.directory/name;fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as f:need(f.write(data)==len(data),'Full immutable result write');f.flush();os.fsync(f.fileno());opened=os.fstat(f.fileno())
        named=path.lstat();need(stat.S_ISREG(named.st_mode)and named.st_nlink==1 and(opened.st_dev,opened.st_ino)==(named.st_dev,named.st_ino),'Stable singly linked result')
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
        self.boundary('after_aggregate_JSON_write',0);return reference(str(path.relative_to(ROOT)))

def call_science(stage,q,master,science,writer,boundary,read_document=read):
    common=dict(max_npz_uncompressed_bytes=CFG['max_npz_uncompressed_bytes'],writer=writer,runtime_guard=boundary)
    if stage=='freeze':
        return science.freeze_remaining_predictions_v1(master['context_ref'],master['remaining_seal_refs'],**common)
    truth=read_document(q['truth_map_ref'])
    need(truth.get('schema')=='ROOT_FLARE_original5VAL_VAL_only_truth_handles_v1'and truth.get('root_authorizer')=='root_agent'
        and set(truth['VAL_truth_refs'])==set(VAL)and truth.get('TEST_authorized')is False,'Exact five VAL truth handles, no TEST handle')
    def resolver(parent):
        need(parent in VAL,'Original VAL parents only');boundary('before_exact_VAL_truth_handle')
        return truth['VAL_truth_refs'][parent]
    return science.score_five_original_VAL_v1(q['barrier_ref'],authority_ref=q['score_authority_ref'],
        max_truth_uncompressed_bytes=CFG['max_truth_uncompressed_bytes'],resolve_val_truth=resolver,**common)

def execute(request_ref,stage):
    need(sys.dont_write_bytecode and sys.flags.no_user_site==1 and sys.flags.optimize==0 and Path.cwd()==ROOT
        and sys.prefix==str(ROOT/'.runtime/env') and Path(__file__).resolve()==ROOT/SOURCE and threading.active_count()==1,'Fresh exact repository -B -s CPU interpreter')
    need(not any(n.split('.')[0]in ('torch','torch_geometric','numpy')for n in sys.modules),'Set single-thread zero-CUDA environment before scientific imports')
    q,master,storage=controls(request_ref,stage)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    os.sched_setaffinity(0,{master['cpu_id']})
    resource.setrlimit(resource.RLIMIT_AS,(CFG['max_RSS_bytes'],CFG['max_RSS_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(CFG['max_stage_seconds'],CFG['max_stage_seconds']))
    resource.setrlimit(resource.RLIMIT_FSIZE,(CFG['max_JSON_bytes'],CFG['max_JSON_bytes']))
    pid=os.getpid();need(os.getpgrp()==os.getsid(0)==pid,'ROOT launches this one process through setsid and owns directwait/group monitor')
    runtime=read(master['runtime_ref'])
    import numpy as np
    need(runtime.get('schema')=='ROOT_FLARE_original5VAL_actual_CPU_runtime_v1'and runtime.get('root_authorizer')=='root_agent'
        and runtime['python_version']==sys.version.split()[0]and runtime['numpy_version']==np.__version__
        and runtime.get('CPU_threads')==1 and runtime.get('CUDA_devices')==0,'Actual admitted CPU runtime, no nominal runtime substitution')
    if str(ROOT/'src')not in sys.path:sys.path.insert(0,str(ROOT/'src'))
    guard=load(GUARD,'_ROOT_FLARE_original5VAL_exact_Census_guard');guard=guard.load_selected_guard_v1(SELECTOR)
    need(guard._POLICY_REF==POLICY and guard.MODE=='Census_observation_two_scope_retention_v1','Exact selected current Census singleton')
    initial_storage_transaction(q,master,storage,guard)
    deadline=time.monotonic()+min(CFG['max_stage_seconds'],(datetime.fromisoformat(q['expires_at_utc'])-datetime.now(timezone.utc)).total_seconds())
    def boundary(event,required=0):
        fresh_q,fresh_master,fresh_storage=controls(request_ref,stage)
        need((fresh_q,fresh_master,fresh_storage)==(q,master,storage),'Uncached immutable control files; no authority/config/runtime substitution')
        raw(master['runtime_ref'])
        need(time.monotonic()<deadline,'Original phase deadline; ROOT external monitor owns stuck-IO termination')
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.sched_getaffinity(0)=={master['cpu_id']}
            and all(os.environ.get(name)=='1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Unchanged single-thread CPU affinity and zero-CUDA execution')
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=CFG['max_RSS_bytes'],'Aggregate4GiB RSS failstop')
        return guard.budget_check_v9(storage,boundary=True,required_write_bytes=required)
    packet=local(master['packet_directory']);output=local(master['output_directory'])
    need(packet.is_dir()and output.is_dir(),'ROOT creates charged fresh scope directories only after genuine reserve')
    operational=JSONWriter(packet,boundary);writer=JSONWriter(output,boundary)
    operational.json(stage+'.invocation.json',dict(schema='FLARE_original5VAL_stage_consumed_v1',input_ref=request_ref,master_contract_ref=q['master_contract_ref'],stage=stage,pid=pid,automatic_retry=False))
    science=load(SCIENCE,'_ROOT_FLARE_original5VAL_exact_science')
    result=call_science(stage,q,master,science,writer,boundary)
    boundary('after_original_aggregate_scientific_callable')
    completed=operational.json(stage+'.completed.json',dict(status='completed_FLARE_original5VAL_'+stage+'_v1',input_ref=request_ref,
        master_contract_ref=q['master_contract_ref'],result_ref=result,pid=pid,query_truth_opened=stage=='score',
        actual_direct_wait_and_ROOT_outer_consumption_pending=True,automatic_retry=False,storage_close_or_credit_authorized=False))
    return dict(completed_ref=completed,result_ref=result,actual_direct_wait_and_ROOT_outer_consumption_pending=True,automatic_retry=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--freeze-ref');g.add_argument('--score-ref');a=p.parse_args(argv)
    if a.freeze_ref is None and a.score_ref is None:print(json.dumps(dict(status='SOURCE_ONLY_INACTIVE',operations=0)));return 0
    stage='freeze'if a.freeze_ref is not None else'score';value=a.freeze_ref if stage=='freeze'else a.score_ref
    print(json.dumps(execute(json.loads(value),stage),sort_keys=True),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
