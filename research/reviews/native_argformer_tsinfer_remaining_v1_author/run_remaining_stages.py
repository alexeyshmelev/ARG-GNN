#!/usr/bin/env python3
"""Root-only remaining native stages: reuse TRAIN fits and VAL000 probe; fresh r003."""
from pathlib import Path
import argparse,copy,datetime,hashlib,importlib.util,json,os,signal,subprocess,sys,traceback

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=Path(__file__).resolve().parent
OLD=ROOT/'research/reviews/native_argformer_tsinfer_fit_v7_execution_root'
PY=ROOT/'.runtime/env/bin/python'
SOURCE=ROOT/'src/arggnn/native_argformer_tsinfer_fit_v7.py'
CONFIG=ROOT/'configs/native_argformer_tsinfer_fit_v7.json'
SOURCE_SHA='b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712'
CONFIG_SHA='d744aad4883062d23f874d15be65068b5f1b5e8a44db20ed2e8a425a4fd336c1'
PLAN_REF={'path':'research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3/execution_plan.json','sha256':'0f388eaee0d9b4ff88a6dca432170a7f5a01e7f1278870c985a827b84fbda6a1'}
BARE_UUID='976a5cd3-df12-2535-fa51-e3ea063688ba'
PARENTS=(('val000','bio_paths_pilot_val_000'),('val001','bio_paths_pilot_val_001'))
REUSE_REF={'path': 'research/reviews/native_argformer_tsinfer_remaining_v1_author/reused_receipts.json', 'sha256': 'b593f13f718600461e4a9c0ccab6ad1dd5add468c832e58c5e8057272ff14cef'}

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):
    path=Path(path)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
def read_pinned(record):
    path=ROOT/record['path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('changed source/metadata: '+str(path))
    return json.loads(raw)
def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19])
def event(**value):print(json.dumps({'utc':utc(),**value}),flush=True)
def setup():
    if sha(SOURCE)!=SOURCE_SHA or sha(CONFIG)!=CONFIG_SHA:raise ValueError('unchanged native source/config required')
    cfg=json.loads(CONFIG.read_bytes());plan=read_pinned(PLAN_REF)
    budget=read_pinned(plan['budget_receipt'])
    spec=importlib.util.spec_from_file_location('unchanged_native_v7_metadata',SOURCE)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    proposals={}
    for _,parent in PARENTS:
        own=budget['admitted_reservations']['native_argformer_tsinfer_fit_v7/'+parent]
        if own['reserved_bytes']!=20_000_000_000:raise ValueError('existing own20GB reservation required')
        for stage in ('probe-memory','fit','retrieve'):
            entry=next(x for x in plan['stage_proposals'] if x['parent']==parent and x['stage']==stage)
            proposal=read_pinned(entry['admission_proposal'])
            registration=read_pinned(entry['experiment_registration'])
            if proposal['experiment_registration']!=entry['experiment_registration'] or registration['scope_parents']!=[parent] or registration['stage']!=stage:raise ValueError('original registered stage differs')
            review=read_pinned(proposal['independent_review'])
            if review['verdict']!='PASS' or review['implementation']!={'path':str(SOURCE.relative_to(ROOT)),'sha256':SOURCE_SHA} or review['config']!={'path':str(CONFIG.relative_to(ROOT)),'sha256':CONFIG_SHA}:raise ValueError('original native source/config independent PASS required')
            proposals[parent,stage]=proposal
    return cfg,plan,budget,native,proposals

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--runtime');parser.add_argument('--expected-runtime-sha256')
    args=parser.parse_args()
    cfg,plan,budget,native,proposals=setup()
    if not args.execute:
        print(json.dumps({'status':'inactive_root_continuation_ready','root_authorization_required':True,
          'native_source':{'path':str(SOURCE.relative_to(ROOT)),'sha256':SOURCE_SHA},
          'native_config':{'path':str(CONFIG.relative_to(ROOT)),'sha256':CONFIG_SHA},
          'run_suffix':'r003','physical_GPU':2,'qualified_GPU_UUID':BARE_UUID,
          'schedule':'VAL000 fit using passed r002 probe -> CPU retrieval; VAL001 fresh probe -> fit -> CPU retrieval. CPU retrieval may overlap VAL001 GPU stages.',
          'completed_TRAIN_receipts_and_VAL000_probe_reused':REUSE_REF,
          'reservations':'reuse existing four20GB; no ledger mutation; old runtime packet unchanged',
          'short_tmp':'recomputed using unchanged admission_bindings; original fixed per-parent approved short_tmp',
          'native_science':'unchanged full4096 probe, all280 transductive MLM, native objectives/epochs and exact CPU retrieval'},indent=2))
        return
    if not args.runtime or not args.expected_runtime_sha256:raise ValueError('root-issued active corrected runtime path and SHA required')
    runtime_ref={'path':str(native.confined(args.runtime).relative_to(ROOT)),'sha256':args.expected_runtime_sha256}
    runtime=read_pinned(runtime_ref)
    original_runtime=read_pinned(plan['runtime_GPU2'])
    expected_runtime=copy.deepcopy(original_runtime);expected_runtime['qualified_GPU_UUID']=BARE_UUID
    if runtime!=expected_runtime:raise ValueError('only root runtime UUID representation correction is permitted; active unchanged qualification required')
    for label,parent in PARENTS:
        own=budget['admitted_reservations']['native_argformer_tsinfer_fit_v7/'+parent]
        for stage in ('probe-memory','fit','retrieve'):
            if parent=='bio_paths_pilot_val_000' and stage=='probe-memory':continue
            bindings=native.admission_bindings(cfg,parent,stage,label+'_'+('probe_memory' if stage=='probe-memory' else stage)+'_r003')
            if bindings['parent_storage_scope']!=own['output'] or [bindings['short_tmp']]!=own['auxiliary_scopes']:raise ValueError('exact existing parent/temp reservation required')
            if Path(bindings['output']).exists() or Path(bindings['launch_log']).exists():raise ValueError('r003 output/log already exists; preserve all evidence')
    reuse=read_pinned(REUSE_REF)
    reused_probe=read_pinned(reuse['VAL000_passed_probe_ref'])
    if reused_probe.get('status')!='passed' or reused_probe.get('action')!='probe-memory' or reused_probe.get('parent_id')!='bio_paths_pilot_val_000':raise ValueError('existing actual VAL000 probe required')
    for row in reuse['completed_TRAIN_fit_and_retrieval']:
        receipt=read_pinned(row['receipt'])
        if receipt.get('status')!='passed' or receipt.get('parent_id')!=row['parent'] or receipt.get('action')!=row['stage']:raise ValueError('completed TRAIN evidence changed')
    activation={'root_authorizer':'root_agent','active':True,'activated_utc':utc(),
      'queue_source':ref(Path(__file__)),'runtime':runtime_ref,'native_source':ref(SOURCE),
      'native_config':ref(CONFIG),'prior_plan':PLAN_REF,'budget_receipt':plan['budget_receipt'],
      'per_parent_probe_before_own_fit':True,'completed_TRAIN_and_VAL000_probe_reused':REUSE_REF,
      'remaining_stages_only':True,'serialize_root_budget_mutations_until_worker_first_read':True,
      'query_truth_authorized':False,'scientific_source_or_config_changed':False}
    save(OUT/'queue_activation.json',activation)
    outcomes=[];retrievals=[];children=[]
    legacy=ROOT/'.runtime/gpup/storage_reservations.json'
    supplemental=ROOT/'.runtime/storage/project_budget_v4.json'
    def start(label,parent,stage,gate=None):
        run=label+'_'+('probe_memory' if stage=='probe-memory' else stage)+'_r003'
        a=copy.deepcopy(proposals[parent,stage]);a.update(native.admission_bindings(cfg,parent,stage,run))
        a['logical_job_name']='argformer7_'+run
        if stage!='retrieve':a['runtime']=runtime_ref
        own=budget['admitted_reservations']['native_argformer_tsinfer_fit_v7/'+parent]
        storage={'kind':'native_argformer_tsinfer_fit_v7_root_storage','authorizer':'root_agent','active':True,
          'implementation':a['implementation'],'config':a['config'],'scope_parents':[parent],
          'reservation_key':own['key'],'reserved_bytes':20_000_000_000,'output':own['output'],
          'write_scopes':[own['output'],*own['auxiliary_scopes']],'stage_boundary_cap_not_OS_quota':True,
          'mode':'separately_audited_successor','migration_authority':own['root_admission'],
          'companion':cfg['storage']['companion_contract'],'reservation':own,
          'legacy_ledger_sha256_at_admission':sha(legacy),'supplemental_ledger_sha256_at_admission':sha(supplemental)}
        sp=OUT/f'{parent}_{stage}_{run}_storage_active.json';save(sp,storage);a['storage']=ref(sp)
        if stage=='fit':a['full_batch_memory_receipt']=ref(gate)
        if stage=='retrieve':a['completed_fit_receipt']=ref(gate)
        ap=OUT/f'{parent}_{stage}_{run}_admission_active.json';save(ap,a)
        command_args=['--execute','--parent',parent,'--stage',stage,'--run-id',run,
          '--expected-source-sha256',SOURCE_SHA,'--expected-config-sha256',CONFIG_SHA,
          '--admission',str(ap.relative_to(ROOT)),'--expected-admission-sha256',sha(ap)]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
          PYTHONPATH=str(ROOT/'src'),ARGGNN_LOGICAL_JOB_NAME=a['logical_job_name'])
        if stage=='retrieve':
            env['CUDA_VISIBLE_DEVICES']=''
            command=[str(PY),'-B','-m','arggnn.native_argformer_tsinfer_fit_v7',*command_args]
        else:
            command=[str(PY),'-B',str(ROOT/'scripts/launch_gpu.py'),'--gpu','2','--job',
              a['logical_job_name'],'arggnn.native_argformer_tsinfer_fit_v7',*command_args]
        log=Path(a['launch_log']);log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('xb')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        record={'parent':parent,'stage':stage,'run_id':run,'pid':child.pid,'start_ticks':ticks(child.pid),
          'started_utc':utc(),'root_admission':ref(ap),'argv':command,'GPU':None if stage=='retrieve' else 2}
        save(OUT/f'{parent}_{stage}_{run}_started.json',record);event(event='stage_started',**record)
        item=(child,stream,record,Path(a['output'])/'receipt.json',a['maximum_walltime_seconds']+900)
        children.append(item);return item
    def finish(item):
        child,stream,record,rp,timeout=item
        try:code=child.wait(timeout=timeout);watchdog=False
        except subprocess.TimeoutExpired:
            watchdog=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        stream.close()
        result={**record,'returncode':code,'watchdog':watchdog,'ended_utc':utc()}
        if rp.is_file():
            result['receipt']=ref(rp);result['scientific_status']=json.loads(rp.read_bytes()).get('status')
        save(OUT/f"{record['parent']}_{record['stage']}_{record['run_id']}_terminal.json",result)
        event(event='stage_terminal',**result)
        if item in children:children.remove(item)
        ok=code==0 and result.get('scientific_status')=='passed'
        outcomes.append({'parent':record['parent'],'stage':record['stage'],'passed':ok,'receipt':result.get('receipt')})
        return ok,rp
    def collect_ready():
        for item in list(retrievals):
            if item[0].poll() is not None:finish(item);retrievals.remove(item)
    stopped=False;error=None
    save(OUT/'queue_started.json',{'utc':utc(),'pid':os.getpid(),'start_ticks':ticks(os.getpid()),
      'physical_GPU':2,'root_activation':ref(OUT/'queue_activation.json')})
    try:
        for label,parent in PARENTS:
            collect_ready()
            if parent=='bio_paths_pilot_val_000':
                probe=ROOT/reuse['VAL000_passed_probe_ref']['path']
                event(event='reuse_passed_VAL000_probe',receipt=reuse['VAL000_passed_probe_ref'])
            else:
                ok,probe=finish(start(label,parent,'probe-memory'))
                if not ok:stopped=True;break
            collect_ready()
            ok,fit=finish(start(label,parent,'fit',probe))
            if not ok:stopped=True;break
            retrievals.append(start(label,parent,'retrieve',fit))
        for item in list(retrievals):finish(item);retrievals.remove(item)
    except BaseException as failure:
        error={'type':type(failure).__name__,'message':str(failure),'traceback':traceback.format_exc()}
        for item in list(children):
            if item[0].poll() is None:os.killpg(item[0].pid,signal.SIGTERM)
            try:finish(item)
            except BaseException:pass
        raise
    finally:
        save(OUT/'queue_terminal.json',{'utc':utc(),'outcomes':outcomes,'stopped_on_GPU_stage_failure':stopped,
          'error':error,'paper_reproduction_claimed':False,'scientific_fallback':False})
    if stopped or any(not x['passed'] for x in outcomes):raise SystemExit(1)

if __name__=='__main__':main()
