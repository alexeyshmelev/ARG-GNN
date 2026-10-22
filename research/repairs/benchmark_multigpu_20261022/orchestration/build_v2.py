"""Build a parallel successor, preserving live GPU5 science and completed outputs."""
import copy
import fcntl
import importlib
import inspect
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from research.repairs.benchmark_multigpu_20261022.runtime import readable_workflow_v2 as q
from research.repairs.benchmark_resume_20261021.native.plan_paths_v1 import repair_plan_paths, rebind_owner_paths

c=q.c
BASE='research/repairs/benchmark_multigpu_20261022'
OLD='configs/benchmarks/executable_workflow_20261021_gpu5_v1.json'
PLAN='configs/benchmarks/executable_workflow_20261022_multigpu_v2.json'
OUT='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_multigpu'
SCI='research.repairs.benchmark_multigpu_20261022.scientific.'
UUID={2:'GPU-976a5cd3-df12-2535-fa51-e3ea063688ba',3:'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',5:'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}
ACTIVE='AM_argformer_american_validation_002_argformer_mlm'


def ref(path):
    return c.file_ref(c.contained_path(path))


def remap(module):
    old='research.repairs.benchmark_resume_20261021.gpu.queue_adapters.'
    return SCI+'queue_adapters.'+module[len(old):] if module.startswith(old) else module


def main():
    directory=c.contained_path(BASE+'/orchestration/build_record_v2')
    directory.mkdir(exist_ok=False)
    old_ref=ref(OLD)
    old=c.read_ref(old_ref)
    q.source_check(old['sources'])
    lock=c.ROOT/'.runtime/locks/benchmark_recovery_capacity_20261020.lock'
    with lock.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        policy=c.read_json(c.POLICY_PATH)
        q.need(policy['approved_readable_workflow_ref']==old_ref and policy['physical_gpus']==[5], 'Exact live GPU5 authority required')
        before=q.save(directory/'policy_before.json',policy,exclusive=True)
        policy.update(additional_physical_gpus=[2,3],user_authorized_physical_gpus=[2,3,5],
            readable_workflow_drain_ref=old_ref,
            readable_workflow_drain_reason='Preserve active GPU5 MLM, transfer only future dispatch to the parallel successor',
            additional_gpu_authorized_at_utc=datetime.now(timezone.utc).isoformat())
        q.save(c.POLICY_PATH,policy)
        state=c.read_json(c.ROOT/old['records_directory']/'state.json')
        q.need([key for key,e in state['tasks'].items() if e['status']=='running']==[ACTIVE], 'Only the known healthy MLM may remain active')
        snapshot=q.save(directory/'old_state_snapshot.json',state,exclusive=True)
    by_id={t['id']:t for t in old['tasks']}
    active=by_id[ACTIVE]
    entry=state['tasks'][ACTIVE]
    identity_ref=ref(c.contained_path(active['output_directory'])/'supervision_identity.json')
    identity=c.read_ref(identity_ref)['child_identity']
    supervisor=q.shared_gpu.identity(entry['pid']) if hasattr(q,'shared_gpu') else q.support.gpu.identity(entry['pid'])
    binding=dict(source_plan_ref=old_ref,source_state_path=old['records_directory']+'/state.json',
        source_state_snapshot_ref=snapshot,source_task_id=ACTIVE,source_task_fingerprint=c.fingerprint(active),
        original_job_ref=entry['job_ref'],original_supervision_identity_ref=identity_ref,
        original_supervisor_pid=supervisor['pid'],original_supervisor_start_ticks=supervisor['start_ticks'],
        owned_process_group_id=identity['pgid'],owned_group_leader_start_ticks=identity['start_ticks'],
        owned_group_leader_session=identity['session'],resource=active['resource'],
        max_rss_bytes=active['max_rss_bytes'],max_output_bytes=active['max_output_bytes'],output_directory=active['output_directory'])
    seed=copy.deepcopy(old)
    seed['tasks']=[dict(id=ACTIVE,resource='external',dependencies=active['dependencies'],workflow_predecessor=binding) if t['id']==ACTIVE else t for t in seed['tasks']]
    seed['workflow_external_predecessors'].append(binding)
    plan,path_evidence=repair_plan_paths(seed,state,OUT)
    success={key for key,e in state['tasks'].items() if e['status']=='succeeded'}
    plan.update(physical_gpu=5,physical_gpus=[2,3,5],records_directory='.runtime/benchmarking/workflow_20261022_multigpu_v2',
        operational_successor_of=old_ref,adopted_state_snapshot_ref=snapshot,
        adopted_completed_task_fingerprints={},dispatch_hold_task_ids=[],publish_progress=True,
        execution_authority='User authorized GPUs2and3 in addition to existingGPU5',
        continuation_scope='Parallel operational repair with unchanged science and healthy native MLM retained')
    scientific=c.read_ref(ref(BASE+'/scientific/routes_v1.json'))
    qualifications=c.read_ref(ref(BASE+'/scientific/qualification_plan_fragment_v1.json'))
    native=c.read_ref(ref(BASE+'/native/integration_v1.json'))
    replacements={r['task_id']:r['replacement_task'] for r in qualifications['fit_rebindings']}
    for index,task in enumerate(plan['tasks']):
        key=task['id']
        if key in success:
            q.need(task==by_id[key], 'Completed descriptor changed')
            plan['adopted_completed_task_fingerprints'][key]=c.fingerprint(task)
            continue
        if key==ACTIVE:continue
        old_entry=state['tasks'][key]
        if old_entry.get('job_ref'):
            q.need(old_entry['status']=='failed' and old_entry.get('owned_process_group_absent') is True, 'Old failed worker must be genuinely drained')
            old_task=by_id[key]
            charged=q.scope_bytes(c.contained_path(old_task['output_directory']))
            q.need(charged==old_entry['retained_bytes'], 'Actual failed scope changed after controller measurement')
            plan['external_storage'].append(dict(scopes=[old_task['output_directory']],cap_bytes=charged,completion_evidence=[],
                failed_state_snapshot_ref=snapshot,original_task_id=key,original_max_output_bytes=old_task['max_output_bytes'],
                actual_wait_ref=ref(c.contained_path(old_task['output_directory'])/'actual_wait.json'),
                retention='Preserve exact original measured failed scope, no new deletion or credit'))
        if key in replacements:
            task=copy.deepcopy(replacements[key]);plan['tasks'][index]=task
        if key=='AM_treeloom_fit':
            task.update(factory_module=SCI+'fit_v3',max_cuda_reserved_bytes=32*2**30,minimum_free_gpu_bytes=40*2**30)
        if task.get('factory_module')!='arggnn.benchmarking.american_competitors':
            task['output_directory']=OUT+'/other/'+key
        q.need(not c.contained_path(task['output_directory']).exists(),'Fresh successor output required')
    plan['tasks'].extend(copy.deepcopy(qualifications['qualification_tasks']))
    augmented_state=copy.deepcopy(state)
    for task in qualifications['qualification_tasks']:augmented_state['tasks'][task['id']]=dict(status='pending')
    plan['retention_owner_rebinding']=rebind_owner_paths(plan,augmented_state)
    qualifier_by_prefix={t['id'].split('__')[0]:t['id'] for t in qualifications['qualification_tasks']}
    lane_previous={3:None,5:'AM_argformer_american_validation_002_retire_native_scratch'}
    parents={};lane_changes=[]
    prepares=[t for t in plan['tasks'] if t['id'] not in success and t.get('factory_module')=='arggnn.benchmarking.american_competitors' and t.get('factory_arguments',{}).get('stage')=='prepare']
    prepares.sort(key=lambda t:(0 if '_validation_' in t['id'] else 1,int(re.search(r'_(\d{3})_argformer_prepare$',t['id']).group(1))))
    for number,task in enumerate(prepares):
        gpu=3 if number%2==0 else 5
        parent=re.search(r'argformer_american_(?:validation|test)_\d{3}',task['id']).group()
        parents[parent]=gpu
        old_deps=task['dependencies'][:]
        task['dependencies']=[key for key in old_deps if not key.endswith('_retire_native_scratch')]
        if lane_previous[gpu]:task['dependencies'].append(lane_previous[gpu])
        lane_previous[gpu]='AM_'+parent+'_retire_native_scratch'
        lane_changes.append(dict(task_id=task['id'],physical_gpu=gpu,before=old_deps,after=task['dependencies']))
    parents['argformer_american_validation_002']=5
    plan['readable_factory_routes']={key:dict(module=remap(value['module']),function=value['function']) for key,value in plan['readable_factory_routes'].items()}
    plan['readable_factory_routes'].update(scientific['readable_factory_routes'])
    plan['readable_scientific_routes']={key:dict(module=remap(value['module']),function=value['function']) for key,value in plan['readable_scientific_routes'].items()}
    for route in native['scientific_routes']:
        plan['readable_scientific_routes'][route['original_module']+':'+route['original_function']]=dict(module=route['module'],function=route['function'])
    modules=set(plan['readable_factory_modules'])|set(scientific['readable_factory_modules'])|set(native['callable_module_allowlist'])
    for task in plan['tasks']:
        key=task['id']
        if key in success or task['resource']=='external':continue
        task['factory_module']=remap(task['factory_module'])
        prefix=key.split('__')[0]
        if prefix in qualifier_by_prefix and 'qualification_ref' in task.get('factory_arguments',{}):
            qual=qualifier_by_prefix[prefix]
            task['factory_arguments']['qualification_ref']={'$task':qual}
            if qual not in task['dependencies']:task['dependencies'].append(qual)
        match=re.search(r'argformer_american_(?:validation|test)_\d{3}',key)
        gpu=parents.get(match.group(),3) if match else (3 if key.startswith('AP__') else 5 if 'c4_treeloom' in key else 2)
        if key=='AM_treeloom_fit':gpu=2
        task.update(physical_gpu=gpu,gpu_uuid=UUID[gpu])
        route=plan['readable_factory_routes'].get(task['factory_module']+':'+task['factory_function'],dict(module=task['factory_module'],function=task['factory_function']))
        function=getattr(importlib.import_module(route['module']),route['function'])
        parameters=inspect.signature(function).parameters
        if route['module'].startswith(SCI) and route['function'] in ('fit_factory','prediction_factory'):
            task['factory_arguments'].update(physical_gpu=gpu,gpu_uuid=UUID[gpu])
        for name,value in [('physical_gpu',gpu),('gpu_uuid',UUID[gpu])]:
            if name in parameters:task['factory_arguments'][name]=value
        if task.get('gpu_after_preparation') or key=='AM_treeloom_fit':
            task.update(max_cuda_reserved_bytes=32*2**30,minimum_free_gpu_bytes=40*2**30)
        modules.add(route['module'])
    plan['readable_factory_modules']=sorted(modules)
    sources={r['path'] for r in old['sources']}
    sources.add('src/arggnn/official_argmix_biological.py')
    sources.update(('src/arggnn/expanded_native_argmix_v1.py','src/arggnn/full_panel_native_baseline_contracts_v1.py'))
    from research.repairs.benchmark_multigpu_20261022.native import queue_european_argmix_v1 as native_queue
    sources.update(native_queue.SOURCE_PATHS)
    sources.update(str(path.relative_to(c.ROOT)) for path in (native_queue.bindings.native.SOURCE/'src').rglob('*.py'))
    for folder in [BASE,'research/repairs/benchmark_followup_20261022']:
        sources.update(str(path.relative_to(c.ROOT)) for path in (c.ROOT/folder).rglob('*.py'))
    plan['sources']=[ref(path) for path in sorted(sources)]
    plan_ref=q.save(PLAN,plan,exclusive=True)
    q.load_plan(plan_ref)
    initial=q.support.initial_state(plan,plan_ref)
    projection=q.storage_projection(plan,initial)
    q.need(projection<980000000000,'Storage ceiling required')
    route_blocks={task['id']:q.support.route_block_reason(plan,task) for task in plan['tasks'] if initial['tasks'][task['id']]['status']=='pending' and task['resource']!='external'}
    q.need(not any(route_blocks.values()),'Unresolved operational routes')
    receipt=dict(plan_ref=plan_ref,old_state_snapshot_ref=snapshot,adopted_successes=len(success),
        pending=sum(e['status']=='pending' for e in initial['tasks'].values()),total_stages=len(plan['tasks']),
        preserved_live_MLM=binding,parallel_retirement_lanes=lane_changes,projected_storage_bytes=projection,
        path_repair=path_evidence,executed=False,old_policy_before_drain_ref=before)
    receipt_ref=q.save(directory/'build_receipt.json',receipt,exclusive=True)
    print(json.dumps(dict(plan_ref=plan_ref,build_receipt_ref=receipt_ref,adopted_successes=len(success),projected_storage_bytes=projection)),flush=True)


if __name__=='__main__':main()
