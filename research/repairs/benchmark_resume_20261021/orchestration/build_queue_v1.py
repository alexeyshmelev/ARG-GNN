"""Prepare the GPU5 successor using actual results and unchanged scientific tasks.

This builder does not authorize or launch jobs. Completed producer descriptors,
results and old attempts remain immutable. ROOT releases the resulting plan.
"""
import copy
import json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking.workflow_v4 import scope_bytes
from research.repairs.benchmark_resume_20261021.gpu import readable_workflow_v1 as queue
from research.repairs.benchmark_resume_20261021.native.plan_paths_v1 import repair_plan_paths, rebind_owner_paths

BASE='research/repairs/benchmark_resume_20261021'
OLD='configs/benchmarks/executable_workflow_20261020_readable_v3.json'
OUTPUT='experiments/results/benchmark_campaign_20261018_workflow/release_20261021_gpu5'
PLAN='configs/benchmarks/executable_workflow_20261021_gpu5_v1.json'
GPU='research.repairs.benchmark_resume_20261021.gpu.'
TRAINING='research.repairs.benchmark_resume_20261021.training.fit_v1'
RECONCILE={'AP__ancestralpaths_european_train_002__ap_infer','AP__ancestralpaths_european_test_000__ap_infer'}


def reference(path):
    return c.file_ref(c.contained_path(path))


def mapped_module(module):
    old='research.repairs.benchmark_recovery_20261020.'
    if module.startswith(old+'training.queue_adapters.'):
        return GPU+'queue_adapters.'+module[len(old+'training.queue_adapters.'):]
    return module


def build():
    old_ref=reference(OLD)
    old=c.read_ref(old_ref)
    queue.source_check(old['sources'])
    original_state=c.read_json(c.contained_path(old['records_directory'])/'state.json')
    queue.need(original_state['plan_ref']==old_ref,'Exact old queue state required')
    queue.need(not Path('/proc/4171358').exists(),'Old GPU0 controller must be absent')
    state=copy.deepcopy(original_state)
    directory=c.contained_path(BASE+'/orchestration/build_v1')
    directory.mkdir(exist_ok=False)
    raw_ref=queue.save(directory/'original_state.json',original_state,exclusive=True)
    reconciled=[]
    by_id={task['id']:task for task in old['tasks']}
    for identifier in sorted(RECONCILE):
        task=by_id[identifier]
        entry=state['tasks'][identifier]
        out=c.contained_path(task['output_directory'])
        wait_ref=reference(out/'actual_wait.json')
        wait=c.read_ref(wait_ref)
        result_ref=reference(out/'result.json')
        result=c.read_ref(result_ref)
        queue.need(entry.get('actual_controller_wait')==0
            and entry.get('owned_process_group_absent') is True
            and wait.get('actual_direct_wait') is True
            and wait.get('actual_direct_returncode')==0
            and wait.get('owned_process_group_absent') is True
            and wait['job_ref']==entry['job_ref']==result['job_ref'],
            'Genuine direct and controller waits plus result must establish completion')
        c.read_ref(result['value'])
        entry.update(status='succeeded',result_ref=result_ref,retained_bytes=scope_bytes(out),
            resumed_reconciliation=dict(original_state_ref=raw_ref,actual_wait_ref=wait_ref,
                reason='Successful scientific work was reaped after the controller pause and GPU0 authority revocation'))
        entry.pop('error',None)
        reconciled.append(identifier)
    snapshot=queue.save(directory/'adoption_state.json',state,exclusive=True)
    success={identifier for identifier,entry in state['tasks'].items() if entry['status']=='succeeded'}
    queue.need(len(success)==183,'Expected 181 recorded successes plus two genuine completed AP jobs')
    plan,path_evidence=repair_plan_paths(old,state,OUTPUT)
    plan.update(physical_gpu=5,physical_gpu_uuid='GPU-3db85505-d061-e45d-d154-e7e8c1cad497',
        records_directory='.runtime/benchmarking/workflow_20261021_gpu5_v1',
        operational_successor_of=old_ref,adopted_state_snapshot_ref=snapshot,
        adopted_completed_task_fingerprints={},dispatch_hold_task_ids=[],publish_progress=True,
        continuation_scope='User-authorized physical GPU5, preserved completed science, repaired operational RAM and output paths',
        execution_authority='Explicit user instruction to continue on physical GPU5 only')
    fit_job=c.read_ref(dict(path='.runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json',
        sha256='187e3177a9ccf587e1038ad4f477d6beaccb5128333977083bf06bd5084fe4a3'))
    fit=copy.deepcopy(fit_job['task'])
    fit.update(factory_module=TRAINING,factory_function='fit_factory',max_rss_bytes=512*2**30,
        max_cuda_reserved_bytes=16*2**30,minimum_free_gpu_bytes=26*2**30,max_output_bytes=80*2**20)
    for at,task in enumerate(plan['tasks']):
        identifier=task['id']
        if identifier in success:
            queue.need(task==by_id[identifier],'Historical completed descriptor changed')
            plan['adopted_completed_task_fingerprints'][identifier]=c.fingerprint(task)
            continue
        if identifier=='AM_treeloom_fit':
            binding=task['standalone_predecessor']
            wait=c.read_json(c.contained_path(binding['output_directory'])/'supervisor_result.json')
            queue.need(wait.get('owned_process_group_absent') is True or wait.get('owned_child_absent') is True,
                'Failed candidate must have actual owned cleanup before a fresh attempt')
            plan['external_storage'].append(dict(scopes=[binding['output_directory']],cap_bytes=binding['max_output_bytes'],
                completion_evidence=[],retention='Full retained charge for failed original fit, no completion or credit'))
            task=copy.deepcopy(fit)
            plan['tasks'][at]=task
        queue.need(task['resource']!='external','Unexpected unresolved external producer')
        entry=state['tasks'][identifier]
        queue.need(not entry.get('job_ref') and not entry.get('reservation_bytes'),
            'Unreconciled previously launched pending task: '+identifier)
        if task.get('factory_module')!='arggnn.benchmarking.american_competitors':
            task['output_directory']=OUTPUT+'/other/'+identifier
        queue.need(not c.contained_path(task['output_directory']).exists(),'Fresh output scope required')
    plan['retention_owner_rebinding']=rebind_owner_paths(plan,state)
    plan['readable_factory_routes']={key:dict(module=mapped_module(value['module']),function=value['function'])
        for key,value in plan['readable_factory_routes'].items()}
    plan['readable_scientific_routes']={key:dict(module=mapped_module(value['module']),function=value['function'])
        for key,value in plan.get('readable_scientific_routes',{}).items()}
    plan['readable_scientific_routes']['arggnn.benchmarking.american_competitors:run_argformer']=dict(
        module=GPU+'native_scientific_gpu5_v1',function='run_argformer')
    c4_module='research.repairs.benchmark_resume_20261021.native.c4.queue_european_argmix_v1'
    plan['readable_scientific_routes']['research.repairs.benchmark_recovery_20261020.scheduler.queue_european_argmix_v1:execute_stage']=dict(
        module=c4_module,function='execute_stage')
    modules={mapped_module(module) for module in plan['readable_factory_modules']}
    modules.add(c4_module)
    modules.add(TRAINING)
    modules.add(GPU+'native_scientific_gpu5_v1')
    plan['readable_factory_modules']=sorted(modules)
    sources={ref['path'] for ref in old['sources']}
    for folder in ('gpu','training','native','orchestration'):
        sources.update(str(path.relative_to(c.ROOT)) for path in (c.ROOT/BASE/folder).rglob('*.py'))
    plan['sources']=[reference(path) for path in sorted(sources)]
    plan_ref=queue.save(PLAN,plan,exclusive=True)
    queue.load_plan(plan_ref)
    initial=queue.support.initial_state(plan,plan_ref)
    projection=queue.storage_projection(plan,initial)
    block_reasons={task['id']:reason for task in plan['tasks']
        if initial['tasks'][task['id']]['status']=='pending' and (reason:=queue.support.route_block_reason(plan,task))}
    queue.need(not block_reasons,'Operational route gaps remain: '+str(block_reasons))
    receipt=dict(plan_ref=plan_ref,original_state_ref=raw_ref,adopted_state_snapshot_ref=snapshot,
        adopted_successes=len(success),completed_AP_reconciliations=reconciled,
        pending_tasks=sum(entry['status']=='pending' for entry in initial['tasks'].values()),
        physical_gpu=5,gpu_uuid=plan['physical_gpu_uuid'],old_controller_absent=True,
        projected_storage_bytes=projection,storage_ceiling_bytes=plan['storage_ceiling_bytes'],
        sources=len(plan['sources']),scope_repair=path_evidence,
        historical_failures_preserved=True,scientific_settings_unchanged=True,execution_started=False)
    receipt_ref=queue.save(directory/'build_receipt.json',receipt,exclusive=True)
    return dict(plan_ref=plan_ref,build_receipt_ref=receipt_ref,adopted_successes=len(success),
        pending_tasks=receipt['pending_tasks'],projected_storage_bytes=projection)


if __name__=='__main__':
    print(json.dumps(build(),sort_keys=True),flush=True)
