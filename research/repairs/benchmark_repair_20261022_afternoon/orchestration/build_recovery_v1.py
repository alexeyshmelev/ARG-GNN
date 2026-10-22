"""Build the complete operational successor without launches or science changes."""
from copy import deepcopy
import json
from pathlib import Path
from collections import Counter
from research.repairs.benchmark_repair_20261022_afternoon.runtime import readable_workflow_v1 as queue
from research.repairs.benchmark_multigpu_20261022.expansion import controller_overlay_v1 as old_overlay
from research.repairs.benchmark_resume_20261021.native import plan_paths_v1 as paths
from arggnn.benchmarking import workflow_predecessor_v5 as predecessor

c=queue.c
GIB=2**30
HERE=Path(__file__).parent
ACTUAL=HERE/'actual_v1'
OUTPUT='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_afternoon'
PLAN='configs/benchmarks/executable_workflow_20261022_afternoon_v1.json'
RECORDS='.runtime/benchmarking/workflow_20261022_afternoon_v1'


def prepare():
    policy=c.read_json(c.POLICY_PATH)
    overlay_ref=policy['approved_parallel_resource_overlay_ref']
    queue.need(policy.get('parallel_resource_overlay_drain_ref')==overlay_ref,'Old overlay future dispatch must be drained')
    config,base,_=old_overlay.check_config(overlay_ref)
    old_ref=config['base_plan_ref']
    source_state_path=c.contained_path(config['records_directory'])/'state.json'
    state=c.read_json(source_state_path)
    queue.need(state['plan_ref']==old_ref,'Preserve original job plan identity')
    queue.need(not (c.ROOT/PLAN).exists(),'Fresh versioned recovery plan required')
    snapshot_ref=queue.save(ACTUAL/'source_state_snapshot.json',state,exclusive=True)
    plan=deepcopy(base)
    view=old_overlay.plan_view(config,base,c.read_ref(config['state_snapshot_ref']))
    plan['external_storage']=deepcopy(view['external_storage'])
    bindings=deepcopy(view['workflow_external_predecessors'])
    by_id={task['id']:task for task in base['tasks']}
    active_ids={identifier for identifier,entry in state['tasks'].items() if entry['status']=='running'}
    queue.need(len(active_ids)==4,'Reconcile the exact four healthy jobs before integration')
    active_bindings={}
    for identifier in sorted(active_ids):
        task=by_id[identifier]
        entry=state['tasks'][identifier]
        job=c.read_ref(entry['job_ref'])
        queue.need(job['plan_ref']==old_ref and job['task']==task,'Exact immutable healthy job required')
        identity_ref=c.file_ref(c.contained_path(task['output_directory'])/'supervision_identity.json')
        identity=c.read_ref(identity_ref)['child_identity']
        supervisor=queue.shared_gpu.identity(entry['pid'])
        queue.need(supervisor is not None and supervisor['pid']==entry['pid'],'Original supervisor must remain live')
        binding=dict(source_plan_ref=old_ref,source_controller_overlay_ref=overlay_ref,
            source_state_path=str(source_state_path.relative_to(c.ROOT)),source_state_snapshot_ref=snapshot_ref,
            source_task_id=identifier,source_task_fingerprint=c.fingerprint(task),original_job_ref=entry['job_ref'],
            original_supervision_identity_ref=identity_ref,original_supervisor_pid=entry['pid'],
            original_supervisor_start_ticks=supervisor['start_ticks'],owned_process_group_id=identity['pid'],
            owned_group_leader_start_ticks=identity['start_ticks'],owned_group_leader_session=identity['session'],
            max_rss_bytes=task['max_rss_bytes'],max_output_bytes=task['max_output_bytes'],
            resource=task['resource'],output_directory=task['output_directory'])
        predecessor._source(binding)
        queue.need(not any(row['source_task_id']==identifier for row in bindings),'No duplicate predecessor identity')
        bindings.append(binding)
        active_bindings[identifier]=binding
    failed=[b for b in bindings if b['source_task_id']=='AM_treeloom_fit']
    queue.need(len(failed)==1,'One exact American failed predecessor must be retained')
    failed=failed[0]
    _,failure=predecessor._source(failed)
    terminal=predecessor._drained(failed,failure)
    queue.need(failure['status']=='failed' and terminal is not None,'Actual failure and owned group absence required')
    queue.need(failure['actual_controller_wait']==1 and terminal[1]['actual_direct_returncode']==1,'Preserve genuine failure waits')
    retained=failure['retained_bytes']
    queue.need(type(retained) is int and 0<=retained<=failed['max_output_bytes'],'Actual failed scope measurement required')
    bindings.remove(failed)
    failure_receipt=queue.save(ACTUAL/'retained_failed_AM_v3.json',dict(
        original_binding=failed,actual_source_state_ref=c.file_ref(failed['source_state_path']),
        actual_wait_ref=terminal[0],actual_retained_bytes=retained,
        conservative_retention_charge_bytes=failed['max_output_bytes'],files_deleted=False,new_credit_bytes=0),exclusive=True)
    plan['external_storage'].append(dict(cap_bytes=failed['max_output_bytes'],scopes=[failed['output_directory']],
        completion_evidence=[],evidence_ref=failure_receipt,reason='Preserve failed32GiB attempt at full original cap, no new credit'))
    plan['workflow_external_predecessors']=bindings
    new_tasks=[]
    for task in base['tasks']:
        identifier=task['id']
        if identifier in active_bindings:
            task=dict(id=identifier,dependencies=deepcopy(task['dependencies']),resource='external',
                workflow_predecessor=deepcopy(active_bindings[identifier]))
        else:
            task=deepcopy(task)
        new_tasks.append(task)
    plan['tasks']=new_tasks
    plan['records_directory']=RECORDS
    plan['max_parallel_rss_bytes']=1792*GIB
    plan['continuation_approval_key']='approved_recovery_workflow_ref'
    plan['adopted_state_snapshot_ref']=snapshot_ref
    plan['adopted_completed_task_fingerprints']={identifier:c.fingerprint(by_id[identifier])
        for identifier,entry in state['tasks'].items() if entry['status']=='succeeded'}
    patch=c.read_json(HERE.parent/'training/task_resource_patch_v4.json')
    task=next(task for task in plan['tasks'] if task['id']=='AM_treeloom_fit')
    for key in ['factory_module','factory_function','physical_gpu','gpu_uuid','max_cuda_reserved_bytes','minimum_free_gpu_bytes','max_rss_bytes']:
        task[key]=patch[key]
    task['factory_arguments']['max_rss_bytes']=patch['factory_argument_max_rss_bytes']
    task['factory_arguments']['resume_checkpoint_ref']=None
    scope_changes=[]
    for task in plan['tasks']:
        if state['tasks'][task['id']]['status']=='succeeded' or task['resource']=='external':continue
        if task.get('factory_module')==paths.COMPETITORS:continue
        before=task['output_directory']
        after=OUTPUT+'/other/'+task['id']
        queue.need(not c.contained_path(before).exists() or task['id']=='AM_treeloom_fit','Unexpected materialized future task: '+task['id'])
        queue.need(not c.contained_path(after).exists(),'Fresh output already exists')
        task['output_directory']=after
        scope_changes.append(dict(task_id=task['id'],before=before,after=after))
    plan,path_receipt=paths.repair_plan_paths(plan,state,OUTPUT)
    plan['retention_owner_rebinding']=paths.rebind_owner_paths(plan,state)
    for task in plan['tasks']:
        if state['tasks'][task['id']]['status']=='succeeded':
            queue.need(task==by_id[task['id']],'Completed descriptor changed')
    fit=next(task for task in plan['tasks'] if task['id']=='AM_treeloom_fit')
    plan['tasks']=[fit]+[task for task in plan['tasks'] if task['id']!='AM_treeloom_fit']
    plan['task_family_dispatch_order']='American failed-fit retry first to use next full512GiB reservation, then inherited stable order'
    plan['operational_successor_of']=old_ref
    plan['recovery_of_controller_overlay_ref']=overlay_ref
    plan['execution_authority']='Separate ROOT recovery approval, retaining every healthy original release'
    plan['recovery_scientific_settings_changed']=False
    plan['recovery_resource_changes']=dict(American_allocator_GiB=64,American_minimum_free_GiB=72,American_host_RSS_GiB=512,
        max_parallel_RSS_GiB=1792,host_margin_GiB=64,original_full_parent_updates=600,original_epochs=30)
    plan['recovery_paths_evidence_ref']=queue.save(ACTUAL/'path_rebinding.json',dict(generic=scope_changes,native=path_receipt),exclusive=True)
    refs={ref['path']:ref for ref in base['sources']}
    manifest=c.read_json(HERE.parent/'training/source_manifest_v4.json')
    for ref in manifest['source_refs']:refs[ref['path']]=ref
    for source in sorted((HERE.parent/'runtime').glob('*.py')):refs[c.file_ref(source)['path']]=c.file_ref(source)
    for source in [Path(__file__),HERE.parent/'training/source_manifest_v4.json',Path(paths.__file__)]:refs[c.file_ref(source)['path']]=c.file_ref(source)
    plan['sources']=[refs[key] for key in sorted(refs)]
    plan['readable_factory_modules']=sorted(set(plan['readable_factory_modules'])|{patch['factory_module']})
    reference=queue.save(PLAN,plan,exclusive=True)
    loaded=queue.load_plan(reference)
    initial=queue.initial_state(loaded,reference)
    projection=queue.storage_projection(loaded,initial)
    cpu,rss=queue.external_resources(loaded)
    queue.need(projection<980000000000,'Preserved complete queue exceeds storage ceiling')
    queue.need(rss==1536*GIB,'Exact four full healthy reservations required')
    receipt=dict(schema='benchmark_complete_recovery_preparation_v1',plan_ref=reference,source_snapshot_ref=snapshot_ref,
        tasks=len(plan['tasks']),adopted_successes=len(plan['adopted_completed_task_fingerprints']),healthy_external_tasks=sorted(active_ids),
        active_full_RSS_bytes=rss,max_parallel_RSS_bytes=plan['max_parallel_rss_bytes'],storage_projection_bytes=projection,
        statuses=dict(Counter(row['status'] for row in initial['tasks'].values())),source_count=len(refs),
        science_changed=False,launches=0,policy_changes=0,old_sources_changed=False)
    queue.save(ACTUAL/'preparation_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':prepare()
