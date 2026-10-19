"""Build an inert source-pinned successor. ROOT alone activates or launches it."""
import argparse
import copy
import json
from pathlib import Path
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_v4 as lifecycle
from . import readable_workflow_v1 as queue
from .shared_gpu0_v1 import identity

BASE='research/repairs/benchmark_recovery_20261020'
MODULE='research.repairs.benchmark_recovery_20261020.scheduler.'
ADAPTER='research.repairs.benchmark_recovery_20261020.training.queue_adapters.'
OLD='configs/benchmarks/executable_workflow_20261019_v9.json'
STORAGE=BASE+'/storage/compression_plan_v3.json'


def reference(path):return c.file_ref(c.contained_path(path))


def binding_for_running(task,entry,source_ref,snapshot_ref):
    output=c.contained_path(task['output_directory'])
    receipt_ref=reference(output/'supervision_identity.json')
    receipt=c.read_ref(receipt_ref);root=receipt['child_identity']
    supervisor=identity(entry['pid'])
    queue.need(receipt['job_ref']==entry['job_ref'] and root['ppid']==supervisor['pid'],
               'Exact old durable child and supervisor required')
    return dict(source_plan_ref=source_ref,source_state_path=c.read_ref(source_ref)['records_directory']+'/state.json',
        source_state_snapshot_ref=snapshot_ref,source_task_id=task['id'],source_task_fingerprint=c.fingerprint(task),
        original_job_ref=entry['job_ref'],original_supervision_identity_ref=receipt_ref,
        original_supervisor_pid=supervisor['pid'],original_supervisor_start_ticks=supervisor['start_ticks'],
        owned_process_group_id=root['pgid'],owned_group_leader_session=root['session'],
        owned_group_leader_start_ticks=root['start_ticks'],resource=task['resource'],
        max_output_bytes=task['max_output_bytes'],max_rss_bytes=task['max_rss_bytes'],output_directory=task['output_directory'])


def rebind_retention_owners(plan,adopted_success_ids):
    """Literal operational owners follow actual producer scopes after rebasing."""
    by_id={task['id']:task for task in plan['tasks']}
    def scope(identifier):
        task=by_id[identifier]
        if 'output_directory' in task:return task['output_directory']
        for key in ('workflow_predecessor','standalone_predecessor'):
            if key in task:return task[key]['output_directory']
        raise ValueError('No authenticated producer scope for '+identifier)
    changes=[]
    for task in plan['tasks']:
        if task['id'] in adopted_success_ids or task['resource']=='external':continue
        arguments=task.get('factory_arguments',{})
        if task.get('factory_module')=='arggnn.benchmarking.retention':
            owner_ids=[arguments[key]['$task'] for key in ('preparation_ref','embedding_ref','prediction_ref')]
            arguments['owner_scopes']=[scope(identifier) for identifier in owner_ids]
            old_remeasure=task.get('remeasure_succeeded_tasks',[])
            task['remeasure_succeeded_tasks']=[identifier for identifier in old_remeasure if by_id[identifier]['resource']!='external']
            changes.append(dict(task_id=task['id'],owner_scopes=arguments['owner_scopes'],
                external_owners_kept_fully_charged=[identifier for identifier in old_remeasure if by_id[identifier]['resource']=='external']))
        if task.get('factory_module')=='arggnn.benchmarking.european_retention_v9':
            arguments['owner_scope']=scope(arguments['owner_task_id'])
            changes.append(dict(task_id=task['id'],owner_scope=arguments['owner_scope']))
    return changes


def defer_optional_external_archive(plan):
    by_id={task['id']:task for task in plan['tasks']}
    deferred=[]
    for task in plan['tasks']:
        if task.get('factory_module')!='arggnn.benchmarking.european_retention_v9':continue
        owner_id=task['factory_arguments']['owner_task_id']
        if by_id[owner_id]['resource']!='external':continue
        queue.need(owner_id=='AP__ancestralpaths_european_train_001__ap_infer',
                   'Only ROOT-approved optional AP001 archive may be deferred')
        queue.need(not any(task['id'] in other['dependencies'] for other in plan['tasks']),
                   'An optional archive cannot have scientific dependents')
        deferred.append(dict(task=copy.deepcopy(task),
            reason='Optional archive of an external adopted producer deferred; original predecessor retained charge remains in full and no credit is granted',
            ROOT_approved=True,storage_credit_bytes=0))
    ids={item['task']['id'] for item in deferred}
    plan['tasks']=[task for task in plan['tasks'] if task['id'] not in ids]
    plan['deferred_optional_archive_tasks']=deferred


def build(version='v1'):
    source_ref=reference(OLD);old=c.read_ref(source_ref)
    queue.source_check(old['sources'])
    source_state=c.read_json(c.contained_path(old['records_directory'])/'state.json')
    queue.need(source_state['plan_ref']==source_ref,'Exact current v9 source state required')
    directory=c.contained_path(BASE+'/scheduler/readable_'+version)
    directory.mkdir(parents=True,exist_ok=False)
    snapshot=queue.save(directory/'source_v9_state_snapshot.json',source_state,exclusive=True)
    plan=copy.deepcopy(old)
    plan.update(readable_execution=True,records_directory='.runtime/benchmarking/workflow_20261020_readable_'+version,
        operational_successor_of=source_ref,continuation_approval_key='approved_readable_workflow_ref',
        continuation_scope='Static ordinary source, honest adoption, shared physical GPU0 and measured remaining-growth admission',
        adopted_state_snapshot_ref=snapshot,adopted_completed_task_fingerprints={},
        dispatch_hold_task_ids=[],publish_progress=True,readable_factory_routes={},readable_scientific_routes={})
    plan.pop('metadata_factory_smoke',None)
    storage_ref=reference(STORAGE);storage=c.read_ref(storage_ref)
    archive_launch=reference('.runtime/benchmarking/recovery_20261020_storage_launch_v3/launch.json')
    old_archive_state,archive_proof=queue.support.adoption.closed_archive_controller_proof(storage_ref,archive_launch)
    archive_snapshot=queue.save(directory/'historical_archive_v3_state_snapshot.json',old_archive_state,exclusive=True)
    archive_proof['state_snapshot_ref']=archive_snapshot
    plan['historical_archive_controller']=archive_proof
    from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v2
    plan,recovery_evidence=closed_archives_v2.add_recovery_tasks(plan,
        'experiments/results/benchmark_campaign_20261018_workflow/recovery_20261020/storage_v2')
    plan['storage_recovery_build_evidence']=recovery_evidence
    native_ref=reference(BASE+'/scheduler/validation001_request.draft_v3.json');native=c.read_ref(native_ref)
    native_binding=dict(kind='native_contrastive',task_id=native['task']['id'],request_ref=native_ref,
        launch_ref=reference('.runtime/benchmarking/recovery_20261020_native_val001_v3/launch.json'),
        supervisor_identity=dict(pid=3514690,start_ticks=512609190),output_directory=native['task']['output_directory'],
        max_rss_bytes=256*2**30,max_output_bytes=13*2**30,max_cuda_reserved_bytes=40*2**30)
    fit_ref=reference(BASE+'/training/fit_request_v1.json');fit=c.read_ref(fit_ref)
    fit_binding=dict(kind='treeloom_candidate_fit',task_id='AM_treeloom_fit',request_ref=fit_ref,
        supervisor_identity=dict(pid=3635227,start_ticks=512659830),output_directory=fit['output_directory'],
        max_rss_bytes=512*2**30,max_output_bytes=64*2**20,max_cuda_reserved_bytes=16*2**30)
    predecessors={row['source_task_id']:row for row in plan.get('workflow_external_predecessors',[])}
    native_ids={native_binding['task_id']:native_binding,'AM_treeloom_fit':fit_binding}
    for i,task in enumerate(plan['tasks']):
        identifier=task['id']
        if task.get('closed_storage_recovery_target'):continue
        entry=source_state['tasks'][identifier]
        if entry['status']=='succeeded':
            plan['adopted_completed_task_fingerprints'][identifier]=c.fingerprint(task)
            continue
        if identifier in native_ids:
            plan['tasks'][i]=dict(id=identifier,dependencies=task['dependencies'],resource='external',standalone_predecessor=native_ids[identifier])
            continue
        if entry['status']=='running':
            queue.need(task['resource']!='external','Running inherited externals must remain external')
            binding=binding_for_running(task,entry,source_ref,snapshot)
            predecessors[identifier]=binding
            plan['tasks'][i]=dict(id=identifier,dependencies=task['dependencies'],resource='external',workflow_predecessor=binding)
            continue
        if task['resource']=='external':continue
        old_output=c.contained_path(task['output_directory'])
        if old_output.exists():
            wait_path=old_output/'actual_wait.json';wait=c.read_json(wait_path) if wait_path.exists() else {}
            retained=lifecycle.scope_bytes(old_output) if wait.get('actual_direct_wait') is True and wait.get('owned_process_group_absent') is True else task['max_output_bytes']
            plan['external_storage'].append(dict(scopes=[task['output_directory']],cap_bytes=retained,completion_evidence=[],
                retention='Preserved prior attempt; actual retained only after direct wait and owned drain, otherwise full cap'))
        task['output_directory']='experiments/results/benchmark_campaign_20261018_workflow/recovery_20261020/readable_'+version+'/'+identifier
        if task['resource']=='gpu0' or 'max_cuda_reserved_bytes' in task:
            tree='treeloom' in identifier.lower() or 'european_preparation' in task.get('factory_module','')
            task['max_cuda_reserved_bytes']=(16 if tree else 40)*2**30
            task['minimum_free_gpu_bytes']=(26 if tree else 50)*2**30
            if task['resource']=='cpu':task['gpu_after_preparation']=True
    plan['workflow_external_predecessors']=list(predecessors.values())
    plan['retention_owner_rebinding']=rebind_retention_owners(plan,set(plan['adopted_completed_task_fingerprints']))
    defer_optional_external_archive(plan)
    for attempt in storage['closed_native_attempts']:
        scope=attempt['scope']
        if not any(scope in row.get('scopes',[]) for row in plan['external_storage']):
            plan['external_storage'].append(dict(scopes=[scope],cap_bytes=attempt['retained_bytes'],completion_evidence=[],
                actual_wait_ref=attempt['wait_ref'],retention='Exact failed native V2 retained bytes; all original files preserved'))
    # Preflight attempts are closed source evidence outside the old v9 baseline.
    for scope in ('experiments/results/benchmark_recovery_20261020/treeloom_recurrent_preflight_v1',
                  'experiments/results/benchmark_recovery_20261020/treeloom_recurrent_preflight_v2'):
        if c.contained_path(scope).exists():
            plan['external_storage'].append(dict(scopes=[scope],cap_bytes=lifecycle.scope_bytes(c.contained_path(scope)),
                completion_evidence=[],retention='Preserved completed ROOT preflight evidence, separately charged'))
    routes={
        'arggnn.benchmarking.european_workflow:make_call':(MODULE+'queue_european_workflow_v1','make_call'),
        'arggnn.benchmarking.european_argmix_workflow:make_call':(MODULE+'queue_european_argmix_v1','make_call'),
        'arggnn.benchmarking.american_workflow:prediction_factory':(ADAPTER+'american_workflow_v2','prediction_factory'),
        'arggnn.benchmarking.empirical_treeloom:factory':(ADAPTER+'empirical','factory')}
    for function in ('fit_factory','prediction_factory','qualification_factory'):
        routes['arggnn.benchmarking.european_preparation_v9:'+function]=(ADAPTER+'european_stage',function)
    plan['readable_factory_routes']={key:dict(module=value[0],function=value[1]) for key,value in routes.items()}
    modules={value[0] for value in routes.values()}
    modules.add(ADAPTER+'american_prediction')
    modules.update(MODULE+name for name in ('queue_european_workflow_v1','queue_european_argmix_v1'))
    plan['readable_factory_modules']=sorted(modules)
    source_paths={ref['path'] for ref in plan['sources']}
    for folder in ('scheduler','training','training/queue_adapters','runtime','storage','reporting'):
        source_paths.update(str(p) for p in (c.ROOT/BASE/folder).glob('*.py'))
    source_paths.update(ref['path'] for ref in storage['source_refs'])
    source_paths.update(ref['path'] for ref in native['source_refs'])
    source_paths.update(ref['path'] for ref in fit['source_refs'])
    source_paths.add(STORAGE)
    adapter_manifest=reference(BASE+'/training/queue_adapters/source_manifest_v2.json')
    plan['readable_adapter_manifest_ref']=adapter_manifest
    source_paths.update(ref['path'] for ref in c.read_ref(adapter_manifest)['source_refs'])
    source_paths.add(adapter_manifest['path'])
    storage_manifest=reference(BASE+'/storage/recovery_tasks_v3.json')
    plan['readable_storage_manifest_ref']=storage_manifest
    storage_metadata=c.read_ref(storage_manifest)
    plan['required_policy_remove_approved_closed_storage_recovery_refs']=storage_metadata['required_policy_remove_approved_closed_storage_recovery_refs']
    source_paths.update(ref['path'] for ref in storage_metadata['source_refs'])
    source_paths.add(storage_manifest['path'])
    queue.source_check([*c.read_ref(adapter_manifest)['source_refs'],*storage_metadata['source_refs'],*native['source_refs'],*fit['source_refs']])
    source_paths.add('src/arggnn/benchmarking/storage_recovery_20261020_v2.py')
    source_paths.add(plan['closed_storage_recovery']['inventory_ref']['path'])
    plan['sources']=[reference(path) for path in sorted(source_paths)]
    plan_ref=queue.save('configs/benchmarks/executable_workflow_20261020_readable_'+version+'.json',plan,exclusive=True)
    queue.load_plan(plan_ref)
    state=queue.support.initial_state(plan,plan_ref)
    projection=queue.storage_projection(plan,state)
    receipt=dict(schema='static_readable_successor_build_v1',plan_ref=plan_ref,source_state_snapshot_ref=snapshot,
        adopted_successes=len(plan['adopted_completed_task_fingerprints']),tasks=len(plan['tasks']),
        deferred_optional_archive_tasks=[item['task']['id'] for item in plan['deferred_optional_archive_tasks']],
        projected_storage_bytes=projection,source_count=len(plan['sources']),ROOT_activation_required=True,
        execution_started=False,outer_standalone_OS_waits_known=False,
        required_policy_remove_approved_closed_storage_recovery_refs=plan['required_policy_remove_approved_closed_storage_recovery_refs'],
        old_archive_controller_drain_proof=archive_proof,
        remaining_adapter_block_reasons={task['id']:reason for task in plan['tasks']
            if state['tasks'][task['id']]['status']=='pending' and (reason:=queue.support.route_block_reason(plan,task))})
    return queue.save(directory/'build_receipt.json',receipt,exclusive=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='v1')
    args=parser.parse_args();print(json.dumps(build(args.version),sort_keys=True))

if __name__=='__main__':main()
