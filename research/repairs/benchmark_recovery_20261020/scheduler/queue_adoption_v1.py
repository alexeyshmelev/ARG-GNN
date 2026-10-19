"""Authenticate standalone outcomes and mirror the sole archive controller."""
import copy
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_v4 as lifecycle
from .shared_gpu0_v1 import identity


def need(ok,message):
    if not ok:raise ValueError(message)


def generation_absent(expected):
    try:current=identity(expected['pid'])
    except (FileNotFoundError,ProcessLookupError):return True
    return any(current.get(key)!=value for key,value in expected.items())


def scientific_root(binding):
    output=c.contained_path(binding['output_directory'])
    if binding.get('kind')=='treeloom_candidate_fit':
        receipt=c.read_json(output/'child_identity.json')
        need(receipt['supervisor_identity']==binding['supervisor_identity'],'Candidate supervisor receipt changed')
        expected=receipt['worker_identity']
        current=lifecycle.identity(expected['pid'])
        need(all(current.get(key)==value for key,value in expected.items())
             and current['pid']==current['pgid']==current['session'],'Exact candidate scientific session required')
        return current
    receipt=c.read_json(output/'supervision_identity.json')
    return receipt['child_identity']


def drained(binding):
    output=c.contained_path(binding['output_directory'])
    if not generation_absent(binding['supervisor_identity']):return False
    if binding.get('kind')=='treeloom_candidate_fit':
        path=output/'supervisor_result.json'
        if not path.is_file():return False
        wait=c.read_json(path)
        need(wait['request_ref']==binding['request_ref'] and wait['supervisor_identity']==binding['supervisor_identity'],
             'Candidate terminal request/supervisor changed')
        return wait.get('lifecycle')=='direct_owned_Popen_wait' and type(wait.get('actual_exit_code')) is int and wait.get('owned_child_absent') is True
    path=output/'actual_wait.json'
    if not path.is_file():return False
    wait=c.read_json(path)
    return wait.get('actual_direct_wait') is True and type(wait.get('actual_direct_returncode')) is int and wait.get('owned_process_group_absent') is True


def completion_evidence(binding):
    return dict(kind='scientific_direct_wait_and_exact_supervisor_generation_absence',
        supervisor_identity=binding['supervisor_identity'],actual_outer_OS_wait_known=False,
        actual_outer_returncode=None,scientific_direct_returncode=0)


def candidate_result(binding):
    if not drained(binding):return None
    output=c.contained_path(binding['output_directory'])
    request=c.read_ref(binding['request_ref'])
    launch=c.read_json(output/'launch.json')
    need(launch['request_ref']==binding['request_ref'] and launch['supervisor_identity']==binding['supervisor_identity'],
         'Exact candidate launch required')
    wait_ref=c.file_ref(output/'supervisor_result.json');wait=c.read_ref(wait_ref)
    if wait['actual_exit_code']!=0 or wait.get('success') is not True or wait.get('resource_violation') is not None:
        return dict(status='failed',error='Standalone TreeLoom candidate failed',external_actual_wait_ref=wait_ref)
    result_ref=c.file_ref(output/'training_completed.json');result=c.read_ref(result_ref)
    job=c.read_ref(request['job_ref']);outer=job['call']['spec']
    need(outer.get('schema')=='treeloom_full20_compact_RAM_fit_v4'
         and isinstance(outer.get('training_spec'),dict),'Exact original compact-RAM fit wrapper required')
    spec=outer['training_spec']
    need(result.get('schema')=='treeloom_fresh_full_training_completion_v1'
         and result.get('request_ref')==binding['request_ref'] and result.get('observation_barrier_ref')==request['observation_barrier_ref']
         and result.get('parent_registry_ref')==spec['parent_registry_ref']
         and result.get('complete_parent_updates')==600 and result.get('completed_epochs')==30
         and result.get('final_saved_optimizer_parameter_states')==49 and result.get('final_saved_optimizer_steps')==[600]
         and result.get('checkpoint_selection')=='fixed_final_epoch30' and result.get('selected_epoch')==30
         and result.get('VAL_truth_read') is False and result.get('TEST_truth_read') is False,
         'Complete unchanged candidate budget, final checkpoint and truth barriers required')
    checkpoint=result['checkpoint_ref'];c.contained_path(checkpoint['path'])
    return dict(status='succeeded',value=result_ref,external_actual_wait_ref=wait_ref,
        external_source_result_ref=result_ref,external_original_job_ref=request['job_ref'],
        controller_completion_evidence=completion_evidence(binding))


def archive_state(plan):
    owner=plan.get('standalone_archive_owner')
    if owner is None:return None
    config=c.read_ref(owner['config_ref'])
    launch=c.read_ref(owner['launch_ref'])
    need(launch['config_ref']==owner['config_ref'] and launch['supervisor_identity']==owner['supervisor_identity'],
         'Exact standalone archive launch and generation required')
    need(c.contained_path(owner['state_path'])==c.contained_path(config['records_directory'])/'state.json',
         'Read only the named standalone archive state')
    state=c.read_json(owner['state_path'])
    need(state['plan_ref']==owner['config_ref'],'Standalone archive source state changed')
    expected={task['id']:task for task in config['recovery_plan']['tasks'] if task.get('closed_storage_recovery_target')}
    actual={task['id']:task for task in plan['tasks'] if task.get('closed_storage_recovery_target')}
    need(expected==actual,'Exact standalone archive task descriptors required')
    return state


def refresh_archives(plan,state):
    source=archive_state(plan)
    if source is None:return
    owner=plan['standalone_archive_owner']
    for task in plan['tasks']:
        if not task.get('closed_storage_recovery_target'):continue
        entry=copy.deepcopy(source['tasks'][task['id']])
        entry['standalone_archive_dispatch_owner']=owner['supervisor_identity']
        state['tasks'][task['id']]=entry


def archive_dispatch_reason(plan,task):
    if task.get('closed_storage_recovery_target') and plan.get('standalone_archive_owner'):
        return 'Standalone archive controller owns this exact batch; no duplicate dispatch'
    return None


def closed_archive_controller_proof(config_ref,launch_ref):
    """Require a real terminal for every launched child before queue handoff."""
    config=c.read_ref(config_ref);launch=c.read_ref(launch_ref)
    need(launch['config_ref']==config_ref,'Exact old archive launch required')
    need(generation_absent(launch['supervisor_identity']),'Old archive controller generation is still present')
    state_path=c.contained_path(config['records_directory'])/'state.json'
    state=c.read_json(state_path)
    need(state['plan_ref']==config_ref,'Old archive source state changed')
    launched=[]
    for task in config['recovery_plan']['tasks']:
        entry=state['tasks'][task['id']]
        need(entry['status'] not in ('running','unknown'),'Old archive child lacks a genuine settled terminal')
        if not entry.get('job_ref'):
            need(entry['status']=='pending' and not entry.get('reservation_bytes')
                 and not c.contained_path(task['output_directory']).exists(),
                 'Only never-launched archive tasks may remain pending')
            continue
        wait_ref=c.file_ref(c.contained_path(task['output_directory'])/'actual_wait.json')
        wait=c.read_ref(wait_ref)
        need(entry['status'] in ('succeeded','failed') and type(entry.get('actual_controller_wait')) is int
             and wait.get('job_ref')==entry['job_ref'] and wait.get('actual_direct_wait') is True
             and wait.get('actual_direct_returncode')==entry['actual_controller_wait']
             and wait.get('owned_process_group_absent') is True and entry.get('owned_process_group_absent') is True,
             'Every launched old archive must preserve actual wait and owned drain')
        launched.append(dict(task_id=task['id'],status=entry['status'],job_ref=entry['job_ref'],actual_wait_ref=wait_ref,
            actual_direct_returncode=wait['actual_direct_returncode'],owned_process_group_absent=True))
    return state,dict(config_ref=config_ref,launch_ref=launch_ref,supervisor_identity=launch['supervisor_identity'],
        exact_supervisor_generation_absent=True,all_launched_children_genuinely_drained=True,
        launched=launched,outer_OS_wait_known=False,old_dispatch_authority_not_transferred=True)
