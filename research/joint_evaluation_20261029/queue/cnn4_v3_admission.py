"""Continue the existing CNN allocation after exact idle-controller handoff."""
import copy
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.cnn4_admission import need, within_shared_allowance


def verify_alias(task):
    wait = r.read(task['alias_wait_ref'])
    need(wait.get('actual_wait') is True and wait.get('actual_returncode') == 0 and
         wait.get('owned_process_group_absent') is True and wait['result_ref'] == task['result_ref'],
         'Completed CNN alias needs exact genuine successful wait')
    need(not processes.session_members(wait['pid']), 'Completed CNN worker session is still live')
    for reference in task['alias_product_refs']:
        need(r.ref(reference['path']) == reference, 'Completed CNN product changed')
    need(r.read(task['result_ref'])['scientific_success'] is True, 'Actual completed CNN scientific output')


def validate_continuation(plan):
    old = r.read(plan['previous_live_plan_ref'])
    review = r.read(plan['ram_review_ref'])
    config = r.read(plan['successor_v2_config_ref'])
    template = r.read(config['successor_template_ref'])
    need(template['required_root_model_chains'] == review['root_gpu_gate_required_model_chains'] ==
         ['american_joint', 'argmix_joint', 'european_four_joint'], 'Three-chain root priority bound required')
    need(plan['foreign_ram_reserve_bytes'] == review['proposed_CNN_foreign_ram_reserve_bytes'] == 1408 * 2**30,
         'Exactly reviewed1408GiB foreign reservation')
    need(plan['records_directory'] == 'research/joint_evaluation_20261029/queue/cnn4_v3_records', 'Fresh CNNv3 records')
    allowed = {'tasks', 'foreign_ram_reserve_bytes', 'records_directory', 'controller_source_refs',
               'previous_live_plan_ref', 'ram_review_ref', 'successor_v2_config_ref', 'handoff_ref', 'handoff_config_ref'}
    need({k:v for k,v in old.items() if k not in allowed} == {k:v for k,v in plan.items() if k not in allowed},
         'Only operational continuation fields may change')
    need([t['id'] for t in plan['tasks']] == [t['id'] for t in old['tasks']], 'Exact15 CNN tasks')
    for task, before in zip(plan['tasks'], old['tasks']):
        if task['kind'] == 'external':
            normalized = copy.deepcopy(task)
            for key in ('result_ref', 'alias_wait_ref', 'alias_product_refs'):
                normalized.pop(key)
            normalized['kind'] = before['kind']
            need(normalized == before, 'Completed alias scientific specification changed')
            verify_alias(task)
        else:
            need(task == before and task['kind'] == 'cpu', 'Pending CNN scientific task changed')
            need(not r.path(task['output_directory']).exists(), 'Unattributed pending CNN output; no replay')
    allocation = r.read(plan['allocation_ref'])
    need(sum(t['max_output_bytes'] for t in plan['tasks']) == allocation['payload_bytes'] == 384*2**20 and
         plan['controller_metadata_allowance_bytes'] == allocation['metadata_bytes'] == 256*2**20 and
         allocation['cnn_reserved_bytes'] == 640*2**20 and allocation['root_confirmed'] is True,
         'Entire existing CNN allocation remains charged including aliases')
    need(within_shared_allowance(plan), 'Combined evaluation use stays within original1GiB')
    need(r.scope_bytes(r.path('research/joint_evaluation_20261029')) + allocation['cnn_reserved_bytes'] <= 2**30,
         'Preserved attempts and full CNN reservation fit original evaluation allowance')
    return allocation


def admit(plan):
    allocation = validate_continuation(plan)
    handoff = r.read(plan['handoff_ref'])
    need(handoff['stopped'] is True and handoff['old_controller_exit_observed'] is True and
         handoff['old_controller_wait_consumed'] is False, 'Observed idle predecessor exit; never fabricate its wait')
    need(handoff['config_ref'] == plan['handoff_config_ref'], 'Exact reviewed handoff')
    previous = r.read(handoff['sealed_original_state_ref'])
    need(previous['plan_ref'] == plan['previous_live_plan_ref'], 'Exact prior CNN plan')
    for task in plan['tasks']:
        row = previous['tasks'][task['id']]
        need((row['status'] == 'succeeded') == (task['kind'] == 'external'), 'All and only succeeded tasks are aliases')
        if task['kind'] == 'external':
            need(row['result_ref'] == task['result_ref'], 'Historical output identity preserved')
        else:
            need(row['status'] == 'pending' and 'pid' not in row, 'Pending tasks were never launched')
    r.save(r.path(plan['records_directory']) / 'admission.json', dict(schema='cnn_four_idle_continuation_admission_v1',
        previous_live_plan_ref=plan['previous_live_plan_ref'], handoff_ref=plan['handoff_ref'],
        cnn_reserved_bytes=allocation['cnn_reserved_bytes'], additional_reservation_bytes=0,
        scientific_plan_unchanged=True, completed_outputs_preserved=True, foreign_ram_gib=1408))
