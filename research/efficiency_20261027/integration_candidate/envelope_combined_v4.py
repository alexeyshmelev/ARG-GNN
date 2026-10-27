"""Exact attempt7 shape,768GiB host cap and76GiB scheduler GPU admission."""
import copy
from arggnn.benchmarking import contracts as c
from .configuration_combined_v4 import configuration, GIB

PREDECESSOR_JOB = '.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt7.json'
ENVELOPE_KEYS = ('max_rss_bytes', 'max_output_bytes', 'max_seconds',
    'minimum_free_gpu_bytes', 'output_directory', 'dependencies', 'preferred_gpus',
    'cpu_threads', 'python', 'id', 'kind')
ARGUMENT_CHANGE_KEYS = {'execution_configuration', 'max_rss_bytes',
    'resume_checkpoint_ref', 'resume_evidence_ref'}
CAP_FIELDS = ('max_rss_bytes', 'call.spec.successor.arguments.max_rss_bytes')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def predecessor_request(prior):
    call = prior.get('call', {})
    spec = call.get('spec', {})
    need(call.get('module') == 'research.efficiency_20261027.integration_candidate.restore_entrypoint_v3'
         and call.get('function') == 'run_task'
         and type(spec) is dict and set(spec) == {'successor', 'preserved_history_ref'},
         'Exact attempt7 nested restore-wrapper request required')
    successor = spec['successor']
    need(type(successor) is dict and set(successor) == {'stage', 'arguments',
         'preprocessing_release_ref', 'execution_release_ref', 'transition_ref', 'attempt_id'}
         and successor['stage'] == 'fit' and type(successor['arguments']) is dict,
         'Exact nested predecessor fit request required')
    return successor


def resource_declaration(prior, predecessor_job_ref):
    old = predecessor_request(prior)['arguments']
    need(prior['max_rss_bytes'] == old['max_rss_bytes'] == 592*GIB
         and prior['cpu_threads'] == 8 and prior['minimum_free_gpu_bytes'] == 56*GIB
         and prior['preferred_gpus'] == [3], 'Authentic prior592GiB CPU8/GPU3/56GiB admission task required')
    successor = {key: copy.deepcopy(prior[key]) for key in ENVELOPE_KEYS}
    successor['max_rss_bytes'] = 768*GIB
    successor['minimum_free_gpu_bytes'] = 76*GIB
    changed_fields = {key: dict(before=592*GIB, after=768*GIB) for key in CAP_FIELDS}
    changed_fields['minimum_free_gpu_bytes'] = dict(before=56*GIB, after=76*GIB)
    return dict(schema='TreeLoom_explicit_successor_resource_declaration_v1',
        predecessor_job_ref=copy.deepcopy(predecessor_job_ref),
        changed_fields=changed_fields,
        successor_scheduler_envelope=successor,
        live_host_and_peer_admission_required_after_predecessor_exit=True,
        predecessor_reservation_not_released_until_actual_exit=True,
        cold_compact_pool=True, default_fallback_allowed=False)


def validate_declaration(value):
    reference = c.file_ref(PREDECESSOR_JOB)
    prior = c.read_json(PREDECESSOR_JOB)
    need(value == resource_declaration(prior, reference),
         'Only declared host592-to-768GiB and scheduler minimum-free-GPU56-to-76GiB changes are allowed')
    return prior


def validate_arguments(prior, arguments, release):
    old = predecessor_request(prior)['arguments']
    selected = configuration(release['configuration'])
    need(release['resource_declaration'] == resource_declaration(
            prior, release['resource_declaration']['predecessor_job_ref']),
         'Exact predecessor and successor resource declarations required')
    need(type(arguments) is dict and set(arguments) == set(old),
         'The original scientific argument shape must be preserved')
    for key in set(old)-ARGUMENT_CHANGE_KEYS:
        need(arguments[key] == old[key], 'Original task input or allowance changed: '+key)
    need(arguments['max_rss_bytes'] == selected['max_rss_bytes'] == 768*GIB
         and arguments['execution_configuration'] == selected,
         'Exact reviewed combined configuration and768GiB cap required')
    for key in ('preparation_headroom_bytes', 'training_headroom_bytes', 'observation_cache_bytes'):
        need(arguments[key] == selected[key], 'Original joint-admission allowance changed: '+key)
    return old


def validate_task(task, prior, release):
    expected = release['resource_declaration']['successor_scheduler_envelope']
    need(expected == resource_declaration(prior,
        release['resource_declaration']['predecessor_job_ref'])['successor_scheduler_envelope'],
        'Bounded declared successor scheduler envelope required')
    for key in ENVELOPE_KEYS:
        need(key in task and task[key] == expected[key],
             'Admitted task differs from the declared successor envelope: '+key)
    return task
