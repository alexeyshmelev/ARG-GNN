"""Restore exact archived native history only inside the admitted successor.

The controller repair sees an absent outer output. Its unchanged runtime creates
that outer scope and resource metadata, then this wrapper validates the complete
archive and inert training specification before restoring the native history.
The existing entry revalidates every live target before starting training.
"""
import copy
import json
from . import entrypoint_combined_v3 as entry
from .envelope_combined_v2 import validate_arguments, validate_task
from . import training_combined_v3 as training

OUTER_METADATA_ALLOWANCE_BYTES=2**20
REQUEST_KEYS={'stage','arguments','preprocessing_release_ref','execution_release_ref','transition_ref','attempt_id'}


def output_budget(prior,history,spec):
    cap=prior['max_output_bytes']
    new=spec['requirements']['output_scope_cap_bytes']
    entry.need(type(cap) is int and cap>0 and type(new) is int and new==64*2**20,
               'Original finite output envelope and unchanged64MiB training reservation required')
    required=history['total_bytes']+new+OUTER_METADATA_ALLOWANCE_BYTES
    entry.need(required<=cap,'Archived history plus new training and outer metadata exceed original output cap')
    return dict(restored_history_bytes=history['total_bytes'],new_training_reservation_bytes=new,
        outer_metadata_allowance_bytes=OUTER_METADATA_ALLOWANCE_BYTES,required_output_bytes=required,
        admitted_output_cap_bytes=cap,remaining_margin_bytes=cap-required,cap_increased=False)


def preflight(request):
    entry.need(type(request) is dict and set(request)=={'successor','preserved_history_ref'},
               'Exact restore wrapper and immutable successor request required')
    successor=request['successor']
    entry.need(type(successor) is dict and set(successor)==REQUEST_KEYS and successor['stage']=='fit',
               'Exact reviewed successor fit request required')
    release=entry.validate_release(successor['execution_release_ref'])
    preprocessing=entry.previous.authenticate_release(successor['preprocessing_release_ref'])
    entry.need(preprocessing['encoder_workers']==4,'Original qualified four-process preparation required')
    arguments=copy.deepcopy(successor['arguments'])
    entry.need(arguments['output_directory']==entry.OUTPUT
               and arguments.get('execution_configuration')==release['configuration'],
               'Exact original native output and reviewed execution configuration required')
    transition=entry.read(successor['transition_ref'])
    entry.need(transition.get('schema')=='TreeLoom_existing_native_successor_boundary_v1'
        and transition.get('native_output_directory')==entry.OUTPUT
        and transition.get('warm_observation_cache_transfer') is False
        and transition.get('ROOT_confirmed_no_other_writer') is True,
        'Explicit immutable existing-native-output transition required')
    entry.validate_exit(transition)
    entry.need(transition.get('checkpoint_ref')==arguments['resume_checkpoint_ref']
        and transition.get('evidence_ref')==arguments['resume_evidence_ref'],
        'Request must bind the captured durable state and provenance')
    checkpoint=entry.previous._ordinary_path(arguments['resume_checkpoint_ref']['path'],regular_file=True)
    entry.need(checkpoint.is_relative_to(entry.c.ROOT/'research/efficiency_20261027')
        and checkpoint.stat().st_size<=8*2**20
        and entry.c.file_ref(checkpoint)==arguments['resume_checkpoint_ref'],
        'Exact immutable bounded captured checkpoint required')
    prior=entry.read(transition['predecessor_job_ref'])
    entry.need(transition['predecessor_job_ref'] == release['resource_declaration']['predecessor_job_ref'],
        'Transition must bind the declared attempt7 predecessor')
    validate_arguments(prior, arguments, release)
    root,history,files=entry.previous.validate_history(request['preserved_history_ref'])
    entry.need(entry.previous._ordinary_path(prior['output_directory'],allow_missing=True)==root.parent,
               'Original scheduler outer output must own the native history')
    predecessor=entry.previous._ordinary_path(transition['predecessor_output_directory'],allow_missing=True)
    entry.need(predecessor.parent==root,'Previous attempt must be a direct native-history child')
    declared=transition['preserved_history_refs']
    entry.need(type(declared) is list and 0<len(declared)<=4096,'Exact bounded transition history required')
    targets={}
    for reference in declared:
        entry.previous._reference(reference)
        target=entry.previous._ordinary_path(reference['path'],allow_missing=True)
        entry.need(target.is_relative_to(root) and target not in targets,'Unique original native history target required')
        targets[target]=reference
    mapped={target:dict(path=str(target.relative_to(entry.c.ROOT)),sha256=source_ref['sha256'])
            for source,target,size,source_ref in files}
    entry.need(mapped==targets,'Archive restored targets and digests must exactly equal captured history')
    sources={target:source for source,target,size,source_ref in files}
    required={predecessor/'latest_checkpoint.json',predecessor/'checkpoints/latest.pt',predecessor/'progress.jsonl'}
    entry.need(required<=set(sources),'Prior latest metadata, checkpoint and complete update history required')
    latest=json.loads(sources[predecessor/'latest_checkpoint.json'].read_text())
    evidence=entry.read(arguments['resume_evidence_ref'])
    entry.need(latest['complete_parent_updates']==evidence['complete_parent_updates']
        and latest['checkpoint_ref']==mapped[predecessor/'checkpoints/latest.pt']
        and latest['checkpoint_ref']['sha256']==arguments['resume_checkpoint_ref']['sha256'],
        'Archived latest checkpoint must equal captured complete-update boundary, never older state')
    leaf=entry.previous.attempt_directory(root,evidence['complete_parent_updates'],successor['attempt_id'])
    entry.need(not any(target.is_relative_to(leaf) or target==root/(leaf.name+'.execution.json') for target in targets),
               'New attempt and execution receipt must be absent from archive')
    probe=dict(arguments,output_directory=str(leaf.relative_to(entry.c.ROOT)))
    spec=training.build_spec(**probe)
    budget=output_budget(prior,history,spec)
    return dict(spec=spec,budget=budget,prior=prior,history=history,leaf=leaf,root=root,release=release)


def validate_resource_callback(resource_check, prior, release, request):
    # The admitted class can be __main__.ResourceCheck under python -m.
    entry.need(callable(resource_check), 'Admitted worker resource callback required')
    task=getattr(resource_check, 'task', None)
    entry.need(type(task) is dict, 'Resource callback must expose its exact admitted task')
    validate_task(task, prior, release)
    entry.need(task.get('call') == dict(
        module='research.efficiency_20261027.integration_candidate.restore_entrypoint_combined_v3',
        function='run_task', spec=request), 'The actual admitted request must equal this reviewed successor')
    return task


def run_task(request,*,resource_check):
    entry.need(callable(resource_check),'Admitted worker resource callback required')
    checked=preflight(request)
    # The unchanged runtime owns admission. Its starting metadata may exist only
    # in the outer scope; validate_history requires the native root still absent.
    validate_resource_callback(resource_check,checked['prior'],checked['release'],request)
    resource_check(checked['budget']['required_output_bytes'])
    entry.previous.restore_history(request['preserved_history_ref'],resource_check=resource_check)
    return entry.run_task(request['successor'],resource_check=resource_check)
