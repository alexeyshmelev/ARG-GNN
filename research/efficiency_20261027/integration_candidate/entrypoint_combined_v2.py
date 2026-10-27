"""Immutable successor entry point for a ROOT-captured durable attempt boundary.

No stop/restart, history restoration, warm-cache claim or live monkeypatch. The
existing native output is retained; only a fresh attempt leaf and its receipt
are written after validation. ROOT owns capture, release, scheduling and launch.
"""
import copy
import json
import os
from pathlib import Path
from arggnn.benchmarking import contracts as c
from research.preprocessing_optimization_20261026.integration import treeloom_resume as previous
from research.efficiency_20261027.execution.deferred_gpu_gate import owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity
from . import training_combined_v2 as training
from .envelope_combined_v1 import validate_arguments
from .qualification_combined_v2 import validate_release

OUTPUT=previous.OUTPUT


def need(ok,message):
    if not ok:
        raise ValueError(message)


def read(reference):
    previous._reference(reference)
    path=previous._ordinary_path(reference['path'],regular_file=True)
    need(path.stat().st_size <= 8*2**20 and c.file_ref(path)==reference,
         'Exact bounded ordinary metadata reference required')
    return json.loads(path.read_text())


def validate_exit(transition):
    wait_path=previous._ordinary_path(transition['predecessor_actual_wait_ref']['path'],regular_file=True)
    need(wait_path.is_relative_to(c.ROOT/'research/efficiency_20261027'),
         'Predecessor actual wait must be an immutable work copy outside mutable lifecycle output')
    stop=read(transition['predecessor_actual_wait_ref'])
    pid=transition['predecessor_pid']
    ticks=transition['predecessor_start_ticks']
    need(type(pid) is int and pid>1 and type(ticks) is str and ticks.isdigit(),
         'Exact ROOT-owned predecessor PID/start identity required')
    need(stop.get('actual_wait') is True and stop.get('owned_group_absent') is True
         and stop.get('pid')==pid and stop.get('process_start_ticks')==ticks
         and type(stop.get('actual_returncode')) is int,
         'ROOT must actually wait for the exact predecessor and retain its return code')
    need(identity(pid)!=ticks and not owned_group_members(pid),
         'Previous worker/group/session must be absent before successor starts')


def validate_transition(reference, arguments, release):
    transition=read(reference)
    need(transition.get('schema')=='TreeLoom_existing_native_successor_boundary_v1'
         and transition.get('native_output_directory')==OUTPUT
         and transition.get('warm_observation_cache_transfer') is False
         and transition.get('ROOT_confirmed_no_other_writer') is True,
         'Explicit immutable existing-native-output transition required')
    validate_exit(transition)
    need(transition.get('checkpoint_ref')==arguments['resume_checkpoint_ref']
         and transition.get('evidence_ref')==arguments['resume_evidence_ref'],
         'Request must use the newly captured durable checkpoint and provenance')
    checkpoint=previous._ordinary_path(arguments['resume_checkpoint_ref']['path'],regular_file=True)
    need(checkpoint.is_relative_to(c.ROOT/'research/efficiency_20261027')
         and checkpoint.stat().st_size<=8*2**20 and c.file_ref(checkpoint)==arguments['resume_checkpoint_ref'],
         'Immutable bounded captured checkpoint outside the mutable latest output required')
    job=read(transition['predecessor_job_ref'])
    need(transition['predecessor_job_ref'] == release['resource_declaration']['predecessor_job_ref'],
         'Transition must bind the declared attempt7 predecessor')
    validate_arguments(job, arguments, release)
    root=previous._ordinary_path(OUTPUT)
    need(root.is_dir() and not (root/'training_completed.json').exists(),
         'Existing incomplete native history namespace required')
    prior=previous._ordinary_path(transition['predecessor_output_directory'])
    need(prior.is_dir() and prior.parent==root, 'Previous attempt history must remain in the native root')
    history=transition['preserved_history_refs']
    need(type(history) is list and 0<len(history)<=4096, 'Bounded preserved history manifest required')
    seen=set()
    total=0
    for reference in history:
        previous._reference(reference)
        path=previous._ordinary_path(reference['path'],regular_file=True)
        need(path.is_relative_to(root) and str(path) not in seen and c.file_ref(path)==reference,
             'Every preserved historical file must remain byte-identical')
        total+=path.stat().st_size
        need(total<=80*2**20,'Bounded historical output manifest required')
        seen.add(str(path))
    required={str(prior/'latest_checkpoint.json'),str(prior/'checkpoints/latest.pt'),str(prior/'progress.jsonl')}
    need(required<=seen,'Durable latest state and complete prior update history must be retained')
    latest=json.loads((prior/'latest_checkpoint.json').read_text())
    evidence=read(arguments['resume_evidence_ref'])
    need(latest['complete_parent_updates']==evidence['complete_parent_updates']
         and latest['checkpoint_ref']['sha256']==arguments['resume_checkpoint_ref']['sha256']
         and c.file_ref(prior/'checkpoints/latest.pt')==latest['checkpoint_ref'],
         'Captured state must equal the latest durable predecessor boundary, never an older checkpoint')
    return transition,evidence


def write(path,value,resource_check):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    resource_check(len(raw))
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return c.file_ref(path)


def run_task(request,*,resource_check):
    need(type(request) is dict and set(request)=={'stage','arguments','preprocessing_release_ref',
        'execution_release_ref','transition_ref','attempt_id'} and request['stage']=='fit',
        'Exact reviewed successor fit request required')
    release=validate_release(request['execution_release_ref'])
    preprocessing=previous.authenticate_release(request['preprocessing_release_ref'])
    need(preprocessing['encoder_workers']==4,'Original qualified four-process preparation required')
    arguments=copy.deepcopy(request['arguments'])
    need(arguments['output_directory']==OUTPUT,'Original benchmark native output contract required')
    need(arguments.get('execution_configuration')==release['configuration'],
         'Request must select exactly the reviewed execution configuration')
    transition,evidence=validate_transition(request['transition_ref'],arguments,release)
    steps=evidence['complete_parent_updates']
    root=previous._ordinary_path(OUTPUT)
    leaf=previous.attempt_directory(root,steps,request['attempt_id'])
    arguments['output_directory']=str(leaf.relative_to(c.ROOT))
    spec=training.build_spec(**arguments)
    # Source/history/exit validation above is read-only and precedes any write.
    receipt=write(root/(leaf.name+'.execution.json'),dict(
        schema='TreeLoom_combined_execution_successor_v1',configuration=release['configuration'],
        transition_ref=request['transition_ref'],execution_release_ref=request['execution_release_ref'],
        preprocessing_release_ref=request['preprocessing_release_ref'],
        coordinated_hook_installation_owner='training_once_before_producers',
        resume_steps=steps,attempt_output_directory=arguments['output_directory'],
        existing_history_preserved=True,warm_observation_cache_transferred=False,
        source_refs=spec['requirements']['source_refs']),resource_check)
    result=training.run_training(spec,resource_check=resource_check)
    need(result['complete_parent_updates']==600
         and result['resumed_from_complete_parent_updates']==steps,'Exact original completion budget required')
    validate_release(request['execution_release_ref'])
    completion={key:value for key,value in result.items() if key!='completion_ref'}
    completion.update(nested_completion_ref=result['completion_ref'],execution_successor_ref=receipt)
    result['completion_ref']=write(root/'training_completed.json',completion,resource_check)
    result['execution_successor_ref']=receipt
    return result
