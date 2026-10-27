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
from research.preprocessing_optimization_20261026.integration.hooks import install_tree, verify_bindings
from research.efficiency_20261027.execution.deferred_gpu_gate import owned_group_members
from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity
from . import training_v1 as training

OUTPUT=previous.OUTPUT
from .configuration_v1 import configuration


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


def validate_transition(reference, arguments):
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
    old=job['call']['spec']['arguments']
    for key in ('parent_registry_ref','train_inputs','max_rss_bytes','observation_cache_bytes',
                'preparation_headroom_bytes','training_headroom_bytes','max_truth_uncompressed_bytes'):
        need(arguments[key]==old[key], 'Original task inputs and memory envelope must be preserved: '+key)
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


def validate_release(reference):
    release=read(reference)
    pins=c.file_ref('research/efficiency_20261027/integration_candidate/source_pins_v1.json')
    need(release.get('schema')=='TreeLoom_execution_successor_release_v1'
         and release.get('status')=='independently_approved'
         and configuration(release.get('configuration'))==release.get('configuration')
         and release.get('source_pins_ref')==pins,
         'ROOT-sealed reviewed release of this exact execution configuration required')
    full=read(release['full_parent_qualification_ref'])
    scope=full.get('scope',{})
    need(full.get('status')=='completed' and scope.get('mode')=='full'
         and scope.get('full_parent_claim') is True and scope.get('native_holds')==77472
         and scope.get('supervised_original_focals')==10000 and scope.get('haplotypes')==1600
         and scope.get('heldout_truth_opened') is False,
         'Completed exact full TRAIN000/all10000-target qualification required')
    for reference in full.get('source_refs',[]):
        need(c.file_ref(reference['path'])==reference,'Full qualification dependency changed')
    selected=release['configuration']
    rows=[row for row in full.get('records',[]) if row.get('kind') in ('explicit_vjp','explicit_cpu_vjp')]
    need(rows and all(row.get('CPU_threads')==8 and row.get('holds_per_block',
             full.get('execution_partitions',{}).get('holds_per_block'))==selected['holds_per_block']
             and row.get('comparison',{}).get('passed') is True
             and row.get('comparison',{}).get('argmax_disagreements')==0
             and row.get('comparison',{}).get('categories',{}).get('gradients',{}).get('tensors')==49
             and row.get('comparison',{}).get('categories',{}).get('parameters',{}).get('tensors')==49
             and row.get('comparison',{}).get('categories',{}).get('moments',{}).get('tensors')==147
             for row in rows), 'Exact selected full-parent all49 gradient/Adam comparison must pass')
    review=read(release['independent_integration_review_ref'])
    need(review.get('status') in ('pass','passed') and review.get('candidate_source_pins_ref')==pins,
         'Passing independent integration review must pin this complete candidate source manifest')
    wait=read(release['full_parent_actual_wait_ref'])
    launch=read(release['full_parent_launch_ref'])
    full_path=Path(release['full_parent_qualification_ref']['path'])
    launch_path=Path(release['full_parent_launch_ref']['path'])
    wait_path=Path(release['full_parent_actual_wait_ref']['path'])
    need(full_path.parent.parent==launch_path.parent==wait_path.parent
         and wait.get('pid')==launch.get('pid')
         and wait.get('process_start_ticks')==launch.get('process_start_ticks'),
         'Qualification, launch and actual wait must belong to the same exact run')
    need(wait.get('actual_wait') is True and wait.get('owned_group_absent') is True
         and wait.get('actual_returncode')==0,
         'Full qualification must be actually reaped with no owned processes remaining')
    training.requirements(release['configuration'])
    return release


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
    transition,evidence=validate_transition(request['transition_ref'],arguments)
    steps=evidence['complete_parent_updates']
    root=previous._ordinary_path(OUTPUT)
    leaf=previous.attempt_directory(root,steps,request['attempt_id'])
    arguments['output_directory']=str(leaf.relative_to(c.ROOT))
    spec=training.build_spec(**arguments)
    # Source/history/exit validation above is read-only and precedes any write.
    bindings=install_tree(preparation_workers=4,max_rss_bytes=arguments['max_rss_bytes'])
    receipt=write(root/(leaf.name+'.execution.json'),dict(
        schema='TreeLoom_explicit_execution_successor_v1',configuration=release['configuration'],
        transition_ref=request['transition_ref'],execution_release_ref=request['execution_release_ref'],
        preprocessing_release_ref=request['preprocessing_release_ref'],bindings=bindings,
        resume_steps=steps,attempt_output_directory=arguments['output_directory'],
        existing_history_preserved=True,warm_observation_cache_transferred=False,
        source_refs=spec['requirements']['source_refs']),resource_check)
    result=training.run_training(spec,resource_check=resource_check)
    need(result['complete_parent_updates']==600
         and result['resumed_from_complete_parent_updates']==steps,'Exact original completion budget required')
    verify_bindings(bindings)
    validate_release(request['execution_release_ref'])
    completion={key:value for key,value in result.items() if key!='completion_ref'}
    completion.update(nested_completion_ref=result['completion_ref'],execution_successor_ref=receipt)
    result['completion_ref']=write(root/'training_completed.json',completion,resource_check)
    result['execution_successor_ref']=receipt
    return result
