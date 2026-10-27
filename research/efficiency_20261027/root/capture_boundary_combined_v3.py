"""Capture only the authenticated attempt7 boundary after qualified release.

No queue/controller changes. Actions are separate and fail closed on an existing
intent. ROOT must inspect the finished profile and approve the operational review
before invoking stop. Capture preserves all output through same-filesystem rename.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time
from arggnn.benchmarking.paper_campaign import processes, runtime as r
from research.preprocessing_optimization_20261026.integration import migrate as prior
from research.preprocessing_optimization_20261026.integration import treeloom_resume as previous
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training

WORK=r.ROOT/'research/efficiency_20261027/integration_migration_combined_v1'
TASK=prior.TASK
PID=1166415
TICKS='574375145'
ATTEMPT=7
MODULE='research.efficiency_20261027.integration_candidate.restore_entrypoint_v3'
need=prior.require
save=prior.exclusive


def authenticate_release(release_path):
    from research.efficiency_20261027.integration_candidate.qualification_combined_v5 import validate_release
    release_ref=r.ref(release_path)
    validate_release(release_ref)
    return release_ref


def current(expected_status):
    plan,state,task=prior.current()
    row=state['tasks'][TASK]
    need(row['status']==expected_status and row['pid']==PID
         and row['process_start_ticks']==TICKS and row['attempt']==ATTEMPT,
         'Only the previously inspected attempt7 worker is in scope')
    need(task['call']['module']==MODULE and task['max_rss_bytes']==592*2**30
         and row['physical_gpu']==3, 'Exact current module/resource/GPU scope required')
    for reference in task['source_refs']:
        need(r.ref(reference['path'])==reference,'Running dependency changed')
    return plan,state,task,row


def validate_live(state):
    proc=Path('/proc')/str(PID)
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    need(fields[19]==TICKS and int(fields[1])==state['controller_pid']
         and int(fields[2])==PID and int(fields[3])==PID,
         'Original PID/start/controller/process-group/session required')
    need((proc/'cwd').resolve()==r.ROOT,'Original project working directory required')


def review(path):
    value=json.loads(r.path(path).read_text())
    need(value.get('status')=='pass' and value.get('operational_source_ref')==r.ref(__file__),
         'Independent review must approve this exact operational source')
    return r.ref(path)


def stop(args):
    release_ref=authenticate_release(args.release)
    review_ref=review(args.review)
    plan,state,task,row=current('running')
    validate_live(state)
    from research.efficiency_20261027.root.transition_admission_v2 import assess
    admission=assess(task, args.peer_directory, before_stop=True)
    need(admission['admitted'],admission['reason'])
    need(not (WORK/'stop_intent.json').exists() and not (WORK/'stop_sent.json').exists(),
         'Inspect an existing dispatch instead of retrying a signal')
    fd=processes._pidfd_open(PID)
    try:
        _,fresh,new_task,new_row=current('running')
        validate_live(fresh)
        need(new_task==task and fresh['plan_revision']==state['plan_revision'],
             'Source or queue revision changed during identity check')
        before=save(WORK/'before_stop.json',dict(task=task,row=row,
            plan_ref=state['plan_ref'],plan_revision=state['plan_revision'],
            release_ref=release_ref,review_ref=review_ref,admission=admission,
            peer_directories=args.peer_directory,observed_unix=time.time()))
        save(WORK/'stop_intent.json',dict(pid=PID,process_start_ticks=TICKS,
            signal='SIGTERM',before_stop_ref=before,observed_unix=time.time()))
        processes._pidfd_send(fd,signal.SIGTERM)
    finally:
        os.close(fd)
    save(WORK/'stop_sent.json',dict(pid=PID,process_start_ticks=TICKS,
        signal='SIGTERM',observed_unix=time.time(),controller_signalled=False,
        unrelated_processes_signalled=False))
    print(json.dumps(dict(status='own_worker_signal_sent',pid=PID,
        next_action='Wait for actual controller exit receipt before capture')))


def predecessor_leaf(task):
    from research.efficiency_20261027.integration_candidate.envelope_combined_v2 import predecessor_request
    request=predecessor_request(task)
    arguments=request['arguments']
    evidence=r.read(arguments['resume_evidence_ref'])
    steps=evidence['complete_parent_updates']
    need(type(steps) is int and steps==268 and request['attempt_id']=='explicit512_attempt7',
         'Exact running attempt7 initial boundary and leaf identity required')
    leaf=prior.NATIVE/('exact_preprocessing_from_update%d_%s' % (steps,request['attempt_id']))
    return arguments,leaf


def checkpoint_state(task):
    import torch
    import random
    import numpy as np
    arguments,leaf=predecessor_leaf(task)
    arguments=copy.deepcopy(arguments)
    registry=training.fresh.FreshParentRegistry.from_ref(arguments['parent_registry_ref'])
    model=training.fresh.FreshTreeLoom(registry).build_model()
    checkpoint=previous._ordinary_path(str(leaf/'checkpoints/latest.pt'),regular_file=True)
    need(checkpoint.stat().st_size<=8*2**20,'Bounded current-attempt checkpoint required')
    before=r.ref(checkpoint)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    need(r.ref(checkpoint)==before,'Checkpoint changed during validation')
    steps=training.validate_saved_state(saved,model,registry)
    need(268<steps<600,'Capture actual new progress before original600-update completion')
    validate_latest(task,checkpoint,steps)
    schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    need(saved['schedule_identity']==training.schedule_identity(schedule)
         and saved['last_completed_parent']==schedule[steps-1][1]
         and saved['next_schedule_index']==steps,'Original full schedule and next update required')
    rng=saved['random_state']
    random.Random().setstate(rng['python'])
    state=rng['numpy']
    np.random.RandomState().set_state((state['name'],np.asarray(state['keys'],dtype=np.uint32),
        state['position'],state['has_gauss'],state['cached_gaussian']))
    torch.Generator(device='cpu').set_state(rng['torch_CPU'])
    need(len(rng['torch_CUDA'])==1 and rng['torch_CUDA'][0].dtype==torch.uint8
         and rng['torch_CUDA'][0].ndim==1 and rng['torch_CUDA'][0].numel()>0,
         'One preserved CUDA RNG state required')
    return arguments,registry,model,saved,steps,checkpoint,schedule


def validate_latest(task, checkpoint, steps):
    arguments,expected=predecessor_leaf(task)
    need(checkpoint==expected/'checkpoints/latest.pt',
         'Selected checkpoint must belong to the exact current attempt leaf')
    meta=previous._ordinary_path(str(expected/'latest_checkpoint.json'),regular_file=True)
    before_ref=r.ref(meta)
    latest=json.loads(meta.read_text())
    need(latest.get('complete_parent_updates')==steps
         and latest.get('checkpoint_ref')==r.ref(checkpoint)
         and r.ref(meta)==before_ref,
         'Checkpoint bytes and durable latest metadata must agree before archival')
    return before_ref


def capture(args):
    release_ref=authenticate_release(args.release)
    review(args.review)
    plan,state,task,row=current('failed')
    need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True,
         'Controller must reap the worker and all owned descendants first')
    need(not Path('/proc',str(PID)).exists() and not processes.session_members(PID),
         'Original process and owned session must be absent')
    before=json.loads((WORK/'before_stop.json').read_text())
    need(before['task']==task and before['release_ref']==release_ref
         and before['plan_revision']==state['plan_revision'],
         'Stopped task/release/revision must equal the recorded intent')
    output=r.path(task['output_directory'])
    need(output==prior.NATIVE.parent and output.is_dir() and not output.is_symlink(),
         'Only the original task output may be archived')
    wait_path=previous._ordinary_path(str(output/'actual_wait.json'),regular_file=True)
    wait_ref=r.ref(wait_path)
    controller_wait=json.loads(wait_path.read_text())
    for key in ('pid','process_start_ticks','attempt','status','actual_wait',
                'owned_process_group_absent','actual_returncode'):
        need(controller_wait.get(key)==row.get(key),
             'Controller actual-wait receipt differs from recorded task state: '+key)
    need(r.ref(wait_path)==wait_ref and type(controller_wait.get('actual_returncode')) is int,
         'Stable actual controller exit receipt required')
    permanent_wait=save(WORK/'predecessor_actual_wait.json',dict(controller_wait,
        owned_group_absent=True,controller_actual_wait_ref=wait_ref))
    a,registry,model,saved,steps,checkpoint,schedule=checkpoint_state(task)
    latest_metadata_ref=validate_latest(task,checkpoint,steps)
    captured=WORK/('update%d.pt'%steps)
    with checkpoint.open('rb') as incoming,captured.open('xb') as outgoing:
        shutil.copyfileobj(incoming,outgoing,length=2**20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    checkpoint_ref=r.ref(captured)
    need(checkpoint_ref['sha256']==r.ref(checkpoint)['sha256'],'Checkpoint copy must be byte-identical')
    state_ref=save(WORK/'stopped_state.json',state)
    evidence=dict(schema='treeloom_durable_optimizer_boundary_v2',checkpoint_ref=checkpoint_ref,
        parent_registry_ref=a['parent_registry_ref'],complete_parent_updates=steps,
        durable_complete_parent_optimizer_boundary=True,no_inflight_update_in_serialized_state=True,
        VAL_truth_read=False,TEST_truth_read=False,source_refs=task['source_refs'],
        original_job_ref=r.ref(row['command'][-1]),stopped_snapshot_ref=state_ref,
        observation_barrier_ref=saved['observation_barrier_ref'],schedule_identity=saved['schedule_identity'],
        next_schedule_index=steps,next_parent=schedule[steps][1],next_epoch=schedule[steps][0],
        next_update=steps+1,original_last_completed_parent=saved['last_completed_parent'],
        final_update=600,remaining_complete_parent_updates=600-steps,
        serialized_rng_keys=list(saved['random_state']),original_checkpoint_location_preserved=True,
        inspected_unix=time.time())
    evidence_ref=save(WORK/'resume_evidence.json',evidence)
    arguments=copy.deepcopy(a)
    arguments.update(resume_checkpoint_ref=checkpoint_ref,resume_evidence_ref=evidence_ref)
    training.validate_resume_checkpoint(saved,model,registry,arguments)
    archive=WORK/'prior_attempt7'
    need(not archive.exists(),'Existing archived attempt must never be replaced')
    files=[]
    historical_refs=[]
    for path in sorted(prior.NATIVE.rglob('*')):
        need(not path.is_symlink() and (path.is_file() or path.is_dir()),
             'History must contain only ordinary files/directories')
        if path.is_file():
            reference=r.ref(path)
            relative=path.relative_to(prior.NATIVE)
            historical_refs.append(reference)
            files.append(dict(relative_path=str(relative),size_bytes=path.stat().st_size,
                source_ref=dict(path=str((archive/'native'/relative).relative_to(r.ROOT)),
                    sha256=reference['sha256'])))
    total=sum(item['size_bytes'] for item in files)
    need(0<len(files)<=previous.MAX_HISTORY_FILES and total<=previous.MAX_HISTORY_BYTES,
         'Preserved history must fit the established bounded restore contract')
    need(total+65*2**20<=task['max_output_bytes'],
         'History plus 64MiB future native output and 1MiB outer metadata must fit the unchanged cap')
    need(not (prior.NATIVE/'training_completed.json').exists(),'Completed training cannot migrate')
    history=dict(schema='exact_preprocessing_preserved_history_v1',
        destination=str(prior.NATIVE.relative_to(r.ROOT)),files=files,
        file_count=len(files),total_bytes=total)
    save(WORK/'history_prepared_before_rename.json',history)
    archived_bytes=r.scope_bytes(output)
    need(archived_bytes<=task['max_output_bytes'],'Prior output exceeds its existing cap')
    predecessor_directory=checkpoint.parent.parent
    transition=dict(schema='TreeLoom_existing_native_successor_boundary_v1',
        native_output_directory=str(prior.NATIVE.relative_to(r.ROOT)),
        predecessor_output_directory=str(predecessor_directory.relative_to(r.ROOT)),
        predecessor_job_ref=r.ref(row['command'][-1]),predecessor_pid=PID,
        predecessor_start_ticks=TICKS,predecessor_actual_wait_ref=permanent_wait,
        checkpoint_ref=checkpoint_ref,evidence_ref=evidence_ref,
        preserved_history_refs=historical_refs,warm_observation_cache_transfer=False,
        ROOT_confirmed_no_other_writer=True)
    transition_ref=save(WORK/'transition.json',transition)
    _,fresh,new_task,new_row=current('failed')
    need(new_task==task and fresh['plan_ref']==state['plan_ref']
         and fresh['plan_revision']==state['plan_revision']
         and new_row.get('actual_returncode')==row['actual_returncode']
         and new_row.get('actual_wait') is True
         and new_row.get('owned_process_group_absent') is True,
         'Stopped task/source/exit/revision changed before archive rename')
    need(not Path('/proc',str(PID)).exists() and not processes.session_members(PID),
         'No owned writer may survive at archive rename')
    need(validate_latest(task,checkpoint,steps)==latest_metadata_ref
         and r.ref(checkpoint)['sha256']==checkpoint_ref['sha256'],
         'Latest durable checkpoint changed before archive rename')
    output.rename(archive)
    for item in files:
        need(r.ref(item['source_ref']['path'])==item['source_ref'],'History changed during same-filesystem rename')
    history_ref=save(WORK/'preserved_history.json',history)
    archive_ref=save(WORK/'archive_manifest.json',dict(archived_bytes=archived_bytes,
        archives=[dict(original=task['output_directory'],archived=str(archive.relative_to(r.ROOT)),bytes=archived_bytes)],
        native_history_ref=history_ref,checkpoint_capture_ref=checkpoint_ref))
    save(WORK/'capture_completed.json',dict(status='completed',complete_parent_updates=steps,
        transition_ref=transition_ref,preserved_history_ref=history_ref,
        archive_manifest_ref=archive_ref,release_ref=release_ref,
        prior_output_absent=not output.exists(),queue_submission_performed=False))
    print(json.dumps(dict(status='captured',updates=steps,history_bytes=total,
        archived_bytes=archived_bytes,next_update=steps+1)))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=['inspect','stop','capture'])
    parser.add_argument('--release')
    parser.add_argument('--review')
    parser.add_argument('--peer-directory',action='append',default=[])
    args=parser.parse_args()
    if args.action=='inspect':
        _,state,task,row=current('running')
        validate_live(state)
        print(json.dumps(dict(status='read_only',pid=PID,attempt=ATTEMPT,
            source_count=len(task['source_refs']),project_reserved_bytes=state['project_reserved_bytes'])))
    else:
        need(args.release and args.review,'Qualified release and exact operational review required')
        if args.action=='stop': stop(args)
        else: capture(args)


if __name__=='__main__': main()
