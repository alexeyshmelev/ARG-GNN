"""Explicit owned-worker migration after release qualification.

Commands are separate: inspect, stop, prepare, submit. No controller state is
edited. Only the identified TreeLoom leader receives a signal through pidfd.
The controller must record actual exit before a repair can be prepared.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import signal
import time
from arggnn.benchmarking.paper_campaign import runtime as r, repairs

ROOT = r.ROOT
WORK = ROOT / 'research/preprocessing_optimization_20261026/integration/migration'
STATE = ROOT / '.runtime/benchmarking/paper_campaign_resume_20261026_v1/state.json'
TASK = 'paper_american_treeloom_fit'
NATIVE = ROOT / 'experiments/results/paper_american_20261024/treeloom_fit/native'
RELEASE = ROOT / 'research/preprocessing_optimization_20261026/integration/release.json'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return r.ref(path)


def current():
    state = json.loads(STATE.read_text())
    plan = r.read(state['plan_ref'])
    receipts = [r.read(ref) for ref in state.get('applied_repairs', {}).values()]
    receipts.sort(key=lambda x:x['prior_plan_revision'])
    require([item['prior_plan_revision'] for item in receipts] == list(range(state.get('plan_revision', 0))),
            'Contiguous authenticated repair history required')
    for receipt in receipts:
        require(receipt['status'] == 'applied' and receipt['prior_plan_sha256'] == repairs._digest(plan),
                'Effective plan repair chain changed')
        request = r.read(receipt['request_ref'])
        by_id = {t['id']: t for t in request['replacement_tasks']}
        plan['tasks'] = [copy.deepcopy(by_id.get(t['id'], t)) for t in plan['tasks']]
        require(receipt['resulting_plan_sha256'] == repairs._digest(plan), 'Effective repaired plan digest changed')
    require(len(receipts) == state.get('plan_revision', 0), 'Complete effective plan reconstruction required')
    task = next(t for t in plan['tasks'] if t['id'] == TASK)
    return plan, state, task


def inspect():
    plan, state, task = current()
    row = state['tasks'][TASK]
    print(json.dumps(dict(counts=state['counts'], task=row,
        effective_call=task['call']['module'], source_count=len(task['source_refs']),
        release_exists=RELEASE.exists()), indent=2))


def stop():
    from .treeloom_resume import authenticate_release
    authenticate_release(r.ref(RELEASE))
    plan, state, task = current()
    row = state['tasks'][TASK]
    require(row['status'] == 'running', 'Only current running own TreeLoom worker can migrate')
    require(task['call']['module'] == 'research.operations_resume_20261026.american.treeloom_resume',
            'Only the explicit current TreeLoom implementation is eligible')
    pid = row['pid']
    proc = Path('/proc') / str(pid)
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    require(stat[19] == str(row['process_start_ticks']), 'PID start identity changed')
    require(int(stat[1]) == state['controller_pid'], 'Current worker must belong to current controller')
    require(proc.joinpath('cwd').resolve() == ROOT, 'Own repository working directory required')
    require(row['physical_gpu'] in (3,4,5), 'Authorized physical GPU required')
    for reference in task['source_refs']:
        require(r.ref(reference['path']) == reference, 'Running source pin changed')
    snapshot = dict(task=task, row=row, plan_ref=state['plan_ref'],
        plan_revision=state.get('plan_revision',0), observed_unix=time.time(),
        release_ref=r.ref(RELEASE), scope=str(NATIVE.parent.relative_to(ROOT)))
    exclusive(WORK/'before_stop.json', snapshot)
    fd = os.pidfd_open(pid)
    try:
        after = proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        require(after[19] == str(row['process_start_ticks']), 'PID changed before pidfd signal')
        current_plan, current_state, current_task = current()
        current_row = current_state['tasks'][TASK]
        require(current_row['status'] == 'running' and current_row['pid'] == pid
            and current_row['process_start_ticks'] == row['process_start_ticks']
            and current_row['attempt'] == row['attempt'] and current_task == task
            and current_state.get('plan_revision', 0) == state.get('plan_revision', 0),
            'Owned task identity or effective plan changed before pidfd signal')
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    exclusive(WORK/'stop_sent.json', dict(pid=pid, signal='SIGTERM',
        process_start_ticks=row['process_start_ticks'], time_unix=time.time(),
        scope=TASK, controller_signalled=False, other_workers_signalled=False))
    print('Only the authenticated TreeLoom leader was signalled. Await actual scheduler exit.')


def checkpoint(task):
    import torch
    import random
    import numpy as np
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training
    from .treeloom_resume import _ordinary_path
    a = copy.deepcopy(task['call']['spec']['arguments'])
    registry = training.fresh.FreshParentRegistry.from_ref(a['parent_registry_ref'])
    rt = training.fresh.FreshTreeLoom(registry)
    model = rt.build_model()
    candidates = []
    for path in NATIVE.glob('*/checkpoints/latest.pt'):
        _ordinary_path(str(path), regular_file=True)
        require(not path.is_symlink() and path.stat().st_size <= 8*2**20, 'Bounded regular checkpoint required')
        before_ref = r.ref(path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        require(r.ref(path) == before_ref, 'Checkpoint changed during validation')
        steps = training.validate_saved_state(saved, model, registry)
        candidates.append((steps, path, saved))
    require(bool(candidates), 'Durable resumed checkpoint absent')
    steps, path, saved = max(candidates, key=lambda x:x[0])
    require(238 < steps < 600, 'Migration must preserve actual new progress before fixed final600')
    schedule = [(e,p) for e in range(1,31) for p in registry.epoch_order(e)]
    require(saved['schedule_identity'] == training.schedule_identity(schedule)
        and saved['last_completed_parent'] == schedule[steps-1][1]
        and saved['next_schedule_index'] == steps, 'Original schedule changed')
    rng = saved['random_state']
    random.Random().setstate(rng['python'])
    n = rng['numpy']
    np.random.RandomState().set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),
        n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.Generator(device='cpu').set_state(rng['torch_CPU'])
    require(len(rng['torch_CUDA']) == 1 and rng['torch_CUDA'][0].dtype == torch.uint8
        and rng['torch_CUDA'][0].ndim == 1 and rng['torch_CUDA'][0].numel() > 0,
        'One saved CUDA RNG state required')
    return a, registry, model, saved, steps, path, schedule


def prepare():
    import shutil
    from .treeloom_resume import authenticate_release, MAX_HISTORY_FILES, MAX_HISTORY_BYTES, _ordinary_path
    from .hooks import source_refs
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1 import training
    release_ref = r.ref(RELEASE)
    authenticate_release(release_ref)
    plan, state, task = current()
    row = state['tasks'][TASK]
    require(row['status'] == 'failed' and row.get('actual_wait') is True
        and row.get('owned_process_group_absent') is True, 'Genuine completed owned exit required')
    before = json.loads((WORK/'before_stop.json').read_text())
    require(row['pid'] == before['row']['pid'] and row['attempt'] == before['row']['attempt'],
        'The stopped attempt must match the scheduler record')
    require(before['task'] == task and before['release_ref'] == release_ref
        and before['row']['process_start_ticks'] == row['process_start_ticks'],
        'Stopped source, release and process identity must remain exact')
    require(not Path('/proc',str(row['pid'])).exists(), 'Old worker still exists')
    output = r.path(task['output_directory'])
    _ordinary_path(task['output_directory'])
    require(output == NATIVE.parent and output.is_dir(), 'Exact old output scope required')
    a, registry, model, saved, steps, old_checkpoint, schedule = checkpoint(task)
    captured = WORK / ('update%d.pt'%steps)
    require(not captured.exists(), 'New immutable captured checkpoint required')
    with old_checkpoint.open('rb') as incoming, captured.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=2**20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    checkpoint_ref = r.ref(captured)
    require(checkpoint_ref['sha256'] == r.ref(old_checkpoint)['sha256'], 'Checkpoint copy changed')
    state_ref = exclusive(WORK/'stopped_state.json', state)
    evidence = dict(schema='treeloom_durable_optimizer_boundary_v2', checkpoint_ref=checkpoint_ref,
        parent_registry_ref=a['parent_registry_ref'], complete_parent_updates=steps,
        durable_complete_parent_optimizer_boundary=True, no_inflight_update_in_serialized_state=True,
        VAL_truth_read=False,TEST_truth_read=False, source_refs=task['source_refs'],
        original_job_ref=r.ref(row['command'][-1]), stopped_snapshot_ref=state_ref,
        observation_barrier_ref=saved['observation_barrier_ref'], schedule_identity=saved['schedule_identity'],
        next_schedule_index=steps, next_parent=schedule[steps][1], next_epoch=schedule[steps][0],
        next_update=steps+1, original_last_completed_parent=saved['last_completed_parent'],
        final_update=600,remaining_complete_parent_updates=600-steps,
        serialized_rng_keys=list(saved['random_state']), original_checkpoint_location_preserved=True,
        inspected_unix=time.time())
    evidence_ref = exclusive(WORK/'resume_evidence.json', evidence)
    a.update(resume_checkpoint_ref=checkpoint_ref,resume_evidence_ref=evidence_ref)
    training.validate_resume_checkpoint(saved,model,registry,a)
    archive = WORK / ('prior_attempt%d'%row['attempt'])
    require(not archive.exists(), 'Immutable archive path must be new')
    files = []
    for path in sorted(NATIVE.rglob('*')):
        require(not path.is_symlink(), 'Native history must contain no symlinks')
        require(path.is_file() or path.is_dir(), 'Native history must contain only ordinary files and directories')
        if path.is_file():
            files.append(dict(relative_path=str(path.relative_to(NATIVE)),
                source_ref=dict(path=str((archive/'native'/path.relative_to(NATIVE)).relative_to(ROOT)),
                    sha256=r.ref(path)['sha256']), size_bytes=path.stat().st_size))
    archive_bytes = r.scope_bytes(output)
    require(archive_bytes <= task['max_output_bytes'], 'Original archived output exceeds reservation')
    history = dict(schema='exact_preprocessing_preserved_history_v1',destination=str(NATIVE.relative_to(ROOT)),files=files,
        file_count=len(files),total_bytes=sum(x['size_bytes'] for x in files))
    require(0 < history['file_count'] <= MAX_HISTORY_FILES and 0 < history['total_bytes'] <= MAX_HISTORY_BYTES,
        'Preserved history must fit the bounded restore contract before archive rename')
    require(not any(item['relative_path'] == 'training_completed.json' for item in files),
        'Completed training must not be migrated')
    exclusive(WORK/'prepared_history_manifest.json', history)
    # Retain all old files atomically on this same NFS filesystem.
    output.rename(archive)
    for item in files:
        require(r.ref(item['source_ref']['path']) == item['source_ref'], 'Archived history changed during rename')
    history_ref = exclusive(WORK/'preserved_history.json',history)
    manifest_ref = exclusive(WORK/'archive_manifest.json',dict(archived_bytes=archive_bytes,
        archives=[dict(original=task['output_directory'],archived=str(archive.relative_to(ROOT)),bytes=archive_bytes)],
        native_history_ref=history_ref, checkpoint_capture_ref=checkpoint_ref))
    replacement = copy.deepcopy(task)
    replacement['call'] = dict(module='research.preprocessing_optimization_20261026.integration.treeloom_resume',
        function='run_task',spec=dict(stage='fit', arguments=a, release_ref=release_ref,
        preserved_history_ref=history_ref,resume_steps=steps,attempt_id='optimization_attempt%d'%(row['attempt']+1)))
    refs = task['source_refs'] + source_refs()
    unique_refs = {}
    for reference in refs:
        require(reference['path'] not in unique_refs or unique_refs[reference['path']] == reference,
                'Conflicting original and candidate source pins')
        unique_refs[reference['path']] = reference
    replacement['source_refs'] = list(unique_refs.values())
    # Historical copies and newly produced output both count against this scope.
    replacement['max_output_bytes'] = task['max_output_bytes'] + history['total_bytes'] + 16*2**20
    reset = []
    closure = {TASK}
    changed = True
    while changed:
        changed = False
        for other in plan['tasks']:
            if other['id'] not in closure and set(other.get('dependencies',[])) & closure:
                closure.add(other['id'])
                changed=True
    for name in sorted(closure-{TASK}):
        other_row = state['tasks'][name]
        if other_row['status'] == 'blocked':
            other = next(t for t in plan['tasks'] if t['id']==name)
            require(not r.path(other['output_directory']).exists(), 'Blocked descendant output requires separate archive')
            reset.append(name)
    request = dict(schema=repairs.SCHEMA,request_id='exact_preprocessing_20261026',
        expected_plan_ref=state['plan_ref'],expected_plan_revision=state.get('plan_revision',0),
        expected_attempts={name:state['tasks'][name].get('attempt',1) for name in [TASK]+reset},
        reset_task_ids=reset,replacement_tasks=[replacement],archived_bytes=archive_bytes,
        archive_manifest_ref=manifest_ref,
        reason='Qualified exact own preprocessing, resume actual durable optimizer boundary. Native competitor implementations, simulations, schedule and metrics unchanged.')
    exclusive(WORK/'repair_request.json',request)
    print(json.dumps(dict(prepared=True,steps=steps,next_parent=schedule[steps][1],
        reset_descendants=len(reset),history_bytes=history['total_bytes'],archive_bytes=archive_bytes)))


def submit():
    plan,state,task=current()
    request=json.loads((WORK/'repair_request.json').read_text())
    directory=r.path(plan['records_directory'])
    body=directory/'repair_requests'/(request['request_id']+'.request.json')
    pending=directory/'repair_requests'/(request['request_id']+'.ref.json')
    require(not body.exists() and not pending.exists(), 'Repair must be submitted exactly once')
    body_ref=exclusive(body,request)
    # Validate through the actual scheduler API before atomically making it visible.
    staged=WORK/(request['request_id']+'.ref.json')
    exclusive(staged,body_ref)
    repairs._prepare(plan,state,directory,staged)
    # Write outside the watched directory, then expose the complete immutable
    # file with one link operation. A controller can never read a partial JSON.
    os.link(staged, pending)
    print('Validated repair submitted to the existing controller. No state rows edited.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','stop','prepare','submit'])
    args=parser.parse_args()
    globals()[args.action]()
