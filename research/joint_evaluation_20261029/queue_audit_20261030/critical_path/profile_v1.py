"""Targeted read-only Relate profile; write only a new immutable audit receipt."""
from pathlib import Path
import datetime
import json
import os
import time
import psutil
from arggnn.benchmarking.paper_campaign import runtime as r

FOLDER = Path(__file__).parent
BASE = Path('research/joint_evaluation_20261029/queue/successor_records_v2')
FAMILY = Path('experiments/paper_argmix_1611_rebuild_20261024_v1')
SUFFIXES = {'generated', 'relate_inputs', 'native_relate', 'inferred', 'support',
            'features', 'treeloom_observation', 'release_new_scratch'}


def main():
    target = FOLDER / 'profile_v1.json'
    if target.exists():
        raise ValueError('Preserve the existing profile')
    plan = json.loads((BASE / 'plan.json').read_text())
    state = json.loads((BASE / 'state.json').read_text())
    state_path = FOLDER / 'main_state_sample_v2.json'
    if state_path.exists():
        raise ValueError('Preserve the existing source snapshot')
    r.save(state_path, state)
    task = next(t for t in plan['tasks'] if state['tasks'][t['id']]['status'] == 'running')
    row = state['tasks'][task['id']]
    parent = task['id'].removeprefix('paper_argmix_').rsplit('.', 1)[0]
    leader = psutil.Process(row['pid'])
    first = {p.pid: sum(p.cpu_times()[:2]) for p in leader.children(recursive=True)}
    start = time.monotonic()
    time.sleep(10)
    workers = []
    for process in leader.children(recursive=True):
        cpu = sum(process.cpu_times()[:2])
        workers.append(dict(pid=process.pid, state=process.status(), rss_bytes=process.memory_info().rss,
            cpu_percent=(cpu-first[process.pid])/(time.monotonic()-start)*100 if process.pid in first else None,
            command=process.cmdline()))
    waits = [json.loads(line) for line in (FAMILY/parent/'native_run/actual_native_waits.jsonl').read_text().splitlines()]
    sections = len(list((FAMILY/parent/'relate_work/inferred/chunk_0/paint').glob('*bin')))
    registry = json.loads(Path('research/joint_evaluation_20261029/queue/native_argmix_retry_active_owner_v1.json').read_text())
    native_plan = r.read(registry['active_plan_ref'])
    native_state = json.loads(Path(registry['state_path']).read_text())

    def floor(name):
        commitments = {}
        for stage in plan['tasks']:
            if not stage['id'].startswith('paper_argmix_'+name+'.') or stage['id'].rsplit('.', 1)[1] not in SUFFIXES:
                continue
            path = r.path(stage['output_directory'])
            actual = r.scope_bytes(path)
            desired = actual if state['tasks'][stage['id']]['status'] == 'succeeded' else max(actual, stage['max_output_bytes'])
            commitments[str(path)] = max(commitments.get(str(path), 0), desired)
            for auxiliary in stage.get('additional_mutable_scopes', []):
                path = r.path(auxiliary['path'])
                commitments[str(path)] = max(commitments.get(str(path), 0), auxiliary['max_bytes'], r.scope_bytes(path))
        return dict(parent=name, current_bytes=r.scope_bytes(FAMILY/name),
                    completion_floor_bytes=sum(commitments.values()), commitments=commitments)

    current = floor(parent)
    future = floor('published_recon_iid_002')
    scope = r.scope_bytes(FAMILY)
    native_extra = sum(max(0, t['max_output_bytes']-r.scope_bytes(r.path(t['output_directory'])))
        for t in native_plan['tasks'] if t['kind'] != 'external' and native_state['tasks'][t['id']]['status'] in ('pending', 'running'))
    projected = scope-current['current_bytes']-future['current_bytes']+current['completion_floor_bytes']+future['completion_floor_bytes']+native_extra
    parents = ['published_recon_iid_'+str(i).zfill(3) for i in range(3)] + ['published_recon_misspec_'+str(i).zfill(3) for i in range(5)]
    changes = []
    for index in range(2, len(parents)):
        name = 'paper_argmix_'+parents[index]+'.generated'
        old = next(t for t in plan['tasks'] if t['id'] == name)
        changes.append(dict(task_id=name, before=old['dependencies'], after=['paper_argmix_'+parents[index-2]+'.release_new_scratch']))
    history = []
    for name in ['published_recon_train_003', 'published_recon_train_004', 'published_recon_val_000', 'published_recon_iid_000']:
        historical = state['tasks']['paper_argmix_'+name+'.native_relate']
        terminal = r.read(historical['result_ref'])
        history.append(dict(parent=name, result_ref=historical['result_ref'], native_seconds=terminal['seconds'], successful=terminal['successful']))
    report = dict(schema='relate_critical_path_profile_v1', observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        profiler_ref=r.ref(__file__), source_plan_ref=r.ref(BASE/'plan.json'), source_state_snapshot_ref=r.ref(state_path),
        active_task_id=task['id'], active_pid=row['pid'], active_start_ticks=row['process_start_ticks'],
        stage='BuildTopology', completed_sections=sum('BuildTopology' in w['stage'] for w in waits), total_sections=sections,
        all_observed_native_waits_successful=all(w['returncode'] == 0 and w['actual_wait_consumed'] for w in waits), workers=workers,
        observed_pss_bytes=row['observed_pss_bytes'], memory_cap_bytes=task['max_rss_bytes'], historical_parent_durations=history,
        host=dict(physical_cores=psutil.cpu_count(logical=False), logical_cores=psutil.cpu_count(), mem_available_bytes=psutil.virtual_memory().available, load_average=os.getloadavg()),
        capacity=dict(scope_actual_bytes=scope, scope_cap_bytes=137438953472, current_parent=current, next_parent=future,
            native_future_reserve_bytes=native_extra, two_parent_completion_commitment_bytes=projected,
            margin_bytes=137438953472-projected, optional_guard_bytes=2**30,
            global_commitment_bytes=state['project_reserved_bytes'], global_cap_bytes=plan['storage_ceiling_bytes']),
        proposed_dependency_changes=changes, max_unretired_parents=2, native_workers_per_parent=8,
        recommendation='Use a rolling two-parent window only with full conversion and retirement completion floors checked before every admission. Keep unchanged seeds, algorithms, native argv, outputs, two task lanes and all resource caps.',
        handoff_requirements=['Same paper_campaign_global.lock lifetime lock.',
            'Freeze and recheck only a naturally childless controller; no scientific worker signals.',
            'Reconcile stale running rows only from matching genuine actual_wait receipts and verified absent sessions; otherwise resume old owner.',
            'Native retry verification and direct external state readers pin the old main identity; authenticate a replacement main-owner registry and update these consumers at their idle boundary.',
            'Do not prewrite outputs still owned by the active main scheduler.'],
        limitation='Two cold full parent caps plus retained baseline exceed the family cap. Current sealed generated output makes the rolling pair feasible. Recompute at activation and every admission; do not treat this profile as a launch authorization.',
        real_worker_launches=0, real_signals_sent=0)
    r.save(target, report)
    print(json.dumps(dict(profile_ref=r.ref(target), completed_sections=report['completed_sections'], total_sections=sections,
                         worker_cpu_percent=[w['cpu_percent'] for w in workers], projected_bytes=projected,
                         margin_bytes=137438953472-projected), sort_keys=True))


if __name__ == '__main__':
    main()
