"""Dependency queue with shared GPU admission and real subprocess completion.

This queue never signals pre-existing jobs. It may stop only its own process
sessions on failure, timeout or explicit cancellation. Historical output remains.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import argmix_claims as claims
from arggnn.benchmarking.paper_campaign import processes
from research.joint_evaluation_20261029.queue import coordination
from arggnn.benchmarking.paper_campaign import repairs
from arggnn.benchmarking.paper_campaign import continuation


def validate(plan):
    if plan['schema'] != 'paper_campaign_v1' or plan['physical_gpus'] != [3, 4, 5]:
        raise ValueError('Explicit current campaign and GPU authority required')
    ids = [t['id'] for t in plan['tasks']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task identifier')
    done = set()
    while len(done) < len(ids):
        available = {t['id'] for t in plan['tasks'] if set(t['dependencies']) <= done}
        if not available - done:
            raise ValueError('Missing or cyclic dependency')
        done |= available
    by_id = {t['id']: t for t in plan['tasks']}
    def references(value):
        if isinstance(value, dict):
            if '$task' in value:
                yield value['$task']
            for item in value.values():
                yield from references(item)
        elif isinstance(value, list):
            for item in value:
                yield from references(item)
    for task in plan['tasks']:
        ancestors = set(task['dependencies'])
        pending = list(ancestors)
        while pending:
            name = pending.pop()
            for dependency in by_id[name]['dependencies']:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    pending.append(dependency)
        if not set(references(task)) <= ancestors:
            raise ValueError('Artifact link lacks a dependency edge: ' + task['id'])
    outputs = []
    for task in plan['tasks']:
        if task['kind'] == 'external':
            if not task.get('result_ref') and not task.get('state_path'):
                raise ValueError('External task needs an actual evidence source')
            continue
        if task['kind'] not in ('cpu', 'gpu'):
            raise ValueError('Unsupported task resource')
        for key in ('max_output_bytes', 'max_rss_bytes', 'max_seconds'):
            if type(task[key]) is not int or task[key] <= 0:
                raise ValueError('Positive finite task resource required: ' + key)
        if not task.get('call') and not task.get('argv'):
            raise ValueError('Task lacks an executable call')
        if task.get('argv') and any(x in task['argv'] for x in ('-c', '-EncodedCommand')):
            raise ValueError('Use a visible script/module rather than inline execution')
        output = r.path(task['output_directory'])
        if not output.is_relative_to(r.ROOT / 'experiments'):
            raise ValueError('Task output must be in project experiments')
        if any(output == prior or output.is_relative_to(prior) or prior.is_relative_to(output) for prior in outputs):
            raise ValueError('Task output scopes overlap')
        outputs.append(output)
        if not r.path(task['python']).is_file():
            raise ValueError('Task interpreter is absent')
        for extra in task.get('additional_mutable_scopes', []):
            if not r.path(extra['path']).is_relative_to(r.ROOT / 'experiments') or extra['max_bytes'] <= 0:
                raise ValueError('Explicit project-contained auxiliary write bound required')
        if task.get('argv') and r.path(task['argv'][0]) != r.path(task['python']):
            raise ValueError('Argv interpreter differs from declared runtime')
    if plan['storage_ceiling_bytes'] > 1_200_000_000_000:
        raise ValueError('Operational storage ceiling exceeded')
    return plan


def resolve(value, state):
    if isinstance(value, dict):
        if '$task' in value:
            row = state['tasks'][value['$task']]
            if row['status'] != 'succeeded':
                raise ValueError('Artifact dependency is unfinished')
            result = row['result_ref']
            if not value.get('raw_result_ref', False):
                saved = r.read(result)
                result = saved.get('value', result)
            if value.get('read'):
                result = r.read(result)
            for key in value.get('path', []):
                result = result[key]
            return result
        return {k: resolve(v, state) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, state) for v in value]
    return value


def gpu_state():
    command = ['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits']
    output = subprocess.check_output(command, text=True, timeout=15)
    result = {}
    for line in output.splitlines():
        index, uuid, free, utilization = [x.strip() for x in line.split(',')]
        index = int(index)
        if index in r.GPU_UUIDS:
            if uuid != r.GPU_UUIDS[index]:
                raise RuntimeError('Physical GPU UUID mapping changed')
            result[index] = dict(uuid=uuid, free_bytes=int(free)*2**20, utilization=int(utilization))
    if set(result) != set(r.GPU_UUIDS):
        raise RuntimeError('Missing authorized GPU')
    return result


def environment(task, gpu):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(r.ROOT / 'src') + ':' + str(r.ROOT), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(task.get('cpu_threads', 4)),
               MKL_NUM_THREADS=str(task.get('cpu_threads', 4)), OPENBLAS_NUM_THREADS=str(task.get('cpu_threads', 4)))
    if gpu is not None:
        env.update(CUDA_VISIBLE_DEVICES=r.GPU_UUIDS[gpu], ARGGNN_PHYSICAL_GPU=str(gpu),
                   ARGGNN_GPU_MEMORY_POLICY='full_device', ARGGNN_GPU_UUID_MAP=json.dumps(r.GPU_UUIDS), WORLD_SIZE='1', LOCAL_RANK='0')
    else:
        env.update(CUDA_VISIBLE_DEVICES='')
        env.pop('ARGGNN_PHYSICAL_GPU', None)
    return env


def external_result(task):
    if task.get('result_ref'):
        r.read(task['result_ref'])
        return task['result_ref']
    source = json.loads(r.path(task['state_path']).read_text())
    row = source['tasks'][task['source_task_id']]
    if row['status'] == 'succeeded':
        r.read(row['result_ref'])
        return row['result_ref']
    if row['status'] in ('failed', 'blocked', 'unknown'):
        raise RuntimeError('External predecessor ' + row['status'])
    return None


def stop_owned(process, expected_start):
    processes.drain(process, terminate=True)
    return process.wait(timeout=5)


def pipeline_floor(plan, state, sizes, active_ids, candidate=None):
    """Keep enough disk for a started native chain to finish and retire scratch.

    The floor is an admission reservation, not a preallocated file or a model
    limit. It prevents other pipelines from consuming a required successor's
    temporary capacity while still permitting work that fits beside it.
    """
    extra = 0
    for pool in plan.get('pipeline_storage_floors', []):
        if all(state['tasks'][name]['status'] == 'succeeded' for name in pool['retirement_ids']):
            continue
        scope = r.path(pool['scope'])
        member_ids = set(pool['task_ids'])
        members = [t for t in plan['tasks'] if t['id'] in member_ids]
        present = sum(sizes[t['id']] for t in members)
        present += sum(max(0, t['max_output_bytes'] - sizes[t['id']])
                       for t in members if t['id'] in active_ids)
        if candidate is not None and candidate['id'] in member_ids:
            present += candidate['max_output_bytes']
        retained = sum(sizes[task['id']] for task_id, parent_scope in pool['retired_parent_scopes'].items()
                       if state['tasks'][task_id]['status'] == 'succeeded'
                       for task in members if r.path(task['output_directory']).is_relative_to(r.path(parent_scope)))
        extra += max(0, retained + pool['next_parent_transient_bytes'] - present)
    return extra


def run(plan_path):
    plan = validate(json.loads(r.path(plan_path).read_text()))
    for source in plan.get('controller_source_refs', []):
        if r.ref(source['path']) != source:
            raise ValueError('Controller source changed: ' + source['path'])
    directory = r.path(plan['records_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'sidecar.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / 'state.json'
    if state_path.exists():
        raise ValueError('Use a reviewed continuation after an existing attempt, never silently replay')
    claims.claim_all(plan)
    seeded = None
    state = dict(schema='paper_campaign_state_v1', plan_ref=r.ref(plan_path), controller_pid=os.getpid(),
                 tasks=seeded if seeded is not None else {t['id']: dict(status='pending') for t in plan['tasks']})
    if seeded is not None:
        state['continuation_ref'] = plan['continuation_ref']
        state['archived_attempt_bytes'] = r.read(plan['continuation_ref'])['archived_attempt_bytes']
    outputs = [r.path(t['output_directory']) for t in plan['tasks'] if t['kind'] != 'external']
    auxiliaries = {}
    for task in plan['tasks']:
        for extra in task.get('additional_mutable_scopes', []):
            scope = r.path(extra['path'])
            if not any(scope == root or scope.is_relative_to(root) for root in outputs):
                auxiliaries[str(scope)] = r.scope_bytes(scope)
    state['auxiliary_initial_bytes'] = auxiliaries
    def used(task):
        logs = directory / 'logs'
        attempts = list(logs.glob(task['id'] + '.attempt*.log'))
        attempts.append(logs / (task['id'] + '.log'))
        return r.scope_bytes(r.path(task['output_directory'])) + sum(r.scope_bytes(p) for p in attempts)
    running = {}
    def cancel(signum, frame):
        raise RuntimeError('Controller received explicit termination signal')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        while True:
            repairs.apply_pending(plan, state, directory)
            for task_id, (process, log, start, start_ticks) in list(running.items()):
                task = next(t for t in plan['tasks'] if t['id'] == task_id)
                row = state['tasks'][task_id]
                code = process.poll()
                if code is None:
                    usage = processes.pss(process.pid)
                    size = used(task)
                    for extra in task.get('additional_mutable_scopes', []):
                        if r.scope_bytes(r.path(extra['path'])) > extra['max_bytes']:
                            size = task['max_output_bytes'] + 1
                    row['observed_pss_bytes'] = usage
                    row['observed_output_bytes'] = size
                    scope_exceeded = any(
                        r.path(task['output_directory']).is_relative_to(r.path(limit['path']))
                        and r.scope_bytes(r.path(limit['path'])) > limit['max_bytes']
                        for limit in plan.get('scope_limits', []))
                    if usage > task['max_rss_bytes'] or size > task['max_output_bytes'] or scope_exceeded:
                        code = stop_owned(process, start_ticks)
                        row['error'] = 'Resource reservation exceeded'
                if code is None and time.time()-start > task['max_seconds']:
                    code = stop_owned(process, start_ticks)
                    row['error'] = 'Wall time limit exceeded'
                if code is None:
                    continue
                drained = processes.drain(process)
                if not drained:
                    row.setdefault('leader_exit_unix', time.time())
                    if time.time() - row['leader_exit_unix'] < 15:
                        continue
                    processes.drain(process, terminate=True)
                    row['error'] = 'Task leader exited with surviving descendants, drained own session'
                log.close()
                output = r.path(task['output_directory'])
                result = output / task.get('completion_file', 'result.json')
                final_bytes = used(task)
                if final_bytes > task['max_output_bytes']:
                    row['error'] = 'Final output exceeds reservation'
                for extra in task.get('additional_mutable_scopes', []):
                    if r.scope_bytes(r.path(extra['path'])) > extra['max_bytes']:
                        row['error'] = 'Final auxiliary scope exceeds reservation'
                success = code == 0 and result.is_file() and 'error' not in row
                row.update(status='succeeded' if success else 'failed', actual_returncode=code,
                           actual_wait=True, owned_process_group_absent=True, finished_unix=time.time(), retained_bytes=used(task))
                if success:
                    row['result_ref'] = r.ref(result)
                else:
                    row.setdefault('error', 'Nonzero exit or absent completion artifact')
                r.save(output / 'actual_wait.json', dict(row))
                del running[task_id]
            for task in plan['tasks']:
                row = state['tasks'][task['id']]
                if row['status'] != 'pending' or task['kind'] != 'external':
                    continue
                try:
                    result = external_result(task)
                    if result:
                        row.update(status='succeeded', result_ref=result, reused=True)
                except Exception as exc:
                    row.update(status='blocked', error=str(exc))
            scope_usage = {limit['path']: r.scope_bytes(r.path(limit['path'])) for limit in plan.get('scope_limits', [])}
            state['scope_usage_bytes'] = scope_usage
            snapshot = gpu_state()
            state['gpu_observation'] = snapshot
            baseline = plan['reserved_existing_project_bytes'] + state.get('archived_attempt_bytes', 0)
            sizes = {t['id']: used(t) for t in plan['tasks'] if t['kind'] != 'external'}
            reserved = sum(sizes.values()) + sum(max(0, r.scope_bytes(Path(scope)) - initial) for scope, initial in auxiliaries.items())
            # Source specs, state and controller stdout have a separately charged allowance.
            reserved += plan.get('controller_metadata_allowance_bytes', 2**30)
            auxiliary_reservations = set()
            for t in plan['tasks']:
                if t['id'] in running:
                    reserved += max(0, t['max_output_bytes'] - sizes[t['id']])
                    for extra in t.get('additional_mutable_scopes', []):
                        scope = str(r.path(extra['path']))
                        if scope not in auxiliary_reservations:
                            reserved += max(0, extra['max_bytes'] - r.scope_bytes(r.path(scope))) if extra.get('growth_allowed', True) else 0
                            auxiliary_reservations.add(scope)
            for task in plan['tasks']:
                row = state['tasks'][task['id']]
                if row['status'] != 'pending' or task['kind'] == 'external':
                    continue
                deps = [state['tasks'][d]['status'] for d in task['dependencies']]
                if any(s in ('failed', 'blocked') for s in deps):
                    row.update(status='blocked', error='Required predecessor failed')
                    continue
                if any(s != 'succeeded' for s in deps):
                    row['waiting_reason'] = 'Required predecessor unfinished'
                    continue
                violated = next((limit for limit in plan.get('scope_limits', [])
                    if r.path(task['output_directory']).is_relative_to(r.path(limit['path']))
                    and scope_usage[limit['path']] > limit['max_bytes']), None)
                if violated:
                    row.update(status='blocked', error='Campaign scope limit exceeded: ' + violated['path'])
                    continue
                new_growth = task['max_output_bytes'] + sum(max(0, extra['max_bytes'] - r.scope_bytes(r.path(extra['path']))) for extra in task.get('additional_mutable_scopes', []) if extra.get('growth_allowed', True))
                if baseline + reserved + new_growth + pipeline_floor(plan, state, sizes, set(running), task) > plan['storage_ceiling_bytes']:
                    row['waiting_reason'] = 'Storage reservation'
                    continue
                active = [t for t in plan['tasks'] if t['id'] in running]
                if len(active) >= plan.get('max_parallel_tasks', 6):
                    row['waiting_reason'] = 'Parallel worker capacity'
                    continue
                free_ram = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
                remaining_growth = plan.get('foreign_ram_reserve_bytes', 0) + sum(max(0, t['max_rss_bytes'] - state['tasks'][t['id']].get('observed_pss_bytes', 0)) for t in active)
                for old in plan.get('existing_workers', []):
                    p = Path('/proc') / str(old['pid']) / 'stat'
                    if p.exists():
                        fields = p.read_text().rsplit(')', 1)[1].split()
                        if fields[19] == old['start_ticks']:
                            remaining_growth += max(0, old['max_rss_bytes'] - processes.pss(old['pid']))
                if free_ram < task['max_rss_bytes'] + remaining_growth + 64*2**30:
                    row['waiting_reason'] = 'Host RAM headroom'
                    continue
                gpu = None
                if task['kind'] == 'gpu':
                    if not coordination.gpu_stage_allowed(plan, task):
                        row['waiting_reason'] = 'Original parent002 large-GPU transition guard'
                        continue
                    candidates = [i for i,g in snapshot.items() if i in plan['allowed_sidecar_gpus'] and g['free_bytes'] >= task['minimum_free_gpu_bytes']
                                  and sum(state['tasks'][t['id']].get('physical_gpu') == i for t in active) < plan.get('max_jobs_per_gpu', 2)]
                    if not candidates:
                        row['waiting_reason'] = 'GPU headroom'
                        continue
                    preferred = task.get('preferred_gpus', [])
                    gpu = min(candidates, key=lambda i: (preferred.index(i) if i in preferred else len(preferred), snapshot[i]['utilization'], -snapshot[i]['free_bytes']))
                elif sum(t['kind'] == 'cpu' for t in active) >= plan.get('max_cpu_tasks', 3):
                    row['waiting_reason'] = 'CPU preparation lanes occupied'
                    continue
                try:
                    concrete = resolve(task, state)
                    for source in concrete.get('source_refs', []):
                        if r.ref(source['path']) != source:
                            raise ValueError('Pinned source changed: ' + source['path'])
                except Exception as exc:
                    row.update(status='blocked', error=str(exc))
                    continue
                output = r.path(task['output_directory'])
                if output.exists() and not claims.permit_claimed_output(plan, task, row):
                    row.update(status='blocked', error='Output already exists, no overwrite')
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                if not task.get('argv') and not output.exists():
                    output.mkdir()
                attempt = row.get('attempt', 1)
                suffix = '' if attempt == 1 else '.attempt' + str(attempt)
                task_file = directory / 'jobs' / (task['id'] + suffix + '.json')
                r.save(task_file, concrete)
                command = concrete.get('argv') or [str(r.path(task['python'])), '-B', '-s', '-m',
                           'arggnn.benchmarking.paper_campaign.runtime', '--task', str(task_file)]
                log_path = directory / 'logs' / (task['id'] + suffix + '.log')
                log_path.parent.mkdir(exist_ok=True)
                log = log_path.open('xb')
                process = subprocess.Popen(command, cwd=r.ROOT, env=environment(task, gpu),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                start_ticks = (Path('/proc') / str(process.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                row.update(status='running', pid=process.pid, process_start_ticks=start_ticks,
                           physical_gpu=gpu, reservation_bytes=task['max_output_bytes'],
                           command=command, log_path=str(log_path.relative_to(r.ROOT)), started_unix=time.time())
                row.pop('waiting_reason', None)
                running[task['id']] = (process, log, time.time(), start_ticks)
                reserved += new_growth
                if gpu is not None:
                    snapshot[gpu]['free_bytes'] -= task['minimum_free_gpu_bytes']
                r.save(state_path, state)
            state['updated_unix'] = time.time()
            state['counts'] = {key: sum(row['status'] == key for row in state['tasks'].values())
                               for key in ('pending', 'running', 'succeeded', 'failed', 'blocked')}
            state['project_reserved_bytes'] = baseline + reserved + pipeline_floor(plan, state, sizes, set(running))
            r.save(state_path, state)
            if not running and all(row['status'] in ('succeeded', 'failed', 'blocked') for row in state['tasks'].values()):
                return state
            time.sleep(10)
    finally:
        for task_id, (process, log, start, ticks) in running.items():
            code = stop_owned(process, ticks)
            log.close()
            state['tasks'][task_id].update(status='failed', actual_returncode=code,
                                          actual_wait=True, owned_process_group_absent=True, finished_unix=time.time(), error='Controller ended')
            task = next(t for t in plan['tasks'] if t['id'] == task_id)
            r.save(r.path(task['output_directory']) / 'actual_wait.json', dict(state['tasks'][task_id]))
        r.save(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        plan = validate(json.loads(r.path(args.plan).read_text()))
        print(json.dumps(dict(valid=True, tasks=len(plan['tasks']))))
    else:
        result = run(args.plan)
        print(json.dumps(result['counts']))
        if result['counts']['failed'] or result['counts']['blocked']:
            sys.exit(1)


if __name__ == '__main__':
    main()
