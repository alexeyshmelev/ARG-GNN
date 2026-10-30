"""Dependency queue with shared GPU admission and real subprocess completion.

This queue never signals pre-existing jobs. It may stop only its own process
sessions on failure, timeout or explicit cancellation. Historical output remains.
"""
from __future__ import annotations
import argparse
import fcntl
import functools
import json
import os
import re
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.resources import probe_guard_v1 as probe
from research.joint_evaluation_20261029.queue import storage_credit_v1 as storage_credit
from research.joint_evaluation_20261029.queue import successor_v2_gpu as default_gpu_ownership
from arggnn.benchmarking.paper_campaign import processes
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
            if not task.get('result_ref') and not task.get('state_path') and not task.get('future_wait_path'):
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


def stop_owned(process, expected_start):
    if not processes.drain(process, terminate=True):
        raise probe.ProbeUnavailable('Owned process session has not drained')
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


def native_checkpoint_alias_bytes(path, observed):
    """Accept only author checkpoint leaf aliases; never traverse a link."""
    relative = path.relative_to(r.ROOT).as_posix()
    allowed = (r'experiments/results/paper_american_20261024/argformer_american_'
               r'((test_00[0-9]|validation_00[0-4])/(mlm|contrastive)/native/checkpoints/'
               r'|validation_001/mlm/native/resume_20261026_batch119500/checkpoints/)'
               r'latest-rank0\.pt')
    if re.fullmatch(allowed, relative) is None:
        raise ValueError('Unregistered mutable-scope symbolic link')
    target = os.readlink(path)
    if re.fullmatch(r'ep[0-9]+-ba[0-9]+-rank0\.pt', target) is None:
        raise ValueError('Native checkpoint alias target may not escape its directory')
    try:
        target_stat = (path.parent / target).lstat()
    except FileNotFoundError:
        return observed.st_size
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError('Native checkpoint alias target must be an ordinary file')
    # Retain the older conservative apparent-size charge for live aliases, plus
    # link metadata. A retired dangling alias contributes metadata only.
    return observed.st_size + target_stat.st_size


def strict_scope_bytes(directory):
    """Count declared scopes without suppressing directory traversal failures."""
    directory = Path(directory)
    try:
        root_stat = directory.lstat()
    except FileNotFoundError:
        return 0
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError('Mutable scope cannot be a symbolic link')
    if not stat.S_ISDIR(root_stat.st_mode):
        return root_stat.st_size
    def fail(error):
        raise error
    total = 0
    for parent, directories, files in os.walk(directory, followlinks=False, onerror=fail):
        for name in directories + files:
            path = Path(parent) / name
            try:
                value = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(value.st_mode):
                total += native_checkpoint_alias_bytes(path, value)
                continue
            if not stat.S_ISDIR(value.st_mode):
                total += value.st_size
    return total


def filesystem_observation(function):
    """Keep unavailable I/O distinct from a proven invalid contract or limit."""
    @functools.wraps(function)
    def observed(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except OSError as error:
            raise probe.ProbeUnavailable(type(error).__name__ + ': ' + str(error)[:300]) from error
    return observed


def observe_io(monitor, name, function, *args, **kwargs):
    return monitor.observe(name, filesystem_observation(function), *args, **kwargs)


TASK_LOCAL_PROBES = ('task_admission:', 'source_read:', 'prepare:', 'lease:')


def global_admission_blocked(monitor):
    return any(not name.startswith(TASK_LOCAL_PROBES) for name in monitor.errors)


def publish_state(monitor, state_path, state):
    # A failed publication never invents a receipt or cancels another worker.
    state['monitor_health'] = monitor.health()
    state['monitor_health']['new_admissions_allowed'] = not global_admission_blocked(monitor)
    state['monitor_health']['task_local_admission_failures'] = {key: value for key, value in monitor.errors.items() if key.startswith(TASK_LOCAL_PROBES)}
    health = observe_io(monitor, 'publish_health', r.save, monitor.path, state['monitor_health'])
    saved = observe_io(monitor, 'publish_state', r.save, state_path, state)
    return health is not None and saved is not None


def resource_observation(plan, task, used):
    size = used(task)
    auxiliary_exceeded = any(r.scope_bytes(r.path(extra['path'])) > extra['max_bytes']
                             for extra in task.get('additional_mutable_scopes', []))
    scope_exceeded = any(r.path(task['output_directory']).is_relative_to(r.path(limit['path']))
                         and r.scope_bytes(r.path(limit['path'])) > limit['max_bytes']
                         for limit in plan.get('scope_limits', []))
    return size, auxiliary_exceeded, scope_exceeded


def exit_observation(task, used):
    size = used(task)
    auxiliary_exceeded = any(r.scope_bytes(r.path(extra['path'])) > extra['max_bytes']
                             for extra in task.get('additional_mutable_scopes', []))
    result = r.path(task['output_directory']) / task.get('completion_file', 'result.json')
    return size, auxiliary_exceeded, r.ref(result) if result.is_file() else None


def publish_wait(path, payload):
    # Replacement may have succeeded before a trailing read failed. A retry must
    # preserve exactly the same immutable payload, including its finish time.
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError('Existing actual wait differs; never overwrite historical evidence')
        return r.ref(path)
    return r.save(path, payload)


def close_log(log):
    log.close()
    return True


def process_start_ticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


def run(plan_path, *, support_module, owner_lock_path, gpu_module=None, allow_existing_state=False):
    support = support_module
    gpu_ownership = gpu_module or default_gpu_ownership
    external_result = support.external_result
    plan = validate(json.loads(r.path(plan_path).read_text()))
    for source in plan.get('controller_source_refs', []):
        if r.ref(source['path']) != source:
            raise ValueError('Controller source changed: ' + source['path'])
    directory = r.path(plan['records_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    lock = r.path(owner_lock_path).open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    state_path = directory / 'state.json'
    if state_path.exists() and not allow_existing_state:
        raise ValueError('Use a reviewed continuation after an existing attempt, never silently replay')
    seeded = support.initial_rows(plan)
    state = dict(schema='paper_campaign_state_v1', plan_ref=r.ref(plan_path), controller_pid=os.getpid(),
                 tasks=seeded if seeded is not None else {t['id']: dict(status='pending') for t in plan['tasks']})
    metadata = support.state_metadata(plan, plan_path)
    if set(metadata) & {'tasks', 'controller_pid', 'schema'}:
        raise ValueError('Bootstrap metadata cannot override worker identity or task rows')
    state.update(metadata)
    # Explicit process-local operational probe: child interpreters retain their
    # original runtime and scientific code. All support accounting reads here
    # must propagate traversal failures instead of silently undercounting.
    previous_scope_probe = r.scope_bytes
    r.scope_bytes = strict_scope_bytes
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
    terminal_payloads = {}
    retiring_logs = {}
    prepared_outputs = set()
    prepared_logs = {}
    gpu_leases = {}
    probe.install_process_probes()
    monitor = probe.ProbeMonitor(directory / 'probe_health.json')
    def cancel(signum, frame):
        raise RuntimeError('Controller received explicit termination signal')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        while True:
            for retired_id, retired_log in list(retiring_logs.items()):
                if observe_io(monitor, 'close_log:' + retired_id, close_log, retired_log):
                    del retiring_logs[retired_id]
            for task_id, (process, log, start, start_ticks) in list(running.items()):
                task = next(t for t in plan['tasks'] if t['id'] == task_id)
                row = state['tasks'][task_id]
                if start_ticks is None:
                    start_ticks = observe_io(monitor, 'identity:' + str(process.pid), process_start_ticks, process.pid)
                    if start_ticks is not None:
                        row['process_start_ticks'] = start_ticks
                        running[task_id] = (process, log, start, start_ticks)
                code = process.poll()
                if code is not None and start_ticks is None:
                    row['error'] = 'Worker exited before its start identity could be recorded'
                    monitor.errors.pop('identity:' + str(process.pid), None)
                if row.get('termination_requested'):
                    code = monitor.observe('stop:' + str(process.pid), stop_owned, process, start_ticks)
                    if code is None:
                        continue
                elif code is None:
                    usage = monitor.observe('pss:' + str(process.pid), processes.pss, process.pid)
                    bounds = observe_io(monitor, 'bounds:' + task_id, resource_observation, plan, task, used)
                    size, auxiliary_exceeded, scope_exceeded = bounds if bounds is not None else (None, False, False)
                    if usage is not None:
                        row['observed_pss_bytes'] = usage
                        row['observed_pss_unix'] = time.time()
                    row['pss_observation_status'] = 'available' if usage is not None else 'unavailable'
                    if size is not None:
                        row['observed_output_bytes'] = size
                    row['storage_observation_status'] = 'available' if bounds is not None else 'unavailable'
                    metadata_runtime_ok = observe_io(monitor, 'metadata_admission', support.metadata_within_bound, plan)
                    if (usage is not None and usage > task['max_rss_bytes']) or (size is not None and size > task['max_output_bytes']) or auxiliary_exceeded or scope_exceeded or metadata_runtime_ok is False:
                        row['error'] = 'Resource reservation exceeded'
                        row['termination_requested'] = True
                        code = monitor.observe('stop:' + str(process.pid), stop_owned, process, start_ticks)
                if code is None and not row.get('termination_requested') and time.time()-start > task['max_seconds']:
                    row['error'] = 'Wall time limit exceeded'
                    row['termination_requested'] = True
                    code = monitor.observe('stop:' + str(process.pid), stop_owned, process, start_ticks)
                if code is None:
                    continue
                drained = monitor.observe('drain:' + str(process.pid), processes.drain, process)
                if drained is None:
                    continue
                if not drained:
                    row.setdefault('leader_exit_unix', time.time())
                    if time.time() - row['leader_exit_unix'] < 15:
                        continue
                    row['error'] = 'Task leader exited with surviving descendants; own-session drain required'
                    terminated = monitor.observe('drain:' + str(process.pid), processes.drain, process, terminate=True)
                    if terminated is None or not terminated:
                        continue
                    row['error'] = 'Task leader exited with surviving descendants, drained own session'
                if task_id not in terminal_payloads:
                    observation = observe_io(monitor, 'exit:' + task_id, exit_observation, task, used)
                    if observation is None:
                        continue
                    final_bytes, auxiliary_exceeded, result_ref = observation
                    finished = dict(row)
                    if final_bytes > task['max_output_bytes']:
                        finished['error'] = 'Final output exceeds reservation'
                    if auxiliary_exceeded:
                        finished['error'] = 'Final auxiliary scope exceeds reservation'
                    success = code == 0 and result_ref is not None and 'error' not in finished
                    finished.update(status='succeeded' if success else 'failed', actual_returncode=code,
                                    actual_wait=True, owned_process_group_absent=True,
                                    finished_unix=time.time(), retained_bytes=final_bytes)
                    if success:
                        finished['result_ref'] = result_ref
                    else:
                        finished.setdefault('error', 'Nonzero exit or absent completion artifact')
                    terminal_payloads[task_id] = finished
                finished = terminal_payloads[task_id]
                saved = observe_io(monitor, 'receipt:' + task_id, publish_wait,
                                   r.path(task['output_directory']) / 'actual_wait.json', finished)
                if saved is None:
                    continue
                row.clear()
                row.update(finished)
                del running[task_id]
                retiring_logs[task_id] = log
                del terminal_payloads[task_id]
                monitor.forget_process(process.pid)
                for prefix in ('bounds:', 'exit:', 'receipt:', 'identity:'):
                    monitor.errors.pop(prefix + (str(process.pid) if prefix == 'identity:' else task_id), None)
                gpu_ownership.release(gpu_leases.pop(task_id, None))
            for task in plan['tasks']:
                row = state['tasks'][task['id']]
                if row['status'] != 'pending' or task['kind'] != 'external':
                    continue
                try:
                    result = filesystem_observation(external_result)(task)
                    if result:
                        evidence = filesystem_observation(support.external_evidence)(task)
                        row.update(status='succeeded', result_ref=result, reused=True, **evidence)
                    monitor.errors.pop('external:' + task['id'], None)
                except (probe.ProbeUnavailable, OSError) as exc:
                    monitor.unavailable('external:' + task['id'], exc)
                    row['waiting_reason'] = 'Producer process observation unavailable'
                except Exception as exc:
                    row.update(status='blocked', error=str(exc))
            def disk_snapshot():
                scopes = {limit['path']: r.scope_bytes(r.path(limit['path'])) for limit in plan.get('scope_limits', [])}
                sizes = {t['id']: used(t) for t in plan['tasks'] if t['kind'] != 'external'}
                auxiliary = {str(r.path(extra['path'])): r.scope_bytes(r.path(extra['path']))
                             for t in plan['tasks'] for extra in t.get('additional_mutable_scopes', [])}
                return scopes, sizes, auxiliary
            disk = observe_io(monitor, 'disk_inventory', disk_snapshot)
            scope_usage, sizes, auxiliary_sizes = disk if disk is not None else ({}, {}, {})
            state['scope_usage_bytes'] = scope_usage if disk is not None else None
            observed_gpus = monitor.observe('gpu_inventory', probe.gpu_observation, gpu_state)
            snapshot = observed_gpus if observed_gpus is not None else {}
            state['gpu_observation'] = observed_gpus
            accounting = observe_io(monitor, 'admission_accounting', support.admission_accounting, plan)
            if accounting is not None:
                state['closed_writer_accounting'] = accounting
            state['closed_writer_accounting_stale'] = accounting is None
            metadata_ok = observe_io(monitor, 'metadata_admission', support.metadata_within_bound, plan)
            free_ram = monitor.observe('host_memory', probe.host_available_bytes)
            foreign_growth = observe_io(monitor, 'foreign_owner_growth', support.foreign_owner_growth, plan)
            allowed_gpus = observe_io(monitor, 'allowed_gpu_owners', gpu_ownership.allowed, plan)
            def old_worker_growth():
                growth = 0
                for old in plan.get('existing_workers', []):
                    p = Path('/proc') / str(old['pid']) / 'stat'
                    if p.exists():
                        fields = p.read_text().rsplit(')', 1)[1].split()
                        if fields[19] == old['start_ticks']:
                            growth += max(0, old['max_rss_bytes'] - processes.pss(old['pid']))
                return growth
            old_growth = observe_io(monitor, 'existing_workers', old_worker_growth)
            publish_state(monitor, state_path, state)
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
                if global_admission_blocked(monitor):
                    row['waiting_reason'] = 'Monitoring degraded; no new admissions'
                    continue
                violated = next((limit for limit in plan.get('scope_limits', [])
                    if r.path(task['output_directory']).is_relative_to(r.path(limit['path']))
                    and scope_usage[limit['path']] > limit['max_bytes']), None)
                if violated:
                    row.update(status='blocked', error='Campaign scope limit exceeded: ' + violated['path'])
                    continue
                new_growth = task['max_output_bytes'] + sum(max(0, extra['max_bytes'] - auxiliary_sizes[str(r.path(extra['path']))]) for extra in task.get('additional_mutable_scopes', []) if extra.get('growth_allowed', True))
                scope_fits = observe_io(monitor, 'task_admission:' + task['id'], support.task_fits_scope,
                                        plan, state, task, [t for t in plan['tasks'] if t['id'] in running])
                if scope_fits is None:
                    row['waiting_reason'] = 'Storage observation unavailable'
                    continue
                if not accounting['admitted'] or not metadata_ok or not scope_fits:
                    row['waiting_reason'] = 'Storage reservation'
                    continue
                active = [t for t in plan['tasks'] if t['id'] in running]
                if len(active) >= plan.get('max_parallel_tasks', 6):
                    row['waiting_reason'] = 'Parallel worker capacity'
                    continue
                remaining_growth = plan['foreign_ram_reserve_bytes'] + foreign_growth + sum(max(0, t['max_rss_bytes'] - state['tasks'][t['id']].get('observed_pss_bytes', 0)) for t in active)
                remaining_growth += old_growth
                if free_ram < task['max_rss_bytes'] + remaining_growth + 64*2**30:
                    row['waiting_reason'] = 'Host RAM headroom'
                    continue
                gpu = None
                if task['kind'] == 'gpu':
                    task_gpus = observe_io(monitor, 'task_admission:' + task['id'], getattr(gpu_ownership, 'allowed_for_task', lambda p,t,a:a), plan, task, allowed_gpus)
                    if task_gpus is None:
                        row['waiting_reason'] = 'Task GPU policy unavailable'
                        continue
                    candidates = [i for i,g in snapshot.items() if i in task_gpus and g['free_bytes'] >= task['minimum_free_gpu_bytes']
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
                    for source in concrete.get('source_refs', []) + plan.get('controller_source_refs', []):
                        if r.ref(source['path']) != source:
                            raise ValueError('Pinned source changed: ' + source['path'])
                except OSError as exc:
                    monitor.unavailable('source_read:' + task['id'], exc)
                    row['waiting_reason'] = 'Source observation unavailable'
                    continue
                except Exception as exc:
                    row.update(status='blocked', error=str(exc))
                    continue
                monitor.errors.pop('source_read:' + task['id'], None)
                output = r.path(task['output_directory'])
                try:
                    if task['id'] not in prepared_outputs and output.exists() and not support.permit_existing_output(plan, task, row):
                        row.update(status='blocked', error='Output already exists, no overwrite')
                        continue
                    # Remember only this controller's verified, never-spawned preparation.
                    # This is not persisted as permission to replay a prior process.
                    prepared_outputs.add(task['id'])
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
                    if task['id'] not in prepared_logs:
                        prepared_logs[task['id']] = log_path.open('xb')
                except ValueError as exc:
                    row.update(status='blocked', error=str(exc))
                    continue
                except OSError as exc:
                    monitor.unavailable('prepare:' + task['id'], exc)
                    row['waiting_reason'] = 'Launch preparation I/O unavailable; no worker spawned'
                    continue
                monitor.errors.pop('prepare:' + task['id'], None)
                observed_lease = observe_io(monitor, 'lease:' + task['id'], lambda: {'lease': gpu_ownership.acquire(gpu) if gpu is not None else None})
                if observed_lease is None:
                    row['waiting_reason'] = 'GPU ownership observation unavailable'
                    continue
                lease = observed_lease['lease']
                if lease is False:
                    row['waiting_reason'] = 'Root or successor owns exclusive GPU3 lock'
                    continue
                gpu_leases[task['id']] = lease
                log = prepared_logs.pop(task['id'])
                try:
                    process = subprocess.Popen(command, cwd=r.ROOT, env=environment(task, gpu),
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                               pass_fds=(() if lease is None else (lease,)))
                except OSError as exc:
                    row.update(status='blocked', error='Process spawn failed; no automatic replay: ' + str(exc))
                    log.close()
                    gpu_ownership.release(gpu_leases.pop(task['id'], None))
                    continue
                started = time.time()
                # Track the real child before any fallible /proc or filesystem read.
                running[task['id']] = (process, log, started, None)
                row.update(status='running', pid=process.pid, process_start_ticks=None,
                           physical_gpu=gpu, reservation_bytes=task['max_output_bytes'], command=command,
                           log_path=str(log_path.relative_to(r.ROOT)), started_unix=started)
                start_ticks = observe_io(monitor, 'identity:' + str(process.pid), process_start_ticks, process.pid)
                if start_ticks is not None:
                    row['process_start_ticks'] = start_ticks
                    running[task['id']] = (process, log, started, start_ticks)
                row.pop('waiting_reason', None)
                if gpu is not None:
                    snapshot[gpu]['free_bytes'] -= task['minimum_free_gpu_bytes']
                publish_state(monitor, state_path, state)
            state['updated_unix'] = time.time()
            state['counts'] = {key: sum(row['status'] == key for row in state['tasks'].values())
                               for key in ('pending', 'running', 'succeeded', 'failed', 'blocked', 'superseded')}
            if accounting is not None:
                state['project_reserved_bytes'] = accounting['total_bytes']
            published = publish_state(monitor, state_path, state)
            if published and not running and not retiring_logs and all(row['status'] in ('succeeded', 'failed', 'blocked', 'superseded') for row in state['tasks'].values()):
                return state
            time.sleep(10)
    finally:
        for task_id, (process, log, start, ticks) in running.items():
            task = next(t for t in plan['tasks'] if t['id'] == task_id)
            if task_id in terminal_payloads:
                finished = terminal_payloads[task_id]
            else:
                code = stop_owned(process, ticks)
                finished = dict(state['tasks'][task_id], status='failed', actual_returncode=code,
                                actual_wait=True, owned_process_group_absent=True,
                                finished_unix=time.time(), error='Controller ended')
            # A genuinely committed terminal receipt is immutable, including
            # when shutdown follows an ambiguous trailing publication read.
            publish_wait(r.path(task['output_directory']) / 'actual_wait.json', finished)
            state['tasks'][task_id].update(finished)
            log.close()
        r.save(state_path, state)
        for log in list(prepared_logs.values()) + list(retiring_logs.values()):
            log.close()
        for lease in gpu_leases.values():
            gpu_ownership.release(lease)
        lock.close()
        r.scope_bytes = previous_scope_probe


# Entrypoints provide reviewed support, lock, GPU policy and adoption behavior explicitly.
