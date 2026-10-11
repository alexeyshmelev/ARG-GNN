"""Separate append-only credits for genuine terminal failed CPU preparations.

Frozen journals/readers remain unchanged. Only the final ceiling predicate of
a SHA-pinned isolated CPU reader is deferred until the combined charge.
No action at import, scientific execution, global process scan or FD scan.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import FunctionType
from scripts import project_budget_v9 as budget
from . import project_storage_closed_scopes_v1 as previous

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE = 'src/arggnn/project_storage_terminal_scopes_v1.py'
JOURNAL = '.runtime/storage/project_budget_terminal_scopes_v1'
MODE = 'journal_terminal_scopes_credit_v1'
CPU_SOURCE = 'src/arggnn/project_storage_cpu_closed_scopes_v1.py'
CPU_SHA256 = '2a33a19c1447442fc99708bf3d73c21e1720a2453a99deb71d1688a4dcc5c6ff'
PREVIOUS_SHA256 = '974bcf342965abb7e7e062547c68123de9bb9414bf9b1f97fd962d1488d3fdd7'
METADATA_CHARGE = 65536
FAILED_PRODUCERS = {
    'src/arggnn/prepare_expanded_pyg_observations_v1.py':
        ('1e266d7e2915b4059aa454d4e1a49a00c5cdcac6e85368fa2ad89913bb714064', 'expanded_pyg_observations_v1'),
    'src/arggnn/prepare_expanded_pyg_observations_v2.py':
        ('5c4bece48c86fd39849109ecea9c6820040379a5e282e17b14630ef2cf1bc33a', 'expanded_pyg_observations_v2'),
}
_CPU = None
_CACHE = None


def _utc(): return datetime.now(timezone.utc).isoformat()
def _ref(path): return budget.v4.reference(ROOT, path)
def _bound(ref): return budget.v4.bound(ROOT, ref, True)
def _pins(view):
    return {k: view[k] for k in ('legacy_ledger_sha256', 'supplemental_ledger_sha256',
        'journal_tail_sha256', 'cpu_overlay_tail_sha256', 'terminal_overlay_tail_sha256')}


def _cpu_reader():
    global _CPU
    raw = budget.v4.bound(ROOT, {'path': CPU_SOURCE, 'sha256': CPU_SHA256})
    budget.v4.bound(ROOT, {'path': previous.SOURCE, 'sha256': PREVIOUS_SHA256})
    if _CPU is None:
        text = raw.decode()
        old = "if base['hard_limit_bytes']!=1000000000000 or not 0<=projected<min(base['limit_bytes'],base['hard_limit_bytes']):"
        new = "if base['hard_limit_bytes']!=1000000000000 or projected<0:"
        if text.count(old) != 1: raise ValueError('Exact frozen CPU final ceiling predicate required')
        ns = dict(__name__='_arggnn_frozen_cpu_terminal_reader', __package__='arggnn',
                  __file__=str(ROOT / CPU_SOURCE))
        exec(compile(text.replace(old, new), str(ROOT / CPU_SOURCE), 'exec'), ns)
        _CPU = ns
    return _CPU


def _paths(reservation):
    return _cpu_reader()['_paths'](reservation)


def _ceiling(projected, view):
    budget.v4.integer(projected, 'terminal project charge')
    if (view['limit_bytes'] != 950000000000 or view['hard_limit_bytes'] != 1000000000000
            or projected >= min(view['limit_bytes'], view['hard_limit_bytes'])):
        raise ValueError('Combined terminal scheduling/human ceiling exceeded')


def _authority(ref, operation, bindings, *, fresh):
    value = _bound(ref)
    required = dict(kind='project_budget_terminal_overlay_root_admission_v1', root_authorizer='root_agent',
        active=True, operation=operation, accounting_mode=MODE, bindings=bindings,
        all_budget_writers_use_scheduler_mutex=True, all_consumers_and_launchers_use_terminal_overlay_guard=True,
        historical_records_mutation_authorized=False, global_rebase_authorized=False,
        scientific_execution_authorized=False, GPU_authorized=False)
    if any(budget.v4.encoded(value.get(k)) != budget.v4.encoded(v) for k, v in required.items()):
        raise ValueError('Exact root terminal overlay authority required')
    if operation == 'close_failed_cpu' and any(value.get(k) is not True for k in (
            'scope_write_authority_revoked_for_all_cooperative_project_jobs',
            'closed_scope_future_writes_and_relaunch_revoked', 'external_NFS_writers_outside_observation')):
        raise ValueError('Failed CPU closure requires cooperative write/relaunch revocation')
    if operation == 'reserve' and any(ref not in value['reservation']['consumer_bindings'] for ref in bindings.values()):
        raise ValueError('Every frozen/new budget source must be reserved')
    if fresh:
        expiry = datetime.fromisoformat(value['expires_at_utc'].replace('Z', '+00:00'))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc): raise ValueError('Expired terminal authority')
    return value


def _load_events(bindings):
    directory = budget.v4.canonical(ROOT, JOURNAL)
    if not directory.exists(): return [], None
    if not directory.is_dir(): raise ValueError('Canonical terminal journal directory required')
    before = directory.lstat(); names = sorted(p.name for p in directory.iterdir())
    if len(names) > 4096: raise ValueError('Terminal event count cap exceeded')
    events, tail, total = [], None, 0
    for number, name in enumerate(names, 1):
        if name != f'{number:06d}.json': raise ValueError('Terminal journal gap/unknown entry')
        raw, sha = budget.v4.file_bytes(ROOT, directory / name); total += len(raw)
        if len(raw) > METADATA_CHARGE or total > 32 * 1024 * 1024:
            raise ValueError('Terminal metadata charge/cap exceeded')
        event = budget.v4.parse(raw)
        if (set(event) != {'schema', 'sequence', 'previous_sha256', 'bindings', 'operation', 'authority', 'value'}
                or event['schema'] != 'project_budget_terminal_overlay_event_v1'
                or type(event['sequence']) is not int or event['sequence'] != number
                or event['previous_sha256'] != tail or event['bindings'] != bindings
                or event['operation'] not in ('reserve', 'close_failed_cpu')):
            raise ValueError('Immutable terminal event chain drift')
        authority = _authority(event['authority'], event['operation'], bindings, fresh=False)
        if authority.get('expected_before', {}).get('terminal_overlay_tail_sha256') != tail:
            raise ValueError('Terminal authority predecessor drift')
        events.append(event); tail = sha
    after = directory.lstat()
    if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Terminal journal changed during replay')
    return events, tail


def _observation(value, pid):
    if (set(value) != {'pid', 'absent', 'method', 'checked_utc'} or type(value['pid']) is not int
            or value['pid'] != pid or value['absent'] is not True
            or value['method'] != 'targeted os.kill(pid,0) ESRCH'):
        raise ValueError('Exact root targeted PID absence observation required')
    stamp = datetime.fromisoformat(value['checked_utc'].replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp > datetime.now(timezone.utc):
        raise ValueError('Nonfuture timezone PID observation required')
    return stamp


def _failed_cpu(request, reservation, *, observe=False):
    fields = {'key', 'storage_ref', 'failure_ref', 'started_ref', 'input_ref', 'launch_ref', 'drain_ref'}
    if set(request) != fields or request['key'] != reservation['key']:
        raise ValueError('Exact failed CPU closure request required')
    paths = _paths(reservation)
    storage, failure, started, admission, launch, drain = (
        _bound(request[name]) for name in ('storage_ref', 'failure_ref', 'started_ref', 'input_ref', 'launch_ref', 'drain_ref'))
    source_ref = admission['source_ref']; known = FAILED_PRODUCERS.get(source_ref['path'])
    if known is None or source_ref != {'path': source_ref['path'], 'sha256': known[0]}:
        raise ValueError('Exact known failed CPU ProcessPool producer required')
    budget.v4.bound(ROOT, source_ref); prefix = known[1]
    pid = budget.v4.integer(launch.get('pid'), 'failed CPU launch PID', 2)
    if (request['failure_ref']['path'] != str((paths[0] / 'failed.json').relative_to(ROOT))
            or request['started_ref']['path'] != str((paths[0] / 'started.json').relative_to(ROOT))
            or (paths[0] / 'completed.json').exists()
            or storage.get('kind') != 'project_budget_v9_storage_record'
            or storage.get('accounting_mode') != 'journal_cpu_completed_scopes_credit_v1'
            or storage.get('reservation') != reservation or admission.get('storage') != storage
            or admission.get('schema') != prefix + '_root_admission'
            or admission.get('root_authorizer') != 'root_agent' or admission.get('execution_authorized') is not True
            or admission.get('GPU_authorized') is not False or admission.get('targets_read') is not False
            or admission.get('outer_included') is not False or admission.get('CPU_workers') != 8
            or admission.get('output_directory') != str(paths[0].relative_to(ROOT))
            or storage.get('output_directory') != admission['output_directory']
            or storage.get('reserved_new_bytes') != reservation['reserved_bytes']
            or failure.get('schema') != prefix + '_failure' or failure.get('input_ref') != request['input_ref']
            or not isinstance(failure.get('error'), str) or not failure['error']
            or not isinstance(failure.get('traceback'), str) or not failure['traceback']
            or started.get('schema') != prefix + '_started' or started.get('input_ref') != request['input_ref']
            or started.get('source_ref') != source_ref or started.get('CPU_workers') != 8
            or launch.get('input_ref') != request['input_ref'] or launch.get('source_ref') != source_ref
            or launch.get('job_id') != admission.get('job_id') or launch.get('CPU_workers') != 8
            or launch.get('GPU_used') is not False or launch.get('outer_included') is not False):
        raise ValueError('Actual failed own-scope CPU preparation/input/launch evidence required')
    runtime = _bound(admission['runtime_ref'])
    if (runtime.get('device') != 'cpu' or runtime.get('CUDA_VISIBLE_DEVICES') != ''
            or runtime.get('job_id') != admission['job_id'] or runtime.get('source_ref') != source_ref
            or runtime.get('output_directory') != admission['output_directory']):
        raise ValueError('Bound original CPU runtime required')
    for ref in (source_ref, admission['config_ref'], admission['runtime_ref'], admission['registration_ref'],
                admission['producer_ref'], admission['original_producer_ref'], admission['storage_adapter_ref']):
        if ref not in reservation['consumer_bindings']: raise ValueError('Failed input dependencies differ from reservation')
        budget.v4.bound(ROOT, ref)
    for path, sha in admission['source_sha256'].items():
        ref = {'path': path, 'sha256': sha}
        if ref not in reservation['consumer_bindings']: raise ValueError('Original source pin absent from reservation')
        budget.v4.bound(ROOT, ref)
    command = [str(ROOT / '.runtime/env/bin/python'), '-B', '-m', source_ref['path'][4:-3].replace('/', '.'),
               '--execute', '--input', request['input_ref']['path'], '--expected-input-sha256', request['input_ref']['sha256']]
    if launch.get('command') != command: raise ValueError('Exact failed CPU launch command required')
    drain_fields = {'kind', 'root_authorizer', 'status', 'evidence_method', 'input_ref', 'source_ref',
        'failure_ref', 'launch_ref', 'events_ref', 'pids', 'observations', 'owned_process_pool_drained', 'GPU_used'}
    if (set(drain) != drain_fields or drain['kind'] != 'project_budget_failed_cpu_drain_receipt_v1'
            or drain['root_authorizer'] != 'root_agent' or drain['status'] != 'failed_CPU_preparation_workers_drained'
            or drain['evidence_method'] != 'pinned_finally_after_owned_ProcessPool_drain'
            or any(drain[name] != request[name] for name in ('input_ref', 'failure_ref', 'launch_ref'))
            or drain['source_ref'] != source_ref or drain['owned_process_pool_drained'] is not True
            or drain['GPU_used'] is not False
            or drain['events_ref']['path'] != str((paths[0] / 'events.jsonl').relative_to(ROOT))):
        raise ValueError('Bound root owned ProcessPool drain receipt required')
    rows = [budget.v4.parse(line) for line in budget.v4.bound(ROOT, drain['events_ref']).splitlines()]
    if not 2 <= len(rows) <= 4096: raise ValueError('Bounded actual preparation events required')
    failed = [row for row in rows if row.get('event') == 'failed']; cleanup = rows[-1]
    # The exact pinned sources call _drain first. scratch.rmdir is reached only
    # after wait=True shutdown and the check of every owned process.is_alive().
    if (len(failed) != 1 or any(failed[0].get(k) != v for k, v in failure.items() if k not in ('schema', 'input_ref'))
            or cleanup.get('event') != 'worker_cleanup_failed' or cleanup.get('error_type') != 'OSError'
            or cleanup.get('error') != "[Errno 39] Directory not empty: '" + str(paths[0] / 'worker_tmp') + "'"
            or 'scratch.rmdir()' not in cleanup.get('traceback', '')
            or str(ROOT / source_ref['path']) not in cleanup.get('traceback', '')
            or any(row.get('event') == 'completed' for row in rows)):
        raise ValueError('Actual post-drain cleanup traceback required; no success substitution')
    pids = drain['pids']
    if not isinstance(pids, list) or not 1 <= len(pids) <= 32 or pids != sorted(set(pids)):
        raise ValueError('Unique sorted bounded owned CPU PIDs required')
    for value in pids: budget.v4.integer(value, 'owned CPU PID', 2)
    recorded = {pid}
    for row in rows:
        for field in ('worker_pid', 'prepare_worker_pid'):
            if field in row: recorded.add(budget.v4.integer(row[field], 'recorded pool PID', 2))
    if not recorded <= set(pids) or len(drain['observations']) != len(pids):
        raise ValueError('Main and every recorded pool PID require root absence evidence')
    cleanup_time = datetime.fromisoformat(cleanup['utc'].replace('Z', '+00:00'))
    if cleanup_time.tzinfo is None: raise ValueError('Timezone drain event required')
    for value, observation in zip(pids, drain['observations']):
        if _observation(observation, value) < cleanup_time:
            raise ValueError('Root PID absence must be observed after actual ProcessPool drain')
    observations = [_cpu_reader()['_absent'](value) for value in pids] if observe else drain['observations']
    return dict(status='failed_CPU_preparation', generation_completed=False,
        failure_verified=True, owned_process_pool_drained=True, GPU_used=False, observations=observations)


def _state_locked(source_sha, launcher_sha, *, extra=(), fresh=False):
    global _CACHE
    cpu = _cpu_reader()
    bindings = dict(implementation={'path': budget.SOURCE_REL, 'sha256': source_sha},
        launcher={'path': budget.LAUNCHER_REL, 'sha256': launcher_sha},
        cpu_overlay={'path': CPU_SOURCE, 'sha256': CPU_SHA256}, terminal_overlay=_ref(SOURCE))
    dependencies = _CACHE[3] if _CACHE is not None else ()
    watched = sorted(set(extra) | set(dependencies) | {SOURCE, JOURNAL, previous.SOURCE})
    before = cpu['_fingerprint'](watched)
    if not fresh and _CACHE is not None and _CACHE[0] == bindings and _CACHE[1] == before:
        return copy.deepcopy(_CACHE[2])
    reservations, closed, base, _, _ = cpu['_state_locked'](source_sha, launcher_sha, extra=extra, fresh=fresh)
    reservations = copy.deepcopy(reservations); closed = set(closed)
    events, tail = _load_events(bindings); legacy = budget.v4.read_json(ROOT, budget.v4.LEGACY)[0]
    occupied = [budget.v4.canonical(ROOT, r['output']) for r in legacy['jobs'].values()]
    for reservation in reservations.values(): occupied.extend(_paths(reservation))
    failed_closed, overlay = {}, {}
    discovered = set(cpu['_CACHE'][3]) | set(extra) | {SOURCE, JOURNAL, previous.SOURCE}
    def collect(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}: discovered.add(value['path'])
            for child in value.values(): collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value: collect(child)
    for event in events:
        authority = _authority(event['authority'], event['operation'], bindings, fresh=False)
        value = event['value']; collect(event); collect(authority)
        if event['operation'] == 'reserve':
            request = authority['reservation']
            if value != {**request, 'root_admission': event['authority']}: raise ValueError('Terminal reservation authority drift')
            paths = _paths(value); key = value['key']
            if (len(reservations) + len(overlay) >= 4096 or key in reservations or key in overlay
                    or key in legacy['jobs'] or any(budget.v4.paths_overlap(a, b) for a in paths for b in occupied)):
                raise ValueError('Terminal reservation key/scope collision')
            overlay[key] = value; occupied.extend(paths)
        else:
            request = authority['closure']; key = request['key']; reservation = reservations.get(key) or overlay.get(key)
            if (reservation is None or key in closed or key in failed_closed
                    or set(value) != {'request', 'reservation_sha256', 'measurement', 'failure_proof',
                        'closed_utc', 'future_writes_and_relaunch_revoked'}
                    or value['request'] != request or value['reservation_sha256'] != budget.digest(reservation)
                    or value['future_writes_and_relaunch_revoked'] is not True):
                raise ValueError('Unknown/duplicate failed closure or certificate drift')
            proof = _failed_cpu(request, reservation); recorded = value['failure_proof']
            if (set(recorded) != set(proof) or any(recorded[k] != proof[k] for k in proof if k != 'observations')
                    or len(recorded['observations']) != len(proof['observations'])):
                raise ValueError('Explicit failure/drain proof drift')
            for observed, original in zip(recorded['observations'], proof['observations']):
                if _observation(observed, original['pid']) < _observation(original, original['pid']):
                    raise ValueError('Applied PID proof predates bound root drain observation')
            measurement = value['measurement']
            if (set(measurement) != {'conservative_scope_bytes', 'entries', 'scope_paths', 'stat_only'}
                    or measurement['scope_paths'] != [str(p) for p in _paths(reservation)] or measurement['stat_only'] is not True):
                raise ValueError('Exact single owned-scope stat measurement required')
            actual = budget.v4.integer(measurement['conservative_scope_bytes'], 'failed retained bytes')
            budget.v4.integer(measurement['entries'], 'failed retained entries')
            if actual > reservation['reserved_bytes']: raise ValueError('Failed retained scope exceeds allowance')
            failed_closed[key] = value
            for name in request:
                if name != 'key': collect(_bound(request[name]))
    reservations.update(overlay)
    credit = sum(reservations[k]['reserved_bytes'] - value['measurement']['conservative_scope_bytes']
                 for k, value in failed_closed.items())
    added = sum(value['reserved_bytes'] for value in overlay.values())
    projected = base['projected_bytes'] + added - credit + len(events) * METADATA_CHARGE
    _ceiling(projected, base)
    view = dict(base, original_cpu_projected_bytes=base['projected_bytes'], projected_bytes=projected,
        terminal_closed_allowance_credit_bytes=credit, terminal_overlay_reserved_bytes=added,
        terminal_overlay_metadata_charge_bytes=len(events) * METADATA_CHARGE,
        terminal_overlay_tail_sha256=tail, terminal_overlay_events=len(events), accounting_mode=MODE,
        global_process_inspection_performed=False, original_journal_mutated=False)
    result = reservations, closed | set(failed_closed), view, bindings, events
    if before != cpu['_fingerprint'](watched): raise ValueError('Accounting namespace changed during replay')
    _CACHE = bindings, cpu['_fingerprint'](sorted(discovered)), copy.deepcopy(result), tuple(sorted(discovered))
    return result


def inspect_v1(*, expected_source_sha256, expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        return _state_locked(expected_source_sha256, expected_launcher_sha256, fresh=True)[2]


def _append(operation, authority_ref, value, state):
    _, _, view, bindings, events = state
    if len(events) >= 4096: raise ValueError('Terminal event count cap exceeded')
    event = dict(schema='project_budget_terminal_overlay_event_v1', sequence=len(events) + 1,
        previous_sha256=view['terminal_overlay_tail_sha256'], bindings=bindings, operation=operation,
        authority=authority_ref, value=value)
    raw = budget.v4.encoded(event) + b'\n'
    if len(raw) > METADATA_CHARGE: raise ValueError('Terminal event exceeds charged metadata allowance')
    directory = budget.v4.canonical(ROOT, JOURNAL); directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / f'{len(events) + 1:06d}.json'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'wb') as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)
    stored, sha = budget.v4.file_bytes(ROOT, path)
    if stored != raw: raise ValueError('Terminal append uncertainty; preserve partial event')
    return sha


def apply_close_failed_cpu_v1(authority_ref, *, expected_source_sha256, expected_launcher_sha256):
    with budget.v4.mutex(ROOT):
        state = _state_locked(expected_source_sha256, expected_launcher_sha256, fresh=True)
        reservations, closed, view, bindings, _ = state
        authority = _authority(authority_ref, 'close_failed_cpu', bindings, fresh=True)
        if authority['expected_before'] != _pins(view): raise ValueError('Fresh historical/CPU/terminal pins required')
        request = authority['closure']; key = request['key']
        if key not in reservations or key in closed: raise ValueError('Unknown/already closed failed CPU key')
        reservation = reservations[key]; proof = _failed_cpu(request, reservation, observe=True)
        measurement = budget.v4.measure_scopes(ROOT, _paths(reservation))  # Exactly one owned-scope inventory.
        for observation in proof['observations']: _cpu_reader()['_absent'](observation['pid'])
        actual = measurement['conservative_scope_bytes']
        if actual > reservation['reserved_bytes']: raise ValueError('Failed output exceeds original allowance')
        projected = view['projected_bytes'] - (reservation['reserved_bytes'] - actual) + METADATA_CHARGE
        _ceiling(projected, view)
        value = dict(request=copy.deepcopy(request), reservation_sha256=budget.digest(reservation),
            measurement=measurement, failure_proof=proof, closed_utc=_utc(), future_writes_and_relaunch_revoked=True)
        tail = _append('close_failed_cpu', authority_ref, value, state)
        return dict(kind='project_budget_terminal_failed_cpu_close_receipt_v1', key=key, retained_bytes=actual,
            allowance_bytes=reservation['reserved_bytes'], credit_bytes=reservation['reserved_bytes'] - actual,
            projected_bytes=projected, terminal_overlay_tail_sha256=tail, value=value, scope_measurements=1,
            original_journal_mutated=False, global_process_scan_performed=False, GPU_used=False,
            status='failed_CPU_preparation', generation_completed=False)


def _adapt(name):
    method = _cpu_reader()[name]
    ns = dict(method.__globals__, SOURCE=SOURCE, MODE=MODE, JOURNAL=JOURNAL,
        _state_locked=_state_locked, _pins=_pins, _paths=_paths, _authority=_authority, _append=_append,
        budget_check_v9=budget_check_v9, check_storage_locked_v1=check_storage_locked_v1)
    value = FunctionType(method.__code__, ns, method.__name__, method.__defaults__, method.__closure__)
    value.__kwdefaults__ = method.__kwdefaults__
    return value


def apply_reserve_v1(authority_ref, *, expected_source_sha256, expected_launcher_sha256):
    # Preserve the frozen exclusivity, overlap, ownership and allowance checks.
    value = _adapt('apply_reserve_v1')(authority_ref, expected_source_sha256=expected_source_sha256,
                                     expected_launcher_sha256=expected_launcher_sha256)
    value['kind'] = 'project_budget_terminal_reserve_receipt_v1'
    value['terminal_overlay_tail_sha256'] = value.pop('cpu_overlay_tail_sha256')
    return value


def check_storage_locked_v1(storage, *, first=False, boundary=False, required_write_bytes=0):
    """Caller holds the canonical scheduler mutex; never reacquire it."""
    return _adapt('check_storage_locked_v1')(storage, first=first, boundary=boundary,
                                             required_write_bytes=required_write_bytes)


def budget_check_v9(storage, *, first=False, boundary=False, required_write_bytes=0):
    with budget.v4.mutex(ROOT):
        return check_storage_locked_v1(storage, first=first, boundary=boundary, required_write_bytes=required_write_bytes)


def validate_storage_v9(spec, cfg, storage, *, source_ref, config_ref, runtime_ref):
    return _adapt('validate_storage_v9')(spec, cfg, storage, source_ref=source_ref, config_ref=config_ref, runtime_ref=runtime_ref)


def _writer_method(method):
    value = FunctionType(method.__code__, dict(method.__globals__, budget_check_v9=budget_check_v9),
                         method.__name__, method.__defaults__, method.__closure__)
    value.__kwdefaults__ = method.__kwdefaults__
    return value


class PilotOutputV9(previous.PilotOutputV9):
    __init__ = _writer_method(previous.PilotOutputV9.__init__)
    save = _writer_method(previous.PilotOutputV9.save)
    discard = _writer_method(previous.PilotOutputV9.discard)


if __name__ == '__main__':
    print(json.dumps(dict(status='source_only_terminal_overlay_not_applied', source=_ref(SOURCE),
        accounting_mode=MODE, frozen_cpu_source={'path': CPU_SOURCE, 'sha256': CPU_SHA256}), indent=2))

