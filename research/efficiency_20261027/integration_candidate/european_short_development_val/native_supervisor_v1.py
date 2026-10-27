"""Plaintext native argv supervisor. Definition only: no child starts on import.

A caller must supply the one admitted AggregateOwner and independently reviewed
native tool paths. No retry, process cleanup outside owned identity, or GPU.
"""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from .aggregate_guard_v1 import AggregateOwner, GuardFailure, METADATA
from .short_plan_v1 import validate_native_config

LOG_LIMIT = 192 * 2**20
DRAIN_LIMIT = 255 * 2**20
MAX_STEPS = 5


def process_table():
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            text = (path / 'stat').read_text()
            fields = text[text.rfind(')') + 2:].split()
            rows[int(path.name)] = dict(pid=int(path.name), ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]), start=int(fields[19]),
                state=fields[0], rss_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows


def owned_family(table, leader, known):
    result = {}
    for pid, old in known.items():
        current = table.get(pid)
        if current is not None and current['start'] == old['start']:
            result[pid] = current
    current = table.get(leader['pid'])
    if current is not None and current['start'] == leader['start']:
        result[current['pid']] = current
    # The unique child session belongs to this exact newly-created leader.
    for pid, row in table.items():
        if row['session'] == leader['session'] and row['start'] >= leader['start']:
            result[pid] = row
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid not in result and row['ppid'] in result:
                result[pid] = row
                changed = True
    return result


def signal_owned(family, signum):
    current = process_table()
    for pid, old in family.items():
        row = current.get(pid)
        if row is not None and row['start'] == old['start'] and row['state'] != 'Z':
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass


def reap_adopted(family, leader_pid):
    waits = []
    for pid in family:
        if pid == leader_pid:
            continue
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited:
                waits.append(dict(pid=waited, wait_status=status))
        except ChildProcessError:
            pass
    return waits


class NativeSupervisor:
    def __init__(self, owner, *, reviewed_native_tools):
        if type(owner) is not AggregateOwner or not isinstance(reviewed_native_tools, dict):
            raise ValueError('One admitted aggregate owner and reviewed tool hash bindings required')
        self.owner = owner
        self.tools = dict(reviewed_native_tools)
        self.steps = 0
        self.log_bytes = owner._last['captured_log_bytes']
        self.terminal_bytes = 0
        self.owner_pid = os.getpid()
        self.owner_identity = process_table()[self.owner_pid]
        self.active = False

    def _argv(self, argv, work, label):
        if (type(argv) is not list or not argv or any(type(x) is not str or '\x00' in x for x in argv)
                or label not in {'convert_vcf', 'annotate_observed_cohorts', 'relate_initial',
                                  'five_population_size_iterations', 'convert_post_size'}):
            raise GuardFailure('Reviewed plaintext native argument list required')
        binary = argv[2] if argv[:2] == ['bash', '-x'] else argv[0]
        binding = self.tools.get(binary)
        if not binding or hashlib.sha256(Path(binary).read_bytes()).hexdigest() != binding:
            raise GuardFailure('Native executable/script is not in reviewed hash bindings')
        if argv[0] == 'bash' and argv[:2] != ['bash', '-x']:
            raise GuardFailure('Only visible bash -x script invocation allowed')
        if any(value in {'-c', '--command', '-EncodedCommand', '-encodedcommand'} for value in argv):
            raise GuardFailure('Inline/encoded command dispatch is forbidden')
        for i, value in enumerate(argv):
            if i in ({2} if argv[:2] == ['bash', '-x'] else {0}):
                continue
            if value.startswith('/'):
                self.owner.path(value)
        return list(argv)

    def __call__(self, argv, work, journal, label, config, guard):
        validate_native_config(config)
        if guard is not self.owner or self.active or self.steps >= MAX_STEPS:
            raise GuardFailure('Same aggregate guard, serial steps and finite command count required')
        now_identity = process_table().get(self.owner_pid)
        if now_identity != self.owner_identity:
            # RSS and state may change; identity is only PID/start/session.
            if now_identity is None or any(now_identity[k] != self.owner_identity[k]
                                           for k in ('pid', 'start', 'session')):
                raise GuardFailure('Native supervisor owner identity changed')
        work = self.owner.path(work)
        temporary = self.owner.path(work / 'tmp')
        if not work.is_dir() or not temporary.is_dir():
            raise GuardFailure('Charged work and temporary directories required')
        argv = self._argv(argv, work, label)
        self.owner(config['native_step_headroom_bytes'])
        self.log_bytes = max(self.log_bytes, self.owner._last['captured_log_bytes'])
        if self.owner._thread is None:
            self.owner.start_monitor()
        # Adopt orphan grandchildren so actual waits can be recorded. This is
        # set only when an independently admitted native step is invoked.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise GuardFailure('Linux child-subreaper ownership setup failed')
        self.active = True
        self.steps += 1
        started = time.monotonic()
        timeout = min(config['relate']['timeout_seconds_per_step'], self.owner.max_seconds)
        record = dict(label=label, argv=argv, status='failed', returncode=None,
                      owner={k:self.owner_identity[k] for k in ('pid','start','session')},
                      requested_scan_interval_seconds=1., maximum_sample_age_seconds=5.,
                      complete_log=True, excluded_log_bytes=0, waits=[], terminal_family_absent=False,
                      storage_reservation_bytes=16*2**30, operating_threshold_bytes=12*2**30,
                      provisional_external_write_bound=True)
        paths = [self.owner.path(work / (label + '.' + stream + '.log')) for stream in ('stdout','stderr')]
        logs = []
        selector = selectors.DefaultSelector()
        child = None
        leader = None
        known = {}
        failure = None
        shutdown_at = None
        max_rss = 0
        try:
            for path in paths:
                logs.append(path.open('xb'))
            environment = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary),
                               TEMP=str(temporary), CUDA_VISIBLE_DEVICES='')
            child = subprocess.Popen(argv, cwd=work, env=environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
            leader = process_table().get(child.pid)
            if leader is None or leader['session'] != child.pid or leader['pgrp'] != child.pid:
                raise GuardFailure('Genuine newly-owned native session identity missing')
            record['leader'] = {k:leader[k] for k in ('pid','start','session','pgrp')}
            known = {child.pid: leader}
            for pipe, output in zip((child.stdout, child.stderr), logs):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, output)
            while True:
                table = process_table()
                family = owned_family(table, leader, known)
                known.update(family)
                record['waits'].extend(reap_adopted(family, child.pid))
                live = {pid:row for pid,row in family.items() if row['state'] != 'Z'}
                rss = table.get(self.owner_pid, {}).get('rss_bytes', self.owner.max_family_rss_bytes) + sum(row['rss_bytes'] for row in live.values())
                max_rss = max(max_rss, rss)
                if failure is None:
                    try:
                        self.owner.check()
                        if rss >= self.owner.max_family_rss_bytes:
                            raise GuardFailure('Owned recursive family RSS bound reached')
                        if time.monotonic() - started >= timeout:
                            raise GuardFailure('Native finite step wall bound reached')
                    except BaseException as error:
                        failure = repr(error)
                if failure is not None:
                    if shutdown_at is None:
                        shutdown_at = time.monotonic()
                        signal_owned(family, signal.SIGTERM)
                    elif time.monotonic() - shutdown_at >= 2:
                        signal_owned(family, signal.SIGKILL)
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    if self.log_bytes + len(chunk) > DRAIN_LIMIT:
                        record['complete_log'] = False
                        record['excluded_log_bytes'] += len(chunk)
                        failure = failure or 'Finite shutdown log bound exceeded; attempt failed'
                        continue
                    if failure is None:
                        try:
                            self.owner.reserve_stream_write(len(chunk))
                        except BaseException as error:
                            failure = repr(error)
                    key.data.write(chunk)
                    key.data.flush()
                    self.log_bytes += len(chunk)
                    if self.log_bytes >= LOG_LIMIT:
                        failure = failure or 'Aggregate captured native log operating bound reached'
                code = child.poll()
                if code is not None and code != 0 and failure is None:
                    failure = 'Native command exited unsuccessfully'
                if code is not None and live and set(live) != {child.pid} and failure is None:
                    failure = 'Native leader returned with owned descendants still active'
                if code is not None and not live and not selector.get_map():
                    break
            code = child.wait()
            record['returncode'] = code
            record['waits'].append(dict(pid=child.pid, returncode=code, method='Popen.wait'))
            remaining = owned_family(process_table(), leader, known)
            record['waits'].extend(reap_adopted(remaining, child.pid))
            remaining = owned_family(process_table(), leader, known)
            record['terminal_family_absent'] = not remaining
            if remaining:
                failure = failure or 'Owned family not fully absent after actual waits'
            if failure is None:
                self.owner.check()
                record['status'] = 'complete'
        except BaseException as error:
            failure = repr(error)
            record['complete_log'] = False
        finally:
            # Even exceptional Python/monitor paths stop only exact owned IDs,
            # genuinely wait, and never signal the caller or unrelated jobs.
            if child is not None and leader is None:
                child.kill()
                record['returncode'] = child.wait()
                record['waits'].append(dict(pid=child.pid, returncode=record['returncode'], method='Popen.wait_after_identity_failure'))
                record['complete_log'] = False
            if child is not None and leader is not None:
                family = owned_family(process_table(), leader, known)
                known.update(family)
                signal_owned(family, signal.SIGKILL)
                record['returncode'] = child.wait()
                if not any(wait.get('pid') == child.pid for wait in record['waits']):
                    record['waits'].append(dict(pid=child.pid, returncode=record['returncode'], method='Popen.wait'))
                for _ in range(50):
                    family = owned_family(process_table(), leader, known)
                    known.update(family)
                    record['waits'].extend(reap_adopted(family, child.pid))
                    if not family:
                        break
                    signal_owned(family, signal.SIGKILL)
                    time.sleep(0.1)
                record['terminal_family_absent'] = not owned_family(process_table(), leader, known)
            selector.close()
            for stream in logs:
                stream.close()
            if child is not None:
                for pipe in (child.stdout, child.stderr):
                    if pipe is not None:
                        pipe.close()
            self.active = False
            record.update(elapsed_seconds=time.monotonic()-started, max_family_rss_bytes=max_rss,
                          aggregate_log_bytes=self.log_bytes, failure=failure,
                          storage_metrics=dict(self.owner.metrics),
                          tracked_family=[{k:row[k] for k in ('pid','start','session','ppid')} for row in known.values()])
            if failure or not record['terminal_family_absent']:
                record['status'] = 'failed'
            journal.append(record)
            self.owner.write_json(work / (label + '.terminal.json'), record, emergency=record['status'] != 'complete')
            # Each step gets a fresh immutable snapshot; final commands.json is
            # written once, after the fifth successfully completed command.
            if self.steps == MAX_STEPS and record['status'] == 'complete':
                self.owner.write_json(work / 'commands.json', journal)
        if record['status'] != 'complete':
            raise GuardFailure('Native attempt failed; retained terminal evidence: ' + str(failure))
