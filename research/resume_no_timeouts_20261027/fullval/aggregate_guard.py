"""One storage-bounded aggregate owner for one full200Mb validation output scope.

Periodic external-write monitoring plus the 4 GiB reserve is provisional; this
is not a filesystem quota or a mathematical native-write bound. No import I/O.
"""
from __future__ import annotations
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
import traceback

RESERVATION = 120 * 2**30 + 128 * 2**20
OPERATING = RESERVATION - 4 * 2**30
METADATA = 256 * 2**20
MAX_SAMPLE_AGE = 5.0
SCAN_INTERVAL = 1.0


class GuardFailure(RuntimeError):
    pass


def confined(scope, path):
    scope, path = Path(scope).absolute(), Path(path).absolute()
    if '..' in path.parts or path.resolve() != path or not path.is_relative_to(scope):
        raise GuardFailure('All writable and temporary paths must be canonical inside charged scope')
    return path


def measure(scope):
    started = time.monotonic()
    logical = allocated = files = metadata = logs = 0
    stack = [Path(scope)]
    while stack:
        path = stack.pop()
        confined(scope, path)
        try:
            info = path.lstat()
        except FileNotFoundError:
            # The guardian atomically replaces exactly this temporary file.
            # Re-measure its destination rather than ignoring arbitrary loss.
            if path != Path(scope) / 'status.tmp':
                raise
            destination = Path(scope) / 'status.json'
            confined(scope, destination)
            info = destination.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise GuardFailure('Atomic status replacement must be an ordinary single-link file')
            # A second destination entry may also be counted. This is safe,
            # conservative accounting for one concurrent status replacement.
            path = destination
        if stat.S_ISLNK(info.st_mode):
            raise GuardFailure('Symlinks are forbidden throughout aggregate output')
        allocated += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            stack.extend(path.iterdir())
        elif stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise GuardFailure('Hardlinked output is forbidden')
            logical += info.st_size
            files += 1
            if path.name.endswith(('.stdout.log', '.stderr.log', '.terminal.json', '.launch.json')) or path.name == 'commands.json':
                metadata += max(info.st_size, info.st_blocks * 512)
            if path.name.endswith(('.stdout.log', '.stderr.log')):
                logs += info.st_size
        else:
            raise GuardFailure('Only ordinary files/directories may occupy aggregate output')
    finished = time.monotonic()
    return dict(logical_bytes=logical, allocated_bytes=allocated,
                charged_bytes=max(logical, allocated), files=files,
                supervisor_metadata_bytes=metadata, captured_log_bytes=logs,
                started_monotonic=started, finished_monotonic=finished,
                scan_seconds=finished-started)


class AggregateOwner:
    """Caller retains this same locked owner through every stage and attempt."""
    def __init__(self, scope, *, execution_check, max_family_rss_bytes, max_seconds):
        if (not callable(execution_check) or type(max_family_rss_bytes) is not int
                or max_family_rss_bytes <= 0 or max_seconds is not None):
            raise ValueError('Current admission, finite family RSS and no total wall cutoff required')
        self.scope = Path(scope).absolute()
        if not self.scope.is_dir() or self.scope.resolve() != self.scope:
            raise GuardFailure('Existing canonical aggregate root required')
        self.execution_check = execution_check
        self.max_family_rss_bytes = max_family_rss_bytes
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.stage = None
        self.stage_started = None
        self._lock = threading.RLock()
        self._stopped = threading.Event()
        self._thread = None
        self._error = None
        self._last = None
        self._pending = 0
        self._closed = False
        self.metrics = dict(scans=0, max_scan_seconds=0., max_sample_age_seconds=0.,
                            max_growth_bytes=0, max_charged_bytes=0)
        lock_path = confined(self.scope, self.scope / '.aggregate_owner.lock')
        self._lease = lock_path.open('a+b')
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.refresh()
            self.check()
        except BaseException:
            self._lease.close()
            raise

    def set_stage(self, name):
        if name not in (None, 'generation', 'native'):
            raise GuardFailure('Only the registered serial generation/native stages')
        self.stage = name
        self.stage_started = time.monotonic()
        self.refresh()
        self.check()

    def path(self, path):
        return confined(self.scope, path)

    def refresh(self):
        sample = measure(self.scope)
        sample['stage_charged_bytes'] = {name: measure(self.scope / name)['charged_bytes']
            if (self.scope / name).exists() else 0 for name in ('generation', 'native')}
        with self._lock:
            previous = self._last
            if sample['supervisor_metadata_bytes'] >= METADATA:
                raise GuardFailure('Aggregate supervisor metadata share exhausted')
            self._last = sample
            # Retain every stream credit for this finite aggregate attempt.
            # This conservatively double-counts flushed logs rather than
            # erasing a reserved chunk that a concurrent scan has not seen.
            # The consequence is an earlier metadata/log operating stop.
            self.metrics['scans'] += 1
            self.metrics['max_scan_seconds'] = max(self.metrics['max_scan_seconds'], sample['scan_seconds'])
            self.metrics['max_charged_bytes'] = max(self.metrics['max_charged_bytes'], sample['charged_bytes'])
            if previous:
                self.metrics['max_growth_bytes'] = max(self.metrics['max_growth_bytes'], sample['charged_bytes']-previous['charged_bytes'])
            return sample

    def check(self):
        if self._closed or self._error is not None:
            raise GuardFailure('Aggregate owner closed or monitor failed: ' + str(self._error))
        sample = self._last
        age = time.monotonic() - sample['started_monotonic'] if sample else math.inf
        self.metrics['max_sample_age_seconds'] = max(self.metrics['max_sample_age_seconds'], age)
        if age > MAX_SAMPLE_AGE:
            raise GuardFailure('Aggregate sample exceeded five-second age limit')
        if sample['charged_bytes'] + self._pending >= OPERATING:
            raise GuardFailure('Aggregate operating stop threshold reached')
        for name, limit in (('generation', 24 * 2**30), ('native', 96 * 2**30)):
            if sample['stage_charged_bytes'][name] >= limit:
                raise GuardFailure('Original full validation stage output cap reached: ' + name)
        self.execution_check()
        return sample

    def __call__(self, prospective_bytes=0):
        if type(prospective_bytes) is not int or prospective_bytes < 0:
            raise GuardFailure('Nonnegative prospective byte bound required')
        # Serial Python writes receive a fresh aggregate measurement. During a
        # native process the independent supervisor also enforces sample age.
        self.refresh()
        self.check()
        if self._last['charged_bytes'] + prospective_bytes >= OPERATING:
            raise GuardFailure('Prospective Python/native write headroom exceeds operating limit')
        if self.stage is not None:
            limit = (24 if self.stage == 'generation' else 96) * 2**30
            if self._last['stage_charged_bytes'][self.stage] + prospective_bytes >= limit:
                raise GuardFailure('Prospective write exceeds original stage output cap')

    def reserve_stream_write(self, size):
        if type(size) is not int or size < 0:
            raise GuardFailure('Nonnegative stream byte bound required')
        self.check()
        with self._lock:
            self.check()
            if self._last['charged_bytes'] + self._pending + size + 4096 >= OPERATING:
                raise GuardFailure('Prospective captured log write exceeds operating limit')
            if self._last['supervisor_metadata_bytes'] + self._pending + size + 4096 >= METADATA:
                raise GuardFailure('Aggregate supervisor metadata share exhausted')
            self._pending += size + 4096

    def start_monitor(self):
        if self._thread is not None:
            raise GuardFailure('Exactly one aggregate monitor allowed')
        self.refresh()
        self.check()
        def scan():
            while not self._stopped.wait(SCAN_INTERVAL):
                try:
                    self.refresh()
                    self.check()
                except BaseException as error:
                    self._error = dict(error=repr(error), error_type=type(error).__name__,
                        errno=getattr(error, 'errno', None), filename=getattr(error, 'filename', None),
                        traceback=traceback.format_exc(), stage=self.stage,
                        observed_unix=time.time())
                    return
        self._thread = threading.Thread(target=scan, name='full-validation-storage', daemon=True)
        self._thread.start()

    def stop_monitor(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=MAX_SAMPLE_AGE)
            if self._thread.is_alive():
                raise GuardFailure('Storage monitor failed to join; attempt cannot complete')

    def write_json(self, path, value, *, emergency=False):
        destination = self.path(path)
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        if len(raw) > 2**20:
            raise GuardFailure('Complete terminal receipt exceeds one MiB bound')
        if not emergency:
            self(len(raw) + 4096)
            if self._last['supervisor_metadata_bytes'] + len(raw) + 4096 > METADATA:
                raise GuardFailure('Complete supervisor receipt exceeds aggregate metadata share')
        else:
            # Failure receipts can use the reserved margin, never ordinary
            # stage admission. Existing failed evidence is not removed.
            sample = measure(self.scope)
            if sample['supervisor_metadata_bytes'] + len(raw) + 4096 > METADATA:
                raise GuardFailure('Complete failure receipt exceeds aggregate metadata share')
            if sample['charged_bytes'] + len(raw) + 4096 > RESERVATION:
                raise GuardFailure('No reserved margin remains for terminal evidence')
        with destination.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self):
        try:
            self.stop_monitor()
            # A final in-flight scan may fail while stop_monitor joins.
            # Completion must observe that failure before releasing ownership.
            if self._error is not None:
                raise GuardFailure('Aggregate monitor failed during close: ' + str(self._error))
        finally:
            self._closed = True
            self._lease.close()
