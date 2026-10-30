"""Narrow observational failures pause admission without inventing measurements.

Install only in a separately reviewed controller process. No process is launched,
signalled, adopted or considered complete by this module. Identity violations,
source drift and verified resource-limit violations retain their original errors.
"""
import functools
import subprocess
import time
from pathlib import Path
import psutil
from arggnn.benchmarking.paper_campaign import processes, runtime as r


class ProbeUnavailable(RuntimeError):
    """A read-only operating-system observation could not be obtained."""


PROBE_ERRORS = (subprocess.TimeoutExpired, subprocess.CalledProcessError,
                FileNotFoundError, PermissionError, ProcessLookupError,
                psutil.NoSuchProcess, psutil.AccessDenied)


def process_observation(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ProbeUnavailable:
            raise
        except PROBE_ERRORS as error:
            raise ProbeUnavailable(type(error).__name__ + ': ' + str(error)[:300]) from error
    return wrapped


def install_process_probes():
    """Convert failures only at the two read-only process inventory boundaries."""
    for name in ('session_members', 'pss'):
        function = getattr(processes, name)
        if not getattr(function, '_observational_probe', False):
            wrapped = process_observation(function)
            wrapped._observational_probe = True
            setattr(processes, name, wrapped)


def gpu_observation(function):
    try:
        return function()
    except ProbeUnavailable:
        raise
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError,
            PermissionError, ValueError) as error:
        raise ProbeUnavailable(type(error).__name__ + ': ' + str(error)[:300]) from error
    except RuntimeError as error:
        if str(error) != 'Missing authorized GPU':
            raise
        raise ProbeUnavailable(str(error)) from error


def host_available_bytes():
    try:
        lines = Path('/proc/meminfo').read_text().splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith('MemAvailable:'))) * 1024
    except (FileNotFoundError, PermissionError, ValueError, StopIteration) as error:
        raise ProbeUnavailable('MemAvailable observation unavailable: ' + type(error).__name__) from error


class ProbeMonitor:
    def __init__(self, path, *, clock=time.time):
        self.path = path
        self.clock = clock
        self.errors = {}
        self.last_good = {}

    @property
    def degraded(self):
        return bool(self.errors)

    def unavailable(self, name, error):
        now = self.clock()
        old = self.errors.get(name, {})
        self.errors[name] = dict(reason=str(error)[:400], first_failure_unix=old.get('first_failure_unix', now),
                                 last_failure_unix=now, failures=old.get('failures', 0) + 1)

    def observe(self, name, function, *args, **kwargs):
        try:
            value = function(*args, **kwargs)
        except ProbeUnavailable as error:
            self.unavailable(name, error)
            return None
        self.errors.pop(name, None)
        self.last_good[name] = dict(observed_unix=self.clock(), available=True)
        return value

    def forget_process(self, pid):
        for name in ('pss:' + str(pid), 'drain:' + str(pid)):
            self.errors.pop(name, None)

    def health(self):
        return dict(schema='queue_probe_health_v1', status='degraded' if self.degraded else 'healthy',
                    new_admissions_allowed=not self.degraded, updated_unix=self.clock(),
                    unavailable=dict(self.errors), last_good=dict(self.last_good),
                    unknown_measurements_reported_as_zero=False, existing_workers_preserved=True)

    def save(self):
        health = self.health()
        r.save(self.path, health)
        return health
