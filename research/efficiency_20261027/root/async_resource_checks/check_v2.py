"""Keep periodic resource scans off our trainer's dispatch thread.

The original callback still measures the entire owned family, checks the same
caps and writes the same status file. Integer prospective writes remain
synchronous. This prototype is not installed into any running benchmark.
"""
from __future__ import annotations
import threading
import math
import time


class AsyncResourceCheck:
    def __init__(self, callback, *, interval_seconds=30., maximum_scan_age_seconds=120., clock=time.monotonic):
        if not callable(callback) or not isinstance(getattr(callback, 'task', None), dict):
            raise ValueError('Original task-bearing resource callback required')
        if not hasattr(callback, 'started') or not hasattr(callback, 'next_check'):
            raise ValueError('Original throttled resource callback required')
        if not 0 < interval_seconds < maximum_scan_age_seconds:
            raise ValueError('Positive interval shorter than finite scan lease required')
        if not all(math.isfinite(v) for v in (interval_seconds, maximum_scan_age_seconds)):
            raise ValueError('Finite intervals required')
        self.callback = callback
        self.task = callback.task
        self.started = callback.started
        self.interval = float(interval_seconds)
        self.maximum_age = float(maximum_scan_age_seconds)
        self.clock = clock
        self._callback_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._last_success = None
        self._closed = False
        self._scans = 0
        self._scan_seconds = 0.
        self._maximum_scan_seconds = 0.
        self._synchronous_writes = 0

    def _latch(self, error):
        with self._state_lock:
            if self._error is None:
                self._error = error

    def _budget(self):
        with self._state_lock:
            error, last = self._error, self._last_success
        if error is not None:
            raise RuntimeError('Periodic original resource check failed') from error
        now = self.clock()
        if now-self.started > self.task['max_seconds']:
            raise RuntimeError('Task wall time limit reached')
        if last is not None and now-last > self.maximum_age:
            raise RuntimeError('Resource scan exceeded its finite freshness lease')

    def _scan(self, stage, *, force):
        acquired = False
        try:
            # Prospective writes wait for an in-flight scan, but may not wait
            # past the existing wall/freshness lease and then renew it.
            while not acquired:
                self._budget()
                acquired = self._callback_lock.acquire(timeout=min(.05, self.maximum_age/4))
            self._budget()
            began = self.clock()
            if force:
                self.callback.next_check = 0.
            self.callback(stage)
            finished = self.clock()
            duration = finished-began
            if duration > self.maximum_age:
                raise RuntimeError('Original resource scan duration exceeded its finite lease')
            # Do not allow a late successful callback to erase a prior timeout.
            self._budget()
            with self._state_lock:
                self._last_success = finished
                self._scans += 1
                self._scan_seconds += duration
                self._maximum_scan_seconds = max(self._maximum_scan_seconds, duration)
                if type(stage) is int:
                    self._synchronous_writes += 1
        except BaseException as error:
            self._latch(error)
            raise
        finally:
            if acquired:
                self._callback_lock.release()

    def start(self):
        if self._thread is not None or self._closed:
            raise RuntimeError('Monitor starts once')
        # One complete measurement precedes background use and GPU dispatch.
        self._scan('TreeLoom_async_monitor_initial', force=True)
        self._thread = threading.Thread(target=self._run, name='TreeLoom_resource_scan', daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self._scan('TreeLoom_async_periodic_resources', force=True)
            except BaseException:
                return

    def _check(self):
        self._budget()
        with self._state_lock:
            last = self._last_success
        if self._closed or self._thread is None:
            raise RuntimeError('Resource monitor is not active')
        now = self.clock()
        if now-self.started > self.task['max_seconds']:
            raise RuntimeError('Task wall time limit reached')
        if last is None or now-last > self.maximum_age:
            raise RuntimeError('Resource scan exceeded its finite freshness lease')
        if not self._thread.is_alive():
            raise RuntimeError('Resource monitor stopped unexpectedly')

    def __call__(self, stage):
        self._check()
        if type(stage) is int:
            if stage < 0:
                raise ValueError('Nonnegative write request required')
            # Original positive/zero byte semantics and original cap check.
            self._scan(stage, force=True)
            self._check()
        # Ordinary stage names do not wait behind a process-memory scan.
        # Training keeps its existing cheap own-RSS and phase admission checks.

    def receipt(self):
        with self._state_lock:
            return dict(periodic_scans=self._scans,
                scan_seconds=self._scan_seconds, maximum_scan_seconds=self._maximum_scan_seconds,
                synchronous_prospective_write_checks=self._synchronous_writes,
                interval_after_scan_completion_seconds=self.interval,
                maximum_scan_age_seconds=self.maximum_age,
                changed_model_or_data=False, original_caps_preserved=True,
                production_deployed=False)

    def close(self):
        if self._closed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.maximum_age)
            if self._thread.is_alive():
                raise RuntimeError('Original resource callback did not finish within finite join')
        self._closed = True
        with self._state_lock:
            error = self._error
        if error is not None:
            raise RuntimeError('Original resource monitor failed before close') from error
        self._scan('TreeLoom_async_monitor_final', force=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        self.close()
        return False
