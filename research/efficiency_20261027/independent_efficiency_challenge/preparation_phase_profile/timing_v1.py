"""Function-boundary timers only; no scientific imports or execution at import."""
from contextlib import contextmanager
from functools import wraps
import threading
import time


class PhaseTimers:
    def __init__(self, emit=lambda row: None):
        self.emit = emit
        self.rows = []
        self.stack = []

    def call(self, phase, function, *args, **kwargs):
        if len(self.rows) + len(self.stack) >= 32:
            raise RuntimeError('At most 32 measured function calls')
        parent = self.stack[-1] if self.stack else None
        self.emit(dict(event='begin', phase=phase, parent_phase=parent,
                       observed_unix=time.time(), thread_id=threading.get_ident()))
        self.stack.append(phase)
        wall, thread, process = time.perf_counter(), time.thread_time(), time.process_time()
        started = time.time()
        success = False
        error_type = None
        try:
            result = function(*args, **kwargs)
            success = True
            return result
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            # Sample clocks before metadata construction/journal emission.
            ended = time.time()
            elapsed = time.perf_counter() - wall
            thread_elapsed = time.thread_time() - thread
            process_elapsed = time.process_time() - process
            self.stack.pop()
            row = dict(event='end', phase=phase, parent_phase=parent,
                wall_start_unix=started, wall_end_unix=ended,
                wall_seconds=elapsed, calling_thread_CPU_seconds=thread_elapsed,
                main_process_CPU_seconds=process_elapsed,
                thread_id=threading.get_ident(), success=success, error_type=error_type)
            self.rows.append(row)
            self.emit(row)

    @contextmanager
    def wrap(self, owner, attribute, phase):
        original = getattr(owner, attribute)
        @wraps(original)
        def measured(*args, **kwargs):
            return self.call(phase, original, *args, **kwargs)
        setattr(owner, attribute, measured)
        try:
            yield original
        finally:
            # Restore the exact original object even after a failed invocation.
            changed = getattr(owner, attribute) is not measured
            setattr(owner, attribute, original)
            if changed:
                raise RuntimeError('Timed binding changed unexpectedly: ' + attribute)

    def require_once(self, phases):
        actual = [row['phase'] for row in self.rows]
        if sorted(actual) != sorted(phases) or not all(row['success'] for row in self.rows):
            raise RuntimeError('One successful original call per declared phase required')
