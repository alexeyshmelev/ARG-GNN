"""One admitted next parent, delivered in unchanged order and cleaned up on exit."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import threading


class PrefetchCancelled(RuntimeError):
    """Internal cooperative cancellation, never a scientific input rejection."""


class SerializedCheck:
    """Serialize the caller's potentially stateful resource callback."""
    def __init__(self, callback, lock=None):
        self.callback = callback
        self.lock = lock if lock is not None else threading.RLock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            return self.callback(*args, **kwargs)


class OneParentLookahead:
    """No eager input read, no retained previous parent, at most one future.

    The caller consumes each parent, admits prediction, optionally starts the
    next preparation, then discards the current parent before calling take again.
    Its admission callback must charge current RSS plus both the prediction and
    preparation headrooms. A failed admission leaves the next parent synchronous.
    Background exceptions are raised when that same next parent is requested.
    """
    def __init__(self, parents, load, *, resource_check, admit_overlap):
        self.parents = tuple(parents)
        if len(set(self.parents)) != len(self.parents):
            raise ValueError('One occurrence per held-out parent required')
        self.load = load
        self.check = resource_check
        self.admit_overlap = admit_overlap
        self.index = 0
        self.future = None
        self.future_parent = None
        self.stop = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='TreeLoom-observed-prefetch')
        self.closed = False
        self.prefetched = 0
        self.synchronous = 0
        self.deferred = 0

    def __enter__(self):
        return self

    def _background_check(self, *args, **kwargs):
        if self.stop.is_set():
            raise PrefetchCancelled('Next-parent preparation cancelled during owned cleanup')
        return self.check(*args, **kwargs)

    def _background_load(self, parent):
        self._background_check('TreeLoom_prefetch_before_observation')
        return self.load(parent, self._background_check)

    def take(self, parent):
        if self.closed or self.index >= len(self.parents) or parent != self.parents[self.index]:
            raise ValueError('Held-out parent requests must follow the original order')
        if self.future is None:
            value = self.load(parent, self.check)
            self.synchronous += 1
        else:
            if parent != self.future_parent:
                raise ValueError('Only the exact next parent may be prefetched')
            future = self.future
            self.future = None
            self.future_parent = None
            try:
                value = future.result()
            finally:
                del future
            self.prefetched += 1
        self.index += 1
        return value

    def start_next(self):
        if self.closed:
            raise ValueError('Closed parent prefetcher')
        if self.future is not None or self.index >= len(self.parents):
            return False
        if not self.admit_overlap():
            self.deferred += 1
            return False
        parent = self.parents[self.index]
        self.future_parent = parent
        self.future = self.executor.submit(self._background_load, parent)
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.future is not None:
            self.future.cancel()
        # Background scientific operations regularly call the cooperative check.
        # No detached worker or latent bank is allowed to survive this context.
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None
        self.future_parent = None

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False
