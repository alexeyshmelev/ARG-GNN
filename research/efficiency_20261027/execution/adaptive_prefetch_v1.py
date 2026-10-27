"""Separate two-parent candidate; no launch, production mutation or reduced workload.

Preparation and training retain their complete 128 GiB reservations. Optional
admission never waits. Each admitted producer additionally owns one conservative
family credit before it can start, so two encoder calls cannot race admission.
"""
from concurrent.futures import ThreadPoolExecutor, CancelledError
from contextlib import contextmanager
import gc
import threading
import time

from research.efficiency_20261027.integration_candidate.memory_v1 import LayoutMemoryGovernor
from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.pool import need
from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import pack_bundle, restore_bundle

CHILD_ALLOWANCE_BYTES = 8 * 2**30


def current_family_memory():
    """Fresh recursive PSS with RSS fallback; measurements are never cached."""
    import psutil
    process = psutil.Process()
    rows = []
    for child in [process] + process.children(recursive=True):
        try:
            memory = child.memory_full_info()
            rows.append(dict(pid=child.pid, bytes=getattr(memory, 'pss', memory.rss),
                             measure='PSS' if hasattr(memory, 'pss') else 'RSS'))
        except psutil.NoSuchProcess:
            continue
    return dict(bytes=sum(row['bytes'] for row in rows), processes=rows,
                recursive_children=max(0, len(rows)-1))


class EncoderCredits:
    """Credits include future child growth and remain until preparation returns.

The encoder's original ProcessPoolExecutor context has reaped its own children
before its call returns. Holding the credit through packing is more conservative.
No operation here waits for a different phase or credit.
"""
    def __init__(self, *, maximum_bytes, family_memory=current_family_memory,
                 allowance_bytes=CHILD_ALLOWANCE_BYTES):
        need(type(maximum_bytes) is int and maximum_bytes > 0, 'Finite family cap required')
        need(type(allowance_bytes) is int and allowance_bytes > 0, 'Positive explicit child allowance')
        self.maximum = maximum_bytes
        self.allowance = allowance_bytes
        self.measure = family_memory
        self.lock = threading.RLock()
        self.active = {}
        self.local = threading.local()
        self.peak_credits = 0
        self.peak_family_bytes = 0
        self.peak_children = 0
        self.encoder_calls = 0

    def _measure(self):
        measured = self.measure()
        need(type(measured['bytes']) is int and measured['bytes'] >= 0,
             'Live process-family memory required')
        self.peak_family_bytes = max(self.peak_family_bytes, measured['bytes'])
        self.peak_children = max(self.peak_children, measured['recursive_children'])
        return measured

    def acquire(self, parent):
        with self.lock:
            measured = self._measure()
            if measured['bytes'] + (len(self.active)+1)*self.allowance > self.maximum:
                return None
            need(parent not in self.active.values() and len(self.active) < 2,
                 'At most two distinct admitted parent child credits')
            token = object()
            self.active[token] = parent
            self.peak_credits = max(self.peak_credits, len(self.active))
            return token

    def release(self, token):
        with self.lock:
            need(token in self.active, 'Release exact owned child credit once')
            del self.active[token]

    @contextmanager
    def owner(self, token):
        need(getattr(self.local, 'token', None) is None, 'One producer credit per thread')
        with self.lock:
            need(token in self.active, 'Admitted producer credit required')
        self.local.token = token
        try:
            yield
        finally:
            del self.local.token

    def before_encoder(self):
        with self.lock:
            token = getattr(self.local, 'token', None)
            need(token in self.active, 'Encoder must run inside its admitted parent credit')
            measured = self._measure()
            need(measured['bytes'] + len(self.active)*self.allowance <= self.maximum,
                 'Live family plus every outstanding child allowance exceeds task limit')
            self.encoder_calls += 1
            return self.active[token]

    def snapshot(self):
        with self.lock:
            return dict(active_child_credits=len(self.active), peak_child_credits=self.peak_credits,
                child_allowance_bytes=self.allowance, maximum_encoder_children=8,
                measured_peak_recursive_children=self.peak_children,
                measured_peak_family_bytes=self.peak_family_bytes, encoder_calls=self.encoder_calls,
                measurement_is_sampled_not_continuous=True)


class AdaptiveMemoryGovernor(LayoutMemoryGovernor):
    """Original actual-RSS/layout semantics with foreground-priority admission."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.foreground_waiters = 0
        self.optional_deferrals = 0
        self.labels = {}

    def _available(self, amount, *, evict):
        while True:
            self.check('adaptive_governor_admission')
            if self.rss() + amount + sum(self.active.values()) <= self.maximum:
                return True
            if not evict:
                return False
            self.eviction_checks += 1
            if self.cache is not None and self.cache.evict_one_unleased():
                gc.collect()
                continue
            gc.collect()
            return self.rss() + amount + sum(self.active.values()) <= self.maximum

    def acquire(self, label, amount=None, *, optional=False, credits=None, parent=None):
        amount = self.reserve if amount is None else amount
        need(type(amount) is int and amount >= 0, 'Finite nonnegative phase reserve')
        owner = threading.get_ident()
        with self.condition:
            if not optional:
                self.foreground_waiters += 1
            try:
                while True:
                    need(not self.closed, 'Memory governor closed')
                    available = (not optional or self.foreground_waiters == 0)
                    available = available and self._available(amount, evict=not optional)
                    credit = None
                    if available and credits is not None:
                        credit = credits.acquire(parent)
                        available = credit is not None
                    if available:
                        token = object()
                        self.active[token] = amount
                        self.owners[token] = owner
                        self.labels[token] = label
                        self.admissions += 1
                        self.peak_reserved = max(self.peak_reserved, sum(self.active.values()))
                        return token, credit
                    if optional:
                        self.optional_deferrals += 1
                        return None
                    other = any(value != owner for value in self.owners.values())
                    need(other, 'Complete foreground reserve unavailable after actual-RSS eviction; '
                                'no wait on own nested reservation or reduced workload')
                    self.waits += 1
                    self.condition.wait(timeout=1.)
            finally:
                if not optional:
                    self.foreground_waiters -= 1
                    self.condition.notify_all()

    def transfer(self, token):
        with self.condition:
            need(token in self.active, 'Owned producer reservation required')
            self.owners[token] = threading.get_ident()

    def release(self, token, *, credits=None, credit=None):
        with self.condition:
            if credits is not None:
                credits.release(credit)
            need(token in self.active, 'Release exact phase once')
            del self.active[token], self.owners[token], self.labels[token]
            self.condition.notify_all()

    @contextmanager
    def phase(self, label, required_bytes=None):
        token, _ = self.acquire(label, required_bytes)
        try:
            yield
        finally:
            self.release(token)

    def snapshot(self):
        with self.condition:
            return dict(super().snapshot(), foreground_waiters=self.foreground_waiters,
                optional_deferrals=self.optional_deferrals,
                active_labels=list(self.labels.values()), optional_admission_never_waits=True)


class AdaptiveCompactParents:
    """One frozen schedule, one compact pool, one future per attempted parent.

Only the coordinator invokes load/refill/training/advance. Producer callbacks
may commit readiness, but only durable_checkpoint advances the schedule cursor.
Future.result is called outside pool, governor, event and resource-check locks.
"""
    def __init__(self, *, schedule, start_cursor, parent_bounds, input_refs,
                 external_arrays, max_cache_bytes, governor, credits, resource_check,
                 event, prepare, pack=pack_bundle, restore=restore_bundle):
        need(sum(parent_bounds.values()) <= max_cache_bytes, 'All authenticated compact bounds must fit')
        need(max_cache_bytes + governor.reserve < governor.maximum, 'Original complete working reserve')
        self.schedule = tuple(schedule)
        need(self.schedule and set(self.schedule) <= set(parent_bounds), 'Registered fixed schedule required')
        need(type(start_cursor) is int and 0 <= start_cursor < len(self.schedule), 'Exact unfinished cursor')
        need(governor.maximum == credits.maximum, 'Same main and process-family cap required')
        self.cursor = start_cursor
        self.bounds = dict(parent_bounds)
        self.inputs = input_refs
        self.external = external_arrays
        self.maximum = max_cache_bytes
        self.governor = governor
        self.credits = credits
        self.check = resource_check
        self.event = event
        self.prepare, self.pack, self.restore = prepare, pack, restore
        self.lock = threading.RLock()
        self.owner = threading.get_ident()
        self.stop = threading.Event()
        self.closed = False
        self.error = None
        self.records, self.futures, self.reservations = {}, {}, {}
        self.states = {parent: 'absent' for parent in self.bounds}
        self.pending = {}
        self.preparations, self.restorations = {}, {}
        self.charged = 0
        self.loaded_cursor = None
        self.training = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='TreeLoom_adaptive_prepare')

    def _coordinator(self):
        need(threading.get_ident() == self.owner and not self.closed, 'Live owning coordinator required')
        self.raise_if_failed()

    def raise_if_failed(self):
        # Never inspect/wait on a Future from a resource callback or governor.
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError('An attempted parent failed; retry is forbidden') from error

    def _reap(self):
        with self.lock:
            completed = [(parent, future) for parent, future in self.futures.items() if future.done()]
        for parent, future in completed:
            need(future.result() == parent, 'Completed future must retain its exact parent')
        self.raise_if_failed()

    def _producer_check(self, stage):
        need(not self.stop.is_set(), 'Cooperative parent producer stop')
        self.raise_if_failed()
        self.check(stage)

    def _produce(self, parent, token, credit):
        try:
            self.governor.transfer(token)
            with self.lock:
                need(self.states[parent] == 'queued', 'Exactly one queued owner before preparation')
                self.states[parent] = 'preparing'
                self.preparations[parent] = 1
            self._producer_check('adaptive_parent_start')
            with self.credits.owner(credit):
                bundle, metadata = self.prepare(parent, resource_check=self._producer_check)
                record = self.pack(bundle, metadata, input_refs=self.inputs[parent],
                    external_arrays=self.external.get(parent), resource_check=self._producer_check)
            need(record['charged_bytes'] <= self.bounds[parent], 'Actual compact charge exceeds sealed bound')
            del bundle, metadata
            gc.collect()
            self._producer_check('adaptive_parent_commit')
            with self.lock:
                need(self.states[parent] == 'preparing', 'Single complete preparation commit')
                self.records[parent] = record
                self.charged += record['charged_bytes']
                del self.pending[parent]
                need(self.charged + sum(self.pending.values()) <= self.maximum, 'Atomic compact charge cap')
                self.states[parent] = 'ready'
                detail = dict(parent_id=parent, parents=len(self.records),
                    compact_charged_bytes=record['charged_bytes'], pool_charged_bytes=self.charged,
                    disk_cache_bytes=0, expanded_bank_sent_over_IPC=False)
            self.event('compact_parent_ready', detail)
            return parent
        except BaseException as error:
            with self.lock:
                self.states[parent] = 'failed'
                self.pending.pop(parent, None)
                if self.error is None:
                    self.error = error
            self.stop.set()
            raise
        finally:
            self.governor.release(token, credits=self.credits, credit=credit)

    def _submit(self, parent, *, optional):
        self._coordinator()
        with self.lock:
            if self.states[parent] != 'absent':
                return False
            if sum(not future.done() for future in self.futures.values()) >= 2:
                return False
        reservation = self.governor.acquire('prepare_and_lossless_pack', optional=optional,
                                             credits=self.credits, parent=parent)
        if reservation is None:
            return False
        token, credit = reservation
        try:
            with self.lock:
                need(not self.closed and self.states[parent] == 'absent', 'Once-only parent claim')
                need(self.charged + sum(self.pending.values()) + self.bounds[parent] <= self.maximum,
                     'Pending authenticated compact charges must fit before submission')
                self.states[parent] = 'queued'
                self.pending[parent] = self.bounds[parent]
                self.reservations[parent] = reservation
                # A producer can start but must wait for this short state lock.
                self.futures[parent] = self.executor.submit(self._produce, parent, token, credit)
        except BaseException as error:
            with self.lock:
                self.states[parent] = 'failed'
                self.pending.pop(parent, None)
                self.error = error
            self.stop.set()
            self.governor.release(token, credits=self.credits, credit=credit)
            raise
        self.event('observation_prefetch_started', dict(parent_id=parent, CPU_workers=2,
                   optional=optional, cursor=self.cursor, maximum_encoder_children=8))
        return True

    def refill(self):
        self._coordinator()
        self._reap()
        # Foreground training must already own its reservation before speculation.
        need(self.training, 'Refill only after foreground training admission')
        submitted = []
        for parent in dict.fromkeys(self.schedule[self.cursor+1:]):
            with self.lock:
                state = self.states[parent]
                occupied = sum(not future.done() for future in self.futures.values())
            if state != 'absent':
                continue
            if occupied >= 2 or not self._submit(parent, optional=True):
                break
            submitted.append(parent)
        return submitted

    def load(self, parent):
        self._coordinator()
        self._reap()
        need(not self.training and self.cursor < len(self.schedule)
             and parent == self.schedule[self.cursor], 'Exact next scheduled parent required')
        with self.lock:
            state = self.states[parent]
        if state == 'absent':
            need(self._submit(parent, optional=False), 'Required parent admission cannot be skipped')
        with self.lock:
            future = self.futures[parent]
        need(future.result() == parent, 'Wait only for demanded exact parent outside every lock')
        self._reap()
        with self.lock:
            record = self.records[parent]
        with self.governor.phase('exact_restore', record['read_peak_upper_bound_bytes']):
            bundle, metadata = self.restore(record, input_refs=self.inputs[parent], resource_check=self.check)
        with self.lock:
            self.restorations[parent] = self.restorations.get(parent, 0)+1
            self.loaded_cursor = self.cursor
        return bundle, metadata

    @contextmanager
    def training_phase(self):
        self._coordinator()
        need(self.loaded_cursor == self.cursor and not self.training, 'Load before training reservation')
        with self.governor.phase('complete_parent_training'):
            self.training = True
            try:
                self.refill()
                yield
                self._reap()
            finally:
                self.training = False

    def durable_checkpoint(self, *, parent, completed_updates):
        self._coordinator()
        need(self.training and self.loaded_cursor == self.cursor
             and parent == self.schedule[self.cursor] and completed_updates == self.cursor+1,
             'Only exact durable complete-parent checkpoint advances the schedule')
        self.cursor = completed_updates
        self.loaded_cursor = None

    def receipt(self):
        with self.lock:
            result = dict(schema='TreeLoom_adaptive_two_parent_candidate_v1', cursor=self.cursor,
                states=dict(self.states), preparations=dict(self.preparations),
                restorations=dict(self.restorations), compact_charged_bytes=self.charged,
                pending_bound_bytes=sum(self.pending.values()), cap_bytes=self.maximum,
                CPU_prefetch_workers=2, lookahead_parents=2, maximum_encoder_children=8,
                disk_cache_bytes=0, compact_evictions=0, cold_rebuilds=0,
                expanded_tensor_IPC=False, production_qualified=False)
        result['governor'] = self.governor.snapshot()
        result['encoder_credits'] = self.credits.snapshot()
        return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.stop.set()
            futures = list(self.futures.items())
        self.governor.close()
        for parent, future in futures:
            if future.cancel():
                token, credit = self.reservations[parent]
                self.governor.release(token, credits=self.credits, credit=credit)
                with self.lock:
                    self.states[parent] = 'failed'
                    self.pending.pop(parent, None)
                    if self.error is None:
                        self.error = CancelledError('Queued producer cancelled during close')
        self.executor.shutdown(wait=True, cancel_futures=False)
        need(not self.governor.active and not self.credits.active,
             'Every producer and child credit must be joined/released before close returns')
