"""Small spawn-only, one-task-per-worker transport; no Torch or scientific imports."""
from __future__ import annotations
import multiprocessing as mp
import time


def _start_ticks(pid):
    with open("/proc/%d/stat" % pid) as stream:
        raw = stream.read(8192)
    return int(raw[raw.rfind(")")+2:].split()[19])
import traceback


def _child(connection, opener, kwargs):
    reader = None
    try:
        reader = opener(**kwargs)
        connection.send(("ready",))
        while True:
            request = connection.recv()
            if request == ("stop",):
                connection.send(("stopped",))
                break
            command, serial, index = request
            if command != "read":
                raise ValueError("Unknown worker command")
            try:
                value = reader(index)
                connection.send(("ok", serial, index, value))
                del value
            except BaseException as error:
                connection.send(("error", serial, index, type(error).__name__,
                                 str(error)[:2000], traceback.format_exc()[-6000:]))
                break
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send(("startup_error", type(error).__name__, str(error)[:2000]))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if reader is not None and callable(getattr(reader, "close", None)):
            reader.close()
        connection.close()


class BoundedOrderedWorkersV1:
    """No feeder queue, background parent thread, retry or unbounded submission."""
    def __init__(self, *, count, workers, opener, opener_kwargs, check,
                 task_timeout_seconds, drain_timeout_seconds):
        if not (type(count) is int and count > 0 and type(workers) is int and 1 <= workers <= 4
                and callable(opener) and callable(check)
                and 0 < task_timeout_seconds <= 3600 and 0 < drain_timeout_seconds <= 30):
            raise ValueError("Explicit bounded worker/task/drain settings required")
        self.count, self.limit = count, min(count, workers)
        self.opener, self.kwargs, self.check = opener, opener_kwargs, check
        self.task_timeout, self.drain_timeout = task_timeout_seconds, drain_timeout_seconds
        self.slots, self.pending, self.serial = [], {}, 0
        self.closed, self.last_drain = False, None
        self.dispatches, self.deliveries, self.discards = [], [], []

    def start(self):
        if self.closed:
            raise RuntimeError("Closed worker pool cannot be reused")
        if self.slots:
            return
        context = mp.get_context("spawn")
        try:
            for _ in range(self.limit):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_child, args=(child, self.opener, self.kwargs),
                                          daemon=False)
                try:
                    process.start()
                except BaseException:
                    parent.close(); child.close()
                    raise
                child.close()
                self.slots.append(dict(connection=parent, process=process, task=None,
                                       start_ticks=None))
                self.slots[-1]["start_ticks"] = _start_ticks(process.pid)
            for slot in self.slots:
                if self._receive(slot, time.monotonic() + self.task_timeout) != ("ready",):
                    raise RuntimeError("CPU observation worker did not become ready")
        except BaseException:
            self.close(failed=True)
            raise

    def pids(self):
        return [s["process"].pid for s in self.slots]

    def live_identities(self):
        return [(s["process"].pid, s["start_ticks"]) for s in self.slots if s["process"].is_alive()]

    def _receive(self, slot, deadline):
        while time.monotonic() < deadline:
            self.check("prefetch_wait", self.live_identities())
            if slot["connection"].poll(min(.1, max(0., deadline-time.monotonic()))):
                try:
                    return slot["connection"].recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError("CPU worker transport closed") from error
            if not slot["process"].is_alive():
                raise RuntimeError("CPU worker exited without its ordered reply")
        raise TimeoutError("Bounded observation worker deadline exceeded")

    def prefetch(self, index, direction, skip=lambda index: False):
        self.start()
        if direction not in (-1, 1):
            raise ValueError("Explicit physical direction required")
        for slot in self.slots:
            if slot["task"] is not None:
                continue
            while 0 <= index < self.count and (index in self.pending or skip(index)):
                index += direction
            if not 0 <= index < self.count:
                return
            self.check("prefetch_dispatch", self.live_identities())
            self.serial += 1
            task = (self.serial, index, time.monotonic() + self.task_timeout)
            slot["connection"].send(("read", task[0], index))
            slot["task"] = task
            self.pending[index] = slot
            self.dispatches.append(index)
            index += direction

    def _take(self, index, discarded=False):
        slot = self.pending[index]
        serial, physical, deadline = slot["task"]
        reply = self._receive(slot, deadline)
        if reply[0] != "ok" or reply[1:3] != (serial, physical):
            raise RuntimeError("CPU worker failure or physical identity drift: " + repr(reply[:3]) + ("; " + repr(reply[3:])[:3000] if reply[0] != "ok" else ""))
        del self.pending[index]
        slot["task"] = None
        (self.discards if discarded else self.deliveries).append(index)
        return reply[3]

    def take(self, index, direction, skip=lambda index: False):
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError("Physical index outside complete parent")
        try:
            self.start()
            if index not in self.pending:
                # Unexpected checkpoint/reverse demand is exact indexed access,
                # not a replacement, omitted block or out-of-order yield.
                for old in list(self.pending):
                    self._take(old, discarded=True)
            self.prefetch(index, direction, lambda candidate: candidate != index and skip(candidate))
            return self._take(index)
        except BaseException:
            self.close(failed=True)
            raise

    def close(self, *, failed=False):
        if self.closed:
            return self.last_drain
        self.closed = True
        forced, errors = [], []
        deadline = time.monotonic() + self.drain_timeout
        if not failed:
            try:
                for index in list(self.pending):
                    slot = self.pending[index]
                    serial, physical, task_deadline = slot["task"]
                    reply = self._receive(slot, min(deadline, task_deadline))
                    if reply[0] != "ok" or reply[1:3] != (serial, physical):
                        raise RuntimeError("Prefetched task failed during release")
                    self.discards.append(index)
                    slot["task"] = None
                self.pending.clear()
                for slot in self.slots:
                    slot["connection"].send(("stop",))
                for slot in self.slots:
                    if self._receive(slot, deadline) != ("stopped",):
                        raise RuntimeError("Missing CPU worker stop acknowledgement")
            except BaseException as error:
                errors.append(type(error).__name__ + ": " + str(error)[:1000])
        for slot in self.slots:
            process = slot["process"]
            process.join(timeout=max(0., deadline-time.monotonic()) if not failed and not errors else 0.)
            if process.is_alive():
                process.terminate()
                forced.append(process.pid)
        for slot in self.slots:
            process = slot["process"]
            process.join(timeout=2.)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.)
            slot["connection"].close()
        self.pending.clear()
        self.last_drain = dict(drained=not any(s["process"].is_alive() for s in self.slots),
                               failed=bool(failed or errors or forced
                                           or any(s["process"].exitcode != 0 for s in self.slots)), forced_worker_pids=forced,
                               worker_pids=self.pids(),
                               exitcodes=[s["process"].exitcode for s in self.slots],
                               errors=errors, max_inflight=self.limit,
                               dispatch_count=len(self.dispatches),
                               delivered_count=len(self.deliveries),
                               discarded_prefetch_count=len(self.discards))
        if not self.last_drain["drained"]:
            raise RuntimeError("Owned observation worker failed to drain")
        for slot in self.slots:
            slot["process"].close()
        self.slots.clear()
        return self.last_drain
