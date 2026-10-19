"""Bounded latest-state reporting; no signals, subprocesses or CUDA queries."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json
import os
import resource
import sys
import threading
import time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")


class StageProgress:
    def __init__(self, path, *, parent_id, interval_seconds=15):
        self.path = Path(path).resolve()
        if not self.path.is_relative_to(ROOT) or self.path.is_symlink():
            raise ValueError("Status must remain inside the admitted project output")
        if not self.path.parent.is_dir() or not 5 <= interval_seconds <= 60:
            raise ValueError("Existing admitted output directory and finite interval required")
        self.parent_id = str(parent_id)
        self.interval = interval_seconds
        self.started = time.monotonic()
        self.main = threading.get_ident()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.active = []
        self.timings = {}
        self.marks = Counter()
        self.latest = {}
        self.error = None
        self.thread = threading.Thread(target=self._loop, name="TreeLoom stage status", daemon=True)
        self.thread.start()

    @contextmanager
    def stage(self, label):
        started = time.monotonic()
        with self.lock:
            if label not in self.timings:
                if len(self.timings) >= 48:
                    raise ValueError("At most 48 named stages")
                self.timings[label] = dict(calls=0, total_seconds=0., max_seconds=0.)
            self.active.append((label, started))
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            with self.lock:
                row = self.timings[label]
                row["calls"] += 1
                row["total_seconds"] += elapsed
                row["max_seconds"] = max(row["max_seconds"], elapsed)
                self.active.pop()

    def mark(self, label, **values):
        with self.lock:
            if label not in self.marks and len(self.marks) >= 64:
                raise ValueError("At most 64 fixed progress marks")
            self.marks[label] += 1
            self.latest.update(values)
            self.latest["last_mark"] = label
            self.latest["last_mark_elapsed_seconds"] = time.monotonic() - self.started
            if self.error is not None:
                raise RuntimeError("Live status reporting failed: " + self.error)

    def snapshot(self):
        frames, frame = [], sys._current_frames().get(self.main)
        while frame is not None and len(frames) < 12:
            frames.append(dict(file=frame.f_code.co_filename[-180:], line=frame.f_lineno,
                               function=frame.f_code.co_name[:80]))
            frame = frame.f_back
        with self.lock:
            now = time.monotonic()
            return dict(schema="treeloom_stage_progress_v1", parent_id=self.parent_id,
                elapsed_seconds=now-self.started, latest=dict(self.latest),
                active_stages=[dict(stage=name, elapsed_seconds=now-started)
                               for name, started in self.active],
                stage_timings={key: dict(value) for key, value in self.timings.items()},
                mark_counts=dict(self.marks), main_thread_stack=frames,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                reporting_error=self.error)

    def flush(self):
        data = (json.dumps(self.snapshot(), sort_keys=True, allow_nan=False)+"\n").encode()
        if len(data) > 65536:
            raise ValueError("Live status exceeds 64 KiB bound")
        temporary = self.path.with_name(self.path.name+".next")
        descriptor = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.flush()
            except Exception as error:
                with self.lock:
                    self.error = type(error).__name__+": "+str(error)[:300]
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError("Stage reporting thread did not stop")
        self.flush()
        if self.error is not None:
            raise RuntimeError("Live status reporting failed: "+self.error)
