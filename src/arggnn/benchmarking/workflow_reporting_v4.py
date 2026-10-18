"""Continuation reporting isolated from a draining predecessor publisher.

Reuse the frozen metric formatter in a private module instance. Both publishers
share the same outer lock, but own different report paths, caches and README
markers. No running predecessor source or module object is modified.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import time

from . import contracts as c

BEGIN = "<!-- BEGIN CONTINUATION BENCHMARK WORKFLOW -->"
END = "<!-- END CONTINUATION BENCHMARK WORKFLOW -->"
STATUS_PATH = "research/benchmark_continuation_current.json"
SCORE_PATH = "research/continuation_matched_benchmark_results.json"

_spec = importlib.util.spec_from_file_location(
    "arggnn.benchmarking._continuation_reporting_core",
    c.ROOT / "src/arggnn/benchmarking/workflow_reporting.py")
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
_core.BEGIN, _core.END = BEGIN, END
_core.STATUS_PATH, _core.SCORE_PATH = STATUS_PATH, SCORE_PATH
_core.REPORT_PATHS = ("README.md", STATUS_PATH, SCORE_PATH)
_core.RUNTIME_PATH = ".runtime/benchmarking/workflow_reporting_v4"
_original_markdown = _core._markdown


def _markdown(report, results):
    return _original_markdown(report, results).replace(
        "## Executable benchmark workflow", "## Active benchmark continuation", 1)


_core._markdown = _markdown


def publish(plan_ref, state_path, force=False):
    """Write v4-only reports under the lock respected by the original writer."""
    shared = c.contained_path(".runtime/benchmarking/workflow_reporting/reporting.lock")
    shared.parent.mkdir(parents=True, exist_ok=True)
    if shared.is_symlink():
        raise ValueError("Shared reporting lock must not be a symlink")
    with shared.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="busy")
        return _core.publish(plan_ref, state_path, force=force)


class ProgressReporter:
    """Publish asynchronously so Git cannot block scientific supervision."""
    def __init__(self, plan, ref, state_path):
        self.enabled = plan.get("publish_progress") is True
        self.ref, self.state_path = ref, str(state_path)
        self.executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="continuation_report")
                         if self.enabled else None)
        self.future = None
        self.last_submit = 0.0
        self.last_event = None
        self.score_ids = {task["id"] for task in plan["tasks"] if
            str(task.get("factory_arguments", {}).get("stage", "")).endswith("score")
            or str(task.get("factory_module", "")).endswith("_scoring")}

    def submit(self, state, *, final=False):
        if not self.enabled:
            return
        try:
            if self.future is not None and not self.future.done():
                return
            if self.future is not None:
                self.future.result()
            event = (state.get("complete", False), tuple(sorted(
                (key, row["status"]) for key, row in state["tasks"].items()
                if row["status"] in ("failed", "unknown")
                or (key in self.score_ids and row["status"] == "succeeded"))))
            now = time.monotonic()
            if not final and event == self.last_event and now - self.last_submit < 60:
                return
            self.future = self.executor.submit(publish, self.ref, self.state_path)
            self.last_event, self.last_submit = event, now
        except Exception as exc:
            _core._record_error(exc, time.time())

    def close(self, state):
        if self.executor is not None:
            try:
                self.executor.submit(publish, self.ref, self.state_path)
            except Exception as exc:
                _core._record_error(exc, time.time())
            self.executor.shutdown(wait=False)
