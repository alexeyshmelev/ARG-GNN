"""Current repaired queue publisher, isolated from two draining predecessors."""
import importlib.util
from . import contracts as c

_spec = importlib.util.spec_from_file_location(
    "arggnn.benchmarking._reporting_v5_isolated",
    c.ROOT / "src/arggnn/benchmarking/workflow_reporting_v4.py")
_private = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_private)
BEGIN = "<!-- BEGIN REPAIRED BENCHMARK WORKFLOW -->"
END = "<!-- END REPAIRED BENCHMARK WORKFLOW -->"
STATUS_PATH = "research/benchmark_repaired_current.json"
SCORE_PATH = "research/repaired_matched_benchmark_results.json"
_private.BEGIN = _private._core.BEGIN = BEGIN
_private.END = _private._core.END = END
_private.STATUS_PATH = _private._core.STATUS_PATH = STATUS_PATH
_private.SCORE_PATH = _private._core.SCORE_PATH = SCORE_PATH
_private._core.REPORT_PATHS = ("README.md", STATUS_PATH, SCORE_PATH)
_private._core.RUNTIME_PATH = ".runtime/benchmarking/workflow_reporting_v5"
publish = _private.publish
ProgressReporter = _private.ProgressReporter

_original_snapshot = _private._core._snapshot
_original_markdown = _private._core._markdown


def _snapshot(plan_ref, state_path):
    report, results = _original_snapshot(plan_ref, state_path)
    from . import workflow_predecessor_v5 as predecessors
    plan = c.read_ref(plan_ref)
    report["preserved_workflow_jobs"] = []
    for binding in plan.get("workflow_external_predecessors", []):
        task, entry = predecessors._source(binding)
        report["preserved_workflow_jobs"].append(dict(
            id=binding["source_task_id"], resource=task["resource"],
            status=entry["status"], source_state_path=binding["source_state_path"],
            original_job_ref=binding["original_job_ref"]))
    return report, results


def _markdown(report, results):
    block = _original_markdown(report, results)
    running = [row for row in report.get("preserved_workflow_jobs", []) if row["status"] == "running"]
    if running:
        gpu = sum(row["resource"] == "gpu0" for row in running)
        cpu = sum(row["resource"] == "cpu" for row in running)
        note = ("Preserved jobs are still running under their original supervisors: "
                + str(gpu) + " on GPU 0 and " + str(cpu) + " on CPU. "
                "Their outputs feed this queue after genuine completion.\n\n")
        block = block.replace("Running workflow jobs:", note + "Running workflow jobs:", 1)
    return block


_private._core._snapshot = _snapshot
_private._core._markdown = _markdown
