"""Compact current-workflow reports; called by ROOT, never a daemon.

Only named bounded metadata is read. Reporting failures are isolated from the
scientific controller. Git publishes only the three report paths below.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from . import contracts as c

BEGIN = "<!-- BEGIN EXECUTABLE BENCHMARK WORKFLOW -->"
END = "<!-- END EXECUTABLE BENCHMARK WORKFLOW -->"
STATUS_PATH = "research/benchmark_execution_current.json"
SCORE_PATH = "research/current_matched_benchmark_results.json"
REPORT_PATHS = ("README.md", STATUS_PATH, SCORE_PATH)
RUNTIME_PATH = ".runtime/benchmarking/workflow_reporting"
INTERVAL_SECONDS = 30 * 60
STATUSES = ("succeeded", "failed", "running", "pending", "blocked", "unknown")
SCORE_SCHEMA = "arggnn_current_matched_benchmark_results_v1"
AMERICAN = "american_fresh_all15_common_score_v1"
EUROPEAN = "ap_paper_native6_paired_comparison_v1"
C4 = "european_release_C4_paired_comparison_v1"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _utc(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat()


def _text(value, limit=500):
    return str(value).replace("\n", " ").replace("\r", " ")[:limit]


def _save(path, value):
    path = c.contained_path(path)
    _need(not path.is_symlink(), "Report path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text() == raw:
        return False
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    try:
        with temporary.open("x") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _score_task(task):
    arguments = task.get("factory_arguments", {})
    return (str(arguments.get("stage", "")).endswith("score")
            or str(task.get("id", "")).endswith("_score")
            or str(task.get("factory_function", "")).startswith("score")
            or str(task.get("factory_module", "")).endswith("_scoring"))


def _summary_reference(value):
    if isinstance(value, dict) and set(value) == {"path", "sha256"}:
        return value
    if isinstance(value, dict):
        for key in ("summary_ref", "comparison_ref", "score_ref"):
            candidate = value.get(key)
            if isinstance(candidate, dict) and set(candidate) == {"path", "sha256"}:
                return candidate
    raise ValueError("Successful score task has no exact summary reference")


def _aggregate(value, *, require_unit=True):
    """Copy exact aggregate scalars only; never rows, truth, or per-parent arrays."""
    allowed = {"mean", "sample_SD", "parents", "independent_parents", "unit", "status",
               "mean_percent", "sample_SD_percentage_points", "direction"}
    required = {"mean", "sample_SD"} | ({"unit"} if require_unit else set())
    _need(isinstance(value, dict) and required <= set(value),
          "Known aggregate schema required")
    result = {key: item for key, item in value.items() if key in allowed}
    _need(all(item is None or type(item) in (str, int, float, bool) for item in result.values()),
          "Only aggregate scalars may be reported")
    for key in ("mean", "sample_SD"):
        item = result[key]
        _need(item is None or (type(item) in (int, float) and math.isfinite(item)),
              "Finite aggregate or explicitly unavailable metric required")
    return result


def _compact(ref):
    summary = c.read_ref(ref)
    schema = summary.get("schema")
    if schema == AMERICAN:
        _need(summary.get("status") == "completed_shared15parent_paired_scores"
              and summary.get("all_predictions_sealed_before_any_heldout_truth") is True,
              "Completed all-method held-out barrier required")
        allowed_metrics = {"accuracy", "macro_F1", "NLL", "Brier"} | {
            ancestry + "_" + metric for ancestry in ("EUR", "EAS", "AFR")
            for metric in ("precision", "recall", "F1")}
        aggregates = {}
        for split, methods in summary["equal_parent_summaries"].items():
            _need(split in ("validation", "test", "all_evaluation"), "Known American split required")
            aggregates[split] = {}
            for method, metrics in methods.items():
                _need(method in ("treeloom", "argformer", "flare"), "Known American method required")
                _need(method == "treeloom" or not ({"NLL", "Brier"} & set(metrics)),
                      "Hard-call transports cannot supply posterior metrics")
                aggregates[split][method] = {name: _aggregate(value) for name, value in metrics.items()
                                              if name in allowed_metrics}
        deltas = {split: {pair: {metric: _aggregate(value) for metric, value in metrics.items()
                                  if metric in allowed_metrics}
                           for pair, metrics in pairs.items()}
                  for split, pairs in summary.get("paired_deltas", {}).items()}
        result = {key: summary[key] for key in (
            "status", "parents", "validation_parents", "test_parents", "records_per_parent",
            "records_per_method", "all_predictions_sealed_before_any_heldout_truth",
            "probabilities_modified", "probability_floor", "aggregation",
            "native_one_hot_transports_excluded_from_NLL_Brier", "exact_paper_reproduction")
                  if key in summary}
        result.update(equal_parent_summaries=aggregates, paired_deltas=deltas,
                      per_parent_values_omitted=True)
    elif schema == EUROPEAN:
        _need(summary.get("all_method_all_parent_prediction_barrier_before_truth") is True,
              "Completed native-six held-out barrier required")
        aggregates = {}
        for method, groups in summary["summary"].items():
            _need(method in ("ancestralpaths", "treeloom"), "Known native-six method required")
            aggregates[method] = {group: {metric: _aggregate(value) for metric, value in metrics.items()
                                         if metric in ("accuracy", "macro_F1", "Brier", "raw_NLL")}
                                  for group, metrics in groups.items()}
        result = {key: summary[key] for key in (
            "records_per_parent", "all_method_all_parent_prediction_barrier_before_truth",
            "uncertainty", "endpoint", "published_CNN_reproduction", "comparison_scope")
                  if key in summary}
        result.update(summary=aggregates, paired_accuracy_differences={
            group: _aggregate(value) for group, value in summary.get("paired_accuracy_differences", {}).items()},
            per_parent_values_omitted=True)
    elif schema == C4:
        _need(summary.get("all_registered_endpoint_predictions_before_any_TEST_truth") is True,
              "Completed cross-endpoint TEST prediction barrier required")
        _need(set(summary["summary"]) == {"argmix", "treeloom"}
              and summary.get("source_class_order") == ["Ana", "CHG", "WHG", "EHG"],
              "Known C4 methods and source class order required")
        metrics = ("accuracy", "macro_F1", "Brier", "raw_NLL")
        aggregates = {}
        for method, groups in summary["summary"].items():
            _need("overall" in groups, "Actual overall C4 aggregates required")
            aggregates[method] = {}
            for group, values in groups.items():
                _need(set(values) == set(metrics)
                      and all(value.get("independent_parents") == 5 for value in values.values()),
                      "Complete five-parent C4 metric aggregates required")
                aggregates[method][group] = {name: _aggregate(values[name], require_unit=False)
                                             for name in metrics}
        result = {key: summary[key] for key in (
            "source_class_order", "all_registered_endpoint_predictions_before_any_TEST_truth",
            "records_per_parent", "uncertainty", "separate_genuine_four_head_TreeLoom_training",
            "published_ARGMix_paper_reproduction", "scope") if key in summary}
        result.update(summary=aggregates, paired_accuracy_differences={
            group: _aggregate(value, require_unit=False)
            for group, value in summary.get("paired_accuracy_differences", {}).items()},
            per_parent_values_omitted=True,
            metric_units={"accuracy": "fraction", "macro_F1": "fraction",
                          "Brier": "summed_squared_probability_error", "raw_NLL": "nats_per_record"},
            paired_accuracy_difference_unit="fraction",
            unit_provenance="Known C4 metric definitions; source aggregate objects have no unit field")
    else:
        return {"schema": schema, "source_summary_ref": ref, "metrics_reported": False,
                "reason": "Summary schema is not supported by this reporter"}
    result.update(schema=schema, source_summary_ref=ref, metrics_reported=True)
    return result


def _external(plan):
    """Named queue observations stay separate from exact workflow task counts."""
    identities = {(row["queue_path"], row["job_id"]) for row in plan.get("external_resources", [])}
    identities.update((task["queue_path"], task["job_id"]) for task in plan["tasks"]
                      if task["resource"] == "external")
    queues, rows, errors = {}, [], []
    for path, job_id in sorted(identities):
        try:
            if path not in queues:
                queues[path] = c.read_json(path)
            matches = [job for job in queues[path]["jobs"] if job["admission"]["job_id"] == job_id]
            _need(len(matches) == 1, "Expected exactly one named external queue job")
            row = matches[0]
            rows.append(dict(job_id=job_id, queue_path=path, status=_text(row["status"], 80)))
        except Exception as exc:
            errors.append(dict(queue_path=path, job_id=job_id, error=type(exc).__name__ + ": " + _text(exc)))
    return rows, errors


def _snapshot(plan_ref, state_path):
    plan = c.read_ref(plan_ref)
    _need(plan.get("schema") == "arggnn_executable_workflow_v1" and plan.get("physical_gpu") == 0,
          "Authenticated GPU0 executable workflow required")
    state = c.read_json(state_path)
    _need(state.get("plan_ref") == plan_ref, "Workflow state belongs to another plan")
    tasks = plan["tasks"]
    _need(len({task["id"] for task in tasks}) == len(tasks)
          and {task["id"] for task in tasks} == set(state["tasks"]), "Exact workflow task set required")
    counts = Counter()
    stages, records, scores, score_errors = {}, [], {}, []
    for task in tasks:
        identity = task["id"]
        entry = state["tasks"][identity]
        status = entry["status"]
        _need(status in STATUSES, "Unknown workflow task status")
        counts[status] += 1
        stage = task.get("factory_arguments", {}).get("stage") or task.get("factory_function") or "external"
        stage = _text(stage, 120)
        stages.setdefault(stage, Counter())[status] += 1
        row = dict(id=identity, stage=stage, resource=task["resource"], status=status)
        for key in ("pid", "started_at", "finished_at", "actual_controller_wait", "owned_process_group_absent"):
            if key in entry:
                row[key] = entry[key]
        for key in ("error", "waiting_reason"):
            if entry.get(key):
                row[key] = _text(entry[key])
        records.append(row)
        if status == "succeeded" and _score_task(task):
            try:
                envelope = c.read_ref(entry["result_ref"])
                scores[identity] = _compact(_summary_reference(envelope["value"]))
            except Exception as exc:
                score_errors.append(dict(task_id=identity, error=type(exc).__name__ + ": " + _text(exc)))
    external, external_errors = _external(plan)
    complete = all(entry["status"] == "succeeded" for entry in state["tasks"].values())
    _need(not state.get("complete") or complete, "State completion disagrees with actual task statuses")
    report = dict(schema="arggnn_benchmark_execution_current_v1", plan_ref=plan_ref,
                  state_path=str(c.contained_path(state_path).relative_to(c.ROOT)),
                  total_tasks=len(tasks), counts={status: counts[status] for status in STATUSES},
                  stages={stage: {status: counts[status] for status in STATUSES}
                          for stage, counts in sorted(stages.items())},
                  tasks=records, running_job_names=[row["id"] for row in records if row["status"] == "running"],
                  complete=complete, run_status=state.get("run_status", "unspecified"),
                  external_queue_jobs=external, reporting_errors=score_errors + external_errors,
                  metric_policy="Only authenticated succeeded summaries with known schemas supply metrics")
    for key in ("active_cpu_lanes_including_external", "reserved_rss_bytes_including_external"):
        if key in state:
            report[key] = state[key]
    result = dict(schema=SCORE_SCHEMA, plan_ref=plan_ref, results=scores)
    return report, result


def _metric(value, unit=None):
    if value["mean"] is None:
        return "positive infinity" if value.get("status") == "positive_infinity" else "unavailable (see source)"
    sd = value["sample_SD"]
    spread = "unavailable" if sd is None else format(sd, ".6g")
    return format(value["mean"], ".6g") + " ± " + spread + " " + str(value.get("unit", unit or "unit unspecified"))


def _markdown(report, results):
    counts = report["counts"]
    lines = [BEGIN, "## Executable benchmark workflow", "",
             "Observed " + report["observed_utc"] + ". " + "; ".join(
                 str(counts[status]) + " " + status for status in STATUSES) + ".",
             "Run status: `" + _text(report["run_status"]) + "`. Physical GPU: 0.", "",
             "[Current queue details](" + STATUS_PATH + ") · [Authenticated matched scores](" + SCORE_PATH + ")", ""]
    lines += ["| Stage | Succeeded | Failed | Running | Pending | Blocked | Unknown |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage, values in report["stages"].items():
        lines.append("| " + stage.replace("|", chr(92) + "|") + " | " + " | ".join(str(values[s]) for s in STATUSES) + " |")
    lines += ["", "Running workflow jobs: " + (", ".join("`" + name + "`" for name in report["running_job_names"]) or "none") + "."]
    external = [row["job_id"] for row in report["external_queue_jobs"] if row["status"] == "running"]
    if report["external_queue_jobs"]:
        lines += ["Named external queue jobs running: " + (", ".join("`" + name + "`" for name in external) or "none")
                  + ". External queue states are listed separately from workflow counts."]
    measured = False
    for identity, score in results["results"].items():
        if not score["metrics_reported"]:
            continue
        measured = True
        lines += ["", "Completed score: `" + identity + "` ([exact source](" + score["source_summary_ref"]["path"] + ")).",
                  "Values below are equal-parent means ± sample SD in the stated metric units."]
        if score["schema"] == AMERICAN:
            lines += ["American matched evaluation: " + str(score["records_per_parent"]) + " records per independent parent. "
                      "Native ARGformer and FLARE hard-call transports have no NLL/Brier. Exact paper reproduction: false.",
                      "", "| Split | Method | Parents | Accuracy | Macro F1 | NLL | Brier |", "| --- | --- | ---: | --- | --- | --- | --- |"]
            for split, methods in score["equal_parent_summaries"].items():
                for method, metrics in methods.items():
                    lines.append("| " + split + " | " + method + " | " + str(metrics.get("accuracy", {}).get("parents", "unavailable")) + " | " + " | ".join(
                        _metric(metrics[name]) if name in metrics else "not reported" for name in ("accuracy", "macro_F1", "NLL", "Brier")) + " |")
        elif score["schema"] == C4:
            lines += ["European C4 matched evaluation: " + str(score["records_per_parent"]) + " records per parent. "
                      + score.get("scope", "") + ". Separate genuine four-head TreeLoom training. "
                      "Published ARGMix paper reproduction: false.",
                      "", "| Method | Parents | Accuracy | Macro F1 | Raw NLL | Brier |",
                      "| --- | ---: | --- | --- | --- | --- |"]
            for method, groups in score["summary"].items():
                metrics = groups["overall"]
                lines.append("| " + method + " | " + str(metrics["accuracy"]["independent_parents"]) + " | "
                             + " | ".join(_metric(metrics[name], score["metric_units"][name])
                                          for name in ("accuracy", "macro_F1", "raw_NLL", "Brier")) + " |")
        else:
            lines += ["European native-six matched evaluation: " + str(score["records_per_parent"]) + " records per parent. "
                      + score.get("comparison_scope", "") + ". Published CNN reproduction: false.",
                      "", "| Method | Parents | Accuracy | Macro F1 | Raw NLL | Brier |", "| --- | ---: | --- | --- | --- | --- |"]
            for method, groups in score["summary"].items():
                metrics = groups["overall"]
                lines.append("| " + method + " | " + str(metrics.get("accuracy", {}).get("independent_parents", "unavailable")) + " | " + " | ".join(_metric(metrics[name]) if name in metrics else "not reported"
                             for name in ("accuracy", "macro_F1", "raw_NLL", "Brier")) + " |")
    if not measured:
        lines += ["", "No supported authenticated matched score has completed in this workflow."]
    if report["reporting_errors"]:
        lines += ["", str(len(report["reporting_errors"])) + " metadata reporting error(s) are recorded in the current queue details."]
    lines += [END]
    return "\n".join(lines) + "\n"


def _readme(block):
    path = c.contained_path("README.md")
    _need(not path.is_symlink(), "README must not be a symlink")
    original = path.read_text()
    starts, ends = original.count(BEGIN), original.count(END)
    _need((starts, ends) in ((0, 0), (1, 1)), "Malformed or duplicate workflow README markers")
    if starts:
        begin, end = original.index(BEGIN), original.index(END) + len(END)
        _need(begin < end - len(END), "Reversed workflow README markers")
        # Preserve every byte outside the marked block, including its trailing newline.
        return original[:begin] + block.rstrip("\n") + original[end:]
    return original + ("" if original.endswith("\n\n") else "\n" if original.endswith("\n") else "\n\n") + block


def _git(*arguments):
    return subprocess.run(["git", *arguments], cwd=c.ROOT, capture_output=True, text=True, timeout=45, check=False)


def _checked_git(*arguments):
    result = _git(*arguments)
    _need(result.returncode == 0, "Git " + arguments[0] + " failed (exit " + str(result.returncode) + "): "
          + _text(result.stderr or result.stdout, 1200))
    return result


def _record_error(exc, now):
    record = dict(schema="arggnn_workflow_reporting_error_v1", observed_utc=_utc(now),
                  error_type=type(exc).__name__, error=_text(exc, 2000))
    try:
        path = c.contained_path(RUNTIME_PATH) / "last_error.json"
        _save(path, record)
        return dict(status="error", error=record["error"], error_path=str(path.relative_to(c.ROOT)))
    except Exception as recording:
        return dict(status="error", error=record["error"], error_recording_failed=type(recording).__name__ + ": " + _text(recording))


def publish(plan_ref, state_path, force=False):
    """Refresh exact current reports and publish milestones, isolating errors.

    Ordinary Git attempts are at least 30 minutes apart. A new failed/unknown
    task, completed workflow, new authenticated score, or explicit force bypasses
    that interval once per distinct event. Report files can refresh in between.
    ROOT alone decides when this callable is activated.
    """
    now = time.time()
    try:
        runtime = c.contained_path(RUNTIME_PATH)
        runtime.mkdir(parents=True, exist_ok=True)
        lock_path = runtime / "reporting.lock"
        _need(not lock_path.is_symlink(), "Reporting lock must not be a symlink")
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(status="busy")
            cache_path = runtime / "cache.json"
            cache = c.read_json(cache_path) if cache_path.exists() else {}
            report, results = _snapshot(plan_ref, state_path)
            fingerprint = c.fingerprint(dict(report=report, results=results))
            if fingerprint == cache.get("written_fingerprint"):
                # Reuse the last meaningful observation time; polling is not a milestone.
                report["observed_utc"] = cache["observed_utc"]
            else:
                report["observed_utc"] = _utc(now)
            block = _markdown(report, results)
            readme = _readme(block)  # Validate all marker structure before changing reports.
            _save(STATUS_PATH, report)
            if c.contained_path(SCORE_PATH).exists():
                _need(c.read_json(SCORE_PATH).get("schema") == SCORE_SCHEMA,
                      "Existing matched score file has another owner/schema")
            _save(SCORE_PATH, results)
            _save("README.md", readme)
            cache.update(written_fingerprint=fingerprint, observed_utc=report["observed_utc"])
            event = dict(plan_ref=plan_ref, failures=[row["id"] for row in report["tasks"]
                         if row["status"] in ("failed", "unknown")], complete=report["complete"],
                         scores={key: value["source_summary_ref"] for key, value in results["results"].items()})
            old_event = cache.get("attempt_event", {})
            urgent = (bool(set(event["failures"]) - set(old_event.get("failures", [])))
                      or (event["complete"] and not old_event.get("complete"))
                      or any(value != old_event.get("scores", {}).get(key) for key, value in event["scores"].items()))
            due = force or urgent or now - cache.get("last_attempt_at", 0) >= INTERVAL_SECONDS
            changed = fingerprint != cache.get("committed_fingerprint")
            if not due or not (changed or cache.get("pending_push")):
                _save(cache_path, cache)
                return dict(status="updated" if changed else "unchanged", git_published=False,
                            counts=report["counts"], reporting_errors=report["reporting_errors"])
            cache.update(last_attempt_at=now, attempt_event=event)
            _save(cache_path, cache)  # Failed Git attempts must not hammer every scheduler poll.
            if changed:
                _checked_git("add", "--", *REPORT_PATHS)
                diff = _git("diff", "--cached", "--quiet", "--", *REPORT_PATHS)
                _need(diff.returncode in (0, 1), "Git report diff failed")
                if diff.returncode == 1:
                    _checked_git("commit", "--only", "-m", "Report executable benchmark workflow progress", "--", *REPORT_PATHS)
                    cache.update(pending_push=True, committed_fingerprint=fingerprint, last_commit_at=now)
                else:
                    cache["committed_fingerprint"] = fingerprint
                _save(cache_path, cache)
            if cache.get("pending_push"):
                _checked_git("push", "origin", "HEAD")
                cache.update(pending_push=False, last_push_at=now)
                _save(cache_path, cache)
            return dict(status="published", git_published=True, counts=report["counts"],
                        reporting_errors=report["reporting_errors"])
    except Exception as exc:
        return _record_error(exc, now)
