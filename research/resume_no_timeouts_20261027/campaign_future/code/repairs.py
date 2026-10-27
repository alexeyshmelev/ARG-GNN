"""Explicit immutable failed-branch repairs without touching healthy workers.

Submissions are repair_requests/ID.ref.json exact file references to
repair_requests/ID.request.json. Requests bind the original plan reference,
current plan revision and every affected attempt. Prior definitions and rows
remain in immutable receipts; source-reference lists are losslessly deduplicated.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
from . import runtime as r

SCHEMA = "paper_campaign_repair_request_v1"
REQUIRED = {"schema", "request_id", "expected_plan_ref", "expected_plan_revision",
            "expected_attempts", "reset_task_ids", "replacement_tasks",
            "archived_bytes", "reason"}


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _bounded_ref(reference, maximum):
    _need(type(reference) is dict and set(reference) == {"path", "sha256"},
          "Exact immutable file reference required")
    path = r.path(reference["path"])
    _need(path.is_file() and path.stat().st_size <= maximum,
          "Finite existing repair metadata required")
    return r.read(reference)


def _compact_tasks(tasks):
    compact, source_sets = [], {}
    for original in tasks:
        task = copy.deepcopy(original)
        if "source_refs" in task:
            sources = task.pop("source_refs")
            digest = _digest(sources)
            source_sets[digest] = sources
            task["source_refs_set_sha256"] = digest
        compact.append(task)
    return compact, source_sets


def _prepare(plan, state, directory, submission):
    _need(not submission.is_symlink() and submission.stat().st_size <= 4096,
          "Ordinary bounded repair submission required")
    submission_ref = r.ref(submission)
    reference = _bounded_ref(submission_ref, 4096)
    request = _bounded_ref(reference, 8 * 2**20)
    _need(type(request) is dict and REQUIRED <= set(request)
          and set(request) <= REQUIRED | {"archive_manifest_ref"}, "Exact repair request keys required")
    name = request["request_id"]
    _need(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name),
          "Explicit simple repair identifier required")
    expected_body = directory / "repair_requests" / (name + ".request.json")
    _need(submission.name == name + ".ref.json" and r.path(reference["path"]) == expected_body,
          "Repair identifier must bind its exact named request and submission")
    _need(request["schema"] == SCHEMA and request["expected_plan_ref"] == state["plan_ref"],
          "Repair must bind the original immutable plan")
    revision = state.get("plan_revision", 0)
    _need(type(request["expected_plan_revision"]) is int
          and request["expected_plan_revision"] == revision, "Stale repair plan revision")
    _need(type(request["reason"]) is str and 0 < len(request["reason"]) <= 4096,
          "Finite readable repair reason required")
    reset, replacements = request["reset_task_ids"], request["replacement_tasks"]
    _need(type(reset) is list and all(type(x) is str for x in reset)
          and len(reset) == len(set(reset)) and type(replacements) is list
          and all(type(x) is dict and type(x.get("id")) is str for x in replacements),
          "Explicit unique reset identifiers and complete replacement tasks required")
    replacement_ids = [task["id"] for task in replacements]
    _need(len(replacement_ids) == len(set(replacement_ids)), "Duplicate replacement identifier")
    targets = set(reset) | set(replacement_ids)
    _need(0 < len(targets) <= 512, "Bounded nonempty failed branch required")
    by_id = {task["id"]: task for task in plan["tasks"]}
    _need(targets <= set(by_id) and targets <= set(state["tasks"]), "Repair cannot add unknown tasks")
    attempts = request["expected_attempts"]
    _need(type(attempts) is dict and set(attempts) == targets
          and all(type(x) is int and x >= 1 for x in attempts.values()),
          "Every affected attempt must be explicitly bound")
    for task_id in targets:
        row, old = state["tasks"][task_id], by_id[task_id]
        _need(row["status"] in ("failed", "blocked"), "Only FAILED or BLOCKED tasks may be repaired")
        _need(old["kind"] != "external", "External evidence tasks cannot be repaired")
        _need(attempts[task_id] == row.get("attempt", 1), "Stale repair task attempt")
        _need(not r.path(old["output_directory"]).exists(), "Old output scope must be archived before repair")
    for task in replacements:
        old = by_id[task["id"]]
        _need(task["kind"] == old["kind"]
              and task["output_directory"] == old["output_directory"]
              and task.get("additional_mutable_scopes", []) == old.get("additional_mutable_scopes", []),
              "Replacement must preserve resource kind and exact output/auxiliary scopes")
    replacement_by_id = {task["id"]: task for task in replacements}
    candidate = copy.deepcopy(plan)
    candidate["tasks"] = [copy.deepcopy(replacement_by_id.get(task["id"], task))
                          for task in plan["tasks"]]
    from .scheduler import validate
    validate(candidate)
    for task_id in targets:
        task = next(task for task in candidate["tasks"] if task["id"] == task_id)
        _need(task["kind"] != "external", "Replacement cannot become external evidence")
        output = r.path(task["output_directory"])
        _need(output.is_relative_to(r.ROOT / "experiments") and not output.exists(),
              "Replacement output must be fresh and inside project experiments")
    for task in replacements:
        for source in task.get("source_refs", []):
            path = r.path(source["path"])
            _need(path.is_file() and path.stat().st_size <= 32 * 2**20
                  and r.ref(path) == source, "Replacement source pin changed or is unbounded")
    _need(type(state.get("archived_attempt_bytes", 0)) is int
          and state.get("archived_attempt_bytes", 0) >= 0, "Existing archive ledger must be nonnegative")
    archived = request["archived_bytes"]
    _need(type(archived) is int and archived >= 0, "Nonnegative explicit archive byte accounting required")
    if archived or request.get("archive_manifest_ref"):
        manifest = _bounded_ref(request.get("archive_manifest_ref"), 8 * 2**20)
        _need(type(manifest) is dict and manifest.get("archived_bytes") == archived,
              "Immutable archive manifest must bind the reported archived_bytes")
    prior_tasks, source_sets = _compact_tasks([by_id[name] for name in sorted(targets)])
    receipt = dict(schema="paper_campaign_repair_receipt_v1", status="applied",
                   request_ref=reference, submission_ref=submission_ref,
                   request_id=name, reason=request["reason"], plan_ref=state["plan_ref"],
                   prior_plan_revision=revision, resulting_plan_revision=revision + 1,
                   prior_plan_sha256=_digest(plan), resulting_plan_sha256=_digest(candidate),
                   prior_tasks=prior_tasks, prior_source_ref_sets=source_sets,
                   prior_rows={name:copy.deepcopy(state["tasks"][name]) for name in sorted(targets)},
                   reset_task_ids=sorted(targets), replacement_task_ids=sorted(replacement_ids),
                   resulting_attempts={name:attempts[name] + 1 for name in sorted(targets)},
                   archived_bytes=archived, archive_manifest_ref=request.get("archive_manifest_ref"),
                   source_ref=r.ref(__file__))
    return candidate, receipt


def apply_pending(plan, state, directory):
    """Apply reviewed requests in place; malformed/stale requests never escape.

    Return receipt references for newly applied or rejected submissions.
    The controller retains process control and persists state after this call.
    It must add state.archived_attempt_bytes to its storage baseline and use
    row.attempt (default 1) for attempt-specific jobs/logs and all-log accounting.
    """
    results = []
    try:
        directory = r.path(directory)
        requests = directory / "repair_requests"
        if not requests.is_dir():
            return results
        submissions = sorted(requests.glob("*.ref.json"))
    except Exception as exc:
        state["repair_scan_error"] = str(exc)[:1024]
        return results
    handled = 0
    for submission in submissions:
        if handled >= 32:
            break
        marker = None
        try:
            _need(not submission.is_symlink() and submission.stat().st_size <= 4096,
                  "Ordinary bounded repair submission required")
            marker = r.ref(submission)["sha256"]
            if marker in state.get("processed_repair_submissions", {}):
                continue
            handled += 1
            candidate, receipt = _prepare(plan, state, directory, submission)
            request_id = receipt["request_id"]
            _need(request_id not in state.get("applied_repairs", {}),
                  "Repair identifier was already applied")
            destination = directory / "repair_receipts" / (request_id + ".json")
            if destination.exists():
                _need(r.read(r.ref(destination)) == receipt,
                      "An immutable receipt already uses this repair identifier")
                receipt_ref = r.ref(destination)
            else:
                receipt_ref = r.save(destination, receipt)
            plan.clear()
            plan.update(candidate)
            for name, attempt in receipt["resulting_attempts"].items():
                state["tasks"][name] = dict(status="pending", attempt=attempt,
                                            repair_receipt_ref=receipt_ref)
            state["plan_revision"] = receipt["resulting_plan_revision"]
            state["archived_attempt_bytes"] = state.get("archived_attempt_bytes", 0) + receipt["archived_bytes"]
            state.setdefault("applied_repairs", {})[request_id] = receipt_ref
            state.setdefault("processed_repair_submissions", {})[marker] = receipt_ref
            results.append(receipt_ref)
        except Exception as exc:
            if marker is None:
                handled += 1
            # Rejections are diagnostics, never a reason to cancel healthy jobs.
            rejection = dict(schema="paper_campaign_repair_receipt_v1", status="rejected",
                             submission_path=str(submission.relative_to(r.ROOT)),
                             submission_sha256=marker, error=str(exc)[:2048],
                             plan_revision=state.get("plan_revision", 0))
            try:
                suffix = marker or _digest(str(submission))
                destination = directory / "repair_receipts" / "rejected" / (submission.stem + "." + suffix[:16] + ".json")
                receipt_ref = r.save(destination, rejection)
                results.append(receipt_ref)
                if marker:
                    state.setdefault("processed_repair_submissions", {})[marker] = receipt_ref
            except Exception as save_error:
                state.setdefault("repair_receipt_errors", {})[str(submission)] = str(save_error)[:1024]
    return results
