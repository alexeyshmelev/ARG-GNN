"""Read only exact v3 predecessors without restarting or opening scientific payloads."""
from __future__ import annotations
import os
from pathlib import Path
from . import contracts as c

def need(ok, message):
    if not ok:
        raise ValueError(message)

def is_ref(value):
    return (isinstance(value, dict) and set(value) == {"path", "sha256"}
            and isinstance(value["path"], str) and isinstance(value["sha256"], str)
            and len(value["sha256"]) == 64)

def proc_identity(pid):
    """Read only the named process, never enumerate unrelated process metadata."""
    try:
        raw = Path("/proc") / str(pid) / "stat"
        text = raw.read_text()
    except FileNotFoundError:
        return None
    tail = text[text.rfind(")") + 2:].split()
    return dict(pid=int(pid), ppid=int(tail[1]), pgid=int(tail[2]), session=int(tail[3]),
                start_ticks=int(tail[19]), state=tail[0])

def _bindings(plan):
    bindings = plan.get("workflow_external_predecessors", [])
    need(isinstance(bindings, list), "Explicit workflow predecessor bindings required")
    ids = [b["source_task_id"] for b in bindings]
    need(len(ids) == len(set(ids)), "Workflow predecessors must be distinct")
    return bindings

def _source(binding):
    plan = c.read_ref(binding["source_plan_ref"])
    tasks = [t for t in plan["tasks"] if t["id"] == binding["source_task_id"]]
    need(len(tasks) == 1 and c.fingerprint(tasks[0]) == binding["source_task_fingerprint"],
         "Exact original scientific predecessor descriptor required")
    task = tasks[0]
    need(task["resource"] == "cpu" and task["max_output_bytes"] == binding["max_output_bytes"]
         and task["max_rss_bytes"] == binding["max_rss_bytes"]
         and c.contained_path(task["output_directory"]) == c.contained_path(binding["output_directory"]),
         "Original predecessor resource and output bounds required")
    job = c.read_ref(binding["original_job_ref"])
    need(job.get("plan_ref") == binding["source_plan_ref"] and job.get("task") == task
         and job.get("task_fingerprint") == binding["source_task_fingerprint"],
         "Exact original predecessor job required")
    expected_state = c.contained_path(plan["records_directory"]) / "state.json"
    need(c.contained_path(binding["source_state_path"]) == expected_state,
         "Read only the original controller's named state")
    state = c.read_json(expected_state)
    need(state["plan_ref"] == binding["source_plan_ref"], "Original controller plan changed")
    entry = state["tasks"][binding["source_task_id"]]
    need(entry.get("job_ref") == binding["original_job_ref"]
         and entry.get("pid") == binding["original_supervisor_pid"],
         "Original job and supervisor identities changed")
    return task, entry

def _drained(binding, entry):
    if entry["status"] not in ("succeeded", "failed"):
        return None
    if (type(entry.get("actual_controller_wait")) is not int
            or entry.get("owned_process_group_absent") is not True):
        return None
    path = c.contained_path(binding["output_directory"]) / "actual_wait.json"
    if not path.exists():
        return None
    ref = c.file_ref(path)
    wait = c.read_ref(ref)
    need(wait.get("job_ref") == binding["original_job_ref"],
         "Original producer wait must bind its original job")
    if (wait.get("actual_direct_wait") is not True
            or type(wait.get("actual_direct_returncode")) is not int
            or wait.get("owned_process_group_absent") is not True):
        return None
    return ref, wait

def external_result(task):
    """Expose the old envelope's actual value, never its envelope as the value."""
    binding = task["workflow_predecessor"]
    need(task["id"] == binding["source_task_id"] and task["resource"] == "external",
         "Explicit external workflow task required")
    _, entry = _source(binding)
    if entry["status"] in ("failed", "blocked"):
        return dict(status="failed", error="Original workflow predecessor failed",
                    original_job_ref=binding["original_job_ref"])
    if entry["status"] != "succeeded":
        return None
    terminal = _drained(binding, entry)
    if terminal is None:
        return None
    wait_ref, wait = terminal
    need(entry["actual_controller_wait"] == 0
         and wait["actual_direct_returncode"] == 0
         and wait.get("supervisor_error") is None,
         "Original producer must genuinely succeed")
    result_ref = entry["result_ref"]
    need(c.contained_path(result_ref["path"]) ==
         c.contained_path(binding["output_directory"]) / "result.json",
         "Original result envelope must remain in its original output scope")
    envelope = c.read_ref(result_ref)
    need(envelope.get("job_ref") == binding["original_job_ref"] and "value" in envelope,
         "Original successful result envelope required")
    value = envelope["value"]
    need(isinstance(value, dict), "Original scientific return value must be metadata")
    # Native inference.json can be hundreds of MB. Its exact FileRef is already
    # authenticated inside this small original envelope. Downstream native readers
    # authenticate that payload with their explicitly declared 512 MiB cap.
    if is_ref(value):
        c.contained_path(value["path"])
    return dict(status="succeeded", value=value, external_actual_wait_ref=wait_ref,
                external_source_result_ref=result_ref,
                external_original_job_ref=binding["original_job_ref"],
                original_supervisor_pid=binding["original_supervisor_pid"])

def external_resources(plan):
    cpu = rss = 0
    for binding in _bindings(plan):
        _, entry = _source(binding)
        if _drained(binding, entry) is None:
            cpu += 1
            rss += binding["max_rss_bytes"]
    return cpu, rss

def storage_reservation_bytes(plan):
    total = 0
    for binding in _bindings(plan):
        _, entry = _source(binding)
        if _drained(binding, entry) is None:
            total += binding["max_output_bytes"]
        else:
            retained = entry.get("retained_bytes")
            need(type(retained) is int and 0 <= retained <= binding["max_output_bytes"],
                 "Only genuine completed retained accounting can release output reserve")
            total += retained
    return total

def live_group_bindings(plan):
    """Return exact still-undrained groups for conservative PSS headroom credit."""
    result = []
    for binding in _bindings(plan):
        _, entry = _source(binding)
        if _drained(binding, entry) is not None:
            continue
        group = proc_identity(binding["owned_process_group_id"])
        supervisor = proc_identity(binding["original_supervisor_pid"])
        if (group is not None and supervisor is not None
                and group["start_ticks"] == binding["owned_group_leader_start_ticks"]
                and group["pgid"] == group["pid"]
                and group["session"] == binding["owned_group_leader_session"]
                and group["ppid"] == supervisor["pid"]
                and supervisor["start_ticks"] == binding["original_supervisor_start_ticks"]):
            result.append(dict(binding, pid=group["pid"], start_ticks=group["start_ticks"],
                               pgid=group["pgid"], session=group["session"]))
    return result
