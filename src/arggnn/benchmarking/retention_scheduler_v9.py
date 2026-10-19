"""Private V9 scheduler compiler with explicit lossless retention receipts.

The frozen scratch-retirement rule is unchanged for ordinary retention tasks.
Only the new AP lossless archive route receives its own completion contract.
"""
from __future__ import annotations
import ast
import copy
import inspect
from pathlib import Path
import textwrap
import types
from . import contracts as c
from research.reviews.workflow_progress_diagnosis_20261018.scheduler_v6 import guard_order_v1 as guarded


ARCHIVE_MODULE = "arggnn.benchmarking.european_retention_v9"
ARCHIVE_SCHEMA = "closed_european_lossless_archive_v1"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_retirement(task, entry, retirement, plan, state):
    """Authenticate the new receipt before original local-owner remeasurement."""
    if task.get("factory_module") != ARCHIVE_MODULE:
        need(retirement.get("successful_new_scratch_only") is True,
             "Actual retirement completion required")
        return
    owner_ids = task.get("remeasure_succeeded_tasks")
    need(task.get("factory_function") == "factory" and task["resource"] == "cpu"
         and isinstance(owner_ids, list) and len(owner_ids) == 1,
         "One explicit local AP lossless retention owner required")
    owner_id = owner_ids[0]
    need(owner_id.startswith("AP__") and owner_id.endswith("__ap_infer")
         and task["id"] == owner_id + "__lossless_archive_v9"
         and task["dependencies"] == [owner_id],
         "Only the exact AP inference archive route is eligible")
    owners = [item for item in plan["tasks"] if item["id"] == owner_id]
    need(len(owners) == 1 and owners[0]["resource"] != "external"
         and state["tasks"][owner_id]["status"] == "succeeded"
         and not state["tasks"][owner_id].get("external"),
         "Only a succeeded local observed inference scope may be remeasured")
    owner = owners[0]
    scope = c.contained_path(owner["output_directory"])
    output = c.contained_path(task["output_directory"])
    need(not output.is_relative_to(scope) and not scope.is_relative_to(output),
         "Archive output must be separately accounted from the owner")
    need(entry.get("status") == "succeeded" and entry.get("actual_controller_wait") == 0
         and entry.get("owned_process_group_absent") is True,
         "Lossless archive needs its actual successful wait and group drain")
    job = c.read_ref(entry["job_ref"])
    need(job.get("task") == task and job.get("task_fingerprint") == c.fingerprint(task)
         and job.get("call", {}).get("module") == ARCHIVE_MODULE
         and job["call"].get("function") == "run",
         "Actual archive job must bind the exact reviewed task")
    spec = job["call"]["spec"]
    need(spec.get("schema") == ARCHIVE_SCHEMA and spec.get("owner_task_id") == owner_id
         and c.contained_path(spec["owner_scope"]) == scope
         and c.contained_path(spec["output_directory"]) == output / "native",
         "Actual archive job must bind exact owner and separately charged output")
    envelope = c.read_ref(entry["result_ref"])
    receipt_ref = envelope.get("value")
    need(envelope.get("job_ref") == entry["job_ref"]
         and isinstance(receipt_ref, dict)
         and c.contained_path(receipt_ref["path"]) == output / "native/restore_manifest.json",
         "Actual lossless manifest must be the archive worker result")
    need(retirement.get("schema") == ARCHIVE_SCHEMA
         and retirement.get("inference_ref") == spec.get("inference_ref")
         and retirement.get("actual_wait_ref") == spec.get("actual_wait_ref")
         and c.contained_path(retirement["owner_scope"]) == scope
         and retirement.get("owner_task_id") == owner_id
         and c.contained_path(retirement["archive_output_directory"]) == output / "native"
         and retirement.get("archives_inside_current_task_output") is True
         and retirement.get("compressed_content_preserves_every_original_byte") is True
         and retirement.get("final_observed_tree_and_inputs_unchanged") is True
         and retirement.get("raw_path_replacement_only_after_verified_durable_archive") is True,
         "Verified durable lossless retention receipt required")
    from research.reviews.benchmark_repairs_20261019.resources_v9.european_archive_v1 import CANDIDATES
    rows = retirement.get("entries")
    need(isinstance(rows, list) and rows, "Actual verified archive entries required")
    seen = set()
    saved = 0
    for row in rows:
        original = c.contained_path(row["original_path"])
        need(original.is_relative_to(scope / "native")
             and str(original.relative_to(scope / "native")) in CANDIDATES
             and original not in seen,
             "Each archive entry must name one allowlisted original intermediate")
        seen.add(original)
        need(row.get("decompressed_hash_and_size_verified") is True
             and _hex(row.get("original_sha256")) and _hex(row.get("archive_sha256"))
             and type(row.get("original_bytes")) is int and row["original_bytes"] >= 0
             and type(row.get("archive_bytes")) is int and row["archive_bytes"] >= 0
             and type(row.get("credited_bytes")) is int and row["credited_bytes"] >= 0,
             "Verified size, digest and nonnegative archive accounting required")
        if row.get("state") == "archive_committed":
            archive = c.contained_path(row["archive_path"])
            need(archive.is_relative_to(output / "native") and archive.is_file()
                 and not archive.is_symlink() and archive.stat().st_size == row["archive_bytes"]
                 and not original.exists(),
                 "Committed gzip must remain in its charged new output scope")
            before, after = row.get("original_conservative_charge_bytes"), row.get("archive_conservative_charge_bytes")
            need(type(before) is int and type(after) is int
                 and before >= after >= 0 and row["credited_bytes"] == before - after,
                 "Committed archive credit must match its verified charge difference")
        else:
            need(row.get("state") == "original_kept_no_storage_gain"
                 and row["credited_bytes"] == 0 and original.is_file(),
                 "Unchanged original files cannot receive storage credit")
        saved += row["credited_bytes"]
    need(type(retirement.get("observed_saved_bytes")) is int
         and retirement["observed_saved_bytes"] == saved,
         "Lossless archive total must match its per-file verified accounting")
    # The unchanged scheduler performs current scope_bytes() remeasurement next.
    # Per-file estimates do not directly reduce the active storage projection.


def compile_scheduler_run(run_function):
    """Combine the actual V6 RAM ordering transform and the narrow archive gate."""
    source = textwrap.dedent(inspect.getsource(run_function))
    tree = ast.parse(source)
    node = tree.body[0]
    expected = ast.parse(
        "need(retirement.get('successful_new_scratch_only') is True,'Actual retirement completion required')").body[0]
    matches = [(parent, index) for parent in ast.walk(node)
               if hasattr(parent, "body") and isinstance(parent.body, list)
               for index, item in enumerate(parent.body)
               if ast.dump(item, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    need(len(matches) == 1, "One original scratch retirement completion gate required")
    parent, index = matches[0]
    parent.body[index] = ast.parse(
        "_validate_retirement_v9(task,entry,retirement,plan,state)").body[0]
    transformed = ast.unparse(ast.fix_missing_locations(tree))
    # Feed only this private transformed source to the unchanged reviewed compiler.
    # Its aggregate/lease/storage/RAM-order assertions still run on the actual AST.
    inspector = types.SimpleNamespace(**vars(inspect))
    inspector.getsource = lambda target: transformed if target is run_function else inspect.getsource(target)
    compiler = types.FunctionType(guarded.compile_scheduler_run.__code__,
        dict(guarded.compile_scheduler_run.__globals__, inspect=inspector),
        guarded.compile_scheduler_run.__name__, guarded.compile_scheduler_run.__defaults__,
        guarded.compile_scheduler_run.__closure__)
    compiled = compiler(run_function)
    def invoke(*args, **kwargs):
        actual = types.FunctionType(compiled.__code__,
            dict(run_function.__globals__, _validate_retirement_v9=validate_retirement),
            run_function.__name__, run_function.__defaults__, run_function.__closure__)
        actual.__kwdefaults__ = copy.copy(run_function.__kwdefaults__)
        return actual(*args, **kwargs)
    invoke.__name__ = run_function.__name__
    invoke.__wrapped__ = compiled
    return invoke
