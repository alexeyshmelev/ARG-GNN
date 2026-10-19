"""Inert signature binding of the actual draft queue with the agreed v2 route.

No factory body is called. Dependency expressions remain symbolic objects.
"""
import importlib
import inspect
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
DIRECTORY = ROOT/"research/repairs/benchmark_recovery_20261020/training/queue_adapters"
PLAN = "configs/benchmarks/executable_workflow_20261020_readable_v2.json"
PREFIX = "research.repairs.benchmark_recovery_20261020.training.queue_adapters."


def reference(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit():
    plan = json.loads((ROOT/PLAN).read_text())
    routes = dict(plan["readable_factory_routes"])
    override = dict(module=PREFIX+"american_workflow_v2", function="prediction_factory")
    routes["arggnn.benchmarking.american_workflow:prediction_factory"] = override
    rows, failures, sources = [], [], {}
    for task in plan["tasks"]:
        if task["resource"] == "external":
            continue
        key = task["factory_module"]+":"+task["factory_function"]
        route = routes.get(key, dict(module=task["factory_module"], function=task["factory_function"]))
        module = importlib.import_module(route["module"])
        function = getattr(module, route["function"])
        signature = inspect.signature(function)
        kwargs = dict(task.get("factory_arguments", {}))
        kwargs["output_directory"] = str(ROOT/task["output_directory"]/"native")
        row = dict(task_id=task["id"], original_route=key, resolved_route=route,
            supplied_keywords=sorted(kwargs), signature=str(signature),
            variadic_keywords=any(item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()))
        try:
            signature.bind(**kwargs)
        except TypeError as error:
            row.update(status="failed", error=str(error))
            failures.append(row)
        else:
            row["status"] = "passed_signature_bind_without_call"
        rows.append(row)
        path = Path(module.__file__).resolve()
        sources[str(path)] = reference(path)
    torch = sys.modules.get("torch")
    initialized = torch is not None and torch.cuda.is_initialized()
    if initialized:
        raise RuntimeError("Metadata-only route audit initialized CUDA")
    result = dict(schema="readable_factory_signature_audit_v2", plan_ref=reference(ROOT/PLAN),
        route_override=override, required_additional_scientific_module=PREFIX+"american_prediction",
        factory_bodies_called=0, symbolic_dependencies_resolved=False, CUDA_initialized=False,
        checked_tasks=len(rows), external_tasks_skipped=sum(t["resource"]=="external" for t in plan["tasks"]),
        variadic_factory_tasks=sum(row["variadic_keywords"] for row in rows),
        failures=failures, rows=rows, inspected_factory_sources=list(sources.values()),
        limitation="Signature binding checks interface shape only. Variadic factories need their existing focused contract tests.")
    output = DIRECTORY/"factory_signature_audit_v2.json"
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(report_ref=reference(output), checked_tasks=len(rows), failures=len(failures),
        factory_bodies_called=0, CUDA_initialized=False), sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
