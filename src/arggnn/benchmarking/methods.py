"""Source-pinned method discovery and pre-admission job specifications.

This module uses the standard library only. Discovery never imports a numerical
runtime, reads a dataset, queries a device or launches a process.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = ROOT / "configs/benchmarks/methods.json"
SCHEMA = "arggnn.benchmark_methods.v1"
GPU0_UUID = "GPU-ff06ca1f-403a-f2d0-b584-71e177452667"


def _project_path(value: str) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Repository path traversal is forbidden")
    path = path if path.is_absolute() else ROOT / path
    if not path.is_relative_to(ROOT) or path.resolve() != path:
        raise ValueError("Canonical repository path required")
    return path


def _validate_ref(record: dict) -> None:
    if (type(record) is not dict or set(record) != {"path", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
        raise ValueError("Expected a path and SHA256 source/artifact reference")
    _project_path(record["path"])


def load_registry(registry_path: str | Path | None = None) -> dict:
    """Read the compact registry, checking its structure without loading methods."""
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA or not isinstance(value.get("methods"), list):
        raise ValueError("Unsupported benchmark method registry")
    identifiers = set()
    for method in value["methods"]:
        identifier = method.get("id", "")
        if (not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None
                or identifier in identifiers):
            raise ValueError("Method IDs must be unique stable identifiers")
        identifiers.add(identifier)
        for field in ("label", "algorithm", "fidelity", "compatible_protocols",
                      "pipeline", "stages", "source_refs", "blocking_requirements"):
            if field not in method:
                raise ValueError(f"Method {identifier} lacks {field}")
        exact = method["fidelity"].get("exact_paper_reproduction")
        if type(exact) is not bool or (exact and not method["fidelity"].get("reproduction_evidence_refs")):
            raise ValueError("Exact-paper claims require explicit verified evidence references")
        if method.get("protocol_ids") != method["compatible_protocols"]:
            raise ValueError("Public protocol_ids must match the explicit compatibility list")
        if not isinstance(method.get("requirements"), list):
            raise ValueError("Readable method requirements required")
        if not isinstance(method["compatible_protocols"], list) or not method["compatible_protocols"]:
            raise ValueError("Explicit compatible protocols required")
        stages = method["stages"]
        if (not isinstance(stages, dict) or not set(method["pipeline"]).issubset(stages)
                or len(method["pipeline"]) != len(set(method["pipeline"]))):
            raise ValueError("The ordered pipeline must name distinct concrete stages")
        for protocol, pipeline in method.get("pipeline_by_protocol", {}).items():
            if protocol not in method["protocol_ids"] or not set(pipeline).issubset(stages):
                raise ValueError("Protocol pipeline must use registered stages")
        for record in method["source_refs"]:
            _validate_ref(record)
        for stage in stages.values():
            entry = stage["entrypoint"]
            if entry["kind"] not in ("callable", "python_script", "python_module", "native_binary"):
                raise ValueError("Unknown method entrypoint kind")
            _validate_ref(entry["source_ref"])
            if entry["source_ref"] not in method["source_refs"]:
                raise ValueError("Every entrypoint must be in the method source closure")
            if stage.get("resource") not in ("cpu", "gpu0"):
                raise ValueError("Only CPU and physical GPU0 stage resources are valid")
            if not isinstance(stage.get("required_artifacts"), list):
                raise ValueError("Explicit stage artifact requirements required")
    return value


def list_methods(registry_path: str | Path | None = None) -> list[dict]:
    """Return independent plain JSON descriptors in registry order."""
    return copy.deepcopy(load_registry(registry_path)["methods"])


def get_method(method_id: str, registry_path: str | Path | None = None) -> dict:
    for method in list_methods(registry_path):
        if method["id"] == method_id:
            return method
    raise KeyError(f"Unknown benchmark method: {method_id}")


def verify_method_sources(method_id: str, registry_path: str | Path | None = None) -> list[dict]:
    """Verify only registered source/config bytes; never inspect scientific files."""
    method = get_method(method_id, registry_path)
    for record in method["source_refs"]:
        path = _project_path(record["path"])
        if not path.is_file() or path.stat().st_size > 32 * 2**20:
            raise ValueError("Missing or oversized registered source: " + record["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Registered method source changed: " + record["path"])
    return method["source_refs"]


def resolve_stage(method_id: str, stage_id: str, registry_path: str | Path | None = None):
    """Lazily return a source-pinned callable. This function does not call it.

    Numerical stages retain their original observed-input, truth and resource
    arguments. Job preparation and execution stay in the separately approved
    benchmark runner.
    """
    method = get_method(method_id, registry_path)
    verify_method_sources(method_id, registry_path)
    entry = method["stages"][stage_id]["entrypoint"]
    if entry["kind"] != "callable":
        raise ValueError("This stage is a native process entrypoint, not an in-process callable")
    module = importlib.import_module(entry["module"])
    function = getattr(module, entry["callable"])
    if not callable(function):
        raise ValueError("Registered stage is not callable")
    return function


def build_job_spec(method_id: str, stage_id: str, *, protocol_id: str,
                   artifacts: dict[str, dict], output_directory: str,
                   registry_path: str | Path | None = None) -> dict:
    """Assemble an inert, truthful plan for a later fresh admission.

    Supplying references does not authenticate authority or mark a job admitted.
    The scheduler accepts only a separately validated actual admission receipt.
    """
    method = get_method(method_id, registry_path)
    if protocol_id not in method["compatible_protocols"]:
        raise ValueError(f"{method_id} is incompatible with protocol {protocol_id}")
    stage = method["stages"][stage_id]
    if protocol_id not in stage.get("compatible_protocols", method["compatible_protocols"]):
        raise ValueError(f"{stage_id} has no implementation for protocol {protocol_id}")
    if not isinstance(artifacts, dict):
        raise ValueError("Stage artifacts must be a mapping of named immutable references")
    if any(not isinstance(key, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None
           or key in {"python", "entrypoint", "output_directory"} for key in artifacts):
        raise ValueError("Artifact names cannot override command fields")
    for record in artifacts.values():
        _validate_ref(record)
    output = _project_path(output_directory)
    missing = [key for key in stage["required_artifacts"] if key not in artifacts]
    entry = stage["entrypoint"]
    argv = None
    if entry.get("argv_template") is not None and not missing:
        fields = {"python": method["python"], "output_directory": str(output),
                  "entrypoint": str(_project_path(entry.get("path", entry["source_ref"]["path"])))}
        fields.update({key: str(_project_path(record["path"])) for key, record in artifacts.items()})
        fields.update({key + "_json": json.dumps(record, sort_keys=True) for key, record in artifacts.items()})
        argv = [part.format_map(fields) for part in entry["argv_template"]]
    blockers = [f"Missing artifact: {name}" for name in missing]
    blockers.extend(method["blocking_requirements"])
    blockers.extend(stage.get("blocking_requirements", []))
    return {
        "schema": "arggnn.benchmark_method_job.v1",
        "method_id": method_id, "stage_id": stage_id, "protocol_id": protocol_id,
        "resource": stage["resource"], "entrypoint": copy.deepcopy(entry),
        "required_artifacts": list(stage["required_artifacts"]),
        "stage_configuration": copy.deepcopy(stage.get("protocol_configuration", {}).get(protocol_id, {})),
        "argv": argv, "artifacts": copy.deepcopy(artifacts),
        "output_directory": str(output), "source_refs": copy.deepcopy(method["source_refs"]),
        "missing_artifacts": missing, "blockers": blockers,
        "source_verification_performed": False,
        "execution_authorized": False, "requires_fresh_admission": True,
        "requires_later_explicit_user_approval": True,
        "required_gpu_policy": None if stage["resource"] == "cpu" else {
            "physical_gpu": 0, "gpu_uuid": GPU0_UUID,
            "process_title": "AI Auto Research Pipeline Test",
            "inherited_audited_GPU0_lease": True,
        },
        "author_environment": copy.deepcopy(stage.get("author_environment", {})),
        "common_data_is_exact_paper_reproduction": False,
    }
