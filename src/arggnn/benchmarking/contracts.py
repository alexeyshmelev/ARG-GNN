"""Small metadata contracts shared by the benchmark CLI and scheduler."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = "configs/benchmarks/execution_policy.json"
MAX_METADATA_BYTES = 2 * 1024 * 1024


class ContractError(ValueError):
    """A plan or result does not satisfy the declared benchmark contract."""


def contained_path(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if ".." in path.parts or not path.resolve().is_relative_to(ROOT):
        raise ContractError("All benchmark files must remain inside the repository")
    return path


def _metadata_bytes(path: str | Path) -> tuple[Path, bytes]:
    path = contained_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_METADATA_BYTES:
            raise ContractError("Expected bounded, ordinary metadata or source")
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ContractError("Metadata grew beyond the read limit")
    return path, raw


def _json_object(raw: bytes) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ContractError("Expected a JSON object")
    return value


def read_json(path: str | Path) -> dict:
    return _json_object(_metadata_bytes(path)[1])


def file_ref(path: str | Path) -> dict[str, str]:
    path, raw = _metadata_bytes(path)
    return {"path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest()}


def read_ref(ref: dict) -> dict:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or not isinstance(ref["sha256"], str)):
        raise ContractError("Expected an exact path/SHA256 metadata reference")
    path, raw = _metadata_bytes(ref["path"])
    observed = {"path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if observed != ref:
        raise ContractError("Metadata changed after its reference was frozen")
    # Hash and parse the same bounded snapshot, never a second file read.
    return _json_object(raw)


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_protocol(protocol: dict) -> dict:
    if protocol.get("schema") != "benchmark_protocol_v1":
        raise ContractError("Unsupported benchmark protocol")
    classes = protocol.get("class_order")
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ContractError("Declare an ordered set of ancestry classes")
    kind = protocol.get("comparison_kind")
    if kind not in {"common_data_reconstruction", "paper_reproduction",
                    "empirical_application"}:
        raise ContractError("Declare the scientific comparison being made")
    if kind == "paper_reproduction" and protocol.get("unresolved_details"):
        raise ContractError("Unresolved publication details preclude exact reproduction")
    if protocol.get("split_unit") != "independent_simulation_parent_before_sampling":
        if kind != "empirical_application":
            raise ContractError("Split independent simulations before sampling trees or windows")
    partitions = protocol.get("split_parents", {})
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        parents = partitions.get(split)
        if not isinstance(parents, list) or len(set(parents)) != len(parents):
            raise ContractError("Each declared split needs unique parent identities")
        if seen.intersection(parents):
            raise ContractError("Simulation parents cannot occur in multiple splits")
        seen.update(parents)
    if protocol.get("observations", {}).get("query_truth_in_features") is not False:
        raise ContractError("Query ancestry truth must be excluded from observed inputs")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("probability_floor") is not False:
        raise ContractError("Native posterior zeroes cannot silently be floored")
    if evaluation.get("parent_aggregation") != "equal_parent_mean_sample_sd":
        raise ContractError("Declare equal-parent means and sample SD across simulations")
    if kind == "empirical_application" and evaluation.get("supervised_accuracy", False):
        raise ContractError("Empirical samples have no known simulated ancestry truth")
    return protocol


def preflight(protocol: dict) -> dict:
    """Inspect small metadata only. Never read genotype arrays, truth or weights."""
    validate_protocol(protocol)
    from .methods import get_method

    findings = []
    for method_id in protocol.get("methods", []):
        method = get_method(method_id)
        if protocol["id"] not in method.get("protocol_ids", []):
            findings.append({"method": method_id, "blocking": True,
                             "message": "Adapter does not declare this ancestry task"})
    sources = []
    for ref in protocol.get("source_refs", []):
        try:
            ok = file_ref(ref["path"]) == ref
            sources.append({"path": ref["path"], "verified": ok})
            if not ok:
                findings.append({"blocking": True, "message": "Changed source: " + ref["path"]})
        except (OSError, ValueError) as error:
            findings.append({"blocking": True, "message": str(error)})
    for requirement in protocol.get("required_artifacts", []):
        ref = requirement.get("ref")
        if ref is None:
            findings.append({"blocking": True, "message": requirement["description"]})
        else:
            try:
                read_ref(ref)
            except (OSError, ValueError) as error:
                findings.append({"blocking": True, "message": str(error)})
    if protocol.get("ready_for_execution") is not True:
        findings.append({"blocking": True,
                         "message": "Protocol execution readiness has not been released"})
    policy = read_json(POLICY_PATH)
    approved = policy.get("benchmark_execution_approved") is True
    return {"protocol_id": protocol["id"], "comparison_kind": protocol["comparison_kind"],
            "scientifically_exact_paper_reproduction":
                protocol["comparison_kind"] == "paper_reproduction",
            "source_checks": sources, "findings": findings,
            "unresolved_details": protocol.get("unresolved_details", []),
            "user_approval_present": approved,
            "ready": approved and not any(f.get("blocking") for f in findings),
            "scientific_payloads_opened": False, "jobs_started": 0}


def validate_module_argv(argv) -> tuple[str, ...]:
    """Bounded syntax validation only, never authority to execute a command."""
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(a, str) and "\0" not in a for a in argv)
            or sum(map(len, argv)) > 32768):
        raise ContractError("Expected a bounded literal argv array")
    first = 1 if argv[0] == "setsid" else 0
    if first >= len(argv):
        raise ContractError("Missing project interpreter after setsid")
    executable = contained_path(argv[first])
    if not executable.is_file() or "python" not in executable.name:
        raise ContractError("Use the admitted project Python interpreter")
    if (argv.count("-m") != 1
            or any(a in argv for a in ("-c", "/bin/sh", "bash", "sh"))):
        raise ContractError("Expected an admitted module entrypoint")
    index = argv.index("-m")
    if index <= first or index + 1 >= len(argv):
        raise ContractError("Missing module name after -m")
    if not argv[index + 1].startswith(("arggnn.", "research.reviews.")):
        raise ContractError("The guarded entrypoint must belong to this project")
    return tuple(argv)


def validate_admission(ref: dict):
    """Validate a fresh approved worker, without inventing historical waits.

    Runtime admission is deliberately deferred to the maintained worker's real
    storage, physical GPU lease, memory and policy checks before science starts.
    Historical normalized admission packets remain unsupported.
    """
    packet = read_ref(ref)
    if packet.get("schema") != "benchmark_guarded_worker_admission_v1":
        raise ContractError("Scientific admission-family adapters are not released for this packet")
    if (packet.get("root_authorizer") != "root_agent"
            or packet.get("deferred_runtime_admission") is not True
            or packet.get("historical_wait_claimed") is not False):
        raise ContractError("A fresh ROOT worker approval with deferred real runtime checks is required")
    source_ref = packet.get("worker_source_ref")
    if (not isinstance(source_ref, dict)
            or source_ref.get("path") != "src/arggnn/benchmarking/execution.py"
            or file_ref(source_ref["path"]) != source_ref):
        raise ContractError("The current guarded worker source must be pinned")
    from .execution import validate_spec, worker_argv
    from .scheduler import AdmittedJob
    spec = validate_spec(packet["input_ref"])
    if spec["job_id"] != packet.get("job_id"):
        raise ContractError("Worker job and approval identity differ")
    argv = worker_argv(packet["input_ref"])
    validate_module_argv(list(argv))
    return AdmittedJob(job_id=spec["job_id"], resource=spec["resource"],
                       argv=tuple(argv), cwd=str(ROOT), input_ref=packet["input_ref"],
                       admission_receipt_ref=ref, max_seconds=spec["max_seconds"])


def check_execution_release(ref: dict) -> dict:
    """Check the canonical user release without selecting a job."""
    canonical = read_json(POLICY_PATH)
    if canonical.get("benchmark_execution_approved") is not True:
        raise ContractError("Benchmarking is paused pending explicit user approval")
    if ref != file_ref(POLICY_PATH):
        raise ContractError("Use the current canonical execution policy")
    policy = read_ref(ref)
    if policy.get("status") != "approved" or policy.get("physical_gpus") != [0]:
        raise ContractError("The execution policy must explicitly approve GPU 0 only")
    if policy.get("approval_origin") != "explicit_user_go_ahead":
        raise ContractError("A new explicit user go-ahead is required")
    end = datetime.fromisoformat(policy["expires_at_utc"])
    if end.tzinfo is None or datetime.now(timezone.utc) >= end:
        raise ContractError("Execution approval has expired")
    if policy.get("smoke_tests_only") is not False:
        raise ContractError("Release smoke-only mode explicitly before benchmarking")
    return policy


def validate_execution_policy(ref: dict, job) -> bool:
    """Last gate before each subprocess. There is no CLI command to approve it."""
    policy = check_execution_release(ref)
    allowed = policy.get("approved_jobs", [])
    expected = {"job_id": job.job_id,
                "admission_receipt_ref": job.admission_receipt_ref,
                "input_ref": job.input_ref}
    if expected not in allowed:
        raise ContractError("This exact admitted job is not approved")
    return True
