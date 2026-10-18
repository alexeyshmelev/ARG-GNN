"""One independently approved CPU generation -> shared inference pipeline.

No GPU lease, CUDA runtime, native path corpus, old accounting or launch at
import. ROOT owns spec/policy creation. Runtime metadata schema:
benchmark_cpu_runtime_v1, generation/inference={prefix:absolute, packages:{...}}.
Optional storage baseline: benchmark_storage_baseline_v1, root_verified=true,
accepted_at_utc (fresh ROOT acceptance), measured_bytes, source_measurement_ref,
outstanding_native_caps_bytes and metadata_margin_bytes. The measurement's own
historical timestamp remains unchanged; ROOT covers subsequent native writes
with their full outstanding caps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from . import contracts as c
from . import american_preparation as generation
from . import american_observation as observation
from .scheduler import AdmittedJob, PersistentScheduler
from .execution import scope_bytes, _own_group_rss, _stop_group

ROOT = c.ROOT
MODULE = "arggnn.benchmarking.cpu_execution"
SOURCE = "src/arggnn/benchmarking/cpu_execution.py"
STATE = ROOT / ".runtime/benchmarking/american_cpu_queue.json"
LOCK = ROOT / ".runtime/benchmarking/american_cpu_controller.lock"
RUNTIMES = {
    "generation": (".runtime/env", ("numpy", "tskit", "msprime", "stdpopsim")),
    "inference": (".runtime/envs/biological_tsinfer_v1", ("numpy", "tskit", "tsinfer", "tsdate")),
}
REQUIRED_SOURCES = {SOURCE, "src/arggnn/benchmarking/scheduler.py",
    "src/arggnn/benchmarking/contracts.py", "src/arggnn/benchmarking/execution.py",
    "src/arggnn/benchmarking/american_preparation.py", "src/arggnn/benchmarking/american_observation.py",
    *(r["path"] for r in generation.SOURCE_REFS), *(r["path"] for r in observation.SOURCE_REFS)}


def validate_spec(ref, *, fresh=True):
    spec = c.read_ref(ref)
    if (spec.get("schema") != "benchmark_american_cpu_pipeline_v1" or spec.get("resource") != "cpu"
            or not isinstance(spec.get("job_id"), str) or not spec["job_id"]
            or spec.get("python") != ".runtime/env/bin/python"):
        raise ValueError("Source-pinned fresh American CPU pipeline required")
    for name, maximum in (("max_seconds", 86400), ("max_rss_bytes", 128 * 2**30),
                          ("max_output_bytes", 64 * 2**30)):
        if type(spec.get(name)) is not int or not 60 <= spec[name] <= maximum:
            raise ValueError("Positive finite CPU pipeline caps exceed the permitted envelope")
    refs = spec.get("source_refs", [])
    if not REQUIRED_SOURCES.issubset({r.get("path") for r in refs}):
        raise ValueError("Complete maintained/numerical CPU source closure required")
    for record in refs:
        path = c.contained_path(record["path"])
        if path.stat().st_size > 32 * 2**20 or c.file_ref(path) != record:
            raise ValueError("CPU worker source changed")
    runtime = c.read_ref(spec["runtime_ref"])
    if runtime.get("schema") != "benchmark_cpu_runtime_v1":
        raise ValueError("Qualified explicit CPU runtime metadata required")
    for kind, (prefix, packages) in RUNTIMES.items():
        entry = runtime.get(kind, {})
        if entry.get("prefix") != str(ROOT / prefix) or set(entry.get("packages", {})) != set(packages):
            raise ValueError("Explicit supported stage interpreter/package metadata required")
    output = c.contained_path(spec["output_directory"])
    if (output.resolve() != output or not output.is_relative_to(ROOT / "experiments/results")
            or output == ROOT / "experiments/results"):
        raise ValueError("Isolated canonical CPU output scope required")
    first, second = spec["generation_spec"], spec["inference_template"]
    plan = generation.prepare(first) if fresh else {"config": {"split": "train"}}
    if not first.get("task_id", "").startswith("argformer_american_train_"):
        raise ValueError("Only the fresh TRAIN task family is registered")
    if (plan["config"]["split"] != "train" or first["output_directory"] != str(output / "generation")
            or second.get("output_directory") != str(output / "inference")
            or second.get("schema") != "american_fresh_observation_stage_v1"
            or second.get("stage") != "shared_inference" or "input_ref" in second
            or second.get("declared_inference_conventions") != observation.INFERENCE_CONVENTIONS):
        raise ValueError("One fresh full TRAIN generation then actual-result-bound inference required")
    if any(type(second.get(k)) is not int or second[k] <= 0 for k in observation.STAGES["shared_inference"]["caps"]):
        raise ValueError("Explicit complete-parent inference count bounds required")
    bound = second["max_sample_store_bytes"] + 2 * (second["max_tables_bytes"] + 16777216) + second["max_node_map_bytes"] + 8 * 2**20
    if bound > second["max_output_bytes"]:
        raise ValueError("Inference serialization bounds exceed its cap")
    for stage in (first, second):
        if any(type(stage.get(k)) is not int or stage[k] <= 0 for k in ("max_seconds", "max_rss_bytes", "max_output_bytes")):
            raise ValueError("Finite positive stage output/RSS/wall bounds required")
        if (stage["max_seconds"] > spec["max_seconds"] or stage["max_rss_bytes"] > spec["max_rss_bytes"]):
            raise ValueError("Stage caps exceed the whole pipeline envelope")
    if first["max_output_bytes"] + second["max_output_bytes"] + 64 * 2**20 > spec["max_output_bytes"]:
        raise ValueError("Both full stage caps plus temporary/log margin must fit")
    return spec


def worker_argv(ref, spec):
    return (str(c.contained_path(spec["python"])), "-B", "-s", "-m", MODULE,
            "--execute", "--job-ref", json.dumps(ref, sort_keys=True))


def validate_admission(ref):
    packet = c.read_ref(ref)
    if (packet.get("schema") != "benchmark_cpu_worker_admission_v1"
            or packet.get("root_authorizer") != "root_agent"
            or packet.get("deferred_runtime_admission") is not True
            or packet.get("historical_wait_claimed") is not False
            or packet.get("worker_source_ref") != c.file_ref(SOURCE)):
        raise ValueError("Fresh source-bound ROOT CPU admission required")
    spec = validate_spec(packet["input_ref"])
    if packet["job_id"] != spec["job_id"]:
        raise ValueError("CPU worker/admission identity mismatch")
    return AdmittedJob(spec["job_id"], "cpu", worker_argv(packet["input_ref"], spec),
                       str(ROOT), packet["input_ref"], ref, spec["max_seconds"])


def _approved(ref, spec):
    policy_ref = c.file_ref(c.POLICY_PATH)
    policy = c.check_execution_release(policy_ref)
    matches = [j for j in policy.get("approved_jobs", []) if j.get("job_id") == spec["job_id"] and j.get("input_ref") == ref]
    if len(matches) != 1:
        raise PermissionError("This exact CPU job is not approved in the current canonical policy")
    packet = c.read_ref(matches[0]["admission_receipt_ref"])
    if (packet.get("schema") != "benchmark_cpu_worker_admission_v1"
            or packet.get("root_authorizer") != "root_agent"
            or packet.get("deferred_runtime_admission") is not True
            or packet.get("historical_wait_claimed") is not False
            or packet.get("worker_source_ref") != c.file_ref(SOURCE)
            or packet.get("job_id") != spec["job_id"] or packet.get("input_ref") != ref
            or c.read_ref(ref) != spec):
        raise PermissionError("Current exact source/input-bound CPU admission required")
    return policy


def _storage_admission(spec, policy):
    if spec.get("storage_baseline_ref"):
        baseline = c.read_ref(spec["storage_baseline_ref"])
        stamp = datetime.fromisoformat(baseline["accepted_at_utc"])
        age = (datetime.now(timezone.utc) - stamp).total_seconds()
        if (baseline.get("schema") != "benchmark_storage_baseline_v1" or baseline.get("root_verified") is not True
                or type(baseline.get("measured_bytes")) is not int or baseline["measured_bytes"] < 0
                or stamp.tzinfo is None or not 0 <= age <= 300):
            raise ValueError("Fresh reliable ROOT complete-project storage baseline required")
        measured = c.read_ref(baseline["source_measurement_ref"])
        native, margin = baseline["outstanding_native_caps_bytes"], baseline["metadata_margin_bytes"]
        if (measured.get("measured_project_bytes") != baseline["measured_bytes"]
                or type(native) is not int or native < measured.get("approved_new_output_caps_bytes", 0)
                or type(margin) is not int or margin < 64 * 2**20
                or policy.get("approved_new_storage_cap_bytes", 0) < native + spec["max_output_bytes"] + margin):
            raise ValueError("Accepted actual measurement plus all outstanding native/CPU caps required")
        used = baseline["measured_bytes"]
    else:
        result = subprocess.run(["du", "-sb", str(ROOT)], check=True, capture_output=True, text=True, timeout=300)
        used = int(result.stdout.split()[0])
    total = policy.get("approved_new_storage_cap_bytes")
    ceiling = min(policy.get("storage_ceiling_bytes", 980000000000), 980000000000,
                  policy.get("human_storage_limit_bytes", 1000000000000) - 20000000000)
    if type(total) is not int or total < spec["max_output_bytes"] or used + total >= ceiling:
        raise RuntimeError("Actual project bytes plus all approved new caps reaches the storage ceiling")
    return {"measured_bytes": used, "all_approved_caps": total, "ceiling_bytes": ceiling}


def _json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return c.file_ref(path)


def _tick(ref, spec):
    output, deadline = c.contained_path(spec["output_directory"]), float(os.environ["ARGGNN_CPU_DEADLINE"])
    last = [-float("inf")]
    def check(*args):
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("CPU pipeline wall cap reached")
        if now - last[0] >= 2:
            last[0] = now
            _approved(ref, spec)
            if (scope_bytes(output) > spec["max_output_bytes"]
                    or max(_own_group_rss(os.getpgrp()), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) > spec["max_rss_bytes"]):
                raise RuntimeError("CPU pipeline own output/RSS cap exceeded")
    return check


def _run_stage(ref, spec, kind, stage):
    output = c.contained_path(spec["output_directory"])
    stage_ref = _json(output / (kind + "_spec.actual.json"), stage)
    python = ROOT / RUNTIMES[kind][0] / "bin/python"
    command = [str(python), "-B", "-s", "-m", MODULE, "--stage", kind,
               "--stage-ref", json.dumps(stage_ref), "--job-ref", json.dumps(ref)]
    error = None
    with (output / (kind + ".log")).open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            status = process.wait(timeout=max(1, min(stage["max_seconds"],
                float(os.environ["ARGGNN_CPU_DEADLINE"]) - time.monotonic())))
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            process.terminate()
            try:
                status = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); status = process.wait()
    _json(output / (kind + "_actual_wait.json"), dict(actual_direct_wait=True,
        actual_direct_returncode=status, stage_spec_ref=stage_ref, error=error, pid=process.pid))
    if error is not None or status != 0:
        raise RuntimeError("CPU stage failed; no retry: " + kind)
    actual = c.read_ref(c.file_ref(output / (kind + "_return.json")))
    if actual.get("stage_spec_ref") != stage_ref:
        raise RuntimeError("Actual CPU stage result/spec identity differs")
    return actual["result_ref"], stage_ref


def run_pipeline(spec, *, job_ref, resource_check, execution_check):
    """Sequential actual receipts from separately waited supported interpreters."""
    output = c.contained_path(spec["output_directory"])
    execution_check()
    first, _ = _run_stage(job_ref, spec, "generation", spec["generation_spec"])
    resource_check("CPU_generation_receipt_bound")
    second = dict(spec["inference_template"], input_ref=first)
    inferred, second_ref = _run_stage(job_ref, spec, "inference", second)
    resource_check("CPU_shared_inference_receipt_bound")
    return _json(output / "pipeline_completed.json", dict(
        schema="american_cpu_generation_inference_completed_v1",
        generation_ready_ref=first, inference_spec_ref=second_ref, inference_ready_ref=inferred,
        native_path_preparation="pending_separate_compact_focal_stage",
        query_truth_scored=False, test_labels_opened=False, automatic_retry=False))


def _stage_child(ref, kind, stage_ref):
    spec = validate_spec(ref, fresh=False)
    _approved(ref, spec)
    runtime = c.read_ref(spec["runtime_ref"])[kind]
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or str(Path(sys.prefix).resolve()) != runtime["prefix"]
            or any(importlib.metadata.version(k) != v for k, v in runtime["packages"].items())):
        raise RuntimeError("Qualified explicit CPU stage interpreter/runtime required")
    resource.setrlimit(resource.RLIMIT_FSIZE, (spec["max_output_bytes"], spec["max_output_bytes"]))
    stage = c.read_ref(stage_ref)
    output = c.contained_path(spec["output_directory"])
    expected = spec["generation_spec"]
    if kind == "inference":
        actual = c.read_ref(c.file_ref(output / "generation_return.json"))
        expected = dict(spec["inference_template"], input_ref=actual["result_ref"])
    if stage != expected:
        raise RuntimeError("CPU stage must derive from its exact preceding actual result")
    function = generation.run if kind == "generation" else observation.run_inference
    result = function(stage, resource_check=_tick(ref, spec), execution_check=lambda: _approved(ref, spec))
    return _json(output / (kind + "_return.json"), dict(result_ref=result, stage_spec_ref=stage_ref))


def _child(ref):
    spec = validate_spec(ref)
    _approved(ref, spec)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or Path(sys.prefix).resolve() != ROOT / ".runtime/env":
        raise RuntimeError("Explicit CPU-only project orchestrator required")
    return run_pipeline(spec, job_ref=ref, resource_check=_tick(ref, spec),
                        execution_check=lambda: _approved(ref, spec))


def execute(ref):
    spec = validate_spec(ref)
    policy = _approved(ref, spec)
    output = c.contained_path(spec["output_directory"])
    if output.exists():
        raise RuntimeError("One-shot CPU scope exists; no retry or reuse")
    storage = _storage_admission(spec, policy)
    _approved(ref, spec)
    output.mkdir(parents=True, exist_ok=False)
    temporary, cache = output / "tmp", output / "cache"
    temporary.mkdir(); cache.mkdir()
    _json(output / "storage_before_start.json", storage)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ARGGNN_GPU") and k != "ARGGNN_PHYSICAL_GPU"}
    env.update(CUDA_VISIBLE_DEVICES="", TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", XDG_CACHE_HOME=str(cache),
        NUMBA_CACHE_DIR=str(cache / "numba"), STDPOPSIM_CACHE=str(cache / "stdpopsim"),
        MPLCONFIGDIR=str(cache / "matplotlib"), HF_HOME=str(cache / "huggingface"),
        TORCH_HOME=str(cache / "torch"), PYTHONPYCACHEPREFIX=str(cache / "pycache"),
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", NUMBA_NUM_THREADS="8",
        PYTHONPATH=str(ROOT) + ":" + str(ROOT / "src") + ":" + str(ROOT / "scripts"))
    deadline = time.monotonic() + spec["max_seconds"] - 30
    env["ARGGNN_CPU_DEADLINE"] = repr(deadline)
    command = [str(c.contained_path(spec["python"])), "-B", "-s", "-m", MODULE,
               "--child", "--job-ref", json.dumps(ref, sort_keys=True)]
    cancelled, error, process = [], None, None
    handlers = {sig: signal.signal(sig, lambda number, frame: cancelled.append(number)) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (output / "cpu.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if cancelled or time.monotonic() >= deadline:
                        raise TimeoutError("CPU worker cancelled or wall cap reached")
                    _approved(ref, spec)
                    if scope_bytes(output) > spec["max_output_bytes"] or _own_group_rss(process.pid) > spec["max_rss_bytes"]:
                        raise RuntimeError("CPU worker own output/RSS cap exceeded")
                    time.sleep(1)
            except BaseException as exc:
                error = type(exc).__name__ + ": " + str(exc)
            finally:
                _stop_group(process)
                status = process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    try:
        os.killpg(process.pid, 0)
        drained = False
    except ProcessLookupError:
        drained = True
    final_bytes = scope_bytes(output)
    if not drained or final_bytes > spec["max_output_bytes"]:
        error = error or "Owned CPU group not drained or final output cap exceeded"
    result = dict(actual_direct_wait=True, actual_direct_returncode=status, supervisor_error=error,
                  owned_process_group_absent=drained, output_bytes=final_bytes, automatic_retry=False, GPU_used=False)
    _json(output / "actual_wait.json", result)
    if error is not None or status != 0:
        raise RuntimeError("CPU pipeline failed; partial outputs preserved: " + str(result))
    return result


def scheduler():
    return PersistentScheduler(STATE, LOCK, validate_admission=validate_admission,
        validate_execution_policy=c.validate_execution_policy, cpu_parallelism=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-ref"); parser.add_argument("--admission-ref")
    parser.add_argument("--stage", choices=tuple(RUNTIMES)); parser.add_argument("--stage-ref")
    parser.add_argument("--policy-ref"); parser.add_argument("--enqueue", action="store_true")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.stage:
        result = _stage_child(json.loads(args.job_ref), args.stage, json.loads(args.stage_ref))
    elif args.child:
        result = _child(json.loads(args.job_ref))
    elif args.job_ref:
        ref = json.loads(args.job_ref)
        result = execute(ref) if args.execute else validate_spec(ref)
    elif args.enqueue:
        result = scheduler().enqueue(json.loads(args.admission_ref))
    else:
        result = scheduler().run(execute=args.execute,
            execution_policy_ref=json.loads(args.policy_ref) if args.policy_ref else None)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))


if __name__ == "__main__":
    main()
