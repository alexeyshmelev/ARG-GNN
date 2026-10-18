"""Finite sequential native ARGformer prediction queue; planning is the default.

Consumes the existing corrected contrastive job. It never retries a failed or
unknown job. Execution needs the exact chain ref in the canonical ROOT policy.
Only completed-stage boundaries may add one deterministic full next job.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from . import contracts as c
from . import argformer_prediction as prediction
from . import argformer_execution as native
from . import execution
from .scheduler import PersistentScheduler, SchedulerBusy

SCHEMA = "benchmark_native_argformer_chain_v1"
GiB = 2**30
METADATA_CAP = 4 * 2**20
STAGES = [
    dict(id="query_train", view="query_paths", split="train", disk=GiB, rss=64*GiB, seconds=7200),
    dict(id="query_val", view="query_paths", split="val", disk=GiB, rss=64*GiB, seconds=3600),
    dict(id="reference_val", view="contrastive", split="val", disk=8*GiB, rss=256*GiB, seconds=21600),
    dict(id="reference_train", view="contrastive", split="train", disk=64*GiB, rss=256*GiB, seconds=86400),
    dict(id="retrieval", disk=2*GiB, rss=256*GiB, seconds=21600),
]


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


def build_spec(*, parent_input_ref, output_root, records_directory,
               queue_path=".runtime/benchmarking/queue.json",
               controller_lock_path=".runtime/benchmarking/controller.lock",
               max_seconds=3*86400):
    """Build metadata only; parent must already be a real current queue job."""
    parent = c.read_ref(parent_input_ref)
    _need((parent.get("callable_module"), parent.get("callable_name")) ==
          ("arggnn.benchmarking.argformer_execution", "run_contrastive"),
          "Existing corrected full native contrastive parent required")
    sources = prediction.requirements()["source_refs"] + [
        c.file_ref(Path(__file__).resolve()), c.file_ref(execution.__file__),
        c.file_ref(c.__file__),
        c.file_ref("src/arggnn/benchmarking/scheduler.py"),
    ]
    return dict(schema=SCHEMA, root_authorizer="root_agent",
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                max_seconds=max_seconds, parent_job_id=parent["job_id"],
                parent_input_ref=parent_input_ref,
                output_root=str(native._output(output_root).relative_to(c.ROOT)),
                records_directory=str(c.contained_path(records_directory).relative_to(c.ROOT)),
                queue_path=str(c.contained_path(queue_path).relative_to(c.ROOT)),
                controller_lock_path=str(c.contained_path(controller_lock_path).relative_to(c.ROOT)),
                stages=STAGES, total_prediction_scope_caps_bytes=76*GiB,
                metadata_cap_bytes=METADATA_CAP,
                worker_source_ref=c.file_ref(execution.__file__),
                native_python=native.NATIVE_PYTHON,
                max_cuda_reserved_bytes=74*GiB, minimum_free_gpu_bytes=78*GiB,
                source_refs=list({r["path"]: r for r in sources}.values()),
                policy_is_the_only_execution_release=True, automatic_retry=False,
                truth_access=False, scoring=False)


def _load(ref):
    spec = c.read_ref(ref)
    _need(spec.get("schema") == SCHEMA and spec.get("root_authorizer") == "root_agent"
          and spec.get("stages") == STAGES and spec.get("automatic_retry") is False
          and spec.get("truth_access") is False and spec.get("scoring") is False
          and spec.get("total_prediction_scope_caps_bytes") == 76*GiB
          and spec.get("metadata_cap_bytes") == METADATA_CAP,
          "Exact finite five-stage native chain required")
    _need(type(spec.get("max_seconds")) is int and 0 < spec["max_seconds"] <= 7*86400,
          "Finite at-most-seven-day chain window required")
    created = datetime.fromisoformat(spec["created_at_utc"])
    _need(created.tzinfo is not None, "Absolute chain creation time required")
    for key in ("records_directory", "queue_path", "controller_lock_path", "output_root"):
        c.contained_path(spec[key])
    native._output(spec["output_root"])
    _need(spec["worker_source_ref"] == c.file_ref(execution.__file__),
          "Frozen current maintained worker required")
    for record in spec["source_refs"]:
        _need(c.file_ref(record["path"]) == record, "Frozen campaign source changed")
    return spec, created.timestamp() + spec["max_seconds"]


def _scheduler(spec):
    return PersistentScheduler(spec["queue_path"], spec["controller_lock_path"],
                               validate_admission=c.validate_admission,
                               validate_execution_policy=c.validate_execution_policy)


def _job_id(spec, stage):
    return spec["parent_job_id"] + "/" + stage["id"]


def _entry(jobs, job_id):
    matches = [j for j in jobs if j["admission"]["job_id"] == job_id]
    _need(len(matches) <= 1, "Duplicate actual queue job identity")
    return matches[0] if matches else None


def inspect(spec, jobs):
    """Read-only queue progress; no completion filename can create success."""
    parent = _entry(jobs, spec["parent_job_id"])
    if parent is None:
        return dict(status="blocked", reason="Existing parent is absent from queue")
    _need(parent["admission"]["input_ref"] == spec["parent_input_ref"],
          "Chain parent cannot be rebound to another attempt")
    if parent["status"] != "succeeded":
        return dict(status="waiting" if parent["status"] in ("queued", "started") else "blocked",
                    job_id=spec["parent_job_id"], predecessor_status=parent["status"])
    _need(parent.get("actual_returncode") == 0, "Actual successful parent OS wait required")
    for stage in spec["stages"]:
        entry = _entry(jobs, _job_id(spec, stage))
        if entry is None:
            return dict(status="materialize_next", stage=stage["id"])
        if entry["status"] != "succeeded":
            return dict(status="queued" if entry["status"] == "queued" else
                        "waiting" if entry["status"] == "started" else "blocked",
                        stage=stage["id"], job_id=_job_id(spec, stage),
                        predecessor_status=entry["status"])
        _need(entry.get("actual_returncode") == 0, "Actual successful stage OS wait required")
    return dict(status="complete", stages=5, truth_opened=False, scoring_performed=False)


def _save_once(spec, name, value):
    directory = c.contained_path(spec["records_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    _need(directory.resolve() == directory, "Canonical own chain metadata directory")
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    _need(len(raw) <= 256*1024 and Path(name).name == name, "Bounded chain JSON record")
    path = directory / name
    if path.exists():
        _need(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
              "Existing chain record differs; no retry or overwrite")
    else:
        size = sum(p.lstat().st_size + 4096 for p in directory.iterdir())
        _need(size + len(raw) + 8192 <= METADATA_CAP, "Whole finite chain metadata cap")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return c.file_ref(path)


def _producer_bundle(spec, entry, tag, stage):
    _need(entry["status"] == "succeeded" and entry.get("actual_returncode") == 0,
          "Actual successful queue record required")
    job = c.read_ref(entry["admission"]["input_ref"])
    output = c.contained_path(job["scientific_spec"]["output_directory"])
    filename = "native_contrastive_completed.json" if stage == "contrastive" else "embedding_completed.json"
    bundle = dict(completion_ref=c.file_ref(output / filename),
                  actual_wait_ref=c.file_ref(c.contained_path(job["output_directory"]) / "actual_wait.json"),
                  queue_result_ref=_save_once(spec, tag+".queue_result.json", entry))
    prediction._producer(bundle, stage)
    return bundle


def _prepare(spec, stage, jobs):
    parent = _entry(jobs, spec["parent_job_id"])
    producer = _producer_bundle(spec, parent, "parent", "contrastive")
    output = c.contained_path(spec["output_root"]) / stage["id"]
    if stage["id"] == "retrieval":
        producers = {s["view"]+"/"+s["split"]:
                     _producer_bundle(spec, _entry(jobs, _job_id(spec, s)), s["id"], "embedding")
                     for s in spec["stages"] if s["id"] != "retrieval"}
        scientific = prediction.build_retrieval_spec(producers, output_directory=output / "native")
        callable_name = "run_retrieval"
    else:
        scientific = prediction.build_embedding_spec(
            producer, view=stage["view"], split=stage["split"], output_directory=output / "native")
        callable_name = "run_embedding"
    job = dict(schema=execution.SCHEMA, job_id=_job_id(spec, stage), resource="gpu0",
               physical_gpu=0, gpu_uuid=native.GPU0_UUID, python=spec["native_python"],
               output_directory=str(output.relative_to(c.ROOT)),
               callable_module="arggnn.benchmarking.argformer_prediction",
               callable_name=callable_name, scientific_spec=scientific,
               source_refs=spec["source_refs"], max_output_bytes=stage["disk"],
               max_seconds=stage["seconds"], max_rss_bytes=stage["rss"],
               max_cuda_reserved_bytes=spec["max_cuda_reserved_bytes"],
               minimum_free_gpu_bytes=spec["minimum_free_gpu_bytes"])
    job_ref = _save_once(spec, stage["id"]+".job.json", job)
    execution.validate_spec(job_ref)  # Source/shape check only, no runtime or payload read.
    receipt = dict(schema="benchmark_guarded_worker_admission_v1", root_authorizer="root_agent",
                   job_id=job["job_id"], input_ref=job_ref,
                   worker_source_ref=spec["worker_source_ref"],
                   deferred_runtime_admission=True, historical_wait_claimed=False)
    receipt_ref = _save_once(spec, stage["id"]+".admission.json", receipt)
    return job_ref, receipt_ref


def _released_policy(policy, chain_ref, job_ref, receipt_ref, job_id, remaining_caps):
    _need(policy.get("approved_native_chain_ref") == chain_ref,
          "ROOT must release this exact immutable chain in the canonical policy")
    updated = dict(policy)
    approved = dict(job_id=job_id, input_ref=job_ref, admission_receipt_ref=receipt_ref)
    prior = [j for j in policy["approved_jobs"] if j["job_id"] == job_id]
    _need(not prior or prior == [approved], "Policy job cannot be rebound")
    updated["approved_jobs"] = list(policy["approved_jobs"])
    if not prior:
        updated["approved_jobs"].append(approved)
    # Earlier retained files are already inside the next real project du.
    # Independently running CPU jobs retain their complete conservative caps.
    cpu_caps = policy.get("independent_cpu_output_cap_bytes", 0)
    _need(type(cpu_caps) is int and cpu_caps >= 0,
          "Independent outstanding CPU output caps must be nonnegative integers")
    updated["approved_new_storage_cap_bytes"] = remaining_caps + cpu_caps + METADATA_CAP
    return updated


def _write_policy(value):
    path = c.contained_path(c.POLICY_PATH)
    raw = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name+".chain.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return c.file_ref(path)


def _check_final(spec, jobs):
    entry = _entry(jobs, _job_id(spec, STAGES[-1]))
    job = c.read_ref(entry["admission"]["input_ref"])
    done = c.read_json(c.contained_path(job["scientific_spec"]["output_directory"]) / "retrieval_completed.json")
    wait = c.read_json(c.contained_path(job["output_directory"]) / "actual_wait.json")
    _need(wait.get("actual_direct_wait") is True and wait.get("actual_direct_returncode") == 0
          and wait.get("supervisor_error") is None and wait.get("job_id") == job["job_id"]
          and done.get("records") == 10000 and done.get("topk") == 1
          and done.get("query_ancestry_truth_opened") is False and done.get("scoring_performed") is False,
          "Actual complete common10000 native top1 prediction required")


@contextmanager
def _chain_lock(spec):
    path = c.contained_path(spec["records_directory"]) / "chain.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def run(chain_ref, *, execute=False, sleep=time.sleep, clock=time.time):
    """Plan by default. Explicit execution adds/runs only a genuinely ready next job."""
    _need(type(execute) is bool, "Explicit execution boolean required")
    spec, end = _load(chain_ref)
    scheduler = _scheduler(spec)
    if not execute:
        return dict(mode="plan", execution_authorized=False, **inspect(spec, scheduler.jobs()))
    policy_ref = c.file_ref(c.POLICY_PATH)
    policy = c.check_execution_release(policy_ref)
    _need(policy.get("approved_native_chain_ref") == chain_ref, "Exact ROOT chain release required")
    with _chain_lock(spec):
        while clock() < end:
            jobs = scheduler.jobs()
            progress = inspect(spec, jobs)
            if progress["status"] == "complete":
                _check_final(spec, jobs)
                return progress
            if progress["status"] == "blocked":
                return progress  # No automatic replay, repair or replacement.
            if progress["status"] == "waiting":
                sleep(5)
                continue
            stage = next(s for s in STAGES if s["id"] == progress["stage"])
            if clock() + stage["seconds"] > end:
                return dict(status="blocked", reason="Insufficient remaining finite chain window")
            try:
                # Same queue-controller lock excludes a live scheduler/worker.
                # Never change canonical policy while that controller owns it.
                with scheduler._lock(scheduler.controller_lock_path):
                    jobs = scheduler.jobs()
                    progress = inspect(spec, jobs)
                    if progress["status"] not in ("materialize_next", "queued"):
                        continue
                    stage = next(s for s in STAGES if s["id"] == progress["stage"])
                    _need(not any(j["status"] in ("started", "unknown") for j in jobs),
                          "No policy mutation during a live or unknown queued worker")
                    policy_ref = c.file_ref(c.POLICY_PATH)
                    policy = c.check_execution_release(policy_ref)
                    if progress["status"] == "materialize_next":
                        job_ref, receipt_ref = _prepare(spec, stage, jobs)
                    else:
                        entry = _entry(jobs, _job_id(spec, stage))
                        job_ref = entry["admission"]["input_ref"]
                        receipt_ref = entry["admission"]["admission_receipt_ref"]
                    remaining = sum(s["disk"] for s in STAGES
                                    if (_entry(jobs, _job_id(spec, s)) or {}).get("status") != "succeeded")
                    policy_ref = _write_policy(_released_policy(
                        policy, chain_ref, job_ref, receipt_ref, _job_id(spec, stage), remaining))
                    scheduler.enqueue(receipt_ref)
                scheduler.run(execute=True, execution_policy_ref=policy_ref,
                              stop_requested=lambda: clock() >= end)
            except SchedulerBusy:
                sleep(5)
        return dict(status="blocked", reason="Finite chain deadline reached", automatic_retry=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-ref", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(json.loads(args.chain_ref), execute=args.execute), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
