"""Root-delegated CUDA bootstrap. Module scope is standard-library only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from . import gpu_pool_v1 as pool
from . import native_argformer_exec as guard

ROOT = pool.ROOT
SOURCE = "src/arggnn/anchor_compact_cuda_bootstrap_v1.py"
PROTOCOL = "configs/anchor_compact_cuda_bootstrap_v1.json"
PROTOCOL_SHA256 = "986577a0ad0da16b8fb73823f3441819828f8055e7b18280dabe2e27cff4a215"
RUNNER = "arggnn.anchor_compact_qualification_runner_v1"
RUNNER_CONFIG = {"path": "configs/anchor_compact_qualification_runner_v1.json",
                 "sha256": "0c8f27a35b2b1725bc817e6b0426346ae36634c7eed0c3e3244b994df424b7b5"}
RUNNER_REVIEW = {"path": "research/reviews/anchor_compact_qualification_runner_v1_independent/review_receipt.json",
                 "sha256": "68c46636840832c5043ffbfe1ec2900b197c7a11e9fdfcd57e9443b2c0f9d0bb"}
ADAPTER_CONFIG = {"path": "configs/anchor_compact_actual_bank_v1.json",
                  "sha256": "fb03f822587dfdb3f9ca9d605a2bde9714e3cc3ada9980d53242be05c86e9894"}
ADAPTER_REVIEW = {"path": "research/reviews/anchor_compact_actual_bank_v1_independent/review_receipt_v2.json",
                  "sha256": "e117a87d0cfdd1ce42a947493d52d7b3df65460e6c56956eb50ea8362876e188"}
CACHE_PATHS = {
    "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data", "CUDA_CACHE_PATH": "cache/cuda",
    "TORCH_HOME": "cache/torch", "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton", "NUMBA_CACHE_DIR": "cache/numba",
    "MPLCONFIGDIR": "config/matplotlib", "PYTHONPYCACHEPREFIX": "cache/pycache",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def bound_bytes(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}
            and isinstance(record["path"], str) and isinstance(record["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", record["sha256"]), "Exact hash-bound record required")
    path = pool.canonical_path(ROOT, record["path"])
    require(path.is_file(), "Regular bound file required")
    with path.open("rb") as handle:
        data = handle.read(1024 * 1024 + 1)
    require(len(data) <= 1024 * 1024 and hashlib.sha256(data).hexdigest() == record["sha256"],
            "Source/permit/receipt SHA256 differs: " + record["path"])
    return data


def read_bound(record):
    return json.loads(bound_bytes(record))


def fresh(record, now, initial=True):
    issued, expires = record.get("issued_unix_seconds"), record.get("expires_unix_seconds")
    require(type(issued) in (int, float) and type(expires) in (int, float)
            and math.isfinite(issued) and math.isfinite(expires)
            and 0 <= now - issued and (not initial or now - issued <= 120)
            and now < expires, "Fresh unexpired root authorization required")


def validate_storage(storage, output, pool_log, now, initial=True):
    require(storage.get("kind") == "root_compact_runner_storage_v1"
            and storage.get("status") == "active_root_admission"
            and storage.get("authorizer") == "root_agent"
            and storage.get("project_root") == str(ROOT)
            and storage.get("runner_config") == RUNNER_CONFIG, "Exact root storage domain required")
    stamp = storage.get("issued_unix_seconds")
    require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= now - stamp and (not initial or now - stamp <= 120),
            "Fresh root storage observation required")
    expiry = storage.get("reservation_expires_unix_seconds")
    require(type(expiry) in (int, float) and math.isfinite(expiry) and now < expiry
            and storage.get("reservation_scope") == [str(output)]
            and storage.get("output_cache_log_cap_bytes") == 1000000000
            and storage.get("reservation_active") is True
            and storage.get("raw_tensors_or_checkpoints_retained") is False
            and storage.get("global_scan_used") is False
            and storage.get("accounting_scope") == "entire_repository_including_ignored_environment"
            and storage.get("unaccounted_writer_scope") == []
            and storage.get("warning_threshold_bytes") == 450000000000
            and storage.get("hard_limit_bytes") == 500000000000, "Complete owned1GB storage admission required")
    base, reservations = storage.get("observed_repository_bytes"), storage.get("outstanding_growth_reservations")
    require(type(base) is int and base >= 0 and isinstance(reservations, list)
            and storage.get("current_reservation_inventory") == reservations,
            "Current complete reservation inventory required")
    total, seen, owned, logs = base, set(), 0, 0
    for item in reservations:
        require(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
                and item["id"] not in seen and type(item.get("bytes")) is int
                and item["bytes"] >= 0 and item.get("status") == "reserved"
                and isinstance(item.get("scope"), list) and item["scope"], "Invalid growth reservation")
        seen.add(item["id"])
        for scope in item["scope"]:
            pool.canonical_path(ROOT, scope)
        total += item["bytes"]
        if item["scope"] == [str(output)]:
            require(item["bytes"] == 1000000000, "Exact runner1GB reserve required")
            owned += 1
        if item["scope"] == [str(pool_log)]:
            require(item["bytes"] == 1048576, "Exact separately accounted pool log reserve required")
            logs += 1
    require(owned == logs == 1 and total == storage.get("projected_repository_peak_bytes")
            and total < 450000000000, "Owned/log reservation or storage arithmetic differs")


def check_parent(permit):
    expected = permit["controller_identity"]
    require(isinstance(expected, dict) and set(expected) == {"pid", "uid", "start_ticks"},
            "Exact controller PID identity required")
    for key in expected:
        pool.integer(expected[key], "controller " + key, 1 if key != "uid" else 0)
    current, own = pool.proc_identity(expected["pid"]), pool.proc_identity(os.getpid())
    require({key: current[key] for key in expected} == expected and current["state"] not in {"Z", "X"}
            and own["uid"] == expected["uid"] == os.getuid() and own["ppid"] == expected["pid"]
            and own["pgid"] == own["sid"] == own["pid"] == os.getpid(),
            "Live own controller parent/UID/session identity differs")


def check_runtime(expected, lease, protocol):
    frozen = protocol["expected_runtime"]
    require(set(expected) == set(frozen) | {"driver_version"}
            and all(expected[key] == value for key, value in frozen.items())
            and platform.python_version() == expected["python"]
            and str(Path(sys.executable)) == expected["python_executable"]
            and hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == expected["python_executable_sha256"]
            and importlib.metadata.version("torch") == expected["torch"], "Fixed precontext runtime differs")
    result = subprocess.run(["nvidia-smi", "--id=" + lease["visible_gpu_uuid"],
        "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True,
        check=True, timeout=15)
    require(result.stdout.strip() == expected["driver_version"], "Fresh root driver runtime differs")


def check_registered_job(admission):
    permit, output = admission["permit"], admission["output"]
    registry = pool.read_json(pool.canonical_path(ROOT, f".runtime/gpup/{permit['pool']}/registry.json"))
    require(registry.get("schema") == "gpu_pool_registry_v1" and registry.get("pool") == permit["pool"]
            and registry.get("lifecycle") == "running"
            and registry.get("physical_gpu") == admission["lease"]["physical_gpu"]
            and all(registry["controller"][key] == value for key, value in permit["controller_identity"].items()),
            "Exact live registered pool controller required")
    requested = permit["pool_submission"]
    require(set(requested) == {"job", "model", "seed", "module", "output", "peak_mib", "storage_bytes", "policy", "log_limit_bytes"}
            and requested["job"] == permit["logical_job"] and requested["model"] == "H020_compact_qualification"
            and type(requested["seed"]) is int and requested["seed"] == 101
            and requested["module"] == "arggnn.anchor_compact_cuda_bootstrap_v1"
            and requested["output"] == str(output) and requested["storage_bytes"] == 1001048576
            and requested["log_limit_bytes"] == 1048576 and requested["policy"] == "reject",
            "Exact root-owned pool submission reservation required")
    for key in ("peak_mib", "storage_bytes", "log_limit_bytes"):
        pool.integer(requested[key], "pool submission " + key, 1)
    jobs = [job for job in registry["jobs"] if job["logical_job"] == permit["logical_job"]]
    require(len(jobs) == 1, "Exactly one registered logical worker required")
    job = jobs[0]
    expected_args = ["--config", admission["config_record"]["path"], "--config-sha256", admission["config_record"]["sha256"],
        "--permit", admission["permit_record"]["path"], "--permit-sha256", admission["permit_record"]["sha256"],
        "--output", str(output)]
    require(job["state"] in {"launching", "running"}
            and job["requested"] == dict(requested, args=expected_args)
            and job["argv"] == [permit["runtime_expected"]["python_executable"], "-m", requested["module"], *expected_args]
            and job["module_source_sha256"] == permit["bootstrap_source"]["sha256"]
            and (job["pid"] is None or job["pid"] == os.getpid())
            and (job["leader_start_ticks"] is None
                 or job["leader_start_ticks"] == pool.proc_identity(os.getpid())["start_ticks"]),
            "Registered own worker argv/source/PID/reservation differs")


def check_inventory(permit, lease):
    devices = permit["project_physical_gpus"]
    require(isinstance(devices, list) and all(type(i) is int for i in devices)
            and devices == sorted(set(devices)) and lease["physical_gpu"] in devices
            and 0 < len(devices) <= 2 and set(devices) <= set(pool.ALLOWED_GPUS),
            "Complete eligible/max-two root device admission required")
    inventory = pool.checked_inventory(pool.real_inventory)
    expected_pids = {physical: set() for physical in devices}
    seen_compute = set()
    require(isinstance(permit["project_compute_identities"], list), "Complete root compute list required")
    for item in permit["project_compute_identities"]:
        require(set(item) == {"pid", "uid", "start_ticks", "physical_gpu", "uuid"}
                and item["physical_gpu"] in devices and inventory[item["physical_gpu"]]["uuid"] == item["uuid"],
                "Exact root compute identity required")
        for key in ("pid", "uid", "start_ticks", "physical_gpu"):
            pool.integer(item[key], "compute " + key, 0 if key == "uid" else 1)
        require(item["pid"] not in seen_compute, "Duplicate root compute PID")
        seen_compute.add(item["pid"])
        observed = pool.proc_identity(item["pid"])
        require(all(observed[key] == item[key] for key in ("pid", "uid", "start_ticks"))
                and observed["uid"] == os.getuid() and observed["state"] not in {"Z", "X"},
                "Root compute PID identity changed")
        expected_pids[item["physical_gpu"]].add(item["pid"])
    require(inventory[lease["physical_gpu"]]["uuid"] == lease["visible_gpu_uuid"]
            and all({p["pid"] for p in inventory[physical]["compute"]} == expected_pids[physical]
                    for physical in devices)
            and all(os.getpid() not in {p["pid"] for p in row["compute"]} for row in inventory.values()),
            "Leased device inventory changed or own context already exists")


def admit_precontext(config_record, permit_record, output_value):
    require(Path(__file__).resolve() == ROOT / SOURCE, "Only canonical bootstrap source admitted")
    require(config_record == {"path": PROTOCOL, "sha256": PROTOCOL_SHA256}, "Exact bootstrap protocol required")
    config, permit = read_bound(config_record), read_bound(permit_record)
    require(config.get("kind") == "anchor_compact_cuda_bootstrap_protocol_v1"
            and config.get("status") == "source_only_actual_execution_not_admitted"
            and config.get("actual_execution_admitted") is False
            and config.get("root_delegated_live_evidence_issuance_requires_fresh_permit") is True,
            "Exact source-only bootstrap protocol required")
    require(permit.get("kind") == "root_compact_cuda_bootstrap_v1_permit"
            and permit.get("status") == "active_root_admission"
            and permit.get("authorizer") == "root_agent" and permit.get("project_root") == str(ROOT)
            and permit.get("bootstrap_config") == config_record
            and permit.get("root_delegate_live_evidence_issuance") is True
            and permit.get("fit_allowed") is False and permit.get("truth_allowed") is False
            and permit.get("launch_after_prior_pools_natural_drain") is True,
            "Fresh root-delegated operational permit required")
    now = time.time()
    fresh(permit, now)
    require(permit["bootstrap_source"]["path"] == SOURCE, "Exact bootstrap source pin required")
    bound_bytes(permit["bootstrap_source"])
    for path, digest in config["source_sha256"].items():
        bound_bytes({"path": path, "sha256": digest})
    review_record = permit["independent_bootstrap_review"]
    require(review_record["path"] == "research/reviews/anchor_compact_cuda_bootstrap_v1_independent/review_receipt.json",
            "Assigned independent bootstrap review required")
    review = read_bound(review_record)
    require(review.get("verdict") == "PASS" and review.get("actual_execution_admitted") is False
            and review.get("bootstrap_sha256") == permit["bootstrap_source"]["sha256"]
            and review.get("bootstrap_config_sha256") == config_record["sha256"],
            "Independent review must bind exact bootstrap/protocol bytes")
    extension = permit["pool_extension"]
    require(extension["source"]["path"] == "scripts/gpu_pool_research_v2.py"
            and extension["config"]["path"] == "configs/gpu_pool_research_v2.json"
            and extension["independent_review"]["path"] == "research/reviews/gpu_pool_research_v2_independent/review_receipt.json",
            "Assigned separate pool integration required")
    bound_bytes(extension["source"])
    extension_config, extension_review = read_bound(extension["config"]), read_bound(extension["independent_review"])
    require(extension_config.get("allowed_modules") == sorted(pool.MODULES | {RUNNER, "arggnn.anchor_compact_cuda_bootstrap_v1"})
            and extension_config.get("bootstrap_worker", {}).get("source") == permit["bootstrap_source"]
            and extension_config["bootstrap_worker"].get("config") == config_record
            and extension_review.get("verdict") == "PASS" and extension_review.get("actual_execution_admitted") is False
            and extension_review.get("launcher_sha256") == extension["source"]["sha256"]
            and extension_review.get("config_sha256") == extension["config"]["sha256"]
            and extension_review.get("bootstrap_sha256") == permit["bootstrap_source"]["sha256"]
            and extension_review.get("bootstrap_config_sha256") == config_record["sha256"],
            "Exact reviewed pool bootstrap integration required")
    output = pool.canonical_path(ROOT, output_value)
    require(str(output) == permit["output_scope"] and output != ROOT
            and not output.exists() and output.parent.is_dir(), "Fresh exact owned output required")
    require(isinstance(permit["pool"], str) and pool.NAME.fullmatch(permit["pool"])
            and len(permit["pool"]) <= 32 and isinstance(permit["logical_job"], str)
            and pool.NAME.fullmatch(permit["logical_job"])
            and os.environ.get("ARGGNN_LOGICAL_JOB_NAME") == permit["logical_job"], "Exact logical pool job required")
    pool_log = pool.canonical_path(ROOT, f".runtime/gpup/{permit['pool']}/jobs/{permit['logical_job']}.log")
    validate_storage(permit["storage_fields"], output, pool_log, now)
    experiment = read_bound(permit["experiment_admission"])
    require(experiment.get("kind") == "root_compact_runner_experiment_v1"
            and experiment.get("status") == "active_root_admission"
            and experiment.get("authorizer") == "root_agent" and experiment.get("project_root") == str(ROOT)
            and experiment.get("hypothesis_id") == "H020" and experiment.get("fit_allowed") is False
            and experiment.get("truth_allowed") is False and experiment.get("ledger_consulted") is True
            and experiment.get("decisions_registered") is True
            and isinstance(experiment.get("repeat_or_new_condition_reason"), str)
            and experiment["repeat_or_new_condition_reason"], "Current root experiment admission required")
    fresh(experiment, now)
    require(experiment["ledger"]["path"] == "research/experiment_ledger.md"
            and experiment["decisions"]["path"] == "research/experiment_decisions.json",
            "Exact current ledger/decision bindings required")
    bound_bytes(experiment["ledger"])
    bound_bytes(experiment["decisions"])
    lease = guard.validate_lease()
    require(lease == permit["lease"]
            and lease["launcher_metadata"]["launcher_pid"] == permit["controller_identity"]["pid"]
            and lease["launcher_metadata"]["job"] == f"gpu_pool_v1:{permit['pool']}:gpu{lease['physical_gpu']}",
            "Exact inherited shared-descriptor lease required")
    check_parent(permit)
    check_inventory(permit, lease)
    check_runtime(permit["runtime_expected"], lease, config)
    admission = dict(config=config, permit=permit, config_record=config_record, permit_record=permit_record,
                     output=output, lease=lease, experiment=experiment)
    check_registered_job(admission)
    return admission


def recheck_inputs(admission, initial=False):
    require(read_bound(admission["config_record"]) == admission["config"]
            and read_bound(admission["permit_record"]) == admission["permit"], "Bootstrap inputs changed")
    permit = admission["permit"]
    fresh(permit, time.time(), initial=initial)
    fresh(admission["experiment"], time.time(), initial=initial)
    for path, digest in admission["config"]["source_sha256"].items():
        bound_bytes({"path": path, "sha256": digest})
    for record in (permit["bootstrap_source"], permit["independent_bootstrap_review"],
                   permit["experiment_admission"], permit["pool_extension"]["source"],
                   permit["pool_extension"]["config"], permit["pool_extension"]["independent_review"],
                   admission["experiment"]["ledger"], admission["experiment"]["decisions"]):
        bound_bytes(record)
    require(guard.validate_lease() == admission["lease"], "Inherited lease changed")
    check_parent(permit)
    validate_storage(permit["storage_fields"], admission["output"],
        pool.canonical_path(ROOT, f".runtime/gpup/{permit['pool']}/jobs/{permit['logical_job']}.log"),
        time.time(), initial=initial)
    if initial:
        check_registered_job(admission)
        check_inventory(permit, admission["lease"])
        check_runtime(permit["runtime_expected"], admission["lease"], admission["config"])


def prepare_scope(output):
    output.mkdir(mode=0o700)
    (output / ".launch").mkdir(mode=0o700)
    (output / "logs").mkdir(mode=0o700)
    for name, relative in CACHE_PATHS.items():
        path = pool.canonical_path(ROOT, output / relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    sys.dont_write_bytecode = True
    # Route the complete same-PID runner log inside its exact1GB owned scope.
    fd = os.open(output / "logs/worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


def load_cuda_modules():
    torch = importlib.import_module("torch")
    policy = importlib.import_module("arggnn.development_execution")
    runner = importlib.import_module(RUNNER)
    return torch, policy, runner


def initialize_context(torch):
    require(not torch.cuda.is_initialized(), "Fresh CUDA context required")
    require(torch.cuda.device_count() == 1, "Exactly one visible leased GPU required")
    torch.cuda.set_device(0)
    torch.cuda.init()
    token = torch.empty(1, device="cuda:0")
    torch.cuda.synchronize()
    del token


def write_evidence(output, name, value, raw=False):
    path = pool.canonical_path(ROOT, output / ".launch" / name)
    data = value if raw else pool.encoded(value) + b"\n"
    require(isinstance(data, bytes) and len(data) <= 1024 * 1024, "Bounded launch evidence required")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def validate_adapter_fields(fields, sealed):
    config = read_bound(ADAPTER_CONFIG)
    prepared = read_bound(config["prepared_manifest"])
    observed = read_bound(config["observed_manifest"])
    bound_bytes(config["original_candidate_config"])
    require(observed.get("truth_or_oracle_read") is False
            and prepared["config_sha256"] == config["original_candidate_config"]["sha256"]
            and prepared["observed_manifest_sha256"] == config["observed_manifest"]["sha256"]
            and prepared["source_sha256"] == config["original_preparation_source_sha256"],
            "Original frozen metadata provenance differs")
    expected = dict(kind="root_compact_actual_bank_qualification_admission_v1",
        status="active_root_admission", authorizer="root_agent", project_root=str(ROOT),
        config=ADAPTER_CONFIG, independent_review=ADAPTER_REVIEW, fit_allowed=False, truth_allowed=False,
        devices=["cpu_float64", "cuda_float32"], stages=["parity", "profile_after_parity_pass"],
        expected_process_title=guard.PROCESS_TITLE, selected_train_tiles=sealed.select_train_tiles_v1(prepared))
    require(fields == expected, "Exact root-selected adapter admission fields required")
    return config


def run_bootstrap(admission):
    permit, output = admission["permit"], admission["output"]
    prepare_scope(output)
    title = guard.apply_and_validate_title()
    torch, policy, runner = load_cuda_modules()
    require(not torch.cuda.is_initialized()
            and torch.__version__ == permit["runtime_expected"]["torch"]
            and torch.version.cuda == permit["runtime_expected"]["torch_cuda"], "Exact fresh Torch runtime required")
    require(policy.configure_deterministic_development() == policy.DETERMINISTIC_POLICY,
            "Exact frozen strict numerical policy required")
    adapter_config = validate_adapter_fields(permit["adapter_admission_fields"], runner.sealed)
    runner.check_owned_storage(output)
    recheck_inputs(admission, initial=True)
    initialize_context(torch)
    snapshot = subprocess.run([str(ROOT / ".runtime/env/bin/nvtop"), "--snapshot"],
                              capture_output=True, check=True, timeout=30)
    snapshot_record = write_evidence(output, "nvtop_initial.json", snapshot.stdout, raw=True)
    live = runner.observe_live({"nvtop_snapshot": snapshot_record})
    require(live["lease"] == admission["lease"] and live["pid"] == os.getpid()
            and live["actual_process_title"] == title == guard.PROCESS_TITLE
            and live["actual_context_uuid"] == admission["lease"]["canonical_uuid"]
            and live["logical_device_count"] == 1 and live["logical_device"] == 0
            and all(live[key] == value for key, value in permit["runtime_expected"].items()),
            "Actual own PID/title/UUID/runtime differs from root permit")
    project_devices = sorted({i for i, gpu in enumerate(live["nvtop_snapshot"])
        for process in gpu.get("processes", [])
        if process.get("cmdline", "").strip() == guard.PROCESS_TITLE})
    require(project_devices == permit["project_physical_gpus"], "Live project device inventory differs")
    source_map = dict(adapter_config["source_sha256"])
    source_map.update({"src/arggnn/anchor_compact_qualification_runner_v1.py":
        admission["config"]["source_sha256"]["src/arggnn/anchor_compact_qualification_runner_v1.py"],
        RUNNER_CONFIG["path"]: RUNNER_CONFIG["sha256"]})
    common = dict(status="active_root_admission", authorizer="root_agent", project_root=str(ROOT),
                  runner_config=RUNNER_CONFIG, issued_unix_seconds=time.time())
    binding = dict(bootstrap_source=permit["bootstrap_source"], bootstrap_config=admission["config_record"],
                   root_permit=admission["permit_record"], independent_bootstrap_review=permit["independent_bootstrap_review"],
                   pool_extension=permit["pool_extension"], experiment_admission=permit["experiment_admission"])
    runtime = dict(common, kind="root_compact_runner_runtime_v1", source_sha256=source_map,
                   **{key: live[key] for key in permit["runtime_expected"]},
                   actual_process_title=live["actual_process_title"], pid=live["pid"],
                   actual_context_uuid=live["actual_context_uuid"], device="cuda",
                   execution_policy=live["execution_policy"], bootstrap_execution=binding)
    launcher = dict(common, kind="root_compact_runner_launcher_v1", lease=live["lease"], pid=live["pid"],
        actual_context_uuid=live["actual_context_uuid"], title_before_cuda=title,
        title_before_cuda_verified=True, root_pool_runner_allowlist_integrated=True,
        launch_after_prior_pools_natural_drain=True, nvtop_snapshot=snapshot_record,
        nvtop_exit_code=0, nvtop_executable=str(ROOT / ".runtime/env/bin/nvtop"),
        project_physical_gpus=project_devices, bootstrap_execution=binding)
    storage = permit["storage_fields"]
    runner.validate_domains(runtime, storage, launcher, config_record=RUNNER_CONFIG,
        source_sha256=source_map, output=output, live=live, now=time.time(), initial=True)
    domains = {key: write_evidence(output, key + ".json", value) for key, value in (
        ("runtime_admission", runtime), ("storage_admission", storage), ("launcher_admission", launcher))}
    adapter = dict(permit["adapter_admission_fields"], **domains)
    adapter_record = write_evidence(output, "adapter_admission.json", adapter)
    root = dict(kind="root_compact_qualification_runner_v1_admission", status="active_root_admission",
        authorizer="root_agent", project_root=str(ROOT), runner_config=RUNNER_CONFIG,
        runner_source={"path": "src/arggnn/anchor_compact_qualification_runner_v1.py",
            "sha256": source_map["src/arggnn/anchor_compact_qualification_runner_v1.py"]},
        independent_runner_review=RUNNER_REVIEW, adapter_admission=adapter_record,
        output_scope=str(output), fit_allowed=False, truth_allowed=False, bootstrap_execution=binding, **domains)
    runner_record = write_evidence(output, "runner_admission.json", root)
    runner.check_owned_storage(output)
    recheck_inputs(admission)
    result = runner.main(["--config", RUNNER_CONFIG["path"], "--config-sha256", RUNNER_CONFIG["sha256"],
                          "--admission", runner_record["path"], "--admission-sha256", runner_record["sha256"]])
    recheck_inputs(admission)
    runner.check_owned_storage(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--permit", required=True)
    parser.add_argument("--permit-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    admission = admit_precontext({"path": args.config, "sha256": args.config_sha256},
        {"path": args.permit, "sha256": args.permit_sha256}, args.output)
    return run_bootstrap(admission)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2)
