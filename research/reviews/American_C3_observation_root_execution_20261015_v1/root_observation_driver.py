"""Root-only C3 observed-genotype inference admission and direct CPU wait.

Import, --source-report and --preflight are read-only. Only root may invoke
--admit/--launch after reviewing the frozen source and coordinating admissions.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, ast, hashlib, importlib.util, json, os, selectors, struct
import subprocess, sys, time

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
REL = "research/reviews/American_C3_observation_root_execution_20261015_v1"
HERE = ROOT / REL
SOURCE = REL + "/root_observation_driver.py"
WORKER = "src/arggnn/american_c3_observe_v1.py"
CLOSURE = ["src/arggnn/american_c3_common_v1.py", WORKER,
           "src/arggnn/american_c3_simulate_v1.py"]
GUARD = "research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"
REPAIR = "research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"
BUDGET = "scripts/project_budget_v9.py"
LAUNCHER = "scripts/launch_reserved_gpu_v6.py"
PYTHON = ROOT / ".runtime/envs/biological_tsinfer_v1/bin/python"
STAGES = ("qualification100kb", "full30mb")
LOG_CAP = 16777216
WORK_CAP = 134217728
METADATA_ALLOWANCE = 1048576
STAGE_SECONDS = {"qualification100kb": 600, "full30mb": 75600}
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(".." not in p.parts and p.resolve() == p, "Canonical nonsymlink project path required")
    p.relative_to(ROOT)
    return p


def ref(value):
    p = local(value)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": p.relative_to(ROOT).as_posix(), "sha256": h.hexdigest()}


def read(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}
            and ref(record["path"]) == record, "Exact SHA-bound JSON required")
    return json.loads(local(record["path"]).read_text())


def save(folder, name, value, guard=None, storage=None):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(data) <= METADATA_ALLOWANCE, "Compact root metadata only")
    if guard is not None:
        guard.budget_check_v9(storage, boundary=True, required_write_bytes=len(data) + 8192)
    p = folder / name
    with p.open("xb") as stream:
        stream.write(data)
    return ref(p)


def frozen():
    record = ref(REL + "/frozen_refs.json")
    value = read(record)
    for item in value["refs"].values():
        require(ref(item["path"]) == item, "Frozen source/config/runtime changed: " + item["path"])
    return record, value["refs"]


def npy_header(value):
    # Header metadata only; never create or expose genotype/position arrays.
    with local(value).open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", "NPY format required")
        version = tuple(stream.read(2))
        require(version in ((1, 0), (2, 0), (3, 0)), "Supported NPY header version required")
        size = struct.unpack("<H" if version == (1, 0) else "<I",
                             stream.read(2 if version == (1, 0) else 4))[0]
        require(size <= 65536, "Bounded NPY header required")
        return ast.literal_eval(stream.read(size).decode("latin1"))


def qualification_ready():
    packet = HERE / "admitted_qualification100kb_v1"
    wait_path = packet / "actual_wait.json"
    if not wait_path.exists():
        return False, "Root must run the short 100kb observation inference and obtain direct exit 0"
    wait = read(ref(wait_path))
    require(wait["driver_ref"] == ref(SOURCE) and wait["stage"] == "qualification100kb",
            "Qualification must use this exact root driver")
    if wait["returncode"] != 0 or wait.get("operational_failure") is not None:
        return False, "The short observation qualification failed; preserve its artifacts"
    require(wait["actual_direct_worker_wait"] is True and wait.get("completed_ref"),
            "Actual successful inference wait and completion required")
    completed = read(wait["completed_ref"])
    require(completed["status"] == "completed_current_C3_inference", "Actual inferred completion required")
    observed = read(completed["observation_ref"])
    require(observed["parent_id"] == "american_c3_qualification_000"
            and observed["sequence_length_bp"] == 100000
            and observed["counts_haploid"] == {"AFR": 8, "EUR": 8, "ASIA": 8, "ADMIX": 40}
            and observed["sites_and_sample_identities_verified"] is True
            and observed["query_ancestry_truth_in_inputs"] is False,
            "Complete original short qualification identity required")
    return True, "Short observation inference completed and verified all original sites/samples"


def preflight_v1(stage):
    require(stage in STAGES, "Exact registered stage required")
    frozen_ref, pins = frozen()
    cfg_ref = pins["config_" + stage]
    cfg = read(cfg_ref)
    input_closure_ref = pins["inputs_" + stage]
    inputs = read(input_closure_ref)
    require(cfg["execution_enabled"] is False and cfg["observation_ref"] == inputs["observation_ref"],
            "Only inactive exact observation config may be activated")
    m = read(cfg["observation_ref"])
    expected = inputs["expected_identity"]
    require(m["schema"] == "american_c3_observed_v1"
            and m["parent_id"] == expected["parent_id"]
            and m["sequence_length_bp"] == expected["sequence_length_bp"]
            and m["counts_haploid"] == expected["counts_haploid"]
            and m["class_order"] == ["EUR", "EAS", "AFR"]
            and m["sample_unit"] == "haploid"
            and m["query_ancestry_truth_in_inputs"] is False
            and m["migrations_and_internal_population_labels_removed"] is True,
            "Exact original observed-only C3 identity required")
    require(set(inputs["consumed_data_refs"]) == {"genotypes_ref", "positions_ref", "alleles_ref", "roster_ref"},
            "Only four observed data inputs allowed")
    for key, record in inputs["consumed_data_refs"].items():
        require(m[key] == record and ref(record["path"]) == record,
                "Observed input changed: " + key)
    roster = read(m["roster_ref"])
    rows = roster["samples"]
    h = sum(expected["counts_haploid"].values())
    require(roster["parent_id"] == expected["parent_id"] and roster["class_order"] == m["class_order"]
            and len(rows) == h and [r["sample_index"] for r in rows] == list(range(h)),
            "Stable parent and original sample-column order required")
    for i in range(0, h, 2):
        a, b = rows[i:i + 2]
        require(a["sample_id"] == b["sample_id"] and (a["haplotype"], b["haplotype"]) == (0, 1),
                "Original diploid pairing required")
    for row in rows:
        require(row["age_generations"] == 0 and row["role"] in ("query", "pure_source_reference"),
                "Original contemporary sample roles required")
        require(row["observed_reference_class"] is None if row["role"] == "query"
                else row["observed_reference_class"] in (0, 1, 2), "No query truth labels")
    g = npy_header(m["genotypes_ref"]["path"])
    p = npy_header(m["positions_ref"]["path"])
    sites = expected["sites"]
    require(g["shape"] == (sites, h) and g["descr"] == "|i1" and g["fortran_order"] is False
            and p["shape"] == (sites,) and p["descr"] == "<f8"
            and p["fortran_order"] is False and 0 < sites <= cfg["max_sites"],
            "Exact observed array metadata required")
    runtime = read(pins["runtime"])
    require(runtime["CUDA_VISIBLE_DEVICES"] == "" and runtime["metadata_only"] is True
            and runtime["packages"]["tsinfer"] == "0.5.1" and runtime["packages"]["tsdate"] == "0.2.7",
            "Existing qualified CPU inference runtime required")
    serialized = cfg["max_sample_store_bytes"] + 2 * (cfg["max_tables_bytes"] + 16777216) + 16777216
    planned = serialized + WORK_CAP + LOG_CAP + METADATA_ALLOWANCE
    require(planned <= cfg["output_cap_bytes"], "Serialization/work/log plan exceeds stage cap")
    require(not local(inputs["output_directory"]).exists(), "Fresh stage output required")
    ready, reason = (True, "Root admission and short inference launch remain unperformed")
    if stage == "full30mb":
        ready, reason = qualification_ready()
    return dict(stage=stage, ready_for_root_admission=ready, prerequisite=reason,
                frozen_refs_ref=frozen_ref, input_closure_ref=input_closure_ref,
                inactive_config_ref=cfg_ref, runtime_ref=pins["runtime"],
                source_closure_refs=[ref(path) for path in CLOSURE], driver_ref=ref(SOURCE),
                output_directory=inputs["output_directory"], expected_identity=expected,
                serialized_count_bound_bytes=serialized, planned_total_bytes=planned,
                output_cap_bytes=cfg["output_cap_bytes"], GPU_authorized=False,
                observed_genotype_arrays_loaded=False, oracle_tree_or_private_truth_payload_opened=False)


def runtime_report():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    cmd = [str(PYTHON), "-B", "-m", "arggnn.american_c3_observe_v1", "--runtime-report"]
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True)
    require(len(result.stdout) <= 65536, "Bounded metadata-only runtime report required")
    return json.loads(result.stdout)


def guard_v1(pins):
    require(ref(GUARD) == pins["guard"] and ref(REPAIR) == pins["repair_authority"],
            "Exact current capacity-v2/repair authority required")
    spec = importlib.util.spec_from_file_location("root_C3_observation_capacity_v2", local(GUARD))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_v2(pins["repair_authority"])
    return module


def admit_v1(stage):
    require(Path.cwd().resolve() == ROOT, "Root must execute in repository")
    plan = preflight_v1(stage)
    require(plan["ready_for_root_admission"], plan["prerequisite"])
    _, pins = frozen()
    report = runtime_report()
    runtime = read(pins["runtime"])
    require(report["runtime"] == runtime and report["source_closure_refs"] == plan["source_closure_refs"],
            "Existing runtime or worker closure changed")
    packet = HERE / ("admitted_" + stage + "_v1")
    require(not packet.exists(), "Fresh root packet required; preserve failed/previous packets")
    packet.mkdir()
    cfg = read(plan["inactive_config_ref"])
    cfg["execution_enabled"] = True
    config_ref = save(packet, "config.json", cfg)
    runtime_ref = save(packet, "runtime.json", runtime)
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(hours=22)).isoformat()
    job = "American-C3-observe-" + stage + "-20261015-v1"
    m = guard_v1(pins)
    with m.budget.v4.mutex(ROOT):
        _, _, view, bindings, _ = m._state_locked(
            pins["budget"]["sha256"], pins["launcher"]["sha256"], fresh=True)
    consumers = [*plan["source_closure_refs"], config_ref, runtime_ref, ref(SOURCE),
                 plan["frozen_refs_ref"], plan["input_closure_ref"], plan["inactive_config_ref"],
                 *bindings.values(), pins["guard"], pins["repair_source"], pins["repair_authority"],
                 pins["capacity_v1"], pins["marker"], pins["activation_barrier"]]
    # Frozen v4 accepts .py/.sh/.json consumers. NPY/JSONL input files stay
    # SHA-pinned transitively in the JSON input closure and are checked preflight.
    consumers = list({(r["path"], r["sha256"]): r for r in consumers}.values())
    require(all(Path(r["path"]).suffix in (".py", ".sh", ".json") for r in consumers),
            "Direct consumer binding must use the existing accepted source/JSON formats")
    authority = dict(kind="H039_failed_scope_capacity_root_admission_v1",
        root_authorizer="root_agent", active=True, operation="reserve", accounting_mode=m.MODE,
        bindings=bindings, expected_before=m._pins(view), activation_barrier_ref=pins["activation_barrier"],
        expires_at_utc=expiry, all_budget_writers_use_scheduler_mutex=True,
        all_consumers_and_launchers_use_H039_failed_scope_guard=True,
        all_old_guard_consumers_and_supervisors_terminal_before_activation=True,
        historical_records_mutation_authorized=False, global_rebase_authorized=False,
        scientific_execution_authorized=False, GPU_authorized=False,
        reservation=dict(key="H041/" + job, output=str(ROOT / plan["output_directory"]),
                         auxiliary_scopes=[str(packet)], reserved_bytes=cfg["output_cap_bytes"],
                         consumer_bindings=consumers))
    reserve_ref = save(packet, "reserve_authority.json", authority)
    receipt = m.apply_reserve_v1(reserve_ref, expected_source_sha256=pins["budget"]["sha256"],
                               expected_launcher_sha256=pins["launcher"]["sha256"])
    save(packet, "reserve_receipt.json", receipt)
    storage = dict(kind="project_budget_v9_storage_record", active=True, consumer_uses_v9_guard=True,
        accounting_mode=m.MODE, job_id=job, output_directory=plan["output_directory"],
        reserved_new_bytes=cfg["output_cap_bytes"], reservation=receipt["value"], expires_at_utc=expiry,
        initial_ledger_pins=m._pins(m.inspect_v1(expected_source_sha256=pins["budget"]["sha256"],
                                               expected_launcher_sha256=pins["launcher"]["sha256"])),
        budget_components=dict(implementation=pins["budget"], launcher=pins["launcher"],
                               consumer_adapter=pins["guard"]))
    spec = dict(root_authorizer="root_agent", hypothesis_id="H041", job_id=job,
        execution_authorized=True, GPU_authorized=False, source_ref=ref(WORKER),
        source_closure_refs=plan["source_closure_refs"], config_ref=config_ref, runtime_ref=runtime_ref,
        storage_adapter_ref=pins["guard"], marker_repair_authority_ref=pins["repair_authority"],
        input_closure_ref=plan["input_closure_ref"], driver_ref=ref(SOURCE),
        storage=storage, output_directory=plan["output_directory"],
        created_at_utc=now.isoformat(), expires_at_utc=expiry)
    admission_ref = save(packet, "admission.json", spec, m, storage)
    admitted = dict(stage=stage, admission_ref=admission_ref, driver_ref=ref(SOURCE),
                    output_cap_bytes=cfg["output_cap_bytes"], execution_started=False)
    save(packet, "admitted.json", admitted, m, storage)
    return admitted


def work_bytes(path):
    total = 0
    count = 0
    stack = [path]
    while stack:
        folder = stack.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                count += 1
                require(count <= 50000 and not entry.is_symlink(), "Bounded private work directory required")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    total += entry.stat(follow_symlinks=False).st_size
    return total


def launch_v1(stage):
    plan = preflight_v1(stage)
    require(plan["ready_for_root_admission"], plan["prerequisite"])
    _, pins = frozen()
    packet = HERE / ("admitted_" + stage + "_v1")
    admitted = read(ref(packet / "admitted.json"))
    require(admitted["driver_ref"] == ref(SOURCE), "Exact admitted root driver required")
    spec = read(admitted["admission_ref"])
    require(spec["input_closure_ref"] == plan["input_closure_ref"]
            and read(spec["config_ref"]) == dict(read(plan["inactive_config_ref"]), execution_enabled=True)
            and read(spec["runtime_ref"]) == read(pins["runtime"]), "Exact activated config/runtime/input required")
    require(datetime.fromisoformat(spec["expires_at_utc"]) > datetime.now(timezone.utc),
            "Fresh root admission required")
    m = guard_v1(pins)
    storage = spec["storage"]
    # Root must serialize launches/admissions. Refresh pins immediately before
    # first worker validation; never edit or replay an existing admission.
    storage["initial_ledger_pins"] = m._pins(m.inspect_v1(
        expected_source_sha256=pins["budget"]["sha256"], expected_launcher_sha256=pins["launcher"]["sha256"]))
    admission_ref = save(packet, "launch_admission.json", spec, m, storage)
    m.budget_check_v9(storage, boundary=True, required_write_bytes=32768)
    work = packet / "work"
    (work / "tmp").mkdir(parents=True)
    (work / "numba_cache").mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               TMPDIR=str(work / "tmp"), TMP=str(work / "tmp"), TEMP=str(work / "tmp"),
               NUMBA_CACHE_DIR=str(work / "numba_cache"), XDG_CACHE_HOME=str(work / "cache"))
    cmd = [str(PYTHON), "-B", "-m", "arggnn.american_c3_observe_v1", "--execute",
           "--admission", admission_ref["path"], "--expected-admission-sha256", admission_ref["sha256"]]
    log_path = packet / "launch.log"
    m.budget_check_v9(storage, boundary=True, required_write_bytes=LOG_CAP + 8192)
    started = time.monotonic()
    failure = None
    retained = 0
    terminate_at = None
    with log_path.open("xb") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        save(packet, "launch.json", dict(stage=stage, pid=proc.pid, command=cmd,
             admission_ref=admission_ref, driver_ref=ref(SOURCE), actual_wait_pending=True), m, storage)
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while selector.get_map():
                if failure is None:
                    if time.monotonic() - started > STAGE_SECONDS[stage]:
                        failure = "Registered operational walltime exceeded"
                    elif work_bytes(work) > WORK_CAP:
                        failure = "Private temporary/cache work allowance exceeded"
                    elif datetime.now(timezone.utc) >= datetime.fromisoformat(spec["expires_at_utc"]):
                        failure = "Root admission expired"
                    if failure is not None:
                        proc.terminate()
                        terminate_at = time.monotonic()
                if terminate_at is not None and proc.poll() is None and time.monotonic() - terminate_at > 30:
                    proc.kill()
                for key, _ in selector.select(timeout=1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    if failure is None and retained + len(block) > LOG_CAP:
                        failure = "Retained launch log cap exceeded"
                        proc.terminate()
                        terminate_at = time.monotonic()
                    if failure is None:
                        m.budget_check_v9(storage, boundary=True, required_write_bytes=len(block) + 8192)
                        log.write(block)
                        log.flush()
                        retained += len(block)
            rc = proc.wait()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        finally:
            selector.close()
            proc.stdout.close()
    completed_path = local(spec["output_directory"]) / "completed.json"
    completed_ref = ref(completed_path) if completed_path.exists() else None
    if rc == 0 and completed_ref is None:
        failure = "Exit 0 without inferred completion"
    wait = dict(stage=stage, pid=proc.pid, returncode=rc, operational_failure=failure,
                actual_direct_worker_wait=True, driver_ref=ref(SOURCE), admission_ref=admission_ref,
                completed_ref=completed_ref, log_ref=ref(log_path), retained_log_bytes=retained,
                elapsed_seconds=time.monotonic() - started, private_work_bytes=work_bytes(work),
                finished_utc=datetime.now(timezone.utc).isoformat())
    # A post-exit guard failure must not erase the actual wait. Preserve compact
    # evidence in the already reserved auxiliary root packet, then re-raise.
    try:
        save(packet, "actual_wait.json", wait, m, storage)
    except Exception as exc:
        save(packet, "actual_wait_guard_failure.json", dict(wait=wait, error_type=type(exc).__name__,
             error=str(exc), root_followup_required=True))
        raise
    return wait


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source-report", action="store_true")
    mode.add_argument("--preflight", choices=STAGES)
    mode.add_argument("--admit", choices=STAGES)
    mode.add_argument("--launch", choices=STAGES)
    args = parser.parse_args()
    if args.preflight:
        result = preflight_v1(args.preflight)
    elif args.admit:
        result = admit_v1(args.admit)
    elif args.launch:
        result = launch_v1(args.launch)
    else:
        frozen_ref, pins = frozen()
        result = dict(status="source_only_inactive_root_review_required", driver_ref=ref(SOURCE),
                      frozen_refs_ref=frozen_ref, inference_worker_ref=pins["worker"],
                      source_closure_refs=[ref(p) for p in CLOSURE], stages=list(STAGES),
                      live_guard_configured=False, admission_created=False, inference_executed=False)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.launch and (result["returncode"] != 0 or result["operational_failure"] is not None):
        return result["returncode"] or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
