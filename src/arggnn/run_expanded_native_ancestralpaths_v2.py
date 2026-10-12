"""Root-executable CPU H038 released AncestralPaths preparation and final fit.

Default is a source report. Root supplies exact input SHA, metadata/target refs,
qualified runtime and terminal-retired storage capability. No action at import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = "src/arggnn/run_expanded_native_ancestralpaths_v2.py"
NATIVE = "src/arggnn/expanded_native_ancestralpaths_v1.py"
CONFIG = "research/reviews/expanded_native_ancestralpaths_v1_author/config.draft.json"
STORAGE = "src/arggnn/project_storage_terminal_retired_scopes_v1.py"
MODE = "journal_terminal_retired_scopes_credit_v1"
SCHEMA = "expanded_native_ancestralpaths_driver_v2_root_admission"
PACKAGES = ("tensorflow-cpu", "keras", "numpy", "tskit", "msprime", "demes", "pandas")


def _path(value):
    path = Path(value)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    if not path.parts or ".." in path.parts:
        raise ValueError("Dedicated project path required")
    for n in range(1, len(path.parts) + 1):
        if ROOT.joinpath(*path.parts[:n]).is_symlink():
            raise ValueError("Project artifact symlink forbidden")
    result = ROOT / path
    result.resolve().relative_to(ROOT)
    return result


def _ref(value):
    path = _path(value)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=h.hexdigest())


def _read(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Exact project reference required")
    payload = _path(ref["path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise ValueError("Bound artifact bytes differ: " + ref["path"])
    return json.loads(payload)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _peak_rss():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def source_report_v2():
    cfg = _read(_ref(CONFIG))
    paths = {SOURCE, NATIVE, STORAGE, "src/arggnn/project_storage_closed_scopes_v1.py",
        "src/arggnn/project_storage_cpu_closed_scopes_v1.py",
        "src/arggnn/project_storage_terminal_scopes_v1.py",
        "src/arggnn/project_storage_terminal_success_scopes_v1.py",
        "src/arggnn/project_budget_v9_consumer.py",
        "scripts/project_budget_v9.py", "scripts/launch_reserved_gpu_v6.py"}
    paths.update(ref["path"] for ref in cfg["helper_refs"].values())
    return dict(status="source_only_no_execution", schema=SCHEMA, source_ref=_ref(SOURCE),
        config_ref=_ref(CONFIG), storage_adapter_ref=_ref(STORAGE),
        source_sha256={path: _ref(path)["sha256"] for path in sorted(paths)},
        generation_config_ref=cfg["generation_config_ref"], registration_ref=cfg["registration_ref"],
        qualified_prefix=cfg["tensorflow_cpu_prefix"], accounting_mode=MODE,
        TRAIN_parents=32, VAL_parents=8, outer_included=False, VAL_selection=False)


def admission_draft_v2():
    """Metadata-only capability template; never hash/open a target archive."""
    report = source_report_v2()
    cfg = _read(report["config_ref"])
    registry = _read(cfg["registration_ref"])["parent_registry"]
    completed, targets, pending = {}, {}, []
    for row in registry:
        if row["split"] == "outer":
            continue
        parent_id = row["parent_id"]
        base = "data/expanded_full_panel_v1/" + parent_id
        path = _path(base + "/expanded_parent_completed.json")
        if path.is_file():
            completed[parent_id] = _ref(path)
            value = _read(completed[parent_id])
            if value.get("Relate_completed") is not True:
                pending.append(parent_id)
            if row["split"] == "train":
                generation = _read(value["generation_ref"])
                targets[parent_id] = dict(path=base + "/truth/full_truth_intervals.npz",
                    sha256=generation["files"]["truth/full_truth_intervals.npz"]["sha256"])
        else:
            completed[parent_id] = dict(path=base + "/expanded_parent_completed.json",
                                       sha256="ROOT_BIND_AFTER_COMPLETION")
            pending.append(parent_id)
    return dict(schema=SCHEMA, root_authorizer="root_agent", execution_authorized=False,
        status="draft_not_admitted", job_id="H038-expanded-native-ancestralpaths-v2",
        output_directory="data/expanded_native_ancestralpaths_v2", output_cap_bytes=16000000000,
        expires_at_utc="ROOT_SET_ACTIVE_EXPIRY", source_ref=report["source_ref"],
        config_ref=report["config_ref"], source_sha256=report["source_sha256"],
        storage_adapter_ref=report["storage_adapter_ref"],
        runtime_ref=dict(path="research/reviews/expanded_native_ancestralpaths_driver_v2_author/runtime.json",
                         sha256="ROOT_BIND_QUALIFIED_RUNTIME"),
        completion_refs=completed, TRAIN_truth_refs=targets, pending_parent_ids=pending,
        TRAIN_targets_authorized=True, VAL_selection=False, outer_included=False, GPU_authorized=False,
        storage="ROOT_BIND_ACTIVE_TERMINAL_RETIRED_STORAGE_RECORD")


def runtime_report_v2():
    cfg = _read(_ref(CONFIG))
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(schema="expanded_native_ap_cpu_runtime_v2", executable=sys.executable,
        prefix=str(Path(sys.prefix).resolve()), qualified_prefix=cfg["tensorflow_cpu_prefix"],
        packages=versions, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_ref=_ref(SOURCE), GPU_used=False)


def _admit(spec, *, native_runtime):
    report = source_report_v2()
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    if (spec.get("schema") != SCHEMA or spec.get("root_authorizer") != "root_agent"
            or spec.get("execution_authorized") is not True or spec.get("status") != "active"
            or spec.get("outer_included") is not False or spec.get("VAL_selection") is not False
            or spec.get("GPU_authorized") is not False or spec.get("TRAIN_targets_authorized") is not True
            or expiry.tzinfo is None or expiry <= datetime.now(timezone.utc)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or Path.cwd().resolve() != ROOT):
        raise ValueError("Active root-admitted CPU native TRAIN capability required")
    if (spec["source_ref"] != report["source_ref"] or spec["config_ref"] != report["config_ref"]
            or spec["storage_adapter_ref"] != report["storage_adapter_ref"]
            or spec["source_sha256"] != report["source_sha256"]):
        raise ValueError("Exact frozen driver/native/config/storage sources required")
    cfg = _read(spec["config_ref"])
    registration = _read(cfg["registration_ref"])
    generation = _read(cfg["generation_config_ref"])
    registry = registration["parent_registry"]
    if registry != generation["parent_registry"]:
        raise ValueError("Exact expanded registered biology required")
    train = [r["parent_id"] for r in registry if r["split"] == "train"]
    val = [r["parent_id"] for r in registry if r["split"] == "val"]
    if len(train) != 32 or len(val) != 8 or set(spec["completion_refs"]) != set(train + val):
        raise ValueError("Exactly completed32 TRAIN and8 VAL references required")
    if set(spec["TRAIN_truth_refs"]) != set(train):
        raise ValueError("External truth capabilities must be exactly32 TRAIN parents")
    for parent_id in train + val:
        ref = spec["completion_refs"][parent_id]
        base = generation["output_root"] + "/" + parent_id
        if ref["path"] != base + "/expanded_parent_completed.json":
            raise ValueError("Fixed registered completion path required")
        completed = _read(ref)
        if (completed["parent_id"] != parent_id or completed["Relate_completed"] is not True
                or completed["generation_completed"] is not True or completed["outer_truth_evaluated"] is not False):
            raise ValueError("Completed registered inner parent required")
        if parent_id in train:
            target = spec["TRAIN_truth_refs"][parent_id]
            original = _read(completed["generation_ref"])
            if (target["path"] != base + "/truth/full_truth_intervals.npz"
                    or target["sha256"] != original["files"]["truth/full_truth_intervals.npz"]["sha256"]):
                raise ValueError("Exact producer-bound external TRAIN target required")
    runtime = _read(spec["runtime_ref"])
    if (runtime.get("schema") != "expanded_native_ap_cpu_runtime_v2"
            or runtime.get("source_ref") != spec["source_ref"]
            or runtime.get("qualified_prefix") != cfg["tensorflow_cpu_prefix"]
            or runtime.get("prefix") != str(_path(cfg["tensorflow_cpu_prefix"]))
            or runtime.get("CUDA_VISIBLE_DEVICES") != "" or runtime.get("GPU_used") is not False):
        raise ValueError("Bound qualified native CPU runtime report required")
    if native_runtime:
        actual = runtime_report_v2()
        if (Path(sys.prefix).resolve() != _path(cfg["tensorflow_cpu_prefix"])
                or runtime["packages"] != actual["packages"]
                or actual["packages"]["tensorflow-cpu"] != "2.15.1"
                or actual["packages"]["keras"] != "2.15.0"):
            raise ValueError("Execute in the qualified TF2.15.1/Keras2.15 CPU environment")
    if type(spec.get("output_cap_bytes")) is not int or spec["output_cap_bytes"] <= 0:
        raise ValueError("Positive root-admitted output cap required")
    storage = spec["storage"]
    if storage["accounting_mode"] != MODE or storage["output_directory"] != spec["output_directory"]:
        raise ValueError("Exact terminal-retired output capability required")
    refs = [{"path": path, "sha256": sha} for path, sha in report["source_sha256"].items()]
    refs += [spec["config_ref"], spec["runtime_ref"], cfg["generation_config_ref"], cfg["registration_ref"]]
    if any(ref not in storage["reservation"]["consumer_bindings"] for ref in refs):
        raise ValueError("Reservation must bind exact source/config/runtime/registry closure")
    return cfg, train, val, expiry, _path(spec["output_directory"])


def _authority(spec, cfg, *, first):
    adapter = importlib.import_module("arggnn.project_storage_terminal_retired_scopes_v1")
    if first:
        adapter.validate_storage_v9(spec, dict(cfg, output_cap_bytes=spec["output_cap_bytes"]), spec["storage"],
            source_ref=spec["source_ref"], config_ref=spec["config_ref"], runtime_ref=spec["runtime_ref"])
    else:
        adapter.budget_check_v9(spec["storage"], boundary=True)
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    def guard(planned=0):
        if datetime.now(timezone.utc) >= expiry:
            raise ValueError("Native execution admission expired")
        return adapter.budget_check_v9(spec["storage"], boundary=True, required_write_bytes=int(planned))
    return guard


def _save(path, value, guard):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    guard(len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return _ref(path)


def execute_v2(spec, input_ref, *, supervised=False):
    cfg, train, val, _, output = _admit(spec, native_runtime=True)
    if supervised:
        launch = _read(_ref(output / "supervisor/launch.json"))
        if launch["input_ref"] != input_ref or launch["source_ref"] != spec["source_ref"]:
            raise ValueError("Exact supervisor-to-worker admission handoff required")
    elif output.exists():
        raise FileExistsError("Existing native outputs preserved; use a fresh admitted output")
    guard = _authority(spec, cfg, first=not supervised)
    if not output.exists():
        guard(16384); output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    guard(32768); run.mkdir(exist_ok=False)
    scratch = output / "runtime_tmp"
    scratch.mkdir(exist_ok=False)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    _save(run / "started.json", dict(schema="expanded_native_ap_driver_started_v2",
        input_ref=input_ref, source_ref=spec["source_ref"], worker_pid=os.getpid(), started_utc=_utc(),
        TRAIN_parents=train, VAL_parents=val, native_epochs=5, VAL_selection=False), guard)
    native = importlib.import_module("arggnn.expanded_native_ancestralpaths_v1")
    features, training, timings = {}, {}, {}
    started = time.monotonic()
    try:
        for parent_id in train + val:
            tick = time.monotonic()
            observed = native.read_completed_parent_v1(parent_id, spec["completion_refs"][parent_id], config=cfg)
            feature = native.prepare_parent_v1(observed, run / "features" / parent_id, config=cfg, guard=guard)
            features[parent_id] = feature["manifest_ref"]
            if parent_id in train:
                rows = native.prepare_train_rows_v1(observed, feature["manifest_ref"],
                    spec["TRAIN_truth_refs"][parent_id], run / "training" / parent_id, config=cfg, guard=guard)
                training[parent_id] = rows["manifest_ref"]
            timings[parent_id] = time.monotonic() - tick
            _save(run / (parent_id + "_prepared.json"), dict(parent_id=parent_id,
                features_ref=features[parent_id], training_ref=training.get(parent_id),
                elapsed_seconds=timings[parent_id], peak_cpu_rss_bytes=_peak_rss()), guard)
            print(json.dumps(dict(stage="parent_prepared", parent_id=parent_id, elapsed_seconds=timings[parent_id])),
                  flush=True)
            del observed, feature
        prep_ref = _save(run / "prepared.json", dict(schema="expanded_native_ap_driver_prepared_v2",
            input_ref=input_ref, feature_refs=features, TRAIN_training_refs=training,
            parent_preparation_seconds=timings, preparation_elapsed_seconds=time.monotonic() - started), guard)
        tick = time.monotonic()
        model, fit = native.fit_all_train_v1(training, run / "fit", config=cfg, guard=guard)
        fit_seconds = time.monotonic() - tick
        seals = {}
        # Every VAL seal reloads the fixed final checkpoint. No VAL truth is bound.
        for parent_id in val:
            seal = native.seal_parent_predictions_v1(fit["manifest_ref"], features[parent_id],
                run / "VAL_predictions" / parent_id, config=cfg, guard=guard)
            seals[parent_id] = seal["manifest_ref"]
            print(json.dumps(dict(stage="VAL_raw_sealed", parent_id=parent_id, seal_ref=seals[parent_id])), flush=True)
        result = dict(schema="expanded_native_ap_driver_completed_v2", input_ref=input_ref,
            source_ref=spec["source_ref"], config_ref=spec["config_ref"], runtime_ref=spec["runtime_ref"],
            prepared_ref=prep_ref, fit_ref=fit["manifest_ref"], VAL_prediction_seals=seals,
            TRAIN_count=32, VAL_count=8, epochs=5, checkpoint_selection="final_epoch5",
            VAL_selection=False, VAL_targets_read=False, outer_included=False, GPU_used=False,
            native_requested_training_rows=5200000, realized_training_rows=fit["training_rows"],
            sampler_shortfall_rows=fit["shortfall_rows"], fit_elapsed_seconds=fit_seconds,
            total_elapsed_seconds=time.monotonic() - started, peak_cpu_rss_bytes=_peak_rss(),
            completed_utc=_utc(), worker_pid=os.getpid())
        _save(output / "completed.json", result, guard)
        return result
    except BaseException as error:
        failure = dict(schema="expanded_native_ap_driver_failed_v2", input_ref=input_ref,
            exception_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            prepared_parent_ids=list(features), elapsed_seconds=time.monotonic() - started,
            peak_cpu_rss_bytes=_peak_rss(), worker_pid=os.getpid())
        print(json.dumps(failure), file=sys.stderr, flush=True)
        try:
            _save(output / "failed.json", failure, guard)
        except BaseException as receipt_error:
            print("Failure receipt could not be written: " + str(receipt_error), file=sys.stderr, flush=True)
        raise


def supervise_v2(spec, input_ref):
    cfg, _, _, _, output = _admit(spec, native_runtime=False)
    if output.exists():
        raise FileExistsError("Supervisor requires fresh admitted output; prior outputs preserved")
    guard = _authority(spec, cfg, first=True)
    guard(32768); output.mkdir(parents=True, exist_ok=False)
    directory = output / "supervisor"; directory.mkdir()
    command = [str(_path(cfg["tensorflow_cpu_prefix"]) / "bin/python"), "-B", "-m",
        "arggnn.run_expanded_native_ancestralpaths_v2", "--execute", "--supervised",
        "--input", input_ref["path"], "--expected-input-sha256", input_ref["sha256"]]
    _save(directory / "launch.json", dict(schema="expanded_native_ap_supervisor_launch_v2",
        input_ref=input_ref, source_ref=spec["source_ref"], command=command,
        supervisor_pid=os.getpid(), launched_utc=_utc(), CPU_only=True), guard)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    captured = bytearray()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(json.dumps(dict(stage="native_worker_launched", worker_pid=process.pid)), flush=True)
    try:
        for line in iter(process.stdout.readline, b""):
            captured.extend(line)
            sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise
    # No supervisor filesystem writer runs concurrently with child scope guards.
    guard(len(captured) + 8192)
    with (directory / "launch.log").open("xb") as stream:
        stream.write(captured); stream.flush(); os.fsync(stream.fileno())
    completed_ref = _ref(output / "completed.json") if (output / "completed.json").is_file() else None
    _save(directory / "launcher_exit.json", dict(schema="expanded_native_ap_supervisor_exit_v2",
        input_ref=input_ref, worker_pid=process.pid, supervisor_pid=os.getpid(), returncode=returncode,
        completed_ref=completed_ref, worker_waited=True, CPU_only=True, completed_utc=_utc()), guard)
    return returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--runtime-report", action="store_true")
    mode.add_argument("--admission-draft", action="store_true")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if args.runtime_report:
        print(json.dumps(runtime_report_v2(), indent=2, sort_keys=True)); return
    if args.admission_draft:
        print(json.dumps(admission_draft_v2(), indent=2, sort_keys=True)); return
    if not args.execute and not args.supervise:
        print(json.dumps(source_report_v2(), indent=2, sort_keys=True)); return
    if not args.input or not args.expected_input_sha256:
        parser.error("Root --input and --expected-input-sha256 required")
    input_ref = dict(path=args.input, sha256=args.expected_input_sha256)
    spec = _read(input_ref)
    if args.supervise:
        raise SystemExit(supervise_v2(spec, input_ref))
    result = execute_v2(spec, input_ref, supervised=args.supervised)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
