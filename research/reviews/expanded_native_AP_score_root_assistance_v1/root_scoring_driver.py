"""Root-owned 10MB CPU native AP/H038 VAL scoring utility.

Default: source/metadata report only. Author did not invoke admission or execution.
Root alone invokes --admit/--launch/--run. No model/bank construction or GPU title.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/expanded_native_AP_score_root_assistance_v1"
SELF = HERE / "root_scoring_driver.py"
HELPER = HERE / "root_score_helper.py"
ALIGNED = ROOT / "research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json"
COMPLETED = ROOT / "data/expanded_native_ancestralpaths_v2/completed.json"
UTILITY = ROOT / "research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
UTILITY_SHA = "97ae77c6a5173b787e62b7fd3226c3184729477dfdba94b711d51dca315725ec"
GUARD = "src/arggnn/project_storage_terminal_retired_scopes_v1.py"
GUARD_SHA = "bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
PYTHON = ROOT / ".runtime/env/bin/python"
CAP = 10_000_000
SCHEMA = "expanded_native_AP_H038_VAL_score_input_v1"
VAL_IDS = tuple(f"ap_expanded_val_{i:03d}" for i in range(8))
# Existing source packages only. No installation or runtime dependency mutation.
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def path(value):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    require(".." not in p.parts and p.resolve() == p, "Canonical nonsymlink project path required")
    p.relative_to(ROOT)
    return p


def ref(value):
    p = path(value)
    require(p.suffix in (".py", ".json"), "Source/metadata references only")
    return {"path": p.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def bound(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"},
            "Exact path/SHA reference required")
    p = path(record["path"])
    require(hashlib.sha256(p.read_bytes()).hexdigest() == record["sha256"],
            "Bound source/metadata bytes changed: " + record["path"])
    return p


def read_json(record):
    require(Path(record["path"]).suffix == ".json", "JSON metadata role required")
    return json.loads(bound(record).read_text())


def load_module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cpu():
    require(Path.cwd().resolve() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Root repository and explicit empty CPU CUDA visibility required")


def utilities():
    require(ref(UTILITY)["sha256"] == UTILITY_SHA and ref(GUARD)["sha256"] == GUARD_SHA,
            "Pinned retired root utility/guard changed")
    return load_module(UTILITY, "root_AP_score_retired_utilities")


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}, sort_keys=True), flush=True)


def source_report_v1():
    aligned_ref, completed_ref = ref(ALIGNED), ref(COMPLETED)
    aligned, completed = read_json(aligned_ref), read_json(completed_ref)
    require(aligned["schema"] == "mutation_routed_full_panel_pyg_input_v1" and
            aligned["arm"] == "aligned", "Exact existing aligned H038 metadata")
    require(completed["schema"] == "expanded_native_ap_driver_completed_v2" and
            completed["VAL_count"] == 8 and completed["epochs"] == 5 and
            completed["VAL_selection"] is False and completed["outer_included"] is False,
            "Complete fixed-epoch AP metadata required")
    sources = dict(aligned["source_sha256"])
    ap_cfg = read_json(completed["config_ref"])
    extra = [SELF, HELPER, UTILITY, ROOT / "src/arggnn/project_budget_v9_consumer.py",
             ROOT / "src/arggnn/ancestralpaths_native_bridge.py",
             ROOT / "src/arggnn/expanded_native_ancestralpaths_v1.py",
             ROOT / "src/arggnn/run_expanded_native_ancestralpaths_v2.py"]
    for p in extra:
        record = ref(p)
        if record["path"] in sources:
            require(sources[record["path"]] == record["sha256"], "Original source pin changed")
        sources[record["path"]] = record["sha256"]
    for p, digest in sources.items():
        require(Path(p).suffix == ".py" and ref(p)["sha256"] == digest,
                "H038/AP source closure differs from reviewed bytes")
    require(ap_cfg["source_ref"]["sha256"] ==
            sources["src/arggnn/expanded_native_ancestralpaths_v1.py"],
            "AP fit adapter source pin changed")
    metadata = {
        "aligned_input": aligned_ref, "AP_completed": completed_ref,
        "H038_config": aligned["refs"]["config"],
        "generation_config": aligned["refs"]["generation_config"],
        "registration": aligned["refs"]["registration"],
        "observation_cache": aligned["refs"]["observation_cache"],
        "target_bindings": aligned["refs"]["target_bindings"],
        "AP_config": completed["config_ref"], "AP_fit": completed["fit_ref"],
    }
    for record in metadata.values():
        require(Path(record["path"]).suffix == ".json", "Metadata consumer role required")
        bound(record)
    return {
        "status": "source_only_no_admission_execution_or_arrays",
        "schema": SCHEMA, "source_ref": ref(SELF), "scoring_helper_ref": ref(HELPER),
        "storage_adapter_ref": ref(GUARD), "source_sha256": sources,
        "metadata_refs": metadata, "output_cap_bytes": CAP,
        "GPU_authorized": False, "model_execution_authorized": False,
        "outer_truth_authorized": False, "TRAIN_truth_authorized": False,
        "hypothesis_id": "H038", "new_hypothesis": False,
    }


def runtime_report_v1():
    return {
        "schema": "expanded_native_AP_H038_score_CPU_runtime_v1",
        "prefix": str(Path(sys.prefix).resolve()),
        "python": list(sys.version_info[:3]),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "torch-geometric")},
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "GPU_used": False, "model_constructed": False,
        "source_ref": ref(SELF),
    }


def tag_value(tag):
    require(tag and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag),
            "Simple fresh root packet tag required")
    return tag


def admission_draft_v1(tag="v1"):
    report = source_report_v1()
    tag = tag_value(tag)
    return {
        **report, "status": "draft", "root_authorizer": "root_agent",
        "execution_authorized": False, "VAL_truth_score_authorized": False,
        "job_id": "native-AP-H038-VAL-score-" + tag,
        "output_directory": "experiments/results/expanded_native_AP_H038_VAL_score_" + tag,
        "fixed_VAL_parent_ids": list(VAL_IDS),
        "prospective_registration_ref": report["metadata_refs"]["registration"],
        "runtime_ref": None, "storage": None,
        "all_eight_AP_seals_before_truth": True,
        "scope": "Present182_FP32_C4_primary_plus_separate_native6_SNP_focals",
        "CPU_title_required": False,
    }


def unique(refs):
    values = list({(r["path"], r["sha256"]): r for r in refs}.values())
    require(len(values) <= 64 and all(Path(r["path"]).suffix in (".py", ".json") for r in values),
            "At most64 metadata/source consumer bindings; no scientific payloads")
    return values


def admit_v1(tag="v1"):
    cpu()
    u = utilities()
    folder = HERE / ("admitted_" + tag_value(tag))
    require(not folder.exists(), "Fresh metadata packet required")
    folder.mkdir()
    draft = admission_draft_v1(tag)
    require(not (ROOT / draft["output_directory"]).exists(), "Fresh scoring output required")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [str(PYTHON), "-B", str(SELF), "--runtime-report"]
    process = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    u.save(folder, "runtime_cli.json", {"command": command, "returncode": process.returncode,
                                      "stderr": process.stderr.decode(errors="replace")})
    require(process.returncode == 0, "Metadata-only CPU runtime report failed")
    runtime = json.loads(process.stdout)
    require(runtime["CUDA_VISIBLE_DEVICES"] == "" and runtime["GPU_used"] is False and
            runtime["source_ref"] == draft["source_ref"], "Exact CPU scoring runtime required")
    runtime_ref = u.save(folder, "runtime.json", runtime)
    draft["runtime_ref"] = runtime_ref
    draft_ref = u.save(folder, "input_draft.json", draft)
    consumers = [{"path": p, "sha256": h} for p, h in draft["source_sha256"].items()]
    consumers += list(draft["metadata_refs"].values()) + [runtime_ref, draft_ref]
    authority = u.authority("reserve")
    consumers += list(authority["bindings"].values())
    consumers = unique(consumers)
    authority["reservation"] = {
        "key": "H038/" + draft["job_id"],
        "output": str(ROOT / draft["output_directory"]), "auxiliary_scopes": [],
        "reserved_bytes": CAP, "consumer_bindings": consumers,
    }
    authority_ref = u.save(folder, "reserve_authority.json", authority)
    receipt = u.m.apply_reserve_v1(
        authority_ref, expected_source_sha256=u.CORE, expected_launcher_sha256=u.BASE)
    u.save(folder, "reserve_receipt.json", receipt)
    view = u.inspect()
    expiry = (datetime.now(timezone.utc) + timedelta(hours=23)).isoformat()
    storage = {
        "kind": "project_budget_v9_storage_record", "active": True,
        "consumer_uses_v9_guard": True, "accounting_mode": u.m.MODE,
        "job_id": draft["job_id"], "output_directory": draft["output_directory"],
        "reserved_new_bytes": CAP, "reservation": receipt["value"],
        "expires_at_utc": expiry, "initial_ledger_pins": u.m._pins(view),
        "budget_components": {
            "implementation": u.ref(u.m.budget.SOURCE_REL),
            "launcher": u.ref(u.m.budget.LAUNCHER_REL),
            "consumer_adapter": u.ref(u.m.SOURCE),
        },
    }
    u.save(folder, "storage.json", storage)
    draft.update(status="active", execution_authorized=True,
                 VAL_truth_score_authorized=True, expires_at_utc=expiry,
                 storage=storage, consumer_bindings=consumers)
    input_ref = u.save(folder, "input.json", draft)
    u.save(folder, "admitted.json", {
        "input_ref": input_ref, "runtime_ref": runtime_ref,
        "output_cap_bytes": CAP, "consumer_binding_count": len(consumers),
        "launched": False, "GPU_used": False,
    })
    emit("ROOT_AP_score_admitted", input_ref=input_ref,
         consumer_binding_count=len(consumers), output_cap_bytes=CAP, launched=False)
    return input_ref


class CompactJSONOutputV1:
    """JSON only, exclusive scope, retired guard before every actual write."""
    def __init__(self, spec, adapter):
        self.storage, self.adapter = spec["storage"], adapter
        self.directory = path(spec["output_directory"])
        require(not self.directory.exists(), "Exclusive fresh score output required")
        adapter.budget_check_v9(self.storage, boundary=True, required_write_bytes=8192)
        self.directory.mkdir(parents=False)
        self.used = 0

    def save(self, name, value):
        require(Path(name).name == name and name.endswith(".json"), "Flat JSON output only")
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        planned = len(payload) + 8192
        require(self.used + planned <= CAP, "Compact scoring output cap reached")
        self.adapter.budget_check_v9(self.storage, boundary=True, required_write_bytes=planned)
        require(planned <= os.statvfs(self.directory).f_bavail * os.statvfs(self.directory).f_frsize,
                "Filesystem free bytes insufficient")
        fd = os.open(self.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        self.used += planned
        return {"path": (self.directory / name).relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest()}


def execute_v1(input_ref):
    cpu()
    spec = read_json(input_ref)
    report = source_report_v1()
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    require(spec["schema"] == SCHEMA and spec["status"] == "active" and
            spec["root_authorizer"] == "root_agent" and spec["execution_authorized"] is True and
            spec["VAL_truth_score_authorized"] is True and
            spec["GPU_authorized"] is False and spec["TRAIN_truth_authorized"] is False and
            spec["outer_truth_authorized"] is False and spec["model_execution_authorized"] is False and
            spec["output_cap_bytes"] == CAP and spec["fixed_VAL_parent_ids"] == list(VAL_IDS) and
            expiry.tzinfo is not None and expiry > datetime.now(timezone.utc),
            "Active root CPU VAL-only scoring capability required")
    for key in ("source_ref", "scoring_helper_ref", "storage_adapter_ref",
                "source_sha256", "metadata_refs"):
        require(spec[key] == report[key], "Frozen scoring source/metadata closure changed: " + key)
    runtime = read_json(spec["runtime_ref"])
    require(runtime == runtime_report_v1(), "Admitted scoring runtime changed")
    consumers = unique(spec["storage"]["reservation"]["consumer_bindings"])
    required = [{"path": p, "sha256": h} for p, h in spec["source_sha256"].items()]
    required += list(spec["metadata_refs"].values()) + [spec["runtime_ref"]]
    require(all(r in consumers for r in required), "Scoring closure absent from own reservation")

    from arggnn import census_tsinfer_fit_v1 as fit
    from arggnn import project_storage_terminal_retired_scopes_v1 as adapter
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1
    reader = fit.ConsumerReaderV1()
    adapter.validate_storage_v9(
        spec, {"output_cap_bytes": CAP}, spec["storage"], source_ref=spec["source_ref"],
        config_ref=spec["metadata_refs"]["H038_config"], runtime_ref=spec["runtime_ref"])
    output = CompactJSONOutputV1(spec, adapter)
    started = time.monotonic()
    parent_reports = {}
    try:
        output.save("started.json", {"input_ref": input_ref, "started_utc": utc(),
                    "worker_pid": os.getpid(), "GPU_used": False, "output_cap_bytes": CAP,
                    "model_or_backbone_bank_constructed": False})
        emit("AP_score_started", worker_pid=os.getpid(), output_cap_bytes=CAP)
        aligned = read_json(spec["metadata_refs"]["aligned_input"])
        generation = read_json(spec["metadata_refs"]["generation_config"])
        registration = read_json(spec["metadata_refs"]["registration"])
        require(generation["parent_registry"] == registration["parent_registry"] and
                registration["hypothesis_id"] == "H038", "Original H038 registry differs")
        registry = generation["parent_registry"]
        require([r["parent_id"] for r in registry if r["split"] == "val"] == list(VAL_IDS),
                "Exact registered internal eight VAL parents")
        require(all(spec["metadata_refs"][k] == aligned["refs"][a] for k, a in (
                    ("H038_config", "config"), ("generation_config", "generation_config"),
                    ("registration", "registration"), ("observation_cache", "observation_cache"),
                    ("target_bindings", "target_bindings"))),
                "Scoring metadata must reuse original aligned H038 refs")
        index = read_full_cache_index_v1(
            spec["metadata_refs"]["observation_cache"], reader, registry=registry,
            entry_refs=aligned["observation_entry_refs"],
            source_refs=aligned["observation_cache_source_refs"])
        helper = load_module(bound(spec["scoring_helper_ref"]), "root_AP_score_functions")
        parents = {}
        for pid in VAL_IDS:
            parents[pid] = helper.read_h038_scoring_only_v1(index, pid, reader)
            emit("H038_scoring_fields_loaded", parent_id=pid, model_bank_constructed=False)
        all_bindings = read_json(spec["metadata_refs"]["target_bindings"])["parents"]
        bindings = {p: all_bindings[p] for p in VAL_IDS}
        del all_bindings

        def on_parent(pid, value):
            parent_reports[pid] = output.save(pid + ".json", value)
            emit("AP_parent_scored", parent_id=pid,
                 primary=value["groups"]["primary_present_day_182"])

        def on_stage(stage, pid=None):
            emit(stage, **({"parent_id": pid} if pid is not None else {}))

        result = helper.score_root_v1(
            completed_ref=spec["metadata_refs"]["AP_completed"],
            parents=parents, bindings=bindings, reader=reader,
            helpers=fit.metric_helpers_v1(), on_parent=on_parent, on_stage=on_stage)
        summary_ref = output.save("summary.json", result)
        completed_ref = output.save("completed.json", {
            "schema": "expanded_native_AP_H038_VAL_score_completed_v1",
            "input_ref": input_ref, "source_ref": spec["source_ref"],
            "scoring_helper_ref": spec["scoring_helper_ref"], "summary_ref": summary_ref,
            "parent_report_refs": parent_reports, "VAL_parents": 8,
            "C4_primary": result["C4_primary_equal_parent"],
            "native6_scope": result["native6_scope"], "GPU_used": False,
            "outer_or_TRAIN_truth_read": False, "model_or_backbone_bank_constructed": False,
            "completed_utc": utc(), "elapsed_seconds": time.monotonic() - started,
            "worker_pid": os.getpid(), "output_cap_bytes": CAP})
        emit("AP_score_completed", completed_ref=completed_ref,
             C4_primary=result["C4_primary_equal_parent"])
        return completed_ref
    except BaseException as error:
        failure = {"stage": "AP_score_failed", "type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(),
                   "worker_pid": os.getpid(), "completed_parent_reports": parent_reports,
                   "elapsed_seconds": time.monotonic() - started, "GPU_used": False}
        emit("AP_score_failed", **{k: v for k, v in failure.items() if k != "stage"})
        try:
            output.save("failed.json", failure)
        except BaseException as publication_error:
            emit("AP_failure_receipt_unpublished", error=str(publication_error))
        raise


def launch_v1(tag="v1"):
    cpu()
    u = utilities()
    folder = HERE / ("admitted_" + tag_value(tag))
    admitted = read_json(ref(folder / "admitted.json"))
    value = read_json(admitted["input_ref"])
    require(not (ROOT / value["output_directory"]).exists(), "No replay or existing output")
    value["storage"]["initial_ledger_pins"] = u.m._pins(u.inspect())
    u.save(folder, "launch_storage.json", value["storage"])
    input_ref = u.save(folder, "launch_input.json", value)
    command = [str(PYTHON), "-B", str(SELF), "--run", "--input", input_ref["path"],
               "--expected-input-sha256", input_ref["sha256"]]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (folder / "launch.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    u.save(folder, "launch.json", {"command": command, "worker_pid": child.pid,
           "input_ref": input_ref, "launched_utc": utc(), "actual_wait_pending": True,
           "GPU_used": False, "scientific_completion_claimed": False})
    emit("ROOT_AP_score_launched", worker_pid=child.pid, input_ref=input_ref)
    interruption = None
    try:
        returncode = child.wait()
    except BaseException as error:
        interruption = {"type": type(error).__name__, "error": str(error)}
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        returncode = child.wait()
    completed = ROOT / value["output_directory"] / "completed.json"
    failed = ROOT / value["output_directory"] / "failed.json"
    receipt = {
        "schema": "ROOT_AP_score_actual_wait_v1", "worker_pid": child.pid,
        "returncode": returncode, "actual_child_waited": True, "input_ref": input_ref,
        "completed_ref": ref(completed) if completed.is_file() else None,
        "failed_ref": ref(failed) if failed.is_file() else None,
        "stdout_log_ref": {
            "path": (folder / "launch.log").relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256((folder / "launch.log").read_bytes()).hexdigest(),
        },
        "interruption": interruption, "waited_utc": utc(), "GPU_used": False,
        "success": returncode == 0 and completed.is_file() and interruption is None,
    }
    u.save(folder, "root_wait.json", receipt)  # Fixed name and actual Popen.wait exit.
    emit("ROOT_AP_score_actual_wait", **receipt)
    require(receipt["success"], "AP scoring failed; see fixed root_wait.json and launch.log")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--admit", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--admission-draft", action="store_true")
    action.add_argument("--runtime-report", action="store_true")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args()
    try:
        if args.admit:
            admit_v1(args.tag)
        elif args.launch:
            launch_v1(args.tag)
        elif args.run:
            require(args.input and args.expected_input_sha256, "Bound root input required")
            execute_v1({"path": args.input, "sha256": args.expected_input_sha256})
        elif args.admission_draft:
            print(json.dumps(admission_draft_v1(args.tag), sort_keys=True, indent=2))
        elif args.runtime_report:
            print(json.dumps(runtime_report_v1(), sort_keys=True, indent=2))
        else:
            print(json.dumps(source_report_v1(), sort_keys=True, indent=2))
    except BaseException as error:
        emit("ROOT_AP_score_driver_error", type=type(error).__name__, error=str(error),
             traceback=traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
