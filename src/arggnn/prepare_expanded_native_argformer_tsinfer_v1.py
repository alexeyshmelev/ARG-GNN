"""Root CPU preparation supervisor for H038 native tsinfer/tsdate ARGformer.

Default metadata only. Separate qualified CPU processes observe then prepare each
admitted inner ARG. Primary fitting belongs to8 independent VAL ARGs;
32 TRAIN remain registry metadata. No fit, GPU, truth capability, retirement or install.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from . import expanded_native_argformer_tsinfer_v1 as native

ROOT = native.ROOT
SOURCE = "src/arggnn/prepare_expanded_native_argformer_tsinfer_v1.py"
ADAPTER = native.SOURCE
CONFIG = "research/reviews/expanded_native_argformer_tsinfer_v1_author/config.draft.json"
STORAGE = "src/arggnn/project_storage_terminal_retired_scopes_v1.py"
MODE = "journal_terminal_retired_scopes_credit_v1"
SCHEMA = "expanded_native_argformer_tsinfer_preparation_v1_root_admission"
STORAGE_PATHS = (STORAGE, "src/arggnn/project_storage_terminal_success_scopes_v1.py",
    "src/arggnn/project_storage_terminal_scopes_v1.py", "src/arggnn/project_storage_cpu_closed_scopes_v1.py",
    "src/arggnn/project_storage_closed_scopes_v1.py", "scripts/project_budget_v9.py",
    "scripts/launch_reserved_gpu_v6.py", "src/arggnn/project_budget_v9_consumer.py")


def source_report_v1():
    cfg = native.default_config_v1()
    paths = set(STORAGE_PATHS + (SOURCE, ADAPTER, native.FROZEN, native.PIPELINE, native.PRODUCER,
                                native.ORIGINAL, native.CONTRACTS))
    paths.update(ref["path"] for ref in cfg["helper_refs"].values())
    return dict(status="source_only_no_launch", schema=SCHEMA, source_ref=native._ref(SOURCE),
        config_ref=native._ref(CONFIG), adapter_ref=native._ref(ADAPTER),
        storage_adapter_ref=native._ref(STORAGE), accounting_mode=MODE,
        source_sha256={p: native._ref(p)["sha256"] for p in sorted(paths)},
        observation_runtime_ref=cfg["observation"]["runtime_ref"],
        generation_config_ref=cfg["observation"]["generation_config_ref"],
        registration_ref=cfg["observation"]["registration_ref"], parent_order=list(native.PARENTS),
        CPU_workers=1, GPU_authorized=False, fit_authorized=False, query_truth_read=False,
        primary_fit_parent_ids=list(native.VAL_PARENTS),
        execution_scopes=["canary", "primary_VAL", "all40_future_protocol_CPU_only"])


def admission_draft_v1():
    cfg = native.default_config_v1(); report = source_report_v1()
    refs, pending = {}, []
    for parent in native.PARENTS:
        p = native._path("data/expanded_full_panel_v1/" + parent + "/expanded_parent_completed.json")
        if p.is_file():
            refs[parent] = native._ref(p)
            value = native._read(refs[parent])
            if value.get("Relate_completed") is not True:
                pending.append(parent)
        else:
            refs[parent] = dict(path=p.relative_to(ROOT).as_posix(), sha256="ROOT_BIND_AFTER_COMPLETION")
            pending.append(parent)
    return dict(schema=SCHEMA, status="draft_not_admitted", root_authorizer="root_agent",
        execution_authorized=False, fit_authorized=False, GPU_authorized=False,
        truth_authorized=False, outer_included=False, parent_order=list(native.PARENTS), workers=1,
        job_id="H038-native-argformer-tsinfer-preparation-v1", source_ref=report["source_ref"],
        config_ref=report["config_ref"], adapter_ref=report["adapter_ref"],
        source_sha256=report["source_sha256"], storage_adapter_ref=report["storage_adapter_ref"],
        observation_runtime_ref=report["observation_runtime_ref"],
        preparation_runtime_ref=dict(path="research/reviews/expanded_native_argformer_tsinfer_v1_author/preparation_runtime.json",
                                     sha256="ROOT_BIND_QUALIFIED_PREPARATION_RUNTIME_REPORT"),
        completion_refs=refs, pending_parent_ids=pending,
        execution_scope="canary", execution_parent_ids=[native.VAL_PARENTS[0]], native_canary_ref=None,
        output_directory="data/expanded_native_argformer_tsinfer_canary_v1",
        output_cap_bytes=cfg["resource_planning"]["per_parent_output_cap_bytes"],
        resource_caps={k: cfg["resource_planning"][k] for k in (
            "per_parent_output_cap_bytes", "CPU_RSS_cap_bytes", "per_stage_wall_seconds",
            "maximum_native_nodes", "maximum_native_trees", "maximum_native_edges",
            "maximum_native_sites", "maximum_full_path_occurrences", "maximum_unique_full_paths")},
        expires_at_utc="ROOT_SET_ACTIVE_EXPIRY",
        storage="ROOT_BIND_ACTIVE_TERMINAL_RETIRED_STORAGE_RECORD",
        resource_planning=cfg["resource_planning"], activation_requires_success_writers_finished=True)


def _admit(input_ref):
    spec = native._read(input_ref); report = source_report_v1()
    cfg = native._read(spec["config_ref"])
    native._registry(cfg)
    expiry = datetime.fromisoformat(spec["expires_at_utc"].replace("Z", "+00:00"))
    native._require(spec.get("schema") == SCHEMA and spec.get("status") == "active"
        and spec.get("root_authorizer") == "root_agent" and spec.get("execution_authorized") is True
        and spec.get("fit_authorized") is False and spec.get("GPU_authorized") is False
        and spec.get("truth_authorized") is False and spec.get("outer_included") is False
        and spec.get("workers") == 1 and spec.get("parent_order") == list(native.PARENTS)
        and spec.get("activation_requires_success_writers_finished") is True
        and expiry.tzinfo is not None and expiry > datetime.now(timezone.utc)
        and Path.cwd().resolve() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "Active root CPU/observations-only native preparation admission required")
    for key in ("source_ref", "config_ref", "adapter_ref", "storage_adapter_ref", "source_sha256", "observation_runtime_ref"):
        native._require(spec[key] == report[key], "Exact source/runtime/config binding required: " + key)
    native._require(spec["resource_planning"] == cfg["resource_planning"]
        and type(spec["output_cap_bytes"]) is int and 0 < spec["output_cap_bytes"] <=
        cfg["resource_planning"]["total_preparation_cap_bytes"]
        and set(spec["completion_refs"]) == set(native.PARENTS), "Exact 40-parent preparation/resource plan required")
    scopes = {"canary": [native.VAL_PARENTS[0]], "primary_VAL": list(native.VAL_PARENTS),
              "all40_future_protocol_CPU_only": list(native.PARENTS)}
    native._require(spec.get("execution_scope") in scopes
        and spec.get("execution_parent_ids") == scopes[spec["execution_scope"]],
        "Exact canary, independent8VAL, or optional40CPU scope required")
    native._limits(cfg, spec["resource_caps"])
    native._require(spec["output_cap_bytes"] <= len(spec["execution_parent_ids"]) *
        spec["resource_caps"]["per_parent_output_cap_bytes"], "Total allowance cannot exceed admitted parent scope")
    if spec["execution_scope"] == "canary":
        native._require(spec.get("native_canary_ref") is None, "Canary must remain a fresh initial CPU scope")
    else:
        canary = native._read(spec["native_canary_ref"])
        native._require(canary.get("schema") == "expanded_native_argformer_tsinfer_CPU_prepared_v1"
            and canary.get("execution_scope") == "canary" and canary.get("source_ref") == spec["source_ref"]
            and canary.get("config_ref") == spec["config_ref"] and canary.get("scope_completed") is True
            and set(canary["prepared_refs"]) == {native.VAL_PARENTS[0]},
            "Completed same-source full952 native canary required before larger CPU scope")
        prepared = native._read(canary["prepared_refs"][native.VAL_PARENTS[0]])
        native._require(prepared["schema"] == "expanded_native_argformer_tsinfer_prepared_v1"
            and prepared["source_ref"] == cfg["source_ref"] and prepared["config_digest"] == native._digest(cfg)
            and prepared["parent_id"] == native.VAL_PARENTS[0] and prepared["split"] == "val"
            and prepared["true_query_ancestry_read"] is False
            and prepared["input_samples"] == 952 and prepared["query_haplotypes"] == 686
            and prepared["reference_haplotypes"] == 266 and prepared["observed_ages_verified"] is True
            and prepared["path_counts"]["mlm"]["train"] >= 512
            and prepared["path_counts"]["contrastive"]["train"] >= 4096,
            "Canary must prove complete observed corpus and literal released batch eligibility")
    for parent in spec["execution_parent_ids"]:
        ref = spec["completion_refs"][parent]
        native._require(ref["path"] == "data/expanded_full_panel_v1/" + parent + "/expanded_parent_completed.json",
                        "Registered completion path required")
        value = native._read(ref)
        native._require(value["parent_id"] == parent and value["generation_completed"] is True
            and value["Relate_completed"] is True and value["outer_truth_evaluated"] is False,
            "All 40 exact completed inner observations must be bound")
    runtime = native._read(spec["preparation_runtime_ref"])
    native._require(runtime.get("schema") == "expanded_argformer_tsinfer_CPU_runtime_v1"
        and runtime.get("stage") == "prepare" and runtime.get("source_ref") == cfg["source_ref"]
        and runtime.get("prefix") == str(native._path(native.PREP_PREFIX))
        and runtime.get("python") == [3, 11, 9] and {
            k: (v.split("+")[0] if v else v) for k, v in runtime.get("packages", {}).items()} == {
            "numpy": "1.26.4", "tskit": "0.6.4", "torch": "2.4.0"}
        and runtime.get("CUDA_VISIBLE_DEVICES") == "" and runtime.get("GPU_used") is False,
        "Qualified native CPU preparation runtime report required")
    storage = spec["storage"]
    native._require(storage["accounting_mode"] == MODE and storage["output_directory"] == spec["output_directory"]
        and storage["reserved_new_bytes"] == spec["output_cap_bytes"], "Exact retired-scopes output reservation required")
    refs = [spec[k] for k in ("source_ref", "config_ref", "adapter_ref", "storage_adapter_ref",
                             "observation_runtime_ref", "preparation_runtime_ref")]
    refs += [dict(path=p, sha256=sha) for p, sha in report["source_sha256"].items()]
    refs += [cfg["observation"][k] for k in ("generation_config_ref", "registration_ref", "recipe_ref", "pipeline_ref")]
    refs += [spec["completion_refs"][p] for p in spec["execution_parent_ids"]]
    if spec.get("native_canary_ref") is not None:
        refs.append(spec["native_canary_ref"])
    native._require(all(ref in storage["reservation"]["consumer_bindings"] for ref in refs),
                    "Reservation must bind exact scientific/runtime/source/completion closure")
    module = importlib.import_module("arggnn.project_storage_terminal_retired_scopes_v1")
    native._require(module.MODE == MODE, "Frozen successor accounting required")
    return spec, cfg, module, expiry


def _guard(spec, cfg, module, expiry, *, first=False):
    if first:
        module.validate_storage_v9(spec, dict(cfg, output_cap_bytes=spec["output_cap_bytes"]), spec["storage"],
            source_ref=spec["source_ref"], config_ref=spec["config_ref"], runtime_ref=spec["preparation_runtime_ref"])
    def guard(planned=0):
        native._require(datetime.now(timezone.utc) < expiry, "Root preparation authority expired")
        return module.budget_check_v9(spec["storage"], boundary=True, required_write_bytes=int(planned))
    return guard


def _owned_size(path):
    """Only this admitted parent subtree, at stopped-worker boundaries."""
    total = 0
    for directory, dirs, names in os.walk(path, followlinks=False):
        for name in dirs + names:
            p = Path(directory) / name
            native._require(not p.is_symlink(), "Owned parent subtree cannot contain symlinks")
        for name in names:
            p = Path(directory) / name
            native._require(p.is_file(), "Regular parent artifacts required")
            total += p.stat().st_size
    return total


def worker_v1(input_ref, parent_id, stage):
    spec, cfg, module, expiry = _admit(input_ref)
    native._require(parent_id in spec["execution_parent_ids"] and stage in ("observe", "prepare"), "Exact inner CPU stage required")
    output = native._path(spec["output_directory"]); parent = output / "parents" / parent_id
    launch_ref = native._ref(parent / (stage + "_launch.json"))
    launch = native._read(launch_ref)
    native._require(launch["input_ref"] == input_ref and launch["parent_id"] == parent_id and launch["stage"] == stage,
                    "Exact supervisor-to-worker stage binding required")
    root_guard = _guard(spec, cfg, module, expiry)
    def guard(planned=0):
        native._require(_owned_size(parent) + int(planned) <=
            spec["resource_caps"]["per_parent_output_cap_bytes"],
            "Per-parent complete native storage cap exceeded; no reduction")
        return root_guard(planned)
    tmp = parent / (stage + "_tmp")
    native._require(tmp.is_dir() and os.environ.get("TMPDIR") == str(tmp), "Owned native CPU scratch required")
    if stage == "observe":
        result = native.observe_parent_v1(parent_id, spec["completion_refs"][parent_id],
                                          parent / "observation", config=cfg, guard=guard, resource_caps=spec["resource_caps"])
    else:
        observation_ref = native._ref(parent / "observation/observation_manifest.json")
        result = native.prepare_parent_v1(parent_id, observation_ref, parent / "prepared", config=cfg, guard=guard, resource_caps=spec["resource_caps"])
    native._require(_owned_size(parent) <= spec["resource_caps"]["per_parent_output_cap_bytes"],
                    "Complete native observation+preparation exceeds per-parent cap")
    print(json.dumps(dict(stage=stage, parent_id=parent_id, manifest_ref=result["manifest_ref"])), flush=True)
    return result


def _RSS(pid):
    with Path("/proc/%d/status" % pid).open() as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL); process.wait()


def _launch(spec, cfg, input_ref, parent_id, stage, guard):
    parent = native._path(spec["output_directory"]) / "parents" / parent_id
    prefix = native.TS_PREFIX if stage == "observe" else native.PREP_PREFIX
    command = [str(native._path(prefix) / "bin/python"), "-B", "-m",
        "arggnn.prepare_expanded_native_argformer_tsinfer_v1", "--worker", "--stage", stage,
        "--parent", parent_id, "--input", input_ref["path"], "--expected-input-sha256", input_ref["sha256"]]
    tmp = parent / (stage + "_tmp"); guard(32768); tmp.mkdir(exist_ok=False)
    native._save(parent / (stage + "_launch.json"), dict(schema="expanded_argformer_tsinfer_CPU_launch_v1",
        input_ref=input_ref, source_ref=spec["source_ref"], parent_id=parent_id, stage=stage,
        command=command, supervisor_pid=os.getpid()), guard)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
        TMPDIR=str(tmp), NUMBA_CACHE_DIR=str(tmp), XDG_CACHE_HOME=str(tmp), MPLCONFIGDIR=str(tmp),
        HF_HUB_OFFLINE="1", WANDB_MODE="disabled")
    started = time.monotonic(); captured = bytearray(); peak = 0; failure = None
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    selector = selectors.DefaultSelector(); selector.register(process.stdout, selectors.EVENT_READ)
    print(json.dumps(dict(stage="CPU_worker_launched", parent_id=parent_id, operation=stage, pid=process.pid)), flush=True)
    try:
        while True:
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                else:
                    if len(captured) + len(chunk) > 16000000:
                        raise ValueError("Bounded native launch log exceeded 16MB; preserve worker failure")
                    captured.extend(chunk); sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            if process.poll() is not None and not selector.get_map():
                break
            if process.poll() is None:
                try:
                    peak = max(peak, _RSS(process.pid))
                except FileNotFoundError:
                    pass
                native._require(peak <= spec["resource_caps"]["CPU_RSS_cap_bytes"], "Native CPU RSS cap exceeded")
                native._require(time.monotonic() - started <= spec["resource_caps"]["per_stage_wall_seconds"],
                                "Native CPU stage wall cap exceeded")
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error)); _stop(process)
    finally:
        selector.close(); process.wait()
    # The supervisor publishes only after worker exit; guard namespace has one writer.
    guard(len(captured) + 8192)
    with (parent / (stage + "_launch.log")).open("xb") as f:
        f.write(captured); f.flush(); os.fsync(f.fileno())
    record = dict(schema="expanded_argformer_tsinfer_CPU_exit_v1", parent_id=parent_id, stage=stage,
        input_ref=input_ref, worker_pid=process.pid, worker_waited=True, returncode=process.returncode,
        peak_CPU_RSS_bytes=peak, elapsed_seconds=time.monotonic() - started, supervisor_failure=failure)
    native._save(parent / (stage + "_exit.json"), record, guard)
    native._require(failure is None and process.returncode == 0, "Native worker failed; no fallback, partial artifacts preserved")
    native._require(_owned_size(parent) <= spec["resource_caps"]["per_parent_output_cap_bytes"],
                    "Per-parent observation+prepared storage cap exceeded")
    return record


def supervise_v1(input_ref):
    spec, cfg, module, expiry = _admit(input_ref)
    output = native._path(spec["output_directory"])
    native._require(not output.exists(), "Fresh admitted native preparation output required; no implicit resume")
    guard = _guard(spec, cfg, module, expiry, first=True)
    guard(32768); output.mkdir(parents=True, exist_ok=False); (output / "parents").mkdir()
    started = time.monotonic(); manifests, timings = {}, {}
    for parent_id in spec["execution_parent_ids"]:
        guard(16384); (output / "parents" / parent_id).mkdir()
        timings[parent_id] = {}
        for stage in ("observe", "prepare"):
            timings[parent_id][stage] = _launch(spec, cfg, input_ref, parent_id, stage, guard)
        manifests[parent_id] = native._ref(output / "parents" / parent_id / "prepared/prepared_manifest.json")
    result = dict(schema="expanded_native_argformer_tsinfer_CPU_prepared_v1", condition=native.CONDITION,
        source_ref=spec["source_ref"], adapter_ref=spec["adapter_ref"], config_ref=spec["config_ref"],
        input_ref=input_ref, prepared_refs=manifests, CPU_stage_outcomes=timings, scope_completed=True,
        execution_scope=spec["execution_scope"], execution_parent_ids=spec["execution_parent_ids"],
        full_registry_parent_order=list(native.PARENTS), primary_fit_parent_ids=list(native.VAL_PARENTS),
        transductive_ARGformer_TRAIN_checkpoints_used_for_VAL=False,
        native_canary_ref=spec.get("native_canary_ref"), resource_caps=spec["resource_caps"],
        fresh_VAL_ARG_fits_authorized=False, fits_started=0, GPU_used=False, truth_read=False,
        outer_included=False, elapsed_seconds=time.monotonic() - started,
        storage_activation_owned_by_root=True)
    result["manifest_ref"] = native._save(output / "prepared_scope.json", result, guard)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--admission-draft", action="store_true")
    parser.add_argument("--stage", choices=("observe", "prepare"))
    parser.add_argument("--parent")
    parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256")
    args = parser.parse_args(argv)
    if args.admission_draft:
        print(json.dumps(admission_draft_v1(), indent=2, sort_keys=True)); return
    if not args.supervise and not args.worker:
        print(json.dumps(source_report_v1(), indent=2, sort_keys=True)); return
    if not args.input or not args.expected_input_sha256:
        parser.error("Root input and expected SHA required")
    input_ref = dict(path=args.input, sha256=args.expected_input_sha256)
    if args.worker:
        if not args.parent or not args.stage:
            parser.error("Bound parent and CPU stage required")
        result = worker_v1(input_ref, args.parent, args.stage)
    else:
        result = supervise_v1(input_ref)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
