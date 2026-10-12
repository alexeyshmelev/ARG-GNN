"""ROOT-owned native admission/launch helper; default source report only.

--admit runs qualified metadata runtime reports, freezes drafts and reserves
exact scopes. --launch uses new input/storage records with current initial pins.
The author never invokes either flag. No retirement or frozen-source mutation.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import time
import sys

R = Path("/workspace-SR003.nfs2/ARG-GNN")
P = R / "research/reviews/expanded_full_panel_v1_root/native_launch_helpers_v1"
SELF = P / "native_admit_launch_v1.py"
UTIL = R / "research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
UTIL_SHA = "97ae77c6a5173b787e62b7fd3226c3184729477dfdba94b711d51dca315725ec"
GUARD = "src/arggnn/project_storage_terminal_retired_scopes_v1.py"
GUARD_SHA = "bbbdf26206e4e1725c79d2558b4426af848df4cbbd5ac2d0f2bc7051c0606bab"
PYTHON = R / ".runtime/env/bin/python"
JOBS = {
    "ancestralpaths": dict(
        module="arggnn.run_expanded_native_ancestralpaths_v2",
        driver="src/arggnn/run_expanded_native_ancestralpaths_v2.py",
        sha="62919ad60cbb76c2feec8bf9ce34a002137b920f3d96278d2f494c96416a0b51",
        runtime=".runtime/conda/envs/ancestralpaths_cpu_v1/bin/python",
        cap=16000000000),
    "argformer-canary": dict(
        module="arggnn.prepare_expanded_native_argformer_tsinfer_v1",
        driver="src/arggnn/prepare_expanded_native_argformer_tsinfer_v1.py",
        sha="33d60ea61755c18733603b9c5e9ae89857c364b6b05349557d37d2fec378cab7",
        adapter="src/arggnn/expanded_native_argformer_tsinfer_v1.py",
        adapter_sha="b3704e780a5c3dbac073ed1cea7cd3fecf660caa54164b9996ee658479b3e600",
        cap=4000000000),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refs_only():
    files = [SELF, UTIL, R / GUARD] + [R / value["driver"] for value in JOBS.values()]
    files.append(R / JOBS["argformer-canary"]["adapter"])
    files.append(R / "src/arggnn/project_budget_v9_consumer.py")
    return {str(p.relative_to(R)): sha(p) for p in files}


def utilities():
    if sha(UTIL) != UTIL_SHA or sha(R / GUARD) != GUARD_SHA:
        raise ValueError("Pinned ROOT utility/retired guard changed")
    for value in JOBS.values():
        if sha(R / value["driver"]) != value["sha"]:
            raise ValueError("Frozen native driver changed")
    value = JOBS["argformer-canary"]
    if sha(R / value["adapter"]) != value["adapter_sha"]:
        raise ValueError("Frozen native inference adapter changed")
    spec = importlib.util.spec_from_file_location("root_native_launch_utilities_v1", UTIL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected(target):
    return list(JOBS) if target == "both" else [target]


def packet(tag, name):
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in tag):
        raise ValueError("Simple fresh ROOT packet tag required")
    return P / ("admitted_" + tag) / name


def capture(u, folder, name, command):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(command, cwd=R, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False)
    u.save(folder, name + "_cli.json", dict(command=command, returncode=process.returncode,
           stderr=process.stderr.decode("utf-8", errors="replace")))
    if process.returncode:
        raise RuntimeError("Qualified metadata CLI failed: " + name)
    value = json.loads(process.stdout)
    return value, u.save(folder, name + ".json", value)


def unique(refs):
    return list({(ref["path"], ref["sha256"]): ref for ref in refs}.values())


def prepare_draft(u, name, tag):
    value = JOBS[name]
    folder = packet(tag, name)
    folder.mkdir(parents=True, exist_ok=False)
    if name == "ancestralpaths":
        runtime, runtime_ref = capture(u, folder, "runtime",
            [str(R / value["runtime"]), "-B", "-m", value["module"], "--runtime-report"])
        if runtime["packages"].get("tensorflow-cpu") != "2.15.1" or runtime["packages"].get("keras") != "2.15.0":
            raise ValueError("AP requires actual qualified TF2.15.1/Keras2.15 CPU report")
    else:
        module = "arggnn.expanded_native_argformer_tsinfer_v1"
        runtime, runtime_ref = capture(u, folder, "preparation_runtime",
            [str(R / ".runtime/conda/envs/argformer_native_v1/bin/python"),
             "-B", "-m", module, "--runtime-report", "prepare"])
        observed, observed_ref = capture(u, folder, "observation_runtime",
            [str(R / ".runtime/envs/biological_tsinfer_v1/bin/python"),
             "-B", "-m", module, "--runtime-report", "observe"])
        if (observed["prefix"] != str(R / ".runtime/envs/biological_tsinfer_v1")
            or observed["python"] != [3, 11, 17]
            or observed["packages"] != {"numpy": "2.4.6", "tskit": "1.0.3", "tsinfer": "0.5.1", "tsdate": "0.2.7"}
            or observed["CUDA_VISIBLE_DEVICES"] != ""):
            raise ValueError("Canary requires actual qualified tsinfer/tsdate CPU report")
    if runtime.get("CUDA_VISIBLE_DEVICES") != "" or runtime.get("GPU_used") is not False:
        raise ValueError("Qualified runtime reports must be CPU-only")
    if name == "ancestralpaths":
        if (runtime.get("schema") != "expanded_native_ap_cpu_runtime_v2"
            or runtime.get("prefix") != str(R / ".runtime/conda/envs/ancestralpaths_cpu_v1")
            or runtime.get("source_ref") != u.ref(value["driver"])):
            raise ValueError("Exact AP v2 qualified runtime/source report required")
    else:
        if (runtime.get("schema") != "expanded_argformer_tsinfer_CPU_runtime_v1"
            or runtime.get("stage") != "prepare" or runtime.get("python") != [3, 11, 9]
            or runtime.get("prefix") != str(R / ".runtime/conda/envs/argformer_native_v1")
            or {k: (v.split("+")[0] if v else v) for k, v in runtime.get("packages", {}).items()} !=
                {"numpy": "1.26.4", "tskit": "0.6.4", "torch": "2.4.0"}
            or runtime.get("source_ref") != u.ref(value["adapter"])):
            raise ValueError("Exact native ARGformer preparation runtime/source report required")
    draft, draft_ref = capture(u, folder, "input_draft",
        [str(PYTHON), "-B", "-m", value["module"], "--admission-draft"])
    if draft["pending_parent_ids"]:
        raise ValueError("Completed parent metadata is still pending")
    if draft["output_cap_bytes"] != value["cap"] or (R / draft["output_directory"]).exists():
        raise ValueError("Exact16GB AP /4GB canary and a fresh native output required")
    config = u.read(draft["config_ref"]["path"])
    refs = [dict(path=p, sha256=h) for p, h in draft["source_sha256"].items()]
    refs += [draft["source_ref"], draft["config_ref"], draft["storage_adapter_ref"],
             runtime_ref, draft_ref, u.ref(SELF), u.ref(UTIL),
             u.ref("src/arggnn/project_budget_v9_consumer.py")]
    if name == "ancestralpaths":
        draft["runtime_ref"] = runtime_ref
        refs += [config["generation_config_ref"], config["registration_ref"]]
        refs += list(draft["completion_refs"].values())
        refs += list(config["helper_refs"].values())
        if len(draft["completion_refs"]) != 40 or len(draft["TRAIN_truth_refs"]) != 32:
            raise ValueError("AP native final five-epoch all40/32TRAIN scope required")
    else:
        draft["preparation_runtime_ref"] = runtime_ref
        draft["actual_observation_runtime_report_ref"] = observed_ref
        refs += [observed_ref, draft["adapter_ref"], draft["observation_runtime_ref"]]
        refs += [config["observation"][k] for k in ("generation_config_ref", "registration_ref", "recipe_ref", "pipeline_ref")]
        refs += [draft["completion_refs"][p] for p in draft["execution_parent_ids"]]
        if draft["execution_scope"] != "canary" or draft["execution_parent_ids"] != ["ap_expanded_val_000"]:
            raise ValueError("Only full952 VAL000 native inference/path-preparation canary admitted here")
    draft.update(status="active", execution_authorized=True, root_authorizer="root_agent",
        root_launch_helper_ref=u.ref(SELF), root_authority_utilities_ref=u.ref(UTIL),
        expires_at_utc=(datetime.now(timezone.utc) + timedelta(hours=23)).isoformat())
    return dict(name=name, folder=folder, draft=draft, consumers=refs, runtime_ref=runtime_ref)


def admit(target, tag):
    u = utilities()
    if Path.cwd().resolve() != R or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("ROOT repository and explicit CPU visibility required")
    items = [prepare_draft(u, name, tag) for name in selected(target)]
    for item in items:
        draft = item["draft"]
        authority = u.authority("reserve")  # Fresh credited inspect/pins under the existing utility.
        item["consumers"] += list(authority["bindings"].values())
        authority["reservation"] = dict(key="H038/" + draft["job_id"],
            output=str(R / draft["output_directory"]), auxiliary_scopes=[],
            reserved_bytes=draft["output_cap_bytes"], consumer_bindings=unique(item["consumers"]))
        authority_ref = u.save(item["folder"], "reserve_authority.json", authority)
        receipt = u.m.apply_reserve_v1(authority_ref, expected_source_sha256=u.CORE,
                                      expected_launcher_sha256=u.BASE)
        u.save(item["folder"], "reserve_receipt.json", receipt)
        item["reservation"] = receipt["value"]
    view = u.inspect()
    for item in items:
        draft = item["draft"]
        storage = dict(kind="project_budget_v9_storage_record", active=True, consumer_uses_v9_guard=True,
            accounting_mode=u.m.MODE, job_id=draft["job_id"], output_directory=draft["output_directory"],
            reserved_new_bytes=draft["output_cap_bytes"], reservation=item["reservation"],
            expires_at_utc=draft["expires_at_utc"], initial_ledger_pins=u.m._pins(view),
            budget_components=dict(implementation=u.ref(u.m.budget.SOURCE_REL),
                launcher=u.ref(u.m.budget.LAUNCHER_REL), consumer_adapter=u.ref(u.m.SOURCE)))
        u.save(item["folder"], "storage.json", storage)
        draft["storage"] = storage
        input_ref = u.save(item["folder"], "input.json", draft)
        u.save(item["folder"], "admitted.json", dict(input_ref=input_ref, runtime_ref=item["runtime_ref"],
               scope=draft.get("execution_scope", "native_AP_32TRAIN_final5epoch_8VAL_seals"),
               output_cap_bytes=draft["output_cap_bytes"], launched=False))
        print(json.dumps(dict(stage="ROOT_native_admitted", name=item["name"], input_ref=input_ref,
              output_cap_bytes=draft["output_cap_bytes"], launched=False)), flush=True)


def waited(u, item, *, interruption=None):
    name, folder, child, output = item
    returncode = child.wait()
    expected = R / output / ("completed.json" if name == "ancestralpaths" else "prepared_scope.json")
    failed = R / output / "failed.json"
    record = dict(kind="ROOT_native_supervisor_direct_wait_v1", name=name, supervisor_pid=child.pid,
        returncode=returncode, actual_child_waited=True,
        completed_ref=u.ref(expected) if expected.is_file() else None,
        failure_manifest_ref=u.ref(failed) if failed.is_file() else None,
        stdout_log_ref=u.ref(folder / "launch.log"), interruption=interruption,
        success=returncode == 0 and expected.is_file() and interruption is None,
        waited_at_utc=datetime.now(timezone.utc).isoformat(), GPU_used=False)
    u.save(folder, "root_wait.json", record)
    print(json.dumps(dict(stage="ROOT_native_supervisor_directly_waited", **record)), flush=True)
    return record


def launch(target, tag):
    u = utilities()
    if Path.cwd().resolve() != R or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("ROOT repository and explicit CPU visibility required")
    running, outcomes = [], []
    def terminated(signum, frame):
        raise KeyboardInterrupt("ROOT dispatch interrupted by signal " + str(signum))
    old_term = signal.signal(signal.SIGTERM, terminated)
    try:
        # Dispatch both independent children before waiting for either.
        for name in selected(target):
            folder = packet(tag, name)
            admitted = u.read(str((folder / "admitted.json").relative_to(R)))
            source_input = admitted["input_ref"]
            if u.ref(source_input["path"]) != source_input:
                raise ValueError("Admitted input bytes changed")
            value = u.read(source_input["path"])
            if value["root_launch_helper_ref"] != u.ref(SELF) or (R / value["output_directory"]).exists():
                raise ValueError("Exact helper and fresh output required; no implicit replay")
            value["storage"]["initial_ledger_pins"] = u.m._pins(u.inspect())
            storage_ref = u.save(folder, "launch_storage.json", value["storage"])
            input_ref = u.save(folder, "launch_input.json", value)
            command = [str(PYTHON), "-B", "-m", JOBS[name]["module"], "--supervise",
                       "--input", input_ref["path"], "--expected-input-sha256", input_ref["sha256"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # This ROOT log is outside each worker's exclusive native output namespace.
            with (folder / "launch.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=R, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((name, folder, child, value["output_directory"]))
            record = dict(kind="ROOT_native_supervisor_launch_v1", name=name, pid=child.pid,
                input_ref=input_ref, storage_ref=storage_ref, command=command,
                launched_utc=datetime.now(timezone.utc).isoformat(),
                root_owns_direct_wait=True, scientific_completion_claimed=False, GPU_used=False)
            u.save(folder, "launch.json", record)
            print(json.dumps(dict(stage="ROOT_native_supervisor_launched", **record)), flush=True)
        while running:
            for item in list(running):
                if item[2].poll() is None:
                    continue
                running.remove(item); outcomes.append(waited(u, item))
            if running:
                time.sleep(2)
    except BaseException as error:
        # Preserve direct exit evidence for every child even on interrupted ROOT dispatch.
        reason = dict(type=type(error).__name__, message=str(error))
        for item in list(running):
            child = item[2]
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            running.remove(item)
            try:
                outcomes.append(waited(u, item, interruption=reason))
            except BaseException as receipt_error:
                print("Child directly waited; ROOT wait receipt publication failed: " + str(receipt_error),
                      file=sys.stderr, flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
    if not all(record["success"] for record in outcomes):
        raise RuntimeError("Native supervisor failed or has no completed manifest; see direct root_wait.json")
    print(json.dumps(dict(stage="ROOT_native_dispatch_completed", selected=selected(target),
        directly_waited=len(outcomes), all_requested_scopes_completed=True, GPU_used=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--admit", choices=("ancestralpaths", "argformer-canary", "both"))
    action.add_argument("--launch", choices=("ancestralpaths", "argformer-canary", "both"))
    parser.add_argument("--tag", default="v1", help="Fresh metadata packet suffix; output/job identities remain frozen")
    args = parser.parse_args()
    if args.admit:
        admit(args.admit, args.tag)
    elif args.launch:
        launch(args.launch, args.tag)
    else:
        print(json.dumps(dict(status="source_only_no_admission_or_launch", source_sha256=refs_only(),
            AP_cap_bytes=16000000000, ARGformer_VAL000_canary_cap_bytes=4000000000,
            authority_utility_sha256=UTIL_SHA, accounting_mode="journal_terminal_retired_scopes_credit_v1",
            commands=["--admit both --tag v1", "--launch both --tag v1"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
