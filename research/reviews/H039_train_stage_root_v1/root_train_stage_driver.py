"""Root-only H039 TRAIN admission; launch occurs only with explicit --launch."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, hashlib, importlib.util, json, os, signal, subprocess, sys, traceback

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/H039_train_stage_root_v1"
PACKET = HERE / "admitted_v1"
SELF = HERE / "root_train_stage_driver.py"
PYTHON = ROOT / ".runtime/env/bin/python"
CONFIG = "configs/directed_clade_exchange_train_stage_v1.json"
WORKER = "arggnn.train_directed_clade_exchange_train_stage_v1"
ORIGINAL_LAUNCHER = "scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"
SHARED_LAUNCHER = "scripts/launch_reserved_gpu_authorized_external_share_v1.py"
UTILITY = "research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py"
JOB = "directed-clade-exchange-train-stage-seed101-v1"
OUTPUT = "experiments/results/directed_clade_exchange_train_stage_v1_seed101"
CAP = 15000000
OFFICIAL_CLOSURE = "research/reviews/next_accuracy_architecture_fresh_v2/train_stage_v1/source_closure.json"
EXPECTED_METADATA = {'aligned_input': {'path': 'research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json', 'sha256': '72b200f5bcfbba13d04517b4f139b22097d1944ae4d855aa6c55ccf50c64d7e6'}, 'root_utility': {'path': 'research/reviews/expanded_full_panel_v1_root/retire_inactive_scopes_before_fast_cache_v1.py', 'sha256': '97ae77c6a5173b787e62b7fd3226c3184729477dfdba94b711d51dca315725ec'}, 'external_GPU_sharing_authorization': {'path': 'research/reviews/H039_train_stage_root_v1/external_GPU_sharing_authorization.json', 'sha256': '56fe3e47542530d45613de6f4356adaebd0cf66f2d21ba2b0b426c7f996684d1'}}
TRAIN = ["ap_expanded_train_%03d" % i for i in range(32)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ref(p):
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    return dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read_bound(r):
    require(r["path"].endswith(".json"), "Metadata JSON only; no arrays or checkpoints")
    require(ref(r["path"]) == r, "Metadata hash changed: " + r["path"])
    return json.loads((ROOT / r["path"]).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), sort_keys=True), flush=True)


def source_report():
    for r in EXPECTED_METADATA.values():
        require(ref(r["path"]) == r, "Frozen driver metadata/helper changed: " + r["path"])
    closure_ref = ref(OFFICIAL_CLOSURE)
    closure = read_bound(closure_ref)
    require(len(closure) == 46, "Exact reviewed 46-source closure required")
    for p, h in closure.items():
        require(p.endswith(".py") and ref(p)["sha256"] == h, "Reviewed source changed: " + p)
    return dict(status="source_only_no_admission_or_launch", source_ref=ref(SELF),
                source_sha256=closure, source_closure_ref=closure_ref, metadata_refs=EXPECTED_METADATA,
                job_id=JOB, output_directory=OUTPUT, output_cap_bytes=CAP,
                qualification_allowance_bytes=2000000, TRAIN_target_binding_parents=32,
                additional_updates_per_arm=320, checkpoint_count=6)


def utilities():
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]
    r = EXPECTED_METADATA["root_utility"]
    require(ref(r["path"]) == r, "Existing root admission helper changed")
    spec = importlib.util.spec_from_file_location("H039_train_root_utility", ROOT / r["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique(rows):
    return list({(r["path"], r["sha256"]): r for r in rows}.values())


def admit(launcher, max_cuda_reserved_bytes):
    report = source_report()
    require(launcher in (ORIGINAL_LAUNCHER, SHARED_LAUNCHER), "Reviewed launcher choice required")
    require(type(max_cuda_reserved_bytes) is int and 20000000000 < max_cuda_reserved_bytes <= 75000000000,
            "Original CUDA cap or root-admitted reduced cap above workspace reserve required")
    inactive_config_ref = ref(CONFIG)
    cfg = read_bound(inactive_config_ref)
    require(cfg.get("execution_guard_ready") is False and cfg.get("status") == "source_only",
            "Current inactive TRAIN config required")
    require(not (ROOT / OUTPUT).exists(), "Fresh exclusive TRAIN output required")
    require(not PACKET.exists(), "Admission already started; no automatic reserve replay")
    u = utilities()
    from arggnn import train_directed_clade_exchange_train_stage_v1 as worker
    require(set(report["source_sha256"]) == worker.ALLOWED_SOURCE_PATHS,
            "Actual worker must retain exact 46-source closure")
    old = read_bound(EXPECTED_METADATA["aligned_input"])
    cfg.update(execution_guard_ready=True, status="active",
        output_cap_status="active15MB_TRAIN_scope_six_checkpoints_no_predictions")
    require(cfg["source_refs"] == [dict(path=p, sha256=report["source_sha256"][p]) for p in worker.NEW_SOURCES],
            "Current TRAIN config source refs must match all ten actual H039 sources")
    if launcher == SHARED_LAUNCHER:
        require(ref(launcher)["sha256"] == "888eacff2765f441b47db3f07d52cb2dcee67e6291991c4756c7ee9a340d2bfc",
                "Root reviewed explicit-share wrapper required")
    cfg["helper_refs"]["actual_full_storage_launcher"] = ref(launcher)
    worker.validate_protocol_v1(cfg)
    original = read_bound(old["refs"]["target_bindings"])
    bindings = dict(original, parents={p: original["parents"][p] for p in TRAIN}, phase="train_only",
        TRAIN_target_binding_parents=32, VAL_included=False, outer_included=False, targets_opened=False)
    require(set(bindings["parents"]) == set(TRAIN) and
            bindings["parents"] == {p: original["parents"][p] for p in TRAIN},
            "Exact unchanged original 32 TRAIN metadata bindings required")
    require(all(b["split"] == "train" for b in bindings["parents"].values()),
            "TRAIN-only bindings required")
    generation = read_bound(old["refs"]["generation_config"])
    require([r["parent_id"] for r in generation["parent_registry"] if r["split"] == "train"] == TRAIN,
            "Original fixed 32 TRAIN registry required")
    completion = read_bound(cfg["warmstart_completion_ref"])
    require(completion["selected"]["checkpoint"] == cfg["warmstart_checkpoint_ref"],
            "Actual selected warmstart reference differs; checkpoint is not opened by admission")
    phase = dict(schema="directed_clade_exchange_split_phase_registration_v1",
        root_authorizer="root_agent", status="registered", hypothesis_id="H039", arms=cfg["arms"], seed=101,
        continuation_passes=10, additional_updates_per_arm=320, checkpoint_passes=[5,10],
        early_stopping=False, VAL_influences_optimization=False, later_all24_seals_before_any_VAL_truth=True,
        selection=cfg["selection"], warmstart_completion_ref=cfg["warmstart_completion_ref"],
        witness_sidecar_manifest_ref=cfg["reused_witness_manifest_ref"])
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(hours=23)).isoformat()
    PACKET.mkdir()
    u.save(PACKET, "admission_started.json", dict(started_utc=now.isoformat(),
           driver_ref=report["source_ref"], launcher_ref=ref(launcher),
           reserve_replay_authorized=False, scientific_execution_started=False))
    try:
        runtime_ref = u.save(PACKET, "runtime.json", dict(status="active", device="cuda",
            process_title="AI Auto Research Pipeline Test", created_at_utc=now.isoformat(),
            expires_at_utc=expiry, job_id=JOB, output_directory=OUTPUT,
            GPU_selected_by_existing_reserved_launcher=True, **cfg["runtime"]))
        phase["registered_utc"] = now.isoformat()
        phase_ref = u.save(PACKET, "phase_registration.json", phase)
        binding_ref = u.save(PACKET, "TRAIN_target_bindings.json", bindings)
        u.save(PACKET, "config_before_activation.json", json.loads((ROOT / CONFIG).read_text()))
        activated_ref = u.save(PACKET, "activated_config.json", cfg)
        # Root --admit installs the activation at the worker's required fixed CONFIG path.
        require(ref(CONFIG) == inactive_config_ref, "Concurrent TRAIN config edit")
        (ROOT / CONFIG).write_bytes((ROOT / activated_ref["path"]).read_bytes())
        config_ref = ref(CONFIG)
        refs = {k: old["refs"][k] for k in ("registration", "generation_config", "observation_cache")}
        refs.update(config=config_ref, runtime=runtime_ref, target_bindings=binding_ref,
            warmstart_completion=cfg["warmstart_completion_ref"],
            witness_sidecar_manifest=cfg["reused_witness_manifest_ref"], phase_registration=phase_ref)
        for r in refs.values():
            read_bound(r)
        worker.validate_split_phase_v1(phase, cfg, refs)
        authority = u.authority("reserve")
        consumers = unique([dict(path=p, sha256=h) for p,h in report["source_sha256"].items()]
            + list(refs.values()) + list(authority["bindings"].values())
            + [cfg["base_fit_config"], report["source_ref"], ref(launcher), report["source_closure_ref"]]
            + list(EXPECTED_METADATA.values()))
        require(len(consumers) <= 64, "Existing 64 consumer-binding limit exceeded")
        authority["reservation"] = dict(key="H039/" + JOB, output=str(ROOT / OUTPUT),
            auxiliary_scopes=[], reserved_bytes=CAP, consumer_bindings=consumers)
        authority_ref = u.save(PACKET, "reserve_authority.json", authority)
        reservation = u.m.apply_reserve_v1(authority_ref,
            expected_source_sha256=u.CORE, expected_launcher_sha256=u.BASE)
        u.save(PACKET, "reserve_receipt.json", reservation)
        storage_ref = u.save(PACKET, "storage.json", dict(kind="project_budget_v9_storage_record",
            active=True, consumer_uses_v9_guard=True, accounting_mode=u.m.MODE,
            job_id=JOB, output_directory=OUTPUT, reserved_new_bytes=CAP,
            reservation=reservation["value"], expires_at_utc=expiry,
            initial_ledger_pins=u.m._pins(u.inspect()),
            budget_components=dict(implementation=u.ref(u.m.budget.SOURCE_REL),
                launcher=u.ref(u.m.budget.LAUNCHER_REL), consumer_adapter=u.ref(u.m.SOURCE))))
        refs["storage"] = storage_ref
        spec = dict(old, schema="directed_clade_exchange_train_stage_input_v1",
            hypothesis_id="H039", root_authorizer="root_agent", phase="train_only",
            split_execution_authorized=True, fit_authorized=True, pyg_authorized=True,
            status="active", arm="joint_three_arm", arms=cfg["arms"], seed=101, device="cuda",
            job_id=JOB, output_directory=OUTPUT, created_at_utc=now.isoformat(),
            expires_at_utc=expiry, refs=refs, source_sha256=report["source_sha256"],
            max_cuda_reserved_bytes=max_cuda_reserved_bytes)
        input_ref = u.save(PACKET, "input.json", spec)
        admitted = dict(schema="H039_TRAIN_root_admission_v1", input_ref=input_ref,
            driver_ref=report["source_ref"], launcher_ref=ref(launcher), reserved_bytes=CAP,
            qualification_allowance_bytes=2000000, consumer_binding_count=len(consumers),
            additional_updates_per_arm=320, checkpoint_count=6, source_count=46,
            TRAIN_target_binding_parents=32, GPU_inventory_inspected=False,
            scientific_execution_started=False, admitted_utc=utc())
        u.save(PACKET, "admitted.json", admitted)
        emit("H039_TRAIN_admitted_not_launched", **admitted)
    except BaseException as error:
        u.save(PACKET, "admission_error.json", dict(type=type(error).__name__, error=str(error),
               traceback=traceback.format_exc(), automatic_reserve_replay=False, recorded_utc=utc()))
        raise


def launch(gpu):
    report = source_report()
    u = utilities()
    admitted = read_bound(ref(PACKET / "admitted.json"))
    require(admitted["driver_ref"] == report["source_ref"], "Admitted root driver source changed")
    launcher_ref = admitted["launcher_ref"]
    require(ref(launcher_ref["path"]) == launcher_ref, "Admitted actual launcher changed")
    spec = read_bound(admitted["input_ref"])
    require(spec["source_sha256"] == report["source_sha256"], "Admitted 46-source closure changed")
    require(not (ROOT / spec["output_directory"]).exists(), "Existing output; no automatic replay")
    require(not (PACKET / "launch.log").exists(), "Existing launcher attempt; no automatic replay")
    for r in spec["refs"].values():
        read_bound(r)
    storage = read_bound(spec["refs"]["storage"])
    storage["initial_ledger_pins"] = u.m._pins(u.inspect())
    spec["refs"]["storage"] = u.save(PACKET, "launch_storage.json", storage)
    input_ref = u.save(PACKET, "launch_input.json", spec)
    command = [str(PYTHON), "-B", "-m", launcher_ref["path"][:-3].replace("/", ".")]
    if gpu is not None:
        command += ["--gpu", str(gpu)]
    command += ["--job", JOB, "--storage-record", spec["refs"]["storage"]["path"],
        "--expected-storage-sha256", spec["refs"]["storage"]["sha256"],
        "--expected-budget-source-sha256", u.CORE, "--expected-launcher-sha256", u.BASE,
        WORKER, "--execute", "--input", input_ref["path"],
        "--expected-input-sha256", input_ref["sha256"]]
    env = dict(os.environ, PYTHONPATH=str(ROOT)+":"+str(ROOT/"scripts")+":"+str(ROOT/"src"),
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    child = None
    interrupted = []
    handlers = {}
    returncode = None
    launch_error = None
    try:
        with (PACKET / "launch.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            u.save(PACKET, "launch.json", dict(launcher_pid=child.pid, requested_GPU=gpu,
                actual_launcher_ref=launcher_ref, input_ref=input_ref, command=command,
                launched_utc=utc(), actual_wait_pending=True))
            def forward(signum, _frame):
                interrupted.append(signum)
                if child.poll() is None:
                    os.killpg(child.pid, signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, forward)
            emit("H039_TRAIN_actual_launcher_started", launcher_pid=child.pid, requested_GPU=gpu)
            returncode = child.wait()
    except BaseException as error:
        launch_error = dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            returncode = child.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    output = ROOT / spec["output_directory"]
    completed = output / "completed.json"
    failed = output / "failed.json"
    logpath = PACKET / "launch.log"
    receipt = dict(schema="H039_TRAIN_actual_reserved_launcher_wait_v1",
        launcher_pid=child.pid if child else None, returncode=returncode,
        actual_direct_launcher_wait=child is not None, actual_launcher_ref=launcher_ref,
        input_ref=input_ref, completed_ref=ref(completed) if completed.is_file() else None,
        failed_ref=ref(failed) if failed.is_file() else None,
        stdout_log_ref=ref(logpath) if logpath.is_file() else None,
        launcher_error=launch_error, received_signals=interrupted,
        success=returncode == 0 and completed.is_file() and not interrupted and launch_error is None,
        waited_utc=utc(), automatic_replay=False)
    if not receipt["success"] and logpath.is_file():
        receipt["actual_launcher_output_tail"] = logpath.read_bytes()[-12000:].decode("utf-8", errors="replace")
    u.save(PACKET, "root_wait.json", receipt)
    emit("H039_TRAIN_actual_launcher_wait", **receipt)
    require(receipt["success"], "TRAIN launch failed/partial; fixed root_wait.json and launch.log retain actual errors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--admit", action="store_true")
    actions.add_argument("--launch", action="store_true")
    parser.add_argument("--launcher", choices=(ORIGINAL_LAUNCHER, SHARED_LAUNCHER), default=ORIGINAL_LAUNCHER)
    parser.add_argument("--max-cuda-reserved-bytes", type=int, default=75000000000)
    parser.add_argument("--gpu", type=int, choices=(2, 3))
    args = parser.parse_args()
    try:
        if args.admit:
            admit(args.launcher, args.max_cuda_reserved_bytes)
        elif args.launch:
            launch(args.gpu)
        else:
            print(json.dumps(source_report(), indent=2, sort_keys=True))
    except BaseException as error:
        emit("H039_TRAIN_root_driver_error", type=type(error).__name__, error=str(error),
             traceback=traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
