"""Root admission and explicit launch for RAM-sealed frozen H039 VAL evaluation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,os,signal,subprocess,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H039_frozen_checkpoint_evaluation_v2"
SELF=HERE/"root_evaluation_driver.py"
PYTHON=ROOT/".runtime/env/bin/python"
WORKER="arggnn.evaluate_directed_clade_exchange_train_stage_v1"
SOURCE="src/arggnn/evaluate_directed_clade_exchange_train_stage_v2.py"
CAP=15000000
ARMS=("plain_continuation","generic_unary","directed")
VAL=tuple("ap_expanded_val_%03d"%i for i in range(8))


def require(ok,message):
    if not ok:raise ValueError(message)


def ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(record):
    require(record["path"].endswith(".json"),"Admission reads metadata only")
    require(ref(record["path"])==record,"Metadata SHA changed: "+record["path"])
    return json.loads((ROOT/record["path"]).read_text())


def save(folder,name,value):
    with (folder/name).open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return ref(folder/name)


def unique(rows):
    return list({(r["path"],r["sha256"]):r for r in rows}.values())


def utc():return datetime.now(timezone.utc).isoformat()


def emit(stage,**values):
    print(json.dumps(dict(stage=stage,**values),sort_keys=True),flush=True)


def packet(passes):
    label="pass5" if passes==[5] else "pass5_10"
    return HERE/("admitted_"+label),"directed-clade-exchange-frozen-evaluation-seed101-"+label+"-v2",        "experiments/results/directed_clade_exchange_frozen_evaluation_v2_seed101_"+label


def source_report():
    manifest_ref=ref(HERE/"source_manifest.json")
    manifest=read(manifest_ref)
    for p,h in manifest["sources"].items():
        require(p.endswith(".py") and ref(p)["sha256"]==h,"Frozen inference/scorer/helper source changed: "+p)
    baseline_ref=ref(HERE/"baseline_metadata.json");baseline=read(baseline_ref)
    for k,r in baseline.items():
        if k.endswith("_ref"):read(r)
    return dict(status="source_only_no_admission_or_launch",driver_ref=ref(SELF),
        worker_ref=dict(path=SOURCE,sha256=manifest["sources"][SOURCE]),source_manifest_ref=manifest_ref,
        baseline_metadata_ref=baseline_ref,required_execution_sources=len(manifest["required_consumer_source_paths"]),
        source_manifest_sources=len(manifest["sources"]),output_cap_bytes=CAP,all24_RAM_seals_before_truth=True)


def utilities(cfg):
    sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),str(ROOT/"src")]
    from arggnn import project_storage_terminal_retired_scopes_v1 as frozen
    helper_ref=cfg["helper_refs"]["storage_adapter"]
    require(ref(helper_ref["path"])==helper_ref,"Actual narrow repair helper changed")
    spec=importlib.util.spec_from_file_location("H039_frozen_eval_root_repair",ROOT/helper_ref["path"])
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    adapter.configure_repair_v1(cfg["marker_repair_authority_ref"])
    u=SimpleNamespace(m=adapter,budget=frozen.budget,pins=frozen._pins,
        CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e",
        BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3")
    u.inspect=lambda:adapter.inspect_v1(expected_source_sha256=u.CORE,expected_launcher_sha256=u.BASE)
    def authority():
        return dict(kind="project_budget_terminal_retired_root_admission_v1",root_authorizer="root_agent",
            active=True,operation="reserve",accounting_mode=frozen.MODE,
            bindings=dict(implementation=ref(frozen.budget.SOURCE_REL),launcher=ref(frozen.budget.LAUNCHER_REL),
                cpu_overlay=ref("src/arggnn/project_storage_cpu_closed_scopes_v1.py"),
                terminal_overlay=ref("src/arggnn/project_storage_terminal_scopes_v1.py"),
                terminal_success_overlay=ref("src/arggnn/project_storage_terminal_success_scopes_v1.py"),
                terminal_retired_overlay=ref(frozen.SOURCE)),
            expected_before=frozen._pins(u.inspect()),all_budget_writers_use_scheduler_mutex=True,
            all_consumers_and_launchers_use_terminal_retired_guard=True,
            all_terminal_success_writers_finished_before_retirement_activation=True,
            historical_records_mutation_authorized=False,global_rebase_authorized=False,
            scientific_execution_authorized=False,GPU_authorized=False,
            expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=23)).isoformat())
    u.authority=authority
    return u


def admit(passes):
    report=source_report();folder,job,out=packet(passes)
    require(not folder.exists() and not (ROOT/out).exists(),"Fresh evaluation admission/output; no reserve replay")
    baseline=read(report["baseline_metadata_ref"])
    train=read(baseline["train_input_ref"])
    train_cfg=read(baseline["train_config_ref"])
    cfg=read(ref(HERE/"config_draft.json"))
    require(cfg["execution_guard_ready"] is False and cfg["status"]=="source_only","Inactive reviewed config")
    cfg.update(status="active",execution_guard_ready=True,evaluation_passes=passes)
    plan=dict(schema="H039_frozen_checkpoint_set_v1",evaluation_passes=passes,
        train_input_ref=baseline["train_input_ref"],train_config_ref=baseline["train_config_ref"],
        train_phase_registration_ref=baseline["train_phase_registration_ref"],
        checkpoint_groups={},group_receipt_refs={})
    for at in passes:
        path=ROOT/baseline["train_output_directory"]/("pass_%02d_all3_frozen_checkpoints.json"%at)
        require(path.is_file(),"Actual all3 frozen pass%02d checkpoint receipt is not yet available"%at)
        group_ref=ref(path);group=read(group_ref)
        require(group["pass_index"]==at and group["steps"]=={a:32*at for a in ARMS}
            and group["all3_actual_states_retained"] is True and set(group["checkpoints"])==set(ARMS)
            and group["VAL_truth_read"] is False and group["VAL_predictions_executed"] is False,
            "Actual frozen all3 fixed-pass receipt required")
        for r in group["checkpoints"].values():
            require(r["path"].startswith(baseline["train_output_directory"]+"/")
                and r["path"].endswith(".pt") and (ROOT/r["path"]).is_file(),"Actual same TRAIN output checkpoint")
        plan["checkpoint_groups"][str(at)]=group["checkpoints"];plan["group_receipt_refs"][str(at)]=group_ref
    require(cfg["model"]==train_cfg["model"] and cfg["selection"]==train_cfg["selection"],"Unchanged model/selection")
    u=utilities(cfg)
    from arggnn import evaluate_directed_clade_exchange_train_stage_v1 as worker
    worker.validate_protocol(cfg)
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=23)).isoformat()
    folder.mkdir()
    save(folder,"admission_started.json",dict(root_authorizer="root_agent",driver_ref=report["driver_ref"],
        started_utc=now.isoformat(),evaluation_passes=passes,no_optimization=True,no_automatic_reserve_replay=True))
    try:
        checkpoint_ref=save(folder,"checkpoint_manifest.json",plan)
        config_ref=save(folder,"config.json",cfg)
        runtime_ref=save(folder,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=out,
            process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,
            GPU_selected_by_existing_v2_shared_launcher=True,**cfg["runtime"]))
        phase_ref=save(folder,"phase_registration.json",dict(
            schema="H039_frozen_checkpoint_evaluation_registration_v1",root_authorizer="root_agent",status="registered",
            hypothesis_id="H039",arms=list(ARMS),seed=101,evaluation_passes=passes,original_evaluation_passes=[5,10],
            selection=cfg["selection"],all24_RAM_seals_before_any_VAL_truth_per_pass=True,
            all_requested_passes_sealed_before_first_VAL_truth=True,optimization_authorized=False,
            dense_prediction_arrays_saved=False,probability_representation=cfg["probability_representation"],
            checkpoint_manifest_ref=checkpoint_ref,train_phase_registration_ref=baseline["train_phase_registration_ref"],
            fixed_pass5_only_selection_deferred=passes==[5],registered_utc=now.isoformat()))
        refs=dict(config=config_ref,runtime=runtime_ref,source_manifest=report["source_manifest_ref"],
            phase_registration=phase_ref,checkpoint_manifest=checkpoint_ref,
            generation_config=baseline["generation_config_ref"],registration=baseline["registration_ref"],
            observation_cache=baseline["observation_cache_ref"],target_bindings=baseline["VAL_target_bindings_ref"],
            witness_sidecar_manifest=baseline["witness_sidecar_manifest_ref"],train_input=baseline["train_input_ref"],
            train_config=baseline["train_config_ref"],train_phase_registration=baseline["train_phase_registration_ref"])
        manifest=read(report["source_manifest_ref"])
        authority=u.authority()
        consumers=unique([dict(path=p,sha256=manifest["sources"][p]) for p in manifest["required_consumer_source_paths"]]
            +list(refs.values())+list(authority["bindings"].values())
            +[report["driver_ref"],cfg["base_fit_config"],cfg["helper_refs"]["storage_adapter"],
                cfg["helper_refs"]["actual_launcher"],cfg["helper_refs"]["v9_consumer_marker"],
                cfg["marker_repair_authority_ref"],cfg["external_GPU_sharing_authorization_ref"]])
        require(len(consumers)<=64,"Actual required execution/metadata consumer closure exceeds64")
        authority["reservation"]=dict(key="H039/"+job,output=str(ROOT/out),auxiliary_scopes=[],
            reserved_bytes=CAP,consumer_bindings=consumers)
        authority_ref=save(folder,"reserve_authority.json",authority)
        reservation=u.m.apply_reserve_v1(authority_ref,expected_source_sha256=u.CORE,expected_launcher_sha256=u.BASE)
        save(folder,"reserve_receipt.json",reservation)
        storage_ref=save(folder,"storage.json",dict(kind="project_budget_v9_storage_record",active=True,
            consumer_uses_v9_guard=True,accounting_mode=u.m.MODE,job_id=job,output_directory=out,
            reserved_new_bytes=CAP,reservation=reservation["value"],expires_at_utc=expiry,
            initial_ledger_pins=u.pins(u.inspect()),budget_components=dict(implementation=ref(u.budget.SOURCE_REL),
                launcher=ref(u.budget.LAUNCHER_REL),consumer_adapter=cfg["helper_refs"]["storage_adapter"])))
        refs["storage"]=storage_ref
        spec=dict(schema="H039_frozen_checkpoint_evaluation_input_v1",root_authorizer="root_agent",status="active",
            hypothesis_id="H039",phase="frozen_checkpoint_evaluation",evaluation_authorized=True,
            VAL_truth_score_authorized=True,optimization_authorized=False,fit_authorized=False,
            TRAIN_truth_authorized=False,outer_truth_authorized=False,pyg_authorized=True,device="cuda",
            arms=list(ARMS),seed=101,evaluation_passes=passes,job_id=job,output_directory=out,
            created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,
            evaluation_source_ref=report["worker_ref"],max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=50000000000)
        input_ref=save(folder,"input.json",spec)
        admitted=dict(schema="H039_frozen_eval_root_admission_v1",driver_ref=report["driver_ref"],
            input_ref=input_ref,evaluation_passes=passes,actual_consumer_binding_count=len(consumers),
            checkpoint_count=3*len(passes),reserved_bytes=CAP,optimization_authorized=False,
            GPU_inventory_inspected=False,GPU_launched=False,admitted_utc=utc())
        save(folder,"admitted.json",admitted);emit("H039_frozen_evaluation_admitted_not_launched",**admitted)
    except BaseException as error:
        save(folder,"admission_error.json",dict(type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),automatic_reserve_replay=False,recorded_utc=utc()))
        raise


def launch(passes,gpu):
    report=source_report();folder,job,out=packet(passes)
    admitted=read(ref(folder/"admitted.json"))
    require(admitted["driver_ref"]==report["driver_ref"],"Admitted root driver changed")
    spec=read(admitted["input_ref"]);cfg=read(spec["refs"]["config"]);u=utilities(cfg)
    require(not (ROOT/out).exists() and not (folder/"launch.log").exists(),"Existing output/attempt; no replay")
    require(cfg["source_manifest_ref"]==report["source_manifest_ref"],"Admitted execution source map changed")
    launcher=cfg["helper_refs"]["actual_launcher"]
    require(ref(launcher["path"])==launcher,"Actual v2 sharing launcher changed")
    storage=read(spec["refs"]["storage"]);storage["initial_ledger_pins"]=u.pins(u.inspect())
    spec["refs"]["storage"]=save(folder,"launch_storage.json",storage)
    input_ref=save(folder,"launch_input.json",spec)
    command=[str(PYTHON),"-B","-m",launcher["path"][:-3].replace("/",".")]
    if gpu is not None:command+=["--gpu",str(gpu)]
    command+=["--job",job,"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],
        "--expected-budget-source-sha256",u.CORE,"--expected-launcher-sha256",u.BASE,
        WORKER,"--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    env=dict(os.environ,PYTHONPATH=str(ROOT)+":"+str(ROOT/"scripts")+":"+str(ROOT/"src"),
        PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    child=None;rc=None;error=None;signals=[];handlers={}
    try:
        with (folder/"launch.log").open("xb") as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(folder,"launch.json",dict(launcher_pid=child.pid,requested_GPU=gpu,command=command,
                input_ref=input_ref,actual_launcher_ref=launcher,actual_wait_pending=True,launched_utc=utc()))
            def forward(signum,_frame):
                signals.append(signum)
                if child.poll() is None:os.killpg(child.pid,signum)
            for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,forward)
            emit("H039_frozen_actual_launcher_started",launcher_pid=child.pid,evaluation_passes=passes)
            rc=child.wait()
    except BaseException as caught:
        error=dict(type=type(caught).__name__,error=str(caught),traceback=traceback.format_exc())
        if child:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            rc=child.wait()
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    completed=ROOT/out/"completed.json";failed=ROOT/out/"failed.json";log=folder/"launch.log"
    receipt=dict(schema="H039_frozen_evaluation_actual_wait_v1",launcher_pid=child.pid if child else None,
        returncode=rc,actual_direct_launcher_wait=child is not None,input_ref=input_ref,
        completed_ref=ref(completed) if completed.is_file() else None,failed_ref=ref(failed) if failed.is_file() else None,
        stdout_log_ref=ref(log) if log.is_file() else None,launcher_error=error,received_signals=signals,
        success=rc==0 and completed.is_file() and error is None and not signals,automatic_replay=False,waited_utc=utc())
    if not receipt["success"] and log.is_file():
        receipt["actual_launcher_output_tail"]=log.read_bytes()[-12000:].decode("utf-8",errors="replace")
    save(folder,"root_wait.json",receipt);emit("H039_frozen_actual_launcher_wait",**receipt)
    require(receipt["success"],"Frozen evaluation failed/partial; actual errors preserved in launch.log/root_wait.json")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument("--admit",action="store_true");actions.add_argument("--launch",action="store_true")
    parser.add_argument("--passes",choices=("5","5,10"),default="5,10")
    parser.add_argument("--gpu",type=int,choices=(2,3))
    args=parser.parse_args();passes=[5] if args.passes=="5" else [5,10]
    try:
        if args.admit:admit(passes)
        elif args.launch:launch(passes,args.gpu)
        else:print(json.dumps(source_report(),indent=2,sort_keys=True))
    except BaseException as error:
        emit("H039_frozen_root_driver_error",type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise SystemExit(1)


if __name__=="__main__":main()
