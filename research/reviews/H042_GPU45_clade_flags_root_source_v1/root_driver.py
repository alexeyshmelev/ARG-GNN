"""Root-only H042 admissions and direct actual launcher wait. Default is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
HERE=ROOT/"research/reviews/H042_GPU45_clade_flags_root_source_v1"
SOURCE="research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"
CONTEXT="research/reviews/H042_GPU45_clade_flags_root_source_v1/context_factory_v1.py"
MAIN="src/arggnn/train_anonymous_clade_flag_control_v1.py"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
POLICY_REF=dict(path="research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",sha256="a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85")
LAUNCHER_REF=dict(path="scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py",sha256="d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
AUTH_REF=dict(path="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json",sha256="0231ae8edb05f8dd34ce127a0de0b97c1dd64fa461ad081e7a83bb92c558d26c")

PREDECESSOR_COMPLETION_REF={'path': 'experiments/results/H040_GPU45_resume_tensor_seed101_v1/completed.json', 'sha256': '3e2f656f8657edc722a3223f8a6827f7ba3e2f626f997c90966846baffe13d29'}
TEMPLATE_REF=dict(path="research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json",sha256="72b200f5bcfbba13d04517b4f139b22097d1944ae4d855aa6c55ccf50c64d7e6")
PREDECESSOR="research/reviews/H040_GPU45_migration_root_20261015_v1/admitted_tensor"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP=40000000
LOG_CAP=4000000
def require(ok,msg):
    if not ok:raise ValueError(msg)
def ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file(),"Canonical existing project file")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(ref(record["path"])==record,"Pinned project metadata changed")
    require(record["path"].endswith(".json"),"JSON metadata only")
    return json.loads((ROOT/record["path"]).read_text())
def save(folder,name,value):
    require(Path(name).name==name,"Flat packet metadata name")
    with (folder/name).open("x") as f:json.dump(value,f,indent=2);f.write("\n")
    return ref(folder/name)
def unique(rows):return list({(r["path"],r["sha256"]):r for r in rows}.values())
def policy_env():
    env=os.environ.copy()
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(POLICY_REF,separators=(",",":"))
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"scripts"),str(ROOT/"src"),env.get("PYTHONPATH","")])
    return env
def load_guard():
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF):require(ref(r["path"])==r,"Exact current root authority required")
    spec=importlib.util.spec_from_file_location("_root_H042_credit_v3",ROOT/GUARD_REF["path"])
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    return m
def predecessor_refs():
    launch_ref=ref(PREDECESSOR+"/launch.json");wait_ref=ref(PREDECESSOR+"/actual_wait.json")
    launch=read(launch_ref);wait=read(wait_ref)
    require(launch["arm"]=="tensor" and launch["physical_gpu"]==4 and wait["actual_direct_launcher_wait"] is True
        and wait["returncode"]==0 and wait["completed_ref"]==PREDECESSOR_COMPLETION_REF
        and launch["launcher_pid"]==wait["launcher_pid"],
        "Root must actually wait for the exact successful H040tensor GPU4 completion before H042")
    return launch_ref,wait_ref
def admit(phase,train_completion_ref=None):
    from arggnn import train_anonymous_clade_flag_control_v1 as worker
    require(phase in ("train","evaluate"),"Single flag phase")
    launch_ref,wait_ref=predecessor_refs()
    packet=HERE/("admitted_"+phase);require(not packet.exists(),"No admission replay")
    output="experiments/results/H042_GPU4_clade_flags_"+phase+"_seed101_v1"
    auxiliary=str(packet.relative_to(ROOT)/"worker_aux")
    require(not (ROOT/output).exists() and not (ROOT/auxiliary).exists(),"Fresh exclusive output and auxiliary")
    old=read(TEMPLATE_REF);cfg=read(ref(HERE/"config.json"));worker.validate_protocol(cfg)
    sources=read(cfg["source_manifest_ref"])["source_sha256"]
    for path,digest in sources.items():require(ref(path)["sha256"]==digest,"Frozen source closure changed")
    require(cfg["hypothesis_id"]=="H042" and cfg["base_fit_config"]["path"]=="configs/census_tsinfer_fit_v1.json","H042/runtime identity")
    if phase=="evaluate":
        require(train_completion_ref is not None,"Singular exact-arm H042 TRAIN completion required")
        trained=read(train_completion_ref)
        require(trained.get("schema")=="anonymous_clade_flag_train_completion_v1"
            and trained.get("status")=="completed_fixed_30_pass_flag_TRAIN"
            and trained["hypothesis_id"]=="H042" and trained["arm"]==worker.ARM
            and trained["steps"]=={worker.ARM:960}
            and set(trained["checkpoint_groups"])=={"5","10","15","20","25","30"}
            and trained["VAL_truth_read"] is False and trained["outer_truth_read"] is False
            and trained["conditioned_on_H040_VAL"] is False,"Fixed30pass H042 train completion")
    else:require(train_completion_ref is None,"TRAIN cannot consume a previous completion")
    original=read(old["refs"]["target_bindings"]);ids=worker.TRAIN if phase=="train" else worker.VAL
    require(set(old["observation_entry_refs"])==set(worker.TRAIN+worker.VAL),"Original40 observed entries")
    bindings=dict(original,parents={pid:original["parents"][pid] for pid in ids},phase=phase,
        VAL_included=phase=="evaluate",outer_included=False)
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    job="H042-clade-flags-"+phase+"-seed101-GPU4"
    config_ref=ref(HERE/"config.json")
    runtime_ref=save(packet,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=output,
        physical_gpu=4,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,
        torch="2.6.0+cu124",torch_geometric="2.7.0",completed_overlay_policy_ref=POLICY_REF))
    binding_ref=save(packet,"phase_target_bindings.json",bindings)
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=config_ref,runtime=runtime_ref,target_bindings=binding_ref)
    manifest=dict(schema="H042_exact_transitive_consumer_closure_v1",hypothesis_id="H042",phase=phase,
        source_manifest_ref=cfg["source_manifest_ref"],source_sha256=sources,refs=refs,
        observation_entry_refs=old["observation_entry_refs"],observation_cache_source_refs=old["observation_cache_source_refs"],
        GPU4_predecessor_launch_ref=launch_ref,GPU4_predecessor_wait_ref=wait_ref,
        train_completion_ref=train_completion_ref)
    manifest_ref=save(packet,"consumer_manifest.json",manifest)
    m=load_guard()
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[ref(MAIN),config_ref,runtime_ref,ref(CONTEXT),ref(SOURCE),cfg["source_manifest_ref"],manifest_ref,
        LAUNCHER_REF,GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF,
        ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),cfg["base_fit_config"]]
    consumers=unique(mandatory+list(budget_bindings.values()))
    require(len(consumers)<=64,"Use exact transitive JSON closure below64 direct refs")
    request=dict(key="H042/"+job,output=str(ROOT/output),auxiliary_scopes=[str(ROOT/auxiliary)],
        reserved_bytes=CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=budget_bindings)
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=CAP,reservation=receipt["value"],expires_at_utc=expiry,
        initial_ledger_pins=receipt["initial_ledger_pins"],completed_overlay_policy_ref=POLICY_REF,
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    refs["storage"]=save(packet,"storage.json",storage)
    spec=dict(schema="anonymous_clade_flag_root_execution_input_v1",control_id=worker.CONTROL_ID,arm=worker.ARM,
        hypothesis_id="H042",root_authorizer="root_agent",status="active",phase=phase,seed=101,
        device="cuda",fit_authorized=True,pyg_authorized=True,job_id=job,output_directory=output,
        created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=50000000000,
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(SOURCE),main_source_ref=ref(MAIN),
        consumer_manifest_ref=manifest_ref,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],GPU45_authorization_ref=AUTH_REF,
        GPU4_predecessor_launch_ref=launch_ref,GPU4_predecessor_wait_ref=wait_ref,
        completed_overlay_policy_ref=POLICY_REF,physical_gpu=4,worker_auxiliary_scope=auxiliary)
    if phase=="evaluate":spec["train_completion_ref"]=train_completion_ref
    input_ref=save(packet,"input.json",spec)
    save(packet,"admitted.json",dict(input_ref=input_ref,launcher_ref=LAUNCHER_REF,driver_ref=ref(SOURCE),
        consumer_manifest_ref=manifest_ref,source_count=len(sources),consumer_binding_count=len(consumers),
        phase=phase,physical_gpu=4,passes=30,updates=960,reserved_bytes=CAP,scientific_execution_started=False))
    print(json.dumps(dict(admitted=phase,input_ref=input_ref,consumers=len(consumers))),flush=True)
def launch(phase):
    require(phase in ("train","evaluate"),"Single flag phase")
    predecessor_refs()
    packet=HERE/("admitted_"+phase);admitted=read(ref(packet/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["launcher_ref"]==LAUNCHER_REF,"Actual driver/launcher changed")
    spec=read(admitted["input_ref"]);require(not (ROOT/spec["output_directory"]).exists(),"No execution replay")
    require(spec["physical_gpu"]==4,"GPU4 only")
    m=load_guard();storage=read(spec["refs"]["storage"])
    storage["initial_ledger_pins"]=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE))
    spec["refs"]["storage"]=save(packet,"launch_storage.json",storage)
    input_ref=save(packet,"launch_input.json",spec)
    auxiliary=ROOT/spec["worker_auxiliary_scope"]
    m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+16384)
    auxiliary.mkdir()
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",LAUNCHER_REF["path"][:-3].replace("/","."),
        "--gpu","4","--job",spec["job_id"],"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],"--expected-budget-source-sha256",CORE,
        "--expected-launcher-sha256",BASE,"arggnn.train_anonymous_clade_flag_control_v1",
        "--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    retained=received=0;logging_error=None
    with (auxiliary/"launch.log").open("xb") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=policy_env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        save(packet,"launch.json",dict(launcher_pid=proc.pid,phase=phase,physical_gpu=4,input_ref=input_ref,
            command=cmd,actual_wait_pending=True,completed_overlay_policy_ref=POLICY_REF))
        for data in iter(lambda:proc.stdout.read1(65536),b""):
            received+=len(data)
            piece=data[:max(0,LOG_CAP-retained)]
            if piece and logging_error is None:
                try:
                    m.budget_check_v9(storage,boundary=True,required_write_bytes=len(piece)+8192)
                    log.write(piece);log.flush();retained+=len(piece)
                except BaseException as error:logging_error=type(error).__name__+": "+str(error)[:2048]
        rc=proc.wait()
        log.flush();os.fsync(log.fileno())
    completed=ROOT/spec["output_directory"]/"completed.json"
    receipt=dict(launcher_pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,
        completed_ref=ref(completed) if completed.exists() else None,log_ref=ref(auxiliary/"launch.log"),
        log_bytes_received=received,log_bytes_retained=retained,log_truncated=received>retained,
        logging_error=logging_error,physical_gpu=4,phase=phase,waited_utc=datetime.now(timezone.utc).isoformat())
    save(packet,"actual_wait.json",receipt)
    print(json.dumps(dict(phase=phase,returncode=rc,actual_wait=True,logging_error=logging_error)),flush=True)
    return rc if rc else (1 if logging_error else 0)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--admit",choices=("train","evaluate"));group.add_argument("--launch",choices=("train","evaluate"))
    parser.add_argument("--train-completion");parser.add_argument("--expected-train-completion-sha256")
    args=parser.parse_args()
    if args.admit:
        completion=None
        if args.train_completion or args.expected_train_completion_sha256:
            require(args.train_completion and args.expected_train_completion_sha256,"Exact singular completion path/SHA")
            completion=dict(path=args.train_completion,sha256=args.expected_train_completion_sha256)
        admit(args.admit,completion);return 0
    if args.launch:return launch(args.launch)
    print(json.dumps(dict(status="source_only_no_admission_or_execution",hypothesis_id="H042",physical_gpu=4,
        guard_ref=GUARD_REF,policy_ref=POLICY_REF,launcher_ref=LAUNCHER_REF)))
    return 0
if __name__=="__main__":raise SystemExit(main())
