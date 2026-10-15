"""Root-only H042 admissions and direct actual launcher wait. Default is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
HERE=ROOT/"research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1"
SOURCE="research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1/root_driver.py"
CONTEXT="research/reviews/H042_GPU45_frozen_flag_evaluation_source_v1/context_factory_v1.py"
MAIN="src/arggnn/H042_GPU45_frozen_flag_evaluation_v1.py"
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
MEMO_REF={'path': 'research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py', 'sha256': 'c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364'}
V4_REF={'path': 'src/arggnn/project_budget_v4.py', 'sha256': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47'}
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
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Charged root metadata must fit64KiB bound")
    with (folder/name).open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return ref(folder/name)
def unique(rows):return list({(r["path"],r["sha256"]):r for r in rows}.values())
def policy_env():
    env=os.environ.copy()
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(POLICY_REF,separators=(",",":"))
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"scripts"),str(ROOT/"src"),env.get("PYTHONPATH","")])
    return env
def load_guard():
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,MEMO_REF,V4_REF):require(ref(r["path"])==r,"Exact current root authority required")
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

class MetadataReader:
    root=ROOT
    def decode_json(self,record):return read(record)

def load_worker():
    from arggnn import H042_GPU45_frozen_flag_evaluation_v1
    return H042_GPU45_frozen_flag_evaluation_v1

def training_refs():
    worker=load_worker()
    refs=dict(train_completion_ref=ref(worker.TRAIN_OUTPUT+"/completed.json"),
              train_launch_ref=ref(worker.TRAIN_PACKET+"/launch.json"),
              train_actual_wait_ref=ref(worker.TRAIN_PACKET+"/actual_wait.json"))
    evidence=worker.validate_train_evidence_v1(refs,MetadataReader())
    return refs,evidence

def admit(gpu):
    require(gpu in (4,5),"Root must explicitly select physical GPU4 or5")
    worker=load_worker();science=worker.original
    train_refs,(trained,train_input,train_cfg)=training_refs()
    launch_ref,wait_ref=predecessor_refs()
    packet=HERE/"admitted_evaluate";require(not packet.exists(),"No duplicate H042 evaluation admission")
    output="experiments/results/H042_GPU"+str(gpu)+"_clade_flags_evaluate_seed101_memo_v1"
    auxiliary=packet/"worker_aux"
    require(not (ROOT/output).exists() and not auxiliary.exists(),"Fresh exclusive output and auxiliary")
    old=read(TEMPLATE_REF)
    source_doc=read(train_cfg["source_manifest_ref"]);sources=dict(source_doc["source_sha256"])
    for path in (MAIN,CONTEXT,SOURCE,MEMO_REF["path"],V4_REF["path"],
                 worker.ORIGINAL_REF["path"],worker.ORIGINAL_CONTEXT_REF["path"],worker.ORIGINAL_DRIVER_REF["path"]):
        digest=ref(path)["sha256"]
        require(path not in sources or sources[path]==digest,"Original source closure changed")
        sources[path]=digest
    for path,digest in sources.items():require(ref(path)["sha256"]==digest,"Frozen source closure changed")
    original=read(old["refs"]["target_bindings"])
    bindings=dict(original,parents={pid:original["parents"][pid] for pid in science.VAL},
                  phase="evaluate",VAL_included=True,outer_included=False)
    require(set(old["observation_entry_refs"])==set(science.TRAIN+science.VAL),"Original40 observation entries required")
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    job="H042-clade-flags-evaluate-seed101-GPU"+str(gpu)+"-memo-v1"
    # Numerical config is exactly TRAIN config; only its source-manifest ref changes.
    evaluation_source_doc=dict(source_doc,source_sha256=sources)
    evaluation_source_doc["metadata_refs"]=dict(source_doc["metadata_refs"],original_TRAIN_config=trained["config_ref"],
        actual_TRAIN_completion=train_refs["train_completion_ref"],actual_TRAIN_wait=train_refs["train_actual_wait_ref"],
        actual_TRAIN_launch=train_refs["train_launch_ref"])
    source_manifest_ref=save(packet,"source_manifest.json",evaluation_source_doc)
    cfg=dict(train_cfg,source_manifest_ref=source_manifest_ref);science.validate_protocol(cfg)
    require(cfg["output_cap_bytes"]==CAP,"Preserve exact original40MB output cap")
    config_ref=save(packet,"config.json",cfg)
    runtime_ref=save(packet,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=output,
        physical_gpu=gpu,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,
        torch="2.6.0+cu124",torch_geometric="2.7.0",completed_overlay_policy_ref=POLICY_REF))
    binding_ref=save(packet,"phase_target_bindings.json",bindings)
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=config_ref,runtime=runtime_ref,target_bindings=binding_ref)
    manifest=dict(schema="H042_exact_transitive_consumer_closure_v1",hypothesis_id="H042",phase="evaluate",
        source_manifest_ref=source_manifest_ref,source_sha256=sources,refs=dict(refs),
        observation_entry_refs=old["observation_entry_refs"],observation_cache_source_refs=old["observation_cache_source_refs"],
        GPU4_predecessor_launch_ref=launch_ref,GPU4_predecessor_wait_ref=wait_ref,
        memo_helper_ref=MEMO_REF,**train_refs)
    manifest_ref=save(packet,"consumer_manifest.json",manifest)
    spec=dict(schema="H042_frozen_flag_evaluation_input_v1",control_id=science.CONTROL_ID,arm=science.ARM,
        hypothesis_id="H042",root_authorizer="root_agent",status="active",phase="evaluate",seed=101,
        device="cuda",fit_authorized=True,pyg_authorized=True,job_id=job,output_directory=output,
        created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=50000000000,
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(SOURCE),main_source_ref=ref(MAIN),
        consumer_manifest_ref=manifest_ref,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],GPU45_authorization_ref=AUTH_REF,
        GPU4_predecessor_launch_ref=launch_ref,GPU4_predecessor_wait_ref=wait_ref,
        completed_overlay_policy_ref=POLICY_REF,physical_gpu=gpu,worker_auxiliary_scope=auxiliary.relative_to(ROOT).as_posix(),
        memo_helper_ref=MEMO_REF,transaction_memo_no_throttling=True,**train_refs)
    worker.validate_capability_v1(spec,MetadataReader(),cfg)
    m=load_guard()
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[ref(MAIN),config_ref,runtime_ref,ref(CONTEXT),ref(SOURCE),source_manifest_ref,manifest_ref,
        LAUNCHER_REF,GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF,MEMO_REF,V4_REF,
        ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),cfg["base_fit_config"],*train_refs.values()]
    consumers=unique(mandatory+list(budget_bindings.values()))
    require(len(consumers)<=64,"Exact transitive closure must remain below64 direct refs")
    request=dict(key="H042/"+job,output=str(ROOT/output),auxiliary_scopes=[str(auxiliary)],
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
    input_ref=save(packet,"input.json",spec)
    save(packet,"admitted.json",dict(input_ref=input_ref,launcher_ref=LAUNCHER_REF,driver_ref=ref(SOURCE),
        consumer_manifest_ref=manifest_ref,source_count=len(sources),consumer_binding_count=len(consumers),
        phase="evaluate",physical_gpu=gpu,reserved_bytes=CAP,scientific_execution_started=False))
    print(json.dumps(dict(admitted="evaluate",input_ref=input_ref,gpu=gpu,consumers=len(consumers))),flush=True)

def launch(gpu):
    packet=HERE/"admitted_evaluate";admitted=read(ref(packet/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["launcher_ref"]==LAUNCHER_REF,"Exact driver/launcher changed")
    spec=read(admitted["input_ref"])
    require(gpu==spec["physical_gpu"] and gpu in (4,5),"Use the exact root-admitted physical GPU")
    require(not (ROOT/spec["output_directory"]).exists(),"No duplicate H042 evaluation execution")
    cfg=read(spec["refs"]["config"]);load_worker().validate_capability_v1(spec,MetadataReader(),cfg)
    m=load_guard();storage=read(spec["refs"]["storage"])
    # Preserve the reservation's original immutable-prefix pins.
    input_ref=save(packet,"launch_input.json",spec)
    auxiliary=ROOT/spec["worker_auxiliary_scope"];require(not auxiliary.exists(),"Fresh exclusive auxiliary scope")
    m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+16384);auxiliary.mkdir()
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",LAUNCHER_REF["path"][:-3].replace("/","."),
        "--gpu",str(gpu),"--job",spec["job_id"],"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],"--expected-budget-source-sha256",CORE,
        "--expected-launcher-sha256",BASE,"arggnn.H042_GPU45_frozen_flag_evaluation_v1",
        "--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    captured=bytearray();received=0
    proc=subprocess.Popen(cmd,cwd=ROOT,env=policy_env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    save(packet,"launch.json",dict(launcher_pid=proc.pid,phase="evaluate",physical_gpu=gpu,input_ref=input_ref,
        command=cmd,actual_wait_pending=True,completed_overlay_policy_ref=POLICY_REF))
    try:
        for data in iter(lambda:proc.stdout.read1(65536),b""):
            received+=len(data);captured.extend(data[:max(0,LOG_CAP-len(captured))])
    finally:
        rc=proc.wait();proc.stdout.close()
    log_ref=None;logging_error=None
    # No supervisor guard scan or charged-scope log writes while the launcher runs.
    try:
        m.budget_check_v9(storage,boundary=True,required_write_bytes=len(captured)+8192)
        with (auxiliary/"launch.log").open("xb") as log:log.write(captured);log.flush();os.fsync(log.fileno())
        log_ref=ref(auxiliary/"launch.log")
    except BaseException as error:logging_error=type(error).__name__+": "+str(error)[:2048]
    completed=ROOT/spec["output_directory"]/"completed.json"
    receipt=dict(launcher_pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,
        completed_ref=ref(completed) if completed.exists() else None,log_ref=log_ref,
        log_bytes_received=received,log_bytes_retained=len(captured),log_truncated=received>len(captured),
        logging_error=logging_error,physical_gpu=gpu,phase="evaluate",waited_utc=datetime.now(timezone.utc).isoformat(),
        descendant_drain_or_no_live_writer_claim=False)
    save(packet,"actual_wait.json",receipt)
    print(json.dumps(dict(phase="evaluate",returncode=rc,actual_wait=True,logging_error=logging_error)),flush=True)
    return rc if rc else (1 if logging_error else 0)

def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group()
    group.add_argument("--admit",action="store_true");group.add_argument("--launch",action="store_true")
    p.add_argument("--gpu",type=int,choices=(4,5));args=p.parse_args()
    if args.admit:
        require(args.gpu in (4,5),"Explicit root GPU choice required");admit(args.gpu);return 0
    if args.launch:
        require(args.gpu in (4,5),"Explicit root GPU choice required");return launch(args.gpu)
    print(json.dumps(dict(status="source_only_no_admission_or_execution",hypothesis_id="H042",phase="evaluate",
        eligible_physical_GPUs=[4,5],requires_original_TRAIN960_completion_and_wait0=True,memo_helper_ref=MEMO_REF),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
