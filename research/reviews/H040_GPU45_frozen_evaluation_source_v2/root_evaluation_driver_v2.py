"""Deferred root-only H040 frozen evaluation admission and GPU5 launcher.

Default invocation is inert. Source preparation never admits or launches a job.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H040_GPU45_frozen_evaluation_source_v2"
SOURCE="research/reviews/H040_GPU45_frozen_evaluation_source_v2/root_evaluation_driver_v2.py"
MAIN="src/arggnn/H040_GPU45_frozen_evaluation_v2.py"
CONTEXT="research/reviews/H040_GPU45_frozen_evaluation_source_v2/context_factory_v2.py"
GUARD_REF={"path":"research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py","sha256":"ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"}
POLICY_REF={"path":"research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json","sha256":"a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85"}
LAUNCHER_REF={"path":"scripts/launch_reserved_gpu_authorized_GPU45_completed_v1.py","sha256":"d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d"}
REPAIR_REF={"path":"research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json","sha256":"f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db"}
AUTH="research/reviews/H040_GPU45_migration_root_20261015_v1/GPU45_authorization.json"
H038_INPUT="research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json"
RESUME_ROOT="research/reviews/H040_GPU45_migration_root_20261015_v1"
ARMS=("tensor","event_mlp")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP=20000000
PACKET=HERE/"admitted_evaluate"
OUTPUT="experiments/results/H040_GPU45_frozen_evaluation_seed101_v2"
def require(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p!=ROOT and ROOT in p.parents and ".." not in p.parts and p.resolve()==p,"Canonical nonsymlink project path required")
    return p
def ref(value):
    p=local(value)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact metadata path/SHA record required")
    raw=local(record["path"]).read_bytes()
    require(len(raw)<=1048576 and hashlib.sha256(raw).hexdigest()==record["sha256"],"Bounded unchanged original metadata SHA required")
    return json.loads(raw)
def save(folder,name,value):
    p=local(folder/name)
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"New root metadata must fit charged 64KiB allowance")
    with p.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return ref(p)
def unique(rows):
    result={}
    for row in rows:
        require(Path(row["path"]).suffix in (".py",".sh",".json"),"Only metadata/source consumer bindings")
        old=result.get(row["path"])
        require(old is None or old==row,"Conflicting direct source pin")
        result[row["path"]]=row
    return list(result.values())
def fixed_refs():
    for record in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF):
        require(ref(record["path"])==record,"Fixed active resource component changed")
def load_worker():
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    from arggnn import H040_GPU45_frozen_evaluation_v2
    return H040_GPU45_frozen_evaluation_v2
class MetadataReader:
    root=ROOT
    def decode_json(self,record):return read(record)
def evidence():
    completions={arm:ref("experiments/results/H040_GPU45_resume_"+arm+"_seed101_v1/completed.json") for arm in ARMS}
    waits={arm:ref(RESUME_ROOT+"/admitted_"+arm+"/actual_wait.json") for arm in ARMS}
    launches={arm:ref(RESUME_ROOT+"/admitted_"+arm+"/launch.json") for arm in ARMS}
    for arm in ARMS:
        completion=read(completions[arm]);wait=read(waits[arm]);launch=read(launches[arm])
        require(type(wait.get("returncode")) is int and wait["returncode"]==0
            and wait.get("actual_direct_launcher_wait") is True
            and wait["completed_ref"]==completions[arm] and wait["launcher_pid"]==launch["launcher_pid"]
            and launch["input_ref"]==completion["input_ref"],"Both genuine resumed completions and actual same-launcher direct waits0 required")
    return completions,waits,launches
def load_guard():
    fixed_refs()
    sys.path[:0]=[str(ROOT),str(ROOT/"src")]
    spec=importlib.util.spec_from_file_location("_H040_root_completed_credit_v3",local(GUARD_REF["path"]))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    return module
def admit():
    fixed_refs()
    require(not PACKET.exists() and not local(OUTPUT).exists(),"Fresh exclusive root packet/output required")
    completions,waits,launches=evidence()
    worker=load_worker()
    trains={arm:read(completions[arm]) for arm in ARMS}
    inputs={arm:read(trains[arm]["input_ref"]) for arm in ARMS}
    cfg=worker.current_resource_config(read(trains["tensor"]["config_ref"]))
    worker.validate_protocol(cfg);require(cfg["output_cap_bytes"]==CAP,"Preserve exact original20MB config cap")
    h038_ref=ref(H038_INPUT);h038=read(h038_ref)
    targets=worker.val_target_bindings_v2(read(h038["refs"]["target_bindings"]))
    sources={}
    for old in inputs.values():
        for path,sha in old["source_sha256"].items():
            require(path not in sources or sources[path]==sha,"Original arm source closures differ")
            sources[path]=sha
    for path in (MAIN,CONTEXT,SOURCE,GUARD_REF["path"],LAUNCHER_REF["path"]):
        sources[path]=ref(path)["sha256"]
    for path,sha in sources.items():require(ref(path)["sha256"]==sha,"Original or current source bytes changed")
    PACKET.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=20)).isoformat()
    job="H040-GPU45-frozen-evaluation-seed101-v2"
    config_ref=save(PACKET,"config.json",cfg)
    runtime_ref=save(PACKET,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=OUTPUT,
        process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,
        torch="2.6.0+cu124",torch_geometric="2.7.0"))
    targets_ref=save(PACKET,"VAL_bindings.json",targets)
    transitive_ref=save(PACKET,"transitive_source_closure.json",dict(
        schema="H040_GPU45_evaluation_transitive_source_closure_v2",source_sha256=sources,
        train_completion_refs=completions,train_actual_wait_refs=waits,train_launch_refs=launches,
        original_H038_input_ref=h038_ref,checkpoint_arrays_and_truth_are_transitive_data_not_consumers=True))
    refs={key:inputs["tensor"]["refs"][key] for key in ("registration","generation_config","observation_cache")}
    refs.update(config=config_ref,runtime=runtime_ref,target_bindings=targets_ref)
    spec=dict(inputs["tensor"],schema="H040_GPU45_frozen_evaluation_input_v2",root_authorizer="root_agent",status="active",
        phase="evaluate",arm="joint_pair",job_id=job,output_directory=OUTPUT,created_at_utc=now.isoformat(),
        expires_at_utc=expiry,physical_gpu=5,refs=refs,source_sha256=sources,root_context_module_ref=ref(CONTEXT),
        main_source_ref=ref(MAIN),GPU45_authorization_ref=ref(AUTH),completed_overlay_policy_ref=POLICY_REF,
        original_H038_input_ref=h038_ref,transitive_source_closure_ref=transitive_ref,
        train_completion_refs=completions,train_actual_wait_refs=waits)
    spec.pop("resume_manifest_ref",None)
    worker.validate_evaluation_contract_v2(spec,cfg,MetadataReader())
    module=load_guard()
    view=module.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    direct=[ref(path) for path in (MAIN,CONTEXT,SOURCE,AUTH,
        "scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py","scripts/launch_reserved_gpu_full_storage_v1.py")]
    consumers=unique(list(view["bindings"].values())+direct+list(refs.values())+
        [GUARD_REF,LAUNCHER_REF,POLICY_REF,REPAIR_REF,transitive_ref,h038_ref,cfg["base_fit_config"]])
    require(len(consumers)<=64,"64-consumer limit exceeded; no admission")
    auxiliary=PACKET/"launcher_artifacts"
    require(not auxiliary.exists(),"Fresh root launcher artifact scope required")
    request=dict(key="H040/"+job,output=str(local(OUTPUT)),auxiliary_scopes=[str(auxiliary)],
        reserved_bytes=CAP,consumer_bindings=consumers)
    authority=module.operation_draft_v1("reserve",request,expected_before=module._pins(view),bindings=view["bindings"])
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(PACKET,"reserve_authority.json",authority)
    receipt=module.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(PACKET,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=module.MODE,job_id=job,output_directory=OUTPUT,reserved_new_bytes=CAP,
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(implementation=ref(module.budget.SOURCE_REL),
        launcher=ref(module.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    refs["storage"]=save(PACKET,"storage.json",storage)
    input_ref=save(PACKET,"input.json",spec)
    save(PACKET,"admitted.json",dict(input_ref=input_ref,driver_ref=ref(SOURCE),launcher_ref=LAUNCHER_REF,
        consumer_binding_count=len(consumers),source_count=len(sources),reserved_bytes=CAP,
        physical_gpu=5,scientific_execution_started=False))
    print(json.dumps(dict(admitted=True,input_ref=input_ref,consumer_binding_count=len(consumers))),flush=True)
def launch(gpu):
    require(gpu==5,"This deferred paired evaluator must use GPU5")
    fixed_refs();completions,waits,_=evidence()
    admitted=read(ref(PACKET/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["launcher_ref"]==LAUNCHER_REF,"Exact deferred driver/launcher required")
    spec=read(admitted["input_ref"])
    require(spec["train_completion_refs"]==completions and spec["train_actual_wait_refs"]==waits
        and not local(spec["output_directory"]).exists(),"Exact completed TRAIN evidence and fresh evaluation output required")
    worker=load_worker();cfg=read(spec["refs"]["config"])
    worker.validate_evaluation_contract_v2(spec,cfg,MetadataReader())
    module=load_guard();storage=read(spec["refs"]["storage"])
    module.budget_check_v9(storage,boundary=True,required_write_bytes=8192)
    spec["refs"]["storage"]=save(PACKET,"launch_storage.json",storage)
    input_ref=save(PACKET,"launch_input.json",spec)
    auxiliary=local(storage["reservation"]["auxiliary_scopes"][0]);require(not auxiliary.exists(),"Fresh launcher log scope required")
    auxiliary.mkdir();log_path=auxiliary/"launch.log"
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",LAUNCHER_REF["path"][:-3].replace("/","."),
        "--gpu","5","--job",spec["job_id"],"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],
        "--expected-budget-source-sha256",CORE,"--expected-launcher-sha256",BASE,
        "arggnn.H040_GPU45_frozen_evaluation_v2","--execute","--input",input_ref["path"],
        "--expected-input-sha256",input_ref["sha256"]]
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"scripts"),str(ROOT/"src"),env.get("PYTHONPATH","")])
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(POLICY_REF,separators=(",",":"))
    with log_path.open("xb") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=log)
        save(PACKET,"launch.json",dict(launcher_pid=proc.pid,physical_gpu=5,input_ref=input_ref,
            command=cmd,actual_wait_pending=True,completed_overlay_policy_ref=POLICY_REF))
        print(json.dumps(dict(launched=True,pid=proc.pid,gpu=5)),flush=True)
        rc=proc.wait()
    completed=local(spec["output_directory"])/"completed.json"
    save(PACKET,"actual_wait.json",dict(launcher_pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,
        completed_ref=ref(completed) if completed.exists() else None,log_ref=ref(log_path),
        waited_utc=datetime.now(timezone.utc).isoformat(),descendant_drain_or_no_live_writer_claim=False))
    print(json.dumps(dict(returncode=rc,actual_wait=True)),flush=True)
    return rc
def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group()
    group.add_argument("--admit",action="store_true");group.add_argument("--launch",action="store_true")
    p.add_argument("--gpu",type=int,choices=(5,),default=5);args=p.parse_args()
    if args.admit:admit();return 0
    if args.launch:return launch(args.gpu)
    print(json.dumps(dict(status="source_only_deferred_root_actions_not_executed",module="arggnn.H040_GPU45_frozen_evaluation_v2",
        physical_gpu=5,requires_both_resumed_completion_and_actual_wait0=True,policy_ref=POLICY_REF),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
