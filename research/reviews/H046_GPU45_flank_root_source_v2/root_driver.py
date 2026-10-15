"""Source-only future H046 admission/launcher proposal. Default is inert."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H046_GPU45_flank_root_source_v2.root_common_v1 import (
    HERE as HERE_REL,require,ref,read,save,unique,policy_env,load_guard_v1,count_plan_v1,
    validate_train_lineage_v1,GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,TEMPLATE_REF,
    MEMO_REF,V4_REF,CORE,BASE,OUTPUT_CAP,LOG_CAP)
HERE=ROOT/HERE_REL
SOURCE=HERE_REL+"/root_driver.py"
CONTEXT=HERE_REL+"/context_factory_v1.py"
CLI_MODULE="arggnn.H046_GPU45_flank_worker_v2"
CLI="src/"+CLI_MODULE.replace(".","/")+".py"
MAIN="src/arggnn/train_continuous_clade_flank_v1.py"
CAP=OUTPUT_CAP
def admit(phase,gpu,train_completion_ref=None,train_launch_ref=None,train_actual_wait_ref=None):
    from arggnn import train_continuous_clade_flank_v1 as worker
    require(phase in ("train","evaluate"),"Single paired phase")
    require(gpu in (4,5),"Root must explicitly choose one eligible GPU4/5")
    packet=HERE/("admitted_"+phase);require(not packet.exists(),"No admission replay")
    output="experiments/results/H046_GPU%d_flank_%s_seed101_v2"%(gpu,phase)
    auxiliary=str(packet.relative_to(ROOT)/"worker_aux")
    require(not (ROOT/output).exists() and not (ROOT/auxiliary).exists(),"Fresh exclusive output and auxiliary")
    old=read(TEMPLATE_REF);cfg=read(ref(HERE/"config_inactive.json"));worker.validate_protocol(cfg)
    sources=read(cfg["source_manifest_ref"])["source_sha256"]
    for path,digest in sources.items():require(ref(path)["sha256"]==digest,"Frozen source closure changed")
    require(cfg["hypothesis_id"]=="H046" and cfg["base_fit_config"]["path"]=="configs/census_tsinfer_fit_v1.json","H046/runtime identity")
    require(cfg.get("status")=="root_frozen_protocol_before_admission"
        and cfg.get("root_resource_caps_are_proposals") is False,"Root must freeze reviewed caps before admission")
    preparation=read(cfg["preparation_completion_ref"])
    wait=read(cfg["preparation_actual_wait_ref"])
    require(wait["actual_direct_launcher_wait"] is True and wait["returncode"]==0
        and wait["completed_ref"]==cfg["preparation_completion_ref"] and wait["owned_group_drained"] is True,
        "Original immutable H044 preparation direct wait/drain required")
    counts=read(cfg["count_manifest_ref"]);plan=count_plan_v1(counts,phase,preparation)
    require(plan["all96_original_grid_probability_bytes"]<=cfg["RAM_prediction_cap_bytes"],"Complete dense96 RAM bound")
    require(all(r["compact_payload_upper_bound_bytes"]<=min(cfg["flank_payload_cap_bytes"],
        cfg["flank_cache_cap_bytes"]) for r in plan["parents"].values()),
        "Declared complete raw payload bound exceeds caps")
    require(all(r["GPU_flank_payload_upper_bound_bytes"]<=cfg["GPU_flank_cap_bytes"]
        for r in plan["parents"].values()),"Declared complete GPU flank payload exceeds cap")
    lineage=dict(phase=phase,train_completion_ref=train_completion_ref,
                 train_launch_ref=train_launch_ref,train_actual_wait_ref=train_actual_wait_ref)
    if phase=="evaluate":
        require(all(lineage[k] is not None for k in ("train_completion_ref","train_launch_ref","train_actual_wait_ref")),
                "Actual paired H046 TRAIN and direct launcher wait refs required")
        validate_train_lineage_v1(type("MetadataReader",(),{"decode_json":staticmethod(read)})(),lineage)
    else:
        require(all(lineage[k] is None for k in ("train_completion_ref","train_launch_ref","train_actual_wait_ref")),
                "TRAIN cannot consume previous completions")
    original=read(old["refs"]["target_bindings"]);ids=worker.TRAIN if phase=="train" else worker.VAL
    require(set(old["observation_entry_refs"])==set(worker.TRAIN+worker.VAL),"Original40 observed entries")
    bindings=dict(original,parents={pid:original["parents"][pid] for pid in ids},phase=phase,
        TRAIN_included=phase=="train",VAL_included=phase=="evaluate",outer_included=False)
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=24)).isoformat()
    job="H046-flank-%s-seed101-v2-GPU%d"%(phase,gpu)
    config_ref=ref(HERE/"config_inactive.json")
    runtime_ref=save(packet,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=output,
        physical_gpu=gpu,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,
        torch="2.6.0+cu124",torch_geometric="2.7.0",completed_overlay_policy_ref=POLICY_REF))
    binding_ref=save(packet,"phase_target_bindings.json",bindings)
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=config_ref,runtime=runtime_ref,target_bindings=binding_ref)
    manifest=dict(schema="H046_exact_transitive_consumer_closure_v1",hypothesis_id="H046",phase=phase,
        source_manifest_ref=cfg["source_manifest_ref"],source_sha256=sources,refs=refs,
        observation_entry_refs=old["observation_entry_refs"],observation_cache_source_refs=old["observation_cache_source_refs"],
        
        preparation_completion_ref=cfg["preparation_completion_ref"],preparation_actual_wait_ref=cfg["preparation_actual_wait_ref"],
        train_completion_ref=train_completion_ref,train_launch_ref=train_launch_ref,train_actual_wait_ref=train_actual_wait_ref)
    manifest_ref=save(packet,"consumer_manifest.json",manifest)
    m,memo=load_guard_v1(consumer=False)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,budget_bindings=state[2],state[3]
    mandatory=[ref(MAIN),config_ref,runtime_ref,ref(CONTEXT),ref(SOURCE),ref(CLI),cfg["source_manifest_ref"],manifest_ref,
        cfg["count_manifest_ref"],cfg["preparation_completion_ref"],cfg["preparation_actual_wait_ref"],MEMO_REF,V4_REF,
        LAUNCHER_REF,GUARD_REF,POLICY_REF,AUTH_REF,REPAIR_REF,
        ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),
        ref("research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"),cfg["base_fit_config"]]
    consumers=unique(mandatory+list(budget_bindings.values())
        +([train_completion_ref,train_launch_ref,train_actual_wait_ref] if phase=="evaluate" else []))
    require(len(consumers)<=64,"Use exact transitive JSON closure below64 direct refs")
    request=dict(key="H046/"+job,output=str(ROOT/output),auxiliary_scopes=[str(ROOT/auxiliary)],
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
    spec=dict(schema="H046_paired_flank_root_input_v1",arm="joint_pair",
        hypothesis_id="H046",root_authorizer="root_agent",status="active",phase=phase,seed=101,
        device="cuda",fit_authorized=True,pyg_authorized=True,job_id=job,output_directory=output,
        created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=50000000000,
        root_context_module_ref=ref(CONTEXT),root_driver_ref=ref(SOURCE),main_source_ref=ref(MAIN),worker_cli_ref=ref(CLI),
        consumer_manifest_ref=manifest_ref,observation_entry_refs=old["observation_entry_refs"],
        observation_cache_source_refs=old["observation_cache_source_refs"],GPU45_authorization_ref=AUTH_REF,
        
        completed_overlay_policy_ref=POLICY_REF,physical_gpu=gpu,worker_auxiliary_scope=auxiliary,
        preparation_completion_ref=cfg["preparation_completion_ref"],preparation_actual_wait_ref=cfg["preparation_actual_wait_ref"])
    if phase=="evaluate":spec.update(train_completion_ref=train_completion_ref,train_launch_ref=train_launch_ref,train_actual_wait_ref=train_actual_wait_ref)
    input_ref=save(packet,"input.json",spec)
    save(packet,"admitted.json",dict(input_ref=input_ref,launcher_ref=LAUNCHER_REF,driver_ref=ref(SOURCE),
        consumer_manifest_ref=manifest_ref,source_count=len(sources),consumer_binding_count=len(consumers),
        phase=phase,physical_gpu=gpu,passes=30,updates=960,reserved_bytes=CAP,output_forecast=plan,scientific_execution_started=False))
    print(json.dumps(dict(admitted=phase,input_ref=input_ref,consumers=len(consumers))),flush=True)
def launch(phase):
    require(phase in ("train","evaluate"),"Single paired phase")
    packet=HERE/("admitted_"+phase);admitted=read(ref(packet/"admitted.json"))
    require(admitted["driver_ref"]==ref(SOURCE) and admitted["launcher_ref"]==LAUNCHER_REF,"Actual driver/launcher changed")
    spec=read(admitted["input_ref"]);require(not (ROOT/spec["output_directory"]).exists(),"No execution replay")
    gpu=spec["physical_gpu"];require(gpu in (4,5),"Exactly one root-selected eligible GPU")
    m,memo=load_guard_v1(consumer=True);storage=read(spec["refs"]["storage"])
    # Preserve the original reservation immutable-prefix pins and storage reference.
    input_ref=save(packet,"launch_input.json",spec)
    auxiliary=ROOT/spec["worker_auxiliary_scope"]
    m.budget_check_v9(storage,boundary=True,required_write_bytes=LOG_CAP+16384)
    auxiliary.mkdir()
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",LAUNCHER_REF["path"][:-3].replace("/","."),
        "--gpu",str(gpu),"--job",spec["job_id"],"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],"--expected-budget-source-sha256",CORE,
        "--expected-launcher-sha256",BASE,CLI_MODULE,
        "--execute","--input",input_ref["path"],"--expected-input-sha256",input_ref["sha256"]]
    captured=bytearray();received=0;supervisor_error=None
    proc=subprocess.Popen(cmd,cwd=ROOT,env=policy_env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    try:
        try:
            save(packet,"launch.json",dict(launcher_pid=proc.pid,phase=phase,physical_gpu=gpu,input_ref=input_ref,
                command=cmd,actual_wait_pending=True,completed_overlay_policy_ref=POLICY_REF))
        except BaseException as error:
            supervisor_error=type(error).__name__+": "+str(error)[:2048]
        for data in iter(lambda:proc.stdout.read1(65536),b""):
            received+=len(data);captured.extend(data[:max(0,LOG_CAP-len(captured))])
    except BaseException as error:
        supervisor_error=supervisor_error or type(error).__name__+": "+str(error)[:2048]
    finally:
        rc=proc.wait();proc.stdout.close()
    log_ref=None;logging_error=None
    # No charged-scope scan or log write by this supervisor while the child runs.
    try:
        m.budget_check_v9(storage,boundary=True,required_write_bytes=len(captured)+8192)
        with (auxiliary/"launch.log").open("xb") as log:log.write(captured);log.flush();os.fsync(log.fileno())
        log_ref=ref(auxiliary/"launch.log")
    except BaseException as error:logging_error=type(error).__name__+": "+str(error)[:2048]
    completed=ROOT/spec["output_directory"]/"completed.json"
    receipt=dict(launcher_pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,
        completed_ref=ref(completed) if completed.exists() else None,log_ref=log_ref,
        log_bytes_received=received,log_bytes_retained=len(captured),log_truncated=received>len(captured),
        logging_error=logging_error,supervisor_error=supervisor_error,physical_gpu=gpu,phase=phase,
        waited_utc=datetime.now(timezone.utc).isoformat(),descendant_drain_or_no_live_writer_claim=False)
    save(packet,"actual_wait.json",receipt)
    print(json.dumps(dict(phase=phase,returncode=rc,actual_wait=True,
        logging_error=logging_error,supervisor_error=supervisor_error)),flush=True)
    return rc if rc else (1 if logging_error or supervisor_error else 0)
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--admit",choices=("train","evaluate"));group.add_argument("--launch",choices=("train","evaluate"))
    parser.add_argument("--gpu",type=int,choices=(4,5))
    for name in ("train-completion","train-launch","train-actual-wait"):
        parser.add_argument("--"+name);parser.add_argument("--expected-"+name+"-sha256")
    args=parser.parse_args(argv)
    if args.admit:
        require(args.gpu in (4,5),"Root must choose/schedule a physical GPU explicitly")
        refs=[]
        for name in ("train_completion","train_launch","train_actual_wait"):
            path=getattr(args,name);digest=getattr(args,"expected_"+name+"_sha256")
            require(bool(path)==bool(digest),"Paired exact path/SHA required")
            refs.append(dict(path=path,sha256=digest) if path else None)
        admit(args.admit,args.gpu,*refs);return 0
    if args.launch:return launch(args.launch)
    print(json.dumps(dict(status="source_only_no_admission_or_execution",hypothesis_id="H046",
        eligible_GPUs=[4,5],paired_arms_one_GPU=True,guard_ref=GUARD_REF,memo_ref=MEMO_REF,
        proposed_output_scope_cap_bytes=CAP)));return 0
if __name__=="__main__":raise SystemExit(main())
