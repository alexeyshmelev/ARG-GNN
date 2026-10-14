
"""Root-only admissions and direct launcher waits for H040 paired training."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
HERE=ROOT/"research/reviews/H040_root_execution_v1"
SOURCE="research/reviews/H040_root_execution_v1/root_driver.py"
CONTEXT="research/reviews/H040_root_execution_v1/context_factory.py"
CONFIG_PROPOSAL="research/reviews/H040_latent_clade_execution_source_v1/proposed_config.json"
GUARD="research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"
LAUNCHER="scripts/launch_reserved_gpu_authorized_external_share_v4.py"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP=20000000
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
    assert ref(r["path"])==r
    return json.loads((ROOT/r["path"]).read_text())
def save(folder,name,v):
    p=folder/name
    with p.open("x") as f:json.dump(v,f,indent=2);f.write("\n")
    return ref(p)
def load_guard():
    s=importlib.util.spec_from_file_location("root_H040_capacity",ROOT/GUARD);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.configure_v2(ref("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"));return m
def unique(rows):return list({(r["path"],r["sha256"]):r for r in rows}.values())
def admit(arm):
    assert arm in ("tensor","event_mlp")
    packet=HERE/("admitted_"+arm);assert not packet.exists()
    output="experiments/results/H040_latent_clade_"+arm+"_seed101"
    assert not (ROOT/output).exists()
    m=load_guard()
    old=read(ref("research/reviews/expanded_full_panel_v1_root/full_panel_pyg_fits_v4/aligned/input.json"))
    cfg=read(ref(CONFIG_PROPOSAL))
    cfg.update(output_cap_bytes=CAP,qualification_allowance_bytes=2000000,
        marker_repair_authority_ref=ref("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"),
        helper_refs=dict(full_reservation_adapter=ref(GUARD),actual_full_storage_launcher=ref(LAUNCHER),v9_consumer_marker=ref("src/arggnn/project_budget_v9_consumer.py")))
    from arggnn import H040_latent_clade_execution_v1 as worker
    worker.validate_protocol(cfg)
    sources=dict(old["source_sha256"])
    for p in (worker.SOURCE,*[r["path"] for r in worker.MECHANISM_REFS],CONTEXT):
        sources[p]=ref(p)["sha256"]
    for p,h in sources.items():assert ref(p)["sha256"]==h
    packet.mkdir()
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    job="H040-latent-clade-"+arm+"-seed101"
    config_ref=save(packet,"config.json",cfg)
    runtime_ref=save(packet,"runtime.json",dict(status="active",device="cuda",job_id=job,output_directory=output,process_title="AI Auto Research Pipeline Test",created_at_utc=now.isoformat(),expires_at_utc=expiry,torch="2.6.0+cu124",torch_geometric="2.7.0"))
    original=read(old["refs"]["target_bindings"])
    bindings=dict(original,parents={p:original["parents"][p] for p in worker.TRAIN},phase="train",VAL_included=False,outer_included=False)
    binding_ref=save(packet,"TRAIN_bindings.json",bindings)
    refs={k:old["refs"][k] for k in ("registration","generation_config","observation_cache")}
    refs.update(config=config_ref,runtime=runtime_ref,target_bindings=binding_ref)
    with m.budget.v4.mutex(ROOT):
        state=m._state_locked(CORE,BASE,fresh=True)
    _,_,view,budget_bindings,_=state
    consumers=unique([dict(path=p,sha256=h) for p,h in sources.items()]+list(refs.values())+list(budget_bindings.values())+
        [ref(SOURCE),ref(GUARD),ref(LAUNCHER),ref("scripts/launch_reserved_gpu_terminal_retired_scopes_v1.py"),
        ref("research/reviews/H039_train_stage_root_v1/external_GPU_sharing_authorization.json"),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        cfg["marker_repair_authority_ref"],ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),cfg["base_fit_config"]])
    assert len(consumers)<=64,("bindings",len(consumers))
    auth=dict(kind="H039_failed_scope_capacity_root_admission_v1",root_authorizer="root_agent",active=True,
        operation="reserve",accounting_mode=m.MODE,bindings=budget_bindings,expected_before=m._pins(view),
        activation_barrier_ref=ref("research/reviews/H039_failed_scope_capacity_v2/terminal_activation_barrier.json"),
        expires_at_utc=expiry,all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_H039_failed_scope_guard=True,
        all_old_guard_consumers_and_supervisors_terminal_before_activation=True,historical_records_mutation_authorized=False,
        global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,
        reservation=dict(key="H040/"+job,output=str(ROOT/output),auxiliary_scopes=[],reserved_bytes=CAP,consumer_bindings=consumers))
    authority_ref=save(packet,"reserve_authority.json",auth)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,job_id=job,
        output_directory=output,reserved_new_bytes=CAP,reservation=receipt["value"],expires_at_utc=expiry,
        initial_ledger_pins=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)),
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=ref(GUARD)))
    refs["storage"]=save(packet,"storage.json",storage)
    spec=dict(old,schema="H040_root_execution_input_v1",root_authorizer="root_agent",status="active",hypothesis_id="H040",
        phase="train",arm=arm,seed=101,device="cuda",fit_authorized=True,pyg_authorized=True,job_id=job,
        output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,refs=refs,source_sha256=sources,
        max_cpu_rss_bytes=512000000000,max_cuda_reserved_bytes=50000000000,root_context_module_ref=ref(CONTEXT))
    input_ref=save(packet,"input.json",spec)
    save(packet,"admitted.json",dict(input_ref=input_ref,launcher_ref=ref(LAUNCHER),driver_ref=ref(SOURCE),source_count=len(sources),consumer_binding_count=len(consumers),arm=arm,passes=30,updates=960,reserved_bytes=CAP,scientific_execution_started=False))
    print(json.dumps(dict(admitted=arm,input_ref=input_ref,consumers=len(consumers))),flush=True)
def launch(arm,gpu):
    assert (arm,gpu) in (("tensor",2),("event_mlp",3))
    packet=HERE/("admitted_"+arm);admitted=read(ref(packet/"admitted.json"));assert admitted["driver_ref"]==ref(SOURCE)
    assert admitted["launcher_ref"]==ref(LAUNCHER)
    m=load_guard();spec=read(admitted["input_ref"]);assert not (ROOT/spec["output_directory"]).exists()
    storage=read(spec["refs"]["storage"]);storage["initial_ledger_pins"]=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE))
    spec["refs"]["storage"]=save(packet,"launch_storage.json",storage)
    input_ref=save(packet,"launch_input.json",spec)
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m",LAUNCHER[:-3].replace("/","."),
        "--gpu",str(gpu),"--job",spec["job_id"],"--storage-record",spec["refs"]["storage"]["path"],
        "--expected-storage-sha256",spec["refs"]["storage"]["sha256"],"--expected-budget-source-sha256",CORE,
        "--expected-launcher-sha256",BASE,"arggnn.H040_latent_clade_execution_v1","--execute","--input",input_ref["path"],
        "--expected-input-sha256",input_ref["sha256"]]
    with (packet/"launch.log").open("x") as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=log,env=os.environ.copy())
        save(packet,"launch.json",dict(launcher_pid=proc.pid,arm=arm,physical_gpu=gpu,input_ref=input_ref,command=cmd,actual_wait_pending=True))
        print(json.dumps(dict(launched=arm,pid=proc.pid,gpu=gpu)),flush=True)
        rc=proc.wait()
    completed=ROOT/spec["output_directory"]/"completed.json"
    save(packet,"actual_wait.json",dict(launcher_pid=proc.pid,returncode=rc,actual_direct_launcher_wait=True,
        completed_ref=ref(completed) if completed.exists() else None,log_ref=ref(packet/"launch.log"),waited_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(arm=arm,returncode=rc,actual_wait=True)),flush=True);return rc
def main():
    p=argparse.ArgumentParser();p.add_argument("--admit",choices=("tensor","event_mlp"));p.add_argument("--launch",choices=("tensor","event_mlp"));p.add_argument("--gpu",type=int)
    a=p.parse_args()
    if a.admit:admit(a.admit);return 0
    if a.launch:return launch(a.launch,a.gpu)
    print("Source only");return 0
if __name__=="__main__":raise SystemExit(main())
