
"""Root-only C3 simulation admission and direct CPU process wait."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
HERE=ROOT/"research/reviews/American_C3_root_execution_20261015_v1"
SOURCE="research/reviews/American_C3_root_execution_20261015_v1/root_simulation_driver.py"
GUARD="research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py"
COMMON="src/arggnn/american_c3_common_v1.py"
WORKER="src/arggnn/american_c3_simulate_v1.py"
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
    assert ref(r["path"])==r
    return json.loads((ROOT/r["path"]).read_text())
def save(folder,name,value):
    p=folder/name
    with p.open("x") as f:json.dump(value,f,indent=2);f.write("\n")
    return ref(p)
def guard():
    s=importlib.util.spec_from_file_location("root_C3_current_guard",ROOT/GUARD);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.configure_v2(ref("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"));return m
def admit(stage):
    assert stage in ("qualification100kb","full30mb")
    packet=HERE/("admitted_"+stage+"_v2");assert not packet.exists();packet.mkdir()
    draft="research/reviews/ARGformer_American_C3_implementation_20261015_v1/simulation_"+stage+"_inactive.json"
    cfg=read(ref(draft));cfg["execution_enabled"]=True
    config_ref=save(packet,"config.json",cfg)
    from arggnn.american_c3_common_v1 import runtime_report
    runtime_ref=save(packet,"runtime.json",runtime_report())
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    job="American-C3-"+stage+"-20261015";output="experiments/results/American_C3_"+stage+"_20261015_v1"
    assert not (ROOT/output).exists()
    m=guard()
    with m.budget.v4.mutex(ROOT):_,_,view,bindings,_=m._state_locked(CORE,BASE,fresh=True)
    closure=[ref(COMMON),ref(WORKER)]
    consumers=[*closure,config_ref,runtime_ref,ref(SOURCE),*bindings.values(),ref(GUARD),
        ref("research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py"),
        ref("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"),
        ref("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py"),
        ref("src/arggnn/project_budget_v9_consumer.py")]
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    auth=dict(kind="H039_failed_scope_capacity_root_admission_v1",root_authorizer="root_agent",active=True,operation="reserve",
        accounting_mode=m.MODE,bindings=bindings,expected_before=m._pins(view),activation_barrier_ref=ref("research/reviews/H039_failed_scope_capacity_v2/terminal_activation_barrier.json"),
        expires_at_utc=expiry,all_budget_writers_use_scheduler_mutex=True,all_consumers_and_launchers_use_H039_failed_scope_guard=True,
        all_old_guard_consumers_and_supervisors_terminal_before_activation=True,historical_records_mutation_authorized=False,
        global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,
        reservation=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[],reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers))
    a=save(packet,"reserve_authority.json",auth);receipt=m.apply_reserve_v1(a,expected_source_sha256=CORE,expected_launcher_sha256=BASE);save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=m.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],reservation=receipt["value"],expires_at_utc=expiry,
        initial_ledger_pins=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE)),
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=ref(GUARD)))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,execution_authorized=True,GPU_authorized=False,
        source_ref=ref(WORKER),source_closure_refs=closure,config_ref=config_ref,runtime_ref=runtime_ref,
        storage_adapter_ref=ref(GUARD),marker_repair_authority_ref=ref("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"),
        storage=storage,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry)
    a=save(packet,"admission.json",spec);save(packet,"admitted.json",dict(admission_ref=a,driver_ref=ref(SOURCE),cap_bytes=cfg["output_cap_bytes"],execution_started=False))
    print(json.dumps(dict(stage=stage,admitted=True,admission_ref=a)),flush=True)
def launch(stage):
    packet=HERE/("admitted_"+stage+"_v2");admitted=read(ref(packet/"admitted.json"));assert admitted["driver_ref"]==ref(SOURCE)
    spec=read(admitted["admission_ref"]);assert not (ROOT/spec["output_directory"]).exists()
    m=guard();spec["storage"]["initial_ledger_pins"]=m._pins(m.inspect_v1(expected_source_sha256=CORE,expected_launcher_sha256=BASE))
    a=save(packet,"launch_admission.json",spec)
    cmd=[str(ROOT/".runtime/env/bin/python"),"-B","-m","arggnn.american_c3_simulate_v1","--execute","--admission",a["path"],"--expected-admission-sha256",a["sha256"]]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with (packet/"launch.log").open("x") as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=f)
        save(packet,"launch.json",dict(pid=proc.pid,command=cmd,admission_ref=a,actual_wait_pending=True))
        print(json.dumps(dict(stage=stage,pid=proc.pid,started=True)),flush=True);rc=proc.wait()
    save(packet,"actual_wait.json",dict(pid=proc.pid,returncode=rc,actual_direct_worker_wait=True,finished_utc=datetime.now(timezone.utc).isoformat(),log_ref=ref(packet/"launch.log")))
    print(json.dumps(dict(stage=stage,returncode=rc,actual_wait=True)),flush=True);return rc
def main():
    p=argparse.ArgumentParser();p.add_argument("--admit",choices=("qualification100kb","full30mb"));p.add_argument("--launch",choices=("qualification100kb","full30mb"));a=p.parse_args()
    if a.admit:admit(a.admit);return0=0;return return0
    if a.launch:return launch(a.launch)
    print("source only");return 0
if __name__=="__main__":raise SystemExit(main())
