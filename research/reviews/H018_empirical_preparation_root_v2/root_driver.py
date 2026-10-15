"""Root-only H018 CPU observation preparation. No model, checkpoint or CUDA."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import fields
import argparse, hashlib, json, os, sys, subprocess
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.reviews.H047_GPU45_cohort_risk_root_source_v1 import root_common_v1 as common
HERE=ROOT/"research/reviews/H018_empirical_preparation_root_v2"
SOURCE=str(Path(__file__).relative_to(ROOT))
OUTPUT="data/H018_empirical_observation_bank_root_v2"
CAP=20000000
def ref(path):
    p=Path(path)
    if not p.is_absolute():
        p=ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def save(name,value):
    path=HERE/name
    with path.open("x") as f:
        json.dump(value,f,indent=2,sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return ref(path)
class Reader:
    root=ROOT
    def read(self,record):
        p=ROOT/record["path"]
        assert p.resolve()==p and p.is_relative_to(ROOT) and p.is_file()
        raw=p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record["sha256"]
        return raw
    def decode_json(self,record):
        return json.loads(self.read(record))
def admit():
    assert not (HERE/"admitted.json").exists() and not (ROOT/OUTPUT).exists()
    reader=Reader()
    cfg=reader.decode_json(ref("research/reviews/H018_empirical_reader_source_v1/config_inactive.json"))
    manifest=reader.decode_json(cfg["source_manifest_ref"])
    for p,d in manifest["source_sha256"].items():
        reader.read(dict(path=p,sha256=d))
    for r in manifest["metadata_refs"].values():
        reader.read(r)
    cfg.update(status="root_admitted_observation_preparation",preparation_authorized=True,
        resources_are_proposals_pending_root_review=False,output_cap_bytes=CAP,max_cpu_rss_bytes=2000000000)
    cfg_ref=save("config_root_v1.json",cfg)
    now=datetime.now(timezone.utc)
    expiry=(now+timedelta(hours=24)).isoformat()
    job="H018-empirical-original-GT-graph-CPU-prepare-v2"
    runtime_ref=save("runtime.json",dict(job_id=job,device="cpu",CUDA_VISIBLE_DEVICES="",
        output_directory=OUTPUT,created_at_utc=now.isoformat(),expires_at_utc=expiry,
        model_or_checkpoint_or_GPU_authorized=False))
    m,_=common.load_guard_v1(consumer=False)
    with m.budget.v4.mutex(ROOT):
        state=m._state_locked(common.CORE,common.BASE,fresh=True)
    view,bindings=state[2],state[3]
    consumers=common.unique([*bindings.values(),*[ref(p) for p in manifest["source_sha256"]],
        *[ref(r["path"]) for r in manifest["metadata_refs"].values()],ref(SOURCE),cfg_ref,runtime_ref,ref(cfg["source_manifest_ref"]["path"]),
        common.GUARD_REF,common.POLICY_REF,common.REPAIR_REF,common.MEMO_REF,common.V4_REF,
        ref("research/reviews/H047_GPU45_cohort_risk_root_source_v1/root_common_v1.py"),
        ref("research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"),
        ref("src/arggnn/project_budget_v9_consumer.py"),ref("research/reviews/H018_empirical_reader_source_v1/source_handoff.json")])
    request=dict(key="H018/"+job,output=str(ROOT/OUTPUT),auxiliary_scopes=[],
        reserved_bytes=CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    auth_ref=save("reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(auth_ref,expected_source_sha256=common.CORE,expected_launcher_sha256=common.BASE)
    save("reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=OUTPUT,reserved_new_bytes=CAP,
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=common.POLICY_REF,
        budget_components=dict(implementation=ref(m.budget.SOURCE_REL),launcher=ref(m.budget.LAUNCHER_REL),
            consumer_adapter=common.GUARD_REF))
    spec=dict(schema="H018_root_CPU_observation_preparation_input_v1",job_id=job,output_directory=OUTPUT,
        config_ref=cfg_ref,runtime_ref=runtime_ref,storage_ref=save("storage.json",storage),
        producer_ref=ref(SOURCE),root_authorizer="root_agent",model_or_checkpoint_or_GPU_authorized=False)
    record=save("input.json",spec)
    save("admitted.json",dict(input_ref=record,reserved_bytes=CAP,actual_preparation_started=False))
    print(json.dumps(dict(event="admitted_H018_CPU_preparation",input_ref=record)),flush=True)
def prepare():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    reader=Reader()
    admitted=reader.decode_json(ref(HERE/"admitted.json"))
    spec=reader.decode_json(admitted["input_ref"])
    assert spec["producer_ref"]==ref(SOURCE) and spec["model_or_checkpoint_or_GPU_authorized"] is False
    cfg=reader.decode_json(spec["config_ref"])
    storage=reader.decode_json(spec["storage_ref"])
    m,memo=common.load_guard_v1(consumer=True)
    m.validate_storage_v9(spec,cfg,storage,source_ref=spec["producer_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    output=m.PilotOutputV9(OUTPUT,storage,reader)
    checks=[]
    def before(stage,plan):
        parts=Path("/proc/self/statm").read_text().split()
        rss=int(parts[1])*os.sysconf("SC_PAGE_SIZE")
        allowance=cfg["CPU_preparation_cap_bytes"]
        assert rss+allowance<=cfg["max_cpu_rss_bytes"]
        checks.append(dict(stage=stage,current_RSS_bytes=rss,planning_allowance_bytes=allowance,plan=plan))
    from research.reviews.H018_empirical_reader_source_v1.reader_v1 import prepare_from_root_context_v1
    result=prepare_from_root_context_v1(dict(root_authorizer="root_agent",admitted=True,cfg=cfg,
        config_ref=spec["config_ref"],reader=reader,before_stage=before))
    before("serialize_complete_CPU_bank",{})
    bank=result["bank"]
    payload=dict(schema="H018_empirical_CPU_observation_bank_v1",input_ref=admitted["input_ref"],
        bank_tensors={f.name:getattr(bank,f.name) for f in fields(bank)},
        flag_node_flags=result["flags"].node_flags,metadata=result["metadata"])
    bank_ref=output.save("observation_bank.pt",payload,kind="torch")
    done=output.save("completed.json",dict(schema="H018_empirical_CPU_preparation_completed_v1",status="completed",
        input_ref=admitted["input_ref"],producer_ref=ref(SOURCE),bank_ref=bank_ref,metadata=result["metadata"],
        resource_checks=checks,memo_receipt=memo.receipt,no_model_or_checkpoint_executed=True,
        no_query_ancestry_truth=True,no_GPU_used=True,completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(event="H018_CPU_preparation_completed",completion_ref=done)),flush=True)
def launch():
    assert not (HERE/"launch.json").exists()
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=""
    command=[str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/SOURCE),"--prepare"]
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    save("launch.json",dict(pid=proc.pid,command=command,actual_wait_pending=True))
    retained=bytearray()
    received=0
    for data in iter(lambda:proc.stdout.read1(65536),b""):
        received+=len(data)
        retained.extend(data[:max(0,1000000-len(retained))])
    rc=proc.wait()
    print(json.dumps(dict(event="H018_actual_direct_CPU_wait",pid=proc.pid,returncode=rc)),flush=True)
    save("actual_wait.json",dict(pid=proc.pid,returncode=rc,actual_direct_Popen_wait=True,
        waited_utc=datetime.now(timezone.utc).isoformat(),stdout_bytes_received=received,stdout_truncated=received>len(retained)))
    with (HERE/"worker_stdout.log").open("xb") as f:
        f.write(retained)
        f.flush()
        os.fsync(f.fileno())
    print(retained.decode(errors="replace")[-6000:],flush=True)
    return rc
def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group()
    g.add_argument("--admit",action="store_true")
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--launch",action="store_true")
    a=p.parse_args()
    if a.admit:
        admit()
    elif a.prepare:
        prepare()
    elif a.launch:
        return launch()
    else:
        print("Inactive root CPU preparation driver")
    return 0
if __name__=="__main__":
    raise SystemExit(main())
