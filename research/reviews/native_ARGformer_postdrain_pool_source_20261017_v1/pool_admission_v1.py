"""Inactive fresh128MiB controller admission. ROOT alone may explicitly invoke."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,sys,time
from . import common_v1 as h
from . import pool_contract_v1 as c
ROOT=h.ROOT;SOURCE=h.HERE+"/pool_admission_v1.py"

def admit_v1(record):
    h.require(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site,"Exact fresh -B -s ROOT admission")
    q=h.read(record);h.active(q);h.source_refs_v1();c.reviewed_v1(q["source_review_ref"])
    h.require(q.get("schema")=="GPU0_postdrain_controller_ROOT_request_v1"and q.get("root_authorizer")=="root_agent"
        and q.get("status")=="active"and q.get("execution_authorized") is True
        and q.get("adapter_manifest_ref")==h.ref(h.MANIFEST)and q.get("storage_policy_selector_ref")==h.SELECTOR_REF
        and q.get("reservation_bytes")==128*2**20,"Exact explicit fresh controller ROOT request")
    h.require(isinstance(q.get("instance_id"),str)and q["instance_id"]and all(ch in "abcdefghijklmnopqrstuvwxyz0123456789_-"for ch in q["instance_id"]),"Safe fresh instance")
    stage=h.local(q["preparation_directory"]);output=h.local(q["controller_output_directory"])
    h.require(stage==h.local(record["path"]).parent and stage.is_dir()and q["preparation_directory"].startswith("research/reviews/GPU0_postdrain_ROOT_preparation_")
        and q["controller_output_directory"].startswith("research/reviews/GPU0_postdrain_controller_actual_")
        and not output.exists()and not stage.is_relative_to(output)and not output.is_relative_to(stage),"Fresh staging and never-reserved controller output")
    cap=dict(q["capability"]);c.preflight_capability_v1(cap,q);H058_job=next(n for n,j in cap["jobs"].items()if j["worker_kind"]=="H058")
    c.validate_handover_v1(cap["handover_ref"],H058_job,fresh=True)
    h.require(cap["controller_job_id"]=="GPU0-postdrain-controller-"+q["instance_id"]
        and cap["expires_unix"]>time.time()+86400+600
        and cap["drain_expires_unix"]<=datetime.fromisoformat(q["expires_at_utc"]).timestamp(),"Fresh full scientific interval and finite drain authority")
    g=h.guard_v1(consumer=False)
    runtime=h.save(stage,q["instance_id"]+".controller_runtime.json",dict(schema="GPU0_postdrain_controller_runtime_v1",python=str(ROOT/".runtime/env/bin/python"),source_ref=h.ref(c.POOL),physical_gpu=0,gpu_uuid=c.UUID))
    closure=h.save(stage,q["instance_id"]+".controller_closure.json",dict(schema="GPU0_postdrain_controller_closure_v1",adapter_manifest_ref=h.ref(h.MANIFEST),ROOT_request_ref=record,source_sha256=h.read(h.ref(h.MANIFEST))["source_sha256"]))
    from .root_driver_v1 import mutation_memo_v1
    memo=mutation_memo_v1(g,q);implementation=h.ref(g.budget.SOURCE_REL);launcher=h.ref(g.budget.LAUNCHER_REL)
    state=memo.read_state_v1(expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    consumers=h.unique([*state[3].values(),h.SELECTOR_REF,h.ref(h.MANIFEST),h.ref(SOURCE),h.ref(c.POOL),h.ref(c.SOURCE),record,q["source_review_ref"],runtime,closure])
    h.require(len(consumers)<=64,"Actual unchanged direct-role union64 checked before reserve")
    request=dict(key="GPU0-postdrain/"+cap["controller_job_id"],output=str(output),auxiliary_scopes=[],reserved_bytes=128*2**20,consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",request,expected_before=g._pins(state[2]),bindings=state[3]);authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    auth=h.save(stage,q["instance_id"]+".controller_reserve_authority.json",authority)
    receipt=memo.reserve_v1(auth,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    h.require(not output.exists(),"Controller output absent through actual reserve")
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,job_id=cap["controller_job_id"],output_directory=q["controller_output_directory"],reserved_new_bytes=128*2**20,reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=h.POLICY,storage_policy_selector_ref=h.SELECTOR_REF,budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=h.GUARD))
    storage_ref=h.save(stage,q["instance_id"]+".controller_storage.json",storage)
    cap.update(root_request_ref=record,plan_ref=record,controller_storage_ref=storage_ref,source_review_ref=q["source_review_ref"])
    cap_ref=h.save(stage,q["instance_id"]+".capability.json",cap);c.validate_cap_v1(cap_ref)
    receipt_ref=h.save(stage,q["instance_id"]+".controller_admitted.json",dict(schema="GPU0_postdrain_controller_admitted_v1",capability_ref=cap_ref,reserve_receipt=receipt,guard_binding_count=len(consumers),scientific_or_controller_execution_started=False))
    return dict(capability_ref=cap_ref,admitted_ref=receipt_ref,one_shot_start_argv=["setsid",str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",c.POOL[:-3].replace("/","."),"start","--execute","--capability",cap_ref["path"],"--sha256",cap_ref["sha256"]])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--admit",action="store_true");p.add_argument("--request-ref");a=p.parse_args(argv)
    if not a.admit:print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False)));return 0
    h.require(a.request_ref,"Exact ROOT request")
    print(json.dumps(admit_v1(json.loads(a.request_ref)),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
