"""ROOT-only future EVAL reservation/admission; default inert, never launches."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_native_EVAL_GPU0_concurrent_source_20261016_v1/eval_v1"
SOURCE=HERE+"/root_admit_eval_v1.py"
def contract():
    p=ROOT/HERE/"contract_eval_v1.py";sp=importlib.util.spec_from_file_location("_C3_EVAL_root_contract",p)
    c=importlib.util.module_from_spec(sp);sys.modules[sp.name]=c;sp.loader.exec_module(c);return c
def admit_v1(request_ref):
    c=contract();q,caps=c.validate_request_v1(request_ref)
    c.require(Path.cwd().resolve()==ROOT and Path(__file__).resolve()==ROOT/SOURCE,"Exact ROOT admission source/cwd")
    packet=c.local(q["packet_directory"]);output=c.local(q["output_directory"]);tmp=c.local(q["short_tmp_directory"]);aux=packet/"worker_aux"
    c.require(not any(p.exists() for p in (packet,output,tmp)),"Fresh exclusive actual EVAL packet/output/TMP")
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    guard=c.capacity_guard_v1()
    packet.mkdir()
    def save(name,value):
        p=packet/name
        with p.open("x") as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
        return c.ref(str(p.relative_to(ROOT)))
    config=dict(schema="C3_full30Mb_EVAL_operational_config_v1",output_cap_bytes=caps["reservation_cap_bytes"],phase_write_budget_bytes=caps["phase_write_budget_bytes"],short_tmp_directory=q["short_tmp_directory"])
    config_ref=save("config.json",config)
    closure_ref=save("source_and_input_closure.json",dict(schema="C3_full30Mb_EVAL_transitive_closure_v1",root_stage_request_ref=request_ref,source_sha256=q["source_sha256"],parent=q["parent"],training=q["training"],native_MLM_dependency=False))
    implementation=c.ref(guard.budget.SOURCE_REL);launcher=c.ref(guard.budget.LAUNCHER_REL)
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    mandatory=[c.ref(SOURCE),c.ref(c.ENTRY),c.ref(c.SOURCE),request_ref,q["caps_ref"],q["science_source_pins_ref"],c.ref(c.BOUNDS),config_ref,closure_ref,c.RUNTIME_REF,c.BASE_CONFIG_REF,c.HOOK_REF,c.CAPACITY_REF,c.POLICY_REF,c.REPAIR_REF,c.RENEWAL_REF,c.RENEWAL_CERT_REF,c.GPU_GUARD_REF,c.ref("src/arggnn/biological_candidate.py"),c.ref("src/arggnn/census_tsinfer_fit_v1.py"),c.ref("src/arggnn/development_execution.py")]
    consumers=c.unique([*state[3].values(),*mandatory])
    reserve_request=dict(key="H041/"+q["job_id"],output=str(output),auxiliary_scopes=[str(aux),str(tmp)],reserved_bytes=caps["reservation_cap_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",reserve_request,expected_before=guard._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    authority_ref=save("reserve_authority.json",authority)
    receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    receipt_ref=save("reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=caps["reservation_cap_bytes"],reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=c.POLICY_REF,budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.CAPACITY_REF))
    storage_ref=save("storage.json",storage)
    spec=dict(schema="C3_full30Mb_EVAL_active_input_v1",status="active",root_authorizer="root_agent",execution_authorized=True,GPU_authorized=True,parent_id=q["parent_id"],split="val",job_id=q["job_id"],phase="eval",output_directory=q["output_directory"],root_driver_ref=c.ref(SOURCE),root_auxiliary_directory=str(aux.relative_to(ROOT)),short_tmp_directory=q["short_tmp_directory"],operational_GPU0_guard_ref=c.GPU_GUARD_REF,main_source_ref=c.ref(c.ENTRY),operational_GPU0_runtime_ref=c.ref(c.ENTRY),root_stage_request_ref=request_ref,caps_ref=q["caps_ref"],refs=dict(config=config_ref,runtime=c.RUNTIME_REF,storage=storage_ref),transitive_source_closure_ref=closure_ref,source_sha256=q["source_sha256"],physical_gpu=0,execution_gpu_uuid=c.GPU_UUID,max_cpu_rss_bytes=caps["max_cpu_rss_bytes"],max_cuda_reserved_bytes=caps["max_cuda_reserved_bytes"],created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=c.POLICY_REF)
    input_ref=save("input.json",spec);method=c.method_v1(input_ref)
    save("admitted.json",dict(schema="C3_full30Mb_EVAL_admission_v1",input_ref=input_ref,method=method,consumer_binding_count=len(consumers),reserve_receipt_ref=receipt_ref,execution_started=False))
    return dict(input_ref=input_ref,method=method,consumer_binding_count=len(consumers))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--admit",action="store_true");p.add_argument("--request-ref");a=p.parse_args(argv)
    if not a.admit:
        print(json.dumps(dict(status="source_only_no_admission",execution=False)));return 0
    if not a.request_ref:raise ValueError("Explicit actual ROOT request ref required")
    print(json.dumps(admit_v1(json.loads(a.request_ref))));return 0
if __name__=="__main__":raise SystemExit(main())
