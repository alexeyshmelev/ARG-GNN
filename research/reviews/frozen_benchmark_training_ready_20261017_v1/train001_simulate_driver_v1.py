"""ROOT entry for one concrete missing full30Mb TRAIN001 simulator stage.

Reuses the reviewed original CPU supervisor execute() and finite cleanup exactly.
ROOT supplies actual request/capacity/directive/review. No automatic retry/close.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from . import train001_simulate_context_v1 as context


def admit(request_ref):
    c,d=context.initialize()
    q,caps,_=c.validate_request(request_ref);packet,output=c.phase_paths(c.PHASE)
    c.require(not packet.exists()and not c.local(output).exists(),"Fresh exclusive packet/output")
    g=c.original_guard()
    implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(c.ROOT):
        state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    packet.mkdir()
    config_ref=c.save(packet,"config.json",c.make_config(q,caps))
    sources={path:c.ref(path)["sha256"]for path in c.source_paths()}
    closure=c.save(packet,"source_and_input_closure.json",dict(
        schema="frozen_TRAIN001_simulator_actual_source_closure_v1",
        source_sha256=sources,root_stage_request_ref=request_ref,
        bindings=q["bindings"],native_MLM_dependency=False))
    consumers=c.unique([*state[3].values(),*c.direct_roles(request_ref,q,config_ref,closure,q["runtime_ref"])])
    job="TreeLoom-frozen-full30Mb-TRAIN001-simulate-"+q["scope_id"]
    aux=packet/"worker_aux"
    reserve=dict(key="H041/"+job,output=str(c.ROOT/output),auxiliary_scopes=[str(aux)],
        reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=c.save(packet,"reserve_authority.json",authority)
    receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    rr=c.save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=job,output_directory=output,
        reserved_new_bytes=caps["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
    sr=c.save(packet,"storage.json",storage)
    spec=dict(schema="frozen_TRAIN001_current_Census_simulator_input_v1",
        status="active",root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,
        parent_id=c.PARENT,phase=c.PHASE,stage=c.STAGE,job_id=job,output_directory=output,
        worker_auxiliary_scope=str(aux.relative_to(c.ROOT)),root_stage_request_ref=request_ref,
        root_driver_ref=c.ref(context.DRIVER),main_source_ref=c.ref(context.WORKER),
        source_sha256=sources,transitive_source_closure_ref=closure,caps_ref=q["caps_ref"],
        refs=dict(config=config_ref,runtime=q["runtime_ref"],storage=sr),
        runtime_ref=q["runtime_ref"],storage=storage,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False)
    ir=c.save(packet,"input.json",spec)
    c.save(packet,"admitted.json",dict(input_ref=ir,storage=storage,reserve_receipt_ref=rr,
        driver_ref=c.ref(context.DRIVER),consumer_binding_count=len(consumers),scientific_execution_started=False))
    return dict(input_ref=ir,consumer_binding_count=len(consumers),execution_started=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--admit-ref");group.add_argument("--execute-ref")
    a=p.parse_args(argv)
    if not(a.admit_ref or a.execute_ref):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)))
        return 0
    c,d=context.initialize()
    c.require(os.getcwd()==str(c.ROOT)and sys.dont_write_bytecode and sys.flags.no_user_site,
        "Exact existing ROOT -B -s interpreter and repository cwd")
    if a.admit_ref:
        result=admit(json.loads(a.admit_ref))
    else:
        record=json.loads(a.execute_ref)
        s,q,_=d.input_spec(record)
        c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"Actual selected CPUs available to ROOT")
        os.sched_setaffinity(0,set(q["cpu_ids"]))
        with c.cpu_slot()as fd:
            d.SLOT_FD=fd
            result=d.execute(record)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
