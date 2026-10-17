"""Fresh65GiB downstream master admission and late-bound stage metadata. Inert."""
from pathlib import Path
import copy,sys
from . import common_v1 as h
from . import native_common_v1 as n
from . import downstream_contract_v1 as c
def admit_master_v1(request_ref):
    q=h.read(request_ref);h.active(q);h.source_refs_v1()
    h.require(q.get("schema")=="native_postdrain_downstream_master_ROOT_request_v1" and q.get("root_authorizer")=="root_agent"
        and q.get("execution_authorized") is True and q.get("template_only") is False and q["reservation_bytes"]==65*2**30
        and q["adapter_manifest_ref"]==h.ref(h.MANIFEST) and q["capacity_guard_ref"]==h.GUARD and q["capacity_policy_ref"]==h.POLICY
        and q["storage_policy_selector_ref"]==h.SELECTOR_REF,"Exact fresh current guard/source65GiB downstream request")
    from .pool_contract_v1 import reviewed_v1,validate_cap_v1
    reviewed_v1(q["source_review_ref"]);cap=validate_cap_v1(q["postdrain_pool_capability_ref"])
    parent=h.read(h.ref(n.PARENT_REGISTRY))["parents"][q["parent_id"]]
    h.require(q["preparation_ref"]==parent["preparation_ref"] and q["MLM_prerequisites"] is not None
        and q["original_MLM7GiB_scope_extended"] is False,"Exact original-parent corpus and genuine MLM prerequisites, no old scope extension")
    stage=h.local(q["root_preparation_directory"]);out=h.local(q["output_directory"]);aux=h.local(q["worker_auxiliary_scope"])
    h.require(stage==h.local(request_ref["path"]).parent and stage.is_dir() and not out.exists() and not aux.exists()
        and q["output_directory"].startswith("experiments/results/native_postdrain_downstream_")
        and q["worker_auxiliary_scope"].startswith("research/reviews/native_postdrain_downstream_AUX_"),"Existing charged preparation, two fresh complete roots")
    h.require(h.datetime.fromisoformat(q["expires_at_utc"]).timestamp()<=cap["expires_unix"],"Complete downstream master inside new pool lifetime")
    config=h.read(q["config_ref"]);base=h.read(h.ref(h.HERE+"/downstream_scientific_base.v1.json"));counts=h.read(q["counts_ref"])
    base["scientific_protocol"]["contrastive"]["updates"]=counts["datasets"]["contrastive/train"]["rows"]//4096
    base["scientific_protocol"]["retrieval"]["reference_rows"]=sum(counts["datasets"]["contrastive/"+s]["rows"]for s in("train","val"))
    h.require(config==base,"Exact all-four original author embeddings/normalization/pooling/top1/batch/chunk/config and65GiB bound")
    g=h.guard_v1(consumer=False)
    from .root_driver_v1 import mutation_memo_v1
    memo=mutation_memo_v1(g,q);implementation=h.ref(g.budget.SOURCE_REL);launcher=h.ref(g.budget.LAUNCHER_REL)
    state=memo.read_state_v1(expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    bindings=h.unique([*state[3].values(),h.ref(h.MANIFEST),h.ref(c.DRIVER),h.ref(c.CONTEXT),request_ref,q["config_ref"],q["runtime_ref"],q["source_review_ref"],h.SELECTOR_REF])
    h.require(len(bindings)<=64,"Actual unchanged direct-role union64 checked before reserve")
    reserve=dict(key="native-postdrain-downstream/"+q["job_id"],output=str(out),auxiliary_scopes=[str(aux)],reserved_bytes=65*2**30,consumer_bindings=bindings)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3]);authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=h.save(stage,"downstream_reserve_authority.json",authority)
    receipt=memo.reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=g.MODE,job_id=q["job_id"],output_directory=q["output_directory"],reserved_new_bytes=65*2**30,reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],completed_overlay_policy_ref=h.POLICY,storage_policy_selector_ref=h.SELECTOR_REF,budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=h.GUARD))
    storage_ref=h.save(stage,"downstream_storage.json",storage)
    master=dict(q,schema="native_VAL000_downstream_ROOT_master_input_v1",root_stage_request_ref=request_ref,source_manifest_ref=h.ref(h.MANIFEST),source_review_passed=True,
        source_ref=h.ref(c.DRIVER),main_source_ref=h.ref(c.DRIVER),source_sha256=h.read(h.ref(h.MANIFEST))["source_sha256"],
        refs=dict(config=q["config_ref"],runtime=q["runtime_ref"],storage=storage_ref),storage_ref=storage_ref,
        future_storage_reserved_bytes=65*2**30,owned_roster_policy_ref=c.OWNED_ROSTER_POLICY_REF,operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF)
    mr=h.save(stage,"downstream_master_input.json",master)
    return dict(master_input_ref=mr,storage_ref=storage_ref,actual_reserve_receipt=receipt,execution_started=False,no_automatic_phase_launch=True)
def stage_input_v1(authority_ref):
    q=h.read(authority_ref);c.active(q);master=h.read(q["master_input_ref"]);c.active(master)
    h.require(q["schema"]=="native_VAL000_downstream_ROOT_phase_authority_v1" and q["phase"]in c.PHASES
        and q["job_id"]==master["job_id"]+"--"+q["phase"] and q["source_manifest_ref"]==h.ref(h.MANIFEST)
        and q["capacity_guard_ref"]==h.GUARD and q["capacity_policy_ref"]==h.POLICY and q["storage_policy_selector_ref"]==h.SELECTOR_REF,
        "Exact finite fresh ROOT phase authority in existing65GiB master, no reservation reset")
    s=dict(schema="native_VAL000_downstream_ROOT_phase_input_v1",root_authorizer="root_agent",status="active",execution_authorized=True,
        job_id=q["job_id"],parent_id=q["parent_id"],phase=q["phase"],root_stage_request_ref=authority_ref,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],output_directory=master["output_directory"]+"/"+q["phase"],
        worker_auxiliary_scope=master["worker_auxiliary_scope"]+"/"+q["phase"],refs=dict(config=master["config_ref"],runtime=master["runtime_ref"],storage=master["storage_ref"]))
    r=h.save(h.local(authority_ref["path"]).parent,"phase_input.json",s);c.inspect_input_v1(r)
    return dict(input_ref=r,execute_only_through_new_native_pool_slot=True,reservation_mutation=False)

def main(argv=None):
    import argparse,json
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--admit-master",action="store_true");action.add_argument("--stage",action="store_true")
    parser.add_argument("--request-ref");args=parser.parse_args(argv)
    if not(args.admit_master or args.stage):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    h.require(Path.cwd()==h.ROOT and sys.dont_write_bytecode and sys.flags.no_user_site and args.request_ref,"Exact finite -B -s ROOT entry")
    result=(admit_master_v1 if args.admit_master else stage_input_v1)(json.loads(args.request_ref))
    print(json.dumps(result,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
