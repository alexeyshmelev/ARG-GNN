"""Root-only prospective980GB activation. Default reads source metadata only.

Source workers never execute --prepare or --activate. The old v3 prefix must be
genuinely validated by ROOT after immediate H047 evaluation admission and a
cooperative future-mutation barrier. No existing policy/source/journal is edited.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/prospective_980GB_capacity_source_v1"
HERE=ROOT/REL
SOURCE=REL+"/root_activate_980_v1.py"
GUARD=REL+"/prospective_980gb_v1.py"
BASE="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py"
BASE_SHA="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef"
BASE_POLICY={"path":"research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json","sha256":"a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85"}
REPAIR="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"
BUDGET="scripts/project_budget_v9.py"
LAUNCHER="scripts/launch_reserved_gpu_v6.py"
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT),"Canonical own source/metadata")
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact metadata SHA")
    raw=(ROOT/record["path"]).read_bytes();require(len(raw)<=65536,"Compact root activation metadata")
    return json.loads(raw)
def save(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();require(len(raw)<=65536,"Bounded activation metadata")
    with (HERE/name).open("xb") as f:f.write(raw)
    return ref(HERE/name)
def barrier(record):
    v=read(record)
    require(v["schema"]=="prospective980_root_future_mutation_barrier_v1" and v["root_authorizer"]=="root_agent"
      and v["active"] is True and v["future_guard_ref"]==ref(GUARD)
      and v["all_future_old_v3_resource_accounting_mutations_cooperatively_revoked"] is True
      and v["all_existing_charges_and_live_guards_preserved"] is True
      and v["no_rebase_deletion_or_old_source_mutation"] is True
      and v["H046_future_evaluation_uses_separate_successor_IO_integration"] is True
      and v["unresolved_old_future_mutation_capabilities"]==[],
      "Actual root prospective freeze/revocation; existing live jobs preserved")
    a=read(v["H047_evaluation_admission_ref"])
    require(v["H047_evaluation_admission_ref"]["path"]=="research/reviews/H047_GPU45_cohort_risk_root_source_v1/admitted_evaluate/admitted.json"
      and a["phase"]=="evaluate" and a["physical_gpu"] in (4,5),"Original H047 evaluation admitted before ancestor freeze")
    return v
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def source_refs():
    pins=read(ref(HERE/"frozen_refs.json"))["refs"]
    require(pins[GUARD]==ref(GUARD) and ref(BASE)["sha256"]==BASE_SHA and ref(BASE_POLICY["path"])==BASE_POLICY,"Exact new source and immutable old reader/policy")
    return pins
def base_snapshot():
    """Root-only read through unchanged guard under its original mutex."""
    source_refs();m=module(BASE,"_root_prospective980_actual_v3_base");m.configure_v2(ref(REPAIR),policy_ref=BASE_POLICY)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(ref(BUDGET)["sha256"],ref(LAUNCHER)["sha256"],fresh=True)
    require(len(state)==7,"Exact original v3 state API")
    return m,state
def prepare(barrier_ref):
    barrier(barrier_ref);old,state=base_snapshot();new=module(GUARD,"_root_prospective980_source_draft")
    require(not (ROOT/new.JOURNAL).exists() and not (HERE/"activation_policy.root_active_v1.json").exists(),"Never replay activation or reuse own journal")
    draft=new.activation_policy_draft_v1(repair_authority_ref=ref(REPAIR),base_pins=old._pins(state[2]),
      base_projected_bytes=state[2]["projected_bytes"],implementation_ref=ref(BUDGET),launcher_ref=ref(LAUNCHER))
    pr=save("activation_policy.draft_root_v1.json",draft)
    save("activation_preparation.root_v1.json",dict(status="inactive_draft_from_actual_root_v3_state",
      draft_ref=pr,barrier_ref=barrier_ref,base_prefix=old._pins(state[2]),base_projected_bytes=state[2]["projected_bytes"],
      no_old_mutation=True,root_actual_guard_read=True,created_utc=datetime.now(timezone.utc).isoformat()))
    return dict(policy_draft_ref=pr,active=False,base_projected_bytes=state[2]["projected_bytes"])
def activate(draft_ref,barrier_ref):
    barrier(barrier_ref);old,state=base_snapshot();new=module(GUARD,"_root_prospective980_activation")
    draft=read(draft_ref)
    expected=new.activation_policy_draft_v1(repair_authority_ref=ref(REPAIR),base_pins=old._pins(state[2]),
      base_projected_bytes=state[2]["projected_bytes"],implementation_ref=ref(BUDGET),launcher_ref=ref(LAUNCHER))
    require(draft==expected and draft["active"] is False,"Root-reviewed exact draft and still-identical original prefix")
    require(not (ROOT/new.JOURNAL).exists() and not (HERE/"activation_policy.root_active_v1.json").exists(),"Fresh prospective activation, no replay")
    policy=copy.deepcopy(draft);policy["active"]=True
    policy_ref=save("activation_policy.root_active_v1.json",policy)
    new.configure_v2(ref(REPAIR),policy_ref=policy_ref)
    view=new.inspect_v1(expected_source_sha256=ref(BUDGET)["sha256"],expected_launcher_sha256=ref(LAUNCHER)["sha256"])
    require(view["completed_overlay_prefix"]==dict(events=0,tail_sha256=None) and view["new_reserved_bytes"]==0
      and view["completed_credit_bytes"]==0,"Activation charges bootstrap only; no new credit/reservation")
    rr=save("activation_receipt.root_v1.json",dict(status="activated_prospective980_root",
      policy_ref=policy_ref,guard_ref=ref(GUARD),base_ref=ref(BASE),base_policy_ref=BASE_POLICY,
      root_barrier_ref=barrier_ref,base_prefix=policy["base_pins"],base_projected_bytes=policy["base_projected_bytes"],
      projected_bytes=view["projected_bytes"],warning_active=view["warning_active"],warning_message=view["warning_message"],
      ceiling_bytes=980000000000,warning_threshold_bytes=950000000000,user_limit_bytes=1000000000000,
      no_existing_sources_or_journals_modified=True,no_rebase_or_deletion=True,
      no_new_admission_or_completed_credit=True,old_live_consumers_keep_original_guards=True,
      created_utc=datetime.now(timezone.utc).isoformat()))
    return dict(policy_ref=policy_ref,activation_receipt_ref=rr,projected_bytes=view["projected_bytes"],
      warning_active=view["warning_active"],warning_message=view["warning_message"])
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument("--prepare",action="store_true");g.add_argument("--activate",action="store_true")
    for flag in ["barrier","draft"]:
        p.add_argument("--"+flag);p.add_argument("--"+flag+"-sha256")
    a=p.parse_args()
    if a.prepare or a.activate:
        require(a.barrier and a.barrier_sha256,"Exact actual root barrier")
        br=dict(path=a.barrier,sha256=a.barrier_sha256)
        if a.prepare:result=prepare(br)
        else:
            require(a.draft and a.draft_sha256,"Exact reviewed inactive policy draft")
            result=activate(dict(path=a.draft,sha256=a.draft_sha256),br)
    else:result=dict(status="inactive_source_only",source_ref=ref(SOURCE),new_guard_ref=ref(GUARD),
       old_guard_ref=dict(path=BASE,sha256=BASE_SHA),warning_bytes=950000000000,ceiling_bytes=980000000000,
       guard_configuration_or_state_reads=0,activation_or_admission=0)
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=="__main__":main()
