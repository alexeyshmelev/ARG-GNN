"""Tiny source/metadata contracts only. No real guard configuration or scientific payload."""
from pathlib import Path
import ast, copy, hashlib, json, types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
BASE="research/reviews/European_outer8_preparation_source_20261016_v3"
CHILD="research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1"
def run():
    checks=[]
    def need(ok,msg):
        if not ok:raise ValueError(msg)
    def good(msg,ok=True):need(ok,msg);checks.append(dict(name=msg,status="PASS"))
    def reject(msg,fn):
        try:fn()
        except (ValueError,KeyError,AssertionError):good(msg);return
        raise AssertionError("Missing rejection: "+msg)
    def ref(path):
        return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    def fake(name):return dict(path="SYNTHETIC_FUTURE_ROOT_METADATA_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
    def defs(path,names,env):
        tree=ast.parse((ROOT/path).read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        need({n.name for n in nodes}==set(names),"Exact named source functions")
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),path,"exec"),env)
    def fn(path,name):
        return next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    def dump(n):return ast.dump(n,include_attributes=False)
    base=BASE+"/prepare_outer000_v1.py";candidate=HERE+"/prepare_outer000_v1.py"
    for name in ("need","local","ref","raw","read","digest","load","current","sources","freeze","put","boundary","sample_scope","main"):
        good(name+"_original_AST_unchanged",dump(fn(base,name))==dump(fn(candidate,name)))
    # Every inherited worker/lifecycle statement is unchanged except explicit selected guard argument.
    for name in ("worker","execute"):
        old=ast.get_source_segment((ROOT/base).read_text(),fn(base,name)).replace('g=guard(True)','g=guard(q["policy_selector_ref"],True)')
        good(name+"_only_selected_guard_call_changed",dump(ast.parse(old).body[0])==dump(fn(candidate,name)))
    for name in ("run_v3","main"):
        old=ast.get_source_segment((ROOT/BASE/"check_prediction_barrier_v2.py").read_text(),fn(BASE+"/check_prediction_barrier_v2.py",name))
        old=old.replace("from .outer_prediction_validator_v2 import ","from research.reviews.European_outer8_preparation_source_20261016_v3.outer_prediction_validator_v2 import ")
        old=old.replace("from .outer_adapters_v2 import ","from research.reviews.European_outer8_preparation_source_20261016_v3.outer_adapters_v2 import ")
        old=old.replace("from .score_outer_v2 import ","from research.reviews.European_outer8_preparation_source_20261016_v3.score_outer_v2 import ")
        good("consumer_"+name+"_math_and_barrier_unchanged",dump(ast.parse(old).body[0])==dump(fn(HERE+"/check_prediction_barrier_v1.py",name)))
    need(ref(CHILD+"/explicit_policy_selector_v1.py")["sha256"]=="2a991ca6e2da3f4ba05888bc44e4cd53025f929abf88760aa4f93571e13bc87b","Pinned selector")
    good("pinned_selector_source_unchanged")
    # Exact new role-building functions on original concrete refs and explicitly synthetic future ROOT refs.
    src=(ROOT/candidate).read_text();tree=ast.parse(src);env={"need":need,"ref":ref}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            key=n.targets[0].id
            if key in {"HERE","BASE_HERE","SOURCE","MANIFEST","REGISTRY","METHODS","PROTOCOL","CONFIG","PRODUCER","ORIGINAL","PIDFD","OWNED","NAMESPACE","V4","SELECTOR"}:
                exec(compile(ast.Module(body=[n],type_ignores=[]),candidate,"exec"),env)
    defs(candidate,{"generation_direct_refs_v1","consumer_direct_refs_v1"},env)
    q={k:fake(k) for k in ("runtime_ref","method_freeze_ref","source_review_ref","policy_selector_ref",
        "registration_ref","registration_decisions_ref","registration_ledger_ref","runtime_qualification_ref",
        "root_request_ref","entry_review_ref","bundle_ref","scorer_review_ref","actual_integration_qualification_ref")}
    current=json.loads((ROOT/"research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json").read_text())["current_guard_roles"]
    current={k:v for k,v in current.items()}
    unique=lambda rows:list({json.dumps(r,sort_keys=True):r for r in rows}.values())
    good("all30_native14v3_current_roles_from_pinned_metadata",len(unique(current.values()))==30)
    # Execute only exact pure _bindings and dedup code objects with inert I/O, no policy/state/guard module.
    childenv=dict(copy=copy,require=need,_ref=ref,_manifest=lambda:None,_raw=lambda r:b"SYNTHETIC_AUTHENTICATION_ONLY",
        SOURCE=CHILD+"/stopped_retention_v1.py",MEASURE_SOURCE=CHILD+"/retained_nodes_v1.py",MANIFEST=CHILD+"/source_manifest.v1.json",
        SELECTOR_SOURCE=CHILD+"/explicit_policy_selector_v1.py",_POLICY_REF=fake("future_child_policy"))
    defs(CHILD+"/stopped_retention_v1.py",{"_bindings","consumer_bindings_v1"},childenv)
    policy=dict(root_activation_ref=fake("future_child_activation"),source_review_ref=fake("future_child_review"),
        permanent_revocation_ref=fake("future_child_parent_revocation"))
    child=childenv["_bindings"](current,policy)
    good("all_current_roles_remain_direct_in_child38",len(unique(child.values()))==38 and all(r in child.values() for r in current.values()))
    models={}
    for policy_name,bindings in (("native14v3",current),("native7_H050_child",child)):
        models[policy_name]={}
        for stage in ("generation","validate","score"):
            additional=env["generation_direct_refs_v1"](q,fake("generation_root_request")) if stage=="generation" else env["consumer_direct_refs_v1"](q,stage)
            union=unique([*bindings.values(),*additional])
            good(policy_name+"_"+stage+"_complete_returned_required_selector_union_under64",len(union)<=64)
            good(policy_name+"_"+stage+"_no_returned_guard_omitted",all(r in union for r in bindings.values()))
            models[policy_name][stage]=dict(returned_guard_count=len(unique(bindings.values())),required_count=len(unique(additional)),
                complete_direct_count=len(union),complete_direct_refs=union,future_refs_explicitly_synthetic=True)
    # Guard selection lifecycle code on a synthetic selector and memo only.
    events=[];sel=fake("selector");chosen=dict(accounting_mode="test",policy_ref=fake("policy"))
    g=types.SimpleNamespace(MODE="test",_POLICY_REF=chosen["policy_ref"],
        budget=types.SimpleNamespace(v4=types.SimpleNamespace(),__name__="test"))
    for name in ("canonical","file_bytes","mutex"):
        f=lambda:None;f.__module__="synthetic_v4";setattr(g.budget.v4,name,f)
    g.budget.v4.__name__="synthetic_v4"
    selector=types.SimpleNamespace(load_selected_guard_v1=lambda r:(events.append("configure") or g))
    memo=types.SimpleNamespace(install_consumer_v1=lambda *a,**k:(events.append("memo") or object()))
    genv=dict(selection=lambda r:copy.deepcopy(chosen),load=lambda r,n:selector if r==env["SELECTOR"] else memo,
        SELECTOR=env["SELECTOR"],NAMESPACE=env["NAMESPACE"],V4=env["V4"],local=lambda p:ROOT/p,need=need,
        _SELECTED_REF=None,_SELECTED_VALUE=None,_SELECTOR_MODULE=None,_GUARD=None)
    defs(candidate,{"guard"},genv)
    good("one_exact_selection_then_one_memo",genv["guard"](sel,True) is g and events==["configure","memo"])
    good("same_configured_module_reused_after_memo_no_reentry",genv["guard"](sel,True) is g and events==["configure","memo"])
    reject("different_selector_ref_after_memo_rejected",lambda:genv["guard"](fake("other_selector"),True))
    old_policy=g._POLICY_REF;g._POLICY_REF=fake("other_policy")
    reject("configured_policy_drift_rejected",lambda:genv["guard"](sel,True));g._POLICY_REF=old_policy

    # Prospective policy/registration checks on in-memory metadata only.
    from datetime import datetime,timezone
    penv=dict(need=need,ref=ref,copy=copy,datetime=datetime,SELECTOR=env["SELECTOR"],PROTOCOL=env["PROTOCOL"],
        REGISTRY=env["REGISTRY"],CONFIG=env["CONFIG"],PRODUCER=env["PRODUCER"],MANIFEST=env["MANIFEST"],
        ROOT=ROOT)
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="APPROVED":
            exec(compile(ast.Module(body=[n],type_ignores=[]),candidate,"exec"),penv)
    approved=penv["APPROVED"]["native14_stopped_retention_v3"]
    chosen_doc=dict(schema="explicit_ROOT_storage_policy_selector_v1",active=True,root_authorizer="root_agent",
        selector_source_ref=env["SELECTOR"],accounting_mode="native14_stopped_retention_v3",
        source_ref=approved["source_ref"],source_manifest_ref=approved["manifest_ref"],policy_ref=approved["policy_ref"],
        measurement_source_ref=fake("measure"),source_review_ref=fake("policy_review"),repair_authority_ref=fake("repair"),
        no_fallback_discovery_upgrade_or_in_run_swap=True,future_ROOT_adapters_only=True,
        scheduling_ceiling_bytes=980000000000,human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000)
    docs={sel["path"]:chosen_doc}
    penv["read"]=lambda r:copy.deepcopy(docs[r["path"]]);penv["raw"]=lambda r:b"SYNTHETIC_AUTHENTICATION_ONLY"
    defs(candidate,{"selection","registration","prospective"},penv)
    good("explicit_pinned_current_selection_metadata_contract",penv["selection"](sel)==chosen_doc)
    for key,value in (("active",False),("accounting_mode","latest"),("source_ref",fake("unapproved_source")),
                      ("policy_ref",fake("wrong_current_policy")),("no_fallback_discovery_upgrade_or_in_run_swap",False),
                      ("human_limit_bytes",2000000000000)):
        old=chosen_doc[key];chosen_doc[key]=value
        reject("selection_"+key+"_drift_blocks",lambda:penv["selection"](sel));chosen_doc[key]=old
    rq=copy.deepcopy(q);rq["created_at_utc"]="2026-10-16T20:00:00+00:00"
    frozen=dict(frozen_at_utc="2026-10-16T18:00:00+00:00")
    reg=dict(schema="European_outer8_prospective_registration_v1",root_authorizer="root_agent",status="registered",
        condition="European_outer8_frozen_predictor_independent_confirmation_v1",hypothesis_id="SYNTHETIC_HYPOTHESIS",
        linked_hypothesis_ids=["H038","H050"],method_freeze_ref=rq["method_freeze_ref"],protocol_ref=ref(env["PROTOCOL"]),
        registry_ref=ref(env["REGISTRY"]),decisions_snapshot_ref=rq["registration_decisions_ref"],
        ledger_snapshot_ref=rq["registration_ledger_ref"],registered_at_utc="2026-10-16T19:00:00+00:00")
    decision={k:reg[k] for k in ("condition","linked_hypothesis_ids","method_freeze_ref","protocol_ref","registry_ref")}
    decision.update(id=reg["hypothesis_id"],outer_tuning_authorized=False,repeat_reason="synthetic independent confirmation only")
    decisions=dict(decisions=[decision])
    docs.update({rq["registration_ref"]["path"]:reg,rq["method_freeze_ref"]["path"]:frozen,rq["registration_decisions_ref"]["path"]:decisions})
    penv["raw"]=lambda r:b"SYNTHETIC_HYPOTHESIS independent registration" if r==rq["registration_ledger_ref"] else b"SYNTHETIC_AUTHENTICATION_ONLY"
    good("actual_registration_shape_requires_decision_and_ledger_snapshots",penv["registration"](rq)==reg)
    for key,value in (("status","inactive_template"),("method_freeze_ref",fake("other_freeze"))):
        old=reg[key];reg[key]=value;reject("registration_"+key+"_drift_blocks",lambda:penv["registration"](rq));reg[key]=old
    decision["outer_tuning_authorized"]=True
    reject("registered_outer_tuning_blocks",lambda:penv["registration"](rq));decision["outer_tuning_authorized"]=False
    decisions["decisions"]=[]
    reject("missing_registered_decision_blocks",lambda:penv["registration"](rq));decisions["decisions"]=[decision]
    reg["registered_at_utc"]="2026-10-16T21:00:00+00:00"
    reject("registration_after_request_blocks",lambda:penv["registration"](rq));reg["registered_at_utc"]="2026-10-16T19:00:00+00:00"

    # No real source configuration, state, mutex, scope, data or model function has executed.
    return dict(schema="European_outer8_current_policy_source_contracts_v1",status="PASS_SOURCE_ONLY",
        check_count=len(checks),checks=checks,direct_role_models=models,
        native_driver_role_union_not_claimed=True,all_future_refs_synthetic_not_authority=True,
        actual_guard_configuration_probe_measurement_accounting_model_dataset_truth_job_or_Git=False)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
