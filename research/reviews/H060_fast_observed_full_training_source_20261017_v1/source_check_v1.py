"""Bounded stdlib AST/source/metadata checks only; no project module import."""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H060_fast_observed_full_training_source_20261017_v1"
OLD=ROOT/"research/reviews/H060_isolated_hold_compatibility_source_20261017_v1"
TX=ROOT/"research/reviews/H060_H061_transaction_admission_source_20261017_v1"


def need(value,message):
    if not value:raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def funcs(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


def same(a,b):
    return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def run():
    checks=[]
    for path in HERE.glob("*.py"):ast.parse(path.read_text(),filename=str(path))
    b=read(HERE/"scientific_binding.v1.json")
    bindings=read(HERE/"source_bindings.v1.json")
    need(set(bindings["consumers"])=={"H060"},"Only H060 fresh consumer")
    spec=bindings["consumers"]["H060"]
    for key in ("contract_ref","driver_ref","worker_ref","source_manifest_ref","source_handoff_ref"):
        need(ref(spec[key]["path"])==spec[key],"Exact immutable existing CPU component: "+key)
    prior=read(ROOT/bindings["actual_admitted_predecessor_input_ref"]["path"])
    need(ref(bindings["actual_admitted_predecessor_input_ref"]["path"])==bindings["actual_admitted_predecessor_input_ref"],
        "Actual predecessor metadata reference")
    need(prior["root_driver_ref"]==ref(str((TX/"root_driver_v1.py").relative_to(ROOT)))
        and prior["main_source_ref"]==ref(str((TX/"worker_entry_v1.py").relative_to(ROOT))),
        "Authenticate actual current transaction/worker join")
    checks.append("actual_admitted_current_binding_and_immutable_CPU_component_sources")

    old=funcs(OLD/"warmstart_train_v2.py");new=funcs(HERE/"warmstart_train_v2.py")
    changed={"run_train_v1","_signature_v1"}
    need(set(new)==set(old)|{"_fast_preparation_refs_v1"},"No new training/target/math function")
    for name in set(old)-changed:need(same(old[name],new[name]),"Original unchanged runner function "+name)
    old_module=ast.parse((OLD/"warmstart_train_v2.py").read_text())
    new_module=ast.parse((HERE/"warmstart_train_v2.py").read_text())
    for name in ("DEFAULTS","TRAIN","VAL","CHECKPOINT_EPOCHS","SCHEMA","ACCEPTANCE","ENGINE","ENGINE_REVIEW"):
        def assignments(tree):
            return [n for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
        need(all(same(a,c)for a,c in zip(assignments(old_module),assignments(new_module))),
            "Original constant/imported semantic identity "+name)
    normalized=copy.deepcopy(new["run_train_v1"])
    index=next(i for i,n in enumerate(normalized.body)if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=="fast_source_refs"for t in n.targets))
    del normalized.body[index]
    ot=next(n for n in old["run_train_v1"].body if isinstance(n,ast.Try))
    nt=next(n for n in normalized.body if isinstance(n,ast.Try))
    # Only original observed startup statements1..3 are replaced. Everything
    # from all20 cache/barrier checks onward, including targets and1200 updates,
    # must remain the original AST.
    nt.body[1:4]=copy.deepcopy(ot.body[1:4])
    need(same(old["run_train_v1"],normalized),"Entire runner AST except disclosed observed startup")
    signature=copy.deepcopy(new["_signature_v1"])
    extensions=[i for i,n in enumerate(signature.body)if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)
        and isinstance(n.value.func.value,ast.Name)and n.value.func.value.id=="source_paths"
        and n.value.func.attr=="extend"and len(n.value.args)==1
        and isinstance(n.value.args[0],ast.Name)and n.value.args[0].id=="REQUIRED_FAST_SOURCE_PATHS"]
    need(len(extensions)==1,"One explicit signature extension")
    del signature.body[extensions[0]]
    for node in ast.walk(signature):
        if isinstance(node,ast.Name)and node.id=="LEGACY_OBSERVATION_HERE":node.id="HERE"
    need(same(old["_signature_v1"],signature),"Original signature plus four refs and explicit unchanged legacy directory")
    checks.append("unchanged_models_warmstart_targets_all20_all5VAL_cache_updates_and60barrier_AST")

    pins=read(HERE/"scientific_source_pins.v2.json")["refs"]
    pinmap={row["path"]:row["sha256"]for row in pins}
    need(len(pinmap)==len(pins),"No scientific pin path ambiguity")
    required=b["fast_preparation_source_refs"]
    need(len(required)==4 and all(pinmap.get(row["path"])==row["sha256"]and ref(row["path"])==row for row in required),
        "All four exact fast source refs selected from fresh full closure")
    need(pinmap.get(b["runner_source_ref"]["path"])==b["runner_source_ref"]["sha256"]
        and ref(b["runner_source_ref"]["path"])==b["runner_source_ref"],"Actual new runner SOURCE pin")
    for row in pins:
        need(Path(row["path"]).suffix in(".py",".json"),
            "Scientific pins contain only original source/protocol/topology metadata")
        if Path(row["path"]).suffix==".json":
            need(row["path"]in{
                "research/reviews/American_TreeLoom_warmstart_training_source_20261017_v2/protocol.inactive.v2.json",
                "research/reviews/H060_inferred_layout_actual_diagnostic_ROOT_20261017_v1/all20_TRAIN_topology_inventory.root_v1.json",
                "research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/compatibility_protocol.v1.json"},
                "Only exact inherited compact protocol/inventory metadata JSON pins")
        need(ref(row["path"])["sha256"]==row["sha256"],"Scientific source remains exact: "+row["path"])
    checks.append("complete_scientific_source_pins_and_all_four_fast_refs")

    component=read(ROOT/b["fast_adapter_review_ref"]["path"])
    fixture=read(ROOT/b["fast_adapter_ROOT_fixture_ref"]["path"])
    need(ref(b["fast_adapter_review_ref"]["path"])==b["fast_adapter_review_ref"]
        and ref(b["fast_adapter_ROOT_fixture_ref"]["path"])==b["fast_adapter_ROOT_fixture_ref"]
        and component["status"]=="SOURCE_PASS"and component["blockers"]==[]
        and fixture["actual_returncode"]==0
        and fixture["source_manifest_ref"]==b["fast_adapter_manifest_ref"],
        "Genuine component evidence only")
    checks.append("genuine_adapter_SOURCE_PASS_and_ROOT_fixture0_not_production_qualification")

    base=funcs(TX/"root_driver_v1.py");current=funcs(HERE/"root_driver_v1.py")
    for name in ("driver","_make_root_mutation_memo","_memo_admission_node","_install_memo_admit",
                 "execute_or_admit","consumer_for_admitted","worker","append_selected_source"):
        need(same(base[name],current[name]),"Reviewed transaction/lifecycle function unchanged: "+name)
    need((HERE/"worker_entry_v1.py").read_bytes()==(TX/"worker_entry_v1.py").read_bytes(),
        "Common worker entry byte-identical")
    profiles=read(HERE/"storage_profiles.v1.json")
    need(set(profiles)=={"Relate_failed64GiB_retention_v1"},"No old-policy fallback profile")
    need(ref(bindings["actual_selector_metadata_ref"]["path"])==bindings["actual_selector_metadata_ref"],
        "Exact current Relate selector capsule")
    checks.append("unchanged_memo_role_preflight_owned_driver_and_actual_worker_gate")

    caps=read(HERE/"caps.inactive.v1.json")
    need(caps==spec["caps"]==read(ROOT/spec["base_directory"]/"caps.inactive.v1.json"),
        "Every inherited CPU/cache/RSS/output/time cap unchanged")
    need((HERE/"caps.inactive.v1.json").read_bytes()==(ROOT/spec["base_directory"]/"caps.inactive.v1.json").read_bytes(),
        "Cap source byte identity")
    checks.append("all_inherited_caps_byte_identical")

    fixture=read(HERE/"role_union_fixture.v1.json")
    guards=list(fixture["actual_guard_role_refs_by_name"].values())
    def unique(rows):
        out=[]
        for row in rows:
            if row not in out:out.append(row)
        return out
    need(len(guards)==46 and len(unique(guards))==39,"Every46named/39distinct guard role retained")
    need(all(row in prior["storage"]["reservation"]["consumer_bindings"]for row in guards),
        "Guard metadata derives from actual admitted union, no omitted role")
    # Execute ONLY the two original pure metadata role builders, not a contract,
    # selector, driver, guard, journal, mutex, resource or admission function.
    runtime=ROOT/spec["base_directory"]
    role_node=funcs(runtime/"root_contract_v1.py")["direct_roles"]
    deps=read(runtime/"dependencies.v1.json")
    prefix="research/reviews/H060_isolated_CPU_ROOT_preparation_fast_observed_FIXTURE/"
    virtual=lambda label:dict(path=prefix+label+".json",sha256=hashlib.sha256(("symbolic-only:"+label).encode()).hexdigest())
    request=virtual("request")
    q={key:virtual(key)for key in ("source_review_ref","scientific_review_ref","runtime_ref",
        "runtime_qualification_ref","root_directive_ref","registration_ref","registration_decisions_ref","training_context_ref")}
    q["storage_policy_selector_ref"]=bindings["actual_selector_metadata_ref"]
    ns=dict(binding=lambda:b,ref=virtual,MANIFEST="manifest",SOURCE="contract_driver",DRIVER="contract_driver",
        ENTRY="worker",HANDOFF="handoff",DEPS=deps,NAMESPACE=deps["NAMESPACE"],PIDFD=deps["PIDFD"],
        OWNED=deps["OWNED"],SUPERVISOR=deps["SUPERVISOR"])
    exec(compile(ast.Module(body=[role_node],type_ignores=[]),"<isolated-direct-role-metadata>","exec"),ns)
    roles=ns["direct_roles"](request,q,virtual("config"),virtual("closure"))
    ap=dict(need=need,copy=copy)
    exec(compile(ast.Module(body=[current["append_selected_source"]],type_ignores=[]),
        "<isolated-selected-source-addition>","exec"),ap)
    child=bindings["selected_child_source_ref"]
    roles=ap["append_selected_source"](roles,child,child)
    union=unique(guards+roles)
    need(len(union)==61 and len(union)<=64 and all(row in union for row in guards),
        "Full39guard+fresh consumer union61<=64, no four-ref direct-role inflation")
    checks.append("all46named_guard_roles_and_full_pure_metadata_union61_below64")

    for name in ("ROOT_request.INACTIVE.template.v1.json","ROOT_directive.INACTIVE.template.v1.json"):
        value=read(HERE/name)
        need(value["active"]is False and value["root_authorizer"]is None
            and value["created_at_utc"]is None and value["expires_at_utc"]is None,
            "No fabricated active ROOT authority")
    need(read(HERE/"runtime_qualification.INACTIVE.template.v1.json")["status"]=="inactive",
        "No fabricated runtime qualification")
    need(read(HERE/"registration.INACTIVE.template.v1.json")["status"]=="INACTIVE_NOT_REGISTERED",
        "No actual registration")
    need(read(HERE/"ROOT_scientific_review.INACTIVE.template.v1.json")["status"]=="UNREVIEWED_INACTIVE",
        "No fabricated new scientific review")
    checks.append("ROOT_templates_inactive_and_fresh_registration_review_qualification_required")

    for name in ("torch","numpy","torch_geometric","tskit"):need(name not in sys.modules,"No scientific imports")
    checks.append("no_scientific_runtime_guard_resource_or_admission_execution")
    return dict(schema="H060_fast_observed_source_only_static_checks_v1",status="STATIC_SOURCE_PASS_ONLY",
        checks=checks,checks_count=len(checks),pure_metadata_role_union=61,role_ceiling=64,
        numerical_or_model_or_production_or_truth_execution=False,actual_guard_or_resource_state_observed=False,
        runtime_qualification_registration_admission_or_execution=False,Git_or_installs=False)


if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
