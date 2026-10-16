"""Independent source-only endpoint and role checks. No live probes or accounting."""
from pathlib import Path
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND=ROOT/"research/reviews/native14_stopped_retention_successor_source_20261016_v1"
def require(ok,message):
    if not ok:raise ValueError(message)
def reference(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def tree(path):return ast.parse((ROOT/path).read_text())
def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def compile_only(n,ns):
    exec(compile(ast.Module(body=[n],type_ignores=[]),"<independent-exact-source-function>","exec"),ns)
def run():
    main=tree(str(CAND.relative_to(ROOT)/"native14_stopped_retention_v1.py"))
    absent=function(main,"_native_socket_absent")
    target="/SYNTHETIC_ONLY/charged/socket.sock"
    cases=[]
    for label,bound,expected in [
        ("canonical_absolute",target,False),
        ("relative_bind_with_charged_CWD","socket.sock",False),
        ("absolute_with_redundant_separator","/SYNTHETIC_ONLY/charged//socket.sock",False),
        ("absolute_with_dot_component","/SYNTHETIC_ONLY/charged/./socket.sock",False)]:
        # Linux records the original filesystem bind address string. Fixture
        # supplies a known equivalent target spelling; no kernel file is read.
        ns=dict(require=require,_kernel_endpoints=lambda:[bound.encode()])
        compile_only(absent,ns)
        try:actual=ns["_native_socket_absent"](target)
        except ValueError:actual=False
        cases.append(dict(case=label,requested_absolute_path=target,fake_kernel_bind_address=bound,expected_absent=expected,actual_absent=actual,incorrect_absence_accepted=actual!=expected))
    assert sum(c["incorrect_absence_accepted"] for c in cases)==3
    # Reconstruct actual current23 roles by executing only binding-expression/
    # function ASTs; no _state_locked, policy validation or project import.
    ref=reference
    def known(path):return ref(path)
    bottom=tree("research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py")
    assignment=next(n for n in function(bottom,"_state_locked").body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="bindings")
    cpu=known("src/arggnn/project_storage_cpu_closed_scopes_v1.py")
    terminal=known("src/arggnn/project_storage_terminal_scopes_v1.py")
    success=known("src/arggnn/project_storage_terminal_success_scopes_v1.py")
    retired=known("src/arggnn/project_storage_terminal_retired_scopes_v1.py")
    ns=dict(budget=types.SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py"),
        source_sha=known("scripts/project_budget_v9.py")["sha256"],launcher_sha=known("scripts/launch_reserved_gpu_v6.py")["sha256"],
        old=types.SimpleNamespace(CPU_SOURCE=cpu["path"],CPU_SHA256=cpu["sha256"],SOURCE=terminal["path"]),
        previous=types.SimpleNamespace(SOURCE=retired["path"],PREVIOUS_SHA256=success["sha256"],previous=types.SimpleNamespace(SOURCE=success["path"],OLD_SHA256=terminal["sha256"])),
        PREVIOUS_SHA256=retired["sha256"],_ref=ref,SOURCE="research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py")
    compile_only(assignment,ns)
    bindings=ns["bindings"];chain_counts=[dict(stage="repaired_H039_base",unique_refs=len({json.dumps(r,sort_keys=True) for r in bindings.values()}))]
    stages=[
        ("research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py","research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json",None),
        ("research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py","research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json",None),
        ("research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py","research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json",None),
        ("research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py","research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json","research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/frozen_parent_activation.root_active_v1.json")
    ]
    for source,policy,activation in stages:
        t=tree(source);fn=function(t,"_bindings")
        needed={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        env=dict(copy=copy,_ref=ref,require=require)
        for node in t.body:
            if isinstance(node,ast.Assign) and all(isinstance(x,ast.Name) for x in node.targets) and any(x.id in needed for x in node.targets):
                # These assignments are literal source/ref constants only.
                allowed=(ast.Constant,ast.Dict,ast.Tuple,ast.List,ast.Call,ast.Name,ast.Load,ast.keyword,ast.Assign,ast.Store)
                assert all(isinstance(x,allowed) for x in ast.walk(node)),source
                compile_only(node,env)
        env["_POLICY_REF"]=known(policy)
        env["_REPAIR_AUTHORITY_REF"]=known("research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json")
        compile_only(fn,env)
        p={} if activation is None else dict(root_activation_ref=known(activation))
        bindings=env["_bindings"](bindings,p)
        chain_counts.append(dict(stage=source,unique_refs=len({json.dumps(r,sort_keys=True) for r in bindings.values()})))
    plan=json.loads((CAND/"direct_role_plan.v1.json").read_text())
    normalized=lambda rs:{json.dumps(r,sort_keys=True) for r in rs}
    assert normalized(bindings.values())==normalized(plan["current_Census23_refs"])
    assert len(normalized(bindings.values()))==23
    manifest=json.loads((CAND/"source_manifest.v1.json").read_text())
    baseline=normalized(list(bindings.values())+manifest["original_reservation"]["consumer_bindings"])
    assert len(baseline)==55 and baseline==normalized(plan["baseline_unique_refs"])
    concrete=baseline|normalized(plan["authored_direct_refs"])
    assert len(concrete)==58 and concrete==normalized(plan["concrete_union_refs"])
    # Full new source _bindings with six explicit future fake authority roles.
    fake=lambda n:dict(path="SYNTHETIC_ONLY/"+n+".json",sha256=hashlib.sha256(n.encode()).hexdigest())
    env=dict(copy=copy,require=require,_native_manifest=lambda:manifest,_ref=ref,_raw=lambda r:None,
        SOURCE=str(CAND.relative_to(ROOT)/"native14_stopped_retention_v1.py"),MANIFEST=str(CAND.relative_to(ROOT)/"source_manifest.v1.json"),
        MEASURE_SOURCE=str(CAND.relative_to(ROOT)/"retained_nodes_v1.py"),_POLICY_REF=fake("policy"))
    compile_only(function(main,"consumer_bindings_v1"),env);compile_only(function(main,"_bindings"),env)
    fresh=dict(root_activation_ref=fake("activation"),permanent_revocation_ref=fake("revocation"),source_review_ref=fake("review"))
    full=env["_bindings"](bindings,fresh)
    state=env["consumer_bindings_v1"](list(full.values()))
    invocation=env["consumer_bindings_v1"](state,[fake("observation"),fake("operation")])
    assert len(state)==62 and len(invocation)==64
    try:env["consumer_bindings_v1"](invocation,[fake("extra")])
    except ValueError:extra_rejected=True
    else:raise AssertionError("65th direct role accepted")
    return dict(schema="independent_native14_endpoint_and_role_source_checks_v1",endpoint_equivalent_address_cases=cases,
        endpoint_guard_fails_relative_and_noncanonical_bind_addresses=True,actual_current_role_chain_counts=chain_counts,
        independent_current23_role_plan_exact=True,baseline55_exact=True,concrete58_exact=True,state62_exact=True,
        invocation64_exact=True,extra_role65_rejected=extra_rejected,live_process_endpoint_or_accounting_probe=False,source_only=True)
if __name__=="__main__":print(json.dumps(run(),sort_keys=True,indent=2))
