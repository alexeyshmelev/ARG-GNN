"""Independent narrow VALv2 source/policy/namespace review. Default inert only."""
from pathlib import Path
import ast,copy,hashlib,importlib,json,os,subprocess,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND="research/reviews/American_C3_VAL003004_native14v3_preparation_source_20261016_v2"
PRIOR="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
OUT="research/reviews/American_C3_VAL003004_native14v3_preparation_independent_review_20261016_v2"
def require(ok,msg):
    if not ok:raise ValueError(msg)
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_bytes())
def dump(name,v):
    with (ROOT/OUT/name).open("x") as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
def node(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def astsame(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def run():
    h=read(CAND+"/source_handoff.v1.json");m=read(CAND+"/source_manifest.v1.json")
    assert ref(CAND+"/source_handoff.v1.json")["sha256"]=="0571347d9bdbc7396d5a712f68f5077c825d22dfe849cc5862623e3cf758cbad"
    assert ref(CAND+"/source_manifest.v1.json")==h["source_manifest_ref"]
    previous_review="research/reviews/American_C3_VAL003004_current_Census_preparation_independent_review_20261016_v1/review.v1.json"
    assert ref(previous_review)["sha256"]=="70e64ea2859c16c931d69e9eeb83483385505498726c860909c0f83cde1715c8"
    assert read(previous_review)["verdict"]=="SOURCE_PASS"
    for r in m["refs"]:assert ref(r["path"])==r
    for r in h["files"]:assert ref(r["path"])=={k:r[k] for k in ["path","sha256"]}
    for name in ["readiness.v1.json","scientific_basis.v1.json","caps.template.inactive.v1.json",
        "infer.scientific_config.template.inactive.v1.json","native_paths.scientific_config.template.inactive.v1.json",
        "plan.scientific_config.template.inactive.v1.json"]:
        assert (ROOT/PRIOR/name).read_bytes()==(ROOT/CAND/name).read_bytes()
    olddt=ast.parse((ROOT/PRIOR/"root_driver_v1.py").read_text());dt=ast.parse((ROOT/CAND/"root_driver_v1.py").read_text())
    oldpt=ast.parse((ROOT/PRIOR/"plan_worker_v1.py").read_text());pt=ast.parse((ROOT/CAND/"plan_worker_v1.py").read_text())
    assert astsame(node(olddt,"execute"),node(dt,"execute"))
    assert astsame(node(oldpt,"execute"),node(pt,"execute"))
    ct=ast.parse((ROOT/CAND/"root_contract_v1.py").read_text())
    oldct=ast.parse((ROOT/PRIOR/"root_contract_v1.py").read_text())
    for name in ["observation","existing_simulation","scientific_config","worker_caps","make_config","hooks"]:
        assert astsame(node(oldct,name),node(ct,name)),name
    # Reproduce139 pure/fake checks, redirecting BOTH author output writes.
    fixture=ROOT/CAND/"fake_contracts_v2.py";ft=ast.parse(fixture.read_text())
    writes=[n for n in ast.walk(ft) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="write_text"]
    assert len(writes)==2
    output_names=[]
    for n in writes:
        assert isinstance(n.func.value,ast.BinOp) and isinstance(n.func.value.right,ast.Constant)
        name=n.func.value.right.value
        assert name in ["direct_role_plan.v2.json","fake_contract_results.v2.json"]
        target=ROOT/OUT/("author_reproduced_"+name);output_names.append(str(target))
        n.func.value=ast.Call(func=ast.Name(id="Path",ctx=ast.Load()),args=[ast.Constant(value=str(target))],keywords=[])
    ast.fix_missing_locations(ft)
    fixture_ns=dict(__file__=str(fixture),__name__="_independent_VALv2_fake_reproduction",print=lambda *a,**k:None)
    exec(compile(ft,"<sealed-author-fixture-only-own-output-redirection>","exec"),fixture_ns)
    assert fixture_ns["result"]["passed"]==139
    # Actual imports and fresh CLI processes use the pinned interpreter but no
    # bind/admit/execute/close arguments. No real policy/guard is configured.
    pkg=CAND.replace("/",".");mods={name:importlib.import_module(pkg+"."+name) for name in
        ["root_contract_v1","root_driver_v1","worker_entry_v1","plan_worker_v1"]}
    c=mods["root_contract_v1"]
    for name,mod in mods.items():
        assert Path(mod.__file__).resolve()==ROOT/CAND/(name+".py")
        if name!="root_contract_v1":assert mod.c is c
    assert c.POLICY_REF is None and c.GUARD_REF==h["prospective_native14v3_guard_ref"]
    assert c.SLOT_REL==PRIOR and c.GPU_PHASES==()
    forbidden=["torch","torch_geometric","numpy","tskit","tsinfer","tsdate","msprime"]
    assert not any(x in sys.modules for x in forbidden)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]))
    cli=[]
    for name in ["root_driver_v1","worker_entry_v1","plan_worker_v1"]:
        result=subprocess.run([str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",pkg+"."+name],
            cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,(name,result.stdout,result.stderr)
        value=json.loads(result.stdout);assert not result.stderr.strip()
        cli.append(dict(module=name,returncode=0,value=value))
    assert c.POLICY_REF is None
    # Actual CLI parser routes explicit --policy-ref into c.bind, under fake c
    # only. No real binder or guard function is called.
    calls=[]
    fc=types.SimpleNamespace(PARENTS=c.PARENTS,STAGES=c.STAGES,
        bind=lambda *args:(calls.append(args) or dict(fake=True)))
    main_ns=dict(mods["root_driver_v1"].main.__globals__,c=fc,print=lambda *a,**k:None)
    main=types.FunctionType(mods["root_driver_v1"].main.__code__,main_ns)
    policy_ref=dict(path=c.POLICY_PATH,sha256="SYNTHETIC_ONLY_NOT_ACTUAL_POLICY_HASH")
    review_ref=dict(path="SYNTHETIC_ONLY/VALv2_review",sha256="0"*64)
    args=["--bind","--parent",c.PARENTS[0],"--stage","infer","--review-ref",json.dumps(review_ref),"--policy-ref",json.dumps(policy_ref)]
    assert main(args)==0 and calls[-1]==(c.PARENTS[0],"infer",review_ref,policy_ref)
    # Independent policy binder negatives using the actual code object and fake
    # metadata. All descriptors explicitly synthetic, never actual policy refs.
    def fake_policy_fixture():
        p_ref=dict(path=c.POLICY_PATH,sha256="SYNTHETIC_ONLY_POLICY")
        act_ref=dict(path="SYNTHETIC_ONLY/activation",sha256="1"*64)
        rev_ref=dict(path="SYNTHETIC_ONLY/revocation",sha256="2"*64)
        sr_ref=dict(path="SYNTHETIC_ONLY/native14_review",sha256="3"*64)
        policy=dict(active=True,schema="native14_exact_stopped_retention_policy_v1",root_authorizer="root_agent",
            source_ref=c.GUARD_REF,source_manifest_ref=c.NATIVE_MANIFEST_REF,measurement_source_ref=c.NATIVE_HELPER_REF,
            journal=".runtime/storage/native14_stopped_retention_v3",root_activation_ref=act_ref,
            permanent_revocation_ref=rev_ref,source_review_ref=sr_ref,base_pins=dict(synthetic=True),base_projected_bytes=1)
        activation=dict(active=True,root_authorizer="root_agent",source_ref=c.GUARD_REF,source_manifest_ref=c.NATIVE_MANIFEST_REF,
            measurement_source_ref=c.NATIVE_HELPER_REF,new_journal=policy["journal"],permanent_revocation_ref=rev_ref,
            source_review_ref=sr_ref,frozen_parent_pins=policy["base_pins"],frozen_parent_projected_bytes=1)
        review=dict(verdict="SOURCE_PASS",source_ref=c.GUARD_REF,source_manifest_ref=c.NATIVE_MANIFEST_REF,measurement_source_ref=c.NATIVE_HELPER_REF)
        revocation=dict(active=True)
        records={p_ref["path"]:policy,act_ref["path"]:activation,sr_ref["path"]:review,rev_ref["path"]:revocation}
        ns=dict(c.bind_policy.__globals__,POLICY_REF=None,raw=lambda r:None,read=lambda r:copy.deepcopy(records[r["path"]]))
        fn=types.FunctionType(c.bind_policy.__code__,ns)
        return p_ref,policy,activation,review,revocation,ns,fn
    negatives=[]
    mutations=[("policy","active",False),("policy","schema","WRONG"),("policy","root_authorizer","WRONG"),
        ("policy","source_ref",dict(path="WRONG",sha256="0"*64)),("policy","source_manifest_ref",{}),
        ("policy","measurement_source_ref",{}),("policy","journal","WRONG"),
        ("activation","active",False),("activation","new_journal","WRONG"),("activation","source_review_ref",{}),
        ("activation","frozen_parent_pins",{}),("activation","frozen_parent_projected_bytes",2),
        ("review","verdict","REQUEST_CHANGES"),("review","source_ref",{}),("review","source_manifest_ref",{}),
        ("review","measurement_source_ref",{}),("revocation","active",False)]
    for group,key,value in mutations:
        p_ref,p,a,r,v,ns,fn=fake_policy_fixture();targets=dict(policy=p,activation=a,review=r,revocation=v)
        targets[group][key]=value
        try:fn(p_ref)
        except ValueError:pass
        else:raise AssertionError((group,key))
        negatives.append(dict(group=group,field=key))
    p_ref,p,a,r,v,ns,fn=fake_policy_fixture();fn(p_ref)
    try:fn(dict(p_ref,sha256="SYNTHETIC_ONLY_DIFFERENT"))
    except ValueError:negatives.append(dict(group="immutable_policy",field="sha256_swap"))
    else:raise AssertionError("policy swap accepted")
    p_ref,p,a,r,v,ns,fn=fake_policy_fixture()
    try:fn(dict(p_ref,path="SYNTHETIC_ONLY/wrong_policy_path"))
    except ValueError:negatives.append(dict(group="policy_ref",field="path"))
    else:raise AssertionError("wrong policy path accepted")
    # Actual original_guard routing code uses exact v3 source/MODE and unchanged
    # API, with pure fake endpoints and no actual configure invocation.
    p_ref,p,a,r,v,ns,fn=fake_policy_fixture();fn(p_ref);configured=[];loads=[]
    def primitive():pass
    primitive.__module__="SYNTHETIC_v4"
    fake_guard=types.SimpleNamespace(MODE="native14_stopped_retention_v3",
        budget=types.SimpleNamespace(v4=types.SimpleNamespace(__name__="SYNTHETIC_v4",
            canonical=primitive,file_bytes=primitive,mutex=primitive)),
        configure_with_renewal_v1=lambda repair,**kw:configured.append((repair,kw)))
    ns.update(bind_policy=fn,raw=lambda r:None,sys=types.SimpleNamespace(path=[]),
        load=lambda record,name:(loads.append((record,name)) or fake_guard))
    actual_guard=types.FunctionType(c.original_guard.__code__,ns)
    assert actual_guard() is fake_guard and loads[0][0]==c.GUARD_REF
    assert configured==[(c.REPAIR_REF,dict(policy_ref=p_ref))]
    # Independently recompute30 base + stage unions from actual direct_roles
    # with synthetic future descriptors, not from author's claimed64 fields.
    base23=m["current_Census23_refs"]
    def fake(name):return dict(path="SYNTHETIC_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
    fake_p=fake("native14_policy")
    base30={r["path"]:r for r in [*base23,c.GUARD_REF,c.NATIVE_MANIFEST_REF,c.NATIVE_HELPER_REF,
        fake_p,fake("native14_activation"),fake("native14_revocation"),fake("native14_review")]}
    assert len(base30)==30
    roles=[];consolidated=c.BASE_CONFIG_REF
    assert consolidated in m["refs"]
    for parent in c.PARENTS:
        for stage in c.STAGES:
            c.select(parent,stage)
            if stage=="infer":
                row=read(CAND+"/readiness.v1.json")["parents"][parent]
                prior={k:row[v] for k,v in [("completion_ref","simulation_completion_ref"),("wait_ref","simulation_wait_ref"),
                    ("outer_ref","simulation_outer_terminal_ref"),("close_ref","simulation_close_ref"),("observation_ref","simulation_observation_ref")]}
            else:prior={k:fake(parent+"/"+stage+"/"+k) for k in ["completion_ref","wait_ref","outer_ref","close_ref","observation_ref"]}
            q=dict(prior=prior,caps_ref=fake(parent+"/"+stage+"/caps"),source_review_ref=fake("VALv2_review"))
            direct=types.FunctionType(c.direct_roles.__code__,dict(c.direct_roles.__globals__,POLICY_REF=fake_p))
            rows=direct(fake(parent+"/"+stage+"/request"),q,fake(parent+"/"+stage+"/config"),fake(parent+"/"+stage+"/closure"),c.RUNTIME_REF)
            union={r["path"]:r for r in [*base30.values(),*rows]}
            assert len(union)==64 and consolidated not in rows
            assert all(x in rows for x in [c.GUARD_REF,fake_p,c.RUNTIME_REF,c.ref(c.SLOT_REL+"/cpu_slot_0.lock"),c.ref(c.SLOT_REL+"/cpu_slot_1.lock")])
            assert c.SOURCE in {r["path"] for r in rows} and c.DRIVER in {r["path"] for r in rows}
            roles.append(dict(parent=parent,stage=stage,base_count=30,direct_rows=len(rows),union_count=len(union)))
    # Consolidation does not weaken authentication: all manifest refs are raw
    # checked on source_paths, then validate_request and actual input source map.
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="raw"
        for n in ast.walk(node(ct,"source_paths")))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="source_paths"
        for n in ast.walk(node(ct,"validate_request")))
    assert "for path,sha in s[\"source_sha256\"].items():raw(dict(path=path,sha256=sha))" in (ROOT/CAND/"root_contract_v1.py").read_text()
    assert not any(x in sys.modules for x in forbidden)
    gt=ast.parse((ROOT/c.GUARD_REF["path"]).read_text())
    apis={name:ast.unparse(node(gt,name).args) for name in ["configure_with_renewal_v1","_state_locked","operation_draft_v1",
        "apply_reserve_v1","apply_close_completed_v1","budget_check_v9","validate_storage_v9"]}
    assert apis["configure_with_renewal_v1"]=="repair_authority_ref, *, policy_ref"
    assert apis["_state_locked"]=="source_sha, launcher_sha, *, fresh=False"
    return dict(schema="American_C3_VALv2_independent_delta_checks_v2",status="PASS_SOURCE_ONLY_CHECKS",
        exact_candidate_and301_ref_pins_verified=True,manifest_refs=len(m["refs"]),handoff_file_pins=len(h["files"]),
        inherited_v1_review_ref=ref(previous_review),author139_fake_checks_reproduced=True,
        entire_supervisor_execute_AST_identical=True,full_plan_execute_AST_identical=True,
        original_science_caps_readiness_configs_byte_identical=True,
        actual_import_modules_route_to_v2=True,real_fresh_pinned_inert_CLI=cli,
        actual_CLI_policy_ref_forwarded_under_fake_bind=True,actual_original_guard_routing_under_fake_endpoints=True,
        policy_negative_cases=negatives,independent_policy_rejections=len(negatives),
        six_independent_direct_unions=roles,full_transitive_baseline_hash_authentication_preserved=True,
        shared_original_v1_slot_paths=True,current_native14v3_API_signatures=apis,
        no_real_policy_guard_accounting_scientific_payload_GPU_probe_or_job_called=True,
        only_real_child_processes_three_default_inert_CLI_smokes=True,
        actual_activation_capacity_admission_execution_not_established=True)
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit source-only --run required")
    value=run();dump("independent_delta_checks.v2.json",value)
    print(json.dumps(dict(status=value["status"],refs=value["manifest_refs"],policy_rejections=value["independent_policy_rejections"])))
