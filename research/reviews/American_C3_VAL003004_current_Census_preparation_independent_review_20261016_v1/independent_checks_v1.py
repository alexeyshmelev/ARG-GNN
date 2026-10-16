"""Independent source/pure/fake-API review. No real guard or scientific actions."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib,io,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
OUT="research/reviews/American_C3_VAL003004_current_Census_preparation_independent_review_20261016_v1"
def require(value,message):
    if not value:raise ValueError(message)
def ref(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_bytes())
def dump(name,value):
    with (ROOT/OUT/name).open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
def node(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
def run():
    handoff=read(CAND+"/source_handoff.v1.json");manifest=read(CAND+"/source_manifest.v1.json")
    assert ref(CAND+"/source_handoff.v1.json")["sha256"]=="e7e7edf6f71321fde24637c944f4edf76ec89a6fc5aef3694aa4af6b71cb62fb"
    assert ref(CAND+"/source_manifest.v1.json")["sha256"]=="a216b2a263c8594e20fc149a3dca13b949828da4cb59a9f6b41c9774f893ac34"
    for record in manifest["refs"]:assert ref(record["path"])==record
    for record in handoff["files"]:
        assert ref(record["path"])=={k:record[k] for k in ["path","sha256"]}
    basis=read(CAND+"/scientific_basis.v1.json");ready=read(CAND+"/readiness.v1.json")
    ct=ast.parse((ROOT/CAND/"root_contract_v1.py").read_text())
    dt=ast.parse((ROOT/CAND/"root_driver_v1.py").read_text())
    wt=ast.parse((ROOT/CAND/"worker_entry_v1.py").read_text())
    pt=ast.parse((ROOT/CAND/"plan_worker_v1.py").read_text())
    original_pt=ast.parse((ROOT/basis["original_plan_worker_ref"]["path"]).read_text())
    assert same(node(pt,"execute"),node(original_pt,"execute"))
    prior_dt=ast.parse((ROOT/basis["finite_supervisor_source_ref"]["path"]).read_text())
    actual_try=next(n for n in node(dt,"execute").body if isinstance(n,ast.Try) and n.finalbody)
    prior_try=next(n for n in node(prior_dt,"execute").body if isinstance(n,ast.Try) and n.finalbody)
    assert same(ast.Module(body=actual_try.finalbody,type_ignores=[]),ast.Module(body=prior_try.finalbody,type_ignores=[]))
    # Reproduce all author fake contracts, redirecting its sole result write to
    # our own directory. Candidate bytes/results remain immutable.
    fixture=ROOT/CAND/"fake_contracts_v1.py";tree=ast.parse(fixture.read_text())
    writes=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="write_text"]
    assert len(writes)==1
    writes[0].func.value=ast.Constant(value=str(ROOT/OUT/"author_fake_contracts_reproduced.v1.json"))
    writes[0].func.value=ast.Call(func=ast.Name(id="Path",ctx=ast.Load()),args=[writes[0].func.value],keywords=[])
    ast.fix_missing_locations(tree)
    author_ns=dict(__file__=str(fixture),__name__="_independent_author_fixture_reproduction",print=lambda *a,**k:None)
    exec(compile(tree,"<author-pure-fixture-output-redirect-only>","exec"),author_ns)
    assert author_ns["result"]["passed"]==98
    # Actual default imports and entrypoints remain inert. Never call select
    # through a real guard and never invoke an actual admit/execute/close mode.
    pkg=CAND.replace("/",".")
    mods={name:importlib.import_module(pkg+"."+name) for name in
        ["root_contract_v1","root_driver_v1","worker_entry_v1","plan_worker_v1"]}
    c=mods["root_contract_v1"]
    for name,m in mods.items():assert Path(m.__file__).resolve()==ROOT/CAND/(name+".py")
    for name in ["root_driver_v1","worker_entry_v1","plan_worker_v1"]:assert mods[name].c is c
    stream=io.StringIO()
    import contextlib
    default=[]
    with contextlib.redirect_stdout(stream):
        for name in ["root_driver_v1","worker_entry_v1","plan_worker_v1"]:
            before=len(stream.getvalue());assert mods[name].main([])==0
            default.append(dict(module=name,result=json.loads(stream.getvalue()[before:])))
    assert not any(n in sys.modules for n in ["numpy","torch","torch_geometric","tskit","tsinfer","tsdate","msprime"])
    # Compute all six actual direct_roles code-object unions with source/public
    # metadata only. Future references are fake descriptors, never fake files.
    roleplan=read(CAND+"/direct_role_plan.v1.json");unions=[]
    def fake(role):return dict(path="SYNTHETIC_ONLY/"+role,sha256=hashlib.sha256(role.encode()).hexdigest())
    for parent in c.PARENTS:
        for stage in c.STAGES:
            c.select(parent,stage)
            if stage=="infer":
                row=ready["parents"][parent]
                prior={k:row[v] for k,v in [("completion_ref","simulation_completion_ref"),("wait_ref","simulation_wait_ref"),
                    ("outer_ref","simulation_outer_terminal_ref"),("close_ref","simulation_close_ref"),("observation_ref","simulation_observation_ref")]}
            else:prior={k:fake(parent+"/"+stage+"/"+k) for k in ["completion_ref","wait_ref","outer_ref","close_ref","observation_ref"]}
            q=dict(prior=prior,caps_ref=fake(parent+"/"+stage+"/caps"),source_review_ref=fake("independent_review"))
            rows=c.direct_roles(fake(parent+"/"+stage+"/request"),q,fake(parent+"/"+stage+"/config"),
                fake(parent+"/"+stage+"/closure"),c.RUNTIME_REF)
            union={r["path"]:r for r in [*roleplan["current_Census23_refs"],*rows]}
            assert len(union)==58 and len(union)<=64
            unions.append(dict(parent=parent,stage=stage,union_count=len(union),
                current_Census_base_count=len(roleplan["current_Census23_refs"]),direct_roles=len(rows),ceiling=64))
    # Independent actual predecessor function negatives, under fake metadata.
    cases=[]
    for parent in ["american_c3_val_003","american_c3_val_004"]:
        for stage in ["native_paths","plan"]:
            previous="infer" if stage=="native_paths" else "native_paths"
            folder=ROOT/CAND/("admitted_val"+parent[-3:]+"_"+previous+"_v1")
            output="experiments/results/American_C3_val"+parent[-3:]+"_full30mb_"+previous+"_current_Census_v1"
            def fr(p):
                p=Path(p);p=str(p.relative_to(ROOT)) if p.is_absolute() else str(p)
                return dict(path=p,sha256=hashlib.sha256(p.encode()).hexdigest())
            cp=fr(output+"/completed.json");wr=fr(folder/"worker_aux/actual_wait.json")
            tr=fr(folder/"outer_terminal.root_actual_v1.json");cr=fr(folder/"closed/close_receipt.json")
            obsref=fr(output+"/observed/manifest.json");sourceobs=fake(parent+"/original_sanitized")
            obs=dict(parent_id=parent,split="val",sequence_length_bp=30000000,sample_unit="haploid",
                class_order=["EUR","EAS","AFR"],counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),
                query_ancestry_truth_in_inputs=False,genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",
                sites_and_sample_identities_verified=True,source_observation_ref=sourceobs)
            completed=dict(status={"infer":"completed_current_C3_inference","native_paths":"completed_native_C3_path_preparation"}[previous],
                parent_id=parent,observation_ref=obsref,query_focal_count=10000,query_ancestry_truth_opened=False,native_split_seed=42,
                native_train_fraction=.9,class_order=["EUR","EAS","AFR"],focals_ref=fake(parent+"/10000_focals"))
            wait=dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id=parent,
                phase=previous+"_val_"+parent[-3:],input_ref=fr(folder/"input.json"),all_owned_processes_and_groups_drained=True,operational_failure=None)
            terminal=dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cp,direct_wait_ref=wr)
            closed=dict(key="H041/American-C3-val"+parent[-3:]+"-full30mb-"+previous+"-current-Census-v1",
                value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=cp,direct_wait_ref=wr)))
            records={cp["path"]:completed,wr["path"]:wait,tr["path"]:terminal,cr["path"]:closed,obsref["path"]:obs}
            ns=dict(Path=Path,ROOT=ROOT,REL=CAND,require=require,read=lambda r:copy.deepcopy(records[r["path"]]),
                ref=fr,local=lambda p:Path(p) if Path(p).is_absolute() else ROOT/p,
                existing_simulation=lambda p:dict(simulation_observation_ref=sourceobs))
            for name in ["observation","actual_prior"]:
                actual=types.FunctionType(getattr(c,name).__code__,ns,name);ns[name]=actual
            assert ns["actual_prior"](parent,stage)["completion_ref"]==cp
            mutations=[(wr,"returncode",1),(wr,"actual_direct_worker_wait",False),
                (wr,"all_owned_processes_and_groups_drained",False),(wr,"parent_id","american_c3_val_000"),
                (tr,"actual_terminal_consumed",False),(tr,"actual_exit_code",1),(obsref,"sequence_length_bp",30000),
                (obsref,"sites_and_sample_identities_verified",False),(obsref,"query_ancestry_truth_in_inputs",True)]
            if stage=="plan":mutations.extend([(cp,"query_focal_count",9999),(cp,"query_ancestry_truth_opened",True),(cp,"native_split_seed",43)])
            for record,key,value in mutations:
                original=records[record["path"]][key];records[record["path"]][key]=value
                try:ns["actual_prior"](parent,stage)
                except ValueError:pass
                else:raise AssertionError((parent,stage,key))
                records[record["path"]][key]=original
                cases.append(dict(parent=parent,stage=stage,rejected_field=key))
    # Exact current API compatibility is checked from original source signatures.
    gt=ast.parse((ROOT/c.GUARD_REF["path"]).read_text())
    expected={"configure_with_renewal_v1":"repair_authority_ref, *, policy_ref",
        "operation_draft_v1":"operation, request, *, expected_before, bindings",
        "apply_reserve_v1":"authority_ref, *, expected_source_sha256, expected_launcher_sha256",
        "apply_close_completed_v1":"authority_ref, *, expected_source_sha256, expected_launcher_sha256",
        "_state_locked":"source_sha, launcher_sha, *, fresh=False"}
    for name,args in expected.items():assert ast.unparse(node(gt,name).args)==args
    # Exact source-level capability restrictions and unchanged producer calls.
    worker=(ROOT/CAND/"worker_entry_v1.py").read_text()
    assert "return producer.execute(cfg,common.Writer(s,g))" in worker
    assert "return plan_worker_v1.execute(record)" in worker
    assert c.GPU_PHASES==() and c.STAGES==("infer","native_paths","plan")
    assert "configure_with_renewal_v1(REPAIR_REF,policy_ref=POLICY_REF)" in (ROOT/CAND/"root_contract_v1.py").read_text()
    return dict(schema="American_C3_VAL003004_independent_source_checks_v1",status="PASS_SOURCE_ONLY_CHECKS",
        exact_handoff_manifest_and_source_refs_verified=True,manifest_ref_count=len(manifest["refs"]),
        handoff_file_pin_count=len(handoff["files"]),author_98_fake_contracts_independently_reproduced=True,
        actual_default_imports_and_inert_entrypoints=default,six_current_Census_source_role_unions=unions,
        original_full_plan_execute_AST_equal=True,original_finite_cleanup_finally_AST_equal=True,
        independent_predecessor_rejection_cases=cases,independent_rejection_count=len(cases),
        exact_current_guard_API_signatures_checked=expected,
        original_science_full_extent_classes_seeds_focal_rule_and_caps_preserved=True,
        no_scientific_payload_or_real_guard_accounting_GPU_process_probe_or_job=True,
        actual_capacity_admission_execution_close_and_paper_exact_reproduction_not_established=True)
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit source-only --run required")
    result=run();dump("independent_checks.v1.json",result)
    print(json.dumps(dict(status=result["status"],manifest_refs=result["manifest_ref_count"],
        independent_rejections=result["independent_rejection_count"],direct_union=58)))
