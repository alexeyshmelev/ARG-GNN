"""Metadata-only and static checks for independent later C3 builder review."""
from pathlib import Path
import ast,copy,hashlib,importlib,json,subprocess,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");sys.path.insert(0,str(ROOT))
REVIEW="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1"
BUILDER="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v1"
MAIN="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
OLD="research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1"
c=importlib.import_module(MAIN.replace("/",".")+".acceptance_contract_v1")
b=importlib.import_module(BUILDER.replace("/",".")+".prepare_evaluation_root_v1")
def ref(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def save(name,value):
    p=ROOT/REVIEW/name
    with p.open("x") as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
    return ref(p.relative_to(ROOT))
checks=[]
def check(name,detail):checks.append(dict(name=name,result="PASS_SOURCE_OR_METADATA_ONLY",detail=detail))
audit=json.loads((ROOT/BUILDER/"actual_plan_readiness.v1.json").read_text())
reads=[]
ns=dict(c.__dict__)
def traced(record,cap=33554432):
    if isinstance(record,dict):
        assert Path(record["path"]).suffix==".json"
        reads.append(copy.deepcopy(record))
    return c.read(record,cap=cap)
ns["read"]=traced
for name in ("terminal0_v1","plan_ready_v1"):
    ns[name]=types.FunctionType(getattr(c,name).__code__,ns,name)
actual=[]
for row in audit["parents"]:
    e=row["plan_evidence"];parent=row["parent_id"]
    try:
        plan=ns["plan_ready_v1"](e,parent)
        actual.append(dict(parent_id=parent,gate_passed=True,outer_route="actual historical close/ROOT approval fallback" if e["outer_terminal_ref"] is None else "explicit actual terminal",
            sequence_length_bp=plan["sequence_length_bp"],haplotypes=plan["haplotypes"],focal_records=plan["focal_records"]))
        assert parent in c.VAL[:2]
    except ValueError as ex:
        assert parent==c.VAL[2] and e["close_receipt_ref"] is None
        actual.append(dict(parent_id=parent,gate_passed=False,missing="source-time audit has no actual close receipt",error=type(ex).__name__+": "+str(ex),
            source_time_observation_utc=audit["observed_utc"],no_current_global_close_search_or_rerun=True))
assert len(actual)==3
check("reproduced_actual_VAL000001_closed_metadata_gates_and_declared_VAL002_incomplete_gate",actual)
tree=ast.parse((ROOT/MAIN/"worker_entry_v1.py").read_text())
old=ast.parse((ROOT/OLD/"worker_entry_v1.py").read_text())
numerical=[]
for name in ("predict_v1","matrix_v1","score_v1"):
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    prior=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(fn,include_attributes=False)==ast.dump(prior,include_attributes=False)
    numerical.append(name)
check("original_predict_matrix_score_function_ASTs_unchanged",numerical)
caps=[]
for phase in b.PHASES:
    value=c.caps_v1(phase);assert value==c.read(c.ref(MAIN+"/caps_"+phase+".INACTIVE.v1.json"))
    assert value["reservation_cap_bytes"]==33554432 and value["CPU_threads"]==32 and value["max_cpu_rss_bytes"]==137438953472
    assert value["operational_deadline_seconds"]==86400 and value["dataloader_workers"]==value["max_cuda_reserved_bytes"]==0
    assert c.make_config(dict(phase=phase,packet_directory="SYNTHETIC"),value)["output_cap_bytes"]==33554432
    caps.append(dict(phase=phase,ref=c.ref(MAIN+"/caps_"+phase+".INACTIVE.v1.json")))
check("all7_exact_original_caps_and_config",caps)
assert c.RUNTIME_REF==dict(path="research/reviews/ARGformer_American_C3_implementation_20261015_v1/shared_cpu_runtime_metadata.json",sha256="e0944a1c2c4b2a26d2edfe68fad11db691b4b6853ebeddd8e3b79692b43fa602")
c.read(c.RUNTIME_REF)
check("exact_shared_runtime_ref",c.RUNTIME_REF)
score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="score_v1")
matrix_call=next(n.lineno for n in ast.walk(score) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="validation_matrix_v1")
truth_assignment=next(n.lineno for n in ast.walk(score) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="truth" for t in n.targets))
resolve_def=next(n.lineno for n in ast.walk(score) if isinstance(n,ast.FunctionDef) and n.name=="resolve")
assert matrix_call<truth_assignment<resolve_def
check("unchanged_worker_authenticates30cell_prediction_registry_before_truth_resolver",dict(matrix_call_line=matrix_call,truth_mapping_line=truth_assignment,resolver_line=resolve_def))
cli=[]
module=BUILDER.replace("/",".")+".prepare_evaluation_root_v1"
for name,args,expected in [("default",[],0),("help",["--help"],0),("missing_active",["--prepare-root"],1)]:
    p=subprocess.run([str(ROOT/".runtime/env/bin/python"),"-B","-s","-m",module,*args],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==expected
    if name=="default":
        value=json.loads(p.stdout);assert value["status"]=="SOURCE_ONLY_INACTIVE" and value["admission"] is False and value["evaluation"] is False
    cli.append(dict(name=name,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
check("actual_pinned_inert_CLI_default_help_missing_active",cli)

expected={
BUILDER+"/source_handoff.v1.json":"d68cdb9164a3f51440aca7454aebd180b109d5d28fdc654020cb85b4c8b19a46",
BUILDER+"/prepare_evaluation_root_v1.py":"98bc9bf6f6909c228e5146a19d6ca092ca2220165dabf759c3537ddb348ec824",
MAIN+"/source_manifest.v1.json":"a76e6eb58fef091dd80ca8ba918c4873b1a31b7f28f28b2e6e84104ec910af35",
"research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1/review.v1.json":"b89ce51b5fc08420bb7b275a21817146a54791438c4d8fa74ccdf810b4f1fe51",
"research/reviews/C3_saved600_validator_ROOT_preparation_independent_review_20261016_v1/review.v1.json":"7e0cf3cbbd9cf368831c089aa377eb70d80265be03a3d280316ba43e839507a7"}
def add(path,sha):
    assert path not in expected or expected[path]==sha
    expected[path]=sha
def collect(value):
    if isinstance(value,dict):
        if set(value)=={"path","sha256"}:add(value["path"],value["sha256"])
        for v in value.values():collect(v)
    elif isinstance(value,list):
        for v in value:collect(v)
for path in [BUILDER+"/source_handoff.v1.json",BUILDER+"/source_manifest.v1.json",BUILDER+"/actual_plan_readiness.v1.json",
    MAIN+"/source_handoff.v1.json",MAIN+"/source_manifest.v1.json"]:
    collect(json.loads((ROOT/path).read_text()))
m=json.loads((ROOT/MAIN/"source_manifest.v1.json").read_text())
for path,sha in m["source_sha256"].items():add(path,sha)
for r in [*reads,*[x["ref"] for x in caps]]:add(r["path"],r["sha256"])
for path in [MAIN+"/input.INACTIVE.template.v1.json",MAIN+"/ROOT_validation_request.INACTIVE.template.v1.json",MAIN+"/caps_validate_saved600.INACTIVE.v1.json"]:
    r=ref(path);add(r["path"],r["sha256"])
pins=[]
for path,sha in sorted(expected.items()):
    p=ROOT/path
    assert not Path(path).is_absolute() and ".." not in p.parts and p.resolve()==p and p.is_file() and not p.is_symlink()
    assert p.suffix in (".json",".py",".sh",".md") and p.stat().st_size<=33554432
    assert ref(path)==dict(path=path,sha256=sha),(path,sha,ref(path))
    pins.append(dict(path=path,sha256=sha,bytes=p.stat().st_size))
ast_refs=[]
for path in [BUILDER+"/prepare_evaluation_root_v1.py",MAIN+"/acceptance_contract_v1.py",MAIN+"/worker_entry_v1.py",MAIN+"/root_driver_v1.py"]:
    ast.parse((ROOT/path).read_text());ast_refs.append(ref(path))
check("all_declared_builder_main_readiness_and_followed_JSON_metadata_pins",dict(pin_count=len(pins),parsed_AST_refs=ast_refs))
pinref=save("hash_verification.v1.json",dict(schema="C3_later_eval_builder_independent_pins_v1",status="PASS_SOURCE_ONLY",refs=pins,pin_count=len(pins),parsed_AST_refs=ast_refs,
    actual_scientific_payload_guard_resource_process_or_admission_opened=False))
assert not any(name in sys.modules for name in ("torch","numpy","torch_geometric"))
result=dict(schema="C3_later_eval_builder_independent_metadata_static_v1",status="PASS_SOURCE_AND_METADATA_ONLY",checks=checks,check_count=len(checks),
    hash_evidence_ref=pinref,actual_VAL003004_readiness_not_inspected=True,all5_plan_gate_covered_by_fake_tests=True,
    actual_current_acceptance_or_execution_readiness_not_asserted=True,
    actual_guard_resource_process_probe_accounting_admission_acceptance_or_evaluation=False,scientific_payloads_opened=False)
print(json.dumps(result,sort_keys=True,indent=2))
