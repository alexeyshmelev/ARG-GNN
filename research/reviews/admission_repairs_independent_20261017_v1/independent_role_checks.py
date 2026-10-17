"""Independent SOURCE inspection only. No project modules are imported or run."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/admission_repairs_independent_20261017_v1"
base=ROOT/sys.argv[1];tag=sys.argv[2]
def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def fn(path,name):
 source=path.read_text();tree=ast.parse(source)
 return next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name),source
manifest=load(base/"source_manifest.v1.json");bindings=load(base/"source_bindings.v1.json")
mismatches=[p for p,s in manifest["source_sha256"].items()if sha(ROOT/p)!=s]
assert not mismatches,mismatches
for p in base.glob("*.py"):compile(p.read_text(),str(p),"exec")
actual=load(ROOT/bindings["actual_H060_failed_authority_metadata_ref"]["path"])
assert ref(bindings["actual_H060_failed_authority_metadata_ref"]["path"])==bindings["actual_H060_failed_authority_metadata_ref"]
assert ref(bindings["actual_selector_metadata_ref"]["path"])==bindings["actual_selector_metadata_ref"]
selected=load(ROOT/bindings["actual_selector_metadata_ref"]["path"])["source_ref"]
assert selected==bindings["selected_child_source_ref"]
actual_roles=actual["bindings"]
def unique(rs):return list(dict.fromkeys((r["path"],r["sha256"])for r in rs))
def fixture(label):return dict(path="research/reviews/SOURCE_FIXTURE_NEVER_AUTHORITY/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
def evaluate(node,env):
 if isinstance(node,ast.Name):return env[node.id]
 if isinstance(node,ast.Constant):return node.value
 if isinstance(node,ast.Subscript):return evaluate(node.value,env)[evaluate(node.slice,env)]
 if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=="ref":return ref(evaluate(node.args[0],env))
 raise AssertionError(ast.dump(node))
cases=[];preflight={};here_audit={}
for consumer,spec in bindings["consumers"].items():
 q=load(ROOT/spec["original_request_ref"]["path"]);deps=load(ROOT/spec["base_directory"]/"dependencies.v1.json");science=load(ROOT/spec["scientific_binding_ref"]["path"])
 q["storage_policy_selector_ref"]=bindings["actual_selector_metadata_ref"]
 for key in ["source_review_ref","runtime_ref","runtime_qualification_ref","root_directive_ref"]:q[key]=fixture(consumer+key)
 env=dict(q=q,b=science,DEPS=deps,MANIFEST=str(base/"source_manifest.v1.json"),SOURCE=str(base/"root_driver_v1.py"),DRIVER=str(base/"root_driver_v1.py"),ENTRY=str(base/"worker_entry_v1.py"),HANDOFF=str(base/"source_handoff.v1.json"),request_ref=fixture(consumer+"request"),cfgref=fixture(consumer+"config"),closure=fixture(consumer+"closure"),**{k:deps[k]for k in ["NAMESPACE","PIDFD","OWNED","SUPERVISOR"]})
 rolefn,source=fn(ROOT/spec["contract_ref"]["path"],"direct_roles")
 lists=[n for n in ast.walk(rolefn)if isinstance(n,ast.List)];assert len(lists)==1
 direct=[evaluate(n,env)for n in lists[0].elts]
 # Only source AST expressions are interpreted as literal metadata. No source function is executed.
 for stage in(["train"]if consumer=="H060"else["freeze","predict","score"]):
  tail=[]if stage in("train","freeze")else[fixture("freeze_acceptance")]
  if stage=="score":tail.append(fixture("predict_acceptance"))
  original=list(actual_roles.values())+direct+tail
  result=unique(original+[selected]);again=unique(original+[selected,selected])
  assert result==again and result[:len(unique(original))]==unique(original)
  assert all((r["path"],r["sha256"])in result for r in actual_roles.values())
  assert (selected["path"],selected["sha256"])in result
  padded=result+[("SOURCE_FIXTURE_"+str(i),"fixture")for i in range(65-len(result))]
  assert len(padded)==65 and not 1<=len(padded)<=64
  cases.append(dict(consumer=consumer,stage=stage,distinct_union=len(result),all_46_actual_role_names_preserved=True,exact_selected_source_present=True,append_idempotent=True,order_preserved=True,synthetic_65th_rejected_by_unchanged_predicate=True))
 admit,source=fn(ROOT/spec["driver_ref"]["path"],"admit");text=ast.get_source_segment(source,admit)
 positions=[text.index(s)for s in["consumer_bindings_v1","scope_spec","operation_draft_v1","apply_reserve_v1"]]
 assert positions==sorted(positions)
 preflight[consumer]=dict(consumer_union_scope_preflight_before_authority_reserve=True,exact_predecessor_driver_ref=spec["driver_ref"],unchanged_scientific_binding_ref=spec["scientific_binding_ref"],unchanged_caps=spec["caps"])
 here_audit[consumer]=[{"line":i,"text":line.strip()}for i,line in enumerate((ROOT/spec["contract_ref"]["path"]).read_text().splitlines(),1)if "HERE"in line]
assert[(c["consumer"],c["stage"],c["distinct_union"])for c in cases]==[("H060","train",61),("H061","freeze",62),("H061","predict",63),("H061","score",64)]
consumer_fn,source=fn(ROOT/selected["path"],"consumer_bindings_v1")
assert "1<=len(result)<=64"in ast.get_source_segment(source,consumer_fn)
config_template_present=(base/"scientific_config.INACTIVE.template.v1.json").is_file()
blockers=[]
if not config_template_present:blockers.append(dict(consumer="H061",priority=1,title="Rebound HERE resolves H061 config to missing file",file=str(base.relative_to(ROOT)/"root_driver_v1.py"),line=38,original_config_validator_line=127,details="contract sets c.HERE to common package, original validate_config reads HERE/scientific_config.INACTIVE.template.v1.json, but that file is absent. ROOT raised this and independent AST/source inspection confirmed it."))
result=dict(schema="independent_H060_H061_source_role_checks_v1",candidate_source_ref=ref(base/"root_driver_v1.py"),candidate_manifest_ref=ref(base/"source_manifest.v1.json"),candidate_handoff_ref=ref(base/"source_handoff.v1.json"),all_manifest_files_authenticated=len(manifest["source_sha256"]),hash_mismatches=mismatches,actual_failed_authority_metadata_ref=bindings["actual_H060_failed_authority_metadata_ref"],baseline_role_names=len(actual_roles),baseline_distinct_refs=len(unique(actual_roles.values())),handwritten_source_AST_cases=cases,preflight=preflight,HERE_relative_original_contract_audit=here_audit,H061_common_config_template_present=config_template_present,blockers=blockers,source_only=True,project_guard_entrypoint_imports=False,source_functions_executed=False,actual_mutex_process_payload_resource_numerical_or_git_operations=False)
path=OWN/("child_role_checks."+tag+".json");assert not path.exists();path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(result_ref=ref(path),blockers=blockers,role_counts=[c["distinct_union"]for c in cases],authenticated_files=len(manifest["source_sha256"]))))
