"""Independent source-derived direct role arithmetic; no guard invocation."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D="research/reviews/H058_GPU0_full_training_successor_source_20261017_v4"
OUT=ROOT/"research/reviews/H058_GPU0_full_training_successor_source_20261017_v4"
def read(p):return (ROOT/p).read_text()
def ref(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
ns=dict(Path=Path)
common=ast.parse(read(D+"/common_v1.py"))
for n in common.body:
 if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"<source_constant>","exec"),ns)
q={k:dict(path="RAM/"+k+".json",sha256="0"*64) for k in ("config_ref","source_review_ref","direct_attachment_authority_ref","native_resource_disposition_ref")}
q["science_handoff_ref"]=ref(D+"/producer_handoff.v1.json")
ns.update(q=q,cfg=dict(source_manifest_ref=ref(D+"/producer_manifest.v1.json")),
 request_ref=dict(path="RAM/request.json",sha256="0"*64),producer=ref(D+"/producer_v1.py"),
 runtime=dict(path="RAM/runtime.json",sha256="0"*64),closure=dict(path="RAM/closure.json",sha256="0"*64),ref=ref,direct=SimpleNamespace())
for n in ast.parse(read(D+"/direct_gpu0_v1.py")).body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="CURRENT_CAP" for t in n.targets):ns["direct"].CURRENT_CAP=ast.literal_eval(n.value)
admit=next(n for n in ast.parse(read(D+"/root_driver_v1.py")).body if isinstance(n,ast.FunctionDef) and n.name=="admit_v1")
for n in admit.body:
 nodes=[]
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="needed" for t in n.targets):nodes=[n]
 if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="needed":nodes=[n]
 if isinstance(n,ast.If):
  nodes=[x for x in n.body if isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=="needed"]
 for x in nodes:exec(compile(ast.Module(body=[x],type_ignores=[]),"<exact_unevaluated_admission_role_construction>","exec"),ns)
capture_path="research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/actual_parent_capture.root_v1.json"
capture=json.loads(read(capture_path))
base=capture["all_actual_parent_bindings"]
parent={x["path"]:x for x in base.values()}
parent[ns["SELECTOR_REF"]["path"]]=ns["SELECTOR_REF"]
union=dict(parent)
for r in ns["needed"]:
 assert r["path"] not in union or union[r["path"]]==r
 union[r["path"]]=r
assert len(base)==45 and len(parent)==39 and len(union)==64 and len(set(union)-set(parent))==25
# Source-only state contract check: one capsule is the entire selected child's addition.
child=ast.parse(read(ns["GUARD"]["path"]))
bindings=next(n for n in child.body if isinstance(n,ast.FunctionDef) and n.name=="_bindings")
assert "Relate64_explicit_selector_capsule" in ast.unparse(bindings)
assert "result=copy.deepcopy(base_bindings)" in ast.unparse(bindings).replace(" ","")
# No production/scientific import: inspect the exact original protocol as source.
protocol_path="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/protocol_v1.py"
protocol=read(protocol_path)
result=dict(status="PASS",frozen_capture_ref=ref(capture_path),frozen_named_roles=len(base),
 current_selected_distinct_roles=len(parent),additional_direct_distinct_roles=25,total_distinct_roles=len(union),
 full_source_manifest_ref=ref(D+"/source_manifest.v1.json"),production_source_ref=ref(D+"/root_driver_v1.py"),
 original_protocol_ref=ref(protocol_path),direct_role_paths=sorted(union),role_arithmetic_is_not_live_state_evidence=True,
 source_guard_calls=False,real_mutex_or_journal_calls=False)
with (OUT/"join_checks.result.v1.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status="PASS",distinct_roles=len(union),result_ref=ref(str((OUT/"join_checks.result.v1.json").relative_to(ROOT)))),sort_keys=True))
