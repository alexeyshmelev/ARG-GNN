"""Consume the delivered independent review through original candidate policy/selector/state predicates, RAM IO only."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/Census_observation_two_scope_retention_source_20261017_v1"
O=ROOT/"research/reviews/Census_observation_two_scope_retention_independent_20261017_v1"
def ref(p):return {"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
review_ref=ref(O/"review.v1.json")
source=(D/"ram_integration_check_v1.py").read_text()
assert hashlib.sha256(source.encode()).hexdigest()=="316c7bed1768aff7e244e23143b49e617ea01a4940f200b041942c2a7e160349"
t=ast.parse(source);nodes=[];replacements=0
for n in t.body:
 if n.lineno>=244:continue
 if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="review"for x in n.targets):
  n.value=ast.Name(id="__DELIVERED_REVIEW_REF__",ctx=ast.Load());nodes.append(n)
  nodes.append(ast.Expr(value=ast.Call(func=ast.Name(id="seed",ctx=ast.Load()),args=[ast.Name(id="review",ctx=ast.Load()),ast.Constant(value=True)],keywords=[])))
  replacements+=1
 else:nodes.append(n)
assert replacements==1
t.body=nodes;ast.fix_missing_locations(t)
g={"__name__":"independent_delivered_review_RAM_consumption","__DELIVERED_REVIEW_REF__":copy.deepcopy(review_ref)}
exec(compile(t,"<audited_RAM_IO_with_delivered_review>","exec"),g)
e=g["env"];module=g["module"]
assert e["load_selected_guard_v1"](g["selectorref"])is module
assert e["_policy"]()["source_review_ref"]==review_ref
loaded=e["_json"](review_ref);assert loaded["independent"]is True and loaded["verdict"]=="SOURCE_PASS"and loaded["blockers"]==[]
assert loaded["source_ref"]==g["records"][e["SOURCE"]]
assert loaded["source_manifest_ref"]==g["records"][e["MANIFEST"]]
assert loaded["measurement_source_ref"]==g["records"][e["MEASURE_SOURCE"]]
S=g["records"][g["budget"].SOURCE_REL]["sha256"];L=g["records"][g["budget"].LAUNCHER_REL]["sha256"]
state=e["_state_locked"](S,L,fresh=True)
assert len(state[3])==47 and all(state[3][k]==v for k,v in g["parentbindings"].items())
assert state[2]["completed_credit_bytes"]==0 and not state[4]
assert state[2]["projected_bytes"]==g["baseview"]["projected_bytes"]+2097152
print(json.dumps(dict(schema="Census_observation_delivered_review_consumption_RAM_v1",status="PASS_SOURCE_ONLY",
 delivered_review_ref=review_ref,actual_candidate_policy_selector_and_state_accepted_exact_delivered_review=True,
 same_candidate_singleton_returned=True,parent_direct_roles_preserved=46,child_direct_roles=47,
 no_RAM_retention_event_or_credit=True,actual_resource_operations=0,source_or_helper_modifications=0),sort_keys=True,indent=2))
