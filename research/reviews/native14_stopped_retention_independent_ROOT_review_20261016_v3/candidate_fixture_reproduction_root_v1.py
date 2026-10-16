"""Narrow pure/fake API contracts for historical-role consolidation only."""
import ast,copy,hashlib,json,types
from pathlib import Path
HERE=Path('research/reviews/native14_stopped_retention_successor_source_20261016_v3');OLD=Path('research/reviews/native14_stopped_retention_successor_source_20261016_v2')
REPORT=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def check(name,ok):
    if not ok:raise AssertionError(name)
    REPORT.append(dict(case=name,status="PASS"))
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError):check(name,True);return
    raise AssertionError(name)
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
    p=Path(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
source=(HERE/"native14_stopped_retention_v1.py").read_text();tree=ast.parse(source)
def fn(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
env=dict(copy=copy,require=require)
for n in tree.body:
    if isinstance(n,ast.Assign):
        try:value=ast.literal_eval(n.value)
        except (ValueError,TypeError):continue
        for target in n.targets:
            if isinstance(target,ast.Name):env[target.id]=value
m=read(HERE/"source_manifest.v1.json");audit=read(m["evidence_audit_ref"]["path"])
historical=m["original_reservation"]["consumer_bindings"]
encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
env["budget"]=types.SimpleNamespace(digest=lambda v:hashlib.sha256(encoded(v)).hexdigest())
calls=[]
def raw(r):calls.append(copy.deepcopy(r));return b""
env.update(_raw=raw,_ref=ref,_completed_json=lambda r:m if r["path"]==str(HERE/"source_manifest.v1.json") else audit)
nodes=[copy.deepcopy(fn(tree,n)) for n in ["_native_manifest","_bindings","consumer_bindings_v1"]]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"<native14-v3-fake-API>","exec"),env)
check("original_reservation_digest_unchanged",env["budget"].digest(m["original_reservation"])==m["original_reservation_sha256"])
check("exact_original45_all_inside222_manifest",len(historical)==45 and all(r in m["runtime_verified_source_and_compact_metadata_refs"] for r in historical))
env["_native_manifest"]()
check("manifest_authenticates_all222_refs",calls==m["runtime_verified_source_and_compact_metadata_refs"])
for label,mutate in [
    ("omitted_original_binding",lambda z:z["original_reservation"]["consumer_bindings"].pop()),
    ("changed_original_binding",lambda z:z["original_reservation"]["consumer_bindings"][0].update(sha256="0"*64)),
    ("changed_original_root",lambda z:z["original_reservation"].update(output="/other/root")),
    ("missing_historical_runtime_auth",lambda z:z["runtime_verified_source_and_compact_metadata_refs"].remove(historical[0]))]:
    saved=copy.deepcopy(m);mutate(m)
    reject("manifest_rejects_"+label,env["_native_manifest"]);m=saved
# Real source function assembles actual current23 + three source refs + four future
# authority roles. Fake records below are software-only and never activation.
historical=m["original_reservation"]["consumer_bindings"]
parent=read(OLD/"direct_role_plan.v1.json")["current_Census23_refs"]
policy=dict(root_activation_ref=dict(path="SOFTWARE/activation.json",sha256="1"*64),
    permanent_revocation_ref=dict(path="SOFTWARE/revocation.json",sha256="2"*64),
    source_review_ref=dict(path="SOFTWARE/review.json",sha256="3"*64))
env["_POLICY_REF"]=dict(path="SOFTWARE/policy.json",sha256="4"*64)
base={"base_%02d"%i:r for i,r in enumerate(parent)}
calls.clear();result=env["_bindings"](base,policy)
check("exact_current23_plus7_is30",len({r["path"] for r in result.values()})==30)
check("no_ancestor_binding_overwritten",all(result[k]==v for k,v in base.items()))
check("no_historical_only_roles_flattened",not any(k.startswith("native14_original_consumer_") for k in result))
# _native_manifest first authenticates222, then _bindings explicitly authenticates45.
check("every_original_binding_explicitly_raw_authenticated_in_original_order",calls[222:267]==historical)
saved_raw=env["_raw"]
def fail_one(r):
    if r==historical[-1]:raise ValueError("SOFTWARE changed historical source")
    return saved_raw(r)
env["_raw"]=fail_one
reject("any_changed_original_binding_blocks_state",lambda:env["_bindings"](base,policy))
env["_raw"]=saved_raw
rows=[dict(path="SOFTWARE/role_%02d.json"%i,sha256="%064x"%i) for i in range(65)]
check("unchanged_deduplicator_accepts_exact64",len(env["consumer_bindings_v1"](rows[:64],rows[:3]))==64)
reject("unchanged_deduplicator_rejects65",lambda:env["consumer_bindings_v1"](rows))
# Only two native-specific function bodies differ, apart from directory identity.
oldt=ast.parse((OLD/"native14_stopped_retention_v1.py").read_text())
unchanged=0
for n in oldt.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name not in {"_bindings","_native_manifest"}:
        normalized=ast.parse(ast.unparse(n).replace(str(OLD),str(HERE))).body[0]
        check("preserved_"+n.name,ast.dump(normalized,include_attributes=False)==ast.dump(fn(tree,n.name),include_attributes=False));unchanged+=1
check("retained_helper_byte_identical",(HERE/"retained_nodes_v1.py").read_bytes()==(OLD/"retained_nodes_v1.py").read_bytes())
report=dict(schema="native14_v3_narrow_historical_auth_consolidation_contracts",status="PASS",cases=len(REPORT),results=REPORT,
    preserved_production_function_ASTs=unchanged,actual_guard_or_accounting_calls=False,actual_scope_or_process_probes=False,
    source_compact_metadata_only=True,historical_files_or_bindings_modified=False)
(Path('research/reviews/native14_stopped_retention_independent_ROOT_review_20261016_v3')/"candidate_fixture_reproduction.root_v1.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status="PASS",checks=len(REPORT),preserved_functions=unchanged),sort_keys=True))
