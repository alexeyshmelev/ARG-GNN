"""RAM-only source/schema/metadata doubles. No project imports or resource operations."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H059_outer000_Relate64_execution_source_20261017_v1"
OLD="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
checks=[];negative=[]
def need(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads((ROOT/p).read_text())
def rr(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def fn(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def run_nodes(tree,names,env):
 nodes=[copy.deepcopy(fn(tree,n)) for n in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"<RAM-source-metadata-doubles>","exec"),env)
def reject(call,label):
 try:call()
 except(ValueError,KeyError,TypeError):negative.append(label);return
 raise ValueError("Malformed metadata accepted: "+label)
b=read(HERE+"/source_bindings.v1.json");new=ast.parse((ROOT/HERE/"prepare_outer000_v1.py").read_text());old=ast.parse((ROOT/OLD/"prepare_outer000_v1.py").read_text())
for p in (ROOT/HERE).glob("*.py"):ast.parse(p.read_text())
checks.append("All new Python ASTs parse without imports")
for name in("worker","execute"):
 node=copy.deepcopy(fn(new,name))
 node.body=[n for n in node.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="verify_generation_roles_v1")]
 need(dump(node)==dump(fn(old,name)),"Original "+name+" AST retained except full current-role assertion")
checks.append("Original scientific worker, native call, runtime checks, pidfd gates, owned wait/drain/cleanup and evidence bodies retained")
for name in("freeze","registration","guard","boundary","current","put"):
 need(dump(fn(new,name))==dump(fn(old,name)),"Original "+name+" retained")
checks.append("Original freeze/registration/raw ledger/current/memo/boundary predicates retained")
source=(ROOT/HERE/"prepare_outer000_v1.py").read_text()
need("auxiliary_scopes=[str(packet)]" in source,"Entire packet charged")
admit=ast.unparse(fn(new,"admit"))
need(admit.index("g.budget.v4.scope_spec")<admit.index("g.operation_draft_v1")<admit.index("g.apply_reserve_v1")<admit.index("packet.mkdir"),"Preflight and reserve precede packet mkdir")
need('put(prep / (packet.name +' in admit and "verify_generation_roles_v1" in source,"Only staging authority before reserve")
need("root_packet_bytes" in ast.unparse(fn(new,"sample_scope")) and "totals.append(admin)" in source,"Whole scope polling includes admin")
checks.append("Original schema preflight before mutation; packet/output absent through reserve; whole packet/AUX/output charged and polled")
frozen=read(b["frozen_method_freeze_ref"]["path"]);registry=read("research/reviews/European_outer8_preparation_source_20261016_v3/outer8_registry.v1.json");cfg=read("configs/expanded_full_panel_v1.json");protocol=read("research/reviews/European_outer8_preparation_source_20261016_v3/protocol.v1.json")
outer=[x for x in cfg["parent_registry"] if x["split"]=="outer"]
need(outer==registry["outer_rows"] and len(outer)==8 and all(x["downstream_role"]=="UNTOUCHED_OUTER" and x["evaluation_authorized"] is False for x in outer),"Exact eight reserved untouched original rows")
need(cfg["simulation"]["length_bp"]==2000000 and cfg["roster"]["haplotypes"]==952 and cfg["roster"]["query_haplotypes"]==686 and cfg["roster"]["pure_reference_haplotypes"]==266,"Full original science counts")
need(frozen["method_ids"]==["H050_stable_leaf_memory","H050_h045_continued_training","AncestralPaths_native_Dense5","ARGMix_current_native","ARGformer_native_within_ARG"] and frozen["H058_automatically_included"] is False and protocol["all8_predictions_for_all5_methods_before_any_outer_truth"] is True,"Original five methods and all40+8 barrier")
caps=read(HERE+"/caps.INACTIVE.v1.json")
need(caps["reservation_bytes"]==4*2**30 and caps["output_bytes"]==4000000000 and caps["auxiliary_bytes"]==128*2**20 and caps["root_packet_bytes"]==8*2**20 and caps["output_bytes"]+caps["auxiliary_bytes"]+caps["root_packet_bytes"]<caps["reservation_bytes"],"Whole unchanged4GiB supports explicit admin failstop")
checks.append("All eight seed rows/config/demography/frozen methods/full9522Mb and48pretruth products unchanged; all original caps retained")
# Exact current selection metadata function executed only with handwritten read/raw doubles.
chosen=read(b["actual_selector_ref"]["path"]);selected=dict(need=need,ACTUAL_SELECTOR_REF=b["actual_selector_ref"],SELECTOR=b["guard_source_ref"],APPROVED={chosen["accounting_mode"]:dict(source_ref=chosen["source_ref"],manifest_ref=chosen["source_manifest_ref"],policy_ref=chosen["policy_ref"])},read=lambda r:copy.deepcopy(chosen),raw=lambda r:r)
run_nodes(new,["selection"],selected)
need(selected["selection"](b["actual_selector_ref"])==chosen,"Actual current selector metadata accepted")
reject(lambda:selected["selection"](dict(path="stale.json",sha256="a"*64)),"Stale selector identity")
for key,bad in(("active",False),("source_ref",dict(path="old.py",sha256="a"*64)),("policy_ref",dict(path="old.json",sha256="b"*64)),("accounting_mode","native7_H050_stopped_retention_v1")):
 prior=chosen[key];chosen[key]=bad;reject(lambda:selected["selection"](b["actual_selector_ref"]),key);chosen[key]=prior
checks.append("Exact Relate64 selector metadata accepted; stale/inactive/source/policy/mode substitutions rejected")
# Independent full union from all actual frozen guard bindings plus source-derived generation roles.
P=lambda name:dict(path="RAM_ONLY/"+name+".json",sha256=hashlib.sha256(name.encode()).hexdigest())
constants={}
for node in new.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name):
    if target.id in("CONFIG","PRODUCER","ORIGINAL","PIDFD","OWNED","NAMESPACE","V4"):
     value=node.value
     if isinstance(value,ast.Call) and isinstance(value.func,ast.Name)and value.func.id=="dict":
      constants[target.id]={x.arg:ast.literal_eval(x.value) for x in value.keywords}
env=dict(need=need,ref=lambda p:P(p) if p in(HERE+"/source_manifest.v1.json",HERE+"/source_handoff.v1.json")else rr(p) if (ROOT/p).is_file() else P(p),HERE=HERE,SOURCE=HERE+"/prepare_outer000_v1.py",MANIFEST=HERE+"/source_manifest.v1.json",REGISTRY="research/reviews/European_outer8_preparation_source_20261016_v3/outer8_registry.v1.json",METHODS="research/reviews/European_outer8_preparation_source_20261016_v3/frozen_method_ids.v1.json",PROTOCOL="research/reviews/European_outer8_preparation_source_20261016_v3/protocol.v1.json",SELECTOR=b["guard_source_ref"],ORIGINAL_SELECTOR=b["original_selector_ref"],**constants)
run_nodes(new,["generation_direct_refs_v1"],env)
q={k:P(k)for k in("runtime_ref","source_review_ref","runtime_qualification_ref")}
q.update(method_freeze_ref=b["frozen_method_freeze_ref"],policy_selector_ref=b["actual_selector_ref"],registration_ref=b["registration_ref"],registration_decisions_ref=b["registration_decisions_ref"],registration_ledger_ref=b["registration_ledger_ref"])
roles=env["generation_direct_refs_v1"](q,P("request"));guards=read(b["guard_role_snapshot_ref"]["path"])["bindings"]
unique=lambda rs:list({(r["path"],r["sha256"]):r for r in rs}.values())
union=unique([*guards.values(),*roles])
need(len(guards)==46 and len(unique(guards.values()))==39 and len(union)<=64 and all(r in union for r in guards.values()),"Every named actual historical guard retained in full role union")
need(all(Path(r["path"]).suffix in {".py",".sh",".json"} for r in roles),"No direct Markdown roles")
# Execute only original scope_spec with RAM path/hash doubles, never original bound/canonical/mutex.
scope=ast.parse((ROOT/"src/arggnn/project_budget_v4.py").read_text());root=ROOT
def bound(root,r,stored=False):
 need(isinstance(r,dict)and set(r)=={"path","sha256"} and isinstance(r["path"],str)and len(r["sha256"])==64,"RAM source/metadata ref")
 return r
def integer(v,name,minimum=0):need(type(v)is int and v>=minimum,"RAM exact integer")
env2=dict(Path=Path,integer=integer,canonical=lambda root,p:root/p if not Path(p).is_absolute()else Path(p),paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,accounting_paths=lambda root:[],bound=bound)
run_nodes(scope,["scope_spec"],env2)
own=dict(key=b["proposed_reservation_key"],output=str(ROOT/b["output_directory"]),auxiliary_scopes=[str(ROOT/b["packet_directory"])],reserved_bytes=caps["reservation_bytes"],consumer_bindings=union)
need(len(env2["scope_spec"](ROOT,own))==2,"Exact original scope schema accepts complete charged packet roles")
bad=copy.deepcopy(own);bad["consumer_bindings"].append(b["registration_ledger_ref"]);reject(lambda:env2["scope_spec"](ROOT,bad),"Direct Markdown")
bad=copy.deepcopy(own);bad["consumer_bindings"] += [P("extra"+str(i))for i in range(65-len(union))];reject(lambda:env2["scope_spec"](ROOT,bad),"Original65role overflow")
bad=copy.deepcopy(own);bad["auxiliary_scopes"]=[bad["output"]];reject(lambda:env2["scope_spec"](ROOT,bad),"Overlapping scope")
checks.append("Full source-derived role union and original scope_spec pass with RAM doubles; Markdown,65roles and overlap reject")
need("registration_ledger_ref" in ast.unparse(fn(new,"registration")) and "registration_ledger_ref"not in ast.unparse(fn(new,"generation_direct_refs_v1")),"Ledger still raw-authenticated behind valid direct JSON")
checks.append("Immutable ledger and decisions remain bound through genuine ROOT request/registration JSON")
for name in("ROOT_request.INACTIVE.template.v1.json","runtime_qualification.INACTIVE.template.v1.json","independent_review.INACTIVE.template.v1.json"):
 v=read(HERE+"/"+name);need(v.get("root_authorizer")is None or name.startswith("independent"),"Inactive authorizer")
checks.append("No fabricated ROOT clocks/authority/runtime/review facts")
result=dict(schema="H059_outer000_Relate64_source_metadata_checks_v1",status="PASS_SOURCE_ONLY_INACTIVE_UNREVIEWED",independent=False,source_only=True,checks=checks,checks_count=len(checks),negative_cases=negative,guard_role_names=46,distinct_guard_roles=39,direct_role_count=len(unique(roles)),full_role_union_count=len(union),original_role_ceiling=64,full_role_union=union,actual_guard_state_resource_capacity_or_scope_absence_observed=False,numerical_payload_reads_science_runtime_guard_process_probe_admission_journal_mutex_GPU_install_or_Git=False)
(ROOT/HERE/"source_check.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status=result["status"],checks=len(checks),negative=len(negative),full_role_union=len(union))))
