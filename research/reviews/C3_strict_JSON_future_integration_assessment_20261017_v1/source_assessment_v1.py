"""Source-only C3 future parse-memo coverage assessment. No actual integration."""
from pathlib import Path
import json,ast,hashlib,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/C3_strict_JSON_future_integration_assessment_20261017_v1"
verified={}
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def raw(record):
 p=ROOT/record["path"];before=p.lstat();assert p.resolve()==p and stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**21
 data=p.read_bytes();after=p.lstat()
 key=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 assert key(before)==key(after)and len(data)==before.st_size and hashlib.sha256(data).hexdigest()==record["sha256"]
 verified[record["path"]]=record["sha256"];return data
v4ref=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
v4tree=ast.parse(raw(v4ref));env={"json":json}
parser=next(n for n in v4tree.body if isinstance(n,ast.FunctionDef)and n.name=="parse")
exec(compile(ast.Module(body=[parser],type_ignores=[]),str(ROOT/v4ref["path"])+"#pure_parser","exec"),env)
def read(r):return env["parse"](raw(r))
reviewref=dict(path="research/reviews/consumer_strict_JSON_parse_memo_independent_20261017_v2/review.v2.json",sha256="dee52072371380a98b39adb1a9aa5235ddd85fb10824097ba0da6441a9a5bea8")
review=read(reviewref);assert review["verdict"]=="SOURCE_PASS"and review["blockers"]==[]
raw(review["component_ref"])
cpath="research/reviews/C3_remaining_cells_GPU0_source_20261017_v5/contract_v1.py"
cref=dict(path=cpath,sha256="06902e8602fcc6c4ca97b6ce1fc9b41445a8890275441bc4c572aada3513efe7")
craw=raw(cref);ct=ast.parse(craw)
mref=dict(path="research/reviews/C3_remaining_cells_GPU0_source_20261017_v5/source_manifest.v1.json",sha256="24fc06c2165003792f24b2e9393840bd0badc68bb2e1af8594b15145cc447a2b")
m=read(mref)
jsonpaths={x["path"]for x in m["source_authentication_refs"]if x["path"].endswith(".json")}
ipath="research/reviews/C3_remaining_cells_GPU0_ROOT_v5_predict_val_000_root01_oct17/input.json"
iref=ref(ipath);i=read(iref);storage=read(i["refs"]["storage"]);q=read(i["root_stage_request_ref"]);own=storage["reservation"]
assert len(own["consumer_bindings"])==len({r["path"]for r in own["consumer_bindings"]})==64
nspath="research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py"
nsraw=raw(dict(path=nspath,sha256="17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f"))
assert "MEMO_REF" in nsraw.decode()and "TransactionMemoAdapter" in nsraw.decode()
history_files=[
"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json",
"research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/source_manifest.v1.json",
"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json"]
rows=[]
for mp in history_files:
 mr=ref(mp);hm=read(mr)
 for key,t in hm["targets"].items():
  r=t["storage_ref"];data=raw(r);value=env["parse"](data)
  nodes=containers=0;stack=[value]
  while stack:
   x=stack.pop();nodes+=1
   if type(x)is dict:containers+=1;stack.extend(x.keys());stack.extend(x.values())
   elif type(x)is list:containers+=1;stack.extend(x)
  rows.append(dict(original_historical_key=key,storage_ref=r,frozen_manifest_ref=mr,
   raw_bytes=len(data),JSON_nodes=nodes,containers=containers,
   in_existing_closure_control_exclusions=r["path"]in jsonpaths))
assert len(rows)==5 and sum(r["raw_bytes"]for r in rows)==47584 and sum(r["JSON_nodes"]for r in rows)==1286 and sum(r["containers"]for r in rows)==243
assert not any(r["in_existing_closure_control_exclusions"]for r in rows)
calls=[]
for sp in [
"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py",
"research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py",
"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py"]:
 sr=ref(sp);tree=ast.parse(raw(sr))
 for fn in tree.body:
  if isinstance(fn,ast.FunctionDef):
   for n in ast.walk(fn):
    if isinstance(n,ast.Call)and ast.unparse(n.func)=="_manifest":
     calls.append(dict(source_ref=sr,owner=fn.name,line=n.lineno))
assert len(calls)==6 and {r["owner"]for r in calls}=={"_bindings","_target"}
# Exact original all30 same-source predicate only. No fabricated wait or completion is constructed.
fn=next(n for n in ct.body if isinstance(n,ast.FunctionDef)and n.name=="new_stage_values_v1")
pred=next(n.value.args[0]for n in fn.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and n.value.func.id=="require"and "Exact same source/election" in ast.unparse(n))
predicate=compile(ast.Expression(body=pred),str(ROOT/cpath)+"#same_source_predicate","eval")
pe=dict(s=i,q=q,parent=q["parent_id"],election=q["cell_election_ref"],pending=q["pending_epochs"],MANIFEST=mref["path"],ref=lambda path:mref)
assert eval(predicate,pe) is True
future_ref={"path":"RAM_ONLY/future_v6_manifest.json","sha256":"6"*64}
pe["ref"]=lambda path:future_ref
assert eval(predicate,pe) is False
result={
 "schema":"C3_future_strict_JSON_integration_source_assessment_v1",
 "decision":"NO_SUCCESSOR_JUSTIFIED_BY_CURRENT_NARROW_COVERAGE",
 "source_only":True,"component_independent_review_ref":reviewref,"component_ref":review["component_ref"],
 "baseline_contract_ref":cref,"baseline_source_manifest_ref":mref,
 "recorded_VAL000_v5_input_ref":iref,"recorded_VAL000_v5_storage_ref":i["refs"]["storage"],
 "recorded_bindings":64,"recorded_unique_binding_paths":64,"original_cap_preserved":64,
 "authenticated_existing_library_precedent":"Direct namespace source17a137 loads exact c972 original memo library through literal SHA. A new direct operational contract could similarly bind an explicit parse library in full closure without dropping any original role. No such new source was authored.",
 "current_closure_JSON_exclusion_count":len(jsonpaths),"complete_actual_future_control_closure_qualified":False,
 "eligible_historical_records":rows,"eligible_raw_bytes":47584,"eligible_JSON_nodes":1286,"eligible_containers":243,
 "static_manifest_call_sites":calls,
 "coverage_limits":[
  "One basic _bindings traversal parses each of these five records once, giving no parsed-cache hit for that basic pass.",
  "Additional hits arise only where historical _target/proof traversal repeats a manifest. Actual event/repetition frequency was not measured.",
  "All current policy/manifest/authority/ledger/journal/inventory predicates, first raw reads and original NFS namespace/scope/final identity work remain outside this narrow parse reuse.",
  "Every successful reuse must deepcopy, and additional final byte validation can reread up to five eligible files under the same original lock.",
  "Raw byte size/node counts are source coverage, not measured time, speedup, negligible-runtime proof or current capacity."
 ],
 "same_source_barrier_assessment":{
  "exact_original_current_VAL000_v5_same_source_predicate_passes":True,
  "exact_original_predicate_rejects_current_VAL000_v5_under_future_v6_self_manifest":True,
  "fixture_future_ref_is_RAM_only_not_actual_authority":True,
  "full_future_route_requirement":"Explicit exact VAL000v5+VAL001/002v6 lineage and proven unchanged numerical/model/data/selection bodies plus full all30pretruth seals would need independent whole-route review.",
  "original_predicate_or_scientific_barrier_relaxed":False
 },
 "practical_conclusion":"The only proposed reuse covers47.6KB/243containers across five historical records, with deepcopy and extra final IO. No evidence establishes material saving sufficient to justify a new operational source and mixed-source matrix lineage integration now. Keep current v5 route unchanged.",
 "future_work_if_new_evidence":"Only genuine separately authorized boundary counters or timing that identifies a material eligible repeated-parse cost would justify revisiting. This assessment authorizes no monitoring/probe/campaign.",
 "new_successor_implementation_authored":False,"existing_jobs_or_sources_changed":False,
 "actual_runtime_import_guard_state_clock_lock_process_resource_accounting_or_Git_operations":False,
 "numerical_or_training_fits":False,"actual_admission_or_live_adoption":False,"speedup_claimed":False,
 "verified_selected_refs":verified}
with(OUT/"assessment.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:result[k]for k in("decision","eligible_raw_bytes","eligible_JSON_nodes","eligible_containers","new_successor_implementation_authored","speedup_claimed")},sort_keys=True))
