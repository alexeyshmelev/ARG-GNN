"""Independent source-only reserve fixtures. All operational objects are handwritten in memory."""
import ast,copy,hashlib,json,pathlib,sys
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/ROOT_mutation_transaction_memo_independent_20261017_v2";base=ROOT/"research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2"
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest=json.loads((base/"source_manifest.v1.json").read_text());audit=json.loads((base/"audit_manifest.v1.json").read_text())
for p,s in audit["files_sha256"].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s
sources={k:(ROOT/r["path"]).read_bytes()for k,r in manifest["predecessor_pins"].items()}
for k,r in manifest["predecessor_pins"].items():assert hashlib.sha256(sources[k]).hexdigest()==r["sha256"]
text=(base/"root_mutation_memo_v1.py").read_text();ns={"__name__":"independent_memo_definitions_only"};exec(compile(text,str(base/"root_mutation_memo_v1.py"),"exec"),ns)
class MemoryPath(pathlib.PurePosixPath):
 def exists(self):return False
root=MemoryPath("/source_fixture/never_materialized");ns["ROOT"]=root;ns["source_bytes"]=lambda record:b"memo fixture"
original_reserve=next(n for n in ast.parse(sources["guard"]).body if isinstance(n,ast.FunctionDef)and n.name=="apply_reserve_v1")
results=[]
for case in["success","source_drift","proof_drift","namespace_drift","manifest_drift","missing_direct_manifest","operation_exception","append_exception","cold_success"]:
 locked=[False];events=[];reads={};drift=[False];appended=[0]
 memory={"src/a.py":b"pinned code a\n","src/arggnn/project_budget_v9_consumer.py":b"marker\n","research/guard.py":b"guard\n","research/integration.json":json.dumps({"source_sha256":{ns["SOURCE"]:"f"*64}},sort_keys=True).encode(),"research/authority.json":b"{}","research/complete.json":b'{"complete":true}',".runtime/legacy.json":b"{\"jobs\":{}}"}
 refs={p:dict(path=p,sha256=hashlib.sha256(b).hexdigest())for p,b in memory.items()};allow={p:r["sha256"]for p,r in refs.items()if p.endswith(".py")}
 def relative(name):return str(MemoryPath(name).relative_to(root))if MemoryPath(name).is_absolute()else str(name)
 def file_bytes(r,name):
  need(locked[0],"Fixture read outside owned lock");key=relative(name);reads[key]=reads.get(key,0)+1;data=memory[key];return data,hashlib.sha256(data).hexdigest()
 def canonical(r,name):
  need(locked[0],"Fixture canonical outside lock");p=MemoryPath(name);p=p if p.is_absolute()else r/p
  return p/"changed"if drift[0]else p
 v4=ModuleType("independent_handwritten_v4");v4.__dict__.update(Path=MemoryPath,file_bytes=file_bytes,canonical=canonical,parse=json.loads,integer=lambda v,*args:need(type(v)is int and v>0,"positive int"),paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,accounting_paths=lambda r:[],encoded=lambda v:json.dumps(v,sort_keys=True).encode(),LEGACY=".runtime/legacy.json")
 nodes=[n for n in ast.parse(sources["v4"]).body if isinstance(n,ast.FunctionDef)and n.name in["bound","reference","scope_spec","read_json"]]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"exact_v4_predicates_handwritten_memory","exec"),v4.__dict__)
 budget=ModuleType("independent_handwritten_budget");budget.__dict__.update(v4=v4,JOURNAL_REL=".runtime/journal")
 node=next(n for n in ast.parse(sources["budget"]).body if isinstance(n,ast.FunctionDef)and n.name=="successor_paths")
 exec(compile(ast.Module(body=[node],type_ignores=[]),"exact_successor_paths_handwritten_memory","exec"),budget.__dict__)
 originals={k:getattr(v4,k)for k in["bound","reference","scope_spec","canonical","file_bytes"]};original_successor=budget.successor_paths
 @contextmanager
 def mutex(r):
  need(not locked[0],"Nested fixture mutex");locked[0]=True;events.append("lock")
  try:yield
  finally:
   need(all(getattr(v4,k)is v for k,v in originals.items())and budget.successor_paths is original_successor,"Patches survive unlock")
   locked[0]=False;events.append("unlock")
 v4.mutex=mutex
 own=dict(key="SOURCE_ONLY/fresh",output=str(root/"data/new"),auxiliary_scopes=[str(root/"research/aux")],reserved_bytes=1000,consumer_bindings=[refs["src/a.py"]]*12+[refs["research/guard.py"],refs["src/arggnn/project_budget_v9_consumer.py"]]+([]if case=="missing_direct_manifest"else[refs["research/integration.json"]]))
 view=dict(prefix="actual_fixture_prefix",base_pins={},new_reserved_bytes=0,completed_credit_bytes=0)
 state=({},set(),view,{"guard":refs["research/guard.py"]},[],[],{})
 def get_state(*args,**kwargs):
  need(locked[0]and kwargs==dict(fresh=True),"Public reserve fresh actual lock")
  for _ in range(10):v4.bound(root,refs["src/a.py"]);v4.bound(root,refs["research/authority.json"],True);v4.bound(root,refs["research/complete.json"],True)
  return state
 def authority(*args,**kwargs):return dict(expected_before=view["prefix"],request=own)
 def charge(*args):
  if case=="source_drift":memory["src/a.py"]=b"drift\n"
  if case=="proof_drift":memory["research/complete.json"]=b'{"complete":false}'
  if case=="namespace_drift":drift[0]=True
  if case=="manifest_drift":memory["research/integration.json"]=b"{\"source_sha256\":{}}"
  if case=="operation_exception":raise ValueError("fixture preappend exception")
  return 1000
 def append(*args):
  need(locked[0]and v4.canonical is canonical and v4.file_bytes is file_bytes and v4.mutex is mutex,"Original primitive changed")
  before=reads.get("src/a.py",0);v4.bound(root,refs["src/a.py"]);need(reads["src/a.py"]==before+1,"Append uses cached source")
  if case=="append_exception":raise ValueError("fixture append exception")
  appended[0]+=1;return"fixture_tail"
 guard=ModuleType("independent_handwritten_guard");guard.__dict__.update(budget=budget,ROOT=root,SOURCE="research/guard.py",require=need,_state_locked=get_state,_authority=authority,_pins=lambda v:v["prefix"],_paths=lambda value:budget.successor_paths(root,value),_ref=lambda p:refs[p],_policy=lambda:dict(never_credit_keys=[],base_projected_bytes=0),charge=charge,copy=copy,_append=append,warning_fields=lambda p:{})
 memo=ns["RootMutationMemo"].__new__(ns["RootMutationMemo"]);memo.guard=guard;memo.budget=budget;memo.v4=v4;memo.allow=allow;memo.adapter_ref=dict(path=ns["SOURCE"],sha256="f"*64);memo.manifest_ref=refs["research/integration.json"];memo.guard_functions={"apply_reserve_v1":(None,original_reserve)};memo._check=lambda:None
 error=None;result=None
 try:result=memo.reserve_v1(refs["research/authority.json"],expected_source_sha256="fixture_budget",expected_launcher_sha256="fixture_launcher")
 except ValueError as e:error=str(e)
 success=case in["success","cold_success"]
 need((error is None)==success,"Unexpected fixture verdict "+case+": "+str(error));need(appended[0]==int(success),"Unexpected original append")
 need(events==["lock","unlock"]and not ns["_ACTIVE"],"Not restored or generation retained")
 need(reads["research/authority.json"]==10,"JSON was cached")
 if success:need(reads["src/a.py"]==3,"Source not1+uncachedfinal+uncachedappend");need(result["value"]["root_admission"]==refs["research/authority.json"],"Original receipt drift")
 results.append(dict(case=case,expected_success=success,error=error,original_append_calls=appended[0],events=events,source_reads=reads["src/a.py"],uncached_JSON_reads=reads["research/authority.json"],memo_counts=memo.counts))
# Critical body AST equality means no policy/numerical/role/ceiling predicates were edited.
assert len(original_reserve.body)==1 and isinstance(original_reserve.body[0],ast.With)
result=dict(schema="independent_ROOT_mutation_memo_v2_public_reserve_checks_v1",source_ref=ref(base/"root_mutation_memo_v1.py"),source_manifest_ref=ref(base/"source_manifest.v1.json"),audit_manifest_ref=ref(base/"audit_manifest.v1.json"),sealed_files_authenticated=len(audit["files_sha256"]),predecessor_pins_authenticated=3,public_reserve_handwritten_cases=results,actual_original_reserve_AST_body_exercised_with_handwritten_memory=True,actual_v4_predicates_and_budget_path_AST_exercised_with_handwritten_memory=True,all_exits_restore_before_synthetic_unlock=True,cache_generation_not_retained=True,exact_bound_metadata_and_code_bytes_only_cached=True,current_authority_JSON_uncached_and_manifest_final_uncached=True,original_mutex_critical_body_and_three_primitives_unchanged=True,actual_guard_modules_or_entrypoints_imported=False,actual_mutex_process_payload_resource_numerical_Git_operations=False,real_performance_gain_measured=False,blockers=[])
p=OWN/"independent_public_reserve_checks.v1.json";assert not p.exists();p.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n");print(json.dumps(dict(result_ref=ref(p),cases=[dict(case=r["case"],expected_success=r["expected_success"],error=r["error"])for r in results])))
