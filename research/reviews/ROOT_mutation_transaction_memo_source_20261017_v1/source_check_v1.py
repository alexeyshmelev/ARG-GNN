"""Stdlib source/AST plus handwritten memory checks; never imports project guards."""
import ast,copy,hashlib,json,pathlib,sys,traceback
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
HERE=pathlib.Path("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1")
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 source=(HERE/"root_mutation_memo_v1.py").read_bytes();tree=ast.parse(source);compile(tree,str(HERE/"root_mutation_memo_v1.py"),"exec")
 ns={"__name__":"inert_memo_source_test"};exec(compile(tree,str(HERE/"root_mutation_memo_v1.py"),"exec"),ns)
 originals={k:pathlib.Path(v["path"]).read_bytes()for k,v in ns["PINS"].items()}
 need(all(hashlib.sha256(originals[k]).hexdigest()==v["sha256"]for k,v in ns["PINS"].items()),"Exact three source pins, no imports")
 class MemoryPath(pathlib.PurePosixPath):
  def exists(self):return False
 root=MemoryPath("/handwritten/never_materialized")
 memory={"src/a.py":b"source a\n","scripts/b.sh":b"source b\n","src/arggnn/project_budget_v9_consumer.py":b"consumer\n",
  "research/reviews/authority.json":b'{"fresh":1}',".runtime/legacy.json":b'{"jobs":{}}'}
 refs={p:dict(path=p,sha256=hashlib.sha256(b).hexdigest())for p,b in memory.items()}
 allow={p:r["sha256"]for p,r in refs.items()if pathlib.PurePosixPath(p).suffix in(".py",".sh")}
 reads={};canon_calls=[0];locked=[False];drift=[False]
 def relative(name):return str(MemoryPath(name).relative_to(root))if MemoryPath(name).is_absolute()else str(MemoryPath(name))
 def canonical(r,name):
  need(locked[0],"Read under synthetic lock");canon_calls[0]+=1
  p=MemoryPath(name);p=p if p.is_absolute()else r/p
  return p/"changed"if drift[0]else p
 def file_bytes(r,name):
  need(locked[0],"Read under synthetic lock");key=relative(name);reads[key]=reads.get(key,0)+1
  value=memory[key];return value,hashlib.sha256(value).hexdigest()
 v4=ModuleType("handwritten_v4")
 v4.__dict__.update(Path=MemoryPath,canonical=canonical,file_bytes=file_bytes,parse=json.loads,
  integer=lambda value,*args:need(type(value)is int and value>=1,"integer"),accounting_paths=lambda r:[],
  paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,encoded=lambda v:json.dumps(v,sort_keys=True).encode(),
  LEGACY=".runtime/legacy.json")
 nodes=[n for n in ast.parse(originals["v4"]).body if isinstance(n,ast.FunctionDef)and n.name in{"bound","reference","scope_spec","read_json"}]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),"actual_v4_predicates_in_memory","exec"),v4.__dict__)
 v4.mutex=lambda r:None
 budget=ModuleType("handwritten_budget");budget.__dict__.update(v4=v4,JOURNAL_REL=".runtime/journal")
 successor=next(n for n in ast.parse(originals["budget"]).body if isinstance(n,ast.FunctionDef)and n.name=="successor_paths")
 exec(compile(ast.Module(body=[successor],type_ignores=[]),"actual_successor_paths_in_memory","exec"),budget.__dict__)
 original_objects={n:getattr(v4,n)for n in("canonical","file_bytes","mutex","bound","reference","scope_spec")}
 original_successor=budget.successor_paths
 own=dict(key="synthetic/fresh",output=str(root/"data/fresh"),auxiliary_scopes=[str(root/"research/reviews/fresh_aux")],
  reserved_bytes=1000,consumer_bindings=[refs[p]for p in allow]+[refs["research/reviews/authority.json"]])
 tx=ns["_Transaction"](root,v4,budget,allow);locked[0]=True
 try:
  tx.install()
  for _ in range(20):
   budget.successor_paths(root,own);v4.reference(root,"src/a.py")
   v4.bound(root,refs["research/reviews/authority.json"],True)
  need(all(getattr(v4,n)is original_objects[n]for n in("canonical","file_bytes","mutex")),"Three mutation primitives unchanged")
  need(all(reads[p]==1 for p in allow),"Repeated code authenticated once before final validation")
  need(reads["research/reviews/authority.json"]==40,"JSON authority never cached")
  before=canon_calls[0]
  for _ in range(5):v4.canonical(root,"src/a.py")
  need(canon_calls[0]==before+5,"Fingerprint/global canonical remains uncached")
  tx.finish();need(all(reads[p]==2 for p in allow),"Each unique code source revalidated uncached")
  need(not tx.enabled,"Cache disabled before append")
  v4.bound(root,refs["src/a.py"]);need(reads["src/a.py"]==3,"Append-time source read uncached")
  good_counts=dict(tx.counts)
 finally:
  tx.close()
  need(not tx.sources and not tx.names and tx.bytes==0,"Cache destroyed before synthetic unlock")
  locked[0]=False
 need(all(getattr(v4,n)is fn for n,fn in original_objects.items())and budget.successor_paths is original_successor,"Originals restored")
 failures=[]
 for case in("source_drift","namespace_drift","operation_exception"):
  locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow);appended=False
  try:
   tx.install();v4.bound(root,refs["src/a.py"]);v4.scope_spec(root,own)
   if case=="source_drift":memory["src/a.py"]=b"changed source\n"
   if case=="namespace_drift":drift[0]=True
   if case=="operation_exception":raise ValueError("handwritten operation failure")
   tx.finish();appended=True
  except ValueError:failures.append(case)
  finally:
   tx.close();need(not tx.sources and not tx.names,"Exceptional cache teardown")
   need(all(getattr(v4,n)is fn for n,fn in original_objects.items()),"Exceptional originals restored")
   need(not appended,"Failure prevents append");memory["src/a.py"]=b"source a\n";drift[0]=False;locked[0]=False
 # New mutex generation has cold source cache; no result persists across transactions.
 locked[0]=True;before=reads["scripts/b.sh"];tx=ns["_Transaction"](root,v4,budget,allow)
 try:tx.install();v4.bound(root,refs["scripts/b.sh"]);need(reads["scripts/b.sh"]==before+1,"New generation reads again");tx.finish()
 finally:tx.close();locked[0]=False
 # Public state API: synthetic original mutex really encloses patching and teardown.
 actual_root=ns["ROOT"];actual_source_reader=ns["source_bytes"];events=[]
 @contextmanager
 def fake_mutex(r):
  need(not locked[0],"No nested mutex");locked[0]=True;events.append("locked")
  try:yield
  finally:
   need(v4.bound is original_objects["bound"]and v4.scope_spec is original_objects["scope_spec"]
    and budget.successor_paths is original_successor,"Restore before original unlock")
   locked[0]=False;events.append("unlocked")
 v4.mutex=fake_mutex;ns["ROOT"]=root;ns["source_bytes"]=lambda record:b"handwritten exact memo source"
 fake_guard=SimpleNamespace(budget=budget)
 def fake_state(*args,**kwargs):
  need(locked[0]and kwargs==dict(fresh=True),"Actual public state lock/fresh flag")
  for _ in range(5):v4.bound(root,refs["src/a.py"])
  return("handwritten_state",)
 fake_guard._state_locked=fake_state;fake_guard._append=lambda *a:None
 memo=ns["RootMutationMemo"].__new__(ns["RootMutationMemo"])
 memo.guard=fake_guard;memo.budget=budget;memo.v4=v4;memo.allow=allow;memo.adapter_ref=refs["src/a.py"];memo._check=lambda:None
 try:
  need(memo.read_state_v1(expected_source_sha256="a",expected_launcher_sha256="b")==("handwritten_state",),"Public state result unchanged")
  def fail_state(*args,**kwargs):v4.bound(root,refs["src/a.py"]);raise ValueError("synthetic state failure")
  fake_guard._state_locked=fail_state
  try:memo.read_state_v1(expected_source_sha256="a",expected_launcher_sha256="b")
  except ValueError:pass
  else:raise AssertionError("Public exceptional state")
  need(events==["locked","unlocked","locked","unlocked"]and not ns["_ACTIVE"],"Every public exit releases and clears generation")
 finally:ns["ROOT"]=actual_root;ns["source_bytes"]=actual_source_reader;v4.mutex=original_objects["mutex"]

 # Exact original critical body, with only the mutex moved to the public adapter.
 reserve=next(n for n in ast.parse(originals["guard"]).body if isinstance(n,ast.FunctionDef)and n.name=="apply_reserve_v1")
 fake_scope=dict(__name__="handwritten_guard")
 derived=ns["critical_reserve"](reserve,fake_scope,lambda *a:None)
 need(derived.__code__.co_filename.endswith("#exact-reserve-critical-body"),"Explicit derivative identity")
 # Confirm public lock/teardown/append ordering statically, without guard imports or mutex probes.
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="RootMutationMemo")
 run=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=="_run")
 calls=sorted((n.lineno,n.col_offset,ast.unparse(n.func))for n in ast.walk(run)if isinstance(n,ast.Call))
 names=[x[2]for x in calls]
 need(names.index("original_mutex")<names.index("_Transaction")<names.index("tx.install"),"Memo starts after actual original lock")
 need("tx.close"in names,"Always teardown inside original lock")
 append=next(n for n in ast.walk(run)if isinstance(n,ast.FunctionDef)and n.name=="append")
 anames=[x[2]for x in sorted((n.lineno,n.col_offset,ast.unparse(n.func))for n in ast.walk(append)if isinstance(n,ast.Call))]
 need(anames.index("tx.finish")<anames.index("original_append"),"Uncached final validation before original append")
 need('Path(record["path"]).suffix in(".py",".sh")'in source.decode(),"JSON excluded at public API")
 return dict(schema="ROOT_mutation_transaction_memo_source_checks_v1",status="SOURCE_SYNTHETIC_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  original_code_source_pins_authenticated=3,actual_project_modules_imported=False,actual_mutex_or_guard_called=False,
  actual_payload_resource_process_job_or_accounting_operations=False,numerical_work=False,
  actual_v4_schema_and_bound_code_exercised_in_memory=True,actual_budget_successor_path_code_exercised_in_memory=True,
  unchanged_original_primitive_objects=True,uncached_JSON_and_global_canonical=True,
  cache_generation_destroyed_before_unlock=True,uncached_preappend_revalidation=True,postvalidation_reads_uncached=True,
  exceptional_cases=failures,public_original_mutex_enter_and_restore_before_unlock=True,
  code_source_cold_each_transaction=True,synthetic_counts=good_counts,
  synthetic_representative_source_reads=dict(path="scripts/b.sh",original_repeated=20,memo_before_validation=1,uncached_final_validation=1),
  real_performance_gain_measured=False,reserve_only=True,caller_held_mutex_and_retirement_supported=False)
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True,indent=2))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
