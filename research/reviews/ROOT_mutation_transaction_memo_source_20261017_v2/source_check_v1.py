"""Stdlib source/AST plus handwritten memory checks; never imports project guards."""
import ast,copy,hashlib,json,pathlib,sys,traceback
from contextlib import contextmanager
from types import ModuleType,SimpleNamespace
HERE=pathlib.Path("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2")
def need(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 source=(HERE/"root_mutation_memo_v1.py").read_bytes();tree=ast.parse(source);compile(tree,str(HERE/"root_mutation_memo_v1.py"),"exec")
 ns={"__name__":"inert_memo_source_test"};exec(compile(tree,str(HERE/"root_mutation_memo_v1.py"),"exec"),ns)
 # Reverse only transaction implementation/source identity/current-authority exclusion.
 old_tree=ast.parse(pathlib.Path("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py").read_bytes())
 new_tree=copy.deepcopy(tree)
 new_tree.body=[n for n in new_tree.body if not(isinstance(n,ast.ClassDef)and n.name=="_Transaction")]
 old_tree.body=[n for n in old_tree.body if not(isinstance(n,ast.ClassDef)and n.name=="_Transaction")]
 for n in new_tree.body:
  if isinstance(n,ast.Expr)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str):n.value.value=old_tree.body[0].value.value
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="HERE"for t in n.targets):n.value.value="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1"
 for n in ast.walk(new_tree):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_Transaction":
   n.keywords=[k for k in n.keywords if k.arg!="uncached_paths"]
 need(ast.dump(new_tree,include_attributes=False)==ast.dump(old_tree,include_attributes=False),"V1 adapter identity/lock/critical-body code unchanged except current-authority exclusion")
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
 tx=ns["_Transaction"](root,v4,budget,allow,uncached_paths=("research/reviews/authority.json",));locked[0]=True
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
  locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow,uncached_paths=("research/reviews/authority.json",));appended=False
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
 locked[0]=True;before=reads["scripts/b.sh"];tx=ns["_Transaction"](root,v4,budget,allow,uncached_paths=("research/reviews/authority.json",))
 try:tx.install();v4.bound(root,refs["scripts/b.sh"]);need(reads["scripts/b.sh"]==before+1,"New generation reads again");tx.finish()
 finally:tx.close();locked[0]=False
 # Historical SHA-bound JSON and direct _cpu_reader source checks.
 from datetime import datetime as RealDatetime,timezone,timedelta
 historical={"data/expanded_full_panel_pyg_cache_v5/cache/manifest.json":b'{"schema":"completed_manifest","items":[1,2,3]}',
  "research/completed_proof.json":b'{"completed":true,"updates":1200}',
  "research/historical_config.json":b'{"CPU_threads":32}',
  "research/prefix_snapshot.json":b'{"prefix":1}',
  ".runtime/storage/journal/000001.json":b'{"sequence":1}',
  "data/scientific_payload.npz":b"never cache numerical payload",
  "research/raw_metadata.json":b'{"raw":1}'}
 memory.update(historical)
 history_refs={p:dict(path=p,sha256=hashlib.sha256(raw).hexdigest())for p,raw in historical.items()}
 terminal_path="src/arggnn/project_storage_terminal_scopes_v1.py"
 cpu_path="src/arggnn/project_storage_cpu_closed_scopes_v1.py"
 previous_path="src/arggnn/project_storage_closed_scopes_v1.py"
 reader_sources={p:pathlib.Path(p).read_bytes()for p in(terminal_path,cpu_path,previous_path)}
 reader_pins={p:hashlib.sha256(raw).hexdigest()for p,raw in reader_sources.items()}
 need(reader_pins=={
  terminal_path:"f06cf7028af7e86deb54caa1e8a01f0110db47b0d992bfe8aec8ee013ab7d79c",
  cpu_path:"2a33a19c1447442fc99708bf3d73c21e1720a2453a99deb71d1688a4dcc5c6ff",
  previous_path:"974bcf342965abb7e7e062547c68123de9bb9414bf9b1f97fd962d1488d3fdd7"},"Exact direct CPU reader source pins")
 memory.update({p:raw for p,raw in reader_sources.items()if p!=terminal_path})
 cpu_node=next(n for n in ast.parse(reader_sources[terminal_path]).body if isinstance(n,ast.FunctionDef)and n.name=="_cpu_reader")
 sentinel={"handwritten_existing_cpu_namespace":True}
 cpu_scope=dict(budget=budget,ROOT=root,CPU_SOURCE=cpu_path,CPU_SHA256=reader_pins[cpu_path],
  previous=SimpleNamespace(SOURCE=previous_path),PREVIOUS_SHA256=reader_pins[previous_path],_CPU=sentinel)
 exec(compile(ast.Module(body=[cpu_node],type_ignores=[]),"exact_cpu_reader_AST_existing_memory_namespace","exec"),cpu_scope)
 locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow,uncached_paths=("research/reviews/authority.json",))
 baseline=dict(reads);parse_count=[0];original_parse=v4.parse
 def parse_counted(raw):parse_count[0]+=1;return json.loads(raw)
 v4.parse=parse_counted
 try:
  tx.install()
  for _ in range(20):
   parsed=v4.bound(root,history_refs["data/expanded_full_panel_pyg_cache_v5/cache/manifest.json"],True)
   parsed["items"].append(999)
   need(v4.bound(root,history_refs["data/expanded_full_panel_pyg_cache_v5/cache/manifest.json"],True)["items"]==[1,2,3],"Parse creates fresh JSON object every call")
   for p in("research/completed_proof.json","research/historical_config.json"):v4.bound(root,history_refs[p],True)
   for p in("research/prefix_snapshot.json",".runtime/storage/journal/000001.json","data/scientific_payload.npz"):v4.bound(root,history_refs[p])
   v4.bound(root,refs["research/reviews/authority.json"],True)
   need(cpu_scope["_cpu_reader"]()is sentinel,"Exact direct CPU reader returns original namespace")
  for p in("data/expanded_full_panel_pyg_cache_v5/cache/manifest.json","research/completed_proof.json","research/historical_config.json",cpu_path,previous_path):
   need(reads[p]-baseline.get(p,0)==1,"Dynamic exact bound enrolment reads once "+p)
  for p in("research/prefix_snapshot.json",".runtime/storage/journal/000001.json","data/scientific_payload.npz"):
   need(reads[p]-baseline.get(p,0)==20 and p not in tx.sources,"Forbidden observations/prefix/payload uncached "+p)
  need(reads["research/reviews/authority.json"]-baseline["research/reviews/authority.json"]==20,"Explicit current authority uncached")
  need(parse_count[0]==100,"Every original JSON parse executes, including every cache hit")
  cached_json="data/expanded_full_panel_pyg_cache_v5/cache/manifest.json";before=reads[cached_json]
  for _ in range(5):v4.file_bytes(root,cached_json);v4.read_json(root,cached_json);v4.reference(root,cached_json)
  need(reads[cached_json]==before+15,"All raw JSON/read_json/reference without expected SHA remain uncached")
  need(all(getattr(v4,n)is original_objects[n]for n in("canonical","file_bytes","mutex")),"Original primitives remain untouched")
  tx.finish()
  for p in("research/completed_proof.json","research/historical_config.json",cpu_path,previous_path):
   need(reads[p]-baseline.get(p,0)==2,"Original uncached unique-ref final validation "+p)
  historical_counts=dict(tx.counts)
 finally:
  tx.close();v4.parse=original_parse;locked[0]=False
 need(not tx.sources and not tx.names and not tx.uncached,"Historical cache destroyed")
 metadata_failures=[]
 for case in("config_drift","proof_drift","expected_sha_conflict","namespace_drift","current_excluded_drift"):
  locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow,uncached_paths=("research/reviews/authority.json",));saved=dict(memory);returned=False
  try:
   tx.install();v4.bound(root,history_refs["research/historical_config.json"],True);v4.bound(root,history_refs["research/completed_proof.json"],True);v4.scope_spec(root,own)
   if case=="config_drift":memory["research/historical_config.json"]=b'{"CPU_threads":99}'
   if case=="proof_drift":memory["research/completed_proof.json"]=b'{"completed":false,"updates":0}'
   if case=="namespace_drift":drift[0]=True
   if case=="expected_sha_conflict":
    path="research/completed_proof.json";memory[path]=b'{"completed":false}'
    v4.bound(root,dict(path=path,sha256=hashlib.sha256(memory[path]).hexdigest()),True)
   if case=="current_excluded_drift":
    memory["research/reviews/authority.json"]=b'{"fresh":2}'
    v4.bound(root,refs["research/reviews/authority.json"],True)
   tx.finish();returned=True
  except ValueError:metadata_failures.append(case)
  finally:
   tx.close();memory.clear();memory.update(saved);drift[0]=False;locked[0]=False
   need(not returned and not tx.sources and not tx.names,"Content/proof/current/namespace drift blocks return and restores")
 need(len(metadata_failures)==5,"All metadata drift cases")
 # Original JSON parse failure never enrols bytes, so another attempt reads uncached.
 invalid_path="research/invalid_json.json";memory[invalid_path]=b"{"
 invalid_ref=dict(path=invalid_path,sha256=hashlib.sha256(memory[invalid_path]).hexdigest())
 locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow);before=reads.get(invalid_path,0)
 try:
  tx.install()
  for _ in range(2):
   try:v4.bound(root,invalid_ref,True)
   except (ValueError,json.JSONDecodeError):pass
   else:raise AssertionError("Invalid JSON original parse must fail")
  need(invalid_path not in tx.sources and reads[invalid_path]==before+2,"Only successful original bound/parse may enrol JSON")
  tx.finish()
 finally:tx.close();locked[0]=False
 # Exact original successor authority time predicate executes again on cached JSON bytes.
 clock=[RealDatetime(2026,10,17,tzinfo=timezone.utc)]
 class Clock:
  fromisoformat=staticmethod(RealDatetime.fromisoformat)
  @staticmethod
  def now(zone):return clock[0]
 authority_path="research/historical_time_authority.json";mode="journal_terminal_success_scopes_credit_v1"
 authority=dict(kind="project_budget_terminal_success_root_admission_v1",root_authorizer="root_agent",active=True,
  operation="reserve",accounting_mode=mode,bindings={},all_budget_writers_use_scheduler_mutex=True,
  all_consumers_and_launchers_use_terminal_success_guard=True,all_terminal_v1_writers_finished_before_successor_activation=True,
  historical_records_mutation_authorized=False,global_rebase_authorized=False,scientific_execution_authorized=False,GPU_authorized=False,
  reservation=dict(consumer_bindings=[]),expires_at_utc=(clock[0]+timedelta(seconds=10)).isoformat())
 memory[authority_path]=json.dumps(authority).encode();time_ref=dict(path=authority_path,sha256=hashlib.sha256(memory[authority_path]).hexdigest())
 success_path=pathlib.Path("src/arggnn/project_storage_terminal_success_scopes_v1.py");success_source=success_path.read_bytes()
 need(hashlib.sha256(success_source).hexdigest()=="2dfbcc2711fc8310fd8a8b9866ee84e9107e1d2c4356396e0ebc0e99fdb7c3b3","Exact authority predicate source")
 authority_node=next(n for n in ast.parse(success_source).body if isinstance(n,ast.FunctionDef)and n.name=="_authority")
 authority_scope=dict(budget=budget,_bound=lambda record:v4.bound(root,record,True),MODE=mode,datetime=Clock,timezone=timezone)
 exec(compile(ast.Module(body=[authority_node],type_ignores=[]),"exact_original_authority_time_predicate","exec"),authority_scope)
 locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow)
 try:
  tx.install();authority_scope["_authority"](time_ref,"reserve",{},fresh=True)
  clock[0]+=timedelta(seconds=11)
  try:authority_scope["_authority"](time_ref,"reserve",{},fresh=True)
  except ValueError:pass
  else:raise AssertionError("Original current-time predicate skipped on JSON cache hit")
  need(reads[authority_path]==1 and tx.counts["json_hits"]>=1,"Time predicate rerun on an actual JSON byte hit")
  tx.finish();need(reads[authority_path]==2,"Time-test JSON final uncached verification")
 finally:tx.close();locked[0]=False
 # Memory ceilings skip caching excess unique bytes instead of growing.
 locked[0]=True;tx=ns["_Transaction"](root,v4,budget,allow,max_entries=1,max_bytes=4096)
 try:
  tx.install()
  for p in("research/completed_proof.json","research/historical_config.json"):v4.bound(root,history_refs[p],True)
  need(len(tx.sources)==1 and tx.bytes<=4096,"Bounded byte/ref cache")
  tx.finish()
 finally:tx.close();locked[0]=False
 metadata_results=dict(historical_counts=historical_counts,exact_cpu_reader_sources=reader_pins,
  cached_JSON_parsed_each_call=True,changed_config_proof_expected_SHA_namespace_current_refs_rejected=metadata_failures,
  actual_original_authority_time_predicate_rerun_on_JSON_hit=True,raw_JSON_reference_journal_prefix_payload_uncached=True,
  original_unique_ref_final_revalidation=True,current_operation_uncached_paths=True,memory_limits_enforced=True)
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
 need('path.suffix in(".py",".sh",".json")'in source.decode(),"Only exact-bound dynamic JSON eligibility")
 need(not any(n.startswith(("arggnn","scripts.project_budget","_child_source_repair","_H058_fresh72h"))for n in sys.modules),"No project/guard/science imports")
 return dict(exact_v1_adapter_delta=True,invalid_JSON_never_enrolled=True,schema="ROOT_mutation_transaction_memo_source_checks_v2",metadata_results=metadata_results,public_reserve_cases=public_reserve_checks(ns,originals),status="SOURCE_SYNTHETIC_CHECKS_PASS_NOT_INDEPENDENT_REVIEW",
  original_code_source_pins_authenticated=3,actual_project_modules_imported=False,actual_mutex_or_guard_called=False,
  actual_payload_resource_process_job_or_accounting_operations=False,numerical_work=False,
  actual_v4_schema_and_bound_code_exercised_in_memory=True,actual_budget_successor_path_code_exercised_in_memory=True,
  unchanged_original_primitive_objects=True,uncached_current_JSON_and_global_canonical=True,
  cache_generation_destroyed_before_unlock=True,uncached_preappend_revalidation=True,postvalidation_reads_uncached=True,
  exceptional_cases=failures,public_original_mutex_enter_and_restore_before_unlock=True,
  code_source_cold_each_transaction=True,synthetic_counts=good_counts,
  synthetic_representative_source_reads=dict(path="scripts/b.sh",original_repeated=20,memo_before_validation=1,uncached_final_validation=1),
  real_performance_gain_measured=False,reserve_only=True,caller_held_mutex_and_retirement_supported=False)
def public_reserve_checks(ns,sources):
 def need(ok,msg):
  if not ok:raise ValueError(msg)
 saved_root=ns["ROOT"];saved_source=ns["source_bytes"]
 class MemoryPath(pathlib.PurePosixPath):
  def exists(self):return False
 root=MemoryPath("/source_fixture/never_materialized");ns["ROOT"]=root;ns["source_bytes"]=lambda record:b"memo fixture"
 original_reserve=next(n for n in ast.parse(sources["guard"]).body if isinstance(n,ast.FunctionDef)and n.name=="apply_reserve_v1")
 results=[]
 for case in["success","source_drift","proof_drift","namespace_drift","manifest_drift","missing_direct_manifest","operation_exception","append_exception","cold_success"]:
  locked=[False];events=[];reads={};drift=[False];appended=[0]
  memory={"src/a.py":b"pinned code a\n","src/arggnn/project_budget_v9_consumer.py":b"marker\n","research/guard.py":b"guard\n","research/integration.json":json.dumps({"source_sha256":{ns["SOURCE"]:"f"*64}},sort_keys=True).encode(),"research/authority.json":b"{}","research/completed_proof.json":b'{"completed":true}',".runtime/legacy.json":b"{\"jobs\":{}}"}
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
   for _ in range(10):v4.bound(root,refs["src/a.py"]);v4.bound(root,refs["research/authority.json"],True);v4.bound(root,refs["research/completed_proof.json"],True)
   return state
  def authority(*args,**kwargs):return dict(expected_before=view["prefix"],request=own)
  def charge(*args):
   if case=="source_drift":memory["src/a.py"]=b"drift\n"
   if case=="proof_drift":memory["research/completed_proof.json"]=b'{"completed":false}'
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
  need(reads["research/authority.json"]==10,"Current authority JSON was cached")
  if success:need(reads["src/a.py"]==3,"Source not1+uncachedfinal+uncachedappend");need(result["value"]["root_admission"]==refs["research/authority.json"],"Original receipt drift")
  results.append(dict(case=case,expected_success=success,error=error,original_append_calls=appended[0],events=events,source_reads=reads["src/a.py"],uncached_JSON_reads=reads["research/authority.json"],memo_counts=memo.counts))
 ns["ROOT"]=saved_root;ns["source_bytes"]=saved_source
 return results
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True,indent=2))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
