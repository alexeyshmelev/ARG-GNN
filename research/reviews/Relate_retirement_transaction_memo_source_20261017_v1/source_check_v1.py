"""Pure source/AST and handwritten memory checks; no live guard, mutex, process or payload."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,json,sys,traceback
HERE=Path("research/reviews/Relate_retirement_transaction_memo_source_20261017_v1")
def check(ok,message):
 if not ok:raise AssertionError(message)
def value_need(ok,message):
 if not ok:raise ValueError(message)
def main():
 raw=(HERE/"retire_transaction_memo_v1.py").read_bytes();tree=ast.parse(raw);compile(tree,str(HERE/"retire_transaction_memo_v1.py"),"exec")
 ns={"__name__":"source_only_retirement_memory"};exec(compile(tree,"source_only_retirement_memory","exec"),ns)
 memo_raw=Path(ns["MEMO_REF"]["path"]).read_bytes()
 check(hashlib.sha256(memo_raw).hexdigest()==ns["MEMO_REF"]["sha256"],"Accepted memo SHA")
 mn={"__name__":"source_only_accepted_memo_memory"};exec(compile(memo_raw,"accepted_memo_code_in_memory","exec"),mn)
 guard_raw=Path(ns["GUARD_REF"]["path"]).read_bytes()
 check(hashlib.sha256(guard_raw).hexdigest()==ns["GUARD_REF"]["sha256"],"Original guard SHA")
 retire=next(x for x in ast.parse(guard_raw).body if isinstance(x,ast.FunctionDef)and x.name=="apply_retire_stopped_v1")
 v4_raw=Path(mn["PINS"]["v4"]["path"]).read_bytes();budget_raw=Path(mn["PINS"]["budget"]["path"]).read_bytes()
 check(hashlib.sha256(v4_raw).hexdigest()==mn["PINS"]["v4"]["sha256"]and hashlib.sha256(budget_raw).hexdigest()==mn["PINS"]["budget"]["sha256"],"Original primitive source SHAs")
 # Reconstruct the derivative in memory and reverse ONLY the two declared additions.
 derived=ns["derive_retirement"](retire,{},lambda f:None,lambda *a:None)
 check(derived.__code__.co_filename.endswith("#entire-original-retirement-with-two-additions"),"Explicit entire derivative identity")
 d=copy.deepcopy(retire);d.name="_memo_exact_retirement"
 calls=[x for x in ast.walk(d.body[-1])if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=="_append"]
 calls[0].func.id="_retirement_append";calls[0].args[0:0]=[ast.Name(id="finite",ctx=ast.Load()),ast.Name(id="auth",ctx=ast.Load())]
 body=d.body[-1].body
 d.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="_retirement_transaction",ctx=ast.Load()),args=[ast.Name(id="finite",ctx=ast.Load())],keywords=[]),optional_vars=None)],body=body)]
 manual={}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(d)],type_ignores=[])),derived.__code__.co_filename,"exec"),manual)
 check(manual["_memo_exact_retirement"].__code__==derived.__code__,"Actual derivative matches only declared transformations")
 restored=copy.deepcopy(d);restored.name=retire.name;restored.body[-1].body=restored.body[-1].body[0].body
 call=next(x for x in ast.walk(restored)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=="_retirement_append")
 call.func.id="_append";check([call.args.pop(0).id,call.args.pop(0).id]==["finite","auth"],"Original finite/auth callback");restored.decorator_list=retire.decorator_list
 check(ast.dump(restored,include_attributes=False)==ast.dump(retire,include_attributes=False),"Entire original AST restored exactly")
 class MemoryPath(PurePosixPath):
  def exists(self):return False
 root=MemoryPath("/HANDWRITTEN_NO_FILES");ns["ROOT"]=root
 TARGET=ns["TARGET_KEY"];OP=ns["RETIRE_OPERATION"];reports=[]
 for case in("success","source_drift","namespace_drift","historical_JSON_drift","JSON_drift","operation_exception","expired_after_final_revalidation","authority_expired_during_finish","ROOT_instruction_expired_during_finish","expired_while_waiting_for_mutex","wrong_key"):
  events=[];clock=[0.];locked=[False];drift=[False];reads={};canon=[0];late_expiry=[False];transactions=[];parses={}
  memory={"src/a.py":b"immutable source\n","scripts/b.sh":b"immutable script\n",
   "src/arggnn/project_budget_v9_consumer.py":b"synthetic marker source\n",
   "research/authority.json":b'{"kind":"original-authority"}',"research/revocation.json":b'{"revoked":true}',
   "research/observation.json":b'{"absent":true}',"research/historical_guard_config.json":b'{"immutable":true}'}
  def ref(path):
   return dict(path=path,sha256=hashlib.sha256(memory[path]).hexdigest())
  code_refs=[ref("src/a.py"),ref("scripts/b.sh"),ref("src/arggnn/project_budget_v9_consumer.py")];allow={r["path"]:r["sha256"]for r in code_refs}
  proof_refs=[ref("research/revocation.json"),ref("research/observation.json")]
  request=dict(key="wrong/key"if case=="wrong_key"else TARGET,revocation_ref=proof_refs[0],observation_ref=proof_refs[1])
  memory["research/authority.json"]=json.dumps(dict(request=request)).encode()
  authority_ref=ref("research/authority.json");historical_ref=ref("research/historical_guard_config.json")
  def relative(p):
   p=MemoryPath(p);return str(p.relative_to(root))if p.is_absolute()else str(p)
  def canonical(r,p):
   check(locked[0],"Only synthetic lock");canon[0]+=1;p=MemoryPath(p);p=p if p.is_absolute()else r/p
   return p/"DRIFT"if drift[0]else p
  def file_bytes(r,p):
   check(locked[0],"Only synthetic lock");key=relative(p);reads[key]=reads.get(key,0)+1;data=memory[key]
   if key=="src/a.py"and reads[key]==2:
    if case=="expired_after_final_revalidation":clock[0]=300.;events.append("deadline_expired_during_uncached_finish")
    if case in("authority_expired_during_finish","ROOT_instruction_expired_during_finish"):late_expiry[0]=True;events.append("UTC_expired_during_uncached_finish")
   return data,hashlib.sha256(data).hexdigest()
  def parse(data):
   sha=hashlib.sha256(data).hexdigest();parses[sha]=parses.get(sha,0)+1;return json.loads(data)
  v4=ModuleType("handwritten_v4");v4.__dict__.update(Path=MemoryPath,canonical=canonical,file_bytes=file_bytes,
   parse=parse,integer=lambda v,*a:value_need(type(v)is int and v>=1,"integer"),accounting_paths=lambda r:[],
   paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,encoded=lambda v:json.dumps(v,sort_keys=True).encode(),LEGACY=".runtime/legacy.json")
  nodes=[n for n in ast.parse(v4_raw).body if isinstance(n,ast.FunctionDef)and n.name in{"bound","reference","scope_spec","read_json"}]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),"original_v4_predicates_handwritten_memory","exec"),v4.__dict__)
  budget=ModuleType("handwritten_budget");budget.__dict__.update(v4=v4,JOURNAL_REL=".runtime/journal",digest=lambda v:"synthetic_digest")
  successor=next(n for n in ast.parse(budget_raw).body if isinstance(n,ast.FunctionDef)and n.name=="successor_paths")
  exec(compile(ast.Module(body=[successor],type_ignores=[]),"original_successor_paths_handwritten_memory","exec"),budget.__dict__)
  originals={n:getattr(v4,n)for n in("canonical","file_bytes","bound","reference","scope_spec")};original_successor=budget.successor_paths
  @contextmanager
  def mutex(r):
   check(not locked[0],"No caller-held synthetic lock");locked[0]=True;events.append("mutex_enter")
   if case=="expired_while_waiting_for_mutex":clock[0]=300.
   try:yield
   finally:
    check(all(getattr(v4,n)is value for n,value in originals.items())and budget.successor_paths is original_successor,"Restored before unlock")
    check(all(not tx.sources and not tx.names and not tx.uncached and tx.bytes==0 and not tx.enabled for tx in transactions),
     "Every byte/path/exclusion cache destroyed BEFORE unlock")
    events.append("restore_before_mutex_exit");locked[0]=False
  v4.mutex=mutex
  own=dict(key=TARGET,output=str(root/"data/original_failed"),auxiliary_scopes=[str(root/"research/original_aux")],
   reserved_bytes=64*2**30,consumer_bindings=code_refs+proof_refs+[historical_ref])
  view=dict(new_reserved_bytes=64*2**30,completed_credit_bytes=0)
  state=({TARGET:own},set(),view,dict(code=code_refs[0]),[],[],None)
  request=dict(key="wrong/key"if case=="wrong_key"else TARGET,revocation_ref=proof_refs[0],observation_ref=proof_refs[1])
  auth=dict(expected_before=dict(prefix=0),request=request)
  def state_locked(*a,**kw):
   check(kw==dict(fresh=True),"Original fresh replay")
   for _ in range(8):budget.successor_paths(root,own);v4.reference(root,"src/a.py");v4.bound(root,historical_ref,True)
   return state
  def proof(*a,**kw):
   events.append("fresh_target_absence"if kw.get("observe")else"uncached_proof_replay")
   for record in proof_refs:v4.bound(root,record,True)
   v4.canonical(root,"fingerprint");return dict(pre_measurement_targets={},post_measurement_targets={})
  helper=SimpleNamespace(require=value_need)
  def measure(*a,**kw):
   events.extend(["first_stat_pass","second_stat_pass","final_census"])
   for _ in range(3):helper.require(True,"bounded original census")
   if case=="operation_exception":raise ValueError("synthetic operation failure")
   return dict(retained=123)
  helper.measure_retained_nodes_v1=measure
  def retained(*a):
   if case=="source_drift":memory["src/a.py"]=b"CHANGED"
   if case=="namespace_drift":drift[0]=True
   if case=="JSON_drift":memory["research/observation.json"]=b'{"absent":false}'
   if case=="historical_JSON_drift":memory["research/historical_guard_config.json"]=b'{"immutable":false}'
   return 123
  loader=SimpleNamespace(exec_module=lambda m:None)
  importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *a:SimpleNamespace(loader=loader),module_from_spec=lambda s:helper))
  obj=ns["RetireStoppedMemo"].__new__(ns["RetireStoppedMemo"])
  def transaction(*args,**kwargs):
   tx=mn["_Transaction"](*args,**kwargs);transactions.append(tx);return tx
  obj.memo=SimpleNamespace(_Transaction=transaction);obj.v4=v4;obj.budget=budget;obj.core=SimpleNamespace(allow=allow)
  obj._tx=None;obj.counts={};obj._append_calls=0;obj._authority_ref=authority_ref
  obj.current_refs=[];obj.manifest_ref=obj.review_ref=obj.memo_review_ref=obj.instruction_ref=proof_refs[0]
  instruction=dict(source_only_ROOT_instruction=True)
  def expire(value):
   events.append("original_authority_expiry"if value is auth else"ROOT_instruction_expiry")
   if late_expiry[0]and((case=="authority_expired_during_finish"and value is auth)
    or(case=="ROOT_instruction_expired_during_finish"and value is instruction)):raise ValueError("synthetic UTC expiry")
  obj.guard=SimpleNamespace(_expire=expire)
  def authenticate(bound):
   events.append("uncached_ROOT_manifest_review_instruction")
   for record in proof_refs:bound(root,record,True)
   return instruction
  obj._authenticate=authenticate
  def append(operation,record,value,s):
   check(locked[0]and obj._tx is not None and not obj._tx.enabled,"Caching disabled before original append")
   check(v4.canonical is canonical and v4.file_bytes is file_bytes and v4.mutex is mutex,"Original primitives untouched")
   v4.bound(root,code_refs[0]);events.append("original_append");return"synthetic_tail"
  obj._original_append=append
  scope=dict(__name__="handwritten_retirement_guard",time=SimpleNamespace(monotonic=lambda:clock[0]),require=value_need,
   OPERATION_WALL_SECONDS=300,budget=budget,ROOT=root,_state_locked=state_locked,_authority=lambda *a,**k:auth,_pins=lambda v:dict(prefix=0),
   RETIRE_OPERATION=OP,TARGET_KEYS=(TARGET,),assert_open=lambda *a:None,
   consumer_bindings_v1=lambda *groups:[v4.bound(root,r)for group in groups for r in group],_stopped_proof=proof,
   importlib=importlib,MEASURE_SOURCE="source_only_fake_measure.py",_paths=lambda v:[MemoryPath(v["output"]),MemoryPath(v["auxiliary_scopes"][0])],
   _target=lambda key:dict(scope_paths=[own["output"],own["auxiliary_scopes"][0]]),_retained_measurement=retained,
   _observe_targets=lambda t:events.append("post_measurement_target_absence")or{},_utc=lambda:"synthetic_closed_utc",
   _recorded_stopped_proof=lambda *a:events.append("uncached_recorded_fingerprint"),copy=copy,charge=lambda *a:999,
   _policy=lambda:dict(base_projected_bytes=0),_expire=expire,
   _append=append,warning_fields=lambda p:dict(no_real_capacity=True))
  fn=ns["derive_retirement"](retire,scope,obj._transaction,obj._append)
  failed=False
  try:result=fn(authority_ref,expected_source_sha256="source_only",expected_launcher_sha256="source_only")
  except ValueError as exc:
   failed=True
   if case=="success":raise AssertionError("Unexpected handwritten success failure: "+str(exc))from exc
  check(failed==(case!="success"),"Expected synthetic outcome "+case)
  check(not locked[0]and obj._tx is None,"Teardown complete")
  check(events[-1]=="restore_before_mutex_exit","Every exit restores before actual synthetic unlock")
  if case=="success":
   check(events.index("uncached_ROOT_manifest_review_instruction")<events.index("original_append"),"Uncached validation before append")
   check(reads["src/a.py"]==3 and reads["research/observation.json"]>=4,"Code final+append reads and current JSON uncached")
   check(reads["research/historical_guard_config.json"]==2 and parses[historical_ref["sha256"]]>=8,
    "Historical exact-bound bytes memoize, parsing reruns")
  else:check("original_append"not in events,"Failure prevents append "+case)
  reports.append(dict(case=case,expected_failure=failed,append_count=events.count("original_append"),restored_before_unlock=True,
   deadline_before_mutex_preserved=True,reads=reads,events=events))
 return dict(schema="Relate_retirement_transaction_source_checks_v1",source_only=True,checks=reports,
  entire_original_AST_restored_after_removing_only_two_declared_additions=True,
  original_prelock_started_deadline_and_finite_checks_preserved=True,original_actual_mutex_context_retained=True,
  expired_deadline_during_final_revalidation_blocks_original_append=True,
  original_authority_and_ROOT_instruction_UTC_expiry_after_finish_block_append=True,original_canonical_file_bytes_mutex_objects_unchanged=True,
  current_authority_proof_revocation_observation_raw_journal_fingerprints_measurement_and_targets_not_cached=True,
  accepted_transaction_clone_and_revalidation_logic_reused=True,restore_and_cache_destroy_before_unlock=True,
  historical_exact_bound_metadata_bytes_cache_but_parse_and_predicates_rerun=True,
  no_actual_project_module_imports_or_guard_mutex_state_resource_process_payload_job_accounting_calls=True,
  actual300second_fit_credit_capacity_or_performance_gain_claimed=False)
if __name__=="__main__":
 try:print(json.dumps(main(),sort_keys=True,indent=2))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
