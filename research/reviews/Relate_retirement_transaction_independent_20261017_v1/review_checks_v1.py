"""Independent retirement composition review. Only source AST and handwritten memory."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from contextlib import contextmanager
from datetime import datetime as RealDatetime,timezone,timedelta
import ast,builtins,copy,hashlib,json,sys,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE_DIR="research/reviews/Relate_retirement_transaction_memo_source_20261017_v1"
OWN="research/reviews/Relate_retirement_transaction_independent_20261017_v1"
MEMO_REVIEW="research/reviews/ROOT_mutation_transaction_memo_independent_20261017_v2/review.v1.json"
def need(ok,message):
 if not ok:raise AssertionError(message)
def demand(ok,message):
 if not ok:raise ValueError(message)
def ref(path):
 raw=(ROOT/path).read_bytes();return dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
def isolate(nodes,scope):
 exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<independent-isolated-AST>","exec"),scope);return scope
def function(raw,name):
 matches=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name==name]
 need(len(matches)==1,"one exact function "+name);return copy.deepcopy(matches[0])
def main():
 base=ROOT/SOURCE_DIR;raw=(base/"retire_transaction_memo_v1.py").read_bytes()
 need(hashlib.sha256(raw).hexdigest()=="1c8a2038980e05d8d2bdea865907c8e10c51b8ccb00fd766999c3631e5662d12","supplied sealed source")
 manifest=json.loads((base/"source_manifest.v1.json").read_bytes())
 need(ref(SOURCE_DIR+"/source_manifest.v1.json")["sha256"]=="43057730975c959b30339f4f482af360252bc382854b35bfffabbe1d9bfabd11","supplied manifest")
 need(ref(SOURCE_DIR+"/source_handoff.v1.json")["sha256"]=="f363ee3509c2856cfc27b16ed19fa4cb751dba86168f0a3c583e1c0f8bc922c1","supplied handoff")
 for path,sha in manifest["source_sha256"].items():
  data=(ROOT/path).read_bytes();need(hashlib.sha256(data).hexdigest()==sha,"every sealed pair "+path)
  if path.endswith(".py"):compile(data,path,"exec")
 tree=ast.parse(raw)
 for n in tree.body:
  if isinstance(n,ast.Assign):
   calls=[ast.unparse(x)for x in ast.walk(n)if isinstance(x,ast.Call)]
   need(all(x=="Path('/workspace-SR003.nfs2/ARG-GNN')"or x.startswith("dict(")for x in calls),"no top-level source/guard IO")
  else:need(isinstance(n,(ast.Import,ast.ImportFrom,ast.Expr,ast.FunctionDef,ast.ClassDef,ast.If)),"inert source shape")
  if isinstance(n,ast.If):need(ast.unparse(n.test)=="__name__ == '__main__'","only inactive main")
 h={"__name__":"independent_retirement_definitions"};exec(compile(tree,"<inert-helper-definitions>","exec"),h)
 h["threading"]=SimpleNamespace(active_count=lambda:1)
 memo_raw=(ROOT/h["MEMO_REF"]["path"]).read_bytes();need(hashlib.sha256(memo_raw).hexdigest()==h["MEMO_REF"]["sha256"],"exact memo2source")
 m={"__name__":"independent_memo_definitions"};exec(compile(memo_raw,"<inert-memo-definitions>","exec"),m)
 guard_raw=(ROOT/h["GUARD_REF"]["path"]).read_bytes();need(hashlib.sha256(guard_raw).hexdigest()==h["GUARD_REF"]["sha256"],"exact originalguard")
 original=function(guard_raw,"apply_retire_stopped_v1")
 captured=[]
 def capture(tree,filename,mode,**kwargs):
  if isinstance(tree,ast.Module):captured.append(copy.deepcopy(tree))
  return builtins.compile(tree,filename,mode,**kwargs)
 h["compile"]=capture
 derived=h["derive_retirement"](original,{},lambda finite:None,lambda *args:None)
 need(len(captured)==1,"capture actual derivative AST")
 actual=captured[0].body[0];restored=copy.deepcopy(actual)
 need(ast.dump(ast.Module(body=restored.body[:-1],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=original.body[:-1],type_ignores=[]),include_attributes=False),"all prelock started/deadline/finite statements exact")
 outer=restored.body[-1];need(ast.dump(outer.items[0].context_expr,include_attributes=False)==ast.dump(original.body[-1].items[0].context_expr,include_attributes=False),"same original actual mutex")
 need(len(outer.body)==1 and isinstance(outer.body[0],ast.With)and ast.unparse(outer.body[0].items[0].context_expr)=="_retirement_transaction(finite)","only transaction context addition")
 outer.body=outer.body[0].body
 calls=[n for n in ast.walk(restored)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_retirement_append"]
 need(len(calls)==1 and [ast.unparse(x)for x in calls[0].args[:2]]==["finite","auth"],"same exact original closure/auth into sole append")
 calls[0].func.id="_append";calls[0].args=calls[0].args[2:];restored.name=original.name
 need(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),"entire original AST restored exactly")
 h.pop("compile")
 append_node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=="RetireStoppedMemo")
 append_node=next(n for n in append_node.body if isinstance(n,ast.FunctionDef)and n.name=="_append")
 sequence=[ast.unparse(n.value.func)for n in append_node.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)]
 need(sequence[-4:]==["tx.finish","self.guard._expire","self.guard._expire","finite"],"finaluncached checks then originalexpiry andfinite")
 need(isinstance(append_node.body[-1],ast.Return)and ast.unparse(append_node.body[-1].value.func)=="self._original_append","originalappend immediately afterfinite")
 memo_review_raw=(ROOT/MEMO_REVIEW).read_bytes();memo_review=json.loads(memo_review_raw)
 need(hashlib.sha256(memo_review_raw).hexdigest()=="5f57cd4b46c3575901f660f47ab1c556aabdb2f30b8c15d7ad339b2cf0a29978","actual separate memo2review")
 need(memo_review["independent"]is True and memo_review["verdict"]=="SOURCE_PASS"and memo_review["source_ref"]==h["MEMO_REF"]and memo_review["source_manifest_ref"]==h["MEMO_MANIFEST_REF"]and not memo_review["blockers"],"separate accepted exactmemo2")
 v4raw=(ROOT/m["PINS"]["v4"]["path"]).read_bytes();budgetraw=(ROOT/m["PINS"]["budget"]["path"]).read_bytes()
 class MP(PurePosixPath):
  def exists(self):return False
 root=MP("/independent/no-files");h["ROOT"]=root;m["ROOT"]=root
 reports=[]
 cases=("success","success_near300","append_uncertainty","wait_300","finish_300","authority_expiry_finish","instruction_expiry_finish","historical_JSON_drift","current_observation_drift","measure_exception","wrong_key","wrong_memo_review","wrong_instruction_helper","blocked_helper_review")
 for case in cases:
  mono=[0.];utc=[0.];locked=[False];events=[];reads={};txs=[];append_count=[0]
  source_flags={"source_ok":True}
  memory={"src/fixture.py":b"immutable fixture code\n","src/arggnn/project_budget_v9_consumer.py":b"fixture consumer",
   "research/historical_completed.json":b'{"completed":true}',"research/current_revocation.json":b'{"revoked":true}',
   "research/current_observation.json":b'{"absent":true}'}
  def memory_ref(path):return dict(path=path,sha256=hashlib.sha256(memory[path]).hexdigest())
  code_refs=[memory_ref("src/fixture.py"),memory_ref("src/arggnn/project_budget_v9_consumer.py")]
  historical_ref=memory_ref("research/historical_completed.json");rev=memory_ref("research/current_revocation.json");obs=memory_ref("research/current_observation.json")
  helper_ref=ref(SOURCE_DIR+"/retire_transaction_memo_v1.py");manifest_ref=ref(SOURCE_DIR+"/source_manifest.v1.json")
  memory[manifest_ref["path"]]=(base/"source_manifest.v1.json").read_bytes()
  helper_review=dict(independent=True,verdict="SOURCE_PASS",source_ref=helper_ref,source_manifest_ref=manifest_ref,blockers=[])
  if case=="blocked_helper_review":helper_review["verdict"]="SOURCE_BLOCKED"
  memory["research/helper_review.json"]=json.dumps(helper_review,sort_keys=True).encode();helper_review_ref=memory_ref("research/helper_review.json")
  mr=copy.deepcopy(memo_review)
  if case=="wrong_memo_review":mr["source_ref"]=dict(path="research/old_memo.py",sha256="0"*64)
  memory["research/memo_review.json"]=json.dumps(mr,sort_keys=True).encode();mr_ref=memory_ref("research/memo_review.json")
  epoch=RealDatetime(2026,10,17,tzinfo=timezone.utc)
  request=dict(key="wrong/key"if case=="wrong_key"else h["TARGET_KEY"],reservation_sha256="fixture-digest",revocation_ref=rev,observation_ref=obs)
  auth=dict(request=request,expected_before=dict(prefix=0),expires_at_utc=(epoch+timedelta(seconds=600)).isoformat())
  memory["research/current_authority.json"]=json.dumps(auth,sort_keys=True).encode();ar=memory_ref("research/current_authority.json")
  instruction=dict(schema="ROOT_exact_Relate64_retirement_transaction_instruction_v1",root_authorizer="root_agent",active=True,
   helper_ref=helper_ref,source_manifest_ref=manifest_ref,independent_review_ref=helper_review_ref,memo_ref=h["MEMO_REF"],memo_review_ref=mr_ref,
   guard_ref=h["GUARD_REF"],key=h["TARGET_KEY"],operation=h["RETIRE_OPERATION"],cache_refs=code_refs,uncached_current_refs=[],
   operation_wall_seconds=300,original_authority_schema_unchanged=True,original_reservation_files_caps_expiry_and_frozen_ancestor_journals_unchanged=True,
   new_accounting_lineage=False,automatic_retry=False,expires_at_utc=(epoch+timedelta(seconds=10 if case=="instruction_expiry_finish"else 600)).isoformat())
  if case=="wrong_instruction_helper":instruction["helper_ref"]=dict(path="research/wrong_helper.py",sha256="0"*64)
  memory["research/ROOT_instruction.json"]=json.dumps(instruction,sort_keys=True).encode();instruction_ref=memory_ref("research/ROOT_instruction.json")
  def relative(name):
   p=MP(name);return str(p.relative_to(root))if p.is_absolute()else str(p)
  def canonical(r,name):p=MP(name);return p if p.is_absolute()else r/p
  def file_bytes(r,name):
   key=relative(name);reads[key]=reads.get(key,0)+1;data=memory[key]
   if key=="src/fixture.py"and reads[key]==2:
    events.append("original_uncached_final_byte_check")
    if case=="finish_300":mono[0]=300.
    if case=="success_near300":mono[0]=299.999
    if case=="authority_expiry_finish":utc[0]=601.
    if case=="instruction_expiry_finish":utc[0]=11.
   return data,hashlib.sha256(data).hexdigest()
  v4=ModuleType("independent_memory_v4");v4.__dict__.update(Path=MP,canonical=canonical,file_bytes=file_bytes,parse=json.loads,
   integer=lambda n,*args:demand(type(n)is int and n>=1,"bounded integer"),paths_overlap=lambda a,b:a==b or a in b.parents or b in a.parents,
   accounting_paths=lambda r:[],encoded=lambda x:json.dumps(x,sort_keys=True).encode(),LEGACY=".runtime/legacy.json")
  isolate([function(v4raw,n)for n in("bound","reference","scope_spec","read_json")],v4.__dict__)
  budget=ModuleType("independent_memory_budget");budget.__dict__.update(v4=v4,JOURNAL_REL=".runtime/journal",digest=lambda x:"fixture-digest")
  isolate([function(budgetraw,"successor_paths")],budget.__dict__)
  original_functions={n:getattr(v4,n)for n in("canonical","file_bytes","bound","reference","scope_spec")};original_successor=budget.successor_paths
  @contextmanager
  def mutex(r):
   demand(not locked[0],"no nestedlock");events.append("mutex_wait_enter");locked[0]=True
   if case=="wait_300":mono[0]=300.
   try:yield
   finally:
    need(all(getattr(v4,k)is v for k,v in original_functions.items())and budget.successor_paths is original_successor,"restore beforeunlock")
    need(all(not tx.sources and not tx.names and not tx.uncached and tx.bytes==0 and not tx.enabled for tx in txs),"destroy beforeunlock")
    locked[0]=False;events.append("mutex_exit")
  v4.mutex=mutex
  class Clock:
   fromisoformat=staticmethod(RealDatetime.fromisoformat)
   @staticmethod
   def now(zone):return epoch+timedelta(seconds=utc[0])
  expire_scope=isolate([function(guard_raw,"_expire")],dict(datetime=Clock,timezone=timezone,require=demand))
  expire=expire_scope["_expire"]
  own=dict(key=h["TARGET_KEY"],output=str(root/"data/original"),auxiliary_scopes=[str(root/"research/original_aux")],reserved_bytes=64*2**30,consumer_bindings=code_refs+[rev,obs,historical_ref])
  view=dict(new_reserved_bytes=64*2**30,completed_credit_bytes=0);state=({h["TARGET_KEY"]:own},set(),view,dict(code=code_refs[0]),[],[],None)
  def state_locked(*args,**kwargs):
   demand(locked[0]and kwargs==dict(fresh=True),"original fresh state inside lock");events.append("fresh_state")
   for _ in range(6):budget.successor_paths(root,own);v4.bound(root,historical_ref,True)
   return state
  def authority(record,operation,bindings,**kwargs):
   demand(kwargs==dict(fresh=True),"original fresh authority")
   parsed=v4.bound(root,record,True);expire(parsed);events.append("original_current_authority");return parsed
  def proof(request,own,**kwargs):
   for record in(rev,obs):v4.bound(root,record,True)
   events.append("observe_targets"if kwargs.get("observe")else"original_proof_recheck")
   return dict(pre_measurement_targets={},post_measurement_targets={})
  measured=SimpleNamespace(require=demand)
  def measure(*args,**kwargs):
   events.extend(["stat_pass1","stat_pass2","final_census"])
   for _ in range(3):measured.require(True,"original finite pernode")
   if case=="historical_JSON_drift":memory[historical_ref["path"]]=b'{"completed":false}'
   if case=="current_observation_drift":memory[obs["path"]]=b'{"absent":false}'
   if case=="measure_exception":raise ValueError("measurement exception")
   return dict(retained=123)
  measured.measure_retained_nodes_v1=measure
  fake_import=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *args:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None)),module_from_spec=lambda spec:measured))
  def original_append(operation,record,value,s):
   demand(locked[0]and obj._tx is not None and not obj._tx.enabled,"cache disabled at originalappend")
   need(all(getattr(v4,n)is original_functions[n]for n in("canonical","file_bytes"))and v4.mutex is mutex,"original primitives")
   protected={ar["path"],rev["path"],obs["path"],manifest_ref["path"],helper_review_ref["path"],mr_ref["path"],instruction_ref["path"]}
   need(not protected.intersection(obj._tx.sources),"all current evidence and ROOTrefs excluded")
   before=reads["src/fixture.py"];v4.bound(root,code_refs[0]);need(reads["src/fixture.py"]==before+1,"append reads uncached")
   if case=="append_uncertainty":events.append("original_append_uncertainty");raise ValueError("uncertain original append; preserve/no replay")
   events.append("original_append");append_count[0]+=1;return "fixture-tail"
  guard=ModuleType("independent_memory_guard")
  guard.__dict__.update(time=SimpleNamespace(monotonic=lambda:mono[0]),require=demand,OPERATION_WALL_SECONDS=300,budget=budget,ROOT=root,
   _state_locked=state_locked,_authority=authority,_pins=lambda v:dict(prefix=0),RETIRE_OPERATION=h["RETIRE_OPERATION"],TARGET_KEYS=(h["TARGET_KEY"],),
   assert_open=lambda *args:None,consumer_bindings_v1=lambda *groups:[v4.bound(root,r)for group in groups for r in group],
   _stopped_proof=proof,importlib=fake_import,MEASURE_SOURCE="source-only-unused.py",_paths=lambda value:[MP(value["output"]),MP(value["auxiliary_scopes"][0])],
   _target=lambda key:dict(scope_paths=[own["output"],own["auxiliary_scopes"][0]]),_retained_measurement=lambda value,own:value["retained"],
   _observe_targets=lambda target:events.append("post_measurement_absence")or{},_utc=lambda:"fixture-closed",
   _recorded_stopped_proof=lambda *args:events.append("original_proof_fingerprint"),copy=copy,charge=lambda *args:999,
   _policy=lambda:dict(base_projected_bytes=0),_expire=expire,_append=original_append,warning_fields=lambda projected:{})
  def source_auth(record):
   demand(record in(helper_ref,h["MEMO_REF"]),"exact pinned helper/memo source authentication");events.append("uncached_source_auth")
  def provenance(module,record,names):
   demand(record==h["GUARD_REF"]and names==("apply_retire_stopped_v1","_state_locked","_append"),"exact original provenancegate requested")
   return dict(apply_retire_stopped_v1=(None,original))
  def tx_factory(*args,**kwargs):
   tx=m["_Transaction"](*args,**kwargs);txs.append(tx);return tx
  memo_view=SimpleNamespace(_Transaction=tx_factory,_ACTIVE=False,source_bytes=source_auth,functions_from_source=provenance)
  obj=h["RetireStoppedMemo"].__new__(h["RetireStoppedMemo"])
  obj.memo=memo_view;obj.guard=guard;obj.budget=budget;obj.v4=v4;obj.core=SimpleNamespace(allow={r["path"]:r["sha256"]for r in code_refs},_check=lambda:events.append("core_original_provenance"))
  obj.manifest_ref=manifest_ref;obj.helper_ref=helper_ref;obj.review_ref=helper_review_ref;obj.memo_review_ref=mr_ref;obj.instruction_ref=instruction_ref
  obj.cache_refs=code_refs;obj.current_refs=[];obj.counts={};obj._tx=None;obj._used=False
  result=None;error=None
  try:result=obj.apply_retire_stopped_v1(ar,expected_source_sha256="fixture-budget",expected_launcher_sha256="fixture-launcher")
  except ValueError as exc:error=str(exc)
  success=case in("success","success_near300");need((error is None)==success,"expected independent outcome "+case+": "+str(error))
  need(append_count[0]==int(success)and not locked[0]and obj._tx is None and not h["_ACTIVE"]and not memo_view._ACTIVE,"public generation/append/teardown")
  if success:
   need(result["kind"]=="Relate64_exact_failed_retention_receipt_v1"and result["retained_bytes"]==123 and result["credit_bytes"]==own["reserved_bytes"]-123,"unchanged original measuredreceipt")
   need(events.index("stat_pass1")<events.index("final_census")<events.index("original_uncached_final_byte_check")<events.index("original_append")<events.index("mutex_exit"),"measurement then finalvalidation thenappend thenunlock")
   need(reads[historical_ref["path"]]==2 and reads[obs["path"]]>=3,"historicalinitial+final/currentevidencefresh")
   try:obj.apply_retire_stopped_v1(ar,expected_source_sha256="fixture-budget",expected_launcher_sha256="fixture-launcher")
   except ValueError:pass
   else:raise AssertionError("one-use helper replay accepted")
   need(append_count[0]==1 and events.count("mutex_wait_enter")==1,"no secondtransaction on replay")
  elif case=="append_uncertainty":
   need(obj._used and result is None,"uncertain append gives no receipt and consumes instance")
   try:obj.apply_retire_stopped_v1(ar,expected_source_sha256="fixture-budget",expected_launcher_sha256="fixture-launcher")
   except ValueError:pass
   else:raise AssertionError("uncertain append replay accepted")
   need(events.count("mutex_wait_enter")==1 and events.count("original_append_uncertainty")==1,"no uncertain append replay")
  elif case in("wrong_memo_review","wrong_instruction_helper","blocked_helper_review"):
   need("mutex_wait_enter"not in events,"auth failure beforeoperation/lock")
  else:need(events[-1]=="mutex_exit","exceptional teardown beforeunlock")
  reports.append(dict(case=case,expected_success=success,error=error,append_count=append_count[0],events=events,current_authority_reads=reads.get(ar["path"],0),current_observation_reads=reads.get(obs["path"],0)))
 need(not any(n.startswith(("arggnn","scripts.project_budget","_H058","_child_source_repair"))for n in sys.modules),"no projectguard imports")
 return dict(schema="independent_Relate_retirement_transaction_checks_v1",source_only=True,source_ref=ref(SOURCE_DIR+"/retire_transaction_memo_v1.py"),
  source_manifest_ref=ref(SOURCE_DIR+"/source_manifest.v1.json"),source_handoff_ref=ref(SOURCE_DIR+"/source_handoff.v1.json"),authenticated_source_pins=len(manifest["source_sha256"]),
  actual_memo2_component_review_ref=ref(MEMO_REVIEW),actual_derivative_AST_captured_and_exactly_reversed=True,original300s_prelock_deadline_preserved=True,
  final_uncached_validation_then_original_authority_instruction_expiry_then_originalfinite_then_originalappend=True,public_helper_cases=reports,
  helper_public_check_authenticate_transaction_append_and_derive_methods_exercised=True,actual_original_retirement_and_v4_AST_used=True,
  handwritten_guard_state_proof_measurement_and_provenancegate_stubs=True,all_current_evidence_ROOTrefs_excluded_and_raw_observations_fresh=True,
  measured_receipt_before_credit=True,one_use_no_replay=True,all_exits_restore_clear_beforeunlock=True,
  actual_guard_state_mutex_process_resource_payload_data_job_Git_operations=False,real300s_fit_performance_or_credit_claim=False)
if __name__=="__main__":
 try:print(json.dumps(main(),indent=2,sort_keys=True))
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
