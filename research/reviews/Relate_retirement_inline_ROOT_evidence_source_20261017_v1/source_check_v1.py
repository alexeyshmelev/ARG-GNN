"""Source AST and handwritten-memory checks only. No guard or operational imports."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from contextlib import contextmanager
from datetime import datetime as RealDateTime,timezone,timedelta
import ast,copy,hashlib,io,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/Relate_retirement_inline_ROOT_evidence_source_20261017_v1"
def need(v,m):
 if not v:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 source=(HERE/"inline_ROOT_retire_v1.py").read_text();tree=ast.parse(source)
 ns={"__name__":"handwritten_inline_helper"};exec(compile(tree,"handwritten_inline_helper","exec"),ns)
 guard_path=ROOT/ns["GUARD_REF"]["path"];memo_path=ROOT/ns["MEMO_REF"]["path"]
 need(sha(guard_path)==ns["GUARD_REF"]["sha256"]and sha(memo_path)==ns["MEMO_REF"]["sha256"],"Exact unchanged guard/memo2")
 gt=ast.parse(guard_path.read_text());original=next(n for n in gt.body if isinstance(n,ast.FunctionDef)and n.name=="apply_retire_stopped_v1")
 # Independently construct/reverse all three declared changes and compare EXACT AST.
 d=copy.deepcopy(original);d.name="_inline_ROOT_exact_retirement"
 call=next(n for n in ast.walk(d.body[-1])if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_append")
 call.func.id="_retirement_append";call.args[0:0]=[ast.Name(id="finite",ctx=ast.Load()),ast.Name(id="auth",ctx=ast.Load())]
 d.body[-1].body.insert(2,ast.parse("authority_ref = _inline_ROOT_authority(state, finite)").body[0])
 d.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="_retirement_transaction",ctx=ast.Load()),args=[ast.Name(id="finite",ctx=ast.Load())],keywords=[]),optional_vars=None)],body=d.body[-1].body)]
 actual=ns["derive_inline_retirement"](original,{},lambda f:None,lambda s,f:None,lambda *a:None)
 manual={};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(d)],type_ignores=[])),actual.__code__.co_filename,"exec"),manual)
 need(manual[d.name].__code__==actual.__code__,"Actual derivative equals independent three-change construction")
 restored=copy.deepcopy(d);restored.name=original.name;restored.body[-1].body=restored.body[-1].body[0].body
 seam=restored.body[-1].body.pop(2)
 need(ast.unparse(seam)=="authority_ref = _inline_ROOT_authority(state, finite)","Exact only fresh-evidence seam")
 call=next(n for n in ast.walk(restored)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_retirement_append")
 call.func.id="_append";need([call.args.pop(0).id,call.args.pop(0).id]==["finite","auth"],"Original finite/auth passed unchanged")
 need(ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False),"Entire original AST restores byte-semantically")
 mn={"__name__":"handwritten_memo2"};exec(compile(memo_path.read_text(),"handwritten_memo2","exec"),mn)
 v4raw=(ROOT/"src/arggnn/project_budget_v4.py").read_text();v4tree=ast.parse(v4raw)
 reports=[]
 cases=("cold121_success","stale_ROOT_rows","wrong_new_authority_path","changed_same_state_prefix","callback_exception","current_observation_drift","cached_source_drift","cached_namespace_drift","deadline_during_final_raw","authority_expiry_during_final_raw","instruction_expiry_during_final_raw","mutex_wait300","post_measurement_target_present","second_instance_same_paths")
 for case in cases:
  events=[];clock=[0.];locked=[False];transactions=[];counts={};memory={};directories=set();late=[False];callback_checks=[0]
  class MemoryPath(PurePosixPath):
   def exists(self):return str(self)in memory or str(self)in directories
   def is_dir(self):return str(self)in directories
   def lstat(self):return info(str(self))
  root=MemoryPath("/HANDWRITTEN_NO_FILES");ns["ROOT"]=root
  def stamp():return(RealDateTime(2026,10,17,tzinfo=timezone.utc)+timedelta(seconds=clock[0])).isoformat()
  class DT:
   @staticmethod
   def now(tz):return RealDateTime.fromisoformat(stamp())
   @staticmethod
   def fromisoformat(value):return RealDateTime.fromisoformat(value)
  ns["datetime"]=DT
  md="research/reviews/Relate_retention_inline_ROOT_actual_MEMORY"
  directories.update([str(root),str(root/md)])
  paths={name:md+"/"+name+".json"for name in("attempt","observation","authority")}
  def key(p):
   p=MemoryPath(p);return str(p if p.is_absolute()else root/p)
  def rel(p):return str(MemoryPath(key(p)).relative_to(root))
  def data(p):return memory[key(p)]
  def mref(p):return dict(path=rel(p),sha256=hashlib.sha256(data(p)).hexdigest())
  def add(p,v):memory[key(p)]=(json.dumps(v,sort_keys=True)+"\n").encode();return mref(p)
  def info(p):
   isdir=p in directories;return SimpleNamespace(st_dev=1,st_ino=sum(p.encode())+1,st_mode=stat.S_IFDIR if isdir else stat.S_IFREG,st_nlink=1,st_size=0 if isdir else len(memory.get(p,b"")),st_mtime_ns=1,st_ctime_ns=1)
  def canonical(r,p):
   need(locked[0],"Only handwritten lock");p=MemoryPath(key(p))
   return p/"DRIFT"if case=="cached_namespace_drift"and late[0]else p
  def file_bytes(r,p):
   need(locked[0],"Only handwritten reader");k=key(p);counts[k]=counts.get(k,0)+1
   if transactions and not transactions[0].enabled and not late[0]:
    late[0]=True;events.append("raw_final_revalidation")
    if case=="deadline_during_final_raw":clock[0]=300.
    if case=="authority_expiry_during_final_raw":clock[0]=200.
    if case=="instruction_expiry_during_final_raw":clock[0]=250.
   raw=memory[k];return raw,hashlib.sha256(raw).hexdigest()
  v4=ModuleType("handwritten_v4");v4.__dict__.update(canonical=canonical,file_bytes=file_bytes,hashlib=hashlib,json=json,Path=MemoryPath)
  for n in v4tree.body:
   if isinstance(n,ast.FunctionDef)and n.name in("parse","bound","reference","read_json"):
    exec(compile(ast.Module(body=[n],type_ignores=[]),"original_v4_memory","exec"),v4.__dict__)
  def scope_spec(r,v):return[v4.canonical(r,v["output"])]
  v4.scope_spec=scope_spec
  budget=ModuleType("handwritten_budget");budget.v4=v4;budget.digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
  def successor_paths(r,v):return v4.scope_spec(r,v)
  budget.successor_paths=successor_paths
  @contextmanager
  def mutex(r):
   need(not locked[0],"No outer lock");locked[0]=True;events.append("mutex_enter")
   if case=="mutex_wait300":clock[0]=300.
   try:yield
   finally:
    need(v4.bound is originals["bound"]and v4.canonical is canonical and v4.file_bytes is file_bytes,"Originals restored before unlock")
    need(all(not tx.sources and not tx.names and not tx.uncached and tx.bytes==0 for tx in transactions),"Every cache destroyed before unlock")
    locked[0]=False;events.append("unlock_after_restore")
  v4.mutex=mutex
  for n in("canonical","file_bytes","mutex"):getattr(v4,n).__module__=v4.__name__
  originals={n:getattr(v4,n)for n in("bound","reference","scope_spec","canonical","file_bytes","mutex")}
  code_ref=add("src/a.py",dict(source="unchanged"));hist_ref=add("research/historical.json",dict(historical=True))
  add(".runtime/raw_journal.json",dict(prefix=1))
  revref=add(md+"/revocation.json",dict(permanent=True))
  iref=add(md+"/instruction.json",dict(instruction=True));helperref=dict(path=ns["SOURCE"],sha256="1"*64)
  callbackref=dict(path=md+"/callback.py",sha256="2"*64)
  own=dict(key=ns["TARGET_KEY"],output=str(root/"original_failed"),auxiliary_scopes=[str(root/"old_aux")],reserved_bytes=64*2**30,consumer_bindings=[code_ref,hist_ref])
  state=({own["key"]:own},set(),dict(new_reserved_bytes=64*2**30,completed_credit_bytes=0),dict(source=code_ref),[],[],None)
  t=dict(original_reservation=copy.deepcopy(own),known_PID_start_rows=[dict(pid=10,start_ticks=20)],known_process_groups=[10],scope_paths=[own["output"],own["auxiliary_scopes"][0]])
  g=ModuleType("handwritten_guard");g.__dict__.update(ROOT=root,budget=budget,MODE="same_original_journal",RETIRE_OPERATION=ns["RETIRE_OPERATION"],TARGET_KEYS=(ns["TARGET_KEY"],),OPERATION_WALL_SECONDS=300,FRESH_TARGET_SECONDS=120,MAX_JSON=65536,
   require=need,_utc=stamp,_stamp=RealDateTime.fromisoformat,datetime=DT,timezone=timezone)
  def expire(v):need(RealDateTime.fromisoformat(v["expires_at_utc"])>DT.now(timezone.utc),"Expired original authority/instruction")
  g._expire=expire;g._target=lambda k:t;g._pins=lambda view:dict(prefix=0);g.assert_open=lambda k,r,c:need(k in r and k not in c,"Original open reservation")
  def state_locked(*a,**kw):
   need(kw==dict(fresh=True),"Original fresh state");events.append("original_fresh_state")
   for _ in range(3):v4.bound(root,code_ref);v4.bound(root,hist_ref,True);v4.reference(root,"src/a.py");v4.read_json(root,".runtime/raw_journal.json")
   clock[0]=121.;return state
  def observe(target):
   events.append("actual_target_observation")
   if case=="post_measurement_target_present"and "final_census"in events:raise ValueError("Known target present")
   return dict(PID_observations=[[10,20,True,stamp()]],group_observations=[[10,True,stamp()]])
  g._observe_targets=observe
  for n in gt.body:
   if isinstance(n,ast.FunctionDef)and n.name in("operation_draft_v1","_authority","_validate_targets","consumer_bindings_v1"):
    exec(compile(ast.Module(body=[n],type_ignores=[]),"original_guard_predicates_memory","exec"),g.__dict__)
  g._raw=lambda r:v4.bound(root,r)
  g._json=lambda r:v4.bound(root,r,True)
  def proof(request,own,observe=False):
   obs=v4.bound(root,request["observation_ref"],True);facts={k:obs[k]for k in("PID_observations","group_observations")}
   g._validate_targets(facts,t,RealDateTime(2026,10,17,tzinfo=timezone.utc),DT.now(timezone.utc),fresh=observe)
   current=g._observe_targets(t)if observe else facts
   return dict(pre_measurement_targets=current,post_measurement_targets=copy.deepcopy(current))
  helper=SimpleNamespace(require=need)
  def measure(*a,**kw):
   events.extend(["first_stat_pass","second_stat_pass","final_census"]);clock[0]+=5
   if case=="cached_source_drift":memory[key("src/a.py")]=b"CHANGED"
   if case=="current_observation_drift":memory[key(paths["observation"])]=b"CHANGED"
   return dict(retained=123)
  helper.measure_retained_nodes_v1=measure
  g._state_locked=state_locked;g._stopped_proof=proof;g._paths=lambda v:[MemoryPath(v["output"]),MemoryPath(v["auxiliary_scopes"][0])]
  g._retained_measurement=lambda *a:123;g._recorded_stopped_proof=lambda *a:None;g._policy=lambda:dict(base_projected_bytes=1)
  g.charge=lambda *a:1;g.warning_fields=lambda p:dict();g.MEASURE_SOURCE="never_imported"
  def append(*a):
   need(transactions and not transactions[0].enabled,"Hits disabled through original append");events.append("original_append");return"tail"
  g._append=append
  g.importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)),module_from_spec=lambda s:helper))
  g.time=SimpleNamespace(monotonic=lambda:clock[0]);g.copy=copy
  obj=ns["InlineRootRetireStoppedMemo"].__new__(ns["InlineRootRetireStoppedMemo"])
  obj.guard=g;obj.budget=budget;obj.v4=v4;obj.memo=SimpleNamespace(_ACTIVE=False)
  def transaction(*a,**kw):
   tx=mn["_Transaction"](*a,**kw);transactions.append(tx)
   need(all(p in tx.uncached for p in paths.values()),"Fresh paths excluded BEFORE attempt write")
   return tx
  obj.memo._Transaction=transaction;obj.core=SimpleNamespace(allow={code_ref["path"]:code_ref["sha256"]})
  obj.manifest_ref=obj.review_ref=obj.memo_review_ref=iref;obj.instruction_ref=iref;obj.callback_ref=callbackref;obj.helper_ref=helperref
  obj.paths=paths;obj.revocation_ref=revref;obj.current_refs=[];obj._used=False;obj._tx=None;obj._callback_calls=obj._append_calls=0;obj._authority_ref=None
  obj.functions={"apply_retire_stopped_v1":(None,original)}
  instruction=dict(expires_at_utc=(RealDateTime(2026,10,17,tzinfo=timezone.utc)+timedelta(seconds=250 if case=="instruction_expiry_during_final_raw"else 900)).isoformat())
  def authenticate(bound):
   need(bound(root,iref,True)==dict(instruction=True),"Current instruction never cached");g._expire(instruction);return instruction
  obj._authenticate=authenticate;obj._check=lambda:None;obj._check_callback=lambda:callback_checks.__setitem__(0,callback_checks[0]+1)
  class Stream(io.BytesIO):
   def __init__(self,p):super().__init__();self.p=p
   def close(self):
    if not self.closed:memory[self.p]=self.getvalue()
    super().close()
   def fileno(self):return 123
   def flush(self):memory[self.p]=self.getvalue()
  class FakeOS:
   O_WRONLY=os.O_WRONLY;O_CREAT=os.O_CREAT;O_EXCL=os.O_EXCL;O_NOFOLLOW=os.O_NOFOLLOW;O_RDONLY=os.O_RDONLY;O_DIRECTORY=os.O_DIRECTORY
   @staticmethod
   def open(p,flags,mode=0):
    p=key(p)
    if flags&os.O_DIRECTORY:return 999
    need(flags&os.O_EXCL and flags&os.O_NOFOLLOW and p not in memory,"Exclusive new token only")
    memory[p]=b"";events.append("attempt_created");return 123
   @staticmethod
   def fdopen(fd,mode):return Stream(key(paths["attempt"]))
   @staticmethod
   def fstat(fd):return info(key(paths["attempt"]))
   @staticmethod
   def fsync(fd):events.append("fsync")
   @staticmethod
   def close(fd):pass
  ns["os"]=FakeOS
  def callback(*,guard,context,finite):
   need(context["expected_before"]==g._pins(state[2])and context["bindings"]==state[3],"SAME actual fresh state context")
   need(context["original_reservation"]is not state[0][own["key"]],"Callback receives a deep copy, never actual state")
   events.append("ROOT_callback");finite()
   if case=="callback_exception":raise ValueError("ROOT callback failed")
   facts=g._observe_targets(t)
   if case=="stale_ROOT_rows":
    for rows in facts.values():rows[0][-1]=RealDateTime(2026,10,17,tzinfo=timezone.utc).isoformat()
   obsref=add(paths["observation"],dict(revocation_ref=revref,observed_at_utc=stamp(),**facts))
   request=dict(key=own["key"],reservation_sha256=budget.digest(own),revocation_ref=revref,observation_ref=obsref)
   auth=g.operation_draft_v1(g.RETIRE_OPERATION,request,expected_before=context["expected_before"],bindings=context["bindings"])
   auth.update(active=True,expires_at_utc=(RealDateTime(2026,10,17,tzinfo=timezone.utc)+timedelta(seconds=200 if case=="authority_expiry_during_final_raw"else 900)).isoformat())
   if case=="changed_same_state_prefix":auth["expected_before"]=dict(prefix=1)
   aref=add(paths["authority"],auth)
   return dict(aref,path=md+"/wrong.json")if case=="wrong_new_authority_path"else aref
  obj.callback=callback
  failure=None
  try:obj.apply_retire_stopped_v1(expected_source_sha256="source",expected_launcher_sha256="launcher")
  except (ValueError,KeyError)as e:failure=str(e)
  success=case in("cold121_success","second_instance_same_paths")
  need((failure is None)==success,"Expected outcome "+case+": "+str(failure))
  if success:
   need(events.index("original_fresh_state")<events.index("ROOT_callback")<events.index("first_stat_pass")<events.index("second_stat_pass")<events.index("final_census")<events.index("raw_final_revalidation")<events.index("original_append"),"Original ordering and final raw validation")
   need(len([e for e in events if e=="ROOT_callback"])==1,"Exactly one callback")
   if case=="second_instance_same_paths":
    obj._used=False;obj._authority_ref=None;obj._callback_calls=obj._append_calls=0
    try:obj.apply_retire_stopped_v1(expected_source_sha256="source",expected_launcher_sha256="launcher")
    except ValueError:pass
    else:raise ValueError("New helper instance replay must fail on immutable attempt path")
  else:need("original_append"not in events,"No append on "+case)
  need(not locked[0]and ns["_ACTIVE"]is False and obj.memo._ACTIVE is False,"All locks/active flags restored")
  need(all(not tx.sources and not tx.names and not tx.uncached for tx in transactions),"Cold cache destroyed")
  reports.append(dict(case=case,status="PASS",failure_expected=not success,events=events))
 # Actual helper whole-join/ROOT instruction predicate on handwritten metadata.
 metadata_reports=[]
 template=json.loads((HERE/"ROOT_instruction.INACTIVE.template.v1.json").read_text())
 need(template["active"]is False and template["root_authorizer"]is None
  and template["created_at_utc"]is None and template["expires_at_utc"]is None,"No active template authority")
 review_template=json.loads((HERE/"whole_join_review.INACTIVE.template.v1.json").read_text())
 need(review_template["independent"]is False and review_template["actual_callback_complete_source_review"]is False,"No fabricated whole-join review")
 for metadata_case in("genuine_handwritten_metadata","inactive_instruction","wrong_key","unreviewed_callback","extra_instruction_field","future_instruction","expired_instruction","callback_outside_preparation","aliased_fresh_paths","instruction_outside_preparation"):
  locked[0]=True;clock[0]=126.;late[0]=True
  obj2=ns["InlineRootRetireStoppedMemo"].__new__(ns["InlineRootRetireStoppedMemo"])
  spec=copy.deepcopy(template)
  callback_ref=add(md+"/actual_callback.py",dict(handwritten_source=True))
  concrete_paths={n:md+"/fresh_"+n+".json"for n in("attempt","observation","authority")}
  if metadata_case=="callback_outside_preparation":callback_ref=add("research/other/callback.py",dict(handwritten_source=True))
  if metadata_case=="aliased_fresh_paths":concrete_paths["authority"]=concrete_paths["observation"]
  manifest_ref=add(md+"/manifest.json",dict(source_sha256={ns["SOURCE"]:spec["helper_ref"]["sha256"],ns["MEMO_REF"]["path"]:ns["MEMO_REF"]["sha256"],ns["MEMO_MANIFEST_REF"]["path"]:ns["MEMO_MANIFEST_REF"]["sha256"]}))
  memo_review_ref=add(md+"/memo_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=ns["MEMO_REF"],source_manifest_ref=ns["MEMO_MANIFEST_REF"]))
  review=dict(schema="Relate64_inline_ROOT_retirement_whole_join_review_v1",independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=spec["helper_ref"],source_manifest_ref=manifest_ref,callback_source_ref=callback_ref,callback_name=ns["CALLBACK_NAME"],fresh_metadata_paths=concrete_paths,permanent_revocation_ref=revref,guard_ref=ns["GUARD_REF"],memo_ref=ns["MEMO_REF"],actual_callback_complete_source_review=True)
  if metadata_case=="unreviewed_callback":review["actual_callback_complete_source_review"]=False
  review_ref=add(md+"/review.json",review)
  spec.update(active=True,root_authorizer="root_agent",source_manifest_ref=manifest_ref,whole_join_review_ref=review_ref,memo_review_ref=memo_review_ref,callback_source_ref=callback_ref,metadata_directory=md,fresh_metadata_paths=concrete_paths,permanent_revocation_ref=revref,created_at_utc=(DT.now(timezone.utc)-timedelta(seconds=1)).isoformat(),expires_at_utc=(DT.now(timezone.utc)+timedelta(seconds=900)).isoformat())
  if metadata_case=="inactive_instruction":spec["active"]=False
  if metadata_case=="wrong_key":spec["key"]="other/key"
  if metadata_case=="extra_instruction_field":spec["extra"]=True
  if metadata_case=="future_instruction":spec["created_at_utc"]=(DT.now(timezone.utc)+timedelta(seconds=1)).isoformat()
  if metadata_case=="expired_instruction":spec["expires_at_utc"]=DT.now(timezone.utc).isoformat()
  instruction_path=("research/other/instruction.json"if metadata_case=="instruction_outside_preparation"else md+"/actual_instruction.json")
  instruction_ref=add(instruction_path,spec)
  for attr,value in dict(v4=v4,guard=g,helper_ref=spec["helper_ref"],manifest_ref=manifest_ref,review_ref=review_ref,memo_review_ref=memo_review_ref,instruction_ref=instruction_ref,callback_ref=callback_ref,callback_name=ns["CALLBACK_NAME"],metadata_directory=md,paths=concrete_paths,revocation_ref=revref,cache_refs=spec["cache_refs"],current_refs=[],memo=SimpleNamespace(source_bytes=lambda r:b"handwritten source read")).items():setattr(obj2,attr,value)
  rejected=False
  try:obj2._authenticate(v4.bound)
  except (ValueError,KeyError,TypeError):rejected=True
  need(rejected==(metadata_case!="genuine_handwritten_metadata"),"Actual helper metadata predicate "+metadata_case)
  locked[0]=False;metadata_reports.append(dict(case=metadata_case,status="PASS",rejection_expected=rejected))
 callback_shape_reports=[]
 baseline=(HERE/"ROOT_authorizer_callback.INACTIVE.template.v1.py").read_bytes()
 shape_cases={"inactive_exact_stub":baseline,
  "top_level_call":baseline+b"\nprint(123)\n",
  "nonstdlib_import":b"import numpy\n"+baseline,
  "unsafe_stdlib_import":b"import antigravity\n"+baseline,
  "evaluated_default":b"def authorize_inline_ROOT_retirement_v1(*,guard,context,finite=evil()): pass\n",
  "decorator":b"@evil()\ndef authorize_inline_ROOT_retirement_v1(*,guard,context,finite): pass\n",
  "second_function":baseline+b"\ndef second(): pass\n"}
 for label,raw in shape_cases.items():
  obj3=ns["InlineRootRetireStoppedMemo"].__new__(ns["InlineRootRetireStoppedMemo"])
  obj3.callback_ref={};obj3.memo=SimpleNamespace(source_bytes=lambda r:raw)
  rejected=False
  try:obj3._callback_bytes()
  except (ValueError,TypeError):rejected=True
  need(rejected==(label!="inactive_exact_stub"),"Actual callback import shape gate "+label)
  callback_shape_reports.append(dict(case=label,status="PASS",rejection_expected=rejected))
 # The checker imports ONLY stdlib and defines source in private memory.
 need(not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit")),"No scientific imports")
 return dict(schema="Relate64_inline_ROOT_source_and_memory_checks_v1",status="SOURCE_STATIC_MEMORY_PASS_ONLY",cases=len(reports),reports=reports,metadata_cases=len(metadata_reports),metadata_reports=metadata_reports,callback_shape_cases=len(callback_shape_reports),callback_shape_reports=callback_shape_reports,entire_original_AST_restored=True,exactly_three_declared_AST_additions=True,original_guard_and_memo2_hashes_verified=True,source_template_not_runtime_qualification=True,actual_guard_or_mutex_or_PID_or_resource_or_accounting_or_authority_execution=False)
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
