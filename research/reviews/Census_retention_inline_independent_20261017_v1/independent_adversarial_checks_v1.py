"""Source AST and handwritten-memory checks only. No guard or operational imports."""
from pathlib import Path,PurePosixPath
from types import ModuleType,SimpleNamespace
from contextlib import contextmanager
from datetime import datetime as RealDateTime,timezone,timedelta
import ast,copy,hashlib,io,json,os,stat,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/Census_retention_inline_source_20261017_v1"
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
 cases=('freshness_exact120', 'freshness_over120', 'install_partial_failure', 'install_complete_failure', 'append_write_error', 'final_proof_error', 'callback_deadline300', 'same_instance_replay')
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
  md="research/reviews/Census_retention_inline_ROOT_actual_MEMORY"
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
  own=dict(key=ns["TARGET_KEYS"][0],output=str(root/"original_failed"),auxiliary_scopes=[],reserved_bytes=20000000000,consumer_bindings=[code_ref,hist_ref])
  state=({own["key"]:own},set(),dict(new_reserved_bytes=20000000000,completed_credit_bytes=0),{("role"+str(i)):code_ref for i in range(47)},[],[],None)
  t=dict(original_reservation=copy.deepcopy(own),known_PID_start_rows=[dict(pid=10,start_ticks=20)],known_process_groups=[10],scope_paths=[own["output"]])
  g=ModuleType("handwritten_guard");g.__dict__.update(ROOT=root,budget=budget,MODE="same_original_journal",RETIRE_OPERATION=ns["RETIRE_OPERATION"],TARGET_KEYS=ns["TARGET_KEYS"],OPERATION_WALL_SECONDS=300,FRESH_TARGET_SECONDS=120,MAX_JSON=65536,
   require=need,_utc=stamp,_stamp=RealDateTime.fromisoformat,datetime=DT,timezone=timezone)
  def expire(v):need(RealDateTime.fromisoformat(v["expires_at_utc"])>DT.now(timezone.utc),"Expired original authority/instruction")
  g._expire=expire;g._target=lambda k:t;g._pins=lambda view:dict(prefix=0);g.assert_open=lambda k,r,c:need(k in r and k not in c,"Original open reservation")
  def state_locked(*a,**kw):
   need(kw==dict(fresh=True),"Original fresh state");events.append("original_fresh_state")
   for _ in range(3):v4.bound(root,code_ref);v4.bound(root,hist_ref,True);v4.reference(root,"src/a.py");v4.read_json(root,".runtime/raw_journal.json")
   clock[0]=300. if case=="fresh_state300"else 121.;return state
  def observe(target):
   events.append("actual_target_observation")
   if case=="post_measurement_target_present"and "final_census"in events:raise ValueError("Known target present")
   return dict(PID_observations=[[10,20,True,stamp()]],group_observations=[[10,True,stamp()]])
  g._observe_targets=observe
  for n in gt.body:
   if isinstance(n,ast.FunctionDef)and n.name in("operation_draft_v1","_authority","_validate_targets","consumer_bindings_v1"):
    exec(compile(ast.Module(body=[n],type_ignores=[]),"original_guard_predicates_memory","exec"),g.__dict__)
  g._raw=lambda r:None # Lower source reader double; real source pins are checked separately, never a real guard.
  g._json=lambda r:v4.bound(root,r,True)
  def proof(request,own,observe=False):
   if observe and case in("freshness_exact120","freshness_over120"):
    clock[0]+=120 if case=="freshness_exact120"else 120.000001
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
  g._state_locked=state_locked;g._stopped_proof=proof;g._paths=lambda v:[MemoryPath(v["output"])]
  g._retained_measurement=lambda *a:123;g._recorded_stopped_proof=lambda *a:None;g._policy=lambda:dict(base_projected_bytes=1)
  g.charge=lambda *a:1;g.warning_fields=lambda p:dict();g.MEASURE_SOURCE="never_imported"
  def append(*a):
   need(transactions and not transactions[0].enabled,"Hits disabled through original append");events.append("original_append")
   if case=="append_overrun":clock[0]=300.
   return"tail"
  old_append=append
  def independent_append(*args):
   result=old_append(*args)
   if case=="append_write_error":raise ValueError("Independent post-write uncertainty")
   return result
  g._append=independent_append
  proof_calls=[0]
  def final_proof(*args):
   proof_calls[0]+=1
   if proof_calls[0]==2 and case=="final_proof_error":
    need(v4.bound is originals["bound"],"Uncached final proof after exact restoration")
    raise ValueError("Independent final raw proof refusal")
  g._recorded_stopped_proof=final_proof
  g.importlib=SimpleNamespace(util=SimpleNamespace(spec_from_file_location=lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)),module_from_spec=lambda s:helper))
  g.time=SimpleNamespace(monotonic=lambda:clock[0]);g.copy=copy
  obj=ns["InlineRootRetireStoppedMemo"].__new__(ns["InlineRootRetireStoppedMemo"])
  obj.guard=g;obj.budget=budget;obj.v4=v4;obj.memo=SimpleNamespace(_ACTIVE=False);obj.key=ns["TARGET_KEYS"][0]
  def transaction(*a,**kw):
   tx=mn["_Transaction"](*a,**kw);transactions.append(tx)
   need(all(p in tx.uncached for p in paths.values()),"Fresh paths excluded BEFORE attempt write")
   return tx
  original_transaction_factory=transaction
  def injected_transaction(*args,**kwargs):
   tx=original_transaction_factory(*args,**kwargs)
   if case=="install_partial_failure":
    def partial():
     replacement=tx.bound;tx.patches.append((v4,"bound",v4.bound,replacement));v4.bound=replacement
     raise ValueError("Independent partial-install fault")
    tx.install=partial
   if case=="install_complete_failure":
    original_install=tx.install
    def after_install():
     original_install();raise ValueError("Independent after-install fault")
    tx.install=after_install
   return tx
  obj.memo._Transaction=injected_transaction;obj.core=SimpleNamespace(allow={code_ref["path"]:code_ref["sha256"]},_check=lambda:need(v4.bound is originals["bound"]and budget.successor_paths is successor_paths,"Original callables restored before final core/source check"))
  obj.manifest_ref=obj.review_ref=obj.memo_review_ref=iref;obj.instruction_ref=iref;obj.callback_ref=callbackref;obj.helper_ref=helperref
  obj.paths=paths;obj.revocation_ref=revref;obj.current_refs=[];obj._used=False;obj._tx=None;obj._callback_calls=obj._append_calls=0;obj._authority_ref=None
  obj.functions={"apply_retire_stopped_v1":(None,original)}
  instruction=dict(expires_at_utc=(RealDateTime(2026,10,17,tzinfo=timezone.utc)+timedelta(seconds=250 if case=="instruction_expiry_during_final_raw"else 900)).isoformat())
  def authenticate(bound):
   need(bound(root,iref,True)==dict(instruction=True),"Current instruction never cached");g._expire(instruction);return instruction
  obj._tx_manifest_for_exclusions=lambda:dict(source_sha256={code_ref["path"]:code_ref["sha256"],hist_ref["path"]:hist_ref["sha256"]})
  instruction.update(expected_before=dict(prefix=1 if case=="prewrite_prefix_mismatch"else 0),expected_guard_roles=copy.deepcopy(state[3]))
  if case=="prewrite_missing_role":state[3].pop("role46")
  if case=="prewrite_cap_overflow":g.consumer_bindings_v1=lambda *a:[dict(path="RAM/extra"+str(i),sha256="a"*64)for i in range(64)]
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
  original_callback=callback
  def independent_callback(**kwargs):
   result=original_callback(**kwargs)
   if case=="callback_deadline300":clock[0]=300.
   return result
  obj.callback=independent_callback
  failure=None
  try:obj.apply_retire_stopped_v1(expected_source_sha256="source",expected_launcher_sha256="launcher")
  except (ValueError,KeyError)as e:failure=str(e)
  success=case in("freshness_exact120","same_instance_replay")
  need((failure is None)==success,"Expected outcome "+case+": "+str(failure))
  if success:
   need(events.index("original_fresh_state")<events.index("attempt_created")<events.index("ROOT_callback")<events.index("first_stat_pass")<events.index("second_stat_pass")<events.index("final_census")<events.index("raw_final_revalidation")<events.index("original_append"),"Original ordering and final raw validation")
   need(len([e for e in events if e=="ROOT_callback"])==1,"Exactly one callback")
   if case=="second_instance_same_paths":
    obj._used=False;obj._authority_ref=None;obj._callback_calls=obj._append_calls=0
    try:obj.apply_retire_stopped_v1(expected_source_sha256="source",expected_launcher_sha256="launcher")
    except ValueError:pass
    else:raise ValueError("New helper instance replay must fail on immutable attempt path")
  else:
   need(("original_append"in events)==(case in("append_overrun","append_write_error")),"Only append-overrun uncertainty may fail after original append "+case)
   if case.startswith("prewrite_")or case in("fresh_state300","mutex_wait300"):need("attempt_created"not in events and "ROOT_callback"not in events,"No writes before original state/prefix/47role/cap/deadline acceptance "+case)
  need(not locked[0]and ns["_ACTIVE"]is False and obj.memo._ACTIVE is False,"All locks/active flags restored")
  need(all(not tx.sources and not tx.names and not tx.uncached for tx in transactions),"Cold cache destroyed")
  if case in("same_instance_replay","append_write_error"):
   prior_events=list(events)
   try:obj.apply_retire_stopped_v1(expected_source_sha256="source",expected_launcher_sha256="launcher")
   except ValueError:pass
   else:raise ValueError("Independent same-instance retry unexpectedly allowed")
   need(events==prior_events,"No callback/lock/append on same-instance replay")
  reports.append(dict(case=case,status="PASS",failure_expected=not success,events=events))

 # Focused actual instruction/review predicate cases with handwritten bound/source readers.
 metadata_reports=[];clock[0]=126.;locked[0]=True
 template=json.loads((HERE/"ROOT_instruction.INACTIVE.template.v1.json").read_text())
 need(template["active"]is False and template["root_authorizer"]is None and template["key"]is None,"Inactive instruction grants no target capability")
 need(json.loads((HERE/"whole_join_review.INACTIVE.template.v1.json").read_text())["independent"]is False,"No invented independent review")
 for label in("valid","inactive","wrong_key","role_omission","review_prefix_mismatch","unreviewed_callback","selector_mismatch","extra_field","expired"):
  refs=lambda name:dict(path=md+"/"+name,sha256="a"*64)
  o=ns["InlineRootRetireStoppedMemo"].__new__(ns["InlineRootRetireStoppedMemo"])
  spec=copy.deepcopy(template);paths2={k:md+"/new_"+k+".json"for k in("attempt","authority","observation")}
  attrs=dict(key=ns["TARGET_KEYS"][0],guard=g,v4=SimpleNamespace(canonical=lambda r,p:MemoryPath(key(p))),memo=SimpleNamespace(source_bytes=lambda r:b"source already separately pinned"),instruction_ref=refs("instruction.json"),manifest_ref=refs("manifest.json"),review_ref=refs("review.json"),memo_review_ref=refs("memo_review.json"),helper_ref=template["helper_ref"],callback_ref=refs("callback.py"),callback_name=ns["CALLBACK_NAME"],paths=paths2,metadata_directory=md,revocation_ref=refs("revocation.json"),current_refs=[],cache_refs=template["cache_refs"])
  for k,v in attrs.items():setattr(o,k,v)
  roles={"role"+str(i):code_ref for i in range(47)}
  spec.update(active=True,root_authorizer="root_agent",key=o.key,source_manifest_ref=o.manifest_ref,whole_join_review_ref=o.review_ref,memo_review_ref=o.memo_review_ref,callback_source_ref=o.callback_ref,metadata_directory=md,fresh_metadata_paths=paths2,permanent_revocation_ref=o.revocation_ref,expected_before=dict(prefix=0),expected_guard_roles=roles,created_at_utc=(DT.now(timezone.utc)-timedelta(seconds=1)).isoformat(),expires_at_utc=(DT.now(timezone.utc)+timedelta(seconds=900)).isoformat())
  review=dict(schema="Census_inline_ROOT_retirement_whole_join_review_v1",independent=True,verdict="SOURCE_PASS",blockers=[],source_ref=o.helper_ref,source_manifest_ref=o.manifest_ref,callback_source_ref=o.callback_ref,callback_name=o.callback_name,fresh_metadata_paths=o.paths,permanent_revocation_ref=o.revocation_ref,guard_ref=ns["GUARD_REF"],memo_ref=ns["MEMO_REF"],selector_ref=ns["SELECTOR_REF"],policy_ref=ns["POLICY_REF"],bridge_ref=ns["BRIDGE_REF"],actual_callback_complete_source_review=True,key=o.key,expected_before=dict(prefix=0),expected_guard_roles=copy.deepcopy(roles))
  manifest=dict(source_sha256={ns["SOURCE"]:o.helper_ref["sha256"],ns["MEMO_REF"]["path"]:ns["MEMO_REF"]["sha256"],ns["MEMO_MANIFEST_REF"]["path"]:ns["MEMO_MANIFEST_REF"]["sha256"],ns["BRIDGE_REF"]["path"]:ns["BRIDGE_REF"]["sha256"]})
  selector=dict(source_ref=ns["GUARD_REF"],policy_ref=ns["POLICY_REF"]);policy=dict(source_ref=ns["GUARD_REF"],active=True)
  if label=="inactive":spec["active"]=False
  if label=="wrong_key":spec["key"]="other/key"
  if label=="role_omission":spec["expected_guard_roles"].pop("role46");review["expected_guard_roles"].pop("role46")
  if label=="review_prefix_mismatch":spec["expected_before"]=dict(prefix=1)
  if label=="unreviewed_callback":review["actual_callback_complete_source_review"]=False
  if label=="selector_mismatch":selector["source_ref"]={}
  if label=="extra_field":spec["extra"]=True
  if label=="expired":spec["expires_at_utc"]=DT.now(timezone.utc).isoformat()
  docs={o.instruction_ref["path"]:spec,o.manifest_ref["path"]:manifest,o.review_ref["path"]:review,o.memo_review_ref["path"]:dict(independent=True,verdict="SOURCE_PASS",source_ref=ns["MEMO_REF"],source_manifest_ref=ns["MEMO_MANIFEST_REF"],blockers=[]),ns["SELECTOR_REF"]["path"]:selector,ns["POLICY_REF"]["path"]:policy}
  failed=False
  try:o._authenticate(lambda root,r,as_json:copy.deepcopy(docs[r["path"]]))
  except(ValueError,KeyError,TypeError):failed=True
  need(failed==(label!="valid"),"Exact new capability and current selector metadata gate "+label)
  metadata_reports.append(dict(case=label,status="PASS",failure_expected=failed))
 locked[0]=False
 # Actual accepted bridge operand transformation executes with RAM source/callable readers only.
 bridge_ns={"__name__":"RAM_accepted_FLARE_bridge"};bridge_file=ROOT/ns["BRIDGE_REF"]["path"]
 need(sha(bridge_file)==ns["BRIDGE_REF"]["sha256"],"Accepted bridge exact SHA")
 exec(compile(bridge_file.read_text(),"<inert-accepted-FLARE-bridge>","exec"),bridge_ns)
 base=SimpleNamespace(**mn);seen=[]
 base.functions_from_source=lambda module,record,names:(seen.append(record)or{})
 base.source_bytes=lambda record:b"RAM separately pinned source reader"
 memo_source=memo_path.read_bytes()
 methods,counts=bridge_ns["derive_memo_methods_v1"](base,memo_source,ns["GUARD_REF"],(),ns["BRIDGE_REF"])
 guard_stub=SimpleNamespace();v4_stub=SimpleNamespace(bound=lambda *a:dict(source_sha256={mn["SOURCE"]:ns["MEMO_REF"]["sha256"]}))
 methods["_check"](SimpleNamespace(guard=guard_stub,v4=v4_stub,budget=SimpleNamespace(),manifest_ref={},adapter_ref=ns["MEMO_REF"]))
 need(counts==[1,1]and seen[0]==ns["GUARD_REF"],"Actual derived memo checks exact current Census rather than old Relate source")
 need(not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit")),"No scientific imports")
 return dict(schema="Census_inline_independent_adversarial_RAM_checks_v1",status="INDEPENDENT_ADVERSARIAL_CASES_USING_DISCLOSED_AUTHOR_RAM_SCAFFOLD",cases=len(reports),reports=reports,metadata_cases=len(metadata_reports),metadata_reports=metadata_reports,accepted_current_Census_bridge_operand_join=True,entire_original_AST_restored=True,exactly_three_declared_AST_additions=True,original_guard_and_memo2_hashes_verified=True,all47roles_prefix_and64cap_before_first_write=True,actual_guard_state_mutex_process_resource_measurement_or_mutation_calls=False,performance_or_deadline_fit_measured=False,source_template_not_runtime_qualification=True)
if __name__=="__main__":print(json.dumps(run(),indent=2,sort_keys=True))
