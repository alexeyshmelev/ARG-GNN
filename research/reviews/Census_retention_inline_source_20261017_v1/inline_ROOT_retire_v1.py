"""Inactive ROOT-only just-in-time immutable evidence, exact Census observation retention.

No import IO, guard/resource/mutex/process observation or accounting operation.
An actual callback, whole-join review and fresh finite ROOT instruction are required.
"""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,os,stat,sys,threading
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/Census_retention_inline_source_20261017_v1"
SOURCE=HERE+"/inline_ROOT_retire_v1.py"
MEMO_REF=dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/root_mutation_memo_v1.py",sha256="487fcd93b73f6668126522b67720041abbbaa8b95ee54c3bc69adffba1a765b2")
MEMO_MANIFEST_REF=dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/source_manifest.v1.json",sha256="7f6a55db07b0967bdf6000e40c1c4eeccca8e67a4050f27cf9732d2f1140eb1c")
GUARD_REF=dict(path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py",sha256="72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb")
TARGET_KEYS=("census_tsinfer_observation_v1/census-tsinfer-inspect-v1","census_tsinfer_observation_v1/census-tsinfer-numerical-v1")
SELECTOR_REF=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
POLICY_REF=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
BRIDGE_REF=dict(path="research/reviews/FLARE_original_VAL002004_conversion_execution_source_20261017_v1/root_memo_bridge_v1.py",sha256="d5d56b504f268c3aa7c4679df47d1c9eddd21d6e0c370c5671f6b26c8f64e2ed")
RETIRE_OPERATION="retire_exact_Census_observation_retained_allowance_v1"
CALLBACK_NAME="authorize_inline_ROOT_retirement_v1"
CALLBACK_STDLIB_MODULES=frozenset(("__future__","copy","datetime","errno","hashlib","json","os","pathlib","stat","time"))
_ACTIVE=False
def need(ok,message):
 if not ok:raise ValueError(message)
def load_memo():
 """Only explicit construction reads/imports this pinned stdlib-only accepted helper."""
 p=ROOT/MEMO_REF["path"];before=p.lstat()
 need(p.resolve()==p and stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,"Exact bounded memo source")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:opened=os.fstat(f.fileno());raw=f.read(2**20+1);after=os.fstat(f.fileno())
 identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(identity(before)==identity(opened)==identity(after)==identity(p.lstat())
  and len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==MEMO_REF["sha256"],"Stable exact accepted memo bytes")
 spec=importlib.util.spec_from_file_location("_Census_retirement_exact_accepted_memo",p)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;exec(compile(raw,str(p),"exec",dont_inherit=True),module.__dict__);return module

def derive_inline_retirement(node,scope,transaction,authority,append):
 """Original AST plus exactly three reversible additions; deadline stays prelock."""
 need(node.name=="apply_retire_stopped_v1"and len(node.body)==5,"Exact original retirement function shape")
 block=node.body[-1]
 need(isinstance(block,ast.With)and len(block.items)==1 and block.items[0].optional_vars is None
  and ast.unparse(block.items[0].context_expr)=="budget.v4.mutex(ROOT)","Original actual mutex stays in original position")
 need(ast.unparse(block.body[0])=="finite()"
  and ast.unparse(block.body[1])=="state = _state_locked(expected_source_sha256, expected_launcher_sha256, fresh=True)"
  and ast.unparse(block.body[2])=="auth = _authority(authority_ref, RETIRE_OPERATION, state[3], fresh=True)",
  "One exact after-fresh-state, before-authority ROOT seam")
 derived=copy.deepcopy(node);derived.name="_inline_ROOT_exact_retirement"
 calls=[n for n in ast.walk(derived.body[-1])if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_append"]
 need(len(calls)==1 and len(calls[0].args)==4 and not calls[0].keywords,"Exactly one original append")
 calls[0].func.id="_retirement_append";calls[0].args[0:0]=[ast.Name(id="finite",ctx=ast.Load()),ast.Name(id="auth",ctx=ast.Load())]
 original_body=derived.body[-1].body
 original_body.insert(2,ast.parse("authority_ref = _inline_ROOT_authority(state, finite)").body[0])
 derived.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="_retirement_transaction",ctx=ast.Load()),
  args=[ast.Name(id="finite",ctx=ast.Load())],keywords=[]),optional_vars=None)],body=original_body)]
 derived.decorator_list=[]
 namespace=dict(scope,_retirement_transaction=transaction,_inline_ROOT_authority=authority,_retirement_append=append,__name__=__name__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),
  SOURCE+"#entire-original-retirement-with-three-additions","exec"),namespace)
 return namespace[derived.name]

class InlineRootRetireStoppedMemo:
 """One attempt under the ORIGINAL mutex; ROOT owns the reviewed real callback."""
 def __init__(self,guard,*,root_instruction_ref):
  need(not _ACTIVE,"No construction during an active retirement transaction")
  self.memo=load_memo();self.guard=guard;self.budget=guard.budget;self.v4=self.budget.v4
  self.instruction_ref=copy.deepcopy(root_instruction_ref)
  spec=self.v4.bound(ROOT,self.instruction_ref,True)
  self.key=spec["key"];need(self.key in TARGET_KEYS,"Only two exact original20GB observation scopes")
  self.manifest_ref=copy.deepcopy(spec["source_manifest_ref"]);self.helper_ref=copy.deepcopy(spec["helper_ref"])
  self.review_ref=copy.deepcopy(spec["whole_join_review_ref"]);self.memo_review_ref=copy.deepcopy(spec["memo_review_ref"])
  self.callback_ref=copy.deepcopy(spec["callback_source_ref"]);self.callback_name=spec["callback_name"]
  self.cache_refs=copy.deepcopy(spec["cache_refs"]);self.current_refs=copy.deepcopy(spec["uncached_current_refs"])
  self.paths=copy.deepcopy(spec["fresh_metadata_paths"]);self.metadata_directory=spec["metadata_directory"]
  self.revocation_ref=copy.deepcopy(spec["permanent_revocation_ref"])
  self._tx=None;self._used=False;self._callback_calls=0;self._append_calls=0;self._authority_ref=None;self.counts={}
  self.callback_module=self.callback=None
  bridge_raw=self.memo.source_bytes(BRIDGE_REF)
  bridge=type(sys)("_Census_exact_accepted_FLARE_memo_bridge")
  exec(compile(bridge_raw,str(ROOT/BRIDGE_REF["path"]),"exec",dont_inherit=True,optimize=0),bridge.__dict__)
  memo_source=self.memo.source_bytes(MEMO_REF)
  methods,counts=bridge.derive_memo_methods_v1(self.memo,memo_source,GUARD_REF,(),BRIDGE_REF)
  need(counts==[1,1],"Exact accepted current-Census authentication bridge")
  Memo=type("ExactCensusRootMutationMemoV2",(self.memo.RootMutationMemo,),methods)
  self.core=Memo(guard,integration_manifest_ref=self.manifest_ref,adapter_ref=MEMO_REF,cache_refs=self.cache_refs)
  self._check(check_callback=False)
  raw=self._callback_bytes();p=ROOT/self.callback_ref["path"]
  name="_exact_ROOT_inline_retirement_authorizer"
  module=importlib.util.module_from_spec(importlib.util.spec_from_file_location(name,p));sys.modules[name]=module
  exec(compile(raw,str(p),"exec",dont_inherit=True,optimize=0),module.__dict__)
  self.callback_module=module;self.callback=getattr(module,self.callback_name)
  self._check()
 def _authenticate(self,bound):
  selector=bound(ROOT,SELECTOR_REF,True);policy=bound(ROOT,POLICY_REF,True)
  need(selector.get("source_ref")==GUARD_REF and selector.get("policy_ref")==POLICY_REF
   and policy.get("source_ref")==GUARD_REF and policy.get("active")is True,"Exact current Census selector and active policy revalidated uncached")
  manifest=bound(ROOT,self.manifest_ref,True)
  need(manifest.get("source_sha256",{}).get(SOURCE)==self.helper_ref["sha256"]
   and manifest["source_sha256"].get(MEMO_REF["path"])==MEMO_REF["sha256"]
   and manifest["source_sha256"].get(MEMO_MANIFEST_REF["path"])==MEMO_MANIFEST_REF["sha256"]
   and manifest["source_sha256"].get(BRIDGE_REF["path"])==BRIDGE_REF["sha256"],"Exact new helper and unchanged memo2 source closure")
  need(self.helper_ref["path"]==SOURCE,"Exact new helper path; no old attempt substitution")
  need(self.callback_name==CALLBACK_NAME,"One exact ROOT authorizer callback name")
  review=bound(ROOT,self.review_ref,True)
  need(review.get("schema")=="Census_inline_ROOT_retirement_whole_join_review_v1"
   and review.get("independent")is True and review.get("verdict")=="SOURCE_PASS"and review.get("blockers")==[]
   and review.get("source_ref")==self.helper_ref and review.get("source_manifest_ref")==self.manifest_ref
   and review.get("callback_source_ref")==self.callback_ref and review.get("callback_name")==self.callback_name
   and review.get("fresh_metadata_paths")==self.paths and review.get("permanent_revocation_ref")==self.revocation_ref
   and review.get("guard_ref")==GUARD_REF and review.get("memo_ref")==MEMO_REF
   and review.get("selector_ref")==SELECTOR_REF and review.get("policy_ref")==POLICY_REF and review.get("bridge_ref")==BRIDGE_REF
   and review.get("actual_callback_complete_source_review")is True,
   "Genuine new whole-join review includes actual ROOT callback; source-template review never suffices")
  memo_review=bound(ROOT,self.memo_review_ref,True)
  need(memo_review.get("independent")is True and memo_review.get("verdict")=="SOURCE_PASS"
   and memo_review.get("source_ref")==MEMO_REF and memo_review.get("source_manifest_ref")==MEMO_MANIFEST_REF
   and memo_review.get("blockers")==[],"Separately accepted exact unchanged memo2 review")
  instruction=bound(ROOT,self.instruction_ref,True)
  expected=dict(schema="ROOT_exact_Census_inline_evidence_instruction_v1",root_authorizer="root_agent",active=True,
   helper_ref=self.helper_ref,source_manifest_ref=self.manifest_ref,whole_join_review_ref=self.review_ref,
   memo_ref=MEMO_REF,memo_manifest_ref=MEMO_MANIFEST_REF,memo_review_ref=self.memo_review_ref,guard_ref=GUARD_REF,
   bridge_ref=BRIDGE_REF,selector_ref=SELECTOR_REF,policy_ref=POLICY_REF,
   expected_before=instruction.get("expected_before"),expected_guard_roles=instruction.get("expected_guard_roles"),
   key=self.key,operation=RETIRE_OPERATION,callback_source_ref=self.callback_ref,callback_name=CALLBACK_NAME,
   metadata_directory=self.metadata_directory,fresh_metadata_paths=self.paths,permanent_revocation_ref=self.revocation_ref,
   cache_refs=self.cache_refs,uncached_current_refs=self.current_refs,
   created_at_utc=instruction.get("created_at_utc"),expires_at_utc=instruction.get("expires_at_utc"),
   operation_wall_seconds=300,fresh_target_seconds=120,original_authority_and_request_schemas_unchanged=True,
   original_reservation_files_caps_expiry_and_frozen_ancestor_journals_unchanged=True,
   ROOT_authorizes_one_actual_callback_after_original_fresh_state=True,
   callback_only_actual_target_observations_and_new_immutable_ROOT_metadata=True,
   original_two_pass_measurement_and_raw_final_revalidation_unchanged=True,
   new_accounting_lineage=False,automatic_retry=False,old_instruction_or_failed_authority_reused=False)
  need(instruction==expected and set(instruction)==set(expected) and instruction["key"]in TARGET_KEYS,"Exact fresh ROOT-only one-attempt instruction; never accounting authority")
  need(review.get("key")==self.key and review.get("expected_before")==instruction["expected_before"]
   and review.get("expected_guard_roles")==instruction["expected_guard_roles"],"Whole review binds exact target and expected state/47roles")
  created=datetime.fromisoformat(instruction["created_at_utc"])
  need(created.tzinfo is not None and created<=datetime.now(timezone.utc),"Genuine finite nonfuture ROOT instruction")
  self.guard._expire(instruction)
  need(isinstance(instruction["expected_before"],dict)and isinstance(instruction["expected_guard_roles"],dict)
   and len(instruction["expected_guard_roles"])==47,"Exact current prefix and all47roles in instruction")
  need(isinstance(self.metadata_directory,str)and self.metadata_directory.startswith("research/reviews/Census_retention_inline_ROOT_actual_")
   and str(Path(self.metadata_directory))==self.metadata_directory,"Separately fresh explicit ROOT metadata namespace")
  directory=self.v4.canonical(ROOT,self.metadata_directory)
  need((ROOT/self.instruction_ref["path"]).parent==directory,
   "Fresh ROOT instruction itself belongs to the same explicit new preparation")
  need(directory.is_dir()and set(self.paths)=={"attempt","observation","authority"}
   and all(isinstance(p,str)and str(Path(p))==p and not Path(p).is_absolute()and (ROOT/p).parent==directory
    and Path(p).suffix==".json"for p in self.paths.values())and len(set(self.paths.values()))==3,
   "Three distinct predetermined fresh flat JSON paths in existing ROOT preparation")
  need(isinstance(self.callback_ref,dict)and set(self.callback_ref)=={"path","sha256"}
   and (ROOT/self.callback_ref["path"]).parent==directory and Path(self.callback_ref["path"]).suffix==".py",
   "Actual reviewed ROOT callback lives in this new ROOT preparation")
  need(len(self.current_refs)<=2048 and all(isinstance(r,dict)and set(r)=={"path","sha256"}for r in self.current_refs),"Explicit bounded current-evidence exclusions")
  for r in (self.helper_ref,MEMO_REF,BRIDGE_REF,self.callback_ref):self.memo.source_bytes(r)
  return instruction
 def _check(self,check_callback=True):
  self.core._check()
  need(not _ACTIVE and not self.memo._ACTIVE and threading.active_count()==1,"Fresh nonnested single-thread ROOT operation")
  need(self.guard.RETIRE_OPERATION==RETIRE_OPERATION and self.guard.TARGET_KEYS==TARGET_KEYS and self.guard._SELECTOR_REF==SELECTOR_REF and self.guard._POLICY_REF==POLICY_REF
   and self.guard.OPERATION_WALL_SECONDS==300 and self.guard.FRESH_TARGET_SECONDS==120,"Exact original key,300s operation and120s target proof")
  self.functions=self.memo.functions_from_source(self.guard,GUARD_REF,("apply_retire_stopped_v1","_state_locked","_append",
   "_authority","_stopped_proof","_observe_targets","_absent","_target","_pins","operation_draft_v1","stopped_observation_draft_v1"))
  self._authenticate(self.v4.bound)
  if check_callback:self._check_callback()
 def _callback_bytes(self):
  """Actual callback import is inert: one definition and stdlib imports only."""
  raw=self.memo.source_bytes(self.callback_ref);tree=ast.parse(raw)
  functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
  need(len(functions)==1 and functions[0].name==CALLBACK_NAME,"One actual ROOT callback definition only")
  fn=functions[0];args=fn.args
  need(not fn.decorator_list and fn.returns is None and not args.posonlyargs and not args.args
   and args.vararg is None and args.kwarg is None and not args.defaults
   and [a.arg for a in args.kwonlyargs]==["guard","context","finite"]
   and all(a.annotation is None for a in args.kwonlyargs)and args.kw_defaults==[None,None,None],
   "Exact inert keyword-only callback signature; no evaluated defaults/annotations/decorators")
  for node in tree.body:
   if isinstance(node,ast.Expr)and isinstance(node.value,ast.Constant)and isinstance(node.value.value,str):continue
   if node is fn:continue
   if isinstance(node,ast.Import):
    need(all(a.name.split(".")[0]in CALLBACK_STDLIB_MODULES for a in node.names),"Actual ROOT callback imports stdlib only")
   elif isinstance(node,ast.ImportFrom):
    need(node.level==0 and node.module and node.module.split(".")[0]in CALLBACK_STDLIB_MODULES,"Actual ROOT callback imports stdlib only")
   else:raise ValueError("No actual callback top-level execution or extra source dependencies")
  return raw
 def _check_callback(self):
  self._callback_bytes()
  found=self.memo.functions_from_source(self.callback_module,self.callback_ref,(self.callback_name,))
  need(found[self.callback_name][0]is self.callback and not hasattr(self.callback,"__wrapped__"),"Exact independently reviewed ROOT callback code object")
 def _put_attempt(self,finite):
  """Small out-of-band ROOT one-attempt token; no accounting authority or credit."""
  finite();p=self.v4.canonical(ROOT,self.paths["attempt"])
  data=(json.dumps(dict(schema="ROOT_inline_Census_single_attempt_consumed_v1",root_instruction_ref=self.instruction_ref,
   callback_source_ref=self.callback_ref,fresh_metadata_paths=self.paths,consumed_utc=self.guard._utc(),
   operation=RETIRE_OPERATION,key=self.key,automatic_retry=False),sort_keys=True)+"\n").encode()
  need(len(data)<=self.guard.MAX_JSON,"Original compact ROOT metadata byte limit")
  fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,"wb")as stream:
   stream.write(data);stream.flush();os.fsync(stream.fileno());opened=os.fstat(stream.fileno())
  named=p.lstat();need(stat.S_ISREG(named.st_mode)and named.st_nlink==1 and(named.st_dev,named.st_ino)==(opened.st_dev,opened.st_ino),"Exact immutable new attempt record")
  fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(fd)
  finally:os.close(fd)
  finite()
 @contextmanager
 def _transaction(self,finite):
  need(self._tx is None,"One original-lock cold cache generation");finite()
  self._authenticate(self.v4.bound)
  records=[self.manifest_ref,self.review_ref,self.memo_review_ref,self.instruction_ref,self.callback_ref,self.revocation_ref,*self.current_refs]
  manifest=self._tx_manifest_for_exclusions()
  exclusions=[r["path"]for r in records]+list(self.paths.values())+[p for p in manifest["source_sha256"]if Path(p).suffix==".json"]
  # Every configured ancestor policy/selector/control and nested JSON ref stays uncached.
  owner=self.guard;seen=set()
  def collect(v):
   if type(v)is dict:
    if set(v)=={"path","sha256"}and Path(v["path"]).suffix==".json":exclusions.append(v["path"])
    else:
     for child in v.values():collect(child)
   elif type(v)in(list,tuple):
    for child in v:collect(child)
  while owner is not None:
   need(id(owner)not in seen and len(seen)<16,"Finite acyclic configured parent chain");seen.add(id(owner))
   for name in("_POLICY_REF","_SELECTOR_REF","_REPAIR_AUTHORITY_REF"):
    record=getattr(owner,name,None)
    if record is not None:collect(record);collect(self.v4.bound(ROOT,record,True))
   owner=getattr(owner,"_BASE",None)
  need(len(set(exclusions))<=2048,"All current paths explicitly excluded within original bound")
  tx=self.memo._Transaction(ROOT,self.v4,self.budget,self.core.allow,uncached_paths=exclusions);self._tx=tx
  try:
   tx.install()
   # Explicit exclusions exist before FIRST staging write. Originals canonical,
   # file_bytes and mutex are unchanged by accepted memo2.
   need(all(not self.v4.canonical(ROOT,p).exists()for p in self.paths.values()),"Never reuse any earlier ROOT attempt/evidence/authority path")
   yield
  finally:
   try:self.counts=dict(tx.counts);tx.close()
   finally:self._tx=None
 def _tx_manifest_for_exclusions(self):
  return self.v4.bound(ROOT,self.manifest_ref,True)
 def _inline_ROOT_authority(self,state,finite):
  need(self._tx is not None and self._callback_calls==0 and self._authority_ref is None,"One inline ROOT callback after original state")
  finite();self._authenticate(self._tx.original["bound"]);self._check_callback()
  instruction=self._authenticate(self._tx.original["bound"])
  need(self.guard._pins(state[2])==instruction["expected_before"]and state[3]==instruction["expected_guard_roles"]
   and len(state[3])==47,"Same exact expected prefix and every47 guard role before any write")
  self.guard.assert_open(self.key,state[0],state[1]);own=state[0][self.key]
  need(own==self.guard._target(self.key)["original_reservation"],"Same exact original observation reservation")
  # Validate the full47-role union and future immutable names BEFORE the attempt/observation/authority writes.
  existing=self.guard.consumer_bindings_v1(list(state[3].values()),
   [self.instruction_ref,self.manifest_ref,self.review_ref,self.memo_review_ref,self.helper_ref,MEMO_REF,BRIDGE_REF,SELECTOR_REF,POLICY_REF,self.callback_ref,self.revocation_ref])
  need(all(p not in{r["path"]for r in existing}for p in self.paths.values())and len(existing)+len(self.paths)<=64,
   "All existing exact refs plus three distinct future records must fit64 before writes; no fabricated future hashes")
  finite();self._put_attempt(finite)
  context=dict(key=self.key,original_reservation=copy.deepcopy(own),expected_before=copy.deepcopy(self.guard._pins(state[2])),
   bindings=copy.deepcopy(state[3]),root_instruction_ref=copy.deepcopy(self.instruction_ref),
   fresh_metadata_paths=copy.deepcopy(self.paths),permanent_revocation_ref=copy.deepcopy(self.revocation_ref))
  observed_not_before=datetime.now(timezone.utc)
  self._callback_calls+=1
  record=self.callback(guard=self.guard,context=context,finite=finite)
  finite();self._check_callback()
  need(isinstance(record,dict)and set(record)=={"path","sha256"}and record["path"]==self.paths["authority"],"Only NEW predetermined original-schema authority ref")
  obs_ref=self._tx.original["reference"](ROOT,self.paths["observation"])
  obs=self._tx.original["bound"](ROOT,obs_ref,True)
  auth=self._tx.original["bound"](ROOT,record,True)
  observed_end=datetime.fromisoformat(obs["observed_at_utc"])
  need(observed_end.tzinfo is not None and observed_not_before<=observed_end<=datetime.now(timezone.utc)
   and all(observed_not_before<=datetime.fromisoformat(row[-1])<=observed_end
    for field in("PID_observations","group_observations")for row in obs[field]),
   "Every actual ROOT observation is newly timestamped AFTER original fresh state and before this authority check")
  need(obs.get("revocation_ref")==self.revocation_ref and auth.get("request")==dict(key=self.key,
   reservation_sha256=self.budget.digest(own),revocation_ref=self.revocation_ref,observation_ref=obs_ref)
   and auth.get("expected_before")==self.guard._pins(state[2])and auth.get("bindings")==state[3],
   "NEW immutable actual observation and authority bind SAME original fresh state; no schema extensions")
  self._authority_ref=copy.deepcopy(record)
  finite();return record
 def _append(self,finite,auth,operation,record,value,state):
  tx=self._tx
  need(tx is not None and self._append_calls==0 and self._callback_calls==1
   and operation==RETIRE_OPERATION and record==self._authority_ref and record["path"]==self.paths["authority"]
   and value["request"]["key"]==self.key,"One original append through SAME new authority")
  self._append_calls+=1
  instruction=self._authenticate(tx.original["bound"]);self._check_callback()
  tx.finish() # disable hits, revalidate EVERY admitted cached byte/name originally.
  # Restore every original before final current metadata/source/proof validation.
  tx.close();self.core._check()
  instruction=self._authenticate(self.v4.bound);self._check_callback()
  fresh_auth=self.guard._authority(record,RETIRE_OPERATION,state[3],fresh=True)
  need(fresh_auth==auth,"Immutable current authority unchanged after raw validation")
  self.guard._recorded_stopped_proof(value["proof"],self.guard._stopped_proof(value["request"],state[0][self.key]),value["closed_utc"])
  self.guard._expire(auth);self.guard._expire(instruction)
  finite() # SAME original prelock deadline, after final raw validation, before append.
  tail=self._original_append(operation,record,value,state)
  finite() # A blocking original fsync/postread may overrun; never return a success receipt then.
  return tail
 def apply_retire_stopped_v1(self,*,expected_source_sha256,expected_launcher_sha256):
  global _ACTIVE
  self._check();need(not self._used,"One operation call per helper; immutable token also forbids new-instance replay")
  self._used=True;self._original_append=self.guard._append
  entire=derive_inline_retirement(self.functions["apply_retire_stopped_v1"][1],self.guard.__dict__,self._transaction,self._inline_ROOT_authority,self._append)
  _ACTIVE=True;self.memo._ACTIVE=True
  try:
   result=entire(None,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
   need(self._callback_calls==self._append_calls==1,"One actual ROOT callback and original append only")
   return result
  finally:self.memo._ACTIVE=False;_ACTIVE=False
if __name__=="__main__":
 print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",guard_imported=False,callback_loaded=False,operations=False)))
