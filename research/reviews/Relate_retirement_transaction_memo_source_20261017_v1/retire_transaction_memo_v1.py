"""Inactive exact failed-Relate retention transaction; original300s lifetime and authority unchanged."""
from pathlib import Path
from contextlib import contextmanager
import ast,copy,hashlib,importlib.util,json,os,stat,sys,threading
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/Relate_retirement_transaction_memo_source_20261017_v1"
SOURCE=HERE+"/retire_transaction_memo_v1.py"
MEMO_REF=dict(path="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/root_mutation_memo_v1.py",sha256="487fcd93b73f6668126522b67720041abbbaa8b95ee54c3bc69adffba1a765b2")
MEMO_MANIFEST_REF={'path': 'research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/source_manifest.v1.json', 'sha256': '7f6a55db07b0967bdf6000e40c1c4eeccca8e67a4050f27cf9732d2f1140eb1c'}
GUARD_REF=dict(path="research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py",sha256="b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164")
TARGET_KEY="ARGMix/prospective1611-accepted-TRAIN000-Relate-v4"
RETIRE_OPERATION="retire_exact_failed_Relate64_retention_v1"
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
 spec=importlib.util.spec_from_file_location("_Relate_retirement_exact_accepted_memo",p)
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;exec(compile(raw,str(p),"exec",dont_inherit=True),module.__dict__);return module
def derive_retirement(node,scope,transaction,append):
 """Entire original AST retained, including prelock deadline; only two explicit additions."""
 need(node.name=="apply_retire_stopped_v1"and len(node.body)==5,"Exact original retirement function shape")
 block=node.body[-1]
 need(isinstance(block,ast.With)and len(block.items)==1 and block.items[0].optional_vars is None
  and ast.unparse(block.items[0].context_expr)=="budget.v4.mutex(ROOT)","Original actual mutex stays in original position")
 derived=copy.deepcopy(node);derived.name="_memo_exact_retirement"
 calls=[n for n in ast.walk(derived.body[-1])if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="_append"]
 need(len(calls)==1 and len(calls[0].args)==4 and not calls[0].keywords,"Exactly one original append")
 calls[0].func.id="_retirement_append";calls[0].args[0:0]=[ast.Name(id="finite",ctx=ast.Load()),ast.Name(id="auth",ctx=ast.Load())]
 original_body=derived.body[-1].body
 derived.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="_retirement_transaction",ctx=ast.Load()),
  args=[ast.Name(id="finite",ctx=ast.Load())],keywords=[]),optional_vars=None)],body=original_body)]
 derived.decorator_list=[]
 namespace=dict(scope,_retirement_transaction=transaction,_retirement_append=append,__name__=__name__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),
  SOURCE+"#entire-original-retirement-with-two-additions","exec"),namespace)
 return namespace[derived.name]
class RetireStoppedMemo:
 """One public exact original-key operation, owning its ORIGINAL mutex; no held-lock API."""
 def __init__(self,guard,*,integration_manifest_ref,helper_ref,review_ref,memo_review_ref,root_instruction_ref,cache_refs,uncached_current_refs=()):
  global _ACTIVE
  need(not _ACTIVE,"No construction during an active retirement transaction")
  self.memo=load_memo();self.guard=guard;self.budget=guard.budget;self.v4=self.budget.v4
  self.manifest_ref=copy.deepcopy(integration_manifest_ref);self.helper_ref=copy.deepcopy(helper_ref)
  self.review_ref=copy.deepcopy(review_ref);self.memo_review_ref=copy.deepcopy(memo_review_ref);self.instruction_ref=copy.deepcopy(root_instruction_ref)
  self.cache_refs=copy.deepcopy(cache_refs);self.current_refs=copy.deepcopy(list(uncached_current_refs))
  need(len(self.current_refs)<=2048 and all(isinstance(r,dict)and set(r)=={"path","sha256"}for r in self.current_refs),
   "Explicit bounded current-evidence exclusions")
  self.counts={};self._tx=None;self._used=False
  need(self.helper_ref["path"]==SOURCE,"Exact new retirement helper path")
  # Reuse accepted constructor/fingerprint/cache-ref validation and read-state API.
  # Retirement joins are OUT OF BAND ROOT refs, never added to old reservation/authority.
  self.core=self.memo.RootMutationMemo(guard,integration_manifest_ref=self.manifest_ref,
   adapter_ref=MEMO_REF,cache_refs=self.cache_refs)
  self._check()
 def _authenticate(self,bound):
  manifest=bound(ROOT,self.manifest_ref,True)
  need(manifest.get("source_sha256",{}).get(SOURCE)==self.helper_ref["sha256"]
   and manifest.get("source_sha256",{}).get(MEMO_REF["path"])==MEMO_REF["sha256"]
   and manifest.get("source_sha256",{}).get(MEMO_MANIFEST_REF["path"])==MEMO_MANIFEST_REF["sha256"],"Exact out-of-band helper/memo manifest")
  review=bound(ROOT,self.review_ref,True)
  need(review.get("independent")is True and review.get("verdict")in("SOURCE_PASS","PASS")
   and review.get("source_ref")==self.helper_ref and review.get("source_manifest_ref")==self.manifest_ref
   and review.get("blockers")==[],"Independent complete retirement helper source review")
  memo_review=bound(ROOT,self.memo_review_ref,True)
  need(memo_review.get("independent")is True and memo_review.get("verdict")in("SOURCE_PASS","PASS")
   and memo_review.get("source_ref")==MEMO_REF and memo_review.get("source_manifest_ref")==MEMO_MANIFEST_REF
   and memo_review.get("blockers")==[],
   "Separately reviewed exact metadata memo dependency")
  instruction=bound(ROOT,self.instruction_ref,True)
  need(instruction.get("schema")=="ROOT_exact_Relate64_retirement_transaction_instruction_v1"
   and instruction.get("root_authorizer")=="root_agent"and instruction.get("active")is True
   and instruction.get("helper_ref")==self.helper_ref and instruction.get("source_manifest_ref")==self.manifest_ref
   and instruction.get("independent_review_ref")==self.review_ref and instruction.get("memo_ref")==MEMO_REF
   and instruction.get("memo_review_ref")==self.memo_review_ref
   and instruction.get("guard_ref")==GUARD_REF and instruction.get("key")==TARGET_KEY
   and instruction.get("operation")==RETIRE_OPERATION and instruction.get("cache_refs")==self.cache_refs
   and instruction.get("uncached_current_refs")==self.current_refs
   and instruction.get("operation_wall_seconds")==300 and instruction.get("original_authority_schema_unchanged")is True
   and instruction.get("original_reservation_files_caps_expiry_and_frozen_ancestor_journals_unchanged")is True
   and instruction.get("new_accounting_lineage")is False and instruction.get("automatic_retry")is False,
   "Actual immutable ROOT instruction, not a new accounting authority/schema/clock")
  self.guard._expire(instruction)
  self.memo.source_bytes(self.helper_ref);self.memo.source_bytes(MEMO_REF)
  return instruction
 def _check(self):
  self.core._check()
  need(not _ACTIVE and not self.memo._ACTIVE and threading.active_count()==1,"Fresh nonnested single-thread ROOT transaction")
  need(self.guard.RETIRE_OPERATION==RETIRE_OPERATION and self.guard.TARGET_KEYS==(TARGET_KEY,)
   and self.guard.OPERATION_WALL_SECONDS==300,"Exact original failed64GiB Relate retirement/cap only")
  self.functions=self.memo.functions_from_source(self.guard,GUARD_REF,("apply_retire_stopped_v1","_state_locked","_append"))
  self._authenticate(self.v4.bound)
 @contextmanager
 def _transaction(self,finite):
  need(self._tx is None,"One bounded cache generation");finite()
  # Read current authority through the ORIGINAL uncached reader only to obtain
  # exclusion paths. All original exact authority/schema checks still follow.
  authority=self.v4.bound(ROOT,self._authority_ref,True);request=authority.get("request",{})
  need(request.get("key")==TARGET_KEY,"Only original failed64GiB key")
  records=[self._authority_ref,self.manifest_ref,self.review_ref,self.memo_review_ref,self.instruction_ref,*self.current_refs]
  for name in("revocation_ref","observation_ref"):
   record=request.get(name);need(isinstance(record,dict)and set(record)=={"path","sha256"},"Current original evidence ref")
   records.append(record)
  finite()
  tx=self.memo._Transaction(ROOT,self.v4,self.budget,self.core.allow,uncached_paths=[r["path"]for r in records]);self._tx=tx
  try:
   tx.install();yield
  finally:
   try:self.counts=dict(tx.counts);tx.close()
   finally:self._tx=None
 def _append(self,finite,auth,operation,record,value,state):
  tx=self._tx
  need(tx is not None and self._append_calls==0 and operation==RETIRE_OPERATION
   and record==self._authority_ref and value["request"]["key"]==TARGET_KEY,"One exact original-key append only")
  self._append_calls+=1
  # Current authority/proof/revocation/observation and raw journal/fingerprint/
  # inventory/stat/process reads stay uncached. Only immutable exact-bound
  # historical metadata/code bytes may memoize; predicates rerun every time.
  # Every cached byte/path is revalidated originally and uncached before append.
  instruction=self._authenticate(tx.original["bound"])
  tx.finish()
  self.guard._expire(auth);self.guard._expire(instruction)
  finite() # EXACT original deadline closure, AFTER revalidation, BEFORE original append.
  return self._original_append(operation,record,value,state)
 def read_state_v1(self,*,expected_source_sha256,expected_launcher_sha256):
  self._check()
  return self.core.read_state_v1(expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
 def apply_retire_stopped_v1(self,authority_ref,*,expected_source_sha256,expected_launcher_sha256):
  global _ACTIVE
  self._check();need(not self._used,"No automatic replay on this helper instance")
  self._used=True;self._append_calls=0;self._authority_ref=copy.deepcopy(authority_ref)
  self._original_append=self.guard._append
  entire=derive_retirement(self.functions["apply_retire_stopped_v1"][1],self.guard.__dict__,self._transaction,self._append)
  _ACTIVE=True;self.memo._ACTIVE=True
  try:
   result=entire(authority_ref,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
   need(self._append_calls==1,"Original exact retirement must append once");return result
  finally:self.memo._ACTIVE=False;_ACTIVE=False
if __name__=="__main__":
 print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",guard_imported=False,operations=False)))
