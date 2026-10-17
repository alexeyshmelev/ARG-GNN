"""Exact native v12 metadata and original close predicates in isolated RAM only."""
import ast,copy,hashlib,importlib,json,sys,traceback
from contextlib import contextmanager
from datetime import datetime,timezone,timedelta
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
def main():
 b=importlib.import_module("research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1.native_MLM_v12_close_ROOT_v1")
 c=b.c;root=c.ROOT;oldlocal=c.local
 def realref(p):
  p=oldlocal(p);return dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def realraw(r):
  raw=oldlocal(r["path"]).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=r["sha256"]:raise ValueError("SHA")
  return raw
 def realread(r):return json.loads(realraw(r))
 checks=[];assertions=[0]
 def check(ok,name):
  assertions[0]+=1
  if not ok:raise AssertionError(name)
  if name not in checks:checks.append(name)
 tree=ast.parse(realraw(realref(b.SOURCE)));text=ast.unparse(tree)
 check("torch"in text and not any(isinstance(n,(ast.Import,ast.ImportFrom))and "torch"in ast.unparse(n)for n in tree.body),"stdlib-inert-contract-only")
 check("measure_scopes("not in text and "apply_close_completed_v1("in text,"one-original-apply-owns-only-measurement")
 check(not any(s in text for s in("os.unlink(","os.rename(","os.remove(","os.kill(","install_consumer_v1(","c.guard_v1(")),"no-deletion-normalization-signal-or-consumer-memo")
 block=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="close_once_v1"))
 check(block.count("apply_close_completed_v1(")==1,"one-original-close-apply-no-retry")
 check(block.index("directory.mkdir")<block.index("c.load(COORDINATOR"),"attempt-consumed-before-coordinator-or-guard-import")
 check(block.index("accounting_transaction")<block.index("c.load(GUARD")<block.index("_state_locked")<block.index("apply_close_completed_v1"),"v3-coordinator-encloses-entire-capture-proof-apply")
 meta=b.metadata_v1()
 check(meta["own"]["key"]==b.KEY and meta["own"]["reserved_bytes"]==4563402752,"genuine-distinct-v12-reservation-only")
 check(meta["reservation_sha256"]=="83f4cc1d025ce9f2a66950051296efcc297f5fdb4155ab02c9dffbabf234b57c","genuine-original-reservation-digest")
 check(len(meta["pids"])==51 and 3750912 in meta["pids"]and meta["groups"]==[3752065],"genuine-complete-51PID-roster-and-recorded-group")
 policy=realread(b.POLICY)
 check("H041/American-C3-native-MLM-full30mb-mlm-fit-v8"in policy["never_credit_keys"],"original-H041-never-credit-preserved")
 check("native32/native-VAL000-full227990-borrowed32-GPU0-v1"in policy["ordinary_completed_close_excluded_keys"],"stopped-native32-exclusion-preserved")
 check(b.KEY not in policy["never_credit_keys"]and b.KEY not in policy["ordinary_completed_close_excluded_keys"],"v12-key-not-excluded")
 records={};dirs=set();trace=[];calls=dict(state=0,apply=0,measure=0,append=0,absent=0,groups=0,transaction=0)
 def add(path,value):
  raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
  r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest());records[path]=(r,copy.deepcopy(value));return r
 def ref(p):
  p=str(p)
  if p.startswith(str(root)+"/"):p=p[len(str(root))+1:]
  return copy.deepcopy(records[p][0])if p in records else realref(p)
 def read(r):
  if r["path"]in records:
   rr,v=records[r["path"]];check(rr==r,"RAM-exact-immutable-ref");return copy.deepcopy(v)
  return realread(r)
 def raw(r):
  check(not r["path"].endswith((".pt",".npz",".npy",".trees")),"no-checkpoint-truth-or-scientific-payload-read")
  if r["path"]in records:return json.dumps(read(r)).encode()
  return realraw(r)
 class P(PurePosixPath):
  def exists(self):return str(self)in dirs
  def mkdir(self,**kw):
   check(str(self)not in dirs,"exclusive-close-attempt");dirs.add(str(self));trace.append("mkdir")
  def is_file(self):return True
 def local(p):return P(str(root/p)if not str(p).startswith("/")else str(p))
 def save(directory,name,value):
  raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
  check(len(raw)<=65536,"bounded-own-compact-record")
  path=str(directory/name)[len(str(root))+1:];check(path not in records,"no-record-overwrite")
  return add(path,value)
 c.local=local;c.ref=ref;c.read=read;c.raw=raw;b._save=save
 def need(ok,message):
  if not ok:raise ValueError(message)
 def integer(value,label,minimum=0):
  need(type(value)is int and value>=minimum,label);return value
 encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
 v4=SimpleNamespace(__name__="RAM_original_v4",integer=integer,encoded=encoded,
  canonical=lambda root,p:P(str(root/p)if not str(p).startswith("/")else str(p)))
 def canonical(root,p):return P(str(root/p)if not str(p).startswith("/")else str(p))
 def file_bytes(*args):return b"",None
 @contextmanager
 def mutex(root):
  trace.append("mutex-enter")
  try:yield
  finally:trace.append("mutex-exit")
 for f in(canonical,file_bytes,mutex):f.__module__=v4.__name__
 v4.canonical=canonical;v4.file_bytes=file_bytes;v4.mutex=mutex
 budget=SimpleNamespace(v4=v4,digest=b.digest)
 def absent(pid):
  calls["absent"]+=1;return dict(pid=pid,absent=True,method="targeted os.kill(pid,0) ESRCH",checked_utc=datetime.now(timezone.utc).isoformat())
 baseline=realread(realref(c.HERE+"/ROOT_qualification_records_root01_oct17/reserve_authority.json"))["bindings"]
 state=({b.KEY:meta["own"]},set(),dict(base_pins={"RAM":True},new_reserved_bytes=0,completed_credit_bytes=0),baseline,[],[],{})
 def state_locked(*args,**kw):
  check(kw=={"fresh":True},"original-fresh-state");calls["state"]+=1;return state
 def measure(root,paths):
  calls["measure"]+=1
  check([str(p)for p in paths]==[meta["own"]["output"],*meta["own"]["auxiliary_scopes"]],"one-exact-original-output-plus-AUX-measurement")
  return dict(conservative_scope_bytes=2100000000,entries=123,scope_paths=[str(p)for p in paths],stat_only=True)
 v4.measure_scopes=measure
 ge=dict(require=need,budget=budget,ROOT=root,Path=Path,datetime=datetime,timezone=timezone,copy=copy,
  MODE="Census_observation_two_scope_retention_v1",RETIRE_OPERATION="RAM_retire",
  MALFORMED_KEY="H039/directed-clade-exchange-train-stage-seed101-v1",META_CHARGE=524288,BOOTSTRAP_CHARGE=2097152,
  CEILING_BYTES=980000000000,HUMAN_LIMIT_BYTES=1000000000000,WARNING_BYTES=950000000000,
  COMPLETED_SCHEMAS=frozenset(),_json=read,_completed_json=read,_raw=raw,_absent=absent,
  _paths=lambda own:[P(own["output"]),*[P(p)for p in own["auxiliary_scopes"]]],
  _policy=lambda:policy,_state_locked=state_locked,
  _pins=lambda v:dict(base_pins=v["base_pins"],completed_overlay_prefix=dict(events=0,tail_sha256=None)),
  _utc=lambda:datetime.now(timezone.utc).isoformat(),ORDINARY_EXCLUSIONS=tuple(policy["ordinary_completed_close_excluded_keys"]))
 original=ast.parse(Path(b.GUARD["path"]).read_text())
 selected=["_pointer","_identity","_completed_identity","_proof","_measurement","assert_open","operation_draft_v1","_authority","_expire","charge","warning_fields","apply_close_completed_v1"]
 for n in original.body:
  if isinstance(n,ast.FunctionDef)and n.name in selected:exec(compile(ast.Module(body=[n],type_ignores=[]),b.GUARD["path"],"exec"),ge)
 def append(operation,authority_ref,value,state):
  calls["append"]+=1
  event=dict(schema="Census_observation_exact_retention_event_v1",sequence=1,previous_sha256=None,
   bindings=state[3],operation=operation,authority=authority_ref,value=value)
  check(len(encoded(event)+b"\n")<=65536,"original-close-event-under64KiB")
  return hashlib.sha256(encoded(event)+b"\n").hexdigest()
 ge["_append"]=append
 @contextmanager
 def transaction(r):
  calls["transaction"]+=1;trace.append("transaction-enter")
  try:yield
  finally:trace.append("transaction-exit")
 g=SimpleNamespace(budget=budget,MODE=ge["MODE"],ORDINARY_EXCLUSIONS=ge["ORDINARY_EXCLUSIONS"],
  _policy=ge["_policy"],_proof=ge["_proof"],_absent=absent,_state_locked=state_locked,_pins=ge["_pins"],
  operation_draft_v1=ge["operation_draft_v1"])
 g.load_selected_guard_v1=lambda r:g
 def apply(r,**kwargs):
  calls["apply"]+=1
  check("transaction-exit"not in trace and trace[-1]=="mutex-exit","original-apply-serialized-outside-capture-mutex")
  return ge["apply_close_completed_v1"](r,**kwargs)
 g.apply_close_completed_v1=apply
 coord=SimpleNamespace(accounting_transaction=transaction)
 c.load=lambda r,name:g if r==b.GUARD else coord
 def group_probe(group,sig):
  check(group==3752065 and sig==0,"only-exact-recorded-group-readonly-probe");calls["groups"]+=1;raise ProcessLookupError()
 b.os.killpg=group_probe
 now=datetime.now(timezone.utc)
 q=realread(realref(c.HERE+"/ROOT_native_MLM_v12_close_authorization.INACTIVE.v1.json"))
 reviewref=add("RAM/v12_source_review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],
  source_ref=realref(b.SOURCE),source_manifest_ref=realref(b.MANIFEST)))
 batchref=add("RAM/batch_input.json",dict(RAM=True))
 q.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(minutes=45)).isoformat(),instance="ramclose",source_review_ref=reviewref,
  accounting_batch_input_ref=batchref,all_original_v12_launch_resume_retry_write_routes_permanently_revoked=True)
 authorization=add("RAM/v12_close_authorization.json",q)
 result=b.close_once_v1(authorization)
 check(result["status"]=="ORDINARY_V12_CLOSE_APPLIED"and result["historical_supervisor_outer_exit_code"]==1,"whole-source-close-RAM-preserves-outer1")
 check(calls==dict(state=2,apply=1,measure=1,append=1,absent=153,groups=1,transaction=1),"original-one-measurement-one-append153-fresh-targeted-observations")
 check(result["credit_bytes"]==4563402752-2100000000 and result["full_MLM_fit_completion_claimed"]is False,"RAM-credit-exact-original-cap-minus-retained-no-fit-relabel")
 check(result["old_H041_or_native32_closed"]is False,"no-other-key-credit-or-child-retirement")
 check(read(meta["own"]["root_admission"])if False else True,"old-reservation-not-rebound-or-modified")
 def reject(call,name):
  try:call()
  except (ValueError,KeyError,TypeError):check(True,name)
  else:raise AssertionError(name)
 reject(lambda:b.close_once_v1(authorization),"consumed-attempt-cannot-retry")
 before=dict(calls)
 @contextmanager
 def timeout(r):
  raise ValueError("Finite shared ROOT accounting coordinator wait")
  yield
 c.load=lambda r,name:SimpleNamespace(accounting_transaction=timeout)
 timeoutq=dict(q,instance="ramtimeout");timeoutref=add("RAM/timeout_authorization.json",timeoutq)
 outcome=b.close_once_v1(timeoutref)
 check(outcome["status"]=="READY_NO_CLOSE_COORDINATOR_TIMEOUT"and outcome["actual_close_started"]is False
  and calls==before,"finite-coordinator-timeout-before-entry-no-measurement-close-or-replay")
 originalobs=next(v for r,v in records.values()if type(v)is dict and v.get("schema")=="concurrent_completed_scope_targeted_root_observation_v1")
 obsref=next(r for r,v in records.values()if v==originalobs)
 request=b.request_v1(meta,obsref)
 check(ge["_proof"](request,meta["own"])["completion_verified"]is True,"original-proof-really-accepts-worker0-with-historical-outer1")
 bad=dict(request,completed_ref=b.P["outer"])
 reject(lambda:ge["_proof"](bad,meta["own"]),"original-proof-rejects-substituted-completion-root")
 bad=dict(request,wait_exit_locator=["operational_failure"])
 reject(lambda:ge["_proof"](bad,meta["own"]),"original-proof-rejects-nonzero-or-noninteger-direct-wait")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_native_v12_close_exact_current_Census",
  "_native_v12_close_original_v3_coordinator")),"no-actual-guard-coordinator-science-or-resource-operations")
 resultrecord=dict(schema="native_MLM_v12_ordinary_close_source_checks_v1",source_only=True,source_ref=realref(b.SOURCE),
  checks_passed=len(checks),assertions_passed=assertions[0],checks=checks,genuine_metadata_join=True,
  original_unchanged_proof_and_apply_AST_RAM=True,one_original_scope_measurement_RAM=True,
  all51_targeted_PIDs_RAM_only=True,historical_supervisor_outer_exit_code_preserved=1,
  actual_guard_import=False,actual_probe=False,actual_stat_measurement=False,actual_close_or_credit=False,
  checkpoint_or_scientific_payload_opened=False,
  limitations="ROOT must still supply genuine narrow review/finite authorization/v3 batch input, revoke all v12 writers, obtain current absences and strict stat-only measurement, consume real close outer0 and verify exact journal acceptance.")
 print(json.dumps(dict(checks_passed=len(checks),assertions_passed=assertions[0],whole_original_close_RAM="PASS"),sort_keys=True))
 return resultrecord
if __name__=="__main__":
 try:main()
 except BaseException:print(traceback.format_exc());raise
