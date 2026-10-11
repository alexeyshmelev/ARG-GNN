"""Independent source-only V3 queue checks; no top-level queue or science imports."""
from pathlib import Path, PurePosixPath
from copy import deepcopy
from types import SimpleNamespace
import ast, hashlib, json, datetime

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3_independent"
INTEGRATED=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root_v3"
OLD=ROOT/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root"
RUNNER=INTEGRATED/"run_four_parent_queue.py"
SOURCE=RUNNER.read_text()
PLAN=json.loads((INTEGRATED/"execution_plan.json").read_bytes())
TREE=ast.parse(SOURCE)
HELPER_NAMES={"read","sha","ref","read_pinned","stage_proposal","preflight_bindings","admission"}
HELPERS=compile(ast.Module(body=[n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name in HELPER_NAMES],type_ignores=[]),str(RUNNER),"exec")
def is_assignment(n,name):
 return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
first=next(i for i,n in enumerate(TREE.body) if is_assignment(n,"cfg"))
last=next(i for i,n in enumerate(TREE.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=="save")
PREFIX=compile(ast.Module(body=TREE.body[first:last+1],type_ignores=[]),str(RUNNER),"exec")
COUNTS={}
CASES=[]
def check(value,label):
 COUNTS[label]=COUNTS.get(label,0)+1
 if not value:raise AssertionError(label)
def raw_json(x):return (json.dumps(x,indent=2,allow_nan=False)+"\n").encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
class FirstWrite(Exception):pass
class VPath:
 fs=None
 def __init__(self,path):
  self.path=path.path if isinstance(path,VPath) else PurePosixPath(str(path))
 def __truediv__(self,other):return VPath(self.path/str(other))
 def __str__(self):return str(self.path)
 def relative_to(self,other):return self.path.relative_to(VPath(other).path)
 def read_bytes(self):return self.fs.load(str(self))
 def read_text(self,*args,**kwargs):return self.read_bytes().decode()
 def exists(self):raise AssertionError("forbidden_process_or_actual_path_exists")
 def stat(self):raise AssertionError("forbidden_process_stat")
class Fixture:
 def __init__(self):
  self.root=VPath("/handmade-independent/native-v7")
  self.base=self.root/"research/reviews/native_argformer_tsinfer_fit_v7_execution_root"
  self.runner=self.root/str(RUNNER.relative_to(ROOT))
  self.plan_path=self.root/str((INTEGRATED/"execution_plan.json").relative_to(ROOT))
  self.review_path=self.root/"handmade/review.json"
  self.source=self.root/"src/arggnn/native_argformer_tsinfer_fit_v7.py"
  self.config=self.root/"configs/native_argformer_tsinfer_fit_v7.json"
  self.files={};self.reads=[];self.writes=[];self.after_read={}
  self.source_pin=self.put(self.source,b"handmade native identity, never imported")
  self.config_value={"storage":{"companion_contract":{"handmade":True}}}
  self.config_pin=self.put(self.config,self.config_value)
  self.queue_pin=self.put(self.runner,SOURCE.encode())
  self.plan=deepcopy(PLAN)
  self.plan["queue_source"]=deepcopy(self.queue_pin)
  self.plan["implementation"]=deepcopy(self.source_pin)
  self.plan["config"]=deepcopy(self.config_pin)
  self.own={}
  for item in self.plan["stage_proposals"]:
   parent,stage=item["parent"],item["stage"]
   self.own["native_argformer_tsinfer_fit_v7/"+parent]={
    "output":str(self.root/"outputs"/parent),"auxiliary_scopes":[str(self.root/"shorttmp"/parent)],
    "root_admission":{"path":"handmade/reserve.json","sha256":"f"*64}}
   reg={"scope_parents":[parent],"stage":stage}
   item["experiment_registration"]=self.put(self.root/item["experiment_registration"]["path"],reg)
   proposal={"scope_parents":[parent],"stage":stage,"implementation":deepcopy(self.source_pin),
    "config":deepcopy(self.config_pin),"experiment_registration":deepcopy(item["experiment_registration"]),
    "active":False,"silent_fallback":False}
   item["admission_proposal"]=self.put(self.root/item["admission_proposal"]["path"],proposal)
  self.plan["budget_receipt"]=self.put(self.base/"budget_receipt.json",{"admitted_reservations":self.own})
  self.plan["release_barrier"]=self.put(self.base/"release_barrier.json",{
   "root_authorizer":"root_agent","active":True,"waiters":[{"pid":7,"uid":5,"start_ticks":9}],
   "maximum_wait_seconds":11,"poll_interval_seconds":1})
  self.activation={"root_authorizer":"root_agent","active":True,"queue_source":deepcopy(self.queue_pin),
   "budget_receipt":deepcopy(self.plan["budget_receipt"])}
  self.put(self.root/".runtime/gpup/storage_reservations.json",{"handmade":"ledger"})
  self.put(self.root/".runtime/storage/project_budget_v4.json",{"handmade":"ledger"})
  self.seal()
 def put(self,path,obj):
  raw=obj if isinstance(obj,bytes) else raw_json(obj)
  self.files[str(path)]=raw
  return {"path":str(path.relative_to(self.root)),"sha256":digest(raw)}
 def seal(self):
  self.plan_pin=self.put(self.plan_path,self.plan)
  self.review_pin=self.put(self.review_path,{"verdict":"PASS","review_role":"independent_reviewer",
   "execution_plan":deepcopy(self.plan_pin),"queue_source":deepcopy(self.queue_pin),
   "synthetic_fixture_only":True})
  self.activation.update(execution_plan=deepcopy(self.plan_pin),independent_review=deepcopy(self.review_pin))
  self.put(self.base/"queue_activation.json",self.activation)
 def load(self,path):
  self.reads.append(path)
  if path not in self.files:raise FileNotFoundError(path)
  result=self.files[path]
  callback=self.after_read.pop(path,None)
  if callback:callback()
  return result
 def first_write(self,path,obj):
  self.writes.append((str(path),deepcopy(obj)))
  raise FirstWrite()
 def save(self,path,obj):
  check(str(path) not in self.files,"exclusive_handmade_save")
  self.writes.append((str(path),deepcopy(obj)))
  self.put(path,obj)
 def namespace(self):
  VPath.fs=self
  ns={"Path":VPath,"ROOT":self.root,"BASE":self.base,"SOURCE":self.source,"CONFIG":self.config,
   "SOURCE_SHA":self.source_pin["sha256"],"CONFIG_SHA":self.config_pin["sha256"],
   "hashlib":hashlib,"json":json,"__file__":str(self.runner),
   "os":SimpleNamespace(getpid=lambda:101),"time":SimpleNamespace(monotonic=lambda:0),
   "ticks":lambda pid:88,"utc":lambda:"handmade-only","save":self.first_write}
  exec(HELPERS,ns)
  return ns
 def prefix(self):
  ns=self.namespace()
  exec(PREFIX,ns)
  raise AssertionError("first-write sentinel missing")
 def good_prefix(self):
  ns=self.namespace()
  try:exec(PREFIX,ns)
  except FirstWrite:pass
  else:raise AssertionError("first-write sentinel missing")
  check(len(self.writes)==1,"positive_first_write_count")
  check(self.writes[0][0]==str(self.base/"queue_wait_started.json"),"positive_first_write_path")
  ns["save"]=self.save
  self.prefix_reads=list(self.reads)
  self.writes.clear();self.reads.clear()
  return ns
 def ledger_reads(self):
  return [s for s in self.reads if s.endswith("/storage_reservations.json") or s.endswith("/project_budget_v4.json")]
def reject_prefix(f,label):
 rejected=False
 try:f.prefix()
 except (AssertionError,KeyError,FileNotFoundError,json.JSONDecodeError):rejected=True
 check(rejected,label+"_rejected")
 check(not f.writes,label+"_before_first_write")
 check(not f.ledger_reads(),label+"_before_ledger_read")
 CASES.append({"case":label,"rejected":True,"queue_writes":0,"ledger_reads":0})
def reject_admission(f,ns,item,label):
 rejected=False
 try:ns["admission"](item["parent"],item["stage"],"handmade_reject")
 except (AssertionError,KeyError,FileNotFoundError,json.JSONDecodeError):rejected=True
 check(rejected,label+"_rejected")
 check(not f.writes,label+"_before_stage_writes")
 check(not f.ledger_reads(),label+"_before_ledger_read")
 CASES.append({"case":label,"rejected":True,"stage_writes":0,"ledger_reads":0})

# Reproduce all three V1 binding counterexamples against V2.
f=Fixture();path=f.base/"release_barrier.json";value=json.loads(f.files[str(path)]);value["waiters"]=[]
f.put(path,value);reject_prefix(f,"prior_empty_unpinned_barrier")
f=Fixture();f.activation["budget_receipt"]=f.put(f.base/"budget_receipt.json",{"admitted_reservations":deepcopy(f.own),"changed":True})
f.put(f.base/"queue_activation.json",f.activation);reject_prefix(f,"prior_activation_different_budget")
f=Fixture();item=f.plan["stage_proposals"][0];path=f.root/item["admission_proposal"]["path"];value=json.loads(f.files[str(path)]);value["silent_fallback"]=True
f.put(path,value);reject_prefix(f,"prior_changed_unpinned_proposal")

# Check every bound pair and both metadata file roles before any queue write.
for i in range(12):
 for field in ("admission_proposal","experiment_registration"):
  for fault in ("changed_bytes","missing","wrong_sha","alias_path"):
   f=Fixture();item=f.plan["stage_proposals"][i];path=f.root/item[field]["path"]
   if fault=="changed_bytes":f.files[str(path)]+=b" "
   elif fault=="missing":del f.files[str(path)]
   elif fault=="wrong_sha":item[field]["sha256"]="0"*64;f.seal()
   else:item[field]=deepcopy(f.plan["stage_proposals"][(i+1)%12][field]);f.seal()
   reject_prefix(f,field+"_"+fault+"_"+str(i))
  f=Fixture();ns=f.good_prefix();item=f.plan["stage_proposals"][i];path=f.root/item[field]["path"]
  f.files[str(path)]+=b" ";reject_admission(f,ns,item,field+"_after_preflight_"+str(i))
for fault in ("missing","duplicate","extra","wrong_parent","wrong_stage"):
 f=Fixture();items=f.plan["stage_proposals"]
 if fault=="missing":items.pop()
 elif fault=="duplicate":items[-1]=deepcopy(items[0])
 elif fault=="extra":items.append(deepcopy(items[0]))
 elif fault=="wrong_parent":items[-1]["parent"]="unexpected_parent"
 else:items[-1]["stage"]="unexpected_stage"
 f.seal();reject_prefix(f,"pair_completeness_"+fault)
# Rehashed mismatching fields still reject, rather than relying only on stale hashes.
for i in range(12):
 for fault in ("proposal_parent","proposal_stage","proposal_registration","registration_parent","registration_stage"):
  f=Fixture();item=f.plan["stage_proposals"][i]
  pp=f.root/item["admission_proposal"]["path"];rp=f.root/item["experiment_registration"]["path"]
  proposal=json.loads(f.files[str(pp)]);reg=json.loads(f.files[str(rp)])
  if fault=="proposal_parent":proposal["scope_parents"]=[item["parent"],"extra"]
  elif fault=="proposal_stage":proposal["stage"]="unexpected_stage"
  elif fault=="proposal_registration":proposal["experiment_registration"]=deepcopy(f.plan["stage_proposals"][(i+1)%12]["experiment_registration"])
  else:
   if fault=="registration_parent":reg["scope_parents"]=["unexpected_parent"]
   else:reg["stage"]="unexpected_stage"
   item["experiment_registration"]=f.put(rp,reg);proposal["experiment_registration"]=deepcopy(item["experiment_registration"])
  item["admission_proposal"]=f.put(pp,proposal);f.seal();reject_prefix(f,fault+"_"+str(i))
for field in ("budget_receipt","release_barrier"):
 for fault in ("changed_bytes","missing","wrong_sha","alias_path"):
  f=Fixture();path=f.root/f.plan[field]["path"]
  if fault=="changed_bytes":f.files[str(path)]+=b" "
  elif fault=="missing":del f.files[str(path)]
  elif fault=="wrong_sha":f.plan[field]["sha256"]="0"*64
  else:f.plan[field]=f.put(f.base/("other_"+field+".json"),f.files[str(path)])
  if field=="budget_receipt":f.activation[field]=deepcopy(f.plan[field])
  f.seal();reject_prefix(f,field+"_"+fault)
# Preflight consumes all pinned metadata before first write and retains snapshot.
f=Fixture();ns=f.namespace()
try:exec(PREFIX,ns)
except FirstWrite:pass
for field in ("budget_receipt","release_barrier"):
 check(f.reads.count(str(f.root/f.plan[field]["path"]))==1,"single_read_"+field)
for item in f.plan["stage_proposals"]:
 for field in ("admission_proposal","experiment_registration"):
  check(f.reads.count(str(f.root/item[field]["path"]))==1,"single_read_all24")
check(f.writes[0][1]["barrier"]==f.plan["release_barrier"],"wait_uses_verified_barrier_ref")
# Simulated drift immediately after a byte read cannot change the parsed snapshot.
f=Fixture();ns=f.namespace();item=f.plan["stage_proposals"][0];pp=f.root/item["admission_proposal"]["path"]
original=json.loads(f.files[str(pp)])
f.after_read[str(pp)]=lambda:f.files.__setitem__(str(pp),b'{"not_the_verified_snapshot":true}')
parsed=ns["stage_proposal"](f.plan,item["parent"],item["stage"])
check(parsed==original,"single_snapshot_hash_and_parse")
check(f.reads.count(str(pp))==1,"single_snapshot_one_read")
# Each valid admission stays parent-scoped, preserves registration pin, and writes storage then admission.
f=Fixture();ns=f.good_prefix()
for i,item in enumerate(f.plan["stage_proposals"]):
 f.reads.clear();f.writes.clear()
 a,ap=ns["admission"](item["parent"],item["stage"],"handmade_"+str(i))
 check(len(f.writes)==2,"positive_two_stage_writes")
 check(a["experiment_registration"]==item["experiment_registration"],"positive_registration_pin")
 check(f.writes[0][1]["scope_parents"]==[item["parent"]],"positive_storage_parent")
 check(f.writes[0][1]["reserved_bytes"]==20000000000,"positive_reserved_bytes")


# The V2 coordinated plan/proposal substitution now rejects before queue writes.
f=Fixture();old_plan_pin=deepcopy(f.plan_pin);item=deepcopy(f.plan["stage_proposals"][0])
pp=f.root/item["admission_proposal"]["path"]
proposal=json.loads(f.files[str(pp)]);proposal["independent_unreviewed_marker"]="changed_after_verified_plan_snapshot"
changed_proposal_raw=raw_json(proposal)
changed_plan=deepcopy(f.plan)
changed_plan["stage_proposals"][0]["admission_proposal"]["sha256"]=digest(changed_proposal_raw)
def substitute_after_plan_read():
 f.files[str(pp)]=changed_proposal_raw
 f.files[str(f.plan_path)]=raw_json(changed_plan)
f.after_read[str(f.plan_path)]=substitute_after_plan_read
reject_prefix(f,"prior_coordinated_plan_proposal_snapshot_substitution")
snapshot_cases=[{"case":"prior_coordinated_plan_proposal_substitution","rejected_before_first_write":True}]
# Plan replacement alone leaves the bound parsed plan and all proposal pins unchanged.
f=Fixture();reviewed=deepcopy(f.plan);changed=deepcopy(f.plan);changed["unreviewed_plan_marker"]=True
f.after_read[str(f.plan_path)]=lambda:f.files.__setitem__(str(f.plan_path),raw_json(changed))
ns=f.good_prefix()
check(ns["plan"]==reviewed,"plan_snapshot_retains_verified_bytes")
check(f.prefix_reads.count(str(f.plan_path))==1,"plan_snapshot_one_read")
check("unreviewed_plan_marker" not in ns["plan"],"plan_snapshot_does_not_parse_replacement")
snapshot_cases.append({"case":"plan_replaced_after_read","parsed_verified_original":True,"plan_reads":1})
# Review replacement after read never changes its parsed PASS/role/subject identity.
f=Fixture();reviewed=json.loads(f.files[str(f.review_path)]);changed=deepcopy(reviewed)
changed.update(verdict="REJECT",review_role="untrusted",execution_plan={"path":"unreviewed","sha256":"0"*64})
f.after_read[str(f.review_path)]=lambda:f.files.__setitem__(str(f.review_path),raw_json(changed))
ns=f.good_prefix()
check(ns["review"]==reviewed,"review_snapshot_retains_verified_bytes")
check(f.prefix_reads.count(str(f.review_path))==1,"review_snapshot_one_read")
snapshot_cases.append({"case":"review_replaced_after_read","parsed_verified_original":True,"review_reads":1})
# CONFIG immediately changes after read: the retained second hash check fails closed.
f=Fixture();changed={"storage":{"companion_contract":{"unreviewed":True}}}
f.after_read[str(f.config)]=lambda:f.files.__setitem__(str(f.config),raw_json(changed))
reject_prefix(f,"CONFIG_replaced_after_snapshot_before_redundant_hash")
# Replacement after CONFIG's retained second hash cannot change its already parsed cfg.
f=Fixture();calls={"count":0};reviewed=deepcopy(f.config_value);original_load=f.load
def config_second_hash(path):
 result=original_load(path)
 if path==str(f.config):
  calls["count"]+=1
  if calls["count"]==2:f.files[path]=raw_json(changed)
 return result
f.load=config_second_hash
ns=f.good_prefix()
check(ns["cfg"]==reviewed,"CONFIG_snapshot_retains_verified_bytes")
check(calls["count"]==2,"CONFIG_one_parse_snapshot_plus_redundant_hash")
snapshot_cases.append({"case":"CONFIG_replaced_after_retained_hash","parsed_verified_original":True,"CONFIG_byte_reads":2,"CONFIG_parses":1})

# Denied snapshot refs or semantic review/admission identities never reach the first queue write.
for field in ("plan","review","CONFIG"):
 for fault in ("stale_bytes","missing","wrong_sha"):
  f=Fixture();path={"plan":f.plan_path,"review":f.review_path,"CONFIG":f.config}[field]
  if fault=="stale_bytes":f.files[str(path)]+=b" "
  elif fault=="missing":del f.files[str(path)]
  elif field=="CONFIG":f.config_pin["sha256"]="0"*64
  else:
   pin="execution_plan" if field=="plan" else "independent_review"
   f.activation[pin]["sha256"]="0"*64;f.put(f.base/"queue_activation.json",f.activation)
  reject_prefix(f,"snapshot_denied_"+field+"_"+fault)
for field,fault in (
 ("activation","inactive"),("activation","root_authorizer"),("activation","queue_ref"),
 ("plan","physical_GPU"),("plan","queue_ref"),
 ("review","verdict"),("review","review_role"),("review","execution_plan"),("review","queue_source")):
 f=Fixture()
 if field=="activation":
  if fault=="inactive":f.activation["active"]=False
  elif fault=="root_authorizer":f.activation["root_authorizer"]="not_root"
  else:f.activation["queue_source"]["sha256"]="0"*64
  f.put(f.base/"queue_activation.json",f.activation)
 elif field=="plan":
  if fault=="physical_GPU":f.plan["physical_GPU"]=3
  else:f.plan["queue_source"]["sha256"]="0"*64
  f.seal()
 else:
  value=json.loads(f.files[str(f.review_path)])
  if fault=="verdict":value["verdict"]="REJECT"
  elif fault=="review_role":value["review_role"]="author"
  else:value[fault]["sha256"]="0"*64
  f.activation["independent_review"]=f.put(f.review_path,value)
  f.put(f.base/"queue_activation.json",f.activation)
 reject_prefix(f,"semantic_denied_"+field+"_"+fault)

# Mock only the unchanged dispatch AST with synthetic stage results and receipts.
tail_start=next(i for i,n in enumerate(TREE.body)
 if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
 and isinstance(n.value.func,ast.Name) and n.value.func.id=="save"
 and n.value.args and isinstance(n.value.args[0],ast.BinOp)
 and isinstance(n.value.args[0].right,ast.Constant) and n.value.args[0].right.value=="queue_started.json")
TAIL=compile(ast.Module(body=TREE.body[tail_start:],type_ignores=[]),str(RUNNER),"exec")
parents=("bio_paths_pilot_train_000","bio_paths_pilot_train_001","bio_paths_pilot_val_000","bio_paths_pilot_val_001")
def dispatch_fixture(failures):
 events=[];saves=[];passed={};receipts={};inflight=set()
 def mock_start(parent,stage,run,probe=None,fit=None):
  events.append(("start",parent,stage))
  if stage=="fit":
   check(probe==receipts[(parent,"probe-memory")] and passed[(parent,"probe-memory")],"fit_requires_own_passed_probe")
  if stage=="retrieve":
   check(fit==receipts[(parent,"fit")] and passed[(parent,"fit")],"retrieve_requires_own_passed_fit")
   inflight.add(parent)
  record={"parent":parent,"stage":stage,"run_id":run}
  return None,None,record,PurePosixPath("/handmade")/parent/stage/"receipt.json",99
 def mock_finish(item):
  record=item[2];parent,stage=record["parent"],record["stage"]
  events.append(("finish",parent,stage))
  ok=(parent,stage) not in failures
  passed[(parent,stage)]=ok;receipts[(parent,stage)]=item[3]
  if stage=="retrieve":inflight.discard(parent)
  return ok,item[3]
 ns={"save":lambda path,value:saves.append((str(path),deepcopy(value))),
  "BASE":PurePosixPath("/handmade/base"),"utc":lambda:"handmade-only",
  "os":SimpleNamespace(getpid=lambda:1),"ticks":lambda pid:1,
  "ref":lambda path:{"path":str(path),"sha256":"0"*64},"retrievals":[],
  "start":mock_start,"finish":mock_finish}
 exitcode=0
 try:exec(TAIL,ns)
 except SystemExit as exc:exitcode=exc.code
 return events,saves,exitcode
events,saves,code=dispatch_fixture(set())
starts=[event for event in events if event[0]=="start"]
check(code==0 and len(starts)==12,"dispatch_all12_stages_mocked")
check([e[2] for e in starts[:4]]==["probe-memory"]*4,"dispatch_all4_probes_first")
check([e[1] for e in starts[:4]]==list(parents),"dispatch_probe_parent_order")
check(all(events.index(("finish",p,"probe-memory"))<events.index(("start",parents[0],"fit")) for p in parents),"dispatch_all_probes_finish_before_first_fit")
check(events.index(("start",parents[0],"retrieve"))<events.index(("start",parents[1],"fit"))<events.index(("finish",parents[0],"retrieve")),"dispatch_CPU_retrieval_overlaps_later_fit")
check(saves[-1][1]["paper_reproduction_claimed"] is False and saves[-1][1]["silent_fallback"] is False,"dispatch_final_claim_guards")
for index,parent in enumerate(parents):
 events,saves,code=dispatch_fixture({(parent,"probe-memory")})
 starts=[e for e in events if e[0]=="start"]
 check(code==1 and len(starts)==index+1,"dispatch_stops_first_failed_probe")
 check(all(e[2]=="probe-memory" for e in starts),"dispatch_failed_probe_no_fit_or_retrieval")
 check(saves[-1][1]["stopped_on_first_failed_probe"] and saves[-1][1]["remaining_stages_unexecuted"],"dispatch_failed_probe_terminal")
for parent in parents:
 events,saves,code=dispatch_fixture({(parent,"fit")})
 starts=[e for e in events if e[0]=="start"]
 check(code==0 and ("start",parent,"retrieve") not in starts,"dispatch_failed_fit_no_own_retrieval")
 check(sum(e[2]=="fit" for e in starts)==4,"dispatch_other_fits_continue_after_failed_fit")
 check(sum(e[2]=="retrieve" for e in starts)==3,"dispatch_only_passed_fits_retrieve")
result={
 "kind":"native_v7_v3_independent_handmade_checks","verdict":"PASS",
 "assertions":sum(COUNTS.values()),"checks_by_category":COUNTS,
 "prior_three_V1_counterexamples_now_rejected":True,
 "prior_V2_coordinated_plan_proposal_substitution_now_rejected_before_first_queue_write":True,
 "rejection_cases":CASES,"snapshot_cases":snapshot_cases,
 "all12_post_preflight_pair_drift_rejected_before_stage_writes_or_ledger_reads":True,
 "single_read_pinned_helpers_verified":True,
 "all_probes_first_own_probe_fit_refs_and_CPU_retrieval_overlap_verified_by_mock_dispatch":True,
 "mock_dispatch_full_pass_cases":1,"mock_dispatch_failed_probe_cases":4,"mock_dispatch_failed_fit_cases":4,
 "actual_native_runtime_import_process_GPU_budget_data_model_or_Git_operations":False
}
for name,value in (("handmade_test_results.json",result),("snapshot_regression_evidence.json",{
 "kind":"native_v7_v3_independent_verified_snapshot_regressions","verdict":"PASS",
 "prior_V2_plan_counterexample_rejected_before_first_write":True,"snapshot_cases":snapshot_cases,
 "synthetic_review_PASS_is_fixture_only":True,"actual_queue_or_science_executed":False})):
 with (OUT/name).open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
print(json.dumps({"verdict":"PASS","assertions":result["assertions"],"rejection_cases":len(CASES),"snapshot_cases":snapshot_cases,"dispatch_cases":9},indent=2))
