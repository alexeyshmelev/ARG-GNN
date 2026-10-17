"""Source and compact metadata plus isolated RAM full600 continuation checks."""
import ast,copy,hashlib,importlib,io,json,sys,traceback
from contextlib import contextmanager,redirect_stdout
from datetime import datetime,timezone,timedelta
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
def main():
 b=importlib.import_module("research.reviews.frozen_TreeLoom_bounded_training_source_20261017_v1.full600_ROOT_continuation_v1")
 c=b.c;root=c.ROOT;oldlocal=c.local
 def realref(path):
  p=oldlocal(path);return dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 def realraw(r):
  p=oldlocal(r["path"]);data=p.read_bytes()
  if hashlib.sha256(data).hexdigest()!=r["sha256"]:raise ValueError("Immutable source/metadata SHA")
  return data
 def realread(r):return json.loads(realraw(r))
 checks=[];assertions=[0]
 def check(ok,name):
  assertions[0]+=1
  if not ok:raise AssertionError(name)
  if name not in checks:checks.append(name)
 tree=ast.parse(realraw(realref(b.SOURCE)));source=ast.unparse(tree)
 check(not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(s in ast.unparse(n)for s in("torch","Census","memo"))for n in tree.body),"inert-stdlib-and-contract-imports")
 check("os.execve("in source and "Popen("not in source,"single-exec-existing-parent-no-new-scientific-supervisor")
 check("c.guard_v1("not in source and "install_consumer_v1("not in source,"ROOT-mutation-never-installs-consumer-memo")
 adm=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="_admit"))
 check(adm.index("accounting_transaction")<adm.index("_state_locked")<adm.index("apply_reserve_v1"),"v3-coordinator-encloses-capture-and-original-apply")
 check(adm.count("apply_reserve_v1(")==1 and "scope_spec("in adm,"once-reserve-and-original-full-scope")
 cont=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="continue_v1"))
 check(cont.index("directory.mkdir")<cont.index("_wait(")<cont.index("_admit(")<cont.index("os.execve("),"attempt-before-finite-wait-before-admit-before-exec")
 check("while True"not in adm and "time.sleep"not in adm,"no-accounting-retry-loop")
 baseauthority=realref(c.HERE+"/ROOT_qualification_records_root01_oct17/reserve_authority.json")
 check(baseauthority["sha256"]=="737442dc4280177a278ba27d433ebcb0e3777896712c83499aa71194f1bc6be4","ROOT-authorized-genuine-qualification-authority-pin")
 baseline=realread(baseauthority)["bindings"]
 check(len(baseline)==47 and len({json.dumps(v,sort_keys=True)for v in baseline.values()})==40,"genuine-47role40unique-baseline-not-template")
 check(c.GUARD not in baseline.values(),"current-adapter-explicitly-additional-to-baseline")
 records={};dirs=set();trace=[];calls=dict(state=0,apply=0,scope=0,transaction=0,exec=0)
 def add(path,value):
  raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
  r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest());records[path]=(r,copy.deepcopy(value));return r
 def ref(path):
  path=str(path)
  if path.startswith(str(root)+"/"):path=path[len(str(root))+1:]
  return copy.deepcopy(records[path][0])if path in records else realref(path)
 def read(r):
  if r["path"]in records:
   rr,value=records[r["path"]];check(rr==r,"RAM-exact-ref-authentication");return copy.deepcopy(value)
  return realread(r)
 def raw(r):
  check(not r["path"].endswith((".npz",".npy",".pt",".trees")),"no-scientific-array-truth-or-checkpoint-read")
  if r["path"]in records:read(r);return json.dumps(records[r["path"]][1]).encode()
  return realraw(r)
 class P(PurePosixPath):
  def exists(self):return str(self)in dirs
  def mkdir(self,**kw):
   check(str(self)not in dirs,"exclusive-attempt-mkdir");dirs.add(str(self));trace.append("mkdir")
  def is_file(self):return True
 def local(p):return P(str(root/p)if not str(p).startswith("/")else str(p))
 def save(directory,name,value):
  rawdata=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
  check(len(rawdata)<=65536,"bounded-flat-own-metadata")
  path=str(directory/name)[len(str(root))+1:]
  check(path not in records,"exclusive-own-record")
  r=add(path,value);trace.append("save:"+name);return r
 now=datetime.now(timezone.utc)
 actualinput=realref(c.HERE+"/ROOT_qualification_records_root01_oct17/input.json")
 actualq=realread(actualinput)
 completed=add("RAM/qualification_completed.json",dict(input_ref=actualinput,phase="qualification",
  complete_parent_updates=1,trainable_tensors=49,qualification_weights_are_not_final_training_weights=True,
  qualification_warmstart_or_optimizer_or_rng_reuse=False,VAL_truth_read=False,TEST_truth_read=False,
  elapsed_seconds=123.,peak_cuda_reserved_bytes=32*2**30,peak_cpu_RSS_bytes=128*2**30))
 waitref=add("RAM/qualification_wait.json",dict(scientific_input_ref=actualinput,actual_raw_wait=0,
  actual_direct_returncode=0,all_owned_processes_drained=True,pipe_eof=True,failure=None,cleanup_errors=[],
  adopted_waits_truncated=False,adopted_waits=[dict(returncode=0)]))
 outerref=add("RAM/qualification_outer.json",dict(actual_exit_code=0,consumed=True))
 parityref=add("RAM/qualification_parity.json",dict(passed=True,
  all49_raw_and_clipped_gradients_and_AdamW_parameters_moments_compared=True,nonzero_memory_gradient=True))
 acceptance=add("research/reviews/RAM/qualification_acceptance.json",dict(
  schema="ROOT_frozen_TreeLoom_actual_qualification_acceptance_v1",root_authorizer="root_agent",
  actual_outer_terminal_consumed=True,actual_exit_code=0,all_owned_processes_drained=True,
  source_manifest_ref=realref(c.MANIFEST),synthetic_all49_parity_passed=True,
  full30Mb_TRAIN000_disposable_optimizer_update_verified=True,qualification_weights_reused=False,
  qualification_input_ref=actualinput,completed_ref=completed,actual_parent_wait_ref=waitref,
  actual_outer_terminal_ref=outerref,synthetic_parity_ref=parityref))
 rows={};labels={}
 for parent in b.PARENTS:
  observed=add("RAM/"+parent+"_observed.json",dict(parent_id=parent,split="train",sequence_length_bp=30000000,
   query_ancestry_truth_in_inputs=False))
  focal=dict(path="experiments/results/"+parent+"/focals.npz",sha256="4"*64)
  native=add("RAM/"+parent+"_native.json",dict(parent_id=parent,observation_ref=observed,focals_ref=focal,
   query_ancestry_truth_opened=False))
  close=add("RAM/"+parent+"_accepted_close.json",dict(actual_close_event_accepted=True))
  rows[parent]=dict(parent_id=parent,observation_ref=observed,native_preparation_ref=native,
   focal_registry_ref=focal,focal_identity_sha256="5"*64,observation_only=True,
   all_native_trees_nodes_sites_samples_retained=True,inference_close_accepted_ref=close,native_close_accepted_ref=close)
  labels[parent]=dict(parent_id=parent,truth_ref=dict(path="experiments/results/"+parent+"/truth/ancestry.npz",sha256="6"*64))
 readiness=add("research/reviews/RAM/all20_readiness.json",dict(schema="TreeLoom_all20_training_readiness_v1",
  all20_complete=True,query_truth_opened=False,TRAIN_parent_ids=b.PARENTS,parents=rows))
 truth=add("research/reviews/RAM/all20_truth_handles.json",dict(schema="TreeLoom_all20_training_truth_handles_v1",
  TRAIN_parent_ids=b.PARENTS,class_order=["EUR","EAS","AFR"],parents=labels,query_truth_opened=False))
 batch=add("RAM/actual_batch_input.json",dict(RAM_only=True))
 request=realread(realref(c.HERE+"/ROOT_full600_continuation_request.INACTIVE.v1.json"))
 request.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(hours=40)).isoformat(),wait_deadline_at_utc=(now+timedelta(seconds=120)).isoformat(),
  instance="fullram",qualification_input_ref=actualinput,accounting_batch_input_ref=batch,
  future_qualification_acceptance_path=acceptance["path"],future_all20_readiness_path=readiness["path"],
  future_all20_truth_handles_path=truth["path"],future_launch_authority_path="research/reviews/RAM/launch_authority.json",
  automatic_single_parent_launch_authorized=True,poll_seconds=5)
 requestref=add("RAM/full600_request.json",request)
 ownerref=realref("research/reviews/research_execution_ROOT_20261017_v34/native_owner_v3_capability.root_active_v1.json")
 readyref=realref("research/reviews/research_execution_ROOT_20261017_v34/native_owner_v3_live_ready.root_v1.json")
 launch=realread(realref(c.HERE+"/ROOT_full600_launch_authority.INACTIVE.v1.json"))
 launch.update(status="active",template_only=False,execution_authorized=True,created_at_utc=now.isoformat(),
  expires_at_utc=(now+timedelta(hours=10)).isoformat(),continuation_request_ref=requestref,
  qualification_acceptance_ref=acceptance,all20_readiness_ref=readiness,all20_truth_handles_ref=truth,
  controller_capability_ref=ownerref,controller_ready_ref=readyref,runtime_ref=actualq["runtime_ref"],
  runtime_qualification_ref=actualq["runtime_qualification_ref"],max_seconds=28800,
  automatic_single_parent_launch_authorized=True)
 launch["measured_runtime_resource_plan"].update(ROOT_approved=True,qualification_completed_ref=completed,
  actual_qualification_elapsed_seconds=123.,actual_qualification_peak_cuda_reserved_bytes=32*2**30,
  actual_qualification_peak_cpu_RSS_bytes=128*2**30,all20_parent_dimensions_reviewed=True,
  estimated_full600_work_seconds=6000.,basis="RAM fixture only; actual measured plan still ROOT work")
 launchref=add("research/reviews/RAM/launch_authority.json",launch)
 c.local=local;c.ref=ref;c.read=read;c.raw=raw;b._save=save
 q,manifest=b._request(requestref);q["_request_ref"]=requestref
 check(q["qualification_input_ref"]==actualinput,"original-source-request-authenticates-real-qualified-input")
 qualified=b._qualification(acceptance,read(acceptance),q)
 data=b._all20(readiness,read(readiness),truth,read(truth),qualified)
 check(list(data["parents"])==b.PARENTS and len(data["train_truth_refs"])==20,"all20-genuine-shape-no-VAL-TEST")
 check(data["barrier"]["schema"]=="frozen_American_full30Mb_all20_observation_barrier_v2"
  and data["barrier"]["prior_disposable_TRAIN000_qualification_genuine_truth_opened"]is True
  and data["barrier"]["this_full_training_truth_opened"]is False,"honest-scoped-v2-barrier-conversion")
 bound=b._launch(launchref,read(launchref),q,qualified,data)
 check(bound["authority"]["max_seconds"]==28800,"measured-plan-and-real-owner-compact-join")
 def reject(call,name,kind=ValueError):
  try:call()
  except kind:check(True,name)
  else:raise AssertionError(name)
 bad=read(acceptance);bad["qualification_weights_reused"]=True
 reject(lambda:b._qualification(acceptance,bad,q),"reject-qualification-state-reuse")
 bad=read(readiness);bad["parents"].pop(b.PARENTS[-1])
 reject(lambda:b._all20(readiness,bad,truth,read(truth),qualified),"reject-missing-parent")
 bad=read(readiness);bad["parents"][b.PARENTS[1]]["parent_id"]="american_c3_val_001"
 reject(lambda:b._all20(readiness,bad,truth,read(truth),qualified),"reject-substituted-VAL-parent")
 bad=read(truth);bad["query_truth_opened"]=True
 reject(lambda:b._all20(readiness,read(readiness),truth,bad,qualified),"reject-fulltraining-truth-before-seal")
 bad=read(launchref);bad["measured_runtime_resource_plan"]["persistent_preparation_or_tensor_cache_assumed"]=True
 reject(lambda:b._launch(launchref,bad,q,qualified,data),"reject-unimplemented-cache-plan")
 bad=read(launchref);bad["measured_runtime_resource_plan"]["estimated_full600_work_seconds"]=28800.
 reject(lambda:b._launch(launchref,bad,q,qualified,data),"reject-insufficient-measured-work-window",b.ReadyOnly)
 bad=read(launchref);bad["expires_at_utc"]=(now+timedelta(days=3)).isoformat()
 reject(lambda:b._launch(launchref,bad,q,qualified,data),"reject-existing-owner-clock-extension",b.ReadyOnly)
 bad=read(launchref);bad["all20_readiness_ref"]=truth
 reject(lambda:b._launch(launchref,bad,q,qualified,data),"launch-must-bind-exact-all20-seal")
 # Finite metadata waiting is exercised with RAM clock, never an actual wait.
 clock=[0.];wall=now.timestamp()
 oldtime=b.time;oldfuture=b._future
 b.time=SimpleNamespace(time=lambda:wall+clock[0],monotonic=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
 b._future=lambda p:(_ for _ in()).throw(b.Pending(p))
 shortq=dict(q,wait_deadline_at_utc=(now+timedelta(seconds=11)).isoformat())
 with redirect_stdout(io.StringIO()):
  reject(lambda:b._wait(shortq,None),"finite-unready-deadline-no-token-poll",b.Pending)
 check(clock[0]==11.,"finite-monotonic-clock-bounded-poll")
 b._future=lambda p:(ref(p),read(ref(p)))
 with redirect_stdout(io.StringIO()):
  sq=dict(q,automatic_single_parent_launch_authorized=False)
  _,_,none,reason=b._wait(sq,None)
 check(none is None and "data readiness only"in reason,"readiness-only-mode-no-admission")
 b.time=oldtime;b._future=oldfuture
 # Complete original full scope and original input contract are executed against RAM guard dependencies.
 state=({},set(),dict(base_pins={"RAM":True},new_reserved_bytes=0,completed_credit_bytes=0),baseline,[],[],{})
 @contextmanager
 def mutex(root):
  trace.append("mutex-enter")
  try:yield
  finally:trace.append("mutex-exit")
 @contextmanager
 def transaction(r):
  check(r==batch,"actual-batch-ref-to-accounting-transaction");calls["transaction"]+=1;trace.append("transaction-enter")
  try:yield
  finally:trace.append("transaction-exit")
 def canonical(root,p):return P(p)
 def file_bytes(*a):return b"",None
 for f in(mutex,canonical,file_bytes):f.__module__="RAM_original_v4"
 v4=SimpleNamespace(__name__="RAM_original_v4",mutex=mutex,canonical=canonical,file_bytes=file_bytes,
  encoded=lambda v:json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
 def extract(path,names,env):
  for n in ast.parse(Path(path).read_text()).body:
   if isinstance(n,ast.FunctionDef)and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),path,"exec"),env)
 scopeenv=dict(Path=Path,canonical=canonical,
  paths_overlap=lambda a,d:str(a)==str(d)or str(a).startswith(str(d)+"/")or str(d).startswith(str(a)+"/"),
  accounting_paths=lambda r:[],bound=lambda root,r:raw(r),
  integer=lambda x,n,m:check(type(x)is int and x>=m,"original64MiB-charge-schema"))
 extract("src/arggnn/project_budget_v4.py",["scope_spec"],scopeenv)
 def scope(root,r):calls["scope"]+=1;return scopeenv["scope_spec"](root,r)
 v4.scope_spec=scope
 g=SimpleNamespace(MODE="Census_observation_two_scope_retention_v1",MAX_JSON=65536,META_CHARGE=524288,budget=SimpleNamespace(v4=v4))
 env=dict(copy=copy,require=c.require,MODE=g.MODE,RETIRE_OPERATION="RAM",_raw=raw)
 extract(c.GUARD["path"],["consumer_bindings_v1","operation_draft_v1"],env)
 g.consumer_bindings_v1=env["consumer_bindings_v1"];g.operation_draft_v1=env["operation_draft_v1"]
 def state_locked(*args,**kw):
  check(kw=={"fresh":True}and trace[-1]=="mutex-enter","fresh-state-under-original-mutex")
  calls["state"]+=1;return state
 g._state_locked=state_locked;g._pins=lambda v:dict(base_pins=v["base_pins"],completed_overlay_prefix=dict(events=0,tail_sha256=None))
 g.load_selected_guard_v1=lambda r:g
 def apply(r,**kw):
  check(trace[-1]=="mutex-exit"and "transaction-exit"not in trace,"original-apply-inside-coordinator-outside-capture-mutex")
  calls["apply"]+=1;authority=read(r)
  return dict(value=dict(authority["request"],root_admission=r),initial_ledger_pins=dict(RAM=True))
 g.apply_reserve_v1=apply
 def load(r,name):
  check(r in(c.GUARD,b.COORDINATOR),"only-guard-and-coordinator-loaded-at-admission")
  trace.append("load:"+name)
  return g if r==c.GUARD else SimpleNamespace(accounting_transaction=transaction)
 c.load=load;c.CURRENT_INPUT_REF=None
 directory=local(c.HERE+"/RAM_full600_admit");dirs.add(str(directory))
 result=b._admit(q,manifest,directory,qualified,data,bound)
 check(result["automatic_exec_allowed"]is True and result["status"]=="FULL600_ADMITTED_SINGLE_PARENT_EXEC_NEXT","full-RAM64-admission-pass")
 actual=read(result["input_ref"]);storage=read(actual["storage_ref"])
 check(calls==dict(state=1,apply=1,scope=1,transaction=1,exec=0),"one-serialized-original-reserve-no-actual-exec")
 check(len(storage["reservation"]["consumer_bindings"])==64,"exact-actual47role40unique-plus-full60064union")
 check(all(r in storage["reservation"]["consumer_bindings"]for r in baseline.values())and c.GUARD in storage["reservation"]["consumer_bindings"],"all47actual-roles-and-current-adapter-retained")
 check(actual["phase"]=="full_training"and list(actual["parents"])==b.PARENTS
  and actual["qualification_is_disposable_and_not_benchmark_result"]is False,"original-full600-input-contract-pass")
 check(not any("checkpoint"in k.lower()for k in actual),"no-qualification-checkpoint-or-warmstart-input")
 check(not local(actual["output_directory"]).exists()and not local(actual["worker_auxiliary_scope"]).exists(),"primary-AUX-remain-absent-until-original-parent")
 reject(lambda:g.consumer_bindings_v1(storage["reservation"]["consumer_bindings"],
  [realref(c.HERE+"/ROOT_full600_launch_authority.INACTIVE.v1.json")]),"original65th-binding-rejected")
 check(not any(n in sys.modules for n in("torch","torch_geometric","_TreeLoom_full600_ROOT_mutation_Census","_TreeLoom_full600_accounting_transaction")),"no-real-guard-coordinator-science-or-GPU-operations")
 resultrecord=dict(schema="ROOT_TreeLoom_full600_continuation_source_checks_v1",source_only=True,
  source_ref=realref(b.SOURCE),checks_passed=len(checks),assertions_passed=assertions[0],checks=checks,
  actual_baseline_authority_ref=baseauthority,baseline_roles=47,baseline_unique_refs=40,full600_complete_binding_union=64,
  original_scope_spec_RAM=True,original_contract_full600_validate_input_RAM=True,
  original_consumer_bindings_and_operation_draft_RAM=True,finite_wait_RAM=True,
  actual_admission=False,actual_launch=False,actual_guard_or_coordinator_import=False,
  scientific_array_or_checkpoint_or_truth_payload_opened=False,
  limitations="RAM evidence only; ROOT still supplies actual qualification0/all20 readiness/truth handles/measured window/fresh or current owner, real accounting and real full600 outcomes.")
 print(json.dumps(dict(checks_passed=len(checks),assertions_passed=assertions[0],full600_complete_binding_union=64),sort_keys=True))
 return resultrecord
if __name__=="__main__":
 try:main()
 except BaseException:print(traceback.format_exc());raise
