"""Independent complete guarded scanner integration review; all active endpoints are synthetic."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib,importlib.util,json,sys,types,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/C3_CPU_interleaved_guarded_integration_independent_review_20261016_v1"
CAND="research/reviews/C3_CPU_interleaved_guarded_integration_source_20261016_v1"
sys.path.insert(0,str(ROOT))
def load_source(path,label):
 sp=importlib.util.spec_from_file_location(label,ROOT/path)
 m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
nc=load_source(CAND+"/acceptance_contract_v1.py","_independent_complete_interleaved_contract")
nb=load_source(CAND+"/prepare_evaluation_root_v1.py","_independent_complete_interleaved_builder")
def file_ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def functions(path):
 text=(ROOT/path).read_text()
 return text,{n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(node):return ast.dump(node,include_attributes=False)
def clone(fn,ns):
 value=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
 value.__kwdefaults__=copy.copy(fn.__kwdefaults__);return value
checks=[]
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SOURCE_AND_SOFTWARE_ONLY",**({"detail":detail}if detail is not None else {})))
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError,AttributeError):return
 raise AssertionError("Expected rejection")
# Inactive source validation authenticates only pinned source and compact metadata.
effective=nc.manifest_v1();oldmanifest=nc.legacy.manifest_v1()
assert len(effective["source_sha256"])==99 and len(oldmanifest["source_sha256"])==81
assert all(effective["source_sha256"][p]==s for p,s in oldmanifest["source_sha256"].items())
assert nc.consumer_guard is nc.legacy.consumer_guard and nc.original_guard is nc.legacy.original_guard
assert nc.consumer_guard.__globals__["original_guard"] is nc.legacy.original_guard
assert nc.legacy.SOURCE==nc.LEGACY_CONTRACT_REF["path"]
check("real_inactive_contract_complete99_closure_and_separate_legacy_namespace")
newworker,newnodes=functions(CAND+"/worker_entry_v1.py")
oldworker,oldnodes=functions(nc.LEGACY_ENTRY_REF["path"])
for name in ("GuardedLoaderV1","validate_saved_v1","matrix_v1","score_v1","execute","main"):
 assert dump(newnodes[name])==dump(oldnodes[name]),name
check("matrix_score_guarded_execute_and_cli_exact_original_AST")
assert "contract=c" in ast.get_source_segment(newworker,newnodes["predict_v1"])
check("only_prediction_body_routes_to_exact_reviewed_bridge")
# Reuse sealed independent full-schema VFS setup and original saved600 validators.
hpath="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1/independent_fake_checks_v1.py"
prior=json.loads((ROOT/"research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1/review.v1.json").read_text())
expected=next(r for section in prior["checks"]for r in section["evidence_refs"]if r["path"]==hpath)
assert file_ref(hpath)==expected
h=(ROOT/hpath).read_text()
h=h.replace('REVIEW="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1"','REVIEW="'+BASE+'"')
hnodes=[]
for node in ast.parse(h).body:
 hnodes.append(node)
 if isinstance(node,ast.ClassDef)and node.name=="FakeFlow":break
hns={}
exec(compile(ast.Module(body=hnodes,type_ignores=[]),"<sealed_independent_VFS_setup_only>","exec"),hns)
OldFlow=hns["FakeFlow"]
legacy_actual=hns["c"];helper=hns["helper"]
class CompleteFlow(OldFlow):
 def __init__(self,phase="predict_val_003",mode="normal"):
  super().__init__(phase,mode)
  self.oldns=self.ns
  # Synthetic acceptance content sits at the exact required immutable logical ref.
  self.mem[nc.SAVED_ACCEPTANCE_REF["path"]]=(copy.deepcopy(self.acceptance),nc.SAVED_ACCEPTANCE_REF["sha256"])
  self.evidence["ROOT_artifact_acceptance_ref"]=nc.SAVED_ACCEPTANCE_REF
  if phase.startswith("predict_val_"):
   own=nc.VAL[int(phase[-3:])]
   self.evidence["closed_plan_evidence_by_parent"]={own:copy.deepcopy(self.plan_evidence[own])}
  legacy_fake=types.SimpleNamespace(**self.oldns)
  nn=dict(vars(nc),legacy=legacy_fake,local=self.local,ref=self.ref,read=self.read,
          save=self.oldns["save"],_ACTIVE_SPEC=None)
  _,nodes=functions(CAND+"/acceptance_contract_v1.py")
  for name in nodes:
   if name!="_legacy_v1":nn[name]=clone(getattr(nc,name),nn)
  # Unchanged delegated bodies use the original synthetic namespace.
  for name in ("plan_ready_v1","terminal0_v1","original_guard","consumer_guard","acceptance_inputs_v1"):
   if name in self.oldns:nn[name]=self.oldns[name]
  self.ns=nn;self.fakec=types.SimpleNamespace(**nn)
  def route_stage(phase,result):
   if phase in nc.OLD_INPUT_REFS:
    inp=nc.OLD_INPUT_REFS[phase];driver=nc.LEGACY_DRIVER_REF;entry=nc.LEGACY_ENTRY_REF
    manifest_ref=nc.LEGACY_MANIFEST_REF;source_map=oldmanifest["source_sha256"]
   else:
    driver=self.ref(nc.DRIVER);entry=self.ref(nc.ENTRY);manifest_ref=self.ref(nc.MANIFEST)
    source_map=effective["source_sha256"];inp=None
   req=self.put(dict(phase=phase,source_manifest_ref=manifest_ref,training_evidence_ref=nc.SAVED_ACCEPTANCE_REF))
   spec=dict(phase=phase,root_driver_ref=driver,main_source_ref=entry,source_sha256=source_map,
    refs=dict(runtime=nc.RUNTIME_REF),root_stage_request_ref=req,
    packet_directory="research/reviews/C3_full30Mb_eval_ROOT_interleaved_"+phase+"_SYNTHETIC")
   if inp:self.mem[inp["path"]]=(spec,inp["sha256"])
   else:inp=self.put(spec)
   env=self.put(dict(stage=phase,status=nc.legacy.STATUSES[phase],root_driver_ref=driver,
                    training_completion_ref=self.training_ref,input_ref=inp,result_ref=result))
   wait=self.put(dict(driver_ref=driver,input_ref=inp,completed_ref=env,returncode=0,
      actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None))
   term=self.put(dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=env,actual_wait_ref=wait))
   return dict(completion_ref=env,actual_wait_ref=wait,outer_terminal_ref=term)
  self.stage=route_stage
  self.predictions={p:self.stage("predict_val_"+p[-3:],self.put(self.sixes[p]))for p in nc.VAL}
  self.matrix_evidence=self.stage("matrix",self.put(self.matrix))
  if phase=="matrix":self.evidence["VAL_predictions"]=copy.deepcopy(self.predictions)
  if phase=="score":self.evidence["matrix"]=copy.deepcopy(self.matrix_evidence)
  self.builder_review=json.loads((ROOT/CAND/"ROOT_review.INACTIVE.template.v1.json").read_text())
  self.builder_review.update(independent=True,verdict="SOURCE_PASS",source_handoff_ref=self.ref(nc.HANDOFF))
  self.builder_review_ref=self.put(self.builder_review)
  original_inspect=self.guard.inspect_v1
  def inspect(**kw):
   value=original_inspect(**kw)
   # Baseline actual inherited role set only; a new returned role must fail65.
   value["bindings"]=copy.deepcopy(oldmanifest["current_guard_roles"])
   if mode=="overflow":value["bindings"]["SYNTHETIC65"]=dict(path=BASE+"/SYNTHETIC65.json",sha256="7"*64)
   return value
  self.guard.inspect_v1=inspect
  bn=dict(vars(nb))
  def bound(record):
   if record["path"]in self.mem:return json.dumps(self.read(record),sort_keys=True,separators=(",",":")).encode()
   return nb.bound(record)
  def reference(path):return self.ref(path)if path in self.mem else nb.reference(path)
  def load(record,label):
   assert record in (nb.PREP_HELPER_REF,self.ref(nc.SOURCE)),record
   return helper if record==nb.PREP_HELPER_REF else self.fakec
  bn.update(bound=bound,reference=reference,load=load)
  for name in ("builder_review_v1","prepare_root_v1"):bn[name]=clone(getattr(nb,name),bn)
  self.bn=bn
  if mode=="late":
   end=datetime.fromisoformat(self.expiry)
   class Clock:
    count=0
    @classmethod
    def fromisoformat(cls,value):return datetime.fromisoformat(value)
    @classmethod
    def now(cls,tz):
     cls.count+=1
     return end-timedelta(seconds=86700)if cls.count>=3 else end-timedelta(days=3)
   bn["datetime"]=Clock
 def value(self,name):
  values=[v[0]for k,v in self.mem.items()if k.startswith("research/reviews/C3_full30Mb_eval_ROOT_interleaved_")and k.endswith("/"+name)]
  assert len(values)==1,(name,len(values))
  return copy.deepcopy(values[0])
 def run(self):
  ev=self.put(self.evidence)
  args=types.SimpleNamespace(phase=self.phase,root_directive=nb.DIRECTIVE,instance="SYNTHETIC_"+self.phase+"_"+self.mode,
   main_review_path=nc.LEGACY_MAIN_REVIEW_REF["path"],main_review_sha256=nc.LEGACY_MAIN_REVIEW_REF["sha256"],
   builder_review_path=self.builder_review_ref["path"],builder_review_sha256=self.builder_review_ref["sha256"],
   evidence_path=ev["path"],evidence_sha256=ev["sha256"],expires_at_utc=self.expiry)
  return self.bn["prepare_root_v1"](args)
summaries=[]
for phase in nc.PHASES:
 f=CompleteFlow(phase);result=f.run()
 req=f.value("request.root_active_v1.json");spec=f.value("input.json");storage=f.value("storage.json")
 assert result["consumer_binding_count"]==64 and f.applies==1
 assert spec["source_sha256"]==effective["source_sha256"]
 assert spec["main_source_ref"]==file_ref(nc.ENTRY)and spec["root_driver_ref"]==file_ref(nc.DRIVER)
 assert storage["reservation"]==f.value("reserve_receipt.json")["value"]
 assert storage["initial_ledger_pins"]==f.value("reserve_receipt.json")["initial_ledger_pins"]
 assert req["expires_at_utc"]==spec["expires_at_utc"]==storage["expires_at_utc"]==f.expiry
 assert f.value("config.json")["output_cap_bytes"]==storage["reserved_new_bytes"]==33554432
 assert f.value("admitted.json")["execution_started"]is False
 assert ("VAL_truth_capability"in req)==(phase=="score")
 if phase.startswith("predict"):
  own=nc.VAL[int(phase[-3:])]
  for p,e in f.plan_evidence.items():
   if p!=own:assert not any(r["path"]in f.reads for r in (e["completion_ref"],e["actual_wait_ref"],e["plan_ref"],e["close_receipt_ref"]))
 shadowed={nc.SAVED_ACCEPTANCE_REF["path"],*(r["path"]for r in nc.OLD_INPUT_REFS.values())}
 assert not any((ROOT/k).exists()for k in f.mem if k not in shadowed)
 assert not any(Path(p).suffix in (".pt",".npy",".npz",".trees")for p in f.reads)
 argv=result["exact_one_ROOT_stage_execution_argv"]
 assert argv[:5]==["setsid",str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/nc.DRIVER),"--launch"]
 f.fakec.set_active_spec_v1=lambda r:f.ns["validate_input_v1"](r)
 dn=dict(argparse=importlib.import_module("argparse"),json=json,c=f.fakec,SOURCE=nc.DRIVER,__doc__="inert complete driver")
 dn["bind_reviewed_supervisor_v1"]=lambda:lambda p:0
 _,driver_nodes=functions(CAND+"/root_driver_v1.py")
 exec(compile(ast.Module(body=[driver_nodes["main"]],type_ignores=[]),"<exact_new_driver_main>","exec"),dn)
 assert dn["main"](argv[4:])==0
 summaries.append(dict(phase=phase,roles=64,pre_reserve_metadata_charge=f.precharge,total_packet_metadata_charge=f.charge,
                       required_plan_count=1 if phase.startswith("predict")else 5))
check("all4_full_new_preparation_request_input_storage_driver_flows",summaries)
for mode in ("wrapped","overflow","race","late"):
 f=CompleteFlow("predict_val_003",mode);reject(f.run)
 assert f.applies==(1 if mode in ("race","late")else 0)
 if mode in ("wrapped","overflow"):assert not any(k.endswith("/reserve_authority.json")for k in f.mem)
 assert not any(k.endswith("/admitted.json")for k in f.mem)if mode!="late"else True
 check("complete_flow_"+mode+"_failstop_no_retry_or_execution")
for phase in ("predict_val_000","predict_val_001","predict_val_002","validate_saved600"):
 f=CompleteFlow("predict_val_003");f.phase=phase;f.rejected_before_writes()
check("old3_and_validation_never_admitted_by_new_builder")
for phase in ("predict_val_003","predict_val_004","matrix","score"):
 for p in ([nc.VAL[int(phase[-3:])]]if phase.startswith("predict")else nc.VAL):
  for change in ("missing","unclosed","undrained","unconsumed"):
   f=CompleteFlow(phase)
   if change=="missing":f.evidence["closed_plan_evidence_by_parent"].pop(p)
   elif change=="unclosed":f.evidence["closed_plan_evidence_by_parent"][p]["close_receipt_ref"]=None
   else:
    key="actual_wait_ref"if change=="undrained"else"outer_terminal_ref"
    value=f.read(f.evidence["closed_plan_evidence_by_parent"][p][key])
    value.update(dict(all_owned_processes_and_groups_drained=False)if change=="undrained"else dict(actual_terminal_consumed=False))
    f.evidence["closed_plan_evidence_by_parent"][p][key]=f.put(value)
   f.rejected_before_writes()
check("all4_phase_plan_missing_close_drain_outer_negatives_before_packet")
for idx,p in enumerate(nc.VAL):
 for change in ("wrong_driver","wrong_input","wrong_map","wait_false","outer_false","undrained","missing_epoch","truth_opened"):
  f=CompleteFlow("matrix");e=f.evidence["VAL_predictions"][p]
  if change in ("wait_false","outer_false","undrained"):
   key="outer_terminal_ref"if change=="outer_false"else"actual_wait_ref";v=f.read(e[key])
   v.update({"actual_exit_code":False}if change=="outer_false"else{"returncode":False}if change=="wait_false"else{"all_owned_processes_and_groups_drained":False})
   e[key]=f.put(v)
  else:
   env=f.read(e["completion_ref"])
   if change=="wrong_driver":env["root_driver_ref"]=file_ref(nc.DRIVER)if idx<3 else nc.LEGACY_DRIVER_REF
   elif change in ("wrong_input","wrong_map"):
    inp=f.read(env["input_ref"])
    if change=="wrong_map":inp["source_sha256"]=effective["source_sha256"]if idx<3 else oldmanifest["source_sha256"]
    elif idx>=3:inp["main_source_ref"]=nc.LEGACY_ENTRY_REF
    env["input_ref"]=f.put(inp)
   else:
    six=copy.deepcopy(f.sixes[p])
    if change=="missing_epoch":six["prediction_seal_refs"].pop()
    else:
     seal=f.read(six["prediction_seal_refs"][-1]);seal["private_truth_opened"]=True;six["prediction_seal_refs"][-1]=f.put(seal)
    env["result_ref"]=f.put(six)
   oldenv=f.read(e["completion_ref"]);e["completion_ref"]=f.put(env)
   # Keep genuine wait/input/driver joins in negatives aimed at source routing.
   wait=f.read(e["actual_wait_ref"]);wait.update(completed_ref=e["completion_ref"],input_ref=env["input_ref"],driver_ref=env["root_driver_ref"]);e["actual_wait_ref"]=f.put(wait)
   outer=f.read(e["outer_terminal_ref"]);outer.update(completed_ref=e["completion_ref"],actual_wait_ref=e["actual_wait_ref"]);e["outer_terminal_ref"]=f.put(outer)
  f.rejected_before_writes()
check("all5_fixed_mixed_producer_input_map_and6seal_negatives")
for change in ("missing","duplicate","wrong_focal","wrong_checkpoint","wrong_prediction","truth_opened"):
 f=CompleteFlow("score");m=copy.deepcopy(f.matrix)
 if change=="missing":m["cells"].pop()
 elif change=="duplicate":m["cells"][-1]=copy.deepcopy(m["cells"][0])
 elif change=="wrong_focal":m["cells"][-1]["focal_registry_ref"]=dict(path=BASE+"/SYNTHETIC_foreign.json",sha256="0"*64)
 elif change=="wrong_checkpoint":m["cells"][-1]["checkpoint_ref"]=f.candidates[nc.EPOCHS[0]]
 elif change=="wrong_prediction":m["cells"][-1]["prediction_ref"]=dict(path=BASE+"/SYNTHETIC_foreign.npy",sha256="0"*64)
 else:
  s=f.read(m["cells"][-1]["prediction_seal_ref"]);s["private_truth_opened"]=True;m["cells"][-1]["prediction_seal_ref"]=f.put(s)
 f.evidence["matrix"]=f.stage("matrix",f.put(m));f.reads=[];f.rejected_before_writes()
 assert not any(r["path"]in f.reads for r in f.simulations.values())
check("all30_matrix_and_focal_checkpoint_prediction_seals_before_truth_metadata")
for phase in ("predict_val_003","predict_val_004","matrix"):
 f=CompleteFlow(phase);f.evidence["VAL_truth_capability"]={};f.rejected_before_writes()
check("truth_capability_score_only")
# Author fixtures have no __main__ execution and write only this independent result.
author=load_source(CAND+"/inert_integration_checks_v1.py","_independent_author_interleaved_inert")
result=author.run();expected=json.loads((ROOT/CAND/"inert_checks.v1.json").read_text())
for x in (result,expected):x.pop("completed_utc",None)
assert result==expected and result["check_count"]==62
with (ROOT/BASE/"author_fixture_reproduction.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
check("author62_new_inert_checks_reproduced")
assert not any(n in sys.modules for n in ("torch","numpy","torch_geometric","tskit"))
value=dict(schema="C3_interleaved_complete_independent_VFS_checks_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 all_active_acceptance_plan_guard_reservation_and_driver_endpoints_synthetic=True,scientific_payload_reads_or_hashes=False,
 real_guard_calls_probes_accounting_admissions_jobs_or_Git=False,sealed_candidate_unchanged=True)
with (ROOT/BASE/"independent_complete_checks.v1.json").open("x")as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=value["status"],checks=len(checks),summaries=summaries)))
