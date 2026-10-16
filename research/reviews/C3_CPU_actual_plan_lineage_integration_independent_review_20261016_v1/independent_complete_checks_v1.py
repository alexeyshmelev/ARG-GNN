"""Complete independent inert checks for actual-plan-lineage C3 repair."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,contextlib,copy,hashlib,importlib,importlib.util,io,json,runpy,sys,types,difflib
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/C3_CPU_actual_plan_lineage_integration_independent_review_20261016_v1"
CAND="research/reviews/C3_CPU_actual_plan_lineage_integration_source_20261016_v1"
sys.path.insert(0,str(ROOT))
def load_source(path,label):
 s=importlib.util.spec_from_file_location(label,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def file_ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def functions(path):
 t=(ROOT/path).read_text();return t,{n.name:n for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def clone(fn,ns):
 f=types.FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__);f.__kwdefaults__=fn.__kwdefaults__;return f
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError,AttributeError):return
 raise AssertionError("Expected rejection")
checks=[]
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SOURCE_ONLY",**({"detail":detail}if detail is not None else {})))
nc=load_source(CAND+"/acceptance_contract_v1.py","_independent_actual_plan_contract")
nb=load_source(CAND+"/prepare_evaluation_root_v1.py","_independent_actual_plan_builder")
effective=nc.manifest_v1();oldmanifest=nc.legacy.manifest_v1()
assert len(effective["source_sha256"])==113 and len(oldmanifest["source_sha256"])==81
assert all(effective["source_sha256"][p]==s for p,s in oldmanifest["source_sha256"].items())
actuale=nc.read(nc.ref(CAND+"/actual_VAL003_evidence.v1.json"))
actualplan=nc.plan_ready_v1(actuale,"american_c3_val_003")
env,spec,req,wait=nc.plan_producer_lineage_v1(actuale,"american_c3_val_003")
assert "driver_ref"not in wait and len(spec["source_sha256"])==323
reject(lambda:nc.legacy.plan_ready_v1(actuale,"american_c3_val_003"))
assert "driver_ref"not in nc.read(actuale["actual_wait_ref"])
check("real003_accepts_unchanged_and_legacy_rejection_preserved")
prior="research/reviews/C3_CPU_interleaved_guarded_integration_source_20261016_v1"
for name in ("acceptance_contract_v1.py","prepare_evaluation_root_v1.py","root_driver_v1.py","worker_entry_v1.py"):
 actual="".join(difflib.unified_diff((ROOT/prior/name).read_text().splitlines(True),(ROOT/CAND/name).read_text().splitlines(True),fromfile=prior+"/"+name,tofile=CAND+"/"+name))
 assert actual==(ROOT/CAND/(name+".patch")).read_text()
_,newworker=functions(CAND+"/worker_entry_v1.py");_,oldworker=functions(prior+"/worker_entry_v1.py")
for name in newworker:assert ast.dump(newworker[name],include_attributes=False)==ast.dump(oldworker[name],include_attributes=False),name
check("all_worker_bodies_and_exact_published_source_deltas")
# Keep future004 strictly virtual, using source-fixture setup only through its synthetic plan.
ptext=(ROOT/CAND/"actual_plan_lineage_checks_v1.py").read_text()
pnodes=[n for n in ast.parse(ptext).body if getattr(n,"lineno",999)<117]
pns={};exec(compile(ast.Module(body=pnodes,type_ignores=[]),"<sealed_fixture_real003_and_virtual004_setup>","exec"),pns)
futuredb=copy.deepcopy(pns["db"]);futuree=copy.deepcopy(pns["e4"]);futureF=pns["F"]
futureplan=copy.deepcopy(futuredb[futuree["plan_ref"]["path"]])
child=json.loads((ROOT/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/inert_fixtures.final.v1.json").read_text())["full_role_models"]["guard_state"]
approved=nc.read(nc.ref(nc.APPROVED_SELECTOR_SOURCES))["approved"]
for row in child:
 if row["path"]==approved[1]["source_manifest_ref"]["path"]:row.update(approved[1]["source_manifest_ref"])

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
  for name in ("terminal0_v1","acceptance_inputs_v1"):
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
    packet_directory="research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_"+phase+"_SYNTHETIC")
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
  # New real plan adapter: actual003 metadata plus explicitly synthetic future004.
  for path,value in futuredb.items():self.mem[path]=(copy.deepcopy(value),futureF(path)["sha256"])
  self.plan_evidence[nc.VAL[3]]=copy.deepcopy(actuale)
  self.plan_evidence[nc.VAL[4]]=copy.deepcopy(futuree)
  self.plans[nc.VAL[3]]=copy.deepcopy(actualplan);self.plans[nc.VAL[4]]=copy.deepcopy(futureplan)
  if phase.startswith("predict"):
   own=nc.VAL[int(phase[-3:])]
   self.evidence["closed_plan_evidence_by_parent"]={own:copy.deepcopy(self.plan_evidence[own])}
  else:self.evidence["closed_plan_evidence_by_parent"]=copy.deepcopy(self.plan_evidence)
  cells=[]
  for parent in nc.VAL:
   seals=[]
   for epoch in nc.EPOCHS:
    seal=copy.deepcopy(self.seals[(epoch,parent)]);seal["focal_registry_ref"]=self.plans[parent]["focal_registry_ref"]
    self.seals[(epoch,parent)]=seal;sr=self.put(seal);seals.append(sr)
    cells.append(dict(epoch=epoch,parent_id=parent,checkpoint_ref=self.candidates[epoch],prediction_seal_ref=sr,
                      prediction_ref=seal["prediction_ref"],focal_registry_ref=seal["focal_registry_ref"]))
   self.sixes[parent]=dict(status="sealed_C3_one_original_VAL_all6_candidates",parent_id=parent,
    training_completion_ref=self.training_ref,private_truth_opened=False,prediction_seal_refs=seals)
  self.matrix["cells"]=cells
  self.predictions={p:self.stage("predict_val_"+p[-3:],self.put(self.sixes[p]))for p in nc.VAL}
  self.matrix_evidence=self.stage("matrix",self.put(self.matrix))
  if phase=="matrix":self.evidence["VAL_predictions"]=copy.deepcopy(self.predictions)
  if phase=="score":self.evidence["matrix"]=copy.deepcopy(self.matrix_evidence)
  for parent in nc.VAL:
   obsref=self.plans[parent]["observation_ref"];obs=self.read(obsref)
   if "source_observation_ref"not in obs:
    assert parent==nc.VAL[4]
    obs["source_observation_ref"]=self.put(dict(SYNTHETIC_ONLY_source_observation=True,parent_id=parent))
    self.mem[obsref["path"]]=(copy.deepcopy(obs),obsref["sha256"])
   self.simulations[parent]=self.put(dict(status="completed_C3_simulation_reconstruction",parent_id=parent,class_order=["EUR","EAS","AFR"],observation_ref=obs["source_observation_ref"]))
  if phase=="score":self.evidence["VAL_simulation_completion_refs"]=copy.deepcopy(self.simulations)
  self.use_child=mode=="child"
  choice=approved[1 if self.use_child else 0]
  self.guard.MODE="native7_H050_stopped_retention_v1"if self.use_child else"native14_stopped_retention_v3"
  self.guard.JOURNAL=".runtime/storage/"+self.guard.MODE
  selected_review=self.put(dict(SYNTHETIC_ONLY=True))
  policy=self.put(dict(choice,active=True,root_authorizer="root_agent",source_review_ref=selected_review,journal=self.guard.JOURNAL))
  self.selector=self.put(dict(choice,active=True,root_authorizer="root_agent",selector_source_ref=nc.SELECTOR_SOURCE_REF,
   repair_authority_ref=nc.REPAIR_REF,policy_ref=policy,source_review_ref=selected_review,
   accounting_mode=self.guard.MODE,journal=self.guard.JOURNAL))
  outer=self
  class Selector:
   def load_selected_guard_v1(self,record):
    assert record==outer.selector
    return outer.guard
  nn["_SELECTOR_MODULE"]=Selector()
  nn["select_storage_metadata_v1"](self.selector)
  self.fakec=types.SimpleNamespace(**nn)
  self.builder_review=json.loads((ROOT/CAND/"ROOT_review.INACTIVE.template.v1.json").read_text())
  self.builder_review.update(independent=True,verdict="SOURCE_PASS",source_handoff_ref=self.ref(nc.HANDOFF),integration_manifest_ref=self.ref(nc.MANIFEST))
  self.builder_review_ref=self.put(self.builder_review)
  original_inspect=self.guard.inspect_v1
  def inspect(**kw):
   value=original_inspect(**kw)
   # Baseline actual inherited role set only; a new returned role must fail65.
   value["bindings"]={str(i):copy.deepcopy(r)for i,r in enumerate(child)}if self.use_child else copy.deepcopy(oldmanifest["current_guard_roles"])
   if mode=="overflow":
    for i in range(20):value["bindings"]["SYNTHETIC65_"+str(i)]=dict(path=BASE+"/SYNTHETIC65_"+str(i)+".json",sha256="7"*64)
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
  values=[v[0]for k,v in self.mem.items()if k.startswith("research/reviews/C3_full30Mb_eval_ROOT_actual_plan_lineage_")and k.endswith("/"+name)]
  assert len(values)==1,(name,len(values))
  return copy.deepcopy(values[0])
 def run(self):
  ev=self.put(self.evidence)
  args=types.SimpleNamespace(phase=self.phase,root_directive=nb.DIRECTIVE,instance="SYNTHETIC_"+self.phase+"_"+self.mode,
   main_review_path=nc.LEGACY_MAIN_REVIEW_REF["path"],main_review_sha256=nc.LEGACY_MAIN_REVIEW_REF["sha256"],
   builder_review_path=self.builder_review_ref["path"],builder_review_sha256=self.builder_review_ref["sha256"],
   evidence_path=ev["path"],evidence_sha256=ev["sha256"],selector_path=self.selector["path"],selector_sha256=self.selector["sha256"],expires_at_utc=self.expiry)
  return self.bn["prepare_root_v1"](args)
summaries=[]
for mode in ("normal","child"):
 for phase in nc.PHASES:
  f=CompleteFlow(phase,mode);result=f.run();spec=f.value("input.json");req=f.value("request.root_active_v1.json");storage=f.value("storage.json")
  expected=61 if mode=="child"else 54
  assert result["consumer_binding_count"]==expected and f.applies==1
  assert storage["storage_policy_selector_ref"]==f.selector and spec["storage_policy_selector_ref"]==f.selector
  assert spec["source_sha256"]==effective["source_sha256"]
  returned=child if mode=="child"else list(oldmanifest["current_guard_roles"].values())
  assert all(r in storage["reservation"]["consumer_bindings"]for r in returned)
  assert storage["initial_ledger_pins"]==f.value("reserve_receipt.json")["initial_ledger_pins"]
  assert storage["reservation"]==f.value("reserve_receipt.json")["value"]
  assert req["expires_at_utc"]==spec["expires_at_utc"]==storage["expires_at_utc"]==f.expiry
  assert f.charge<=524288
  summaries.append(dict(phase=phase,guard_choice=mode,roles=expected,root_metadata_charge=f.charge))
check("all4_full_builder_flows_current_and_child_actual003_virtual004",summaries)
for mode in ("wrapped","overflow","race","late"):
 f=CompleteFlow("predict_val_003",mode);reject(f.run)
 assert f.applies==(1 if mode in ("race","late")else 0)
 if mode in ("wrapped","overflow"):assert not any(p.endswith("/reserve_authority.json")for p in f.mem)
 check("complete_failstop_"+mode)
for phase in nc.PHASES:
 parents=[nc.VAL[int(phase[-3:])]]if phase.startswith("predict")else nc.VAL
 for parent in parents:
  f=CompleteFlow(phase);f.evidence["closed_plan_evidence_by_parent"].pop(parent);f.rejected_before_writes()
check("all4_missing_own_or_all5_closed_plans_reject_before_packet")
for badfield in ("driver_ref","returncode","all_owned_processes_and_groups_drained"):
 original=nc.read(actuale["actual_wait_ref"]);changed=copy.deepcopy(original)
 changed.update({"driver_ref":env["root_driver_ref"]}if badfield=="driver_ref"else{"returncode":False}if badfield=="returncode"else{"all_owned_processes_and_groups_drained":False})
 read0=nc.read;nc.read=lambda r,*a,**kw:copy.deepcopy(changed)if r==actuale["actual_wait_ref"]else read0(r,*a,**kw)
 reject(lambda:nc.plan_ready_v1(actuale,nc.VAL[3]));nc.read=read0
assert nc.read(actuale["actual_wait_ref"])==wait and "driver_ref"not in wait
check("independent_real003_manufactured_field_boolean_wait_and_drain_negatives")
# Reproduce both sealed source fixtures without their __main__ writers.
m=load_source(CAND+"/inert_integration_checks_v1.py","_independent_author_new_inert")
ar=m.run();expected=nc.read(nc.ref(CAND+"/inert_integration_checks.v1.json"))
for value in (ar,expected):value.pop("completed_utc",None)
assert ar==expected and ar["check_count"]==61
out=ROOT/BASE/"author_inert_reproduction.v1.json"
if out.exists():assert json.loads(out.read_text())==ar
else:
 with out.open("x")as f:json.dump(ar,f,sort_keys=True,indent=2);f.write("\n")
buf=io.StringIO()
with contextlib.redirect_stdout(buf):runpy.run_path(str(ROOT/CAND/"actual_plan_lineage_checks_v1.py"),run_name="_independent_author_actual")
ar=json.loads(buf.getvalue());expected=nc.read(nc.ref(CAND+"/actual_plan_lineage_checks.final.v1.json"))
normalizations=[]
for mode,model in expected["role_models"].items():
 for row in model["roles"]:
  if row["path"]==nc.MANIFEST:
   assert row["sha256"]=="340705eb5307a7fb3c53f09e10457e25b4eee0f6ddf6832ff6d3a977973a7223"
   normalizations.append(dict(mode=mode,preseal_manifest_ref=copy.deepcopy(row),sealed_manifest_ref=nc.ref(nc.MANIFEST)))
   row.update(nc.ref(nc.MANIFEST))
assert len(normalizations)==2 and ar==expected and ar["check_count"]==46
out=ROOT/BASE/"author_actual_reproduction.v1.json"
if out.exists():assert json.loads(out.read_text())==ar
else:
 with out.open("x")as f:json.dump(ar,f,sort_keys=True,indent=2);f.write("\n")
check("author61mixed_barrier_and46real_lineage_checks_reproduced",dict(only_two_preseal_self_manifest_evidence_rows_normalized=normalizations))
assert not any(n in sys.modules for n in ("torch","numpy","torch_geometric","tskit","scripts.project_budget_v9"))
result=dict(schema="C3_actual_plan_lineage_complete_independent_inert_checks_v1",status="PASS_SOURCE_ONLY",check_count=len(checks),checks=checks,
 all_guard_selector_policy_reservation_and_future004_artifacts_synthetic=True,real003_compact_metadata_read_without_modification=True,
 no_actual_guard_probe_accounting_admission_retention_jobs_payload_or_Git=True)
with(ROOT/BASE/"independent_complete_checks.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],checks=len(checks),flow_summaries=summaries)))
