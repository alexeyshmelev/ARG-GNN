"""Independent full-schema VAL004 source checks; all mutations/accounting endpoints virtual."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace,FunctionType
import ast,contextlib,copy,difflib,hashlib,importlib.util,io,json,runpy,sys
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/American_C3_VAL004_fresh7500MB_authority_source_20261016_v1"
B="research/reviews/American_C3_VAL004_fresh7500MB_independent_review_20261016_v1"
pins=json.loads((R/B/"hash_verification.v1.json").read_text())
known={x["path"]:dict(path=x["path"],sha256=x["sha256"])for x in pins["refs"]}
def actual_ref(p):return dict(path=p,sha256=hashlib.sha256((R/p).read_bytes()).hexdigest())
def load(p,label):
 s=importlib.util.spec_from_file_location(label,R/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def funcs(p):
 t=(R/p).read_text()
 return t,{n.name:n for n in ast.parse(t).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SOURCE_ONLY",**({"detail":detail}if detail is not None else {})))
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):return
 raise AssertionError("Expected rejection")
checks=[]
real=load(C+"/root_contract_v1.py","_VAL004_independent_inactive_contract")
prior=real.actual_prior("american_c3_val_004","native_paths")
assert prior==real.read(real.ref(C+"/frozen_infer_prerequisite.v1.json"))
native=real.scientific_config("american_c3_val_004","native_paths",prior,False)
assert native==real.read(real.ref(C+"/native_paths.scientific_config.template.inactive.v1.json"))
check("actual_frozen_VAL004_simulation_infer_wait_drain_outer_and_ordinary_close_metadata")
old="research/reviews/American_C3_VAL003004_downward8GB_source_20261016_v1"
for name in ("root_contract_v1.py","root_driver_v1.py","worker_entry_v1.py","plan_worker_v1.py"):
 delta="".join(difflib.unified_diff((R/old/name).read_text().splitlines(True),(R/C/name).read_text().splitlines(True),fromfile=old+"/"+name,tofile=C+"/"+name))
 assert delta==(R/C/(name+".patch")).read_text(),name
ot,on=funcs(old+"/root_driver_v1.py");nt,nn=funcs(C+"/root_driver_v1.py")
for name in ("execute","sample_scope","boundary","worker_python","validate_completion"):
 assert ast.dump(on[name],include_attributes=False)==ast.dump(nn[name],include_attributes=False),name
for name in ("cpu_slot","validate_slot_resources","scientific_config","existing_simulation","observation"):
 _,a=funcs(old+"/root_contract_v1.py");_,b=funcs(C+"/root_contract_v1.py")
 assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
check("published_patches_exact_and_original_lifecycle_slot_science_bodies_preserved")
cap=real.read(real.ref(C+"/caps.template.inactive.v1.json"))
oldcap=real.read(real.ref(old+"/caps.template.inactive.v1.json"))
assert cap["stages"]["native_paths"]==dict(oldcap["stages"]["native_paths"],reservation_bytes=7500000000)
assert cap["stages"]["plan"]==oldcap["stages"]["plan"]
cfg=real.read(real.read(real.ref(C+"/scientific_basis.v1.json"))["native_paths_config_ref"])
assert all(native[k]==cfg[k]for k in cfg if k not in ("execution_enabled","observation_ref","path_sampling_seed"))
assert native["path_sampling_seed"]==93005 and native["output_cap_bytes"]==16000000000
# Pure analytic bound from original source, without importing/executing its scientific module.
txt,nd=funcs("src/arggnn/american_c3_paths_v1.py")
ns={};exec(compile(ast.Module(body=[nd["storage_bound"]],type_ignores=[]),"<original_pure_analytic_bound>","exec"),ns)
assert ns["storage_bound"](native)==15218324396
check("original_all_native_counts_products_seeds_runtime_and_analytic_ceiling_preserved")
# Reproduce author's top-level pure fixture under redirected output; it writes no files.
buf=io.StringIO()
with contextlib.redirect_stdout(buf):author=runpy.run_path(str(R/C/"inert_fixtures_v1.py"),run_name="_independent_author_fixture")
ar=author["result"];expected=json.loads((R/C/"inert_fixtures.sealed_source.v1.json").read_text())
assert ar==expected and ar["passed"]==54
ap=R/B/"author_fixture_reproduction.v1.json"
if ap.exists():assert json.loads(ap.read_text())==ar
else:
 with ap.open("x")as f:json.dump(ar,f,sort_keys=True,indent=2);f.write("\n")
check("author54_source_only_checks_reproduced")
state_model=json.loads((R/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/inert_fixtures.final.v1.json").read_text())["full_role_models"]["guard_state"]
for row in state_model:
 if row["path"]==real.APPROVED_GUARDS[1]["source_manifest_ref"]["path"]:row.update(real.APPROVED_GUARDS[1]["source_manifest_ref"])
# Do not read slot state/inodes or acquire slots. Preserve their literal pinned receipt/body above.
class Flow:
 def __init__(self,stage="native_paths",mode="normal"):
  self.c=load(C+"/root_contract_v1.py","_VAL004_independent_virtual_"+stage+"_"+mode)
  c=self.c;self.mem={};self.dirs=set();self.events=[];self.applies=0;self.stage=stage;self.mode=mode
  self.original_read=c.read;self.original_raw=c.raw;self.original_ref=c.ref;self.sizes={}
  outer=self
  class VPath(type(Path())):
   def exists(self):
    key=str(self.relative_to(R))
    if key.startswith(C+"/admitted_")or key.startswith("experiments/results/American_C3_val004_full30mb_")and key.endswith("_fresh7500_v1"):
     return str(self)in outer.dirs or key in outer.mem
    return Path(self).exists()
   def is_symlink(self):return False
   def mkdir(self,*a,**kw):
    assert str(self).startswith(str(R/C)+"/admitted_")
    assert str(self)not in outer.dirs;outer.dirs.add(str(self));outer.events.append("virtual_mkdir")
  def local(path):
   p=Path(path);p=p if p.is_absolute()else R/p
   assert p.is_relative_to(R)and".."not in p.parts
   return VPath(p)
  def ref(path):
   p=local(path);key=str(p.relative_to(R))
   if key in outer.mem:return copy.deepcopy(outer.mem[key][0])
   return known[key]if key in known else outer.original_ref(key)
  def raw(record):
   if record["path"]in outer.mem:
    rr,value,data=outer.mem[record["path"]];assert rr==record
    return data
   assert Path(record["path"]).suffix in (".py",".json",".lock",".md",".sh")
   return outer.original_raw(record)
  def read(record,cap=32*2**20):
   data=raw(record);assert len(data)<=cap
   return json.loads(data)
  def save(folder,name,value,cap=2**20):
   key=str((local(folder)/name).relative_to(R));assert key not in outer.mem
   data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();assert len(data)<=cap
   rr=dict(path=key,sha256=hashlib.sha256(data).hexdigest())
   outer.mem[key]=(rr,copy.deepcopy(value),data);outer.sizes[key]=len(data)
   outer.events.append("virtual_save:"+name);return rr
  self.ref=ref;self.read=read;self.local=local;self.save=save
  c.ref=ref;c.raw=raw;c.read=read;c.local=local;c.save=save
  for rr in state_model:
   if rr["path"]not in known:self.mem[rr["path"]]=(copy.deepcopy(rr),{},b"{}")
  c.source_paths=lambda:tuple(r["path"]for r in self.original_read(self.original_ref(c.MANIFEST))["refs"])
  self.fixed=datetime(2026,10,17,0,0,tzinfo=timezone.utc)
  class Clock:
   @classmethod
   def now(cls,tz):return outer.fixed
   @classmethod
   def fromisoformat(cls,s):return datetime.fromisoformat(s)
  c.datetime=Clock
  self.put=lambda label,value:save(R/B,"SYNTHETIC_"+stage+"_"+mode+"_"+label+".json",value)
  review=self.put("review",dict(verdict="SOURCE_PASS",blockers=[],source_handoff_ref=ref(C+"/source_handoff.v1.json")))
  choice=c.APPROVED_GUARDS[1]
  policy=self.put("policy",dict(choice,source_review_ref=review))
  selector=self.put("selector",dict(choice,active=True,root_authorizer="root_agent",selector_source_ref=c.SELECTOR_SOURCE_REF,
    repair_authority_ref=c.REPAIR_REF,source_review_ref=review,policy_ref=policy,accounting_mode="native7_H050_stopped_retention_v1",
    journal=".runtime/storage/native7_H050_stopped_retention_v1"))
  budget=SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py",
                         v4=SimpleNamespace(mutex=lambda root:contextlib.nullcontext()))
  self.g=SimpleNamespace(MODE="native7_H050_stopped_retention_v1",JOURNAL=".runtime/storage/native7_H050_stopped_retention_v1",budget=budget)
  class Selected:
   def load_selected_guard_v1(self,record):
    assert record==selector;return outer.g
  c._SELECTOR_MODULE=Selected()
  auth=c.read(ref(C+"/root_authority.INACTIVE.template.v1.json"))
  auth.update(active=True,adapter_manifest_ref=ref(C+"/source_manifest.v1.json"),source_review_ref=review,storage_policy_selector_ref=selector,
   created_at_utc=self.fixed.isoformat(),expires_at_utc=(self.fixed+timedelta(hours=24)).isoformat(),old_v3_and_down8_future_routes_revoked=True)
  self.auth=auth;self.authority=self.put("authority",auth);self.review=review;self.selector=selector
  self.original_guard=c.original_guard
  c.original_guard=lambda:self.g
  if stage=="plan":
   _,folder,out,key=c.predecessor_paths_v1("american_c3_val_004","plan")
   obs=prior["observation_ref"]
   complete=dict(status="completed_native_C3_path_preparation",observation_ref=obs,parent_id="american_c3_val_004",
     query_focal_count=10000,query_ancestry_truth_opened=False,native_split_seed=42,native_train_fraction=.9,
     class_order=["EUR","EAS","AFR"],focals_ref=dict(path=B+"/SYNTHETIC_NEVER_READ_focals.npz",sha256="e"*64))
   cp=save(R/out,"completed.json",complete)
   wr=save(R/folder/"worker_aux","actual_wait.json",dict(completed_ref=cp,actual_direct_worker_wait=True,returncode=0,parent_id="american_c3_val_004",
    phase="native_paths_val_004",input_ref=dict(path=folder+"/input.json",sha256="d"*64),all_owned_processes_and_groups_drained=True,operational_failure=None))
   tr=save(R/folder,"outer_terminal.root_actual_v1.json",dict(root_authorizer="root_agent",actual_terminal_consumed=True,actual_exit_code=0,completed_ref=cp,direct_wait_ref=wr))
   save(R/folder/"closed","close_receipt.json",dict(key=key,value=dict(future_writes_and_relaunch_revoked=True,proof=dict(completed_ref=cp,direct_wait_ref=wr))))
  self.request=c.bind("american_c3_val_004",stage,review,selector,self.authority)
  q=c.read(self.request);self.q=q
  returned={str(i):copy.deepcopy(r)for i,r in enumerate(state_model)}
  if mode=="overflow":
   for i in range(20):returned["extra"+str(i)]=self.put("extra"+str(i),{})
  own={}
  if mode=="old_key":own[c.never_admitted_stage_keys_v1()[1]]={}
  if mode=="new_closed_key":own[c.never_admitted_stage_keys_v1()[2]]={}
  closed={q["prior"]["closed_key"]}
  if mode=="prior_open":closed=set()
  state=(own,closed,dict(SYNTHETIC_EXACT_PREFIX=True),returned)
  self.g._state_locked=lambda *a,**kw:state
  self.g._pins=lambda view:copy.deepcopy(view)
  self.g.operation_draft_v1=lambda op,req,**kw:dict(operation=op,request=copy.deepcopy(req),**kw)
  def apply(record,**kw):
   self.applies+=1;assert self.applies==1
   a=c.read(record)
   if mode=="race":raise ValueError("synthetic exact-prefix race")
   return dict(value=copy.deepcopy(a["request"]),initial_ledger_pins=dict(SYNTHETIC_RETURNED_PREFIX=True))
  self.g.apply_reserve_v1=apply
  text,nodes=funcs(C+"/root_driver_v1.py")
  dn=dict(c=c,need=c.require,local=local,ref=ref,read=read,ROOT=R,SOURCE=c.DRIVER)
  exec(compile(ast.Module(body=[nodes["admit"]],type_ignores=[]),"<exact_VAL004_admit_virtual_endpoints>","exec"),dn)
  self.admit=lambda:dn["admit"](self.request)
 def charge(self):
  return sum(size+8192 for p,size in self.sizes.items()if p.startswith(C+"/")and not p.endswith("/completed.json"))
summaries=[]
for stage in real.STAGES:
 f=Flow(stage);r=f.admit();s,q,cfg=f.c.input_spec(r["input_ref"])
 assert f.applies==1 and r["consumer_binding_count"]==54
 assert all(x in s["storage"]["reservation"]["consumer_bindings"]for x in state_model)
 assert s["storage"]["initial_ledger_pins"]==dict(SYNTHETIC_RETURNED_PREFIX=True)
 assert s["storage"]["reservation"]["reserved_bytes"]==(7500000000 if stage=="native_paths"else 33554432)
 assert s["created_at_utc"]==q["created_at_utc"]==f.auth["created_at_utc"]
 assert s["expires_at_utc"]==q["expires_at_utc"]==f.auth["expires_at_utc"]
 assert f.charge()<=524288,f.charge()
 summaries.append(dict(stage=stage,returned_roles=len(state_model),complete_roles=r["consumer_binding_count"],source_count=len(s["source_sha256"]),root_metadata_charge=f.charge()))
 for missing in (f.c.SELECTOR_SOURCE_REF,f.selector,f.authority,known[f.c.MANIFEST]):
  oldstore=copy.deepcopy(s["storage"]);bad=copy.deepcopy(oldstore)
  bad["reservation"]["consumer_bindings"]=[x for x in bad["reservation"]["consumer_bindings"]if x!=missing]
  sr=s["refs"]["storage"];ir=r["input_ref"];oldsr=f.mem[sr["path"]];oldir=f.mem[ir["path"]]
  def override(rr,value):
   raw=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
   f.mem[rr["path"]]=(rr,copy.deepcopy(value),raw)
  override(sr,bad);badinput=copy.deepcopy(s);badinput["storage"]=bad;override(ir,badinput)
  reject(lambda:f.c.input_spec(ir));f.mem[sr["path"]]=oldsr;f.mem[ir["path"]]=oldir
check("both_complete_bind_admit_input_flows_full_source_maps_and_role_closure",summaries)
for mode in ("overflow","old_key","new_closed_key","prior_open","race"):
 f=Flow("native_paths",mode);reject(f.admit)
 assert f.applies==(1 if mode=="race"else 0)
 if mode in ("old_key","new_closed_key","prior_open"):assert not f.dirs
 if mode=="overflow":assert not any(p.endswith("/reserve_authority.json")for p in f.mem)
 assert not any(p.endswith("/admitted.json")for p in f.mem)
check("full_admit_role65_old_new_key_prior_closed_and_single_prefix_race_failstops")
# Strict exact authority duration and runtime reserve boundaries.
for stage in real.STAGES:
 f=Flow(stage);c=f.c;auth=copy.deepcopy(f.auth);end=datetime.fromisoformat(auth["expires_at_utc"])
 for now in (end-timedelta(seconds=c.CAPS["wall_seconds"]+300),end):
  reject(lambda:c.authority_metadata(auth,f.review,f.selector,stage,now=now))
 for patch in (dict(old_v3_and_down8_future_routes_revoked=False),dict(query_truth_access_authorized=True),
   dict(created_at_utc="2026-10-16T19:00:00+00:00",expires_at_utc="2026-10-17T19:00:00+00:00")):
  bad=dict(auth,**patch);reject(lambda:c.authority_metadata(bad,f.review,f.selector,stage,now=f.fixed))
check("both_authority_full_runtime_margin_and_old_clock_truth_revocation_negatives")
assert not any(n in sys.modules for n in ("torch","numpy","torch_geometric","tskit","scripts.project_budget_v9"))
v=dict(schema="VAL004_fresh7500_independent_complete_virtual_checks_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 actual_original_frozen_simulation_and_infer_compact_metadata_authenticated=True,
 all_guard_selector_accounting_mutation_and_new_product_endpoints_synthetic=True,
 no_actual_guard_resource_slot_probe_accounting_admission_retention_job_or_Git=True,scientific_payloads_never_read_or_hashed=True)
with (R/B/"independent_complete_checks.v1.json").open("x")as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=v["status"],checks=len(checks),flow_summaries=summaries)))
