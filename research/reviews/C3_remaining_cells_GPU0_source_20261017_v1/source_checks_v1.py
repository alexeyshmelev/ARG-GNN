"""Source-only complete entry/contract joins. Every operational object is a RAM double."""
from pathlib import Path as RealPath
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json,posixpath,types
ROOT=RealPath("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/C3_remaining_cells_GPU0_source_20261017_v1"
OLD=ROOT/"research/reviews/native_VAL000_downstream_execution_source_20261016_v2"
SCI=ROOT/"research/reviews/native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2"
checks=[];negative=[];docs={};files={};dirs=set();events=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):negative.append(label);return
 raise AssertionError("Invalid accepted: "+label)
def encoded(v):return (json.dumps(v,sort_keys=True,indent=2)+"\n").encode()
def put(path,value):
 data=encoded(value);r=dict(path=path,sha256=hashlib.sha256(data).hexdigest());files[path]=data;docs[path]=copy.deepcopy(value);return r
def source_ref(path):
 p=RealPath(path);p=p if p.is_absolute()else ROOT/p
 need(p.suffix in(".py",".json",".md"),"Only explicit source/metadata ref "+p.name)
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def ref(path):
 key=str(path);key=str(RealPath(key).relative_to(ROOT))if key.startswith(str(ROOT)+"/")else key
 if key in files:return dict(path=key,sha256=hashlib.sha256(files[key]).hexdigest())
 return source_ref(key)
def raw(r):
 if r["path"]in files:
  need(ref(r["path"])==r,"RAM ref exact");return files[r["path"]]
 p=ROOT/r["path"]
 # This is a source allowlist, never arbitrary scientific bytes.
 need(r["path"]in allowed,"Explicit immutable source allowlist "+r["path"])
 b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==r["sha256"],"Source exact "+p.name);return b
def read(r):
 if r["path"]in docs:
  need(ref(r["path"])==r,"RAM JSON exact");return copy.deepcopy(docs[r["path"]])
 return json.loads(raw(r))
class RP:
 def __init__(self,p):self.p=str(p)
 def __str__(self):return self.p
 def __fspath__(self):return self.p
 def __truediv__(self,x):return RP(self.p.rstrip("/")+"/"+str(x).lstrip("/"))
 def __eq__(self,x):return self.p==str(x)
 def __hash__(self):return hash(self.p)
 @property
 def name(self):return posixpath.basename(self.p)
 @property
 def suffix(self):return RealPath(self.p).suffix
 @property
 def parent(self):return RP(posixpath.dirname(self.p))
 def relative_to(self,x):return RP(str(RealPath(self.p).relative_to(str(x))))
 def is_relative_to(self,x):return RealPath(self.p).is_relative_to(str(x))
 def exists(self):return self.p in dirs or str(self.relative_to(ROOT))in files
 def is_dir(self):return self.p in dirs
 def is_symlink(self):return False
 def resolve(self):return self
 def mkdir(self,**kw):
  if self.p in dirs:raise ValueError("replay directory")
  dirs.add(self.p)
 def iterdir(self):
  children={str(ROOT/k)for k in files}|dirs
  return [RP(k)for k in sorted(children)if posixpath.dirname(k)==self.p]
 def lstat(self):
  data=files[str(self.relative_to(ROOT))]
  return types.SimpleNamespace(st_nlink=1,st_size=len(data),st_mode=0o100400)
def local(path):
 p=str(path);p=p if p.startswith("/")else str(ROOT)+"/"+p
 if posixpath.normpath(p)!=p or not p.startswith(str(ROOT)+"/"):raise ValueError("RAM containment")
 return RP(p)
def save(directory,name,value):return put(str((directory/name).relative_to(ROOT)),value)
def ram_module(path,name,bindings):
 tree=ast.parse(RealPath(path).read_text());body=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom)and(n.level or (n.module or "").startswith("research.")):
   for a in n.names:
    if a.name=="*":continue
    if a.asname or a.name not in bindings:need((a.asname or a.name)in bindings,"Explicit project import RAM replacement")
   continue
  body.append(n)
 tree.body=body;ast.fix_missing_locations(tree);m=types.ModuleType(name);m.__dict__.update(bindings)
 exec(compile(tree,"<whole_module_RAM_imports:"+RealPath(path).name+">","exec"),m.__dict__);return m
allowed={str(p.relative_to(ROOT))for p in D.iterdir()if p.suffix in(".py",".json",".md")}
allowed|={str(p.relative_to(ROOT))for p in SCI.iterdir()if p.suffix==".py"}
common=ram_module(ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/common_v1.py","RAM_common",{})
common.raw=raw;common.ref=ref;common.read=read;common.local=local;common.save=save

RP.is_file=lambda self:str(self.relative_to(ROOT))in files
RP.stat=RP.lstat
c=ram_module(D/"contract_v1.py","RAM_C3_remaining_contract",dict(original=common))
manifest=json.loads((D/"source_manifest.v1.json").read_text());allowed|={r["path"]for r in manifest["source_authentication_refs"]}
for r in manifest["executed_source_refs"]+manifest["source_authentication_refs"]:raw(r)
pins=c.pins_v1();p=pins["pins"]
for rr in pins["old_CPU_inputs"].values():allowed.add(read(rr)["root_stage_request_ref"]["path"])
frozen=read(p["FROZEN"]);acceptance=read(p["ARTIFACT_ACCEPTANCE"])
training=read(frozen["training_completion_ref"]);bundle=dict(training_completion_ref=frozen["training_completion_ref"])
class Legacy:
 def training_ready_v1(self,r):need(r==p["ARTIFACT_ACCEPTANCE"],"Only original saved600 acceptance");return copy.deepcopy(bundle),copy.deepcopy(training)
 def plan_ready_v1(self,e,parent):need(e["parent_id"]==parent,"Actual same-parent original plan validator dispatch");return {}
class Guard:
 MODE="Census_observation_two_scope_retention_v1";MAX_JSON=65536;META_CHARGE=524288
 def __init__(self):
  self._SELECTOR_REF=None;self.reserved=None;self.validated=0
  self.budget=types.SimpleNamespace(v4=types.SimpleNamespace(__name__="RAM_v4"))
 def load_selected_guard_v1(self,r):self._SELECTOR_REF=r;return self
 def _json(self,r):return dict(implementation_ref=common.V4,launcher_ref=dict(path="scripts/launch_reserved_gpu_v6.py",sha256="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"))
 def validate_storage_v9(self,s,cfg,storage,**kw):
  need(storage["reservation"]==self.reserved and cfg["output_cap_bytes"]==32*2**20,"Actual stored same whole32MiB reservation");self.validated+=1;return {}
 def budget_check_v9(self,*a,**k):return {}
guard=Guard();lock={"depth":0}
@contextlib.contextmanager
def mutex(root):
 lock["depth"]+=1
 try:yield
 finally:lock["depth"]-=1
def canonical(*a):pass
def file_bytes(*a):pass
for fn in(mutex,canonical,file_bytes):fn.__module__="RAM_v4"
guard.budget.v4.mutex=mutex;guard.budget.v4.canonical=canonical;guard.budget.v4.file_bytes=file_bytes
c.load=lambda r,name:guard if r==p["CENSUS_GUARD"]else Legacy()if r==p["SAVED_CONTRACT"]else (_ for _ in()).throw(AssertionError("Forbidden source import "+str(r)))
now=datetime.now(timezone.utc);created=(now-timedelta(seconds=30)).isoformat();expiry=(now+timedelta(hours=1)).isoformat()
review=put("RAM/current_whole_review.json",dict(independent=True,verdict="SOURCE_PASS",candidate_manifest_ref=ref(c.MANIFEST),blockers=[]))
# Every original producer is terminal/drained and permanently revoked in RAM; nonzero retained as history.
parents={};allseals={}
for i,parent in enumerate(c.VAL):
 inp=(pins["old_CPU_inputs"]|pins["completed_inputs"])[parent];spec=read(inp);phase="predict_val_"+parent[-3:];driver=spec["root_driver_ref"]
 launch=put("RAM/"+parent+"/old_launch.json",dict(input_ref=inp,driver_ref=driver,phase=phase,actual_launch=True,pid=1000+i))
 oldrc=-15 if i<3 else 0
 wait=put("RAM/"+parent+"/old_wait.json",dict(input_ref=inp,driver_ref=driver,phase=phase,pid=1000+i,launch_ref=launch,returncode=oldrc,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,finished_utc=created,operational_failure=None if oldrc==0 else"Original cancelled, preserved"))
 outer=put("RAM/"+parent+"/old_outer.json",dict(actual_exit_code=1 if oldrc else 0,actual_terminal_consumed=True,actual_wait_ref=wait,actual_direct_worker_returncode=oldrc,actual_tool_session_id=700+i))
 rev=put("RAM/"+parent+"/revocation.json",dict(root_authorizer="root_agent",input_ref=inp,old_wait_ref=wait,old_outer_ref=outer,future_writes_and_relaunch_permanently_revoked=True,preserve_all_files_and_original_charge=True,revoked_at_utc=created))
 epochs=list(c.EPOCHS[:4])if i<3 else list(c.EPOCHS);refs=[]
 for epoch in epochs:
  cp=next(x["checkpoint_ref"]for x in training["checkpoint_candidates"]if x["epoch"]==epoch)
  pref=dict(path=spec["output_directory"]+"/"+parent+f"_epoch{epoch:03d}_predictions.npz",sha256="1"*64)
  seal=put(spec["output_directory"]+"/"+parent+f"_epoch{epoch:03d}_prediction_seal.json",dict(epoch=epoch,parent_id=parent,status="sealed_C3_original_heldout_predictions",split="val",records=10000,checkpoint_ref=cp,training_completion_ref=c.tref_v1(),private_truth_opened=False,probabilities_modified=False,prediction_ref=pref))
  refs.append(seal)
 allseals[parent]=refs
 parents[parent]=dict(original_disposition=dict(input_ref=inp,launch_ref=launch,wait_ref=wait,outer_ref=outer,permanent_revocation_ref=rev),inventory_sealed_at_utc=created,retained_prediction_seal_refs=refs,pending_epochs=[25,30]if i<3 else[])
election=put("RAM/post_stop_election.json",dict(schema="C3_ROOT_poststop_retained_cells_acceptance_v1",root_authorizer="root_agent",status="active",created_at_utc=created,expires_at_utc=expiry,saved600_acceptance_ref=p["ARTIFACT_ACCEPTANCE"],inventory_sealed_after_all_original_terminals_and_drains=True,private_truth_opened=False,original_nonzero_waits_preserved=True,parents=parents))
e,ret,pending=c.election_values_v1(election)
need(sum(map(len,ret.values()))==24 and sum(map(len,pending.values()))==6,"Exact original24retained plus6missing")
for parent in c.VAL[:3]:need(read(parents[parent]["original_disposition"]["wait_ref"])["returncode"]==-15,"Original nonzero CPUwait preserved "+parent)
peerlaunch=read(p["H058_LAUNCH"]);peerid=next(r for r in peerlaunch["ownership"]["owned_identities"]if r["pid"]==peerlaunch["pid"])
peer=put("RAM/actual_peer_capsule.json",dict(schema="C3_ROOT_exact_live_H058_peer_v1",root_authorizer="root_agent",status="active",created_at_utc=created,expires_at_utc=expiry,H058_input_ref=p["H058_INPUT"],H058_launch_ref=p["H058_LAUNCH"],H058_request_ref=p["H058_REQUEST"],controller_terminal_qualification_ref=p["H058_TERMINAL"],controller_capability_ref=p["CONTROLLER_CAP"],H058_full_cap_plus_margin_bytes=30*2**30,C3_reserved_bytes=26*2**30,headroom_bytes=4*2**30,H058_source_or_registry_mutation_authorized=False,terminal_peer_absence_preserves_full30GiB=True,worker_identity=peerid))
c.peer_values_v1(peer)
# Actual current47-role read-only snapshot from historical46 + exactly published child capsule.
snapshot=json.loads((ROOT/"research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_acceptance.root_v1.json").read_text())
def rolemap(v):
 if isinstance(v,dict):
  if isinstance(v.get("bindings"),dict)and len(v["bindings"])==46:return v["bindings"]
  for x in v.values():
   out=rolemap(x)
   if out:return out
 if isinstance(v,list):
  for x in v:
   out=rolemap(x)
   if out:return out
roles=copy.deepcopy(rolemap(snapshot));roles["Census_observation_explicit_selector_capsule"]=p["CENSUS_SELECTOR"]
need(len(roles)==47,"All actual published child47roles")
guard._pins=lambda view:copy.deepcopy(view)
def state(*a,**kw):
 need(lock["depth"]==1 and kw.get("fresh")is True,"Actual ROOT state capture API under original mutex in RAM")
 return ({},set(),dict(prefix="RAM_original_prefix"),copy.deepcopy(roles),[],[],{})
guard._state_locked=state
gtree=ast.parse((ROOT/p["CENSUS_GUARD"]["path"]).read_text());ns=dict(copy=copy,require=c.require,_raw=lambda r:None,MODE=guard.MODE,RETIRE_OPERATION="retire_exact_Census_observation_retained_allowance_v1")
for name in("consumer_bindings_v1","operation_draft_v1"):
 fn=next(n for n in gtree.body if isinstance(n,ast.FunctionDef)and n.name==name);exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact_selected_child_"+name+"_RAM>","exec"),ns);setattr(guard,name,ns[name])
def reserve(r,**kw):
 need(lock["depth"]==0,"Actual original reserve API called outside capture lock")
 a=read(r);own=dict(copy.deepcopy(a["request"]),root_admission=r);guard.reserved=own
 need(all(x in own["consumer_bindings"]for x in roles.values()),"All47returned parent roles actually retained")
 return dict(value=own,initial_ledger_pins=dict(base_pins={},completed_overlay_prefix={}))
guard.apply_reserve_v1=reserve
prepared={};requests={};unions={}
for phase in c.PHASES:
 i=int(phase[-3:])if phase!="matrix"else 0;parent=c.VAL[i]if phase!="matrix"else c.PARENT
 job="C3remaining-RAM-"+phase;packet="research/reviews/C3_remaining_cells_GPU0_ROOT_RAM_"+phase
 authority=put("RAM/"+phase+"/direct_authority.json",dict(schema="C3_remaining_ROOT_direct_attachment_v1",root_authorizer="root_agent",status="active",created_at_utc=created,expires_at_utc=expiry,job_id=job,peer_launch_capsule_ref=peer,borrowed_physical_inode_attachment_authorized=True,old_controller_and_H058_preserved=True,old_controller_submissions_remain_revoked=True,evaluator_cuda_bytes=8*2**30,evaluator_margin_bytes=2*2**30,C3_reserved_bytes=26*2**30,H058_reserved_bytes=30*2**30,headroom_bytes=4*2**30))
 q=dict(schema="C3_remaining_cells_ROOT_request_v1",root_authorizer="root_agent",status="active",execution_authorized=True,created_at_utc=created,expires_at_utc=expiry,source_manifest_ref=ref(c.MANIFEST),source_review_ref=review,phase=phase,parent_id=parent,pending_epochs=[25,30]if phase!="matrix"else[],plan_evidence=read(read(pins["old_CPU_inputs"][parent])["root_stage_request_ref"])["plan_evidence"]if phase!="matrix"else{},job_id=job,output_directory="experiments/results/C3_remaining_cells_GPU0_RAM_"+phase,worker_auxiliary_scope=packet+"/worker_aux",storage_policy_selector_ref=p["CENSUS_SELECTOR"],Census_verification_ref=p["CENSUS_VERIFICATION_OUTER"],capacity_guard_ref=p["CENSUS_GUARD"],capacity_policy_ref=read(p["CENSUS_SELECTOR"])["policy_ref"],caps=c.caps_v1(phase),cell_election_ref=election,peer_launch_capsule_ref=peer,direct_attachment_authority_ref=authority,original_CPU_jobs_or_live_H058_mutation_authorized=False,automatic_restart_or_fallback=False)
 if phase=="matrix":requests[phase]=q;continue
 qref=put("RAM/"+phase+"/request.json",q);c.ROOT_INPUT_REF=None;c.STATE=None
 result=c.prepare_v1(qref);prepared[phase]=result;requests[phase]=q;unions[phase]=len(guard.reserved["consumer_bindings"])
 need(result["status"]=="ADMITTED_NOT_EXECUTED"and result["execute_argv"][5]==c.HERE.replace("/",".")+".execution_v1","Full real prepare/admit/input/contract argv join "+phase)
 need(c.validate_input_v1(result["input_ref"])[1]["phase"]==phase,"Full actual source/config/runtime/storage binding "+phase)
need(set(unions.values())=={64},"All3 prediction unions fit64 without role loss")
reject(lambda:guard.consumer_bindings_v1([dict(path="RAM/extra_"+str(i),sha256="0"*64)for i in range(65)]),"Original65ref union refusal retained")
worker=ram_module(D/"worker_v1.py","RAM_C3_remaining_worker",dict(c=c))
for parent in c.VAL[:3]:
 phase="predict_val_"+parent[-3:];sr=prepared[phase]["input_ref"];s=read(sr);q=requests[phase]
 seals=[]
 for epoch in q["pending_epochs"]:
  r=put(s["output_directory"]+"/"+parent+f"_epoch{epoch:03d}_prediction_seal.json",dict(epoch=epoch,parent_id=parent,training_completion_ref=c.tref_v1(),private_truth_opened=False,prediction_ref=dict(path=s["output_directory"]+"/new.npz",sha256="2"*64)));seals.append(r)
 result=put(s["output_directory"]+"/result.json",dict(status="sealed_C3_original_remaining_cells",parent_id=parent,epochs=q["pending_epochs"],training_completion_ref=c.tref_v1(),cell_election_ref=election,private_truth_opened=False,prediction_seal_refs=seals))
 done=put(s["output_directory"]+"/completed.json",dict(status="completed_C3_remaining_cells",parent_id=c.PARENT,original_VAL_parent_id=parent,child_input_ref=sr,actual_execution_source_manifest_ref=ref(c.MANIFEST),cell_election_ref=election,epochs=q["pending_epochs"],training_completion_ref=c.tref_v1(),query_ancestry_truth_opened=False,result_ref=result))
 wait=put("RAM/"+parent+"/new_wait.json",dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,complete_owned_roster_known=True,completed_ref=done,input_ref=sr,adapter_manifest_ref=ref(c.MANIFEST),operational_failure=None,group_observations=[dict(absent=True)]))
 outer=put("RAM/"+parent+"/new_outer.json",dict(root_authorizer="root_agent",actual_exit_code=0,actual_terminal_consumed=True,actual_wait_ref=wait,actual_direct_worker_returncode=0,actual_tool_session_id=900))
 ev=dict(input_ref=sr,completion_ref=done,wait_ref=wait,outer_ref=outer)
 need(set(worker.new_stage_values_v1(ev,parent,election,pending[parent]))=={25,30},"Complete new wait0/drain/outer0 cells join "+parent)
 for field,value in(("returncode",-15),("actual_direct_worker_wait",False),("actual_OS_wait_status",256),("all_owned_processes_and_groups_drained",False),("complete_owned_roster_known",False),("operational_failure","error"),("adapter_manifest_ref",dict(path="other",sha256="0"*64))):
  w=read(wait);w[field]=value;wr=put("RAM/"+parent+"/bad_"+field+".json",w);o=read(outer);o["actual_wait_ref"]=wr;er=dict(ev,wait_ref=wr,outer_ref=put("RAM/"+parent+"/rebound_"+field+".json",o))
  reject(lambda:worker.new_stage_values_v1(er,parent,election,pending[parent]),"New stage rejects "+parent+" "+field)
 # Record for full matrix fixture below.
 requests["matrix"].setdefault("new_stage_evidence",{})[parent]=ev
# The real matrix admission is possible only after complete new-stage evidence exists.
c.ROOT_INPUT_REF=None;c.STATE=None
prepared["matrix"]=c.prepare_v1(put("RAM/matrix/complete_request.json",requests["matrix"]))
unions["matrix"]=len(guard.reserved["consumer_bindings"])
need(unions["matrix"]==62 and c.validate_input_v1(prepared["matrix"]["input_ref"])[1]["phase"]=="matrix","Full actual matrix builder/admission/input join after all new-stage terminals")
# Retained acceptance negatives and subset update: completed25 must be reused.
for field,value in(("inventory_sealed_after_all_original_terminals_and_drains",False),("original_nonzero_waits_preserved",False),("private_truth_opened",True)):
 bad=read(election);bad[field]=value
 reject(lambda:c.election_values_v1(put("RAM/bad_election_"+field+".json",bad)),"Post-stop election rejects "+field)
bad=read(election);bad["parents"][c.VAL[0]]["pending_epochs"]=[20,25,30]
reject(lambda:c.election_values_v1(put("RAM/rerun20_election.json",bad)),"Answered20 cannot become pending")
parent=c.VAL[0];bad=read(election);spec=read(pins["old_CPU_inputs"][parent]);cp=next(x["checkpoint_ref"]for x in training["checkpoint_candidates"]if x["epoch"]==25)
r=put(spec["output_directory"]+"/completed25_seal.json",dict(epoch=25,parent_id=parent,status="sealed_C3_original_heldout_predictions",split="val",records=10000,checkpoint_ref=cp,training_completion_ref=c.tref_v1(),private_truth_opened=False,probabilities_modified=False,prediction_ref=dict(path=spec["output_directory"]+"/completed25.npz",sha256="3"*64)))
bad["parents"][parent]["retained_prediction_seal_refs"].append(r);bad["parents"][parent]["pending_epochs"]=[30]
need(c.election_values_v1(put("RAM/completed25_election.json",bad))[2][parent]==[30],"An epoch25 finished before post-stop inventory is reused; only30 pending")

# Concrete metadata election producer derives pending cells rather than trusting a supplied list.
inventory=dict(schema="C3_ROOT_poststop_original_prediction_inventory_v1",root_authorizer="root_agent",private_truth_opened=False,parents=copy.deepcopy(parents))
for row in inventory["parents"].values():row.pop("pending_epochs")
ir=put("RAM/root_poststop_inventory.json",inventory)
draftq=dict(schema="C3_ROOT_draft_poststop_cell_election_request_v1",root_authorizer="root_agent",status="active",created_at_utc=created,expires_at_utc=expiry,source_manifest_ref=ref(c.MANIFEST),source_review_ref=review,inventory_ref=ir)
draft=c.draft_election_v1(put("RAM/root_draft_request.json",draftq))
need(draft["parents"][c.VAL[0]]["pending_epochs"]==[25,30]and draft["original_invocation_success_claimed"]is False,"Actual concrete election producer derives exact pending and preserves historical failure")
badinv=copy.deepcopy(inventory);badrev=read(badinv["parents"][c.VAL[0]]["original_disposition"]["permanent_revocation_ref"]);badrev["future_writes_and_relaunch_permanently_revoked"]=False
badinv["parents"][c.VAL[0]]["original_disposition"]["permanent_revocation_ref"]=put("RAM/not_revoked.json",badrev)
reject(lambda:c.draft_election_v1(put("RAM/bad_draft_request.json",dict(draftq,inventory_ref=put("RAM/live_inventory.json",badinv)))),"Live original writer cannot enter retained-cell election")

# Complete actual gated driver -> worker entry, stopping at the first forbidden numerical import.
import builtins
class ReachedScience(Exception):pass
class ReachedBarrier(Exception):pass
seen=[];barrier_refs=[]
class Native:
 def apply_and_validate_title(self):seen.append("title")
 def validate_lease(self):seen.append("lease");return dict(physical_gpu=0,visible_gpu_uuid=c.GPU0_UUID)
native=Native()
fake_direct=types.SimpleNamespace(install_before_torch_v1=lambda *args:native)
real_import=builtins.__import__
def fixture_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name in("numpy","torch","torch_geometric"):raise ReachedScience(name)
 if name=="arggnn.american_c3_gnn_predict_isolated_v2":
  def barrier(refs,**kw):barrier_refs.extend(refs);need(kw["training_completion_ref"]==c.tref_v1(),"Actual original30cell barrier gets original saved600 lineage");raise ReachedBarrier()
  return types.SimpleNamespace(seal_validation_matrix_v1=barrier)
 if name.startswith("research."):
  if "worker_v1"in fromlist:return types.SimpleNamespace(worker_v1=actual_worker)
  if "direct_gpu0_v1"in fromlist:return types.SimpleNamespace(direct_gpu0_v1=fake_direct)
  if name.endswith(".worker_v1"):return actual_worker
  raise AssertionError("Unexpected actual project import "+name)
 return real_import(name,globals,locals,fromlist,level)
fb=dict(vars(builtins),__import__=fixture_import)
actual_worker=ram_module(D/"worker_v1.py","RAM_actual_remaining_worker",dict(c=c,__builtins__=fb))
actual_worker.sys=types.SimpleNamespace(modules={},path=[])
actual_worker.resource=types.SimpleNamespace(RLIMIT_FSIZE=1,RUSAGE_SELF=0,setrlimit=lambda *a:None,getrusage=lambda *a:types.SimpleNamespace(ru_maxrss=0))
class FOS:
 environ={}
 def read(self,fd,n):seen.append("gate");return b"G"
 def close(self,fd):pass
 def getpid(self):return 9991
 def getpgrp(self):return 9991
 def getsid(self,pid):return 9991
fos=FOS();actual_worker.os=fos
class WPath(RP):
 @classmethod
 def cwd(cls):return ROOT
driver=ram_module(D/"execution_v1.py","RAM_actual_remaining_driver",dict(c=c,__builtins__=fb,**{k:v for k,v in c.__dict__.items()if not k.startswith("__")}))
driver.os=fos;driver.Path=WPath;driver.tempfile=types.SimpleNamespace(tempdir=None)
driver.sys=types.SimpleNamespace(dont_write_bytecode=True,platform="linux")
c.original._sizes=lambda *a,**kw:[]
prior_load=c.load
class Memo:
 def install_consumer_v1(self,*a,**kw):need(kw["consumer_role"]=="read_only_consumer","Only actual read-only consumer namespace memo role");return dict(installed=True)
c.load=lambda r,name:Memo()if r==common.NAMESPACE_MEMO else prior_load(r,name)
for phase in c.PHASES:
 sr=prepared[phase]["input_ref"];spec=read(sr);q=read(spec["root_stage_request_ref"]);c.ROOT_INPUT_REF=None;c.STATE=None;c.set_amendment_v1(sr)
 paths=c.worker_env_paths_v1(spec);dirs.update(paths.values());fos.environ=dict(paths,CUDA_VISIBLE_DEVICES=""if phase=="matrix"else c.GPU0_UUID)
 if phase=="matrix":ar=None
 else:ar=put(spec["worker_auxiliary_scope"]+"/attachment.json",dict(job_id=spec["job_id"],direct_attachment_authority_ref=q["direct_attachment_authority_ref"]))
 try:driver.worker_v1(sr,77,ar)
 except ReachedScience as ex:need(phase in c.GPU_PHASES and str(ex)=="numpy","Actual full gate/contract/lease/title/writer reaches first forbidden science import "+phase)
 except ReachedBarrier:need(phase=="matrix"and len(barrier_refs)==30,"Actual full matrix worker assembles all30 before unchanged barrier/payload boundary")
 else:raise AssertionError("Actual worker did not reach expected source-only boundary")
need({json.dumps(r,sort_keys=True)for r in barrier_refs}=={json.dumps(r,sort_keys=True)for rows in allseals.values()for r in rows}|{json.dumps(r,sort_keys=True)for parent in c.VAL[:3]for r in read(read(requests["matrix"]["new_stage_evidence"][parent]["completion_ref"])["result_ref"])["prediction_seal_refs"]},"Exact retained24 plus chosen new6 go to unchanged30cell barrier")
# Exact original H058 capacity API, with only actual operations replaced by RAM providers.
hsbindings={k:v for k,v in c.__dict__.items()if not k.startswith("__")}
hsbindings["validate_native_resource_disposition_v1"]=lambda *a,**kw:dict(native_process_identities=[])
hs=ram_module(ROOT/p["H058_DIRECT"]["path"],"RAM_original_H058_capacity",hsbindings)
hs.os=types.SimpleNamespace(getuid=lambda:1000)
base=types.SimpleNamespace(proc_identity=lambda pid:dict(pid=pid,uid=1000,start_ticks=peerid["start"],state="R"))
hs.controller_v1=lambda *a:({},dict(updated_unix=now.timestamp()),dict(process_identities=[]),{})
direct=ram_module(D/"direct_gpu0_v1.py","RAM_new_C3_direct",dict(c=c,**{k:v for k,v in c.__dict__.items()if not k.startswith("__")}))
direct.os=types.SimpleNamespace(getuid=lambda:1000)
direct._PEER=types.SimpleNamespace(controller_v1=hs.controller_v1,capacity_v1=hs.capacity_v1)
q=requests[c.GPU_PHASES[0]]
def inventory(own=0,external=0):
 compute=[dict(pid=peerid["pid"],name="AI Auto Research Pipeline Test",used_mib=28*1024)]
 if own:compute.append(dict(pid=9992,name="AI Auto Research Pipeline Test",used_mib=own))
 if external:compute.append(dict(pid=5555,name="external",used_mib=external))
 return [dict(index=0,uuid=c.GPU0_UUID,total_mib=80*1024,used_mib=sum(x["used_mib"]for x in compute),compute=compute)]
for own in(0,8*1024,10*1024):
 base.real_inventory=lambda own=own:inventory(own)
 sample=direct.capacity_v1(q,base,worker_pid=9992)
 need(sample["projected_mib"]==70*1024 and sample["C3_slot_reused"]is False,"Actual capacity source preserves26+30+10+4 at own"+str(own))
base.real_inventory=lambda:inventory(0,12*1024)
reject(lambda:direct.capacity_v1(q,base,worker_pid=9992),"Additional12GiB external occupancy cannot fit invented80GiB fixture")
base.real_inventory=lambda:inventory(10*1024+1)
reject(lambda:direct.capacity_v1(q,base,worker_pid=9992),"Evaluator beyond8+2GiB fails")
base.proc_identity=lambda pid:dict(pid=pid,uid=1000,start_ticks=peerid["start"]+1,state="R")
reject(lambda:direct.capacity_v1(q,base,worker_pid=9992),"Positive peer PID reuse fails before capacity attribution")
# Complete inherited lifecycle body remains exact except namespace, dispatch and known v7 RSS fix.
def source_functions(path):
 return {n.name:n for n in ast.parse(RealPath(path).read_text()).body if isinstance(n,ast.FunctionDef)}
old=source_functions(ROOT/"research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2/execution_v1.py")
new=source_functions(D/"execution_v1.py")
for name in("execute_v1","main"):
 norm=ast.parse(ast.unparse(new[name]).replace(c.HERE.replace("/","."),"research.reviews.native_saved_MLM_current_Census64_integration_source_20261017_v2").replace("worker_v1","worker_context_v1"))
 # The text replacement also touches function names; compare the intended dependency namespace only.
 expected=ast.unparse(old[name]).replace("worker_context_v1","worker_v1").replace("native_saved_MLM_current_Census64_integration_source_20261017_v2","C3_remaining_cells_GPU0_source_20261017_v1")
 expectedtree=ast.parse(expected)
 for node in ast.walk(expectedtree):
  if isinstance(node,ast.Constant)and node.value==".runtime/conda/envs/argformer_native_v1":node.value=".runtime/env"
  if isinstance(node,ast.keyword):
   if node.arg=="CONDA_PREFIX":node.arg="VIRTUAL_ENV"
   if node.arg in("BLIS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):node.value=ast.Constant(value="32")
 expected=ast.unparse(expectedtree)
 need(ast.dump(new[name],include_attributes=False)==ast.dump(ast.parse(expected).body[0],include_attributes=False),"Full original finite lifecycle/CLI AST retained "+name)


# An unchanged source input, not another same-parent plan, is mandatory.
badq=copy.deepcopy(requests[c.GPU_PHASES[0]]);badq["plan_evidence"]=dict(badq["plan_evidence"],extra="substitute")
reject(lambda:c.validate_request_v1(put("RAM/substituted_plan_request.json",badq)),"Same-parent plan evidence cannot silently substitute original admitted input")
badq=copy.deepcopy(requests[c.GPU_PHASES[0]]);badq["Census_verification_ref"]=put("RAM/not_verified.json",dict(actual_outer_exit_code=0))
reject(lambda:c.validate_request_v1(put("RAM/unverified_selector_request.json",badq)),"Explicit current selector cannot be used without exact consumed verification0 receipt")

for module in(c,):
 with contextlib.redirect_stdout(io.StringIO())as out:rc=module.main([])
 need(rc==0 and json.loads(out.getvalue())["status"]=="SOURCE_ONLY_INACTIVE","Actual inactive builder entry")
for phase in c.PHASES:
 template=copy.deepcopy(requests[phase]);template.update(status="inactive",execution_authorized=False,template_only=True,created_at_utc=None,expires_at_utc=None,source_review_ref=None,cell_election_ref=None)
 if phase in c.GPU_PHASES:template.update(peer_launch_capsule_ref=None,direct_attachment_authority_ref=None,pending_epochs=None)
 else:template["new_stage_evidence"]=None
 (D/("ROOT_"+phase+".INACTIVE.template.v1.json")).write_text(json.dumps(template,sort_keys=True,indent=2)+"\n")
for pth in D.glob("*.py"):compile(pth.read_bytes(),str(pth),"exec");need(True,"Compile "+pth.name)
result=dict(schema="C3_remaining_cells_author_source_checks_v1",verdict="AUTHOR_SOURCE_CHECKS_ONLY",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,direct_unions=unions,raw_parent_roles=47,unique_parent_refs=40,complete_gated_worker_science_boundary_cases=3,complete_retained24_plus_new6_matrix_boundary=True,conservative_capacity_source_fixture_GiB=70,source_authentication_refs=len(manifest["source_authentication_refs"]),actual_guard_state_resource_calls=False,numerical_imports_or_calls=False,prediction_checkpoint_or_truth_payloads_read=False)
(D/"source_checks.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))
