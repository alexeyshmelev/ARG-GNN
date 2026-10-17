"""Source-only complete entry/contract joins. Every operational object is a RAM double."""
from pathlib import Path as RealPath
from datetime import datetime,timezone,timedelta
import ast,copy,contextlib,hashlib,io,json,posixpath,types
ROOT=RealPath("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2"
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
c=ram_module(D/"contract_v1.py","RAM_contract",dict(original=common))
manifest=json.loads((D/"source_manifest.v1.json").read_text());put(c.MANIFEST,manifest)
config=json.loads((D/"scientific_config.FROZEN.v1.json").read_text());configref=put(c.HERE+"/scientific_config.FROZEN.v1.json",config)
need(config==c.EXPECTED_CONFIG,"Real frozen config equals complete real scientific contract")
need(config["output_cap_bytes"]==64*2**30 and config["group_scope_caps"]==dict(contrastive=8*2**30,embedding=56*2**30,RAM_retrieve_and_seal=2**30),"64GiB aggregate stricter intersection with original65GiB groups")
need(config["scientific_protocol"]==json.loads((OLD/"scientific_config.FROZEN.v1.json").read_text())["scientific_protocol"],"All original author scientific settings unchanged")
selector=put("RAM/actual_future_selector.json",dict(active=True,root_authorizer="root_agent",source_ref=c.GUARD,selector_source_ref=c.GUARD,source_manifest_ref=c.CENSUS_MANIFEST_REF,measurement_source_ref=c.CENSUS_MEASUREMENT_REF,accounting_mode=c.CENSUS_MODE,journal="RAM/journal"))
policy=put("RAM/actual_future_policy.json",dict(active=True,source_ref=c.GUARD,journal="RAM/journal",source_manifest_ref=c.CENSUS_MANIFEST_REF,measurement_source_ref=c.CENSUS_MEASUREMENT_REF))
docs[selector["path"]]["policy_ref"]=policy;docs[selector["path"]]["source_review_ref"]=put("RAM/Census_review.json",dict(verdict="SOURCE_PASS"))
selector=put(selector["path"],docs[selector["path"]])
review=put("RAM/whole_review.json",dict(independent=True,verdict="SOURCE_PASS",candidate_manifest_ref=ref(c.MANIFEST),blockers=[]))
acceptance=put("RAM/artifact_acceptance.json",dict(saved=True,historical_MLM_outer_exit_code=1))
now=datetime.now(timezone.utc);created=(now-timedelta(minutes=1)).isoformat();expiry=(now+timedelta(hours=12)).isoformat()
implementation=put("RAM/implementation.json",{});launcher=put("RAM/launcher.json",{})
docs[policy["path"]].update(implementation_ref=implementation,launcher_ref=launcher);policy=put(policy["path"],docs[policy["path"]])
docs[selector["path"]]["policy_ref"]=policy;selector=put(selector["path"],docs[selector["path"]])
# The real selector configurator calls only RAM raw/load/selected-guard objects.
class Guard:
 MODE=c.CENSUS_MODE
 MAX_JSON=65536;META_CHARGE=524288
 def __init__(self):
  self._SELECTOR_REF=None;self.captured=None;self.reserved=None;self.scope_checks=0
  self.budget=types.SimpleNamespace(v4=types.SimpleNamespace(__name__="RAM_v4"))
 def _json(self,r):return read(r)
 def load_selected_guard_v1(self,r):self._SELECTOR_REF=r;return self
 def validate_storage_v9(self,s,cfg,storage,**kw):
  need(storage["reservation"]==self.reserved and cfg["output_cap_bytes"]==64*2**30,"Actual stored64GiB reservation verified by RAM guard")
  self.scope_checks+=1;return {}
 def budget_check_v9(self,*a,**k):self.scope_checks+=1;return {}
guard=Guard()
@contextlib.contextmanager
def mutex(root):yield
def canonical(*a):pass
def file_bytes(*a):pass
for fn in(mutex,canonical,file_bytes):fn.__module__="RAM_v4"
guard.budget.v4.mutex=mutex;guard.budget.v4.canonical=canonical;guard.budget.v4.file_bytes=file_bytes
realraw=c.raw
def configurable_raw(r,prior=c.raw):
 if r in(c.GUARD,c.CENSUS_MANIFEST_REF,c.CENSUS_MEASUREMENT_REF):return b"RAM source pin"
 return prior(r)
c.raw=configurable_raw
c.load=lambda r,name:guard
need(c.configure_selector_v1(selector)is guard and c.POLICY==policy,"Real explicit selector/source/policy configurator joins")
bad=copy.deepcopy(selector);bad["sha256"]="0"*64
reject(lambda:c.configure_selector_v1(bad),"Selector bytes mismatch")
reject(lambda:c.configure_selector_v1(put("RAM/other_selector.json",dict(docs[selector["path"]],source_ref=dict(path="old",sha256="old")))),"Old guard source substitution")
need(c.selected_guard_v1(consumer=False)is guard,"ROOT original unwrapped guard role")
# Original scientific integration executes only its inert complete source module.
science=ram_module(D/"science_binding_v1.py","RAM_science_binding",dict(c=c))
loaded={}
def scientific_load(r,name):
 need(r==c.SAVED_INTEGRATION_REF,"Only exact passed science source loaded")
 m=ram_module(ROOT/r["path"],name,{})
 m.own_ref_v1=lambda:copy.deepcopy(c.SAVED_INTEGRATION_REF)
 loaded[name]=m;return m
c.load=scientific_load
# Review metadata is RAM; source/entry bytes are actual frozen source.
science_review=c.SAVED_SCIENCE_REVIEW_REF
realread=c.read
c.read=lambda r,prior=c.read:dict(verdict="SOURCE_PASS",blocking_findings=[])if r==science_review else prior(r)
realraw=c.raw
c.raw=lambda r,prior=c.raw:b"RAM review"if r==science_review else prior(r)
integration=science.integration_v1(selector);entry=science.entry_v1(selector)
binding=dict(storage_policy_selector_ref=selector,saved_MLM_integration_source_ref=c.SAVED_INTEGRATION_REF)
integration.bind_stage_v1(binding,binding)
need(entry.integration is integration,"Real entry and worker share same private saved scientific integration")
need(integration.PINS["current_selector"]!=selector,"Original old selector pin stays intact and is never passed as actual")
reject(lambda:integration.bind_stage_v1(dict(binding,storage_policy_selector_ref=integration.PINS["current_selector"]),binding),"Old selector cannot satisfy actual private bind")
need(integration.SOURCE.endswith("native_VAL000_contrastive_saved_MLM_integration_source_20261017_v2/saved_mlm_integration_v1.py"),"Original science provenance unchanged")
# Full worker and builder modules, replacing only project imports with the actual RAM-linked modules.
worker=ram_module(D/"worker_context_v1.py","RAM_worker",dict(c=c,saved_science=science))
builder=ram_module(D/"root_builder_v1.py","RAM_builder",dict(c=c,science=science,check_external_prerequisites_v1=worker.check_external_prerequisites_v1,prerequisite_metadata_v1=worker.prerequisite_metadata_v1,PendingPrerequisites=worker.PendingPrerequisites))
# Explicit scientific bridge functions return only RAM metadata for saved eligibility and actual qualification.
class Eligibility:
 def record(self):return dict(eligible_saved_artifact=True,historical_outer_exit=1)
checkpoint=dict(path="RAM/never_read_checkpoint.pt",sha256="1"*64)
integration.saved_acceptance_v1=lambda ctx,req:(None,Eligibility(),checkpoint)
qualification_ref=put("RAM/qualification.json",{})
triples={};qualified=dict(phases=dict(contrastive={}))
integration.fit_completion_dispatch_v1=lambda r,s:copy.deepcopy(qualified)
prep=dict(native_dataset_refs={},full_path_id_order_refs={})
for view,split in(("contrastive","train"),("contrastive","val"),("query_paths","train"),("query_paths","val")):
 key=view+"/"+split;prep["native_dataset_refs"][key]=dict(path="RAM/"+key+".pkl",sha256="2"*64);prep["full_path_id_order_refs"][key]=dict(path="RAM/"+key+".order.json",sha256="3"*64)
c.read=lambda r,prior=c.read:copy.deepcopy(prep)if r==c.PREP_REF else prior(r)
def triple(phase,stage,**extra):
 inp=put("RAM/"+phase+"/input.json",{})
 done=put("RAM/"+phase+"/completed.json",dict(parent_id=c.PARENT,query_ancestry_truth_opened=False,stage=stage,child_input_ref=inp,checkpoint_ref=checkpoint,**extra))
 wait=put("RAM/"+phase+"/wait.json",dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,complete_owned_roster_known=True,completed_ref=done,operational_failure=None,group_observations=[dict(absent=True)],input_ref=inp,adapter_manifest_ref=ref(c.MANIFEST),actual_direct_child_wait=True,child_completion_ref=done,process_group_ESRCH=True))
 outer=put("RAM/"+phase+"/outer.json",dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,all_owned_processes_and_groups_drained=True,actual_wait_ref=wait,actual_tool_session_id=42,schema="native_saved_MLM_downstream_actual_outer_terminal_v1",terminal_consumed=True,actual_tool_terminal_exit_code=0,tool_session_id=42,child_completion_ref=done))
 return dict(completion_ref=done,wait_ref=wait,outer_terminal_ref=outer)
initialization=put("RAM/inline_initialization.json",{})
contrastive=triple("contrastive","contrastive",saved_MLM_initialization_ref=initialization)
qualified["phases"]["contrastive"].update(checkpoint_ref=checkpoint,child_completion_ref=contrastive["completion_ref"],actual_wait_ref=contrastive["wait_ref"],consumed_outer_ref=contrastive["outer_terminal_ref"])
emb={}
for key in prep["native_dataset_refs"]:
 view,split=key.split("/");emb[key]=triple("embedding_"+view+"_"+split,"embedding",view=view,split=split,raw_dataset_ref=prep["native_dataset_refs"][key],row_order_ref=prep["full_path_id_order_refs"][key])
ram=triple("RAM","RAM_retrieve_and_seal",status="completed_native_RAM_retrieval_and_truth_free_seal",full_reference_rows=9721236,full_query_rows=10000,native_top1_arithmetic_or_ties_modified=False)
request=dict(schema="native_saved_MLM_Census64_ROOT_request_v1",root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,source_manifest_ref=ref(c.MANIFEST),source_review_ref=review,created_at_utc=created,expires_at_utc=expiry,storage_policy_selector_ref=selector,capacity_guard_ref=c.GUARD,capacity_policy_ref=policy,instance="ram_entry_join",ROOT_saved_MLM_artifact_acceptance_ref=acceptance)
master=builder.master_metadata_v1(request,config_ref=configref,storage_ref=None,request_ref=put("RAM/master_request.json",dict(request,action="prepare_master")))
guard.reserved=dict(key="H056/"+master["job_id"],output=str(local(master["output_directory"])),auxiliary_scopes=[str(local(master["worker_auxiliary_scope"]))],reserved_bytes=64*2**30,consumer_bindings=[ref(c.MANIFEST),ref(c.DRIVER),configref,master["runtime_ref"]])
storage=dict(job_id=master["job_id"],output_directory=master["output_directory"],reserved_new_bytes=64*2**30,reservation=guard.reserved,accounting_mode=c.CENSUS_MODE,storage_policy_selector_ref=selector,completed_overlay_policy_ref=policy,budget_components=dict(consumer_adapter=c.GUARD))
master["storage_ref"]=put("RAM/storage.json",storage);masterref=put("RAM/master.json",master)
c.frozen_science_metadata_v1=lambda:(prep,{}, {})
allowed.add(c.IO_REVIEW_REF["path"])
c.read=lambda r,prior=c.read:dict(verdict="PASS_SOURCE_OPTION_ONLY_EXECUTION_NOT_AUTHORIZED")if r==c.IO_REVIEW_REF else prior(r)
# Manifest closure raw authentication is checked structurally and uses RAM source doubles here.
c.raw=lambda r,prior=c.raw:b"RAM inherited source"if r in manifest["source_authentication_refs"]else prior(r)
for phase in c.PHASES:
 prereq={}if phase=="contrastive"else dict(contrastive=contrastive)
 if phase in("RAM_retrieve_and_seal","finalize"):prereq["embeddings"]=emb
 if phase=="finalize":prereq["RAM_retrieve_and_seal"]=ram
 q=dict(request,action="prepare_phase",master_input_ref=masterref,phase=phase,max_cuda_reserved_bytes=0 if phase=="finalize"else 32*2**30,prerequisites=prereq,saved_MLM_qualification_ref=None if phase=="contrastive"else qualification_ref)
 a=builder.phase_metadata_v1(q,master,masterref);aref=put("RAM/"+phase+"/authority.json",a)
 s=dict(schema="native_VAL000_downstream_ROOT_phase_input_v1",root_authorizer="root_agent",status="active",template_only=False,execution_authorized=True,created_at_utc=created,expires_at_utc=expiry,root_stage_request_ref=aref,job_id=a["job_id"],phase=phase,output_directory=master["output_directory"]+"/"+phase,worker_auxiliary_scope=master["worker_auxiliary_scope"]+"/"+phase,storage_policy_selector_ref=selector,refs=dict(config=configref,runtime=master["runtime_ref"],storage=master["storage_ref"]))
 sref=put("RAM/"+phase+"/phase_input.json",s);c.ROOT_INPUT_REF=None;c.set_amendment_v1(sref)
 need(c.validate_input_v1(sref)[0]==s,"Real builder to real complete contract "+phase)
 need(worker.check_external_prerequisites_v1(a)["historical_MLM_outer_exit_code"]==1,"Real complete worker prerequisite join "+phase)
 badq=copy.deepcopy(q);badq["expires_at_utc"]=(now+timedelta(days=2)).isoformat()
 reject(lambda:builder.phase_metadata_v1(badq,master,masterref),"No phase master clock extension "+phase)
 c.STATE=None
# Full actual stored master validation uses original RAM guard.
need(builder._master_v1(masterref,request)==master,"Builder joins genuine master source/review/config/storage and guard")
badmaster=copy.deepcopy(master);badmaster["source_review_ref"]=put("RAM/cap_only.json",dict(verdict="SOURCE_PASS"))
reject(lambda:builder._master_v1(put("RAM/bad_master.json",badmaster),request),"Cap-only or different master review rejected")
# Genuine qualification builder must use returned metadata dictionary, never unpack its keys.
integration.qualify_saved_MLM_for_embeddings_v1=lambda **kwargs:dict(schema="RAM_qualification",actual=kwargs,historical_MLM_outer_exit_code=1)
builder._save_v1=lambda directory,name,value:put("RAM/result/"+name,value)
qualrequest=dict(request,action="prepare_qualification",master_input_ref=masterref,initialization_ref=initialization,contrastive_prerequisite=contrastive)
qref=put("RAM/qualification_request.json",qualrequest)
need(builder.prepare_qualification_v1(qref)["historical_MLM_outer_exit_code"]==1,"Real qualification builder route dictionary join")
reject(lambda:worker.check_external_prerequisites_v1(dict(a,phase="embedding_query_paths_train",prerequisites={})),"Missing actual contrastive blocks")
badouter=copy.deepcopy(read(contrastive["outer_terminal_ref"]));badouter["tool_session_id"]=43
badtriple=dict(contrastive,outer_terminal_ref=put("RAM/mismatched_outer_session.json",badouter))
reject(lambda:worker.prerequisite_metadata_v1(badtriple,"contrastive"),"Two alias field sets cannot claim different actual tool sessions")
reject(lambda:builder._request_v1(put("RAM/inactive_request.json",dict(request,action="prepare_master",status="inactive",execution_authorized=False)),"prepare_master"),"Inactive template cannot prepare master")
capreview=put("RAM/cap_review.json",dict(independent=True,verdict="SOURCE_PASS",candidate_manifest_ref=dict(path="cap_only_manifest.json",sha256="0"*64),blockers=[]))
reject(lambda:builder._request_v1(put("RAM/cap_review_request.json",dict(request,action="prepare_master",source_review_ref=capreview)),"prepare_master"),"Cap-only independent review cannot authorize complete operational source")
reject(lambda:builder._request_v1(put("RAM/wrong_policy_request.json",dict(request,action="prepare_master",capacity_policy_ref=put("RAM/unrelated_policy.json",{}))),"prepare_master"),"Explicit ROOT policy cannot disagree with actual activated selector policy")
# Complete real ROOT builder preparation route with RAM journal/mutex/record filesystem only.
snapshot=json.loads((ROOT/"research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_acceptance.root_v1.json").read_text())
def find_roles(value):
 if isinstance(value,dict):
  if isinstance(value.get("bindings"),dict)and len(value["bindings"])==46:return value["bindings"]
  for item in value.values():
   found=find_roles(item)
   if found is not None:return found
 if isinstance(value,list):
  for item in value:
   found=find_roles(item)
   if found is not None:return found
roles=find_roles(snapshot)
need(roles is not None,"Immutable actual46-role ROOT metadata snapshot used without guard/state call")
roles=copy.deepcopy(roles);roles["Census_observation_explicit_selector_capsule"]=selector
need(len(roles)==47,"Planned child preserves all46 roles plus explicit new selector")
locking=dict(depth=0)
@contextlib.contextmanager
def capture_mutex(root):
 locking["depth"]+=1
 try:yield
 finally:locking["depth"]-=1
capture_mutex.__module__="RAM_v4";guard.budget.v4.mutex=capture_mutex
def state(*args,**kwargs):
 need(locking["depth"]==1 and kwargs.get("fresh")is True,"Builder captures original selected child under original mutex fresh")
 return ({},set(),dict(prefix="RAM_frozen_parent"),copy.deepcopy(roles),[],[],{})
guard._state_locked=state;guard._pins=lambda view:copy.deepcopy(view)
gsource=(ROOT/c.GUARD["path"]).read_text()
consumerfn=next(n for n in ast.parse(gsource).body if isinstance(n,ast.FunctionDef)and n.name=="consumer_bindings_v1")
consumerenv=dict(require=c.require,copy=copy,_raw=lambda r:None)
exec(compile(ast.Module(body=[consumerfn],type_ignores=[]),"<exact_consumer_union_RAM_source_auth>","exec"),consumerenv)
guard.consumer_bindings_v1=consumerenv["consumer_bindings_v1"]
reject(lambda:guard.consumer_bindings_v1([dict(path="RAM/source_"+str(i)+".py",sha256=str(i).zfill(64))for i in range(65)]),"Real selected-child union function refuses65 distinct refs")
guard.operation_draft_v1=lambda action,reservation,**kwargs:dict(operation=action,request=copy.deepcopy(reservation),expected_before=kwargs["expected_before"])
def reserve(authority_ref,**kwargs):
 need(locking["depth"]==0,"Original reserve invoked outside capture mutex so prefix races remain failstop")
 authority=read(authority_ref);need(authority["expected_before"]==dict(prefix="RAM_frozen_parent"),"Same exact captured prefix in reserve authority")
 own=copy.deepcopy(authority["request"]);own["root_admission"]=authority_ref
 need(all(r in own["consumer_bindings"]for r in roles.values())and c.GUARD in own["consumer_bindings"],"Full actual role union plus selected child source preserved")
 guard.reserved=own
 return dict(value=own,initial_ledger_pins=dict(base_pins={},completed_overlay_prefix={}),kind="RAM_reserve_receipt")
guard.apply_reserve_v1=reserve
# Real _save validates every existing own directory/file and conservative node charge.
savefn=next(n for n in ast.parse((D/"root_builder_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="_save_v1")
exec(compile(ast.Module(body=[savefn],type_ignores=[]),"<real_builder_save_RAM_fs>","exec"),builder.__dict__)
fullrequest=dict(request,instance="ram_full_route",action="prepare_master")
fullref=put("RAM/full_master_request.json",fullrequest)
admitted=builder.prepare_master_v1(fullref)
need(admitted["status"]=="MASTER_ADMITTED_ONLY"and admitted["scientific_execution_started"]is False,"Complete real prepare_master source route is admission metadata only in RAM")
fullmaster=read(admitted["master_input_ref"])
union=guard.reserved["consumer_bindings"];raw_roles=len(roles);unique_roles=len(guard.consumer_bindings_v1(list(roles.values())))
need(len(union)<=64 and all(r in union for r in roles.values()),"Complete builder direct binding union fits unchanged64 ceiling in explicit future-selector RAM fixture")
fullphase=dict(fullrequest,action="prepare_phase",master_input_ref=admitted["master_input_ref"],phase="contrastive",max_cuda_reserved_bytes=32*2**30,prerequisites={},saved_MLM_qualification_ref=None)
fullphaseref=put("RAM/full_phase_request.json",fullphase);c.ROOT_INPUT_REF=None
prepared=builder.prepare_phase_v1(fullphaseref)
need(prepared["status"]=="PHASE_PREPARED_ONLY"and prepared["execute_argv"][:1]==["setsid"]and "--execute"in prepared["execute_argv"],"Complete real prepare_phase returns concrete original setsid argv without executing")
need(prepared["execute_argv"][5]==c.HERE.replace("/",".")+".execution_v1","Exact builder to new driver namespace")
reject(lambda:builder.prepare_phase_v1(fullphaseref),"Single phase namespace cannot replay")
validinput=c.read(prepared["input_ref"]);validmaster=c.read(admitted["master_input_ref"]);original_config=docs[configref["path"]]
for field,value in(("output_cap_bytes",65*2**30),("scientific_protocol",dict(config["scientific_protocol"],contrastive=dict(config["scientific_protocol"]["contrastive"],batch=2048)))):
 altered=copy.deepcopy(original_config);altered[field]=value
 badcfgref=put("RAM/bad_config_"+field+".json",altered);badm=copy.deepcopy(validmaster);badm["config_ref"]=badcfgref
 bmref=put("RAM/bad_config_master_"+field+".json",badm)
 olda=read(validinput["root_stage_request_ref"]);bada=copy.deepcopy(olda);bada["master_input_ref"]=bmref;bada["config_ref"]=badcfgref
 bs=copy.deepcopy(validinput);bs["root_stage_request_ref"]=put("RAM/bad_authority_"+field+".json",bada);bs["refs"]["config"]=badcfgref
 bsref=put("RAM/bad_input_"+field+".json",bs);c.ROOT_INPUT_REF=None;c.set_amendment_v1(bsref)
 reject(lambda:c.validate_input_v1(bsref),"Real full contract rejects modified frozen "+field)
# Test70KiB own record failstop through the real bounded serializer.
reject(lambda:builder._save_v1(local(fullmaster["ROOT_record_directory"]),"oversized.json",dict(payload="x"*70000)),"Bounded own ROOT metadata rejects oversized file")
need(guard.scope_checks>=3,"Master and qualification authenticate actual stored reservation via same selected RAM guard")
binding_result=dict(raw_returned_roles=raw_roles,unique_returned_role_refs=unique_roles,complete_direct_union_refs=len(union),unchanged_ceiling=64,actual_future_selector_ref_unknown=True,fixture_only=True,all_returned_roles_preserved=True,source_manifest_ref=ref(c.MANIFEST))
(D/"direct_binding_union.SOURCE_ONLY.v1.json").write_text(json.dumps(binding_result,sort_keys=True,indent=2)+"\n")

# V2 exact original event bound and all seven complete builder entries, with no operational guard call.
guard_tree=ast.parse((ROOT/c.GUARD["path"]).read_text())
constants={}
for node in guard_tree.body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Name)and target.id in("MAX_JSON","META_CHARGE"):constants[target.id]=ast.literal_eval(node.value)
need(constants==dict(MAX_JSON=65536,META_CHARGE=524288),"Actual pinned Census final event JSON maximum and event metadata charge")
append=next(n for n in guard_tree.body if isinstance(n,ast.FunctionDef)and n.name=="_append")
rawnode=next(n for n in append.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="raw"for t in n.targets))
need(ast.dump(rawnode.value,include_attributes=False)==ast.dump(ast.parse("budget.v4.encoded(event)+b'\\n'",mode="eval").body,include_attributes=False),"Original event byte bound covers exact serializer plus final newline")
boundnode=next(n for n in append.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and any(isinstance(x,ast.Name)and x.id=="MAX_JSON"for x in ast.walk(n.value)))
need(append.body.index(boundnode)<next(i for i,n in enumerate(append.body)if any(isinstance(x,ast.Attribute)and x.attr=="open"for x in ast.walk(n))),"Original event-size rejection precedes every event filesystem write")
need(builder.MAX_ROOT_RECORD_BYTES==454656 and builder.RESERVE_EVENT_JSON_MAX_BYTES==65536
 and builder.RESERVE_EVENT_NODE_CHARGE_BYTES==4096
 and builder.MAX_ROOT_RECORD_BYTES+builder.RESERVE_EVENT_JSON_MAX_BYTES+builder.RESERVE_EVENT_NODE_CHARGE_BYTES==524288,
 "Complete conservative ROOT plus maximum event and event-node bound fits original allowance exactly")
for phase in c.PHASES[1:]:
 prereq=dict(contrastive=contrastive)
 if phase in("RAM_retrieve_and_seal","finalize"):prereq["embeddings"]=emb
 if phase=="finalize":prereq["RAM_retrieve_and_seal"]=ram
 pq=dict(fullphase,phase=phase,max_cuda_reserved_bytes=0 if phase=="finalize"else 32*2**30,
  prerequisites=prereq,saved_MLM_qualification_ref=qualification_ref)
 c.ROOT_INPUT_REF=None;answer=builder.prepare_phase_v1(put("RAM/v2_full_"+phase+"_request.json",pq))
 need(answer["status"]=="PHASE_PREPARED_ONLY"and c.validate_input_v1(answer["input_ref"])[0]["phase"]==phase,
  "V2 complete real prepare_phase to full contract "+phase)
# Replay the exact v1 reviewer shape with large permitted extra prerequisites.
# V2 must stop before ROOT-directory plus worst event can exceed512KiB.
c.ROOT_INPUT_REF=None;annotation_request=dict(fullrequest,instance="ram_v1_overlap_shape")
annotation_master=builder.prepare_master_v1(put("RAM/v2_annotation_master_request.json",annotation_request))
accepted_phases=0;stopped=False
for phase in c.PHASES:
 prereq=dict(ROOT_operational_annotation="x"*35000)
 if phase!="contrastive":prereq["contrastive"]=contrastive
 if phase in("RAM_retrieve_and_seal","finalize"):prereq["embeddings"]=emb
 if phase=="finalize":prereq["RAM_retrieve_and_seal"]=ram
 pq=dict(annotation_request,action="prepare_phase",master_input_ref=annotation_master["master_input_ref"],phase=phase,
  max_cuda_reserved_bytes=0 if phase=="finalize"else 32*2**30,prerequisites=prereq,
  saved_MLM_qualification_ref=None if phase=="contrastive"else qualification_ref)
 c.ROOT_INPUT_REF=None
 try:builder.prepare_phase_v1(put("RAM/v2_annotation_"+phase+"_request.json",pq));accepted_phases+=1
 except ValueError as exc:
  need("unchanged512KiB event allowance"in str(exc),"V1 overlap reproduction fails for exact combined allowance")
  stopped=True;break
need(stopped and accepted_phases<7,"V2 rejects reachable v1 all-seven35KB allowance-overlap case")
MAX_BEFORE=guard.MAX_JSON;guard.MAX_JSON=65537
try:
 rejects_request=dict(fullrequest,instance="ram_bad_event_max")
 reject(lambda:builder.prepare_master_v1(put("RAM/v2_wrong_event_max_request.json",rejects_request)),"Selected child event-bound mismatch fails before reserve")
finally:guard.MAX_JSON=MAX_BEFORE

# Default production entrypoints are genuinely inert; no process or filesystem operation is invoked.
driver=ram_module(D/"execution_v1.py","RAM_driver",dict(c=c,**{k:v for k,v in c.__dict__.items()if not k.startswith("__")}))
direct=ram_module(D/"direct_gpu0_v1.py","RAM_direct",dict(c=c,**{k:v for k,v in c.__dict__.items()if not k.startswith("__")}))
for module in(builder,driver):
 with contextlib.redirect_stdout(io.StringIO())as out:result=module.main([])
 need(result==0 and json.loads(out.getvalue())["status"]=="SOURCE_ONLY_INACTIVE","Whole actual inactive entry "+module.__name__)
# AST preservation: no new lifecycle/direct/scientific numerical behavior.
def normalized_ast(path):
 s=RealPath(path).read_text().replace("native_saved_MLM_current_Census64_integration_source_20261017_v2","native_VAL000_downstream_execution_source_20261016_v2")
 return ast.parse(s)
need(ast.dump(normalized_ast(D/"direct_gpu0_v1.py"),include_attributes=False)==ast.dump(ast.parse((OLD/"direct_gpu0_v1.py").read_text()),include_attributes=False),"Complete direct borrowed-GPU0 source AST unchanged except namespace")
oldtree=ast.parse((OLD/"execution_v1.py").read_text());newtree=normalized_ast(D/"execution_v1.py")
aliases={"actual_direct_child_wait","child_completion_ref","process_group_ESRCH","saved_MLM_wait_aliases_derived_from_same_actual_wait_and_full_owned_drain"}
for n in ast.walk(newtree):
 if isinstance(n,ast.Call):n.keywords=[k for k in n.keywords if k.arg not in aliases]
need(ast.dump(oldtree,include_attributes=False)==ast.dump(newtree,include_attributes=False),"Complete lifecycle driver AST unchanged except actual wait aliases and namespace")
for name in("GuardedWriter","finalize_worker_v1"):
 o=next(n for n in ast.parse((OLD/"worker_context_v1.py").read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
 n=next(n for n in ast.parse((D/"worker_context_v1.py").read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
 need(ast.dump(o,include_attributes=False)==ast.dump(n,include_attributes=False),"Original full "+name+" AST preserved")
for p in sorted(D.glob("*.py")):compile(p.read_bytes(),str(p),"exec");need(True,"Compile "+p.name)
result=dict(schema="native_saved_MLM_Census64_source_entry_join_checks_v1",verdict="PASS_SOURCE_CHECKS_ONLY",positive_count=len(checks),negative_count=len(negative),checks=checks,negative_cases=negative,actual_guard_imports_or_calls=False,scientific_payloads_read_or_hashed=False,numerical_execution=False,activation=False,admission=False,measured_fit=False,process_or_GPU_inspection=False)
(D/"entry_join_checks.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))
