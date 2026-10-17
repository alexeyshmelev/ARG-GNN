"""Independent native Census64 source/AST/RAM review. Never imports project modules or executes resources."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,builtins,contextlib,copy,hashlib,io,json,posixpath,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CAND=ROOT/"research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v2"
OUT=ROOT/"research/reviews/native_saved_MLM_current_Census64_independent_20261017_v2"
checks=[];negative=[];opened=[];observed=[]
def need(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
def reject(fn,label):
 try:fn()
 except (ValueError,TypeError,KeyError,AssertionError):negative.append(label);return
 raise AssertionError("Accepted invalid: "+label)
def pinned(record):
 p=ROOT/record["path"]
 need(p.suffix in(".py",".json",".md",".yaml",".patch",".txt"),"Explicit source/metadata extension "+p.name)
 data=p.read_bytes()
 need(len(data)<=8*2**20,"Bounded source/metadata "+p.name)
 need(hashlib.sha256(data).hexdigest()==record["sha256"],"Exact source/metadata pin "+p.name)
 opened.append(record);return data
def source_ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
manifest=json.loads((CAND/"source_manifest.v1.json").read_text())
for r in manifest["executed_source_refs"]+manifest["source_authentication_refs"]:pinned(r)
audit=json.loads((CAND/"audit_manifest.v2.json").read_text())
for r in audit["refs"]:pinned(r)
seals={"source_handoff.v2.json":"6d249bc5b3dbd656193ada33add52e26a380c5f15b880327a70a3199cf101bdb","source_manifest.v1.json":"bbfe7ae355c7b08031853523612b6bd9e77e50793d2b41097edbe36f91dd5953","audit_manifest.v2.json":"00d19b94be3d971366a2ecabcb7b295be337efdcd6917721a487c75b06f7ed6c","entry_join_checks_v1.py":"770c9cf5d2b0047a86e29a9aabd28d050a8fd0bc5191d74f87e641170593fc1e","entry_join_checks.result.v1.json":"5e4047f903fb730b7dfb15917c12f6845b4863738d890678f9fd54ecb9446708","source_delta.v2.patch":"1b2e584066857e821163cbd765be31d76d9140e886794393b4fd3998a282d32f"}
for n,h in seals.items():need(source_ref(CAND/n)["sha256"]==h,"Exact received seal "+n)
# Complete predecessor-preserving source delta, with only declared builder edits allowed.
V1=ROOT/"research/reviews/native_saved_MLM_current_Census64_integration_source_20261017_v1"
oldmanifest=json.loads((V1/"source_manifest.v1.json").read_text())
need(all(r in manifest["source_authentication_refs"]for r in oldmanifest["source_authentication_refs"]),"Every210 predecessor source/metadata auth ref retained exactly")
need((V1/"scientific_config.FROZEN.v1.json").read_bytes()==(CAND/"scientific_config.FROZEN.v1.json").read_bytes(),"Scientific configuration byte-identical to sealed v1")
for name in ("__init__.py","contract_v1.py","direct_gpu0_v1.py","execution_v1.py","worker_context_v1.py","science_binding_v1.py"):
 oldtree=ast.parse((V1/name).read_text())
 newtext=(CAND/name).read_text().replace(CAND.name,V1.name)
 need(ast.dump(oldtree,include_attributes=False)==ast.dump(ast.parse(newtext),include_attributes=False),"Complete source AST namespace-only delta "+name)
newtree=ast.parse((CAND/"root_builder_v1.py").read_text().replace(CAND.name,V1.name))
class RevertDeclaredBuilderDelta(ast.NodeTransformer):
 def visit_Module(self,n):
  constants={"EVENT_METADATA_ALLOWANCE_BYTES","RESERVE_EVENT_JSON_MAX_BYTES","RESERVE_EVENT_NODE_CHARGE_BYTES","MAX_ROOT_RECORD_BYTES"}
  result=[]
  for item in n.body:
   if isinstance(item,ast.Assign)and any(isinstance(t,ast.Name)and t.id in constants for t in item.targets):
    if any(isinstance(t,ast.Name)and t.id=="MAX_ROOT_RECORD_BYTES"for t in item.targets):result.append(ast.parse("MAX_ROOT_RECORD_BYTES=524288").body[0])
    continue
   result.append(item)
  n.body=result;return self.generic_visit(n)
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Attribute)and n.value.func.attr=="require"and any(isinstance(x,ast.Attribute)and x.attr=="MAX_JSON"for x in ast.walk(n)):return None
  return self.generic_visit(n)
 def visit_Call(self,n):
  n.keywords=[k for k in n.keywords if k.arg!="metadata_allocation"];return self.generic_visit(n)
 def visit_Constant(self,n):
  if n.value=="ROOT records plus worst original64KiB journal event and4096B node stay within unchanged512KiB event allowance":return ast.copy_location(ast.Constant(value="ROOT record metadata stays within unchanged512KiB reservation-event allowance"),n)
  return n
back=RevertDeclaredBuilderDelta().visit(newtree)
need(ast.dump(back,include_attributes=False)==ast.dump(ast.parse((V1/"root_builder_v1.py").read_text()),include_attributes=False),"Entire ROOT builder delta limited to declared accounting cap/assertion/disclosure")
need(manifest["v1_SOURCE_BLOCKED_review_ref"]["sha256"]=="b9389dfcc068d944712a72d35bfb70656deaf44f7ef231ac771380cc1e0c3ed0","Predecessor source rejection remains authenticated")

# Audited author full-module RAM harness. Remove only its two output writes and print.
# Its operation and project imports are all RAM doubles; no candidate file is rewritten.
t=ast.parse((CAND/"entry_join_checks_v1.py").read_text());suppressed=[]
for i,n in enumerate(t.body):
 if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call):
  f=n.value.func
  if (isinstance(f,ast.Attribute)and f.attr=="write_text")or(isinstance(f,ast.Name)and f.id=="print"):
   suppressed.append(n.lineno);t.body[i]=ast.Pass()
need(suppressed==[296,371,372],"Only audited author output statements suppressed")
ast.fix_missing_locations(t);env={"__name__":"independently_inspected_author_RAM_harness"}
exec(compile(t,"<audited_author_full_RAM_harness>","exec"),env)
need(env["result"]["positive_count"]==1449 and env["result"]["negative_count"]==22,"Audited v2 author RAM baseline 1449/22 reproduced")
for k in ("c","builder","worker","science","integration","driver","direct","guard","docs","files","dirs","read","put","local","ref","RP","request","configref","selector","policy","review","acceptance","prep","contrastive","emb","ram","qualified","initialization","locking","roles","savefn"):
 globals()[k]=env[k]
author_count=len(env["checks"]);author_negative=len(env["negative"])
# Separate our accounting from assertions emitted by audited support.
events=[]
def new_master(instance,created=None,expires=None):
 q=dict(request,instance=instance,action="prepare_master")
 if created is not None:q["created_at_utc"]=created
 if expires is not None:q["expires_at_utc"]=expires
 result=builder.prepare_master_v1(put("RAM/independent/"+instance+"/request.json",q))
 return q,result,read(result["master_input_ref"])
def phase_request(q,admitted,phase,created=None,expires=None):
 prereq={}if phase=="contrastive"else dict(contrastive=contrastive)
 if phase in("RAM_retrieve_and_seal","finalize"):prereq["embeddings"]=emb
 if phase=="finalize":prereq["RAM_retrieve_and_seal"]=ram
 p=dict(q,action="prepare_phase",master_input_ref=admitted["master_input_ref"],phase=phase,max_cuda_reserved_bytes=0 if phase=="finalize"else 32*2**30,prerequisites=prereq,saved_MLM_qualification_ref=None if phase=="contrastive"else env["qualification_ref"])
 if created is not None:p["created_at_utc"]=created
 if expires is not None:p["expires_at_utc"]=expires
 return p
q,admitted,master=new_master("independent_all_seven")
phase_prepared={}
for phase in c.PHASES:
 p=phase_request(q,admitted,phase);c.ROOT_INPUT_REF=None
 result=builder.prepare_phase_v1(put("RAM/independent/"+phase+"/request.json",p))
 phase_prepared[phase]=result
 need(result["execute_argv"][5]==c.HERE.replace("/",".")+".execution_v1","Concrete original driver module for "+phase)
 s=read(result["input_ref"]);s2,a,cfg,storage,_=c.validate_input_v1(result["input_ref"])
 need(s==s2 and a["prerequisites"]["MLM"]["historical_MLM_outer_exit_code"]==1,"Actual prepared full contract historical outer1 for "+phase)
 need(worker.check_external_prerequisites_v1(a)["saved_artifact_only"]is True,"Actual complete prerequisite worker path "+phase)
# ROOT qualification uses same genuine contrastive initialization and triple.
qr=dict(q,action="prepare_qualification",master_input_ref=admitted["master_input_ref"],initialization_ref=initialization,contrastive_prerequisite=contrastive)
qual=builder.prepare_qualification_v1(put("RAM/independent/qualify.json",qr))
need(qual["historical_MLM_outer_exit_code"]==1,"Separate qualification keeps historical exit1")
# Full worker entry through real validation and launch gate, stopping before science.
class WPath(RP):
 @classmethod
 def cwd(cls):return ROOT
class FakeOS:
 environ={}
 def read(self,fd,n):events.append(("gate_read",fd,n));return b"G"
 def close(self,fd):events.append(("gate_close",fd))
 def getpid(self):return 9001
 def getpgrp(self):return 9001
 def getsid(self,pid):return 9001
fakeos=FakeOS()
seen=[]
class Context:
 def __init__(self,inputref,s,a,cfg,storage,ticket,native):
  need(c.ROOT_INPUT_REF==inputref and s["phase"]==a["phase"],"Actual worker context/input/amendment join")
  seen.append(("context",s["phase"]))
 def run(self):seen.append(("science_boundary",None))
fakeworker=types.SimpleNamespace(NativeContext=Context,finalize_worker_v1=lambda *args:seen.append(("finalize_boundary",args[1]["phase"])))
fakedirect=types.SimpleNamespace(install_before_torch_v1=lambda *args:seen.append(("lease_title_before_science",args[0]["phase"])))
real_import=builtins.__import__
def no_project_import(name,globals=None,locals=None,fromlist=(),level=0):
 if name.startswith("research."):
  if name.endswith(".worker_context_v1"):return fakeworker
  if fromlist==("direct_gpu0_v1",)or list(fromlist)==["direct_gpu0_v1"]:return types.SimpleNamespace(direct_gpu0_v1=fakedirect)
  raise AssertionError("Unexpected actual project import "+name)
 return real_import(name,globals,locals,fromlist,level)
driver.__dict__["__builtins__"]=dict(vars(builtins),__import__=no_project_import)
driver.Path=WPath;driver.os=fakeos;driver.tempfile=types.SimpleNamespace(tempdir=None)
driver.sys=types.SimpleNamespace(dont_write_bytecode=True,platform="linux")
# function's import opcode resolves builtins retained when created; reconstruct in same globals.
fn=next(n for n in ast.parse((CAND/"execution_v1.py").read_text()).body if isinstance(n,ast.FunctionDef)and n.name=="worker_v1")
exec(compile(ast.Module(body=[fn],type_ignores=[]),"<unaltered_complete_worker_entry_RAM>","exec"),driver.__dict__)
for phase,result in phase_prepared.items():
 s=read(result["input_ref"]);c.ROOT_INPUT_REF=None;c.set_amendment_v1(result["input_ref"])
 a=read(s["root_stage_request_ref"]);paths=c.worker_env_paths_v1(s)
 for v in paths.values():dirs.add(v)
 fakeos.environ=dict(paths,CUDA_VISIBLE_DEVICES=""if phase=="finalize"else c.GPU0_UUID)
 if phase=="finalize":ticketref=None
 else:ticketref=put(s["worker_auxiliary_scope"]+"/attachment.json",dict(job_id=s["job_id"],borrowed_physical_inode=True,direct_attachment_authority_ref=a["direct_attachment_authority_ref"]))
 need(driver.worker_v1(result["input_ref"],77,ticketref)==0,"Complete actual gate-to-worker contract path "+phase)
need(len([x for x in seen if x[0]=="lease_title_before_science"])==6,"All six GPU phases lease/title gate before science")
need(len([x for x in seen if x[0]=="finalize_boundary"])==1,"Actual finalizer entry stays CPU route")
# Context request mapper, scientific dispatch and output mkdir all RAM; numerical boundaries are sentinels.
worker.source_stages_v1=lambda:types.SimpleNamespace()
c.guard_v1=lambda a:types.SimpleNamespace()
# Instantiate real context against each actual prepared input and inspect request authority.
for phase,result in phase_prepared.items():
 if phase=="finalize":continue
 s=read(result["input_ref"]);c.ROOT_INPUT_REF=None;c.set_amendment_v1(result["input_ref"]);_,a,cfg,storage,_=c.validate_input_v1(result["input_ref"])
 ctx=worker.NativeContext(result["input_ref"],s,a,cfg,storage,{},None)
 context,req=ctx.context_and_request()
 need(req["storage_policy_selector_ref"]==selector and context["storage_policy_selector_ref"]==selector,"Actual scientific context/request selector "+phase)
 need(context["MLM_prerequisite_metadata"]["historical_MLM_outer_exit_code"]==1,"Actual scientific context saves outer1 "+phase)
 context["validate_phase_authority_v1"](copy.deepcopy(req))
 bad=copy.deepcopy(req);bad["storage_policy_selector_ref"]=dict(path="old_route",sha256="0"*64)
 reject(lambda:context["validate_phase_authority_v1"](bad),"Scientific request rejects alternate selector "+phase)
# Independent negative joins mutate one genuine field at a time.
basebundle=copy.deepcopy(contrastive)
for field,value in (("returncode",1),("actual_direct_worker_wait",False),("actual_OS_wait_status",256),("all_owned_processes_and_groups_drained",False),("complete_owned_roster_known",False),("operational_failure","UNKNOWN"),("group_observations",[dict(absent=False)]),("actual_direct_child_wait",False),("process_group_ESRCH",False),("completed_ref",dict(path="other",sha256="0"*64)),("child_completion_ref",dict(path="other",sha256="0"*64))):
 bad=read(basebundle["wait_ref"]);bad[field]=value
 bundle=dict(basebundle,wait_ref=put("RAM/independent/bad_wait_"+field+".json",bad))
 outer=read(basebundle["outer_terminal_ref"]);outer["actual_wait_ref"]=bundle["wait_ref"]
 bundle["outer_terminal_ref"]=put("RAM/independent/rebound_outer_"+field+".json",outer)
 reject(lambda:worker.prerequisite_metadata_v1(bundle,"contrastive"),"Actual contrastive prerequisite rejects "+field)
for field,value in (("actual_exit_code",1),("actual_terminal_consumed",False),("actual_direct_worker_returncode",1),("all_owned_processes_and_groups_drained",False),("actual_wait_ref",dict(path="other",sha256="0"*64)),("actual_tool_session_id",0),("schema","generic_outer"),("terminal_consumed",False),("actual_tool_terminal_exit_code",1),("tool_session_id",99),("child_completion_ref",dict(path="other",sha256="0"*64))):
 bad=read(basebundle["outer_terminal_ref"]);bad[field]=value
 bundle=dict(basebundle,outer_terminal_ref=put("RAM/independent/bad_outer_"+field+".json",bad))
 reject(lambda:worker.prerequisite_metadata_v1(bundle,"contrastive"),"Actual consumed outer prerequisite rejects "+field)
for phase in c.PHASES:
 a=read(read(phase_prepared[phase]["input_ref"])["root_stage_request_ref"])
 reject(lambda:worker.check_external_prerequisites_v1(dict(a,ROOT_saved_MLM_artifact_acceptance_ref=None)),"Missing separate saved-artifact acceptance blocks "+phase)
 if phase!="contrastive":
  reject(lambda:worker.check_external_prerequisites_v1(dict(a,saved_MLM_qualification_ref=None)),"Missing separate qualification blocks "+phase)
# Exact source adaptation captures the compile AST without executing any science.
captured=[]
def capture_compile(tree,*args,**kwargs):
 if isinstance(tree,ast.AST):captured.append(copy.deepcopy(tree))
 return compile(tree,*args,**kwargs)
science.__dict__["compile"]=capture_compile
science._INTEGRATION=None;science._ENTRY=None;science._SELECTED=None
c.load=env["scientific_load"]
c.raw=lambda r,prior=c.raw:(ROOT/r["path"]).read_bytes()if r in(c.SAVED_INTEGRATION_REF,c.SAVED_SCIENTIFIC_ENTRY_REF)else prior(r)
again=science.integration_v1(selector);entry=science.entry_v1(selector)
need(len(captured)==2,"Exactly function selector bridge and original entry module compiled")
original_tree=ast.parse((ROOT/c.SAVED_INTEGRATION_REF["path"]).read_text())
bind=copy.deepcopy(next(n for n in original_tree.body if isinstance(n,ast.FunctionDef)and n.name=="bind_stage_v1"))
target=ast.dump(ast.parse('PINS["current_selector"]',mode="eval").body,include_attributes=False)
class OnlySelector(ast.NodeTransformer):
 def visit_Subscript(self,n):
  if ast.dump(n,include_attributes=False)==target:return ast.copy_location(ast.Name(id="_actual_Census_selector_ref_v1",ctx=ast.Load()),n)
  return self.generic_visit(n)
expected=OnlySelector().visit(bind)
need(ast.dump(captured[0].body[0],include_attributes=False)==ast.dump(expected,include_attributes=False),"Only exact two selector operands differ in actual compiled scientific binding")
ot=ast.parse((ROOT/c.SAVED_SCIENTIFIC_ENTRY_REF["path"]).read_text())
for node in ot.body:
 if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
  actual=next(n for n in captured[1].body if isinstance(n,type(node))and n.name==node.name)
  need(ast.dump(node,include_attributes=False)==ast.dump(actual,include_attributes=False),"Actual scientific entry function AST exact "+node.name)
need(again.PINS["current_selector"]!=selector and entry.integration is again,"Historical scientific selector pin intact; same private entry integration")

# Exact cap intersections and checkpoint peak from independently invented size rows.
c.active_amendment_v1=lambda:None
c.STATE=(master,{}, {}, {})
sizes={}
c.original._sizes=lambda p,**kwargs:sizes.get(str(p),[])
o=str(local(master["output_directory"]));u=str(local(master["worker_auxiliary_scope"]))
sizes[o]=[(("contrastive","x"),8*2**30,"regular"),(("embedding_contrastive_train","x"),55*2**30,"regular"),(("RAM_retrieve_and_seal","x"),2**30,"regular")]
need(sum(c.charged_census_v1(dict(phase="finalize")).values())==64*2**30,"Exactly64GiB intersection accepts unchanged 8+55+1 groups")
reject(lambda:c.charged_census_v1(dict(phase="finalize"),required_write_bytes=1),"64GiB aggregate rejects one byte even below summed65 groups")
sizes[o]=[(("embedding_contrastive_train","x"),56*2**30+1,"regular")]
reject(lambda:c.charged_census_v1(dict(phase="embedding_contrastive_train")),"Embedding56GiB cap remains strict")
sizes[o]=[(("contrastive","checkpoints","one.pt"),3*2**30+4096,"regular"),(("contrastive","checkpoints","two.pt"),3*2**30+4096,"regular")]
need(c.charged_census_v1(dict(phase="contrastive"))[master["output_directory"]]==6*2**30+8192,"Exactly two regular3GiB checkpoints allowed transiently")
sizes[o].append((("contrastive","checkpoints","three.pt"),4097,"regular"))
reject(lambda:c.charged_census_v1(dict(phase="contrastive")),"Third regular checkpoint rejected")
sizes[o]=[(("contrastive","checkpoints","one.pt"),3*2**30+4097,"regular")]
reject(lambda:c.charged_census_v1(dict(phase="contrastive")),"3GiB checkpoint one-byte oversize rejected")
sizes[o]=[(("forged_phase","x"),1,"regular")]
reject(lambda:c.charged_census_v1(dict(phase="contrastive")),"Unknown phase cannot escape group cap")
result=dict(schema="native_Census64_independent_source_checks_v1",independent_positive_count=len(checks),independent_negative_count=len(negative),checks=checks,negative_cases=negative,audited_author_baseline_positive_count=author_count,audited_author_baseline_negative_count=author_negative,source_authentication_refs=len(manifest["source_authentication_refs"]),executed_source_refs=len(manifest["executed_source_refs"]),artifact_refs=len(audit["refs"]),opened_source_metadata_refs=len(opened),all_seven_full_phase_builder_and_worker_joins=True,actual_guard_state_resource_calls=False,numerical_execution=False,payload_reads_or_hashes=False,candidate_source_modified=False)
(OUT/"checks.result.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()if k not in("checks","negative_cases")},sort_keys=True))
