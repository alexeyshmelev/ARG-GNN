"""Real immutable selector and namespace memo on synthetic budget/guard modules only."""
from pathlib import Path
import ast,contextlib,copy,hashlib,importlib.util,json,sys,types,stat,os
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/American_C3_VAL004_fresh7500MB_authority_source_20261016_v1"
B="research/reviews/American_C3_VAL004_fresh7500MB_independent_review_20261016_v1"
def load(p,label):
 sp=importlib.util.spec_from_file_location(label,R/p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def module(path,name,names,extra):
 m=types.ModuleType(name);m.__file__=str(R/path);m.__dict__.update(extra)
 compiled=compile((R/path).read_bytes(),str(R/path),"exec",dont_inherit=True,optimize=0)
 for name in names:
  code=next(x for x in compiled.co_consts if isinstance(x,types.CodeType)and x.co_name==name)
  setattr(m,name,types.FunctionType(code,m.__dict__,name))
 return m
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):return
 raise AssertionError("Expected selector/memo rejection")
memo=load("research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","_independent_VAL004_real_memo")
checks=[]
for index in (0,1):
 c=load(C+"/root_contract_v1.py","_independent_VAL004_real_selector_contract_"+str(index))
 v4path=c.V4_REF["path"];v9path="scripts/project_budget_v9.py"
 v4=module(v4path,"_independent_VAL004_synthetic_v4_"+str(index),{"canonical","file_bytes","mutex","reference","scope_spec"},dict(Path=Path,contextlib=contextlib,stat=stat,os=os,hashlib=hashlib,MAX_JSON=32*1024*1024))
 budget=module(v9path,"_independent_VAL004_synthetic_budget_"+str(index),{"successor_paths"},dict(v4=v4,Path=Path))
 original=tuple(getattr(v4,n)for n in ("canonical","file_bytes","mutex"))
 docs={};events=[]
 def put(label,value):
  raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
  rr=dict(path=B+"/SYNTHETIC_selector_"+str(index)+"_"+label+".json",sha256=hashlib.sha256(raw).hexdigest())
  docs[rr["path"]]=(rr,copy.deepcopy(value),raw);return rr
 def bound(root,rr,*a):
  if rr["path"]in docs:
   expected,value,raw=docs[rr["path"]];assert expected==rr;return raw
  p=R/rr["path"];assert p.suffix in (".py",".json");raw=p.read_bytes()
  assert hashlib.sha256(raw).hexdigest()==rr["sha256"];return raw
 v4.bound=bound
 def read(rr,cap=33554432):return json.loads(bound(R,rr))
 choice=c.APPROVED_GUARDS[index]
 source_review=put("source_review",dict(verdict="SOURCE_PASS",source_ref=choice["source_ref"],
  source_manifest_ref=choice["source_manifest_ref"],measurement_source_ref=choice["measurement_source_ref"]))
 mode="native14_stopped_retention_v3"if index==0 else"native7_H050_stopped_retention_v1"
 journal=".runtime/storage/"+mode
 lineage=dict(base_ref=dict(path=B+"/SYNTHETIC_parent.py",sha256="a"*64),
  base_policy_ref=dict(path=B+"/SYNTHETIC_parent_policy.json",sha256="b"*64),
  base_pins=dict(SYNTHETIC=True),base_projected_bytes=123456,base_accounting_mode="SYNTHETIC_PARENT")
 policy=put("policy",dict(choice,**lineage,source_review_ref=source_review,active=True,root_authorizer="root_agent",
  marker_repair_authority_ref=c.REPAIR_REF,journal=journal,scheduling_ceiling_bytes=980000000000,
  human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000))
 guard=types.SimpleNamespace(budget=budget,MODE=mode,JOURNAL=journal,_POLICY_REF=None)
 def configure(repair,*,policy_ref):
  assert repair==c.REPAIR_REF and policy_ref==policy;guard._POLICY_REF=copy.deepcopy(policy_ref);events.append("synthetic_guard_configured")
 guard.configure_with_renewal_v1=configure
 class Loader:
  def exec_module(self,target):target.__dict__.update(vars(guard))
 fake_lib=types.SimpleNamespace(util=types.SimpleNamespace(
  spec_from_file_location=lambda name,path:types.SimpleNamespace(loader=Loader()),
  module_from_spec=lambda spec:guard))
 selector=module(c.SELECTOR_SOURCE_REF["path"],"_independent_VAL004_real_selector_"+str(index),
  {"require","_read","selector_draft_v1","load_selected_guard_v1"},
  dict(ROOT=R,SOURCE=c.SELECTOR_SOURCE_REF["path"],budget=budget,copy=copy,importlib=fake_lib,
   APPROVED_PATHS=tuple(x["source_ref"]["path"]for x in c.APPROVED_GUARDS),
   _SELECTED_REF=None,_SELECTED_VALUE=None,_GUARD=None))
 selector._read=read
 value=selector.selector_draft_v1(**choice,policy_ref=policy,source_review_ref=source_review,
   repair_authority_ref=c.REPAIR_REF,accounting_mode=mode,journal=journal,frozen_parent_lineage=lineage)
 value["active"]=True;selected=put("selector",value)
 c.read=read;c.raw=lambda rr:bound(R,rr)
 def c_load(rr,name):
  if rr==c.SELECTOR_SOURCE_REF:return selector
  if rr==c.NAMESPACE_REF:return memo
  raise AssertionError("Unexpected actual adapter load "+str(rr))
 c.load=c_load
 c.bind_selector(selected);g=c.original_guard();assert len(events)==1
 assert c.original_guard()is g and g.budget is budget and g._POLICY_REF==policy
 reject(lambda:c.bind_selector(put("swapped_selector",dict(value,policy_ref=put("other_policy",{})))))
 # Required real execution ordering: any input validation/guard access before install.
 for _ in range(3):assert c.original_guard()is g
 assert len(events)==1 and all(getattr(v4,n)is f for n,f in zip(("canonical","file_bytes","mutex"),original))
 g2,handle=c.consumer_guard()
 assert g2 is g and handle.receipt["installed"]is True
 assert any(getattr(v4,n)is not f for n,f in zip(("canonical","file_bytes","mutex"),original))
 reject(lambda:c.bind_selector(selected))
 reject(c.original_guard)
 reject(c.consumer_guard)
 # Synthetic cleanup only.
 handle.uninstall()
 assert all(getattr(v4,n)is f for n,f in zip(("canonical","file_bytes","mutex"),original))
 assert c.original_guard()is g and len(events)==1
 checks.append(dict(name="real_selector_and_real_memo_"+mode,status="PASS_SOURCE_ONLY",
  source_choice=index,guard_configured_once=True,before_memo_repeated_access_same_guard=True,
  after_memo_reselection_factory_and_reinstall_reject=True,synthetic_original_functions_restored=True))
assert not any(n in sys.modules for n in ("scripts.project_budget_v9","arggnn.project_budget_v4","torch","numpy","tskit"))
value=dict(schema="VAL004_fresh7500_real_selector_memo_synthetic_check_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 real_selector_code_real_namespace_installer=True,all_budget_guard_modules_and_selector_policy_review_documents_synthetic=True,
 actual_guard_configure_state_mutex_resource_probe_accounting_admission_retention_jobs_or_Git=False)
with(R/B/"independent_selector_memo.v1.json").open("x")as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=value["status"],checks=len(checks))))
