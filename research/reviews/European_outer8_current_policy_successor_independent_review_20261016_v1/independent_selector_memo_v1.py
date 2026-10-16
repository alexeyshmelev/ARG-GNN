"""Exact real selector/memo and successor wrapper code; all guard/budget state is synthetic."""
from pathlib import Path
import contextlib,copy,hashlib,importlib.util,json,sys,types,stat,os
R=Path("/workspace-SR003.nfs2/ARG-GNN")
C="research/reviews/European_outer8_current_policy_successor_source_20261016_v1"
B="research/reviews/European_outer8_current_policy_successor_independent_review_20261016_v1"
def load(path,label):
 sp=importlib.util.spec_from_file_location(label,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def module(path,label,names,extra):
 m=types.ModuleType(label);m.__file__=str(R/path);m.__dict__.update(extra)
 compiled=compile((R/path).read_bytes(),str(R/path),"exec",dont_inherit=True,optimize=0)
 for name in names:
  code=next(x for x in compiled.co_consts if isinstance(x,types.CodeType)and x.co_name==name)
  setattr(m,name,types.FunctionType(code,m.__dict__,name))
 return m
def reject(fn):
 try:fn()
 except (ValueError,KeyError,TypeError,AssertionError):return
 raise AssertionError("Expected selector/memo rejection")
memo=load("research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","_independent_European_real_memo")
checks=[]
for index,mode in enumerate(("native14_stopped_retention_v3","native7_H050_stopped_retention_v1")):
 c=load(C+"/prepare_outer000_v1.py","_independent_European_real_selector_source_"+str(index))
 v4path=c.V4["path"];v9path="scripts/project_budget_v9.py"
 v4=module(v4path,"_independent_European_synthetic_v4_"+str(index),{"canonical","file_bytes","mutex","reference","scope_spec"},
  dict(Path=Path,contextlib=contextlib,stat=stat,os=os,hashlib=hashlib,MAX_JSON=32*1024*1024))
 budget=module(v9path,"_independent_European_synthetic_budget_"+str(index),{"successor_paths"},dict(v4=v4,Path=Path))
 original=tuple(getattr(v4,n)for n in ("canonical","file_bytes","mutex"))
 docs={};events=[]
 def put(label,value,identifier=None):
  raw=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
  rr=identifier or dict(path=B+"/SYNTHETIC_selector_"+str(index)+"_"+label+".json",sha256=hashlib.sha256(raw).hexdigest())
  docs[rr["path"]]=(rr,copy.deepcopy(value),raw);return rr
 def bound(root,rr,*a):
  if rr["path"]in docs:
   expected,value,raw=docs[rr["path"]];assert expected==rr;return raw
  path=R/rr["path"]
  assert str(path).startswith(str(R/"research/reviews/"))and path.suffix in (".py",".json")
  raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==rr["sha256"];return raw
 v4.bound=bound
 def read(rr,*args):return json.loads(bound(R,rr))
 choice=c.APPROVED[mode]
 measure_path="research/reviews/"+("native14_stopped_retention_successor_source_20261016_v3"if index==0 else"native7_H050_stopped_retention_successor_source_20261016_v1")+"/retained_nodes_v1.py"
 exact=dict(source_ref=choice["source_ref"],source_manifest_ref=choice["manifest_ref"],measurement_source_ref=c.ref(measure_path))
 review=put("review",dict(verdict="SOURCE_PASS",**exact))
 lineage=dict(base_ref=dict(path=B+"/SYNTHETIC_parent.py",sha256="a"*64),base_policy_ref=dict(path=B+"/SYNTHETIC_parent_policy.json",sha256="b"*64),
  base_pins=dict(SYNTHETIC=True),base_projected_bytes=123456,base_accounting_mode="SYNTHETIC_PARENT")
 journal=".runtime/storage/"+mode
 policy=put("policy",dict(exact,**lineage,source_review_ref=review,active=True,root_authorizer="root_agent",
  marker_repair_authority_ref=c.REPAIR,journal=journal,scheduling_ceiling_bytes=980000000000,
  human_limit_bytes=1000000000000,warning_threshold_bytes=950000000000),identifier=choice["policy_ref"])
 # Current fixed policy identifier is virtualized to synthetic metadata. No actual active policy or guard is read.
 g=types.SimpleNamespace(budget=budget,MODE=mode,JOURNAL=journal,_POLICY_REF=None)
 def configure(repair,*,policy_ref):
  assert repair==c.REPAIR and policy_ref==policy;g._POLICY_REF=copy.deepcopy(policy_ref);events.append("synthetic_configure")
 g.configure_with_renewal_v1=configure
 class Loader:
  def exec_module(self,target):target.__dict__.update(vars(g))
 fake_lib=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=lambda name,path:types.SimpleNamespace(loader=Loader()),
  module_from_spec=lambda spec:g))
 selector=module(c.SELECTOR["path"],"_independent_European_real_selector_"+str(index),
  {"require","_read","selector_draft_v1","load_selected_guard_v1"},
  dict(ROOT=R,SOURCE=c.SELECTOR["path"],budget=budget,copy=copy,importlib=fake_lib,
   APPROVED_PATHS=tuple(v["source_ref"]["path"]for v in c.APPROVED.values()),_SELECTED_REF=None,_SELECTED_VALUE=None,_GUARD=None))
 selector._read=read
 value=selector.selector_draft_v1(**exact,policy_ref=policy,source_review_ref=review,repair_authority_ref=c.REPAIR,
  accounting_mode=mode,journal=journal,frozen_parent_lineage=lineage)
 value["active"]=True;selected=put("selected",value)
 c.read=read;c.raw=lambda rr:bound(R,rr)
 calls=[]
 def c_load(rr,label):
  if rr==c.SELECTOR:
   calls.append("selector");return selector
  if rr==c.NAMESPACE:
   calls.append("memo");return memo
  raise AssertionError("Unexpected actual adapter load")
 c.load=c_load
 # One selection/configuration before installation, generation uses originals.
 for i in range(3):
  assert c.guard(selected,False)is g and not hasattr(g,"_outer_memo")
 assert events==["synthetic_configure"]and calls==["selector"]
 assert all(getattr(v4,n)is fn for n,fn in zip(("canonical","file_bytes","mutex"),original))
 # The exact real memo installs once on synthetic source-derived modules.
 assert c.guard(selected,True)is g
 handle=g._outer_memo
 assert handle.receipt["installed"]is True and calls==["selector","memo"]
 assert any(getattr(v4,n)is not fn for n,fn in zip(("canonical","file_bytes","mutex"),original))
 # The real selector correctly refuses entry after installation; the new wrapper must not reenter it.
 reject(lambda:selector.load_selected_guard_v1(selected))
 for consumer in (True,True,False,True):
  assert c.guard(selected,consumer)is g and g._outer_memo is handle
 assert events==["synthetic_configure"]and calls==["selector","memo"]
 reject(lambda:c.guard(put("same_value_different_record",value),True))
 g._POLICY_REF=put("drifted_policy",{});reject(lambda:c.guard(selected,True));g._POLICY_REF=policy
 g.MODE="SYNTHETIC_DRIFT";reject(lambda:c.guard(selected,True));g.MODE=mode
 reject(lambda:c.guard(put("bad_ceiling",dict(value,scheduling_ceiling_bytes=980000000001)),True))
 handle.uninstall()
 assert all(getattr(v4,n)is fn for n,fn in zip(("canonical","file_bytes","mutex"),original))
 # First attempt on an already wrapped namespace cannot select or configure.
 d=load(C+"/prepare_outer000_v1.py","_independent_European_prewrapped_"+str(index))
 d.read=read;d.raw=c.raw;d.load=c_load
 wrapped=lambda *a,**k:None
 v4.mutex=wrapped
 selector._SELECTED_REF=selector._SELECTED_VALUE=selector._GUARD=None
 before=len(events);reject(lambda:d.guard(selected,True));assert len(events)==before
 v4.mutex=original[2]
 checks.append(dict(name="real_selector_and_real_memo_"+mode,status="PASS_SOURCE_ONLY",
  exact_selected_guard_configures_once_before_memo=True,wrapper_reuses_cached_guard_after_memo=True,
  namespace_memo_installed_once=True,after_memo_real_selector_reentry_rejects=True,
  selector_reference_mode_policy_or_ceiling_drift_rejects=True,prewrapped_initial_selection_rejects=True,
  actual_active_policy_not_read=True,current_policy_identifier_virtualized=index==0))
assert not any(n in sys.modules for n in ("scripts.project_budget_v9","arggnn.project_budget_v4","torch","numpy","tskit"))
result=dict(schema="European_outer8_current_policy_real_selector_memo_synthetic_check_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 exact_real_selector_and_namespace_installer=True,all_budget_guard_state_runtime_and_authority_facts_synthetic=True,
 no_actual_guard_configure_state_mutex_resource_probe_accounting_admission_retention_jobs_or_Git=True)
with(R/B/"independent_selector_memo.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=result["status"],checks=len(checks))))
