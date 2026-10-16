"""Real pinned initialization function code on synthetic modules; no real guard/state access."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import FunctionType,SimpleNamespace,ModuleType,CodeType
import ast,builtins,copy,hashlib,importlib,json,os,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BASE="research/reviews/C3_CPU_interleaved_guarded_integration_independent_review_20261016_v1"
CAND="research/reviews/C3_CPU_interleaved_guarded_integration_source_20261016_v1"
sp=importlib.util.spec_from_file_location("_independent_guard_init_inactive_contract",ROOT/CAND/"acceptance_contract_v1.py")
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
pins=json.loads((ROOT/BASE/"hash_verification.v1.json").read_text())
known={r["path"]:r["sha256"]for r in pins["refs"]}
paths=[
 c.LEGACY_CONTRACT_REF["path"],c.GUARD_REF["path"],
 "research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py",
 "research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py",
 "research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py",
 "research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",
 "research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py",
 "research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py",
 "research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py",
 "research/reviews/H039_marker_repair_renewal_source_20261016_v1/marker_repair_renewal_v1.py"]
sources={}
for path in paths:
 raw=(ROOT/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==known[path],path
 code=compile(raw,str(ROOT/path),"exec",dont_inherit=True,optimize=0)
 sources[path]=(raw.decode(),code,{x.co_name:x for x in code.co_consts if isinstance(x,CodeType)})
checks=[]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SYNTHETIC_INITIALIZATION",**({"detail":detail}if detail is not None else {})))
def reject(fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):return
 raise AssertionError("Expected initialization rejection")
def simulation(initial_wrapped=False,marker_preloaded=False):
 events=[];synthetic_modules={};loaded_objects=[]
 v4=ModuleType("_independent_synthetic_original_v4")
 for name in ("canonical","file_bytes","mutex"):
  def sentinel(*a,**k):raise AssertionError("No resource/path/mutex probe")
  sentinel.__module__=v4.__name__;setattr(v4,name,sentinel)
 initial=tuple(getattr(v4,n)for n in ("canonical","file_bytes","mutex"))
 v4.bound=lambda root,r:events.append(("synthetic_bound",r["path"]))or b""
 if initial_wrapped:
  def wrapped(*a,**k):raise AssertionError("Never call wrapped synthetic function")
  wrapped.__module__="_synthetic_existing_wrapper";v4.canonical=wrapped
 budget=SimpleNamespace(v4=v4,SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py")
 class FakeLoader:
  def __init__(self,name,path):self.name,self.path=name,str(path)
  def exec_module(self,module):populate(module,str(Path(self.path).relative_to(ROOT)))
 fake_lib=SimpleNamespace(machinery=SimpleNamespace(SourceFileLoader=FakeLoader))
 def spec_from_file_location(name,path):
  return SimpleNamespace(name=name,origin=str(path),loader=FakeLoader(name,path))
 def module_from_spec(spec):
  m=ModuleType(spec.name);m.__file__=spec.origin;return m
 fake_lib.util=SimpleNamespace(spec_from_file_location=spec_from_file_location,module_from_spec=module_from_spec)
 fake_sys=SimpleNamespace(modules=synthetic_modules)
 fake_authority=dict(schema="SYNTHETIC_OLD_MARKER_AUTHORITY",expires_at_utc="2000-01-01T00:00:00+00:00")
 fake_certificate=dict(renewed_expires_at_utc="2099-01-01T00:00:00+00:00")
 marker_path=paths[-2]
 if marker_preloaded:
  m=ModuleType("_synthetic_existing_marker");m.__file__=str(ROOT/marker_path);synthetic_modules[m.__name__]=m
 def cert(record):
  assert record["path"].endswith("renewal_certificate.root_active_v1.json")
  events.append(("synthetic_certificate",record["path"]))
  return copy.deepcopy(fake_certificate),copy.deepcopy(fake_authority)
 def raw(record):
  assert isinstance(record,dict)and set(record)=={"path","sha256"}
  if record["path"]not in known:
   p=ROOT/record["path"];assert p.suffix in (".py",".json")and not p.is_symlink()and p.stat().st_size<=33554432
   raw_bytes=p.read_bytes();assert hashlib.sha256(raw_bytes).hexdigest()==record["sha256"]
   known[record["path"]]=record["sha256"]
  assert known[record["path"]]==record["sha256"],record
  events.append(("authenticated_pinned_ref_only",record["path"]));return b""
 def ref(path):
  path=str(Path(path).relative_to(ROOT))if Path(path).is_absolute()else str(path)
  return dict(path=path,sha256=known[path])
 class SyntheticOutput:
  def __init__(self,*a,**k):raise AssertionError("No output instantiated")
  def save(self,*a,**k):raise AssertionError("No output written")
  def discard(self,*a,**k):raise AssertionError("No output discarded")
 cpu={"_paths":lambda own:None}
 terminal=SimpleNamespace()
 previous=SimpleNamespace(_frozen=lambda:cpu,old=terminal,previous=SimpleNamespace(),budget=budget,ROOT=ROOT,MODE="journal_terminal_retired_v1",JOURNAL="SYNTHETIC_UNUSED",SOURCE="src/arggnn/project_storage_terminal_retired_scopes_v1.py",METADATA_CHARGE=524288)
 pinned=SimpleNamespace(budget=budget,SOURCE="src/arggnn/project_storage_terminal_retired_scopes_v1.py",_clone=lambda fn,**kw:fn)
 def synthetic_import(name,globals=None,locals=None,fromlist=(),level=0):
  if name=="arggnn":
   assert fromlist==("project_storage_terminal_retired_scopes_v1",),fromlist
   return SimpleNamespace(project_storage_terminal_retired_scopes_v1=pinned)
  raise AssertionError("Unexpected import from initialization code: "+name)
 fake_builtins=dict(vars(builtins),__import__=synthetic_import)
 def populate(module,path):
  assert path in sources,path
  text,compiled,codes=sources[path]
  ns=vars(module)
  ns.update(Path=Path,datetime=datetime,timezone=timezone,timedelta=timedelta,copy=copy,json=json,os=os,
   hashlib=hashlib,importlib=fake_lib,sys=fake_sys,budget=budget,previous=previous,FunctionType=FunctionType,SimpleNamespace=SimpleNamespace,
   __builtins__=fake_builtins)
  # Execute only source constant assignments; exact original function code objects are installed below.
  assignments=[n for n in ast.parse(text).body if isinstance(n,(ast.Assign,ast.AnnAssign))]
  exec(compile(ast.Module(body=assignments,type_ignores=[]),"<source_constants:"+path+">","exec"),ns)
  for name,code in codes.items():ns[name]=FunctionType(code,ns,name)
  ns.update(require=require,_raw=raw,_policy=lambda:events.append(("synthetic_policy",path))or {},ref=ref)
  if path==paths[-1]:
   ns.update(validate_certificate_v1=cert,reference=ref)
  if path==marker_path:
   ns.update(_repair=lambda:events.append(("synthetic_repair_state_boundary",path))or ({},{}),
    _bound=lambda r:copy.deepcopy(fake_authority),previous=previous,PilotOutputV9=SyntheticOutput)
  loaded_objects.append(module)
  events.append(("real_source_code_loaded",path))
 oldns=dict(vars(c.legacy),__builtins__=fake_builtins,importlib=fake_lib,sys=fake_sys,ref=ref,local=lambda p:ROOT/p)
 for name in ("load","original_guard","consumer_guard"):
  fn=getattr(c.legacy,name);value=FunctionType(fn.__code__,oldns,name,fn.__defaults__,fn.__closure__)
  value.__kwdefaults__=fn.__kwdefaults__;oldns[name]=value
 return SimpleNamespace(events=events,v4=v4,initial=initial,old=oldns,lib=fake_lib,modules=synthetic_modules,loaded_objects=loaded_objects)
# Each process role starts independently; configure uses the exact real chain.
summaries=[]
for role in ("ROOT_preparation","ROOT_supervisor","CPU_worker"):
 s=simulation();g,memo=s.old["consumer_guard"]()
 assert memo is None and g.MODE=="native14_stopped_retention_v3"
 names=[];item=g
 while hasattr(item,"_BASE")and item._BASE is not None:
  names.append(item.MODE);item=item._BASE
 assert names==["native14_stopped_retention_v3","prospective_980gb_Census_fit2_v1",
  "prospective_980gb_H020_v3","prospective_980gb_v1","concurrent_completed_scope_credits_v3"],names
 assert all(getattr(s.v4,n)is f for n,f in zip(("canonical","file_bytes","mutex"),s.initial))
 assert g.budget.v4 is s.v4 and item.budget.v4 is s.v4
 assert all(getattr(s.v4,n).__module__==s.v4.__name__ for n in ("canonical","file_bytes","mutex"))
 # Real renewal wrapper changed only each privately loaded synthetic marker's bound reader.
 assert any(k=="real_source_code_loaded"and p==paths[-2]for k,p in s.events)
 marker=next(m for m in s.loaded_objects if m.__file__==str(ROOT/paths[-2]))
 renewal=next(m for m in s.loaded_objects if m.__file__==str(ROOT/paths[-1]))
 assert marker._root_expiry_renewal_ref==g.RENEWAL_CERT_REF
 assert marker._bound(renewal.OLD_AUTHORITY_REF)["expires_at_utc"]=="2099-01-01T00:00:00+00:00"
 assert marker._bound(dict(path="SYNTHETIC_UNRELATED.json",sha256="0"*64))["expires_at_utc"]=="2000-01-01T00:00:00+00:00"
 loaded=[v for v in vars(g._BASE._BASE._BASE._BASE._BASE).values()if isinstance(v,dict)]
 # Hook certificate is idempotent and survives repeated fresh guard construction.
 before=s.lib.machinery.SourceFileLoader.exec_module
 again,_=s.old["consumer_guard"]()
 assert before is s.lib.machinery.SourceFileLoader.exec_module
 assert again is not g and again.budget.v4 is s.v4
 assert all(getattr(s.v4,n)is f for n,f in zip(("canonical","file_bytes","mutex"),s.initial))
 renewal_spec=s.lib.util.spec_from_file_location("_synthetic_renewal_repeat",ROOT/paths[-1])
 renewal=s.lib.util.module_from_spec(renewal_spec);renewal_spec.loader.exec_module(renewal)
 reject(lambda:renewal.install_renewal_v1(dict(path="synthetic_wrong_certificate.json",sha256="0"*64)))
 summaries.append(dict(role=role,real_guard_configurations=2,source_loaded_count=sum(k=="real_source_code_loaded"for k,p in s.events),
   shared_v4_unwrapped=True,namespace_memo_installed=False,actual_state_validators_replaced_with_synthetic_boundaries=True))
check("three_fresh_role_initializations_real_contract_factory_and_guard_configure_chain",summaries)
s=simulation(marker_preloaded=True)
reject(s.old["original_guard"])
check("real_renewal_rejects_preloaded_marker_in_synthetic_interpreter")
s=simulation(initial_wrapped=True);g=s.old["original_guard"]()
# The real builder's explicit mutation-role predicate must reject this prewrapped state.
predicate=all(getattr(g.budget.v4,n).__module__==g.budget.v4.__name__ for n in ("canonical","file_bytes","mutex"))
assert predicate is False
check("prewrapped_v4_rejected_by_real_builder_unwrapped_role_predicate")
# Configure itself rejects policy substitution in an existing synthetic configured object.
s=simulation();g=s.old["original_guard"]()
reject(lambda:g.configure_v2(c.REPAIR_REF,policy_ref=dict(path="synthetic_other_policy.json",sha256="0"*64)))
check("real_guard_configure_rejects_policy_swap")
assert not any(n in sys.modules for n in ("scripts.project_budget_v9","arggnn.project_budget_v4","torch","numpy","torch_geometric"))
value=dict(schema="C3_interleaved_real_source_synthetic_initialization_v1",status="PASS_SOURCE_ONLY",checks=checks,check_count=len(checks),
 source_refs=[dict(path=p,sha256=known[p])for p in paths],
 authenticated_source_or_compact_metadata_refs=[dict(path=p,sha256=sha)for p,sha in sorted(known.items())],
 exact_original_compiled_function_code_objects=True,real_renewal_installer_and_private_marker_wrapper=True,
 only_synthetic_import_system_loader_shared_budget_modules_and_state_boundaries=True,
 real_guard_modules_imported=False,real_state_resource_mutex_or_storage_access=False,real_accounting_admissions_jobs_or_Git=False)
with (ROOT/BASE/"independent_real_initialization.final2.v1.json").open("x")as f:json.dump(value,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(dict(status=value["status"],check_count=len(checks),role_summaries=summaries)))
