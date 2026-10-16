"""Source-only blocker reproduction: real memo install on synthetic module objects."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,json,types,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");sys.path.insert(0,str(ROOT))
BASE="research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v11"
CAND="research/reviews/native_VAL000_full_state_resume_source_20261016_v11"
def require(ok,message):
 if not ok:raise ValueError(message)
def definitions(path,names,namespace):
 tree=ast.parse((ROOT/path).read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 nodes=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=="__future__"]+nodes
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/path),"exec"),namespace)
def module(path,name,names,extra):
 m=types.ModuleType(name);m.__file__=str(ROOT/path);m.__dict__.update(extra)
 compiled=compile((ROOT/path).read_bytes(),str(ROOT/path),"exec",dont_inherit=True,optimize=0)
 for name in names:
  codes=[code for code in compiled.co_consts if isinstance(code,types.CodeType) and code.co_name==name]
  assert len(codes)==1
  setattr(m,name,types.FunctionType(codes[0],m.__dict__,name))
 return m
v4path="src/arggnn/project_budget_v4.py";v9path="scripts/project_budget_v9.py"
v4=module(v4path,"_independent_synthetic_original_v4",{"canonical","file_bytes","mutex","reference","scope_spec"},
 dict(Path=Path,contextlib=contextlib))
budget=module(v9path,"_independent_synthetic_original_v9",{"successor_paths"},dict(v4=v4,Path=Path))
sp=importlib.util.spec_from_file_location("_independent_real_namespace_installer",ROOT/"research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py")
memo=importlib.util.module_from_spec(sp);sp.loader.exec_module(memo)
events=[];handles=[]
def old_factory(q,*,consumer):
 assert consumer is True
 events.append(dict(event="old_factory_before_install",modules={n:getattr(v4,n).__module__ for n in ("canonical","file_bytes","mutex")}))
 handle=memo.install_consumer_v1(v4,budget,consumer_role="read_only_consumer",
  expected_source_path=ROOT/v4path,expected_source_sha256=hashlib.sha256((ROOT/v4path).read_bytes()).hexdigest())
 handles.append(handle)
 events.append(dict(event="old_factory_after_real_memo_install",modules={n:getattr(v4,n).__module__ for n in ("canonical","file_bytes","mutex")}))
 return types.SimpleNamespace(budget=budget)
real=types.SimpleNamespace(budget=budget,configure_v2=lambda *a,**kw:events.append(dict(event="fake_fresh_configure_no_guard_activity")))
renewal=types.SimpleNamespace(install_renewal_v1=lambda r:None)
cn={}
for n in ast.parse((ROOT/CAND/"amendment_contract_v1.py").read_text()).body:
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
  try:cn[n.targets[0].id]=ast.literal_eval(n.value)
  except (ValueError,TypeError):pass
def load(record,name):
 if record==cn["FRESH_GUARD_REF"]:return real
 if name=="_native11_preserved_renewal":return renewal
 raise AssertionError("Unexpected source-only load "+name)
g=dict(cn,require=require,load=load,ROOT=ROOT,RENEWAL={"SYNTHETIC":1},CERT={"SYNTHETIC":2},REPAIR={"SYNTHETIC":3},
 ORIGINAL_STATE=({},{"SYNTHETIC_OLD_REQUEST":1},{}),FRESH_STATE=({}, {}, {}),
 original=types.SimpleNamespace(guard_v1=old_factory),FreshContinuationGuardV11=lambda *a:types.SimpleNamespace())
definitions(CAND+"/amendment_contract_v1.py",{"fresh_guard_runtime_v11","guard_v1"},g)
try:g["guard_v1"]({})
except ValueError as error:
 assert str(error)=="Original unwrapped native14v3 guard"
 failure=type(error).__name__+": "+str(error)
else:raise AssertionError("Dual-consumer construction unexpectedly passed")
assert len(handles)==1 and handles[0].receipt["installed"] is True
assert v4 is budget.v4
for n in ("canonical","file_bytes","mutex"):assert getattr(v4,n).__module__!=v4.__name__
# Independently prove repeated old-consumer construction would fail as well.
try:old_factory({},consumer=True)
except ValueError as error:
 assert "Unwrapped original function identities" in str(error)
 repeated_failure=type(error).__name__+": "+str(error)
else:raise AssertionError("Second memo installation unexpectedly passed")
# Synthetic module cleanup only, no original/imported guard modules were altered.
handles[0].uninstall()
assert all(getattr(v4,n).__module__==v4.__name__ for n in ("canonical","file_bytes","mutex"))
assert not any(n in sys.modules for n in ("torch","numpy","composer","torch_geometric"))
out=dict(schema="native11_dual_guard_memo_blocker_source_reproduction_v1",status="CONFIRMED_SOURCE_BLOCKER",
 candidate_contract_ref=dict(path=CAND+"/amendment_contract_v1.py",sha256=hashlib.sha256((ROOT/CAND/"amendment_contract_v1.py").read_bytes()).hexdigest()),
 events=events,first_failure=failure,repeated_old_factory_failure=repeated_failure,
 real_pinned_consumer_namespace_and_memo_code_used=True,all_wrapped_modules_synthetic=True,
 actual_guard_configure_policy_read_runtime_guard_process_lease_GPU_checkpoint_payload_accounting_admission_job_or_Git=False,
 proposed_repair="Configure both exact guard consumers while shared v4 is unwrapped; install at most one shared memo handle afterward and reuse/cache the retained/fresh read-check guard objects for that interpreter. Do not reconstruct original.guard_v1 inside charged_census after memo installation. Preserve both original-root zero-write and fresh-scope prospective-write checks, original source/caps/policies and no cross-operation memo cache.")
print(json.dumps(out,sort_keys=True,indent=2))
