"""Narrow original-accept/v1-fail/v2-pass deepcopy repair reproduction.

Reuses disclosed independent RAM scaffold and exact source-derived original frames.
No live guard, scheduler mutex, resource, process, accounting or training operation.
"""
from pathlib import Path
from types import ModuleType,FunctionType,CodeType,SimpleNamespace
from contextlib import contextmanager
import ast,copy,hashlib,json,marshal,os,stat,sys,threading,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v2"
review_path="research/reviews/consumer_strict_JSON_parse_memo_independent_20261017_v1/independent_checks_v1.py"
helperraw=(ROOT/review_path).read_bytes()
assert hashlib.sha256(helperraw).hexdigest()=="2c68c4de480efa82556c0f735628de948bccc551d8ac1bed60b46ab61bdc860a"
helper_nodes=[n for n in ast.parse(helperraw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in("need","sha","packed","bounded","inert_module","bare","RAMLock")]
verified={}
exec(compile(ast.Module(body=helper_nodes,type_ignores=[]),str(ROOT/review_path)+"#selected-helper-definitions","exec"),globals())
def ref(path):return dict(path=path,sha256=sha((ROOT/path).read_bytes()))
V1="research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v1/parse_memo_v1.py"
V2="research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v2/parse_memo_v2.py"
assert sha(bounded(V1))=="312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c"
checks=[]
def passed(name,ok=True):need(ok,name);checks.append(name)
def setup(path,depth):
 raw=bounded(path);p=inert_module(path,"_repair_component_"+str(depth)+"_"+("v1"if path==V1 else"v2"),raw)
 v4raw=bounded(p.V4_REF["path"],p.V4_REF["sha256"]);memo_raw=bounded(p.MEMO_REF["path"],p.MEMO_REF["sha256"]);nsraw=bounded(p.NS_REF["path"],p.NS_REF["sha256"])
 v9path="scripts/project_budget_v9.py";v9raw=bounded(v9path,"483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e")
 v4=bare(p.V4_REF["path"],"_repair_RAM_v4_"+str(depth),v4raw,["bound","parse","reference","scope_spec","canonical","file_bytes","mutex"],dict(Path=Path,json=json,MAX_JSON=32*2**20))
 budget=bare(v9path,"_repair_RAM_budget",v9raw,["successor_paths"],dict(v4=v4))
 adapter=ModuleType("_repair_RAM_original_adapter");adapter.__file__=str(ROOT/p.MEMO_REF["path"]);exec(compile(memo_raw,adapter.__file__,"exec",dont_inherit=True,optimize=0),adapter.__dict__);adapter.threading=SimpleNamespace(local=SimpleNamespace,Lock=RAMLock)
 ns=inert_module(p.NS_REF["path"],"_repair_RAM_namespace",nsraw);ns._load_pinned_memo_v1=lambda:adapter
 handle=ns.install_consumer_v1(v4,budget,consumer_role="read_only_consumer",expected_source_path=ROOT/p.V4_REF["path"],expected_source_sha256=p.V4_REF["sha256"])
 guardpath="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py";guardref=ref(guardpath);guardraw=bounded(guardpath,guardref["sha256"])
 policy=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
 selector=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
 guard=bare(guardpath,"_repair_RAM_Census",guardraw,["require","_raw","_json","_completed_json"],dict(ROOT=ROOT,budget=budget,MAX_JSON=65536,MAX_COMPLETED_JSON=2**20,_POLICY_REF=policy,_SELECTOR_REF=selector,_REPAIR_AUTHORITY_REF=None))
 data=("["*depth+"0"+"]"*depth).encode()
 target=dict(path="research/reviews/frozen_nested_value.json",sha256=sha(data))
 frozen=packed(dict(targets={},runtime_authenticated_historical_source_and_metadata_refs=[target]))
 frozenref=dict(path="research/reviews/frozen_provenance_manifest.json",sha256=sha(frozen))
 original_bytes=p._bytes
 def read_source_or_frozen(record):
  if record["path"]==frozenref["path"]:
   p._record(record);need(record==frozenref,"Exact RAM historical manifest ref");return frozen
  return original_bytes(record)
 p._bytes=read_source_or_frozen
 memo=p.StrictJSONParseMemo(handle,v4,budget,guard_source_entries=[(guard,guardref)],frozen_history_manifest_refs=[frozenref],eligible_historical_refs=[target],current_control_refs=[policy,selector])
 provider=dict(raw=data,sha=target["sha256"]);events=[]
 def reader(root,path):
  need(root==ROOT and path==target["path"],"Exact original final RAM reader target")
  events.append("final_uncached_bytes");return provider["raw"],provider["sha"]
 handle.original["file_bytes"]=reader
 def new_state():
  key=handle._key(ROOT,target["path"])[0]
  return dict(raw={key:(ROOT,target["path"],("RAM_identity",),(data,target["sha256"]))},canonical={},entries=0,charge=0,raw_hits=0,raw_misses=0,raw_bytes_missed=0,canonical_hits=0,canonical_misses=0,capacity_bypasses=0,capacity_bypass_bytes=0)
 @contextmanager
 def RAM_old_mutex(root):
  need(root==ROOT and handle._state()is None,"Same original RAM transaction")
  events.append("old_mutex_enter");handle.tls.state=new_state()
  try:
   yield
   need(v4.bound is memo.original_bound and all(getattr(mod,name)is original for mod,name,original,_ in memo.guard_readers),"All original seams restored before old final snapshot")
   events.append("old_final_snapshot_frontier")
  finally:handle.tls.state=None;events.append("old_mutex_exit")
 memo.expected_mutex=RAM_old_mutex;v4.mutex=RAM_old_mutex;memo.install()
 return SimpleNamespace(p=p,v4=v4,guard=guard,memo=memo,handle=handle,target=target,data=data,provider=provider,events=events,constructor_actual=True)
def leaf(value):
 ids=[]
 while type(value)is list:
  need(len(value)==1,"One nested array item");ids.append(id(value));value=value[0]
 return value,ids
def change_leaf(value,new):
 while type(value[0])is list:value=value[0]
 value[0]=new
def disjoint(a,b):
 return not(set(leaf(a)[1])&set(leaf(b)[1]))
need(sys.getrecursionlimit()==1000,"Original independently reproduced default recursion limit1000")
v1=setup(V1,600)
original=v1.memo.original_parse(v1.data)
passed("original_v4_parser_accepts_exact1201B_depth600_SHA",len(v1.data)==1201 and v1.target["sha256"]=="250f8f81de72fe62fb3c3202430524b7a6938c02969b8be425e8b265d6899eea"and leaf(original)[0]==0)
try:
 with v1.v4.mutex(ROOT):
  need(v1.memo._owned(v1.target,v1.data)==v1.target["path"],"Actual constructor and old raw ownership")
  v1.guard._json(v1.target)
except RecursionError as error:
 failure=dict(exception=type(error).__name__,message=str(error),traceback=traceback.format_exc(),actual_component_constructor=True,exact_original_guard_reader=True)
 passed("preserved_v1_actual_component_seam_rejects_original_accepted_deep_JSON")
else:raise AssertionError("Expected v1 copy regression absent")
need(v1.memo.active is None,"V1 failure cleanup preserved")
v1.memo.uninstall()
v2=setup(V2,600)
with v2.v4.mutex(ROOT):
 a=v2.guard._json(v2.target);b=v2.guard._json(v2.target);c=v2.v4.bound(ROOT,v2.target,True)
 passed("v2_original_accept_passes_first_copy_recursion_and_repeated_guard_bound_seams",all(leaf(x)[0]==0 and len(leaf(x)[1])==600 for x in(a,b,c)))
 passed("uncacheable_current_transaction_has_no_retained_parsed_value",not v2.memo.active["parsed"] and v2.target["path"]in v2.memo.active["uncacheable"] and all("parsed"not in x for x in v2.memo.active["uncacheable"].values())and v2.memo.active["parsed_bytes"]==0 and v2.memo.active["raw_bytes"]==1201)
 passed("all_returned_nested_arrays_are_independent",disjoint(a,b)and disjoint(a,c)and disjoint(b,c))
 change_leaf(a,1);passed("mutating_fallback_return_cannot_change_later_values",leaf(b)[0]==leaf(c)[0]==0)
 passed("only_first_failed_copy_attempt_then_original_parser_bypass",v2.memo.active["counts"]["deepcopy_recursion_fallbacks"]==1 and v2.memo.active["counts"]["uncached_parses"]==2)
passed("fallback_retains_final_original_byte_validation_and_old_snapshot_order",v2.events==["old_mutex_enter","final_uncached_bytes","old_final_snapshot_frontier","old_mutex_exit"]and v2.memo.active is None)
with v2.v4.mutex(ROOT):
 v2.guard._json(v2.target);passed("uncacheable_marker_is_cold_at_next_transaction",v2.memo.active["counts"]["deepcopy_recursion_fallbacks"]==1)
try:
 with v2.v4.mutex(ROOT):
  v2.guard._json(v2.target);v2.provider["raw"]=v2.data+b" "
except ValueError:passed("uncacheable_path_final_byte_drift_still_rejects")
else:raise AssertionError("Uncacheable final drift accepted")
passed("final_drift_cleans_marker_and_restores_seams",v2.memo.active is None and v2.v4.bound is v2.memo.original_bound)
v2.memo.uninstall()
# A shallower value first copies successfully. A later deeper caller makes only deepcopy fail.
hit=setup(V2,300)
def deeper(n,fn):
 if n:return deeper(n-1,fn)
 return fn()
passed("original_parser_still_accepts_at_deeper_caller",leaf(deeper(450,lambda:hit.memo.original_parse(hit.data)))[0]==0)
with hit.v4.mutex(ROOT):
 first=hit.guard._json(hit.target)
 passed("first_shallower_copy_is_actually_cached",hit.target["path"]in hit.memo.active["parsed"]and hit.memo.active["counts"]["deepcopy_recursion_fallbacks"]==0)
 second=deeper(450,lambda:hit.guard._json(hit.target))
 third=hit.guard._json(hit.target)
 passed("cache_hit_deepcopy_recursion_evicts_and_returns_fresh_original_parse",not hit.memo.active["parsed"]and hit.target["path"]in hit.memo.active["uncacheable"]and hit.memo.active["counts"]["deepcopy_recursion_fallbacks"]==1 and disjoint(first,second)and disjoint(second,third))
 change_leaf(second,9);passed("cache_hit_fallback_canonical_never_escapes_or_aliases",leaf(first)[0]==leaf(third)[0]==0)
passed("hit_fallback_keeps_final_original_byte_check",hit.events[-3:]==["final_uncached_bytes","old_final_snapshot_frontier","old_mutex_exit"])
hit.memo.uninstall()
# Narrow non-RecursionError behavior uses the actual component seam with a RAM copy provider.
other=setup(V2,3);saved_copy=other.p.copy
def memory_error(_):raise MemoryError("RAM_COPY_MEMORYERROR")
other.p.copy=SimpleNamespace(deepcopy=memory_error)
try:
 with other.v4.mutex(ROOT):other.guard._json(other.target)
except MemoryError:passed("other_deepcopy_exception_propagates_unchanged")
else:raise AssertionError("MemoryError swallowed")
finally:other.p.copy=saved_copy
passed("other_exception_cleanup_preserved",other.memo.active is None and other.v4.bound is other.memo.original_bound)
other.memo.uninstall()
need(sha(bounded(V1))=="312c9318912097d2b4c08e146bef41b1d055d089e1765707603a7c90da3a9d6c","V1 immutable")
tree1=ast.parse(bounded(V1));tree2=ast.parse(bounded(V2))
cls1=next(x for x in tree1.body if isinstance(x,ast.ClassDef)and x.name=="StrictJSONParseMemo")
cls2=next(x for x in tree2.body if isinstance(x,ast.ClassDef)and x.name=="StrictJSONParseMemo")
m1={x.name:x for x in cls1.body if isinstance(x,ast.FunctionDef)};m2={x.name:x for x in cls2.body if isinstance(x,ast.FunctionDef)}
changed=[n for n in m1 if ast.dump(m1[n],include_attributes=False)!=ast.dump(m2[n],include_attributes=False)]
passed("only_parse_fallback_final_loop_and_transaction_state_methods_changed",set(changed)=={"parse_record","_final","_mutex_wrapper"}and set(m2)-set(m1)=={"_mark_copy_uncacheable"})
passed("caps_and_original_ref_parser_control_eligibility_methods_unchanged",all(getattr(v1.p,k)==getattr(v2.p,k)for k in("MAX_ENTRIES","MAX_RAW_BYTES","MAX_PARSED_BYTES","V4_REF","MEMO_REF","NS_REF","FORBIDDEN_WORDS")))
result=dict(schema="consumer_strict_JSON_parse_memo_v2_narrow_repair_checks_v1",status="AUTHOR_NARROW_SOURCE_CHECKS_PASS",checks=checks,count=len(checks),recursion_limit=1000,deep_fixture_depth=600,deep_fixture_bytes=1201,deep_fixture_sha256=v1.target["sha256"],original_accept_v1_fail_v2_pass=True,v1_failure=failure,cache_hit_later_copy_failure_test=True,selected_independent_scaffold_ref=ref(review_path),actual_component_constructor_and_original_source_frames=True,original_parser_executed=True,synthetic_valid_JSON_and_frozen_provenance_are_RAM=True,original_mutex_IO_snapshot_frontier_are_RAM=True,actual_guard_state_resource_process_lock_accounting_or_GPU_operations=False,actual_live_adoption=False,clock_reads_or_speedup_claims=False,broad_author_campaign_repeated=False,changed_class_methods=changed,v1_source_preserved=True)
with(OUT/"narrow_checks.result.v2.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:result[k]for k in("status","count","original_accept_v1_fail_v2_pass","actual_live_adoption")},sort_keys=True))
