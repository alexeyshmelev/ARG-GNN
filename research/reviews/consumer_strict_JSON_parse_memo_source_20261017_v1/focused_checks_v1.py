"""Focused source/metadata tests. No original guard, mutex or resource operation."""
from pathlib import Path
from types import ModuleType,FunctionType,CodeType
from contextlib import contextmanager
import ast,copy,hashlib,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v1"
def load_inert(path,name):
 raw=(ROOT/path).read_bytes();m=ModuleType(name);m.__file__=str(ROOT/path);sys.modules[name]=m
 exec(compile(raw,m.__file__,"exec",dont_inherit=True,optimize=0),m.__dict__)
 return m
p=load_inert(str((HERE/"parse_memo_v1.py").relative_to(ROOT)),"_strict_json_component_source_tests")
def ref(path):return dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
def bare_module(path,name,names,constants):
 raw=(ROOT/path).read_bytes();m=ModuleType(name);m.__file__=str(ROOT/path);m.__dict__.update(constants)
 for n in names:
  code=p._code(raw,path,n)
  # Defaults are literals in these exact original functions.
  node=next(x for x in ast.parse(raw).body if isinstance(x,ast.FunctionDef)and x.name==n)
  defaults=tuple(ast.literal_eval(x)for x in node.args.defaults)or None
  fn=FunctionType(code,m.__dict__,n,defaults);setattr(m,n,fn)
 return m
v4=bare_module(p.V4_REF["path"],"_RAM_v4_pure_source_frames",
 ["bound","parse","reference","scope_spec","canonical","file_bytes","mutex"],
 dict(Path=Path,json=json,MAX_JSON=32*2**20))
budget_path="scripts/project_budget_v9.py"
budget=bare_module(budget_path,"_RAM_budget_original_source_frame",["successor_paths"],dict(v4=v4))
ns=load_inert(p.NS_REF["path"],"_strict_JSON_original_namespace_source")
handle=ns.install_consumer_v1(v4,budget,consumer_role="read_only_consumer",expected_source_path=ROOT/p.V4_REF["path"],expected_source_sha256=p.V4_REF["sha256"])
guard_path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
policy=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json",sha256="5e82c0895c43b77d7e69d440fa9b09dbe19f7f2565b3ae9f866d0075e6f9fad5")
selector=dict(path="research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json",sha256="fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4")
guard=bare_module(guard_path,"_RAM_Census_original_pure_reader_frames",
 ["require","_raw","_json","_completed_json"],dict(ROOT=ROOT,budget=budget,MAX_JSON=65536,MAX_COMPLETED_JSON=2**20,_POLICY_REF=policy,_SELECTOR_REF=selector,_REPAIR_AUTHORITY_REF=None))
manifest_path="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json"
manifest=v4.parse((ROOT/manifest_path).read_bytes())
eligible=manifest["targets"]["census_tsinfer_observation_v1/census-tsinfer-inspect-v1"]["storage_ref"]
raw=(ROOT/eligible["path"]).read_bytes()
assert hashlib.sha256(raw).hexdigest()==eligible["sha256"]
baseline=v4.parse(raw)
kwargs=dict(guard_source_entries=[(guard,ref(guard_path))],frozen_history_manifest_refs=[ref(manifest_path)],
 eligible_historical_refs=[eligible],current_control_refs=[policy,selector])
memo=p.StrictJSONParseMemo(handle,v4,budget,**kwargs)
assert memo.original_parse.__code__==p._code((ROOT/p.V4_REF["path"]).read_bytes(),p.V4_REF["path"],"parse")
checks=[]
def passed(name):checks.append(name)
def rejects(name,fn):
 try:fn()
 except (ValueError,RuntimeError):passed(name)
 else:raise AssertionError("Did not reject "+name)
for label,bad in [("duplicate_JSON_key",b'{"x":1,"x":2}'),("nonfinite_JSON_NaN",b'{"x":NaN}'),("nonfinite_JSON_Infinity",b'{"x":Infinity}')]:
 rejects(label,lambda bad=bad:memo.original_parse(bad))
passed("constructor_derives_actual_original_parser_and_exact_reader_shapes")
# Below this point the original lock/IO provider is replaced only in RAM.
# No original guard or resource function is called.
provider={"raw":raw,"sha":eligible["sha256"]}
final_reads=[]
def memory_reader(root,path):
 assert root==ROOT and path==eligible["path"];final_reads.append(path);return provider["raw"],provider["sha"]
handle.original["file_bytes"]=memory_reader
def seed():
 key=handle._key(ROOT,eligible["path"])[0]
 return dict(raw={key:(ROOT,eligible["path"],("RAM_IDENTITY",),(raw,eligible["sha256"]))},canonical={},entries=0,charge=0,raw_hits=0,raw_misses=0,raw_bytes_missed=0,canonical_hits=0,canonical_misses=0,capacity_bypasses=0,capacity_bypass_bytes=0)
state_box={"state":seed()}
@contextmanager
def RAM_original_mutex(root):
 assert root==ROOT
 assert handle._state()is None
 handle.tls.state=state_box["state"]
 try:
  yield
  # Real original memo snapshot is not executed. Check only source-derived restoration frontier.
  assert v4.bound is memo.original_bound
  assert guard._json is next(x[2]for x in memo.guard_readers if x[1]=="_json")
 finally:
  handle.tls.state=None
memo.expected_mutex=RAM_original_mutex;v4.mutex=RAM_original_mutex;memo.install()
with v4.mutex(ROOT):
 a=guard._json(eligible);assert a==baseline
 a["reservation"]["consumer_bindings"][0]["path"]="MUTATED_TEST_COPY"
 b=guard._json(eligible);assert b==baseline and b is not a
 c=v4.bound(ROOT,eligible,True);assert c==baseline and c is not b
 assert memo.active["counts"]["parse_misses"]==1 and memo.active["counts"]["parse_hits"]==2
passed("genuine_historical_JSON_original_parse_then_two_cache_hits_with_isolated_copies")
assert memo.active is None and len(final_reads)==1
passed("success_restores_exact_original_JSON_ref_functions_and_final_uncached_byte_check")
with v4.mutex(ROOT):
 assert guard._json(eligible)==baseline
 assert memo.active["counts"]["parse_misses"]==1
passed("next_original_transaction_starts_cold")
with v4.mutex(ROOT):
 memo.controls.add(eligible["path"])
 before=memo.active["counts"]["uncached_parses"]
 assert guard._json(eligible)==baseline
 assert memo.active["counts"]["uncached_parses"]==before+1 and not memo.active["parsed"]
 memo.controls.remove(eligible["path"])
passed("explicit_current_control_path_uses_original_parser_without_cache")
with v4.mutex(ROOT):
 badstate=handle._state()["raw"];entry=next(iter(badstate.values()))
 badstate[next(iter(badstate))]=(entry[0],entry[1],entry[2],(bytes(bytearray(raw)),eligible["sha256"]))
 rejects("raw_object_ownership_tamper",lambda:memo.parse_record(eligible,raw))
# Replace state after deliberately malformed RAM tuple.
state_box["state"]=seed()
with v4.mutex(ROOT):
 bad=dict(eligible,sha256="0"*64)
 rejects("original_bound_rejects_conflicting_expected_SHA",lambda:v4.bound(ROOT,bad,True))
passed("reference_SHA_rejection_unchanged")
try:
 with v4.mutex(ROOT):
  guard._json(eligible);raise RuntimeError("RAM body failure")
except RuntimeError:pass
assert memo.active is None and v4.bound is memo.original_bound
passed("body_failure_restores_functions_and_destroys_parse_state")
try:
 with v4.mutex(ROOT):
  guard._json(eligible);provider["raw"]=raw+b" "
except ValueError:passed("final_original_byte_tamper_blocks_return")
else:raise AssertionError("tamper return")
provider["raw"]=raw
assert memo.active is None and v4.bound is memo.original_bound
passed("final_validation_failure_restores_functions_and_destroys_state")
memo.uninstall();assert v4.mutex is RAM_original_mutex
passed("uninstall_only_without_active_operation")
# Constructor still sees exact original installed handle wrapper, not test provider mutex.
v4.mutex=handle.mutex_wrapper
rejects("historical_ref_in_current_controls_blocks_constructor",lambda:p.StrictJSONParseMemo(handle,v4,budget,**dict(kwargs,current_control_refs=[policy,selector,eligible])))
wrong=dict(eligible,sha256="0"*64)
rejects("historical_manifest_SHA_mismatch_blocks_constructor",lambda:p.StrictJSONParseMemo(handle,v4,budget,**dict(kwargs,eligible_historical_refs=[wrong])))
rejects("arbitrary_JSON_without_frozen_historical_provenance_rejected",lambda:p.StrictJSONParseMemo(handle,v4,budget,**dict(kwargs,eligible_historical_refs=[dict(path="research/reviews/arbitrary.json",sha256="0"*64)])))
result=dict(schema="consumer_strict_JSON_source_focused_checks_v1",source_only=True,
 checks=checks,count=len(checks),original_parser_source_ref=p.V4_REF,genuine_historical_input_ref=eligible,
 component_constructor_source_auth_and_original_shape_validation=True,
 real_guard_import_or_configuration=False,real_guard_state_mutex_resource_process_or_accounting_operations=False,
 original_namespace_memo_handle_installed_only_on_RAM_module=True,
 actual_lock_and_io_final_snapshot_replaced_with_RAM_providers=True,
 original_parser_executed=True,genuine_eligible_metadata_bytes_used=True,
 no_actual_guard_or_training_speed_claim=True,full_actual_future_consumer_route_qualified=False)
with(HERE/"focused_checks.result.v1.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(result,sort_keys=True))
