"""Narrow seal metadata join checks. Future native-close documents exist only in RAM."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/FLARE_VAL001_seal_ROOT_preparation_source_20261017_v1"
B=ROOT/"research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
checks=[];actual_refs={};records={}
def check(name,ok=True):assert ok,name;checks.append(name)
def nodes(s,names):
 result=[copy.deepcopy(n)for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert {n.name for n in result}==set(names);return result
def install(s,names,ns):exec(compile(ast.Module(body=nodes(s,names),type_ignores=[]),"<exact source metadata functions only>","exec"),ns)
def key(r):return r["path"],r["sha256"]
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 assert p.suffix in(".json",".py",".md",".txt",".patch",".sh")and not p.is_symlink()and p.is_file()and p.stat().st_size<=2**20
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):
 if key(r)in records:return copy.deepcopy(records[key(r)])
 assert Path(r["path"]).suffix==".json"and ref(r["path"])==r
 actual_refs[r["path"]]=r;return json.loads((ROOT/r["path"]).read_text())
def raw(r):assert ref(r["path"])==r;return (ROOT/r["path"]).read_bytes()
def fake(label):return dict(path="RAM_ONLY_NOT_ACTUAL/"+label+".json",sha256=hashlib.sha256(label.encode()).hexdigest())
s=(D/"prepare_ROOT_v1.py").read_text();tree=ast.parse(s)
ns=dict(ROOT=ROOT,Path=Path,datetime=datetime,timezone=timezone,copy=copy,hashlib=hashlib,json=json)
for n in tree.body:
 if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"<constants only>","exec"),ns)
install(s,("require","validate_native_close_v1","construct_metadata_v1"),ns)
pins=ns["PINS"];own=ref(D/"prepare_ROOT_v1.py")
# Exact original context/config predicates; payload opener is deliberately unavailable.
p=ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py"
core=types.ModuleType("_RAM_original_completion_metadata");basis=read(ref(p.parent/"VAL000_basis.v1.json"))
core.__dict__.update(HERE=p.parent,ROOT=ROOT,copy=copy,json=json,MISSING=tuple("american_c3_val_%03d"%i for i in range(1,5)),require=ns["require"],OPTIONS=basis["configurations"]["native_config"]["native_options"],COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",CLASSES=("EUR","EAS","AFR"),_tools=lambda:(read,lambda *a:(_ for _ in()).throw(AssertionError("No payload access")),ref))
install(p.read_text(),("_local","_local_ref","_sources","_context","_gzip_bound","_conversion_bound","_native_output_bound","inactive_stage_configs_v1"),core.__dict__)
class RamPath(PurePosixPath):
 def is_dir(self):return True
c=types.ModuleType("_RAM_original_contract")
deps=read(ref("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/dependencies.v1.json"))
c.__dict__.update(ROOT=ROOT,HERE=str(B.relative_to(ROOT)),SOURCE=pins["driver"]["path"],DRIVER=pins["driver"]["path"],ENTRY=str(B.relative_to(ROOT))+"/worker_entry_v1.py",MANIFEST=pins["manifest"]["path"],HANDOFF=pins["handoff"]["path"],require=ns["require"],read=read,raw=raw,ref=ref,local=lambda p:RamPath(str(ROOT))/p,current=lambda *a:None,datetime=datetime,timezone=timezone,copy=copy,hashlib=hashlib,json=json,DEPS=deps,STAGES=("native","seal"),VAL=(ns["PARENT"],),completion=lambda:core)
for k in("BASE_RUNTIME","COMPLETION","COMPLETION_REVIEW"):setattr(c,k,deps[k])
contract=(ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/root_contract_v1.py").read_text()
install(contract,("sources","reviewed","scientific_review","stage_caps","stage_evidence","make_config","validate_request"),c.__dict__)
# Execute the exact three-operand bridge solely as functions/AST; no runtime module/guard.
bridge={"ast":ast,"copy":copy}
for n in ast.parse((B/"predecessor_bridge_v1.py").read_text()).body:
 if isinstance(n,(ast.Assign,ast.FunctionDef)):exec(compile(ast.Module(body=[n],type_ignores=[]),"<exact bridge>","exec"),bridge)
orig=c.stage_evidence;node=nodes(contract,("stage_evidence",))[0];derived=bridge["derive_stage_node_v1"](node)
bridged=dict(c.__dict__,V4_DRIVER_REF=bridge["V4_DRIVER_REF"],V4_ENTRY_REF=bridge["V4_ENTRY_REF"],V4_SOURCE_SHA256=read(bridge["V4_MANIFEST_REF"])["source_sha256"])
exec(compile(ast.Module(body=[derived],type_ignores=[]),"<exact predecessor AST>","exec"),bridged)
c.stage_evidence=bridge["build_stage_bridge_v1"](orig,bridged[derived.name],read)
# Source wrapper's selected namespace is an exact Relate64 equality gate, never fallback.
c.bind_selector=lambda r:ns["require"](r==pins["selector"],"Exact existing Relate64 metadata selector")
original_validate=c.validate_request
def validate(r):
 q=read(r);ns["require"](q["phase_budget"]==ns["PHASE_BUDGET"]and q["prior_failure_acceptance_ref"]==pins["prior_failure"],"Exact phase/failure metadata")
 ns["require"](q["storage_policy_selector_ref"]==pins["selector"]and q["stage"]=="seal"and q["parent_id"]==ns["PARENT"],"Exact source wrapper stage and selector")
 return original_validate(r)
c.validate_request=validate
# Genuine native metadata and exact reservation copied; no live-close path is read.
inp=read(pins["native_input"]);reserve=inp["storage"]["reservation"]
close_ref=fake("native_close_receipt");close_outer_ref=fake("native_close_outer11560");accept_ref=fake("native_close_acceptance")
records[key(close_ref)]=dict(kind="concurrent_completed_scope_close_receipt_v1",key=ns["NATIVE_KEY"],artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,request=dict(completed_ref=pins["native_completed"],direct_wait_ref=pins["native_wait"]),reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()))
records[key(close_outer_ref)]=dict(tool_session_id=11560,actual_terminal_consumed=True,actual_exit_code=0,ordinary_close_receipt_ref=close_ref)
records[key(accept_ref)]=dict(schema="ROOT_FLARE_VAL001_native_close_acceptance_v1",root_authorizer="root_agent",accepted=True,native_input_ref=pins["native_input"],actual_wait_ref=pins["native_wait"],completed_ref=pins["native_completed"],native_outer_terminal_ref=pins["native_outer"],reservation_key=ns["NATIVE_KEY"],storage_policy_selector_ref=pins["selector"],original_files_scopes_charges_clocks_and_failure_history_preserved=True,all_native_future_write_resume_relaunch_capabilities_revoked=True,seal_execution_authorized=False,ordinary_close_receipt_ref=close_ref,ordinary_close_outer_terminal_ref=close_outer_ref)
now=datetime(2026,10,17,9,tzinfo=timezone.utc)
instruction=dict(schema="ROOT_FLARE_VAL001_seal_metadata_preparation_v1",root_authorizer="root_agent",active=True,metadata_preparation_only=True,admission_authorized=False,launch_authorized=False,seal_request_metadata_authorized=True,ROOT_reviewed_preparer_source=True,preparation_source_ref=own,stage="seal",parent_id=ns["PARENT"],storage_policy_selector_ref=pins["selector"],native_close_acceptance_ref=accept_ref,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=4)).isoformat(),cpu_ids=[219,220,221,222],**ns["NAMES"])
written={}
def put(path,value):
 assert path.parent==ROOT/ns["NAMES"]["preparation_directory"]
 p=str(path.relative_to(ROOT));assert p not in written
 raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();r=dict(path=p,sha256=hashlib.sha256(raw).hexdigest());records[key(r)]=copy.deepcopy(value);written[p]=r;return r
result=ns["construct_metadata_v1"](instruction,fake("instruction"),c=c,put_metadata=put,own_ref=own,now=now)
q,caps,runtime=c.validate_request(result["request_ref"])
check("full_original_seal_request_predicates_pass_with_RAM_future_close_only",q["native_run_ref"]==pins["native_completed"])
check("seven_metadata_documents_constructed_only_in_RAM",len(written)==7)
check("original_full_sites_focals_markers_and_raw_posterior_seal_config",read(result["expected_config_review_only_ref"])["max_markers"]==26609)
check("exact_original_seal_caps",caps["reservation_bytes"]==40388608 and caps["scientific_seconds"]==3600 and caps["rss_bytes"]==4*2**30)
check("canonical_outer_preserves_original69102_evidence",read(read(q["previous_stage_evidence_ref"])["outer_terminal_ref"])["original_consumed_outer_ref"]==pins["native_outer"])
base=copy.deepcopy(records)
def rejects(label,inst_changes=None,record_change=None):
 records.clear();records.update(copy.deepcopy(base));before=len(written)
 if record_change:
  r,k,v=record_change;records[key(r)][k]=v
 try:ns["construct_metadata_v1"](dict(instruction,**(inst_changes or {})),fake("instruction"),c=c,put_metadata=put,own_ref=own,now=now)
 except (ValueError,KeyError,TypeError):pass
 else:raise AssertionError("Expected rejection: "+label)
 check(label,len(written)==before)
for label,changes in[("missing_native_close_acceptance",dict(native_close_acceptance_ref=None)),("inactive_instruction",dict(active=False)),("admission_authority_rejected",dict(admission_authorized=True)),("stale_Census_or_other_selector_rejected_no_fallback",dict(storage_policy_selector_ref=fake("future_Census"))),("wrong_stage_rejected",dict(stage="native")),("incomplete_CPU_group_rejected",dict(cpu_ids=[1,2,3])),("insufficient_remaining_authority_rejected",dict(expires_at_utc=(now+timedelta(hours=2)).isoformat()))]:rejects(label,changes)
for label,r,k,v in[("planning_acceptance_rejected",accept_ref,"accepted",False),("future_write_authority_rejected",accept_ref,"all_native_future_write_resume_relaunch_capabilities_revoked",False),("unconsumed_close11560_rejected",close_outer_ref,"actual_terminal_consumed",False),("nonzero_close11560_rejected",close_outer_ref,"actual_exit_code",1),("unbound_receipt_rejected",close_outer_ref,"ordinary_close_receipt_ref",fake("other")),("wrong_native_close_key_rejected",close_ref,"key","other")]:rejects(label,record_change=(r,k,v))
forbidden={"guard","admit","execute","activate","apply_reserve_v1","apply_close_completed_v1","_state_locked","operation_draft_v1","bind_selected_context_v1"}
check("source_contains_no_guard_admission_launch_or_state_calls",not any(isinstance(n,ast.Call)and((isinstance(n.func,ast.Attribute)and n.func.attr in forbidden)or(isinstance(n.func,ast.Name)and n.func.id in forbidden))for n in ast.walk(tree)))
check("import_inert_and_only_preparation_mkdir",not any(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)for n in tree.body)and len([n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="mkdir"])==1)
check("no_scientific_or_guard_runtime_imports",not any(n=="torch"or n.startswith(("numpy","arggnn","composer"))for n in sys.modules))
records.clear();records.update(copy.deepcopy(base))
real_close=ref("research/reviews/FLARE_VAL001_native_ROOT_close_20261017_v1/ordinary_close_receipt.root_v1.json")
real_outer=ref("research/reviews/research_execution_ROOT_20261017_v26/FLARE_VAL001_native_ordinary_close_actual_outer_consumed.root_v1.json")
assert real_close["sha256"]=="fc0ba0a0fa888176cbd7fb34f1b4a15b78f0587b82ac29553acaa09b09b0e88b"
assert real_outer["sha256"]=="c501b2b0e953936e8db958816ab992955b66e36e05cef0e72c1f3351c692c323"
records[key(accept_ref)].update(ordinary_close_receipt_ref=real_close,ordinary_close_outer_terminal_ref=real_outer)
ns["validate_native_close_v1"](instruction,c)
check("genuine_ROOT_supplied_close11560_receipt_and_consumed_outer_join_with_RAM_acceptance_only")
result=dict(schema="FLARE_VAL001_seal_preparer_narrow_source_RAM_checks_v1",status="PASS_SOURCE_AST_RAM_ONLY",checks=len(checks),cases=checks,source_ref=own,actual_JSON_metadata_refs=list(actual_refs.values()),future_acceptance_RAM_only=True,real_close11560_ROOT_metadata_refs_verified=True,actual_helper_not_invoked=True,native_close11560_tool_not_polled=True,guard_resource_process_payload_tensor_truth_numerical_admission_launch_or_Git=False)
with (D/"source_checks.result.v2.json").open("x")as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k not in("cases","actual_JSON_metadata_refs")},sort_keys=True))
