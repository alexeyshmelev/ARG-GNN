"""Only source and bounded JSON metadata plus RAM constructed documents. Never invoke preparer."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import ast,copy,hashlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");D=ROOT/"research/reviews/FLARE_VAL001_native_ROOT_preparation_source_20261017_v1"
C=ROOT/"research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v2"
checks=[]
def check(name,ok=True):assert ok,name;checks.append(dict(name=name,status="PASS"))
def rejects(name,action):
    try:action()
    except (ValueError,KeyError,TypeError,AssertionError):check(name)
    else:raise AssertionError("Expected fail-closed: "+name)
def ref(p):
    p=Path(p);p=p if p.is_absolute()else ROOT/p
    assert p.suffix in(".json",".py",".md",".txt",".patch",".sh")and p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def jread(p):return json.loads((p if isinstance(p,Path)else ROOT/p).read_text())
actual_reads={}
def actual(r):
    assert Path(r["path"]).suffix==".json"and ref(r["path"])==r
    actual_reads[r["path"]]=r;return jread(r["path"])
def node(text,name):return copy.deepcopy(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name))
text=(D/"prepare_ROOT_v1.py").read_text();tree=ast.parse(text);ns=dict(ROOT=ROOT,Path=Path,datetime=datetime,timezone=timezone,copy=copy)
for n in tree.body:
    if isinstance(n,ast.Assign):exec(compile(ast.Module(body=[n],type_ignores=[]),"<preparer_source_constants_only>","exec"),ns)
for name in("require","construct_metadata_v1"):
    exec(compile(ast.Module(body=[node(text,name)],type_ignores=[]),"<exact_metadata_constructor_only_no_preparer_invocation>","exec"),ns)
check("prepare_v1_and_main_never_executed","prepare_v1"not in ns and "main"not in ns)
# Existing wrapper/contract factory and predecessor gate run only with explicit RAM adapters.
candidate=(C/"integration_check_v2.py").read_text();prefix=candidate.split('result=dict(schema="FLARE_predecessor_bridge_v2_actual_metadata_RAM_check"',1)[0];assert prefix!=candidate
fixtures=dict(__name__="_exact_existing_metadata_path_RAM_only");exec(compile(prefix,"<existing_metadata_tests_no_result_writer>","exec"),fixtures)
check("existing49source_RAM_cases_reproduced_without_writes",len(fixtures["cases"])==49)
bridge=fixtures["bridge"];key=fixtures["key"]
core_path=ROOT/"research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py";core_text=core_path.read_text()
core=types.ModuleType("_preparer_RAM_original_context_and_config")
core.__dict__.update(HERE=core_path.parent,ROOT=ROOT,copy=copy,json=json,MISSING=tuple("american_c3_val_%03d"%i for i in range(1,5)),require=bridge.require,OPTIONS=fixtures["basis"]["configurations"]["native_config"]["native_options"],COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1",CLASSES=("EUR","EAS","AFR"),_tools=lambda:(actual,lambda *a:(_ for _ in()).throw(AssertionError("No payloads")),ref))
for name in("_local","_local_ref","_sources","_context","_gzip_bound","_conversion_bound","_native_output_bound","inactive_stage_configs_v1"):
    exec(compile(ast.Module(body=[node(core_text,name)],type_ignores=[]),"<original_source_only_context_config>","exec"),core.__dict__)
# Exact genuine completed-conversion certificate chain, read JSON only.
pins=ns["PINS"];predecessor=actual(pins["predecessor"]);accept=actual(predecessor["ordinary_close_terminal_acceptance_ref"])
chain=[pins["predecessor"],predecessor["input_ref"],predecessor["actual_wait_ref"],predecessor["completion_ref"],predecessor["outer_terminal_ref"],predecessor["ordinary_close_receipt_ref"],predecessor["ordinary_close_terminal_acceptance_ref"],accept["ordinary_close_outer_terminal_ref"],accept["final_original_journal_acceptance_ref"],pins["review"],pins["prior_failure"]]
base_records={key(r):actual(r)for r in chain}
check("genuine74c2_predecessor_and46bf_final_certificate_join",pins["predecessor"]["sha256"]=="74c2d2c7d2597356ec764fa2196ba3c3a5597e12478a60b0368b55ad9cb2ae7e"and accept["final_original_journal_acceptance_ref"]==pins["final_certificate"])
core._context(pins["context"],parent_id=ns["PARENT"]);check("actual_original_context_reuse_is_supported_by_full_original_validator")
now=datetime(2026,10,17,8,10,tzinfo=timezone.utc)
terminal_ref=dict(path="RAM_ONLY_NOT_ACTUAL/consumed72068_terminal.json",sha256="7"*64)
terminal=dict(tool_session_id=72068,actual_terminal_consumed=True,actual_exit_code=0)
own_ref=ref(D/"prepare_ROOT_v1.py")
instruction=dict(schema="ROOT_FLARE_VAL001_native_metadata_preparation_v1",root_authorizer="root_agent",active=True,metadata_preparation_only=True,admission_authorized=False,launch_authorized=False,native_request_metadata_authorized=True,ROOT_reviewed_preparer_source=True,preparation_source_ref=own_ref,stage="native",parent_id=ns["PARENT"],storage_policy_selector_ref=pins["selector"],H058_admission72068_actual_terminal_ref=terminal_ref,created_at_utc=now.isoformat(),expires_at_utc=(now+timedelta(hours=8)).isoformat(),cpu_ids=[219,220,221,222],**ns["NAMES"])
instruction_ref=dict(path="RAM_ONLY_NOT_ACTUAL/ROOT_instruction.json",sha256="8"*64)
def fixture():
    c,records,unused,_,_=fixtures["fixture"]();records.update(copy.deepcopy(base_records));records[key(terminal_ref)]=copy.deepcopy(terminal)
    c.completion=lambda:core
    old_read=c.read
    def read(r):return copy.deepcopy(records[key(r)])if key(r)in records else actual(r)
    c.read=read;written={}
    def put(path,value):
        assert path.parent==ROOT/ns["NAMES"]["preparation_directory"]
        name=str(path.relative_to(ROOT));assert name not in written
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode();r=dict(path=name,sha256=hashlib.sha256(raw).hexdigest())
        records[key(r)]=copy.deepcopy(value);written[name]=(r,copy.deepcopy(value));return r
    return c,records,written,put
c,records,written,put=fixture()
result=ns["construct_metadata_v1"](copy.deepcopy(instruction),instruction_ref,c=c,put_metadata=put,own_ref=own_ref,now=now)
q,caps,runtime=c.validate_request(result["request_ref"])
check("full_written_metadata_graph_passes_exact_original_request_validator",q["stage"]=="native"and q["conversion_ref"]==bridge.COMPLETED_REF and q["previous_stage_evidence_ref"]==pins["predecessor"])
check("all5constructed_files_live_only_in_RAM_staging",len(written)==5 and all(p.startswith(ns["NAMES"]["preparation_directory"]+"/")for p in written))
check("registration_uses_original_binder_schema_and_same_count_context",records[key(result["registration_ref"])]["schema"]=="FLARE_root_target_registration_v1"and records[key(result["registration_ref"])]["count_manifest_ref"]==core._context(pins["context"],parent_id=ns["PARENT"])["targets"][ns["PARENT"]]["count_manifest_ref"])
check("native_caps_full_science_and_original_phase_budget_exact",caps["reservation_bytes"]==2005743919 and caps["output_bytes"]==1997355311 and caps["cpu_threads"]==4 and caps["rss_bytes"]==24*2**30 and q["phase_budget"]==dict(setup_seconds=7200,scientific_seconds=14400,wall_seconds=15000,combined_seconds=22200,evidence_reserve_seconds=300,authority_margin_seconds=600))
check("no_fabricated_admitted_input_or_execution_argv",result["admission_or_launch_performed"]is False and not any("/input.json"in p or "/admitted.json"in p for p in written)and "execution_argv"not in result)
for label,changes in(("ROOT_review_absent",dict(ROOT_reviewed_preparer_source=False)),("metadata_instruction_inactive",dict(active=False)),("admission_instruction",dict(admission_authorized=True)),("wrong_selector",dict(storage_policy_selector_ref=instruction_ref)),("reconversion",dict(stage="convert")),("wrong_namespace",dict(packet_directory="research/reviews/other")),("badCPUcount",dict(cpu_ids=[1,2,3])),("short_authority",dict(expires_at_utc=(now+timedelta(hours=6)).isoformat())),("extended_authority",dict(expires_at_utc=(now+timedelta(hours=9)).isoformat()))):
    cc,rr,ww,pp=fixture();rejects("constructor_rejects_"+label,lambda changes=changes:ns["construct_metadata_v1"](dict(instruction,**changes),instruction_ref,c=cc,put_metadata=pp,own_ref=own_ref,now=now));check("rejection_before_any_RAM_metadata_write_"+label,ww=={})
cc,rr,ww,pp=fixture();rr[key(terminal_ref)]["actual_terminal_consumed"]=False
rejects("unconsumed72068_terminal_blocks_all_metadata_writes",lambda:ns["construct_metadata_v1"](instruction,instruction_ref,c=cc,put_metadata=pp,own_ref=own_ref,now=now));check("no_RAM_writes_before_consumed72068",ww=={})
cc,rr,ww,pp=fixture();rr[key(pins["final_certificate"])]["final_original_journal_accepted"]=False
rejects("invalid_final_journal_certificate_blocks",lambda:ns["construct_metadata_v1"](instruction,instruction_ref,c=cc,put_metadata=pp,own_ref=own_ref,now=now));check("no_RAM_writes_without_genuine_final_acceptance",ww=={})
# Inspect actual prepare path without calling it: no resource/admission/launch operations.
prepare=node(text,"prepare_v1")
forbidden={"guard","admit","execute","apply_reserve_v1","apply_close_completed_v1","_state_locked","operation_draft_v1","bind_selected_context_v1"}
check("preparer_has_no_guard_admission_resource_or_converter_calls",not any(isinstance(n,ast.Call)and((isinstance(n.func,ast.Attribute)and n.func.attr in forbidden)or(isinstance(n.func,ast.Name)and n.func.id in forbidden))for n in ast.walk(tree)))
check("actual_prepare_keeps_targets_absent_and_never_makes_reserved_paths",ast.unparse(prepare).count("prep.mkdir(")==1 and not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="mkdir"and ast.unparse(n.func.value)!="prep"for n in ast.walk(prepare)))
check("source_import_remains_inert",not any(isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)for n in tree.body))
check("no_real_scientific_or_guard_runtime_import",not any(n=="torch"or n.startswith(("numpy","arggnn","composer"))for n in sys.modules))
final=dict(schema="FLARE_VAL001_ROOT_preparer_source_AST_RAM_checks_v1",status="PASS_SOURCE_AST_RAM_ONLY",checks=len(checks)+49,preparer_checks=len(checks),existing_candidate_checks=49,cases=checks,actual_JSON_metadata_refs=list(actual_reads.values()),constructed_request=result["request_ref"],constructed_graph_RAM_only=True,helper_prepare_and_main_never_invoked=True,guard_resource_process_payload_tensor_truth_numerical_admission_launch_or_Git=False,source_ref=own_ref)
with (D/"source_checks.result.v1.json").open("x")as f:json.dump(final,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps({k:v for k,v in final.items()if k not in("cases","actual_JSON_metadata_refs")},sort_keys=True))
