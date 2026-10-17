"""Source/AST/RAM checks only. No resource guard, real receipt, payload or process access."""
from pathlib import Path
import ast,copy,hashlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v1"
OUT=ROOT/HERE
PINS={
"contract":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v1/root_contract_v1.py","301d36326a34e4ee23717cb5d033dee5a11cb836b679088a47b4eaa7120422e3"),
"engine":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/root_driver_v1.py","c1d6a5bb835800f30d0c1e487c885ede13db253c18286b20bc115d8d70a7447f"),
"worker":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2/worker_entry_v1.py","13f5d7a124fd188e6c48a17bab4c0c6389f98abed9ec0fcfff6caa1d8228e9b6"),
"v4driver":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/root_driver_v1.py","bcd7eba4f02dd7425a4d43ded8df71402e83cb5ab297886172a26a1d6d395232"),
"v4entry":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/worker_entry_v1.py","cc6a9b5b98034985876a88ecff6c0da6d0b16f2d23ed9195b11b4a0c0a78751b"),
"v4manifest":("research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v4/source_manifest.v1.json","3365dcad1e1398c94ded1ab81b3bc6cd75ca096d9462bcaf59fa81800fef7865"),
"auth":("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py","2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98"),
"converter":("src/arggnn/american_c3_flare_convert_v1.py","6e55e295226a183548a23258e7230855842bb876b4d2a90875d8f2ad53181ff5"),
"native":("src/arggnn/american_c3_flare_run_v2.py","50c48a25834f5b4246f1a28deb2ff564e05e4db2308203578448d966d5273b21"),
"seal":("src/arggnn/american_c3_flare_evaluate_v1.py","ce7567d0f71054cfd96c9a44727e20501ad6972e4af7340de7ccc231a9e8aed7")}
src={}
for key,(path,sha)in PINS.items():
    raw=(ROOT/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==sha,("Changed exact source",path)
    src[key]=raw.decode()
bridge_source=(OUT/"predecessor_bridge_v1.py").read_text()
ns={"__name__":"_RAM_only_FLARE_bridge"}
exec(compile(bridge_source,"<RAM_bridge_source>","exec"),ns)
require=ns["require"];derive=ns["derive_stage_node_v1"];build=ns["build_stage_bridge_v1"]
old=next(n for n in ast.parse(src["contract"]).body if isinstance(n,ast.FunctionDef)and n.name=="stage_evidence")
new=derive(old)
def same(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
# Reverse precisely the three permitted changes and compare the entire original AST.
reverse={"V4_DRIVER_REF":ast.parse("ref(DRIVER)",mode="eval").body,
"V4_ENTRY_REF":ast.parse("ref(ENTRY)",mode="eval").body,
"V4_SOURCE_SHA256":ast.parse("sources()['source_sha256']",mode="eval").body}
class Reverse(ast.NodeTransformer):
    def visit_Name(self,n):return copy.deepcopy(reverse[n.id])if n.id in reverse else n
reversed_node=Reverse().visit(copy.deepcopy(new));reversed_node.name=old.name
assert same(old,reversed_node)
cases=[dict(name="Exact original stage_evidence AST differs only in three predecessor-source operands",status="PASS")]
def passed(name):cases.append(dict(name=name,status="PASS"))
for key in ("worker_entry_v1.py","prepare_context_v1.py"):
    assert (OUT/key).read_bytes()==(ROOT/"research/reviews/FLARE_original_remaining_CPU_execution_source_20261017_v2"/key).read_bytes()
passed("Original v2 worker and context preparation preserved byte-for-byte")
driver=(OUT/"root_driver_v1.py").read_text()
assert 'c.STAGES=("native","seal");c.VAL=(bridge.PARENT,)'in driver
assert 'and q.get("conversion_ref")==bridge.COMPLETED_REF'in driver and 'q.get("storage_policy_selector_ref")==bridge.SELECTOR_REF'in driver
passed("Wrapper restricted to VAL001 native/seal, exact conversion and current selector")
assert 'install_stage_bridge_v1(c,auth)'in driver and '_load_exact_source(ENGINE_REF' in driver
assert 'functions_from_source(c,CONTRACT_REF,("stage_evidence",))'in bridge_source
passed("Exact old source/hash/callable authentication precedes bridge installation")
assert 'return original(record,stage=stage,parent=parent,completion_ref=completion_ref)'in bridge_source
passed("Every nonconversion predecessor delegates unchanged original metadata validator")
old_manifest={"original_v4_code.py":"a"*64};new_manifest={"new_bridge_code.py":"b"*64}
CURRENT_DRIVER={"path":HERE+"/root_driver_v1.py","sha256":"c"*64}
CURRENT_ENTRY={"path":HERE+"/worker_entry_v1.py","sha256":"d"*64}
E={"path":"RAM/stage_evidence.json","sha256":"e"*64}
OUTER={"path":"RAM/conversion_outer0.json","sha256":"f"*64}
CLOSE={"path":"RAM/ordinary_close.json","sha256":"1"*64}
ACCEPT={"path":"RAM/close_acceptance.json","sha256":"2"*64}
CLOSE_OUTER={"path":"RAM/close_outer0.json","sha256":"3"*64}
FINAL={"path":"RAM/final_original_journal.json","sha256":"4"*64}
def key(r):return (r["path"],r["sha256"])
def base_records():
    reserve={"key":ns["KEY"],"output":str(ROOT/ns["COMPLETED_REF"]["path"]).rsplit("/",1)[0],"reserved_bytes":889641408}
    e=dict(schema="FLARE_remaining_stage_ROOT_terminal_evidence_v1",root_authorizer="root_agent",stage="convert",parent_id=ns["PARENT"],completion_ref=ns["COMPLETED_REF"],input_ref=ns["INPUT_REF"],actual_wait_ref=ns["WAIT_REF"],outer_terminal_ref=OUTER,ordinary_close_receipt_ref=CLOSE,ordinary_close_terminal_acceptance_ref=ACCEPT)
    wait=dict(actual_direct_worker_wait=True,returncode=0,all_owned_processes_and_groups_drained=True,operational_failure=None,input_ref=ns["INPUT_REF"],completed_ref=ns["COMPLETED_REF"])
    outer=dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=ns["COMPLETED_REF"],direct_wait_ref=ns["WAIT_REF"])
    inp=dict(schema="FLARE_remaining_CPU_admitted_input_v1",parent_id=ns["PARENT"],stage="convert",root_driver_ref=ns["V4_DRIVER_REF"],main_source_ref=ns["V4_ENTRY_REF"],source_sha256=old_manifest,output_directory=str(Path(ns["COMPLETED_REF"]["path"]).parent),storage={"reservation":reserve})
    closed=dict(kind="concurrent_completed_scope_close_receipt_v1",key=ns["KEY"],artifacts_deleted=0,value=dict(future_writes_and_relaunch_revoked=True,request=dict(completed_ref=ns["COMPLETED_REF"],direct_wait_ref=ns["WAIT_REF"]),reservation_sha256=hashlib.sha256(json.dumps(reserve,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()))
    accept=dict(schema="ROOT_FLARE_exact_v4_conversion_close_acceptance_v1",root_authorizer="root_agent",accepted=True,input_ref=ns["INPUT_REF"],actual_wait_ref=ns["WAIT_REF"],completed_ref=ns["COMPLETED_REF"],conversion_outer_terminal_ref=OUTER,ordinary_close_receipt_ref=CLOSE,reservation_key=ns["KEY"],original_files_scopes_charges_clocks_and_failure_history_preserved=True,all_conversion_future_write_resume_relaunch_capabilities_revoked=True,native_or_seal_execution_authorized=False,ordinary_close_outer_terminal_ref=CLOSE_OUTER,final_original_journal_acceptance_ref=FINAL)
    close_outer=dict(actual_terminal_consumed=True,actual_exit_code=0,ordinary_close_receipt_ref=CLOSE)
    final=dict(root_authorizer="root_agent",final_original_journal_accepted=True,ordinary_close_receipt_ref=CLOSE,reservation_key=ns["KEY"],storage_policy_selector_ref=ns["SELECTOR_REF"])
    return {key(r):v for r,v in ((E,e),(ns["WAIT_REF"],wait),(OUTER,outer),(ns["INPUT_REF"],inp),(CLOSE,closed),(ACCEPT,accept),(CLOSE_OUTER,close_outer),(FINAL,final))}
def funcs(records):
    read=lambda r:copy.deepcopy(records[key(r)])
    scope=dict(ROOT=ROOT,require=require,read=read,local=lambda p:ROOT/p,hashlib=hashlib,json=json,
        DRIVER=CURRENT_DRIVER["path"],ENTRY=CURRENT_ENTRY["path"],
        ref=lambda p:CURRENT_DRIVER if p==CURRENT_DRIVER["path"]else CURRENT_ENTRY,
        sources=lambda:{"source_sha256":new_manifest},V4_DRIVER_REF=ns["V4_DRIVER_REF"],V4_ENTRY_REF=ns["V4_ENTRY_REF"],V4_SOURCE_SHA256=old_manifest)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(old),copy.deepcopy(new)],type_ignores=[])),"<original_metadata_AST_RAM_only>","exec"),scope)
    return scope["stage_evidence"],scope[new.name],read
def run(records):
    original,derived,read=funcs(records)
    return build(original,derived,read)(E,stage="convert",parent=ns["PARENT"],completion_ref=ns["COMPLETED_REF"])
records=base_records();original,_,_=funcs(records)
try:original(E,stage="convert",parent=ns["PARENT"],completion_ref=ns["COMPLETED_REF"])
except ValueError:passed("B1 reproduced: original current-source validator rejects genuine different v4 predecessor")
else:raise AssertionError("Original accepted different source")
assert run(records)==records[key(E)]
passed("Exact pinned v4 conversion accepted only with worker0/drain, conversionouter0, exact close and accepted closeouter0/final journal")
failures=[
("arbitrary input",E,"input_ref",{"path":"RAM/other.json","sha256":"5"*64}),
("arbitrary direct wait",E,"actual_wait_ref",{"path":"RAM/otherwait.json","sha256":"6"*64}),
("wrong v4 driver",ns["INPUT_REF"],"root_driver_ref",CURRENT_DRIVER),
("wrong v4 worker",ns["INPUT_REF"],"main_source_ref",CURRENT_ENTRY),
("wrong v4 source closure",ns["INPUT_REF"],"source_sha256",new_manifest),
("unknown direct wait",ns["WAIT_REF"],"actual_direct_worker_wait",False),
("nonzero worker",ns["WAIT_REF"],"returncode",1),
("unresolved drain",ns["WAIT_REF"],"all_owned_processes_and_groups_drained",False),
("worker failure",ns["WAIT_REF"],"operational_failure","failed"),
("unconsumed conversion outer",OUTER,"actual_terminal_consumed",False),
("nonzero conversion outer",OUTER,"actual_exit_code",1),
("wrong close key",CLOSE,"key","other-key"),
("missing ROOT acceptance",ACCEPT,"accepted",False),
("close grants science",ACCEPT,"native_or_seal_execution_authorized",True),
("unconsumed close outer",CLOSE_OUTER,"actual_terminal_consumed",False),
("nonzero close outer",CLOSE_OUTER,"actual_exit_code",1),
("wrong close outer receipt",CLOSE_OUTER,"ordinary_close_receipt_ref",E),
("unaccepted final journal",FINAL,"final_original_journal_accepted",False),
("wrong final selector",FINAL,"storage_policy_selector_ref",E)]
for label,r,k,v in failures:
    changed=base_records();changed[key(r)][k]=v
    try:run(changed)
    except (ValueError,KeyError):passed("Reject "+label)
    else:raise AssertionError("Accepted "+label)
for label,edit in [
("wrong close reservation hash",lambda r:r[key(CLOSE)]["value"].update(reservation_sha256="0"*64)),
("close lacks revocation",lambda r:r[key(CLOSE)]["value"].update(future_writes_and_relaunch_revoked=False)),
("missing close acceptance ref",lambda r:r[key(E)].pop("ordinary_close_terminal_acceptance_ref")),
("missing final journal ref",lambda r:r[key(ACCEPT)].pop("final_original_journal_acceptance_ref"))]:
    changed=base_records();edit(changed)
    try:run(changed)
    except (ValueError,KeyError):passed("Reject "+label)
    else:raise AssertionError("Accepted "+label)
# Nonconversion path never calls the special acceptance reader.
seen=[]
fn=build(lambda *a,**kw:seen.append(kw)or "unchanged-native",lambda *a,**kw:(_ for _ in ()).throw(AssertionError("wrong branch")),lambda r:(_ for _ in ()).throw(AssertionError("special read")))
assert fn(E,stage="native",parent=ns["PARENT"],completion_ref=E)=="unchanged-native"and seen[0]["stage"]=="native"
passed("Seal's native predecessor uses untouched original branch with no v4 exception")
for text in ("ref(DRIVER)","ref(ENTRY)","sources()['source_sha256']"):
    bad=copy.deepcopy(old)
    pattern=ast.parse(text,mode="eval").body
    class Remove(ast.NodeTransformer):
        def visit(self,n):
            if same(n,pattern):return ast.copy_location(ast.Constant(value=None),n)
            return super().visit(n)
    bad=Remove().visit(bad)
    try:derive(bad)
    except ValueError:passed("Reject drift at original operand "+text)
    else:raise AssertionError("Accepted altered source shape")
result=dict(schema="FLARE_exact_v4_predecessor_bridge_source_check_v1",status="PASS_SOURCE_AST_RAM_ONLY",case_count=len(cases),cases=cases,source_refs={k:dict(path=p,sha256=s)for k,(p,s)in PINS.items()},real_receipts_read=False,resource_guards_imported_or_executed=False,process_inspected=False,science_or_payload_or_truth_read=False,operational_IO=False,original_metadata_AST_executed_only_with_RAM_doubles=True,independent_review=False,actual_close_pending_ROOT=True,actual_launch_authority=False)
p=OUT/"source_check.result.v1.json"
if p.exists():raise ValueError("Immutable source result exists")
p.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status=result["status"],case_count=len(cases),result_ref=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))))
