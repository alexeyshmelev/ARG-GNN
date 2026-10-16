from pathlib import Path
import ast,copy,hashlib,io,itertools,json,types,contextlib,difflib
from datetime import datetime,timezone,timedelta
R=Path("/workspace-SR003.nfs2/ARG-GNN")
D=R/"research/reviews/native_VAL000_GPU0_owned_roster_independent_review_20261016_v4"
S="research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4"
P="research/reviews/native_VAL000_GPU0_same_charge_metadata_bridge_source_20261016_v3"
O="research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2"
def read(p): return (R/p).read_text()
def doc(p): return json.loads(read(p))
def ref(p): return dict(path=p,sha256=hashlib.sha256((R/p).read_bytes()).hexdigest())
def checkref(r): assert ref(r["path"])==r,r;return r
handoff=checkref(dict(path=S+"/source_handoff.v4.json",sha256="113e4fea3f185d2e0bcf6ae072239266721e57354b9f2d135b65b7c385ce94ea"))
m=doc(S+"/source_manifest.v1.json")
checkref(dict(path=S+"/source_manifest.v1.json",sha256="4fc7f605f36d18b82ed155c24b0095a83b4b8b3a9649d8d65f81b23bc8cfae37"))
checked=[]
for r in m["new_executed_source_refs"]: checked.append(checkref(r))
for k,v in doc(S+"/source_handoff.v4.json").items():
    if isinstance(v,dict) and set(v)=={"path","sha256"}:checked.append(checkref(v))
oldmanifest=doc(O+"/source_manifest.v1.json")
# Source text only. Do not inspect data, checkpoints, interpreters or installed packages.
original_source_refs=[r for r in oldmanifest["refs"] if Path(r["path"]).suffix in (".py",".yaml")]
for r in original_source_refs:checked.append(checkref(r))
for name in ("__init__.py","direct_gpu0_v1.py","corpus_metadata_bridge_v1.py"):
    assert ast.dump(ast.parse(read(S+"/"+name).replace(S,P).replace(S.replace(chr(47),chr(46)),P.replace(chr(47),chr(46)))),include_attributes=False)==ast.dump(ast.parse(read(P+"/"+name)),include_attributes=False),name
new=ast.parse(read(S+"/execution_only_v1.py"));old=ast.parse(read(P+"/execution_only_v1.py"))
fn=lambda tree,name:next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
for name in ("worker_v1","rss_identity_v1","_RAMLog","_subreaper","_children"):
    n=ast.parse(ast.unparse(fn(new,name)).replace(S,P).replace(S.replace(chr(47),chr(46)),P.replace(chr(47),chr(46))))
    o=ast.parse(ast.unparse(fn(old,name)))
    assert ast.dump(n,include_attributes=False)==ast.dump(o,include_attributes=False),name
old64=[n for n in ast.walk(fn(old,"execute_v1")) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==64]
new64=[n for n in ast.walk(fn(new,"execute_v1")) if isinstance(n,ast.Constant) and type(n.value)is int and n.value==64]
assert len(old64)==11 and not new64
audit=[]
for n in sorted(old64,key=lambda n:(n.lineno,n.col_offset)):
    audit.append(dict(predecessor_line=n.lineno,column=n.col_offset,value=64))
contract=ast.parse(read(S+"/amendment_contract_v1.py"))
common=ast.parse(read(O+"/common_v1.py"))
encoding=lambda f:next(n.value for n in ast.walk(f) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ("data",) for t in n.targets))
assert ast.dump(encoding(fn(contract,"compact_v1")),include_attributes=False)==ast.dump(encoding(fn(common,"save")),include_attributes=False)
assert ast.literal_eval(next(n.value for n in contract.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="OWNED_ROSTER_CAP" for t in n.targets)))==4096
execfn=fn(new,"execute_v1")
# Execute exact post-cleanup control flow with fake source-bound evidence, in-memory writer and inert path API.
rcindex=next(i for i,n in enumerate(execfn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="rc" for t in n.targets))
tail=ast.FunctionDef(name="_tail",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ("failure","observations","peak_scope")],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(execfn.body[rcindex:]),decorator_list=[])
tailcode=compile(ast.fix_missing_locations(ast.Module(body=[tail],type_ignores=[])),"<exact-v4-postcleanup-tail-with-fakes>","exec")
compact=fn(contract,"compact_v1")
ref0=dict(path="SYNTHETIC_SOURCE_ONLY",sha256="0"*64)
class FakePath:
    def __truediv__(self,key): return self
    def is_file(self): return False
def req(ok,message):
    if not ok:raise ValueError(message)
def simulate(count,adopt_count,raw_status=0,identity_absence=True,group_absent=True,eof=True,overflow_full=False,authority_margin=True):
    history={i+100:dict(pid=i+100,start=i+1,group=i+100) for i in range(count)}
    events=[];saved={}
    tree=types.SimpleNamespace(known=history,evidence=lambda:dict(owned_identities=[dict(pid=r["pid"],start=r["start"]) for r in history.values()]))
    ns=dict(os=types.SimpleNamespace(waitstatus_to_exitcode=lambda s:0 if s==0 else -15),proc=types.SimpleNamespace(pid=100),
        raw_status=raw_status,failure=None,observations=[dict(pgid=r["group"],absent=group_absent) for r in history.values()][:4096],
        itertools=itertools,tree=tree,leader_seed=None,OWNED_ROSTER_CAP=4096,adopted_wait_count=adopt_count,
        adopted=[dict(pid=i,status=0,returncode=0) for i in range(min(adopt_count,4096))],adopted_waits_truncated=adopt_count>4096,
        identity_absence=identity_absence,pipe_eof=eof,physical_fd=None,logical_fd=None,input_ref=ref0,
        current_amendment_ref_v1=lambda:ref0,ref=lambda p:ref0,MANIFEST="SYNTHETIC",
        s=dict(refs=dict(storage=ref0),worker_auxiliary_scope="SYNTHETIC"),gate_released=True,
        OWNED_ROSTER_POLICY_REF=ref0,cleanup_errors=[],CAPS=dict(cleanup_seconds=30,work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150,max_checkpoint_bytes=3*2**30),
        cleanup_finished=30.,cleanup_started=0.,start=0.,time=types.SimpleNamespace(monotonic=lambda:30.,time=lambda:0.),
        expires_unix=1000000. if authority_margin else 1.,attachment_ref=ref0,require=req,json=json,MINIMAL_WAIT_CAP_BYTES=65536,
        TERMINAL_RECEIPT_CAP_BYTES=8*2**20,logs=types.SimpleNamespace(record=lambda:dict(tail="",total_pipe_bytes=0,retained_raw_bytes=0)),
        out=FakePath(),aux=FakePath(),local=lambda p:p,active_amendment_v1=lambda:None,
        charged_census_v1=lambda *a,**kw:{"SYNTHETIC":0},storage={},peak=0,peak_scope=0,capacity_samples=0,last_capacity=None,
        load=lambda *a:types.SimpleNamespace(normalize_latest=lambda *a,**k:events.append("normalize")),
        g=types.SimpleNamespace(budget_check_v9=lambda *a,**kw:None))
    def save(d,name,value):
        data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        assert len(data)<=8*2**20
        saved[name]=data;events.append(name);return ref0
    ns["save"]=save
    exec(compile(ast.Module(body=[compact],type_ignores=[]),"<exact-compact-with-fake-guard-and-memory-writer>","exec"),ns)
    realcompact=ns["compact_v1"]
    def compactcall(spec,name,value,g,storage,**kw):
        if name=="actual_wait.json" and overflow_full:value=dict(value,synthetic_oversize="x"*(8*2**20))
        return realcompact(spec,name,value,g,storage,**kw)
    ns["compact_v1"]=compactcall
    exec(tailcode,ns)
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream): result=ns["_tail"](None,ns["observations"],0)
    records=[json.loads(x) for x in stream.getvalue().splitlines()]
    facts=records[0]["native32_actual_minimal_wait_facts"]
    assert facts["actual_OS_wait_status"]==raw_status
    assert facts["minimal_wait_only_not_scientific_completion"] and facts["no_future_stage_authorized"]
    if overflow_full and facts["all_owned_processes_and_groups_drained"]:
        assert events[:2]==["actual_wait_minimal.json","normalize"]
        assert "actual_wait.json" not in saved and "actual_wait_minimal.json" in saved
        assert result["actual_wait_ref"] is None and result["no_future_stage_authorized"]
    if not facts["all_owned_processes_and_groups_drained"] or not authority_margin:assert not saved and not events
    return dict(facts=facts,events=events,saved_sizes={k:len(v) for k,v in saved.items()},result=result)
cases={}
cases["full4096"]=simulate(4096,4096)
assert cases["full4096"]["facts"]["all_owned_processes_and_groups_drained"] is True
cases["identity4097"]=simulate(4097,4096)
cases["adopted4097"]=simulate(4096,4097)
cases["unknownwait"]=simulate(1,0,raw_status=None)
cases["unknownidentityabsence"]=simulate(1,0,identity_absence=None)
cases["unknowngroup"]=simulate(1,0,group_absent=None)
cases["noEOF"]=simulate(1,0,eof=False)
cases["expired_evidence_margin"]=simulate(1,0,authority_margin=False)
cases["full_receipt_overflow"]=simulate(4096,4096,overflow_full=True)
for k in ("identity4097","adopted4097","unknownwait","unknownidentityabsence","unknowngroup","noEOF"):
    assert cases[k]["facts"]["all_owned_processes_and_groups_drained"] is False
# Independently rerun submitted source-only fixture after inspecting its entire source.
fixture=ast.parse(read(S+"/synthetic_roster_fixture_v1.py"))
ns={"__name__":"independent_rerun"}
exec(compile(fixture,"<inspected-submitted-synthetic-source>","exec"),ns)
submitted=ns["run"]()
assert submitted==doc(S+"/synthetic_roster_fixture.v1.json")
policy=doc(S+"/owned_roster_policy.INACTIVE.v4.json")
assert all(policy[k]==4096 for k in ("cumulative_identity_ceiling","adopted_wait_ceiling","cleanup_signal_retention_ceiling","group_census_retention_ceiling","complete_ownership_evidence_ceiling"))
template=doc(S+"/ROOT_amendment.INACTIVE.template.v4.json")
assert template["execution_authorized"] is False and template["independent_source_review_passed"] is False and template["template_only"] is True
assert template["original_caps"]["cleanup_seconds"]==30 and template["original_caps"]["reservation_bytes"]==7*2**30 and template["original_caps"]["auxiliary_bytes"]==768*2**20
# Hash metadata terminal only. Preserve historical truncated roster and no optimizer claim.
outerref=checkref(doc(S+"/source_handoff.v4.json")["latest_actual_failure_outer_terminal_ref"])
outer=doc(outerref["path"])
assert (outer["actual_tool_session_id"],outer["actual_exit_code"],outer["actual_direct_worker_returncode"])==(93626,241,-15)
assert outer["complete_owned_roster_known"] is False and outer["owned_roster_truncated"] is True and outer["optimizer_updates"]=="UNVERIFIED"
# Source inventory of relevant untouched non-roster 64 constants.
nonroster=[]
for p in (O+"/common_v1.py",S+"/corpus_metadata_bridge_v1.py"):
    for n in ast.walk(ast.parse(read(p))):
        if isinstance(n,ast.Constant) and type(n.value)is int and n.value==64:nonroster.append(dict(path=p,line=n.lineno,column=n.col_offset))

fullwindow=next(n for n in ast.walk(fn(common,"validate_request_v1")) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="require" and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=="Full selected operational window plus drain fits unchanged absolute expiry").args[0]
expiry=datetime.fromisoformat(template["expires_at_utc"]).timestamp()
entry_cutoff=expiry-172800-600
env=dict(q={"expires_at_utc":template["expires_at_utc"]},CAPS={"max_seconds":172800},datetime=datetime)
boundary_results=[]
for offset,expected in ((-1,True),(0,False),(1,False)):
    env["time"]=types.SimpleNamespace(time=lambda offset=offset:entry_cutoff+offset)
    accepted=eval(compile(ast.Expression(fullwindow),"<exact-original-full-window-entry-boundary>","eval"),env)
    assert accepted is expected
    boundary_results.append(dict(seconds_from_entry_cutoff=offset,accepted=accepted))
anchor=datetime.fromisoformat(template["original48h_window_anchor_utc"])
anchor_end=anchor+timedelta(seconds=172800)
deadline_checks=dict(entry_cutoff_utc=datetime.fromtimestamp(entry_cutoff,timezone.utc).isoformat(),entry_condition_strict_less_than=True,
    exact_original_entry_expression_cases=boundary_results,original_anchor_utc=anchor.isoformat(),original48h_end_utc=anchor_end.isoformat(),
    anchored_scientific_work_cutoff_utc=(anchor_end-timedelta(seconds=300)).isoformat(),absolute_expiry_utc=template["expires_at_utc"])

report=dict(schema="native_VAL000_owned_roster_independent_static_and_synthetic_v4",status="PASS_SOURCE_AND_SYNTHETIC_ONLY",
    reviewed_handoff_ref=handoff,checked_refs=checked,deadline_checks=deadline_checks,original_source_text_ref_count=len(original_source_refs),all11_predecessor64_sites=audit,
    successor_numeric64_roster_sites=len(new64),nonroster64_constants_preserved=nonroster,namespace_only_changes_in_direct_and_bridge=True,
    worker_RAMLog_subreaper_RSS_helpers_AST_preserved=True,pretty_encoding_precheck_matches_original_save_AST=True,
    original_author_sources_compiler_workers_and_model_hashes_preserved=True,
    exact_postcleanup_tail_with_in_memory_fakes=cases,submitted_fixture_independently_rerun=submitted,
    project_or_scientific_modules_imported=False,live_process_GPU_runtime_data_truth_checkpoint_probes=False,
    packages_jobs_accounting_Git_operations=False,only_own_review_directory_written=True)
(D/"static_verification.v4.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps(dict(status=report["status"],refs=len(checked),original_source_text_refs=len(original_source_refs),
    synthetic_cases=list(cases),full4096_saved_sizes=cases["full4096"]["saved_sizes"],
    full_receipt_overflow_minimal_saved=cases["full_receipt_overflow"]["saved_sizes"],
    artifact_ref=ref(str((D/"static_verification.v4.json").relative_to(R)))),indent=2))
