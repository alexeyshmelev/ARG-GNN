"""Self-contained synthetic ownership tests. No live process, device or data access.
Reads source files only, extracts original definitions, substitutes fake process
maps, fake pidfds and fake clocks. No project/scientific module imports.
Run explicitly with --run. Prints one compact software-fixture report.
"""
from pathlib import Path
import ast,hashlib,itertools,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FOLDER="research/reviews/native_VAL000_GPU0_owned_roster_source_20261016_v4"
OWNED="research/reviews/American_C3_native_MLM_amendment_v12/owned_lifecycle_v2.py"

def run():
    owned_text=(ROOT/OWNED).read_text();driver_text=(ROOT/FOLDER/"execution_only_v1.py").read_text()
    owned_tree=ast.parse(owned_text);driver_tree=ast.parse(driver_text)
    definitions=[n for n in owned_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("_same","signal_identity","OwnedTree")]
    owner,leader=77,100000
    rows={leader:dict(pid=leader,start=1,ppid=owner,group=leader,session=leader,state="R")}
    children={owner:[leader],leader:[]}
    def read_stat(pid):return dict(rows[pid]) if pid in rows else None
    def read_children(pid):return list(children.get(pid,[]))
    def forbidden(*args,**kwargs):raise AssertionError("No real OS/process API in synthetic fixture")
    ns=dict(Path=Path,os=types.SimpleNamespace(getpid=forbidden,close=forbidden),
        _stat=read_stat,_children=read_children,pidfd_open_v1=forbidden,pidfd_send_signal_v1=forbidden)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),"<original-owned-lifecycle-with-fake-process-maps>","exec"),ns)
    Tree=ns["OwnedTree"];tree=Tree(leader,owner=owner,read_stat=read_stat,read_children=read_children)
    for pid in range(leader+1,leader+4096):
        rows[pid]=dict(pid=pid,start=pid+123,ppid=leader,group=pid,session=pid,state="R")
        children[leader].append(pid)
    # Fake child enumeration also offers an unrelated same-group identity.
    unrelated=99999;rows[unrelated]=dict(pid=unrelated,start=55,ppid=13,group=leader,session=leader,state="R")
    children[owner].append(unrelated)
    tree.refresh()
    assert len(tree.known)==4096 and unrelated not in tree.known
    assert len(tree.evidence()["owned_identities"])==4096
    assert len({r["group"] for r in itertools.islice(tree.known.values(),4096)})==4096
    # Histories are intentionally retained even after children exit.
    for pid in list(rows):
        if pid not in (leader,unrelated):del rows[pid]
    assert tree.live()==[leader] and len(tree.known)==4096
    execute=next(n for n in driver_tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_v1")
    assignments={n.targets[0].id:n.value for n in ast.walk(execute) if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    def expression(name,values):return eval(compile(ast.Expression(assignments[name]),"<actual-successor-expression>","eval"),values)
    base=dict(tree=tree,adopted=[None]*4096,adopted_wait_count=4096,adopted_waits_truncated=False,OWNED_ROSTER_CAP=4096)
    assert expression("roster_complete",base) is True
    # An extra historical identity must fail closed, even if all current rows vanish.
    extra=leader+5000;rows[extra]=dict(pid=extra,start=9,ppid=leader,group=extra,session=extra,state="R")
    children[leader].append(extra);tree.refresh();assert len(tree.known)==4097
    assert expression("roster_complete",base) is False
    del rows[leader];del rows[extra]
    assert tree.identities_absent() is True
    drain_env=dict(raw_status=0,roster_complete=False,identity_absence=True,pipe_eof=True,observations=[dict(absent=True)])
    assert expression("drained",drain_env) is False
    drain_env["roster_complete"]=True;assert expression("drained",drain_env) is True
    for change in [dict(raw_status=None),dict(identity_absence=False),dict(pipe_eof=False),dict(observations=[dict(absent=None)])]:
        test=dict(drain_env,**change);assert expression("drained",test) is False
    # Run exact new adopted-status retention statements in a fake arithmetic namespace.
    increment=next(n for n in ast.walk(execute) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=="adopted_wait_count")
    retention=next(n for n in ast.walk(execute) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and ast.unparse(n.test)=="len(adopted) < OWNED_ROSTER_CAP")
    body=compile(ast.Module(body=[increment,retention],type_ignores=[]),"<actual-bounded-adopted-status-recording>","exec")
    waits=dict(adopted_wait_count=0,adopted_waits_truncated=False,adopted=[],OWNED_ROSTER_CAP=4096,failure=None,
        os=types.SimpleNamespace(waitstatus_to_exitcode=lambda s:0 if s==0 else -15))
    for pid in range(1,4098):
        waits.update(pid=pid,status=0);exec(body,waits)
    assert waits["adopted_wait_count"]==4097 and len(waits["adopted"])==4096 and waits["adopted_waits_truncated"] is True
    smalltree=types.SimpleNamespace(known={1:dict(pid=1,start=1)})
    assert expression("roster_complete",dict(base,tree=smalltree,**{k:waits[k] for k in ("adopted_wait_count","adopted_waits_truncated","adopted")})) is False
    # Original pidfd logic rejects reuse between lookup/open and sends via fake pidfd only.
    expected=dict(pid=123,start=100)
    opened=[];sent=[];closed=[]
    assert ns["signal_identity"](expected,15,read_stat=lambda p:dict(expected),open_fd=lambda p,f:(opened.append(p) or 777),
        send_fd=lambda fd,s,a,f:sent.append((fd,s)),close_fd=closed.append) is True
    assert opened==[123] and sent==[(777,15)] and closed==[777]
    count=[0]
    def reuse_after_open(pid):
        count[0]+=1;return dict(pid=pid,start=100 if count[0]==1 else 101)
    sent.clear();closed.clear()
    assert ns["signal_identity"](expected,15,read_stat=reuse_after_open,open_fd=lambda p,f:888,
        send_fd=lambda *args:sent.append(args),close_fd=closed.append) is False
    assert sent==[] and closed==[888]
    # Size actual successor observation/minimal expressions with4096 maximal rows.
    payloadref=dict(path="SYNTHETIC_SOURCE_METADATA_ONLY",sha256="0"*64)
    fullrows=[dict(pid=2147483647-i,start=18446744073709551615-i) for i in range(4096)]
    maxenv=dict(input_ref=payloadref,current_amendment_ref_v1=lambda:payloadref,ref=lambda p:payloadref,MANIFEST="fixture",
        s=dict(refs=dict(storage=payloadref)),proc=types.SimpleNamespace(pid=2147483647),raw_status=0,rc=0,drained=True,
        identity_absence=True,observations=[dict(pgid=2147483647-i,absent=True,method="targeted ESRCH within aggregate cleanup deadline") for i in range(4096)],
        ownership=dict(owned_identities=fullrows,reused_numeric_PIDs_excluded=list(range(4096)),signals_use_pidfd_only=True),
        adopted=[dict(pid=r["pid"],status=0,returncode=0) for r in fullrows],adopted_wait_count=4096,adopted_waits_truncated=False,
        failure=None,gate_released=True,pipe_eof=True,OWNED_ROSTER_POLICY_REF=payloadref,roster_complete=True,
        cleanup_errors=["x"*1024]*16,CAPS=dict(cleanup_seconds=30,work_cutoff_margin_seconds=300,terminal_evidence_margin_seconds=150),
        cleanup_finished=30.,cleanup_started=0.,start=0.,time=types.SimpleNamespace(monotonic=lambda:30.,time=lambda:0.),
        expires_unix=1000000.,attachment_ref=payloadref,tree=types.SimpleNamespace(known={r["pid"]:r for r in fullrows}),OWNED_ROSTER_CAP=4096)
    observation=expression("observation",maxenv)
    full=dict(observation,completed_ref=payloadref,actual_wait_minimal_ref=payloadref,log_ref=payloadref,
        peak_combined_owned_RSS_bytes=384*2**30,sampled_peak_scope_bytes=7*2**30,
        final_scope_census={"SYNTHETIC":7*2**30},capacity_samples=100000000,last_capacity_sample={"synthetic":True})
    minimal=expression("minimal_facts",maxenv)
    full_bytes=len((json.dumps(full,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    minimal_bytes=len((json.dumps(minimal,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    assert full_bytes<8*2**20 and minimal_bytes<65536
    assert minimal["actual_OS_wait_status"]==0 and minimal["returncode"]==0
    unknown=expression("minimal_facts",dict(maxenv,raw_status=None,rc=None,drained=False,roster_complete=False))
    assert unknown["actual_direct_worker_wait"] is False and unknown["actual_OS_wait_status"] is None and unknown["returncode"] is None
    # Exact metadata writer with fake guard/storage/save functions: a bounded
    # minimal wait remains after an oversized full receipt is rejected.
    contract_text=(ROOT/FOLDER/"amendment_contract_v1.py").read_text()
    compact=next(n for n in ast.parse(contract_text).body if isinstance(n,ast.FunctionDef) and n.name=="compact_v1")
    saved={};checked=[]
    def require0(ok,message):
        if not ok:raise ValueError(message)
    def save0(directory,name,value):
        raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        assert checked[-1]==len(raw)+8192
        saved[name]=raw
        return dict(path="SYNTHETIC_NO_FILE/"+name,sha256=hashlib.sha256(raw).hexdigest())
    writing=dict(json=json,active_amendment_v1=lambda:None,local=lambda p:p,require=require0,
        charged_census_v1=lambda s,required_write_bytes:checked.append(required_write_bytes),save=save0)
    exec(compile(ast.Module(body=[compact],type_ignores=[]),"<exact-compact-writer-with-fake-guard-and-save>","exec"),writing)
    guard=types.SimpleNamespace(budget_check_v9=lambda *args,**kwargs:None)
    writing["compact_v1"](dict(worker_auxiliary_scope="SYNTHETIC"),"actual_wait_minimal.json",minimal,guard,{},cap=65536)
    rejected=False
    try:
        writing["compact_v1"](dict(worker_auxiliary_scope="SYNTHETIC"),"actual_wait.json",dict(oversized="x"*(8*2**20)),guard,{},cap=8*2**20)
    except ValueError:rejected=True
    assert rejected and set(saved)=={"actual_wait_minimal.json"}
    assert json.loads(saved["actual_wait_minimal.json"])["actual_OS_wait_status"]==0
    return dict(schema="native_VAL000_owned_roster_synthetic_software_fixture_v4",status="PASS_SOFTWARE_ONLY",
        original_owned_lifecycle_source_sha256=hashlib.sha256(owned_text.encode()).hexdigest(),
        successor_driver_source_sha256=hashlib.sha256(driver_text.encode()).hexdigest(),
        full4096_history_and_full_group_evidence=True,unrelated_same_group_identity_rejected=True,
        exited_identities_remain_historical_not_live=True,overflow4097_cannot_claim_drain=True,
        adopted_retention_stays4096_and_explicit_count4097=True,deadline_unknown_wait_or_group_cannot_claim_drain=True,
        original_pidfd_signal_and_reused_identity_exclusion_preserved=True,
        actual_minimal_wait_expression_preserves_unknown_status=True,
        bounded_minimal_wait_survives_oversized_full_receipt=True,exact_pretty_encoded_write_size_checked=True,
        synthetic_full4096_evidence_bytes=full_bytes,synthetic_minimal_wait_bytes=minimal_bytes,
        only_source_files_read=True,live_processes_or_GPU_or_payload_or_scientific_module_imports=False)

if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit --run required, synthetic source fixture only")
    print(json.dumps(run(),sort_keys=True,indent=2))
