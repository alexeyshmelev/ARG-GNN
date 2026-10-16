"""Real pinned-interpreter inert import/CLI and actual command-construction smoke.

No scientific, guard, checkpoint, process/device observation or accounting API is
called. The actual supervisor code object runs only against explicitly fake
endpoints and stops at fake Popen after capturing its actual constructed argv.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,importlib,json,os,subprocess,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
PKG=HERE.replace("/",".")
PYTHON=ROOT/".runtime/env/bin/python"

def run():
    assert sys.dont_write_bytecode and Path(sys.executable).resolve()==PYTHON.resolve()
    names=["amendment_contract_v1","execution_only_v1","direct_gpu0_v1",
        "corpus_metadata_bridge_v1","resume_scientific_bridge_v5","alias_namespace_v7","fresh_entry_v9"]
    mods={name:importlib.import_module(PKG+"."+name) for name in names}
    c=mods["amendment_contract_v1"];e=mods["execution_only_v1"];d=mods["direct_gpu0_v1"]
    assert c.HERE==HERE and c.ACTUAL_DRIVER==HERE+"/execution_only_v1.py"
    assert c.DRIVER=="research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/root_driver_v1.py"
    assert e.validate_input_v1 is c.validate_input_v1 and e.normalize_original_alias_v7 is c.normalize_original_alias_v7
    assert e.normalize_resumed_alias_v7 is c.normalize_resumed_alias_v7
    assert d.active_amendment_v1 is c.active_amendment_v1
    assert mods["corpus_metadata_bridge_v1"].c is c and mods["resume_scientific_bridge_v5"].c is c
    assert mods["fresh_entry_v9"].c is c
    assert mods["fresh_entry_v9"].ExactNativeEntryGuardV9.__module__==PKG+".fresh_entry_v9"
    for name,m in mods.items():assert Path(m.__file__).resolve()==ROOT/HERE/(name+".py")
    forbidden=("torch","torch_geometric","numpy","composer","tskit","tsinfer","tsdate","msprime")
    assert not any(n in sys.modules for n in forbidden)
    imports=[]
    for name in names[:-1]:
        source=(ROOT/HERE/(name+".py")).read_text();tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith("research.reviews.native_VAL000_full_state_resume_source_"):
                assert n.module==PKG or n.module.startswith(PKG+".")
                imports.append(dict(file=name,module=n.module,line=n.lineno))
            if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith("research.reviews.native_VAL000_full_state_resume_source_"):
                assert n.value==PKG or n.value.startswith(PKG+".")
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]))
    cli=subprocess.run([str(PYTHON),"-B","-s","-m",PKG+".execution_only_v1"],
        cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert cli.returncode==0,(cli.returncode,cli.stderr)
    cli_value=json.loads(cli.stdout)
    assert cli_value==dict(status="SOURCE_ONLY_INACTIVE",execution=False,admission=False)
    assert not cli.stderr.strip(),cli.stderr
    events=[];commands=[];prints=[]
    class FakePath:
        def __init__(self,value):self.value=str(value)
        def __str__(self):return self.value
        def __truediv__(self,other):return FakePath(self.value+"/"+str(other))
        def exists(self):return False
        def mkdir(self):events.append("fake_mkdir")
    def fake_local(value):
        p=Path(value);return FakePath(p if p.is_absolute() else ROOT/p)
    def fake_ref(value):return dict(path=str(value),sha256="0"*64)
    fake_record=dict(path="SYNTHETIC_INPUT",sha256="1"*64)
    expiry=(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    spec=dict(phase="mlm",job_id="SYNTHETIC_ONLY",output_directory=HERE+"/SYNTHETIC_output",
        worker_auxiliary_scope=HERE+"/SYNTHETIC_aux",
        original_charged_output_directory=HERE+"/SYNTHETIC_original_output",
        original_charged_auxiliary_scope=HERE+"/SYNTHETIC_original_aux",
        expires_at_utc=expiry,refs=dict(config=fake_record,runtime=fake_record,storage=fake_record))
    q=dict(capacity_policy_ref=fake_record,caps=dict(max_cuda_reserved_bytes=32*2**30))
    storage=dict(budget_components=dict(implementation=fake_record,launcher=fake_record))
    class FakeGuard:
        def validate_storage_v9(self,*args,**kwargs):events.append("fake_validate_storage")
        def budget_check_v9(self,*args,**kwargs):events.append("fake_budget_check")
    fake_owned=types.SimpleNamespace(_children=lambda pid:[],_stat=lambda pid:dict(pid=pid,start=1,group=pid,session=pid))
    def fake_load(record,name):
        if name=="pidfd_backend_v1":
            obj=types.SimpleNamespace(self_probe_v1=lambda:events.append("fake_pidfd_probe"))
            sys.modules[name]=obj;return obj
        if name=="_native32_original_full_parent_owned_lifecycle":return fake_owned
        raise AssertionError("Unexpected real source/API request "+name)
    fake_ticket=dict(capacity_at_attachment=dict(synthetic=True),
        physical_lease_metadata=dict(job="SYNTHETIC_ONLY"))
    fake_direct=types.ModuleType(PKG+".direct_gpu0_v1")
    fake_direct.install_cancellation_check_v1=lambda callback:events.append("fake_cancellation_callback")
    fake_direct.install_capacity_evidence_sink_v5=lambda callback:events.append("fake_capacity_callback")
    fake_direct.attach_v1=lambda s,q:(80,81,fake_ticket)
    def fake_popen(command,**kwargs):
        commands.append(dict(command=command,pass_fds=list(kwargs["pass_fds"]),start_new_session=kwargs["start_new_session"]))
        raise RuntimeError("SYNTHETIC_CAPTURE_STOP_BEFORE_ANY_CHILD_LAUNCH")
    fake_os=types.SimpleNamespace(environ={},pathsep=os.pathsep,getpid=lambda:990001,
        pipe=lambda:(71,72),close=lambda fd:None)
    fake_signal=types.SimpleNamespace(SIGTERM=15,SIGINT=2,SIGKILL=9,signal=lambda *args:None)
    fake_selector=types.SimpleNamespace(close=lambda:None)
    namespace=dict(e.execute_v1.__globals__)
    namespace.update(validate_input_v1=lambda record:(spec,q,{},storage,{}),
        require_work_window_v2=lambda s:None,work_deadline_monotonic_v1=lambda:100000,
        guard_v1=lambda value:FakeGuard(),normalize_original_alias_v7=lambda *args:events.append("fake_original_alias_endpoint"),
        local=fake_local,worker_env_paths_v1=lambda s:{},charged_census_v1=lambda s:{},
        compact_v1=lambda *args,**kwargs:fake_ref("SYNTHETIC_RECEIPT"),active_amendment_v1=lambda:None,
        current_amendment_ref_v1=lambda:fake_ref("SYNTHETIC_v8_AMENDMENT"),ref=fake_ref,load=fake_load,
        _subreaper=lambda:events.append("fake_subreaper"),os=fake_os,signal=fake_signal,
        time=types.SimpleNamespace(monotonic=lambda:100.,time=__import__("time").time),
        selectors=types.SimpleNamespace(DefaultSelector=lambda:fake_selector),
        subprocess=types.SimpleNamespace(Popen=fake_popen,PIPE=-1,STDOUT=-2),print=lambda *args,**kwargs:prints.append(args))
    actual=types.FunctionType(e.execute_v1.__code__,namespace,e.execute_v1.__name__)
    assert actual.__code__ is e.execute_v1.__code__
    package=importlib.import_module(PKG);saved=getattr(package,"direct_gpu0_v1")
    saved_module=sys.modules[PKG+".direct_gpu0_v1"];saved_pidfd=sys.modules.get("pidfd_backend_v1")
    try:
        setattr(package,"direct_gpu0_v1",fake_direct);sys.modules[PKG+".direct_gpu0_v1"]=fake_direct
        result=actual(fake_record)
    finally:
        setattr(package,"direct_gpu0_v1",saved);sys.modules[PKG+".direct_gpu0_v1"]=saved_module
        if saved_pidfd is None:sys.modules.pop("pidfd_backend_v1",None)
        else:sys.modules["pidfd_backend_v1"]=saved_pidfd
    assert len(commands)==1,dict(commands=commands,result=result,events=events,prints=prints)
    command=commands[0]["command"];index=command.index("-m")
    assert command[index+1]==PKG+".execution_only_v1"
    assert command[0]==str(ROOT/c.NATIVE_PYTHON_REF["path"])
    assert "--worker-input-ref" in command and "--gate-fd" in command and "--amendment-ref" in command
    assert commands[0]["start_new_session"] is True and commands[0]["pass_fds"]==[71,80,81]
    assert result["actual_direct_worker_wait"] is False and result["actual_direct_worker_returncode"] is None
    assert result["status"]=="FAILED_UNRESOLVED_CLEANUP"
    assert "fake_original_alias_endpoint" in events and "fake_cancellation_callback" in events
    assert not any(n in sys.modules for n in forbidden)
    return dict(schema="native_VAL000_v9_real_inert_import_CLI_and_actual_dispatch_smoke",status="PASS_SOFTWARE_ONLY",
        actual_interpreter=sys.executable,contract_file=c.__file__,HERE=c.HERE,ACTUAL_DRIVER=c.ACTUAL_DRIVER,
        original_guard_DRIVER=c.DRIVER,actual_direct_contract_is_v8=True,actual_bridges_contract_is_v8=True,
        imported_self_routes=imports,default_CLI_returncode=cli.returncode,default_CLI_result=cli_value,
        actual_supervisor_code_object_preserved_for_fake_endpoints=True,
        actual_constructed_worker_argv=commands[0],
        fake_Popen_raised_before_any_worker_launch=True,actual_wait_not_invented=True,
        no_real_alias_guard_scientific_model_corpus_checkpoint_ledger_process_or_GPU_API_called=True,
        no_bytecode=True)
if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit inert-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
