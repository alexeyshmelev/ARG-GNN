"""Supplemental inert tests of actual v9 public helpers and CLI routes.

This unsealed sidecar is outside the sealed executed-source manifest. It never
calls capacity_v1, attach_v1, an accounting guard, actual process probes, CUDA,
or a scientific executor. All worker execution endpoints below are explicit fakes.
"""
from pathlib import Path
from unittest.mock import patch
import copy,hashlib,importlib,json,os,subprocess,sys,tempfile,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
PKG=HERE.replace("/",".")
PYTHON=ROOT/".runtime/env/bin/python"
SEALED_NAMES=("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py",
    "execution_only_v1.py","corpus_metadata_bridge_v1.py",
    "resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py",
    "source_manifest.v1.json","source_handoff.v9.json","source_static.v9.json",
    "ROOT_fresh_entry_API.INACTIVE.v9.json")

def run():
    assert sys.dont_write_bytecode and sys.flags.no_user_site
    assert Path(sys.executable).resolve()==PYTHON.resolve()
    def sealed_hashes():
        return {n:hashlib.sha256((ROOT/HERE/n).read_bytes()).hexdigest() for n in SEALED_NAMES}
    before=sealed_hashes()
    c=importlib.import_module(PKG+".amendment_contract_v1")
    e=importlib.import_module(PKG+".execution_only_v1")
    d=importlib.import_module(PKG+".direct_gpu0_v1")
    f=importlib.import_module(PKG+".fresh_entry_v9")
    b=importlib.import_module(PKG+".corpus_metadata_bridge_v1")
    r=importlib.import_module(PKG+".resume_scientific_bridge_v5")
    native=importlib.import_module("research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2.native_child_v1")
    checks=[]
    refs=[]
    for path in (HERE+"/fresh_entry_v9.py",HERE+"/alias_namespace_v7.py",c.MANIFEST,
        c.RESUME_POLICY_REF["path"],
        "research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py"):
        record=c.ref(path)
        assert hashlib.sha256(c.raw(record)).hexdigest()==record["sha256"]
        refs.append(record)
    alias=c.alias_module_v7()
    assert Path(alias.__file__)==ROOT/c.ALIAS_SOURCE
    assert callable(alias.inspect_latest_v7) and callable(alias.normalize_latest_v7)
    checks.append(dict(name="actual_ref_raw_and_private_alias_import",status="PASS",refs=refs,
        actual_private_alias_file=alias.__file__))

    assert d.capacity_v1.__globals__ is d.__dict__
    assert d.install_capacity_evidence_sink_v5.__globals__ is d.__dict__
    assert d.CAPACITY_EVIDENCE_SINK is None
    seen=[]
    def sink(value):seen.append(value)
    try:
        d.install_capacity_evidence_sink_v5(sink)
        assert d.capacity_v1.__globals__["CAPACITY_EVIDENCE_SINK"] is sink
        assert d.attach_v1.__globals__["CAPACITY_EVIDENCE_SINK"] is sink
        try:d.install_capacity_evidence_sink_v5(lambda value:None)
        except ValueError:pass
        else:raise AssertionError("Second actual sink installation accepted")
    finally:d.CAPACITY_EVIDENCE_SINK=None
    assert not seen
    rows=[dict(pid=800000+i,used_mib=i,name="SYNTHETIC_"+("x"*200),classification="fake") for i in range(40)]
    sample=dict(GPU0=dict(compute=rows),external_compute=rows,classification=rows)
    unchanged=copy.deepcopy(sample)
    bounded=d.bounded_capacity_sample_v5(sample)
    assert sample==unchanged
    for key in ("external_compute","classification"):
        assert len(bounded[key])==32 and bounded[key+"_actual_count"]==40 and bounded[key+"_truncated"] is True
        assert all(len(v["name"])==128 for v in bounded[key])
    assert len(bounded["GPU0"]["compute"])==32
    assert bounded["GPU0"]["actual_compute_row_count"]==40 and bounded["GPU0"]["compute_rows_truncated"] is True
    try:d.bounded_capacity_sample_v5(dict(SYNTHETIC_OVERSIZE="x"*65537))
    except ValueError:pass
    else:raise AssertionError("Unbounded actual capacity formatter accepted")
    checks.append(dict(name="actual_capacity_sink_function_globals_and_pure_bounded_formatter",
        status="PASS",capacity_v1_and_attach_v1_never_called=True,sink_never_invoked=True,
        synthetic_rows_only=True,second_install_rejected=True,oversized_json_rejected=True))

    cli_env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]))
    cli=[]
    for args in ([],["--help"]):
        result=subprocess.run([str(PYTHON),"-B","-s","-m",PKG+".execution_only_v1",*args],
            cwd=ROOT,env=cli_env,capture_output=True,text=True,timeout=30)
        assert result.returncode==0 and not result.stderr.strip(),(args,result.returncode,result.stderr)
        if args:
            assert "--worker-input-ref" in result.stdout and "--gate-fd" in result.stdout
            assert "--attachment-ref" in result.stdout and "--execute" in result.stdout
        else:assert json.loads(result.stdout)==dict(status="SOURCE_ONLY_INACTIVE",execution=False,admission=False)
        cli.append(dict(args=args,returncode=result.returncode,stdout=result.stdout))
    marker=dict(path="SYNTHETIC_ONLY",sha256="0"*64)
    dispatched=[];saved_amendment=c.AMENDMENT_REF;saved_sys_path=list(sys.path)
    def fake_worker(*args):dispatched.append(args);return 37
    try:
        with patch.object(e,"worker_v1",fake_worker):
            rc=e.main(["--worker-input-ref",json.dumps(marker),"--amendment-ref",json.dumps(marker),
                "--gate-fd","73","--attachment-ref",json.dumps(marker)])
        assert rc==37 and dispatched==[(marker,73,marker)]
        assert c.AMENDMENT_REF==marker
    finally:c.AMENDMENT_REF=saved_amendment;sys.path[:]=saved_sys_path
    checks.append(dict(name="actual_default_help_and_worker_parser_dispatch",status="PASS",
        actual_pinned_CLI=cli,actual_main_called_real_set_amendment_with_synthetic_metadata_only=True,
        actual_main_dispatched_to_explicit_fake_worker=True,no_worker_started=True))

    nested=next(v for v in native.run_v1.__code__.co_consts if isinstance(v,types.CodeType) and v.co_name=="fast_guard")
    events=[]
    lease=dict(physical_gpu=0,borrowed_physical_inode=True,SYNTHETIC_ONLY=True)
    class NativeGuard:
        sentinel="SYNTHETIC_FORWARDED_ATTRIBUTE"
        def apply_and_validate_title(self):events.append("fake_title")
        def validate_lease(self):events.append("fake_lease");return lease
    record=dict(path="SYNTHETIC_ENTRY",sha256="1"*64)
    proxy=f.ExactNativeEntryGuardV9(NativeGuard(),native.run_v1,
        capture=lambda:events.append("fake_capture") or record,
        verify=lambda value:events.append("fake_verify") or c.require(value==record,"Synthetic exact entry"))
    assert proxy.expected_fast_code is nested and proxy.sentinel=="SYNTHETIC_FORWARDED_ATTRIBUTE"
    def cell(value):
        def inner():return value
        return inner.__closure__[0]
    values=dict(guard=proxy,lease=lease,require=c.require,spec=dict(SYNTHETIC_ONLY=True))
    fast=types.FunctionType(nested,dict(c=types.SimpleNamespace(active=lambda spec:events.append("fake_active"))),
        closure=tuple(cell(values[name]) for name in nested.co_freevars))
    assert fast.__code__ is nested
    fast("after_native_device")
    assert events==["fake_active","fake_title","fake_lease"]
    events.clear();fast(f.WORKER_EVENT)
    assert events==["fake_active","fake_title","fake_lease","fake_capture","fake_verify","fake_lease","fake_verify"]
    assert proxy.worker_entry_ref==record
    try:fast(f.WORKER_EVENT)
    except ValueError:pass
    else:raise AssertionError("Entry event replay accepted")
    assert events.count("fake_capture")==1
    checks.append(dict(name="actual_imported_original_nested_native_code_and_real_v9_proxy",
        status="PASS",actual_original_native_module=native.__file__,actual_nested_freevars=list(nested.co_freevars),
        normal_event_forwarded=True,entry_capture_and_double_lease_verify=True,replay_rejected=True,
        original_outer_scientific_run_never_called=True))

    class StopBeforeScience(RuntimeError):pass
    worker_events=[]
    fake_paths={"TMPDIR":str(ROOT/HERE)}
    spec=dict(job_id="SYNTHETIC_JOB",worker_auxiliary_scope=HERE,
        output_directory=HERE)
    qualifier=dict(direct_attachment_authority_ref=marker)
    ticket=dict(job_id="SYNTHETIC_JOB",borrowed_physical_inode=True,direct_attachment_authority_ref=marker)
    fake_guard=NativeGuard()
    fake_os=types.SimpleNamespace(read=lambda fd,n:b"G",close=lambda fd:None,
        getpid=lambda:700001,getpgrp=lambda:700001,getsid=lambda pid:700001,environ=fake_paths)
    def fake_scientific_stop(root_ref,s,q,cfg,ticket_arg,entry_guard):
        assert (root_ref,s,q,cfg,ticket_arg)==(marker,spec,qualifier,{},ticket)
        assert isinstance(entry_guard,f.ExactNativeEntryGuardV9)
        assert entry_guard.native_guard is fake_guard and entry_guard.expected_fast_code is nested
        worker_events.append("actual_worker_imported_real_v9_modules_and_real_original_native_code")
        raise StopBeforeScience("SYNTHETIC_STOP_BEFORE_SCIENTIFIC_EXECUTOR")
    old_temp=tempfile.tempdir
    try:
        with patch.object(e,"os",fake_os),patch.object(e,"validate_input_v1",lambda value:(spec,qualifier,{}, {}, {})),\
            patch.object(e,"worker_env_paths_v1",lambda value:fake_paths),patch.object(e,"read",lambda value:ticket),\
            patch.object(d,"install_before_torch_v1",lambda *args:fake_guard),\
            patch.object(b,"install_private_authentication_bridge_v1",lambda:worker_events.append("fake_bridge_endpoint")),\
            patch.object(e,"guard_v1",lambda value:object()),patch.object(e,"make_entry_tick_v9",lambda value:lambda:None),\
            patch.object(r,"run_resumed_native_v5",fake_scientific_stop):
            try:e.worker_v1(marker,73,dict(path=HERE+"/attachment.json",sha256="0"*64))
            except StopBeforeScience:pass
            else:raise AssertionError("Synthetic worker did not stop")
    finally:tempfile.tempdir=old_temp
    assert worker_events==["fake_bridge_endpoint","actual_worker_imported_real_v9_modules_and_real_original_native_code"]
    checks.append(dict(name="actual_worker_code_real_module_routing_to_real_fresh_guard_constructor",
        status="PASS",actual_worker_function_module=e.worker_v1.__module__,
        actual_direct_module=d.__file__,actual_corpus_bridge_module=b.__file__,
        actual_resume_bridge_module=r.__file__,actual_fresh_entry_module=f.__file__,
        all_operational_and_scientific_endpoints_explicit_fakes=True,stopped_before_scientific_executor=True))
    forbidden=("torch","torch_geometric","numpy","composer","tskit","tsinfer","tsdate","msprime")
    assert not any(n in sys.modules for n in forbidden)
    after=sealed_hashes();assert before==after
    return dict(schema="native_VAL000_v9_supplemental_real_public_helpers_inert_evidence_v1",
        status="PASS_SOFTWARE_ONLY",actual_interpreter=sys.executable,checks=checks,
        sealed_files_unchanged=after,counts=dict(check_groups=len(checks)),
        supplemental_unsealed_sidecar_not_added_to_executed_source_manifest=True,
        no_actual_capacity_attach_guard_resource_probe_accounting_checkpoint_CUDA_or_scientific_run=True,
        no_scientific_progress_claim=True,no_bytecode=True)

if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit inert-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
