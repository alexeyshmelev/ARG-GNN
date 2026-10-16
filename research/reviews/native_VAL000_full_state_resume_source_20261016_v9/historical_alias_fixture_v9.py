"""Exact historical metadata auth and real tiny v8-normalized alias consumption."""
from pathlib import Path
import copy,hashlib,importlib,importlib.util,json,os,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
def run():
    c=importlib.import_module(HERE.replace("/",".")+".amendment_contract_v1")
    helper=importlib.import_module(HERE.replace("/",".")+".alias_namespace_v7")
    # Public metadata only. No actual target lstat/read or real process probe.
    terminal,wait,launch=c.validate_v8_failure_lineage_v9()
    old=c.read(c.V8_AMENDMENT_REF);qualified=c.read(c.V8_QUALIFICATION_REF)
    historical=c.validate_known_owner_scope_v6(old,qualified,c.read(c.V4_STDOUT_PROJECTION_REF))
    assert c.HISTORICAL_ROOT_SCOPE_V9["current_entry_absence_established"] is False
    assert len(c.HISTORICAL_ROOT_SCOPE_V9["registry"]["identities"])==73
    assert terminal["known_worker_identity"]==dict(pid=3660358,start=484268347)
    assert terminal["actual_supervisor_start_ticks"]=="UNKNOWN"
    work=ROOT/HERE/"tiny_v8_normalized_absence_fixture_v9";work.mkdir(exist_ok=False)
    checkpoint=work/"output"/"checkpoints";checkpoint.mkdir(parents=True);aux=work/"aux";aux.mkdir()
    target=checkpoint/"ep0-ba6500-rank0.pt";target.write_bytes(b"TINY_SYNTHETIC_ONLY_ORIGINAL6500")
    original_bytes=target.read_bytes()
    os.symlink(target.name,checkpoint/"latest-rank0.pt")
    def ref(path):
        path=Path(path);path=path if path.is_absolute() else work/path
        return dict(path=str(path.relative_to(work)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def read(record):
        path=work/record["path"];data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==record["sha256"]
        return json.loads(data)
    def save(name,value):
        path=aux/name;data=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
        with path.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        fd=os.open(aux,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return ref(path)
    s=importlib.util.spec_from_file_location("_v9_tiny_actual_v4",ROOT/"src/arggnn/project_budget_v4.py")
    v4=importlib.util.module_from_spec(s);s.loader.exec_module(v4)
    original_ref=ref(target);identity=helper.identity(target.lstat())
    fake_v8=dict(path="SYNTHETIC_ONLY_V8_AMENDMENT",sha256="1"*64)
    fake_manifest=dict(path="SYNTHETIC_ONLY_V8_MANIFEST",sha256="2"*64)
    before_box={}
    def preserve(value):
        before_box.update(value=value,record=save("synthetic_v8_before.json",value))
        return before_box["record"]
    after=helper.normalize_latest_v7(work,checkpoint,prove_drained=lambda:True,preserve=preserve,
        revalidate_pre_unlink=lambda:None,strict_boundary=lambda:dict(
            original_unchanged_budget_check_boundary_True_passed=True,
            original_whole_scope_no_follow_charge_caps_passed=True),
        max_bytes=1024,max_regular_files=1,expected_target_ref=original_ref,expected_target_identity=identity,
        context=dict(actual_execution_amendment_ref=fake_v8,adapter_manifest_ref=fake_manifest,
            historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",historical_v3_unknown_descendants_absent=None))
    after_ref=save("synthetic_v8_strict_after.json",after)
    ns=dict(c.normalize_original_alias_v7.__globals__,ROOT=work,ORIGINAL_STATE=(True,),RESUME_STATE=dict(checkpoint_identity=identity),
        V8_AMENDMENT_REF=fake_v8,V8_MANIFEST_REF=fake_manifest,
        V8_ALIAS_BEFORE_REF=before_box["record"],V8_ALIAS_AFTER_REF=after_ref,
        V8_TERMINAL_REF=dict(path="SYNTHETIC_ONLY_FAILED_V8_TERMINAL",sha256="3"*64),
        RESUME_CHECKPOINT_REF=original_ref,read=read,local=lambda path:work/path,
        alias_module_v7=lambda:helper,validate_v8_failure_lineage_v9=lambda:True,
        alias_strict_boundary_v7=lambda *args:v4.measure_scopes(work,[checkpoint.parent,aux]))
    consume=types.FunctionType(c.normalize_original_alias_v7.__code__,ns)
    result=consume({},None,None,None)
    assert result["unlink_replayed"] is False and target.read_bytes()==original_bytes
    assert not (checkpoint/"latest-rank0.pt").exists()
    # Reappearance is rejected and kept. v9 has no startup alias removal code.
    os.symlink(target.name,checkpoint/"latest-rank0.pt")
    try:consume({},None,None,None)
    except ValueError:pass
    else:raise AssertionError("reappeared alias accepted")
    assert (checkpoint/"latest-rank0.pt").is_symlink() and target.read_bytes()==original_bytes
    # Wrong historical receipt digest is rejected under its genuine hash.
    wrong=dict(ns,V8_ALIAS_BEFORE_REF=dict(before_box["record"],sha256="0"*64))
    reject=types.FunctionType(c.normalize_original_alias_v7.__code__,wrong)
    try:reject({},None,None,None)
    except AssertionError:pass
    else:raise AssertionError("wrong receipt digest accepted")
    return dict(schema="native_VAL000_v9_historical_auth_and_tiny_normalized_alias_fixture",
        status="PASS_SOFTWARE_ONLY",actual_v8_public_metadata_lineage_authenticated=True,
        old73_known_scope_authenticated_as_history_only=True,old_timestamps_not_rewritten=True,
        actual_v8_parent_start_remains_UNKNOWN=True,
        actual_tiny_v8_normalization_metadata_then_v9_absence_consumption=True,
        original_tiny_regular_bytes_unchanged=True,v9_unlink_replay_absent=True,
        reappeared_alias_rejected_and_preserved=True,wrong_receipt_hash_rejected=True,
        actual_original_checkpoint_lstat_read_write_or_process_GPU_probe_or_job=False)
if __name__=="__main__":
    import sys
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit metadata/tiny-fixture-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
