"""Representative fake-time actual-code tests. No real probes, CUDA or training."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib,json,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v9"
def run():
    f=importlib.import_module(HERE.replace("/",".")+".fresh_entry_v9")
    checks=[]
    class Clock:
        elapsed=0.
        def now(self):return datetime(2026,10,16,18,tzinfo=timezone.utc)+timedelta(seconds=self.elapsed)
        def mono(self):return self.elapsed
    class FakePath:
        def __init__(self,text):self.text=str(text)
        def __str__(self):return self.text
        def __truediv__(self,name):return FakePath(self.text+"/"+name)
        def is_dir(self):return True
        def exists(self):return False
    native_source=(ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py").read_text()
    tree=ast.parse(native_source);run_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_v1")
    space={};exec(compile(ast.Module(body=[run_node],type_ignores=[]),"<actual-unchanged-native-run-source-only>","exec"),space)
    native_run=space["run_v1"]
    nested=next(x for x in native_run.__code__.co_consts if isinstance(x,types.CodeType) and x.co_name=="fast_guard")
    def cell(value):
        def inner():return value
        return inner.__closure__[0]
    def setup(case):
        clock=Clock();events=[];records={};entry_refs={};prints=[]
        registry=dict(schema="SYNTHETIC_ONLY_registry",identities=[dict(pid=700001,start_ticks=99),dict(pid=700002,start_ticks=None)],
            process_groups=[700003],historical_v3_descendant_coverage="UNKNOWN_TRUNCATED64",
            unknown_historical_descendants_claimed_listed=False)
        spec=dict(worker_auxiliary_scope="SYNTHETIC_ONLY_OWN_AUX")
        manifest=dict(path="SYNTHETIC_ONLY_MANIFEST",sha256="0"*64)
        amend=dict(path="SYNTHETIC_ONLY_AMENDMENT",sha256="1"*64)
        def rr(path):return dict(path=str(path),sha256=hashlib.sha256(str(path).encode()).hexdigest())
        def save(directory,name,value):
            events.append("durable_receipt_attempt")
            if case=="receipt_write_failure":raise OSError("SYNTHETIC receipt write failure")
            data=json.dumps(value,sort_keys=True).encode()
            r=dict(path=str(directory)+"/"+name,sha256=hashlib.sha256(data).hexdigest())
            records[r["path"]]=(r,copy.deepcopy(value));return r
        def read(r):
            stored,value=records[r["path"]]
            assert r==stored
            return copy.deepcopy(value)
        fc=types.SimpleNamespace(require=f.c.require,HERE=HERE,MANIFEST=manifest["path"],
            active_amendment_v1=lambda:None,entry_known_registry_v9=lambda:copy.deepcopy(registry),
            current_amendment_ref_v1=lambda:amend,ref=lambda path:manifest if path==manifest["path"] else rr(path),
            local=lambda path:FakePath(path),charged_census_v1=lambda *a,**k:events.append("complete_caps_projection"),
            save=save,read=read,V8_KNOWN_BUNDLE_REF=rr("SYNTHETIC_ONLY_HISTORICAL_BUNDLE"),
            V8_TERMINAL_REF=rr("SYNTHETIC_ONLY_FAILED_V8_TERMINAL"))
        def tick():assert clock.elapsed<10000
        def kill0(pid):
            clock.elapsed+=.001
            if (case=="known_PID_alive" and pid==700001) or (case=="unknown_start_numeric_PID_exists" and pid==700002):return
            raise ProcessLookupError()
        def pid_probe(row):return f.probe_PID_v9(row,kill0=kill0,read_stat=lambda pid:99,now=clock.now)
        def group0(value):
            clock.elapsed+=.001
            if case=="old_group_exists":return
            raise ProcessLookupError()
        def group_probe(group):return f.probe_group_v9(group,killpg0=group0,now=clock.now)
        def collect(registry,*,stage,context,tick):
            return f.collect_v9(registry,stage=stage,context=context,tick=tick,pid_probe=pid_probe,group_probe=group_probe,
                wall_now=clock.now,monotonic=clock.mono)
        class Guard:
            def budget_check_v9(self,*args,**kwargs):
                assert kwargs["boundary"] is True and kwargs["required_write_bytes"]==f.MAX_ENTRY_BYTES+8192
                events.append("slow_full_original_guard_140s");clock.elapsed+=140.
        def fsync(fd):
            if case=="receipt_fsync_ages_past120":clock.elapsed+=121.
        fake_os=types.SimpleNamespace(getpid=lambda:800001,open=lambda *a:88,close=lambda fd:None,fsync=fsync,
            O_RDONLY=0,O_DIRECTORY=65536,O_NOFOLLOW=131072)
        ns=dict(f.capture_entry_v9.__globals__,c=fc,collect_v9=collect,ENTRY_REFS=entry_refs,os=fake_os,
            require_fresh_v9=lambda value,**kw:f.require_fresh_v9(value,now=clock.now()),
            print=lambda *a,**k:prints.append(a))
        verify=types.FunctionType(f.require_entry_ref_v9.__code__,ns)
        ns["require_entry_ref_v9"]=verify
        capture=types.FunctionType(f.capture_entry_v9.__code__,ns)
        return clock,events,records,entry_refs,spec,capture,verify,tick,Guard(),prints
    # Startup and native setup both exceed120s. Each exact actual entry obtains
    # NEW observations after its own140s full guard. Nothing edits an old row.
    clock,events,records,refs,spec,capture,verify,tick,g,prints=setup("success")
    old_root_snapshot=dict(schema="ROOT_native_VAL000_fresh_known_PID_start_absence_v6",checked_utc=clock.now().isoformat())
    supervisor=capture(spec,{},g,{},stage="supervisor",tick=tick)
    verify(supervisor,spec,stage="supervisor",tick=tick)
    assert clock.elapsed>140 and refs["supervisor"]==supervisor
    clock.elapsed+=250.
    class NativeGuard:
        calls=0
        def apply_and_validate_title(self):pass
        def validate_lease(self):self.calls+=1;return dict(SYNTHETIC_ONLY=True)
    ng=NativeGuard()
    proxy=f.ExactNativeEntryGuardV9(ng,native_run,
        capture=lambda:capture(spec,{},g,{},stage="worker",tick=tick),
        verify=lambda record:verify(record,spec,stage="worker",tick=tick))
    values=dict(guard=proxy,lease=dict(SYNTHETIC_ONLY=True),require=f.c.require,spec={})
    fast=types.FunctionType(nested,dict(c=types.SimpleNamespace(active=lambda s:None)),
        closure=tuple(cell(values[name]) for name in nested.co_freevars))
    fast(f.WORKER_EVENT)
    assert proxy.worker_entry_ref==refs["worker"] and ng.calls==2
    assert clock.elapsed>530 and old_root_snapshot["checked_utc"].endswith("+00:00")
    try:f.require_fresh_v9(old_root_snapshot,now=clock.now())
    except ValueError:pass
    else:raise AssertionError("old ROOT snapshot alone accepted")
    # New entry rows are original120s freshness, independently timestamped.
    worker_value=records[refs["worker"]["path"]][1]
    assert worker_value["PID_observations"][0]["checked_utc"]!=old_root_snapshot["checked_utc"]
    f.require_fresh_v9(worker_value,now=clock.now())
    try:f.require_fresh_v9(worker_value,now=clock.now()+timedelta(seconds=120.01))
    except ValueError:pass
    else:raise AssertionError("freshness relaxed past120")
    checks.append(dict(name="supervisor_and_exact_native_worker_after_more_than120s_setup",status="PASS",
        fake_total_setup_seconds=clock.elapsed,old_ROOT_rows_unchanged=True,new_entry_refs=copy.deepcopy(refs)))
    # Failures at targeted probes, receipt persistence or post-lease freshness
    # prevent returning from the actual native pre-CUDA fast_guard.
    for case in ["known_PID_alive","unknown_start_numeric_PID_exists","old_group_exists",
        "receipt_write_failure","receipt_fsync_ages_past120","post_probe_lease_delay_past120"]:
        clock,events,records,refs,spec,capture,verify,tick,g,prints=setup(case)
        marker=[]
        class NativeGuard:
            calls=0
            def apply_and_validate_title(self):pass
            def validate_lease(self):
                self.calls+=1
                if case=="post_probe_lease_delay_past120" and self.calls==2:clock.elapsed+=121.
                return dict(SYNTHETIC_ONLY=True)
        ng=NativeGuard()
        proxy=f.ExactNativeEntryGuardV9(ng,native_run,capture=lambda:capture(spec,{},g,{},stage="worker",tick=tick),
            verify=lambda record:verify(record,spec,stage="worker",tick=tick))
        values=dict(guard=proxy,lease=dict(SYNTHETIC_ONLY=True),require=f.c.require,spec={})
        fast=types.FunctionType(nested,dict(c=types.SimpleNamespace(active=lambda s:None)),
            closure=tuple(cell(values[name]) for name in nested.co_freevars))
        try:
            fast(f.WORKER_EVENT);marker.append("WOULD_ENTER_CUDA_MODEL_TRAINING")
        except (ValueError,RuntimeError,OSError):pass
        else:raise AssertionError(case)
        assert not marker
        checks.append(dict(name=case,status="PASS_REJECTED_BEFORE_CUDA",receipt_count=len(records)))
    # PID reuse is accepted only as exact OLD identity absence. Unknown starts
    # require ESRCH and are never filled from the replacement process.
    row=f.probe_PID_v9(dict(pid=700001,start_ticks=99),kill0=lambda p:None,read_stat=lambda p:100,
        now=lambda:datetime(2026,10,16,18,tzinfo=timezone.utc))
    assert row["known_identity_absent"] is True and row["numeric_PID_absent"] is False and row["observed_different_start_ticks"]==100
    # Only the original nested code object can invoke the late boundary.
    proxy=f.ExactNativeEntryGuardV9(ng,native_run,capture=lambda:marker.append("BAD_CAPTURE"),verify=lambda r:None)
    def fast_guard(event):return proxy.validate_lease()
    try:fast_guard(f.WORKER_EVENT)
    except ValueError:pass
    else:raise AssertionError("wrong nested code accepted")
    assert "BAD_CAPTURE" not in marker
    checks.append(dict(name="only_exact_original_nested_code_event_and_honest_PID_reuse",status="PASS"))
    # Entire finite cleanup finally remains exactly v8.
    old=ast.parse((ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v8/execution_only_v1.py").read_text())
    new=ast.parse((ROOT/HERE/"execution_only_v1.py").read_text())
    def final(tree):
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="execute_v1")
        return next(n.finalbody for n in run.body if isinstance(n,ast.Try) and n.finalbody)
    assert ast.dump(ast.Module(body=final(old),type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=final(new),type_ignores=[]),include_attributes=False)
    return dict(schema="native_VAL000_v9_actual_code_fake_time_entry_fixture",status="PASS_SOFTWARE_ONLY",
        checks=checks,actual_original_native_run_code_object_and_nested_guard_used=True,
        original120s_freshness_not_relaxed=True,old_ROOT_snapshot_alone_rejected=True,
        finite_cleanup_finally_AST_unchanged=True,actual_probe_process_GPU_model_checkpoint_or_training_access=False)
if __name__=="__main__":
    import sys
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit fake-only --run required")
    print(json.dumps(run(),sort_keys=True,indent=2))
