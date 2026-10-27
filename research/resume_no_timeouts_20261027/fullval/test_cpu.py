"""Tiny CPU-only race, I/O, no-deadline and owned-native lifecycle checks."""
import ast
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from unittest.mock import patch
from . import aggregate_guard as guard, native_supervisor as native, contracts
from research.efficiency_20261027.execution.european_full_val_runner import aggregate_guard_v2 as prior
from research.efficiency_20261027.execution import test_native_terminal_v5 as fixtures

HERE=Path(__file__).resolve().parent
ROOT=contracts.ROOT

def ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def run():
    base=HERE/'cpu_fixtures_v5'
    base.mkdir()
    checks=[]
    def scope(name):
        path=base/name
        path.mkdir()
        return path
    def rename_race(module,path):
        temporary=path/'status.tmp'
        destination=path/'status.json'
        temporary.write_text('{"status":"running"}\n')
        original=Path.lstat
        renamed=[]
        def race(p,*args,**kwargs):
            if p==temporary and not renamed:
                temporary.replace(destination)
                renamed.append(True)
            return original(p,*args,**kwargs)
        with patch.object(Path,'lstat',race):
            try:
                value=module.measure(path)
            except FileNotFoundError as error:
                return dict(rejected=True,filename=error.filename)
        assert renamed and value['charged_bytes']>=destination.stat().st_size
        return dict(rejected=False,charged_bytes=value['charged_bytes'])
    original=rename_race(prior,scope('prior_status_race'))
    assert original['rejected'] and original['filename'].endswith('/status.tmp')
    checks.append(dict(name='original_monitor_reproduces_atomic_status_tmp_rename_race',**original))
    fixed=rename_race(guard,scope('fixed_status_race'))
    assert not fixed['rejected']
    checks.append(dict(name='successor_accounts_atomic_status_replacement',**fixed))

    path=scope('io_failures')
    (path/'payload.bin').write_bytes(b'payload')
    original_lstat=Path.lstat
    for code in (errno.EIO,errno.EACCES,errno.ENOENT):
        def failure(p,*args,**kwargs):
            if p==path/'payload.bin':raise OSError(code,os.strerror(code),str(p))
            return original_lstat(p,*args,**kwargs)
        try:
            with patch.object(Path,'lstat',failure):guard.measure(path)
        except OSError as error:assert error.errno==code
        else:raise AssertionError('A genuine payload I/O failure was ignored')
    checks.append(dict(name='payload_ENOENT_EIO_EACCES_remain_fatal'))

    proc=scope('proc')
    for pid in (11,12,13):(proc/str(pid)).mkdir()
    fields=['S','1','11','11']+['0']*15+['123','0','2']
    (proc/'11'/'stat').write_text('11 (fixture process) '+' '.join(fields))
    (proc/'13'/'stat').write_text('13 (fixture process) '+' '.join(fields))
    # PID 12 exits between directory enumeration and its stat read.
    table=native.process_table(proc)
    assert set(table)=={11,13} and table[11]['start']==123
    checks.append(dict(name='transient_proc_exit_between_listing_and_read_is_tolerated'))
    original_read=Path.read_text
    def proc_error(p,*args,**kwargs):
        if p==proc/'13'/'stat':raise OSError(errno.EIO,'injected I/O failure',str(p))
        return original_read(p,*args,**kwargs)
    try:
        with patch.object(Path,'read_text',proc_error):native.process_table(proc)
    except OSError as error:assert error.errno==errno.EIO
    else:raise AssertionError('Real /proc I/O error was ignored')
    checks.append(dict(name='true_proc_IO_error_remains_fatal'))

    path=scope('no_elapsed_cutoffs')
    owner=guard.AggregateOwner(path,execution_check=lambda:None,max_family_rss_bytes=512*2**20,max_seconds=None)
    owner.started-=365*86400
    for stage in ('generation','native',None):
        owner.set_stage(stage)
        owner.stage_started-=365*86400
        owner.check()
    owner._last['started_monotonic']-=10
    try:owner.check()
    except guard.GuardFailure as error:assert 'age' in str(error)
    else:raise AssertionError('Stale storage liveness check lost')
    owner.refresh()
    owner._last['charged_bytes']=guard.OPERATING
    try:owner.check()
    except guard.GuardFailure as error:assert 'operating' in str(error)
    else:raise AssertionError('Storage operating safety lost')
    owner.close()
    checks.append(dict(name='aggregate_and_stage_elapsed_time_do_not_terminate_healthy_work'))
    checks.append(dict(name='storage_liveness_and_resource_safety_checks_preserved'))

    path=scope('monitor_diagnostics')
    owner=guard.AggregateOwner(path,execution_check=lambda:None,max_family_rss_bytes=512*2**20,max_seconds=None)
    original_refresh=owner.refresh
    failed=threading.Event()
    def refresh():
        if threading.current_thread().name=='full-validation-storage':
            failed.set()
            raise OSError(errno.EIO,'injected read failure',str(path/'payload.bin'))
        return original_refresh()
    with patch.object(owner,'refresh',refresh),patch.object(guard,'SCAN_INTERVAL',.001):
        owner.start_monitor()
        assert failed.wait(2)
        owner._thread.join(timeout=2)
    assert owner._error['filename'].endswith('/payload.bin') and 'Traceback' in owner._error['traceback']
    try:owner.close()
    except guard.GuardFailure:pass
    else:raise AssertionError('Latched monitor failure lost at close')
    assert owner._lease.closed
    checks.append(dict(name='monitor_records_filename_errno_traceback_and_close_propagates_failure'))

    def no_wall_owner(path,**kwargs):
        kwargs['max_seconds']=None
        return guard.AggregateOwner(path,**kwargs)
    sentinel_script=base/'sentinel.sh'
    sentinel_script.write_text('sleep 60\n')
    sentinel=subprocess.Popen(['bash',str(sentinel_script)],start_new_session=True)
    sentinel_leader=native.process_table().get(sentinel.pid)
    assert sentinel_leader is not None
    try:
        with patch.object(fixtures,'AggregateOwner',no_wall_owner),patch.object(fixtures,'GuardFailure',guard.GuardFailure):
            def tiny_legacy_timeout(config):
                config['relate']['timeout_seconds_per_step']=.001
                return config
            with patch.object(native,'validate_native_config',side_effect=tiny_legacy_timeout):
                checks.append(fixtures.case(base,'legacy_per_step_timeout_not_enforced',native,'sleep 0.1\n','complete'))
            checks.append(fixtures.case(base,'immediate_native_exit',native,'printf immediate\\n\n','complete'))
            checks.append(fixtures.case(base,'stale_native_snapshot_rechecked',native,'printf ready > ready\nwhile [ ! -f release ]; do sleep 0.3; done\nprintf finished\\n\n','complete',stale=True))
            checks.append(fixtures.case(base,'native_nonzero_exit',native,'exit 7\n','failed',expected_return=7))
            checks.append(fixtures.case(base,'late_adopted_nonzero_exit',native,'(sleep 0.2; exit 9) &\nexit 0\n','failed',defer_zombie_reap=True))
            checks.append(fixtures.case(base,'persistent_owned_descendant_terminated',native,'(trap "" TERM; sleep 30) &\nexit 0\n','failed',drain_seconds=.35))
        assert sentinel.poll() is None
        checks.append(dict(name='unrelated_separate_session_process_preserved'))
    finally:
        family=native.owned_family(native.process_table(),sentinel_leader,{sentinel.pid:sentinel_leader})
        native.signal_owned(family,signal.SIGTERM)
        sentinel.wait(timeout=5)
        for _ in range(50):
            family=native.owned_family(native.process_table(),sentinel_leader,family)
            native.reap_adopted(family,sentinel.pid)
            if not family:break
            time.sleep(.02)
        assert not native.owned_family(native.process_table(),sentinel_leader,family)

    # Audit the executable comparison syntax, not only elapsed-time test values.
    for name in ('aggregate_guard.py','native_supervisor.py','worker.py','guardian.py'):
        tree=ast.parse((HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Compare) and any(isinstance(op,(ast.Lt,ast.LtE,ast.Gt,ast.GtE)) for op in node.ops):
                text=ast.unparse(node)
                forbidden_names={'MAX_SECONDS','max_seconds','deadline','monotonic_deadline'}
                assert not any((isinstance(child,ast.Name) and child.id in forbidden_names)
                    or (isinstance(child,ast.Attribute) and child.attr=='max_seconds')
                    for child in ast.walk(node)),text
    frozen=contracts.validate_plan()
    config,tools,chain=contracts.native_bindings()
    retained_ref,retained_bytes=contracts.retained_attempts()
    assert len(tools)==7 and frozen['recipe']['parameters']['length_bp']==200000000
    checks.append(dict(name='original_200Mb_recipe_seeds_scientific_source_and_seven_tools_verified',retained_bytes=retained_bytes))
    return checks

def main():
    checks=run()
    receipt=dict(passed=True,check_count=len(checks),checks=checks,
        simulation_executions=0,Relate_executions=0,GPU_executions=0,
        fixtures_only=True,original_run_untouched=True,
        source_refs=[ref(HERE/name) for name in ('aggregate_guard.py','native_supervisor.py','contracts.py','worker.py','guardian.py','test_cpu.py')])
    (HERE/'CPU_TESTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,check_count=len(checks),receipt=ref(HERE/'CPU_TESTS.json'))))

if __name__=='__main__':main()
