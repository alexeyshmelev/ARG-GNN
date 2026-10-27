"""Actual tiny shell exit-race tests, isolated from all scientific/native jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from research.efficiency_20261027.integration_candidate.european_short_development_val import native_supervisor_v3 as old
from research.efficiency_20261027.integration_candidate.european_short_development_val import native_supervisor_v4 as new
from research.efficiency_20261027.integration_candidate.european_short_development_val.aggregate_guard_v3 import AggregateOwner, GuardFailure

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/efficiency_20261027/execution'

def ref(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def case(base,name,module,source,expected,*,stale=False,drain_seconds=5.0,expected_return=0):
    scope=base/name
    scope.mkdir()
    (scope/'tmp').mkdir()
    script=scope/'fixture.sh'
    script.write_text(source)
    owner=AggregateOwner(scope,execution_check=lambda:None,max_family_rss_bytes=512*2**20,max_seconds=30)
    supervisor=module.NativeSupervisor(owner,reviewed_native_tools={str(script):ref(script)['sha256']})
    original_table=module.process_table
    injected=[]
    signals=[]
    original_send=module._pidfd_send
    def tracked_send(fd,sig):
        signals.append(int(sig))
        return original_send(fd,sig)
    def delayed_table():
        table=original_table()
        launch=scope/'convert_vcf.launch.json'
        if stale and not injected and launch.exists() and (scope/'ready').exists():
            leader=json.loads(launch.read_text())['leader']['pid']
            if any(row['session']==leader and row['pid']!=leader and row['state']!='Z' for row in table.values()):
                (scope/'release').write_text('release\n')
                time.sleep(0.8)
                injected.append(True)
        return table
    config=json.loads((ROOT/'configs/benchmarks/ap200mb_native_runtime_v2.json').read_text())
    journal=[]
    error=None
    started=time.monotonic()
    with patch.object(module,'process_table',side_effect=delayed_table), patch.object(module,'_pidfd_send',side_effect=tracked_send), patch.object(module,'TERMINAL_DRAIN_SECONDS',drain_seconds,create=True):
        try:
            supervisor(['bash','-x',str(script)],scope,journal,'convert_vcf',config,owner)
        except GuardFailure as caught:
            error=str(caught)
        finally:
            owner.close()
    assert len(journal)==1
    record=journal[0]
    assert record['status']==expected,(name,record)
    assert record['returncode']==expected_return,(name,record['returncode'])
    assert record['terminal_family_absent'] is True,(name,record)
    assert any(row.get('method')=='Popen.wait' for row in record['waits'])
    assert record['complete_log'] is True,(name,record)
    if expected=='complete':
        assert error is None and not signals,(name,error,signals)
    else:
        assert error is not None
    if stale:assert injected,[name,'stale window not exercised']
    assert not module.owned_family(original_table(),record['leader'],{row['pid']:row for row in record['tracked_family']}),(name,'family survived')
    return dict(name=name,status=record['status'],expected_status=expected,returncode=record['returncode'],
        stale_snapshot_injected=bool(injected),signals_sent=signals,terminal_family_absent=True,
        elapsed_seconds=time.monotonic()-started,terminal_ref=ref(scope/'convert_vcf.terminal.json'),
        script_ref=ref(script),stdout=(scope/'convert_vcf.stdout.log').read_text(),waits=record['waits'])

def run(base):
    base.mkdir()
    sentinel_script=base/'unrelated_sentinel.sh'
    sentinel_script.write_text('sleep 30\n')
    sentinel=subprocess.Popen(['bash',str(sentinel_script)],cwd=base,start_new_session=True)
    checks=[]
    try:
        race='printf ready > ready\nwhile [ ! -f release ]; do sleep 0.3; done\nprintf finished\\n\n'
        checks.append(case(base,'old_stale_snapshot_reproduced',old,race,'failed',stale=True))
        checks.append(case(base,'new_stale_snapshot_rechecked',new,race,'complete',stale=True))
        checks.append(case(base,'immediate_leader_exit',new,'printf immediate\\n\n','complete'))
        benign='(sleep 0.3; printf late_stdout\\n; printf late_stderr\\n >&2) &\nprintf leader_done\\n\nexit 0\n'
        row=case(base,'natural_descendant_and_pipe_drain',new,benign,'complete')
        assert 'late_stdout' in row['stdout'] and any('wait_status' in wait for wait in row['waits'])
        checks.append(row)
        checks.append(case(base,'persistent_owned_descendant_rejected',new,'(trap "" TERM; sleep 30) &\nexit 0\n','failed',drain_seconds=0.35))
        checks.append(case(base,'nonzero_leader_rejected',new,'printf error\\n >&2\nexit 7\n','failed',expected_return=7))
        checks.append(case(base,'nonzero_adopted_descendant_rejected',new,'(sleep 0.2; exit 9) &\nexit 0\n','failed'))
        assert sentinel.poll() is None,'Unrelated separate-session fixture was interrupted'
        return checks
    finally:
        # This separate fixture is ours; finish it only after checking isolation.
        # Its shell has one sleep child; signal exact generations, then wait.
        table=new.process_table()
        leader=table.get(sentinel.pid)
        if leader is not None:
            family=new.owned_family(table,leader,{sentinel.pid:leader})
            new.signal_owned(family,new.signal.SIGTERM)
            sentinel.wait(timeout=5)
            for _ in range(50):
                table=new.process_table()
                family=new.owned_family(table,leader,family)
                new.reap_adopted(family,sentinel.pid)
                if not family:break
                time.sleep(0.02)
            assert not new.owned_family(new.process_table(),leader,family),'Sentinel family survived fixture cleanup'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--fixture-root',default=str(HERE/'native_terminal_v4_fixtures_attempt2'))
    parser.add_argument('--receipt',default=str(HERE/'NATIVE_TERMINAL_V4_CPU.json'))
    args=parser.parse_args()
    base=Path(args.fixture_root)
    destination=Path(args.receipt)
    assert base.is_relative_to(HERE) and destination.is_relative_to(HERE)
    checks=run(base)
    report=dict(schema='European_native_terminal_race_actual_CPU_v4',passed=True,checks_passed=len(checks),checks=checks,
        source_refs=[ref(Path(module.__file__)) for module in (old,new)]+[ref(Path(__file__))],
        actual_tiny_shell_fixtures=True,scientific_payloads_opened=False,native_tool_executions=0,GPU_calls=0,
        unrelated_separate_session_fixture_untouched=True,all_owned_fixture_families_reaped=True,
        production_terminal_drain_seconds=new.TERMINAL_DRAIN_SECONDS,
        persistent_descendant_test_only_drain_seconds=0.35,
        limitation='Actual shell lifecycle checks do not replace full-pipeline admission or scientific qualification.')
    with destination.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(receipt=ref(destination),passed=True,checks_passed=len(checks))))

if __name__=='__main__':main()
