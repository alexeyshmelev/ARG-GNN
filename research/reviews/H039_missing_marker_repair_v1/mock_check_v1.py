"""Focused in-memory marker-reader replay; no live state/PID/inventory/journal actions."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import copy
import hashlib
import importlib.util
import json

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/reviews/H039_missing_marker_repair_v1'


def main():
    spec=importlib.util.spec_from_file_location('marker_repair_focused_check',HERE/'marker_repair_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original_bound=m._bound
    repair_ref=dict(path='research/reviews/H039_missing_marker_repair_v1/mock_repair.json',sha256='a'*64)
    barrier_ref=dict(path='research/reviews/H039_missing_marker_repair_v1/mock_barrier.json',sha256='b'*64)
    stamp=datetime.now(timezone.utc).isoformat()
    barrier=dict(kind='H039_missing_marker_old_guard_terminal_barrier_v1',root_authorizer='root_agent',
        known_root_owned_guard_pids=[999999],observations=[dict(pid=999999,absent=True,
        method='targeted os.kill(pid,0) ESRCH',checked_utc=stamp)],roster_complete=True,
        all_old_guard_consumers_and_supervisors_terminal=True,all_old_guard_future_writes_and_relaunch_revoked=True)
    authority=m.repair_authority_draft_v1(barrier_ref)
    authority.update(active=True,expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    fixtures={repair_ref['path']:authority,barrier_ref['path']:barrier}
    m._bound=lambda r:copy.deepcopy(fixtures[r['path']]) if r['path'] in fixtures else original_bound(r)
    m.configure_repair_v1(repair_ref)
    _,own=m._repair();before=copy.deepcopy(own)
    paths=m._paths(own)
    assert own==before and m.budget.digest(own)==m.RESERVATION_SHA and paths==[Path(own['output'])]
    assert all(r['path']!=m.MARKER_REF['path'] for r in own['consumer_bindings'])
    drift=copy.deepcopy(own);drift['reserved_bytes']+=1
    try:m._paths(drift)
    except ValueError:pass
    else:raise AssertionError('Malformed reservation drift accepted')
    cpu,retired=m._readers()
    assert cpu['_state_locked'].__globals__['_paths'] is m._paths
    assert retired.old._state_locked.__globals__['_paths'] is m._paths
    assert retired.previous._state_locked.__globals__['_paths'] is m._paths
    assert retired._state_locked.__globals__['_paths'] is m._paths
    def forbidden(*args,**kwargs):raise AssertionError('Live action attempted')
    m.budget.v4.mutex=forbidden;m.budget.v4.measure_scopes=forbidden;m._append=forbidden
    cpu['_absent']=forbidden;cpu['_fingerprint']=lambda extra:('stable_mock',)
    cpu['_load_events']=lambda bindings:([],None)
    base=dict(projected_bytes=900000000000,limit_bytes=950000000000,hard_limit_bytes=1000000000000,
        legacy_ledger_sha256='mock-legacy',supplemental_ledger_sha256='mock-supplemental',
        journal_tail_sha256=None,cpu_overlay_tail_sha256=None,terminal_overlay_tail_sha256=None)
    cpu['_ORIGINAL_STATE']=lambda *args,**kwargs:({'jobs':{}},{'reservations':{}},{},{},base,[])
    m.budget.reviewed=lambda *args,**kwargs:(None,None)
    m.budget.v4.read_json=lambda *args,**kwargs:({'jobs':{}},'mock-legacy')
    # Existing prefix/chain parser is preserved in source. This focused replay
    # supplies one original immutable retired reservation event in memory.
    event=dict(operation='reserve',authority=m.AUTHORITY_REF,value=own)
    retired._state_locked.__globals__['_mixed_events']=lambda bindings:([],[],[event],'mock-terminal-tail')
    source_sha='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
    launcher_sha='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
    reservations,closed,view,bindings,events=m._state_locked(source_sha,launcher_sha,fresh=True)
    assert reservations[m.KEY]==before and events==[event] and m.KEY not in closed
    assert view['projected_bytes']==900015065536
    assert view['malformed_reservation_retained_allowance_bytes']==15000000
    assert view['marker_repair_accounting_credit_bytes']==0
    assert bindings['terminal_retired_overlay']==m.PREVIOUS_REF and len(bindings)==6
    try:m.check_storage_locked_v1(dict(reservation=own))
    except ValueError as e:assert 'fully charged and revoked' in str(e)
    else:raise AssertionError('Historical malformed scope allowed to execute')
    result=dict(kind='H039_missing_marker_source_mock_check_v1',status='PASS',
        source_ref=m._ref(m.SOURCE),exact_original_reservation_and_authority_verified=True,
        temporary_parser_marker_only=True,stored_reservation_unchanged=True,
        historical_journal_bindings_unchanged=True,retained_allowance_bytes=15000000,
        accounting_credit_bytes=0,synthetic_projected_bytes=view['projected_bytes'],
        synthetic_original_projected_bytes=900000000000,metadata_charge_preserved=65536,
        malformed_scope_execution_rejected=True,one_case_drift_rejected=True,
        live_scope_inventory=0,live_PID_observations=0,actual_journal_operations=0,
        actual_admission=False,actual_execution=False,root_repair_authority_activated=False)
    HERE.joinpath('mock_check.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
