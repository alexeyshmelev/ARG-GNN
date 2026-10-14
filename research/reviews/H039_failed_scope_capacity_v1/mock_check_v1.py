"""One in-memory failed-H039 retirement + future650MB reserve replay.

Reads only pinned source/original JSON/log evidence. Journal, accounting state,
PID observation, inventory and append are mocked. No actual admission/execution.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
from types import SimpleNamespace

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py'
REPORT=ROOT/'research/reviews/H039_failed_scope_capacity_v1/mock_check.json'


def main():
    spec=importlib.util.spec_from_file_location('h039_failed_capacity_author_check',ROOT/SOURCE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    scope=ROOT/m.OUTPUT
    original_bound=m._bound
    own=json.loads((ROOT/m.FOLDER/'storage.json').read_text())['reservation']
    stamp=m._utc()
    observed=lambda pid:dict(pid=pid,absent=True,method='targeted os.kill(pid,0) ESRCH',checked_utc=stamp)
    terminal_ref=dict(path='research/reviews/H039_failed_scope_capacity_v1/mock_terminal.json',sha256='a'*64)
    barrier_ref=dict(path='research/reviews/H039_failed_scope_capacity_v1/mock_barrier.json',sha256='b'*64)
    launch_ref=dict(path=m.FOLDER+'launch.json',sha256=m.EVIDENCE['launch'][1])
    request=m.retirement_request_v1(terminal_ref)
    fixtures={
      terminal_ref['path']:dict(kind='H039_v1_failed_scope_terminal_observation_v1',root_authorizer='root_agent',
        key=m.KEY,evidence_refs=request['evidence_refs'],pids=m.PIDS,observations=[observed(p) for p in m.PIDS],
        all_original_and_resume_capabilities_revoked=True),
      barrier_ref['path']:dict(kind='H039_failed_scope_capacity_activation_barrier_v1',root_authorizer='root_agent',
        all_old_guard_consumers_and_supervisors_terminal=True,all_old_guard_future_writes_and_relaunch_revoked=True,
        root_owned_process_roster_complete=True,pids=m.PIDS,observations=[observed(p) for p in m.PIDS],launch_refs=[launch_ref]),
    }
    m._bound=lambda r:copy.deepcopy(fixtures[r['path']]) if r['path'] in fixtures else original_bound(r)
    proof=m._failed_H039(request,own)
    assert proof['total_additional_optimizer_updates']==0 and proof['launcher_returncode']==1 and proof['worker_returncode']==1
    source_sha='483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e'
    launcher_sha='a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3'
    bindings=dict(implementation=dict(path=m.budget.SOURCE_REL,sha256=source_sha),
      launcher=dict(path=m.budget.LAUNCHER_REL,sha256=launcher_sha),
      cpu_overlay=dict(path=m.old.CPU_SOURCE,sha256=m.old.CPU_SHA256),
      terminal_overlay=dict(path=m.old.SOURCE,sha256=m.previous.previous.OLD_SHA256),
      terminal_success_overlay=dict(path=m.previous.previous.SOURCE,sha256=m.previous.PREVIOUS_SHA256),
      terminal_retired_overlay=dict(path=m.previous.SOURCE,sha256=m.PREVIOUS_SHA256),H039_failed_scope_overlay=m._ref(m.SOURCE))
    retired={k:v for k,v in bindings.items() if k!='H039_failed_scope_overlay'}
    success={k:v for k,v in retired.items() if k!='terminal_retired_overlay'}
    terminal={k:v for k,v in success.items() if k!='terminal_success_overlay'}
    def authority(operation):
        value=dict(kind='H039_failed_scope_capacity_root_admission_v1',root_authorizer='root_agent',active=True,
          operation=operation,accounting_mode=m.MODE,bindings=bindings,all_budget_writers_use_scheduler_mutex=True,
          all_consumers_and_launchers_use_H039_failed_scope_guard=True,
          all_old_guard_consumers_and_supervisors_terminal_before_activation=True,
          historical_records_mutation_authorized=False,global_rebase_authorized=False,
          scientific_execution_authorized=False,GPU_authorized=False,activation_barrier_ref=barrier_ref)
        if operation=='retire_failed_H039_v1':
            value.update(retirement=request,scope_write_authority_revoked_for_all_cooperative_project_jobs=True,
              closed_scope_future_writes_and_relaunch_revoked=True,
              all_original_and_resume_capabilities_for_retired_scope_revoked=True,
              preserve_all_failed_scope_artifacts=True,external_NFS_writers_outside_observation=True)
        return value
    measurement=dict(conservative_scope_bytes=35000000,entries=50,scope_paths=[str(scope)],stat_only=True)
    retirement_value=dict(request=request,reservation_sha256=m.budget.digest(own),measurement=measurement,
        failure_proof=proof,retired_utc=stamp,future_writes_and_relaunch_revoked=True)
    new_request=dict(key='mock/future650MB',output=str(ROOT/'data/mock_H039_capacity_future650MB'),auxiliary_scopes=[],
                     reserved_bytes=650000000,consumer_bindings=[*bindings.values(),m.V9_CONSUMER])
    events=[];raws=[];tail=None
    def event(phase,operation,value,auth):
        nonlocal tail
        authpath=f'research/reviews/H039_failed_scope_capacity_v1/mock_authority{len(events)+1}.json'
        auth['expected_before']={'terminal_overlay_tail_sha256':tail}
        fixtures[authpath]=auth
        row=dict(schema='project_budget_terminal_overlay_event_v1',sequence=len(events)+1,previous_sha256=tail,
          bindings=phase,operation=operation,authority=dict(path=authpath,sha256='c'*64),value=value)
        raw=m.budget.v4.encoded(row)+b'\n';tail=hashlib.sha256(raw).hexdigest();events.append(row);raws.append((raw,tail))
    for phase in (terminal,success,retired):event(phase,'reserve',{},dict(expected_before={}))
    event(bindings,'retire_failed_H039_v1',retirement_value,authority('retire_failed_H039_v1'))
    auth=authority('reserve');auth['reservation']=new_request
    newauthref=dict(path='research/reviews/H039_failed_scope_capacity_v1/mock_authority5.json',sha256='c'*64)
    event(bindings,'reserve',{**new_request,'root_admission':newauthref},auth)
    class Journal:
        def exists(self):return True
        def is_dir(self):return True
        def lstat(self):return SimpleNamespace(st_dev=1,st_ino=1,st_mtime_ns=1,st_ctime_ns=1)
        def iterdir(self):return [Path(f'{i:06d}.json') for i in range(1,len(raws)+1)]
        def __truediv__(self,name):return Path(m.JOURNAL)/name
    real_canonical=m.budget.v4.canonical
    m.budget.v4.canonical=lambda root,p:Journal() if str(p)==m.JOURNAL else real_canonical(root,p)
    real_file_bytes=m.budget.v4.file_bytes
    real_read_json=m.budget.v4.read_json
    m.budget.v4.file_bytes=lambda root,p:(raws[int(Path(p).stem)-1]
        if str(Path(p).parent).endswith(m.JOURNAL) else real_file_bytes(root,p))
    m.budget.v4.read_json=lambda root,p:({'jobs':{}},'mock-legacy') if str(p)==m.budget.v4.LEGACY else real_read_json(root,p)
    for module in (m.old,m.previous.previous,m.previous):
        module._authority=lambda r,op,b,*,fresh:m._bound(r)
    def forbidden(*a,**k):raise AssertionError('Live action attempted in source-only mock')
    m.budget.v4.measure_scopes=forbidden;m.budget.v4.mutex=forbidden;m._append=forbidden
    m._frozen=lambda:dict(_fingerprint=lambda paths:('stable_mock',),_absent=forbidden)
    base=dict(projected_bytes=949900000000,limit_bytes=950000000000,hard_limit_bytes=1000000000000,
      legacy_ledger_sha256='mock-legacy',supplemental_ledger_sha256='mock-supplemental',
      journal_tail_sha256='mock-v9',cpu_overlay_tail_sha256='mock-cpu',terminal_overlay_tail_sha256=raws[2][1])
    global _CACHE
    _CACHE=(None,None,None,())
    def base_reader(*args,**kwargs):return {m.KEY:own},set(),base,retired,events[:3]
    def clone(method,**changes):
        assert method is m.previous._state_locked
        prior=changes['_mixed_events'](retired)
        assert [len(x) for x in prior[:3]]==[1,1,1] and prior[3]==raws[2][1]
        return base_reader
    m._clone=clone
    reservations,closed,view,actual_bindings,replayed=m._state_locked(source_sha,launcher_sha,fresh=True)
    assert replayed==events and replayed[:3]==events[:3] and actual_bindings==bindings
    assert m.KEY in closed and reservations[m.KEY]==own and reservations['mock/future650MB']['reserved_bytes']==650000000
    assert view['projected_bytes']==940585131072
    assert view['H039_failed_scope_allowance_credit_bytes']==9965000000
    assert view['H039_failed_scope_metadata_charge_bytes']==131072
    # The next old-guard event must be rejected even with an otherwise valid chain.
    event(retired,'reserve',{},dict(expected_before={}))
    try:m._mixed_events(bindings)
    except ValueError as error:assert 'no old writer after activation' in str(error)
    else:raise AssertionError('Late old writer accepted')
    drift=copy.deepcopy(request);drift['key']='H039/directed-clade-exchange-full-panel-seed101-v2'
    try:m._failed_H039(drift,own)
    except ValueError:pass
    else:raise AssertionError('H039 v2 incorrectly accepted')
    report=dict(kind='H039_failed_scope_capacity_source_mock_check_v1',status='PASS',source_ref=bindings['H039_failed_scope_overlay'],
      exact_original_failure_evidence_verified=True,total_additional_optimizer_updates_verified=0,
      immutable_mock_prefix_phases=3,failed_H039_v1_credit_replayed=True,future650MB_reserve_replayed=True,
      synthetic_retained_bytes=35000000,synthetic_start_projected_bytes=949900000000,
      synthetic_end_projected_bytes=view['projected_bytes'],synthetic_credit_bytes=9965000000,
      synthetic_metadata_charge_bytes=131072,old_writer_after_activation_rejected=True,H039_v2_rejected=True,
      live_PID_observations=0,live_scope_inventories=0,actual_journal_operations=0,actual_admission=False,
      actual_scientific_execution=False,actual_retained_bytes_measured=False)
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':main()
