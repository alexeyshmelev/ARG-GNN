"""Independent idle-controller replacement tests; every process operation is mocked."""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import copy
import hashlib
import json
import os
import signal
from research.joint_evaluation_20261029.root import recover_argmix_controllers_20261030 as subject

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
REVIEW=ROOT/'research/joint_evaluation_20261029/review_finish/recovery_20261030'
SCRATCH=REVIEW/'handoff_fixtures_v1'
SCRATCH.mkdir(exist_ok=False)
passed=[]
old=dict(pid=2147483600,process_start_ticks='123456',command=[str(ROOT/'.runtime/env/bin/python3.11')])
base=dict(pid=old['pid'],process_start_ticks=old['process_start_ticks'],cwd=str(ROOT),executable=str(Path(old['command'][0]).resolve()),children=[],state='S')

def rejects(action):
    try:action()
    except ValueError:return
    raise AssertionError('Unsafe input accepted')

with patch.object(subject,'snapshot',return_value=base):assert subject.verify_idle(old)==base
passed.append('same_pid_start_project_cwd_executable_and_childless_identity_accepted')
for key,value in [('process_start_ticks','wrong'),('cwd','/outside'),('executable','/wrong'),('children',[dict(pid=2)]),('state','Z')]:
    with patch.object(subject,'snapshot',return_value=dict(base,**{key:value})):
        rejects(lambda:subject.verify_idle(old))
    passed.append('identity_rejects_'+key)

source=SCRATCH/'source.txt';source.write_text('immutable fixture source\n')
plan=SCRATCH/'plan.json';plan.write_text(json.dumps(dict(source_refs=[subject.r.ref(source)])))
plan_ref=subject.r.ref(plan)
assert subject.source_check(plan,plan_ref)==plan_ref
rejects(lambda:subject.source_check(plan,dict(plan_ref,sha256='0'*64)))
passed.append('exact_reviewed_plan_hash_required')
source.write_text('tampered fixture source\n')
rejects(lambda:subject.source_check(plan,plan_ref))
passed.append('plan_source_hash_changes_rejected')
source.write_text('immutable fixture source\n')

class Child:
    pid=2147483601
    def poll(self):return None

def replacement_case(name, *, initial_bad=False, stopped_child=False, exit_observed=True, changed_plan=False, old_log=False):
    folder=SCRATCH/name;folder.mkdir()
    old_path=folder/'old_launch.json';old_path.write_text(json.dumps(old))
    spec=dict(name='synthetic',old=old_path,new_plan=plan,module='research.joint_evaluation_20261029.root.final_review_continuation_v1',log=folder/'new.log',receipt=folder/'new.json')
    if old_log:spec['log'].write_text('preserve prior attempt\n')
    signals=[];events=[]
    snapshots=[dict(base,children=[dict(pid=2)]) if initial_bad else dict(base),dict(base,state='T',children=[dict(pid=2)] if stopped_child else []),dict(base,pid=Child.pid,process_start_ticks='new')]
    source_checks=[]
    def check(path,expected):
        source_checks.append((path,expected))
        if changed_plan and len(source_checks)==2:raise ValueError('Plan changed before spawn')
        return plan_ref
    def take_snapshot(pid):return snapshots.pop(0)
    def send(fd,sig):assert fd==987654;signals.append(sig)
    before=old_path.read_bytes()
    with patch.object(subject,'source_check',side_effect=check), patch.object(subject,'snapshot',side_effect=take_snapshot), \
         patch.object(subject.processes,'_pidfd_open',return_value=987654) as opened, \
         patch.object(subject.processes,'_pidfd_send',side_effect=send), patch.object(subject.os,'close') as closed, \
         patch.object(subject.select,'select',return_value=([987654] if exit_observed else [],[],[])), \
         patch.object(subject.subprocess,'Popen',return_value=Child()) as spawn, patch.object(subject.time,'sleep'):
        failure=None
        try:result=subject.replace(spec,events.append,plan_ref)
        except ValueError as error:failure=str(error);result=None
    assert old_path.read_bytes()==before
    if old_log:
        assert failure and not signals and opened.call_count==0 and spawn.call_count==0
        assert spec['log'].read_text()=='preserve prior attempt\n'
    elif initial_bad:
        assert failure and not signals and spawn.call_count==0
    elif stopped_child:
        assert failure and signals==[signal.SIGSTOP,signal.SIGCONT] and spawn.call_count==0
    elif not exit_observed or changed_plan:
        assert failure and signals==[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT] and spawn.call_count==0
    else:
        assert not failure and signals==[signal.SIGSTOP,signal.SIGTERM,signal.SIGCONT] and spawn.call_count==1
        assert len(source_checks)==2 and all(expected==plan_ref for path,expected in source_checks)
        assert result['plan_ref']==plan_ref and result['original_actual_wait_consumed'] is False
        assert result['original_idle_exit_observed'] is True and result['worker_signals_sent']==[]
        assert spawn.call_args.kwargs['start_new_session'] is True
        assert [event['action'] for event in events]==['verified_idle_controller','stopped_verified_idle_controller','original_controller_exit_observed','replacement_started']
    passed.append(name)

replacement_case('mocked_idle_stop_recheck_exit_and_explicit_replacement_preserve_old_receipt')
replacement_case('active_child_before_stop_prevents_all_signals',initial_bad=True)
replacement_case('child_appearing_before_stop_is_resumed_without_termination',stopped_child=True)
replacement_case('unobserved_old_exit_prevents_replacement',exit_observed=False)
replacement_case('plan_changed_after_stop_prevents_replacement',changed_plan=True)
replacement_case('previous_recovery_log_prevents_replay',old_log=True)

def ref(path):
    data=path.read_bytes()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
receipt=dict(schema='independent_idle_controller_handoff_checks_v1',passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),
             handoff_source_ref=ref(Path(subject.__file__)),test_source_ref=ref(Path(__file__)),checks=passed,
             all_process_signals_and_launches_mocked=True,actual_signals_sent=0,actual_production_launched=False,truth_opened=False,
             actual_plan_bindings_pending_final_review=True)
out=REVIEW/'handoff_checks_result_v1.json'
with out.open('x') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(dict(passed=True,checks=len(passed),receipt_ref=ref(out)),sort_keys=True))
