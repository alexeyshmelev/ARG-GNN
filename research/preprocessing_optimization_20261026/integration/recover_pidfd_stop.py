"""Recover the recorded pre-signal native-pidfd API failure exactly once.

Reuse the controller's already reviewed portable pidfd syscalls. Preserve the
sealed migration, prior before_stop snapshot and approved592GiB variant.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from arggnn.benchmarking.paper_campaign import processes, runtime as r
from . import migrate as original
from . import migrate_resource592 as resource_ops

PINS=[
 {'path':'src/arggnn/benchmarking/paper_campaign/processes.py','sha256':'9860e688fe204196b72d5a868edd4a0df9287f349dce1beb39b48d6a94604f84'},
 {'path':'research/preprocessing_optimization_20261026/integration/migrate_resource592.py','sha256':'fcd0af3bb713d72f6703e5030dd79107c151abd38cfaef5b237c5691829e9177'},
 {'path':'research/preprocessing_optimization_20261026/integration/migration/before_stop.json','sha256':'bc0675b00a76bdcbd54c9db126323e27959523a72cb198504b1b078e226a28a2'}]
REVIEW=r.ROOT/'research/preprocessing_optimization_20261026/review/pidfd_recovery_review.json'
INTENT=original.WORK/'pidfd_recovery_intent.json'
need=original.require


def authenticate(require_review=True):
    resource_ops.authenticate()
    for ref in PINS:
        need(r.ref(ref['path'])==ref,'Recovery dependency changed: '+ref['path'])
    need(not hasattr(os,'pidfd_open') and not hasattr(signal,'pidfd_send_signal'),
         'Recovery is scoped to the confirmed missing-native-API failure')
    need(not (original.WORK/'stop_sent.json').exists() and not INTENT.exists(),
         'No prior successful signal or recovery dispatch may exist')
    if require_review:
        review=json.loads(REVIEW.read_text())
        need(review['status']=='pass' and review['operational_source_ref']==r.ref(__file__)
             and review['pinned_inputs']==PINS,'Current independent recovery review required')
    return r.read(PINS[2])


def check_saved_identity(before,state,task,row):
    need(row['status']=='running' and task==before['task']
         and state['plan_ref']==before['plan_ref']
         and state.get('plan_revision',0)==before['plan_revision'],
         'Original effective task and plan must still match before_stop')
    for field in ('pid','process_start_ticks','attempt','physical_gpu'):
        need(row[field]==before['row'][field],'Original worker identity changed: '+field)
    need(before['release_ref']==r.ref(original.RELEASE)
         and row['physical_gpu'] in (3,4,5),'Original release and authorized GPU required')


def validate_target(before):
    plan,state,task=original.current()
    row=state['tasks'][original.TASK]
    check_saved_identity(before,state,task,row)
    proc=Path('/proc')/str(row['pid'])
    fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
    need(fields[19]==str(row['process_start_ticks']) and int(fields[1])==state['controller_pid']
         and int(fields[2])==row['pid'] and int(fields[3])==row['pid'],
         'Exact original PID/start/controller/process-group/session required')
    need(proc.joinpath('cwd').resolve()==r.ROOT,'Original project working directory required')
    for ref in task['source_refs']:
        need(r.ref(ref['path'])==ref,'Original running source changed')
    return row


def recover():
    before=authenticate()
    snapshot=resource_ops.snapshot(True)
    snapshot_ref=resource_ops.save_snapshot(snapshot)
    need(snapshot['admissible_projection'],'Fresh592GiB admission projection rejected')
    row=validate_target(before)
    fd=processes._pidfd_open(row['pid'])
    try:
        rechecked=validate_target(before)
        need(all(rechecked[key]==row[key] for key in ('pid','process_start_ticks','attempt','physical_gpu')),
             'Original worker changed after pidfd acquisition')
        need(not (original.WORK/'stop_sent.json').exists(),'A stop receipt appeared before dispatch')
        # Exclusive intent makes an uncertain send fail closed on every retry.
        # A later attempt must inspect this record rather than send twice.
        intent_ref=original.exclusive(INTENT,dict(schema='portable_pidfd_stop_intent_v1',
            pid=row['pid'],process_start_ticks=row['process_start_ticks'],attempt=row['attempt'],
            signal='SIGTERM',task=original.TASK,before_stop_ref=PINS[2],
            operational_source_ref=r.ref(__file__),operational_review_ref=r.ref(REVIEW),
            pinned_inputs=PINS,admission_snapshot_ref=snapshot_ref,time_unix=time.time(),
            no_prior_stop_sent=True,dispatch_outcome_record='stop_sent.json'))
        processes._pidfd_send(fd,signal.SIGTERM)
    finally:
        os.close(fd)
    original.exclusive(original.WORK/'stop_sent.json',dict(pid=row['pid'],signal='SIGTERM',
        process_start_ticks=row['process_start_ticks'],time_unix=time.time(),scope=original.TASK,
        controller_signalled=False,other_workers_signalled=False,portable_pidfd=True,
        recovery_intent_ref=intent_ref,before_stop_ref=PINS[2]))
    print('Only the authenticated original TreeLoom leader received SIGTERM through portable pidfd. Await actual scheduler exit.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['inspect','recover'])
    args=parser.parse_args()
    if args.action=='inspect':
        before=authenticate(require_review=False)
        row=validate_target(before)
        print(json.dumps(dict(status='ready_for_review',pid=row['pid'],
            process_start_ticks=row['process_start_ticks'],native_pidfd_open=False,native_pidfd_send=False,
            before_stop_ref=PINS[2],no_signal_sent=True)))
    else:
        recover()
