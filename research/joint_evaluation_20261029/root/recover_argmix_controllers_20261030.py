"""Replace two verified idle project controllers after the reviewed ARGMix repair."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import psutil
from arggnn.benchmarking.paper_campaign import runtime as r, processes

BASE = Path('research/joint_evaluation_20261029/root')
SPECS = (
    dict(name='finishing', old=BASE/'finish_records_v1/root_replacement_storage_v2.json',
         new_plan=BASE/'finish_plan_v5.json', module='research.joint_evaluation_20261029.root.finish_coordinator_storage_v2',
         log=BASE/'finish_records_v1/controller_argmix_recovery_v3.log',
         receipt=BASE/'finish_records_v1/root_replacement_argmix_v3.json'),
    dict(name='final_review', old=BASE/'final_review_records_v1/root_launch.json',
         new_plan=BASE/'final_review_plan_v2.json', module='research.joint_evaluation_20261029.root.final_review_continuation_v1',
         log=BASE/'final_review_records_v1/controller_argmix_recovery_v2.log',
         receipt=BASE/'final_review_records_v1/root_replacement_argmix_v2.json'),
)


def need(value, message):
    if not value:
        raise ValueError(message)


def snapshot(pid):
    p = Path('/proc') / str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, process_start_ticks=fields[19], state=fields[0],
                cwd=str((p/'cwd').resolve()), executable=str((p/'exe').resolve()),
                children=[dict(pid=c.pid, create_time=c.create_time()) for c in psutil.Process(pid).children(recursive=True)])


def verify_idle(old):
    current = snapshot(old['pid'])
    need(current['process_start_ticks'] == old['process_start_ticks'], 'Original controller identity changed')
    need(current['cwd'] == str(r.ROOT) and current['executable'] == str(Path(old['command'][0]).resolve()),
         'Exact project controller cwd and interpreter')
    need(not current['children'], 'Controller has active children, preserve it')
    need(current['state'] not in ('Z','X'), 'Original controller already exited')
    return current


def source_check(plan_path, expected):
    reference = r.ref(plan_path)
    need(reference == expected, 'Exact independently reviewed plan required')
    plan = r.read(reference)
    for source in plan['source_refs']:
        need(r.ref(source['path']) == source, 'Reviewed source changed')
    return reference


def replace(spec, emit, expected_plan_ref):
    old = r.read(r.ref(spec['old']))
    new_ref = source_check(spec['new_plan'], expected_plan_ref)
    for key in ('log','receipt'):
        need(not r.path(spec[key]).exists(), 'Preserve previous recovery attempt')
    fd = processes._pidfd_open(old['pid'])
    stopped = False
    try:
        before = verify_idle(old)
        emit(dict(action='verified_idle_controller', name=spec['name'], process=before, original_launch_ref=r.ref(spec['old'])))
        processes._pidfd_send(fd, signal.SIGSTOP)
        stopped = True
        for _ in range(40):
            current = verify_idle(old)
            if current['state'] in ('T','t'):
                break
            time.sleep(0.05)
        need(current['state'] in ('T','t') and not current['children'], 'Verify stopped controller is childless')
        emit(dict(action='stopped_verified_idle_controller', name=spec['name'], process=current))
        processes._pidfd_send(fd, signal.SIGTERM)
        processes._pidfd_send(fd, signal.SIGCONT)
        stopped = False
        ready, _, _ = select.select([fd], [], [], 20)
        need(bool(ready), 'Original idle controller exit not observed, do not launch replacement')
        emit(dict(action='original_controller_exit_observed', name=spec['name'], pid=old['pid'],
                  actual_wait_consumed=False, worker_signals_sent=[]))
    finally:
        if stopped:
            processes._pidfd_send(fd, signal.SIGCONT)
        os.close(fd)
    source_check(spec['new_plan'], expected_plan_ref)
    command = [str(r.ROOT/'.runtime/env/bin/python3.11'),'-B','-s','-u','-m',spec['module'],
               '--plan',str(spec['new_plan'])]
    with r.path(spec['log']).open('xb') as log:
        child = subprocess.Popen(command, cwd=r.ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    current = snapshot(child.pid)
    receipt = dict(pid=child.pid,process_start_ticks=current['process_start_ticks'],command=command,plan_ref=new_ref,
                   started_unix=time.time(),original_launch_ref=r.ref(spec['old']),
                   original_idle_exit_observed=True,original_actual_wait_consumed=False,
                   signals_to_verified_idle_controller=['SIGSTOP','SIGTERM','SIGCONT'],worker_signals_sent=[])
    with r.path(spec['receipt']).open('x') as stream:
        json.dump(receipt,stream,indent=2)
        stream.write('\n')
    emit(dict(action='replacement_started', name=spec['name'], receipt_ref=r.ref(spec['receipt'])))
    time.sleep(1)
    need(child.poll() is None, 'Replacement exited at startup, inspect its preserved log')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--review',required=True)
    args=parser.parse_args()
    review_ref=r.ref(args.review)
    review=r.read(review_ref)
    need(review['passed'] is True and review['handoff_source_ref'] == r.ref(__file__), 'Exact independent handoff-source review required')
    for spec in SPECS:
        old=r.read(r.ref(spec['old']))
        verify_idle(old)
        source_check(spec['new_plan'], review['new_plan_refs'][spec['name']])
    if not args.apply:
        print(json.dumps(dict(valid=True,controllers=2,signals_sent=0,review_ref=review_ref)),flush=True)
        return
    journal=r.path(BASE/'controller_recovery_20261030_events.jsonl')
    with journal.open('x') as stream:
        def emit(value):
            value['time_unix']=time.time()
            stream.write(json.dumps(value,sort_keys=True)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(value,sort_keys=True),flush=True)
        emit(dict(action='begin',review_ref=review_ref))
        for spec in SPECS:
            replace(spec,emit,review['new_plan_refs'][spec['name']])
        emit(dict(action='complete',workers_signalled=0))


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        print(json.dumps({'handoff_error':str(error),'error_type':type(error).__name__}),flush=True)
        raise
