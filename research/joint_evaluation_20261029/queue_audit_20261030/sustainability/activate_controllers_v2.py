"""Apply reviewed queue repairs only after freezing and rechecking an idle owner.

No scientific worker receives a signal. Observing an orphaned controller exit is
not reported as a consumed subprocess wait. All replacement argv remain visible.
"""
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

BASE = Path('research/joint_evaluation_20261029/queue_audit_20261030/sustainability')
NAMES = {'treeloom_receipts', 'native_argmix', 'finishing', 'final_review'}


def need(value, message):
    if not value:
        raise ValueError(message)


def identity(old, target, *, frozen=False):
    process = psutil.Process(old['pid'])
    path = Path('/proc') / str(old['pid'])
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    need(fields[19] == old['process_start_ticks'], 'Controller PID was reused')
    need((path / 'cwd').resolve() == r.ROOT, 'Controller is outside the project')
    interpreter = Path(target['old_interpreter']).resolve()
    need(interpreter == (r.ROOT / '.runtime/env/bin/python3.11').resolve(), 'Project interpreter required')
    if 'command' in old:
        need(Path(old['command'][0]).resolve() == interpreter, 'Old launch interpreter differs')
    need((path / 'exe').resolve() == interpreter, 'Controller interpreter differs')
    thread_states = {}
    for entry in (path / 'task').iterdir():
        try:
            thread_states[entry.name] = (entry / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except FileNotFoundError:
            if frozen:
                raise ValueError('Thread identity changed during frozen inspection')
    if frozen:
        need(thread_states and all(state in ('T', 't') for state in thread_states.values()), 'All controller threads must be stopped')
    children = [child.pid for child in process.children(recursive=True)]
    need(not children, 'Controller has active children; preserve all work')
    need(fields[0] not in ('Z', 'X'), 'Controller already exited; inspect before adoption')
    return {'pid':old['pid'], 'process_start_ticks':fields[19], 'state':fields[0], 'children':children, 'thread_states':thread_states}


def check(target):
    old = r.read(target['old_launch_ref'])
    for reference in target['source_refs'] + target['configuration_refs']:
        r.read(reference) if reference['path'].endswith('.json') else need(r.ref(reference['path']) == reference, 'Source changed')
    command = target['argv']
    need(isinstance(command, list) and len(command) >= 6, 'Explicit replacement argv required')
    need(r.path(command[0]).is_file() and command[1:3] == ['-B', '-m'], 'Reviewed project Python module required')
    need(command[3].startswith('research.joint_evaluation_20261029.'), 'Project queue module required')
    need(not any(value in command for value in ('-c', '-EncodedCommand')), 'Plaintext module execution only')
    return old


def activate(name, target, manifest_ref, review_ref, apply):
    old = check(target)
    before = identity(old, target)
    folder = r.path(BASE / 'activations')
    receipt_path = folder / (name + '.json')
    journal_path = folder / (name + '.events.jsonl')
    log_path = folder / (name + '.log')
    need(not any(path.exists() for path in (receipt_path, journal_path, log_path)), 'Preserve existing activation, no replay')
    if not apply:
        return {'target':name, 'valid':True, 'old_identity':before, 'signals_sent':0}
    folder.mkdir(parents=True, exist_ok=True)
    with journal_path.open('x') as journal:
        def event(action, **values):
            row = dict(action=action, time_unix=time.time(), **values)
            journal.write(json.dumps(row, sort_keys=True) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        fd = processes._pidfd_open(old['pid'])
        paused = False
        try:
            before = identity(old, target)
            event('verified_childless', identity=before)
            processes._pidfd_send(fd, signal.SIGSTOP)
            paused = True
            for _ in range(40):
                current = identity(old, target)
                if current['state'] in ('T', 't'):
                    break
                time.sleep(0.05)
            need(current['state'] in ('T', 't'), 'Controller did not pause')
            current = identity(old, target, frozen=True)
            event('frozen_childless', identity=current)
            # Do not resume user code: a deferred Python signal handler could
            # otherwise fork a worker between the frozen check and termination.
            processes._pidfd_send(fd, signal.SIGKILL)
            paused = False
            ready, _, _ = select.select([fd], [], [], 20)
            need(bool(ready), 'Exit not observed; do not launch another owner')
            event('old_exit_observed', actual_wait_consumed=False, worker_signals_sent=[])
        finally:
            if paused:
                processes._pidfd_send(fd, signal.SIGCONT)
            os.close(fd)
        check(target)
        env = os.environ.copy()
        env.update(PYTHONPATH=str(r.ROOT / 'src') + ':' + str(r.ROOT),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        with log_path.open('xb') as log:
            child = subprocess.Popen(target['argv'], cwd=r.ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        fields = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        receipt = dict(schema='reviewed_idle_queue_controller_replacement_v1', target=name,
            pid=child.pid, process_start_ticks=fields[19], command=target['argv'],
            old_launch_ref=target['old_launch_ref'], manifest_ref=manifest_ref, review_ref=review_ref,
            started_unix=time.time(), old_exit_observed=True, old_actual_wait_consumed=False,
            worker_signals_sent=[], controller_signals=['SIGSTOP', 'SIGKILL'])
        r.save(receipt_path, receipt)
        event('replacement_started', receipt_ref=r.ref(receipt_path))
        time.sleep(1)
        need(child.poll() is None, 'New controller exited; preserve log and inspect')
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--target', required=True, choices=sorted(NAMES))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest_ref = r.ref(args.manifest)
    manifest = r.read(manifest_ref)
    review_ref = r.ref(args.review)
    review = r.read(review_ref)
    need(review['passed'] is True and review['launcher_ref'] == r.ref(__file__) and
         review['manifest_ref'] == manifest_ref, 'Exact independent activation review required')
    need(set(manifest['targets']) <= NAMES and args.target in manifest['targets'], 'Reviewed target required')
    print(json.dumps(activate(args.target, manifest['targets'][args.target], manifest_ref, review_ref, args.apply), sort_keys=True))


if __name__ == '__main__':
    main()
