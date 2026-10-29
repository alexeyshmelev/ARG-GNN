"""Root-launched idle-boundary handoff for the exact original controller.

Discovery is read-only. A controller is frozen only after both its saved state
and process tree are idle. The frozen tree is checked again before SIGTERM.
If the check fails, SIGCONT restores it immediately. No worker is signalled.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_support as support


def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]), start_ticks=fields[19])


def descendants(pid):
    output = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,stat='], text=True, timeout=15)
    rows = [line.split() for line in output.splitlines()]
    owned = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent, status in rows:
            if int(parent) in owned and int(child) not in owned:
                owned.add(int(child)); changed = True
    return sorted(int(child) for child, parent, status in rows if int(child) in owned and int(child) != pid and not status.startswith('Z'))


def load_original(config):
    return json.loads(r.path(config['original_state_path']).read_text())


def boundary_reason(state, config, children):
    support.verify_original(state, config)
    if any(row['status'] == 'running' for row in state['tasks'].values()):
        return 'Original controller still owns running work'
    if children:
        return 'Original controller has live descendants'
    for name in config['required_successful_boundary_tasks']:
        row = state['tasks'][name]
        if row['status'] != 'succeeded' or row.get('actual_wait') is not True or row.get('actual_returncode') != 0:
            return 'Awaiting original prediction and retirement receipts'
    gate = state['tasks'][config['required_claim_block_task']]
    if gate['status'] != 'blocked' or gate.get('error') != 'Output already exists, no overwrite' or 'pid' in gate:
        return 'Awaiting exact native claim boundary'
    return None


def check_config(config):
    for source in config['source_refs']:
        support.need(r.ref(source['path']) == source, 'Reviewed successor source changed: ' + source['path'])
    r.read(config['effective_plan_ref'])
    for sidecar in config['sidecars']:
        plan = r.read(sidecar['plan_ref'])
        state = json.loads(r.path(sidecar['state_path']).read_text())
        support.need(state['plan_ref'] == sidecar['plan_ref'], 'Sidecar owner changed')
        support.need(sidecar['state_path'] == plan['records_directory'] + '/state.json', 'Sidecar state path differs')


def attempt_idle_stop(config):
    """Return false without signals while busy; use pidfd to prevent PID reuse."""
    pid = config['prior_controller']['pid']
    expected = config['prior_controller']['start_ticks']
    current = identity(pid)
    support.need(current is not None and current['start_ticks'] == expected and current['state'] != 'Z', 'Original controller is no longer the sealed live process')
    state = load_original(config)
    reason = boundary_reason(state, config, descendants(pid))
    if reason:
        return dict(stopped=False, reason=reason, signals=[])
    # Binding before the signal, then validating the proc identity, avoids PID reuse.
    fd = processes._pidfd_open(pid)
    frozen = False
    signals = []
    try:
        current = identity(pid)
        support.need(current is not None and current['start_ticks'] == expected, 'Original identity changed before freeze')
        processes._pidfd_send(fd, signal.SIGSTOP)
        frozen = True
        signals.append('SIGSTOP')
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = identity(pid)
            if current is not None and current['state'] in ('T', 't'):
                break
            time.sleep(0.02)
        support.need(current is not None and current['state'] in ('T', 't') and current['start_ticks'] == expected, 'Original controller did not enter inspected stopped state')
        state = load_original(config)
        reason = boundary_reason(state, config, descendants(pid))
        if reason:
            return dict(stopped=False, reason='Race detected; resumed: ' + reason, signals=signals + ['SIGCONT'])
        # Confirm every historical worker session is empty while no launch can race.
        for row in state['tasks'].values():
            if 'pid' in row:
                support.need(row.get('actual_wait') is True and row.get('owned_process_group_absent') is True, 'Historical wait incomplete')
                support.need(not processes.session_members(row['pid']), 'Historical worker session remains live')
        before = r.ref(config['original_state_path'])
        processes._pidfd_send(fd, signal.SIGTERM)
        signals.append('SIGTERM')
        processes._pidfd_send(fd, signal.SIGCONT)
        frozen = False
        signals.append('SIGCONT')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = identity(pid)
            if current is None or current['state'] == 'Z':
                break
            time.sleep(0.1)
        support.need(current is None or current['state'] == 'Z', 'Idle controller did not exit; never escalate or signal workers')
        after = r.ref(config['original_state_path'])
        support.need(after == before, 'Original state changed during idle shutdown')
        support.need(not processes.session_members(pid), 'Original controller session still live')
        return dict(stopped=True, signals=signals, original_state_ref=after,
                    old_controller_wait_consumed=False, old_controller_exit_observed=True)
    finally:
        if frozen:
            processes._pidfd_send(fd, signal.SIGCONT)
        os.close(fd)


def run(config_path, inspect_only=False):
    config_ref = r.ref(config_path)
    config = r.read(config_ref)
    check_config(config)
    template = r.read(config['successor_template_ref'])
    state = load_original(config)
    reason = boundary_reason(state, config, descendants(config['prior_controller']['pid']))
    if inspect_only:
        from research.joint_evaluation_20261029.queue import successor_scheduler
        successor_scheduler.validate(template)
        plan = dict(template, handoff_config_ref=config_ref)
        accounting = support.commitment(plan)
        print(json.dumps(dict(ready_to_watch=True, boundary_reason=reason,
                              commitment=accounting, signals_sent=0, workers_launched=0), sort_keys=True))
        return
    directory = r.path(template['records_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'handoff.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    support.need(not (directory/'handoff.json').exists() and not (directory/'state.json').exists(), 'Prior handoff exists; inspect it rather than replay')
    previous = None
    while True:
        check_config(config)
        result = attempt_idle_stop(config)
        if result['stopped']:
            break
        if result['reason'] != previous or result['signals']:
            r.save(directory/'watch_status.json', dict(time_unix=time.time(), watcher_pid=os.getpid(), config_ref=config_ref, **result))
            print(json.dumps(result), flush=True)
            previous = result['reason']
        time.sleep(20)
    original = r.read(result['original_state_ref'])
    sealed = r.save(directory/'original_state_sealed.json', original)
    handoff_ref = r.save(directory/'handoff.json', dict(schema='idle_campaign_handoff_v1', time_unix=time.time(),
                       config_ref=config_ref, sealed_original_state_ref=sealed,
                       prior_controller=config['prior_controller'], **result))
    final = dict(template, handoff_ref=handoff_ref, handoff_config_ref=config_ref)
    final_path = directory/'plan.json'
    r.save(final_path, final)
    from research.joint_evaluation_20261029.queue import successor_scheduler
    outcome = successor_scheduler.run(final_path)
    print(json.dumps(outcome['counts']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    run(args.config, args.inspect)


if __name__ == '__main__':
    main()
