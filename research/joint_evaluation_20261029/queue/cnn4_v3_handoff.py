"""Read-only inspection or root-authorized exact idle CNN controller handoff.

Signals only the identity-checked controller after two idle checks; a raced
worker causes immediate resume. No existing worker, source, plan or row changes.
"""
import argparse
import fcntl
import json
import os
import signal
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue.successor_v2_handoff import identity, descendants
from research.joint_evaluation_20261029.queue.cnn4_v3_admission import need, validate_continuation


def state_for(config):
    value = json.loads(r.path(config['original_state_path']).read_text())
    need(value['plan_ref'] == config['previous_live_plan_ref'] and
         value['controller_pid'] == config['prior_controller']['pid'], 'Exact original CNN owner and plan')
    return value


def boundary_reason(config, template, state, children):
    if children or any(row['status'] == 'running' for row in state['tasks'].values()):
        return 'Original CNN controller still has running work or live descendants'
    need(set(state['tasks']) == {task['id'] for task in template['tasks']}, 'Original CNN task set changed')
    for task in template['tasks']:
        row = state['tasks'][task['id']]
        if task['kind'] == 'external':
            need(row['status'] == 'succeeded' and row['result_ref'] == task['result_ref'] and
                 row.get('actual_wait') is True and row.get('actual_returncode') == 0 and
                 row.get('owned_process_group_absent') is True, 'Exact successful historical CNN alias')
            need(not processes.session_members(row['pid']), 'Completed CNN worker session remains live')
            wait = r.read(task['alias_wait_ref'])
            need(all(wait[k] == row[k] for k in wait), 'Original CNN success differs from real wait receipt')
        else:
            need(row['status'] == 'pending' and 'pid' not in row, 'CNN advanced since template; preserve and prepare a new continuation')
    return None


def check_config(config):
    for source in config['source_refs']:
        need(r.ref(source['path']) == source, 'CNN handoff source changed')
    template = r.read(config['template_ref'])
    need(template['previous_live_plan_ref'] == config['previous_live_plan_ref'], 'Exact predecessor plan')
    validate_continuation(template)
    return template


def attempt_idle_stop(config, template):
    expected = config['prior_controller']
    current = identity(expected['pid'])
    need(current is not None and current['start_ticks'] == expected['start_ticks'] and current['state'] != 'Z',
         'Original CNN controller no longer matches sealed identity')
    reason = boundary_reason(config, template, state_for(config), descendants(expected['pid']))
    if reason:
        return dict(stopped=False, reason=reason, signals=[])
    fd = processes._pidfd_open(expected['pid'])
    frozen, sent = False, []
    try:
        current = identity(expected['pid'])
        need(current is not None and current['start_ticks'] == expected['start_ticks'], 'CNN PID changed before stop')
        processes._pidfd_send(fd, signal.SIGSTOP)
        frozen = True
        sent.append('SIGSTOP')
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            current = identity(expected['pid'])
            if current is not None and current['state'] in ('T', 't'):
                break
            time.sleep(0.02)
        need(current is not None and current['start_ticks'] == expected['start_ticks'] and current['state'] in ('T', 't'),
             'Exact CNN controller must be frozen before second idle check')
        reason = boundary_reason(config, template, state_for(config), descendants(expected['pid']))
        if reason:
            return dict(stopped=False, reason='Race detected; resumed: ' + reason, signals=sent + ['SIGCONT'])
        before = r.ref(config['original_state_path'])
        processes._pidfd_send(fd, signal.SIGTERM)
        sent.append('SIGTERM')
        processes._pidfd_send(fd, signal.SIGCONT)
        frozen = False
        sent.append('SIGCONT')
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = identity(expected['pid'])
            if current is None or current['state'] == 'Z':
                break
            time.sleep(0.1)
        need(current is None or current['state'] == 'Z', 'Idle CNN controller did not exit; no escalation')
        after = r.ref(config['original_state_path'])
        need(before == after, 'Original CNN state changed during idle shutdown')
        need(not descendants(expected['pid']), 'CNN controller still has live descendants')
        return dict(stopped=True, signals=sent, original_state_ref=after,
                    old_controller_exit_observed=True, old_controller_wait_consumed=False)
    finally:
        if frozen:
            processes._pidfd_send(fd, signal.SIGCONT)
        os.close(fd)


def run(config_path, inspect=False):
    config_ref = r.ref(config_path)
    config = r.read(config_ref)
    template = check_config(config)
    state = state_for(config)
    from research.joint_evaluation_20261029.queue import cnn4_v3_scheduler as scheduler
    scheduler.validate(template)
    reason = boundary_reason(config, template, state, descendants(config['prior_controller']['pid']))
    if inspect:
        print(json.dumps(dict(valid=True, boundary_reason=reason, source_pins=len(config['source_refs']),
            config_ref=config_ref, template_ref=config['template_ref'], foreign_ram_gib=1408,
            reused_tasks=sum(t['kind']=='external' for t in template['tasks']),
            signals_sent=0, workers_launched=0)), flush=True)
        return
    directory = r.path(template['records_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'handoff.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(not (directory/'handoff.json').exists() and not (directory/'state.json').exists(), 'Prior handoff evidence; no replay')
        while True:
            check_config(config)
            result = attempt_idle_stop(config, template)
            if result['stopped']:
                break
            r.save(directory/'watch_status.json', dict(config_ref=config_ref, watcher_pid=os.getpid(), **result))
            time.sleep(20)
        sealed = r.save(directory/'original_state_sealed.json', r.read(result['original_state_ref']))
        receipt = r.save(directory/'handoff.json', dict(schema='cnn4_exact_idle_handoff_v1',
            config_ref=config_ref, sealed_original_state_ref=sealed, prior_controller=config['prior_controller'], **result))
        concrete = dict(template, handoff_ref=receipt, handoff_config_ref=config_ref)
        final_path = directory/'plan.json'
        r.save(final_path, concrete)
        outcome = scheduler.run(final_path)
        print(json.dumps(outcome['counts']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    run(args.config, args.inspect)


if __name__ == '__main__':
    main()
