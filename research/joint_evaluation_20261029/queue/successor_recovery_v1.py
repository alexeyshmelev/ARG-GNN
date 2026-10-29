"""Authenticate the failed idle handoff and enter the unchanged successor queue.

This additive recovery sends no signal and never claims to have waited for either
former controller. The final original state and every historical worker receipt
are verified before a new, exclusive recovery record directory is populated.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue import successor_v2_handoff as handoff
from research.joint_evaluation_20261029.queue import successor_v2_support as support
from research.joint_evaluation_20261029.queue import successor_recovery_scheduler_v1 as scheduler
from research.joint_evaluation_20261029.queue import storage_credit_v1 as storage_credit

BASE = 'research/joint_evaluation_20261029/queue/'
RECORDS = BASE + 'successor_recovery_records_v1'
GLOBAL_LOCK = '.runtime/benchmarking/paper_campaign_global.lock'


def need(value, message):
    support.need(value, message)


def absent_process(identity, label):
    # Refuse even a reused PID; recovery never acts on another process.
    need(handoff.identity(identity['pid']) is None, label + ' PID is present')
    need(not processes.session_members(identity['pid']), label + ' session remains live')
    need(not handoff.descendants(identity['pid']), label + ' descendants remain live')


def same_worker_record(name, row, wait, prior_state, prior_ref):
    """Only the authenticated 26 October adoption annotation may differ."""
    if row == wait:
        return
    annotated = copy.deepcopy(wait)
    annotated['continued_from_state_ref'] = prior_ref
    need(row == annotated and prior_state['tasks'].get(name) == wait,
         'Historical worker row differs from real wait: ' + name)


def verify_workers(original, original_plan, manifest):
    tasks = {task['id']: task for task in original_plan['tasks']}
    need(set(original['tasks']) == set(tasks), 'Exact original task namespace required')
    prior_ref = manifest['prior_continuation_state_ref']
    prior = r.read(prior_ref)
    launched = {name for name, row in original['tasks'].items() if 'pid' in row}
    need(launched == set(manifest['historical_wait_refs']), 'Historical worker set changed')
    for name, row in original['tasks'].items():
        need(row['status'] in ('pending', 'blocked', 'failed', 'succeeded'), 'No running or unknown original row')
        if name not in launched:
            need(row['status'] != 'failed', 'Failed worker without preserved process evidence')
            if row['status'] == 'succeeded':
                r.read(row['result_ref'])
            continue
        need(row['status'] in ('succeeded', 'failed') and row.get('actual_wait') is True
             and row.get('owned_process_group_absent') is True, 'Incomplete historical wait: ' + name)
        wait_ref = manifest['historical_wait_refs'][name]
        need(r.path(wait_ref['path']) == r.path(tasks[name]['output_directory']) / 'actual_wait.json',
             'Historical wait outside exact output scope: ' + name)
        wait = r.read(wait_ref)
        same_worker_record(name, row, wait, prior, prior_ref)
        need(not processes.session_members(row['pid']), 'Historical worker session remains live: ' + name)
        current = handoff.identity(row['pid'])
        need(current is None or current['start_ticks'] != row['process_start_ticks'],
             'Historical worker still exists: ' + name)
        if row['status'] == 'succeeded':
            need(row['actual_returncode'] == 0, 'Successful row has nonzero wait: ' + name)
            r.read(row['result_ref'])
    return len(launched)


def preview_rows(original, template):
    tasks = {task['id']: task for task in template['tasks']}
    need(set(tasks) == set(original['tasks']) | set(template['added_task_ids']), 'Declared successor namespace required')
    rows, prior = support.reset_rows(original, tasks, template['alias_task_ids'])
    for name in template['added_task_ids']:
        need(name not in rows and tasks[name]['kind'] == 'external', 'Only declared external additions')
        rows[name] = dict(status='pending', external_owner=True)
    rows, superseded = support.apply_supersession(rows, r.read(template['supersession_ref']))
    prior.update(superseded)
    for name, row in rows.items():
        if row['status'] == 'pending' and tasks[name]['kind'] != 'external':
            need(not r.path(tasks[name]['output_directory']).exists(), 'Pending internal output exists: ' + name)
        if original['tasks'].get(name, {}).get('status') == 'failed':
            need(row == original['tasks'][name], 'Genuine failed history changed')
    return rows, prior


def verify_failure_history(manifest, config, template):
    launch = r.read(manifest['failed_watcher_launch_ref'])
    need(launch['config_ref'] == manifest['original_config_ref'], 'Failed watcher used a different config')
    need({key: launch[key] for key in ('pid', 'start_ticks')} == manifest['failed_watcher'], 'Failed watcher identity changed')
    for field in ('failed_watcher_log_ref', 'original_controller_log_ref', 'original_scheduler_source_ref'):
        need(r.ref(manifest[field]['path']) == manifest[field], 'Failure evidence changed: ' + field)
    watcher_log = r.path(manifest['failed_watcher_log_ref']['path']).read_text()
    original_log = r.path(manifest['original_controller_log_ref']['path']).read_text()
    need(watcher_log.endswith('ValueError: Original state changed during idle shutdown\n'), 'Different watcher failure')
    need('scheduler.py", line 287, in run' in original_log
         and original_log.endswith('RuntimeError: Controller received explicit termination signal\n'),
         'Different original shutdown evidence')
    failed_directory = r.path(template['records_directory'])
    need(not any((failed_directory / name).exists() for name in
                 ('handoff.json', 'original_state_sealed.json', 'plan.json', 'state.json', 'adoption_audit.json')),
         'Failed watcher already produced successor state; never replay')
    absent_process(config['prior_controller'], 'Original controller')
    absent_process(manifest['failed_watcher'], 'Failed watcher')


def inspect(manifest_ref):
    manifest = r.read(manifest_ref)
    need(manifest['schema'] == 'failed_idle_successor_recovery_v1', 'Unknown recovery schema')
    need(manifest['records_directory'] == RECORDS, 'Exact additive recovery scope required')
    need(manifest['historical_rewrite_authorized'] is False and manifest['signals_authorized'] is False,
         'Recovery has no signaling or historical rewrite authority')
    for reference in manifest['source_refs']:
        need(r.ref(reference['path']) == reference, 'Recovery source changed')
    config = r.read(manifest['original_config_ref'])
    handoff.check_config(config)
    template = r.read(config['successor_template_ref'])
    scheduler.validate(template)
    verify_failure_history(manifest, config, template)
    need(manifest['original_final_state_ref']['path'] == config['original_state_path'], 'Wrong final original state')
    original = r.read(manifest['original_final_state_ref'])
    need(handoff.boundary_reason(original, config, []) is None, 'Original idle boundary no longer valid')
    need(dict(Counter(row['status'] for row in original['tasks'].values())) == manifest['original_status_counts'],
         'Original terminal counts changed')
    worker_count = verify_workers(original, r.read(config['effective_plan_ref']), manifest)
    rows, prior = preview_rows(original, template)
    directory = r.path(RECORDS)
    need(not any((directory / name).exists() for name in
                 ('recovery_audit.json', 'handoff.json', 'original_state_sealed.json', 'plan.json', 'state.json', 'adoption_audit.json')),
         'Recovery attempt already exists; inspect it rather than replay')
    accounting = storage_credit.commitment(dict(template, handoff_config_ref=manifest['original_config_ref']), manifest['storage_credit_ref'])
    need(r.ref(config['original_state_path']) == manifest['original_final_state_ref'], 'Original state changed during inspection')
    audit = dict(schema='verified_failed_idle_handoff_recovery_audit_v1', inspected_unix=time.time(),
                 recovery_manifest_ref=manifest_ref, original_final_state_ref=manifest['original_final_state_ref'],
                 failed_watcher_launch_ref=manifest['failed_watcher_launch_ref'],
                 failed_watcher_log_ref=manifest['failed_watcher_log_ref'],
                 original_controller_log_ref=manifest['original_controller_log_ref'],
                 original_scheduler_source_ref=manifest['original_scheduler_source_ref'],
                 historical_wait_refs=manifest['historical_wait_refs'], verified_worker_waits=worker_count,
                 original_status_counts=manifest['original_status_counts'],
                 prospective_status_counts=dict(Counter(row['status'] for row in rows.values())),
                 selective_reset_or_supersession_ids=sorted(prior),
                 predecessor_processes_absent=True, historical_sessions_absent=True,
                 old_controller_wait_consumed=False, failed_watcher_wait_consumed=False,
                 old_controller_returncode=None, failed_watcher_returncode=None,
                 pre_shutdown_state_snapshot_available=False, exact_shutdown_field_diff_available=False,
                 shutdown_explanation='Original scheduler finally saves its in-memory state after SIGTERM. Traceback places interruption in output-size scanning, after refreshing scope and GPU observations. The watcher retained its before hash only in memory; exact changed fields cannot be reconstructed.',
                 signals_sent=[], workers_launched=0, historical_files_modified=False,
                 accounting=accounting, storage_credit_ref=manifest['storage_credit_ref'],
                 storage_ceiling_overridden=False, frozen_task_definitions_unchanged=True)
    return manifest, config, template, original, audit


@contextmanager
def global_owner_lock():
    with r.path(GLOBAL_LOCK).open('r+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_new(filename, value):
    path = r.path(filename)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return r.ref(path)


def run(manifest_path, inspect_only=False):
    manifest_ref = r.ref(manifest_path)
    if inspect_only:
        with global_owner_lock():
            *_, audit = inspect(manifest_ref)
            audit['global_owner_lock_verified'] = True
        return audit
    directory = r.path(RECORDS)
    directory.mkdir(exist_ok=True)
    with (directory / 'recovery.lock').open('a+') as recovery_lock:
        fcntl.flock(recovery_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with global_owner_lock():
            manifest, config, template, original, audit = inspect(manifest_ref)
            audit['global_owner_lock_verified'] = True
            audit_ref = write_new(directory / 'recovery_audit.json', audit)
            operational = r.path(template['records_directory'])
            operational.mkdir(exist_ok=True)
            sealed_ref = write_new(operational / 'original_state_sealed.json', original)
            need(sealed_ref['sha256'] == manifest['original_final_state_ref']['sha256'], 'Sealed final state differs')
            handoff_ref = write_new(operational / 'handoff.json', dict(
                schema='failed_idle_campaign_handoff_recovery_v1', time_unix=time.time(),
                config_ref=manifest['original_config_ref'], recovery_manifest_ref=manifest_ref,
                recovery_audit_ref=audit_ref, sealed_original_state_ref=sealed_ref,
                original_state_ref=manifest['original_final_state_ref'], prior_controller=config['prior_controller'],
                old_controller_wait_consumed=False, old_controller_exit_observed=True,
                old_controller_returncode=None, failed_watcher_wait_consumed=False,
                signals=[], historical_waits_rewritten=False))
            final = dict(template, handoff_ref=handoff_ref,
                         handoff_config_ref=manifest['original_config_ref'], recovery_manifest_ref=manifest_ref,
                         storage_credit_ref=manifest['storage_credit_ref'],
                         controller_source_refs=template['controller_source_refs'] + manifest['source_refs'])
            final_ref = write_new(operational / 'plan.json', final)
        # The frozen scheduler reacquires the same global lock before adoption.
        # Any competing owner wins exclusively; this call then fails without launching.
        result = scheduler.run(final_ref['path'])
        return dict(counts=result['counts'], recovery_manifest_ref=manifest_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    print(json.dumps(run(args.manifest, args.inspect), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
