"""Root-invoked watcher: replace only a verified naturally childless ARGformer owner.

Check-only mode never signals or launches. Apply mode requires an independently
reviewed exact candidate. A missed boundary resumes the original controller.
Workers are never adopted, signalled or restarted by this watcher.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from arggnn.benchmarking.paper_campaign import runtime as r, processes
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import boundary_v1 as boundary
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import support_v1 as support
from research.joint_evaluation_20261029.queue_audit_20261030.resources import probe_guard_v1 as probe

need = boundary.need


class HandoffRequiresInspection(RuntimeError):
    pass


def configuration(plan_path, review_path=None):
    plan_ref = r.ref(plan_path)
    plan = r.read(plan_ref)
    spec = boundary.validate_plan(plan)
    need(spec['launch_argv'] == [str(r.path('.runtime/env/bin/python')), '-B', '-s', '-u', '-m',
         'research.joint_evaluation_20261029.queue_audit_20261030.backfill.scheduler_v1', '--plan', plan_ref['path']],
         'Exact readable replacement command required')
    if review_path is not None:
        review = r.read(r.ref(review_path))
        need(review['passed'] is True and review['watcher_ref'] == r.ref(__file__) and
             review['candidate_plan_ref'] == plan_ref, 'Independent review must bind exact watcher and candidate')
    return plan, spec


def attempt(plan_path, spec, journal):
    # The cheap childless check must succeed before any signal. Opening a pidfd
    # and rechecking identity prevents a recycled numeric PID from being used.
    boundary.controller(spec)
    fd = processes._pidfd_open(spec['predecessor_controller']['pid'])
    paused = False
    killed = False
    try:
        boundary.controller(spec)
        configuration(plan_path)
        journal('pre_freeze_childless')
        processes._pidfd_send(fd, signal.SIGSTOP)
        paused = True
        for unused in range(40):
            try:
                frozen = boundary.controller(spec, frozen=True)
                break
            except ValueError as error:
                if 'must be frozen' not in str(error):
                    raise
                time.sleep(0.05)
        else:
            raise ValueError('Controller did not freeze every thread')
        observed = boundary.inspect_boundary(spec)
        need(observed['controller']['thread_states'] == frozen['thread_states'], 'Frozen thread set changed')
        # Revalidate the entire unchanged executable contract while frozen.
        configuration(plan_path)
        accounting = support.admission_accounting(r.read(r.ref(plan_path)))
        need(accounting['admitted'] is True, 'Shared full-writer storage reservation does not fit')
        boundary.controller(spec, frozen=True)
        second = boundary.inspect_boundary(spec)
        need(second == observed, 'Frozen state or genuine waits changed')
        folder = r.path(spec['metadata_directory']) / 'boundary_attempts' / str(time.time_ns())
        folder.mkdir(parents=True, exist_ok=False)
        sealed_ref = support.immutable_json(folder / 'original_state.json', observed['source'])
        reconciliation_ref = support.immutable_json(folder / 'reconciliation.json',
            {key: observed[key] for key in ('rows', 'actual_wait_refs', 'reconciled_stale_running_ids')})
        evidence_ref = support.immutable_json(folder / 'frozen_controller.json', frozen)
        boundary.controller(spec, frozen=True)
        need(json.loads(r.path(spec['original_state_path']).read_text()) == observed['source'], 'State changed before termination')
        journal('frozen_verified_boundary', evidence_ref=evidence_ref,
                sealed_state_ref=sealed_ref, reconciliation_ref=reconciliation_ref)
        # Keep the verified childless owner frozen through exit. SIGTERM could
        # run its Python handler and admit a new child before the handler fires.
        processes._pidfd_send(fd, signal.SIGKILL)
        killed = True
        paused = False
        ready, unused_write, unused_error = select.select([fd], [], [], 20)
        need(bool(ready), 'Old exit not observed; do not start a replacement')
        need(not processes.session_members(spec['predecessor_controller']['pid']), 'Old controller session did not drain')
        handoff = dict(schema='argformer_verified_idle_handoff_v1',
            operational_continuation_ref=r.read(r.ref(plan_path))['operational_continuation_ref'],
            predecessor_controller=spec['predecessor_controller'], sealed_state_ref=sealed_ref,
            reconciliation_ref=reconciliation_ref, frozen_controller_ref=evidence_ref,
            old_exit_observed=True, old_actual_wait_consumed=False, worker_signals_sent=[],
            controller_signals=['SIGSTOP', 'SIGKILL'], recorded_unix=time.time())
        support.immutable_json(spec['handoff_receipt_path'], handoff)
        journal('old_exit_observed', actual_wait_consumed=False)
    except Exception as error:
        if killed:
            raise HandoffRequiresInspection('Old controller exit was requested; inspect sealed evidence before any retry: ' + str(error)) from error
        raise
    finally:
        if paused:
            processes._pidfd_send(fd, signal.SIGCONT)
            journal('boundary_missed_original_resumed', worker_signals_sent=[])
        os.close(fd)
    # No retry after an old owner has exited. A failed spawn or bootstrap is
    # preserved for explicit inspection, with sealed evidence and logs intact.
    need(killed, 'No verified old exit')
    configuration(plan_path)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(r.ROOT / 'src') + ':' + str(r.ROOT),
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with r.path(spec['replacement_log_path']).open('xb') as log:
        child = subprocess.Popen(spec['launch_argv'], cwd=r.ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    fields = (Path('/proc') / str(child.pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    receipt = dict(schema='argformer_idle_boundary_replacement_started_v1', pid=child.pid,
        process_start_ticks=fields[19], command=spec['launch_argv'], candidate_plan_ref=r.ref(plan_path),
        handoff_ref=r.ref(spec['handoff_receipt_path']), worker_signals_sent=[], started_unix=time.time())
    support.immutable_json(spec['replacement_started_path'], receipt)
    journal('replacement_started', receipt_ref=r.ref(spec['replacement_started_path']))
    return receipt


def progress_outcome(spec):
    source = json.loads(r.path(spec['original_state_path']).read_text())
    need(source['plan_ref'] == spec['original_plan_ref'] and
         source['controller_pid'] == spec['predecessor_controller']['pid'], 'Original state ownership changed')
    row = source['tasks'][boundary.PREPARE]
    if row['status'] != 'pending' or 'pid' in row:
        return dict(status='opportunity_closed', reason='TEST009 is already progressing or terminal under the original owner',
                    test009_prepare_status=row['status'], worker_signals_sent=[])
    old = spec['predecessor_controller']
    path = Path('/proc') / str(old['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        fields = None
    if fields is None or fields[19] != old['process_start_ticks'] or fields[0] in ('Z', 'X'):
        terminal = all(row['status'] in ('succeeded', 'failed', 'blocked') for row in source['tasks'].values())
        return dict(status='opportunity_closed' if terminal else 'owner_absent_requires_inspection',
                    reason='Original owner is no longer live; no automatic handoff or replay', worker_signals_sent=[])
    return None


def run(plan_path, review_path, *, once=False):
    plan, spec = configuration(plan_path, review_path)
    probe.install_process_probes()
    directory = r.path(spec['metadata_directory'])
    lock = (directory / 'watcher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    need(not r.path(spec['handoff_receipt_path']).exists() and not r.path(spec['replacement_started_path']).exists(),
         'Existing handoff must be inspected; no automatic replay')
    log = (directory / 'watcher_events.jsonl').open('a')
    previous = None
    def journal(action, **values):
        log.write(json.dumps(dict(action=action, time_unix=time.time(), **values), sort_keys=True) + '\n')
        log.flush()
        os.fsync(log.fileno())
    try:
        while True:
            try:
                outcome = progress_outcome(spec)
                if outcome is not None:
                    r.save(directory / 'watcher_status.json', outcome)
                    journal(outcome['status'], reason=outcome['reason'], worker_signals_sent=[])
                    return outcome
                # Active descendants never receive a signal. Exact wait checks
                # run after freezing because receipt publication may be mid-step.
                boundary.controller(spec)
                result = attempt(plan_path, spec, journal)
                r.save(directory / 'watcher_status.json', dict(status='replacement_started', receipt=result))
                return result
            except Exception as error:
                if isinstance(error, HandoffRequiresInspection) or r.path(spec['handoff_receipt_path']).exists():
                    journal('handoff_requires_inspection', error=str(error))
                    raise
                reason = type(error).__name__ + ': ' + str(error)
                if reason != previous:
                    previous = reason
                    r.save(directory / 'watcher_status.json', dict(status='waiting_for_safe_boundary', reason=reason,
                        worker_signals_sent=[], predecessor_controller=spec['predecessor_controller'], updated_unix=time.time()))
                    print('waiting_for_safe_boundary: ' + reason, flush=True)
                    journal('waiting_for_safe_boundary', reason=reason)
                if once:
                    return dict(status='waiting_for_safe_boundary', reason=reason)
                time.sleep(2)
    finally:
        log.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--review')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    plan, spec = configuration(args.plan, args.review)
    if not args.apply:
        try:
            evidence = boundary.inspect_boundary(spec)
            result = dict(status='safe_boundary_observed', counts=len(evidence['rows']))
        except Exception as error:
            result = dict(status='waiting_for_safe_boundary', reason=type(error).__name__ + ': ' + str(error))
        print(json.dumps(dict(result, production_launches=0, signals_sent=0), sort_keys=True))
        return
    need(args.review is not None, 'Apply requires exact independent review')
    print(json.dumps(run(args.plan, args.review, once=args.once), sort_keys=True))


if __name__ == '__main__':
    main()
