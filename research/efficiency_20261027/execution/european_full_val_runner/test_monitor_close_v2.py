"""Deterministic tiny-thread shutdown regressions; no scientific/native calls."""
import argparse
import fcntl
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch
from . import aggregate_guard_v1 as prior, aggregate_guard_v2 as guard, contracts_v2 as c

HERE = Path(__file__).parent


def owner_for(module, scope):
    return module.AggregateOwner(scope, execution_check=lambda: None,
        max_family_rss_bytes=2**30, max_seconds=10)


def lease_released(owner, scope):
    assert owner._closed and owner._lease.closed
    with (scope/'.aggregate_owner.lock').open('a+b') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def late_failure(module, scope):
    owner = owner_for(module, scope)
    entered = threading.Event()
    closing = threading.Event()
    refresh = owner.refresh
    stop = owner.stop_monitor
    def delayed_failure():
        if threading.current_thread().name == 'full-validation-storage':
            entered.set()
            assert closing.wait(1)
            raise module.GuardFailure('Synthetic final in-flight monitor error')
        return refresh()
    def begin_close():
        closing.set()
        return stop()
    failure = None
    with patch.object(module, 'SCAN_INTERVAL', .001), patch.object(owner, 'refresh', side_effect=delayed_failure), \
         patch.object(owner, 'stop_monitor', side_effect=begin_close):
        owner.start_monitor()
        assert entered.wait(1)
        owner.check()
        try:
            owner.close()
        except module.GuardFailure as error:
            failure = str(error)
        finally:
            closing.set()
            owner._thread.join(timeout=1)
    assert not owner._thread.is_alive() and owner._error is not None
    lease_released(owner, scope)
    return failure


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipt', default=str(HERE/'MONITOR_CLOSE_CPU_v2.json'))
    args = parser.parse_args()
    destination = Path(args.receipt)
    assert destination.is_relative_to(HERE)
    checks = []
    with tempfile.TemporaryDirectory(prefix='CPU_close_', dir=HERE) as temporary:
        base = Path(temporary)
        def scope(name):
            path = base/name
            path.mkdir()
            return path
        assert late_failure(prior, scope('v1_reproduces')) is None
        checks.append('sealed_v1_reproduces_ignored_late_monitor_failure')
        assert 'Synthetic final in-flight monitor error' in late_failure(guard, scope('v2_rejects'))
        checks.append('v2_propagates_monitor_failure_after_final_foreground_check_and_releases_lease')
        path = scope('clean')
        owner = owner_for(guard, path)
        owner.start_monitor()
        owner.check()
        owner.close()
        assert owner._error is None and not owner._thread.is_alive()
        lease_released(owner, path)
        checks.append('clean_close_joins_and_releases_lease')
        path = scope('latched')
        owner = owner_for(guard, path)
        owner._error = 'Previously observed monitor failure'
        try:
            owner.close()
        except guard.GuardFailure as error:
            assert owner._error in str(error)
        else:
            raise AssertionError('Previously latched failure lost')
        lease_released(owner, path)
        checks.append('previously_latched_monitor_failure_propagates')
        path = scope('join_timeout')
        owner = owner_for(guard, path)
        entered = threading.Event()
        release = threading.Event()
        refresh = owner.refresh
        def blocked_refresh():
            if threading.current_thread().name == 'full-validation-storage':
                entered.set()
                assert release.wait(1)
            return refresh()
        with patch.object(guard, 'SCAN_INTERVAL', .001), patch.object(owner, 'refresh', side_effect=blocked_refresh):
            owner.start_monitor()
            assert entered.wait(1)
            try:
                with patch.object(guard, 'MAX_SAMPLE_AGE', .01):
                    try:
                        owner.close()
                    except guard.GuardFailure as error:
                        assert 'failed to join' in str(error)
                    else:
                        raise AssertionError('Join timeout lost')
                lease_released(owner, path)
            finally:
                release.set()
                owner._thread.join(timeout=1)
        assert not owner._thread.is_alive()
        checks.append('join_timeout_propagates_and_lease_cleanup_survives')
        path = scope('worker_finalizer')
        owner = owner_for(guard, path)
        owner._error = 'Terminal monitor failure'
        def worker_shape():
            try:
                return 'finished'
            finally:
                owner.close()
        try:
            worker_shape()
        except guard.GuardFailure:
            pass
        else:
            raise AssertionError('Worker finalizer permitted successful return')
        lease_released(owner, path)
        checks.append('worker_finally_shape_cannot_return_success_after_monitor_failure')
    report = dict(schema='European_full_VAL_monitor_close_CPU_v2', passed=True,
        check_count=len(checks), checks=checks,
        source_refs=[c.ref(HERE/name) for name in ('aggregate_guard_v1.py', 'aggregate_guard_v2.py',
            'worker_v2.py', 'test_monitor_close_v2.py')],
        real_scientific_payloads_opened=False, scientific_native_model_GPU_executions=0)
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(passed=True, checks=len(checks), receipt=c.ref(destination))))

if __name__ == '__main__':
    main()
