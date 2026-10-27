"""Small CPU-only tests of slow scans, write admission and failure propagation."""
import threading
import time
import unittest
from unittest.mock import patch
from .check_v2 import AsyncResourceCheck


class Original:
    def __init__(self):
        self.task = {'max_seconds': 10000., 'max_output_bytes': 1000, 'max_rss_bytes': 1000}
        self.started = time.monotonic()
        self.next_check = 0.
        self.calls = []
        self.concurrent = 0
        self.maximum_concurrent = 0
        self.block = threading.Event()
        self.release = threading.Event()
        self.do_block = False
        self.error = None

    def __call__(self, stage):
        self.concurrent += 1
        self.maximum_concurrent = max(self.maximum_concurrent, self.concurrent)
        try:
            self.calls.append(stage)
            if stage == 'TreeLoom_async_periodic_resources' and self.do_block:
                self.block.set()
                if not self.release.wait(3.):
                    raise RuntimeError('Synthetic callback wait expired')
            if self.error is not None:
                raise self.error
            if type(stage) is int and stage > self.task['max_output_bytes']:
                raise RuntimeError('Task output reservation exceeded')
            self.next_check = time.monotonic()+30.
        finally:
            self.concurrent -= 1


class Cases(unittest.TestCase):
    def make(self, original=None, **kwargs):
        original = original or Original()
        return original, AsyncResourceCheck(original, interval_seconds=.01, maximum_scan_age_seconds=1., **kwargs)

    def test_dispatch_never_waits_for_slow_scan(self):
        original, check = self.make()
        original.do_block = True
        check.start()
        try:
            self.assertTrue(original.block.wait(1.))
            for _ in range(1000):
                check('model_block')
            self.assertFalse(original.release.is_set())
            self.assertEqual(original.maximum_concurrent, 1)
        finally:
            original.release.set()
            check.close()

    def test_positive_and_zero_writes_stay_synchronous(self):
        original, check = self.make()
        with check:
            check(17)
            check(0)
            self.assertIn(17, original.calls)
            self.assertIn(0, original.calls)
        self.assertEqual(original.maximum_concurrent, 1)
        self.assertEqual(check.receipt()['synchronous_prospective_write_checks'], 2)

    def test_write_cap_failure_propagates(self):
        original, check = self.make()
        check.start()
        with self.assertRaisesRegex(RuntimeError, 'output reservation'):
            check(1001)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            check('following_block')
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            check.close()

    def test_negative_bytes_rejected_before_callback(self):
        original, check = self.make()
        with check:
            before = list(original.calls)
            with self.assertRaises(ValueError):check(-1)
            self.assertEqual(before, original.calls)

    def test_periodic_failure_reaches_training(self):
        original, check = self.make()
        check.start()
        original.error = ValueError('synthetic memory cap')
        check._thread.join(timeout=1.)
        with self.assertRaisesRegex(RuntimeError, 'failed') as result:check('next_block')
        self.assertIs(result.exception.__cause__, original.error)
        with self.assertRaisesRegex(RuntimeError, 'failed'):check.close()

    def test_wall_time_is_immediate(self):
        original, check = self.make()
        check.start()
        check.started -= 20000
        with self.assertRaisesRegex(RuntimeError, 'wall time'):check('next_block')
        with self.assertRaisesRegex(RuntimeError, 'wall time'):check.close()

    def test_scan_freshness_is_fail_closed(self):
        original, check = self.make()
        check.start()
        with check._state_lock:check._last_success -= 2.
        with self.assertRaisesRegex(RuntimeError, 'freshness'):check('next_block')
        with self.assertRaisesRegex(RuntimeError, 'freshness'):check.close()

    def test_lifecycle(self):
        original, check = self.make()
        with self.assertRaisesRegex(RuntimeError, 'not active'):check('before')
        check.start()
        with self.assertRaisesRegex(RuntimeError, 'once'):check.start()
        check.close()
        check.close()
        self.assertFalse(check._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'not active'):check('after')
        self.assertEqual(original.calls[0], 'TreeLoom_async_monitor_initial')
        self.assertEqual(original.calls[-1], 'TreeLoom_async_monitor_final')

    def test_serializes_write_with_periodic_scan(self):
        original, check = self.make()
        original.do_block = True
        check.start()
        done = threading.Event()
        writer = None
        try:
            self.assertTrue(original.block.wait(1.))
            writer = threading.Thread(target=lambda: (check(5), done.set()))
            writer.start()
            self.assertFalse(done.wait(.02))
            original.release.set()
            writer.join(timeout=1.)
            self.assertTrue(done.is_set())
            self.assertEqual(original.maximum_concurrent, 1)
        finally:
            original.release.set()
            if writer is not None:writer.join(timeout=1.)
            check.close()

    def test_initial_failure_never_starts_thread(self):
        original, check = self.make()
        original.error = RuntimeError('initial cap')
        with self.assertRaisesRegex(RuntimeError, 'initial cap'):check.start()
        self.assertIsNone(check._thread)

    def test_slow_initial_scan_cannot_renew_lease(self):
        original = Original()
        now = [original.started]
        callback = original.__call__
        class SlowInitial(Original):
            def __call__(self, stage):
                callback(stage)
                now[0] += 130.
        slow = SlowInitial()
        slow.started = now[0]
        check = AsyncResourceCheck(slow, interval_seconds=30., maximum_scan_age_seconds=120., clock=lambda:now[0])
        with self.assertRaisesRegex(RuntimeError, 'duration'):
            check.start()
        self.assertIsNone(check._last_success)
        self.assertIsNone(check._thread)

    def test_blocked_write_expires_before_scan_releases(self):
        original = Original()
        original.do_block = True
        check = AsyncResourceCheck(original, interval_seconds=.01, maximum_scan_age_seconds=.1)
        check.start()
        errors = []
        done = threading.Event()
        def attempt_write():
            try:check(7)
            except BaseException as error:errors.append(error)
            finally:done.set()
        self.assertTrue(original.block.wait(1.))
        writer = threading.Thread(target=attempt_write)
        writer.start()
        try:
            self.assertTrue(done.wait(.5))
            self.assertFalse(original.release.is_set())
            self.assertEqual(len(errors),1)
            self.assertNotIn(7, original.calls)
            self.assertIsNotNone(check._error)
        finally:
            original.release.set()
            writer.join(timeout=1.)
            with self.assertRaisesRegex(RuntimeError, 'failed'):check.close()

    def test_finite_configuration(self):
        for interval, age in [(0,1),(1,1),(2,1),(1,float('inf')),(float('nan'),2)]:
            with self.assertRaises(ValueError):
                AsyncResourceCheck(Original(), interval_seconds=interval, maximum_scan_age_seconds=age)


if __name__ == '__main__':
    unittest.main(verbosity=2)
