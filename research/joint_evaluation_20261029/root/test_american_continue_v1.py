"""Operational continuation regression tests; no GPU jobs or live mutations."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from research.joint_evaluation_20261029.root import american_continue as a


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=a.r.ROOT / 'research/joint_evaluation_20261029/root')
        self.folder = Path(self.temporary.name)
        self.plan_ref = dict(path='plan', sha256='exact')
        self.spec = dict(name='mock_stage', command=[a.PYTHON, '-B', '-m', 'mock_module'], products=[], gpu=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact18_stage_structure(self):
        stages = a.expected_stages({})
        self.assertEqual(len(stages), 18)
        self.assertEqual(sum(s['gpu'] for s in stages), 17)
        self.assertEqual(len({s['name'] for s in stages}), 18)
        self.assertEqual(stages[-1]['command'][-3:], ['score', '--freeze', a.FREEZE])

    def test_inherited_lock_real_wait_and_reuse(self):
        process = Mock(pid=123456)
        process.wait.return_value = 0
        with (self.folder / 'mutex').open('a+b') as lock, patch.object(a.subprocess, 'Popen', return_value=process) as launch, patch.object(a, 'identity', return_value=dict(process_start_ticks='123')), patch.object(a.processes, 'session_members', return_value=[]):
            first = a.stage(self.folder, self.spec, self.plan_ref, lock)
            self.assertEqual(launch.call_args.kwargs['pass_fds'], (lock.fileno(),))
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            process.wait.assert_called_once_with()
            second = a.stage(self.folder, self.spec, self.plan_ref, lock)
            self.assertEqual(first, second)
            launch.assert_called_once()

    def test_failed_stage_preserved_without_replay(self):
        process = Mock(pid=123456)
        process.wait.return_value = 7
        with (self.folder / 'mutex').open('a+b') as lock, patch.object(a.subprocess, 'Popen', return_value=process) as launch, patch.object(a, 'identity', return_value=dict(process_start_ticks='123')), patch.object(a.processes, 'session_members', return_value=[]):
            for _ in range(2):
                with self.assertRaises(ValueError):
                    a.stage(self.folder, self.spec, self.plan_ref, lock)
            launch.assert_called_once()
            self.assertEqual(a.r.read(a.r.ref(self.folder / 'mock_stage/actual_wait.json'))['actual_returncode'], 7)

    def test_partial_launch_never_replayed(self):
        directory = self.folder / self.spec['name']
        directory.mkdir()
        (directory / 'worker.log').write_text('Preserved launch evidence')
        with patch.object(a.subprocess, 'Popen') as launch, self.assertRaises(ValueError):
            a.stage(self.folder, self.spec, self.plan_ref)
        launch.assert_not_called()

    def test_gpu_stage_requires_lock(self):
        with self.assertRaises(ValueError):
            a.stage(self.folder, self.spec, self.plan_ref)

    def test_missing_product_receipt_still_records_real_wait(self):
        spec = copy.deepcopy(self.spec)
        spec['products'] = [str(self.folder / 'missing_product.json')]
        process = Mock(pid=123456)
        process.wait.return_value = 0
        with (self.folder / 'mutex').open('a+b') as lock, patch.object(a.subprocess, 'Popen', return_value=process), patch.object(a, 'identity', return_value=dict(process_start_ticks='123')), patch.object(a.processes, 'session_members', return_value=[]):
            with self.assertRaises(ValueError):
                a.stage(self.folder, spec, self.plan_ref, lock)
        row = a.r.read(a.r.ref(self.folder / 'mock_stage/actual_wait.json'))
        self.assertTrue(row['actual_wait'])
        self.assertFalse(row['products_complete'])

    def test_wait_binding_rejects_missing_expected_products(self):
        row = dict(actual_wait=True, actual_returncode=0, owned_process_group_absent=True, pid=123456,
                   products_complete=True, command=self.spec['command'], plan_ref=self.plan_ref, products=[])
        spec = copy.deepcopy(self.spec)
        target = self.folder / 'product'; target.write_text('actual product')
        spec['products'] = [str(target)]
        with patch.object(a.processes, 'session_members', return_value=[]), self.assertRaises(ValueError):
            a.validate_wait(row, spec, self.plan_ref)

    def test_exact_cache_wait_or_live_controller(self):
        job_ref = dict(path='job', sha256='job')
        started_ref = dict(path='started', sha256='started')
        started = dict(pid=7, process_start_ticks='8', command=['cache'], job_ref=job_ref)
        job = dict(argv=['cache'], products=[])
        plan = dict(cache_actual_wait_path=str(self.folder / 'cache_wait'), cache_started_ref=started_ref,
                    cache_job_ref=job_ref, cache_controller_identity=dict(pid=6, process_start_ticks='5'))
        with patch.object(a.r, 'read', side_effect=lambda ref: started if ref == started_ref else job), patch.object(a, 'alive', return_value=True), patch.object(a, 'identity', return_value=None):
            self.assertIsNone(a.cache_state(plan))
        with patch.object(a.r, 'read', side_effect=lambda ref: started if ref == started_ref else job), patch.object(a, 'alive', return_value=False), self.assertRaisesRegex(ValueError, 'disappeared'):
            a.cache_state(plan)
        with patch.object(a.r, 'read', side_effect=lambda ref: started if ref == started_ref else job), patch.object(a, 'alive', return_value=True), patch.object(a, 'identity', return_value=dict(process_start_ticks='different')), self.assertRaisesRegex(ValueError, 'reused'):
            a.cache_state(plan)

    def test_live_descendants_block_completed_wait(self):
        with patch.object(a.processes, 'session_members', return_value=[dict(pid=8)]), self.assertRaises(ValueError):
            a.successful_wait(dict(actual_wait=True, actual_returncode=0, owned_process_group_absent=True, pid=7))


if __name__ == '__main__':
    unittest.main()
