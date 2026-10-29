"""Operational recovery checks; no production job, signal, or historical write."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue import successor_recovery_v1 as recovery

MANIFEST = recovery.BASE + 'successor_recovery_manifest_v1.json'


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.wait = dict(status='succeeded', pid=90210, process_start_ticks='12345',
                         actual_wait=True, actual_returncode=0, owned_process_group_absent=True,
                         result_ref=dict(path='out/result.json', sha256='result'))
        self.prior_ref = dict(path='prior.json', sha256='prior')
        self.prior = dict(tasks={'task': copy.deepcopy(self.wait)})

    def test_identical_wait(self):
        recovery.same_worker_record('task', self.wait, self.wait, self.prior, self.prior_ref)

    def test_authenticated_adoption_annotation(self):
        row = dict(self.wait, continued_from_state_ref=self.prior_ref)
        recovery.same_worker_record('task', row, self.wait, self.prior, self.prior_ref)

    def test_changed_wait_fails(self):
        row = dict(self.wait, actual_returncode=1)
        with self.assertRaisesRegex(ValueError, 'differs from real wait'):
            recovery.same_worker_record('task', row, self.wait, self.prior, self.prior_ref)

    def test_unauthenticated_adoption_fails(self):
        row = dict(self.wait, continued_from_state_ref=self.prior_ref)
        prior = dict(tasks={'task': dict(self.wait, actual_returncode=7)})
        with self.assertRaisesRegex(ValueError, 'differs from real wait'):
            recovery.same_worker_record('task', row, self.wait, prior, self.prior_ref)

    def test_extra_annotation_fails(self):
        row = dict(self.wait, continued_from_state_ref=self.prior_ref, invented=True)
        with self.assertRaisesRegex(ValueError, 'differs from real wait'):
            recovery.same_worker_record('task', row, self.wait, self.prior, self.prior_ref)

    def test_live_or_reused_controller_fails(self):
        for ticks in ('12345', 'other_process'):
            with self.subTest(ticks=ticks), patch.object(recovery.handoff, 'identity', return_value={'start_ticks': ticks}):
                with self.assertRaisesRegex(ValueError, 'PID is present'):
                    recovery.absent_process(dict(pid=90210, start_ticks='12345'), 'old')

    def test_surviving_controller_session_fails(self):
        with patch.object(recovery.handoff, 'identity', return_value=None), patch.object(recovery.processes, 'session_members', return_value=[90211]):
            with self.assertRaisesRegex(ValueError, 'session remains live'):
                recovery.absent_process(dict(pid=90210, start_ticks='12345'), 'old')

    def test_surviving_controller_descendant_fails(self):
        with patch.object(recovery.handoff, 'identity', return_value=None), patch.object(recovery.processes, 'session_members', return_value=[]), patch.object(recovery.handoff, 'descendants', return_value=[90211]):
            with self.assertRaisesRegex(ValueError, 'descendants remain live'):
                recovery.absent_process(dict(pid=90210, start_ticks='12345'), 'old')

    def worker_fixture(self, row=None, members=None):
        row = copy.deepcopy(self.wait if row is None else row)
        wait_ref = dict(path='out/actual_wait.json', sha256='wait')
        manifest = dict(prior_continuation_state_ref=self.prior_ref, historical_wait_refs={'task': wait_ref})
        values = {'prior.json': self.prior, 'out/actual_wait.json': self.wait, 'out/result.json': {}}
        with patch.object(r, 'read', side_effect=lambda ref: values[ref['path']]), patch.object(recovery.processes, 'session_members', return_value=members or []), patch.object(recovery.handoff, 'identity', return_value=None):
            return recovery.verify_workers(dict(tasks={'task': row}), dict(tasks=[dict(id='task', output_directory='out')]), manifest)

    def test_worker_receipt_passes(self):
        self.assertEqual(self.worker_fixture(), 1)

    def test_worker_unconsumed_wait_fails(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete historical wait'):
            self.worker_fixture(dict(self.wait, actual_wait=False))

    def test_worker_surviving_session_fails(self):
        with self.assertRaisesRegex(ValueError, 'session remains live'):
            self.worker_fixture(members=[90211])

    def test_worker_changed_returncode_fails(self):
        with self.assertRaisesRegex(ValueError, 'differs from real wait'):
            self.worker_fixture(dict(self.wait, actual_returncode=1))

    def test_existing_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT/'research/joint_evaluation_20261029/review_finish') as temporary:
            path = Path(temporary)/'receipt.json'
            path.write_text('{"retained": true}\n')
            with self.assertRaises(FileExistsError):
                recovery.write_new(path, {'retained': False})
            self.assertEqual(path.read_text(), '{"retained": true}\n')

    def test_live_inspection_is_read_only_and_preserves_failure(self):
        manifest_ref = r.ref(MANIFEST)
        manifest = r.read(manifest_ref)
        before = {name: r.ref(manifest[name]['path']) for name in
                  ('original_final_state_ref', 'failed_watcher_log_ref', 'original_controller_log_ref')}
        with patch.object(recovery, 'write_new', side_effect=AssertionError('inspection wrote a file')), patch.object(recovery.scheduler, 'run', side_effect=AssertionError('inspection launched scheduler')):
            result = recovery.run(MANIFEST, inspect_only=True)
        self.assertEqual(result['verified_worker_waits'], 64)
        self.assertEqual(result['prospective_status_counts'], dict(succeeded=70, pending=155, blocked=16, failed=1, superseded=11))
        self.assertEqual(result['signals_sent'], [])
        self.assertFalse(result['old_controller_wait_consumed'])
        self.assertFalse(result['exact_shutdown_field_diff_available'])
        self.assertTrue(result['global_owner_lock_verified'])
        for name, reference in before.items():
            self.assertEqual(r.ref(manifest[name]['path']), reference)

    def test_new_record_transaction_uses_unchanged_scheduler(self):
        with tempfile.TemporaryDirectory(dir=r.ROOT/'research/joint_evaluation_20261029/review_finish') as temporary:
            root = Path(temporary)
            (root/'.runtime/benchmarking').mkdir(parents=True)
            (root/recovery.GLOBAL_LOCK).touch()
            original = {'tasks': {'done': {'status': 'succeeded'}}}
            config = dict(prior_controller=dict(pid=90210, start_ticks='12345'))
            template = dict(schema='paper_campaign_v1', records_directory='failed_records',
                            tasks=[{'id': 'unchanged'}], storage_ceiling_bytes=1_200_000_000_000, controller_source_refs=[])
            with patch.object(r, 'ROOT', root), patch.object(recovery, 'RECORDS', 'recovery_records'):
                original_ref = recovery.write_new(root/'original.json', original)
                manifest = dict(original_config_ref=dict(path='original_config.json', sha256='config'), original_final_state_ref=original_ref, storage_credit_ref={'path': 'credit.json', 'sha256': 'credit'}, source_refs=[])
                manifest_ref = recovery.write_new(root/'manifest.json', manifest)
                audit = dict(verified_worker_waits=64, workers_launched=0)
                def unchanged_scheduler(plan_path):
                    # The real frozen scheduler can reacquire the original lock.
                    with recovery.global_owner_lock():
                        plan = r.read(r.ref(plan_path))
                    stable = {k: v for k, v in plan.items() if k not in ('records_directory', 'handoff_ref', 'handoff_config_ref', 'recovery_manifest_ref', 'storage_credit_ref')}
                    self.assertEqual(stable, {k: v for k, v in template.items() if k != 'records_directory'})
                    receipt = r.read(plan['handoff_ref'])
                    self.assertFalse(receipt['old_controller_wait_consumed'])
                    self.assertIsNone(receipt['old_controller_returncode'])
                    self.assertEqual(receipt['signals'], [])
                    self.assertEqual(r.read(receipt['sealed_original_state_ref']), original)
                    self.assertEqual(r.ref(root/'original.json'), original_ref)
                    self.assertEqual(plan['handoff_config_ref'], manifest['original_config_ref'])
                    self.assertEqual(plan['records_directory'], template['records_directory'])
                    self.assertEqual(plan['storage_credit_ref'], manifest['storage_credit_ref'])
                    self.assertEqual(Path(plan_path).parent.name, 'failed_records')
                    self.assertTrue((root/'recovery_records/recovery_audit.json').exists())
                    return {'counts': {'succeeded': 1}}
                with patch.object(recovery, 'inspect', return_value=(manifest, config, template, original, audit)), patch.object(recovery.scheduler, 'run', side_effect=unchanged_scheduler) as called:
                    result = recovery.run('manifest.json')
                self.assertEqual(result['counts'], {'succeeded': 1})
                called.assert_called_once()
                self.assertEqual(r.ref(root/'manifest.json'), manifest_ref)


if __name__ == '__main__':
    unittest.main()
