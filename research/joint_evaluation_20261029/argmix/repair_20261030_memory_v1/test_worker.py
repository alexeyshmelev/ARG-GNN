"""Exact-science and guard-only regression checks; no production stage executes."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from arggnn.benchmarking.paper_campaign import runtime as records
from research.joint_evaluation_20261029.argmix.repair_20261030_memory_v1 import worker
from research.joint_evaluation_20261029.argmix import pipeline, runtime

BASE = 'research/joint_evaluation_20261029/argmix/repair_20261030_memory_v1'
AMENDMENT = BASE + '/resource_amendment_v1.json'
FREEZE = 'research/joint_evaluation_20261029/argmix/freeze_v1.json'
ADMISSION = 'research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json'
ORIGINAL_CHECK_RAM = pipeline.check_ram


class FakeProcess:
    def __init__(self, rss, children=(), gone=False):
        self.rss, self.family, self.gone = rss, list(children), gone
    def children(self, recursive):
        assert recursive is True
        return self.family
    def memory_info(self):
        if self.gone:
            raise worker.psutil.NoSuchProcess(999999)
        return SimpleNamespace(rss=self.rss)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.ref = records.ref(AMENDMENT)
        self.amendment = records.read(self.ref)

    def test_live_amendment_and_old_admission(self):
        amendment, protocol, freeze_ref = worker.validate_amendment(self.ref, FREEZE, ADMISSION)
        self.assertEqual(amendment, self.amendment)
        self.assertEqual(protocol['workers'], 16)
        self.assertEqual(runtime.RAM_CAP, 64*2**30)
        self.assertEqual(records.read(amendment['original_admission_ref'])['owned_RAM_max_bytes'], 64*2**30)
        self.assertEqual(freeze_ref, records.ref(FREEZE))

    def test_all_scientific_entrypoints_are_original_objects(self):
        names = {'prepare-cache':'prepare_cache','train':'train','predict':'predict',
                 'barrier':'prediction_barrier','score':'score','validation-score':'validation_score'}
        for stage, name in names.items():
            self.assertIs(worker.STAGES[stage], getattr(pipeline, name))
        from research.joint_evaluation_20261029.argmix.observations import ObservationSource
        from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel
        self.assertIs(pipeline.ObservationSource, ObservationSource)
        self.assertIs(pipeline.JointAncestryModel, JointAncestryModel)

    def test_only_two_resource_bindings_change_in_memory(self):
        before = dict(pipeline.__dict__)
        with patch.object(pipeline, 'RAM_CAP', 64*2**30), patch.object(pipeline, 'check_ram', ORIGINAL_CHECK_RAM):
            worker.apply_ram_authority()
            changed = {name for name, value in before.items() if pipeline.__dict__[name] is not value}
            self.assertEqual(changed, {'RAM_CAP','check_ram'})
            self.assertEqual(runtime.RAM_CAP, 64*2**30)
            self.assertIs(pipeline.check_ram, worker.effective_check_ram)

    def test_original_and_new_guard_sum_identical(self):
        root = FakeProcess(2*2**30, [FakeProcess(3*2**30), FakeProcess(5*2**30), FakeProcess(0,gone=True)])
        with patch.object(worker.psutil, 'Process', return_value=root):
            self.assertEqual(ORIGINAL_CHECK_RAM(), worker.effective_check_ram())
            self.assertEqual(worker.effective_check_ram(), 10*2**30)

    def test_measured_family_fails_old_cap_and_passes_new(self):
        root = FakeProcess(17_368_395_776, [FakeProcess(17_368_395_776) for _ in range(16)])
        with patch.object(worker.psutil, 'Process', return_value=root):
            with self.assertRaisesRegex(ValueError, 'registered64GiB'):
                ORIGINAL_CHECK_RAM()
            self.assertEqual(worker.effective_check_ram(), 295_262_728_192)

    def test_new_guard_remains_finite(self):
        with patch.object(worker.psutil, 'Process', return_value=FakeProcess(worker.EFFECTIVE_RAM_CAP)):
            self.assertEqual(worker.effective_check_ram(), worker.EFFECTIVE_RAM_CAP)
        with patch.object(worker.psutil, 'Process', return_value=FakeProcess(worker.EFFECTIVE_RAM_CAP+1)):
            with self.assertRaisesRegex(ValueError, 'amended384GiB'):
                worker.effective_check_ram()

    def test_original_pool_parameters_unchanged(self):
        with patch.object(pipeline, 'ProcessPoolExecutor', return_value='pool') as pool:
            result = pipeline.pool_for({'workers':16,'cache_namespace':'unchanged'}, {'parent':'binding'}, 'build', 24*2**30)
        self.assertEqual(result, 'pool')
        self.assertEqual(pool.call_args.kwargs['max_workers'], 16)
        self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')
        self.assertIs(pool.call_args.kwargs['initializer'], pipeline.worker_init)
        self.assertEqual(pool.call_args.kwargs['initargs'], ({'parent':'binding'},'unchanged','build',24*2**30))

    def test_qualifier_receives_amended_guard_only(self):
        with patch.object(pipeline, 'RAM_CAP', 64*2**30), patch.object(pipeline, 'check_ram', ORIGINAL_CHECK_RAM):
            worker.apply_ram_authority()
            from research.joint_evaluation_20261029.argmix import gpu_qualification
            self.assertIs(gpu_qualification.check_ram, worker.effective_check_ram)
            with patch.object(gpu_qualification, 'qualify') as qualify:
                args = SimpleNamespace(stage='qualify-gpu')
                worker.dispatch(args, 'protocol', 'freeze')
                qualify.assert_called_once_with(args, 'protocol', 'freeze')

    def altered(self, key, value):
        changed = copy.deepcopy(self.amendment)
        changed[key] = value
        original_read = records.read
        with patch.object(records, 'read', side_effect=lambda ref: changed if ref == self.ref else original_read(ref)):
            worker.validate_amendment(self.ref, FREEZE, ADMISSION)

    def test_unapproved_amendment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'root-authorized'):
            self.altered('root_authorized', False)

    def test_different_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Exact reviewed'):
            self.altered('owned_RAM_max_bytes', 512*2**30)

    def test_weaker_host_gate_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Exact reviewed'):
            self.altered('minimum_host_available_bytes', 128*2**30)

    def test_different_worker_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sixteen workers'):
            self.altered('workers', 8)

    def test_storage_increase_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Storage and GPU'):
            self.altered('cache_max_bytes', 48*2**30)

    def test_missing_replacement_measurement_rejected(self):
        profile = records.read(self.amendment['memory_profile_ref'])
        profile['observations'] = [row for row in profile['observations'] if row['stage'] != 'second_large_parent_replaced']
        with self.assertRaisesRegex(ValueError, 'replacement transient'):
            worker.measured_bound(profile)

    def test_profile_bound_cannot_be_understated(self):
        with self.assertRaisesRegex(ValueError, 'Measured cold'):
            self.altered('conservative_17_process_peak_bytes', 1)

    def test_original_failed_identity_is_rejected(self):
        failed = {'pid': 999998, 'process_start_ticks': '12345'}
        stat = '999998 (original) ' + ' '.join(['S'] + ['0']*18 + ['12345'])
        with patch.object(Path, 'read_text', return_value=stat):
            with self.assertRaisesRegex(ValueError, 'original process identity'):
                worker.require_old_attempt_drained(failed)

    def test_unrelated_reused_pid_is_allowed(self):
        failed = {'pid': 999998, 'process_start_ticks': '12345'}
        stat = '999998 (unrelated) ' + ' '.join(['R'] + ['0']*18 + ['99999'])
        with patch.object(Path, 'read_text', return_value=stat), patch.object(worker.processes, 'session_members', side_effect=AssertionError('Do not inspect unrelated reused session')):
            worker.require_old_attempt_drained(failed)

    def test_absent_leader_surviving_session_is_rejected(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError), patch.object(worker.processes, 'session_members', return_value=[{'pid':999997}]):
            with self.assertRaisesRegex(ValueError, 'session still owns'):
                worker.require_old_attempt_drained({'pid':999998,'process_start_ticks':'12345'})

    def test_all_original_task_command_variants_check_only(self):
        plan = records.read(records.ref('research/joint_evaluation_20261029/argmix/task_plan_v2.json'))
        checked = 0
        for task in plan['tasks']:
            commands = list(task.get('argv_by_physical_gpu', {}).values()) or [task['argv']]
            for command in commands:
                argv = ['worker'] + command[4:] + ['--resource-amendment', AMENDMENT,
                    '--resource-amendment-sha256', self.ref['sha256'], '--check-only']
                with self.subTest(task=task['id']), patch('sys.argv', argv), patch.object(worker, 'dispatch', side_effect=AssertionError('Production execution forbidden')), contextlib.redirect_stdout(io.StringIO()) as captured:
                    worker.main()
                    self.assertTrue(json.loads(captured.getvalue())['valid'])
                checked += 1
        self.assertEqual(checked, 37)

    def test_no_original_or_retained_cache_bytes_change(self):
        seal = records.read(records.ref(FREEZE))
        for ref in seal['source_refs']:
            self.assertEqual(records.ref(ref['path'])['sha256'], ref['sha256'])
        for ref in self.amendment['retained_cache_files']:
            self.assertEqual(records.ref(ref['path'])['sha256'], ref['sha256'])
            self.assertEqual(records.path(ref['path']).stat().st_size, ref['bytes'])
        failed = records.read(self.amendment['failed_cache_wait_ref'])
        self.assertEqual(failed['actual_returncode'], -15)
        self.assertFalse(failed['products_complete'])


if __name__ == '__main__':
    unittest.main()
