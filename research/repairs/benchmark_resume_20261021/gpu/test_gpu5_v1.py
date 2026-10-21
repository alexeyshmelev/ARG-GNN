"""Synthetic CPU-only checks for GPU5 release, inventory and inherited leases."""
import ast
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import inspect
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import contracts_gpu5_v1 as contracts
from . import shared_gpu5_v1 as hardware
from . import queue_gpu5_v1 as pool
from . import readable_workflow_v1 as workflow
from . import native_scientific_gpu5_v1 as native
from .queue_adapters import common

HERE = Path(__file__).parent
OLD = contracts.ROOT/'research/repairs/benchmark_recovery_20261020'


class InventoryTests(unittest.TestCase):
    def sample(self, line):
        runner = Mock(return_value=SimpleNamespace(stdout=line))
        result = hardware.gpu_sample(run=runner)
        self.assertEqual(runner.call_args.args[0][1], '--id=5')
        return result

    def test_exact_gpu5_inventory(self):
        sample = self.sample('5, '+hardware.GPU5_UUID+', 81920, 10240, 71678\n')
        self.assertEqual(sample['physical_gpu'],5)
        self.assertEqual(sample['free_bytes'],71678*2**20)
        self.assertEqual(sample['driver_reserved_bytes'],2*2**20)

    def test_wrong_index_or_uuid_or_multi_device_rejected(self):
        for line in ['0, '+hardware.GPU5_UUID+', 80, 10, 70',
                     '5, GPU-foreign, 80, 10, 70',
                     '5, '+hardware.GPU5_UUID+', 80, 10, 70\n5, '+hardware.GPU5_UUID+', 80, 10, 70']:
            with self.subTest(line=line), self.assertRaises(RuntimeError):self.sample(line)

    def test_inconsistent_inventory_rejected(self):
        with self.assertRaises(RuntimeError):self.sample('5, '+hardware.GPU5_UUID+', 80, 90, 70')

    def test_compute_rows_must_name_target_uuid(self):
        with patch.object(pool.subprocess,'run',return_value=SimpleNamespace(stdout='GPU-foreign, 123, 200')):
            with self.assertRaises(RuntimeError):pool.process_bytes()


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(path='canonical.json',sha256='a'*64)
        self.policy = dict(benchmark_execution_approved=True,status='approved',physical_gpus=[5],
            approval_origin='explicit_user_go_ahead',smoke_tests_only=False,
            expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())

    def check(self, policy=None, reference=None):
        policy = self.policy if policy is None else policy
        with patch.object(contracts,'read_json',return_value=policy), \
             patch.object(contracts,'file_ref',return_value=self.reference), \
             patch.object(contracts,'read_ref',return_value=policy):
            return contracts.check_execution_release(self.reference if reference is None else reference)

    def test_exact_release(self):self.assertEqual(self.check(),self.policy)

    def test_all_release_gates_remain(self):
        failures=[dict(physical_gpus=[0]),dict(physical_gpus=[0,5]),dict(physical_gpus=['5']),
            dict(benchmark_execution_approved=False),dict(status='held'),dict(approval_origin='historical'),
            dict(smoke_tests_only=True),dict(expires_at_utc='2000-01-01T00:00:00+00:00'),
            dict(expires_at_utc='2099-01-01T00:00:00')]
        for delta in failures:
            with self.subTest(delta=delta),self.assertRaises(contracts.ContractError):self.check(dict(self.policy,**delta))
        with self.assertRaises(contracts.ContractError):self.check(reference=dict(path='stale',sha256='b'*64))

    def test_pool_requires_current_exact_plan(self):
        ref=dict(path='plan',sha256='a'*64)
        with patch.object(contracts,'read_ref',return_value=dict(physical_gpu=5)), \
             patch.object(contracts,'file_ref',return_value={}), \
             patch.object(contracts,'check_execution_release',return_value=dict(approved_readable_workflow_ref=ref)):
            self.assertEqual(pool.validate_job_target(dict(plan_ref=ref)),dict(physical_gpu=5))
            with self.assertRaises(RuntimeError):pool.validate_job_target(dict(plan_ref=dict(ref,sha256='b'*64)))
        with patch.object(contracts,'read_ref',return_value=dict(physical_gpu=0)):
            with self.assertRaises(RuntimeError):pool.validate_job_target(dict(plan_ref=ref))


class PoolTests(unittest.TestCase):
    def test_record_and_growth_bound(self):
        pool.validate_record_target(dict(physical_gpu=5,gpu_uuid=hardware.GPU5_UUID))
        for row in [{},dict(physical_gpu=0,gpu_uuid=hardware.GPU5_UUID),dict(physical_gpu=5,gpu_uuid='foreign')]:
            with self.assertRaises(RuntimeError):pool.validate_record_target(row)
        task=dict(max_cuda_reserved_bytes=16*2**30,minimum_free_gpu_bytes=18*2**30)
        with patch.object(pool,'gpu_sample',return_value=dict(free_bytes=30*2**30)), \
             patch.object(pool,'_outstanding_growth_locked',return_value=5*2**30):
            self.assertIsNotNone(pool._admission_reason_locked(dict(physical_gpu=5),task))
            with self.assertRaises(RuntimeError):pool._admission_reason_locked(dict(physical_gpu=0),task)
        with patch.object(pool,'gpu_sample',return_value=dict(free_bytes=31*2**30)), \
             patch.object(pool,'_outstanding_growth_locked',return_value=5*2**30):
            self.assertIsNone(pool._admission_reason_locked(dict(physical_gpu=5),task))

    def test_real_inherited_descriptor_inode_and_generation(self):
        from arggnn.benchmarking import workflow_v4 as lifecycle
        reference=dict(path='synthetic_job',sha256='a'*64)
        plan=dict(physical_gpu=5,tasks=[])
        task=dict(id='synthetic',max_cuda_reserved_bytes=2**30,minimum_free_gpu_bytes=2**30)
        job=dict(plan_ref=dict(path='synthetic_plan',sha256='b'*64),task=task)
        owner=hardware.identity(os.getpid())
        root=dict(owner,pgid=os.getpid(),session=os.getpid())
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_pool_') as directory,ExitStack() as stack:
            temporary=Path(directory)
            stack.enter_context(patch.object(contracts,'ROOT',temporary))
            stack.enter_context(patch.object(contracts,'read_ref',return_value=plan))
            stack.enter_context(patch.object(pool,'POOL',temporary/'.runtime/locks/pool'))
            stack.enter_context(patch.object(pool,'CAPACITY',temporary/'.runtime/locks/capacity.lock'))
            stack.enter_context(patch.object(pool,'validate_job_target',return_value=plan))
            stack.enter_context(patch.object(pool,'gpu_sample',return_value=dict(free_bytes=80*2**30)))
            stack.enter_context(patch.object(pool,'process_bytes',return_value={}))
            stack.enter_context(patch.object(lifecycle,'identity',return_value=root))
            descriptor=stack.enter_context(pool.reserve(job,reference))
            pool.set_root(descriptor,root)
            stack.enter_context(patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES=hardware.GPU5_UUID,
                ARGGNN_PHYSICAL_GPU='5',ARGGNN_MAX_CUDA_RESERVED_BYTES=str(2**30),
                ARGGNN_READABLE_GPU_FD=str(descriptor)),clear=True))
            self.assertEqual(pool.validate_child(job,reference,first_cuda=True),job)
            self.assertEqual(len(pool.inherited(descriptor)),2)
            with patch.object(pool,'identity',return_value=dict(owner,start_ticks=owner['start_ticks']+1)):
                with self.assertRaisesRegex(RuntimeError,'generation'):pool.validate_child(job,reference)
            path=pool.record_path(task)
            path.rename(path.with_suffix('.original'))
            path.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'logical GPU'):pool.validate_child(job,reference)


class AdapterTests(unittest.TestCase):
    def fake_torch(self):
        cuda=SimpleNamespace(device_count=Mock(return_value=1),
            get_device_properties=Mock(return_value=SimpleNamespace(total_memory=80*2**30)),
            set_per_process_memory_fraction=Mock())
        return SimpleNamespace(cuda=cuda,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True)))

    def test_gpu5_allocator_setup(self):
        torch=self.fake_torch()
        with patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU='5',CUDA_VISIBLE_DEVICES=hardware.GPU5_UUID,
            ARGGNN_READABLE_GPU_FD='999',ARGGNN_MAX_CUDA_RESERVED_BYTES=str(16*2**30)),clear=True):
            common.finish_gpu_setup(torch)
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(.2,0)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_gpu0_visibility_fails_before_cuda_calls(self):
        torch=self.fake_torch()
        with patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU='0',CUDA_VISIBLE_DEVICES=hardware.GPU5_UUID),clear=True):
            with self.assertRaises(contracts.ContractError):common.finish_gpu_setup(torch)
        torch.cuda.device_count.assert_not_called()

    def test_native_wrapper_rejects_wrong_physical_before_scope(self):
        output=Mock()
        with patch.object(native.api,'validate_spec',return_value=output), \
             patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU='0',CUDA_VISIBLE_DEVICES=hardware.GPU5_UUID),clear=True):
            with self.assertRaises(ValueError):native.run_argformer(dict(method='argformer',stage='mlm'),
                bounded_runner=Mock(),resource_check=Mock())
        output.mkdir.assert_not_called()

    def test_adapter_descriptors_are_new_namespace_and_complete(self):
        call=common.descriptor('american_prediction','run_predictions',{})
        self.assertTrue(call['module'].startswith('research.repairs.benchmark_resume_20261021.gpu.queue_adapters.'))
        paths={r['path'] for r in call['source_refs']}
        for name in common.SOURCE_NAMES:self.assertIn(common.DIRECTORY+'/'+name,paths)
        for reference in call['source_refs']:self.assertEqual(contracts.file_ref(reference['path']),reference)

    def test_native_science_and_counters_unchanged(self):
        old=(OLD/'scheduler/queue_native_worker_v1.py').read_text()
        new=(HERE/'queue_native_worker_v1.py').read_text()
        old_nodes={n.name:ast.get_source_segment(old,n) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
        new_nodes={n.name:ast.get_source_segment(new,n) for n in ast.parse(new).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(old_nodes),set(new_nodes))
        for name,source in old_nodes.items():
            if name=='_native':continue
            for helper in ['native_telemetry_v1','activation_checkpoint_v1','native_seal_v1']:
                source=source.replace('from .'+helper+' import ',
                    'from research.repairs.benchmark_recovery_20261020.scheduler.'+helper+' import ')
            self.assertEqual(source,new_nodes[name],name)

    def test_source_files_parse_and_no_frozen_gpu0_guards(self):
        for path in HERE.rglob('*.py'):
            ast.parse(path.read_text(),filename=str(path))
        for name in ['queue_execute_v1.py','queue_gpu5_v1.py','queue_native_worker_v1.py',
                     'queue_adapters/common.py','queue_adapters/european_execution.py']:
            source=(HERE/name).read_text()
            self.assertNotIn('GPU0_UUID',source,name)
            self.assertNotIn('gpu_0.lock',source,name)
            self.assertNotIn('--id=0',source,name)
        self.assertIn('plan.get("physical_gpu") == 5',inspect.getsource(workflow.load_plan))


class ReportingTests(unittest.TestCase):
    def test_newest_training_progress_wins_over_old_preparation(self):
        from . import queue_reporting_v1 as reporting
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_report_') as directory:
            output=Path(directory);(output/'native').mkdir()
            preparation=output/'native/preparation_live.json'
            training=output/'native/training_live.json'
            preparation.write_text(json.dumps(dict(schema='treeloom_stage_progress_v1',parent_id='TRAIN000',
                latest=dict(preparation_stage='old preparation'))))
            training.write_text(json.dumps(dict(schema='treeloom_stage_progress_v1',parent_id='TRAIN000',
                latest=dict(last_mark='bounded_backward'),active_stages=[dict(stage='complete_parent_decoder')],
                peak_RSS_bytes=1234)))
            os.utime(preparation,ns=(100,100));os.utime(training,ns=(200,200))
            (output/'native/latest_checkpoint.json').write_text(json.dumps(dict(complete_parent_updates=3,epoch=1,parent_id='TRAIN000')))
            row=reporting.tree_progress(output)
            self.assertEqual(row['phase'],'training')
            self.assertEqual(row['stage'],'TreeLoom training: complete_parent_decoder')
            self.assertEqual(row['completed_parent_updates'],3)
            plan=dict(tasks=[dict(id='fit',resource='gpu0',output_directory=str(output))])
            state=dict(tasks=dict(fit=dict(status='running')))
            self.assertEqual(reporting.local_status(plan,state)['fit']['phase'],'training')
            os.utime(preparation,ns=(300,300))
            self.assertEqual(reporting.tree_progress(output)['phase'],'preparation')

    def test_missing_progress_is_explicit(self):
        from . import queue_reporting_v1 as reporting
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_report_') as directory:
            self.assertNotIn('phase',reporting.tree_progress(Path(directory)))


if __name__=='__main__':unittest.main(verbosity=2,testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
