"""Finite CPU tests for full-device admission and preserved runtime guards."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from . import queue_gpu_v1 as gpu
from . import scientific_setup_v1 as setup
from . import native_telemetry_v1 as telemetry

GIB=2**30


def task(index=2):
    return dict(id='test_full_device',resource='gpu0',physical_gpu=index,
        gpu_uuid=gpu.GPU_UUIDS[index],gpu_memory_policy='full_device',
        minimum_free_gpu_bytes=64*GIB)


def sample(free=70*GIB):
    return dict(physical_gpu=2,uuid=gpu.GPU_UUIDS[2],total_bytes=80*GIB,
        free_bytes=free,used_bytes=80*GIB-free)


class FullDeviceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.stack=ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.temp.cleanup)
        self.stack.enter_context(patch.object(gpu,'CAPACITY',self.root/'capacity.lock'))
        self.stack.enter_context(patch.object(gpu,'physical_path',side_effect=lambda index:self.root/('gpu_'+str(index)+'.lock')))
        self.stack.enter_context(patch.object(gpu,'pool_directory',side_effect=lambda index:self.root/('pool_'+str(index))))
        self.stack.enter_context(patch.object(gpu,'gpu_sample',return_value=sample()))
        self.plan=dict(physical_gpus=[2,3,5],tasks=[])
        self.job=dict(task=task(),plan_ref=dict(path='test_plan.json',sha256='a'*64))
        self.ref=dict(path='test_job.json',sha256='b'*64)
        self.stack.enter_context(patch.object(gpu,'validate_job_target',return_value=self.plan))

    def test_actual_free_threshold_has_no_allocator_margin_or_floor(self):
        candidate=task()
        candidate['max_cuda_reserved_bytes']=80*GIB
        self.assertIsNone(gpu._admission_reason_locked(self.plan,candidate))
        with patch.object(gpu,'gpu_sample',return_value=sample(63*GIB)):
            self.assertIn('actual free memory',gpu._admission_reason_locked(self.plan,candidate))

    def test_legacy_shared_lease_blocks_full_device(self):
        with gpu.physical_path(2).open('a') as legacy:
            fcntl.flock(legacy,fcntl.LOCK_SH|fcntl.LOCK_NB)
            self.assertIn('preserved',gpu.admission_reason(self.plan,task()))
            with self.assertRaises(BlockingIOError):
                with gpu.reserve(self.job,self.ref):self.fail('Legacy job must retain its GPU')
        with gpu.reserve(self.job,self.ref) as descriptor:
            row=json.loads(os.pread(descriptor,65536,0))
            self.assertEqual(row['physical_lease'],'exclusive')
            self.assertFalse(row['artificial_allocator_limit'])

    def test_exclusive_lease_prevents_second_owned_worker(self):
        with gpu.reserve(self.job,self.ref):
            with self.assertRaises(BlockingIOError):
                with gpu.reserve(self.job,self.ref):self.fail('Second owner must wait')
            with gpu.physical_path(2).open('a') as legacy:
                with self.assertRaises(BlockingIOError):fcntl.flock(legacy,fcntl.LOCK_SH|fcntl.LOCK_NB)

    def test_low_free_memory_and_full_usage_are_observed_without_cancellation(self):
        from research.repairs.benchmark_repair_20261022_afternoon.runtime_v2 import queue_gpu_v1 as legacy
        with gpu.reserve(self.job,self.ref) as descriptor:
            tracker=SimpleNamespace(capture=lambda:[os.getpid()])
            with patch.object(gpu,'gpu_sample',return_value=sample(GIB)),patch.object(legacy,'process_bytes',return_value={os.getpid():79*GIB}):
                for unused in range(4):gpu.monitor(self.job,self.ref,descriptor,tracker)
            row=json.loads(os.pread(descriptor,65536,0))
            self.assertEqual(row['last_sample']['owned_driver_bytes'],79*GIB)
            self.assertEqual(row['last_sample']['free_bytes'],GIB)

    def test_full_device_environment_rejects_stale_caps_and_wrong_uuid(self):
        environment=dict(ARGGNN_PHYSICAL_GPU='2',CUDA_VISIBLE_DEVICES=gpu.GPU_UUIDS[2],ARGGNN_GPU_MEMORY_POLICY='full_device')
        with patch.dict(os.environ,environment,clear=True):
            gpu.validate_policy_environment(task())
            os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES']=str(32*GIB)
            with self.assertRaises(RuntimeError):gpu.validate_policy_environment(task())
            del os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES']
            os.environ['CUDA_VISIBLE_DEVICES']=gpu.GPU_UUIDS[3]
            with self.assertRaises(RuntimeError):gpu.validate_policy_environment(task())

    def test_scientific_setup_retains_precision_and_authenticates_without_fraction(self):
        cuda=SimpleNamespace(device_count=lambda:1,get_device_properties=lambda index:SimpleNamespace(total_memory=80*GIB),
            set_per_process_memory_fraction=Mock(side_effect=AssertionError('Artificial fraction must never be set')))
        torch=SimpleNamespace(cuda=cuda,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),cudnn=SimpleNamespace(allow_tf32=True)))
        with patch.object(setup.gpu_binding,'verify_before_cuda') as identity_check,patch.object(setup,'validate_worker_lease') as lease,patch.object(setup,'validate_policy_environment'):
            setup.finish_gpu_setup(torch,task())
            identity_check.assert_called_once_with(torch,task())
            lease.assert_called_once_with(first_cuda=True)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        cuda.set_per_process_memory_fraction.assert_not_called()

    def test_full_batch_telemetry_has_no_cap_and_preserves_actual_counters(self):
        cuda=SimpleNamespace(memory_reserved=lambda index:79*GIB,memory_allocated=lambda index:78*GIB,
            max_memory_allocated=lambda index:78*GIB,max_memory_reserved=lambda index:79*GIB,
            get_device_properties=lambda index:SimpleNamespace(total_memory=80*GIB))
        state=SimpleNamespace(timestamp=SimpleNamespace(batch=SimpleNamespace(value=17)))
        check=Mock()
        with patch.object(gpu,'validate_policy_environment'):
            callback=telemetry.memory_callback(object,SimpleNamespace(cuda=cuda),self.root,check)
            callback.batch_start(state,None)
            callback.batch_end(state,None)
        first=json.loads((self.root/'first_full_batch_memory.json').read_text())
        self.assertEqual(first['batch'],17)
        self.assertEqual(first['reserved_bytes'],79*GIB)
        self.assertFalse(first['artificial_allocator_limit'])
        self.assertTrue(first['full_native_batch_and_microbatch_preserved'])
        check.assert_called_once_with('native_full_batch_completed')

    def test_non_gpu_guards_and_original_source_remain_preserved(self):
        source=Path('research/repairs/benchmark_repair_20261022_afternoon/runtime_v2')
        successor=Path(__file__).parent
        import ast
        for name in ('readable_workflow_v1.py','queue_execute_v1.py'):
            old=ast.parse((source/name).read_text())
            new=ast.parse((successor/name).read_text())
            old_functions={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
            new_functions={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
            intentional={'current_policy','load_plan'} if name=='readable_workflow_v1.py' else {'child','execute'}
            for identifier,node in old_functions.items():
                if identifier not in intentional:
                    self.assertEqual(ast.dump(node),ast.dump(new_functions[identifier]),identifier)
        execution=(successor/'queue_execute_v1.py').read_text()
        for guard in ("tracker.measure() <= task['max_rss_bytes']","lifecycle.scope_bytes(output) <= task['max_output_bytes']",'Scientific wall limit exceeded','No output reuse or automatic retry','actual_direct_returncode'):
            self.assertIn(guard,execution)
        workflow=(successor/'readable_workflow_v1.py').read_text()
        for guard in ('plan["host_ram_margin_bytes"]>=64*2**30','plan["max_parallel_rss_bytes"]<=1792*2**30','980000000000','source_check(job["source_refs"])'):
            self.assertIn(guard,workflow)


if __name__=='__main__':unittest.main()
