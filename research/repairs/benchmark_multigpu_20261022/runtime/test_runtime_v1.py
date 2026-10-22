"""CPU-only parallel release, device isolation and inherited-lease checks."""
import ast
from contextlib import ExitStack
from datetime import datetime,timedelta,timezone
import fcntl
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from . import contracts_parallel_v1 as c
from . import shared_gpu_v1 as hardware
from . import queue_gpu_v1 as pool
from . import readable_workflow_v1 as workflow
from . import queue_execute_v1 as execution
from . import queue_native_worker_v1 as native

HERE=Path(__file__).parent
OLD=c.ROOT/'research/repairs/benchmark_resume_20261021/gpu'


def task(index,identifier='synthetic'):
    return dict(id=identifier,resource='gpu0',physical_gpu=index,gpu_uuid=hardware.GPU_UUIDS[index],
        max_cuda_reserved_bytes=16*2**30,minimum_free_gpu_bytes=18*2**30)


class DeviceTests(unittest.TestCase):
    def test_all_explicit_device_targets(self):
        for index in (2,3,5):
            with self.subTest(index=index):self.assertEqual(hardware.target(task(index)),(index,hardware.GPU_UUIDS[index]))

    def test_wrong_or_implicit_device_rejected(self):
        for value in [{},dict(physical_gpu=0),dict(physical_gpu=True),dict(physical_gpu='2'),
                      dict(physical_gpu=2,gpu_uuid=hardware.GPU_UUIDS[3]),dict(physical_gpu=5,gpu_uuid='foreign')]:
            with self.subTest(value=value),self.assertRaises(RuntimeError):hardware.target(value)

    def test_environment_must_match_exact_task(self):
        for index in (2,3,5):
            with patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU=str(index),CUDA_VISIBLE_DEVICES=hardware.GPU_UUIDS[index]),clear=True):
                self.assertEqual(hardware.validate_visible_target(task(index))[0],index)
                other=3 if index==2 else 2
                with self.assertRaises(RuntimeError):hardware.validate_visible_target(task(other))
        with patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU='2',CUDA_VISIBLE_DEVICES='2,3'),clear=True):
            with self.assertRaises(RuntimeError):hardware.validate_visible_target()

    def test_inventory_each_device_and_foreign_rejection(self):
        for index in (2,3,5):
            runner=Mock(return_value=SimpleNamespace(stdout=f'{index}, {hardware.GPU_UUIDS[index]}, 81920, 10, 81908'))
            self.assertEqual(hardware.gpu_sample(index,run=runner)['physical_gpu'],index)
            self.assertEqual(runner.call_args.args[0][1],'--id='+str(index))
        for line in ['3, '+hardware.GPU_UUIDS[3]+', 81920, 10, 81908',
                     '2, '+hardware.GPU_UUIDS[5]+', 81920, 10, 81908',
                     '2, '+hardware.GPU_UUIDS[2]+', 10, 20, 40']:
            with self.assertRaises(RuntimeError):hardware.gpu_sample(2,run=Mock(return_value=SimpleNamespace(stdout=line)))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(path='old_plan',sha256='a'*64);self.new=dict(path='parallel_plan',sha256='b'*64)
        self.reference=dict(path='canonical',sha256='c'*64)
        self.policy=dict(benchmark_execution_approved=True,status='approved',physical_gpus=[5],
            additional_physical_gpus=[2,3],user_authorized_physical_gpus=[2,3,5],
            approval_origin='explicit_user_go_ahead',smoke_tests_only=False,
            expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            approved_readable_workflow_ref=self.old,approved_parallel_workflow_ref=self.new)

    def check(self,policy=None,reference=None):
        value=self.policy if policy is None else policy
        with patch.object(c,'read_json',return_value=value),patch.object(c,'read_ref',return_value=value), \
             patch.object(c,'file_ref',return_value=self.reference):
            return c.check_execution_release(self.reference if reference is None else reference)

    def test_additional_authority_keeps_primary_fields(self):
        result=self.check()
        self.assertEqual(result['physical_gpus'],[5]);self.assertEqual(result['approved_readable_workflow_ref'],self.old)
        from research.repairs.benchmark_resume_20261021.gpu import contracts_gpu5_v1 as primary
        with patch.object(primary,'read_json',return_value=self.policy),patch.object(primary,'read_ref',return_value=self.policy), \
             patch.object(primary,'file_ref',return_value=self.reference):
            self.assertEqual(primary.check_execution_release(self.reference)['approved_readable_workflow_ref'],self.old)

    def test_all_parallel_authority_and_lifetime_gates(self):
        failures=[dict(physical_gpus=[2,3,5]),dict(additional_physical_gpus=[2]),
            dict(user_authorized_physical_gpus=[2,3]),dict(approved_parallel_workflow_ref=None),
            dict(benchmark_execution_approved=False),dict(status='held'),dict(smoke_tests_only=True),
            dict(approval_origin='historical'),dict(expires_at_utc='2000-01-01T00:00:00+00:00')]
        for delta in failures:
            with self.subTest(delta=delta),self.assertRaises(c.ContractError):self.check(dict(self.policy,**delta))
        with self.assertRaises(c.ContractError):self.check(reference=dict(path='stale',sha256='d'*64))

    def test_parallel_mapping_ignores_primary_drain(self):
        policy=dict(self.policy,readable_workflow_drain_ref=self.old)
        with patch.object(c,'check_execution_release',return_value=policy),patch.object(c,'file_ref',return_value=self.reference):
            current=workflow.current_policy()
            self.assertEqual(current['approved_workflow_ref'],self.new)
            self.assertNotIn('workflow_drain_ref',current)
            policy['parallel_workflow_drain_ref']=self.new
            self.assertEqual(workflow.current_policy()['workflow_drain_ref'],self.new)
        self.assertEqual(policy['approved_readable_workflow_ref'],self.old)

    def test_job_requires_exact_parallel_approval(self):
        job=dict(plan_ref=self.new,task=task(2))
        with patch.object(c,'read_ref',return_value=dict(physical_gpus=[2,3,5])), \
             patch.object(c,'file_ref',return_value=self.reference),patch.object(c,'check_execution_release',return_value=self.policy):
            self.assertEqual(pool.validate_job_target(job)['physical_gpus'],[2,3,5])
            with self.assertRaises(RuntimeError):pool.validate_job_target(dict(job,plan_ref=self.old))


class PoolTests(unittest.TestCase):
    def test_device_pool_and_physical_lease_separation(self):
        self.assertEqual(len({pool.record_path(task(index)) for index in (2,3,5)}),3)
        self.assertEqual(len({pool.physical_path(index) for index in (2,3,5)}),3)
        self.assertEqual(len(pool.accounting_pool_directories(2)),1)
        self.assertEqual(len(pool.accounting_pool_directories(5)),2)
        self.assertTrue(str(pool.CAPACITY).endswith('benchmark_recovery_capacity_20261020.lock'))

    def test_legacy_GPU5_growth_charged_only_on_GPU5(self):
        owner=hardware.identity(os.getpid());plan=dict(tasks=[],physical_gpus=[2,3,5])
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_legacy_') as directory,ExitStack() as stack:
            stack.enter_context(patch.object(c,'ROOT',Path(directory)))
            stack.enter_context(patch.object(pool,'process_bytes',return_value={owner['pid']:4*2**30}))
            legacy=pool.accounting_pool_directories(5)[1];legacy.mkdir(parents=True)
            descriptor=os.open(legacy/'existing.json',os.O_RDWR|os.O_CREAT,0o600)
            try:
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                row=dict(job_ref=dict(path='legacy',sha256='a'*64),cap_bytes=40*2**30,members=[owner],
                    physical_gpu=5,gpu_uuid=hardware.GPU_UUIDS[5])
                os.write(descriptor,json.dumps(row).encode())
                self.assertEqual(pool._outstanding_growth_locked(plan,5),38*2**30)
                self.assertEqual(pool._outstanding_growth_locked(plan,2),0)
                self.assertEqual(pool._outstanding_growth_locked(plan,3),0)
            finally:os.close(descriptor)

    def test_simultaneous_device_reservations_and_inherited_guards(self):
        from arggnn.benchmarking import workflow_v4 as lifecycle
        plan=dict(physical_gpus=[2,3,5],tasks=[]);owner=hardware.identity(os.getpid())
        root=dict(owner,pgid=os.getpid(),session=os.getpid())
        with tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_pools_') as directory,ExitStack() as stack:
            temporary=Path(directory)
            stack.enter_context(patch.object(c,'ROOT',temporary))
            stack.enter_context(patch.object(c,'read_ref',return_value=plan))
            stack.enter_context(patch.object(pool,'CAPACITY',temporary/'.runtime/locks/capacity.lock'))
            stack.enter_context(patch.object(pool,'validate_job_target',return_value=plan))
            stack.enter_context(patch.object(pool,'gpu_sample',return_value=dict(free_bytes=80*2**30)))
            stack.enter_context(patch.object(pool,'process_bytes',return_value={}))
            stack.enter_context(patch.object(lifecycle,'identity',return_value=root))
            held=[]
            for index in (2,3,5):
                reference=dict(path='job'+str(index),sha256=str(index)*64)
                job=dict(plan_ref=dict(path='plan',sha256='a'*64),task=task(index))
                descriptor=stack.enter_context(pool.reserve(job,reference))
                pool.set_root(descriptor,root);held.append((job,reference,descriptor))
            self.assertEqual(len({os.fstat(fd).st_ino for _,_,fd in held}),3)
            for job,reference,descriptor in held:
                index=job['task']['physical_gpu']
                with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES=hardware.GPU_UUIDS[index],
                    ARGGNN_PHYSICAL_GPU=str(index),ARGGNN_MAX_CUDA_RESERVED_BYTES=str(16*2**30),
                    ARGGNN_READABLE_GPU_FD=str(descriptor)),clear=True):
                    self.assertEqual(pool.validate_child(job,reference,first_cuda=True),job)
                    self.assertEqual(len(pool.inherited(descriptor)),2)
                    with patch.object(pool,'identity',return_value=dict(owner,start_ticks=owner['start_ticks']+1)):
                        with self.assertRaisesRegex(RuntimeError,'generation'):pool.validate_child(job,reference)
                    other=held[1][2] if index==2 else held[0][2]
                    os.environ['ARGGNN_READABLE_GPU_FD']=str(other)
                    with self.assertRaisesRegex(RuntimeError,'assigned physical'):pool.validate_child(job,reference)

    def test_per_device_growth_and_floor_admission(self):
        with patch.object(pool,'gpu_sample',return_value=dict(free_bytes=30*2**30)), \
             patch.object(pool,'_outstanding_growth_locked',return_value=5*2**30):
            self.assertIsNotNone(pool._admission_reason_locked(dict(physical_gpus=[2,3,5]),task(2)))
        with patch.object(pool,'gpu_sample',return_value=dict(free_bytes=31*2**30)), \
             patch.object(pool,'_outstanding_growth_locked',return_value=5*2**30):
            self.assertIsNone(pool._admission_reason_locked(dict(physical_gpus=[2,3,5]),task(3)))


class SourceTests(unittest.TestCase):
    def test_all_runtime_source_parses(self):
        for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))

    def test_native_science_functions_are_unchanged(self):
        before=(OLD/'queue_native_worker_v1.py').read_text();after=(HERE/'queue_native_worker_v1.py').read_text()
        old={node.name:ast.get_source_segment(before,node) for node in ast.parse(before).body if isinstance(node,ast.FunctionDef)}
        new={node.name:ast.get_source_segment(after,node) for node in ast.parse(after).body if isinstance(node,ast.FunctionDef)}
        self.assertEqual(set(old),set(new))
        for name in old:
            if name!='_native':self.assertEqual(old[name],new[name],name)
        guard=inspect.getsource(native._native)
        self.assertIn('validate_visible_target()',guard)
        self.assertLess(guard.index('validate_worker_lease()'),guard.index('import torch'))
        self.assertLess(guard.index('validate_worker_lease(first_cuda=True)'),guard.index('torch.cuda.device_count'))

    def test_native_worker_rewrite_and_callback_remain_explicit(self):
        source=inspect.getsource(execution)
        self.assertIn('research.repairs.benchmark_multigpu_20261022.runtime.queue_native_worker_v1',source)
        self.assertIn('def acquire_gpu0()',source)
        self.assertIn('ARGGNN_PHYSICAL_GPU=str(gpu.target(task)[0])',source)
        self.assertNotIn('GPU5_UUID',source)
        self.assertIn('shared_gpu.target(task)',inspect.getsource(workflow.load_plan))
        self.assertIn('entry.pop("waiting_reason",None)',inspect.getsource(workflow.run))


if __name__=='__main__':unittest.main(verbosity=2)
