"""Focused synthetic historical-adoption and v2 route checks; metadata only."""
import copy
import inspect
import unittest
from unittest.mock import patch
from . import readable_workflow_v2 as queue
from . import queue_execute_v2 as execution
from . import queue_native_admission_v2 as admission
from . import queue_native_worker_v2 as native
from . import queue_reporting_v2 as reporting


class AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.plan_ref=dict(path='synthetic_plan',sha256='a'*64)
        self.snapshot_ref=dict(path='synthetic_snapshot',sha256='b'*64)
        self.previous_ref=dict(path='synthetic_previous_plan',sha256='c'*64)
        self.result_ref=dict(path='synthetic_completed_result',sha256='d'*64)
        historical=dict(id='historical',resource='gpu0',dependencies=[],factory_arguments={},
            factory_module='arggnn.benchmarking.synthetic',factory_function='factory',
            output_directory='experiments/results/synthetic_history_v2/historical',
            max_seconds=60,max_rss_bytes=2**30,max_output_bytes=2**20,
            max_cuda_reserved_bytes=16*2**30,minimum_free_gpu_bytes=18*2**30)
        fresh=dict(historical,id='fresh',physical_gpu=2,gpu_uuid=queue.shared_gpu.GPU_UUIDS[2],
            output_directory='experiments/results/synthetic_history_v2/fresh')
        self.plan=dict(schema=queue.SCHEMA,physical_gpu=5,physical_gpus=[2,3,5],cpu_parallelism=1,
            max_parallel_rss_bytes=4*2**30,host_ram_margin_bytes=0,sources=[dict(path='source',sha256='e'*64)],
            tasks=[historical,fresh],readable_execution=True,
            adopted_state_snapshot_ref=self.snapshot_ref,
            adopted_completed_task_fingerprints=dict(historical=queue.c.fingerprint(historical)))
        self.snapshot=dict(plan_ref=self.previous_ref,tasks=dict(historical=dict(status='succeeded',result_ref=self.result_ref)))
        self.previous=dict(tasks=[copy.deepcopy(historical)])

    def read(self,reference):
        if reference==self.plan_ref:return self.plan
        if reference==self.snapshot_ref:return self.snapshot
        if reference==self.previous_ref:return self.previous
        if reference==self.result_ref:return dict(value={})
        raise ValueError('Unexpected synthetic reference')

    def test_completed_original_descriptor_needs_no_new_target(self):
        with patch.object(queue.c,'read_ref',side_effect=self.read),patch.object(queue,'source_check'), \
             patch.object(queue.shared_gpu,'target',wraps=queue.shared_gpu.target) as target:
            self.assertEqual(queue.load_plan(self.plan_ref),self.plan)
            target.assert_called_once_with(self.plan['tasks'][1])
            state=queue.initial_state(self.plan,self.plan_ref)
            self.assertEqual(state['tasks']['historical']['status'],'succeeded')

    def test_pending_task_still_requires_device(self):
        self.plan['tasks'][1].pop('physical_gpu')
        self.plan['tasks'][1].pop('gpu_uuid')
        with patch.object(queue.c,'read_ref',side_effect=self.read),patch.object(queue,'source_check'):
            with self.assertRaisesRegex(RuntimeError,'Explicit physical'):queue.load_plan(self.plan_ref)

    def test_changed_historical_descriptor_cannot_forge_current_fingerprint(self):
        self.plan['tasks'][0]['max_seconds']=61
        self.plan['adopted_completed_task_fingerprints']['historical']=queue.c.fingerprint(self.plan['tasks'][0])
        with patch.object(queue.c,'read_ref',side_effect=self.read):
            with self.assertRaisesRegex(ValueError,'exact original/current'):queue.authenticated_adopted_completed(self.plan)

    def test_unsucceeded_snapshot_cannot_be_exempt(self):
        self.snapshot['tasks']['historical']['status']='pending'
        with patch.object(queue.c,'read_ref',side_effect=self.read):
            with self.assertRaisesRegex(ValueError,'succeeded snapshot'):queue.authenticated_adopted_completed(self.plan)

    def test_current_state_cannot_reopen_historical_success(self):
        state=dict(tasks=dict(historical=dict(status='pending',result_ref=self.result_ref)))
        with patch.object(queue.c,'read_ref',side_effect=self.read):
            with self.assertRaisesRegex(ValueError,'cannot become pending'):queue.validate_adopted_state(self.plan,state)

    def test_historical_completed_job_cannot_execute_directly(self):
        job_ref=dict(path='synthetic_old_job',sha256='f'*64)
        task=self.plan['tasks'][0]
        job=dict(schema='arggnn_workflow_job_v1',plan_ref=self.plan_ref,task=task,
            task_fingerprint=queue.c.fingerprint(task),source_refs=[])
        def read(reference):return job if reference==job_ref else self.read(reference)
        with patch.object(queue.c,'read_ref',side_effect=read),patch.object(queue,'load_plan',return_value=self.plan):
            with self.assertRaisesRegex(ValueError,'cannot execute again'):queue.job_spec(job_ref)

    def test_v2_native_validation_chain_is_consistent(self):
        self.assertIs(admission.workflow,queue)
        self.assertIn('runtime.queue_native_worker_v2',inspect.getsource(execution))
        self.assertNotIn('from .queue_native_admission_v1',inspect.getsource(native))
        self.assertIn('from .queue_native_admission_v2',inspect.getsource(native))

    def test_parallel_reporter_has_distinct_destinations(self):
        self.assertEqual(reporting.START,'<!-- readable-parallel-queue:start -->')
        source=inspect.getsource(reporting)
        self.assertIn('research/benchmark_parallel_current.json',source)
        self.assertNotIn('research/benchmark_readable_current.json',source)
        self.assertIn('from .queue_reporting_v2 import publish',inspect.getsource(queue.ProgressReporter))
