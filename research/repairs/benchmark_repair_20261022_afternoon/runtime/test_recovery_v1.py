"""CPU-only recovery authority, predecessor binding and reservation checks."""
import ast
import copy
from datetime import datetime,timedelta,timezone
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch
from . import readable_workflow_v1 as queue,queue_gpu_v1 as pool,queue_execute_v1 as execution
from . import queue_native_admission_v1 as admission,queue_native_worker_v1 as native
from . import overlay_predecessor_v1 as predecessor,queue_reporting_v1 as reporting


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.old=dict(path='old',sha256='a'*64);self.new=dict(path='new',sha256='b'*64)
        self.ref=dict(path='canonical',sha256='c'*64)
        self.policy=dict(benchmark_execution_approved=True,status='approved',physical_gpus=[5],additional_physical_gpus=[2,3],
            user_authorized_physical_gpus=[2,3,5],approval_origin='explicit_user_go_ahead',smoke_tests_only=False,
            expires_at_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
            approved_readable_workflow_ref=self.old,approved_parallel_workflow_ref=self.old,approved_recovery_workflow_ref=self.new)

    def test_separate_recovery_policy_and_drain(self):
        c=queue.c
        with patch.object(c,'read_json',return_value=self.policy),patch.object(c,'read_ref',return_value=self.policy),patch.object(c,'file_ref',return_value=self.ref):
            self.assertEqual(c.check_execution_release(self.ref)['approved_parallel_workflow_ref'],self.old)
            self.policy['parallel_workflow_drain_ref']=self.old
            self.policy['parallel_resource_overlay_drain_ref']=self.old
            self.assertEqual(queue.current_policy()['approved_workflow_ref'],self.new)
            self.assertNotIn('workflow_drain_ref',queue.current_policy())
            self.policy['recovery_workflow_drain_ref']=self.new
            self.assertEqual(queue.current_policy()['workflow_drain_ref'],self.new)
            self.policy.pop('approved_recovery_workflow_ref')
            with self.assertRaises(c.ContractError):c.check_execution_release(self.ref)

    def test_pool_uses_recovery_approval_for_64GiB_task(self):
        task=dict(physical_gpu=2,gpu_uuid=pool.GPU_UUIDS[2],max_cuda_reserved_bytes=64*2**30,minimum_free_gpu_bytes=72*2**30)
        with patch.object(queue.c,'read_ref',return_value=dict(physical_gpus=[2,3,5])),patch.object(queue.c,'file_ref',return_value=self.ref),patch.object(queue.c,'check_execution_release',return_value=self.policy):
            pool.validate_job_target(dict(task=task,plan_ref=self.new))
            with self.assertRaises(RuntimeError):pool.validate_job_target(dict(task=task,plan_ref=self.old))
        with patch.object(pool,'gpu_sample',return_value=dict(free_bytes=79*2**30)),patch.object(pool,'_outstanding_growth_locked',return_value=0):
            self.assertIsNone(pool._admission_reason_locked(dict(physical_gpus=[2,3,5]),task))

    def test_full_caps_wait_until_one_256GiB_reservation_drains(self):
        rows=[dict(full_cap_bytes=cap*2**30,remaining_growth_bytes=0) for cap in (512,512,256,256)]
        plan=dict(max_parallel_rss_bytes=1792*2**30,host_ram_margin_bytes=64*2**30)
        task=dict(max_rss_bytes=512*2**30)
        with patch.object(queue.support,'managed_ram_rows',return_value=rows),patch.object(queue.support.accounting,'kernel_available_ram_bytes',return_value=1900*2**30):
            self.assertFalse(queue.support.ram_allows(plan,{},task))
            rows.pop()
            self.assertTrue(queue.support.ram_allows(plan,{},task))

    def test_static_recovery_native_chain_and_science(self):
        self.assertIs(admission.workflow,queue)
        self.assertIn('benchmark_repair_20261022_afternoon.runtime.queue_native_worker_v1',inspect.getsource(execution))
        old=(queue.c.ROOT/'research/repairs/benchmark_multigpu_20261022/runtime/queue_native_worker_v2.py').read_text()
        new=inspect.getsource(native)
        old_nodes={node.name:ast.get_source_segment(old,node) for node in ast.parse(old).body if isinstance(node,ast.FunctionDef)}
        new_nodes={node.name:ast.get_source_segment(new,node) for node in ast.parse(new).body if isinstance(node,ast.FunctionDef)}
        self.assertEqual(set(old_nodes),set(new_nodes))
        for name,source in old_nodes.items():self.assertEqual(source.replace('from .queue_native_admission_v2','from .queue_native_admission_v1'),new_nodes[name])
        self.assertIn('Historical completed tasks cannot execute again',inspect.getsource(queue.job_spec))
        self.assertIn('1792*2**30',inspect.getsource(queue.load_plan))
        self.assertEqual(reporting.START,'<!-- readable-recovery-afternoon-queue:start -->')
        self.assertIn('research/benchmark_recovery_afternoon_current.json',inspect.getsource(reporting))


class OverlayPredecessorTests(unittest.TestCase):
    def setUp(self):
        self.overlay=dict(path='overlay',sha256='a'*64);self.planref=dict(path='base',sha256='b'*64)
        self.jobref=dict(path='job',sha256='c'*64);self.identityref=dict(path='identity',sha256='d'*64);self.snapshotref=dict(path='snapshot',sha256='e'*64)
        self.task=dict(id='fit',resource='cpu',max_rss_bytes=512*2**30,max_output_bytes=2**20,output_directory='experiments/results/synthetic_recovery/fit')
        self.binding=dict(source_controller_overlay_ref=self.overlay,source_plan_ref=self.planref,source_task_id='fit',
            source_task_fingerprint=queue.c.fingerprint(self.task),resource='cpu',max_rss_bytes=512*2**30,max_output_bytes=2**20,
            output_directory=self.task['output_directory'],original_job_ref=self.jobref,original_supervision_identity_ref=self.identityref,
            owned_process_group_id=20,owned_group_leader_session=20,owned_group_leader_start_ticks=30,original_supervisor_pid=10,
            source_state_path='.runtime/synthetic_overlay/state.json',source_state_snapshot_ref=self.snapshotref)
        self.state=dict(plan_ref=self.planref,tasks=dict(fit=dict(status='running',job_ref=self.jobref,pid=10)))
        self.values={'overlay':dict(schema='parallel_controller_resource_overlay_v1',base_plan_ref=self.planref,records_directory='.runtime/synthetic_overlay'),
            'base':dict(tasks=[self.task]),'job':dict(plan_ref=self.planref,task=self.task,task_fingerprint=queue.c.fingerprint(self.task)),
            'identity':dict(job_ref=self.jobref,child_identity=dict(pid=20,pgid=20,session=20,start_ticks=30,ppid=10)),
            'snapshot':copy.deepcopy(self.state)}

    def read_json(self,path):
        return dict(approved_parallel_resource_overlay_ref=self.overlay) if path==queue.c.POLICY_PATH else self.state

    def test_authenticated_overlay_job_state(self):
        with patch.object(queue.c,'read_ref',side_effect=lambda ref:self.values[ref['path']]),patch.object(queue.c,'read_json',side_effect=self.read_json):
            self.assertEqual(predecessor.source(self.binding),(self.task,self.state['tasks']['fit']))
            self.state['tasks']['fit']['status']='succeeded'
            self.assertEqual(predecessor.source(self.binding)[1]['status'],'succeeded')

    def test_wrong_overlay_path_job_and_snapshot_rejected(self):
        changes=[('source_controller_overlay_ref',dict(path='foreign',sha256='f'*64)),('source_state_path','.runtime/foreign/state.json'),('original_supervisor_pid',11)]
        with patch.object(queue.c,'read_ref',side_effect=lambda ref:self.values[ref['path']]),patch.object(queue.c,'read_json',side_effect=self.read_json):
            for key,value in changes:
                with self.subTest(key=key),self.assertRaises(ValueError):predecessor.source(dict(self.binding,**{key:value}))
            self.values['job']['task_fingerprint']='foreign'
            with self.assertRaisesRegex(ValueError,'original predecessor job'):predecessor.source(self.binding)
            self.values['job']['task_fingerprint']=queue.c.fingerprint(self.task)
            self.values['snapshot']['tasks']['fit']['status']='pending'
            with self.assertRaisesRegex(ValueError,'frozen healthy'):predecessor.source(self.binding)

    def test_legacy_binding_uses_unchanged_original_checks(self):
        with patch.object(predecessor,'_ORIGINAL_SOURCE',return_value='legacy') as original:
            self.assertEqual(predecessor.source(dict(source_task_id='legacy')),'legacy')
            original.assert_called_once()
