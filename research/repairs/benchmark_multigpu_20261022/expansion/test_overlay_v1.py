"""Metadata-only tests for controller view and separate approval/drain hooks."""
import copy
from contextlib import ExitStack
import inspect
import unittest
from unittest.mock import patch
from . import controller_overlay_v1 as overlay
from . import queue_reporting_overlay_v1 as reporting


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.ref=dict(path='base',sha256='a'*64);self.snapref=dict(path='snapshot',sha256='b'*64)
        self.configref=dict(path='overlay',sha256='c'*64)
        self.live=dict(id='AM_treeloom_fit',resource='gpu0',max_rss_bytes=512*overlay.GIB,max_output_bytes=80*2**20,output_directory='live')
        self.done=dict(id='done',resource='cpu',output_directory='done')
        self.historical=dict(source_task_id='AM_treeloom_fit',output_directory='historical',max_output_bytes=overlay.GIB)
        self.base=dict(tasks=[self.live,self.done],workflow_external_predecessors=[self.historical],external_storage=[],
            records_directory='.runtime/base',max_parallel_rss_bytes=1280*overlay.GIB,host_ram_margin_bytes=64*overlay.GIB,physical_gpus=[2,3,5])
        self.snapshot=dict(plan_ref=self.ref,tasks=dict(AM_treeloom_fit=dict(status='running',job_ref=dict(path='job',sha256='d'*64),pid=42),
            done=dict(status='succeeded',result_ref=dict(path='result',sha256='e'*64))))
        binding=dict(source_task_id='AM_treeloom_fit',source_plan_ref=self.ref,source_task_fingerprint=overlay.c.fingerprint(self.live),
            original_job_ref=self.snapshot['tasks']['AM_treeloom_fit']['job_ref'],original_supervisor_pid=42)
        self.config=dict(schema=overlay.SCHEMA,base_plan_ref=self.ref,state_snapshot_ref=self.snapref,active_predecessors=[binding],
            historical_retention_carries=[dict(original_binding=self.historical,original_current_charge_bytes=1234,cap_bytes=overlay.GIB)],
            records_directory='.runtime/overlay',max_parallel_rss_bytes=1536*overlay.GIB)

    def patches(self):
        stack=ExitStack()
        stack.enter_context(patch.object(overlay.predecessor,'_source',return_value=({},dict(retained_bytes=1234))))
        stack.enter_context(patch.object(overlay.predecessor,'_drained',return_value=({},{})))
        stack.enter_context(patch.object(overlay.queue,'authenticated_adopted_completed',return_value={}))
        return stack

    def test_only_explicit_controller_view_changes(self):
        original=copy.deepcopy(self.base)
        with self.patches():view=overlay.plan_view(self.config,self.base,self.snapshot)
        self.assertEqual(self.base,original)
        self.assertEqual(view['tasks'][1],self.done)
        self.assertEqual(view['tasks'][0]['resource'],'external')
        self.assertEqual(view['external_storage'][0]['cap_bytes'],overlay.GIB)
        self.assertEqual(len(view['workflow_external_predecessors']),1)
        self.assertEqual(view['max_parallel_rss_bytes'],1536*overlay.GIB)
        allowed={'tasks','workflow_external_predecessors','external_storage','records_directory','max_parallel_rss_bytes',
                 'adopted_state_snapshot_ref','adopted_completed_task_fingerprints'}
        self.assertTrue({key for key in view if view.get(key)!=self.base.get(key)}<=allowed)

    def test_historical_charge_cannot_decrease(self):
        self.config['historical_retention_carries'][0]['cap_bytes']=1234
        with self.patches(),self.assertRaisesRegex(ValueError,'cannot decrease'):
            overlay.plan_view(self.config,self.base,self.snapshot)

    def test_policy_hook_keeps_original_worker_identity_and_requires_overlay_approval(self):
        policy=dict(approved_workflow_ref=self.ref,workflow_drain_ref=self.ref,approved_parallel_resource_overlay_ref=self.configref)
        queue=overlay.queue;module,source=queue.MODULE,queue.SOURCE
        with self.patches(),patch.object(overlay,'check_config',return_value=(self.config,self.base,self.snapshot)), \
             patch.object(queue,'current_policy',return_value=policy),patch.object(queue,'load_plan',return_value=self.base), \
             patch.object(queue,'ProgressReporter',queue.ProgressReporter):
            overlay.install_controller_view(self.configref)
            self.assertEqual((queue.MODULE,queue.SOURCE),(module,source))
            self.assertNotIn('workflow_drain_ref',queue.current_policy())
            self.assertEqual(policy['workflow_drain_ref'],self.ref)
            self.assertEqual(queue.load_plan(self.ref,approved=True)['max_parallel_rss_bytes'],1536*overlay.GIB)
            policy['parallel_resource_overlay_drain_ref']=self.configref
            self.assertEqual(queue.current_policy()['workflow_drain_ref'],self.ref)
            policy['approved_parallel_resource_overlay_ref']=None
            with self.assertRaisesRegex(ValueError,'Separate exact'):queue.current_policy()

    def test_distinct_reporting_and_original_materialization_contract(self):
        self.assertEqual(reporting.START,'<!-- readable-parallel-resource-overlay:start -->')
        self.assertIn('research/benchmark_parallel_resource_overlay_current.json',inspect.getsource(reporting))
        source=inspect.getsource(overlay.queue.materialize)
        self.assertIn('plan_ref=plan_ref',source)
        self.assertIn('plan["records_directory"]',source)
        job=inspect.getsource(overlay.queue.job_spec)
        self.assertNotIn('records_directory',job)
