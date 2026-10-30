"""Synthetic boundary tests. Every subprocess launch and signal is mocked."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch, Mock
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.queue_audit_20261030.backfill import boundary_v1 as b, support_v1 as s, gpu_v1 as g, watch_v1 as w
from research.joint_evaluation_20261029.queue_audit_20261030.common import probe_scheduler_v1 as engine

HERE=Path(__file__).resolve().parent

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE,prefix='synthetic_')
        self.root=Path(self.temp.name)
        self.patch=patch.object(r,'ROOT',self.root);self.patch.start()
        self.signals=patch.object(w.processes,'_pidfd_send');self.signal_mock=self.signals.start()
        self.spawn=patch.object(w.subprocess,'Popen');self.spawn_mock=self.spawn.start()
        self.sessions=patch.object(b.processes,'session_members',return_value=[]);self.sessions.start()
        self.first='paper_american_argformer_american_test_007_mlm'
        task=dict(id=self.first,kind='gpu',dependencies=[],python='python',output_directory='experiments/old',max_output_bytes=123)
        prepare=dict(id=b.PREPARE,kind='cpu',dependencies=[b.OLD_EDGE],python='python',output_directory='experiments/new',max_output_bytes=321)
        self.plan=dict(tasks=[task,prepare],records_directory='records',controller_source_refs=[],existing_workers=[])
        self.plan_ref=s.immutable_json('old_plan.json',self.plan)
        command=[str(self.root/'python'),'-B','-s','-m','arggnn.benchmarking.paper_campaign.runtime','--task',str(self.root/'records/jobs'/ (self.first+'.json'))]
        self.running=dict(status='running',pid=77,process_start_ticks='88',command=command,physical_gpu=4,reservation_bytes=123,started_unix=1,log_path='records/logs/'+self.first+'.log')
        result=s.immutable_json('experiments/old/result.json',{'value':'genuine'})
        self.wait=dict(self.running,status='succeeded',actual_wait=True,actual_returncode=0,owned_process_group_absent=True,result_ref=result)
        s.immutable_json('experiments/old/actual_wait.json',self.wait)
        self.source=dict(plan_ref=self.plan_ref,controller_pid=42,tasks={self.first:self.running,b.PREPARE:dict(status='pending')})
        self.baseline=s.immutable_json('baseline.json',self.source)
        self.spec=dict(original_plan_ref=self.plan_ref,baseline_state_ref=self.baseline,predecessor_controller=dict(pid=42,process_start_ticks='43'),original_state_path='state.json')
        s.immutable_json('state.json',self.source)
    def tearDown(self):
        self.sessions.stop();self.spawn.stop();self.signals.stop();self.patch.stop();self.temp.cleanup()
    def test_reconcile_only_real_wait(self):
        result=b.reconcile(self.spec,self.source)
        self.assertEqual(result['rows'][self.first],self.wait)
        self.assertEqual(result['reconciled_stale_running_ids'],[self.first])
        self.assertEqual(json.loads((self.root/'state.json').read_text()),self.source)
        self.signal_mock.assert_not_called();self.spawn_mock.assert_not_called()
    def test_missing_wait_refuses(self):
        (self.root/'experiments/old/actual_wait.json').unlink()
        with self.assertRaises(FileNotFoundError):b.reconcile(self.spec,self.source)
    def test_mismatched_wait_identity_refuses(self):
        r.save('experiments/old/actual_wait.json',dict(self.wait,pid=78))
        with self.assertRaisesRegex(ValueError,'does not belong'):b.reconcile(self.spec,self.source)
    def test_session_alive_refuses(self):
        with patch.object(b.processes,'session_members',return_value=[77]):
            with self.assertRaisesRegex(ValueError,'still live'):b.reconcile(self.spec,self.source)
    def test_real_failed_wait_preserved(self):
        failed=dict(self.wait,status='failed',actual_returncode=9,error='original failure')
        failed.pop('result_ref');r.save('experiments/old/actual_wait.json',failed)
        self.assertEqual(b.reconcile(self.spec,self.source)['rows'][self.first],failed)
    def test_no_fake_success_from_result(self):
        r.save('experiments/old/actual_wait.json',dict(self.wait,actual_wait=False))
        with self.assertRaises(ValueError):b.reconcile(self.spec,self.source)
    def test_completed_prefix_exact(self):
        source=copy.deepcopy(self.source);source['tasks'][self.first]=self.wait
        baseline=s.immutable_json('terminal_baseline.json',source);self.spec['baseline_state_ref']=baseline
        self.assertEqual(b.reconcile(self.spec,source)['rows'][self.first],self.wait)
        source['tasks'][self.first]['finished_unix']=99
        with self.assertRaisesRegex(ValueError,'history changed'):b.reconcile(self.spec,source)
    def test_prepare_started_closes_opportunity(self):
        source=copy.deepcopy(self.source);source['tasks'][b.PREPARE]=dict(status='running',pid=99)
        with self.assertRaisesRegex(ValueError,'opportunity closed'):b.reconcile(self.spec,source)
    def test_pending_partial_output_refuses(self):
        (self.root/'experiments/new').mkdir();(self.root/'experiments/new/unowned').write_text('partial')
        self.plan['claim_task_ids']=[]
        with self.assertRaises(ValueError):b.pending_output(self.plan,self.plan['tasks'][1],dict(status='pending'))
    def test_pending_launch_record_refuses(self):
        path=self.root/'records/jobs'/ (b.PREPARE+'.json');path.parent.mkdir(parents=True);path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'prepared launch'):b.pending_output(self.plan,self.plan['tasks'][1],dict(status='pending'))
    def test_only_exact_dependency_change(self):
        spec=dict(self.spec,schema='argformer_idle_boundary_continuation_v1',source_refs=[])
        ref=s.immutable_json('spec.json',spec);candidate=b.expected_plan(ref)
        self.assertEqual(candidate['tasks'][1]['dependencies'],[])
        self.assertEqual(candidate['tasks'][0],self.plan['tasks'][0])
        b.validate_plan(candidate)
        for key,value in [('max_output_bytes',1),('python','other'),('output_directory','experiments/other')]:
            changed=copy.deepcopy(candidate);changed['tasks'][1][key]=value
            with self.assertRaises(ValueError):b.validate_plan(changed)
    def test_public_plan_binding_preserved(self):
        ref=s.immutable_json('spec.json',self.spec)
        with patch.object(r,'ref',return_value={'path':'candidate','sha256':'new'}):
            with patch.object(r,'read',return_value=self.spec):
                meta=s.state_metadata({'operational_continuation_ref':ref},'candidate')
        self.assertEqual(meta['plan_ref'],self.plan_ref);self.assertEqual(meta['operational_plan_ref']['sha256'],'new')
    def test_no_signals_for_active_children(self):
        with patch.object(b,'controller',side_effect=ValueError('Active predecessor descendants')):
            with self.assertRaises(ValueError):w.attempt('candidate',self.spec,Mock())
        self.signal_mock.assert_not_called();self.spawn_mock.assert_not_called()
    def test_freeze_race_resumes_only_owner(self):
        with patch.object(b,'controller',side_effect=[{}, {}, ValueError('Active predecessor descendants')]),patch.object(w,'configuration',return_value=({},self.spec)),patch.object(w.processes,'_pidfd_open',return_value=123),patch.object(w.os,'close'):
            with self.assertRaises(ValueError):w.attempt('candidate',self.spec,Mock())
        self.assertEqual([call.args[1] for call in self.signal_mock.call_args_list],[signal.SIGSTOP,signal.SIGCONT]);self.spawn_mock.assert_not_called()
    def test_missing_frozen_wait_resumes(self):
        with patch.object(b,'controller',return_value={'thread_states':{'1':'T'}}),patch.object(b,'inspect_boundary',side_effect=FileNotFoundError('actual_wait absent')),patch.object(w,'configuration',return_value=({},self.spec)),patch.object(w.processes,'_pidfd_open',return_value=123),patch.object(w.os,'close'):
            with self.assertRaises(FileNotFoundError):w.attempt('candidate',self.spec,Mock())
        self.assertEqual([call.args[1] for call in self.signal_mock.call_args_list],[signal.SIGSTOP,signal.SIGCONT]);self.spawn_mock.assert_not_called()
    def test_duplicate_lifetime_lock_refuses_before_bootstrap(self):
        plan=dict(self.plan,records_directory='new_records')
        ref=s.immutable_json('engine_plan.json',plan)
        lock=(self.root/'owner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        support=Mock()
        try:
            with patch.object(engine,'validate',side_effect=lambda value:value):
                with self.assertRaises(BlockingIOError):engine.run(ref['path'],support_module=support,owner_lock_path='owner.lock',gpu_module=g,allow_existing_state=True)
        finally:lock.close()
        support.initial_rows.assert_not_called();self.spawn_mock.assert_not_called()
    def test_gpu3_mutex_exclusion(self):
        first=g.acquire(3)
        try:self.assertIs(g.acquire(3),False)
        finally:g.release(first)
        third=g.acquire(3);self.assertIsInstance(third,int);g.release(third)
    def test_gpu3_only_test009(self):
        self.assertEqual(g.allowed_for_task({},dict(id=self.first),[3,4,5]),[4,5])
        self.assertEqual(g.allowed_for_task({},dict(id=b.PREFIX+'mlm'),[3,4,5]),[3,4,5])
    def test_root_priority_blocks_gpu3(self):
        with patch.object(g.support,'accounting_template',return_value={}),patch.object(g.shared,'root_release',return_value=None):self.assertEqual(g.allowed({}),[4,5])
    def test_unfinished_treeloom_blocks_gpu3(self):
        with patch.object(g.support,'accounting_template',return_value={}),patch.object(g.shared,'root_release',return_value={'valid':True}),patch.object(r,'read',return_value={}),patch.object(g,'treeloom_gpu_complete',return_value=False):self.assertEqual(g.allowed({'operational_continuation_ref':{}}),[4,5])
    def test_probe_unavailable_does_not_relax_priority(self):
        with patch.object(g.support,'accounting_template',return_value={}),patch.object(g.shared,'root_release',side_effect=PermissionError('unavailable')):
            with self.assertRaises(PermissionError):g.allowed({})
    def test_opportunity_closed_without_signals(self):
        source=copy.deepcopy(self.source);source['tasks'][b.PREPARE]=dict(status='running',pid=99)
        r.save('state.json',source)
        self.assertEqual(w.progress_outcome(self.spec)['status'],'opportunity_closed')
        self.signal_mock.assert_not_called();self.spawn_mock.assert_not_called()
    def test_absent_owner_no_automatic_handoff(self):
        self.spec['predecessor_controller']['pid']=999999999
        source=copy.deepcopy(self.source);source['controller_pid']=999999999;r.save('state.json',source)
        self.assertEqual(w.progress_outcome(self.spec)['status'],'owner_absent_requires_inspection')
        self.signal_mock.assert_not_called();self.spawn_mock.assert_not_called()
    def test_all_metadata_writers_reserved(self):
        with patch.object(r,'read',return_value={'metadata_max_bytes':16*2**20,'storage_credit_ref':{}}),patch.object(s,'accounting_template',return_value={}),patch.object(s.credit,'commitment',return_value={'items':{'base':100},'ceiling_bytes':1200000000000}):
            result=s.admission_accounting({'operational_continuation_ref':{}})
        self.assertEqual(result['total_bytes'],100+64*2**20)
        self.assertTrue(result['admitted'])
    def test_accounting_io_failure_is_not_zero(self):
        previous=r.scope_bytes
        def failed_commitment(*args,**kwargs):
            return {'items':{'base':r.scope_bytes(self.root)},'ceiling_bytes':1200000000000}
        with patch.object(r,'read',return_value={'metadata_max_bytes':16*2**20,'storage_credit_ref':{}}),patch.object(s,'accounting_template',return_value={}),patch.object(engine,'strict_scope_bytes',side_effect=PermissionError('denied')),patch.object(s.credit,'commitment',side_effect=failed_commitment):
            with self.assertRaises(PermissionError):s.admission_accounting({'operational_continuation_ref':{}})
        self.assertIs(r.scope_bytes,previous)
    def test_successful_frozen_handoff_has_only_mocked_owner_signals(self):
        frozen={'thread_states':{'1':'T'}}
        merged=b.reconcile(self.spec,self.source)
        observed=dict(controller=frozen,source=self.source,**merged)
        self.spec.update(metadata_directory='metadata',handoff_receipt_path='metadata/handoff.json',replacement_log_path='metadata/replacement.log',replacement_started_path='metadata/started.json',launch_argv=['python','-m','reviewed'])
        ref=s.immutable_json('candidate.json',{'operational_continuation_ref':{'path':'spec','sha256':'fake'}})
        self.spawn_mock.return_value.pid=os.getpid()
        with patch.object(b,'controller',return_value=frozen),patch.object(b,'inspect_boundary',return_value=observed),patch.object(w,'configuration',return_value=({},self.spec)),patch.object(w.processes,'_pidfd_open',return_value=123),patch.object(w.os,'close'),patch.object(w.select,'select',return_value=([123],[],[])),patch.object(s,'admission_accounting',return_value={'admitted':True}):
            result=w.attempt(ref['path'],self.spec,Mock())
        self.assertEqual([call.args[1] for call in self.signal_mock.call_args_list],[signal.SIGSTOP,signal.SIGKILL])
        self.assertEqual(result['worker_signals_sent'],[]);self.spawn_mock.assert_called_once()
        handoff=json.loads((self.root/'metadata/handoff.json').read_text());self.assertFalse(handoff['old_actual_wait_consumed'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(schema='argformer_boundary_synthetic_tests_v1',passed=result.wasSuccessful(),tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),production_launches=0,real_signals=0,
                test_source_ref=r.ref(__file__))
    target=HERE/'test_results_v3.json'
    with target.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(report,sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
