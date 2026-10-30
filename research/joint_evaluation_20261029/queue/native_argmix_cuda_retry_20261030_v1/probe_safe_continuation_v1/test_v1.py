"""Mocked fault injection only; never launch, signal or inspect a production worker."""
import copy
import fcntl
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from research.joint_evaluation_20261029.queue_audit_20261030.resources import probe_guard_v1 as guard
from research.joint_evaluation_20261029.queue.native_argmix_cuda_retry_20261030_v1.probe_safe_continuation_v1 import scheduler_v1 as s, support_v1 as support, predict_cli_v1 as wrapper

HERE = Path(__file__).parent


class ProbeTests(unittest.TestCase):
    def test_outage_preserves_last_good_and_never_returns_zero(self):
        monitor = guard.ProbeMonitor(None, clock=lambda: 10)
        self.assertEqual(monitor.observe('pss', lambda: 123), 123)
        failure = mock.Mock(side_effect=guard.ProbeUnavailable('temporary'))
        self.assertIsNone(monitor.observe('pss', failure))
        self.assertTrue(monitor.degraded)
        self.assertEqual(monitor.health()['last_good']['pss']['observed_unix'], 10)
        self.assertFalse(monitor.health()['unknown_measurements_reported_as_zero'])
        self.assertEqual(monitor.observe('pss', lambda: 124), 124)
        self.assertFalse(monitor.degraded)

    def test_gpu_transients_and_identity_violation_are_distinct(self):
        for error in [subprocess.TimeoutExpired('nvidia-smi',15), subprocess.CalledProcessError(1,'nvidia-smi'), ValueError('invalid literal'), RuntimeError('Missing authorized GPU')]:
            with self.assertRaises(guard.ProbeUnavailable):
                guard.gpu_observation(mock.Mock(side_effect=error))
        with self.assertRaisesRegex(RuntimeError,'UUID'):
            guard.gpu_observation(mock.Mock(side_effect=RuntimeError('Physical GPU UUID mapping changed')))

    def test_process_observation_converts_only_observational_errors(self):
        for error in [subprocess.TimeoutExpired('ps',15), FileNotFoundError('proc vanished'), PermissionError('proc denied')]:
            with self.assertRaises(guard.ProbeUnavailable):
                guard.process_observation(mock.Mock(side_effect=error))()
        with self.assertRaisesRegex(RuntimeError,'escaped'):
            guard.process_observation(mock.Mock(side_effect=RuntimeError('Task escaped original process group within its session')))()

    def test_foreign_exit_race_is_normal(self):
        row=dict(status='running',pid=999999,process_start_ticks='123')
        with mock.patch.object(support.prior,'verify_original',return_value=({'original_plan_ref':{}},{'tasks':{'x':row}})), mock.patch.object(support.r,'read',return_value={'tasks':[{'id':'x','max_rss_bytes':99}]}), mock.patch.object(Path,'read_text',side_effect=FileNotFoundError):
            self.assertEqual(support.foreign_owner_growth({}),0)

    def test_foreign_stat_access_failure_is_observational(self):
        row=dict(status='running',pid=999999,process_start_ticks='123')
        with mock.patch.object(support.prior,'verify_original',return_value=({'original_plan_ref':{}},{'tasks':{'x':row}})), mock.patch.object(support.r,'read',return_value={'tasks':[{'id':'x','max_rss_bytes':99}]}), mock.patch.object(Path,'read_text',side_effect=PermissionError):
            with self.assertRaises(guard.ProbeUnavailable):support.foreign_owner_growth({})

    def test_foreign_live_identity_mismatch_remains_fatal(self):
        row=dict(status='running',pid=999999,process_start_ticks='123')
        text='999999 (fixture) '+' '.join(['S']+['0']*18+['124'])
        with mock.patch.object(support.prior,'verify_original',return_value=({'original_plan_ref':{}},{'tasks':{'x':row}})), mock.patch.object(support.r,'read',return_value={'tasks':[{'id':'x','max_rss_bytes':99}]}), mock.patch.object(Path,'read_text',return_value=text):
            with self.assertRaisesRegex(ValueError,'identity changed'):
                support.foreign_owner_growth({})

    def test_title_precedes_cli_invocation_and_other_stages_rejected(self):
        from arggnn.benchmarking.paper_argmix import paper_argmix_cli
        events=[]
        with mock.patch.object(wrapper.sys,'argv',['fixture','predict']), mock.patch.object(wrapper.setproctitle,'setproctitle',side_effect=lambda value:events.append(('title',value))), mock.patch.object(paper_argmix_cli,'main',side_effect=lambda:events.append(('cli',))):
            wrapper.main()
        self.assertEqual(events,[('title','AI Auto Research Pipeline Test'),('cli',)])
        with mock.patch.object(wrapper.sys,'argv',['fixture','train']):
            with self.assertRaises(ValueError):wrapper.main()


class SchedulerTests(unittest.TestCase):
    def scenario(self, *, gpu=False, gpu_error=None, pss_error=False, drain_error=False, limit=False, admission_error=False, terminate_drain_error=False, source_error=False, metadata_error=False, terminate_then_absent=False, stop_error=False):
        with tempfile.TemporaryDirectory(prefix='test_fixture_',dir=HERE) as temporary:
            root=Path(temporary);records=root/'records';output=root/'output';output.mkdir()
            (output/'completed.json').write_text('{}')
            task=dict(id='fixture',kind='gpu' if gpu else 'cpu',dependencies=[],argv=['fixture-only'],
                output_directory=str(output),completion_file='completed.json',max_output_bytes=1000,
                max_rss_bytes=100,minimum_free_gpu_bytes=1,max_seconds=1000)
            plan=dict(tasks=[task],records_directory=str(records),controller_source_refs=[],
                retry_manifest_ref={},operational_continuation_ref={},reserved_existing_project_bytes=0,
                foreign_ram_reserve_bytes=0,existing_workers=[],max_parallel_tasks=1,max_cpu_tasks=1,
                max_jobs_per_gpu=1,storage_ceiling_bytes=1200000000000)
            if source_error:
                task['source_refs']=[{'path':str(output/'completed.json'),'sha256':'changed'}]
            plan_path=root/'plan.json';plan_path.write_text(json.dumps(plan))
            child=mock.Mock(pid=os.getpid())
            child.poll.side_effect=([0]*6 if drain_error or terminate_drain_error else [None,0])
            snapshots=[];calls={'gpu':0,'pss':0,'drain':0,'sleep':0,'terminate':0,'clock':0,'metadata':0}
            def query_gpu():
                calls['gpu']+=1
                if admission_error and calls['gpu']==1:raise guard.ProbeUnavailable('inventory unavailable before admission')
                if calls['gpu']==2 and gpu_error is not None:raise gpu_error
                return {3:dict(uuid=s.r.GPU_UUIDS[3],free_bytes=1000,utilization=0)}
            def query_pss(pid):
                calls['pss']+=1
                if pss_error and calls['pss']==1:raise guard.ProbeUnavailable('ps temporarily unavailable')
                return 101 if limit else 10
            def drain(process, terminate=False):
                calls['drain']+=1
                if terminate_drain_error:
                    if not terminate:return bool(terminate_then_absent and calls['terminate'])
                    calls['terminate']+=1
                    if calls['terminate']==1:raise guard.ProbeUnavailable('inventory unavailable during requested drain')
                if drain_error and calls['drain']==1:raise guard.ProbeUnavailable('drain inventory temporarily unavailable')
                return True
            def sleep(seconds):
                calls['sleep']+=1
                self.assertLess(calls['sleep'],8,'Fixture loop must be bounded')
            def metadata(plan):
                calls['metadata']+=1
                if metadata_error and calls['metadata'] in (2,3):
                    raise guard.ProbeUnavailable('metadata process observation unavailable')
                return True
            def now():
                calls['clock']+=20
                return calls['clock']
            original_path=s.r.path
            def path(value):
                if str(value).endswith('/sidecar_owner.lock'):return root/'fixture.lock'
                return original_path(value)
            def save(filename,value):
                snapshots.append((str(filename),copy.deepcopy(value)));return {'path':str(filename),'sha256':'fixture'}
            with ExitStack() as stack:
                patches=[mock.patch.object(s,'validate',side_effect=lambda value:value),mock.patch.object(s.r,'path',side_effect=path),
                    mock.patch.object(s.r,'save',side_effect=save),mock.patch.object(s.r,'scope_bytes',return_value=0),
                    mock.patch.object(s.support,'initial_rows',return_value={'fixture':{'status':'pending'}}),
                    mock.patch.object(s.support,'admission_accounting',return_value={'admitted':True,'total_bytes':0}),
                    mock.patch.object(s.support,'metadata_within_bound',side_effect=metadata),mock.patch.object(s.support,'foreign_owner_growth',return_value=0),
                    mock.patch.object(s.support,'task_fits_scope',return_value=True),mock.patch.object(s.continuation,'permit_existing_output',return_value=True),
                    mock.patch.object(s.probe,'install_process_probes'),mock.patch.object(s.probe,'host_available_bytes',return_value=10**15),
                    mock.patch.object(s.processes,'pss',side_effect=query_pss),mock.patch.object(s.processes,'drain',side_effect=drain),
                    mock.patch.object(s,'gpu_state',side_effect=query_gpu),mock.patch.object(s.gpu_ownership,'allowed',return_value=[3]),
                    mock.patch.object(s,'pipeline_floor',return_value=0),mock.patch.object(s.time,'sleep',side_effect=sleep),
                    mock.patch.object(s.signal,'signal'),mock.patch.object(s.time,'time',side_effect=now)]
                for patch in patches:stack.enter_context(patch)
                popen=stack.enter_context(mock.patch.object(s.subprocess,'Popen',return_value=child))
                stop=stack.enter_context(mock.patch.object(s,'stop_owned',side_effect=[guard.ProbeUnavailable('drain inventory unavailable'),-15] if stop_error else None,return_value=-15))
                release=stack.enter_context(mock.patch.object(s.gpu_ownership,'release'))
                stack.enter_context(mock.patch.object(s.gpu_ownership,'acquire',return_value=777 if gpu else None))
                error=None;result=None
                try:result=s.run(str(plan_path))
                except RuntimeError as observed:error=observed
                return dict(result=result,error=error,snapshots=snapshots,calls=calls,launches=popen.call_count,
                            stops=stop.call_count,releases=release.call_args_list)

    def test_gpu_probe_failure_preserves_cpu_worker_and_ram_checks(self):
        result=self.scenario(gpu_error=subprocess.TimeoutExpired('nvidia-smi',15))
        self.assertIsNone(result['error']);self.assertEqual(result['stops'],0)
        self.assertEqual(result['result']['counts']['succeeded'],1)
        self.assertGreater(result['calls']['pss'],0)
        self.assertTrue(any(value.get('status')=='degraded' for name,value in result['snapshots'] if name.endswith('probe_health.json')))

    def test_pss_outage_is_unknown_and_does_not_kill(self):
        result=self.scenario(pss_error=True)
        self.assertEqual(result['stops'],0)
        degraded=[value for name,value in result['snapshots'] if name.endswith('state.json') and value.get('monitor_health',{}).get('status')=='degraded']
        self.assertTrue(degraded)
        self.assertNotIn('observed_pss_bytes',degraded[0]['tasks']['fixture'])

    def test_unknown_drain_retains_gpu_lease_until_verified_completion(self):
        result=self.scenario(gpu=True,drain_error=True)
        self.assertEqual(result['stops'],0)
        self.assertEqual(result['calls']['drain'],2)
        self.assertEqual(result['releases'],[mock.call(777)])
        unknown=[value for name,value in result['snapshots'] if name.endswith('state.json') and value.get('monitor_health',{}).get('status')=='degraded']
        self.assertEqual(unknown[0]['tasks']['fixture']['status'],'running')

    def test_degraded_monitoring_pauses_admission_then_recovers(self):
        result=self.scenario(admission_error=True)
        self.assertEqual(result['launches'],1)
        states=[value for name,value in result['snapshots'] if name.endswith('state.json')]
        degraded=[value for value in states if value.get('monitor_health',{}).get('status')=='degraded']
        self.assertTrue(degraded)
        self.assertTrue(all(value['tasks']['fixture']['status']=='pending' for value in degraded))
        self.assertEqual(result['stops'],0)
        self.assertEqual(result['result']['counts']['succeeded'],1)

    def test_termination_drain_probe_failure_is_retried_without_false_wait(self):
        result=self.scenario(gpu=True,terminate_drain_error=True)
        self.assertEqual(result['stops'],0)
        self.assertEqual(result['calls']['terminate'],2)
        self.assertEqual(result['releases'],[mock.call(777)])
        degraded=[value for name,value in result['snapshots'] if name.endswith('state.json') and value.get('monitor_health',{}).get('status')=='degraded']
        self.assertTrue(degraded)
        self.assertNotIn('actual_wait',degraded[0]['tasks']['fixture'])
        self.assertEqual(result['result']['counts']['failed'],1)

    def test_drain_outage_after_signals_cannot_turn_forced_cleanup_into_success(self):
        result=self.scenario(gpu=True,terminate_drain_error=True,terminate_then_absent=True)
        self.assertEqual(result['stops'],0)
        self.assertEqual(result['calls']['terminate'],1)
        self.assertEqual(result['result']['counts']['failed'],1)
        self.assertEqual(result['releases'],[mock.call(777)])

    def test_task_source_drift_still_blocks_launch(self):
        result=self.scenario(source_error=True)
        self.assertEqual(result['launches'],0)
        self.assertEqual(result['result']['counts']['blocked'],1)
        self.assertIn('Pinned source changed',result['result']['tasks']['fixture']['error'])

    def test_metadata_probe_failure_preserves_owned_worker(self):
        result=self.scenario(metadata_error=True)
        self.assertEqual(result['stops'],0)
        self.assertEqual(result['result']['counts']['succeeded'],1)
        self.assertTrue(any(value.get('status')=='degraded' for name,value in result['snapshots'] if name.endswith('probe_health.json')))

    def test_lifetime_lock_rejects_second_owner_before_adoption(self):
        with tempfile.TemporaryDirectory(prefix='test_lock_',dir=HERE) as temporary:
            root=Path(temporary);lock_path=root/'owner.lock';plan_path=root/'plan.json'
            plan_path.write_text(json.dumps({'controller_source_refs':[],'records_directory':str(root/'records')}))
            original_path=s.r.path
            def path(value):
                return lock_path if str(value).endswith('/sidecar_owner.lock') else original_path(value)
            with lock_path.open('a+') as owner, mock.patch.object(s,'validate',side_effect=lambda value:value), mock.patch.object(s.r,'path',side_effect=path), mock.patch.object(s.support,'initial_rows') as adopt, mock.patch.object(s.subprocess,'Popen') as launch:
                fcntl.flock(owner,fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):s.run(str(plan_path))
                adopt.assert_not_called();launch.assert_not_called()

    def test_verified_limit_drain_outage_keeps_stop_intent_and_gpu_lease(self):
        result=self.scenario(gpu=True,limit=True,stop_error=True)
        self.assertEqual(result['stops'],2)
        self.assertEqual(result['result']['counts']['failed'],1)
        self.assertEqual(result['releases'],[mock.call(777)])
        degraded=[value for name,value in result['snapshots'] if name.endswith('state.json') and value.get('monitor_health',{}).get('status')=='degraded']
        self.assertTrue(degraded)
        row=degraded[0]['tasks']['fixture']
        self.assertTrue(row['termination_requested'])
        self.assertEqual(row['status'],'running')
        self.assertNotIn('actual_wait',row)
        self.assertEqual(row['error'],'Resource reservation exceeded')

    def test_verified_ram_excess_still_stops_owned_fixture(self):
        result=self.scenario(limit=True)
        self.assertEqual(result['stops'],1)
        self.assertEqual(result['result']['counts']['failed'],1)

    def test_identity_violation_still_fails_and_drains_owned_fixture(self):
        result=self.scenario(gpu_error=RuntimeError('Physical GPU UUID mapping changed'))
        self.assertIn('UUID',str(result['error']))
        self.assertEqual(result['stops'],1)


if __name__=='__main__':
    unittest.main()
