"""Isolated operational checks; subprocess and Git dispatch are always mocked."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import final_review_continuation_v1 as continuation


class FinalReviewContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=r.ROOT/'research/joint_evaluation_20261029/review_finish')
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(r, 'ROOT', self.root); self.root_patch.start()
        self.records_patch = patch.object(continuation, 'RECORDS', 'records'); self.records_patch.start()
        self.dispatch_guards = [patch.object(continuation.subprocess, name, side_effect=AssertionError('Unmocked dispatch: ' + name)) for name in ('Popen', 'run', 'check_output')]
        for guard in self.dispatch_guards:guard.start()
        self.folder = self.root/'records'; self.folder.mkdir()
        self.finish_ref = self.write('finish.json', {'schema': 'fixture_finish'})
        self.source = self.root/'audit.py'; self.source.write_text('# Fixture source; never executed.\n')
        self.source_ref = r.ref(self.source)
        self.contract_ref = self.write('contract.json', {'scope': 'registered comparisons'})
        self.plan = dict(schema='final_review_continuation_plan_v1', trigger_path='trigger.json',
            finish_plan_ref=self.finish_ref, comparison_paths=['comparison_'+str(i)+'.json' for i in range(5)],
            audit_argv=['/fixture/python', '-B', '-m', 'fixture.audit', '--benchmark', 'all'],
            audit_output='audit.json', audit_source_ref=self.source_ref,
            publisher_argv=['/fixture/python', '-B', '-m', 'fixture.publisher'], expected_native_argmix_condition='equation_corrected',
            source_refs=[self.source_ref], european_review_refs=[], scientific_contract_ref=self.contract_ref)
        self.plan_ref = self.write('plan.json', self.plan)
        (self.root/'README.md').write_text('Unrelated introduction.\n<!-- joint-finish-operations:start -->\nPending.\n<!-- joint-finish-operations:end -->\nUnrelated ending.\n')

    def tearDown(self):
        for guard in reversed(self.dispatch_guards):guard.stop()
        self.records_patch.stop(); self.root_patch.stop(); self.temporary.cleanup()

    def write(self, name, value):
        path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
        return r.ref(path)

    def trigger(self, **changes):
        value=dict(plan_ref=self.finish_ref, all_computations_complete=True,
                   publication={'pushed': True}, completed_comparison_artifacts=self.plan['comparison_paths'])
        value.update(changes)
        return self.write('trigger.json', value)

    def audit_result(self, **changes):
        result=dict(schema='independent_final_benchmark_audit_v1', passed=True,
                    reports={'american': {}, 'argmix': {'native_condition_disclosure':dict(condition='equation_corrected', registered_condition='equation_corrected', published_numerical_table_reproduction=False)}}, source_ref=self.source_ref)
        result.update(changes)
        return self.write('audit.json', result)

    def wait(self, trigger, **changes):
        value=dict(actual_wait=True, actual_returncode=0, owned_process_group_absent=True,
                   command=self.plan['audit_argv'], trigger_ref=trigger, result_ref=self.audit_result(), pid=99999999)
        value.update(changes)
        return self.write('records/actual_wait.json', value)

    def run_main(self):
        with patch.object(sys, 'argv', ['fixture', '--plan', 'plan.json']), patch.dict(sys.modules, {'setproctitle':SimpleNamespace(setproctitle=lambda title:None)}):
            continuation.main()

    def simulate_audit(self, code=0, produce=True):
        child=SimpleNamespace(pid=99999999, wait=lambda:code)
        def launch(argv, **kwargs):
            self.assertEqual(argv, self.plan['audit_argv'])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertTrue(kwargs['start_new_session'])
            if produce:self.audit_result()
            return child
        original_read_text=Path.read_text
        def fixture_read_text(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return 'MemAvailable: 999999999 kB\n'
            if str(path)=='/proc/99999999/stat':return '99999999 (fixture) S '+'0 '*18+'123'
            return original_read_text(path,*args,**kwargs)
        with patch.object(continuation.subprocess, 'Popen', side_effect=launch) as dispatched, patch.object(Path, 'read_text', new=fixture_read_text), patch.object(continuation.processes, 'session_members', return_value=[]):
            result=continuation.audit_once(self.plan,self.folder,self.trigger(),self.plan_ref)
        self.assertEqual(dispatched.call_count,1)
        return result

    def test_pending_gate_does_not_dispatch(self):
        self.assertIsNone(continuation.readiness(self.plan))
        with patch.object(sys,'argv',['fixture','--plan','plan.json','--inspect']):continuation.main()
        self.assertFalse((self.folder/'started.json').exists())

    def test_wrong_finish_plan_fails(self):
        self.trigger(plan_ref={'path':'other.json','sha256':'different'})
        with self.assertRaisesRegex(ValueError,'actually finish'):continuation.readiness(self.plan)

    def test_incomplete_gate_fails(self):
        self.trigger(all_computations_complete=False)
        with self.assertRaisesRegex(ValueError,'actually finish'):continuation.readiness(self.plan)

    def test_unpushed_gate_fails(self):
        self.trigger(publication={'pushed':False})
        with self.assertRaisesRegex(ValueError,'GitHub'):continuation.readiness(self.plan)

    def test_partial_comparison_gate_fails(self):
        self.trigger(completed_comparison_artifacts=self.plan['comparison_paths'][:-1])
        with self.assertRaisesRegex(ValueError,'Exact complete comparison'):continuation.readiness(self.plan)

    def test_changed_audit_source_fails(self):
        self.source.write_text('# Changed fixture.\n')
        with self.assertRaisesRegex(ValueError,'source changed'):continuation.sources(self.plan)

    def test_interrupted_audit_never_replays(self):
        self.write('records/started.json', {'pid':99999999})
        with self.assertRaisesRegex(ValueError,'must not be replayed'):continuation.audit_once(self.plan,self.folder,self.trigger())

    def test_unaffiliated_audit_output_never_overwrites(self):
        self.audit_result()
        with self.assertRaisesRegex(ValueError,'must not be overwritten'):continuation.audit_once(self.plan,self.folder,self.trigger())

    def test_previous_failed_wait_never_replays(self):
        trigger=self.trigger();self.wait(trigger,actual_returncode=1)
        with self.assertRaisesRegex(ValueError,'Previous audit failed'):continuation.audit_once(self.plan,self.folder,trigger)

    def test_previous_wait_wrong_trigger_fails(self):
        trigger=self.trigger();self.wait({'path':'other','sha256':'other'})
        with self.assertRaisesRegex(ValueError,'previous audit binding'):continuation.audit_once(self.plan,self.folder,trigger)

    def test_successful_wait_reuses_exact_result(self):
        trigger=self.trigger();reference=self.wait(trigger);expected=r.read(reference)['result_ref']
        self.assertEqual(continuation.audit_once(self.plan,self.folder,trigger),expected)

    def test_successful_simulation_records_running_and_wait(self):
        result=self.simulate_audit()
        self.assertTrue(r.read(result)['passed'])
        status=r.read(r.ref(self.folder/'status.json'))
        self.assertEqual(status['status'],'running_independent_audit')
        self.assertEqual(status['plan_ref'],self.plan_ref)
        self.assertEqual(status['audit_pid'],99999999)
        wait=r.read(r.ref(self.folder/'actual_wait.json'))
        self.assertTrue(wait['actual_wait']);self.assertEqual(wait['actual_returncode'],0)
        self.assertEqual(wait['process_start_ticks'],'123')

    def test_failed_simulation_preserves_actual_wait(self):
        with self.assertRaisesRegex(ValueError,'did not pass'):self.simulate_audit(code=1,produce=False)
        wait=r.read(r.ref(self.folder/'actual_wait.json'))
        self.assertEqual(wait['actual_returncode'],1)
        self.assertNotIn('result_ref',wait)

    def test_zero_exit_without_report_preserves_partial_wait(self):
        with self.assertRaisesRegex(ValueError,'did not pass'):self.simulate_audit(code=0,produce=False)
        wait=r.read(r.ref(self.folder/'actual_wait.json'))
        self.assertEqual(wait['actual_returncode'],0)
        self.assertNotIn('result_ref',wait)

    def test_main_records_operational_audit_failure(self):
        self.trigger()
        with patch.object(continuation,'audit_once',side_effect=ValueError('fixture audit failure')):
            with self.assertRaisesRegex(ValueError,'fixture audit failure'):self.run_main()
        status=r.read(r.ref(self.folder/'status.json'))
        self.assertEqual(status['status'],'audit_failed')
        self.assertEqual(status['error'],'fixture audit failure')
        self.assertFalse((self.folder/'benchmark_completion.json').exists())

    def test_main_rejects_failed_or_partial_or_wrong_source_report(self):
        for changes in ({'passed':False},{'reports':{'american':{}}},{'source_ref':{'path':'other','sha256':'other'}}):
            with self.subTest(changes=changes):
                self.trigger();reference=self.audit_result(**changes)
                with patch.object(continuation,'audit_once',return_value=reference):
                    with self.assertRaisesRegex(ValueError,'independent numerical audit'):self.run_main()
                self.assertEqual(r.read(r.ref(self.folder/'status.json'))['status'],'audit_failed')
                self.assertFalse((self.folder/'benchmark_completion.json').exists())

    def test_main_publication_pending_then_complete(self):
        self.trigger();reference=self.audit_result()
        for path in self.plan['comparison_paths']:self.write(path,{'fixture_comparison':True})
        observed=[]
        def publish(plan,folder):
            observed.append(r.read(r.ref(folder/'status.json'))['status'])
            return True
        with patch.object(continuation,'audit_once',return_value=reference),patch.object(continuation,'publish',side_effect=publish):self.run_main()
        self.assertEqual(observed,['audit_passed_publication_pending'])
        self.assertEqual(r.read(r.ref(self.folder/'status.json'))['status'],'registered_benchmark_complete_and_pushed')
        completion=r.read(r.ref(self.folder/'benchmark_completion.json'))
        self.assertFalse(completion['paper_complete']);self.assertFalse(completion['exact_published_table_replication_claimed'])
        self.assertEqual(len(completion['comparison_refs']),5)

    def publication_fixture(self):
        self.write(continuation.BASE+'/root/CURRENT_STATUS.json', {'fixture':True})
        self.audit_result();self.write('records/benchmark_completion.json',{'passed':True});self.write('records/actual_wait.json',{'actual_wait':True})
        return continuation.publication_paths(self.plan,self.folder)

    def test_foreign_staging_blocks_before_readme_edit(self):
        self.publication_fixture();before=(self.root/'README.md').read_bytes();calls=[]
        def run(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=1)
        with patch.object(continuation.subprocess,'run',side_effect=run):self.assertFalse(continuation.publish(self.plan,self.folder))
        self.assertEqual((self.root/'README.md').read_bytes(),before)
        self.assertEqual(calls,[['git','diff','--cached','--quiet']])

    def test_publication_only_exact_small_registered_paths(self):
        paths=self.publication_fixture();calls=[];quiet=iter((0,1))
        def run(argv,**kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=next(quiet) if argv==['git','diff','--cached','--quiet'] else 0)
        def output(argv,**kwargs):
            if argv==['git','diff','--cached','--name-only','-z']:return ('\0'.join(paths)+'\0').encode()
            self.assertEqual(argv,['git','rev-parse','HEAD']);return 'fixture-commit\n'
        with patch.object(continuation.subprocess,'run',side_effect=run),patch.object(continuation.subprocess,'check_output',side_effect=output):self.assertTrue(continuation.publish(self.plan,self.folder))
        self.assertIn(self.plan['publisher_argv'], calls)
        current=r.read(r.ref(continuation.BASE+'/root/CURRENT_STATUS.json'))
        self.assertTrue(current['independent_final_review_complete'])
        self.assertEqual(current['native_ARGMix_final_condition_disclosure']['condition'], 'equation_corrected')
        self.assertIn(['git','add','--',*paths],calls)
        self.assertIn(['git','commit','--only','-m','Publish independent final benchmark audit','--',*paths],calls)
        self.assertIn(['git','push','origin','HEAD'],calls)
        self.assertEqual(len(paths),6);self.assertEqual(paths[0],'README.md')
        text=(self.root/'README.md').read_text();self.assertTrue(text.startswith('Unrelated introduction.\n'));self.assertTrue(text.endswith('\nUnrelated ending.\n'))
        self.assertTrue(r.read(r.ref(self.folder/'publication.json'))['pushed'])

    def test_foreign_stage_race_blocks_commit_and_push(self):
        paths=self.publication_fixture();calls=[]
        def run(argv,**kwargs):calls.append(argv);return SimpleNamespace(returncode=0)
        with patch.object(continuation.subprocess,'run',side_effect=run),patch.object(continuation.subprocess,'check_output',return_value=('\0'.join(paths+['foreign.txt'])+'\0').encode()):self.assertFalse(continuation.publish(self.plan,self.folder))
        self.assertFalse(any('commit' in command or 'push' in command for command in calls))
        self.assertFalse((self.folder/'publication.json').exists())

    def test_main_rejects_wrong_registered_condition(self):
        self.trigger()
        reference=self.audit_result(reports={'american':{},'argmix':{'native_condition_disclosure':dict(condition='author_key_i',registered_condition='author_key_i',published_numerical_table_reproduction=False)}})
        with patch.object(continuation,'audit_once',return_value=reference):
            with self.assertRaisesRegex(ValueError,'Exact registered ARGMix condition'):self.run_main()
        self.assertFalse((self.folder/'benchmark_completion.json').exists())

    def test_report_scope_cannot_escape_project(self):
        self.publication_fixture();plan=dict(self.plan,audit_output='../outside.json')
        with self.assertRaisesRegex(ValueError,'Project-contained'):continuation.publication_paths(plan,self.folder)

    def test_large_report_is_not_published(self):
        self.publication_fixture()
        with (self.root/'audit.json').open('wb') as stream:stream.truncate(8*2**20+1)
        with self.assertRaisesRegex(ValueError,'small registered reports'):continuation.publication_paths(self.plan,self.folder)


if __name__=='__main__':unittest.main()
