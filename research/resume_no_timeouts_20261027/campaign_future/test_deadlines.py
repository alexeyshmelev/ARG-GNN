"""Narrow optional-deadline and unchanged-resource tests; never launch tasks."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
from research.resume_no_timeouts_20261027.campaign_future.code import deadlines,runtime,scheduler

OUT=runtime.ROOT/'research/resume_no_timeouts_20261027/campaign_future'

class DeadlineTests(unittest.TestCase):
    def task(self, maximum=None):
        return dict(id='synthetic',kind='cpu',dependencies=[],max_seconds=maximum,
            max_output_bytes=1024,max_rss_bytes=4096,python='.runtime/env/bin/python3.11',
            output_directory='experiments/future_deadline_test_not_created',
            call=dict(module='not_imported',function='not_executed',spec={}))

    def test_explicit_optional_validation(self):
        self.assertIsNone(deadlines.validate_max_seconds(None))
        self.assertEqual(deadlines.validate_max_seconds(1),1)
        for value in (0,-1,True,False,1.0,'none',float('inf'),float('nan')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                deadlines.validate_max_seconds(value)

    def test_elapsed_boundary_and_unlimited(self):
        self.assertFalse(deadlines.elapsed_limit_exceeded(10,10))
        self.assertTrue(deadlines.elapsed_limit_exceeded(10.001,10))
        for elapsed in (0,10,86400,10**30):
            self.assertFalse(deadlines.elapsed_limit_exceeded(elapsed,None))
        self.assertIs(scheduler.elapsed_limit_exceeded,deadlines.elapsed_limit_exceeded)
        self.assertIs(runtime.elapsed_limit_exceeded,deadlines.elapsed_limit_exceeded)

    def test_scheduler_validation(self):
        plan=dict(schema='paper_campaign_v1',physical_gpus=[3,4,5],storage_ceiling_bytes=995000000000,tasks=[self.task()])
        scheduler.validate(plan)
        plan['tasks'][0]['max_seconds']=5
        scheduler.validate(plan)
        for value in (False,0,-1,3.5,'null'):
            plan['tasks'][0]['max_seconds']=value
            with self.assertRaises(ValueError):scheduler.validate(plan)
        del plan['tasks'][0]['max_seconds']
        with self.assertRaises(KeyError):scheduler.validate(plan)

    def check(self, task, size=0, memory=1):
        process=Mock()
        process.memory_full_info.return_value=SimpleNamespace(pss=memory)
        process.memory_info.return_value=SimpleNamespace(rss=memory)
        process.children.return_value=[]
        with patch.object(runtime.time,'monotonic',return_value=10**10), \
             patch.object(runtime,'scope_bytes',return_value=size), \
             patch.object(runtime,'save') as save, patch('psutil.Process',return_value=process):
            guard=runtime.ResourceCheck(task)
            guard.started=0
            guard('check')
            return save

    def test_runtime_no_limit_still_records_progress(self):
        save=self.check(self.task(None))
        save.assert_called_once()
        self.assertEqual(save.call_args.args[1]['elapsed_seconds'],10**10)

    def test_runtime_finite_limit_enforced(self):
        with self.assertRaisesRegex(RuntimeError,'wall time'):
            self.check(self.task(5))

    def test_unlimited_preserves_storage_limit(self):
        with self.assertRaisesRegex(RuntimeError,'output reservation'):
            self.check(self.task(None),size=1025)

    def test_unlimited_preserves_memory_limit(self):
        with self.assertRaisesRegex(RuntimeError,'RSS reservation'):
            self.check(self.task(None),memory=4097)

    def test_future_template_and_scientific_fields_unchanged(self):
        prior=json.loads((OUT/'effective_plan_snapshot.json').read_text())
        future=json.loads((OUT/'campaign_deadline_null_template.json').read_text())
        scheduler.validate(future)
        expected=copy.deepcopy(prior)
        expected['controller_source_refs']=future['controller_source_refs']
        for task in expected['tasks']:
            if task['kind']!='external':task['max_seconds']=None
        self.assertEqual(expected,future)
        for ref in future['controller_source_refs']:
            self.assertEqual(runtime.ref(ref['path']),ref)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DeadlineTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    runtime.save(OUT/'test_receipt.json',dict(schema='optional_campaign_deadline_tests_v1',
        status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),production_tasks_launched=0,
        production_paths_mutated=False,test_source_ref=runtime.ref(__file__)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
