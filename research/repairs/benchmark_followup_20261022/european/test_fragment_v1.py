"""Metadata-only acceptance checks. No qualifier or scientific stage is run."""
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest

from research.repairs.benchmark_resume_20261021.gpu import readable_workflow_v1 as workflow
from research.repairs.benchmark_resume_20261021.gpu.queue_adapters import common, european_stage as stage
from .prepare_qualification_fragment_v1 import HERE, FACTORY


class FragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fragment=workflow.c.read_json(HERE/'qualification_plan_fragment_v1.json')
        cls.plan=workflow.c.read_ref(cls.fragment['base_plan_ref'])
        cls.tasks={task['id']:task for task in cls.plan['tasks']}

    def test_exact_frozen_21_source_contract(self):
        self.assertEqual(self.fragment['exact_candidate_source_refs'],common.source_refs())
        self.assertEqual(len(common.source_refs()),21)
        for reference in self.fragment['required_source_refs']:
            self.assertEqual(workflow.c.file_ref(reference['path']),reference)

    def test_two_fresh_CPU_tasks_and_concrete_current_calls(self):
        fresh=self.fragment['qualification_tasks']
        calls=self.fragment['qualification_descriptors']
        self.assertEqual(len(fresh),2)
        self.assertEqual(len({task['id'] for task in fresh}),2)
        for task,wrapped in zip(fresh,calls):
            self.assertNotIn(task['id'],self.tasks)
            self.assertEqual(task['factory_module'],FACTORY)
            self.assertEqual(task['factory_function'],'qualification_factory')
            self.assertEqual(task['resource'],'cpu')
            self.assertFalse((workflow.c.ROOT/task['output_directory']).exists())
            self.assertEqual(task['max_rss_bytes'],128*2**30)
            self.assertEqual(task['max_seconds'],7200)
            call=wrapped['descriptor']
            self.assertEqual(call['module'],FACTORY)
            self.assertEqual(call['function'],'run_qualification')
            self.assertTrue(call['spec']['CPU_only'])
            self.assertEqual(call['spec']['candidate_source_refs'],common.source_refs())
            self.assertEqual(call['spec']['output_directory'],task['output_directory']+'/native')
            for dependency in task['dependencies']:self.assertIn(dependency,self.tasks)

    def test_fit_changes_only_qualification_binding_and_dependency(self):
        for binding in self.fragment['fit_rebindings']:
            old=self.tasks[binding['task_id']]
            replacement=copy.deepcopy(binding['replacement_task'])
            self.assertEqual(workflow.c.fingerprint(old),binding['previous_task_fingerprint'])
            new_id=replacement['factory_arguments']['qualification_ref']['$task']
            self.assertIn(new_id,{task['id'] for task in self.fragment['qualification_tasks']})
            original_id=old['factory_arguments']['qualification_ref']['$task']
            self.assertNotIn(original_id,replacement['dependencies'])
            self.assertIn(new_id,replacement['dependencies'])
            replacement['dependencies']=[original_id if dep==new_id else dep for dep in replacement['dependencies']]
            replacement['factory_arguments']['qualification_ref']=copy.deepcopy(old['factory_arguments']['qualification_ref'])
            self.assertEqual(replacement,old)
            self.assertEqual(binding['previous_failed_state']['status'],'failed')
            self.assertNotIn('job_ref',binding['previous_failed_state'])
            self.assertNotIn('reservation_bytes',binding['previous_failed_state'])
            self.assertFalse((workflow.c.ROOT/old['output_directory']).exists())

    def test_original_qualifications_remain_original_and_inputs_match(self):
        for diagnosis,wrapped in zip(self.fragment['diagnoses'],self.fragment['qualification_descriptors']):
            historical=workflow.c.read_ref(diagnosis['historical_qualification_ref'])
            self.assertEqual(historical['schema'],'european_v9_complete_parent_CPU_qualification_v1')
            self.assertEqual(len(historical['candidate_source_refs']),9)
            self.assertEqual(historical['observed_inputs'],wrapped['descriptor']['spec']['observed_inputs'])
            self.assertTrue(diagnosis['observed_inputs_unchanged'])
            self.assertTrue(diagnosis['preparation_contract_unchanged'])
            with self.assertRaisesRegex(workflow.c.ContractError,'Actual same-source same-input'):
                stage._require_pass(diagnosis['historical_qualification_ref'],wrapped['descriptor']['spec'])

    def test_fragment_has_no_execution_or_authority_change(self):
        self.assertTrue(self.fragment['no_execution_authority'])
        for name in ['canonical_policy_modified','current_plan_or_state_modified','benchmark_launched','processes_signalled']:
            self.assertFalse(self.fragment[name])
        torch=sys.modules.get('torch')
        self.assertTrue(torch is None or not torch.cuda.is_initialized())


def main():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FragmentTests))
    transcript=stream.getvalue();print(transcript,flush=True)
    with (HERE/'metadata_tests_v1.txt').open('x') as handle:handle.write(transcript)
    receipt=dict(schema='European_qualification_fragment_metadata_acceptance_v1',
        status='PASS' if result.wasSuccessful() else 'FAIL',created_utc=datetime.now(timezone.utc).isoformat(),
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        fragment_ref=workflow.c.file_ref(HERE/'qualification_plan_fragment_v1.json'),
        source_refs=[workflow.c.file_ref(HERE/name) for name in ['prepare_qualification_fragment_v1.py','test_fragment_v1.py']],
        transcript_ref=workflow.c.file_ref(HERE/'metadata_tests_v1.txt'),
        qualifier_executed=False,scientific_stage_executed=False,CUDA_initialized=False,
        processes_signalled=False,policy_or_queue_state_modified=False)
    with (HERE/'metadata_acceptance_v1.json').open('x') as handle:
        json.dump(receipt,handle,indent=2,sort_keys=True);handle.write('\n')
    print(json.dumps(dict(status=receipt['status'],tests=result.testsRun,
        receipt_ref=workflow.c.file_ref(HERE/'metadata_acceptance_v1.json')),sort_keys=True),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
