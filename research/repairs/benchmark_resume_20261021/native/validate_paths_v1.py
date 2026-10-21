"""Bounded metadata-only validation of the output namespace repair.

Uses existing sealed public inference/focal/preparation/completion metadata. It
opens no checkpoint tensors, numerical arrays or held-out ancestry truth and
never invokes a scientific worker, launcher, signal or Git command.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from . import plan_paths_v1 as repair
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import american_competitors as original

BASE='research/repairs/benchmark_resume_20261021/native'
SOURCE_PLAN='configs/benchmarks/executable_workflow_20261020_readable_v3.json'
SOURCE_STATE='.runtime/benchmarking/workflow_20261020_readable_v3/state.json'
OUTPUT_ROOT='experiments/results/benchmark_campaign_20261018_workflow/release_20261021_gpu5'
TARGET='AM_argformer_american_validation_001_argformer_embeddings'


def save(name,value):
    path=c.contained_path(BASE+'/'+name)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    return c.file_ref(path)


def resolved_arguments(task,state):
    return {key:(repair.result_value(value,state) if isinstance(value,dict) and '$task' in value else deepcopy(value))
            for key,value in task['factory_arguments'].items()}


class PathChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=c.read_json(SOURCE_PLAN);cls.state=c.read_json(SOURCE_STATE)
        cls.state_before=deepcopy(cls.state)
        cls.revised,cls.evidence=repair.repair_plan_paths(cls.plan,cls.state,OUTPUT_ROOT)
        cls.old=next(task for task in cls.plan['tasks'] if task['id']==TARGET)
        cls.new=next(task for task in cls.revised['tasks'] if task['id']==TARGET)
        cls.arguments=resolved_arguments(cls.old,cls.state)
        cls.descriptor=original.argformer_factory(**cls.arguments,
            output_directory=str(c.contained_path(cls.new['output_directory'])/'native'))

    def test_original_failure_is_only_the_parent_component(self):
        with self.assertRaisesRegex(ValueError,'Fresh canonical same-parent'):
            original.argformer_factory(**self.arguments,
                output_directory=str(c.contained_path(self.old['output_directory'])/'native'))
        self.assertEqual(self.descriptor['spec']['stage'],'embeddings')
        self.assertEqual(self.descriptor['spec']['parent_id'],'argformer_american_validation_001')

    def test_science_and_completed_checkpoint_are_identical(self):
        reference=original.argformer_factory(**self.arguments,output_directory=str(
            c.contained_path(OUTPUT_ROOT)/'metadata_comparison'/'argformer_american_validation_001'/'native'))
        actual=deepcopy(self.descriptor);expected=deepcopy(reference)
        actual['spec'].pop('output_directory');expected['spec'].pop('output_directory')
        self.assertEqual(actual,expected)
        producer=original.read(self.arguments['producer_ref'])
        self.assertEqual(self.descriptor['spec']['checkpoint_ref'],producer['checkpoint_ref'])
        self.assertEqual(producer['actual_optimizer_updates'],2099)
        self.assertEqual(producer['optimizer_unique_step_values'],[2099])

    def test_all_fresh_competitor_scopes_and_completed_descriptors(self):
        before={task['id']:task for task in self.plan['tasks']}
        after={task['id']:task for task in self.revised['tasks']}
        self.assertEqual(len(self.evidence['scope_changes']),67)
        for row in self.evidence['scope_changes']:
            scope=c.contained_path(row['after']);parent=row['parent_id']
            self.assertIn(parent,scope.parts)
            self.assertEqual(original._out(str(scope/'native'),parent),scope/'native')
            self.assertFalse(scope.exists())
        for identifier,entry in self.state['tasks'].items():
            if entry['status']=='succeeded':self.assertEqual(before[identifier],after[identifier])
        self.assertEqual(self.state,self.state_before)

    def test_retirement_and_archive_owners_use_actual_producers(self):
        by_id={task['id']:task for task in self.revised['tasks']}
        for task in self.revised['tasks']:
            if self.state['tasks'][task['id']]['status']=='succeeded':continue
            arguments=task.get('factory_arguments',{})
            if task.get('factory_module')==repair.RETENTION:
                owners=[repair.owner_scope(by_id[arguments[key]['$task']]) for key in
                        ('preparation_ref','embedding_ref','prediction_ref')]
                self.assertEqual(arguments['owner_scopes'],owners)
                for owner in owners:
                    self.assertIn('benchmark_campaign_20261018_workflow',Path(owner).parts)
            if task.get('factory_module')==repair.AP_ARCHIVE:
                self.assertEqual(arguments['owner_scope'],repair.owner_scope(by_id[arguments['owner_task_id']]))
        retirement=by_id['AM_argformer_american_validation_001_retire_native_scratch']
        self.assertIn('/release_v7/',retirement['factory_arguments']['owner_scopes'][0])
        self.assertEqual(c.contained_path(retirement['factory_arguments']['owner_scopes'][1])/'native',
                         c.contained_path(self.descriptor['spec']['output_directory']))

    def test_cross_parent_join_and_running_scope_are_rejected(self):
        mixed=deepcopy(self.plan)
        task=next(task for task in mixed['tasks'] if task['id']==TARGET)
        task['factory_arguments']['focal_preparation_ref']['$task']='AM_argformer_american_validation_002_focals'
        with self.assertRaisesRegex(ValueError,'same-parent'):
            repair.repair_plan_paths(mixed,self.state,OUTPUT_ROOT)
        running=deepcopy(self.state);running['tasks'][TARGET]['status']='running'
        with self.assertRaisesRegex(ValueError,'Only never-launched'):
            repair.repair_plan_paths(self.plan,running,OUTPUT_ROOT)


def main():
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PathChecks))
    if not result.wasSuccessful():raise SystemExit(1)
    state_snapshot=save('source_state_snapshot_v1.json',PathChecks.state)
    mapping=save('path_mapping_v1.json',dict(PathChecks.evidence,source_plan_ref=c.file_ref(SOURCE_PLAN),
        source_state_snapshot_ref=state_snapshot))
    receipt=dict(schema='same_parent_competitor_paths_validation_v1',source_plan_ref=c.file_ref(SOURCE_PLAN),
        source_state_snapshot_ref=state_snapshot,path_mapping_ref=mapping,tests_passed=result.testsRun,
        materialization_only_failure_reproduced=True,actual_adopted_contrastive_accepted=True,
        producer_ref=PathChecks.arguments['producer_ref'],checkpoint_ref=PathChecks.descriptor['spec']['checkpoint_ref'],
        embedding_serialization_bound_bytes=PathChecks.descriptor['spec']['embedding_serialization_bound_bytes'],
        source_refs=[c.file_ref(BASE+'/plan_paths_v1.py'),c.file_ref(__file__),
            c.file_ref('src/arggnn/benchmarking/american_competitors.py'),c.file_ref('src/arggnn/benchmarking/retention.py')],
        completed_stage_descriptors_unchanged=True,original_scientific_factory_unchanged=True,
        physical_device_not_changed_by_this_helper=True,scientific_launches=0,CUDA_contexts=0,
        checkpoint_payloads_opened=0,heldout_truth_opened=False,signals_sent=0,Git_operations=0)
    print('VALIDATION_RECEIPT',save('validation_v1.json',receipt))

if __name__=='__main__':main()
