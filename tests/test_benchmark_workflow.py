import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking import workflow as w
from arggnn.benchmarking import american_workflow as a


class WorkflowTest(unittest.TestCase):
    def test_resolve_requires_actual_success_and_selects_actual_products(self):
        state={'tasks':{'fit':{'status':'succeeded','result_ref':{'path':'result','sha256':'x'}}}}
        with patch.object(w.c,'read_ref',return_value={'value':{'checkpoint':{'path':'actual','sha256':'y'}}}):
            self.assertEqual(w.resolve({'$task':'fit','path':['checkpoint']},state),{'path':'actual','sha256':'y'})
            state['tasks']['fit']['status']='failed'
            with self.assertRaises(ValueError):w.resolve({'$task':'fit'},state)

    def test_shared_american_complete_dag_and_registered_test_barrier(self):
        tasks=a.build_tasks(prefix='smoke',output_root='experiments/results/smoke_plan_only')
        self.assertEqual(sum(t['resource']=='external' for t in tasks),6)
        self.assertEqual(sum(t['id'].endswith('_generation') for t in tasks),29)
        self.assertEqual(sum(t['id'].endswith('_focals') for t in tasks),35)
        self.assertEqual(sum(t['id'].endswith('_argformer_mlm') for t in tasks),15)
        self.assertEqual(sum(t['id'].endswith('_treeloom_prediction') for t in tasks),15)
        self.assertEqual(sum(t['id'].endswith('_flare_native') for t in tasks),15)
        score=next(t for t in tasks if t['id']=='smoke_score')
        self.assertEqual(len(score['dependencies']),45)
        self.assertEqual({m:len(p) for m,p in score['factory_arguments']['prediction_refs_by_method'].items()},
                         {'treeloom':15,'argformer':15,'flare':15})
        for t in tasks:
            if 'output_directory' in t and 'argformer_american_' in t['id']:
                self.assertTrue(any(x.startswith('argformer_american_') and x[-3:].isdigit()
                                    for x in Path(t['output_directory']).parts))

    def plan(self,tasks):
        return dict(schema=w.SCHEMA,physical_gpu=0,cpu_parallelism=2,sources=[{'path':'source','sha256':'x'}],
            tasks=tasks,max_parallel_rss_bytes=100,host_ram_margin_bytes=0)

    def test_reject_undeclared_artifact_reference_and_cuda_caps(self):
        task=dict(id='x',resource='gpu0',dependencies=[],factory_module='arggnn.benchmarking.american_workflow',
            factory_function='fit_factory',factory_arguments={},max_seconds=1,max_output_bytes=1,max_rss_bytes=1,
            minimum_free_gpu_bytes=74*2**30,max_cuda_reserved_bytes=74*2**30,
            output_directory='experiments/results/synthetic_workflow_test')
        with patch.object(w.c,'read_ref',side_effect=lambda r:self.plan([task])),patch.object(w,'source_check'):
            w.load_plan({'path':'plan','sha256':'x'})
            task['factory_arguments']={'x':{'$task':'unknown'}}
            with self.assertRaisesRegex(ValueError,'declared predecessor'):w.load_plan({'path':'plan','sha256':'x'})
            task['factory_arguments']={};task['max_cuda_reserved_bytes']=81*2**30
            with self.assertRaisesRegex(ValueError,'allocator caps'):w.load_plan({'path':'plan','sha256':'x'})

    def test_external_failed_undrained_keeps_ram_and_lane(self):
        with tempfile.TemporaryDirectory(dir=w.c.ROOT/'.runtime') as temp:
            root=Path(temp);queue=root/'queue.json';wait=root/'wait.json'
            queue.write_text(json.dumps({'jobs':[{'admission':{'job_id':'j'},'status':'failed'}]}))
            wait.write_text(json.dumps({'actual_direct_wait':True,'owned_process_group_absent':False}))
            plan={'external_resources':[dict(queue_path=str(queue),wait_path=str(wait),job_id='j',resource='cpu',max_rss_bytes=50)]}
            self.assertEqual(w.external_resources(plan),(1,50))
            wait.write_text(json.dumps({'actual_direct_wait':True,'owned_process_group_absent':True}))
            self.assertEqual(w.external_resources(plan),(0,0))

    def test_actual_generation_factory_uses_full_declared_recipe_without_execution(self):
        call=a.generation_factory(task_id='argformer_american_validation_000_simulate',
            output_directory='experiments/results/synthetic_factory/argformer_american_validation_000/new',
            limits=dict(max_seconds=7200,max_rss_bytes=128*2**30,max_output_bytes=20*2**30))
        self.assertEqual(call['function'],'run')
        plan=a.generation.prepare(call['spec'])
        self.assertEqual(plan['config']['sequence_length_bp'],30000000)
        self.assertEqual(plan['config']['sample_counts']['ADMIX'],1000)
        self.assertFalse(Path(plan['output_directory']).exists())


if __name__=='__main__':unittest.main()
