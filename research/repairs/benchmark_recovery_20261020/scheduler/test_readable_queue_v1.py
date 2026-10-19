"""Synthetic CPU checks. No scientific runs, real CUDA or external signalling."""
import ast
import copy
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from . import queue_adoption_v1 as adoption, queue_gpu0_v1 as gpu, queue_support_v1 as support
from . import readable_workflow_v1 as workflow


class AdoptionTests(unittest.TestCase):
    def test_candidate_two_field_identity(self):
        binding=dict(kind='treeloom_candidate_fit',output_directory='experiments/results/synthetic',supervisor_identity=dict(pid=10,start_ticks=20))
        root=dict(pid=30,start_ticks=40,pgid=30,session=30,ppid=10)
        with patch.object(adoption.c,'read_json',return_value=dict(supervisor_identity=binding['supervisor_identity'],worker_identity=dict(pid=30,start_ticks=40))),patch.object(adoption.lifecycle,'identity',return_value=root):
            self.assertEqual(adoption.scientific_root(binding),root)
        with patch.object(adoption.c,'read_json',return_value=dict(supervisor_identity=binding['supervisor_identity'],worker_identity=dict(pid=30,start_ticks=41))),patch.object(adoption.lifecycle,'identity',return_value=root):
            with self.assertRaises(ValueError):adoption.scientific_root(binding)

    def test_unknown_outer_wait_stays_unknown(self):
        evidence=adoption.completion_evidence(dict(supervisor_identity=dict(pid=10,start_ticks=20)))
        self.assertFalse(evidence['actual_outer_OS_wait_known'])
        self.assertIsNone(evidence['actual_outer_returncode'])
        self.assertEqual(evidence['scientific_direct_returncode'],0)

    def test_exact_supervisor_generation_required(self):
        with patch.object(adoption,'identity',return_value=dict(pid=10,start_ticks=20)):
            self.assertFalse(adoption.generation_absent(dict(pid=10,start_ticks=20)))
            self.assertTrue(adoption.generation_absent(dict(pid=10,start_ticks=21)))

    def test_native_rejects_partial_budget(self):
        binding=dict(kind='native_contrastive',task_id='T',output_directory='experiments/results/synthetic',request_ref={'path':'request'},launch_ref={'path':'launch'},supervisor_identity=dict(pid=10,start_ticks=20))
        task=dict(id='T');spec=dict(expected_updates=2099,parent_id='P',preparation_ref={},focal_registry_ref={})
        records={'request':dict(task=task,call=dict(spec=spec)),'launch':dict(request_ref=binding['request_ref'],identity=binding['supervisor_identity']),
          'wait':dict(actual_direct_wait=True,actual_direct_returncode=0,owned_process_group_absent=True,supervisor_error=None,job_ref={'path':'job'}),
          'job':dict(task=task,call=dict(spec=spec),shared_native_request_ref=binding['request_ref']),
          'result':dict(job_ref={'path':'job'},value={'path':'science'}),
          'science':dict(actual_optimizer_updates=2098,optimizer_unique_step_values=[2098],completed_epochs=1,stage='contrastive',parent_id='P',preparation_ref={},focal_registry_ref={},query_ancestry_truth_opened=False)}
        def ref(path):return {'path':'wait' if Path(path).name=='actual_wait.json' else 'result'}
        with patch.object(adoption,'drained',return_value=True),patch.object(support.gpu,'identity',side_effect=FileNotFoundError),patch.object(Path,'is_file',return_value=True),patch.object(support.c,'file_ref',side_effect=ref),patch.object(support.c,'read_ref',side_effect=lambda r:records[r['path']]):
            with self.assertRaisesRegex(ValueError,'full unchanged budget'):support.standalone_result(binding)
            records['science'].update(actual_optimizer_updates=2099,optimizer_unique_step_values=[2099])
            result=support.standalone_result(binding)
            self.assertEqual(result['status'],'succeeded')
            self.assertNotIn('actual_controller_wait',result)
            self.assertFalse(result['controller_completion_evidence']['actual_outer_OS_wait_known'])

    def test_archive_mirror_reservation_and_success_once(self):
        task=dict(id='A',closed_storage_recovery_target='x',max_rss_bytes=1)
        owner=dict(config_ref={'path':'cfg'},launch_ref={'path':'launch'},state_path='.runtime/synthetic/state.json',supervisor_identity=dict(pid=1,start_ticks=2))
        plan=dict(tasks=[task],standalone_archive_owner=owner)
        config=dict(records_directory='.runtime/synthetic',recovery_plan=dict(tasks=[task]))
        source=dict(plan_ref=owner['config_ref'],tasks=dict(A=dict(status='running',reservation_bytes=18)))
        state=dict(tasks={})
        with patch.object(adoption.c,'read_ref',side_effect=lambda ref:config if ref==owner['config_ref'] else dict(config_ref=owner['config_ref'],supervisor_identity=owner['supervisor_identity'])),patch.object(adoption.c,'read_json',return_value=source):
            adoption.refresh_archives(plan,state);self.assertEqual(state['tasks']['A']['reservation_bytes'],18)
            source['tasks']['A'].update(status='succeeded',retained_bytes=3,actual_controller_wait=0,owned_process_group_absent=True)
            adoption.refresh_archives(plan,state);self.assertEqual(state['tasks']['A']['retained_bytes'],3)
            self.assertIn('no duplicate dispatch',adoption.archive_dispatch_reason(plan,task))

    def test_candidate_exact_outer_schema_and_final600(self):
        binding=dict(kind='treeloom_candidate_fit',request_ref={'path':'request'},output_directory='experiments/results/synthetic',supervisor_identity=dict(pid=10,start_ticks=20))
        records={'request':dict(job_ref={'path':'job'},observation_barrier_ref={'path':'barrier'}),
          'job':dict(call=dict(spec=dict(schema='treeloom_full20_compact_RAM_fit_v4',training_spec=dict(parent_registry_ref={'path':'registry'})))),
          'wait':dict(actual_exit_code=0,success=True,resource_violation=None),
          'result':dict(schema='treeloom_fresh_full_training_completion_v1',request_ref=binding['request_ref'],observation_barrier_ref={'path':'barrier'},
            parent_registry_ref={'path':'registry'},complete_parent_updates=600,completed_epochs=30,final_saved_optimizer_parameter_states=49,
            final_saved_optimizer_steps=[600],checkpoint_selection='fixed_final_epoch30',selected_epoch=30,VAL_truth_read=False,TEST_truth_read=False,
            checkpoint_ref={'path':'experiments/results/synthetic/checkpoint.pt'})}
        def file_ref(path):return {'path':'wait' if Path(path).name=='supervisor_result.json' else 'result'}
        with patch.object(adoption,'drained',return_value=True),patch.object(adoption.c,'read_json',return_value=dict(request_ref=binding['request_ref'],supervisor_identity=binding['supervisor_identity'])),patch.object(adoption.c,'file_ref',side_effect=file_ref),patch.object(adoption.c,'read_ref',side_effect=lambda ref:records[ref['path']]):
            result=adoption.candidate_result(binding)
            self.assertEqual(result['status'],'succeeded')
            self.assertFalse(result['controller_completion_evidence']['actual_outer_OS_wait_known'])
            records['result']['complete_parent_updates']=599
            with self.assertRaisesRegex(ValueError,'Complete unchanged candidate'):adoption.candidate_result(binding)

    def test_archive_handoff_requires_child_terminal(self):
        task=dict(id='A',output_directory='experiments/results/synthetic')
        config=dict(records_directory='.runtime/synthetic',recovery_plan=dict(tasks=[task]))
        launch=dict(config_ref={'path':'config'},supervisor_identity=dict(pid=10,start_ticks=20))
        state=dict(plan_ref={'path':'config'},tasks=dict(A=dict(status='running',job_ref={'path':'job'},reservation_bytes=18)))
        with patch.object(adoption,'generation_absent',return_value=True),patch.object(adoption.c,'read_ref',side_effect=lambda ref:config if ref['path']=='config' else launch),patch.object(adoption.c,'read_json',return_value=state):
            with self.assertRaisesRegex(ValueError,'genuine settled terminal'):
                adoption.closed_archive_controller_proof({'path':'config'},{'path':'launch'})


class CapacityTests(unittest.TestCase):
    def test_standalone_unallocated_cuda_growth(self):
        binding=dict(max_cuda_reserved_bytes=40*2**30)
        plan=dict(tasks=[dict(standalone_predecessor=binding)])
        root=dict(pid=10,start_ticks=20,pgid=10,session=10)
        class Tracker:
            def __init__(self,pid):pass
            def capture(self):return [10]
        with patch.object(adoption,'drained',return_value=False),patch.object(adoption,'scientific_root',return_value=root),patch('arggnn.benchmarking.workflow_v4.identity',return_value=root),patch('research.reviews.benchmark_repairs_20261019.supervision.finite_pss_v1.repaired_owned_group_class',return_value=Tracker):
            self.assertEqual(gpu.standalone_outstanding_growth(plan,{10:18*2**30}),24*2**30)
        with patch.object(adoption,'drained',return_value=False),patch.object(adoption,'scientific_root',side_effect=FileNotFoundError):
            self.assertEqual(gpu.standalone_outstanding_growth(plan,{10:18*2**30}),42*2**30)

    def test_free_memory_deducts_growth(self):
        with patch.object(gpu,'gpu_sample',return_value=dict(free_bytes=50*2**30)),patch.object(gpu,'_outstanding_growth_locked',return_value=24*2**30):
            self.assertIsNone(gpu._admission_reason_locked({},dict(max_cuda_reserved_bytes=16*2**30,minimum_free_gpu_bytes=26*2**30)))
            self.assertIsNotNone(gpu._admission_reason_locked({},dict(max_cuda_reserved_bytes=17*2**30,minimum_free_gpu_bytes=27*2**30)))

    def test_ram_subtracts_remaining_growth_only(self):
        rows=[dict(full_cap_bytes=512,remaining_growth_bytes=212)]
        with patch.object(support,'managed_ram_rows',return_value=rows),patch.object(support.accounting,'kernel_available_ram_bytes',return_value=400):
            plan=dict(max_parallel_rss_bytes=1000,host_ram_margin_bytes=50)
            self.assertTrue(support.ram_allows(plan,{},dict(max_rss_bytes=100)))
            self.assertFalse(support.ram_allows(plan,{},dict(max_rss_bytes=150)))

    def test_no_concealed_native_payload(self):
        support.validate_native_command(['python','-B','-s','-m','package.worker','--spec','input.json'])
        for argv in (['python','-c','print(1)'],['powershell','-EncodedCommand','abc']):
            with self.assertRaises(ValueError):support.validate_native_command(argv)

    def test_deferred_cpu_has_no_early_gpu_admission(self):
        source=Path(workflow.__file__).read_text()
        self.assertFalse(support.gpu_dispatch_needed(dict(resource='cpu',gpu_after_preparation=True)))
        self.assertTrue(support.gpu_dispatch_needed(dict(resource='gpu0')))
        self.assertIn('not support.archive_controller_live(plan)',source)

    def test_retention_owners_follow_actual_producers(self):
        from .build_readable_queue_v1 import rebind_retention_owners
        prepare=dict(id='P',resource='external',workflow_predecessor=dict(output_directory='old/adopted/P'))
        embedding=dict(id='E',resource='gpu0',output_directory='fresh/E')
        prediction=dict(id='R',resource='gpu0',output_directory='fresh/R')
        retirement=dict(id='C',resource='cpu',factory_module='arggnn.benchmarking.retention',
            remeasure_succeeded_tasks=['P','E'],factory_arguments=dict(preparation_ref={'$task':'P'},embedding_ref={'$task':'E'},prediction_ref={'$task':'R'},owner_scopes=['stale/P','stale/E','stale/R']))
        archive=dict(id='A',resource='cpu',factory_module='arggnn.benchmarking.european_retention_v9',factory_arguments=dict(owner_task_id='E',owner_scope='stale/E'))
        plan=dict(tasks=[prepare,embedding,prediction,retirement,archive])
        rebind_retention_owners(plan,set())
        self.assertEqual(retirement['factory_arguments']['owner_scopes'],['old/adopted/P','fresh/E','fresh/R'])
        self.assertEqual(retirement['remeasure_succeeded_tasks'],['E'])
        self.assertEqual(archive['factory_arguments']['owner_scope'],'fresh/E')


class StaticSourceTests(unittest.TestCase):
    def test_no_runtime_source_execution_in_new_queue(self):
        folder=Path(workflow.__file__).parent
        for path in sorted(folder.glob('queue_*v1.py'))+[Path(workflow.__file__)]:
            tree=ast.parse(path.read_text())
            bad=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'exec','eval','compile','FunctionType'}]
            self.assertFalse(bad,path.name)

    def test_literal_author_dense_is_exact(self):
        source=ast.parse((support.c.ROOT/'.runtime/author_sources/ancestralpaths_77db3cfae100/training.py').read_text())
        static=ast.parse((Path(workflow.__file__).parent/'queue_european_models_v1.py').read_text())
        original=next(node for node in source.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in node.targets))
        function=next(node for node in static.body if isinstance(node,ast.FunctionDef) and node.name=='build_author_dense')
        copied=next(node for node in function.body if isinstance(node,ast.Assign))
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(copied,include_attributes=False))
        compile_original=[node for node in source.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='compile'][-1]
        compile_copied=next(node for node in function.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='compile')
        self.assertEqual(ast.dump(compile_original,include_attributes=False),ast.dump(compile_copied,include_attributes=False))


if __name__=='__main__':unittest.main()
