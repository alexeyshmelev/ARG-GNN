"""Focused CPU source/factory checks for the GPU3/4/5 AM migration.

No preparation, full training, checkpoint restoration or GPU launch is performed.
A synthetic in-memory update-seven payload checks that resume is not fixed at six.
"""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from . import fit_v1 as am
from . import full_vram_v1 as gpu
from . import checkpoint_binding_v1 as checkpoint
from research.repairs.benchmark_recovery_20261020.training.candidate_common import ROOT,file_ref,read_ref

CHECKPOINT_REF=dict(path='experiments/results/benchmark_campaign_20261018_workflow/release_20261022_afternoon/other/AM_treeloom_fit/native/checkpoints/latest.pt',
    sha256='fcb60aafb004c0a577cc199db765db1436c81d30a2297f22c51bf4018a274bf1')


def function_source(path,name):
    text=path.read_text();lines=text.splitlines()
    node=next(node for node in ast.parse(text).body if isinstance(node,ast.FunctionDef) and node.name==name)
    return '\n'.join(lines[node.lineno-1:node.end_lineno])


class MigrationChecks(unittest.TestCase):
    def test_science_and_CUDA_initialization_fix_preserved(self):
        prior=ROOT/am.PREDECESSOR_FIT_REF['path']
        old=function_source(prior,'run_fit');new=function_source(Path(am.__file__),'run_fit')
        boundary="    check('TreeLoom_upfront_full_RAM_admission')"
        self.assertEqual(old[old.index(boundary):],new[new.index(boundary):])
        old_helper=ROOT/'research/repairs/benchmark_full_vram_20261022/training/full_vram_v2.py'
        self.assertEqual(function_source(old_helper,'enable_gpu'),function_source(Path(gpu.__file__),'enable_gpu'))
        self.assertEqual(set(gpu.gpu_binding.GPU_UUIDS),{3,4,5})
        self.assertEqual(gpu.gpu_binding.GPU_UUIDS[4],'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71')
        self.assertNotIn('RESUME_CHECKPOINT_REF',Path(am.__file__).read_text())
        for ref in am.source_refs():self.assertEqual(file_ref(ref['path']),ref)
        with self.assertRaises(ValueError):gpu.permission(2,'GPU-976a5cd3-df12-2535-fa51-e3ea063688ba')
        for index,uuid in gpu.gpu_binding.GPU_UUIDS.items():
            permission=gpu.permission(index,uuid)
            self.assertEqual(permission['authorized_physical_gpus'],[3,4,5])

    def test_factory_binds_real_completed_checkpoint_to_GPU4(self):
        job=read_ref(am.JOB);historical=job['call']['spec'];inputs=historical['training_spec']['train_inputs']
        arguments=dict(preparation_refs={parent:item['native_preparation_ref'] for parent,item in inputs.items()},
            generation_refs={parent:file_ref(Path(item['truth_ref']['path']).parents[2]/'observation_ready.json')
                for parent,item in inputs.items()},
            output_directory='experiments/results/benchmark_gpu345_CPU_factory_check_AM/native',
            max_rss_bytes=512*2**30,qualification_ref=historical['qualification_ref'],
            physical_gpu=4,gpu_uuid=gpu.gpu_binding.GPU_UUIDS[4],resume_checkpoint_ref=CHECKPOINT_REF)
        call=am.fit_factory(**arguments);spec=call['spec']
        self.assertEqual(call['module'],'research.repairs.benchmark_gpu345_20261022.training.fit_v1')
        self.assertEqual(spec['physical_gpu'],4)
        self.assertEqual(spec['resume_checkpoint_ref'],CHECKPOINT_REF)
        self.assertEqual(spec['max_rss_bytes'],512*2**30)
        self.assertEqual(spec['observation_cache_bytes'],352*2**30)
        self.assertEqual(spec['training_headroom_bytes'],128*2**30)
        self.assertEqual(spec['max_output_bytes'],64*2**20)
        self.assertNotIn('max_cuda_reserved_bytes',spec)
        gpu.validate_permission(spec)
        self.assertFalse((ROOT/arguments['output_directory']).exists())
        arguments['resume_checkpoint_ref']=None
        with self.assertRaises(ValueError):am.fit_factory(**arguments)

    def test_resume_accepts_later_completed_position_and_rejects_partial_counters(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        actual=checkpoint.inspect_completed_checkpoint(CHECKPOINT_REF,job_ref=am.JOB,barrier_ref=am.BARRIER)
        self.assertEqual(actual['complete_parent_updates'],6)
        self.assertEqual(actual['optimizer_parameter_states'],49)
        self.assertEqual(actual['updates_remaining'],594)
        saved=torch.load(ROOT/CHECKPOINT_REF['path'],map_location='cpu',weights_only=True)
        fresh,original,registry,plans=am.metadata_bindings(read_ref(am.JOB),read_ref(am.BARRIER))
        schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
        later=copy.deepcopy(saved)
        later.update(complete_parent_updates=7,next_schedule_index=7,
            epoch=schedule[6][0],last_completed_parent=schedule[6][1])
        for state in later['optimizer_state']['state'].values():state['step']=torch.tensor(7.)
        with patch.object(torch,'load',return_value=later):
            synthetic=checkpoint.inspect_completed_checkpoint(CHECKPOINT_REF,job_ref=am.JOB,barrier_ref=am.BARRIER)
        self.assertEqual(synthetic['complete_parent_updates'],7)
        self.assertEqual(synthetic['updates_remaining'],593)
        self.assertEqual(synthetic['next_parent_id'],schedule[7][1])
        next(iter(later['optimizer_state']['state'].values()))['step']=torch.tensor(6.)
        with patch.object(torch,'load',return_value=later),self.assertRaises(ValueError):
            checkpoint.inspect_completed_checkpoint(CHECKPOINT_REF,job_ref=am.JOB,barrier_ref=am.BARRIER)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
