"""Focused CPU checks for exact GPU3/4/5 authority and inherited guard preservation."""
import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import contracts_recovery_v1 as contracts
from . import shared_gpu_v1 as hardware
from . import scientific_gpu_binding_v1 as scientific
from . import scientific_setup_v1 as setup
from . import queue_gpu_v1 as queue

UUID4='GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71'


class GPU345Tests(unittest.TestCase):
    def test_exact_device_four_and_no_device_two(self):
        self.assertEqual(set(hardware.GPU_UUIDS),{3,4,5})
        self.assertEqual(scientific.GPU_UUIDS,hardware.GPU_UUIDS)
        descriptor=dict(physical_gpu=4,gpu_uuid=UUID4)
        self.assertEqual(hardware.target(descriptor),(4,UUID4))
        self.assertEqual(scientific.target(4,UUID4),descriptor)
        for index in (0,1,2):
            with self.assertRaises(RuntimeError):hardware.target(dict(physical_gpu=index,gpu_uuid=UUID4))
            with self.assertRaises(ValueError):scientific.target(index,UUID4)
        environment=dict(ARGGNN_PHYSICAL_GPU='4',CUDA_VISIBLE_DEVICES=UUID4,ARGGNN_GPU_MEMORY_POLICY='full_device')
        with tempfile.TemporaryFile() as lease,patch.dict(os.environ,environment,clear=True):
            os.environ['ARGGNN_READABLE_GPU_FD']=str(lease.fileno())
            queue.validate_policy_environment(dict(descriptor,gpu_memory_policy='full_device'))
            scientific.validate_env(descriptor)
            os.environ['ARGGNN_MAX_CUDA_RESERVED_BYTES']='1'
            with self.assertRaises(RuntimeError):queue.validate_policy_environment()

    def test_device_four_inventory_uses_exact_hardware_uuid(self):
        result=SimpleNamespace(stdout='4, '+UUID4+', 81920, 1024, 80896\n')
        calls=[]
        def run(argv,**kwargs):calls.append(argv);return result
        sample=hardware.gpu_sample(4,run=run)
        self.assertEqual(sample['physical_gpu'],4)
        self.assertEqual(sample['uuid'],UUID4)
        self.assertIn('--id=4',calls[0])
        with patch.object(hardware.subprocess,'run',return_value=SimpleNamespace(stdout=UUID4+', 123, 4096\n')):
            self.assertEqual(hardware.process_bytes(4),{123:4096*2**20})

    def test_current_authority_supersedes_legacy_compatibility_fields(self):
        reference=dict(path='policy.json',sha256='a'*64)
        policy=dict(benchmark_execution_approved=True,status='approved',approval_origin='explicit_user_go_ahead',
            expires_at_utc='2027-10-22T00:00:00+00:00',smoke_tests_only=False,gpu_memory_policy='full_device',
            currently_authorized_physical_gpus=[3,4,5],legacy_gpu_authority_status='preserved_for_running_workers_only',
            physical_gpus=[5],additional_physical_gpus=[2,3],user_authorized_physical_gpus=[2,3,5])
        for key in ('approved_readable_workflow_ref','approved_parallel_workflow_ref','approved_recovery_workflow_ref',
                    'approved_recovery_continuation_workflow_ref','approved_full_vram_workflow_ref','approved_gpu345_workflow_ref'):
            policy[key]=reference
        with patch.object(contracts,'read_json',return_value=policy),patch.object(contracts,'file_ref',return_value=reference),patch.object(contracts,'read_ref',return_value=policy):
            self.assertIs(contracts.check_execution_release(reference),policy)
            policy['currently_authorized_physical_gpus']=[2,3,5]
            with self.assertRaises(contracts.ContractError):contracts.check_execution_release(reference)
            policy['currently_authorized_physical_gpus']=[3,4,5]
            policy['legacy_gpu_authority_status']='active_future_authority'
            with self.assertRaises(contracts.ContractError):contracts.check_execution_release(reference)

    def test_new_process_binding_updates_imported_native_helper_without_source_edits(self):
        from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding as legacy_scientific
        from research.repairs.benchmark_multigpu_20261022.native import device_binding_v1 as legacy_native
        old_scientific={name:getattr(legacy_scientific,name) for name in ('GPU_UUIDS','target','validate_env','verify_before_cuda')}
        old_native=legacy_native.UUID_BY_DEVICE
        imported_native_helper=legacy_native.require_job_device
        try:
            setup.install_device_bindings()
            self.assertEqual(legacy_scientific.target(4,UUID4),dict(physical_gpu=4,gpu_uuid=UUID4))
            with patch.dict(os.environ,dict(ARGGNN_PHYSICAL_GPU='4',CUDA_VISIBLE_DEVICES=UUID4),clear=True):
                self.assertEqual(imported_native_helper(),(4,UUID4))
            with self.assertRaises(ValueError):legacy_scientific.target(2,UUID4)
        finally:
            for name,value in old_scientific.items():setattr(legacy_scientific,name,value)
            legacy_native.UUID_BY_DEVICE=old_native

    def test_unchanged_sources_and_non_device_runtime_guards(self):
        successor=Path(__file__).parent
        original=Path('research/repairs/benchmark_full_vram_20261022/runtime')
        receipt=json.loads((successor/'source_copy_provenance_v1.json').read_text())
        for filename,digest in receipt['unchanged_original_sources'].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(),digest,filename)
        for filename,intentional in [('readable_workflow_v1.py',{'load_plan','current_policy','materialize'}),
                                    ('queue_execute_v1.py',{'child'}),('queue_native_worker_v1.py',set()),
                                    ('native_telemetry_v1.py',set())]:
            old={node.name:node for node in ast.parse((original/filename).read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
            new={node.name:node for node in ast.parse((successor/filename).read_text()).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
            for identifier,node in old.items():
                if identifier not in intentional:self.assertEqual(ast.dump(node),ast.dump(new[identifier]),filename+':'+identifier)
        for filename in ('queue_gpu_v1.py','queue_execute_v1.py','queue_native_worker_v1.py','scientific_setup_v1.py','native_telemetry_v1.py'):
            tree=ast.parse((successor/filename).read_text())
            for node in ast.walk(tree):
                self.assertFalse(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and node.func.attr=='set_per_process_memory_fraction',filename)


if __name__=='__main__':unittest.main()
