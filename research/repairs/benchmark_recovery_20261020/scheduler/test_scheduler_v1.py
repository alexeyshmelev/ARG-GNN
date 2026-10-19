"""Synthetic CPU checks. No CUDA imports, actual Popen calls, or process signals."""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import native_continuation_v1 as route
from . import shared_gpu0_v1 as gpu

ROOT = route.c.ROOT
DIRECTORY = Path(__file__).resolve().parent


def function(path, name):
    return next(node for node in ast.parse(Path(path).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == name)


class SharingTests(unittest.TestCase):
    def test_exact_GPU0_inventory(self):
        fake=lambda *a,**k:SimpleNamespace(stdout='0, '+gpu.GPU0_UUID+', 81920, 24576, 57344\n')
        row=gpu.gpu_sample(fake)
        self.assertEqual(row['free_bytes'],56*2**30)
        fake=lambda *a,**k:SimpleNamespace(stdout='1, '+gpu.GPU0_UUID+', 81920, 24576, 57344\n')
        with self.assertRaisesRegex(RuntimeError,'identity'):gpu.gpu_sample(fake)

    def test_driver_reserved_memory_is_explicit_and_not_counted_as_free(self):
        fake=lambda *a,**k:SimpleNamespace(stdout='0, '+gpu.GPU0_UUID+', 81920, 21158, 59996\n')
        row=gpu.gpu_sample(fake)
        self.assertEqual(row['driver_reserved_bytes'],766*2**20)
        self.assertEqual(row['free_bytes'],59996*2**20)
        gpu.check_start(row)

    def test_start_requires_50GiB_and_full_explicit_envelope(self):
        gpu.check_start(dict(free_bytes=50*2**30))
        with self.assertRaisesRegex(RuntimeError,'50GiB'):gpu.check_start(dict(free_bytes=50*2**30-1))
        self.assertEqual((gpu.CUDA_CAP,gpu.DRIVER_MARGIN,gpu.LIVE_FREE),(40*2**30,2*2**30,8*2**30))

    def test_sustained_live_floor_and_recovery(self):
        monitor=gpu.HeadroomMonitor()
        for free in (7,7,8,7,7):monitor.observe(dict(free_bytes=free*2**30))
        with self.assertRaisesRegex(RuntimeError,'only the new native'):monitor.observe(dict(free_bytes=7*2**30))
        self.assertEqual(monitor.minimum_free_bytes,7*2**30)

    def test_driver_cap_counts_only_new_owned_PIDs(self):
        fake=lambda *a,**k:SimpleNamespace(stdout=gpu.GPU0_UUID+', 111, 1024\n'+gpu.GPU0_UUID+', 222, 60000\n')
        self.assertEqual(gpu.native_driver_bytes({111},fake),2**30)
        with self.assertRaisesRegex(RuntimeError,'driver margin'):gpu.native_driver_bytes({222},fake)

    def test_scientific_functions_stay_literal_equivalent(self):
        original=ROOT/'src/arggnn/benchmarking/american_competitor_worker.py'
        successor=DIRECTORY/'native_worker_v1.py'
        for name in ('checkpoint_counters','_configuration','_verify_encoder_transfer','_embeddings','_author'):
            self.assertEqual(ast.dump(function(original,name)),ast.dump(function(successor,name)),name)

    def test_seal_has_exactly_one_readable_RAM_load_substitution(self):
        original=function(ROOT/'src/arggnn/american_c3_evaluate_v1.py','native_top1_seal')
        successor=function(DIRECTORY/'native_seal_v1.py','native_top1_seal')
        successor.args.args.pop()
        original_nested=next(row for row in original.body if isinstance(row,ast.FunctionDef) and row.name=='embedding')
        successor_nested=next(row for row in successor.body if isinstance(row,ast.FunctionDef) and row.name=='embedding')
        self.assertEqual(ast.unparse(successor_nested.body[0]),'out = load_embedding(record)')
        successor_nested.body[0]=original_nested.body[0]
        self.assertEqual(ast.dump(original),ast.dump(successor))

    def test_all_new_executable_wrappers_are_plaintext_without_dynamic_code_loaders(self):
        forbidden={'exec','eval','compile','FunctionType','run_path','run_module','exec_module'}
        for relative in route.SOURCE_FILES:
            tree=ast.parse((ROOT/relative).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,'attr',None)
                    self.assertNotIn(name,forbidden,(relative,name))
                    for item in node.keywords:
                        self.assertFalse(item.arg=='shell' and isinstance(item.value,ast.Constant) and item.value.value is True)
        native=(DIRECTORY/'native_worker_v1.py').read_text()
        self.assertIn('validate_worker_lease(first_cuda=True)',native)
        self.assertIn('resource.setrlimit(resource.RLIMIT_FSIZE',native)
        self.assertNotIn('efficient_downstream_v1',native)

    def test_remaining_RAM_growth_does_not_subtract_existing_resident_twice(self):
        self.assertEqual(route.remaining_growth(100,dict(complete=True,enforcement_PSS_bytes=80)),20)
        self.assertEqual(route.remaining_growth(100,dict(complete=False,enforcement_PSS_bytes=80)),100)
        self.assertEqual(route.remaining_growth(100,dict(complete=True,enforcement_PSS_bytes=120)),0)

    def test_fresh_canonical_lease_and_logical_reservation_share_lifetime(self):
        with tempfile.TemporaryDirectory(dir=DIRECTORY) as temporary:
            root=Path(temporary)
            request=dict(borrowed_lease=None,task=dict(id='synthetic_native'))
            reference=dict(path='request.json',sha256='c'*64)
            with patch.object(gpu.c,'ROOT',root),patch.object(gpu,'gpu_sample',return_value=dict(free_bytes=56*2**30)):
                with gpu.reserve_native(reference,request) as (descriptor,sample):
                    row=gpu.validate_reservation(request,descriptor,os.getpid())
                    physical=row['owned_physical_lease_fd']
                    self.assertIsInstance(physical,int)
                    self.assertFalse(row['borrowed_physical_lease'])
                    self.assertEqual(json.loads((root/'.runtime/locks/gpu_0.lock').read_text())['job'],'synthetic_native')
                    probe=os.open(root/'.runtime/locks/gpu_0.lock',os.O_RDWR)
                    try:
                        import fcntl
                        with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    finally:os.close(probe)
                with self.assertRaises(OSError):os.fstat(physical)
                with self.assertRaises(OSError):os.fstat(descriptor)

    def test_borrow_validation_failure_does_not_take_physical_lease(self):
        with tempfile.TemporaryDirectory(dir=DIRECTORY) as temporary:
            root=Path(temporary)
            request=dict(borrowed_lease={'synthetic':True},task=dict(id='synthetic_native'))
            with patch.object(gpu.c,'ROOT',root),patch.object(gpu,'validate_borrow',side_effect=RuntimeError('old owner gone')),patch.object(gpu,'gpu_sample') as sample:
                with self.assertRaisesRegex(RuntimeError,'owner gone'):
                    with gpu.reserve_native({},request):self.fail('Invalid borrowed owner admitted')
                sample.assert_not_called()
                self.assertFalse((root/'.runtime/locks/gpu_0.lock').exists())

    def _supervisor(self,low=False):
        temporary=tempfile.TemporaryDirectory(dir=DIRECTORY)
        self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name)
        output=directory/'output'
        task=dict(id='synthetic_contrastive',output_directory=str(output),max_seconds=60,
            max_rss_bytes=2**30,max_output_bytes=2**20,max_cuda_reserved_bytes=74*2**30)
        spec=dict(stage='contrastive',parent_id='synthetic_parent',expected_updates=2,
                  output_directory=str(output/'native'))
        reference=dict(path='request.json',sha256='a'*64)
        request=dict(task=task,call=dict(spec=spec,module='arggnn.benchmarking.american_competitors',function='run_argformer'),
            plan_ref=dict(path='plan.json',sha256='b'*64),source_refs=[],borrowed_lease={})
        event=[]
        class Process:
            pid=777777
            def __init__(self,command,**kwargs):
                self.polls=0;self.returncode=None
                self.barrier=os.dup(int(kwargs['env']['ARGGNN_SHARED_STARTUP_FD']))
                event.append(('command',command,kwargs))
                (output/'native/scientific_completed.json').write_text(json.dumps(dict(
                    actual_optimizer_updates=2,optimizer_unique_step_values=[2],completed_epochs=1,parent_id='synthetic_parent')))
                (output/'native/first_full_batch_memory.json').write_text('{}\n')
            def poll(self):
                self.polls+=1
                if self.polls>(4 if low else 1):self.returncode=0
                return self.returncode
            def wait(self):
                os.close(self.barrier);self.returncode=0
                return 0
        class Tracker:
            def __init__(self,pid):self.root=dict(pid=pid,start_ticks=1);self.direct_wait=None;self.stopped=False
            def measure(self):return 1024
            def capture(self):return {} if self.stopped else {777777:self.root}
            def stop_and_wait(self,process):
                event.append(('owned_cleanup',process.pid));self.stopped=True
                self.direct_wait=dict(actual_returncode=process.wait())
            def receipt(self):return dict(synthetic=True,direct_wait=self.direct_wait)
        @contextmanager
        def reservation(*args):
            with (directory/'logical.lock').open('w+') as handle:
                handle.write('{}');handle.flush()
                yield handle.fileno(),dict(free_bytes=56*2**30)
        with patch.object(route.c,'ROOT',directory),patch.object(route,'validate_request',return_value=request), \
             patch.object(route,'capacity_admission',return_value={}),patch.object(route.lifecycle,'enable_subreaper'), \
             patch.object(gpu,'reserve_native',reservation),patch.object(gpu,'authority',return_value=request), \
             patch.object(gpu,'validate_borrow'),patch.object(gpu,'validate_reservation'),patch.object(gpu,'gpu_sample',return_value=dict(free_bytes=(7 if low else 56)*2**30)), \
             patch.object(gpu,'native_driver_bytes',return_value=2**30),patch.object(route,'OwnedGroup',Tracker), \
             patch.object(route.subprocess,'Popen',side_effect=Process),patch.object(route.lifecycle,'scope_bytes',return_value=1024), \
             patch.object(route.signal,'signal',return_value=None),patch.object(route.time,'sleep'):
            if low:
                with self.assertRaisesRegex(RuntimeError,'below8GiB'):route.supervise(reference)
            else:route.supervise(reference)
        return output,event

    def test_supervisor_fixed_device_bounded_cap_and_truthful_handoff(self):
        output,event=self._supervisor()
        command,kwargs=event[0][1:]
        self.assertEqual(command[4],route.WORKER)
        self.assertNotIn('-c',command)
        self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],gpu.GPU0_UUID)
        self.assertEqual(kwargs['env']['ARGGNN_MAX_CUDA_RESERVED_BYTES'],str(40*2**30))
        self.assertNotIn('ARGGNN_GPU_LEASE_FD',kwargs['env'])
        self.assertEqual(len(kwargs['pass_fds']),2)
        wait=json.loads((output/'actual_wait.json').read_text())
        self.assertTrue(wait['actual_direct_wait'] and wait['owned_process_group_absent'])
        self.assertEqual(wait['actual_direct_returncode'],0)
        self.assertTrue((output/'successor_handoff.json').is_file())
        self.assertTrue((output/'result.json').is_file())

    def test_sustained_headroom_breach_cleans_only_actual_owned_native_process(self):
        output,event=self._supervisor(low=True)
        self.assertEqual([row for row in event if row[0]=='owned_cleanup'],[('owned_cleanup',777777)])
        self.assertFalse((output/'result.json').exists())
        wait=json.loads((output/'actual_wait.json').read_text())
        self.assertTrue(wait['owned_process_group_absent'])
        self.assertIn('below8GiB',wait['supervisor_error'])


if __name__=='__main__':unittest.main()
