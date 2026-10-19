import contextlib
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from arggnn.benchmarking import deferred_gpu_v9 as d


class DeferredTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=d.c.ROOT / '.runtime')
        self.root = Path(self.temp.name)
        (self.root / '.runtime/locks').mkdir(parents=True)
        self.task = dict(id='deferred_test', resource='cpu', deferred_gpu0=True,
            factory_module='arggnn.benchmarking.european_preparation_v9',
            factory_function='fit_factory', minimum_free_gpu_bytes=74*2**30,
            max_cuda_reserved_bytes=74*2**30)
        self.initialized = False
        self.titles = []
        self.calls = []
        self.opens = 0
        self.closed = 0
        self.fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_initialized=lambda: self.initialized))
        self.patches = [patch.object(d.c, 'ROOT', self.root),
            patch.dict('sys.modules', torch=self.fake_torch,
                       setproctitle=types.SimpleNamespace(setproctitle=self.titles.append)),
            patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':d.GPU0_UUID})]
        for p in self.patches:p.start()
        for k in ['ARGGNN_GPU_LEASE_FD','ARGGNN_PHYSICAL_GPU','ARGGNN_MAX_CUDA_RESERVED_BYTES']:
            os.environ.pop(k,None)

    def tearDown(self):
        self.initialized = False
        while d._PROCESS_LIFETIME_LEASES:d._PROCESS_LIFETIME_LEASES.pop().close()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    @contextlib.contextmanager
    def lease(self,spec):
        self.opens += 1
        fd=os.open(self.root/'.runtime/locks/gpu_0.lock',os.O_CREAT|os.O_RDWR,0o600)
        os.ftruncate(fd,0)
        os.write(fd,json.dumps(dict(launcher_pid=os.getpid(),job=spec['job_id'],uuid=d.GPU0_UUID)).encode())
        try:yield fd
        finally:os.close(fd);self.closed+=1

    def phase(self, **kwargs):
        return d.DeferredGPU0(self.task,self.root,self.calls.append,lease_factory=self.lease,**kwargs)

    def test_no_lease_during_preparation_then_exact_once(self):
        with self.phase() as phase:
            self.assertEqual(self.opens,0)
            self.assertNotIn('ARGGNN_GPU_LEASE_FD',os.environ)
            fd=phase.acquire()
            self.assertEqual(self.opens,1)
            self.assertEqual(os.environ['ARGGNN_PHYSICAL_GPU'],'0')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],d.GPU0_UUID)
            self.assertEqual(self.titles,[d.TITLE])
            self.assertEqual(int(os.environ['ARGGNN_GPU_LEASE_FD']),fd)
            with self.assertRaises(RuntimeError):phase.acquire()
        self.assertEqual(self.closed,1)

    def test_early_cuda_context_is_rejected(self):
        with self.phase() as phase:
            self.initialized=True
            with self.assertRaisesRegex(RuntimeError,'before a PyTorch CUDA'):
                phase.acquire()
        self.assertEqual(self.opens,0)

    def test_contention_preserves_process_and_checks_limits(self):
        waits=[]
        tries=[0]
        @contextlib.contextmanager
        def contend(spec):
            tries[0]+=1
            if tries[0]==1:raise BlockingIOError()
            with self.lease(spec) as fd:yield fd
        phase=d.DeferredGPU0(self.task,self.root,self.calls.append,lease_factory=contend,sleep=waits.append)
        with phase:phase.acquire()
        self.assertEqual(waits,[2]);self.assertEqual(tries[0],2)
        self.assertEqual(self.calls.count('waiting_for_deferred_GPU0'),2)

    def test_cuda_lease_retained_even_if_final_receipt_fails(self):
        phase=self.phase();phase.__enter__();fd=phase.acquire();self.initialized=True
        with patch.object(phase,'_record',side_effect=OSError('test output failure')):
            with self.assertRaises(OSError):phase.__exit__(None,None,None)
        os.fstat(fd)
        self.assertEqual(self.closed,0)
        self.assertEqual(len(d._PROCESS_LIFETIME_LEASES),1)

    def test_cpu_failure_closes_lease_before_cuda(self):
        with self.assertRaises(ValueError):
            with self.phase() as phase:
                phase.acquire();raise ValueError('test')
        self.assertEqual(self.closed,1)

    def test_incorrect_factory_or_gpu_cap_rejected(self):
        for changed in [dict(factory_module='unreviewed'),dict(max_cuda_reserved_bytes=75*2**30),dict(resource='gpu0')]:
            with self.assertRaises(RuntimeError):d.validate_task(dict(self.task,**changed))


if __name__=='__main__':unittest.main()
