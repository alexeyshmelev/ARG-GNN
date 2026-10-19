"""Compiled v9 routing with fake CUDA/backend/Popen and temporary lease FD only."""
from contextlib import contextmanager,ExitStack
import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arggnn.benchmarking import deferred_gpu_v9 as deferred
from arggnn.benchmarking import workflow_v9 as runtime


class CompiledDeferredIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.directory=Path(self.temporary.name);self.output=self.directory/"output";self.output.mkdir()
        (self.directory/".runtime/locks").mkdir(parents=True)
        self.addCleanup(self.temporary.cleanup)
        self.names=("MODULE","current_policy","_own_group_rss","execute","child","run","load_plan",
                    "external_result","external_resources","storage_projection","available_ram_bytes",
                    "ProgressReporter","scope_bytes")
        saved={name:getattr(runtime.base,name) for name in self.names}
        self.addCleanup(lambda:[setattr(runtime.base,name,value) for name,value in saved.items()])
        self.task=dict(id="compiled_deferred_fixture",resource="cpu",deferred_gpu0=True,
            factory_module="arggnn.benchmarking.european_preparation_v9",factory_function="fit_factory",
            minimum_free_gpu_bytes=74*2**30,max_cuda_reserved_bytes=74*2**30,
            max_rss_bytes=2**30,max_output_bytes=2**20,max_seconds=60,output_directory=str(self.output))
        self.plan_ref=dict(path="mock/plan",sha256="a"*64)
        self.job=dict(task=self.task,call=dict(module="mock_deferred_backend",function="run_fit",spec={},
            python=str(Path(sys.prefix)/"bin/python")),plan_ref=self.plan_ref)
        jobpath=self.directory/"job.json";jobpath.write_text(json.dumps(self.job)+"\n")
        self.ref=runtime.c.file_ref(jobpath)
        self.events=[];self.initialized=False;self.reserved=1024;self.opens=0;self.closes=0
        self.fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:self.initialized,
                                                             memory_reserved=lambda index:self.reserved))
        self.titles=[];self.mode="normal";self.clock=100
        self.original_phase=deferred.DeferredGPU0
        self.lifetime_before=len(deferred._PROCESS_LIFETIME_LEASES)
        self.addCleanup(self.release_leases)

    def release_leases(self):
        self.initialized=False
        while len(deferred._PROCESS_LIFETIME_LEASES)>self.lifetime_before:
            deferred._PROCESS_LIFETIME_LEASES.pop().close()

    @contextmanager
    def lease(self,spec):
        self.opens+=1;self.events.append("lease_open")
        fd=os.open(self.directory/".runtime/locks/gpu_0.lock",os.O_CREAT|os.O_RDWR,0o600)
        os.ftruncate(fd,0)
        os.write(fd,json.dumps(dict(launcher_pid=os.getpid(),job=spec["job_id"],uuid=deferred.GPU0_UUID)).encode())
        try:yield fd
        finally:os.close(fd);self.closes+=1

    def phase(self,task,output,check):
        return self.original_phase(task,output,check,lease_factory=self.lease,
                                   sleep=lambda seconds:self.fail("No real contention in CPU fixture"))

    def monotonic(self):
        self.clock+=6
        return self.clock

    def backend(self,spec,*,resource_check,acquire_gpu0):
        self.events.append("backend_CPU")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"],deferred.GPU0_UUID)
        self.assertEqual(os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"],"1")
        self.assertNotIn("ARGGNN_GPU_LEASE_FD",os.environ)
        self.assertNotIn("ARGGNN_PHYSICAL_GPU",os.environ)
        self.assertEqual(self.opens,0)
        self.assertEqual(self.titles,[deferred.TITLE])
        self.assertEqual(json.loads((self.output/"gpu_phase.json").read_text())["phase"],"cpu_preparation")
        if self.mode=="early_CUDA":
            self.initialized=True
            resource_check("early_context_fixture")
            self.fail("Compiled child did not reject early CUDA")
        resource_check("CPU_fixture")
        self.events.append("barrier_fixture")
        acquire_gpu0()
        self.events.append("backend_GPU")
        self.assertEqual(self.opens,1)
        self.assertEqual(os.environ["ARGGNN_PHYSICAL_GPU"],"0")
        os.fstat(int(os.environ["ARGGNN_GPU_LEASE_FD"]))
        self.initialized=True
        if self.mode=="excess":self.reserved=self.task["max_cuda_reserved_bytes"]+1
        resource_check("GPU_fixture")
        return dict(synthetic=True,complete=True)

    def child(self,*,missing_title=False):
        runtime.install_bindings()
        original_import=builtins.__import__
        def importing(name,*args,**kwargs):
            if missing_title and name=="setproctitle":raise ImportError("synthetic missing title package")
            return original_import(name,*args,**kwargs)
        actual_import=runtime.base.importlib.import_module
        def module(name,*args,**kwargs):
            if name=="mock_deferred_backend":return SimpleNamespace(run_fit=self.backend)
            return actual_import(name,*args,**kwargs)
        def policy():self.events.append("current_installed_policy");return {}
        def memory(pgid):self.events.append("current_installed_PSS");return 0
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,CUDA_VISIBLE_DEVICES=deferred.GPU0_UUID,
                PYTORCH_NVML_BASED_CUDA_CHECK="1",ARGGNN_WORKFLOW_DEADLINE="1000"))
            for key in ("ARGGNN_GPU_LEASE_FD","ARGGNN_PHYSICAL_GPU","ARGGNN_MAX_CUDA_RESERVED_BYTES"):
                os.environ.pop(key,None)
            stack.enter_context(patch.dict(sys.modules,torch=self.fake_torch,
                setproctitle=SimpleNamespace(setproctitle=self.titles.append)))
            stack.enter_context(patch.object(builtins,"__import__",side_effect=importing))
            stack.enter_context(patch.object(deferred,"c",SimpleNamespace(ROOT=self.directory)))
            stack.enter_context(patch.object(deferred,"DeferredGPU0",side_effect=self.phase))
            stack.enter_context(patch.object(runtime.base,"job_spec",return_value=self.job))
            stack.enter_context(patch.object(runtime.base,"current_policy",side_effect=policy))
            stack.enter_context(patch.object(runtime.base,"_own_group_rss",side_effect=memory))
            stack.enter_context(patch.object(runtime.base,"scope_bytes",return_value=0))
            stack.enter_context(patch.object(runtime.base.resource,"setrlimit"))
            stack.enter_context(patch.object(runtime.base.time,"monotonic",side_effect=self.monotonic))
            stack.enter_context(patch.object(runtime.base.importlib,"import_module",side_effect=module))
            return runtime.child(self.ref)

    def test_compiled_child_uses_current_callbacks_and_late_lease_with_fixed_visibility(self):
        result_ref=self.child()
        result=runtime.c.read_ref(result_ref)
        self.assertEqual(result["value"],dict(synthetic=True,complete=True))
        self.assertEqual(result["job_ref"],self.ref)
        self.assertLess(self.events.index("barrier_fixture"),self.events.index("lease_open"))
        self.assertIn("current_installed_policy",self.events)
        self.assertIn("current_installed_PSS",self.events)
        self.assertEqual(self.titles,[deferred.TITLE,deferred.TITLE])
        self.assertEqual(self.closes,0)
        self.assertEqual(len(deferred._PROCESS_LIFETIME_LEASES),self.lifetime_before+1)
        self.assertEqual(json.loads((self.output/"gpu_phase.json").read_text())["phase"],"worker_finishing")

    def test_compiled_child_rejects_context_before_lease(self):
        self.mode="early_CUDA"
        with self.assertRaisesRegex(ValueError,"before its lease"):
            self.child()
        self.assertEqual(self.opens,0)
        self.assertFalse((self.output/"result.json").exists())

    def test_compiled_child_enforces_GPU_allocator_cap_after_late_acquisition(self):
        self.mode="excess"
        with self.assertRaisesRegex(ValueError,"CUDA allocation cap exceeded"):
            self.child()
        self.assertEqual(self.opens,1)
        self.assertEqual(self.closes,0)
        self.assertFalse((self.output/"result.json").exists())

    def test_deferred_child_requires_title_package_before_backend_import(self):
        with self.assertRaisesRegex(ValueError,"title package"):
            self.child(missing_title=True)
        self.assertNotIn("backend_CPU",self.events)
        self.assertEqual(self.opens,0)

    def test_compiled_supervisor_sets_fixed_GPU0_env_and_takes_no_preparation_lease(self):
        self.output.rmdir()
        events=self.events;output=self.output;ref=self.ref
        test=self
        class FakeProcess:
            pid=777777
            returncode=None
            def __init__(self,argv,**kwargs):
                test.assertEqual(argv[4],runtime.MODULE)
                env=kwargs["env"]
                test.assertEqual(env["CUDA_VISIBLE_DEVICES"],deferred.GPU0_UUID)
                test.assertEqual(env["PYTORCH_NVML_BASED_CUDA_CHECK"],"1")
                test.assertNotIn("ARGGNN_GPU_LEASE_FD",env)
                test.assertNotIn("ARGGNN_PHYSICAL_GPU",env)
                test.assertIn("ARGGNN_V4_STARTUP_FD",env)
                self.barrier=os.dup(int(env["ARGGNN_V4_STARTUP_FD"]))
                test.assertEqual(kwargs["pass_fds"],(int(env["ARGGNN_V4_STARTUP_FD"]),))
                self.polls=0
                (output/"result.json").write_text(json.dumps(dict(value={"synthetic":True},job_ref=ref))+"\n")
            def poll(self):
                self.polls+=1
                if self.polls>1:self.returncode=0
                return self.returncode
            def wait(self,**kwargs):
                self.returncode=0
                if self.barrier is not None:os.close(self.barrier);self.barrier=None
                return 0
        class FakeTracker:
            def __init__(self,pid):
                self.root=dict(pid=pid,start_ticks=1,pgid=pid,session=pid,ppid=os.getpid(),state="S")
                self.direct_wait=None
            def measure(self):events.append("current_owned_PSS");return 0
            def stop_and_wait(self,process):
                self.direct_wait=dict(actual_Popen_wait=True,actual_returncode=process.wait())
            def receipt(self):return dict(root_identity=self.root,direct_Popen_wait=self.direct_wait)
        with patch.object(runtime.base,"job_spec",return_value=self.job), \
                patch.object(runtime._core,"_BASE_POLICY",side_effect=lambda:events.append("current_policy") or
                             dict(approved_resource_continuation_ref=self.plan_ref)), \
                patch.object(runtime._core,"OwnedGroup",FakeTracker), \
                patch.object(runtime._core,"enable_subreaper",return_value=True), \
                patch.object(runtime._core.subprocess,"Popen",side_effect=FakeProcess), \
                patch.object(runtime.base,"gpu0_lease",side_effect=AssertionError("CPU phase acquired lease")) as lease, \
                patch.object(runtime.base.os,"killpg",side_effect=ProcessLookupError), \
                patch.object(runtime.base.time,"sleep"):
            result=runtime.execute(self.ref)
        self.assertTrue(result["actual_direct_wait"])
        self.assertEqual(result["actual_direct_returncode"],0)
        self.assertTrue(result["owned_process_group_absent"])
        self.assertIn("current_owned_PSS",events)
        self.assertIn("current_policy",events)
        lease.assert_not_called()
        self.assertEqual(json.loads((output/"supervision_identity.json").read_text())["job_ref"],self.ref)


if __name__=="__main__":unittest.main()
