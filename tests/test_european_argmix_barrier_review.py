"""Independent all-endpoint barrier checks; synthetic metadata/probabilities only."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from arggnn.benchmarking import european_argmix_release as release
from arggnn.benchmarking import european_argmix_workflow as workflow


class AllEndpointBarrierReviewTest(unittest.TestCase):
    def setUp(self):
        self.parents=["test"+str(i) for i in range(5)]
        self.plan=dict(test_parent_ids=self.parents)
        self.values={"plan":self.plan};self.arrays={};self.reads=[];self.loads=[]
        self.groups=[]
        for endpoint,classes,methods in (
            ("AP6",release.sampling.NATIVE_PATHS,("ancestralpaths","treeloom")),
            ("C4",release.CLASSES,("argmix","treeloom"))):
            group={}
            for method in methods:
                group[method]={}
                for parent in self.parents:
                    ref=endpoint+"/"+method+"/"+parent
                    probabilities=ref+"/probabilities"
                    group[method][parent]=ref
                    value=dict(parent_id=parent,split="test",rows=10000,
                        registry_ref=endpoint+"/"+parent+"/registry",
                        predictions_ref=probabilities)
                    value["native_path_names" if endpoint=="AP6" or method=="treeloom" else "source_class_order"]=list(classes)
                    self.values[ref]=value
                    self.arrays[probabilities]=dict(example_indices=np.arange(10000),
                        probabilities=np.full((10000,len(classes)),1./len(classes),dtype=np.float32))
            self.groups.append(group)
        self.native6,self.c4=self.groups

    def read(self,ref,**kwargs):
        self.reads.append(ref)
        self.assertTrue(kwargs.get("observed"))
        return self.values[ref]

    def load(self,ref,**kwargs):
        self.loads.append(ref)
        self.assertTrue(kwargs.get("observed"))
        return self.arrays[ref]

    def test_all20_products_are_validated_without_any_target_capability(self):
        with patch.object(release.m,"_read",side_effect=self.read),patch.object(release.m,"_arrays",side_effect=self.load):
            release.verify_cross_endpoint_barrier(self.plan,self.native6,self.c4)
        self.assertEqual(len(self.reads),20);self.assertEqual(len(self.loads),20)
        self.assertEqual(set(self.reads),set(self.values)-{"plan"})

    def test_last_malformed_C4_product_blocks_C4_truth_and_native6_scorer(self):
        last=self.c4["treeloom"][self.parents[-1]]
        self.arrays[last+"/probabilities"]["probabilities"][0]=0
        with patch.object(release.m,"_read",side_effect=self.read), \
             patch.object(release.m,"_arrays",side_effect=self.load), \
             patch.object(release,"_truth",side_effect=AssertionError("private targets must remain closed")):
            with self.assertRaisesRegex(ValueError,"sealed canonical"):
                release.score_c4("plan",self.c4,self.native6,{}, "never-created",
                    guard=lambda _:None,execution_check=lambda:None,write_headroom_bytes=1,
                    max_truth_uncompressed_bytes=1)
        self.assertEqual(len(self.loads),20)
        spec=dict(stage="release_native6_score",output_directory="never-created",
            arguments=dict(plan_ref="plan",predictions=self.native6,c4_predictions=self.c4,
                           test_targets={},write_headroom_bytes=1,max_truth_uncompressed_bytes=1))
        with patch.object(release.m,"_read",side_effect=self.read), \
             patch.object(release.m,"_arrays",side_effect=self.load), \
             patch.object(release.m,"score_native_comparison",side_effect=AssertionError("native scorer cannot reach targets")):
            with self.assertRaisesRegex(ValueError,"sealed canonical"):
                workflow.execute_stage(spec,resource_check=lambda _:None,execution_check=lambda:None)

    def test_mismatched_final_C4_registry_is_rejected(self):
        last=self.c4["treeloom"][self.parents[-1]]
        self.values[last]["registry_ref"]="different_eligible_support"
        with patch.object(release.m,"_read",side_effect=self.read),patch.object(release.m,"_arrays",side_effect=self.load):
            with self.assertRaisesRegex(ValueError,"identical canonical"):
                release.verify_cross_endpoint_barrier(self.plan,self.native6,self.c4)
        self.assertEqual(len(self.loads),19)


class NativeCudaStartupReviewTest(unittest.TestCase):
    def fake_cuda(self, events, total=80*2**30):
        def count():
            events.append("count")
            return 1
        def properties(device):
            self.assertEqual(device,0)
            events.append("properties")
            return SimpleNamespace(total_memory=total)
        def fraction(value,device):
            events.append(("fraction",value,device))
        return SimpleNamespace(device_count=count,get_device_properties=properties,
                               set_per_process_memory_fraction=fraction)

    def test_admission_sets_positive_bounded_fraction_before_model_allocation(self):
        events=[]
        torch=SimpleNamespace(cuda=self.fake_cuda(events))
        def title():
            events.append("title_checked")
            return release.TITLE
        setproctitle=SimpleNamespace(getproctitle=title)
        environment=dict(CUDA_VISIBLE_DEVICES=release.GPU0_UUID,
            ARGGNN_PHYSICAL_GPU="0",ARGGNN_MAX_CUDA_RESERVED_BYTES=str(64*2**30))
        with patch.dict(sys.modules,{"setproctitle":setproctitle}),patch.dict(os.environ,environment), \
             patch.object(release.native,"runtime_admission"), \
             patch.object(release.native,"source_imports",return_value=(torch,)):
            self.assertEqual(release.admit_native({}, {},require_gpu=True),(torch,))
        self.assertEqual(events,["title_checked","count","properties",("fraction",.8,0)])
        for cap in ("0","-1",str(75*2**30)):
            events.clear()
            with patch.dict(sys.modules,{"setproctitle":setproctitle}), \
                 patch.dict(os.environ,dict(environment,ARGGNN_MAX_CUDA_RESERVED_BYTES=cap)), \
                 patch.object(release.native,"runtime_admission"), \
                 patch.object(release.native,"source_imports",return_value=(torch,)):
                with self.assertRaisesRegex(ValueError,"allocator cap"):
                    release.admit_native({}, {},require_gpu=True)
            self.assertFalse(any(isinstance(e,tuple) and e[0]=="fraction" for e in events))

    def run_hook(self, script, device, *, environment=None,total=80*2**30):
        events=[]
        setproctitle=SimpleNamespace(setproctitle=lambda value:events.append(("title",value)))
        torch=SimpleNamespace(cuda=self.fake_cuda(events,total))
        env=dict(CUDA_VISIBLE_DEVICES=release.GPU0_UUID,
            ARGGNN_PHYSICAL_GPU="0",ARGGNN_MAX_CUDA_RESERVED_BYTES=str(64*2**30))
        env.update(environment or {})
        argv=[str(release.native.SOURCE/"src"/script),"--device",device]
        with patch.dict(sys.modules,{"setproctitle":setproctitle,"torch":torch}), \
             patch.dict(os.environ,env),patch.object(sys,"argv",argv), \
             patch("os._exit",side_effect=SystemExit(78)),patch("traceback.print_exc"):
            try:
                exec(compile(release._child_startup_source(),"<native-sitecustomize>","exec"),{})
            except SystemExit as error:
                events.append(("exit",error.code))
        self.assertEqual(argv,[str(release.native.SOURCE/"src"/script),"--device",device])
        return events

    def test_original_GPU_child_argv_gets_title_then_cap(self):
        for script in ("run_inference.py","run_joint_inference.py"):
            self.assertEqual(self.run_hook(script,"cuda"),[
                ("title",release.TITLE),"count","properties",("fraction",.8,0)])

    def test_explicit_CPU_export_makes_no_CUDA_calls(self):
        self.assertEqual(self.run_hook("export_torchscript.py","cpu"),[("title",release.TITLE)])

    def test_invalid_child_admission_exits_before_original_source(self):
        for env in (dict(CUDA_VISIBLE_DEVICES="GPU-other"),dict(ARGGNN_PHYSICAL_GPU="1"),
                    dict(ARGGNN_MAX_CUDA_RESERVED_BYTES="0"),
                    dict(ARGGNN_MAX_CUDA_RESERVED_BYTES=str(75*2**30))):
            self.assertEqual(self.run_hook("run_inference.py","cuda",environment=env),
                             [("title",release.TITLE),("exit",78)])
        events=self.run_hook("run_inference.py","cuda",total=32*2**30)
        self.assertEqual(events,[("title",release.TITLE),"count","properties",("exit",78)])

    def test_stage_local_hook_hash_and_parent_environment_restored_on_error(self):
        with tempfile.TemporaryDirectory(dir=release.m.ROOT/"tests") as temporary:
            base=Path(temporary)
            with patch.dict(os.environ,{"PYTHONPATH":"original-parent"}):
                stage=base/"failure";stage.mkdir()
                with self.assertRaisesRegex(RuntimeError,"mock CLI failed"):
                    with release._native_python_startup(stage,lambda _:None) as ref:
                        self.assertEqual(release.m._ref(stage/"python_startup/sitecustomize.py"),ref)
                        self.assertEqual(os.environ["PYTHONPATH"],
                            str(stage/"python_startup")+os.pathsep+"original-parent")
                        raise RuntimeError("mock CLI failed")
                self.assertEqual(os.environ["PYTHONPATH"],"original-parent")
            with patch.dict(os.environ):
                os.environ.pop("PYTHONPATH",None)
                stage=base/"success";stage.mkdir()
                with release._native_python_startup(stage,lambda _:None):
                    self.assertEqual(os.environ["PYTHONPATH"],str(stage/"python_startup"))
                self.assertNotIn("PYTHONPATH",os.environ)


class NativeOwnedRunnerReviewTest(unittest.TestCase):
    def module(self):
        namespace={"_run_local":lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("escaping stock runner called"))}
        exec("def cmd_infer(command):\n    _run_local(command)\n    return 0\n",namespace)
        return SimpleNamespace(__file__=str(release.native.SOURCE/"src/commands/infer.py"),
                               cmd_infer=namespace["cmd_infer"])

    def test_private_same_code_and_argv_use_inherited_group_runner(self):
        import ast
        import inspect
        from arggnn.benchmarking import workflow as owned
        module=self.module();original=module.cmd_infer;calls=[]
        command=["original-python","-u","original-native-script.py","--device","cuda"]
        wait=dict(actual_direct_wait=True,actual_direct_returncode=0,
                  native_process_absent=True,outer_owned_group_cleanup_pending=True)
        def runner(argv,**kwargs):
            self.assertIs(argv,command)
            self.assertEqual(kwargs["cwd"],release.native.SOURCE)
            self.assertEqual(kwargs["env"],{})
            self.assertEqual(kwargs["limits"],{})
            calls.append(kwargs)
            return wait
        with patch.object(release.importlib,"import_module",return_value=module):
            with release._native_local_runner(Path("/mock/current-stage"),runner,lambda:None) as waits:
                self.assertIsNot(module.cmd_infer,original)
                self.assertIs(module.cmd_infer.__code__,original.__code__)
                self.assertEqual(module.cmd_infer(command),0)
        self.assertIs(module.cmd_infer,original)
        self.assertEqual(waits[0]["argv"],command)
        self.assertEqual(waits[0]["actual_wait"],wait)
        self.assertEqual(len(calls),1)
        # ROOT's existing injected runner launches within its inherited group.
        tree=ast.parse(inspect.getsource(owned.child))
        nested=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="bounded_runner")
        starts=[kw.value for n in ast.walk(nested) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=="Popen"
                for kw in n.keywords if kw.arg=="start_new_session"]
        self.assertFalse(any(not isinstance(v,ast.Constant) or v.value is not False for v in starts))

    def test_failed_native_wait_restores_original_binding(self):
        module=self.module();original=module.cmd_infer
        wait=dict(actual_direct_wait=True,actual_direct_returncode=1,
                  native_process_absent=True,outer_owned_group_cleanup_pending=True)
        with patch.object(release.importlib,"import_module",return_value=module):
            with self.assertRaisesRegex(ValueError,"successful inherited-group"):
                with release._native_local_runner(Path("/mock/current-stage"),lambda *a,**k:wait,lambda:None):
                    module.cmd_infer(["native-python","native-script"])
        self.assertIs(module.cmd_infer,original)

    def test_workflow_injects_existing_runner_into_release_inference(self):
        runner=lambda *a,**k:None
        spec=dict(stage="release_infer",arguments=dict(write_headroom_bytes=1),output_directory="/mock/current-stage")
        with patch.object(release,"infer_release",return_value="mock-completion") as infer:
            result=workflow.execute_stage(spec,resource_check=lambda _:None,
                execution_check=lambda:None,bounded_runner=runner)
        self.assertEqual(result,"mock-completion")
        self.assertIs(infer.call_args.kwargs["bounded_runner"],runner)


if __name__=="__main__":unittest.main()
