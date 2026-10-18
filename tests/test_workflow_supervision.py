"""Real tiny CPU processes under fake admission; never run numerical science."""
from contextlib import ExitStack
import importlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType
import unittest
from unittest.mock import patch

from arggnn.benchmarking import workflow as w
from arggnn.benchmarking import american_workflow as american

REAL_POPEN=subprocess.Popen
REAL_SLEEP=time.sleep


class WorkflowSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="workflow_supervision_",dir=w.c.ROOT/"experiments/results")
        self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.ref=dict(path="synthetic/no_live_admission.json",sha256="a"*64)
        self.processes=[]
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)

    def task(self,name,*,deps=(),seconds=10,cap=2**20,args=None):
        return dict(id=name,resource="cpu",dependencies=list(deps),
            factory_module="arggnn.benchmarking.synthetic_supervision_fixture",factory_function="factory",
            factory_arguments=args or {},output_directory=str(self.root/name),max_seconds=seconds,
            max_output_bytes=cap,max_rss_bytes=512*2**20)

    def job(self,task):
        return dict(schema="arggnn_workflow_job_v1",plan_ref=self.ref,task=task,source_refs=[],
                    call=dict(module="arggnn.benchmarking.synthetic_supervision_fixture",function="run",
                              python=str(Path(sys.prefix)/"bin/python"),spec={}))

    def tiny_script(self,job,*,exitcode=0,long=False,excess=False):
        out=job["task"]["output_directory"]
        upstream=job["call"]["spec"].get("upstream")
        code="import json, pathlib, time, sys\n"
        code+="out=pathlib.Path("+repr(out)+")\n"
        code+="(out/'ready').write_text('tiny CPU fixture')\n"
        if excess:code+="(out/'excess.bin').write_bytes(b'x'*(2**20))\n"
        if long:code+="time.sleep(30)\n"
        if exitcode:code+="sys.exit("+str(exitcode)+")\n"
        code+="(out/'native').mkdir(exist_ok=True)\n"
        code+="product=out/'native/product.json'\n"
        code+="product.write_text(json.dumps(dict(synthetic_test_only=True,stage="+repr(job["task"]["id"])+",upstream="+repr(upstream)+")))\n"
        code+="from arggnn.benchmarking import contracts as c\n"
        code+="(out/'result.json').write_text(json.dumps(dict(value=c.file_ref(product),job_ref="+repr(self.ref)+")))\n"
        return code

    def execute_tiny(self,task,*,exitcode=0,long=False,excess=False,release=None):
        job=self.job(task);script=self.tiny_script(job,exitcode=exitcode,long=long,excess=excess)
        def popen(argv,**kwargs):
            self.assertIn("--child",argv)
            p=REAL_POPEN([sys.executable,"-B","-s","-c",script],**kwargs)
            self.processes.append(p);return p
        policy=dict(approved_workflow_ref=self.ref)
        with patch.object(w,"job_spec",return_value=job),patch.object(w.subprocess,"Popen",side_effect=popen), \
             patch.object(w.c,"check_execution_release",side_effect=release or (lambda _:policy)), \
             patch.object(w.time,"sleep",side_effect=lambda n:REAL_SLEEP(min(n,.01))):
            return w.execute(self.ref)

    def test_real_worker_wait0_and_exit7_are_recorded_without_science(self):
        task=self.task("success")
        result=self.execute_tiny(task)
        self.assertTrue(result["actual_direct_wait"])
        self.assertEqual(result["actual_direct_returncode"],0)
        self.assertTrue(result["owned_process_group_absent"])
        self.assertFalse(result["automatic_retry"])
        with self.assertRaises(ProcessLookupError):os.killpg(self.processes[-1].pid,0)
        task=self.task("failure")
        with self.assertRaises(ValueError):self.execute_tiny(task,exitcode=7)
        wait=json.loads((Path(task["output_directory"])/"actual_wait.json").read_text())
        self.assertEqual(wait["actual_direct_returncode"],7)
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertFalse((Path(task["output_directory"])/"result.json").exists())

    def test_output_limit_preserves_partial_output_and_actual_failed_wait(self):
        task=self.task("too_large",cap=65536)
        with self.assertRaises(ValueError):self.execute_tiny(task,excess=True,long=True)
        root=Path(task["output_directory"]);wait=json.loads((root/"actual_wait.json").read_text())
        self.assertIn("output",wait["supervisor_error"].lower())
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertTrue((root/"excess.bin").exists())
        self.assertNotEqual(wait["actual_direct_returncode"],0)

    def test_outer_policy_revocation_stops_busy_child_and_records_actual_wait(self):
        task=self.task("revoked")
        def withdrawn(_):
            if (Path(task["output_directory"])/"ready").exists():
                raise ValueError("synthetic canonical policy withdrawn")
            return dict(approved_workflow_ref=self.ref)
        with self.assertRaisesRegex(ValueError,"synthetic canonical policy withdrawn"):
            self.execute_tiny(task,long=True,release=withdrawn)
        wait=json.loads((Path(task["output_directory"])/"actual_wait.json").read_text())
        self.assertTrue(wait["actual_direct_wait"])
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertIsNotNone(wait["supervisor_error"])

    def test_wall_deadline_stops_busy_child_and_keeps_unknown_result_absent(self):
        task=self.task("deadline",seconds=1)
        with self.assertRaises(ValueError):self.execute_tiny(task,long=True)
        root=Path(task["output_directory"]);wait=json.loads((root/"actual_wait.json").read_text())
        self.assertIn("timed out",wait["supervisor_error"].lower())
        self.assertTrue(wait["actual_direct_wait"])
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertFalse((root/"result.json").exists())

    def test_child_injected_native_runner_records_actual_wait_and_rejects_device_override(self):
        for override in (False,True):
            task=self.task("native_override" if override else "native_wait")
            out=Path(task["output_directory"]);out.mkdir()
            job=self.job(task)
            module=ModuleType(job["call"]["module"])
            def run(spec,*,resource_check,bounded_runner):
                env=dict(CUDA_VISIBLE_DEVICES=w.GPU0_UUID) if override else {}
                return bounded_runner([sys.executable,"-B","-s","-c","print('tiny native CPU fixture')"],
                    str(w.c.ROOT),env,dict(max_seconds=2),str(out/"native.log"))
            module.run=run
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="",ARGGNN_WORKFLOW_DEADLINE=str(time.monotonic()+10))
            for key in ("ARGGNN_PHYSICAL_GPU","ARGGNN_GPU_LEASE_FD","ARGGNN_MAX_CUDA_RESERVED_BYTES"):env.pop(key,None)
            for key in ("TMPDIR","TMP","TEMP","XDG_CACHE_HOME","TORCH_HOME","HF_HOME","MPLCONFIGDIR",
                        "NUMBA_CACHE_DIR","CUDA_CACHE_PATH","TRITON_CACHE_DIR"):
                cache=out/"cache";cache.mkdir(exist_ok=True);env[key]=str(cache)
            with patch.dict(os.environ,env,clear=True),patch.dict(sys.modules,{module.__name__:module}), \
                 patch.object(w,"job_spec",return_value=job),patch.object(w.resource,"setrlimit"), \
                 patch.object(w.c,"check_execution_release",return_value=dict(approved_workflow_ref=self.ref)), \
                 patch.object(w.time,"sleep",side_effect=lambda n:REAL_SLEEP(min(n,.01))):
                if override:
                    with patch.object(w.subprocess,"Popen") as spawn:
                        with self.assertRaisesRegex(ValueError,"override"):w.child(self.ref)
                        spawn.assert_not_called()
                else:
                    result=w.child(self.ref)
                    value=w.c.read_ref(result)["value"]
                    self.assertTrue(value["actual_direct_wait"])
                    self.assertEqual(value["actual_direct_returncode"],0)
                    self.assertTrue(value["native_process_absent"])
                    self.assertTrue(value["outer_owned_group_cleanup_pending"])

    def plan(self,tasks,*,measured=0,ceiling=980000000000):
        baseline=w.save(self.root/"baseline.json",dict(measured_project_bytes=measured))
        return dict(schema=w.SCHEMA,physical_gpu=0,cpu_parallelism=1,
            sources=[w.c.file_ref(w.c.__file__)],tasks=tasks,records_directory=str(self.root/"records"),
            storage_baseline_ref=baseline,metadata_margin_bytes=2**20,external_storage=[],external_resources=[],
            max_parallel_rss_bytes=2**30,host_ram_margin_bytes=0,storage_ceiling_bytes=ceiling)

    def fake_factory_module(self):
        module=ModuleType("arggnn.benchmarking.synthetic_supervision_fixture")
        def factory(*,output_directory,upstream=None):
            return dict(module=module.__name__,function="run",python=str(Path(sys.prefix)/"bin/python"),
                        spec=dict(output_directory=output_directory,upstream=upstream),source_refs=[])
        module.factory=factory
        return module

    def controller_popen(self,*,failure_ids=(),long_ids=()):
        def popen(argv,**kwargs):
            self.assertIn("--execute-job",argv)
            job_ref=json.loads(argv[-1]);job=w.c.read_ref(job_ref)
            script=self.tiny_script(job,exitcode=7 if job["task"]["id"] in failure_ids else 0,
                                    long=job["task"]["id"] in long_ids)
            boot="import json,sys,subprocess,time\nfrom unittest.mock import patch\n"
            boot+="from arggnn.benchmarking import workflow as w\n"
            boot+="job=json.loads("+repr(json.dumps(job))+")\njob_ref=json.loads("+repr(json.dumps(job_ref))+")\n"
            boot+="real=subprocess.Popen; nap=time.sleep\n"
            boot+="def tiny(argv,**kwargs):return real([sys.executable,'-B','-s','-c',"+repr(script)+"],**kwargs)\n"
            boot+="with patch.object(w,'job_spec',return_value=job),patch.object(w.subprocess,'Popen',side_effect=tiny),patch.object(w.c,'check_execution_release',return_value={'approved_workflow_ref':job['plan_ref']}),patch.object(w.time,'sleep',side_effect=lambda n:nap(min(n,.01))):\n"
            boot+=" result=w.execute(job_ref)\n print(json.dumps(result),flush=True)\n"
            kwargs["env"]=dict(os.environ,PYTHONPATH=str(w.c.ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1")
            p=REAL_POPEN([sys.executable,"-B","-s","-c",boot],**kwargs)
            self.processes.append(p);return p
        return popen

    def controller_context(self,plan,*,failure_ids=(),long_ids=(),release=None):
        stack=ExitStack();module=self.fake_factory_module()
        stack.enter_context(patch.dict(sys.modules,{module.__name__:module}))
        stack.enter_context(patch.object(w,"load_plan",return_value=plan))
        stack.enter_context(patch.object(w.c,"check_execution_release",side_effect=release or (lambda _:dict(approved_workflow_ref=self.ref))))
        stack.enter_context(patch.object(w,"available_ram_bytes",return_value=4*2**30))
        stack.enter_context(patch.object(w.subprocess,"Popen",side_effect=self.controller_popen(failure_ids=failure_ids,long_ids=long_ids)))
        stack.enter_context(patch.object(w.time,"sleep",side_effect=lambda n:REAL_SLEEP(min(n,.01))))
        return stack

    def test_controller_consumes_real_waits_then_resolves_dependent_actual_product(self):
        a=self.task("a");b=self.task("b",deps=("a",),args=dict(upstream={"$task":"a","read":True,"path":["stage"]}))
        plan=self.plan([a,b])
        with self.controller_context(plan):state=w.run(self.ref,execute_jobs=True,poll_seconds=.01)
        self.assertTrue(state["complete"])
        self.assertEqual({k:v["status"] for k,v in state["tasks"].items()},{"a":"succeeded","b":"succeeded"})
        self.assertEqual(len(self.processes),2)
        value=w.c.read_ref(w.c.read_ref(state["tasks"]["b"]["result_ref"])["value"])
        self.assertEqual(value["upstream"],"a")
        for entry in state["tasks"].values():
            self.assertEqual(entry["actual_controller_wait"],0)
            self.assertTrue(entry["owned_process_group_absent"])
            self.assertGreater(entry["retained_bytes"],0)

    def test_failed_dependency_blocks_without_retry_while_independent_task_finishes(self):
        a=self.task("bad");b=self.task("dependent",deps=("bad",));independent=self.task("independent")
        plan=self.plan([a,b,independent])
        with self.controller_context(plan,failure_ids=("bad",)):
            state=w.run(self.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual({k:v["status"] for k,v in state["tasks"].items()},
                         dict(bad="failed",dependent="blocked",independent="succeeded"))
        self.assertEqual(len(self.processes),2)
        self.assertFalse((self.root/"dependent").exists())
        self.assertTrue((self.root/"bad/actual_wait.json").exists())

    def test_storage_failstop_starts_no_process_or_scientific_scope(self):
        plan=self.plan([self.task("blocked")],measured=980000000000-2**20)
        with self.controller_context(plan):state=w.run(self.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual(state["run_status"],"blocked_on_storage_capacity")
        self.assertEqual(state["tasks"]["blocked"]["status"],"pending")
        self.assertEqual(self.processes,[])
        self.assertFalse((self.root/"blocked").exists())

    def test_controller_exception_signals_and_waits_its_active_supervisor(self):
        task=self.task("busy",seconds=60);plan=self.plan([task])
        def release(_):
            if (self.root/"busy/ready").exists():raise ValueError("synthetic controller release withdrawn")
            return dict(approved_workflow_ref=self.ref)
        with self.controller_context(plan,long_ids=("busy",),release=release):
            with self.assertRaisesRegex(ValueError,"synthetic controller release withdrawn"):
                w.run(self.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual(len(self.processes),1)
        self.assertIsNotNone(self.processes[0].poll())
        wait=json.loads((self.root/"busy/actual_wait.json").read_text())
        self.assertTrue(wait["actual_direct_wait"])
        self.assertTrue(wait["owned_process_group_absent"])
        self.assertIsNotNone(wait["supervisor_error"])

    def test_unknown_interrupted_scope_requires_reconciliation_before_poll_or_dispatch(self):
        task=self.task("unknown");plan=self.plan([task])
        w.save(Path(plan["records_directory"])/"state.json",dict(plan_ref=self.ref,
            tasks={"unknown":dict(status="unknown",reservation_bytes=task["max_output_bytes"])}))
        def forbidden_poll(_):raise ValueError("Unsafe unknown restart reached polling")
        with self.controller_context(plan,release=forbidden_poll):
            with self.assertRaisesRegex(ValueError,"reconciliation"):
                w.run(self.ref,execute_jobs=True,poll_seconds=.01)
        self.assertEqual(self.processes,[])

    def test_concrete_american_DAG_factory_keyword_formats_bind_without_execution(self):
        tasks=american.build_tasks(prefix="format_test",output_root=str(self.root/"campaign"))
        for task in tasks:
            if task["resource"]=="external":continue
            factory=getattr(importlib.import_module(task["factory_module"]),task["factory_function"])
            inspect.signature(factory).bind(**task["factory_arguments"],
                                            output_directory=str(Path(task["output_directory"])/"native"))
        self.assertEqual(sum(t["id"].endswith("_argformer_mlm") for t in tasks),15)
        self.assertEqual(self.processes,[])

    def test_real_compact_factory_materializes_pinned_sources_without_scientific_payloads(self):
        parent="argformer_american_validation_004";base=self.root/parent
        def metadata(name,value):return w.save(base/"inference"/name,value,exclusive=True)
        rows=[dict(sample_index=i,observed_node_id=i,sample_id="sample_%d"%(i//2),haplotype=i%2,
                   age_generations=0.,role="pure_source_reference" if i<600 else "query",
                   observed_reference_class=i//200 if i<600 else None) for i in range(1600)]
        roster=metadata("roster.json",dict(parent_id=parent,class_order=["EUR","EAS","AFR"],samples=rows))
        payloads={k:dict(path=str((base/"inference"/k).relative_to(w.c.ROOT)),sha256="a"*64)
                  for k in ("tree_ref","genotypes_ref","positions_ref","alleles_ref")}
        observed=metadata("observation.json",dict(schema="american_c3_observed_v1",parent_id=parent,split="val",
            class_order=["EUR","EAS","AFR"],sequence_length_bp=30000000,genealogy_kind=american.observation.GENEALOGY,
            query_ancestry_truth_in_inputs=False,sites_and_sample_identities_verified=True,
            migrations_and_internal_population_labels_removed=True,roster_ref=roster,**payloads))
        completion=metadata("completed.json",dict(status="completed_current_C3_inference",observation_ref=observed))
        ready=metadata("ready.json",dict(schema="american_fresh_inference_ready_v1",parent_id=parent,split="val",
            observation_ref=observed,stage_completion_ref=completion,requires_tsinfer_tsdate=False,
            query_truth_in_training_inputs=False,truth_scoring_authorized=False,exact_paper_reproduction=False,
            declared_conventions=american.generation.CONVENTIONS,
            declared_inference_conventions=american.observation.INFERENCE_CONVENTIONS,seeds=dict(sampling=92001)))
        task=self.task(parent+"_focals")
        task.update(output_directory=str(base/"focal_job"),factory_module="arggnn.benchmarking.american_workflow",
            factory_function="focal_factory",factory_arguments=dict(input_ref=ready,
                limits=dict(max_seconds=43200,max_rss_bytes=64*2**30,max_output_bytes=31*2**20)))
        plan=self.plan([task])
        with patch.object(american.focals,"_exact_focals",side_effect=AssertionError("No scientific execution permitted")):
            job_ref=w.materialize(self.ref,plan,task,dict(tasks={}))
        job=w.c.read_ref(job_ref)
        self.assertEqual(job["call"]["module"],"arggnn.benchmarking.american_focals")
        self.assertEqual(job["call"]["spec"]["declared_focal_conventions"],american.focals.CONVENTIONS)
        self.assertGreaterEqual(len(job["call"]["source_refs"]),6)
        self.assertFalse(Path(task["output_directory"]).exists())
        self.assertTrue(all(not w.c.contained_path(r["path"]).exists() for r in payloads.values()))
        self.assertEqual(self.processes,[])

    def test_real_generation_factory_materializes_exact_full_recipe_without_launch(self):
        parent="argformer_american_validation_004"
        task=self.task(parent+"_generation")
        task.update(factory_module="arggnn.benchmarking.american_workflow",factory_function="generation_factory",
            factory_arguments=dict(task_id=parent+"_simulate",limits=dict(max_seconds=7200,max_rss_bytes=128*2**30,max_output_bytes=20*2**30)))
        task["output_directory"]=str(self.root/parent/"generation_job")
        plan=self.plan([task])
        job_ref=w.materialize(self.ref,plan,task,dict(tasks={}))
        job=w.c.read_ref(job_ref);call=job["call"]
        self.assertEqual(call["module"],"arggnn.benchmarking.american_preparation")
        self.assertEqual(call["spec"]["max_sites"],5000000)
        self.assertEqual(call["spec"]["declared_conventions"],american.generation.CONVENTIONS)
        prepared=american.generation.prepare(call["spec"])
        self.assertEqual(prepared["config"]["sequence_length_bp"],30000000)
        self.assertEqual(prepared["config"]["sample_counts"],dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000))
        self.assertFalse(Path(task["output_directory"]).exists())
        self.assertEqual(self.processes,[])


if __name__=="__main__":unittest.main()
