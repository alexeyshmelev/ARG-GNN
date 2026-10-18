"""Tiny fake-backend tests for exact fresh native budgets and process evidence."""
from collections import OrderedDict
from contextlib import ExitStack
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import american_competitors as api
from arggnn.benchmarking import american_competitor_worker as worker


WAIT=dict(actual_direct_wait=True,actual_direct_returncode=0,native_process_absent=True,
          outer_owned_group_cleanup_pending=True)


class FreshCompetitorContractTest(unittest.TestCase):
    def setUp(self):
        self.parent="argformer_american_test_999"
        self.refs={k:dict(path="mock/"+k+".json",sha256="a"*64)
                   for k in ("inference","focals","observation","focal_registry","preparation","mlm","contrastive","embeddings","checkpoint")}
        self.source=dict(parent_id=self.parent,split="test",observation_ref=self.refs["observation"])
        self.fp=dict(focals_ref=self.refs["focal_registry"])
        self.prep=dict(schema="american_fresh_ARGformer_preparation_v1",status="completed_native_C3_path_preparation",
                       parent_id=self.parent,observation_ref=self.refs["observation"],focals_ref=self.refs["focal_registry"],
                       query_ancestry_truth_opened=False,class_order=["EUR","EAS","AFR"],genealogy_kind=api.observation.GENEALOGY,
                       MLM_optimizer_updates=100,contrastive_optimizer_updates=2,vocabulary_nodes=107000,
                       datasets={k:dict(rows=n) for k,n in (("mlm/train",10240),("mlm/val",1138),
                          ("contrastive/train",8192),("contrastive/val",1024),("query_paths/train",9000),("query_paths/val",1000))})
        self.products={self.refs["preparation"]["path"]:self.prep}
        for stage,epochs,updates in (("mlm",5,100),("contrastive",1,2),("embeddings",1,2)):
            self.products[self.refs[stage]["path"]]=dict(schema="american_fresh_ARGformer_stage_completed_v1",stage=stage,
                parent_id=self.parent,preparation_ref=self.refs["preparation"],focal_registry_ref=self.refs["focal_registry"],
                query_ancestry_truth_opened=False,native_wait=dict(WAIT),checkpoint_ref=self.refs["checkpoint"],
                completed_epochs=epochs,actual_optimizer_updates=updates,optimizer_unique_step_values=[updates],
                outputs={k:dict(rows=self.prep["datasets"][k]["rows"]) for k in api.VIEWS},pooling="head_norm",max_samples=0)
        self.limits=dict(max_seconds=86400,max_rss_bytes=128*2**30,max_output_bytes=8*2**30)
        self.stack=ExitStack()
        self.stack.enter_context(patch.object(api,"inputs",return_value=(self.source,{},self.fp)))
        self.stack.enter_context(patch.object(api,"requirements",return_value=dict(source_refs=[])))
        self.real_read=api.read
        self.stack.enter_context(patch.object(api,"read",side_effect=lambda ref:self.products.get(ref["path"]) or self.real_read(ref)))
        self.addCleanup(self.stack.close)

    def build(self,stage,output=None,**extra):
        args=dict(stage=stage,inference_ref=self.refs["inference"],focal_preparation_ref=self.refs["focals"],
                  output_directory=output or "experiments/results/mock/"+self.parent+"/"+stage,limits=self.limits)
        if stage!="prepare":args["preparation_ref"]=self.refs["preparation"]
        if stage in ("contrastive","embeddings","retrieval"):
            args["producer_ref"]=self.refs[{"contrastive":"mlm","embeddings":"contrastive","retrieval":"embeddings"}[stage]]
        args.update(extra)
        return api.build_argformer_spec(**args)

    def test_full_native_budgets_and_exact_validation(self):
        mlm=self.build("mlm");self.assertEqual(mlm["expected_updates"],100)
        contrastive=self.build("contrastive");self.assertEqual(contrastive["expected_updates"],2)
        self.assertEqual(self.build("embeddings")["checkpoint_ref"],self.refs["checkpoint"])
        self.assertEqual(self.build("retrieval")["producer_ref"],self.refs["embeddings"])
        api.validate_spec(mlm)
        bad=copy.deepcopy(mlm);bad["expected_updates"]=99
        with self.assertRaisesRegex(ValueError,"Exact full native"):
            api.validate_spec(bad)
        self.assertEqual(api.SETTINGS["MLM"]["epochs"],5)
        self.assertEqual(api.SETTINGS["contrastive"]["temperature"],.3)
        self.assertTrue(api.SETTINGS["contrastive"]["uniform_class_weight_fallback"])

    def test_predecessor_misbinding_partial_budget_and_output_cap_reject(self):
        previous=self.products[self.refs["mlm"]["path"]]
        for key,value in (("parent_id","another_ARG"),("query_ancestry_truth_opened",True),
                          ("actual_optimizer_updates",99),("focal_registry_ref",self.refs["checkpoint"])):
            before=previous[key];previous[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.build("contrastive")
            previous[key]=before
        self.limits["max_output_bytes"]=100
        with self.assertRaisesRegex(ValueError,"full native embedding"):
            self.build("embeddings")

    def test_actual_native_wait_evidence_and_outer_group_pending(self):
        api.validate_wait(WAIT)
        for key,value in (("actual_direct_wait",False),("actual_direct_returncode",1),
                          ("native_process_absent",False),("outer_owned_group_cleanup_pending",False)):
            with self.subTest(key=key),self.assertRaises(ValueError):api.validate_wait(dict(WAIT,**{key:value}))

    def test_saved_actual_counters_accept_ordereddict_and_reject_partial(self):
        state=OrderedDict(model=OrderedDict(weight=object()),timestamp={"Timestamp":dict(epoch=5,batch=100)},
                          optimizers={"DecoupledAdamW":dict(state={0:dict(step=100,exp_avg=object(),exp_avg_sq=object())})})
        self.assertEqual(worker.checkpoint_counters(state,5,100)["actual_optimizer_updates"],100)
        state["optimizers"]["DecoupledAdamW"]["state"][0]["step"]=99
        with self.assertRaises(ValueError):worker.checkpoint_counters(state,5,100)

    def test_fake_runner_is_injected_and_real_completion_envelope_is_bound(self):
        with tempfile.TemporaryDirectory(prefix="native_fake_",dir=str(api.ROOT/"experiments/results")) as temporary:
            out=Path(temporary)/self.parent/"mlm"
            spec=self.build("mlm",output=out)
            calls=[]
            def runner(argv,cwd,env,limits,log_path):
                calls.append(argv)
                self.assertEqual(argv[0],str(api.ROOT/api.NATIVE_PYTHON))
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"],api.GPU0_UUID)
                self.assertEqual(limits,self.limits)
                value=dict(self.products[self.refs["mlm"]["path"]],observation_ref=self.refs["observation"])
                (out/"scientific_completed.json").write_text(json.dumps(value))
                Path(log_path).write_text("fake backend only\n")
                return dict(WAIT)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=api.GPU0_UUID,ARGGNN_PHYSICAL_GPU="0"):
                result=api.run_argformer(spec,bounded_runner=runner,resource_check=lambda _:None)
            value=self.real_read(result)
            self.assertEqual(len(calls),1)
            self.assertEqual(value["native_wait"],WAIT)
            self.assertTrue(value["outer_worker_wait_still_required"])
            self.assertFalse(value["query_ancestry_truth_opened"])

    def test_flare_uses_original_options_same_focals_and_injected_native_runner(self):
        from arggnn import american_c3_flare_common_v1 as common
        from arggnn import american_c3_flare_evaluate_v1 as evaluator
        with tempfile.TemporaryDirectory(prefix="flare_fake_",dir=str(api.ROOT/"experiments/results")) as temporary:
            base=Path(temporary)/self.parent
            base.mkdir()
            jar=base/"fixture.jar";jar.write_bytes(b"fake jar, never executed")
            java=base/"fixture_java";java.write_text("fake executable, never executed");java.chmod(0o700)
            binary=dict(schema="root_qualified_FLARE_binary_v1",root_authorizer="root_agent",
                repository=common.PIN_REPOSITORY,tag=common.PIN_TAG,source_commit=common.PIN_COMMIT,
                native_sources_modified=False,source_commit_build_verified=True,
                reported_version=common.EXPECTED_VERSION,jar_ref=api.ref(jar))
            bp=base/"binary.json";bp.write_text(json.dumps(binary))
            version=b'openjdk version "21"\n'
            runtime=dict(binary_provenance_ref=api.ref(bp),jar_ref=api.ref(jar),java_executable=str(java),
                         java_executable_sha256=api.ref(java)["sha256"],java_version_output_sha256=hashlib.sha256(version).hexdigest())
            converted=dict(status="completed_C3_FLARE_observation_conversion",parent_id=self.parent,
                           observation_ref=self.refs["observation"],focal_registry_ref=self.refs["focal_registry"],
                           native_options=common.NATIVE_OPTIONS,projection=common.PROJECTION,
                           native_source_commit=common.PIN_COMMIT,reference_haplotypes=600,query_haplotypes=1000,
                           common_focal_records=10000,common_focal_registry_unchanged=True,
                           query_ancestry_truth_opened=False,native_expected_markers=2)
            for key in ("ref_vcf_ref","gt_vcf_ref","ref_panel_ref","map_ref"):
                file=base/key;file.write_text("fake input");converted[key]=api.ref(file)
            cp=base/"conversion.json";cp.write_text(json.dumps(converted))
            out=base/"native"
            spec=api.build_flare_spec(stage="native",inference_ref=self.refs["inference"],
                focal_preparation_ref=self.refs["focals"],output_directory=out,
                limits=dict(self.limits,max_output_bytes=3*2**30),conversion_ref=api.ref(cp),native_runtime=runtime)
            calls=[]
            def runner(argv,cwd,env,limits,log_path):
                calls.append(argv)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"],"")
                if argv[-1]=="-version":Path(log_path).write_bytes(version)
                else:
                    Path(log_path).write_text("fake native output")
                    for suffix in (".anc.vcf.gz",".model",".global.anc.gz",".log"):
                        (out/"native"/("flare"+suffix)).write_text(common.EXPECTED_VERSION if suffix==".log" else "fake")
                return dict(WAIT)
            seen=[]
            def seal(cfg,writer):
                seen.append(cfg)
                return writer.json("completed.json",dict(status="fake_test_seal_only",focal_registry_ref=cfg["focal_registry_ref"]))
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=""),patch.object(evaluator,"seal_native",side_effect=seal), \
                 patch("resource.setrlimit"):
                result=api.run_flare(spec,bounded_runner=runner,resource_check=lambda _:None)
            self.assertEqual(len(calls),2)
            self.assertIn("min-mac=50",calls[1]);self.assertIn("em=true",calls[1])
            self.assertIn("probs=true",calls[1]);self.assertIn("gen=10",calls[1])
            self.assertFalse(any(arg.startswith(("gt-ancestries=","model=")) for arg in calls[1]))
            self.assertEqual(seen[0]["focal_registry_ref"],self.refs["focal_registry"])
            self.assertIsNone(seen[0]["external_truth_ref"])
            native=self.real_read(api.ref(out/"native_completed.json"))
            self.assertEqual(native["native_wait"],WAIT)
            self.assertFalse(native["query_ancestry_truth_opened"])

    def test_inert_import_in_fresh_cpu_process(self):
        code=("import sys; sys.path.insert(0,%r); " % str(Path(api.__file__).resolve().parents[2])
              + "from arggnn.benchmarking import american_competitors, american_competitor_worker; "
                "assert not any(k in sys.modules for k in ('numpy','torch','tskit','torch_geometric'))")
        subprocess.run([sys.executable,"-B","-s","-c",code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__=="__main__":unittest.main()
