"""Focused CPU fixtures for current-release full-length C4 adapters."""
from pathlib import Path
from unittest.mock import patch
import inspect
import io
import tempfile
import unittest
import numpy as np
from arggnn.benchmarking import european_argmix_release as release
from arggnn.benchmarking import european_argmix_workflow as workflow
from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking import european_native_models as m


def rows_fixture():
    cohorts = [("Ana",50),("CHG",28),("WHG",102),("EHG",86),
               ("present_day",182),("Bronze_Age",162),("BAA",14),("Neo",302),("Yam",26)]
    rows = []
    for cohort, count in cohorts:
        for _ in range(count):
            i=len(rows)
            rows.append(dict(sample_index=i, observed_node_id=i, inferred_tree_node_id=i,
                source_observed_node_id=i,sample_id="s"+str(i//2),haplotype=i%2,
                observed_cohort=cohort,age_generations=0.,
                role="query" if cohort in sampling.COHORTS else "pure_source_reference",
                observed_reference_class=None))
    return rows


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix=".european_C4_tests_",dir=m.ROOT/"tests")
        self.base=Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()

    def test_direct_source_join_is_true_same_SNP_not_prediction_collapse(self):
        rows=[r for r in rows_fixture() if r["role"]=="query"]
        identities=[dict(sample_index=r["sample_index"],sample_id=r["sample_id"],
                        haplotype=r["haplotype"]) for r in rows]
        native=np.ones((2,686),dtype=np.int16)
        source=np.zeros((2,686),dtype=np.int16)
        native[1,:]=2
        source[1,:]=1
        # BAA is represented by the corresponding native via-BAA path.
        baa=[i for i,r in enumerate(rows) if r["observed_cohort"]=="BAA"]
        native[0,baa]=5
        native[1,baa]=6
        registry=dict(schema="ap_paper_sampled_registry_v1",parent_id="p",split="train",
            examples=2,query_identities=identities,records=[
                dict(example_index=1,sample_id=rows[baa[0]]["sample_id"],
                     haplotype=rows[baa[0]]["haplotype"],sample_index=rows[baa[0]]["sample_index"],
                     position_original_bp=100.,cohort="BAA"),
                dict(example_index=0,sample_id=rows[0]["sample_id"],haplotype=rows[0]["haplotype"],
                     sample_index=rows[0]["sample_index"],position_original_bp=99.,cohort="present_day")])
        schema=dict(parent_id="p",split="train",native_paths_1based=list(range(1,7)),
            query_sample_indices=[r["sample_index"] for r in rows],
            query_keys=[[r["sample_id"],r["haplotype"]] for r in rows],
            native_to_C4=[0,1,2,3,0,1])
        truth=dict(intervals_bp=np.array([[0.,100.],[100.,release.LENGTH]]),
            native_paths_1based=native,source_classes_0based=source,
            query_raw_node_ids=np.array(schema["query_sample_indices"]))
        self.assertEqual(release.sampled_source_labels(registry,schema,truth).tolist(),[0,1])
        truth["source_classes_0based"][1,baa[0]]=0
        with self.assertRaisesRegex(ValueError,"ontology"):
            release.sampled_source_labels(registry,schema,truth)

    def test_TEST_target_denial_precedes_private_metadata(self):
        registry=dict(schema="ap_paper_sampled_registry_v1",split="test",labels_present=False)
        seen=[]
        def read(ref,**kwargs):
            seen.append(ref)
            return registry
        with patch.object(m,"_read",side_effect=read):
            with self.assertRaisesRegex(ValueError,"TRAIN"):
                release.prepare_sampled_c4_targets("observed","PRIVATEgeneration","PRIVATEschema",
                    "PRIVATEtruth",self.base/"unused",guard=lambda _:None,
                    execution_check=lambda:None,write_headroom_bytes=1,
                    max_truth_uncompressed_bytes=1)
        self.assertEqual(seen,["observed"])

    def test_full_200Mb_SNP_join_uses_half_open_intervals_and_byte_cap(self):
        queries=[r for r in rows_fixture() if r["role"]=="query"]
        labels=np.zeros((2,686),dtype=np.int16)
        labels[1]=3
        intervals=np.array([[0.,10.],[10.,release.LENGTH]],dtype=np.float64)
        positions=np.array([0,9,10,release.LENGTH-1],dtype=np.int64)
        nodes=np.array([r["source_observed_node_id"] for r in queries])
        kwargs=dict(max_snp_truth_bytes=4*686*2)
        out=release.native_snp_truth(intervals,labels,nodes,queries,positions,
                                    positions.astype(np.float64),**kwargs)
        self.assertEqual(out["truth__hap_labels"][:,0].tolist(),[0,0,3,3])
        self.assertEqual(out["truth__hap_labels"].dtype,np.int16)
        self.assertEqual(len(out["truth__sample_ids"]),343)
        with self.assertRaisesRegex(ValueError,"without cropping"):
            release.native_snp_truth(intervals,labels,nodes,queries,positions,
                positions.astype(np.float64),max_snp_truth_bytes=1)

    def test_streaming_prediction_match_keeps_canonical_order_and_half_open_edges(self):
        registry=dict(examples=3,records=[
            dict(example_index=2,query_node_id=7,position_local_bp=10.),
            dict(example_index=0,query_node_id=7,position_local_bp=2.),
            dict(example_index=1,query_node_id=8,position_local_bp=5.)])
        chunk1=(np.array([7,8]),np.array([0.,0.]),np.array([10.,10.]),
                np.array([[1.,0,0,0],[0,1.,0,0]],dtype=np.float32))
        chunk2=(np.array([7]),np.array([10.]),np.array([20.]),
                np.array([[0,0,1.,0]],dtype=np.float32))
        result=release.match_sampled_intervals(registry,[chunk1,chunk2])
        self.assertEqual(result.argmax(1).tolist(),[0,1,2])
        with self.assertRaisesRegex(ValueError,"no denominator"):
            release.match_sampled_intervals(registry,[chunk1])
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            release.match_sampled_intervals(registry,[chunk1,chunk1,chunk2])

    def test_C4_metrics_preserve_zero_probability_and_truth_row_convention(self):
        p=np.array([[1.,0,0,0],[1.,0,0,0]],dtype=np.float32)
        result=release.c4_metrics(p,np.array([0,1]))
        self.assertEqual(result["accuracy"],.5)
        self.assertIsNone(result["raw_NLL"])
        self.assertTrue(result["raw_NLL_is_infinite"])
        self.assertEqual(result["confusion"][1][0],1)
        self.assertFalse(result["probabilities_clipped_or_renormalized"])

    def test_cross_endpoint_missing_predictions_block_before_private_truth(self):
        seen=[]
        plan=dict(test_parent_ids=["p"+str(i) for i in range(5)])
        with patch.object(m,"_read",side_effect=lambda ref,**kw:seen.append(ref)):
            with self.assertRaisesRegex(ValueError,"Every registered"):
                release.verify_cross_endpoint_barrier(plan,{}, {})
        self.assertEqual(seen,[])

    def test_C4_registry_uses_only_observed_interior_SNPs_before_truth(self):
        import tskit
        tables=tskit.TableCollection(sequence_length=1000.)
        for _ in range(952):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
        for left,right in [(0.,200.),(200.,800.),(800.,1000.)]:
            root=tables.nodes.add_row(time=1000.)
            for child in range(952):
                tables.edges.add_row(left,right,root,child)
        for p in range(1000):
            tables.sites.add_row(position=p,ancestral_state="0")
        tables.sort()
        ts=tables.tree_sequence()
        path=self.base/"observed.trees"
        ts.dump(path)
        rows=rows_fixture()
        tree_ref=m._ref(path)
        rows_ref=m._json(self.base/"rows.json",rows,lambda _:None)
        metadata=dict(site_join=[dict(native_site_id=p,local_position_0based=p,
            original_position_0based=p) for p in range(1000)],
            sample_index_to_inferred_tree_node_id=[
                dict(sample_index=i,inferred_tree_node_id=i) for i in range(952)],
            coverage_original_bp0=[0.,1000.],local_to_original_offset_bp=0.)
        inference=dict(parent_id="test0",truth_or_oracle_read=False,
            inferred_tree=tree_ref,observed_sample_manifest=rows_ref,
            post_five_iterations=metadata)
        inference_ref=m._json(self.base/"inference.json",inference,lambda _:None)
        plan=sampling.paper_sampling_plan(["train0","train1","train2"],
                                         ["test0","test1","test2","test3","test4"],seed=8)
        plan_ref=m._json(self.base/"plan.json",plan,lambda _:None)
        result=release.prepare_c4_registry(plan_ref,inference_ref,self.base/"support",
            guard=lambda _:None,execution_check=lambda:None,write_headroom_bytes=1,
            max_metadata_bytes=4*2**20)
        registry=m._read(result["registry_ref"],observed=True)
        self.assertEqual(registry["examples"],10000)
        self.assertEqual(registry["cohort_quotas"],{c:2000 for c in sampling.COHORTS})
        self.assertTrue(all(200 <= r["position_local_bp"] < 800 for r in registry["records"]))
        self.assertTrue(all(r["native_site_id"] == r["position_local_bp"] for r in registry["records"]))
        self.assertEqual(registry["native_C4_eligible_support"]["excluded_edge_SNPs"],400)
        self.assertFalse(registry["native_C4_eligible_support"]["eligibility_uses_predictions_or_truth"])

    def test_native_stored_shards_match_author_loader_and_global_shuffle_without_duplicate(self):
        import ast
        import torch
        from arggnn.benchmarking import european_argmix_shards as shards
        source=release.native.AUTHOR/"src/train_joint_model.py"
        tree=ast.parse(source.read_text())
        loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                    and n.name=="_load_gt_gru_split_pt")
        constants=[n for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(v,ast.Name) and
                (v.id.startswith("GT_GRU_") or v.id=="CANONICAL_SPECIAL_IDS")
                for v in n.targets)]
        namespace={"torch":torch}
        exec(compile(ast.Module(body=constants+[loader],type_ignores=[]),
                     str(source),"exec"),namespace)
        original_loader=namespace["_load_gt_gru_split_pt"]
        paths,refs,all_values=[],[],[]
        for shard_index,n in enumerate((1,2)):
            shape=(n,1024,16)
            arrays=dict(
                input_ids=(np.arange(np.prod(shape)).reshape(shape)%9).astype(np.int64),
                tmrca=np.full((n,1024,16,16),13+shard_index,dtype=np.int16),
                labels=np.full((n,1024),shard_index,dtype=np.int64),
                row_mask=np.ones((n,1024),dtype=np.uint8),
                cm_features=np.full((n,1024,2),.125*(shard_index+1),dtype=np.float32),
                metadata=np.asarray([2,0,1024,512,15,0,14,0,1,2,3,9],dtype=np.int64))
            arrays["row_mask"][:,-2:]=0
            path=self.base/("shard"+str(shard_index)+".pt")
            with path.open("xb") as stream:
                np.savez(stream,**arrays)
            paths.append(path)
            refs.append(m._ref(path))
            all_values.append(arrays)
        dense=self.base/"dense.pt"
        merged={key:(all_values[0][key] if key=="metadata" else np.concatenate(
            [v[key] for v in all_values])) for key in all_values[0]}
        with dense.open("xb") as stream:
            np.savez(stream,**merged)
        expected=original_loader(str(dense),expected_split_code=0)
        manifest=shards.write_manifest(self.base/"logical.json",["p0","p1"],refs,
                                      split_code=0,guard=lambda _:None)
        before=torch.get_rng_state().clone()
        actual=shards.load_sharded_split(str(m._checked(manifest)),expected_split_code=0,
            guard=lambda _:None,execution_check=lambda:None)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(actual["num_chunks"],expected["num_chunks"])
        keys=("input_ids","tmrca","labels","row_mask","cm_features")
        for key in keys:
            self.assertEqual(actual[key].shape,expected[key].shape)
            self.assertEqual(actual[key].dtype,expected[key].dtype)
            for i in range(3):
                self.assertTrue(torch.equal(actual[key][i],expected[key][i]),key)
        actual_ds=torch.utils.data.TensorDataset(*(actual[key] for key in keys))
        dense_ds=torch.utils.data.TensorDataset(*(expected[key] for key in keys))
        torch.manual_seed(50)
        dense_batches=list(torch.utils.data.DataLoader(dense_ds,batch_size=2,shuffle=True,num_workers=0))
        torch.manual_seed(50)
        actual_batches=list(torch.utils.data.DataLoader(actual_ds,batch_size=2,shuffle=True,num_workers=0))
        self.assertEqual(len(actual_batches),len(dense_batches))
        for actual_batch,dense_batch in zip(actual_batches,dense_batches):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual_batch,dense_batch)))
        self.assertEqual(sorted(p.name for p in self.base.iterdir()),
                         ["dense.pt","logical.json","shard0.pt","shard1.pt"])


    def test_private_author_main_restores_argv_and_logging_after_its_log_closes(self):
        import logging
        import sys
        from types import SimpleNamespace
        from arggnn.benchmarking import european_argmix_shards as shards
        def fake_main():
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("synthetic_private_main").info("fixture finished")
        author=SimpleNamespace(main=fake_main,
            __file__=str(release.native.AUTHOR/"src/train_joint_model.py"))
        old_argv=sys.argv[:]
        old_handlers=logging.getLogger().handlers[:]
        shards.run_original_main(author,[],self.base,guard=lambda _:None,
                                 execution_check=lambda:None)
        self.assertEqual(sys.argv,old_argv)
        self.assertEqual(logging.getLogger().handlers,old_handlers)
        self.assertIn("fixture finished",(self.base/"train_joint_model.log").read_text())
        receipt=m._read(m._ref(self.base/"train_joint_model.command.json"))
        self.assertTrue(receipt["original_main_code_object_unchanged"])
        self.assertFalse(receipt["dense_merged_or_extracted_tensor_corpus_written"])



class WorkflowTests(unittest.TestCase):
    def test_actual_frozen_AP_extension_has_bound_factories_and_joint_truth_barriers(self):
        import json
        import importlib
        from arggnn.benchmarking import european_workflow as ap
        data=json.loads((m.ROOT/"configs/benchmarks/executable_workflow_20261018.json").read_text())
        frozen_ap_tasks=[t for t in data["tasks"] if t["id"].startswith("AP__")]
        frozen_score=next(t for t in frozen_ap_tasks if t["id"]=="AP__all_test__ap_score")
        native_cfg=next(t for t in frozen_ap_tasks
            if t["factory_arguments"].get("stage")=="ap_infer")["factory_arguments"]["arguments"]["native_config_ref"]
        fit=next(t for t in frozen_ap_tasks if t["id"]=="AP__all_train__treeloom_fit")
        pred=next(t for t in frozen_ap_tasks if t["id"]=="AP__all_test__treeloom_predict")
        omit_fit={"protocol_id","sampling_plan_ref","train_inputs"}
        omit_pred={"protocol_id","sampling_plan_ref","fit_completion_ref","heldout_inputs"}
        tree_parameters=dict(fit={k:v for k,v in fit["factory_arguments"].items() if k not in omit_fit},
            predict={k:v for k,v in pred["factory_arguments"].items() if k not in omit_pred})
        # The published plan already contains the C4 extension. Rebuild the
        # pristine AP input with its original builder rather than extending an
        # already-replaced score task or weakening the non-mutation assertion.
        frozen_by_stage={t["id"].rsplit("__",1)[1]:t for t in frozen_ap_tasks}
        feature_args=frozen_by_stage["ap_features"]["factory_arguments"]["arguments"]
        native_fit_args=frozen_by_stage["ap_fit"]["factory_arguments"]["arguments"]
        cap_keys={"max_seconds","max_output_bytes","max_rss_bytes",
                  "minimum_free_gpu_bytes","max_cuda_reserved_bytes"}
        ap_limits={stage:{key:value for key,value in frozen_by_stage[stage].items()
                          if key in cap_keys}
                   for stage in (*ap.STAGES,"treeloom_fit","treeloom_predict")}
        ap_tasks=ap.build_tasks(prefix="AP",output_root="experiments/results/synthetic_AP_workflow",
            parent_records=[t["factory_arguments"]["arguments"]["parent"]
                            for t in frozen_ap_tasks
                            if t["factory_arguments"].get("stage")=="ap_generate"],
            sampling_plan_ref=fit["factory_arguments"]["sampling_plan_ref"],
            native_config_ref=native_cfg,
            python_data=frozen_by_stage["ap_generate"]["factory_arguments"]["python"],
            python_tensorflow=frozen_by_stage["ap_fit"]["factory_arguments"]["python"],
            stage_limits=ap_limits,
            stage_parameters=dict(write_headroom_bytes=feature_args["write_headroom_bytes"],
                max_metadata_bytes=feature_args["max_metadata_bytes"],
                max_feature_bytes=feature_args["max_feature_bytes"],
                max_truth_uncompressed_bytes=frozen_score["factory_arguments"]["arguments"]["max_truth_uncompressed_bytes"],
                native_training_seed=native_fit_args["seed"],
                native_tensorflow_threads=native_fit_args["threads"]),
            treeloom_parameters=tree_parameters)
        self.assertEqual(len(ap_tasks),44)
        score=next(t for t in ap_tasks if t["id"]=="AP__all_test__ap_score")
        self.assertEqual(score["factory_module"],ap.MODULE)
        self.assertEqual(score["factory_arguments"]["stage"],"ap_score")
        self.assertEqual(len(score["dependencies"]),6)
        stages=("validation_generate","validation_infer",*workflow.STAGES,
                "c4_treeloom_fit","c4_treeloom_predict")
        limits={s:dict(max_seconds=100,max_output_bytes=2**30,max_rss_bytes=2**30,
                      minimum_free_gpu_bytes=2**30,max_cuda_reserved_bytes=2**30) for s in stages}
        kwargs=dict(write_headroom_bytes=1,max_metadata_bytes=128*2**20,
            max_truth_uncompressed_bytes=16*2**30,max_snp_truth_bytes=8*2**30,
            native_build_ref={"path":"build","sha256":"a"*64},
            native_cpu_ref={"path":"cpu","sha256":"b"*64},
            native_archive_command_limits={"max_seconds":100},
            max_native_archive_bytes=128*2**30,max_merged_archive_bytes=384*2**30,
            max_h5_chunk_rows=65536)
        result=workflow.build_tasks(ap_tasks=ap_tasks,ap_prefix="AP",prefix="C4",
            output_root="experiments/results/synthetic_C4_workflow",
            config_ref=m._ref(m.ROOT/"configs/benchmarks/recipes/argmix_release_ap200mb_c4_v2.json"),
            sampling_plan_ref=fit["factory_arguments"]["sampling_plan_ref"],
            native_config_ref=native_cfg,python_data=".runtime/env/bin/python",
            python_native=".runtime/official_argmix_abi210_v1/cu129/env/bin/python",
            stage_limits=limits,stage_parameters=kwargs,treeloom_parameters=tree_parameters)
        self.assertEqual(len(result),88)
        ids={t["id"] for t in result}
        self.assertEqual(len(ids),88)
        self.assertTrue(all(set(t["dependencies"]) <= ids for t in result))
        final_scores=[t for t in result if t["factory_arguments"].get("stage")
                      in ("release_native6_score","release_c4_score")]
        self.assertEqual(len(final_scores),2)
        self.assertTrue(all(len(t["dependencies"])==12 for t in final_scores))
        for task in result:
            module=importlib.import_module(task["factory_module"])
            factory=getattr(module,task["factory_function"])
            inspect.signature(factory).bind(**task["factory_arguments"],
                output_directory=task["output_directory"]+"/native")
        self.assertNotIn("c4_predictions",score["factory_arguments"]["arguments"])
        c4_fit=next(t for t in result if t["id"]=="C4__all_train__c4_treeloom_fit")
        self.assertEqual(c4_fit["factory_module"],"arggnn.benchmarking.european_c4_training")
        self.assertEqual(c4_fit["factory_arguments"]["protocol_id"],"ancestralpaths_paper200mb_c4")
        val_gen=next(t for t in result if t["id"].endswith("__validation_generate"))
        parent=val_gen["factory_arguments"]["arguments"]["parent"]
        self.assertEqual(parent["split"],"validation")
        self.assertEqual(set(parent["seeds"]),{"ancestry","mutations","relate_initial","relate_size_base"})


if __name__ == "__main__":
    unittest.main()
