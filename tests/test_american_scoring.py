"""Synthetic scoring checks; no simulation, model, native command or real labels."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from arggnn.benchmarking import american_scoring as scoring


class SyntheticScoreTest(unittest.TestCase):
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return scoring.generation._reference(path)
    def npz(self,path,**arrays):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as f:np.savez_compressed(f,**arrays)
        return scoring.generation._reference(path)
    def fixture(self,base):
        parents=[dict(parent_id="argformer_american_"+split+"_%03d"%i,split=split,simulation_stream=split+str(i))
                 for split,n in (("train",20),("validation",5),("test",10)) for i in range(n)]
        registry=self.write(base/"registry.json",dict(schema=scoring.fresh.SCHEMA,
            split_unit="independent_simulation_parent",class_order=["EUR","EAS","AFR"],sequence_length_bp=30000000,parents=parents))
        generation,preparation,predictions={},{},{m:{} for m in scoring.METHODS}
        for entry in parents[20:]:
            parent=entry["parent_id"];root=base/parent;split="val" if entry["split"]=="validation" else "test"
            original=self.write(root/"generation/observed.json",dict(parent_id=parent))
            generation[parent]=self.write(root/"generation/observation_ready.json",dict(
                schema="american_fresh_shared_observation_ready_v1",parent_id=parent,split=split,
                query_truth_in_training_inputs=False,truth_scoring_authorized=False,observation_ref=original))
            observed=self.write(root/"inference/observed.json",dict(source_observation_ref=original,
                parent_id=parent,split=split,class_order=["EUR","EAS","AFR"],
                genealogy_kind=scoring.focals.observation.GENEALOGY,sites_and_sample_identities_verified=True,
                query_ancestry_truth_in_inputs=False,sequence_length_bp=30000000))
            inferred=self.write(root/"inference/ready.json",dict(schema="american_fresh_inference_ready_v1",
                parent_id=parent,split=split,requires_tsinfer_tsdate=False,
                input_ref=generation[parent],observation_ref=observed,query_truth_in_training_inputs=False))
            positions=np.repeat(np.array([5.,15000000.]),5000)
            records=dict(record_ids=np.arange(10000,dtype=np.int64)+60000,
                sample_nodes=600+np.arange(10000,dtype=np.int64)%1000,
                sample_indices=600+np.arange(10000,dtype=np.int64)%1000,positions_bp0=positions,
                parent_id=np.asarray(parent),class_order=np.asarray(["EUR","EAS","AFR"]))
            focal=self.npz(root/"focals/focals.npz",**records,
                           first_occurrence_intervals_bp=np.column_stack((positions-1,positions+1)))
            preparation[parent]=self.write(root/"focals/preparation.json",dict(schema=scoring.focals.RESULT_SCHEMA,
                status="completed_compact_query_focals",parent_id=parent,split=split,query_focal_count=10000,
                original_record_ids_preserved=True,query_ancestry_truth_opened=False,input_ref=inferred,
                class_order=["EUR","EAS","AFR"],genealogy_kind=scoring.focals.observation.GENEALOGY,
                observation_ref=observed,focals_ref=focal))
            labels=np.zeros((2,1000),dtype=np.uint8);labels[1]=1
            truth=self.npz(root/"generation/private/truth/ancestry.npz",intervals_bp=np.array([[0.,15000000.],[15000000.,30000000.]]),
                query_sample_indices=np.arange(600,1600,dtype=np.int64),source_classes_0based=labels,
                parent_id=records["parent_id"],class_order=records["class_order"])
            self.write(root/"generation/private/generation_completed.json",dict(status="completed_C3_simulation_reconstruction",
                parent_id=parent,class_order=["EUR","EAS","AFR"],observation_ref=original,private_truth_ref=truth))
            for method in scoring.METHODS:
                if method=="treeloom":
                    prob=np.full((10000,3),.1,dtype=np.float32);prob[np.arange(10000),np.repeat([0,1],5000)]=.8
                else:prob=np.eye(3,dtype=np.float32)[np.full(10000,0 if method=="argformer" else 1)]
                pred=self.npz(root/method/"predictions.npz",**records,probabilities=prob)
                seal=dict(parent_id=parent,query_ancestry_truth_opened=False,records=10000,class_order=["EUR","EAS","AFR"],
                          focal_registry_ref=focal,prediction_ref=pred)
                if method=="treeloom":seal.update(schema="treeloom_fresh_prediction_seal_v1",parent_registry_ref=registry,
                    split=entry["split"],observation_ref=observed,all_native_trees_nodes_sites_samples_retained=True,checkpoint_selection="fixed_final_epoch30")
                elif method=="argformer":
                    prep=self.write(root/method/"preparation.json",dict(parent_id=parent,observation_ref=observed,
                        focals_ref=focal,query_ancestry_truth_opened=False))
                    seal.update(status="sealed_native_C3_top1",scores_and_retrieval_indices_modified=False,preparation_ref=prep)
                else:
                    converted=self.write(root/method/"conversion.json",dict(parent_id=parent,observation_ref=observed,
                        focal_registry_ref=focal,query_ancestry_truth_opened=False))
                    seal.update(status="sealed_C3_native_FLARE_calls_and_raw_posteriors",common_focal_registry_unchanged=True,
                        native_AN1_AN2_calls_used_unchanged=True,one_hot_is_native_call_transport=True,
                        rounded_ANP_renormalized=False,raw_rounded_ANP_modified=False,conversion_ref=converted)
                predictions[method][parent]=self.write(root/method/"seal.json",seal)
        return dict(parent_registry_ref=registry,prediction_refs_by_method=predictions,
                    generation_ready_refs_by_parent=generation,focal_preparation_refs_by_parent=preparation,
                    output_directory=str(base/"scores"))

    def test_all45_arrays_precede_every_truth_and_metrics_share_exact_support(self):
        with tempfile.TemporaryDirectory(prefix="scoring_fake_",dir=str(scoring.ROOT/"experiments/results")) as temporary:
            args=self.fixture(Path(temporary));spec=scoring.build_spec(**args)
            loaded=[];truths=[];actual_load=scoring._load_npz
            def observed_load(record,cap,**kwargs):
                if not kwargs.get("public",True):
                    self.assertEqual(len(loaded),45)
                    barrier=Path(spec["output_directory"])/"all45_prediction_barrier.json"
                    self.assertTrue((scoring.ROOT/barrier).is_file());truths.append(record)
                elif "probabilities" in kwargs.get("keys",()):loaded.append(record)
                return actual_load(record,cap,**kwargs)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=""),patch.object(scoring,"_load_npz",side_effect=observed_load):
                result=scoring.run(spec,resource_check=lambda _:None)
            summary=scoring.generation._read(result)
            self.assertEqual(len(truths),15)
            self.assertEqual(summary["records_per_method"],150000)
            self.assertEqual(summary["equal_parent_summaries"]["validation"]["treeloom"]["accuracy"]["parents"],5)
            self.assertEqual(summary["equal_parent_summaries"]["test"]["treeloom"]["accuracy"]["parents"],10)
            total=summary["equal_parent_summaries"]["all_evaluation"]
            self.assertEqual(total["treeloom"]["accuracy"]["mean"],1.)
            self.assertEqual(total["argformer"]["accuracy"]["mean"],.5)
            self.assertEqual(total["flare"]["accuracy"]["mean"],.5)
            self.assertEqual(total["treeloom"]["accuracy"]["sample_SD_percentage_points"],0.)
            self.assertAlmostEqual(total["treeloom"]["NLL"]["mean"],-math.log(.8),places=6)
            self.assertAlmostEqual(total["treeloom"]["Brier"]["mean"],.06,places=6)
            self.assertNotIn("NLL",total["argformer"])
            self.assertNotIn("NLL",total["flare"])
            self.assertEqual(summary["paired_deltas"]["all_evaluation"]["treeloom_minus_argformer"]["accuracy"]["mean"],.5)
            first=next(iter(summary["per_parent"]["argformer"].values()))
            self.assertEqual(first["result"]["confusion"],[[5000,0,0],[5000,0,0],[0,0,0]])
            self.assertFalse(first["result"]["probabilistic"]["available"])

    def test_last_bad_prediction_never_resolves_any_truth(self):
        with tempfile.TemporaryDirectory(prefix="scoring_bad_",dir=str(scoring.ROOT/"experiments/results")) as temporary:
            args=self.fixture(Path(temporary));parents=list(args["prediction_refs_by_method"]["flare"])
            parent=parents[-1];seal_ref=args["prediction_refs_by_method"]["flare"][parent]
            seal=scoring._read(seal_ref)
            payload=scoring._load_npz(seal["prediction_ref"],8*2**20)
            payload["positions_bp0"]=payload["positions_bp0"].copy();payload["positions_bp0"][0]+=1
            broken=self.npz(Path(temporary)/parent/"flare/broken.npz",**payload)
            seal["prediction_ref"]=broken
            args["prediction_refs_by_method"]["flare"][parent]=self.write(Path(temporary)/parent/"flare/broken_seal.json",seal)
            spec=scoring.build_spec(**args)
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=""),patch.object(scoring,"_resolve_truth") as resolver:
                with self.assertRaises(ValueError):scoring.run(spec,resource_check=lambda _:None)
                resolver.assert_not_called()
            self.assertFalse((Path(temporary)/"scores/all45_prediction_barrier.json").exists())

    def test_halfopen_boundaries_all_query_columns_and_zero_probability_semantics(self):
        records=dict(parent_id=np.asarray("synthetic"),class_order=np.asarray(["EUR","EAS","AFR"]),
                     positions_bp0=np.asarray([0.,15000000.,29999999.]),sample_indices=np.asarray([600,800,1599],dtype=np.int64))
        labels=np.asarray([[0]*1000,[1]*1000],dtype=np.uint8)
        truth=dict(parent_id=records["parent_id"],class_order=records["class_order"],
                   intervals_bp=np.asarray([[0.,15000000.],[15000000.,30000000.]]),
                   query_sample_indices=np.arange(600,1600,dtype=np.int64),source_classes_0based=labels)
        y=scoring.join_truth(records,truth,resource_check=lambda _:None)
        np.testing.assert_array_equal(y,[0,1,1])
        labels[0,999]=255
        with self.assertRaises(ValueError):scoring.join_truth(records,truth,resource_check=lambda _:None)
        result=scoring.score_arrays(np.asarray([1]),dict(probabilities=np.asarray([[1.,0.,0.]],dtype=np.float32)),genuine_posterior=True)
        self.assertIsNone(result["probabilistic"]["NLL"])
        self.assertEqual(result["probabilistic"]["NLL_status"],"positive_infinity")
        self.assertFalse(result["probabilistic"]["probability_floor"])

    def test_equal_parent_sample_SD_uses_parent_unit(self):
        result=scoring._summary([.1,.3],"accuracy")
        self.assertAlmostEqual(result["mean"],.2)
        self.assertAlmostEqual(result["sample_SD"],math.sqrt(.02))
        self.assertAlmostEqual(result["sample_SD_percentage_points"],math.sqrt(.02)*100)
        self.assertEqual(result["parents"],2)

    def test_inert_import_in_fresh_cpu_interpreter(self):
        code=("import sys;sys.path.insert(0,%r); " % str(Path(scoring.__file__).resolve().parents[2])
              + "from arggnn.benchmarking import american_scoring; "
                "assert not any(k in sys.modules for k in ('numpy','torch','tskit','torch_geometric'))")
        subprocess.run([sys.executable,"-B","-s","-c",code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__=="__main__":unittest.main()
