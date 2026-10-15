"""Synthetic-only FP32 CPU and mocked metadata checks for shared C3 adapter."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
import ast,hashlib,json,os,unittest
import numpy as np
import torch
from torch.nn import functional as F
from arggnn import american_c3_gnn_shared30mb_v1 as m
from arggnn.american_c3_gnn_dataset_v1 import AmericanC3BlockV1
from arggnn.american_c3_gnn_observations_v1 import encode_block
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
def reference(name):return {"path":"synthetic/"+name,"sha256":"0"*64}
def plan_fixture():
    cfg=m.shared_plan_config_v1(observation_ref=reference("observed.json"),
        focal_registry_ref=reference("focals.npz"),native_preparation_ref=reference("native.json"))
    return dict(schema="american_c3_whole_tree_block_index_v1",parent_id=m.VAL[0],
        observation_ref=cfg["observation_ref"],focal_registry_ref=cfg["focal_registry_ref"],config=cfg,
        sequence_length_bp=30000000,class_order=["EUR","EAS","AFR"],haplotypes=1600,
        original_reference_counts=[200,200,200],effective_reference_counts=[200,200,200],
        focal_records=10000,all_trees_and_samples_retained=True,targets_opened=False,trees=3,sites=2,
        blocks=[dict(start_tree=i,stop_tree=i+1,nodes=1602,sites=int(i<2)) for i in range(3)])
class Synthetic(unittest.TestCase):
    def test_exact_dimensions_and_no_dummy_class(self):
        d=m.dimensions_v1()
        self.assertEqual((d["site_input_width"],d["observed_leaf_width"],d["output_classes"]),(15,5,3))
        self.assertEqual((d["haplotypes"],d["reference_haplotypes"],d["query_haplotypes"]),(1600,600,1000))
        self.assertFalse(d["dummy_fourth_class"]);self.assertFalse(d["posterior_floor_or_renormalization"])
    def test_count_resource_plan_preserves_every_block_and_whole_history(self):
        p=plan_fixture();r=m.resource_plan_v1(p)
        self.assertEqual((r["trees"],r["sites"],r["blocks"],r["total_active_marginal_node_occurrences"]),(3,2,3,4806))
        self.assertEqual(r["CPU_focal_forward_state_bytes"],1280000)
        self.assertEqual(r["one_direction_recurrent_state_bytes"],204800)
        self.assertTrue(r["count_plan_is_not_admission_or_fit_readiness"])
        bad=dict(p,sequence_length_bp=2000000)
        with self.assertRaises(ValueError):m.resource_plan_v1(bad)
    def test_same_observation_native_focal_binding_and_query_truth_rejected(self):
        p=plan_fixture();cfg=p["config"];parent=m.VAL[0]
        observed=dict(schema="american_c3_observed_v1",parent_id=parent,split="val",sequence_length_bp=30000000,
            class_order=["EUR","EAS","AFR"],sample_unit="haploid",
            counts_haploid={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000},query_ancestry_truth_in_inputs=False,
            genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG")
        prep=dict(status="completed_native_C3_path_preparation",parent_id=parent,
            observation_ref=cfg["observation_ref"],focals_ref=cfg["focal_registry_ref"],
            class_order=observed["class_order"],genealogy_kind=observed["genealogy_kind"],
            query_ancestry_truth_opened=False,query_focal_count=10000)
        def read(r):return observed if r==cfg["observation_ref"] else prep
        with patch.object(m,"original_registry_v1",return_value={parent:{"split":"val"}}),patch.object(m,"read",side_effect=read):
            self.assertFalse(m.validate_shared_metadata_v1(cfg,parent_id=parent)["scientific_payload_opened"])
            observed["query_ancestry_truth_in_inputs"]=True
            with self.assertRaises(ValueError):m.validate_shared_metadata_v1(cfg,parent_id=parent)
    def test_TEST_and_TRAIN_denied_before_checkpoint_or_truth_resolution(self):
        with patch.object(m,"read",side_effect=AssertionError("must not read")),patch.object(m,"eligible_checkpoint_v1",side_effect=AssertionError("must not load checkpoint")):
            for parent in (m.TEST[0],m.TRAIN[0]):
                with self.assertRaises(ValueError):
                    m.seal_shared_val_prediction_v1(None,None,None,parent_id=parent,checkpoint_ref=None,
                        training_completion_ref=None,writer=None,device="cpu",runtime_guard=None)
    def test_existing_training_and_numeric_engines_reused(self):
        from arggnn.american_c3_gnn_train_v1 import run_train_v1
        self.assertIs(m.run_development_train_v1,run_train_v1)
        s=Path(m.__file__).read_text();ast.parse(s)
        self.assertNotIn("private_truth_ref",s);self.assertNotIn("simulated_true_with_migrations",s)
        self.assertNotIn(".softmax",s);self.assertNotIn("optimizer.step",s)
        self.assertIn("return seal_prediction_v1(",s)
    def test_existing_prediction_seal_schema_preserved_with_disclosed_branch(self):
        model=m.AmericanC3WholeTreeGNNV1(**m.MODEL_KWARGS)
        plan=plan_fixture()
        forward=NS(dataset=NS(plan_ref=reference("plan.json"),plan=plan,direction="forward"))
        reverse=NS(dataset=NS(plan_ref=reference("plan.json"),plan=plan,direction="reverse"))
        training={"trained_parent_ids":["american_c3_short_train_%03d"%i for i in range(20)],
            "train_sequence_lengths_bp":[2000000]*20}
        saved=[]
        class Writer:
            def save(self,name,value,kind="json"):saved.append((name,value,kind));return reference(name)
        def original_seal(*args,**kw):
            self.assertEqual(kw["split"],"val")
            return kw["writer"].save("original_seal.json",{"status":"sealed_C3_original_heldout_predictions",
                "protocol_sha256":"original","focal_identity_sha256":"unchanged","probabilities_modified":False})
        with patch.object(m,"eligible_checkpoint_v1",return_value=(training,{"epoch":30})),patch.object(m,"validate_shared_metadata_v1"),patch.object(m,"seal_prediction_v1",side_effect=original_seal):
            m.seal_shared_val_prediction_v1(model,forward,reverse,parent_id=m.VAL[0],
                checkpoint_ref=reference("checkpoint.pt"),training_completion_ref=reference("completed.json"),
                writer=Writer(),device="cpu",runtime_guard=lambda event:None)
        seal=saved[0][1]
        self.assertEqual(seal["status"],"sealed_C3_original_heldout_predictions")
        self.assertEqual(seal["protocol_sha256"],"original")
        self.assertEqual(seal["focal_identity_sha256"],"unchanged")
        self.assertEqual(seal["training_branch"],m.DEVELOPMENT_BRANCH)
        self.assertFalse(seal["probabilities_modified"])
        self.assertFalse(seal["original_full30Mb_matched_training_claimed"])
    def test_full1600_haplotype_three_class_whole_tree_FP32_CPU(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"),"")
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70001)
            model=m.AmericanC3WholeTreeGNNV1(**m.MODEL_KWARGS)
        h=1600;parent=np.full(h+2,h+1,dtype=np.int64);parent[0]=h;parent[-1]=-1
        clocks=np.zeros(h+2);clocks[h]=5.;clocks[h+1]=20.
        tree=NativeObservedTreeV1(parent,clocks,np.arange(h,dtype=np.int64),np.arange(h,dtype=np.int64))
        genotype=((np.arange(2)[:,None]+np.arange(h)[None,:])%2).astype(np.int8)
        refs=np.asarray([2]*200+[0]*200+[1]*200+[-1]*1000,dtype=np.int64)
        bank,receipt=encode_block([tree,tree],np.asarray([[0.,15000000.],[15000000.,30000000.]]),
            genotype,np.asarray([2,2],dtype=np.int8),np.asarray([1.,15000001.]),refs,np.zeros(h),
            num_classes=3,query_columns=np.arange(600,1600,dtype=np.int64))
        self.assertEqual(bank.sample_index.shape,(2,1600));self.assertEqual(len(bank.parent),3204)
        self.assertEqual(bank.log_donor_allele_frequency.shape,(2,2,6))
        self.assertEqual(receipt["haplotypes"],1600);self.assertTrue(receipt["unary_branches_preserved"])
        geometry=torch.zeros((2,3),dtype=torch.float32)
        block=AmericanC3BlockV1(bank,parent_id="synthetic_only",start_tree=0,stop_tree=2,
            sequence_length_bp=30000000,complete_parent=True,genomic_geometry=geometry,
            focal_rows=torch.arange(1000),focal_hold=torch.zeros(1000,dtype=torch.long),
            focal_sample_column=torch.arange(600,1600))
        logits=model.forward_complete(block)
        self.assertEqual(logits.shape,(2,1600,3))
        labels=torch.arange(1000)%3
        loss=F.cross_entropy(logits[0,600:],labels);loss.backward()
        grads=[p.grad for p in model.parameters()]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
        self.assertTrue(any(bool((g!=0).any()) for g in grads))
        model.zero_grad(set_to_none=True);model.eval()
        with torch.inference_mode():
            context=model.encode_bank(bank)
            dense_f,_=model.scan_context(context,geometry,direction="forward")
            dense_r,_=model.scan_context(context,geometry,direction="reverse")
            ff0,hidden=model.scan_context(context[:1],geometry[:1],direction="forward")
            ff1,_=model.scan_context(context[1:],geometry[1:],direction="forward",hidden=hidden)
            rr1,hidden=model.scan_context(context[1:],geometry[1:],direction="reverse")
            rr0,_=model.scan_context(context[:1],geometry[:1],direction="reverse",hidden=hidden)
            self.assertTrue(torch.allclose(dense_f,torch.cat([ff0,ff1]),atol=1e-5,rtol=1e-5))
            self.assertTrue(torch.allclose(dense_r,torch.cat([rr0,rr1]),atol=1e-5,rtol=1e-5))
            prob=logits[:,600:].softmax(-1).numpy()
            self.assertEqual(prob.shape,(2,1000,3));self.assertEqual(prob.dtype,np.float32)
            self.assertTrue(np.allclose(prob.sum(-1),1.,atol=1e-6))
        self.__class__.model_count=dict(parameters=sum(p.numel() for p in model.parameters()),
            FP32_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            synthetic_trees=2,synthetic_whole_nodes=3204,haplotypes=1600,
            classes=3,all_parameter_gradients_finite=True,full_vs_two_block_directional_carry_pass=True)
if __name__=="__main__":unittest.main()
