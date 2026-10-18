"""Four-head shared-coordinate contracts and CPU private-binding parity."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arggnn.benchmarking import european_c4_training as current


class CurrentC4TrainingTest(unittest.TestCase):
    def test_native_registry_keeps_every_original_identity(self):
        import numpy as np
        original=dict(record_ids=np.asarray([9,2,7],dtype=np.int64),
            sample_indices=np.asarray([266,267,266]),sample_nodes=np.asarray([500,501,500]),
            positions_bp0=np.asarray([50.,150.,50.]),parent_id=np.asarray("train"),
            class_order=np.asarray(current.observed.NATIVE_SIX_CLASSES))
        with patch.object(current.observed,"_sampled_records",return_value=original):
            view=current._sampled_records({})
        self.assertEqual(view["class_order"].tolist(),list(current.CLASSES))
        self.assertEqual(original["class_order"].tolist(),list(current.observed.NATIVE_SIX_CLASSES))
        for key in original.keys()-{"class_order"}:self.assertIs(view[key],original[key])

    def test_exact_shared_three_train_five_test_plan_only(self):
        plan=dict(schema="ap_paper_sampled_plan_v1",train_parent_ids=["train0","train1","train2"],
            test_parent_ids=["test"+str(i) for i in range(5)],training_total=50000,
            testing_per_parent=10000,native_path_names=list(current.observed.NATIVE_SIX_CLASSES))
        with patch.object(current.original.gpu,"_public_ref"),patch.object(current.c,"read_ref",return_value=plan):
            self.assertIs(current._parent_plan({},current.PROTOCOL),plan)
            for key,value in (("training_total",49999),("native_path_names",list(current.CLASSES))):
                with patch.object(current.c,"read_ref",return_value=dict(plan,**{key:value})):
                    with self.assertRaises(current.c.ContractError):current._parent_plan({},current.PROTOCOL)
            with self.assertRaises(current.c.ContractError):current._parent_plan({},"ancestralpaths_paper200mb_native6")

    def test_source_endpoint_target_binding_and_repeated_draws(self):
        import numpy as np
        import torch
        registry=dict(path="mock/registry.json",sha256="a"*64)
        metadata=dict(schema="european_release_C4_sampled_targets_v1",parent_id="train",split="train",
            registry_ref=registry,source_class_order=list(current.CLASSES),rows=3,
            arrays_ref=dict(path="mock/targets.npz",sha256="b"*64),native_targets_ref={},source_truth_ref={})
        bundle=SimpleNamespace(sampled_registry_ref=registry,focal_hold=torch.tensor([2,0,2]),
            focal_sample_column=torch.tensor([268,266,268]),bank=SimpleNamespace(query_column=torch.arange(266,952)))
        model=SimpleNamespace(output_classes=current.CLASSES)
        for key,value in (("schema","ap_paper_TRAIN_native_targets_v1"),("parent_id","other"),
                          ("split","test"),("registry_ref",{}),
                          ("source_class_order",list(current.observed.NATIVE_SIX_CLASSES))):
            with patch.object(current.observed,"_json_ref",return_value=dict(metadata,**{key:value})), \
                 patch.object(current.observed,"_bound",side_effect=AssertionError("array must stay closed")):
                with self.assertRaises(current.c.ContractError):current._targets({},"train",bundle,model,max_metadata_bytes=2**20)
        with tempfile.TemporaryDirectory(dir=current.c.ROOT/"tests") as directory:
            path=Path(directory)/"labels.npz";np.savez(path,source_labels_0based=np.asarray([3,0,3],dtype=np.int64))
            with patch.object(current.observed,"_json_ref",return_value=metadata),patch.object(current.observed,"_bound",return_value=path):
                targets=current._targets({},"train",bundle,model,max_metadata_bytes=2**20)
            self.assertEqual(targets.labels,(3,0,3));self.assertEqual(targets.readout_rows,(2,0,2))
            self.assertEqual(targets.query_columns,(2,0,2))
            targets.validate_binding(parent_id="train",split="train",sampled_registry_ref=registry)

    def test_private_four_head_complete_parent_binding_and_executor(self):
        import numpy as np
        import torch
        import tskit
        torch.set_num_threads(2);tables=tskit.TableCollection(300.)
        for i in range(952):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
        clade=tables.nodes.add_row(time=50.);root=tables.nodes.add_row(time=1000.)
        for a,b,pair in ((0.,100.,(0,1)),(100.,200.,(0,2)),(200.,300.,(0,1))):
            for i in range(952):tables.edges.add_row(a,b,clade if i in pair else root,i)
            tables.edges.add_row(a,b,root,clade)
        for pos,node in ((50.,300),(250.,301)):
            site=tables.sites.add_row(pos,ancestral_state="A");tables.mutations.add_row(site,node,derived_state="C")
        tables.sort();ts=tables.tree_sequence();parent="synthetic_C4_train"
        refs=dict(inference_ref=dict(path="mock/inference.json",sha256="a"*64))
        state=dict(manifest=dict(parent_id=parent,split="train",sequence_length_bp=200000000,truth_or_oracle_read=False),
            ts=ts,rows=[dict(sample_index=i) for i in range(952)],nodes=np.arange(952,dtype=np.int64),
            genotype=np.asarray(ts.genotype_matrix(),dtype=np.int8),positions=np.asarray([50.,250.]),
            allele_count=np.asarray([2,2],dtype=np.int8),labels=np.asarray([i%4 if i<266 else -1 for i in range(952)]),
            ages=np.zeros(952),query_columns=np.arange(266,952,dtype=np.int64),coordinate_offset_bp=1000.,source_refs=refs)
        records=dict(record_ids=np.arange(3,dtype=np.int64),sample_nodes=np.asarray([266,267,266]),
            sample_indices=np.asarray([266,267,266]),positions_bp0=np.asarray([1050.,1250.,1050.]),
            parent_id=np.asarray(parent),class_order=np.asarray(current.observed.NATIVE_SIX_CLASSES))
        base_loader=current.original._training_targets;base_prepare=current.observed.prepare_observation
        bound=current._bound();view=bound.european
        self.assertIsNot(bound,current.original);self.assertIsNot(view,current.observed)
        self.assertIs(current.original._training_targets,base_loader)
        self.assertIs(current.observed.prepare_observation,base_prepare)
        with patch.object(current.observed,"_sampled_records",return_value=records):
            bundle,metadata=view.prepare_observation(state,protocol_id=current.PROTOCOL,parent_id=parent,split="train",
                source_refs=refs,sampled_registry_ref=dict(path="mock/registry.json",sha256="b"*64),
                max_nodes=10000,max_holds=10,max_episode_payload_bytes=2**20,resource_check=lambda *args:None)
        self.assertIs(type(bundle),view.EuropeanWholeParent);self.assertEqual(bundle.focal_hold.tolist(),[0,2,0])
        model=view.build_model(current.PROTOCOL)
        self.assertEqual(sum(p.numel() for p in model.parameters()),32325)
        self.assertEqual(model.output_classes,current.CLASSES)
        with torch.no_grad():
            for adapter in (model.memory,model.clade_coupling,model.episode_coupling):adapter.O.weight.normal_(0.,.02)
            model.eval();complete=model(bundle.bank,episodes=bundle.original)
            expected=complete[bundle.focal_hold,torch.tensor([0,1,0])]
            actual=view._executor_class()(model,bundle,metadata,holds_per_block=2,haplotypes_per_gru=8,
                resource_check=lambda *args:None,history_holds_per_window=2).forward()
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(actual[0],actual[2],atol=0.,rtol=0.)

    def test_factories_and_inert_import(self):
        expected=dict(requirements=dict(source_refs=[]))
        for factory,builder,function in ((current.fit_factory,"build_fit_spec","run_fit"),
                                         (current.prediction_factory,"build_prediction_spec","run_predictions")):
            with patch.object(current,builder,return_value=expected):descriptor=factory(test_argument=True)
            self.assertEqual(descriptor["module"],"arggnn.benchmarking.european_c4_training")
            self.assertEqual(descriptor["function"],function)
        code="import sys;sys.path[:0]=%r;from arggnn.benchmarking import european_c4_training;assert not any(x in sys.modules for x in ('torch','numpy','tskit','torch_geometric'))" % [str(current.c.ROOT/"src"),str(current.c.ROOT)]
        subprocess.run([sys.executable,"-s","-B","-c",code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__=="__main__":unittest.main()
