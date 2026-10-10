"""Common observed-reference information, metadata and symmetry contracts."""
import unittest
import math
import torch
from arggnn.data import GraphSequence
from arggnn.models import (IndependentGNN,GNNSetPool,GNNGRU,IntervalLineageGNN,count_parameters)
from arggnn.reference_encoder import ReferenceFeatureAdapter,ReferenceGNN,ReferenceIntervalGNN


def example():
    x=torch.zeros(7,6,dtype=torch.float64)
    x[2,0],x[3,1]=1,1
    x[:2,2],x[:4,3]=1,1
    times=torch.tensor([0,0,0,0,10,20,50],dtype=torch.float64)
    branches=torch.tensor([10,10,20,50,10,30,0],dtype=torch.float64)
    x[:,4]=times.log1p()/math.log1p(10000)
    x[:,5]=branches.log1p()/math.log1p(10000)
    edges=torch.tensor([[4,0],[4,1],[5,4],[5,2],[6,5],[6,3]]).T
    return GraphSequence(x,torch.cat((edges,edges.flip(0)),dim=1),torch.zeros(7,dtype=torch.long),
        torch.tensor([0,1,2,3,-1,-1,-1]),torch.zeros(7,1,dtype=torch.float64),
        torch.tensor([[0.,1.]],dtype=torch.float64),torch.tensor([0,1]),
        torch.tensor([.15,.45,.8],dtype=torch.float64),torch.zeros(3,dtype=torch.long))


def changed(seq,**updates):
    values={k:None if v is None else v.clone() for k,v in seq.as_dict().items()}
    values.update(updates)
    return GraphSequence(**values)


class ReferenceAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.seq=example()
        self.kwargs=dict(reference_columns=(0,1),time_column=4,branch_column=5)
        self.adapter=ReferenceFeatureAdapter(6,**self.kwargs)

    def test_all_node_tmrca_counts_and_original_prefix(self):
        stats=self.adapter.node_statistics(self.seq)
        self.assertEqual(stats.features.shape,(7,2,10))
        expected=torch.tensor([[20,50],[20,50],[0,50],[50,0],[20,50],[20,50],[50,50]],dtype=torch.float64)
        physical=torch.expm1(stats.minimum_encoded_tmrca*math.log1p(10000))
        torch.testing.assert_close(physical,expected)
        torch.testing.assert_close(stats.descendant_reference_counts[6],torch.tensor([1.,1.],dtype=torch.float64))
        torch.testing.assert_close(stats.descendant_reference_counts[0],torch.zeros(2,dtype=torch.float64))
        torch.testing.assert_close(stats.nearest_votes[6],torch.tensor([.5,.5],dtype=torch.float64))
        augmented=self.adapter.precompute(self.seq)
        self.assertEqual(augmented.x.shape,(7,26))
        torch.testing.assert_close(augmented.x[:,:6],self.seq.x,rtol=0,atol=0)
        for name in ('edge_index','node_graph','node_ids','clades','intervals','query_node_ids','positions','target_graph'):
            self.assertIs(getattr(augmented,name),getattr(self.seq,name))
        self.assertIs(augmented.window_bounds,self.seq.window_bounds)

    def test_node_permutation_and_stable_id_renaming(self):
        permutation=torch.tensor([6,0,4,2,5,1,3])
        inverse=torch.empty_like(permutation)
        inverse[permutation]=torch.arange(7)
        seq=self.seq
        permuted=changed(seq,x=seq.x[permutation],edge_index=inverse[seq.edge_index],node_graph=seq.node_graph[permutation],
                         node_ids=seq.node_ids[permutation],clades=seq.clades[permutation])
        renamed=changed(seq,node_ids=torch.where(seq.node_ids>=0,100-seq.node_ids,seq.node_ids),query_node_ids=100-seq.query_node_ids)
        original=self.adapter.precompute(seq)
        torch.testing.assert_close(self.adapter.precompute(permuted).x,original.x[permutation],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(self.adapter.precompute(renamed).x,original.x,rtol=0,atol=0)

    def test_refinement_and_graph_serialization(self):
        seq=self.seq
        split=changed(seq,x=seq.x.repeat(2,1),edge_index=torch.cat((seq.edge_index,seq.edge_index+7),dim=1),
            node_graph=torch.tensor([0]*7+[1]*7),node_ids=seq.node_ids.repeat(2),clades=seq.clades.repeat(2,1),
            intervals=torch.tensor([[0.,.3],[.3,1.]],dtype=torch.float64),target_graph=torch.tensor([0,1,1]))
        augmented=self.adapter.precompute(seq)
        split_augmented=self.adapter.precompute(split)
        torch.testing.assert_close(split_augmented.x,augmented.x.repeat(2,1),rtol=0,atol=0)
        reordered=changed(split,node_graph=1-split.node_graph,intervals=split.intervals.flip(0),target_graph=1-split.target_graph)
        torch.testing.assert_close(self.adapter.precompute(reordered).x,split_augmented.x,rtol=0,atol=0)
        model=ReferenceIntervalGNN(6,2,hidden_dim=8,num_layers=2,**self.kwargs).double().eval()
        torch.testing.assert_close(model(augmented),model(split_augmented),rtol=1e-10,atol=1e-10)

    def test_kernel_log_numerics_at_tiny_scale(self):
        adapter=ReferenceFeatureAdapter(6,**self.kwargs,kernel_scales=(1e-6,))
        stats=adapter.node_statistics(self.seq)
        self.assertTrue(bool(torch.isfinite(stats.features).all()))
        torch.testing.assert_close(stats.kernel_evidence[6,:,0],torch.tensor([.5,.5],dtype=torch.float64))
        torch.testing.assert_close(stats.kernel_evidence.sum(dim=1),torch.ones(7,1,dtype=torch.float64))

    def test_wrappers_share_augmented_information_and_finite_gradients(self):
        augmented=self.adapter.precompute(self.seq)
        for cls in (ReferenceGNN,ReferenceIntervalGNN):
            model=cls(6,2,hidden_dim=8,num_layers=2,**self.kwargs).double()
            result=model(augmented)
            self.assertEqual(result.shape,(3,2,2))
            result.square().mean().backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            with self.assertRaises(ValueError):
                model(self.seq)
        wrapped=ReferenceGNN(6,2,hidden_dim=8,num_layers=2,**self.kwargs)
        direct=IndependentGNN(self.adapter.output_dim,2,hidden_dim=8,num_layers=2)
        self.assertEqual(count_parameters(wrapped),count_parameters(direct))
        self.assertEqual(len(list(self.adapter.parameters())),0)

    def test_common_family_input_compatibility(self):
        from arggnn.temporal_baselines import LeafTemporalTransformer,LeafCorrespondenceSupraMPNN
        from arggnn.graphssm_baseline import GraphSSMAdaptation
        augmented=self.adapter.precompute(self.seq)
        for cls in (IndependentGNN,GNNSetPool,GNNGRU,IntervalLineageGNN,
                    LeafTemporalTransformer,LeafCorrespondenceSupraMPNN,GraphSSMAdaptation):
            model=cls(self.adapter.output_dim,2,hidden_dim=8,num_layers=2).double().eval()
            output=model(augmented)
            self.assertEqual(output.shape,(3,2,2))
            self.assertTrue(bool(torch.isfinite(output).all()))

    def test_manifest_provenance_and_reaugmentation_rejection(self):
        names=['ref_A','ref_B','query','leaf','log_time','log_branch']
        provenance=self.adapter.manifest_provenance(names,'1'*64)
        self.assertEqual(provenance['original_feature_names'],names)
        self.assertEqual(provenance['added_feature_names'][0],'ref_A.descendant_reference_fraction')
        self.assertEqual(len(provenance['added_feature_names']),20)
        self.assertEqual(provenance['learnable_adapter_parameters'],0)
        self.assertEqual(provenance['truth_annotations_used'],[])
        self.assertEqual(len(provenance['configuration_sha256']),64)
        self.assertEqual(len(provenance['implementation_sha256']),64)
        with self.assertRaises(ValueError):
            self.adapter.precompute(self.adapter.precompute(self.seq))
        with self.assertRaises(ValueError):
            self.adapter.manifest_provenance(names,'bad')

    def test_reference_vocabulary_can_differ_from_output_classes(self):
        # Two observed panel categories can support a downstream three-class head.
        model=ReferenceGNN(6,3,hidden_dim=8,num_layers=2,**self.kwargs).double()
        self.assertEqual(model(model.prepare_sequence(self.seq)).shape,(3,2,3))


if __name__=='__main__':
    unittest.main()
