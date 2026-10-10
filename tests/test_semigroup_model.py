"""Established flow laws and candidate graph-field representation contracts."""
import unittest
from types import SimpleNamespace
import torch
from arggnn.semigroup_model import (SemigroupGraphModel,SpanSetSemigroupModel,
    DissipativeGenerator,augmented_affine_generator,affine_transition,apply_affine_transition)


def example():
    generator=torch.Generator().manual_seed(81)
    return SimpleNamespace(x=torch.randn(7,3,generator=generator,dtype=torch.float64),
        edge_index=torch.tensor([[0,2,1,2,3,6,4,6,5,6],[2,0,2,1,6,3,6,4,6,5]]),
        node_graph=torch.tensor([0,0,0,1,1,1,1]),node_ids=torch.tensor([0,1,-1,0,1,2,-1]),
        clades=torch.tensor([[0,0],[1,0],[1,0],[0,0],[1,0],[0,1],[1,1]],dtype=torch.float64),
        intervals=torch.tensor([[0.,.4],[.4,1.]],dtype=torch.float64),query_node_ids=torch.tensor([0,1]),
        positions=torch.tensor([.1,.25,.65,.9],dtype=torch.float64),target_graph=torch.tensor([0,0,1,1]),
        window_bounds=torch.tensor([0.,1.],dtype=torch.float64))


def changed(seq,**updates):
    values={k:None if v is None else v.clone() for k,v in vars(seq).items()}
    values.update(updates)
    return SimpleNamespace(**values)


def split_first(seq):
    edges0=seq.edge_index[:,seq.node_graph[seq.edge_index[0]]==0]
    edges1=seq.edge_index[:,seq.node_graph[seq.edge_index[0]]==1]+3
    return changed(seq,x=torch.cat((seq.x[:3],seq.x[:3],seq.x[3:])),
        edge_index=torch.cat((edges0,edges0+3,edges1),dim=1),
        node_graph=torch.tensor([0]*3+[1]*3+[2]*4),node_ids=torch.cat((seq.node_ids[:3],seq.node_ids[:3],seq.node_ids[3:])),
        clades=torch.cat((seq.clades[:3],seq.clades[:3],seq.clades[3:])),
        intervals=torch.tensor([[0.,.15],[.15,.4],[.4,1.]],dtype=torch.float64),target_graph=torch.tensor([0,1,2,2]))


class SemigroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.seq=example()

    def model(self,**kwargs):
        return SemigroupGraphModel(3,3,hidden_dim=8,num_layers=2,latent_dim=3,generator_rank=2,
                                   initial_rate=2.,**kwargs).double().eval()

    def test_exact_affine_semigroup_law_and_zero_time(self):
        a=torch.tensor([[-1.,.3],[-.3,-2.]],dtype=torch.float64)
        b=torch.tensor([.7,-.2],dtype=torch.float64)
        m=augmented_affine_generator(a,b)
        whole=affine_transition(m,.7)
        split=affine_transition(m,.4)@affine_transition(m,.3)
        torch.testing.assert_close(whole,split,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(affine_transition(m,0.),torch.eye(3,dtype=torch.float64))
        with self.assertRaises(ValueError):
            affine_transition(m,-.1)

    def test_diagonal_affine_forcing_already_encodes_order(self):
        a=torch.tensor([[-1.]],dtype=torch.float64)
        m1=augmented_affine_generator(a,torch.tensor([1.],dtype=torch.float64))
        m2=augmented_affine_generator(a,torch.tensor([2.],dtype=torch.float64))
        self.assertGreater(float((m1@m2-m2@m1).abs().max()),0)
        z=torch.zeros(1,dtype=torch.float64)
        first=apply_affine_transition(affine_transition(m2,.3),apply_affine_transition(affine_transition(m1,.4),z))
        second=apply_affine_transition(affine_transition(m1,.4),apply_affine_transition(affine_transition(m2,.3),z))
        expected=(1-torch.exp(torch.tensor(-.4)))*(1-torch.exp(torch.tensor(-.3)))
        self.assertAlmostEqual(float((first-second).abs()),float(expected),places=7)

    def test_full_matrix_dissipativity_and_contraction(self):
        generator=DissipativeGenerator(8,latent_dim=3,rank=2,initial_rate=2.).double()
        h=torch.randn(5,2,8,dtype=torch.float64)
        a,b=generator(h)
        symmetric=(a+a.transpose(-1,-2))/2
        maximum=torch.linalg.eigvalsh(symmetric).max()
        self.assertLess(float(maximum),0)
        homogeneous=torch.matrix_exp(a*.6)
        bound=torch.exp(-generator.rate*generator.minimum_damping*.6)
        self.assertTrue(bool((torch.linalg.matrix_norm(homogeneous,ord=2)<=bound+1e-10).all()))
        self.assertTrue(bool(torch.isfinite(b).all()))

    def test_shapes_gradients_and_diagonal_control(self):
        for kind in ('full','diagonal'):
            model=self.model(matrix_kind=kind)
            result=model(self.seq)
            self.assertEqual(result.shape,(4,2,3))
            result.square().mean().backward()
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            if kind=='diagonal':
                a=model.generator_bank(self.seq).a
                torch.testing.assert_close(a,torch.diag_embed(a.diagonal(dim1=-2,dim2=-1)))
                self.assertEqual(model.generator.inactive_parameter_count(),0)
                self.assertIsNone(model.generator.u)
                self.assertIsNone(model.generator.v)
                self.assertIsNone(model.generator.w)
            else:
                self.assertEqual(model.generator.inactive_parameter_count(),0)

    def test_refinement_invariance_at_common_positions(self):
        split=split_first(self.seq)
        for kind in ('full','diagonal'):
            for aggregation in ('ordered','span_set','homogeneous_set'):
                model=self.model(matrix_kind=kind,aggregation=aggregation)
                torch.testing.assert_close(model(self.seq),model(split),rtol=1e-10,atol=1e-10)
                before=model.encode_context(self.seq);after=model.encode_context(split)
                torch.testing.assert_close(before.forward,after.forward,rtol=1e-10,atol=1e-10)
                torch.testing.assert_close(before.backward,after.backward,rtol=1e-10,atol=1e-10)

    def test_node_graph_identity_permutations_and_reflection(self):
        seq=self.seq;permutation=torch.tensor([6,0,3,1,5,2,4]);inverse=torch.empty_like(permutation)
        inverse[permutation]=torch.arange(7)
        permuted=changed(seq,x=seq.x[permutation],edge_index=inverse[seq.edge_index],
            node_graph=seq.node_graph[permutation],node_ids=seq.node_ids[permutation],clades=seq.clades[permutation])
        renamed=changed(seq,node_ids=torch.where(seq.node_ids>=0,100-seq.node_ids,seq.node_ids),query_node_ids=100-seq.query_node_ids)
        serialized=changed(seq,node_graph=1-seq.node_graph,intervals=seq.intervals.flip(0),target_graph=1-seq.target_graph)
        reflected=changed(seq,intervals=torch.stack((1-seq.intervals[:,1],1-seq.intervals[:,0]),dim=1),positions=1-seq.positions)
        model=self.model()
        for case in (permuted,renamed,serialized,reflected):
            torch.testing.assert_close(model(seq),model(case),rtol=1e-10,atol=1e-10)

    def test_set_discards_arrangement_ordered_retains_it(self):
        seq=self.seq
        first_edges=seq.edge_index[:,seq.node_graph[seq.edge_index[0]]==0]
        three=changed(seq,x=torch.cat((seq.x,seq.x[:3]+.5)),edge_index=torch.cat((seq.edge_index,first_edges+7),dim=1),
            node_graph=torch.tensor([0]*3+[1]*4+[2]*3),node_ids=torch.cat((seq.node_ids,seq.node_ids[:3])),
            clades=torch.cat((seq.clades,seq.clades[:3])),
            intervals=torch.tensor([[0.,1/3],[1/3,2/3],[2/3,1.]],dtype=torch.float64),
            positions=torch.tensor([.2,.5,.9],dtype=torch.float64),target_graph=torch.tensor([0,1,2]))
        rearranged=changed(three,intervals=three.intervals[torch.tensor([1,0,2])],target_graph=torch.tensor([1,0,2]))
        bag=self.model(aggregation='span_set')
        torch.testing.assert_close(bag.encode_context(three).forward,bag.encode_context(rearranged).forward,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(bag(three)[2],bag(rearranged)[2],rtol=1e-10,atol=1e-10)
        ordered=self.model(aggregation='ordered')
        difference=(ordered.encode_context(three).forward[-1]-ordered.encode_context(rearranged).forward[-1]).abs().max()
        self.assertGreater(float(difference),1e-6)

    def test_homogeneous_set_matches_ordered_on_constant_generator_field(self):
        seq=self.seq
        # An unchanged original tree and its duplicate have a constant bank.
        nodes=torch.tensor([0,1,2])
        edges=seq.edge_index[:,seq.node_graph[seq.edge_index[0]]==0]
        one=changed(seq,x=seq.x[nodes],edge_index=edges,node_graph=torch.zeros(3,dtype=torch.long),
                    node_ids=seq.node_ids[nodes],clades=seq.clades[nodes],
                    intervals=torch.tensor([[0.,1.]],dtype=torch.float64),target_graph=torch.zeros(4,dtype=torch.long))
        ordered=self.model(aggregation='ordered')
        homogeneous=self.model(aggregation='homogeneous_set')
        homogeneous.load_state_dict(ordered.state_dict())
        torch.testing.assert_close(ordered(one),homogeneous(one),rtol=1e-10,atol=1e-10)
        repeated=changed(one,x=one.x.repeat(2,1),edge_index=torch.cat((one.edge_index,one.edge_index+3),dim=1),
            node_graph=torch.tensor([0]*3+[1]*3),node_ids=one.node_ids.repeat(2),clades=one.clades.repeat(2,1),
            intervals=torch.tensor([[0.,.3],[.3,1.]],dtype=torch.float64),target_graph=torch.tensor([0,0,1,1]))
        torch.testing.assert_close(ordered(repeated),homogeneous(repeated),rtol=1e-10,atol=1e-10)

    def test_scalar_regression_and_configuration_provenance(self):
        model=SemigroupGraphModel(3,1,hidden_dim=8,num_layers=2,latent_dim=3,generator_rank=2,matrix_kind='diagonal').double()
        output=model(self.seq)
        self.assertEqual(output.shape,(4,2,1))
        output.square().mean().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
        manifest=model.configuration_manifest()
        self.assertEqual(manifest['num_outputs'],1)
        self.assertEqual(manifest['structurally_inactive_parameters'],0)
        self.assertEqual(len(manifest['configuration_sha256']),64)
        self.assertEqual(len(manifest['implementation_sha256']),64)

    def test_unidirectional_and_set_wrapper(self):
        self.assertEqual(self.model(bidirectional=False)(self.seq).shape,(4,2,3))
        model=SpanSetSemigroupModel(3,3,hidden_dim=8,num_layers=2,latent_dim=3,generator_rank=2).double()
        self.assertEqual(model.aggregation,'span_set')
        self.assertEqual(model(self.seq).shape,(4,2,3))


if __name__=='__main__':
    unittest.main()
