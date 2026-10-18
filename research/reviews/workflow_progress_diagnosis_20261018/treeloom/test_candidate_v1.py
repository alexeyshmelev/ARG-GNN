"""Tiny complete CPU forests only; no real observed file/model/GPU execution."""
from dataclasses import fields
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from . import candidate_v1 as candidate,indices_v1 as ix,cache_v1 as cache
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_observations_v1 as original
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import (
    episode_preparation_v1 as ep,cut_preparation_v1 as cp)
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1

HERE=Path(__file__).resolve().parent


def fixture(isolated=True):
    age=np.array([0.,400.,0.,400.,0.,100.,349.,351.])
    trees=[]
    for at in range(4):
        if at==2 and isolated:
            trees.append(NativeObservedTreeV1(np.full(8,-1,dtype=np.int64),age.copy(),np.arange(8),np.arange(8)))
            continue
        unary=at==1
        n=12 if unary else 11;root=n-1
        parent=np.full(n,root,dtype=np.int64);parent[root]=-1
        pair=(4,6) if at==3 else (4,5)
        parent[list(pair)]=8;parent[[0,2]]=9
        if unary:parent[8]=10
        time=np.r_[age,350. if at==3 else 200.,250.,*([300.] if unary else []),1000.]
        trees.append(NativeObservedTreeV1(parent,time,np.arange(8),np.arange(8)))
    intervals=np.array([[0.,100.],[100.,200.],[200.,300.],[300.,400.]])
    positions=np.array([50.,150.,350.]) if isolated else np.array([50.,150.,250.,350.])
    g=np.random.default_rng(9).integers(0,2,size=(len(positions),8),dtype=np.int8)
    return trees,intervals,g,np.full(len(g),2,dtype=np.int8),positions,np.array([0,1,2,0,-1,-1,-1,-1]),age


def prepare_original(values):
    bank,meta=original.encode_block(*values,num_classes=3,query_columns=np.arange(4,8))
    isolated=tuple(meta.get("isolated_zero_site_holds",()))
    arrays,episode_meta=ep.build_clade_episode_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy(),
        bank.spans.numpy(),max_payload_bytes=2**20,isolated_hold_indices=isolated)
    episodes=OrderedCladeEpisodesV1.from_verified_arrays(arrays,episode_meta)
    cut,cut_meta=cp.prepare_cut_plan_v1(bank,expected_haplotypes=8,max_nodes=1000,max_holds=10,
                                     isolated_hold_indices=isolated)
    return bank,episodes,cut,dict(observation=meta,episodes=episode_meta,cut=cut_meta)


class ExactPreparationCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tensor(self,a,b):
        self.assertEqual(a.dtype,b.dtype);self.assertEqual(a.shape,b.shape)
        self.assertEqual(a.numpy().tobytes(),b.numpy().tobytes())

    def assert_bank(self,a,b):
        for f in fields(a):
            x,y=getattr(a,f.name),getattr(b,f.name)
            if isinstance(x,tuple):
                self.assertEqual(len(x),len(y))
                for left,right in zip(x,y):self.assert_tensor(left,right)
            else:self.assert_tensor(x,y)

    def assert_structures(self,left,right):
        self.assert_bank(left[0],right[0])
        self.assert_tensor(left[1].flags.node_flags,right[1].flags.node_flags)
        for f in fields(left[1]):
            if f.name!="flags":self.assert_tensor(getattr(left[1],f.name),getattr(right[1],f.name))
        for a,b in zip(left[2].tensors(),right[2].tensors()):self.assert_tensor(a,b)
        self.assertEqual(left[2].group_offsets,right[2].group_offsets)
        self.assertEqual(left[2].episode_of_hold,right[2].episode_of_hold)
        self.assertEqual(left[2].source_fingerprints,right[2].source_fingerprints)
        self.assertEqual(left[3],right[3])

    def test_complete_connected_and_isolated_unary_cut_episode_byte_parity(self):
        for isolated in (False,True):
            with self.subTest(isolated=isolated):
                values=fixture(isolated)
                expected=prepare_original(values)
                actual=candidate.prepare_synthetic_or_admitted(*values,num_classes=3,
                    query_columns=np.arange(4,8),max_nodes=1000,max_holds=10,
                    max_episode_payload_bytes=2**20,resource_check=lambda *a:None)
                self.assert_structures(expected,actual)
                actual[2].validate(actual[0])

    def test_stable_grouping_and_first_occurrence_order(self):
        values=np.array([3,1,3,2,1])
        unique,inverse=ix.first_occurrence(values)
        np.testing.assert_array_equal(unique,[3,1,2])
        np.testing.assert_array_equal(inverse,[0,1,0,2,1])
        original_levels=original._level_arrays(np.array([2,0,1,0,2]))
        for a,b in zip(original_levels,ix.stable_levels(np.array([2,0,1,0,2]))):
            np.testing.assert_array_equal(a,b)
        p=np.array([4,4,5,5,6,6,-1])
        head,sibling=ix.adjacency(p)
        np.testing.assert_array_equal(head,[-1,-1,-1,-1,1,3,5])
        np.testing.assert_array_equal(sibling,[-1,0,-1,2,-1,4,-1])

    def test_once_only_lossless_mmap_cache_exact_roundtrip_and_caps(self):
        values=fixture(True)
        expected=candidate.prepare_synthetic_or_admitted(*values,num_classes=3,
            query_columns=np.arange(4,8),max_nodes=1000,max_holds=10,
            max_episode_payload_bytes=2**20,resource_check=lambda *a:None)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            directory=Path(temporary)/"bank"
            with self.assertRaisesRegex(ValueError,"caller reservation"):
                cache.write_cache(directory,*expected[:3],expected[3],input_refs={"observed":"synthetic"},
                                  max_output_bytes=1)
            self.assertFalse(directory.exists())
            ref=cache.write_cache(directory,*expected[:3],expected[3],input_refs={"observed":"synthetic"},
                                  max_output_bytes=128*2**20)
            with self.assertRaisesRegex(ValueError,"fresh cache"):
                cache.write_cache(directory,*expected[:3],expected[3],input_refs={"observed":"synthetic"},
                                  max_output_bytes=128*2**20)
            with self.assertRaisesRegex(ValueError,"current RAM"):
                cache.read_cache(ref,max_expanded_bytes=1,max_stored_bytes=128*2**20,input_refs={"observed":"synthetic"})
            loaded=cache.read_cache(ref,max_expanded_bytes=128*2**20,max_stored_bytes=128*2**20,
                                    input_refs={"observed":"synthetic"})
            self.assert_structures(expected,loaded)
            self.assertIs(loaded[0].branch_support_true,loaded[0].branch_support_sham)
            del loaded

    def test_empty_proper_clades_and_external_genotype_cache(self):
        age=np.zeros(5)
        trees=[NativeObservedTreeV1(np.r_[np.full(5,5),-1],np.r_[age,1000.],
                                   np.arange(5),np.arange(5)) for _ in range(2)]
        values=(trees,np.array([[0.,100.],[100.,200.]]),np.array([[0,1,0,1,0],[1,0,1,0,1]],dtype=np.int8),
                np.array([2,2],dtype=np.int8),np.array([50.,150.]),np.array([0,1,2,-1,-1]),age)
        bank,meta=original.encode_block(*values,num_classes=3,query_columns=np.arange(3,5))
        arrays,em=ep.build_clade_episode_arrays_v1(bank.parent.numpy(),bank.sample_index.numpy(),
                                                bank.spans.numpy(),max_payload_bytes=2**20)
        episodes=OrderedCladeEpisodesV1.from_verified_arrays(arrays,em)
        cut,cm=cp.prepare_cut_plan_v1(bank,expected_haplotypes=5,max_nodes=1000,max_holds=10)
        expected=(bank,episodes,cut,dict(observation=meta,episodes=em,cut=cm))
        actual=candidate.prepare_synthetic_or_admitted(*values,num_classes=3,query_columns=np.arange(3,5),
            max_nodes=1000,max_holds=10,max_episode_payload_bytes=2**20,resource_check=lambda *a:None)
        self.assert_structures(expected,actual)
        with tempfile.TemporaryDirectory(dir=HERE) as temporary:
            base=Path(temporary);source=base/"original_genotype.npy"
            np.save(source,bank.genotype.numpy(),allow_pickle=False)
            external={"bank.genotype":dict(path=str(source),sha256=cache.sha(source))}
            ref=cache.write_cache(base/"cache",*actual[:3],actual[3],input_refs={"observed":"empty-synthetic"},
                                  max_output_bytes=128*2**20,external_arrays=external)
            loaded=cache.read_cache(ref,max_expanded_bytes=128*2**20,max_stored_bytes=128*2**20,
                                   input_refs={"observed":"empty-synthetic"})
            self.assert_structures(actual,loaded)
            del loaded
            import json
            receipt=json.loads(Path(ref["path"]).read_text())
            self.assertIn("external_ref",receipt["arrays"]["bank.genotype"])
            target=base/"cache"/receipt["arrays"]["bank.parent"]["ref"]["path"]
            with target.open("r+b") as stream:
                stream.seek(-1,2);value=stream.read(1);stream.seek(-1,2);stream.write(bytes([value[0]^1]))
            with self.assertRaisesRegex(ValueError,"must authenticate"):
                cache.read_cache(ref,max_expanded_bytes=128*2**20,max_stored_bytes=128*2**20,
                                 input_refs={"observed":"empty-synthetic"})

    def test_complete_storage_bound_does_not_depend_on_focal_count(self):
        bound=cache.storage_upper_bound(77472,227474)
        self.assertEqual(bound["packed_node_upper_bound"],77472*3199)
        self.assertGreater(bound["original_width_cache_upper_bound_bytes"],
                           bound["complete_compact_cache_upper_bound_bytes"])
        self.assertGreater(bound["complete_compact_cache_upper_bound_bytes"],15*2**30)


if __name__=="__main__":unittest.main()
