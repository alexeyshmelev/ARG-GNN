"""Small deterministic synthetic equivalence only. No production input/timing/model."""
from pathlib import Path
from dataclasses import fields
import ast
import copy
import sys
import unittest
import numpy as np
import torch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1, branch_concordance_support_v1
from arggnn.american_c3_gnn_dataset_isolated_v2 import AmericanC3BlockV1
from arggnn.american_c3_gnn_stream_train_isolated_v2 import _feature_identity
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1 import observations_v1 as old
from research.reviews.H060_throughput_source_analysis_20261017_v1 import observations_legacy_exact_v1 as new
from research.reviews.H060_throughput_source_analysis_20261017_v1.legacy_support_exact_v1 import branch_concordance_support_exact_v1 as exact
HERE=Path(__file__).resolve().parent


def native(parent,clock,sn=None,sid=None):
    sn=np.arange(4,dtype=np.int64) if sn is None else np.asarray(sn,dtype=np.int64)
    sid=np.arange(len(sn),dtype=np.int64) if sid is None else np.asarray(sid,dtype=np.int64)
    return NativeObservedTreeV1(np.asarray(parent,dtype=np.int64),np.asarray(clock,dtype=np.float64),sn,sid)


def random_tree(rng,h):
    p=[-1]*h; clock=[0.]*h; active=list(range(h))
    while len(active)>1:
        chosen=rng.choice(len(active),size=min(int(rng.integers(2,5)),len(active)),replace=False)
        children=[active[int(i)] for i in chosen]
        active=[u for i,u in enumerate(active) if i not in set(chosen.tolist())]
        ancestor=len(p);p.append(-1);clock.append(max(clock[u] for u in children)+1.)
        for u in children:p[u]=ancestor
        active.append(ancestor)
        if rng.random()<.5:
            u=active.pop();ancestor=len(p);p.append(-1);clock.append(clock[u]+1.)
            p[u]=ancestor;active.append(ancestor)
    sn=rng.permutation(h);sid=rng.permutation(np.arange(h)*11+7)
    return native(p,clock,sn,sid)


class ExactLegacyTests(unittest.TestCase):
    def assert_same(self,tree,g,k,ids):
        expected=branch_concordance_support_v1(tree,g,k,ids)
        for chunk in (1,3,32):
            got=exact(tree,g,k,ids,marker_chunk_size=chunk)
            self.assertEqual(got.dtype,expected.dtype)
            self.assertEqual(got.shape,expected.shape)
            self.assertEqual(got.tobytes(),expected.tobytes())

    def test_random_small_topologies_and_alleles_byte_identity(self):
        rng=np.random.default_rng(918)
        for h in (3,4,7,11):
            for repeat in range(3):
                tree=random_tree(rng,h)
                ids=rng.permutation(tree.sample_id)
                k=rng.integers(1,5,size=17,dtype=np.int64)
                g=np.asarray([rng.integers(-1,int(a),size=h) for a in k],dtype=np.int8)
                self.assert_same(tree,g,k,ids)

    def test_empty_monomorphic_missing_unary_and_sampled_ancestor(self):
        tree=native([4,4,5,5,6,7,7,-1],[0,0,0,0,1,1,2,3])
        self.assert_same(tree,np.empty((0,4),dtype=np.int8),np.empty(0,dtype=np.int8),np.arange(4))
        self.assert_same(tree,np.array([[0,0,0,0],[-1,-1,-1,-1],[0,-1,-1,-1],[0,1,2,3]]),np.array([1,2,2,4]),np.arange(4))
        ancestor=native([2,2,-1],[0,0,1],sn=[0,1,2],sid=[21,8,13])
        self.assert_same(ancestor,np.array([[0,1,2],[1,0,1],[-1,0,1]]),np.array([3,2,2]),np.array([8,13,21]))

    def compare_encoder(self,trees,intervals,g,k,x):
        args=(trees,np.asarray(intervals,dtype=np.float64),np.asarray(g,dtype=np.int8).reshape(-1,4),
            np.asarray(k,dtype=np.int8),np.asarray(x,dtype=np.float64),np.array([0,1,-1,-1]),np.zeros(4))
        a,ma=old.encode_block(*args,num_classes=3)
        b,mb=new.encode_block(*args,num_classes=3)
        self.assertEqual(ma,mb)
        for field in fields(a):
            av,bv=getattr(a,field.name),getattr(b,field.name)
            avs=av if isinstance(av,tuple) else (av,)
            bvs=bv if isinstance(bv,tuple) else (bv,)
            self.assertEqual(len(avs),len(bvs))
            for aa,bb in zip(avs,bvs):
                self.assertEqual(aa.dtype,bb.dtype,field.name)
                self.assertEqual(aa.shape,bb.shape,field.name)
                self.assertEqual(aa.numpy().tobytes(),bb.numpy().tobytes(),field.name)
        metadata=dict(parent_id="synthetic",block_index=0,start_tree=0,stop_tree=len(trees),
            sequence_length_bp=int(intervals[-1][1]),complete_parent=True,
            genomic_geometry=torch.zeros((len(trees),3)),focal_rows=torch.tensor([0,1]),
            focal_hold=torch.tensor([0,len(trees)-1]),focal_sample_column=torch.tensor([2,3]))
        self.assertEqual(_feature_identity(AmericanC3BlockV1(a,**metadata)),
            _feature_identity(AmericanC3BlockV1(b,**metadata)))

    def test_full_encoder_and_original_feature_digest_byte_identity(self):
        a=native([4,4,5,5,6,6,-1],[0,0,0,0,1,1,2])
        b=native([4,4,5,5,6,7,7,-1],[0,0,0,0,1,1,2,3])
        iso=native([-1]*4,[0]*4,sn=[2,0,3,1])
        self.compare_encoder([a,b],[[0,5],[5,10]],[[0,0,1,1],[0,1,2,3],[0,1,2,-1]], [2,4,3],[1,2,9])
        self.compare_encoder([a,iso,b],[[0,5],[5,10],[10,15]],[[0,0,1,1],[0,1,2,3]],[2,4],[1,11])
        self.compare_encoder([a],[[0,5]],[],[],[])
        self.compare_encoder([iso],[[0,5]],[],[],[])

    def test_bad_input_and_count_bound_fail_closed(self):
        tree=native([4,4,5,5,6,6,-1],[0,0,0,0,1,1,2])
        args=(tree,np.array([[0,1,0,1]]),np.array([2]),np.arange(4))
        with self.assertRaises(ValueError):exact(*args,max_count_elements=1)
        for f in (branch_concordance_support_v1,exact):
            with self.assertRaises(ValueError):f(tree,np.array([[0,2,0,1]]),np.array([2]),np.arange(4))
            with self.assertRaises(ValueError):f(tree,np.array([[0,1,0,1]]),np.array([2]),np.array([0,0,2,3]))
            with self.assertRaises(ValueError):f(native([4,4,5,5,6,6,-1],[0,0,0,0,1,1,1]),*args[1:])
        iso=native([-1]*4,[0]*4)
        for f in (old.encode_block,new.encode_block):
            with self.assertRaises(ValueError):f([iso],[[0,5]],args[1],args[2],np.array([1.]),
                np.array([0,1,-1,-1]),np.zeros(4),num_classes=3)

    def test_encoder_and_provider_source_delta_is_only_disclosed_route(self):
        old_ast=ast.parse((ROOT/"research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/observations_v1.py").read_text())
        new_ast=ast.parse((HERE/"observations_legacy_exact_v1.py").read_text())
        def funcs(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        a,b=funcs(old_ast),funcs(new_ast)
        self.assertEqual(set(a),set(b))
        del a["encode_block"].body[1] # old connected-only early dispatch
        for name in a:self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)
        op=(ROOT/"research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/observation_provider_v1.py").read_text()
        np=(HERE/"observation_provider_candidate_v1.py").read_text()
        self.assertEqual(op.replace("from .observations_v1 import native_tree,encode_block",
            "from .observations_legacy_exact_v1 import native_tree,encode_block"),np)
        # Candidate supplies no launcher/config authority or target/model function.
        for path in HERE.glob("*.py"):
            if path.name.startswith("test_"):continue
            self.assertNotIn("subprocess",path.read_text())
            self.assertNotIn("execution_enabled=True",path.read_text())


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    unittest.main()
