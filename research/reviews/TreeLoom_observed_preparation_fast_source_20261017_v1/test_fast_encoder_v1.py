"""Unexecuted small encoder/integration fixtures for ROOT; no production input.

No model constructor, learned forward/backward, truth, checkpoint or timing run.
"""
from pathlib import Path
import ast
from dataclasses import fields
import hashlib
import sys
import unittest
import numpy as np
import torch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.observations_v1 import encode_block as legacy_encode
from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1 import observations_fast_v1 as fast
from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1 import full_preparation_fast_v1 as full
from research.reviews.TreeLoom_observed_preparation_fast_source_20261017_v1 import parent_startup_fast_v1 as startup
HERE=ROOT/"research/reviews/TreeLoom_observed_preparation_fast_source_20261017_v1"
OLD=ROOT/"research/reviews/H060_isolated_hold_compatibility_source_20261017_v1"


def native(parent,clock,sn=None):
    h=4 if sn is None else len(sn)
    return NativeObservedTreeV1(np.asarray(parent,dtype=np.int64),
        np.asarray(clock,dtype=np.float64),
        np.arange(h,dtype=np.int64) if sn is None else np.asarray(sn,dtype=np.int64),
        np.arange(h,dtype=np.int64))


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}


class FastEncoderFixtures(unittest.TestCase):
    def compare(self,trees,intervals,g,k,x,ref=None,age=None):
        h=len(trees[0].sample_node)
        ref=np.array([0,1,-1,-1] if ref is None else ref,dtype=np.int64)
        age=np.array([0,0,100,100] if age is None else age,dtype=np.float64)
        args=(trees,np.asarray(intervals,dtype=np.float64),
            np.asarray(g,dtype=np.int8).reshape(-1,h),np.asarray(k,dtype=np.int8),
            np.asarray(x,dtype=np.float64),ref,age)
        kwargs=dict(num_classes=3,query_columns=np.flatnonzero(ref<0))
        old,old_meta=legacy_encode(*args,**kwargs)
        new,new_meta=fast.encode_block(*args,**kwargs)
        self.assertEqual([f.name for f in fields(new)],[f.name for f in fields(old)])
        for field in fields(old):
            a,b=getattr(old,field.name),getattr(new,field.name)
            if isinstance(a,tuple):
                self.assertEqual(len(a),len(b))
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)),field.name)
            elif field.name in ("branch_support_true","branch_support_sham"):
                torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
            else:
                self.assertTrue(torch.equal(a,b),field.name)
            values=b if isinstance(b,tuple) else (b,)
            self.assertTrue(all(t.device.type=="cpu" and not t.requires_grad for t in values))
        for key,value in old_meta.items():
            self.assertEqual(new_meta[key],value,key)
        self.assertEqual(new_meta["observed_preparation_route"],"explicit_level_batched_global_v1")
        self.assertEqual(new_meta["raw_layout_calls_per_tree"],1)
        self.assertFalse(new_meta["historical_support_byte_identity_claimed"])
        return new,new_meta

    def test_all_connected_route_and_every_unrelated_field(self):
        a=native([4,4,5,5,6,6,-1],[0,0,100,100,200,200,500])
        b=native([4,4,5,5,6,7,7,-1],[0,0,100,100,200,200,300,500])
        bank,_=self.compare([a,b],[[0,5],[5,10]],
            [[0,0,1,1],[0,1,2,3],[0,1,2,-1],[-1,0,1,-1]],[2,4,3,2],[1,2,6,9])
        self.assertEqual(tuple(bank.log_donor_allele_frequency.shape),(4,4,6))
        self.assertEqual(bank.sample_index.shape[0],2)

    def test_mixed_connected_and_isolated_zero_site_holds(self):
        a=native([4,4,5,5,6,6,-1],[0,0,100,100,200,200,500])
        isolated=native([-1,-1,-1,-1],[0,100,0,100],[2,0,3,1])
        bank,meta=self.compare([a,isolated,a],[[0,5],[5,10],[10,15]],
            [[0,1,2,3],[0,0,1,1],[1,0,1,0]],[4,2,2],[1,2,11])
        self.assertEqual(meta["isolated_zero_site_holds"],[1])
        self.assertEqual(bank.parent.numel(),18)

    def test_empty_connected_and_all_isolated_encodings(self):
        a=native([4,4,5,5,6,6,-1],[0,0,100,100,200,200,500])
        self.compare([a],[[0,10]],[],[],[])
        isolated=native([-1,-1,-1,-1],[0,100,0,100],[2,0,3,1])
        self.compare([isolated],[[0,10]],[],[],[])

    def test_raw_chronology_checks_and_FP32_clock_ties(self):
        a=native([2,2,-1],[100000000.,0.,100000001.],[0,1])
        bank,_=self.compare([a],[[0,10]],[[0,1]],[2],[1],ref=[0,-1],age=[100000000.,0.])
        self.assertEqual(bank.node_time[0],bank.node_time[2])
        bad=native([2,2,-1],[100000000.,0.,100000000.],[0,1])
        for encoder in (legacy_encode,fast.encode_block):
            with self.assertRaises(ValueError):
                encoder([bad],[[0,10]],np.array([[0,1]]),np.array([2]),np.array([1.]),
                    np.array([0,-1]),np.array([100000000.,0.]),num_classes=3)

    def test_original_domain_rejects_isolated_SNPs_and_bad_joins(self):
        isolated=native([-1,-1,-1,-1],[0,0,100,100])
        for encoder in (legacy_encode,fast.encode_block):
            with self.assertRaises(ValueError):
                encoder([isolated],[[0,10]],np.array([[0,1,0,1]]),np.array([2]),np.array([1.]),
                    np.array([0,1,-1,-1]),np.array([0,0,100,100]),num_classes=3)
        connected=native([4,4,5,5,6,6,-1],[0,0,100,100,200,200,500])
        bad=NativeObservedTreeV1(connected.parent,connected.node_time,connected.sample_node,np.array([1,0,2,3]))
        for encoder in (legacy_encode,fast.encode_block):
            with self.assertRaises(ValueError):
                encoder([bad],[[0,10]],np.array([[0,1,0,1]]),np.array([2]),np.array([1.]),
                    np.array([0,1,-1,-1]),np.array([0,0,100,100]),num_classes=3)

    def test_global_recipe_AST_and_explicit_encoder_binding(self):
        old=functions(OLD/"full_preparation_v1.py")["prepare_from_loaded_observation_v1"]
        new=functions(HERE/"full_preparation_fast_v1.py")["prepare_from_loaded_observation_v1"]
        self.assertEqual(ast.dump(old),ast.dump(new))
        self.assertIs(full.encode_block,fast.encode_block)
        self.assertIs(full.native_tree,fast.native_tree)
        encode=functions(HERE/"observations_fast_v1.py")["encode_block"]
        names=[n.func.id for n in ast.walk(encode)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(names.count("_layout_with_isolated_samples"),1)
        self.assertEqual(names.count("level_batched_branch_support_v1"),1)
        self.assertNotIn("_connected_encode_block",names)
        self.assertNotIn("branch_concordance_support_v1",names)

    def test_encoder_AST_keeps_original_checks_and_other_features(self):
        import copy
        old=functions(OLD/"observations_v1.py")["encode_block"]
        new=functions(HERE/"observations_fast_v1.py")["encode_block"]
        # Restore only the disclosed dispatch/layout/support/metadata edits.
        new.body.insert(1,copy.deepcopy(old.body[1]))
        old_loop=next(n for n in old.body if isinstance(n,ast.For))
        new_loop=next(n for n in new.body if isinstance(n,ast.For))
        i=next(i for i,n in enumerate(new_loop.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=="native_layout" for t in n.targets))
        original_layout=next(n for n in old_loop.body if isinstance(n,ast.Assign)
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
            and n.value.func.id=="_layout_with_isolated_samples")
        new_loop.body[i:i+2]=[copy.deepcopy(original_layout)]
        old_support=next(n for n in old_loop.body if isinstance(n,ast.If)
            and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name)
                and c.func.id=="branch_concordance_support_v1" for c in ast.walk(n)))
        i=next(i for i,n in enumerate(new_loop.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=="prepared" for t in n.targets))
        new_loop.body[i:i+2]=[copy.deepcopy(old_support)]
        del new.body[-2]  # additional truthful fast-route metadata only
        self.assertEqual(ast.dump(old),ast.dump(new))

    def test_one_load_legacy_seal_fast_global_order_and_no_truth_API(self):
        f=functions(HERE/"parent_startup_fast_v1.py")["seal_then_prepare_one_parent_v1"]
        named=[n for n in ast.walk(f) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(sum(n.func.id=="load_observation" for n in named),1)
        load=next(n.lineno for n in named if n.func.id=="load_observation")
        prepare=next(n.lineno for n in named if n.func.id=="prepare_from_loaded_observation_v1")
        digest=next(n.lineno for n in named if n.func.id=="fast_global_identity_v1")
        seal=next(n.lineno for n in ast.walk(f) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=="seal")
        self.assertLess(load,seal);self.assertLess(seal,prepare);self.assertLess(prepare,digest)
        self.assertTrue(f.body[-1].finalbody)
        self.assertFalse(any("truth" in arg.arg for arg in f.args.args+f.args.kwonlyargs))
        self.assertFalse(any(n.func.id.startswith(("resolve_","train_","score_")) for n in named))

    def test_exact_additional_source_binding_and_digest_recipe(self):
        refs=[dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
            for path in startup.REQUIRED_FAST_SOURCE_PATHS]
        checked=startup._check_fast_sources_v1(refs)
        self.assertEqual(len(checked),4)
        with self.assertRaises(ValueError):
            startup._check_fast_sources_v1(refs[:-1])
        bad=[dict(r) for r in refs];bad[0]["sha256"]="0"*64
        with self.assertRaises(ValueError):
            startup._check_fast_sources_v1(bad)
        old=functions(OLD/"warmstart_train_v2.py")["_global_identity_v1"]
        new=functions(HERE/"parent_startup_fast_v1.py")["fast_global_identity_v1"]
        # Only the function/docstring and two explicit imported tensor helpers
        # differ; serialization order, dtypes, shapes and bytes stay identical.
        old.name=new.name;old.body[0:0]=[ast.Expr(value=ast.Constant(value="doc"))]
        new.body[0]=old.body[0]
        for node in ast.walk(old):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=="full"):
                node.func=ast.Name(id=node.func.attr,ctx=ast.Load())
        def constants(tree):
            return [node.value for node in ast.walk(tree) if isinstance(node,ast.Constant)]
        self.assertEqual(constants(old),constants(new))
        # The two original require messages differ only in human wording.
        old_calls=[n for n in ast.walk(old) if isinstance(n,ast.Call)]
        new_calls=[n for n in ast.walk(new) if isinstance(n,ast.Call)]
        for a,b in zip(old_calls,new_calls):
            if isinstance(a.func,ast.Name) and a.func.id=="require":
                a.args[1]=b.args[1]
        self.assertEqual(ast.dump(old),ast.dump(new))


if __name__=="__main__":
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    unittest.main()
