"""Small CPU parity on a complete952-haplotype synthetic bank; no GPU fit claim."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from arggnn.benchmarking import european_treeloom as european


class EuropeanBoundedAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        import tskit
        torch.set_num_threads(2)
        tables = tskit.TableCollection(300.)
        for i in range(952):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
        clade = tables.nodes.add_row(time=50.)
        root = tables.nodes.add_row(time=1000.)
        for a,b,pair in ((0.,100.,(0,1)),(100.,200.,(0,2)),(200.,300.,(0,1))):
            for i in range(952):
                tables.edges.add_row(a,b,clade if i in pair else root,i)
            tables.edges.add_row(a,b,root,clade)
        for position,node in ((50.,300),(250.,301)):
            site = tables.sites.add_row(position,ancestral_state="A")
            tables.mutations.add_row(site,node,derived_state="C")
        tables.sort()
        cls.ts = tables.tree_sequence()
        cls.refs = dict(inference_ref=dict(path="mock/inference.json",sha256="a"*64),
                        observed_vcf_ref=dict(path="mock/input.vcf",sha256="b"*64),
                        sample_manifest_ref=dict(path="mock/rows.json",sha256="c"*64))
        cls.parent = "synthetic_european_train"
        cls.state = dict(manifest=dict(parent_id=cls.parent,split="train",
                            sequence_length_bp=200000000,truth_or_oracle_read=False),
            ts=cls.ts,rows=[dict(sample_index=i) for i in range(952)],
            nodes=np.arange(952,dtype=np.int64),
            genotype=np.asarray(cls.ts.genotype_matrix(),dtype=np.int8),
            positions=np.asarray([50.,250.],dtype=np.float64),
            allele_count=np.asarray([2,2],dtype=np.int8),
            labels=np.asarray([i%4 if i<266 else -1 for i in range(952)],dtype=np.int64),
            ages=np.zeros(952,dtype=np.float64),query_columns=np.arange(266,952,dtype=np.int64),
            coordinate_offset_bp=1000.,source_refs=cls.refs)
        indices = np.asarray([266,267,268,266,269,270],dtype=np.int64)
        cls.records = dict(record_ids=np.arange(6,dtype=np.int64),sample_nodes=indices.copy(),
            sample_indices=indices,positions_bp0=np.asarray([1050.,1150.,1250.,1050.,1150.,1250.]),
            parent_id=np.asarray(cls.parent),class_order=np.asarray(european.NATIVE_SIX_CLASSES))
        cls.sampled_ref = dict(path="mock/sample_registry.npz",sha256="d"*64)

    def prepare(self,records=None):
        with patch.object(european,"_sampled_records",return_value=self.records if records is None else records):
            return european.prepare_observation(self.state,
                protocol_id="ancestralpaths_paper200mb_native6",parent_id=self.parent,split="train",
                source_refs=self.refs,sampled_registry_ref=self.sampled_ref,
                max_nodes=10000,max_holds=10,max_episode_payload_bytes=2**20,
                resource_check=lambda *args:None)

    def test_complete_bank_and_exact_repeated_sample_coordinate_join(self):
        import torch
        bundle,metadata = self.prepare()
        self.assertEqual(bundle.native_holds,self.ts.num_trees)
        self.assertEqual(bundle.bank.sample_index.shape,(3,952))
        self.assertEqual(bundle.bank.query_column.numel(),686)
        self.assertEqual(bundle.bank.log_donor_allele_frequency.shape[-1],8)
        self.assertEqual(bundle.bank.genotype.shape,(2,952))
        self.assertEqual(bundle.focal_hold.tolist(),[0,1,2,0,1,2])
        self.assertEqual(bundle.focal_sample_column.tolist(),[266,267,268,266,269,270])
        self.assertEqual(bundle.sampled_registry_ref,self.sampled_ref)
        self.assertFalse(metadata["private_truth_opened"])
        self.assertTrue(metadata["structural_recipes_global_before_partition"])
        self.assertEqual(float(bundle.bank.spans.double().sum()),300.)
        self.assertEqual(metadata["focal_records"],6)
        changed = bundle.focal_sample_column
        changed[0] = 267
        with self.assertRaises(european.c.ContractError):
            bundle.validate()

    def test_tiled_full_context_logits_match_original_native_six_model(self):
        import torch
        bundle,metadata = self.prepare()
        model = european.build_model("ancestralpaths_paper200mb_native6")
        self.assertEqual(sum(p.numel() for p in model.parameters()),32391)
        self.assertEqual(model.output_classes,european.NATIVE_SIX_CLASSES)
        # Exercise nonzero memory/clade/episode channels beyond their zero-O
        # initialization, without changing any selected architecture or state key.
        with torch.no_grad():
            for adapter in (model.memory,model.clade_coupling,model.episode_coupling):
                adapter.O.weight.normal_(0.,.02)
        model.eval()
        with torch.no_grad():
            complete = model(bundle.bank,episodes=bundle.original)
            query_index = {int(column):i for i,column in enumerate(bundle.bank.query_column)}
            query = torch.tensor([query_index[int(column)] for column in bundle.focal_sample_column])
            expected = complete[bundle.focal_hold,query]
            executor = european._executor_class()(model,bundle,metadata,
                holds_per_block=2,haplotypes_per_gru=8,resource_check=lambda *args:None,
                history_holds_per_window=2)
            actual = executor.forward()
        self.assertEqual(actual.shape,(6,6))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        # Same coordinate/query repeated draws remain two identical emissions.
        torch.testing.assert_close(actual[0],actual[3],atol=0.,rtol=0.)

    def test_general_history_matches_original_values_and_force_scale_gradients(self):
        import torch
        bundle,_ = self.prepare()
        model = european.build_model("ancestralpaths_paper200mb_native6")
        memory = model.memory
        force = torch.randn(len(bundle.cut.representatives),8,requires_grad=True)
        original = memory._history(force,bundle.bank.spans,bundle.cut)
        expected = torch.autograd.grad(original.square().sum(),(force,memory.ell_theta))
        copied = force.detach().clone().requires_grad_(True)
        actual = european._general_history(memory,copied,memory.ell_theta,bundle.bank.spans,bundle.cut,
            device="cpu",holds_per_window=2,resource_check=lambda *args:None)
        gradients = torch.autograd.grad(actual.square().sum(),(copied,memory.ell_theta))
        torch.testing.assert_close(actual,original,atol=2e-6,rtol=2e-5)
        for result,wanted in zip(gradients,expected):
            torch.testing.assert_close(result,wanted,atol=2e-5,rtol=2e-4)

    def test_whole_parent_limits_and_truth_extra_state_are_rejected(self):
        with patch.object(european,"_sampled_records",return_value=self.records):
            kwargs = dict(protocol_id="ancestralpaths_paper200mb_native6",parent_id=self.parent,
                split="train",source_refs=self.refs,sampled_registry_ref=self.sampled_ref,
                max_nodes=100,max_holds=10,max_episode_payload_bytes=2**20,
                resource_check=lambda *args:None)
            with self.assertRaises(european.c.ContractError):
                european.prepare_observation(self.state,**kwargs)
            kwargs["max_nodes"] = 10000
            with self.assertRaises(european.c.ContractError):
                european.prepare_observation(dict(self.state,query_truth=object()),**kwargs)

    def test_task_heads_and_inert_import(self):
        for protocol,count in (("ancestralpaths_paper200mb_c4",32325),
                               ("european_c4_prospective1611",32325)):
            model = european.build_model(protocol)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            self.assertEqual(tuple(model.output_classes),("Ana","CHG","WHG","EHG"))
        code = ("import sys;sys.path.insert(0,%r);" % str(Path(european.__file__).resolve().parents[2])
                + "from arggnn.benchmarking import european_treeloom;"
                + "assert not any(n in sys.modules for n in ('torch','numpy','tskit','torch_geometric'))")
        subprocess.run([sys.executable,"-B","-s","-c",code],check=True,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__ == "__main__":
    unittest.main()
