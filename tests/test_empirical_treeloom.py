"""Small H36 CPU C4-state/whole-window parity and descriptive pooling."""
from decimal import Decimal,ROUND_HALF_EVEN
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from arggnn.benchmarking import empirical_treeloom as empirical


class CurrentEmpiricalTreeLoomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from arggnn.mutation_routed_tree_observations_v1 import NativeObservedTreeV1
        from arggnn.anonymous_clade_flag_control_v1 import build_clade_flag_arrays_v1,AnonymousCladeFlagsV1
        from research.reviews.H056_H36_implementation_20261016_v1 import contract_v1,preparation_h36_v1
        torch.set_num_threads(2)
        cls.roster=contract_v1.planned_roster_v1();trees=[]
        ages=[float(Decimal.from_float(r["age_generations"]).quantize(Decimal("0.000001"),rounding=ROUND_HALF_EVEN)) for r in cls.roster]
        for pair in ((0,1),(0,1),(0,2)):
            parent=np.full(38,37,dtype=np.int64);parent[list(pair)]=36;parent[37]=-1
            trees.append(NativeObservedTreeV1(parent,np.asarray([*ages,600.,1000.]),np.arange(36),np.arange(36)))
        genotype=np.asarray([[(i+site)%2 for i in range(36)] for site in range(3)],dtype=np.int16)
        arrays=preparation_h36_v1.prepare_arrays_v1(trees,np.asarray([[0.,100.],[100.,300.],[300.,600.]]),
            np.asarray([50.,150.,450.]),genotype,np.full(3,2,dtype=np.int64),cls.roster)
        cls.bank,_=preparation_h36_v1.to_torch_bank_v1(arrays)
        flags,meta=build_clade_flag_arrays_v1(cls.bank.parent.numpy(),cls.bank.sample_index.numpy())
        cls.flags=AnonymousCladeFlagsV1.from_verified_arrays(flags,meta)
        cls.flag_meta=meta

    def test_exact_tensor_payload_serialization_and_observed_loader(self):
        import torch
        from dataclasses import fields
        from research.reviews.H056_H36_implementation_20261016_v1.reader_h36_v1 import bank_array_manifest_v1
        protocol=dict(path="mock/protocol.json",sha256="a"*64)
        input_ref=dict(path="mock/input.json",sha256="b"*64)
        metadata=dict(model_input_tensor_manifest=bank_array_manifest_v1(self.bank),
                      exact_clade_flags=self.flag_meta)
        done=dict(protocol_ref=protocol,input_ref=input_ref,metadata=metadata)
        saved=dict(schema="H056_empirical_CPU_observation_bank_v1",protocol_ref=protocol,
            chromosome="20",input_ref=input_ref,metadata=metadata,
            bank_tensors={f.name:getattr(self.bank,f.name) for f in fields(self.bank)},
            flag_node_flags=self.flags.node_flags)
        with tempfile.TemporaryDirectory(dir=empirical.c.ROOT/"tests") as directory:
            path=Path(directory)/"observation_bank.pt"
            with path.open("wb") as stream:torch.save(saved,stream)
            item=dict(bank_ref=empirical.generation._reference(path),
                roster_ref=dict(path="mock/roster.json",sha256="c"*64),
                counts=dict(native_trees=3,sites=3,vertex_occurrences=114))
            checks=[]
            with patch.object(empirical,"_metadata",return_value=done), \
                 patch.object(empirical.current.observed,"_json_ref",return_value=self.roster):
                bank,flags,result,roster=empirical.load_bank("20",item,max_bank_read_bytes=2**20,
                                                           resource_check=lambda *args:checks.append(args))
            self.assertEqual(bank_array_manifest_v1(bank),metadata["model_input_tensor_manifest"])
            torch.testing.assert_close(flags.node_flags,self.flags.node_flags,atol=0.,rtol=0.)
            self.assertEqual(roster,self.roster);self.assertIs(result,done)
            self.assertEqual(len(checks),1)
            # Release mmap-backed synthetic tensors before unlinking on NFS.
            del bank,flags

    def test_current_c4_state_keys_shapes_and_strict_saved_state(self):
        import torch
        model=empirical.build_variable_model()
        original=empirical.current.observed.build_model(empirical.current.PROTOCOL)
        self.assertEqual(sum(p.numel() for p in model.parameters()),32325)
        self.assertEqual(len(list(model.parameters())),49)
        self.assertEqual({k:tuple(v.shape) for k,v in model.state_dict().items()},
                         {k:tuple(v.shape) for k,v in original.state_dict().items()})
        state={k:v.clone() for k,v in original.state_dict().items()}
        with torch.no_grad():
            for value in state.values():value.add_(.01)
        model.load_state_dict(state,strict=True)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,state[key],atol=0.,rtol=0.)
        native6=empirical.current.observed.build_model("ancestralpaths_paper200mb_native6")
        with self.assertRaises(RuntimeError):model.load_state_dict(native6.state_dict(),strict=True)

    def test_all36_haplotypes_one_complete_bidirectional_gru_context(self):
        import torch
        bundle,episodes,cut=empirical.make_bundle(self.bank,self.flags,chromosome="20",source_refs={},
            max_nodes=1000,max_holds=10,max_episode_payload_bytes=2**20)
        self.assertEqual(bundle.bank.sample_index.shape,(3,36))
        self.assertEqual(bundle.bank.query_column.tolist(),list(range(16,36)))
        torch.testing.assert_close(bundle.episodes.flags.node_flags,self.flags.node_flags,atol=0.,rtol=0.)
        self.assertTrue(episodes["all_original_vertices_preserved"])
        model=empirical.build_variable_model();model.eval();calls=[]
        with torch.no_grad():
            for adapter in (model.memory,model.clade_coupling,model.episode_coupling):adapter.O.weight.normal_(0.,.02)
        hook=model.genomic_gru.register_forward_pre_hook(lambda _,args:calls.append(tuple(args[0].shape)))
        with torch.inference_mode():output=model(self.bank,episodes=bundle).softmax(-1)
        hook.remove()
        self.assertEqual(calls,[(3,36,35)])
        self.assertEqual(tuple(output.shape),(3,20,4));self.assertTrue(bool(torch.isfinite(output).all()))
        torch.testing.assert_close(output.sum(-1),torch.ones(3,20),atol=1e-6,rtol=0.)
        with torch.inference_mode():all_holds=model.forward_holds(self.bank,episodes=bundle).softmax(-1)
        torch.testing.assert_close(output,all_holds[:,16:36],atol=0.,rtol=0.)
        with self.assertRaises(empirical.c.ContractError):empirical.make_bundle(self.bank,self.flags,chromosome="20",
            source_refs={},max_nodes=100,max_holds=10,max_episode_payload_bytes=2**20)

    def test_covered_bp_diploid_chromosome_and_equal_people_pooling(self):
        import numpy as np
        subjects=[r["subject_id"] for r in self.roster[16::2]];results={};expected=[];weights=[1.,3.,6.]
        for i,chrom in enumerate(empirical.CHROMOSOMES):
            values=np.zeros((20,4));values[:,0]=1.
            values[10:,0]=1.-.1*(i+1);values[10:,1]=.1*(i+1)
            values[0]=[.8,.2,0.,0.];values[1]=[.6,.4,0.,0.]
            results[chrom]=dict(haplotype_proportions=values,covered_bp=weights[i])
            expected.append(values.reshape(10,2,4).mean(1))
        summary=empirical.descriptive_summary(results,subjects)
        pooled=np.einsum("c,cpa->pa",weights,expected)/sum(weights)
        np.testing.assert_allclose(summary["pooled_person_probabilities"],pooled)
        np.testing.assert_allclose(summary["later_minus_earlier"],pooled[5:].mean(0)-pooled[:5].mean(0))
        self.assertFalse(summary["ancestry_accuracy_available"]);self.assertFalse(summary["mixture_calibration_claimed"])
        self.assertFalse(summary["uncertainty_or_significance_claimed"]);self.assertEqual(summary["optimization_steps"],0)
        with self.assertRaises(empirical.c.ContractError):empirical.descriptive_summary({"20":results["20"]},subjects)

    def test_native6_or_historical_final_fit_metadata_rejected(self):
        base=dict(schema="european_treeloom_fit_completion_v1",status="completed_full_parent_sampled_fit",
            protocol_id=empirical.current.PROTOCOL,class_order=list(empirical.CLASSES),
            optimizer_updates=90,selected_epoch=30,completed_epochs=30,sampling_plan_ref={},
            checkpoint_selection="fixed_final_epoch30",TEST_truth_read=False,
            checkpoint_ref=dict(path="mock/current.pt",sha256="a"*64))
        for key,value in (("protocol_id","ancestralpaths_paper200mb_native6"),
                          ("class_order",list(empirical.current.observed.NATIVE_SIX_CLASSES)),
                          ("optimizer_updates",600),("selected_epoch",25),
                          ("checkpoint_ref",dict(path="mock/old.pt",sha256="bfd8a87632015a900d9da6009d4b0ee037958460038d7949d0998e80b770796b"))):
            with patch.object(empirical.current.original.gpu,"_public_ref"), \
                 patch.object(empirical.c,"read_ref",return_value=dict(base,**{key:value})),patch.object(empirical.current,"_parent_plan"):
                with self.assertRaises(empirical.c.ContractError):empirical._completion({})

    def test_factory_and_inert_import(self):
        expected=dict(requirements=dict(source_refs=[]))
        with patch.object(empirical,"build_spec",return_value=expected):descriptor=empirical.factory(test_argument=True)
        self.assertEqual(descriptor["module"],"arggnn.benchmarking.empirical_treeloom");self.assertEqual(descriptor["function"],"run")
        code="import sys;sys.path[:0]=%r;from arggnn.benchmarking import empirical_treeloom;assert not any(x in sys.modules for x in ('torch','numpy','tskit','torch_geometric'))" % [str(empirical.c.ROOT/"src"),str(empirical.c.ROOT)]
        subprocess.run([sys.executable,"-s","-B","-c",code],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__=="__main__":unittest.main()
