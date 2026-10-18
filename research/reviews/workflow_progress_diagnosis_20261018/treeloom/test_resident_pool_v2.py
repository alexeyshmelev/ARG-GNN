"""Tiny CPU codec and original600-update schedule checks; no scientific files."""
from dataclasses import dataclass,fields
import copy
import random
import tempfile
import json
import types
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from . import resident_pool_v2 as pool
from . import prepare_integration_v2 as integration
from .test_candidate_v1 import fixture,prepare_original
from research.reviews.TreeLoom_preparation_reuse_source_20261017_v1 import optimized_preparation_v1 as original
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
from arggnn.benchmarking import treeloom_training as training


@dataclass(frozen=True)
class TinyWholeParent:
    bank:object
    episodes:object
    cut:object
    parent_id:str
    split:str
    sequence_length_bp:int
    native_holds:int
    original_sites:int
    focal_hold:object
    focal_sample_column:object
    focal_identity_sha256:str
    observation_ref:dict
    focal_registry_ref:dict
    native_preparation_ref:dict
    bank_stamps:tuple
    episode_stamps:tuple
    focal_stamps:tuple
    def validate(self):
        assert self.bank_stamps==core._stamps(original._bank_tensors(self.bank))
        assert self.episode_stamps==core._stamps(original._episode_tensors(self.episodes))
        assert self.focal_stamps==core._stamps((self.focal_hold,self.focal_sample_column))
        self.cut.validate(self.bank)


def tiny(parent="fresh_train_000",refs=None):
    bank,episodes,cut,_=prepare_original(fixture(True))
    refs=refs or {k:dict(path="mock/"+k,sha256="a"*64) for k in training.fresh.INPUT_REFS}
    hold=torch.tensor([0,1,3]);sample=torch.tensor([4,5,6])
    bundle=TinyWholeParent(bank,episodes,cut,parent,"train",30000000,4,3,hold,sample,"c"*64,
        **refs,bank_stamps=core._stamps(original._bank_tensors(bank)),
        episode_stamps=core._stamps(original._episode_tensors(episodes)),
        focal_stamps=core._stamps((hold,sample)))
    return bundle,dict(parent_id=parent,declared_split="train",parent_registry_ref={"mock":"registry"},
                       all_native_trees_nodes_sites_samples_retained=True,private_truth_opened=False)


class CompactRAMReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_lossless_independent_pack_restore_and_mutation_detection(self):
        bundle,meta=tiny();refs={k:getattr(bundle,k) for k in training.fresh.INPUT_REFS}
        record=pool.pack_bundle(bundle,meta,input_refs=refs)
        restored,got=pool.restore_bundle(record,input_refs=refs)
        self.assertEqual(pool.exact_signature(bundle,meta),pool.exact_signature(restored,got))
        for f in fields(bundle.bank):
            a=getattr(bundle.bank,f.name);b=getattr(restored.bank,f.name)
            if isinstance(a,tuple):
                for x,y in zip(a,b):
                    if x.numel():self.assertNotEqual(x.data_ptr(),y.data_ptr())
            elif a.numel():self.assertNotEqual(a.data_ptr(),b.data_ptr())
        # Restored tensors are independent from immutable compact masters.
        restored.bank.node_time[0]=987.
        again,again_meta=pool.restore_bundle(record,input_refs=refs)
        self.assertEqual(pool.exact_signature(bundle,meta),pool.exact_signature(again,again_meta))
        with self.assertRaisesRegex(ValueError,"input refs"):
            pool.restore_bundle(record,input_refs={"other":"parent"})
        key="bank.node_time";record["stored"][key].flags.writeable=True
        record["stored"][key].flat[0]+=1.
        with self.assertRaisesRegex(ValueError,"authenticate"):
            pool.restore_bundle(record,input_refs=refs,verify_integrity=True)

    def test_all20_prepare_once_original600_updates_and_headroom_failstop(self):
        parents=tuple("fresh_train_%03d"%i for i in range(20))
        class Registry:
            reference={"mock":"registry"}
            def ids(self,split):return parents if split=="train" else ()
            def split_for(self,parent):return "train" if parent in parents else "test"
            def epoch_order(self,epoch):
                result=list(parents);random.Random(70002+epoch).shuffle(result);return result
        registry=Registry()
        def ref(parent,kind):return dict(path="mock/"+parent+"/"+kind,sha256="b"*64)
        inputs={p:{k:ref(p,k) for k in (*training.fresh.INPUT_REFS,"truth_ref")} for p in parents}
        observed={p:{k:inputs[p][k] for k in training.fresh.INPUT_REFS} for p in parents}
        prepared=[];events=[];rss=[0]
        compact=pool.CompactParentPool(parent_bounds={p:80*2**20 for p in parents},
            input_refs=observed,external_arrays={},max_cache_bytes=2*2**30,
            max_rss_bytes=4*2**30,preparation_headroom_bytes=128*2**20,
            rss_bytes=lambda:rss[0],resource_check=lambda s:None,event=lambda *args:events.append(args))
        def load(parent):
            def prepare():
                prepared.append(parent);return tiny(parent,observed[parent])
            return compact.load(parent,prepare)
        seals={}
        for parent in parents:
            bundle,meta=load(parent)
            seals[parent]=dict(focal_identity_sha256=bundle.focal_identity_sha256,**observed[parent])
        del bundle,meta
        compact.require_all_prepared()
        barrier=dict(parent_registry_ref=registry.reference,all20_observations_sealed_before_truth=True,
                     private_truth_opened=False,parents=seals)
        optimizer=SimpleNamespace(count=0);visits=[]
        class Runtime:
            def train_step(self,model,bundle,target,opt,meta,*,resource_check):
                visits.append(bundle.parent_id);opt.count+=1
                return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,native_focals=10000)
        def join(parent,bundle,meta):
            receipt=dict(parent_id=parent,split="train",parent_registry_ref=registry.reference,
                         truth_ref=inputs[parent]["truth_ref"],focal_registry_ref=bundle.focal_registry_ref,
                         focal_identity_sha256=bundle.focal_identity_sha256,records=10000,
                         labels_in_observation_bank=False)
            return SimpleNamespace(parent_id=parent,focal_registry_ref=bundle.focal_registry_ref,
                                   focal_identity_sha256=bundle.focal_identity_sha256,receipt=receipt)
        checkpoints=[]
        def checkpoint(epoch,steps,*args):
            checkpoints.append((epoch,steps));return ref(str(epoch),"checkpoint")
        result=training.train_loop(registry=registry,runtime=Runtime(),model=None,optimizer=optimizer,
            train_inputs=inputs,observation_barrier=barrier,load_parent=load,join_targets=join,
            checkpoint=checkpoint,event=lambda *args:None,resource_check=lambda stage:None,
            optimizer_counts=lambda opt:[opt.count]*49)
        self.assertEqual(result["complete_parent_updates"],600)
        self.assertEqual(prepared,list(parents))
        self.assertEqual(visits,[p for epoch in range(1,31) for p in registry.epoch_order(epoch)])
        self.assertEqual(compact.preparations,{p:1 for p in parents})
        self.assertEqual(compact.restorations,{p:30 for p in parents})
        self.assertEqual(checkpoints,[(e,e*20) for e in training.PASSES])
        rss[0]=4*2**30
        with self.assertRaisesRegex(ValueError,"never evict"):
            load(parents[0])
        self.assertEqual(len(compact.records),20)
        self.assertEqual(compact.preparations,{p:1 for p in parents})


    def test_qualification_gate_and_factory_never_resolve_truth(self):
        from arggnn.benchmarking import treeloom_preparation_v4 as wrapper
        from arggnn.benchmarking import american_workflow as american
        refs={k:dict(path="mock/"+k,sha256="a"*64) for k in training.fresh.INPUT_REFS}
        valid=dict(schema="one_full_AMtrain000_exact_compact_CPU_qualification_v2",
            status="passed_complete_parent_exact_preparation_and_two_reuses",
            parent_id=wrapper.PARENT,observed_input_refs=refs,
            candidate_source_refs=wrapper.candidate_source_refs(),private_truth_opened=False,
            models_executed=False,CUDA_called=False,scientific_outputs_created=0,disk_cache_bytes=0,
            full_parent_preparation_count=1,exact_restores=2,
            all_original_tensor_and_metadata_signatures_equal=True,original_complete_parent_retained=True,
            compact_charged_bytes=1000000,observed_peak_rss_bytes=1000000,elapsed_seconds=1.)
        with mock.patch.object(wrapper.c,"read_ref",return_value=valid):
            self.assertIs(wrapper._require_pass({"path":"mock/pass","sha256":"a"*64},refs),valid)
        for key,value in (("status","failed"),("CUDA_called",True),("models_executed",True),
                          ("full_parent_preparation_count",2),("exact_restores",1),
                          ("elapsed_seconds",7201.),("candidate_source_refs",[])):
            bad=dict(valid,**{key:value})
            with mock.patch.object(wrapper.c,"read_ref",return_value=bad):
                with self.assertRaises(wrapper.c.ContractError):wrapper._require_pass({},refs)
        ready=dict(parent_id=wrapper.PARENT,split="train",observation_ref=refs["observation_ref"])
        observed=dict(parent_id=wrapper.PARENT,split="train",query_ancestry_truth_in_inputs=False)
        with mock.patch.object(american,"observed_binding",return_value=refs), \
                mock.patch.object(american,"train_binding",side_effect=AssertionError("truth capability")), \
                mock.patch.object(wrapper.c,"read_ref",side_effect=[ready,observed]), \
                mock.patch.object(wrapper,"_descriptor",side_effect=lambda function,spec:dict(function=function,spec=spec)):
            descriptor=wrapper.qualification_factory(preparation_ref={},generation_ref={},
                output_directory="experiments/results/NEVER_CREATED_synthetic_qual_v4",
                max_rss_bytes=128*2**30,max_seconds=7200,max_output_bytes=16*2**20)
        self.assertEqual(descriptor["function"],"run_qualification")
        self.assertEqual(descriptor["spec"]["observed_input_refs"],refs)
        self.assertFalse(descriptor["spec"]["truth_or_model_allowed"])
        self.assertTrue(descriptor["spec"]["CPU_only"])
        self.assertEqual(descriptor["spec"]["max_output_bytes"],15*2**20)


    def test_private_fresh_namespace_binding_and_workflow_envelope_preserves_receipt(self):
        original_function=original.prepare_from_loaded_observation_v1
        globals_copy=dict(original_function.__globals__)
        class PrivateBundle:pass
        private_train=("fresh_real_train_000",)
        globals_copy.update(TRAIN=private_train,VAL=("fresh_real_test_000",),
                            AmericanTreeLoomWholeParentV1=PrivateBundle)
        private_function=types.FunctionType(original_function.__code__,globals_copy,
            original_function.__name__,original_function.__defaults__,original_function.__closure__)
        runtime=SimpleNamespace(preparation=SimpleNamespace(prepare_from_loaded_observation_v1=private_function))
        integration.install_runtime_preparation(runtime)
        prepared=runtime.preparation.prepare_from_loaded_observation_v1
        self.assertEqual(prepared.__globals__["TRAIN"],private_train)
        self.assertIs(prepared.__globals__["AmericanTreeLoomWholeParentV1"],PrivateBundle)
        self.assertIsNot(prepared,original_function)
        from arggnn.benchmarking import treeloom_preparation_v4 as wrapper
        from arggnn.benchmarking import contracts as contracts
        from research.reviews.workflow_progress_diagnosis_20261018.treeloom import qualify_parent_v2 as qualifier
        # Exercise the actual maintained wrapper return and then the workflow envelope filename.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            out=Path(temporary);science={"status":"synthetic_saved_receipt"}
            spec=dict(schema=wrapper.QUALIFICATION_SCHEMA,parent_id=wrapper.PARENT,
                candidate_source_refs=wrapper.candidate_source_refs(),CPU_only=True,
                truth_or_model_allowed=False,disk_cache_bytes=0,output_directory=str(out),
                observed_input_refs={})
            def fake_qualify(actual,*,resource_check):
                (out/"qualification.json").write_text(json.dumps(science)+"\n")
                return science
            with mock.patch.object(qualifier,"qualify",side_effect=fake_qualify), \
                    mock.patch.object(wrapper,"_require_pass",return_value=science):
                returned=wrapper.run_qualification(spec,resource_check=lambda stage:None)
            self.assertTrue(returned["path"].endswith("/qualification.json"))
            # Frozen workflow.child saves its envelope at output/result.json.
            (out/"result.json").write_text(json.dumps(dict(status="succeeded",value=returned))+"\n")
            self.assertEqual(contracts.read_ref(returned),science)
            self.assertNotEqual((out/"result.json").resolve(),contracts.contained_path(returned["path"]))

    def test_pool_bound_rejected_before_prepare_and_private_source_clone(self):
        with self.assertRaisesRegex(ValueError,"before preparation"):
            pool.CompactParentPool(parent_bounds={"p":100},input_refs={},external_arrays={},
                max_cache_bytes=99,max_rss_bytes=1000,preparation_headroom_bytes=100,
                rss_bytes=lambda:0,resource_check=lambda s:None,event=lambda *args:None)
        # The full-parent wrapper preserves original source globals and only clones its function.
        self.assertIs(original.encode_block.__module__.endswith("optimized_observations_v1"),True)
        self.assertNotEqual(integration.compiled_complete(),original.prepare_from_loaded_observation_v1)


if __name__=="__main__":unittest.main()
