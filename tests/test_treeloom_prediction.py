"""CPU checkpoint/identity contracts; no GPU or full-parent fit is executed."""
import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from arggnn.benchmarking import treeloom_prediction as prediction


class SavedPredictionContractTest(unittest.TestCase):
    def setUp(self):
        self.parents = ("fresh_validation_000", "fresh_test_000")
        self.reference = dict(path="mock/registry.json", sha256="a"*64)
        self.registry = SimpleNamespace(reference=self.reference,
            split_for=lambda parent: "validation" if parent == self.parents[0] else
                "test" if parent == self.parents[1] else "train",
            ids=lambda split: tuple("fresh_train_%03d" % i for i in range(20)) if split == "train" else
                (self.parents[0],) if split == "validation" else (self.parents[1],))
        self.ref = lambda path: dict(path="mock/"+path, sha256="b"*64)

    def make_checkpoint(self):
        import torch
        class Tiny49(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameters49 = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.ones(32084))] +
                    [torch.nn.Parameter(torch.ones(1)) for _ in range(48)])
            def checkpoint_contract(self):
                return dict(genuine_test_contract=True, expected_parameters=32132)
        model = Tiny49()
        optimizer = dict(state={i:dict(step=torch.tensor(600.),
            exp_avg=torch.zeros_like(p), exp_avg_sq=torch.ones_like(p))
            for i,p in enumerate(model.parameters())},
            param_groups=[dict(params=list(range(49)),lr=.001,weight_decay=.0001,
                betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False)])
        completion = dict(observation_barrier_ref=self.ref("barrier.json"))
        saved = dict(schema="treeloom_fresh_full_checkpoint_v1",epoch=30,
            complete_parent_updates=600,model_seed=70001,parent_registry_ref=self.reference,
            observation_barrier_ref=completion["observation_barrier_ref"],
            settings=copy.deepcopy(prediction.training.SETTINGS),
            contract=model.checkpoint_contract(),checkpoint_selection="fixed_final_epoch30",
            VAL_truth_read=False,TEST_truth_read=False,
            model_state={k:v.detach().clone() for k,v in model.state_dict().items()},
            optimizer_state=optimizer)
        return model, saved, completion

    def test_exact_saved600_counters_and_model_load(self):
        model, saved, completion = self.make_checkpoint()
        result = prediction.validate_saved_checkpoint(saved, model, self.registry, completion)
        self.assertIs(result, model)
        defects = [
            lambda s: s.update(epoch=25),
            lambda s: s.update(complete_parent_updates=599),
            lambda s: s.update(TEST_truth_read=True),
            lambda s: s["optimizer_state"]["state"][48].update(step=599),
            lambda s: s["optimizer_state"]["state"].pop(48),
            lambda s: s["optimizer_state"]["param_groups"][0].update(lr=.01),
            lambda s: s["model_state"].pop(next(iter(s["model_state"]))),
        ]
        for defect in defects:
            broken = copy.deepcopy(saved)
            defect(broken)
            with self.subTest(defect=defect), self.assertRaises((prediction.c.ContractError, KeyError)):
                prediction.validate_saved_checkpoint(broken, model, self.registry, completion)
        import torch
        broken = copy.deepcopy(saved)
        broken["optimizer_state"]["state"][0]["exp_avg_sq"][0] = -1.
        with self.assertRaises(prediction.c.ContractError):
            prediction.validate_saved_checkpoint(broken, model, self.registry, completion)
        broken = copy.deepcopy(saved)
        broken["model_state"][next(iter(broken["model_state"]))][0] = torch.nan
        with self.assertRaises(prediction.c.ContractError):
            prediction.validate_saved_checkpoint(broken, model, self.registry, completion)

    def test_heldout_identity_refs_and_truth_denial_before_reads(self):
        parent = self.parents[0]
        item = {k:self.ref(k) for k in prediction.fresh.INPUT_REFS}
        observed = dict(parent_id=parent,split="val",sequence_length_bp=30000000,
            class_order=list(prediction.fresh.CLASSES),genealogy_kind="same",
            query_ancestry_truth_in_inputs=False)
        prepared = dict(parent_id=parent,class_order=list(prediction.fresh.CLASSES),
            genealogy_kind="same",query_ancestry_truth_opened=False,
            observation_ref=item["observation_ref"],focals_ref=item["focal_registry_ref"])
        def read(ref):
            return observed if ref == item["observation_ref"] else prepared
        with patch.object(prediction.c,"read_ref",side_effect=read) as reader:
            prediction._heldout_bindings(self.registry,{parent:item})
            self.assertEqual(reader.call_count,2)
            observed["class_order"] = ["AFR","EUR","EAS"]
            with self.assertRaises(prediction.c.ContractError):
                prediction._heldout_bindings(self.registry,{parent:item})
        for inputs in ({parent:dict(item,truth_ref=self.ref("private_truth"))},
                       {"fresh_train_000":item},
                       {parent:dict(item,observation_ref=dict(path="private/truth.npz",sha256="b"*64))}):
            with patch.object(prediction.c,"read_ref",side_effect=AssertionError("must not read")):
                with self.assertRaises(prediction.c.ContractError):
                    prediction._heldout_bindings(self.registry,inputs)

    def test_final_completion_and_observation_barrier_are_both_required(self):
        checkpoint_ref, completion_ref = self.ref("epoch_30.pt"), self.ref("completed.json")
        barrier_ref = self.ref("barrier.json")
        complete = dict(schema="treeloom_fresh_full_training_completion_v1",
            status="completed_full20parents30passes600updates",parent_registry_ref=self.reference,
            checkpoint_ref=checkpoint_ref,selected_epoch=30,complete_parent_updates=600,completed_epochs=30,
            checkpoint_selection="fixed_final_epoch30",final_saved_optimizer_parameter_states=49,
            final_saved_optimizer_steps=[600],VAL_truth_read=False,TEST_truth_read=False,
            observation_barrier_ref=barrier_ref)
        barrier = dict(parent_registry_ref=self.reference,
            train_parent_ids=list(self.registry.ids("train")),
            parents=dict.fromkeys(self.registry.ids("train"),{}),
            all20_observations_sealed_before_truth=True,private_truth_opened=False)
        with patch.object(prediction.c,"read_ref",side_effect=lambda ref:complete if ref==completion_ref else barrier):
            self.assertIs(prediction._completion_binding(
                self.registry,checkpoint_ref,completion_ref),complete)
            for mapping,key,value in ((complete,"selected_epoch",25),
                                      (complete,"checkpoint_ref",self.ref("other.pt")),
                                      (barrier,"all20_observations_sealed_before_truth",False)):
                old=mapping[key];mapping[key]=value
                with self.assertRaises(prediction.c.ContractError):
                    prediction._completion_binding(self.registry,checkpoint_ref,completion_ref)
                mapping[key]=old

    def test_import_is_inert(self):
        code = ("import sys;sys.path.insert(0,%r);" % str(Path(prediction.__file__).resolve().parents[2])
                + "from arggnn.benchmarking import treeloom_prediction;"
                + "assert not any(n in sys.modules for n in ('torch','numpy','tskit','torch_geometric'))")
        subprocess.run([sys.executable,"-s","-B","-c",code],check=True,
                       env=dict(os.environ,CUDA_VISIBLE_DEVICES=""))


if __name__ == "__main__":
    unittest.main()
