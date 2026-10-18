"""Fake90-update schedule and early reference contracts; no GPU/data launch."""
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from arggnn.benchmarking import european_training as training


class EuropeanFullTrainingTest(unittest.TestCase):
    def test_three_full_parents_thirty_passes_actual90_steps_and_final_epoch(self):
        parents = ("ap_train_000","ap_train_001","ap_train_002")
        visits,events,checkpoints = [],[],[]
        count = [0]
        def load(parent):
            return SimpleNamespace(parent_id=parent),{}
        def targets(parent,bundle,metadata):
            self.assertEqual(parent,bundle.parent_id)
            return object()
        def step(bundle,target,metadata):
            count[0] += 1
            visits.append(bundle.parent_id)
            return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True)
        def checkpoint(epoch,updates):
            checkpoints.append((epoch,updates))
            return dict(path="mock/epoch_%02d.pt"%epoch,sha256="a"*64)
        kwargs = dict(parents=parents,load_parent=load,join_targets=targets,step=step,
            optimizer_counts=lambda:[count[0]]*49,checkpoint=checkpoint,check=lambda *args:None,
            event=events.append)
        result = training.fixed_training_loop(**kwargs)
        expected = []
        for epoch in range(1,31):
            order = list(parents);random.Random(101+epoch).shuffle(order);expected.extend(order)
        self.assertEqual(visits,expected)
        self.assertEqual(result["optimizer_updates"],90)
        self.assertEqual(result["completed_epochs"],30)
        self.assertEqual(result["selected_epoch"],30)
        self.assertEqual(checkpoints,[(e,3*e) for e in (5,10,15,20,25,30)])
        self.assertEqual(len(events),90)
        count[0] = 0;visits.clear()
        kwargs["optimizer_counts"] = lambda:[count[0]]*48+[0]
        with self.assertRaises(training.c.ContractError):
            training.fixed_training_loop(**kwargs)
        self.assertEqual(len(visits),1)

    def test_no_target_key_or_unregistered_parent_in_heldout_inputs(self):
        ref = dict(path="mock/input.json",sha256="a"*64)
        item = dict.fromkeys(training.INPUTS,ref)
        training._input_shapes({"test":item},["test"],training=False)
        for inputs in ({"test":dict(item,training_targets_ref=ref)},
                       {"test":dict(item,truth_ref=ref)},{"train":item}):
            with self.assertRaises(training.c.ContractError):
                training._input_shapes(inputs,["test"],training=False)
        training._input_shapes({"train":dict(item,training_targets_ref=ref)},["train"],training=True)

    def test_targets_cannot_bind_other_parent_registry_or_class_order(self):
        metadata = dict(schema="ap_paper_TRAIN_native_targets_v1",parent_id="train",split="train",
            registry_ref=dict(path="mock/registry.json",sha256="a"*64),
            no_label_dependent_selection=True,native_path_names=list(training.european.NATIVE_SIX_CLASSES))
        bundle = SimpleNamespace(sampled_registry_ref=metadata["registry_ref"])
        model = SimpleNamespace(output_classes=training.european.NATIVE_SIX_CLASSES)
        for key,value in (("parent_id","other"),("split","test"),
                          ("registry_ref",dict(path="mock/other.json",sha256="a"*64)),
                          ("native_path_names",["Ana","CHG","WHG","EHG"])):
            broken = copy.deepcopy(metadata);broken[key]=value
            with patch.object(training.european,"_json_ref",return_value=broken), \
                 patch.object(training.european,"_bound",side_effect=AssertionError("must deny before array")):
                with self.assertRaises(training.c.ContractError):
                    training._training_targets(dict(path="mock/targets.json",sha256="b"*64),
                        "train",bundle,model,max_metadata_bytes=2**20)

    def test_factories_return_workflow_call_shapes(self):
        expected = dict(requirements=dict(source_refs=[]))
        for factory,builder,name in ((training.fit_factory,"build_fit_spec","run_fit"),
                                     (training.prediction_factory,"build_prediction_spec","run_predictions")):
            with patch.object(training,builder,return_value=expected) as build:
                call = factory(test_argument=True)
                build.assert_called_once_with(test_argument=True)
                self.assertEqual(call["function"],name)
                self.assertEqual(call["module"],"arggnn.benchmarking.european_training")
                self.assertEqual(call["python"],".runtime/env/bin/python")
                self.assertIs(call["spec"],expected)


if __name__ == "__main__":
    unittest.main()
