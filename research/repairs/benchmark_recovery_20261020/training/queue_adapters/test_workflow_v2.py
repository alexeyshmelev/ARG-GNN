"""Focused synthetic metadata checks for the original workflow handoff."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
from . import american_workflow_v2 as workflow


def reference(name):
    return dict(path="experiments/results/synthetic_metadata/"+name, sha256="0"*64)


class WorkflowBindingTests(unittest.TestCase):
    def test_original_factory_kwargs_preserve_low_level_scientific_spec(self):
        fit = dict(checkpoint_ref=reference("final.pt"), completion_ref=reference("completed.json"))
        prepared_ref = reference("preparation.json")
        prepared = dict(observation_ref=reference("observed.json"), focals_ref=reference("focals.npz"))
        registry = reference("registry.json")
        wrapper_source = reference("wrapper.py")
        def file_ref(path):
            return registry if path == workflow.REGISTRY else wrapper_source
        def factory(**kwargs):
            return dict(module="synthetic_low_level", function="run_predictions", spec=kwargs,
                        python=".runtime/env/bin/python", source_refs=[])
        with patch.object(workflow.c, "read_ref", return_value=prepared), \
             patch.object(workflow.c, "file_ref", side_effect=file_ref), \
             patch.object(workflow.model, "prediction_factory", side_effect=factory):
            result = workflow.prediction_factory(fit_result=fit, preparation_refs={"heldout":prepared_ref},
                output_directory="experiments/results/new/native", max_rss_bytes=256*2**30)
        expected = dict(parent_registry_ref=registry, checkpoint_ref=fit["checkpoint_ref"],
            training_completion_ref=fit["completion_ref"],
            heldout_inputs={"heldout":dict(observation_ref=prepared["observation_ref"],
                focal_registry_ref=prepared["focals_ref"], native_preparation_ref=prepared_ref)},
            output_directory="experiments/results/new/native", max_rss_bytes=256*2**30,
            preparation_headroom_bytes=128*2**30, prediction_headroom_bytes=64*2**30,
            observation_cache_bytes=0, max_output_bytes=64*2**20)
        self.assertEqual(result["spec"], expected)
        self.assertEqual(result["source_refs"], [wrapper_source])
        self.assertIs(workflow._fit_result(fit), fit)

    def test_adopted_completion_ref_preserves_actual_checkpoint_and_identity(self):
        complete_ref = reference("completed.json")
        complete = dict(schema="treeloom_fresh_full_training_completion_v1",
            status="completed_full20parents30passes600updates", checkpoint_ref=reference("final.pt"),
            parent_registry_ref=reference("registry.json"), complete_parent_updates=600)
        with patch.object(workflow.c, "read_ref", return_value=complete):
            result = workflow._fit_result(complete_ref)
        self.assertEqual(result, dict(complete, completion_ref=complete_ref))
        self.assertNotIn("completion_ref", complete)
        for invalid in (dict(complete, schema="unrelated"),
                        dict(complete, status="incomplete"),
                        dict(complete, completion_ref=reference("another_completion.json"))):
            with patch.object(workflow.c, "read_ref", return_value=invalid):
                with self.assertRaises(ValueError):
                    workflow._fit_result(complete_ref)
        # The unchanged low-level factory retains full600/49-state validators.
        self.assertIsNotNone(workflow.model._completion_binding)
        self.assertIsNotNone(workflow.model.validate_saved_checkpoint)

    def test_ordinary_wrapper_source(self):
        source = Path(workflow.__file__).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                function = node.func
                name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
                self.assertNotIn(name, {"exec", "eval", "compile", "FunctionType", "_private_module"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
