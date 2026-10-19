"""V8 route, source-gate and scientific receipt checks; all numerical work mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking import treeloom_preparation_v8 as wrapper
from arggnn.benchmarking import american_workflow as american


class WrapperV8Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.refs = {key: dict(path="mock/" + key, sha256="a" * 64)
                     for key in ("observation_ref", "focal_registry_ref", "native_preparation_ref")}

    def qualified(self):
        return dict(schema="one_full_AMtrain000_exact_compact_CPU_qualification_v2",
            status="passed_complete_parent_exact_preparation_and_two_reuses", parent_id=wrapper.PARENT,
            observed_input_refs=self.refs, candidate_source_refs=wrapper.candidate_source_refs(),
            private_truth_opened=False, models_executed=False, CUDA_called=False,
            scientific_outputs_created=0, disk_cache_bytes=0, full_parent_preparation_count=1,
            exact_restores=2, all_original_tensor_and_metadata_signatures_equal=True,
            original_complete_parent_retained=True, compact_charged_bytes=1000,
            observed_peak_rss_bytes=1000, elapsed_seconds=1.)

    def test_factory_preserves_native_scope_and_routes_to_repaired_qualification(self):
        output = str(self.directory / "native")
        spec = dict(output_directory=output, observed_input_refs=self.refs,
                    candidate_source_refs=wrapper.frozen.candidate_source_refs())
        with patch.object(wrapper.lifecycle, "qualification_factory", return_value=dict(spec=spec)) as factory, \
                patch.object(wrapper, "_source_refs", return_value=wrapper.candidate_source_refs()):
            descriptor = wrapper.qualification_factory(preparation_ref={}, generation_ref={},
                output_directory=output, max_rss_bytes=128*wrapper.GIB)
        self.assertEqual(descriptor["module"], wrapper.MODULE)
        self.assertEqual(descriptor["function"], "run_qualification")
        self.assertEqual(descriptor["spec"]["output_directory"], output)
        self.assertEqual(factory.call_args.kwargs["output_directory"], output)
        self.assertEqual(descriptor["spec"]["candidate_source_refs"], wrapper.candidate_source_refs())
        paths = {ref["path"] for ref in descriptor["spec"]["candidate_source_refs"]}
        self.assertIn(wrapper.REPAIR + "cut_hoist_v1.py", paths)
        self.assertIn(wrapper.REPAIR + "integration_v1.py", paths)
        self.assertIn("src/arggnn/benchmarking/treeloom_preparation_v8.py", paths)

    def test_scientific_qualification_ref_is_preserved_below_workflow_envelope(self):
        output = self.directory / "native"
        result = self.qualified()
        spec = dict(schema=wrapper.frozen.QUALIFICATION_SCHEMA, parent_id=wrapper.PARENT,
            candidate_source_refs=wrapper.candidate_source_refs(), CPU_only=True,
            truth_or_model_allowed=False, disk_cache_bytes=0, output_directory=str(output),
            observed_input_refs=self.refs)
        def qualify(actual, *, resource_check):
            self.assertEqual(actual, spec)
            output.mkdir()
            (output / "qualification.json").write_text(json.dumps(result) + "\n")
            return result
        with patch.object(wrapper.repaired, "qualify", side_effect=qualify) as runner:
            ref = wrapper.run_qualification(spec, resource_check=lambda stage: None)
        self.assertTrue(ref["path"].endswith("/native/qualification.json"))
        envelope = self.directory / "result.json"
        envelope.write_text(json.dumps(dict(value=ref, job_ref={})) + "\n")
        self.assertEqual(wrapper.c.read_ref(ref), result)
        self.assertNotEqual(wrapper.c.contained_path(ref["path"]), envelope)
        runner.assert_called_once()

    def test_gate_rejects_previous_candidate_sources_and_envelope_as_qualification(self):
        previous = dict(self.qualified(), candidate_source_refs=wrapper.frozen.candidate_source_refs())
        for bad in (previous, dict(value=self.qualified(), job_ref={})):
            with patch.object(wrapper.c, "read_ref", return_value=bad):
                with self.assertRaises(wrapper.c.ContractError):
                    wrapper._require_pass({}, self.refs)

    def test_all20_fit_factory_and_run_fit_use_same_qualification_and_repaired_training(self):
        parents = tuple("argformer_american_train_%03d" % at for at in range(20))
        preparation_refs = {parent: {"preparation": parent} for parent in parents}
        generation_refs = {parent: {"generation": parent} for parent in parents}
        qualification = self.directory / "native_qualification.json"
        qualification.write_text(json.dumps(self.qualified()) + "\n")
        qualification_ref = wrapper.c.file_ref(qualification)
        output = self.directory / "fit_native"
        registry_ref = dict(path="mock/registry", sha256="b" * 64)
        actual_read = wrapper.c.read_ref
        current_sources = wrapper.candidate_source_refs()
        def read(ref):
            if ref == registry_ref:
                return dict(parents=[dict(parent_id=parent, split="train") for parent in parents])
            return actual_read(ref)
        def binding(parent, preparation_ref, generation_ref):
            self.assertEqual(preparation_ref, preparation_refs[parent])
            self.assertEqual(generation_ref, generation_refs[parent])
            return dict(self.refs, truth_ref={"parent": parent})
        def build_spec(**kwargs):
            self.assertEqual(set(kwargs["train_inputs"]), set(parents))
            self.assertEqual(kwargs["max_rss_bytes"], 512*wrapper.GIB)
            return dict(output_directory=str(output), train_inputs=kwargs["train_inputs"])
        with patch.object(wrapper.c, "file_ref", return_value=registry_ref), \
                patch.object(wrapper.c, "read_ref", side_effect=read), \
                patch.object(american, "observed_binding", return_value=self.refs), \
                patch.object(american, "train_binding", side_effect=binding) as train_binding, \
                patch.object(wrapper.preparation, "build_spec", side_effect=build_spec), \
                patch.object(wrapper, "_descriptor", side_effect=lambda function, spec:
                    dict(module=wrapper.MODULE, function=function, spec=spec)), \
                patch.object(wrapper, "candidate_source_refs", return_value=current_sources):
            descriptor = wrapper.fit_factory(preparation_refs=preparation_refs, generation_refs=generation_refs,
                output_directory=str(output), max_rss_bytes=512*wrapper.GIB,
                qualification_ref=qualification_ref)
        self.assertEqual(train_binding.call_count, 20)
        self.assertEqual(descriptor["function"], "run_fit")
        self.assertEqual(descriptor["spec"]["qualification_ref"], qualification_ref)
        completed = dict(status="completed_full20parents30passes600updates", complete_parent_updates=600,
                         checkpoint_ref=dict(path="mock/checkpoint", sha256="c" * 64))
        def train(spec, *, resource_check):
            self.assertEqual(spec, descriptor["spec"]["training_spec"])
            output.mkdir()
            (output / "training_completed.json").write_text(json.dumps(completed) + "\n")
            return completed
        with patch.object(wrapper.repaired, "run_training", side_effect=train) as runner:
            result = wrapper.run_fit(descriptor["spec"], resource_check=lambda stage: None)
        self.assertEqual(result["complete_parent_updates"], 600)
        self.assertEqual(wrapper.c.read_ref(result["completion_ref"]), completed)
        runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
