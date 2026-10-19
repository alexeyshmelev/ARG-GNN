"""European qualification source/observation gates and deferred callback routing."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from arggnn.benchmarking import european_preparation_v9 as wrapper


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.refs={key:dict(path="experiments/results/mock/"+key+".json",sha256="a"*64)
                   for key in wrapper.OBSERVED}
        self.spec=dict(protocol_id=wrapper.integration.AP,sampling_plan_ref=dict(path="mock/plan",sha256="b"*64),
            max_sites=10,max_metadata_bytes=1024,preparation_limits=dict(max_nodes=1000,max_holds=10,
                                                                       max_episode_payload_bytes=2**20))
        self.bound=SimpleNamespace(parent_plan=lambda *args:dict(train_parent_ids=["p0","p1","p2"]))

    def passed(self):
        return dict(schema="european_v9_complete_parent_CPU_qualification_v1",
            status="passed_full_observed_parent_CPU_preparation",protocol_id=self.spec["protocol_id"],
            parent_id="p0",sampling_plan_ref=self.spec["sampling_plan_ref"],observed_inputs=self.refs,
            preparation_contract=wrapper.qualification.contract(self.spec),
            candidate_source_refs=wrapper.candidate_source_refs(),all_native_trees_nodes_sites_samples_retained=True,
            sampled_rows_and_repeated_draws_preserved=True,private_truth_opened=False,training_labels_opened=False,
            model_instantiated=False,CUDA_called=False,GPU_fit_qualified=False,full_parent_preparation_count=1,
            disk_cache_bytes=0,scientific_outputs_created=0,observed_peak_rss_bytes=1000,
            elapsed_seconds=1.,observed_tensor_bytes=1000,complete_signature_sha256="c"*64)

    def test_gate_binds_protocol_plan_sources_observations_and_no_truth(self):
        valid=self.passed()
        with patch.object(wrapper.c,"read_ref",return_value=valid), \
                patch.object(wrapper.integration,"backend",return_value=self.bound):
            self.assertIs(wrapper._require_pass({},self.spec,self.refs),valid)
        for bad in (dict(valid,protocol_id=wrapper.integration.C4),dict(valid,candidate_source_refs=[]),
                    dict(valid,training_labels_opened=True),dict(valid,full_parent_preparation_count=2),
                    dict(valid,preparation_contract={}),dict(valid,observed_inputs={})):
            with self.subTest(bad=bad),patch.object(wrapper.c,"read_ref",return_value=bad), \
                    patch.object(wrapper.integration,"backend",return_value=self.bound):
                with self.assertRaises(wrapper.c.ContractError):wrapper._require_pass({},self.spec,self.refs)

    def test_both_qualification_factories_route_CPU_native_scope_without_target_refs(self):
        output="experiments/results/NEVER_CREATED_european_v9_qualification/native"
        for protocol in wrapper.integration.PROTOCOLS:
            with patch.object(wrapper.integration,"backend",return_value=self.bound), \
                    patch.object(wrapper,"_descriptor",side_effect=lambda function,spec:
                                 dict(module=wrapper.MODULE,function=function,spec=spec)):
                call=wrapper.qualification_factory(protocol_id=protocol,sampling_plan_ref=self.spec["sampling_plan_ref"],
                    parent_id="p0",observed_inputs=self.refs,output_directory=output,
                    max_sites=10,max_metadata_bytes=1024,preparation_limits=self.spec["preparation_limits"])
            self.assertEqual(call["function"],"run_qualification")
            self.assertEqual(call["spec"]["protocol_id"],protocol)
            self.assertEqual(call["spec"]["output_directory"],output)
            self.assertEqual(set(call["spec"]["observed_inputs"]),set(wrapper.OBSERVED))
            self.assertTrue(call["spec"]["CPU_only"])
            self.assertFalse(wrapper.c.contained_path(output).exists())

    def test_saved_scientific_qualification_ref_survives_workflow_envelope(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            output=Path(temporary)/"native";result=self.passed()
            spec=dict(self.spec,schema=wrapper.QUAL_SCHEMA,parent_id="p0",CPU_only=True,
                candidate_source_refs=wrapper.candidate_source_refs(),observed_inputs=self.refs,
                output_directory=str(output))
            def qualify(actual,*,resource_check):
                output.mkdir();(output/"qualification.json").write_text(json.dumps(result)+"\n")
                return result
            with patch.object(wrapper.qualification,"qualify",side_effect=qualify), \
                    patch.object(wrapper.integration,"backend",return_value=self.bound):
                ref=wrapper.run_qualification(spec,resource_check=lambda stage:None)
            (Path(temporary)/"result.json").write_text(json.dumps(dict(value=ref,job_ref={}))+"\n")
            self.assertTrue(ref["path"].endswith("/native/qualification.json"))
            self.assertEqual(wrapper.c.read_ref(ref),result)

    def test_fit_requires_and_forwards_handoff_callback_predictions_do_not(self):
        fit=Mock(return_value={"fit":True});predict=Mock(return_value={"predict":True})
        bound=SimpleNamespace(run_fit=fit,run_predictions=predict);science={"same":"science"}
        resource=lambda stage:None;acquire=lambda:None
        with patch.object(wrapper,"_validate",return_value=(bound,science)):
            self.assertEqual(wrapper.run_fit({},resource_check=resource,acquire_gpu0=acquire),{"fit":True})
            self.assertEqual(wrapper.run_predictions({},resource_check=resource),{"predict":True})
        fit.assert_called_once_with(science,resource_check=resource,acquire_gpu0=acquire)
        predict.assert_called_once_with(science,resource_check=resource)


if __name__=="__main__":unittest.main()
