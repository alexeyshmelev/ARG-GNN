"""V6 bounded metadata/lifecycle regression; stop exact qualifier before expensive work."""
import copy
import os
from pathlib import Path
import tempfile
from types import FunctionType,SimpleNamespace
import unittest
from unittest import mock
from arggnn.benchmarking import treeloom_preparation_v6 as candidate
from arggnn.benchmarking import treeloom_preparation_v5 as frozen
from .test_v5_factory_lineage import fixture


def descriptor(function,spec):
    return dict(module="arggnn.benchmarking.treeloom_preparation_v4",
                function=function,spec=spec,source_refs=[])


class V6FactoryLifecycleRegression(unittest.TestCase):
    def test_distinct_raw_inferred_and_exact1600_identity_metadata(self):
        refs,values=fixture();seen=[]
        def read(ref):
            seen.append(ref["path"])
            self.assertIn(ref["path"],values,"Scientific/private payload capability")
            return values[ref["path"]]
        with mock.patch.object(frozen,"_read_metadata",side_effect=read):
            observed,lineage=candidate.observed_lineage(refs["preparation"],refs["generation"])
        self.assertNotEqual(refs["raw"],refs["inferred"])
        self.assertEqual(observed["observation_ref"],refs["inferred"])
        self.assertEqual(lineage["samples"],1600)
        self.assertEqual(set(seen),set(values))
        self.assertFalse(lineage["scientific_payloads_opened"])
        self.assertFalse(lineage["private_truth_opened"])

    def test_all_existing_metadata_lineage_rejections_through_new_factory(self):
        refs,original=fixture()
        def source(v):v[refs["inferred"]["path"]]["source_observation_ref"]=refs["inferred"]
        def genotype(v):v[refs["inferred"]["path"]]["genotypes_ref"]=refs["positions"]
        def sample(v):v[refs["inferred_roster"]["path"]]["samples"][600]["sample_id"]="wrong_query"
        def label(v):v[refs["inferred_roster"]["path"]]["samples"][600]["observed_reference_class"]=0
        def recipe(v):v[refs["inference"]["path"]]["recipe_ref"]=refs["campaign"]
        def chain(v):v[refs["inference"]["path"]]["input_ref"]=refs["preparation"]
        for corrupt in (source,genotype,sample,label,recipe,chain):
            values=copy.deepcopy(original);corrupt(values)
            with self.subTest(corrupt=corrupt.__name__), \
                    mock.patch.object(frozen,"_read_metadata",side_effect=lambda ref:values[ref["path"]]):
                with self.assertRaises(candidate.c.ContractError):
                    candidate.qualification_factory(preparation_ref=refs["preparation"],
                        generation_ref=refs["generation"],
                        output_directory="experiments/results/NEVER_CREATED_synthetic_v6/native",
                        max_rss_bytes=128*2**30,max_seconds=7200,max_output_bytes=16*2**20)

    def test_materialize_then_supervisor_then_exact_frozen_output_mkdir(self):
        from . import qualify_parent_v2 as qualifier
        refs,values=fixture()
        class ReachedAfterExactMkdir(Exception):pass
        def stop_before_expensive_work():
            raise ReachedAfterExactMkdir("frozen qualifier out.mkdir succeeded")
        # Clone the exact frozen code object; interrupt at its first monotonic()
        # immediately after out.mkdir, before its try/import torch/load sequence.
        exact=FunctionType(qualifier.qualify.__code__,
            dict(qualifier.qualify.__globals__,time=SimpleNamespace(monotonic=stop_before_expensive_work)),
            qualifier.qualify.__name__,qualifier.qualify.__defaults__,qualifier.qualify.__closure__)
        self.assertIs(exact.__code__,qualifier.qualify.__code__)
        with tempfile.TemporaryDirectory(prefix="test_v6_lifecycle_",dir=candidate.c.ROOT/"experiments/results") as temporary:
            task=Path(temporary)/"future_task";native=task/"native"
            self.assertFalse(task.exists())
            with mock.patch.object(frozen,"_read_metadata",side_effect=lambda ref:values[ref["path"]]), \
                    mock.patch.object(frozen,"_descriptor",side_effect=descriptor):
                call=candidate.qualification_factory(preparation_ref=refs["preparation"],
                    generation_ref=refs["generation"],output_directory=str(native),
                    max_rss_bytes=128*2**30,max_seconds=7200,max_output_bytes=16*2**20)
            self.assertFalse(task.exists());self.assertFalse(native.exists())
            self.assertEqual(candidate.c.contained_path(call["spec"]["output_directory"]),native)
            # Actual supervisor layout before the scientific child starts.
            task.mkdir();(task/"tmp").mkdir();(task/"cache").mkdir()
            (task/"stage.log").write_text("supervisor exists before scientific child\n")
            self.assertFalse(native.exists())
            with mock.patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
                with self.assertRaises(ReachedAfterExactMkdir):
                    exact(call["spec"],resource_check=lambda stage:None)
            self.assertTrue(native.is_dir())
            self.assertFalse((native/"qualification").exists())
            self.assertTrue((task/"tmp").is_dir() and (task/"cache").is_dir())
            self.assertEqual((task/"stage.log").read_text(),"supervisor exists before scientific child\n")
            self.assertEqual(call["module"],"arggnn.benchmarking.treeloom_preparation_v4")
            self.assertIs(candidate.run_qualification,frozen.run_qualification)
            self.assertIs(candidate.run_fit,frozen.run_fit)

    def test_existing_native_scope_is_rejected_and_fit_delegates_frozen(self):
        with tempfile.TemporaryDirectory(prefix="test_v6_existing_",dir=candidate.c.ROOT/"experiments/results") as temporary:
            native=Path(temporary)/"native";native.mkdir()
            with self.assertRaises(candidate.c.ContractError):
                candidate.qualification_factory(preparation_ref={},generation_ref={},
                    output_directory=str(native),max_rss_bytes=128*2**30)
        with mock.patch.object(frozen,"fit_factory",return_value=dict(source_refs=[],module="frozen",function="run_fit")) as delegated:
            call=candidate.fit_factory(preparation_refs={},generation_refs={},
                output_directory="experiments/results/NEVER_CREATED_synthetic_v6_fit/native",
                max_rss_bytes=512*2**30,qualification_ref={})
        delegated.assert_called_once()
        self.assertIn(candidate.c.file_ref(candidate.__file__),call["source_refs"])


if __name__=="__main__":unittest.main()
