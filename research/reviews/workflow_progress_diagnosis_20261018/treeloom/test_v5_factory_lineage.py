"""Metadata-only regression: raw generation and inferred predictor refs differ."""
import copy
from pathlib import Path
import unittest
import tempfile
import json
from unittest import mock
from arggnn.benchmarking import treeloom_preparation_v5 as candidate
from arggnn.benchmarking import treeloom_preparation_v4 as frozen
from arggnn.benchmarking import american_observation as observation


def fixture():
    def ref(key,suffix="json"):
        return dict(path="mock/"+key+"."+suffix,sha256=(key.encode().hex()+"0"*64)[:64])
    names=("generation","preparation","raw","inferred","inference","raw_roster","inferred_roster",
           "genotypes","positions","alleles","focals","raw_tree","inferred_tree","campaign","comparison","recipe")
    refs={name:ref(name,"npy" if name in ("genotypes","positions") else "json") for name in names}
    labels=[2]*200+[0]*200+[1]*200+[None]*1000
    rows=[dict(sample_index=i,sample_id="sample_%04d"%i,haplotype=i%2,
               age_generations=400 if i%3==0 else 0,observed_reference_class=labels[i],
               role="pure_source_reference" if i<600 else "query",observed_node_id=i)
          for i in range(1600)]
    class_order=["EUR","EAS","AFR"];parent=candidate.PARENT
    roster=dict(parent_id=parent,class_order=class_order,samples=rows)
    inferred_roster=copy.deepcopy(roster)
    for row in inferred_roster["samples"]:row["observed_node_id"]+=10000
    inferred_roster["sample_index_to_inferred_node"]=[row["observed_node_id"] for row in inferred_roster["samples"]]
    shared=dict(parent_id=parent,split="train",class_order=class_order,sequence_length_bp=30000000,
        schema="american_c3_observed_v1",genotypes_ref=refs["genotypes"],positions_ref=refs["positions"],
        alleles_ref=refs["alleles"],sample_unit="haploid",counts_haploid={"ADMIX":1000,"AFR":200,"ASIA":200,"EUR":200},
        query_ancestry_truth_in_inputs=False)
    raw=dict(shared,genealogy_kind="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG",
             roster_ref=refs["raw_roster"],tree_ref=refs["raw_tree"])
    inferred=dict(shared,genealogy_kind=observation.GENEALOGY,roster_ref=refs["inferred_roster"],
                  tree_ref=refs["inferred_tree"],source_observation_ref=refs["raw"],
                  sites_and_sample_identities_verified=True)
    common=dict(parent_id=parent,split="train",seeds=dict(sampling=77,ancestry=1,mutations=2),
                campaign_ref=refs["campaign"],comparison_ref=refs["comparison"],recipe_ref=refs["recipe"],
                query_truth_in_training_inputs=False)
    generation=dict(common,schema="american_fresh_shared_observation_ready_v1",
                    observation_ref=refs["raw"],simulated_true_topology_is_predictor_input=False)
    inference=dict(common,schema="american_fresh_inference_ready_v1",
                   input_ref=refs["generation"],observation_ref=refs["inferred"])
    prepared=dict(common,schema="american_fresh_compact_focal_preparation_v1",
        status="completed_compact_query_focals",input_ref=refs["inference"],
        observation_ref=refs["inferred"],genealogy_kind=observation.GENEALOGY,
        query_ancestry_truth_opened=False,class_order=class_order,sequence_length_bp=30000000,
        query_focal_count=10000,all_original_samples_and_trees_traversed=True,
        original_record_ids_preserved=True,path_sampling_seed=77,focals_ref=refs["focals"])
    values=dict(generation=generation,preparation=prepared,raw=raw,inferred=inferred,inference=inference,
                raw_roster=roster,inferred_roster=inferred_roster)
    return refs,{refs[key]["path"]:value for key,value in values.items()}


class V5FactoryLineageRegression(unittest.TestCase):
    def test_distinct_raw_inferred_refs_and_changed_node_mapping_preserve_all1600_identities(self):
        refs,values=fixture();seen=[]
        def read(ref):
            seen.append(ref["path"])
            self.assertIn(ref["path"],values,"No tree/genotype/focal/private payload capability")
            return values[ref["path"]]
        with mock.patch.object(candidate,"_read_metadata",side_effect=read):
            observed,lineage=candidate.observed_lineage(refs["preparation"],refs["generation"])
        self.assertNotEqual(refs["raw"],refs["inferred"])
        self.assertEqual(observed["observation_ref"],refs["inferred"])
        self.assertEqual(lineage["raw_observation_ref"],refs["raw"])
        self.assertEqual(lineage["samples"],1600)
        self.assertTrue(lineage["exact_ordered_sample_identity_roles_labels_ages"])
        self.assertEqual(set(seen),set(values))
        self.assertFalse(lineage["scientific_payloads_opened"])
        self.assertFalse(lineage["private_truth_opened"])

    def test_wrong_source_observation_genotype_query_identity_or_recipe_is_rejected(self):
        refs,original=fixture()
        def corrupt_source(v):v[refs["inferred"]["path"]]["source_observation_ref"]=refs["inferred"]
        def corrupt_genotype(v):v[refs["inferred"]["path"]]["genotypes_ref"]=refs["positions"]
        def corrupt_sample(v):v[refs["inferred_roster"]["path"]]["samples"][600]["sample_id"]="wrong_query"
        def corrupt_label(v):v[refs["inferred_roster"]["path"]]["samples"][600]["observed_reference_class"]=0
        def corrupt_recipe(v):v[refs["inference"]["path"]]["recipe_ref"]=refs["campaign"]
        def corrupt_chain(v):v[refs["inference"]["path"]]["input_ref"]=refs["preparation"]
        for corrupt in (corrupt_source,corrupt_genotype,corrupt_sample,corrupt_label,corrupt_recipe,corrupt_chain):
            values=copy.deepcopy(original);corrupt(values)
            with self.subTest(corrupt=corrupt.__name__), \
                    mock.patch.object(candidate,"_read_metadata",side_effect=lambda ref:values[ref["path"]]):
                with self.assertRaises(candidate.c.ContractError):
                    candidate.observed_lineage(refs["preparation"],refs["generation"])

    def test_factory_reuses_frozen_v4_entrypoints_and_does_not_create_output(self):
        refs,values=fixture()
        out="experiments/results/NEVER_CREATED_synthetic_v5_lineage/native"
        path=candidate.c.contained_path(out);self.assertFalse(path.exists())
        def descriptor(function,spec):
            return dict(module="arggnn.benchmarking.treeloom_preparation_v4",function=function,spec=spec)
        with mock.patch.object(candidate,"_read_metadata",side_effect=lambda ref:values[ref["path"]]), \
                mock.patch.object(candidate,"_descriptor",side_effect=descriptor):
            result=candidate.qualification_factory(preparation_ref=refs["preparation"],
                generation_ref=refs["generation"],output_directory=out,
                max_rss_bytes=128*2**30,max_seconds=7200,max_output_bytes=16*2**20)
        self.assertEqual(result["module"],"arggnn.benchmarking.treeloom_preparation_v4")
        self.assertEqual(result["spec"]["observed_input_refs"]["observation_ref"],refs["inferred"])
        self.assertEqual(result["spec"]["candidate_source_refs"],frozen.candidate_source_refs())
        self.assertFalse(path.exists())
        self.assertIs(candidate.run_qualification,frozen.run_qualification)
        self.assertIs(candidate.run_fit,frozen.run_fit)
        with mock.patch.object(frozen,"fit_factory",return_value=dict(source_refs=[],module="frozen",function="run_fit")) as delegated:
            result=candidate.fit_factory(preparation_refs={},generation_refs={},output_directory=out,
                                        max_rss_bytes=512*2**30,qualification_ref=refs["focals"])
        delegated.assert_called_once()
        self.assertIn(candidate.c.file_ref(candidate.__file__),result["source_refs"])


    def test_existing_supervisor_root_supplies_fresh_child_to_frozen_qualifier(self):
        from research.reviews.workflow_progress_diagnosis_20261018.treeloom import qualify_parent_v2 as qualifier
        refs,values=fixture()
        with tempfile.TemporaryDirectory(prefix="test_v5_supervisor_",dir=candidate.c.ROOT/"experiments/results") as temporary:
            root=Path(temporary);(root/"tmp").mkdir();(root/"cache").mkdir()
            (root/"stage.log").write_text("existing supervisor log\n")
            child=root/"qualification";self.assertFalse(child.exists())
            with mock.patch.object(candidate,"_read_metadata",side_effect=lambda ref:values[ref["path"]]), \
                    mock.patch.object(candidate,"_descriptor",side_effect=lambda function,spec:dict(function=function,spec=spec)):
                descriptor=candidate.qualification_factory(preparation_ref=refs["preparation"],
                    generation_ref=refs["generation"],output_directory=str(root),
                    max_rss_bytes=128*2**30,max_seconds=7200,max_output_bytes=16*2**20)
            self.assertTrue(root.exists())
            self.assertEqual(candidate.c.contained_path(descriptor["spec"]["output_directory"]),child)
            self.assertFalse(child.exists(),"Metadata materialization must not create the fresh scientific child")
            science={"status":"synthetic_fresh_child_receipt"}
            def fresh_child_qualify(spec,*,resource_check):
                output=candidate.c.contained_path(spec["output_directory"])
                self.assertEqual(output,child)
                self.assertFalse(output.exists(),"Frozen qualifier's fresh-path precondition")
                output.mkdir()
                (output/"qualification.json").write_text(json.dumps(science)+"\n")
                return science
            with mock.patch.object(qualifier,"qualify",side_effect=fresh_child_qualify), \
                    mock.patch.object(frozen,"_require_pass",return_value=science):
                returned=frozen.run_qualification(descriptor["spec"],resource_check=lambda stage:None)
            self.assertEqual(candidate.c.contained_path(returned["path"]),child/"qualification.json")
            (root/"result.json").write_text(json.dumps(dict(status="succeeded",value=returned))+"\n")
            self.assertEqual(candidate.c.read_ref(returned),science)
            self.assertEqual((root/"stage.log").read_text(),"existing supervisor log\n")
            self.assertTrue((root/"tmp").is_dir() and (root/"cache").is_dir())


if __name__=="__main__":unittest.main()
