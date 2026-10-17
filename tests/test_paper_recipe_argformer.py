"""Paper/source/plan checks only; no scientific imports or simulator calls."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from arggnn.benchmarking import argformer_recipe as recipe


def plan(**kwargs):
    settings = dict(parent_id="argformer_paper_train_000", split="train",
                    ancestry_seed=180001, mutation_seed=1180001)
    settings.update(kwargs)
    return recipe.plan_simulation(**settings)


class AmericanPaperRecipeTests(unittest.TestCase):
    def test_known_paper_settings_are_distinct_from_unresolved_units(self):
        value = recipe.load_recipe()
        simulation = value["parameters"]["simulation"]
        self.assertEqual(simulation["demographic_model"], "AmericanAdmixture_4B18")
        self.assertEqual(simulation["sequence_length_bp"], 30000000)
        self.assertEqual(simulation["mutation_rate_per_bp_generation"], 1.25e-8)
        self.assertEqual(simulation["recombination_rate_per_bp_generation"], 1e-8)
        self.assertEqual(sum(simulation["sample_counts_as_written"].values()), 1600)
        self.assertIsNone(simulation["sample_unit"])
        self.assertIsNone(simulation["mutation_model"])
        self.assertFalse(value["paper_matched_recipe_ready"])

    def test_every_recorded_setting_has_evidence_and_agreement(self):
        value = recipe.load_recipe()
        recipe.verify_sources(value)
        parameters = [item["parameter"] for item in value["evidence"]]
        self.assertEqual(len(parameters), len(set(parameters)))
        for item in value["evidence"]:
            self.assertIn(item["agreement"], recipe.AGREEMENT)
            self.assertIsInstance(item["material"], bool)
            self.assertTrue("source_ref" in item or "citation" in item)
        for name in ("simulation.sample_unit", "truth.ancestry_definition",
                     "evaluation.path_coordinate_and_truth_rule",
                     "native.contrastive.temperature", "native.contrastive.class_weights"):
            self.assertIn(name, parameters)

    def test_generator_is_real_but_remains_a_reconstruction_candidate(self):
        value = recipe.load_recipe()
        generator = value["generator"]
        self.assertEqual(generator["module"], "arggnn.american_c3_simulate_v1")
        tree = ast.parse((recipe.ROOT / generator["source_ref"]["path"]).read_text())
        self.assertIn(generator["function"], {n.name for n in tree.body if isinstance(n, ast.FunctionDef)})
        defaults = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DEFAULTS" for t in n.targets))
        keywords = {k.arg: k.value for k in defaults.keywords}
        self.assertEqual(ast.literal_eval(keywords["mutation_rate"]), 1.25e-8)
        self.assertEqual(ast.literal_eval(keywords["recombination_rate"]), 1e-8)
        self.assertIn("reconstruction", ast.literal_eval(keywords["schema"]))
        self.assertEqual(generator["fixed_reconstruction_choices"]["mutation_model"], "JC69")
        self.assertTrue(generator["output_requires_inference"])

    def test_unresolved_sample_unit_produces_no_implicit_haplotype_count(self):
        value = plan()
        self.assertIsNone(value["sample_unit_decision"]["selected"])
        self.assertIsNone(value["sample_unit_decision"]["resolved_haplotypes"])
        self.assertFalse(value["execution_authorized"])
        self.assertFalse(value["ready_for_execution"])
        self.assertFalse(value["actual_simulator_invoked"])

    def test_explicit_haploid_and_diploid_choices_do_not_become_paper_facts(self):
        for unit, total in (("haploid", 1600), ("diploid", 3200)):
            value = plan(sample_unit=unit)
            decision = value["sample_unit_decision"]
            self.assertEqual(sum(decision["resolved_haplotypes"].values()), total)
            self.assertFalse(decision["confirmed_by_paper"])
            self.assertFalse(value["paper_matched_recipe_ready"])
            self.assertFalse(value["legacy_reconstruction_relabelled"])

    def test_material_conflicts_block_but_new_seeds_do_not(self):
        blockers = {item["parameter"] for item in recipe.material_blockers()}
        self.assertIn("native.MLM.corruption_rule", blockers)
        self.assertIn("native.contrastive.temperature", blockers)
        self.assertIn("native.contrastive.class_weights", blockers)
        self.assertIn("truth.ancestry_definition", blockers)
        self.assertIn("evaluation.retrieval_k", blockers)
        self.assertNotIn("simulation.new_random_seeds", blockers)
        self.assertNotIn("inference.runtime_versions_and_threads", blockers)
        self.assertTrue(plan()["new_seed_choice"]["allowed_discretionary_choice"])

    def test_paired_plan_generates_once_and_keeps_truth_external(self):
        paired = plan()["paired_data_contract"]
        self.assertEqual(paired["methods"], ["argformer", "flare", "treeloom"])
        self.assertTrue(paired["generate_once_per_parent"])
        self.assertTrue(paired["shared_observed_genotypes_roster_and_inferred_genealogy"])
        self.assertTrue(paired["known_ancestry_truth_kept_separate"])
        self.assertFalse(paired["query_truth_in_observed_inputs"])
        self.assertTrue(paired["inferred_genealogy_required_before_predictors"])
        self.assertTrue(paired["same_prespecified_query_coordinate_support_for_all_methods"])

    def test_flare_is_included_without_inventing_paper_options(self):
        value = recipe.load_recipe()
        self.assertIn("flare", value["methods"])
        self.assertIsNone(value["parameters"]["native_FLARE"]["paper_version"])
        self.assertEqual(value["parameters"]["native_FLARE"]["current_release_options"]["min-mac"], 50)
        self.assertFalse(value["parameters"]["native_FLARE"]["private_truth_inputs"])

    def test_temperature_overlay_is_explicit_inert_and_does_not_resolve_other_conflicts(self):
        value = recipe.plan_native_config_overlay()
        self.assertEqual(value["structured_overlay"]["model"]["contrastive"]["temperature"], .05)
        self.assertFalse(value["selected_for_execution"])
        self.assertFalse(value["author_yaml_modified"])
        blockers = {x["parameter"] for x in value["remaining_material_blockers"]}
        self.assertIn("native.MLM.corruption_rule", blockers)
        self.assertIn("native.contrastive.class_weights", blockers)
        self.assertNotIn("native.contrastive.temperature", blockers)
        wiring = recipe.plan_native_config_overlay("paper_temperature_with_explicit_weight_loader_wiring")
        self.assertEqual(wiring["structured_overlay"]["model"]["train_loader"], chr(36)+"{train_loader}")
        self.assertIn("native.contrastive.class_weights",
                      {x["parameter"] for x in wiring["remaining_material_blockers"]})

    def test_bad_seeds_and_implicit_ploidy_aliases_reject(self):
        for changes in ({"ancestry_seed": 0}, {"ancestry_seed": True},
                        {"mutation_seed": 180001}, {"sample_unit": "samples"},
                        {"split": "random_windows"}):
            with self.assertRaises(ValueError):
                plan(**changes)

    def test_source_drift_blocks_the_inert_plan(self):
        with patch.object(Path, "read_bytes", return_value=b"changed source"):
            with self.assertRaisesRegex(ValueError, "source changed"):
                plan()

    def test_fresh_process_planning_imports_no_scientific_runtime(self):
        code = (
            "import sys; sys.path.insert(0, " + repr(str(recipe.ROOT / "src")) + "); "
            "from arggnn.benchmarking.argformer_recipe import plan_simulation; "
            "p=plan_simulation(parent_id='paper_plan_test',split='test',"
            "ancestry_seed=12345,mutation_seed=67890); "
            "assert p['actual_simulator_invoked'] is False; "
            "assert not any(n in sys.modules for n in "
            "('msprime','stdpopsim','tskit','tsinfer','tsdate','numpy','torch','tensorflow')); "
            "assert 'arggnn.american_c3_simulate_v1' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-B", "-s", "-c", code],
                                cwd=recipe.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
