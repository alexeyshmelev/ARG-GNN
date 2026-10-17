import copy
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from arggnn.benchmarking.contracts import ContractError, fingerprint
from arggnn.benchmarking.paired import (SharedObservedParent, plan_comparison,
    paper_alignment, bind_prepared_parent, validate_paired_inputs, validate_independent_parents, registered_plan, load_recipe)
from arggnn.benchmarking import treeloom


def ref(name):
    return {"path": "fixtures/observed/" + name + ".json", "sha256": fingerprint(name)}


def recipe():
    return {"schema": "paper_recipe_v1", "id": "fixture", "paper_url": "https://example.org/paper",
            "methods": ["treeloom", "flare"], "generator": {"module": "fake", "function": "never_call"},
            "evidence": [{"parameter": "length", "value": 100, "agreement": "paper_explicit",
                          "material": True, "citation": "paper methods", "note": "fixture"}]}


def design():
    return {"recipe_id": "fixture", "root_seed": 101, "split_counts": {"train": 2, "validation": 1, "test": 2},
            "class_order": ["A", "B"], "endpoint": "ancestry", "test_previously_used": False}


def parent():
    stream = plan_comparison(recipe(), design())["parents"][0]
    return SharedObservedParent("fixture_train_000", "train", ref("recipe"), ref("genotypes"),
        ref("roster"), ref("panel"), ref("support"), fingerprint("targets"), stream["simulation_stream"], stream["ancestry_stream"],
        ref("generation"), ref("exposure_history"), ref("inferred_tree"), ref("map"))


class PairedPlanTests(unittest.TestCase):
    def test_generate_once_for_all_methods_with_disjoint_parent_seeds(self):
        value = plan_comparison(recipe(), design())
        self.assertEqual(value["shared_simulation_count"], 5)
        self.assertEqual(sum(p["generation_count"] for p in value["parents"]), 5)
        seeds = [seed for p in value["parents"] for seed in p["seeds"].values()]
        self.assertEqual(len(seeds), len(set(seeds)))
        for p in value["parents"]:
            self.assertEqual({c["method_id"] for c in p["consumers"]}, {"treeloom", "flare"})
            self.assertNotEqual(p["observed_product_id"], p["truth_product_id"])
        self.assertFalse(value["execution_authorized"])
        self.assertFalse(value["scientific_payloads_opened"])

    def test_new_seed_is_reproducible_without_requiring_original_seed(self):
        self.assertEqual(plan_comparison(recipe(), design()), plan_comparison(recipe(), design()))
        d = design(); d["root_seed"] += 1
        self.assertNotEqual(plan_comparison(recipe(), d)["parents"][0]["seeds"],
                            plan_comparison(recipe(), design())["parents"][0]["seeds"])

    def test_material_conflict_stays_visible(self):
        r = recipe()
        r["evidence"].append({"parameter": "ploidy", "value": None, "agreement": "unresolved",
                              "material": True, "citation": "paper", "note": "unknown"})
        self.assertFalse(paper_alignment(r)["paper_alignment_established"])
        self.assertEqual(paper_alignment(r)["material_issues"][0]["parameter"], "ploidy")

    def test_reused_test_and_bad_counts_rejected(self):
        for change in ({"test_previously_used": True}, {"split_counts": {"train": True, "validation": 0, "test": 1}}):
            d = design(); d.update(change)
            with self.assertRaises(ContractError): plan_comparison(recipe(), d)

    def test_method_inputs_share_exact_observations_without_truth(self):
        p = parent()
        inputs = {m: p.method_input(m) for m in ("treeloom", "flare")}
        validate_paired_inputs(p, inputs, methods=("treeloom", "flare"))
        for value in inputs.values():
            self.assertFalse(value["query_truth_in_inputs"])
            self.assertNotIn("truth_ref", value["shared_observations"])
            self.assertNotIn("generation_receipt_ref", value["shared_observations"])
            self.assertNotIn("exposure_history_ref", value["shared_observations"])
        inputs["flare"]["shared_observations"]["genotypes_ref"] = ref("other_genotypes")
        with self.assertRaises(ContractError):
            validate_paired_inputs(p, inputs, methods=("treeloom", "flare"))

    def test_missing_method_extra_truth_and_changed_support_rejected(self):
        p = parent()
        valid = {m: p.method_input(m) for m in ("treeloom", "flare")}
        extra_truth = copy.deepcopy(valid)
        extra_truth["flare"]["truth_ref"] = ref("forbidden")
        for bad in ({"treeloom": valid["treeloom"]}, extra_truth):
            with self.assertRaises(ContractError):
                validate_paired_inputs(p, bad, methods=("treeloom", "flare"))
        changed = dataclasses.replace(p, support_fingerprint=fingerprint("truncated"))
        with self.assertRaises(ContractError):
            validate_paired_inputs(p, {"treeloom": p.method_input("treeloom"), "flare": changed.method_input("flare")},
                                   methods=("treeloom", "flare"))

    def test_graphs_need_inferred_genealogy(self):
        p = dataclasses.replace(parent(), genealogy_ref=None)
        self.assertEqual(p.method_input("flare")["representation"], "phased_genotypes")
        with self.assertRaises(ContractError): p.method_input("treeloom")

    def test_parent_recipe_and_split_are_bound_to_plan(self):
        p = parent(); plan = plan_comparison(recipe(), design())
        plan["recipe_ref"] = p.recipe_ref.as_dict()
        self.assertEqual(set(bind_prepared_parent(plan, p)), {"treeloom", "flare"})
        for bad in (dataclasses.replace(p, split="test"), dataclasses.replace(p, recipe_ref=ref("different_recipe"))):
            with self.assertRaises(ContractError): bind_prepared_parent(plan, bad)

    def test_native_six_path_model_and_protocol_split_adapter(self):
        self.assertEqual(treeloom.EUROPEAN_CONDITIONS["ancestralpaths_paper200mb_native6"],
                         "AP_paper_scale_200Mb_native6")
        implementation = Mock()
        implementation.memory_core._stamps.return_value = ()
        with patch.object(treeloom, "_european_module", return_value=implementation):
            for split, native in (("validation", "val_prediction"), ("test", "evaluation_prediction")):
                treeloom.make_european_bundle("bank", "episodes", "cut",
                    protocol_id="ancestralpaths_paper200mb_native6", parent_id="parent", split=split, source_refs={})
                args = implementation.FrozenEuropeanBundleV1.call_args.args
                self.assertEqual(args[3], "AP_paper_scale_200Mb_native6")
                self.assertEqual(args[5], native)

    def test_seed_changes_product_identity_and_binding(self):
        original = plan_comparison(recipe(), design())
        new_design = design()
        new_design["root_seed"] += 1
        changed = plan_comparison(recipe(), new_design)
        for name in ("observed_product_id", "truth_product_id", "simulation_stream", "ancestry_stream"):
            self.assertNotEqual(original["parents"][0][name], changed["parents"][0][name])
        changed["recipe_ref"] = parent().recipe_ref.as_dict()
        with self.assertRaises(ContractError):
            bind_prepared_parent(changed, parent())

    def test_renaming_an_ancestry_realization_does_not_make_it_independent(self):
        p = parent()
        with self.assertRaises(ContractError):
            validate_independent_parents([p, dataclasses.replace(p, parent_id="other", split="test")])

    def test_native_comparator_and_flare_map_required(self):
        r = recipe()
        r["methods"] = ["treeloom"]
        with self.assertRaises(ContractError):
            plan_comparison(r, design())
        with self.assertRaises(ContractError):
            dataclasses.replace(parent(), genetic_map_ref=None).method_input("flare")

    def test_registered_plans_verify_sources_and_join_actual_family_preparation(self):
        for identifier, expected in (("argformer_american", 35),
                                     ("argmix_european", 9),
                                     ("ancestralpaths_european", 8)):
            plan = registered_plan(identifier)
            self.assertEqual(plan["shared_simulation_count"], expected)
            self.assertTrue(plan["source_pins_verified"])
            self.assertFalse(plan["execution_authorized"])
            if identifier == "argformer_american":
                self.assertEqual(plan["parents"][0]["generation_plan"]["generator_scientific_arguments"]["ancestry_seed"],
                                 plan["parents"][0]["seeds"]["ancestry"])
                self.assertFalse(plan["parents"][0]["generation_plan"]["actual_simulator_invoked"])
            else:
                self.assertEqual(plan["generation_preparation"]["recipe_id"], identifier)
                if identifier == "ancestralpaths_european":
                    self.assertEqual(plan["parents"][-1]["generator_parent_argument"]["split"], "evaluation")
                else:
                    self.assertIsNone(plan["parents"][-1]["generator_parent_argument"])
                    self.assertIn("fixed registry", plan["parents"][-1]["generator_integration_pending"])

    def test_native_endpoint_and_paper_counts_cannot_be_silently_replaced(self):
        from arggnn.benchmarking.contracts import read_json
        for identifier in ("ancestralpaths_european", "argmix_european"):
            r = load_recipe(identifier)
            d = read_json(f"configs/benchmarks/paper_comparisons/{identifier}.json")
            altered = copy.deepcopy(d)
            altered["class_order"] = ["A", "B"]
            with self.assertRaises(ContractError):
                plan_comparison(r, altered)
            altered = copy.deepcopy(d)
            altered["split_counts"]["train"] += 1
            with self.assertRaises(ContractError):
                plan_comparison(r, altered)
