"""Inert American ARGformer paper-recipe planning; standard library only.

No simulator, scientific runtime, dataset reader, launcher or admission API is
imported. The existing simulator remains explicitly a reconstruction candidate.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = ROOT / "configs/benchmarks/recipes/argformer_american.json"
AGREEMENT = {"paper_explicit", "author_code", "paper_unspecified", "conflict", "unresolved"}


def load_recipe() -> dict:
    recipe = json.loads(RECIPE_PATH.read_text())
    if recipe.get("schema") != "paper_recipe_v1" or recipe.get("id") != "argformer_american":
        raise ValueError("Expected the distinct American ARGformer paper recipe")
    for item in recipe["evidence"]:
        if (item.get("agreement") not in AGREEMENT or type(item.get("material")) is not bool
                or not ("source_ref" in item or "citation" in item)):
            raise ValueError("Every setting needs evidence and an explicit agreement status")
    return recipe


def verify_sources(recipe: dict | None = None) -> None:
    """Authenticate only compact local paper/source/config files."""
    recipe = load_recipe() if recipe is None else recipe
    for record in recipe["source_refs"]:
        relative = Path(record["path"])
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or path.resolve() != path
                or not path.is_file() or path.stat().st_size > 32 * 2**20
                or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]):
            raise ValueError("Paper-recipe source changed: " + record["path"])


def material_blockers(recipe: dict | None = None) -> list[dict]:
    recipe = load_recipe() if recipe is None else recipe
    return [
        {"parameter": item["parameter"], "agreement": item["agreement"], "note": item["note"]}
        for item in recipe["evidence"]
        if item["material"] and item["agreement"] in {"conflict", "unresolved", "paper_unspecified"}
        and not item.get("discretionary", False)
    ]


def plan_simulation(*, parent_id: str, split: str, ancestry_seed: int,
                    mutation_seed: int, sample_unit: str | None = None) -> dict:
    """Return a blocked plan bound to real existing generator source.

    Fresh seeds are a permitted new experimental choice, not missing paper
    evidence. A caller-selected ploidy convention is recorded as a decision,
    never promoted to a confirmed paper fact.
    """
    recipe = load_recipe()
    verify_sources(recipe)
    return _plan_simulation(recipe, parent_id=parent_id, split=split,
                            ancestry_seed=ancestry_seed, mutation_seed=mutation_seed,
                            sample_unit=sample_unit)


def plan_simulations(parents: list[dict]) -> list[dict]:
    """Plan a complete independent-parent study, verifying shared sources once."""
    recipe = load_recipe()
    verify_sources(recipe)
    return [_plan_simulation(recipe, parent_id=parent["parent_id"], split=parent["split"],
                             ancestry_seed=parent["seeds"]["ancestry"],
                             mutation_seed=parent["seeds"]["mutations"])
            for parent in parents]


def _plan_simulation(recipe: dict, *, parent_id: str, split: str, ancestry_seed: int,
                     mutation_seed: int, sample_unit: str | None = None) -> dict:
    if not isinstance(parent_id, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", parent_id) is None:
        raise ValueError("Explicit stable new parent ID required")
    if split not in {"train", "validation", "test"}:
        raise ValueError("Independent parent split must precede paths/windows")
    if (type(ancestry_seed) is not int or type(mutation_seed) is not int
            or not 0 < ancestry_seed < 2**32 or not 0 < mutation_seed < 2**32
            or ancestry_seed == mutation_seed):
        raise ValueError("Two distinct positive32-bit ancestry/mutation seeds required")
    if sample_unit not in {None, "haploid", "diploid"}:
        raise ValueError("Unresolved sample unit, or an explicit documented convention, required")
    parameters = copy.deepcopy(recipe["parameters"])
    counts = parameters["simulation"]["sample_counts_as_written"]
    haplotypes = None if sample_unit is None else {
        population: count * (1 if sample_unit == "haploid" else 2)
        for population, count in counts.items()
    }
    arguments = {
        "schema": "american_c3_simulation_reconstruction_v1",
        "execution_enabled": False, "species": "HomSap", "model": "AmericanAdmixture_4B18",
        "sequence_length_bp": 30000000, "mutation_rate": 1.25e-8, "recombination_rate": 1e-8,
        "admixture_generation": 12.0, "class_order": ["EUR", "EAS", "AFR"],
        "sample_counts": counts, "sample_unit": sample_unit,
        "parent_id": parent_id, "split": "val" if split == "validation" else split,
        "ancestry_seed": ancestry_seed, "mutation_seed": mutation_seed,
    }
    return {
        "schema": "argformer_american_inert_simulation_plan_v1",
        "recipe_id": recipe["id"], "generator": copy.deepcopy(recipe["generator"]),
        "generator_scientific_arguments": arguments,
        "actual_simulator_invoked": False, "scientific_runtime_imported": False,
        "execution_authorized": False, "ready_for_execution": False,
        "paper_matched_recipe_ready": False, "legacy_reconstruction_relabelled": False,
        "new_seed_choice": {"ancestry_seed": ancestry_seed, "mutation_seed": mutation_seed,
                            "allowed_discretionary_choice": True},
        "sample_unit_decision": {"selected": sample_unit, "confirmed_by_paper": False,
                                 "resolved_haplotypes": haplotypes},
        "existing_generator_fixed_choices": copy.deepcopy(recipe["generator"]["fixed_reconstruction_choices"]),
        "blocking_material_evidence": material_blockers(recipe),
        "unresolved_details": copy.deepcopy(recipe["unresolved_details"]),
        "paired_data_contract": {
            "generate_once_per_parent": True,
            "shared_observed_genotypes_roster_and_inferred_genealogy": True,
            "methods": copy.deepcopy(recipe["methods"]),
            "known_ancestry_truth_kept_separate": True,
            "query_truth_in_observed_inputs": False,
            "inferred_genealogy_required_before_predictors": True,
            "same_prespecified_query_coordinate_support_for_all_methods": True,
        },
        "requires_later_explicit_execution_approval": True,
        "actual_resource_limits_and_guarded_writer": "not supplied by this source-only scientific plan",
    }


def plan_native_config_overlay(variant: str = "paper_temperature_only") -> dict:
    """Describe an unselected data-only overlay; never edit native YAML or code."""
    recipe = load_recipe()
    verify_sources(recipe)
    candidates = recipe["native_config_overlay_candidates"]
    if variant not in candidates:
        raise ValueError("Unknown explicitly audited native overlay candidate")
    candidate = copy.deepcopy(candidates[variant])
    return {
        "schema": "argformer_american_native_overlay_plan_v1",
        "recipe_id": recipe["id"], "variant": variant, **candidate,
        "selected_for_execution": False, "ready_for_execution": False,
        "execution_authorized": False, "author_source_modified": False,
        "author_yaml_modified": False, "configuration_materialized": False,
        "remaining_material_blockers": [
            item for item in material_blockers(recipe)
            if item["parameter"] not in candidate["addresses_parameters"]
        ],
        "requires_explicit_variant_registration_and_qualification": True,
    }
