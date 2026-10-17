"""Paper-recipe plans and shared observed inputs for paired comparisons.

Planning is pure metadata work. This module neither imports scientific runtimes
nor starts jobs. One simulation and one observation preparation serve every
method. Truth is a separate product accessible only to authorized training and
scoring stages, never an observed-input descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import re
from typing import Mapping

from .contracts import ContractError, file_ref, fingerprint, read_json
from .data import ImmutableRef

AGREEMENTS = {"paper_explicit", "author_code", "paper_unspecified", "conflict", "unresolved"}
REPRESENTATIONS = {"treeloom": "whole_trees", "argformer": "ancestral_paths",
                   "argmix": "native_subgraphs", "ancestralpaths": "native_ancestor_features",
                   "flare": "phased_genotypes"}


def load_recipe(identifier: str) -> dict:
    if not isinstance(identifier, str) or re.fullmatch(r"[a-z][a-z0-9_]*", identifier) is None:
        raise ContractError("Use a registered paper recipe ID")
    recipe = read_json(f"configs/benchmarks/recipes/{identifier}.json")
    if recipe.get("schema") != "paper_recipe_v1" or recipe.get("id") != identifier:
        raise ContractError("Wrong paper recipe identity")
    methods = recipe.get("methods", [])
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(REPRESENTATIONS):
        raise ContractError("Declare each native comparator and TreeLoom once")
    if not recipe.get("evidence"):
        raise ContractError("A paper recipe needs setting-level source evidence")
    for item in recipe["evidence"]:
        if (item.get("agreement") not in AGREEMENTS or type(item.get("material")) is not bool
                or not (item.get("source_ref") or item.get("citation"))):
            raise ContractError("Recipe settings require source evidence and an agreement status")
    return recipe


def paper_alignment(recipe: dict) -> dict:
    """Summarize material disagreements without turning assumptions into facts."""
    issues = [{"parameter": item["parameter"], "agreement": item["agreement"],
               "stages": item.get("stages", ["generation", "inference", "training", "evaluation"]),
               "reason": item["note"]}
              for item in recipe["evidence"]
              if item["material"] and not item.get("discretionary", False)
              and item["agreement"] in {"conflict", "unresolved", "paper_unspecified"}]
    return {"material_issues": issues, "paper_alignment_established": not issues,
            "fresh_seeds_allowed": True,
            "agreement_with_printed_numbers_required": False}


def _seed(recipe_id: str, parent: str, purpose: str, root_seed: int) -> int:
    raw = f"arggnn.paper-comparison|{recipe_id}|{parent}|{purpose}|{root_seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1) + 1


def _validate_paper_design(recipe: dict, design: dict) -> None:
    """Check native endpoints and paper-specified replicate budgets."""
    parameters = recipe.get("parameters", {})
    identifier = recipe["id"]
    if identifier == "argformer_american":
        classes = parameters["ontology"]["class_order"]
        counts = None  # Our supervised design, explicitly separate from the paper.
        endpoint = "three_source_local_ancestry"
    elif identifier == "argmix_european":
        classes = parameters["truth_endpoint"]["class_order"]
        replicates = parameters["replicates"]
        counts = dict(train=replicates["train"], validation=replicates["validation"],
                      test=replicates["iid_test"])
        endpoint = "four_source_local_ancestry_by_cohort"
    elif identifier == "ancestralpaths_european":
        classes = ["Ana", "CHG", "WHG", "EHG", "Ana_via_BAA", "CHG_via_BAA"]
        replicates = parameters["replicates"]
        counts = dict(train=replicates["train"], validation=0, test=replicates["evaluation"])
        endpoint = "native_six_ancestral_paths"
    else:
        return
    if design["class_order"] != classes or design["endpoint"] != endpoint:
        raise ContractError("The paper comparison must retain its declared native endpoint")
    if counts is not None and design["split_counts"] != counts:
        raise ContractError("Paper-specified independent replicate counts changed")


def plan_comparison(recipe: dict, design: dict) -> dict:
    """Return an inert shared-data plan. Splits are independent simulations.

    Scientific training budgets remain in the method recipes. The design records
    which replicate counts come from the paper and which are our study choices.
    No legacy observation or already-exposed TEST is silently reused.
    """
    if design.get("recipe_id") != recipe["id"]:
        raise ContractError("Design and paper recipe must agree")
    seed = design.get("root_seed")
    if type(seed) is not int or not 0 < seed < 2**32:
        raise ContractError("Explicit positive 32-bit study seed required")
    counts = design.get("split_counts", {})
    if set(counts) != {"train", "validation", "test"}:
        raise ContractError("Declare train, validation and untouched test parent counts")
    if any(type(n) is not int or not 0 <= n <= 1000 for n in counts.values()) or counts["test"] < 1:
        raise ContractError("Bounded independent parent counts and a nonempty final TEST required")
    classes = design.get("class_order", [])
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ContractError("Declare the actual prediction endpoint")
    if design.get("test_previously_used") is not False:
        raise ContractError("This paper comparison requires fresh untouched test parents")
    _validate_paper_design(recipe, design)
    methods = tuple(recipe["methods"])
    if (len(methods) < 2 or len(set(methods)) != len(methods)
            or "treeloom" not in methods or not set(methods) <= set(REPRESENTATIONS)):
        raise ContractError("A supported native comparator and TreeLoom are required")
    parents, seeds = [], set()
    recipe_identity = fingerprint(recipe)
    for split in ("train", "validation", "test"):
        for index in range(counts[split]):
            parent = f"{recipe['id']}_{split}_{index:03d}"
            choices = {purpose: _seed(recipe["id"], parent, purpose, seed)
                       for purpose in ("ancestry", "mutations", "observation", "sampling")}
            if seeds.intersection(choices.values()) or len(set(choices.values())) != len(choices):
                raise ContractError("Seed collision: choose another study seed before generating data")
            seeds.update(choices.values())
            simulation_stream = fingerprint({"recipe": recipe_identity, "seeds": choices})
            ancestry_stream = fingerprint({"recipe": recipe_identity, "ancestry_seed": choices["ancestry"]})
            shared = {"recipe_fingerprint": recipe_identity, "parent_id": parent, "split": split,
                      "simulation_stream": simulation_stream, "ancestry_stream": ancestry_stream}
            parents.append({**shared, "seeds": choices,
                "generator": copy.deepcopy(recipe["generator"]),
                "generation_count": 1, "observation_preparation_count": 1,
                "observed_product_id": fingerprint({**shared, "role": "observed"}),
                "truth_product_id": fingerprint({**shared, "role": "truth"}),
                "consumers": [{"method_id": method, "representation": REPRESENTATIONS[method],
                               "uses_same_observed_product": True, "query_truth_in_inputs": False}
                              for method in methods]})
    return {"schema": "paired_paper_comparison_plan_v1", "recipe_id": recipe["id"],
            "paper_url": recipe["paper_url"], "methods": list(methods),
            "design": copy.deepcopy(design), "alignment": paper_alignment(recipe),
            "recipe_fingerprint": recipe_identity, "parents": parents,
            "shared_simulation_count": len(parents),
            "evaluation": {"class_order": classes,
                           "endpoint": design["endpoint"],
                           "support_frozen_from": "observations_before_predictions",
                           "posthoc_prediction_intersection": False,
                           "uncovered_targets_reported": True,
                           "aggregation": "equal_parent_mean_sample_sd"},
            "execution_authorized": False, "jobs_started": 0,
            "scientific_payloads_opened": False,
            "implementation_readiness": "requires_native_recipe_and_runtime_integration"}


def registered_plan(identifier: str) -> dict:
    recipe = load_recipe(identifier)
    design = read_json(f"configs/benchmarks/paper_comparisons/{identifier}.json")
    plan = plan_comparison(recipe, design)
    # Family planners authenticate real paper/source pins and supply the actual
    # generator parameter contract, without importing or invoking the generator.
    if identifier == "argformer_american":
        from .argformer_recipe import plan_simulations
        family = plan_simulations(plan["parents"])
        for parent, generation in zip(plan["parents"], family, strict=True):
            parent["generation_plan"] = generation
    else:
        from .european_recipes import prepare_recipe
        plan["generation_preparation"] = prepare_recipe(identifier)
        for parent in plan["parents"]:
            if identifier == "ancestralpaths_european":
                parent["generator_parent_argument"] = {
                    "parent_id": parent["parent_id"],
                    "split": "evaluation" if parent["split"] == "test" else parent["split"],
                    "seeds": {key: parent["seeds"][key] for key in ("ancestry", "mutations")},
                }
            else:
                # The old ARGMix candidate reads its own fixed config registry.
                # It cannot consume this fresh paper plan's parent/seed schema.
                parent["generator_parent_argument"] = None
                parent["generator_integration_pending"] = (
                    "ARGMix paper configuration materializer requires the missing "
                    "age/map/error/sampling settings. The historical fixed registry "
                    "is not compatible with these fresh parent IDs and seeds.")
    plan["source_pins_verified"] = True
    plan["recipe_ref"] = file_ref(f"configs/benchmarks/recipes/{identifier}.json")
    plan["design_ref"] = file_ref(f"configs/benchmarks/paper_comparisons/{identifier}.json")
    return plan


def _observed_reference(value) -> ImmutableRef:
    ref = ImmutableRef.from_value(value)
    if any(part in {"truth", "query_truth", "targets", "ancestry_truth"}
           for part in ref.path.lower().split("/")):
        raise ContractError("Observed inputs cannot contain query ancestry truth")
    return ref


@dataclass(frozen=True)
class SharedObservedParent:
    """Small references to one complete observed parent shared by all methods.

    The upstream producer and pinned decoder authenticate file contents. This
    descriptor binds simulation streams, the generation receipt, exposure
    history and the roster containing parent-qualified haplotype/diploid IDs.
    The future guarded decoder must validate those receipts and their contents.
    This descriptor authenticates declared identity and does not mistake a filename for proof
    that its contents are sanitized. Reference labels may include admixed cohort
    labels (NF/Yam), never their hidden local ancestry assignments.
    """
    parent_id: str
    split: str
    recipe_ref: ImmutableRef
    genotypes_ref: ImmutableRef
    roster_ref: ImmutableRef
    reference_query_ref: ImmutableRef
    evaluation_support_ref: ImmutableRef
    support_fingerprint: str
    simulation_stream: str
    ancestry_stream: str
    generation_receipt_ref: ImmutableRef
    exposure_history_ref: ImmutableRef
    genealogy_ref: ImmutableRef | None = None
    genetic_map_ref: ImmutableRef | None = None

    def __post_init__(self):
        if (not isinstance(self.parent_id, str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", self.parent_id) is None
                or self.split not in {"train", "validation", "test"}
                or re.fullmatch(r"[0-9a-f]{64}", self.support_fingerprint or "") is None):
            raise ContractError("Explicit parent, split and frozen target identity required")
        for name in ("simulation_stream", "ancestry_stream"):
            if not isinstance(getattr(self, name), str) or re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ContractError("Bind the actual independent ancestry and complete simulation streams")
        for name in ("recipe_ref", "generation_receipt_ref", "exposure_history_ref"):
            object.__setattr__(self, name, ImmutableRef.from_value(getattr(self, name)))
        for name in ("genotypes_ref", "roster_ref", "reference_query_ref",
                     "evaluation_support_ref", "genealogy_ref", "genetic_map_ref"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _observed_reference(value))
            elif name not in {"genealogy_ref", "genetic_map_ref"}:
                raise ContractError("Every method requires the complete shared observations")

    def as_dict(self):
        values = {"parent_id": self.parent_id, "split": self.split,
                  "support_fingerprint": self.support_fingerprint,
                  "simulation_stream": self.simulation_stream, "ancestry_stream": self.ancestry_stream}
        for name in ("recipe_ref", "generation_receipt_ref", "exposure_history_ref",
                     "genotypes_ref", "roster_ref", "reference_query_ref",
                     "evaluation_support_ref", "genealogy_ref", "genetic_map_ref"):
            value = getattr(self, name)
            values[name] = None if value is None else value.as_dict()
        return values

    @property
    def identity(self):
        return fingerprint(self.as_dict())

    def method_input(self, method_id: str) -> dict:
        if method_id not in REPRESENTATIONS:
            raise ContractError("Unknown native representation")
        if method_id == "flare" and self.genetic_map_ref is None:
            raise ContractError("Native FLARE requires the common genetic map")
        if method_id != "flare" and self.genealogy_ref is None:
            raise ContractError("Infer the common genealogy before graph/path preparation")
        # Generation/exposure sidecars are for the trusted coordinator only.
        # Generation receipts may list truth products, so numerical method inputs
        # receive neither those sidecars nor their handles.
        observed = {key: value for key, value in self.as_dict().items()
                    if key not in {"generation_receipt_ref", "exposure_history_ref"}}
        return {"method_id": method_id, "representation": REPRESENTATIONS[method_id],
                "shared_parent_identity": self.identity, "shared_observations": observed,
                "query_truth_in_inputs": False}


def validate_paired_inputs(parent: SharedObservedParent, inputs: Mapping[str, dict], *, methods):
    """Reject separately simulated data, swapped panels, split leaks or target drops."""
    if set(inputs) != set(methods) or len(set(methods)) != len(methods):
        raise ContractError("All declared methods must consume this parent")
    for method_id, value in inputs.items():
        if value != parent.method_input(method_id):
            raise ContractError("Methods must use the same observed parent, panel and frozen support")


def validate_independent_parents(parents) -> None:
    """Reject renamed reuse of one realized genealogy across independent parents."""
    names, streams = set(), set()
    for parent in parents:
        if parent.parent_id in names or parent.ancestry_stream in streams:
            raise ContractError("Independent simulation parents cannot reuse an ancestry random stream")
        names.add(parent.parent_id)
        streams.add(parent.ancestry_stream)


def bind_prepared_parent(plan: dict, parent: SharedObservedParent) -> dict:
    """Join a prepared observed descriptor to its exact declared paper plan."""
    matching = [item for item in plan["parents"] if item["parent_id"] == parent.parent_id]
    if (len(matching) != 1 or matching[0]["split"] != parent.split
            or parent.recipe_ref.as_dict() != plan.get("recipe_ref")
            or matching[0]["simulation_stream"] != parent.simulation_stream
            or matching[0]["ancestry_stream"] != parent.ancestry_stream):
        raise ContractError("Prepared parent does not belong to this recipe and independent split")
    inputs = {method: parent.method_input(method) for method in plan["methods"]}
    validate_paired_inputs(parent, inputs, methods=plan["methods"])
    return inputs
