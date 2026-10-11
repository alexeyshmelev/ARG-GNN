"""Root-only expansion of the released full952 author simulation/Relate recipe.

Default CLI is source-only. One admitted parent per execution. No model, GPU,
outer evaluation or changes to the frozen producer functions are introduced.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import sys
import hashlib
import json
import os

from . import prepare_ancestralpaths_full_panel_discovery_v1 as original

ROOT = original.ROOT
SOURCE = "src/arggnn/prepare_expanded_full_panel_v1.py"
KIND = "expanded_full_panel_v1"
SPLITS = (("train", 32), ("val", 8), ("outer", 8))
PURPOSES = ("ancestry", "mutations", "relate_initial", "relate_size_base")
PRIOR_METADATA = (
    "configs/biological_paths_execution_pilot_v1.json",
    "configs/biological_paths_relate_v1.json",
    "configs/ancestralpaths_full_panel_discovery_v1.json",
    "data/ancient_development64_v1/manifest.json",
    "data/ancient_inferred_development64_v1/manifest.json",
    "data/ancient_common_input_pilot_v2/manifest.json",
)


def _prior_streams_v1():
    """Metadata-only prior registered seed exclusion, no target/array reads."""
    values, refs = set(), []
    for relative in PRIOR_METADATA:
        path = original.confined(relative)
        metadata = json.loads(path.read_text())
        refs.append(original.ref(relative))
        for row in metadata.get("parent_registry", []) + metadata.get("replicates", []):
            values.update(int(value) for value in row.get("seeds", {}).values())
            values.update(row.get("relate_size_iteration_seeds", []))
            if "inference_seed" in row:
                values.add(int(row["inference_seed"]))
        for row in metadata.get("parent_seeds", {}).values():
            values.add(int(row["initial"]))
            values.update(int(row["size_base"]) + offset for offset in range(6))
    return values, refs


def parent_registry_v1():
    rows = []
    for split, count in SPLITS:
        for index in range(count):
            seeds, domains = {}, {}
            for purpose in PURPOSES:
                domain = f"arggnn|expanded_full_panel_v1|root=101|split={split}|parent={index:03d}|purpose={purpose}"
                digest = hashlib.sha256(domain.encode("ascii")).digest()
                modulus = 2**31 - 100 if purpose.startswith("relate") else 2**32 - 1
                seeds[purpose] = int.from_bytes(digest[:8], "big") % modulus + 1
                domains[purpose] = digest.hex()
            rows.append(dict(parent_id=f"ap_expanded_{split}_{index:03d}", split=split,
                downstream_role={"train": "TRAIN", "val": "INTERNAL_VAL", "outer": "UNTOUCHED_OUTER"}[split],
                replicate=index, seeds=seeds, seed_domain_sha256=domains,
                relate_size_iteration_seeds=[seeds["relate_size_base"] + offset for offset in range(1, 6)],
                relate_final_reestimate_seed=seeds["relate_size_base"] + 5,
                evaluation_authorized=False if split == "outer" else None))
    values = [value for row in rows for value in list(row["seeds"].values()) + row["relate_size_iteration_seeds"]]
    prior, _ = _prior_streams_v1()
    if len(set(values)) != len(values) or set(values) & prior:
        raise ValueError("Frozen new stream collision; no automatic reseeding or replacement")
    return rows


def build_config_v1():
    """Original biology, age roster and native recipe; new registry/splits only."""
    config = deepcopy(original.default_config())
    prior, metadata_refs = _prior_streams_v1()
    registry = parent_registry_v1()
    config.update(kind=KIND, status="source_only_registry_frozen_no_generation", output_root="data/expanded_full_panel_v1",
        default_parent_id="ap_expanded_train_000", splits=dict(SPLITS), parent_registry=registry,
        seed_scheme="SHA256 ASCII arggnn|expanded_full_panel_v1|root=101|split={train,val,outer}|parent={000..}|purpose={purpose};first8 big-endian modulo original simulation/Relate ranges;sizebase+1..5;finalreestimate reuses+5",
        expansion=dict(source_adapter=original.ref(SOURCE), reused_producer=original.ref(original.SOURCE),
            original_recipe_config=original.ref(original.CONFIG), prior_seed_metadata=metadata_refs,
            parent_registry_sha256=original.digest(registry), previously_registered_unique_seed_values=len(prior),
            new_unique_seed_values=48*9, stream_collision_count=0,
            generated_parent_count=0, existing_parents_reused=False,
            changed_conditions=["independent_parent_registry", "split_counts", "output_root"],
            unchanged_conditions=["476diploid_author_roster", "all_original_sampling_ages", "author_demography", "2Mb_JC69_rates", "complete686query_C4_targets", "952observed_haplotypes", "Relate1.2.4_initial_and5size_iterations"],
            target_policy="TRAINloss;INTERNAL_VALselection_afterpredictionseal;8outer_reserved_no_evaluation_or_tuning",
            inherited_native_sampler_metadata="retained_original3sequence_stride_receipt;H038wholeforestdoesnotusefocalsampler",
            full_panel_model_preparation_included=False))
    return config


def validate_config_v1(config):
    if config != build_config_v1():
        raise ValueError("Frozen expanded registry, source/runtime or original author recipe differs")
    return dict(kind=KIND, split_counts=dict(SPLITS), parents=48, haplotypes=952,
        query_haplotypes=686, pure_reference_haplotypes=266, mixed_reference_haplotypes=0,
        present_query_haplotypes=182, pure_reference_class_counts=[50,28,102,86],
        new_unique_seed_values=432, stream_collisions=0, simulations_executed=False,
        Relate_executed=False, outer_truth_read=False, model_prepared=False)


def _admission_v1(config, parent, stage, admission):
    expected_output = config["output_root"] + "/" + parent["parent_id"]
    if (admission.get("schema") != "expanded_full_panel_v1_root_admission"
            or admission.get("root_authorizer") != "root_agent"
            or admission.get("execution_authorized") is not True or admission.get("status") != "active"
            or admission.get("parent_id") != parent["parent_id"] or stage not in admission.get("allowed_stages", [])
            or admission.get("output_directory") != expected_output
            or admission.get("source_ref") != original.ref(SOURCE)
            or admission.get("config_content_sha256") != original.digest(config)
            or admission.get("parent_registry_sha256") != original.digest(config["parent_registry"])
            or admission.get("outer_evaluation_authorized") is not False
            or original.ref(admission["runtime_ref"]["path"]) != admission["runtime_ref"]):
        raise ValueError("Exact active root source/config/registry/runtime/parent/output admission required")
    expiration = datetime.fromisoformat(admission["expires_at_utc"].replace("Z", "+00:00"))
    if expiration.tzinfo is None or expiration <= datetime.now(timezone.utc):
        raise ValueError("Root execution admission expired or lacks timezone")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU producer requires CUDA_VISIBLE_DEVICES empty")
    return original.confined(expected_output)


def _seal_observation_entry_v1(parent, output, inference, guard):
    """Observed-only handoff for the strict full952 loader; no target reads."""
    post = inference["post_five_iterations"]
    contract = {key: post[key] for key in ("coverage_original_bp0", "local_to_original_offset_bp",
        "retained_fixed100bp_indexes", "original_parent_interval_bp0", "normalization_length_bp",
        "physical_coordinate_transform")}
    source_roster = original.ref(str((output/"observed/sample_manifest.json").relative_to(ROOT)))
    native_inference_ref = original.ref(str((output/"relate/inference.json").relative_to(ROOT)))
    wrapper = dict(schema="mutation_routed_full_panel_inference_receipt_v1", parent_id=parent["parent_id"],
        split=parent["split"], original_observed_sample_manifest=source_roster,
        inferred_tree=inference["inferred_tree"], observed_sample_manifest=inference["observed_sample_manifest"],
        sample_index_to_inferred_tree_node_id=post["sample_index_to_inferred_tree_node_id"],
        coordinate_contract=contract, native_inference_receipt=native_inference_ref,
        truth_or_oracle_read=False, native_sample_dates_unchanged=True)
    wrapper_path = output/"full_panel_observed_inference_receipt.json"
    original.json_write(wrapper_path, wrapper, guard)
    refs = {key: original.ref(str((output/"observed"/filename).relative_to(ROOT))) for key,filename in (
        ("source_roster","sample_manifest.json"),("vcf","genotypes.vcf"),
        ("site_coordinates","site_coordinate_manifest.json"),("scoring_coordinates","scoring_coordinates.npz"),
        ("observation_manifest","observation_manifest.json"))}
    refs.update(native_tree=inference["inferred_tree"], derived_roster=inference["observed_sample_manifest"],
                inference_receipt=original.ref(str(wrapper_path.relative_to(ROOT))))
    entry = dict(schema="mutation_routed_full_panel_observed_entry_v1", parent_id=parent["parent_id"],
        split=parent["split"], observations_only=True, truth_or_oracle_read=False,
        refs=refs, coordinate_contract=contract, empirical_GRCh37_coordinate_claimed=False,
        samples=952, query_haplotypes=686, pure_reference_haplotypes=266,
        original_native_inference_receipt=native_inference_ref)
    path = output/"full_panel_observed_entry.json"
    original.json_write(path,entry,guard)
    return original.ref(str(path.relative_to(ROOT)))


def _storage_adapter_v1(ref):
    allowed = {"src/arggnn/project_storage_full_reservation_v1.py", "src/arggnn/project_storage_closed_scopes_v1.py"}
    if ref["path"] not in allowed or original.ref(ref["path"]) != ref:
        raise ValueError("Exact root-pinned compatible storage consumer required")
    name="arggnn._expanded_full_root_storage_adapter_v1"
    spec=importlib.util.spec_from_file_location(name,original.confined(ref["path"]))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def execute_parent_v1(config, parent_id, stage, admission, *, guard):
    """Root invokes one newly registered simulation/Relate parent with its guard.

    Generation computes and seals external truth. Relate only reads observed
    files. Outer parents are never read for model tuning/evaluation here.
    """
    validate_config_v1(config)
    if stage not in ("simulate", "relate", "all") or not callable(guard):
        raise ValueError("One existing producer stage and root write guard required")
    registry = {row["parent_id"]: row for row in config["parent_registry"]}
    parent = registry[parent_id]
    output = _admission_v1(config, parent, stage, admission)
    original.native_tools(config)
    generation = None
    if stage in ("simulate", "all"):
        if output.exists():
            raise FileExistsError("Expanded parents are fresh; existing outputs cannot be replaced or reseeded")
        guard(32768)
        output.mkdir(parents=True)
        original.json_write(output / "expanded_registry_seal.json", dict(
            config_content_sha256=original.digest(config), parent_registry_sha256=original.digest(config["parent_registry"]),
            parent_registry=config["parent_registry"], source_ref=original.ref(SOURCE),
            original_producer_ref=original.ref(original.SOURCE), selected_parent=parent,
            all_streams_registered_before_generation=True, outer_evaluation_authorized=False), guard)
        generation = original.generate_parent(config, parent, output, guard)
    else:
        seal = json.loads((output / "expanded_registry_seal.json").read_text())
        if (seal["config_content_sha256"] != original.digest(config)
                or seal["parent_registry_sha256"] != original.digest(config["parent_registry"])
                or seal["selected_parent"] != parent):
            raise ValueError("Earlier generation is not this registered parent")
        generation = json.loads((output / "generation.json").read_text())
    inference = None
    if stage in ("relate", "all"):
        # Read only generation metadata to bind the observed whitelist. Never
        # open the target archives or raw ancestry trees during inference.
        refs = {name: {"path": str((output/"observed"/name).relative_to(ROOT)),
                       "sha256": generation["files"]["observed/"+name]["sha256"]}
                for name in original.OBSERVED_FILES}
        inference = original.infer_parent(config, parent, output, guard, refs)
    entry_ref = None if inference is None else _seal_observation_entry_v1(parent,output,inference,guard)
    result = dict(schema="expanded_full_panel_v1_parent_completion", parent_id=parent_id, split=parent["split"],
        downstream_role=parent["downstream_role"], generation_completed=True, Relate_completed=inference is not None,
        source_ref=original.ref(SOURCE), original_producer_ref=original.ref(original.SOURCE),
        config_content_sha256=original.digest(config), parent_registry_sha256=original.digest(config["parent_registry"]),
        generation_ref=original.ref(str((output/"generation.json").relative_to(ROOT))),
        inference_ref=None if inference is None else original.ref(str((output/"relate/inference.json").relative_to(ROOT))),
        observed_only_inference=True, observation_entry_ref=entry_ref, outer_truth_evaluated=False, models_prepared=False, GPU_used=False)
    name = "expanded_simulation_completed.json" if stage == "simulate" else "expanded_parent_completed.json"
    original.json_write(output / name, result, guard)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--parent", default="ap_expanded_train_000")
    parser.add_argument("--stage", choices=("simulate", "relate", "all"), default="all")
    parser.add_argument("--admission")
    args = parser.parse_args()
    config = build_config_v1() if args.config is None else json.loads(original.confined(args.config).read_text())
    result = validate_config_v1(config)
    if args.expected_source_sha256 and args.expected_source_sha256 != original.sha(original.confined(SOURCE)):
        parser.error("Expected source bytes differ")
    if args.expected_config_sha256 and (not args.config or args.expected_config_sha256 != original.sha(original.confined(args.config))):
        parser.error("Expected config bytes differ")
    if not args.execute:
        print(json.dumps(dict(result, parent_registry_sha256=original.digest(config["parent_registry"]), parent_registry=config["parent_registry"]),indent=2,sort_keys=True))
        return
    if not args.config or not args.admission or not args.expected_config_sha256 or not args.expected_source_sha256:
        parser.error("Root execution needs frozen --config/source hashes and active --admission")
    admission = json.loads(original.confined(args.admission).read_text())
    storage_adapter = _storage_adapter_v1(admission["storage_adapter_ref"])
    config_ref = original.ref(args.config)
    storage_adapter.validate_storage_v9(admission, config, admission["storage"], source_ref=admission["source_ref"],
                        config_ref=config_ref, runtime_ref=admission["runtime_ref"])
    def guard(planned=0):
        return storage_adapter.budget_check_v9(admission["storage"], boundary=True, required_write_bytes=int(planned))
    print(json.dumps(execute_parent_v1(config,args.parent,args.stage,admission,guard=guard),indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
