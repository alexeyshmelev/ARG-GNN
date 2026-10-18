"""Fresh shared American inference and native paths over pinned CPU algorithms.

run_inference consumes the public fresh-generation receipt. run_native_preparation
consumes its public inferred-observation receipt. Neither opens private ancestry,
the simulated true tree, a legacy controller or a model checkpoint. There is no
launch, numerical import or dataset access at import.
"""
from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path

from . import american_preparation as generation

ROOT = generation.ROOT
GENEALOGY = "inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
UNARY_POLICY = "direct_tskit_simplify_all_samples_keep_unary_false_no_site_filter_no_trimming_no_gap_split_v1"
INFERENCE_CONVENTIONS = {
    "tsinfer_threads": 8, "ancestral_allele": 0, "unary_preprocessing": UNARY_POLICY,
    "tsdate": "installed_supported_defaults_with_paper_mutation_rate",
}
FOCAL_CONVENTIONS = {
    "unique_raw_paths_before_truncation": True,
    "count": 10000, "query_selection": "label_blind_without_replacement",
    "coordinate": "midpoint_of_first_observed_occurrence",
    "native_path_train_fraction": .9, "native_path_shuffle_seed": 42,
}
SOURCE_REFS = (
    {"path": "src/arggnn/benchmarking/american_preparation.py",
     "sha256": "a3778cd8b11cd7064233c4bf55f14e38815c95ffb6341e36208917b9e3989b02"},
    {"path": "src/arggnn/american_c3_observe_v3.py",
     "sha256": "4f381c24b75f6ab94d80cf457605375ad01adaed9dcc46dadc27f3dea4f3e630"},
    {"path": "src/arggnn/american_c3_paths_v1.py",
     "sha256": "3b32f16e0a81160852e5b93fe9033abf9bc2eeb56c3b15ca552d73f4aacf374c"},
    {"path": ".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",
     "sha256": "e7e66a029694b21a2ba1f9c9a6c4fbc193d81b8831282580ba9639a9e224726c"},
)
STAGES = {
    "shared_inference": {
        "module": "arggnn.american_c3_observe_v3",
        "input_schema": "american_fresh_shared_observation_ready_v1",
        "result_schema": "american_fresh_inference_ready_v1",
        "completion_status": "completed_current_C3_inference",
        "caps": ("max_sites", "max_tables_bytes", "max_node_map_bytes", "max_sample_store_bytes"),
    },
    "native_preparation": {
        "module": "arggnn.american_c3_paths_v1",
        "input_schema": "american_fresh_inference_ready_v1",
        "result_schema": "american_fresh_native_preparation_ready_v1",
        "completion_status": "completed_native_C3_path_preparation",
        "caps": ("max_unique_paths", "max_path_tokens", "max_path_occurrences",
                 "max_vocabulary_nodes", "max_raw_path_nodes"),
    },
}


def _observed_metadata(ref):
    if any(part.lower() in {"private", "truth", "oracle", "census"}
           for part in Path(ref["path"]).parts):
        raise ValueError("Shared inference/native preparation accepts public observed references only")
    return generation._read(ref)


def _same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def prepare(spec):
    """Return an inert one-parent plan with no scientific or ancestry I/O."""
    if spec.get("schema") != "american_fresh_observation_stage_v1" or spec.get("stage") not in STAGES:
        raise ValueError("Explicit fresh American observation/native stage required")
    stage, definition = spec["stage"], STAGES[spec["stage"]]
    caps = (*definition["caps"], "max_output_bytes", "max_seconds", "max_rss_bytes")
    if any(type(spec.get(k)) is not int or spec[k] <= 0 for k in caps):
        raise ValueError("Finite positive whole-parent output/RSS/wall/count limits required")
    source = _observed_metadata(spec["input_ref"])
    if (source.get("schema") != definition["input_schema"]
            or source.get("query_truth_in_training_inputs") is not False
            or source.get("truth_scoring_authorized") is not False
            or source.get("split") not in ("train", "val", "test")
            or not source.get("parent_id", "").startswith("argformer_american_")
            or source.get("exact_paper_reproduction") is not False):
        raise ValueError("Fresh public same-parent observation lineage required")
    if not _same(source.get("declared_conventions"), generation.CONVENTIONS):
        raise ValueError("Generation convention lineage changed")
    observation = _observed_metadata(source["observation_ref"])
    if (observation.get("schema") != "american_c3_observed_v1"
            or observation.get("parent_id") != source["parent_id"]
            or observation.get("split") != source["split"]
            or observation.get("class_order") != ["EUR", "EAS", "AFR"]
            or observation.get("sequence_length_bp") != 30000000
            or observation.get("query_ancestry_truth_in_inputs") is not False
            or observation.get("migrations_and_internal_population_labels_removed") is not True):
        raise ValueError("Full shared 30 Mb C3 observation/role contract required")
    output = generation._path(spec["output_directory"])
    input_root = generation._path(spec["input_ref"]["path"]).parent
    if (not output.is_relative_to(ROOT / "experiments/results")
            or source["parent_id"] not in output.parts or output.exists()
            or output.is_relative_to(input_root)):
        raise ValueError("Fresh isolated stage output outside the immutable input scope required")
    config = dict(execution_enabled=True, observation_ref=source["observation_ref"],
                  output_cap_bytes=spec["max_output_bytes"])
    config.update({key: spec[key] for key in definition["caps"]})
    if stage == "shared_inference":
        if not _same(spec.get("declared_inference_conventions"), INFERENCE_CONVENTIONS):
            raise ValueError("Freeze declared tsinfer/tsdate preprocessing conventions")
        if (source.get("requires_tsinfer_tsdate") is not True
                or observation.get("genealogy_kind") != "simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"):
            raise ValueError("Exactly one shared inference over generated genotypes is required")
        config.update(schema="american_c3_inferred_observation_v1", mutation_rate=1.25e-8,
                      num_threads=8, unary_preprocessing_policy=UNARY_POLICY,
                      inference_recipe="Fresh shared tsinfer+tsdate; declared ancestral allele zero and exact all-sample unary collapse; full observed sites/samples retained")
        bound = (spec["max_sample_store_bytes"] + 2 * (spec["max_tables_bytes"] + 16777216)
                 + spec["max_node_map_bytes"] + 8 * 1024 * 1024)
        if bound > spec["max_output_bytes"]:
            raise ValueError("Declared full sample-store/tree/node-map serialization bounds exceed output cap")
    else:
        if not _same(spec.get("declared_focal_conventions"), FOCAL_CONVENTIONS):
            raise ValueError("Freeze the declared label-blind native focal conventions")
        if (observation.get("genealogy_kind") != GENEALOGY
                or observation.get("sites_and_sample_identities_verified") is not True
                or source.get("requires_tsinfer_tsdate") is not False):
            raise ValueError("Complete shared inferred observation required before native paths")
        config.update(schema="american_c3_native_path_preparation_v1", genealogy_kind=GENEALOGY,
                      query_focal_count=10000, path_sampling_seed=source["seeds"]["sampling"])
    generation._verify_sources()
    for ref in SOURCE_REFS:
        if generation._reference(ref["path"]) != ref:
            raise ValueError("Pinned observation/native source changed: " + ref["path"])
    return dict(stage=stage, source=source, config=config, output_directory=str(output),
                execution_authorized=False, exact_paper_reproduction=False)


class _StageWriter(generation._Writer):
    def prepare(self, name, planned):
        if name == "completed.json":
            name = "stage_completed.json"
        if Path(name).parts[0] in {"private", "truth"}:
            raise ValueError("Observed/native stages must never produce private truth artifacts")
        return super().prepare(name, planned)

    def pickle(self, name, value, planned):
        path = self.prepare(name, planned)
        with path.open("xb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return self.finish(path, planned)


def _run(spec, expected_stage, *, resource_check, execution_check):
    if not callable(resource_check) or not callable(execution_check):
        raise ValueError("Actual caller admission and resource checks required")
    plan = prepare(spec)
    if plan["stage"] != expected_stage:
        raise ValueError("Concrete stage callable/spec mismatch")
    execution_check()
    import os
    if Path.cwd().resolve() != ROOT or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CPU-only repository worker required")
    resource_check("american_" + expected_stage + "_before_scientific_import")
    definition = STAGES[expected_stage]
    module = importlib.import_module(definition["module"])
    if expected_stage == "native_preparation" and module.storage_bound(plan["config"]) > spec["max_output_bytes"]:
        raise ValueError("Complete native path production bounds exceed admitted output cap")
    writer = _StageWriter(generation._path(plan["output_directory"]), spec["max_output_bytes"], resource_check)
    completion_ref = module.execute(plan["config"], writer)
    resource_check("american_" + expected_stage + "_complete")
    completion = _observed_metadata(completion_ref)
    if completion.get("status") != definition["completion_status"] or writer.pending:
        raise ValueError("Genuine complete same-parent stage receipt required")
    source = plan["source"]
    if expected_stage == "shared_inference":
        inferred = _observed_metadata(completion["observation_ref"])
        if (inferred.get("parent_id") != source["parent_id"] or inferred.get("split") != source["split"]
                or inferred.get("class_order") != ["EUR", "EAS", "AFR"]
                or inferred.get("sequence_length_bp") != 30000000
                or inferred.get("genealogy_kind") != GENEALOGY
                or inferred.get("sites_and_sample_identities_verified") is not True
                or inferred.get("query_ancestry_truth_in_inputs") is not False):
            raise ValueError("Completed inference must preserve the exact shared parent and all observations")
    public = dict(
        schema=definition["result_schema"], parent_id=source["parent_id"], split=source["split"],
        input_ref=spec["input_ref"], observation_ref=(
            completion["observation_ref"] if expected_stage == "shared_inference" else source["observation_ref"]),
        requires_tsinfer_tsdate=False, query_truth_in_training_inputs=False,
        truth_scoring_authorized=False, exact_paper_reproduction=False,
        paper_known_parameters=source["paper_known_parameters"],
        declared_conventions=source["declared_conventions"], seeds=source["seeds"],
        campaign_ref=source["campaign_ref"], comparison_ref=source["comparison_ref"], recipe_ref=source["recipe_ref"],
        stage_completion_ref=completion_ref, source_refs=list(SOURCE_REFS),
        declared_inference_conventions=dict(INFERENCE_CONVENTIONS))
    if expected_stage == "native_preparation":
        if (completion.get("parent_id") != source["parent_id"]
                or completion.get("observation_ref") != source["observation_ref"]
                or completion.get("genealogy_kind") != GENEALOGY
                or completion.get("query_ancestry_truth_opened") is not False
                or completion.get("query_focal_count") != 10000
                or completion.get("contrastive_labels_reference_only") is not True):
            raise ValueError("Complete native corpus and label-blind 10,000-focal barrier required")
        public.update(native_preparation_ref=completion_ref,
            native_dataset_refs=completion["native_dataset_refs"], focals_ref=completion["focals_ref"],
            vocab_ref=completion["vocab_ref"], declared_focal_conventions=dict(FOCAL_CONVENTIONS))
    name = "inference_ready.json" if expected_stage == "shared_inference" else "native_preparation_ready.json"
    return writer.json(name, public)


def run_inference(spec, *, resource_check, execution_check):
    """Execute full shared tsinfer/unary-collapse/tsdate exactly once on CPU."""
    return _run(spec, "shared_inference", resource_check=resource_check, execution_check=execution_check)


def run_native_preparation(spec, *, resource_check, execution_check):
    """Execute complete native path views and shared 10,000 observed focals."""
    return _run(spec, "native_preparation", resource_check=resource_check, execution_check=execution_check)
