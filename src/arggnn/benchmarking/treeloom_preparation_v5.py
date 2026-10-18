"""V5 metadata-only raw-to-inferred qualification lineage; v4 numerical code frozen."""
from __future__ import annotations
import copy
import math
from . import contracts as c
from . import treeloom_preparation_v4 as frozen

PARENT=frozen.PARENT
GIB=2**30
IDENTITY=("sample_index","sample_id","haplotype","age_generations","role","observed_reference_class")


def _need(ok,message):
    if not ok:raise c.ContractError(message)


def _read_metadata(ref):
    path=c.contained_path(ref["path"])
    _need(path.suffix==".json" and path.stat().st_size<=4*2**20,
          "Only bounded public observation/roster JSON metadata may be inspected")
    return c.read_ref(ref)


def observed_lineage(preparation_ref,generation_ref):
    """Authenticate explicit generation->inference->focal chain, never scientific arrays."""
    from . import american_observation as observation
    prepared=_read_metadata(preparation_ref)
    generation=_read_metadata(generation_ref)
    raw_ref=generation["observation_ref"];inferred_ref=prepared["observation_ref"]
    raw=_read_metadata(raw_ref);inferred=_read_metadata(inferred_ref)
    inference_ref=prepared["input_ref"];inference=_read_metadata(inference_ref)
    rows=(_read_metadata(raw["roster_ref"]),_read_metadata(inferred["roster_ref"]))
    _need(generation.get("schema")=="american_fresh_shared_observation_ready_v1"
          and inference.get("schema")=="american_fresh_inference_ready_v1"
          and prepared.get("schema")=="american_fresh_compact_focal_preparation_v1"
          and prepared.get("status")=="completed_compact_query_focals"
          and raw.get("schema")==inferred.get("schema")=="american_c3_observed_v1"
          and all(value.get("parent_id")==PARENT and value.get("split")=="train"
                  for value in (generation,inference,prepared,raw,inferred)),
          "Same existing AMtrain000 public generation/inference/focal lineage required")
    _need(inference.get("input_ref")==generation_ref
          and inference.get("observation_ref")==inferred_ref
          and inferred.get("source_observation_ref")==raw_ref
          and inferred_ref!=raw_ref
          and inferred.get("genealogy_kind")==prepared.get("genealogy_kind")==observation.GENEALOGY
          and raw.get("genealogy_kind")=="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"
          and inferred.get("tree_ref")!=raw.get("tree_ref")
          and inferred.get("sites_and_sample_identities_verified") is True,
          "Explicit raw-to-inferred lineage must retain the inferred predictor genealogy")
    _need(generation.get("query_truth_in_training_inputs") is False
          and generation.get("simulated_true_topology_is_predictor_input") is False
          and inference.get("query_truth_in_training_inputs") is False
          and prepared.get("query_ancestry_truth_opened") is False
          and prepared.get("query_truth_in_training_inputs") is False
          and all(value.get("query_ancestry_truth_in_inputs") is False for value in (raw,inferred)),
          "Public observed lineage must exclude query ancestry truth")
    _need(all(raw.get(key)==inferred.get(key) for key in
                  ("genotypes_ref","positions_ref","alleles_ref","class_order",
                   "sequence_length_bp","sample_unit","counts_haploid"))
          and raw.get("class_order")==prepared.get("class_order")==["EUR","EAS","AFR"]
          and raw.get("sequence_length_bp")==prepared.get("sequence_length_bp")==30000000
          and raw.get("sample_unit")=="haploid"
          and prepared.get("query_focal_count")==10000
          and prepared.get("all_original_samples_and_trees_traversed") is True
          and prepared.get("original_record_ids_preserved") is True,
          "Every original genotype/site/class/sample and focal identity must be preserved")
    _need(all(generation.get(key)==inference.get(key) for key in
                  ("seeds","campaign_ref","comparison_ref","recipe_ref"))
          and all(generation.get(key)==prepared.get(key) for key in
                  ("campaign_ref","comparison_ref","recipe_ref"))
          and prepared.get("path_sampling_seed")==generation.get("seeds",{}).get("sampling"),
          "Same declared recipe and random-stream lineage required")
    raw_roster,inferred_roster=rows
    _need(all(roster.get("parent_id")==PARENT and roster.get("class_order")==raw["class_order"]
              for roster in rows)
          and len(raw_roster.get("samples",()))==len(inferred_roster.get("samples",()))==1600,
          "All1600 original observed reference/query identities required")
    old_rows=raw_roster["samples"];new_rows=inferred_roster["samples"]
    expected_labels=[2]*200+[0]*200+[1]*200+[None]*1000
    node_maps=[]
    for roster_rows in (old_rows,new_rows):
        _need([row.get("sample_index") for row in roster_rows]==list(range(1600))
              and [row.get("observed_reference_class") for row in roster_rows]==expected_labels
              and [row.get("role") for row in roster_rows]==["pure_source_reference"]*600+["query"]*1000
              and len({(row.get("sample_id"),row.get("haplotype")) for row in roster_rows})==1600
              and all(isinstance(row.get("sample_id"),str) and row["sample_id"]
                      and type(row.get("haplotype")) is int and row["haplotype"] in (0,1)
                      and type(row.get("age_generations")) in (int,float)
                      and math.isfinite(row["age_generations"]) and row["age_generations"]>=0
                      and type(row.get("observed_node_id")) is int and row["observed_node_id"]>=0
                      for row in roster_rows),
              "Exact original reference labels, query masking, sample ordering and finite ages")
        node_map=[row["observed_node_id"] for row in roster_rows]
        _need(len(set(node_map))==1600,"Distinct observed sample-node identities")
        node_maps.append(node_map)
    _need(all(tuple(old.get(key) for key in IDENTITY)==tuple(new.get(key) for key in IDENTITY)
              for old,new in zip(old_rows,new_rows))
          and inferred_roster.get("sample_index_to_inferred_node")==node_maps[1],
          "Same all1600 sample/reference/query identity join despite changed inferred node mapping")
    refs=dict(observation_ref=copy.deepcopy(inferred_ref),
              focal_registry_ref=copy.deepcopy(prepared["focals_ref"]),
              native_preparation_ref=copy.deepcopy(preparation_ref))
    lineage=dict(schema="AMtrain000_public_raw_to_inferred_lineage_v5",
        generation_ref=copy.deepcopy(generation_ref),raw_observation_ref=copy.deepcopy(raw_ref),
        inference_ref=copy.deepcopy(inference_ref),inferred_observation_ref=copy.deepcopy(inferred_ref),
        preparation_ref=copy.deepcopy(preparation_ref),
        raw_roster_ref=copy.deepcopy(raw["roster_ref"]),inferred_roster_ref=copy.deepcopy(inferred["roster_ref"]),
        unchanged_shared_observed_refs={k:copy.deepcopy(raw[k]) for k in
                                       ("genotypes_ref","positions_ref","alleles_ref")},
        samples=1600,query_samples=1000,reference_class_counts=[200,200,200],
        exact_ordered_sample_identity_roles_labels_ages=True,
        raw_tree_is_not_predictor_input=True,inferred_genealogy_required=True,
        scientific_payloads_opened=False,private_truth_opened=False)
    return refs,lineage


def _descriptor(function,spec):
    result=frozen._descriptor(function,spec)
    result["source_refs"]=list({ref["path"]:ref for ref in
                               [*result["source_refs"],c.file_ref(__file__)]}.values())
    return result


def qualification_factory(*,preparation_ref,generation_ref,output_directory,max_rss_bytes,
                          max_seconds=7200,max_output_bytes=16*2**20):
    from . import american_workflow as american
    from . import treeloom_training as original
    _need(max_rss_bytes==128*GIB and 0<max_seconds<=7200
          and 2*2**20<=max_output_bytes<=16*2**20,
          "One full-parent CPU qualification has finite128GiB/7200s/16MiB caps")
    refs,lineage=observed_lineage(preparation_ref,generation_ref)
    supervisor_output=c.contained_path(output_directory)
    output=supervisor_output/"qualification"
    _need(supervisor_output.is_relative_to(c.ROOT/"experiments/results") and not output.exists(),
          "Fresh qualification child below the supervisor output scope")
    spec=dict(schema=frozen.QUALIFICATION_SCHEMA,parent_id=PARENT,
        observed_input_refs=refs,preparation_limits=copy.deepcopy(original.LIMITS),
        parent_registry_ref=c.file_ref(american.REGISTRY),
        output_directory=str(output.relative_to(c.ROOT)),max_rss_bytes=max_rss_bytes,
        max_seconds=max_seconds,max_output_bytes=max_output_bytes-2**20,
        max_parent_compact_bytes=frozen._parent_bound(),
        candidate_source_refs=frozen.candidate_source_refs(),
        CPU_only=True,truth_or_model_allowed=False,disk_cache_bytes=0,
        observed_lineage_validation=lineage)
    return _descriptor("run_qualification",spec)


def fit_factory(*,preparation_refs,generation_refs,output_directory,max_rss_bytes,qualification_ref):
    # Frozen v4 fit already checks actual PASS +all20 observed bindings before truth.
    result=frozen.fit_factory(preparation_refs=preparation_refs,generation_refs=generation_refs,
        output_directory=output_directory,max_rss_bytes=max_rss_bytes,qualification_ref=qualification_ref)
    result["source_refs"]=list({ref["path"]:ref for ref in
                               [*result["source_refs"],c.file_ref(__file__)]}.values())
    return result


# Exact frozen numerical/qualification entrypoints; no live module is patched.
run_qualification=frozen.run_qualification
run_fit=frozen.run_fit
