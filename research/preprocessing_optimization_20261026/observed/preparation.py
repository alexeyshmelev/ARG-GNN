"""Complete original TreeLoom parent preparation using qualified exact kernels."""
import importlib
from arggnn.benchmarking import european_treeloom as original
_implementation=original._implementation
_need=original._need
PROTOCOLS=original.PROTOCOLS
_STATE_KEYS=original._STATE_KEYS
NATIVE_SIX_CLASSES=original.NATIVE_SIX_CLASSES
IDENTITY=original.IDENTITY
EuropeanWholeParent=original.EuropeanWholeParent
_sampled_records=original._sampled_records

def prepare_observation(state, *, protocol_id, parent_id, split, source_refs,
                        sampled_registry_ref, max_nodes, max_holds,
                        max_episode_payload_bytes, resource_check,
                        max_sampled_metadata_bytes=128*2**20):
    implementation = _implementation(protocol_id)
    condition = implementation.CONDITIONS[PROTOCOLS[protocol_id]]
    _need(callable(resource_check) and type(state) is dict and set(state) == _STATE_KEYS
          and all(type(v) is int and v > 0 for v in (max_nodes,max_holds,max_episode_payload_bytes))
          and source_refs == state["source_refs"]
          and state["manifest"]["parent_id"] == parent_id
          and state["manifest"]["split"] == split,
          "Actual same-parent observed state, split, refs and complete bounds required")
    import numpy as np
    import torch
    observations = importlib.import_module(
        "research.preprocessing_optimization_20261026.tree.observations")
    episode_preparation = importlib.import_module(
        "research.preprocessing_optimization_20261026.tree.episodes")
    cut_preparation = importlib.import_module(
        "research.preprocessing_optimization_20261026.tree.cut")
    from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
    from arggnn.american_c3_gnn_targets_v1 import identity_sha
    ts, h = state["ts"], condition["haplotypes"]
    _need(ts.num_trees <= max_holds and ts.num_samples == h
          and state["genotype"].shape == (ts.num_sites,h)
          and np.array_equal(state["positions"],ts.tables.sites.position)
          and np.array_equal(state["nodes"],ts.samples())
          and len(state["query_columns"]) == condition["queries"]
          and state["manifest"]["sequence_length_bp"] == condition["length_bp"]
          and state["manifest"]["truth_or_oracle_read"] is False,
          "Every original observed native tree/site/sample must remain present")
    trees, intervals, nodes = [], [], 0
    workspace = observations.native_workspace_v1(ts)
    node_time = np.asarray(ts.nodes_time,dtype=np.float64)
    for tree in ts.trees():
        if tree.index % 256 == 0:
            resource_check("European_TreeLoom_complete_native_tree_list")
        native = observations.native_tree(tree,state["nodes"],node_time=node_time,workspace=workspace)
        nodes += len(native.parent)
        _need(nodes <= max_nodes, "Complete native parent exceeds node cap; never truncate")
        trees.append(native)
        intervals.append(tuple(tree.interval))
    bank, observation_metadata = observations.encode_block(
        trees,intervals,state["genotype"],state["allele_count"],state["positions"],
        state["labels"],state["ages"],num_classes=4,age_split=350.,
        query_columns=state["query_columns"],resource_check=resource_check)
    del trees
    _need(bank.parent.numel() == nodes and len(bank.spans) == ts.num_trees
          and len(bank.genotype) == ts.num_sites, "Full original observed bank required")
    isolated = tuple(observation_metadata.get("isolated_zero_site_holds",()))
    resource_check("European_TreeLoom_global_episodes")
    arrays, episode_metadata = episode_preparation.build_clade_episode_arrays_v1(
        bank.parent.numpy(),bank.sample_index.numpy(),bank.spans.numpy(),
        max_payload_bytes=max_episode_payload_bytes,isolated_hold_indices=isolated)
    episodes = OrderedCladeEpisodesV1.from_verified_arrays(arrays,episode_metadata)
    resource_check("European_TreeLoom_global_cut_plan")
    cut, cut_metadata = cut_preparation.prepare_cut_plan_v1(
        bank,expected_haplotypes=h,max_nodes=max_nodes,max_holds=max_holds,isolated_hold_indices=isolated)
    refs = dict(source_refs,parent_id=parent_id,query_truth_in_inputs=False,
                all_observed_trees_nodes_sites_samples_retained=True,
                supported_length_bp=float(bank.spans.double().sum()),
                requested_length_bp=condition["length_bp"])
    native_split = {"validation":"val_prediction","test":"evaluation_prediction",
                    "evaluation":"evaluation_prediction"}.get(split,split)
    original = implementation.FrozenEuropeanBundleV1(
        bank,episodes,cut,PROTOCOLS[protocol_id],parent_id,native_split,refs,
        implementation.memory_core._stamps(implementation._bank_tensors(bank)),
        implementation.memory_core._stamps(implementation._episode_tensors(episodes)))
    original.validate()
    records = _sampled_records(sampled_registry_ref,
        observation_ref=source_refs["inference_ref"],parent_id=parent_id,split=split,
        max_metadata_bytes=max_sampled_metadata_bytes)
    output_classes = (NATIVE_SIX_CLASSES if protocol_id == "ancestralpaths_paper200mb_native6"
                      else condition["output_classes"])
    n = len(records.get("record_ids",[]))
    _need(set(IDENTITY) <= set(records) and n > 0
          and all(records[k].dtype == np.int64 and records[k].shape == (n,)
                  for k in ("record_ids","sample_nodes","sample_indices"))
          and len(set(records["record_ids"].tolist())) == n
          and records["positions_bp0"].dtype == np.float64 and records["positions_bp0"].shape == (n,)
          and records["parent_id"].shape == () and records["parent_id"].item() == parent_id
          and records["class_order"].tolist() == list(output_classes)
          and np.isin(records["sample_indices"],state["query_columns"]).all()
          and np.array_equal(records["sample_nodes"],state["nodes"][records["sample_indices"]]),
          "Exact original sampled row/sample/class identity required; repeated draws preserved")
    local = records["positions_bp0"] - state["coordinate_offset_bp"]
    _need(np.isfinite(local).all() and ((local >= 0) & (local < ts.sequence_length)).all(),
          "No invented native tail or coordinate extrapolation")
    holds = np.searchsorted(np.asarray(list(ts.breakpoints())),local,side="right")-1
    focal_hold = torch.as_tensor(holds.copy(),dtype=torch.long)
    focal_sample = torch.as_tensor(records["sample_indices"].copy(),dtype=torch.long)
    bundle = EuropeanWholeParent(original,focal_hold,focal_sample,dict(sampled_registry_ref),identity_sha(records),
        implementation.memory_core._stamps((focal_hold,focal_sample)))
    bundle.validate()
    return bundle, dict(parent_id=parent_id,protocol_id=protocol_id,
        sampled_registry_ref=sampled_registry_ref,focal_identity_sha256=bundle.focal_identity_sha256,
        focal_records=n,episode_metadata=episode_metadata,cut_metadata=cut_metadata,
        observation_metadata=observation_metadata,private_truth_opened=False,
        all_native_trees_nodes_sites_samples_retained=True,
        structural_recipes_global_before_partition=True,artificial_block_boundaries=False)

