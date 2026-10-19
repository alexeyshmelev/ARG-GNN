"""Fresh full-parent recipe: unchanged domain/structure, explicit fast global encoder."""
from collections.abc import Mapping
import numpy as np
import torch
from arggnn.american_c3_common_v1 import require,CLASSES
from arggnn.american_c3_gnn_targets_v1 import identity_sha
from research.reviews.frozen_benchmark_training_ready_20261017_v1.american_full30mb_frozen_v1 import TRAIN,VAL
from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as _original
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as memory_core
from .observations import native_tree,encode_block,native_workspace_v1
from .episodes import build_episodes
from .cuts import prepare_cut_plan_v1
AmericanTreeLoomWholeParentV1=_original.AmericanTreeLoomWholeParentV1
_short_ref=_original._short_ref
_bank_tensors=_original._bank_tensors
_episode_tensors=_original._episode_tensors
GENEALOGY=_original.GENEALOGY
HAPLOTYPES=_original.HAPLOTYPES
FOCALS=_original.FOCALS

def prepare_from_loaded_observation_v1(state, *, split, observation_ref,
        focal_registry_ref, native_preparation_ref, max_nodes, max_holds,
        max_episode_payload_bytes,resource_check=lambda stage:None):
    """Root supplies existing sanitized load_observation output, never truth.

    One full parent is encoded. This does not load files, accept a block range,
    prune, subsample, replay native inference or open a target resolver.
    Bounds reject the whole parent. They never authorize workload reduction.
    """
    for ref in (observation_ref, focal_registry_ref, native_preparation_ref):
        _short_ref(ref)
    require(isinstance(state, Mapping)
        and all(type(x) is int and x > 0 for x in
            (max_nodes, max_holds, max_episode_payload_bytes)),
        "Explicit root resource failstop bounds")
    manifest, ts = state["manifest"], state["ts"]
    parent_id = manifest["parent_id"]
    length = int(ts.sequence_length)
    require(ts.sequence_length == length, "Exact integer native physical support")
    require(manifest["schema"] == "american_c3_observed_v1"
        and manifest["genealogy_kind"] == GENEALOGY
        and manifest["query_ancestry_truth_in_inputs"] is False
        and manifest["class_order"] == list(CLASSES),
        "Same sanitized inferred native observation, no query ancestry in inputs")
    require((split == "train" and parent_id in TRAIN and length == 30000000)
        or (split == "val_prediction" and parent_id in VAL and length == 30000000),
        "Exact independent TRAIN or original held-out VAL parent")
    require(np.array_equal(state["labels"],
            np.array([2]*200+[0]*200+[1]*200+[-1]*1000, dtype=np.int64))
        and np.array_equal(state["query_columns"], np.arange(600, 1600))
        and len(state["nodes"]) == 1600
        and len(state["focals"]["record_ids"]) == FOCALS
        and state["focals"]["parent_id"].item() == parent_id
        and state["focals"]["class_order"].tolist() == list(CLASSES)
        and np.array_equal(state["focals"]["sample_nodes"],
            state["nodes"][state["focals"]["sample_indices"]])
        and np.isin(state["focals"]["sample_indices"], np.arange(600, 1600)).all(),
        "All original reference/query identities and native10000focal join")
    genotype = np.asarray(state["genotype"])
    require(genotype.shape == (ts.num_sites, HAPLOTYPES)
        and (genotype >= 0).all()
        and np.array_equal(state["positions"], ts.tables.sites.position),
        "All original complete observed SNP calls; no new filtering/imputation")
    require(ts.num_trees == len(state["spans"]) <= max_holds,
        "Full-parent native hold limit, never a block truncate")
    trees, intervals, nodes = [], [], 0
    global_node_time=np.asarray(ts.nodes_time,dtype=np.float64)
    workspace=native_workspace_v1(ts)
    resource_check("prepare_native_tree_list")
    for tree in ts.trees():
        if tree.index%512==0:resource_check("prepare_native_tree_list")
        native = native_tree(tree, state["nodes"],node_time=global_node_time,workspace=workspace)
        nodes += len(native.parent)
        require(nodes <= max_nodes, "Full-parent native node limit")
        trees.append(native)
        intervals.append(tuple(tree.interval))
    resource_check("encode_full_observed_bank")
    bank, observation_metadata = encode_block(trees, intervals, genotype,
        state["allele_count"], state["positions"], state["labels"], state["ages"],
        num_classes=3, age_split=350., query_columns=state["query_columns"],resource_check=resource_check)
    require(bank.parent.numel() == nodes
        and len(bank.spans) == ts.num_trees
        and len(bank.genotype) == ts.num_sites,
        "No native node, tree or SNP was dropped")
    isolated_holds=tuple(observation_metadata.get("isolated_zero_site_holds",()))
    resource_check("prepare_global_clade_episodes")
    episodes, episode_metadata = build_episodes(
        bank, max_payload_bytes=max_episode_payload_bytes, isolated_hold_indices=isolated_holds)
    resource_check("prepare_global_cut_plan")
    cut, cut_metadata = prepare_cut_plan_v1(bank,
        expected_haplotypes=HAPLOTYPES, max_nodes=max_nodes, max_holds=max_holds,isolated_hold_indices=isolated_holds)
    positions = state["focals"]["positions_bp0"]
    require(np.isfinite(positions).all() and ((positions >= 0) & (positions < length)).all(),
        "No native focal coordinate extrapolation")
    exact_focal_hold = np.searchsorted(np.asarray(list(ts.breakpoints())),
        positions, side="right")-1
    require(np.array_equal(state["focal_tree"], exact_focal_hold),
        "Same original half-open native focal/tree join")
    focal_hold = torch.as_tensor(exact_focal_hold.copy(), dtype=torch.long)
    focal_sample = torch.as_tensor(
        np.asarray(state["focals"]["sample_indices"]).copy(), dtype=torch.long)
    identity = identity_sha(state["focals"])
    bundle = AmericanTreeLoomWholeParentV1(bank, episodes, cut,
        parent_id, split, length, ts.num_trees, ts.num_sites,
        focal_hold, focal_sample, identity,
        dict(observation_ref), dict(focal_registry_ref), dict(native_preparation_ref),
        memory_core._stamps(_bank_tensors(bank)),
        memory_core._stamps(_episode_tensors(episodes)),
        memory_core._stamps((focal_hold, focal_sample)))
    resource_check("validate_complete_observed_bundle")
    bundle.validate()
    metadata=dict(schema="American_TreeLoom_full_observation_preparation_v1",
        parent_id=parent_id, split=split, observation_only=True,
        observation_ref=dict(observation_ref), focal_registry_ref=dict(focal_registry_ref),
        native_preparation_ref=dict(native_preparation_ref),
        focal_identity_sha256=identity, focal_records=FOCALS,
        all_native_trees_nodes_sites_samples_retained=True,
        structural_recipes_global_before_partition=True,
        artificial_block_boundaries=False,
        observation_metadata=observation_metadata,
        episode_metadata=episode_metadata, cut_metadata=cut_metadata,
        private_truth_opened=False, model_executed=False)

    if isolated_holds:
        metadata.update(isolated_zero_site_holds=list(isolated_holds),
            native_hold_compatibility="connected_or_all_sampled_all_isolated_zero_SNP_only",
            no_pruning_virtual_roots_or_fake_edges=True)
    return bundle,metadata
