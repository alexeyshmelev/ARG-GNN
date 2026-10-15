"""Inert integration helpers for a fresh three-arm engine; no reader or launcher.

Imports are stdlib only until a caller explicitly requests model/preparation work.
No function grants execution, storage, target access, or checkpoint authority.
"""
from __future__ import annotations
from itertools import combinations
import hashlib
import statistics

ARMS = ("triple_support_pair_memory", "adjacent_marginal_null", "stable_leaf_memory")
PASSES = (5, 10, 15, 20, 25, 30)
TRAIN = tuple("ap_expanded_train_%03d" % i for i in range(32))
VAL = tuple("ap_expanded_val_%03d" % i for i in range(8))
READOUT = dict(query_block=32, checkpoint_chain=True, checkpoint_memory=True)
PARAMETERS_PER_ARM = 32446
BACKBONE_PARAMETER_PREFIX = "common.backbone."
CHECKPOINT_FILES = len(ARMS)*len(PASSES)
VAL_PRODUCTS = CHECKPOINT_FILES*len(VAL)
MODEL_MODULE = "research.reviews.order2_cut_memory_implementation_20261015_v1.order2_cut_memory_v1"


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _model():
    from importlib import import_module
    return import_module(MODEL_MODULE)


def protocol_shape_v1():
    """Structural schedule only. Root must separately register scientific criteria."""
    return dict(arms=list(ARMS), passes=30, TRAIN_parent_ids=list(TRAIN),
                VAL_parent_ids=list(VAL), updates_per_pass=32, updates_per_arm=960,
                updates_all_arms=2880, checkpoint_passes=list(PASSES),
                saved_checkpoints=CHECKPOINT_FILES,
                VAL_prediction_products_required_before_truth=VAL_PRODUCTS,
                common_decoder="graph_to_transition", parameters_per_arm=PARAMETERS_PER_ARM,
                common_grid_parameters=31933, added_memory_parameters=513,
                execution_mode="three_arm_sequential_one_parent", readout=dict(READOUT))


def expected_product_keys_v1():
    return frozenset((arm, at, pid) for arm in ARMS for at in PASSES for pid in VAL)


def raw_validation_probability_bytes_v1(scoring_centers_by_parent):
    """Raw FP32 [P,686,4] RAM only; excludes Python, proofs, banks and allocator."""
    _require(set(scoring_centers_by_parent) == set(VAL), "All original eight VAL parents")
    _require(all(type(p) is int and 0 < p <= 20000
                 for p in scoring_centers_by_parent.values()), "Complete bounded original grids")
    return CHECKPOINT_FILES*16*686*sum(scoring_centers_by_parent.values())


def bounds_from_counts_v1(native_nodes, native_holds, sites, grid_centers, *, haplotypes=952):
    """Named raw tables and planning estimates, never an RSS/VRAM admission proof."""
    n,t,s,p,h = native_nodes,native_holds,sites,grid_centers,haplotypes
    _require(all(type(x) is int and x > 0 for x in (n,t,s,p,h))
             and n <= 10000000 and t <= 10000 and s <= 2000000
             and p <= 20000 and h == 952, "Explicit original complete preparation bounds")
    q = 686
    original_extra = 2*n+20*p*q+36*p-28
    flag_plan = 2058*n+128*h
    original_preparation = original_extra+flag_plan+96*s+128*p*32+512*8192*32
    cut_plan = 89*t*h+h
    cut_preparation_estimate = 128*n+512*t*h+512*h+64*1024**2
    return dict(
        native_nodes=n,native_holds=t,haplotypes=h,sites=s,grid_centers=p,queries=q,
        full_grid_CPU_features_bytes=20*p*q,
        original_grid_extra_retained_tensor_upper_bytes=original_extra,
        cut_plan_tensor_upper_bytes=cut_plan,
        extra_retained_tensor_upper_bytes=original_extra+cut_plan,
        GPU_persistent_grid_upper_bytes=original_extra-20*p*q+cut_plan,
        one_GPU_feature_query_chunk_bytes=4*p*32*5,
        named_preparation_allowance_bytes=original_preparation+cut_preparation_estimate,
        inherited_grid_preparation_allowance_bytes=original_preparation,
        additional_cut_preparation_work_estimate_bytes=cut_preparation_estimate,
        cut_recipe_JSON_planning_allowance_bytes=65536+256*t,
        total_recipe_JSON_planning_allowance_bytes=131072+65536+256*t,
        one_rank8_FP32_hold_leaf_bank_bytes=32*t*h,
        both_direction_rank8_banks_bytes=64*t*h,
        rank8_compact_forcing_upper_bytes=32*t*h,
        compact32channel_injection_upper_bytes=128*t*h,
        preparation_estimate_is_not_an_RSS_bound=True,
        allocator_and_full_model_peak_unqualified=True,
        backbone_marker_chain_optimizer_autograd_and_workspaces_included=False)


def parent_bounds_v1(bank, positions):
    _require(tuple(bank.sample_index.shape)[1:] == (952,)
             and tuple(bank.query_column.shape) == (686,), "Complete original panel")
    return bounds_from_counts_v1(int(bank.parent.numel()), len(bank.spans),
                                int(bank.genotype.shape[0]), len(positions))


def prepare_parent_v1(original_h044_bundle, bin_edges_bp, *,
                      max_nodes=10000000, max_holds=10000):
    """Cache-miss hook. Prepare once on CPU before phase truth; share across arms."""
    from arggnn.marker_grid_reset_chain_gnn_v1 import prepare_marker_grid_reset_v1
    grid = prepare_marker_grid_reset_v1(original_h044_bundle, bin_edges_bp)
    return _model().prepare_order2_bundle_v1(
        grid, max_nodes=max_nodes, max_holds=max_holds)


def extra_bytes_v1(bundle):
    """Matches original engine accounting: grid/flags/cut, excluding markers/bank."""
    bundle.validate()
    return bundle.tensor_bytes()


def _state_hash_v1(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()+b"\0"+str(a.dtype).encode()+b"\0")
        digest.update(str(a.shape).encode()+b"\0"+a.tobytes())
    return digest.hexdigest()


def build_family_for_engine_v1(payload, device, guard, *, checkpoint_ref, completion_ref):
    """Authenticated payload is supplied by root's scientific reader."""
    import torch
    guard()
    models, old_optimizers, proof = _model().make_matched_family_v1(
        payload, checkpoint_ref=checkpoint_ref, completion_ref=completion_ref,
        device="cpu", **READOUT)
    del old_optimizers
    hashes = {a: _state_hash_v1(models[a]) for a in ARMS}
    _require(set(models) == set(ARMS) and len(set(hashes.values())) == 1,
             "All three initial state dictionaries are identical")
    for arm in ARMS:
        model = models[arm]
        _require(sum(p.numel() for p in model.parameters()) == PARAMETERS_PER_ARM
                 and all(p.requires_grad and p.dtype == torch.float32
                         for p in model.parameters()), "Full FP32 model is trainable")
        guard()
        model.to(device=device, dtype=torch.float32)
    for left, right in combinations(ARMS, 2):
        _require(not ({p.data_ptr() for p in models[left].parameters()}
                      & {p.data_ptr() for p in models[right].parameters()}),
                 "All three arm pairs have disjoint learned storage")
    optimizers = {
        a: torch.optim.AdamW(models[a].parameters(), lr=.001, weight_decay=.0001,
                             betas=(.9,.999), eps=1e-8) for a in ARMS}
    _require(all(not opt.state for opt in optimizers.values()), "All fresh AdamW states empty")
    proof = dict(proof, three_pairwise_storage_disjoint=True,
                 all_three_initial_state_hashes_equal=True)
    return models, optimizers, proof, hashes


def evaluation_model_v1(arm, device):
    import torch
    _require(arm in ARMS, "One registered memory arm")
    return (_model().Order2CutMarkerGridV1(arm=arm, **READOUT)
            .to(device=device, dtype=torch.float32).eval().requires_grad_(False))


def paired_effects_v1(full):
    """Descriptive selected-arm effects; root owns preregistered pass/fail criteria."""
    _require(set(full) == set(ARMS), "All three selected diagnostics")
    comparisons = {}
    for control in ARMS[1:]:
        rows = []
        for pid in VAL:
            cells = {}
            for arm in (ARMS[0], control):
                matches = [r for r in full[arm]["cells"] if r["parent_id"] == pid]
                _require(len(matches) == 1, "Exactly one original same-parent cell")
                cells[arm] = matches[0]["groups"]["primary_present_day_182"]
            true, null = cells[ARMS[0]], cells[control]
            tf, nf = true["boundaries"]["f1"], null["boundaries"]["f1"]
            rows.append(dict(parent_id=pid,
                accuracy_delta=true["accuracy"]-null["accuracy"],
                boundary_F1_delta=None if tf is None or nf is None else tf-nf))
        ad = [r["accuracy_delta"] for r in rows]
        bd = [r["boundary_F1_delta"] for r in rows if r["boundary_F1_delta"] is not None]
        comparisons["triple_minus_"+control] = dict(
            per_parent_paired_differences=rows,
            mean_Present182_accuracy_delta=statistics.mean(ad),
            paired_accuracy_sample_SD_percentagepoints=100*statistics.stdev(ad),
            mean_1kb_boundary_F1_delta=statistics.mean(bd) if len(bd)==8 else None,
            paired_boundary_F1_sample_SD_percentagepoints=(
                100*statistics.stdev(bd) if len(bd)==8 else None))
    return dict(comparisons=comparisons, threshold_decision=None,
                threshold_registration_required=True, one_optimizer_seed=True,
                reused8VAL_development_not_independent_confirmation=True,
                no_priority_information_dominance_or_biological_rate_claim=True)
