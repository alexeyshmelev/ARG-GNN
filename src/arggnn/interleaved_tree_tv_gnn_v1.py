"""H033 confidence-conditioned tree/TV composition on observed native trees.

Two matched utility arms, (BA)^3 and B^3 A^3. Tree elimination, Gaussian
conditioning, TV and operator splitting are established. Intermediate class
fields are learned predictor surrogates, not ancestry truth at each node age.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .persistence_trace_kernel_v1 import _tree
from .reference_logcavity_gnn_v1 import (
    prepare_reference_logcavity_bank_v1, unit_class_fields_v1,
)

ARMS = ("interleaved", "blocked")
TV_ITERATIONS = 8
TV_STEP = .25
INTERIOR_ABSORPTION = .01


@dataclass(frozen=True)
class TreeSolveHoldV1:
    parent: Tensor
    roots: Tensor
    query_nodes: Tensor             # distinct ACTUAL native sample rows
    reference_contrast: Tensor     # local H030 unit source contrasts [N,8]
    source_mask: Tensor            # one at genuine source placement, zero elsewhere
    conductance: Tensor            # child-indexed symmetric fixed edge weights
    weighted_degree: Tensor
    upward_levels: tuple
    upward_children: tuple
    downward_levels: tuple
    readout_indexes: Tensor

    def to(self, device):
        simple = ("parent", "roots", "query_nodes", "reference_contrast",
                  "source_mask", "conductance", "weighted_degree", "readout_indexes")
        sequences = ("upward_levels", "upward_children", "downward_levels")
        return replace(self, **{k: getattr(self, k).to(device) for k in simple},
            **{k: tuple(x.to(device) for x in getattr(self, k)) for k in sequences})


@dataclass(frozen=True)
class PackedTreeBlockV1:
    parent: Tensor
    roots: Tensor
    query_nodes: Tensor
    source_mask: Tensor
    conductance: Tensor
    weighted_degree: Tensor
    upward_levels: tuple
    upward_children: tuple
    downward_levels: tuple
    readout_indexes: Tensor
    origin_node_indexes: Tensor

    def to(self, device):
        simple = ("parent", "roots", "query_nodes", "source_mask", "conductance",
                  "weighted_degree", "readout_indexes", "origin_node_indexes")
        sequences = ("upward_levels", "upward_children", "downward_levels")
        return replace(self, **{k: getattr(self, k).to(device) for k in simple},
            **{k: tuple(x.to(device) for x in getattr(self, k)) for k in sequences})


@dataclass(frozen=True)
class InterleavedTreeTVBankV1:
    blocks: tuple
    reference_contrast: Tensor
    tv_weights: Tensor              # min(1,100/physical consecutive-bin gap)
    age_groups: tuple
    readout_count: int
    query_count: int
    metadata: dict

    def to(self, device):
        return replace(self, blocks=tuple(x.to(device) for x in self.blocks),
            reference_contrast=self.reference_contrast.to(device),
            tv_weights=self.tv_weights.to(device),
            age_groups=tuple(x.to(device) for x in self.age_groups))


def _levels_v1(parent):
    children = [[] for _ in parent]
    for child, ancestor in enumerate(parent):
        if ancestor >= 0:
            children[ancestor].append(child)
    pending = [len(x) for x in children]
    current = [i for i, count in enumerate(pending) if not count]
    upward, gathered = [], []
    seen = 0
    while current:
        upward.append(torch.tensor(current, dtype=torch.long))
        gathered.append(torch.tensor([c for n in current for c in children[n]], dtype=torch.long))
        seen += len(current)
        next_level = []
        for child in current:
            ancestor = parent[child]
            if ancestor >= 0:
                pending[ancestor] -= 1
                if not pending[ancestor]:
                    next_level.append(ancestor)
        current = next_level
    if seen != len(parent):
        raise ValueError("Complete acyclic native forest required")
    roots = [i for i, ancestor in enumerate(parent) if ancestor < 0]
    current, downward = roots, []
    while current:
        current = [c for n in current for c in children[n]]
        if current:
            downward.append(torch.tensor(current, dtype=torch.long))
    return torch.tensor(roots, dtype=torch.long), tuple(upward), tuple(gathered), tuple(downward)


def prepare_interleaved_tree_tv_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, *, census=350.,
        sample_identity_namespace, **geometry):
    """Reuse immutable H030 source/cut policy, then select actual Q sample rows.

    This preparation has no query ancestry/target input. Native nodes and cut
    edges all survive, including holds without a 100bp-center readout.
    """
    base = prepare_reference_logcavity_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, census=census,
        sample_identity_namespace=sample_identity_namespace, **geometry)
    if (readout_x_bp.numel() > 2500 or not bool((readout_x_bp[1:] > readout_x_bp[:-1]).all())):
        raise ValueError("Ordered original readouts within one existing 250kb tile required")
    fixed_coefficients = torch.tensor([1., .1], dtype=torch.float32)
    holds, offsets, offset, maximum = [], [], 0, 0
    empty_families = 0
    for index, tree in enumerate(trees):
        samples, _, _ = _tree(tree)
        native_parent, times = tree.parent.tolist(), tree.node_times.tolist()
        n = len(native_parent) + sum(ancestor >= 0 and times[child] < census < times[ancestor]
                                    for child, ancestor in enumerate(native_parent))
        parent = base.forest.parent[offset:offset+n].clone()
        active = parent >= 0
        parent[active] -= offset
        roots, upward, children, downward = _levels_v1(parent.tolist())
        actual_queries = torch.tensor([samples[int(q)] for q in query_ids], dtype=torch.long)
        if actual_queries.unique().numel() != query_ids.numel():
            raise ValueError("Emitter evidence must lift to distinct actual native Q samples")
        source = base.log_source[offset:offset+n]
        fields, supported = unit_class_fields_v1(source)
        contrast = (fields-base.class_prior[None]) * supported.repeat_interleave(4, -1)
        empty_families += int((~supported).sum())
        source_mask = torch.isfinite(source).any(-1).float()
        if bool(source_mask[actual_queries].any()):
            raise ValueError("Actual Q sample nodes and genuine reference source roles must be disjoint")
        degree = base.forest.child_count[offset:offset+n] + active.float()
        cost = (base.edge_basis[offset:offset+n] * fixed_coefficients).sum(-1)
        affinity = torch.exp(-cost).mean(-1)
        conductance = torch.zeros(n, dtype=torch.float32)
        edges = active.nonzero().flatten()
        conductance[edges] = affinity[edges] / torch.maximum(degree[edges], degree[parent[edges]])
        weighted_degree = torch.zeros(n, dtype=torch.float32)
        weighted_degree.index_add_(0, edges, conductance[edges])
        weighted_degree.index_add_(0, parent[edges], conductance[edges])
        if (not bool(torch.isfinite(contrast).all()) or bool((conductance < 0).any())
                or bool((weighted_degree > 1.+1e-6).any())):
            raise ValueError("Finite source contrasts and weighted degree at most one required")
        readouts = (base.forest.readout_hold == index).nonzero().flatten()
        offsets.append(offset)
        holds.append(TreeSolveHoldV1(parent, roots, actual_queries, contrast, source_mask, conductance,
            weighted_degree, upward, children, downward, readouts))
        offset += n
        maximum = max(maximum, n)
    if offset != len(base.forest.parent):
        raise ValueError("H033 packed rows differ from the complete H030 native/cut forest")
    gap = (readout_x_bp[1:] - readout_x_bp[:-1]).float()
    weights = torch.minimum(torch.ones_like(gap), 100./gap)
    metadata = dict(base.metadata, kind="interleaved_tree_tv_actual_sample_evidence",
        max_hold_vertices=maximum, readouts=len(readout_x_bp), query_count=len(query_ids),
        query_readout="unique_actual_native_sample_node_not_census_group",
        reference_RHS="local_H030_masked_unit_contrasts;younger350_older_actual",
        fixed_age_coefficients=[[1., .1], [1., .1]], learned_age_cost_parameters=0,
        conductance="mean_fixed_family_exp_negative_cost_div_max_endpoint_degree",
        maximum_weighted_degree=max(float(h.weighted_degree.max()) for h in holds),
        local_empty_family_rows=empty_families, graph_state="learned_predictor_surrogate",
        fixed_interior_absorption=INTERIOR_ABSORPTION, Q_source_roles_disjoint=True,
        source_precision="binary_genuine_placement_mask_not_reference_multiplicity",
        K_spectral_bounds=[INTERIOR_ABSORPTION, INTERIOR_ABSORPTION+4.],
        population_or_query_ancestry_truth_inputs=False, recurrent_paths_executed=False,
        TV_iterations=TV_ITERATIONS, TV_step=TV_STEP, tile_boundary_carry=False)
    blocks = _pack_readout_forests_v1(holds, offsets, base.forest.readout_hold)
    metadata.update(packed_block_count=len(blocks), readout_block_size=32,
        replicated_vertices=sum(len(b.parent) for b in blocks),
        maximum_packed_vertices=max(len(b.parent) for b in blocks),
        packed_forest="one_genuine_observed_full_tree_copy_per_retained_readout;no_padding")
    return InterleavedTreeTVBankV1(blocks, torch.cat([h.reference_contrast for h in holds]),
        weights, base.forest.age_groups, len(readout_x_bp), len(query_ids), metadata)


def _pack_readout_forests_v1(holds, offsets, readout_hold):
    """Ragged block-diagonal copies, one genuine tree per original readout.

    Origin indexes map repeated reference projections back to shared node rows.
    Query positions flatten readout-major, preserving actual Q identities.
    No padding, fabricated samples or cross-copy graph edges are introduced.
    """
    blocks = []
    for indexes in torch.arange(len(readout_hold), dtype=torch.long).split(32):
        chosen = [int(readout_hold[b]) for b in indexes]
        parts = {k: [] for k in ("parent", "roots", "query_nodes", "source_mask",
            "conductance", "weighted_degree", "origin_node_indexes")}
        levels = {k: [] for k in ("upward_levels", "upward_children", "downward_levels")}
        cursor = 0
        for h in chosen:
            hold = holds[h]
            parent = hold.parent.clone()
            valid = parent >= 0
            parent[valid] += cursor
            parts["parent"].append(parent)
            parts["roots"].append(hold.roots+cursor)
            parts["query_nodes"].append(hold.query_nodes+cursor)
            for key in ("source_mask", "conductance", "weighted_degree"):
                parts[key].append(getattr(hold, key))
            parts["origin_node_indexes"].append(torch.arange(len(parent), dtype=torch.long)+offsets[h])
            for key in levels:
                for level, nodes in enumerate(getattr(hold, key)):
                    if level == len(levels[key]):
                        levels[key].append([])
                    levels[key][level].append(nodes+cursor)
            cursor += len(parent)
        fields = {k: torch.cat(v) for k, v in parts.items()}
        fields.update({k: tuple(torch.cat(v) for v in values) for k, values in levels.items()})
        blocks.append(PackedTreeBlockV1(**fields, readout_indexes=indexes))
    return tuple(blocks)


def _eliminate_v1(hold, tau, strength, rhs):
    """FP32 SPD tree solve, in-place inside custom autograd/no-grad only.

    d_i = epsilon + source_mask_i + tau_i(on Q) + lambda*degree_i.
    Eliminate c->p: d_p -= e_c^2/d_c; h_p += e_c*h_c/d_c.
    Substitute root z=h/d, then z_c=(h_c+e_c*z_p)/d_c.
    """
    batch, n, _ = rhs.shape
    precision = (INTERIOR_ABSORPTION+hold.source_mask+
                 strength*hold.weighted_degree).expand(batch, n).clone()
    precision[:, hold.query_nodes] += tau
    message = rhs.clone()
    edge = strength*hold.conductance
    for nodes, children in zip(hold.upward_levels, hold.upward_children):
        if children.numel():
            ancestors = hold.parent[children]
            inverse = precision[:, children].reciprocal()
            precision.index_add_(1, ancestors, -edge[children].square()[None]*inverse)
            message.index_add_(1, ancestors,
                (edge[children][None]*inverse)[..., None]*message[:, children])
    result = torch.zeros_like(message)
    result[:, hold.roots] = message[:, hold.roots]/precision[:, hold.roots, None]
    for nodes in hold.downward_levels:
        result[:, nodes] = (message[:, nodes] +
            edge[nodes][None, :, None]*result[:, hold.parent[nodes]]) / precision[:, nodes, None]
    return result


class _TreeStepV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, reference, tau, strength, hold):
        rhs = (hold.source_mask[:, None]*reference)[None].expand(len(x), -1, -1).clone()
        rhs[:, hold.query_nodes] += tau[..., None]*x
        z = _eliminate_v1(hold, tau, strength, rhs)
        ctx.hold = hold
        ctx.save_for_backward(x, tau, strength, z)
        return z[:, hold.query_nodes]

    @staticmethod
    def backward(ctx, grad_output):
        x, tau, strength, z = ctx.saved_tensors
        hold = ctx.hold
        rhs = torch.zeros_like(z)
        rhs[:, hold.query_nodes] = grad_output
        v = _eliminate_v1(hold, tau, strength, rhs)
        vq, zq = v[:, hold.query_nodes], z[:, hold.query_nodes]
        grad_x = tau[..., None]*vq
        grad_reference = hold.source_mask[:, None]*v.sum(0)
        grad_tau = (vq*(x-zq)).sum(-1)
        edges = (hold.parent >= 0).nonzero().flatten()
        parents = hold.parent[edges]
        grad_strength = -(hold.conductance[edges][None, :, None] *
            (v[:, edges]-v[:, parents])*(z[:, edges]-z[:, parents])).sum()
        return grad_x, grad_reference, grad_tau, grad_strength, None


def tree_step_v1(x, reference, tau, strength, hold):
    return _TreeStepV1.apply(x, reference, tau, strength, hold)


def packed_tree_step_v1(x, reference, tau, strength, block):
    # Gathering inside checkpoint avoids retaining repeated r fields per block.
    value = tree_step_v1(x.reshape(1, -1, 4), reference[block.origin_node_indexes],
        tau.reshape(1, -1), strength, block)
    return value.reshape_as(x)


def _difference_transpose_v1(dual):
    # D x = x[1:]-x[:-1], so D^T p = [-p0,p0-p1,...,p_last].
    return torch.cat((-dual[:1], dual[:-1]-dual[1:], dual[-1:]), 0)


def tv_step_v1(x, weights, strength):
    """Exactly eight projected-dual updates, not an exact TV proximal claim.

    p0=0; yk=x-D^T pk; p{k+1}=clip(pk+.25*Dyk, +/-lambda*w).
    y8=x-D^T p8. Unweighted D has ||D||^2<=4, so .25 is safe;
    physical-distance weights enter the dual box, not the operator.
    """
    if len(x) <= 1:
        return x
    bound = strength*weights[:, None, None]
    dual = torch.zeros_like(x[:-1])
    for _ in range(TV_ITERATIONS):
        value = x-_difference_transpose_v1(dual)
        candidate = dual+TV_STEP*(value[1:]-value[:-1])
        dual = torch.maximum(-bound, torch.minimum(bound, candidate))
    return x-_difference_transpose_v1(dual)


class InterleavedTreeTVGNNV1(nn.Module):
    def __init__(self, *, initial_tree_strength=.1, initial_tv_strength=.1,
                 readout_block_size=32, checkpoint_tree=True):
        super().__init__()
        if (not 0 < initial_tree_strength < 1 or not 0 < initial_tv_strength < 1
                or readout_block_size != 32 or checkpoint_tree is not True):
            raise ValueError("Bounded strengths and declared32-readout checkpointing required")
        self.reference_projection = nn.Linear(8, 4, bias=False)
        self.context_head = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(self.context_head.weight)
        self.precision_coefficients = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.raw_tree_strength = nn.Parameter(torch.tensor(
            math.log(initial_tree_strength/(1.-initial_tree_strength)), dtype=torch.float32))
        self.raw_tv_strength = nn.Parameter(torch.tensor(
            math.log(initial_tv_strength/(1.-initial_tv_strength)), dtype=torch.float32))
        self.readout_block_size = readout_block_size
        self.checkpoint_tree = checkpoint_tree

    def parameter_counts(self, control):
        if control not in ARMS:
            raise ValueError("Only two matched H033 arms exist")
        count = sum(p.numel() for p in self.parameters())
        if count != 52:
            raise ValueError("H033 allocated parameter count changed")
        return dict(total=count, active_parameter_blocks=count,
            reference_projection=32, context_head=16, precision=2, bounded_strengths=2,
            known_zero_older_ANA_reference_weights=4, source_contrast_rank_upper_bound=5,
            known_reference_family_mean_input_nullity=True, realized_effective_rank_measured=False,
            same_allocated_capacity=True, same_effective_capacity_claimed=False)

    def tree_operator(self, bank, x, tau, strength):
        reference = self.reference_projection(bank.reference_contrast)
        values = []
        for block in bank.blocks:
            indexes = block.readout_indexes
            if self.training and torch.is_grad_enabled() and self.checkpoint_tree:
                value = checkpoint(packed_tree_step_v1, x[indexes], reference, tau[indexes],
                    strength, block, use_reentrant=False, preserve_rng_state=False)
            else:
                value = packed_tree_step_v1(x[indexes], reference, tau[indexes], strength, block)
            values.append(value)
        # Blocks follow original readout order, with no missing/padded rows.
        return torch.cat(values, 0)

    def forward(self, bank, frozen_baseline_logits, *, control):
        if (control not in ARMS or frozen_baseline_logits.dtype != torch.float32
                or frozen_baseline_logits.shape != (bank.readout_count, bank.query_count, 4)
                or frozen_baseline_logits.device != self.raw_tree_strength.device
                or bank.tv_weights.device != frozen_baseline_logits.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Finite original FP32 H025 logits and matched H033 bank required")
        baseline = frozen_baseline_logits.detach()
        x0 = baseline-baseline.mean(-1, keepdim=True)
        top = baseline.topk(2, dim=-1).values
        margin = (top[..., 0]-top[..., 1]).detach()
        tau = 1.+torch.sigmoid(self.precision_coefficients[0]+self.precision_coefficients[1]*margin)
        tree_strength, tv_strength = self.raw_tree_strength.sigmoid(), self.raw_tv_strength.sigmoid()
        x = x0
        if control == "interleaved":
            for _ in range(3):
                x = tv_step_v1(self.tree_operator(bank, x, tau, tree_strength),
                    bank.tv_weights, tv_strength)
        else:
            for _ in range(3):
                x = self.tree_operator(bank, x, tau, tree_strength)
            for _ in range(3):
                x = tv_step_v1(x, bank.tv_weights, tv_strength)
        delta = x-x0
        centered = delta.clone()
        for group in bank.age_groups:
            centered[:, group] = delta[:, group]-delta[:, group].mean(1, keepdim=True)
        result = baseline+self.context_head(centered)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite H033 class-valued correction")
        self.last_diagnostics = dict(condition=control,
            tree_strength=float(tree_strength.detach()), TV_strength=float(tv_strength.detach()),
            precision_coefficients=self.precision_coefficients.detach().cpu().tolist(),
            precision_min=float(tau.detach().min()), precision_max=float(tau.detach().max()),
            centered_correction_l2=float(centered.detach().norm()), tree_calls=3,
            TV_calls=3, TV_dual_updates=24, readout_block_size=self.readout_block_size,
            max_hold_vertices=bank.metadata["max_hold_vertices"],
            maximum_packed_vertices=bank.metadata["maximum_packed_vertices"],
            replicated_vertices=bank.metadata["replicated_vertices"],
            local_confidence_calibration_possible=True, graph_state_is_predictor_surrogate=True,
            recurrent_paths_executed=False, tile_boundary_carry=False)
        return result
