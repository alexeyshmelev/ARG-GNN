"""H030 unit-scale learned reference evidence from log-domain tree kernels.

Positive metric affinities, sum-product/rerooting and temporal carrier recurrence
are established machinery. These are reference mixtures, not Felsenstein
substitution likelihoods, ancestry posteriors or causal recombination effects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn

from .edge_innovation_gnn_v1 import (
    EdgeInnovationGNNV1, EdgeInnovationBankV1, prepare_edge_innovation_bank_v1,
)
from .persistence_trace_kernel_v1 import _tree

ARMS = ("logcavity_only", "static", "carrier_gru", "carrier_matrix", "sham_ancestor")


def masked_logaddexp_v1(a: Tensor, b: Tensor) -> Tensor:
    """Explicitly empty sums remain -inf with finite zero empty-support VJPs."""
    available = torch.isfinite(a) | torch.isfinite(b)
    aa = torch.where(available, a, torch.zeros_like(a))
    bb = torch.where(available, b, torch.zeros_like(b))
    value = torch.logaddexp(aa, bb)
    return torch.where(available, value, torch.full_like(value, -torch.inf))


def masked_segment_logsumexp_v1(value: Tensor, index: Tensor, segments: int) -> Tensor:
    """Stable reduction with a detached numerical shift and no tiny-mass divide."""
    shape = (segments, value.shape[1])
    expanded = index[:, None].expand_as(value)
    maxima = value.new_full(shape, -torch.inf)
    maxima.scatter_reduce_(0, expanded, value.detach(), reduce="amax", include_self=True)
    available = torch.isfinite(maxima)
    shift = torch.where(available, maxima, torch.zeros_like(maxima))
    term = (value - shift[index]).exp()
    total = value.new_zeros(shape).index_add(0, index, term)
    safe_total = torch.where(available, total, torch.ones_like(total))
    result = shift + safe_total.log()
    return torch.where(available, result, torch.full_like(result, -torch.inf))


def unit_class_fields_v1(log_evidence: Tensor):
    """Normalize each observed age family over available classes in log space.

    Empty family values are zero and their mask is false. An absent class remains
    exactly zero. Logsumexp's all-empty branch is replaced before differentiation.
    """
    value = log_evidence.reshape(-1, 2, 4)
    classes = torch.isfinite(value)
    family = classes.any(-1)
    safe = torch.where(family[..., None], value, torch.zeros_like(value))
    # Normalize relative logs directly. Subtracting an absolute log normalizer
    # from values around -1000 loses FP32 precision even when logsumexp is stable.
    shift = safe.detach().max(-1, keepdim=True).values
    relative = safe - shift
    normalizer = torch.logsumexp(relative, -1, keepdim=True)
    fields = torch.where(classes, (relative - normalizer).exp(), torch.zeros_like(safe))
    return fields.reshape(-1, 8), family


@dataclass(frozen=True)
class ReferenceLogCavityBankV1:
    forest: EdgeInnovationBankV1
    log_source: Tensor
    edge_basis: Tensor           # [N,2families,2basis], additive potential differences
    class_prior: Tensor          # [8], uniform only over represented classes per family
    children_by_rank: tuple      # fixed sibling ordering, never a learned identity
    metadata: dict

    def to(self, device):
        return replace(self, forest=self.forest.to(device),
            log_source=self.log_source.to(device), edge_basis=self.edge_basis.to(device),
            class_prior=self.class_prior.to(device),
            children_by_rank=tuple(x.to(device) for x in self.children_by_rank))


def prepare_reference_logcavity_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, *, census=350.,
        sample_identity_namespace, young_scale=350., old_scale=1400., **geometry):
    """Target-free adapter retaining the pinned H029 native/cut/episode geometry."""
    if census != 350. or young_scale <= 0 or old_scale <= 0:
        raise ValueError("Declared census350 and positive reference-family scales required")
    forest = prepare_edge_innovation_bank_v1(trees, intervals_bp, readout_x_bp,
        query_ids, reference_ids, reference_classes, census=census,
        sample_identity_namespace=sample_identity_namespace, **geometry)
    n = len(forest.parent)
    log_source = torch.full((n, 8), -torch.inf, dtype=torch.float32)
    ages = []
    counts = [torch.tensor(forest.metadata["younger_class_counts"]),
              torch.tensor(forest.metadata["older_class_counts"])]
    offset = 0
    for tree in trees:
        samples, _, _ = _tree(tree)
        p = tree.parent.tolist(); t = tree.node_times.tolist()
        native_p, native_t = p.copy(), t.copy()
        for child, ancestor in enumerate(native_p):
            if ancestor >= 0 and native_t[child] < census < native_t[ancestor]:
                point = len(p)
                p.append(ancestor); t.append(census); p[child] = point
        for identity, cls in zip(reference_ids.tolist(), reference_classes.tolist()):
            node = samples[identity]
            cohort = int(t[node] > census)
            if cohort == 0:
                while t[node] < census:
                    node = p[node]
                    if node < 0:
                        raise ValueError("Younger reference has no valid observed census cut")
                if t[node] != census:
                    raise ValueError("Younger proxy must use its analytic census point")
            # Older proxy remains on its exact supplied native sampling row.
            weight = torch.tensor(-math.log(int(counts[cohort][cls])), dtype=torch.float32)
            log_source[offset + node, cohort * 4 + cls] = torch.logaddexp(
                log_source[offset + node, cohort * 4 + cls], weight)
        ages.extend(t); offset += len(p)
    if offset != n:
        raise ValueError("Log source rows differ from complete native/cut forest rows")
    time = torch.tensor(ages, dtype=torch.float64)
    ancestor = forest.parent.clamp_min(0)
    above = (time - census).clamp_min(0)
    roots = forest.parent < 0
    basis = []
    for scale in (young_scale, old_scale):
        potential = torch.stack((above / scale, torch.log1p(above / scale)), -1)
        delta = potential[ancestor] - potential
        delta[roots] = 0
        if bool((delta < 0).any()):
            raise ValueError("Positive age potential differences required")
        basis.append(delta)
    edge_basis = torch.stack(basis, 1).float()
    # Below350 changes give a common reference-distance factor and are
    # unidentifiable after normalization. We omit that rate rather than fit it.
    children = [[] for _ in range(n)]
    for child, parent in enumerate(forest.parent.tolist()):
        if parent >= 0:
            children[parent].append(child)
    sibling = tuple(torch.tensor([cs[rank] for cs in children if len(cs) > rank])
                    for rank in range(max(map(len, children))))
    prior = torch.cat([(c > 0).float() / (c > 0).sum() for c in counts])
    metadata = dict(forest.metadata,
        kind="reference_logcavity_whole_native_forest_unit_kernel_fields",
        source="observed_R96_proxy_log_weight_minus_log_family_class_count",
        field_normalization="logsumexp_over_available_reference_classes_separately_per_age_family",
        empty_support="explicit_minus_inf_log_evidence_and_zero_fields_false_masks",
        older_absent_classes="no_source_no_pseudocount_no_floor",
        positive_edge_cost="softplus_shared_family_coefficients_times_additive_clipped_age_potential_differences",
        below350_kernel_rate="omitted_unidentifiable_common_factor",
        kernel_subdivision_invariant=True, nonlinear_GNN_subdivision_invariant=False,
        young_scale=young_scale, old_scale=old_scale,
        max_sibling_rank=len(sibling), causal_edit_attribution=False)
    return ReferenceLogCavityBankV1(forest, log_source, edge_basis, prior, sibling, metadata)


class ReferenceLogCavityGNNV1(EdgeInnovationGNNV1):
    """Exact reference-mixture cavities feed learned whole-tree value propagation."""
    def __init__(self, *, initial_linear_cost=1., initial_log_cost=.1, **kwargs):
        super().__init__(**kwargs)
        if initial_linear_cost <= 0 or initial_log_cost <= 0:
            raise ValueError("Positive shared reference-age potential coefficients required")
        initial = torch.tensor([[initial_linear_cost, initial_log_cost]] * 2)
        self.raw_cost = nn.Parameter(torch.log(torch.expm1(initial)))
        self.kernel_head = nn.Linear(8, 4, bias=False)
        nn.init.zeros_(self.kernel_head.weight)

    def parameter_counts(self, control):
        if control not in ARMS:
            raise ValueError("Unknown H030 arm")
        def active(name):
            if control == "logcavity_only":
                return name == "raw_cost" or name.startswith("kernel_head.")
            if name.startswith("kernel_head."):
                return False
            if control != "carrier_gru" and name.startswith("carrier_gru."):
                return False
            if control == "static" and (name == "raw_ell" or name.startswith(
                    ("value.", "key.", "retention.", "write_gate.", "decoder_edge."))):
                return False
            return True
        blocks = sum(p.numel() for name,p in self.named_parameters() if active(name))
        zero_head = 2 * self.hidden_dim * 4 if control == "static" else 0
        return dict(total=sum(p.numel() for p in self.parameters()),
            active=blocks-zero_head, active_parameter_blocks=blocks,
            zero_memory_stream_head_scalars=zero_head)

    def log_cavities(self, bank):
        forest = bank.forest
        cost = (bank.edge_basis * torch.nn.functional.softplus(self.raw_cost)[None]).sum(-1)
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("Nonfinite positive reference edge costs")
        cost = cost.repeat_interleave(4, -1)
        n = len(forest.parent)
        upward = bank.log_source.new_full((n, 8), -torch.inf)
        for nodes, children, ranks in zip(forest.upward_levels, forest.upward_children, forest.upward_ranks):
            child_sum = masked_segment_logsumexp_v1(
                upward[children] - cost[children], ranks, len(nodes))
            value = masked_logaddexp_v1(bank.log_source[nodes], child_sum)
            upward.index_copy_(0, nodes, value)
        incoming = upward - cost
        # Leave-one-child-out uses prefix/suffix log sums, never total-minus-child
        # positive mass subtraction. Arbitrary polytomies and empty classes work.
        prefix = incoming.new_full((n,8), -torch.inf)
        accumulator = incoming.new_full((n,8), -torch.inf)
        for children in bank.children_by_rank:
            parents = forest.parent[children]
            old = accumulator[parents]
            prefix.index_copy_(0, children, old)
            accumulator.index_copy_(0, parents, masked_logaddexp_v1(old, incoming[children]))
        siblings = incoming.new_full((n,8), -torch.inf)
        accumulator = incoming.new_full((n,8), -torch.inf)
        for children in reversed(bank.children_by_rank):
            parents = forest.parent[children]
            old = accumulator[parents]
            siblings.index_copy_(0, children, masked_logaddexp_v1(prefix[children], old))
            accumulator.index_copy_(0, parents, masked_logaddexp_v1(old, incoming[children]))
        outside = incoming.new_full((n,8), -torch.inf)
        for nodes in forest.downward_levels:
            parents = forest.parent[nodes]
            cavity = masked_logaddexp_v1(
                masked_logaddexp_v1(outside[parents], bank.log_source[parents]), siblings[nodes])
            outside.index_copy_(0, nodes, cavity - cost[nodes])
        return upward, outside

    def unit_contrasts(self, bank, log_evidence):
        fields, supported = unit_class_fields_v1(log_evidence)
        contrast = (fields - bank.class_prior[None]) * supported.repeat_interleave(4, -1)
        return contrast, fields, supported

    def encode_units(self, bank, log_up, log_out):
        forest = bank.forest
        up_input, _, _ = self.unit_contrasts(bank, log_up)
        out_input, _, _ = self.unit_contrasts(bank, log_out)
        local_input, _, _ = self.unit_contrasts(bank, bank.log_source)
        base = self.reference_projection(up_input)
        local = self.reference_projection(local_input)
        n,d = len(forest.parent),self.hidden_dim
        up = base.new_zeros(n,d); message_sum = base.new_zeros(n,d); messages = base.new_zeros(n,d)
        ug = self.up_gate(forest.geometry).sigmoid()
        dg = self.down_gate(forest.geometry).sigmoid()
        for nodes, children, ranks in zip(forest.upward_levels,forest.upward_children,forest.upward_ranks):
            acc = base.new_zeros(len(nodes),d)
            if children.numel():
                value = self.up_value(up[children]) * ug[children]
                value = value / forest.child_count[forest.parent[children]][:,None]
                messages.index_copy_(0,children,value);acc.index_add_(0,ranks,value)
            message_sum.index_copy_(0,nodes,acc)
            up.index_copy_(0,nodes,(base[nodes]+acc).tanh())
        outside = base.new_zeros(n,d)
        for nodes in forest.downward_levels:
            parents = forest.parent[nodes]
            # Parent local values include only its own reference sources. Using
            # parent subtree-normalized base here would leak the excluded child.
            cavity = outside[parents] + local[parents] + message_sum[parents] - messages[nodes]
            value = self.reference_projection(out_input[nodes]) + self.down_value(cavity)*dg[nodes]
            outside.index_copy_(0,nodes,value.tanh())
        return up,outside

    def forward(self, bank, frozen_baseline_logits, *, control):
        if control not in ARMS or bank.log_source.dtype != torch.float32:
            raise ValueError("Matched H030 FP32 source bank required")
        forest = bank.forest
        if (frozen_baseline_logits.dtype != torch.float32 or
                frozen_baseline_logits.shape != (len(forest.readout_hold),forest.query_nodes.shape[1],4)
                or frozen_baseline_logits.device != self.raw_cost.device
                or bank.log_source.device != self.raw_cost.device
                or not bool(torch.isfinite(frozen_baseline_logits).all())):
            raise ValueError("Original finite FP32 frozen H025 support required")
        log_up,log_out = self.log_cavities(bank)
        full = masked_logaddexp_v1(log_up,log_out)
        contrast, fields, supported = self.unit_contrasts(bank,full)
        readout = forest.query_nodes[forest.readout_hold]
        if control == "logcavity_only":
            context = contrast[readout]
            head = self.kernel_head
            forward = reverse = None
        else:
            up,outside = self.encode_units(bank,log_up,log_out)
            empty = up.new_zeros(len(up),self.rank)
            forward = empty if control == "static" else self.scan(
                forest,up,outside,forest.forward_frames,control)
            reverse = empty if control == "static" else self.scan(
                forest,up,outside,forest.reverse_frames,control)
            decoded = self.decode(forest,up,outside,forward,reverse)
            context = decoded[readout]
            if control != "static":
                context = context * torch.stack((torch.ones_like(forest.forward_distance),
                    torch.exp(-forest.forward_distance/self.ell_bp()),
                    torch.exp(-forest.reverse_distance/self.ell_bp())),-1)[:,None,:,None]
            context = context.flatten(2)
            head = self.context_head
        centered = context.clone()
        for group in forest.age_groups:
            centered[:,group]=context[:,group]-context[:,group].mean(1,keepdim=True)
        result=frozen_baseline_logits.detach()+head(centered)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite H030 reference-only whole-tree residual")
        self.last_diagnostics=dict(positive_cost_coefficients=torch.nn.functional.softplus(
            self.raw_cost).detach().cpu().tolist(),unit_field_l2=float(fields.detach().norm()),
            centered_context_l2=float(centered.detach().norm()),
            empty_family_fields=int((~supported).sum()),
            forward_read_l2=0. if forward is None else float(forward.detach().norm()),
            reverse_read_l2=0. if reverse is None else float(reverse.detach().norm()))
        return result
