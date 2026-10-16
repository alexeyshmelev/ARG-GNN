"""Unregistered same-marker proposal, isolated SOURCE-ONLY review artifact.

No file reader, truth input, training loop, evaluator, resource logic or CLI.
Authoring did not import/execute this module. All numerical behavior is untested.
Fixed source/age/allele bookkeeping is distinct from the optional learned gate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.aggr import MaxAggregation, SumAggregation

GROUPS, ALLELES, BLOCK_CAP = 8, 4, 16
LOG_GAMMA_MIN, LOG_GAMMA_MAX = -16.0, -0.0001
EMPTY_SENTINEL = -1.0e30


def _require(ok, text):
    if not ok:
        raise ValueError(text)


def _cpu_integer(x, name):
    _require(isinstance(x, Tensor) and x.device.type == "cpu"
             and x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
             and not x.requires_grad, name + " requires observed CPU integers")
    return x.to(dtype=torch.long)


@dataclass(frozen=True)
class HoldLayoutV1:
    """One complete observed tree. Integer IDs are joins, never embeddings."""
    native_node: Tensor             # [V], local row -> original forest node
    parent: Tensor                  # [V], local parent, root=-1
    sample_node: Tensor             # [H], persistent sample-column order
    query_column: Tensor            # [Q], actual unknown query columns
    query_node: Tensor              # [Q], actual query sampled nodes
    reference_column: Tensor        # [R], pure observed references only
    reference_node: Tensor          # [R]
    reference_group: Tensor         # [R], 4*(age>=350)+class
    reference_group_onehot: Tensor  # [R,8], fixed FP32
    route_permutation: Tensor       # [H], target node <- genotype source column
    up_levels: tuple[Tensor, ...]
    up_children: tuple[Tensor, ...]
    up_parent_rank: tuple[Tensor, ...]
    children: tuple[Tensor, ...]    # local child list for every original node
    nonleaf_preorder: tuple[int, ...]
    site_start: int                 # original contiguous site membership
    site_stop: int

    @property
    def nodes(self):
        return self.parent.numel()

    @property
    def haplotypes(self):
        return self.sample_node.numel()

    def to(self, device):
        scalar = ("native_node", "parent", "sample_node", "query_column", "query_node",
                  "reference_column", "reference_node", "reference_group",
                  "reference_group_onehot", "route_permutation")
        sequences = ("up_levels", "up_children", "up_parent_rank", "children")
        changes = {k: getattr(self, k).to(device) for k in scalar}
        changes.update({k: tuple(x.to(device) for x in getattr(self, k)) for k in sequences})
        return replace(self, **changes)


def build_alignment_null_v1(reference_class, sample_age, diploid_id, phase, *, seed):
    """CPU observation-only complete-diploid permutation, constant across SNPs.

    diploid_id is an observed join, not a learned feature. Each diploid has
    phases0/1 and agrees in observed class and younger/older age family.
    Unknown/query diploids are fixed. No random reseeding to improve movement.
    """
    rc = _cpu_integer(reference_class, "reference_class")
    did, ph = _cpu_integer(diploid_id, "diploid_id"), _cpu_integer(phase, "phase")
    _require(sample_age.device.type == "cpu" and not sample_age.requires_grad
             and rc.ndim == 1 and did.shape == ph.shape == sample_age.shape == rc.shape
             and bool(torch.isfinite(sample_age).all()) and bool((sample_age >= 0).all())
             and bool(((rc >= -1) & (rc < 4)).all()) and type(seed) is int,
             "Explicit observed source/date/diploid roster required")
    by_id = {}
    for col, (d, p) in enumerate(zip(did.tolist(), ph.tolist())):
        _require(p in (0, 1) and p not in by_id.setdefault(d, {}), "Unique phases0/1 required")
        by_id[d][p] = col
    strata = {}
    for d, pair in by_id.items():
        _require(set(pair) == {0, 1}, "Every diploid requires both observed haplotypes")
        i, j = pair[0], pair[1]
        ci, cj = int(rc[i]), int(rc[j])
        ai, aj = int(sample_age[i] >= 350), int(sample_age[j] >= 350)
        _require(ci == cj and ai == aj, "Diploid source/age-family metadata disagree")
        if ci >= 0:
            strata.setdefault((ci, ai), []).append(d)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.arange(rc.numel(), dtype=torch.long)
    summary = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        order = torch.randperm(len(ids), generator=gen).tolist()
        moved = 0
        for at, other in enumerate(order):
            moved += int(at != other)
            for p in (0, 1):
                perm[by_id[ids[at]][p]] = by_id[ids[other]][p]
        summary.append(dict(source=key[0], older_family=key[1],
                            diploids=len(ids), moved_diploids=moved))
    return perm, dict(seed=seed, strata=summary,
        constant_across_sites=True, unknown_columns_fixed=True,
        complete_diploid_sequences_preserved=True,
        exact_dates_within_age_family_not_preserved=True,
        called_moved_mass_not_computed=True)


def prepare_parent_layouts_v1(parent, sample_index, site_hold, reference_class,
                             sample_age, query_column, genotype, *,
                             route_permutation=None, allele_count: Optional[Tensor] = None):
    """CPU observation-only indexing/validation, no scientific file reads.

    Supply fields from the exact verified MutationRoutedWholeTreeInputsV1 bank.
    Optional allele_count comes from the original observed coordinate sidecar.
    The caller separately certifies observed-label provenance and full-panel IDs.
    """
    p = _cpu_integer(parent, "parent")
    samples = _cpu_integer(sample_index, "sample_index")
    sh = _cpu_integer(site_hold, "site_hold")
    rc = _cpu_integer(reference_class, "reference_class")
    qc = _cpu_integer(query_column, "query_column")
    gt = _cpu_integer(genotype, "genotype")
    _require(p.ndim == 1 and 0 < p.numel() < 2**31 and samples.ndim == 2,
             "Bounded original forest and complete sample rows required")
    n, (t, h) = p.numel(), samples.shape
    _require(t > 0 and h > 1 and rc.shape == sample_age.shape == (h,)
             and sample_age.device.type == "cpu" and not sample_age.requires_grad
             and bool(torch.isfinite(sample_age).all()) and bool((sample_age >= 0).all())
             and qc.ndim == sh.ndim == 1 and qc.numel() > 0
             and qc.unique().numel() == qc.numel()
             and bool(((qc >= 0) & (qc < h)).all())
             and bool(((rc >= -1) & (rc < 4)).all())
             and bool((rc[qc] == -1).all()),
             "Known proxy metadata only; query source classes MUST be unknown")
    _require(gt.ndim == 2 and gt.shape == (sh.numel(), h)
             and bool(((gt >= -1) & (gt < ALLELES)).all())
             and bool(((sh >= 0) & (sh < t)).all())
             and bool((sh[1:] >= sh[:-1]).all())
             and bool(((p >= -1) & (p < n)).all())
             and bool(((samples >= 0) & (samples < n)).all()), "Observed index/allele bounds differ")
    if allele_count is not None:
        ac = _cpu_integer(allele_count, "allele_count")
        _require(ac.shape == sh.shape and bool(((ac >= 1) & (ac <= ALLELES)).all())
                 and bool(((gt < ac[:, None]) | (gt == -1)).all()),
                 "Original per-site allele cardinalities disagree")
    groups = 4 * (sample_age >= 350).long() + rc
    refs = torch.nonzero(rc >= 0, as_tuple=False).flatten()
    perm = torch.arange(h) if route_permutation is None else _cpu_integer(route_permutation, "route_permutation")
    _require(perm.shape == (h,) and torch.equal(perm.sort().values, torch.arange(h))
             and torch.equal(perm[rc < 0], torch.arange(h)[rc < 0])
             and torch.equal(groups[perm[refs]], groups[refs]),
             "Only within-source/age-family reference permutations permitted")
    # Native trees are disjoint. Every root is traversed exactly once, and no
    # anonymous ancestor identity is joined across holds.
    pv = p.tolist()
    kids = [[] for _ in range(n)]
    roots = []
    for v, par in enumerate(pv):
        if par < 0:
            roots.append(v)
        else:
            _require(par != v, "Self-parent forbidden")
            kids[par].append(v)
    _require(len(roots) == t, "Exactly one complete tree per hold required")
    root_of, trees = [-1] * n, {}
    for root in roots:
        preorder, stack = [], [root]
        while stack:
            v = stack.pop()
            _require(root_of[v] < 0, "Cycle or duplicate native traversal")
            root_of[v] = root
            preorder.append(v)
            stack.extend(reversed(kids[v]))
        trees[root] = preorder
    _require(all(r >= 0 for r in root_of), "Cyclic or omitted native forest vertices")
    sizes = torch.bincount(sh, minlength=t).tolist()
    start, used, layouts = 0, set(), []
    for hold, row in enumerate(samples.tolist()):
        root = root_of[row[0]]
        _require(root not in used and len(set(row)) == h and all(root_of[v] == root for v in row),
                 "Unique full sampled panel in each disjoint tree required")
        used.add(root)
        native = trees[root]
        local = {v: at for at, v in enumerate(native)}
        _require({v for v in native if not kids[v]} == set(row),
                 "All and only original sampled leaves required; sampled interiors unsupported")
        local_children = [torch.tensor([local[c] for c in kids[v]], dtype=torch.long) for v in native]
        height, by_height = [0] * len(native), {}
        for v in reversed(native):
            at = local[v]
            height[at] = max((height[local[c]] + 1 for c in kids[v]), default=0)
            by_height.setdefault(height[at], []).append(at)
        levels, children, ranks = [], [], []
        for depth in sorted(by_height):
            ns = by_height[depth]
            cs, rs = [], []
            for rank, at in enumerate(ns):
                cs.extend(local_children[at].tolist())
                rs.extend([rank] * local_children[at].numel())
            levels.append(torch.tensor(ns, dtype=torch.long))
            children.append(torch.tensor(cs, dtype=torch.long))
            ranks.append(torch.tensor(rs, dtype=torch.long))
        sn = torch.tensor([local[v] for v in row], dtype=torch.long)
        rg = groups[refs]
        layouts.append(HoldLayoutV1(
            native_node=torch.tensor(native, dtype=torch.long),
            parent=torch.tensor([-1 if pv[v] < 0 else local[pv[v]] for v in native]),
            sample_node=sn, query_column=qc.clone(), query_node=sn[qc],
            reference_column=refs.clone(), reference_node=sn[refs], reference_group=rg,
            reference_group_onehot=torch.nn.functional.one_hot(rg, GROUPS).float(),
            route_permutation=perm.clone(), up_levels=tuple(levels),
            up_children=tuple(children), up_parent_rank=tuple(ranks),
            children=tuple(local_children),
            nonleaf_preorder=tuple(at for at, v in enumerate(native) if kids[v]),
            site_start=start, site_stop=start + sizes[hold]))
        start += sizes[hold]
    return tuple(layouts)


def _safe_logaddexp(a, b):
    active = torch.isfinite(a) | torch.isfinite(b)
    # Avoid evaluating logaddexp(-inf,-inf), whose gradient is undefined.
    aa, bb = torch.where(active, a, torch.zeros_like(a)), torch.where(active, b, torch.zeros_like(b))
    value = torch.logaddexp(aa, bb)
    return torch.where(active, value, torch.full_like(value, -torch.inf))


def _safe_logsumexp(x, dim):
    active = torch.isfinite(x).any(dim=dim, keepdim=True)
    value = torch.logsumexp(torch.where(active, x, torch.zeros_like(x)), dim=dim)
    return torch.where(active.squeeze(dim), value, torch.full_like(value, -torch.inf))


def _safe_logcumsumexp(x):
    finite = torch.isfinite(x)
    active = finite.long().cumsum(0) > 0
    # V<2**31 and effective edge log gamma>=-16 ensure every real path
    # exceeds -3.5e10, far above the sentinel. Empty prefixes stay masked.
    safe = torch.where(finite, x, torch.full_like(x, EMPTY_SENTINEL))
    value = torch.logcumsumexp(safe, dim=0)
    return torch.where(active, value, torch.full_like(value, -torch.inf))


class SameMarkerPathEvidenceV1(nn.Module):
    """Known positive tree-path aggregation plus fixed same-marker lookup.

    No trainable parameters. External log_gamma may be fixed or differentiable.
    All-path weighting is established message passing, not a novelty claim.
    """
    def __init__(self, marker_block_size=16):
        super().__init__()
        _require(type(marker_block_size) is int and 1 <= marker_block_size <= BLOCK_CAP,
                 "Explicit marker block in1..16 required")
        self.marker_block_size = marker_block_size
        self.sum_aggr, self.max_aggr = SumAggregation(), MaxAggregation()

    def _segment_logsumexp(self, values, index, size):
        if values.shape[0] == 0:
            return values.new_full((size, values.shape[1]), -torch.inf)
        finite = torch.isfinite(values)
        count = self.sum_aggr(finite.to(values.dtype), index=index, dim_size=size, dim=0)
        maximum = self.max_aggr(values, index=index, dim_size=size, dim=0)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        shifted = torch.where(finite, values - maximum.index_select(0, index), torch.zeros_like(values))
        total = self.sum_aggr(torch.where(finite, shifted.exp(), torch.zeros_like(values)),
                              index=index, dim_size=size, dim=0)
        value = maximum + total.clamp_min(torch.finfo(values.dtype).tiny).log()
        return torch.where(count > 0, value, torch.full_like(value, -torch.inf))

    def _block_statistics(self, log_gamma, genotype, layout):
        b, h = genotype.shape
        n, q = layout.nodes, layout.query_column.numel()
        # Exactly32 channels/site, independent of panel size and number of SNPs.
        assigned = genotype.index_select(1, layout.route_permutation)
        reference = assigned.index_select(1, layout.reference_column)
        rh = torch.nn.functional.one_hot(reference.long().clamp_min(0), ALLELES).bool()
        rh = rh & (reference >= 0)[..., None]
        group = layout.reference_group_onehot.bool()
        seed_active = rh[:, :, None, :] & group[None, :, :, None]
        ref_seed = torch.where(seed_active, log_gamma.new_zeros(()),
                               log_gamma.new_full((), -torch.inf))
        f = b * GROUPS * ALLELES
        seed = log_gamma.new_full((n, f), -torch.inf)
        seed.index_copy_(0, layout.reference_node, ref_seed.permute(1, 0, 2, 3).reshape(-1, f))
        up = log_gamma.new_full((n, f), -torch.inf)
        for nodes, ch, rank in zip(layout.up_levels, layout.up_children, layout.up_parent_rank):
            values = up.index_select(0, ch) + log_gamma.index_select(0, ch)[:, None]
            aggregate = self._segment_logsumexp(values, rank, nodes.numel())
            value = _safe_logaddexp(seed.index_select(0, nodes), aggregate)
            # Each original node is written once. Gathered child tensors are
            # independent copies, as in the existing complete-tree sweep.
            up.index_copy_(0, nodes, value)
        outside = log_gamma.new_full((n, f), -torch.inf)
        for node in layout.nonleaf_preorder:
            ch = layout.children[node]
            message = up.index_select(0, ch) + log_gamma.index_select(0, ch)[:, None]
            prefix = _safe_logcumsumexp(message)
            suffix = _safe_logcumsumexp(message.flip(0)).flip(0)
            empty = message.new_full((1, f), -torch.inf)
            left = torch.cat((empty, prefix[:-1]), dim=0)
            right = torch.cat((suffix[1:], empty), dim=0)
            # Clone avoids saving a view of a buffer subsequently written at
            # descendant rows. No inherited history or gradient is detached.
            local = _safe_logaddexp(seed[node], outside[node].clone())
            cavity = _safe_logaddexp(_safe_logaddexp(left, right), local[None, :])
            outside.index_copy_(0, ch, cavity + log_gamma.index_select(0, ch)[:, None])
        log_n = outside.index_select(0, layout.query_node).reshape(q, b, GROUPS, ALLELES).permute(1, 0, 2, 3)
        log_d = _safe_logsumexp(log_n, -1)
        query = genotype.index_select(1, layout.query_column).long()
        allele = query.clamp_min(0)[:, :, None, None].expand(b, q, GROUPS, 1)
        log_match = log_n.gather(-1, allele).squeeze(-1)
        # Counts use this route only; the original backbone remains unchanged.
        counts = torch.einsum("rg,bra->bga", layout.reference_group_onehot, rh.to(log_gamma.dtype))
        called = counts.sum(-1)
        valid = (query >= 0)[:, :, None] & (called > 0)[:, None, :]
        safe_d = torch.where(torch.isfinite(log_d), log_d, torch.zeros_like(log_d))
        weighted = (log_match - safe_d).exp()
        frequency = counts / called.clamp_min(1)[..., None]
        ordinary = frequency[:, None].expand(b, q, GROUPS, ALLELES).gather(-1, allele).squeeze(-1)
        contrast = torch.where(valid, weighted - ordinary, torch.zeros_like(weighted))
        relative_log_mass = (safe_d - called.clamp_min(1).log()[:, None, :]).clamp_max(0)
        relative_log_mass = torch.where(valid, relative_log_mass, torch.zeros_like(relative_log_mass))
        features = torch.cat((contrast, relative_log_mass, valid.to(log_gamma.dtype)), -1)
        mask = torch.cat((valid, valid, (query >= 0)[:, :, None].expand_as(valid)), -1)
        sums = torch.where(mask, features, torch.zeros_like(features)).sum(0)
        maxima = torch.where(mask, features, torch.full_like(features, -torch.inf)).amax(0)
        return sums, maxima, mask.to(log_gamma.dtype).sum(0)

    def forward(self, layout, genotype, log_gamma):
        """Return pooled `[Q,48]`: mean24 then max24 over original hold SNPs.

        genotype is ONLY this hold `[S_hold,H]`; -1 means missing. Graph and
        group layouts were checked on CPU before transfer. In training, every
        differentiable marker block is checkpointed with all gradients retained.
        """
        _require(isinstance(layout, HoldLayoutV1) and genotype.ndim == 2
                 and genotype.shape == (layout.site_stop-layout.site_start, layout.haplotypes)
                 and genotype.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
                 and log_gamma.shape == (layout.nodes,) and log_gamma.dtype == torch.float32
                 and genotype.device == log_gamma.device == layout.parent.device,
                 "Exact one-hold same-device FP32 observation contract required")
        # This explicit numerical restriction is part of SPEC_FIX.md.
        effective = log_gamma.clamp(LOG_GAMMA_MIN, LOG_GAMMA_MAX)
        q = layout.query_column.numel()
        sums, counts = effective.new_zeros((q, 24)), effective.new_zeros((q, 24))
        maxima = effective.new_full((q, 24), -torch.inf)
        for start in range(0, genotype.shape[0], self.marker_block_size):
            block = genotype[start:start+self.marker_block_size]
            if torch.is_grad_enabled() and effective.requires_grad:
                values = checkpoint(lambda lg, gt: self._block_statistics(lg, gt, layout),
                                    effective, block, use_reentrant=False,
                                    preserve_rng_state=False)
            else:
                values = self._block_statistics(effective, block, layout)
            add, high, number = values
            sums, maxima, counts = sums + add, torch.maximum(maxima, high), counts + number
        mean = sums / counts.clamp_min(1)
        high = torch.where(counts > 0, maxima, torch.zeros_like(maxima))
        return torch.cat((mean, high), -1)


class BoundedEdgeAttenuationV1(nn.Module):
    """UNPROVEN learned gate, separate from fixed evidence engineering.

    Existing9 geometry/support channels + child/parent first upward states.
    Root is not an edge and its value is never consumed. All unary edges remain.
    """
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(9 + 32 + 32, 1)
        nn.init.zeros_(self.projection.weight)
        desired = -math.log(0.95)
        fraction = (desired + LOG_GAMMA_MAX) / (-LOG_GAMMA_MIN + LOG_GAMMA_MAX)
        nn.init.constant_(self.projection.bias, math.log(fraction / (1-fraction)))
        self.float()

    def forward(self, edge9, upward, parent):
        _require(edge9.shape == (parent.numel(), 9) and upward.shape == (parent.numel(), 32)
                 and edge9.dtype == upward.dtype == torch.float32,
                 "Original FP32 edge9 and first upward node states required")
        parent_state = upward.index_select(0, parent.clamp_min(0))
        score = self.projection(torch.cat((edge9, upward, parent_state), -1)).squeeze(-1)
        return LOG_GAMMA_MAX + (LOG_GAMMA_MIN-LOG_GAMMA_MAX) * score.sigmoid()


class SameMarkerAlleleSupportAdapterV1(nn.Module):
    """Optional714-parameter extension; no model/checkpoint loader or runner.

    Gate74 + rank projection384 + zero-output projection256. Analytic count,
    not an instantiated-model measurement. Returns query-node residuals only.
    """
    def __init__(self, marker_block_size=16):
        super().__init__()
        self.gate = BoundedEdgeAttenuationV1()
        self.evidence = SameMarkerPathEvidenceV1(marker_block_size)
        self.input = nn.Linear(48, 8, bias=False)
        self.output = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.output.weight)
        self.float()

    def forward(self, layout, genotype, edge9, first_upward):
        log_gamma = self.gate(edge9, first_upward, layout.parent)
        fixed_summary = self.evidence(layout, genotype, log_gamma)
        return self.output(self.input(fixed_summary).tanh())
