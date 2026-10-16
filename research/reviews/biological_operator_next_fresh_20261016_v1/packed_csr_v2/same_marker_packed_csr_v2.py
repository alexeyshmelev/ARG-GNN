"""SOURCE-ONLY packed successor for the EXISTING unregistered same-marker proposal.

No scientific imports/execution occurred during authoring. No file reader,
truth input, experiment, device probe, optimizer or launcher is provided.
The old source artifact stays preserved. No full-parent GPU layout .to exists.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, fields
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.aggr import SumAggregation, MaxAggregation

G, A, BLOCK_CAP = 8, 4, 16
LOG_MIN, LOG_MAX = -16., -.0001


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cpu_index(x, name):
    require(isinstance(x, Tensor) and x.device.type == "cpu" and x.ndim >= 1
            and x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
            and not x.requires_grad, "Observed CPU integer tensor required: " + name)
    return x


def i32(values):
    return torch.tensor(values, dtype=torch.int32).contiguous()


def stamp(x):
    return (x.data_ptr(), x._version, tuple(x.shape), x.dtype, x.device.type)


@dataclass(frozen=True)
class PackedHoldDeviceV2:
    """Transient ONE-HOLD contiguous indexes, constructed inside checkpoint."""
    native_node: Tensor
    parent: Tensor
    query_column: Tensor
    query_node: Tensor
    reference_column: Tensor
    reference_node: Tensor
    reference_group: Tensor
    group_onehot: Tensor
    route_permutation: Tensor
    level_node: Tensor
    level_child: Tensor
    level_parent_rank: Tensor
    # node/edge start-stop and max degree: immutable CPU metadata, no GPU .item.
    level_slices: tuple[tuple[int, int, int, int, int], ...]
    nodes: int
    haplotypes: int


@dataclass(frozen=True)
class PackedHoldCPUV2:
    """One complete observed tree; no tensor per individual graph node."""
    native_node: Tensor              # [V], original global forest row
    parent: Tensor                   # [V], local parent, root=-1
    sample_node: Tensor              # [H], original sampled-column order
    query_column: Tensor             # [Q]
    query_node: Tensor               # [Q], local
    global_query_node: Tensor        # [Q], original forest row
    reference_column: Tensor         # [R], pure references only
    reference_node: Tensor           # [R], local
    reference_group: Tensor          # [R], younger0..3 then older4..7
    route_permutation: Tensor        # [H], reference-only assigned sequence join
    children_ptr: Tensor             # [V+1], ordinary child CSR (CPU proof only)
    children_index: Tensor           # [V-1], ordinary child CSR (CPU proof only)
    level_node: Tensor               # [V], height-grouped nodes
    level_child: Tensor              # [V-1], grouped by height then recipient
    level_parent_rank: Tensor        # [V-1], recipient rank within height group
    level_slices: tuple[tuple[int, int, int, int, int], ...]
    site_start: int
    site_stop: int
    immutable_stamps: tuple

    @property
    def nodes(self):
        return self.parent.numel()

    @property
    def haplotypes(self):
        return self.sample_node.numel()

    def verify_immutable(self):
        now = tuple((f.name, stamp(getattr(self, f.name))) for f in fields(self)
                    if isinstance(getattr(self, f.name), Tensor))
        require(now == self.immutable_stamps, "CPU packed observation layout mutated")

    def _transient_device_view(self, device):
        """Internal: call only INSIDE the whole-hold checkpointed function.

        Eleven contiguous index transfers per hold, independent of V. Child CSR
        and sampled rows stay on CPU; no device tensor is cached on this object.
        """
        self.verify_immutable()
        names = ("native_node", "parent", "query_column", "query_node",
                 "reference_column", "reference_node", "reference_group",
                 "route_permutation", "level_node", "level_child", "level_parent_rank")
        vectors = {k: getattr(self, k).to(device=device, dtype=torch.long) for k in names}
        group = torch.nn.functional.one_hot(vectors["reference_group"], G).float()
        return PackedHoldDeviceV2(**vectors, group_onehot=group,
            level_slices=self.level_slices, nodes=self.nodes, haplotypes=self.haplotypes)


@dataclass(frozen=True)
class PackedParentCPUV2:
    """CPU-only metadata for ONE original parent, with a counted numeric cap."""
    holds: tuple[PackedHoldCPUV2, ...]
    global_query_node: Tensor
    numeric_tensor_bytes: int
    tensor_object_count: int
    source_stamps: tuple
    preparation_named_array_allowance_bytes: int
    python_scratch_or_RSS_measured: bool = False

    def verify_for_supplied_cpu_bank(self, **observations):
        now = tuple((key, stamp(value)) for key, value in sorted(observations.items()))
        require(now == self.source_stamps, "Original CPU observation field stamps differ")
        for hold in self.holds:
            hold.verify_immutable()


def prepare_packed_parent_v2(*, parent, sample_index, site_hold, reference_class,
                            sample_age, query_column, genotype, route_permutation=None,
                            allele_count=None, max_layout_tensor_bytes=134217728,
                            max_prepare_named_array_bytes=536870912):
    """Future CPU observation-only preparation, no filesystem/payload reader.

    Every returned index buffer is contiguous int32. Temporary global child CSR
    uses Python array buffers, NOT one Torch tensor or Python child list per node.
    At most one parent should be cached by a future root-controlled caller.
    Integer/source purity/provenance certificates remain external root duties.
    """
    inputs = dict(parent=parent, sample_index=sample_index, site_hold=site_hold,
                  reference_class=reference_class, sample_age=sample_age,
                  query_column=query_column, genotype=genotype)
    for key in ("parent", "sample_index", "site_hold", "reference_class", "query_column", "genotype"):
        cpu_index(inputs[key], key)
    require(parent.ndim == 1 and 0 < parent.numel() < 2**31 and sample_index.ndim == 2,
            "Bounded complete original CPU forest required")
    n, (t, h) = parent.numel(), sample_index.shape
    require(t > 0 and 1 < h < 2**31 and sample_age.shape == reference_class.shape == (h,)
            and sample_age.device.type == "cpu" and not sample_age.requires_grad
            and bool(torch.isfinite(sample_age).all()) and bool((sample_age >= 0).all())
            and query_column.ndim == site_hold.ndim == 1 and query_column.numel() > 0
            and query_column.unique().numel() == query_column.numel()
            and bool(((query_column >= 0) & (query_column < h)).all())
            and bool(((reference_class >= -1) & (reference_class < 4)).all())
            and bool((reference_class[query_column.long()] == -1).all()),
            "Known-reference labels only, with all query classes unknown")
    require(genotype.shape == (site_hold.numel(), h)
            and bool(((genotype >= -1) & (genotype < A)).all())
            and bool(((site_hold >= 0) & (site_hold < t)).all())
            and bool((site_hold[1:] >= site_hold[:-1]).all())
            and bool(((parent >= -1) & (parent < n)).all())
            and bool(((sample_index >= 0) & (sample_index < n)).all()),
            "Observed original index/categorical bounds differ")
    if allele_count is not None:
        cpu_index(allele_count, "allele_count")
        require(allele_count.shape == site_hold.shape
                and bool(((allele_count >= 1) & (allele_count <= A)).all())
                and bool(((genotype < allele_count[:, None]) | (genotype == -1)).all()),
                "Original observed cardinality differs")
        inputs["allele_count"] = allele_count
    groups = 4 * (sample_age >= 350).long() + reference_class.long()
    refs = torch.nonzero(reference_class >= 0, as_tuple=False).flatten().tolist()
    perm = torch.arange(h) if route_permutation is None else cpu_index(route_permutation, "route_permutation")
    require(perm.shape == (h,) and torch.equal(perm.long().sort().values, torch.arange(h))
            and torch.equal(perm.long()[reference_class < 0], torch.arange(h)[reference_class < 0])
            and torch.equal(groups[perm.long()[refs]], groups[refs]),
            "Reference-only within-source/age-family route permutation required")
    if route_permutation is not None:
        inputs["route_permutation"] = route_permutation
    # Named numeric scratch only. Python integer/list overhead is explicitly NOT
    # included or represented as a measured RSS certificate.
    allowance = 80 * n + 32 * t * h
    require(type(max_prepare_named_array_bytes) is int and allowance <= max_prepare_named_array_bytes,
            "One-parent named preparation allowance exceeds explicit cap")
    pv = parent.tolist()  # one global integer list, O(N), released after preparation
    child_count = array("i", [0]) * n
    for node, p in enumerate(pv):
        require(p != node, "Self-parent forbidden")
        if p >= 0:
            child_count[p] += 1
    ptr = array("i", [0]) * (n + 1)
    for node in range(n):
        ptr[node + 1] = ptr[node] + child_count[node]
    require(ptr[-1] < 2**31, "Bounded original edge count required")
    child_index = array("i", [0]) * ptr[-1]
    cursor = array("i", ptr[:-1])
    for node, p in enumerate(pv):
        if p >= 0:
            child_index[cursor[p]] = node
            cursor[p] += 1
    del cursor, child_count
    visited = bytearray(n)
    root_used, holds, total_payload, total_objects = set(), [], 0, 0
    starts, site_counts = 0, torch.bincount(site_hold.long(), minlength=t).tolist()
    query = query_column.tolist()
    for hold_number, row in enumerate(sample_index.tolist()):
        root, steps = row[0], 0
        while pv[root] >= 0:
            root, steps = pv[root], steps + 1
            require(steps <= n, "Cyclic parent chain")
        require(root not in root_used and len(set(row)) == h,
                "Each hold requires a distinct original tree and unique samples")
        root_used.add(root)
        native, stack = [], [root]
        while stack:
            node = stack.pop()
            require(not visited[node], "Duplicate/cyclic original tree traversal")
            visited[node] = 1
            native.append(node)
            stack.extend(reversed(child_index[ptr[node]:ptr[node+1]]))
        local = {node: at for at, node in enumerate(native)}
        require({node for node in native if ptr[node] == ptr[node+1]} == set(row),
                "All and only original sampled leaves required, including every vertex")
        v = len(native)
        lp, flat_children, child_ptr, height = [], [], [0], [0] * v
        for node in native:
            lp.append(-1 if pv[node] < 0 else local[pv[node]])
            flat_children.extend(local[c] for c in child_index[ptr[node]:ptr[node+1]])
            child_ptr.append(len(flat_children))
        by_height = {}
        for node in reversed(native):
            at = local[node]
            height[at] = max((height[local[c]] + 1
                              for c in child_index[ptr[node]:ptr[node+1]]), default=0)
            by_height.setdefault(height[at], []).append(at)
        level_node, level_child, rank, slices = [], [], [], []
        for depth in sorted(by_height):
            nodes = by_height[depth]
            ns, es, degree = len(level_node), len(level_child), 0
            level_node.extend(nodes)
            for recipient, at in enumerate(nodes):
                children = flat_children[child_ptr[at]:child_ptr[at+1]]
                level_child.extend(children)
                rank.extend([recipient] * len(children))
                degree = max(degree, len(children))
            slices.append((ns, len(level_node), es, len(level_child), degree))
        sn = [local[node] for node in row]
        data = dict(native_node=i32(native), parent=i32(lp), sample_node=i32(sn),
            query_column=i32(query), query_node=i32([sn[c] for c in query]),
            global_query_node=i32([row[c] for c in query]), reference_column=i32(refs),
            reference_node=i32([sn[c] for c in refs]), reference_group=i32(groups[refs].tolist()),
            route_permutation=i32(perm.tolist()), children_ptr=i32(child_ptr),
            children_index=i32(flat_children), level_node=i32(level_node),
            level_child=i32(level_child), level_parent_rank=i32(rank))
        numeric = sum(x.numel() * x.element_size() for x in data.values())
        total_payload += numeric
        total_objects += len(data)
        require(total_payload <= max_layout_tensor_bytes, "Packed one-parent numeric layout cap exceeded")
        proof = tuple((key, stamp(value)) for key, value in data.items())
        holds.append(PackedHoldCPUV2(**data, level_slices=tuple(slices),
            site_start=starts, site_stop=starts + site_counts[hold_number], immutable_stamps=proof))
        starts += site_counts[hold_number]
    require(all(visited) and len(root_used) == t, "All original forest vertices must be represented once")
    all_query = torch.cat([hold.global_query_node for hold in holds]).contiguous()
    total_payload += all_query.numel() * all_query.element_size()
    require(total_payload <= max_layout_tensor_bytes, "Complete query-row join exceeds packed cap")
    return PackedParentCPUV2(tuple(holds), all_query, total_payload, total_objects + 1,
        tuple((key, stamp(value)) for key, value in sorted(inputs.items())), allowance)


def safe_logadd(a, b):
    active = torch.isfinite(a) | torch.isfinite(b)
    aa = torch.where(active, a, torch.zeros_like(a))
    bb = torch.where(active, b, torch.zeros_like(b))
    value = torch.logaddexp(aa, bb)
    return torch.where(active, value, torch.full_like(value, -torch.inf))


def safe_logsum(x, dim):
    active = torch.isfinite(x).any(dim=dim, keepdim=True)
    value = torch.logsumexp(torch.where(active, x, torch.zeros_like(x)), dim=dim)
    return torch.where(active.squeeze(dim), value, torch.full_like(value, -torch.inf))


def segmented_inclusive_logprefix(x, group, max_degree):
    """Contiguous-group parallel scan; no subtraction or tensor per node.

    Hillis-Steele doubles the covered within-group prefix. Cost O(E*F*log D)
    for max degree D, working storage O(E*F). Real-path sums never use a finite
    sentinel here: safe_logadd handles all-empty pairs directly.
    """
    result, stride = x, 1
    while stride < max_degree:
        if stride >= x.shape[0]:
            break
        same = (group[stride:] == group[:-stride])[:, None]
        merged = safe_logadd(result[stride:], result[:-stride])
        result = torch.cat((result[:stride], torch.where(same, merged, result[stride:])), 0)
        stride *= 2
    return result


def segmented_exclusive_logs(x, group, max_degree):
    """Return sibling mass before/after each child, with empty-group masks."""
    prefix = segmented_inclusive_logprefix(x, group, max_degree)
    suffix = segmented_inclusive_logprefix(x.flip(0), group.flip(0), max_degree).flip(0)
    empty = x.new_full((1, x.shape[1]), -torch.inf)
    adjacent = group[1:] == group[:-1]
    left_mask = torch.cat((torch.zeros(1, dtype=torch.bool, device=group.device), adjacent))[:, None]
    right_mask = torch.cat((adjacent, torch.zeros(1, dtype=torch.bool, device=group.device)))[:, None]
    left = torch.where(left_mask, torch.cat((empty, prefix[:-1]), 0), -torch.inf)
    right = torch.where(right_mask, torch.cat((suffix[1:], empty), 0), -torch.inf)
    return left, right


class PackedSameMarkerEvidenceV2(nn.Module):
    """Parameter-free evidence maths, using known Torch/PyG aggregations."""
    def __init__(self, marker_block_size=16):
        super().__init__()
        require(type(marker_block_size) is int and 1 <= marker_block_size <= BLOCK_CAP,
                "Marker block cap1..16 required")
        self.marker_block_size = marker_block_size
        self.sum_aggr, self.max_aggr = SumAggregation(), MaxAggregation()

    def segment_logsum(self, values, index, size):
        if values.shape[0] == 0:
            return values.new_full((size, values.shape[1]), -torch.inf)
        finite = torch.isfinite(values)
        count = self.sum_aggr(finite.to(values.dtype), index=index, dim_size=size, dim=0)
        maximum = self.max_aggr(values, index=index, dim_size=size, dim=0)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        shift = torch.where(finite, values - maximum.index_select(0, index), torch.zeros_like(values))
        total = self.sum_aggr(torch.where(finite, shift.exp(), torch.zeros_like(values)),
                              index=index, dim_size=size, dim=0)
        value = maximum + total.clamp_min(torch.finfo(values.dtype).tiny).log()
        return torch.where(count > 0, value, torch.full_like(value, -torch.inf))

    def _block_evidence(self, log_gamma, genotype, view):
        b, h = genotype.shape
        n, q = view.nodes, view.query_column.numel()
        assigned = genotype.index_select(1, view.route_permutation)
        reference = assigned.index_select(1, view.reference_column)
        allele_hot = torch.nn.functional.one_hot(reference.long().clamp_min(0), A).bool()
        allele_hot = allele_hot & (reference >= 0)[..., None]
        active = allele_hot[:, :, None, :] & view.group_onehot.bool()[None, :, :, None]
        f = b * G * A
        ref_seed = torch.where(active, log_gamma.new_zeros(()), log_gamma.new_full((), -torch.inf))
        seed = log_gamma.new_full((n, f), -torch.inf)
        seed.index_copy_(0, view.reference_node, ref_seed.permute(1, 0, 2, 3).reshape(-1, f))
        up = log_gamma.new_full((n, f), -torch.inf)
        for ns, ne, es, ee, degree in view.level_slices:
            nodes = view.level_node[ns:ne]
            child, rank = view.level_child[es:ee], view.level_parent_rank[es:ee]
            message = up.index_select(0, child) + log_gamma.index_select(0, child)[:, None]
            aggregate = self.segment_logsum(message, rank, ne-ns)
            up.index_copy_(0, nodes, safe_logadd(seed.index_select(0, nodes), aggregate))
        outside = log_gamma.new_full((n, f), -torch.inf)
        for ns, ne, es, ee, degree in reversed(view.level_slices):
            if es == ee:
                continue
            nodes = view.level_node[ns:ne]
            child, rank = view.level_child[es:ee], view.level_parent_rank[es:ee]
            message = up.index_select(0, child) + log_gamma.index_select(0, child)[:, None]
            left, right = segmented_exclusive_logs(message, rank, degree)
            local = safe_logadd(seed.index_select(0, nodes), outside.index_select(0, nodes))
            cavity = safe_logadd(safe_logadd(left, right), local.index_select(0, rank))
            outside.index_copy_(0, child, cavity + log_gamma.index_select(0, child)[:, None])
        log_n = outside.index_select(0, view.query_node).reshape(q, b, G, A).permute(1, 0, 2, 3)
        log_d = safe_logsum(log_n, -1)
        query = genotype.index_select(1, view.query_column).long()
        choose = query.clamp_min(0)[:, :, None, None].expand(b, q, G, 1)
        log_match = log_n.gather(-1, choose).squeeze(-1)
        counts = torch.einsum("rg,bra->bga", view.group_onehot, allele_hot.to(log_gamma.dtype))
        called = counts.sum(-1)
        valid = (query >= 0)[:, :, None] & (called > 0)[:, None, :]
        safe_d = torch.where(torch.isfinite(log_d), log_d, torch.zeros_like(log_d))
        matching = (log_match-safe_d).exp()
        frequency = counts / called.clamp_min(1)[..., None]
        ordinary = frequency[:, None].expand(b, q, G, A).gather(-1, choose).squeeze(-1)
        contrast = torch.where(valid, matching-ordinary, torch.zeros_like(matching))
        mass = (safe_d-called.clamp_min(1).log()[:, None]).clamp_max(0)
        mass = torch.where(valid, mass, torch.zeros_like(mass))
        features = torch.cat((contrast, mass, valid.to(log_gamma.dtype)), -1)
        mask = torch.cat((valid, valid, (query >= 0)[:, :, None].expand_as(valid)), -1)
        return features, mask

    def marker_features_v2(self, log_gamma, genotype, cpu_layout):
        """Bounded qualification API: `[B,Q,24]` features and bool mask.

        B must be1..16. This is NOT a whole-parent materialization API. Root-edge
        log_gamma is ignored by tree routing. Group order:younger C4 then older.
        """
        require(isinstance(cpu_layout, PackedHoldCPUV2) and genotype.ndim == 2
                and 1 <= genotype.shape[0] <= self.marker_block_size
                and genotype.shape[1] == cpu_layout.haplotypes
                and log_gamma.shape == (cpu_layout.nodes,) and log_gamma.dtype == torch.float32,
                "One bounded observed hold block required")
        view = cpu_layout._transient_device_view(log_gamma.device)
        return self._block_evidence(log_gamma.clamp(LOG_MIN, LOG_MAX), genotype, view)

    def block_statistics(self, log_gamma, genotype, cpu_layout):
        features, mask = self.marker_features_v2(log_gamma, genotype, cpu_layout)
        sums = torch.where(mask, features, torch.zeros_like(features)).sum(0)
        maximum = torch.where(mask, features, -torch.inf).amax(0)
        return sums, maximum, mask.to(log_gamma.dtype).sum(0)

    def forward(self, log_gamma, genotype, cpu_layout):
        """One-hold `[Q,48]` summary with nested marker-block checkpoints.

        Caller also checkpoints the whole hold to discard gathered gate/graph
        inputs across holds. Each inner closure captures CPU layout ONLY; every
        device view is created inside that block checkpoint and never cached.
        This is the intended block-lifetime bound; numerical/resource qualification is required.
        """
        require(isinstance(cpu_layout, PackedHoldCPUV2) and genotype.ndim == 2
                and genotype.shape[1] == cpu_layout.haplotypes
                and log_gamma.shape == (cpu_layout.nodes,) and log_gamma.dtype == torch.float32
                and genotype.device == log_gamma.device,
                "Single-hold observed categorical/FP32 contract differs")
        effective = log_gamma.clamp(LOG_MIN, LOG_MAX)
        q = cpu_layout.query_column.numel()
        sums, count = effective.new_zeros((q, 24)), effective.new_zeros((q, 24))
        maximum = effective.new_full((q, 24), -torch.inf)
        for start in range(0, genotype.shape[0], self.marker_block_size):
            block = genotype[start:start+self.marker_block_size]
            def run_block(lg, gt):
                return self.block_statistics(lg, gt, cpu_layout)
            if torch.is_grad_enabled():
                add, high, number = checkpoint(run_block, effective, block,
                    use_reentrant=False, preserve_rng_state=False)
            else:
                add, high, number = run_block(effective, block)
            sums, count, maximum = sums+add, count+number, torch.maximum(maximum, high)
        return torch.cat((sums/count.clamp_min(1),
                          torch.where(count > 0, maximum, torch.zeros_like(maximum))), -1)


class PackedSameMarkerAdapterV2(nn.Module):
    """Same714 allocated parameters/maths asv1; whole-hold checkpoint lifetime.

    Gate74, rank-input384, output256. No general operator/accuracy/speed claim.
    The only public forest API keeps packed metadata on CPU and transfers one
    transient hold inside checkpointed work. It must never cache the device view.
    """
    def __init__(self, marker_block_size=16):
        super().__init__()
        self.gate = nn.Module()
        self.gate.projection = nn.Linear(73, 1)
        nn.init.zeros_(self.gate.projection.weight)
        fraction = (-math.log(.95)+LOG_MAX)/(-LOG_MIN+LOG_MAX)
        nn.init.constant_(self.gate.projection.bias, math.log(fraction/(1-fraction)))
        self.evidence = PackedSameMarkerEvidenceV2(marker_block_size)
        self.input = nn.Linear(48, 8, bias=False)
        self.output = nn.Linear(8, 32, bias=False)
        nn.init.zeros_(self.output.weight)
        self.float()

    def _one_hold(self, cpu_layout, global_upward, global_geometry, global_support, global_genotype):
        view = cpu_layout._transient_device_view(global_upward.device)
        # All gathered copies and gate inputs are INSIDE checkpoint, so they
        # cannot accumulate as saved numerical arrays across all trees/sites.
        up = global_upward.index_select(0, view.native_node)
        edge9 = torch.cat((global_geometry.index_select(0, view.native_node),
                           global_support.index_select(0, view.native_node)), -1)
        parent_up = up.index_select(0, view.parent.clamp_min(0))
        score = self.gate.projection(torch.cat((edge9, up, parent_up), -1)).squeeze(-1)
        log_gamma = LOG_MAX+(LOG_MIN-LOG_MAX)*score.sigmoid()
        gt = global_genotype[cpu_layout.site_start:cpu_layout.site_stop]
        summary = self.evidence(log_gamma, gt, cpu_layout)
        return self.output(self.input(summary).tanh())

    def forward_native_hold_v2(self, cpu_layout, global_upward, global_geometry,
                               global_support, global_genotype):
        """Return `[Q,32]`; original forest arguments are SHARED tensor refs.

        No original observation or history is detached. In grad-enabled execution
        whole-hold nonreentrant checkpointing is mandatory, including zero-output
        initialization. CPU closure state contains no device layout/messages.
        Root supplies authenticated immutable original bank fields.
        """
        require(isinstance(cpu_layout, PackedHoldCPUV2), "One packed CPU hold required")
        cpu_layout.verify_immutable()
        require(global_upward.dtype == global_geometry.dtype == global_support.dtype == torch.float32
                and global_upward.ndim == 2 and global_upward.shape[1] == 32
                and global_geometry.shape == (global_upward.shape[0], 4)
                and global_support.shape == (global_upward.shape[0], 5)
                and global_genotype.ndim == 2 and global_genotype.shape[1] == cpu_layout.haplotypes
                and global_genotype.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
                and global_upward.device == global_geometry.device == global_support.device == global_genotype.device,
                "Original same-device full-bank observation/state contract differs")
        def run(up, geometry, support, genotype):
            return self._one_hold(cpu_layout, up, geometry, support, genotype)
        if torch.is_grad_enabled():
            return checkpoint(run, global_upward, global_geometry, global_support, global_genotype,
                              use_reentrant=False, preserve_rng_state=False)
        return run(global_upward, global_geometry, global_support, global_genotype)
