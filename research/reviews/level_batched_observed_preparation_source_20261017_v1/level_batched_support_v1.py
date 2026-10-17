"""CPU observed-only descendant counts and five original branch summaries.

The caller supplies the unchanged, already validated native layout tuple.
This module has no file loader, target, parameter, launcher or lifecycle API.
Counts use int64 Torch/PyG sums; summary accumulation is ordinary FP32.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch_geometric.utils import scatter


def _require(value, message):
    if not value:
        raise ValueError(message)


def _integer_cpu(value, name, ndim=1):
    tensor = torch.as_tensor(value)
    _require(tensor.device.type == "cpu" and tensor.ndim == ndim
        and tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        name + " must be an integer CPU tensor/array of the original rank")
    return tensor


@dataclass(frozen=True)
class BranchLevelLayoutV1:
    parent: torch.Tensor
    sample_node: torch.Tensor
    sample_id: torch.Tensor
    branches: torch.Tensor
    roots: torch.Tensor
    levels: tuple
    children: tuple
    parent_ranks: tuple


def prepare_branch_level_layout_v1(validated_native_layout):
    """Reuse original raw chronology/topology validation; never sort by FP32 age.

    Input is the original nine-field _layout / _layout_with_isolated_samples
    result, computed once by the caller. Connected trees retain true unary
    edges and sampled ancestors. The H060 all-sampled/all-isolated zero-site
    case is represented without a virtual root. Mixed disconnected forests
    remain outside the original supported domain.
    """
    _require(len(validated_native_layout) == 9, "Original nine-field validated native layout required")
    p, _, sn, sid, _, _, height, _, _ = validated_native_layout
    p = _integer_cpu(p, "parent").to(torch.long).clone()
    sn = _integer_cpu(sn, "sample_node").to(torch.long).clone()
    sid = _integer_cpu(sid, "sample_id").to(torch.long).clone()
    height = _integer_cpu(height, "height").to(torch.long).clone()
    n = p.numel()
    _require(n > 0 and height.shape == p.shape and bool((height >= 0).all())
        and bool(((p >= -1) & (p < n)).all()) and sn.numel() > 0
        and sn.shape == sid.shape and bool(((sn >= 0) & (sn < n)).all())
        and bool((sid >= 0).all()) and sn.unique().numel() == sn.numel()
        and sid.unique().numel() == sid.numel(), "Complete original native node/sample/height joins")
    branches = torch.nonzero(p >= 0).flatten()
    roots = torch.nonzero(p < 0).flatten()
    _require(roots.numel() == 1 or (roots.numel() == n and sn.numel() == n),
        "Original connected tree or all-sampled/all-isolated hold only")
    _require(bool((height[p[branches]] > height[branches]).all()),
        "Topological height must strictly increase across every real native edge")
    # One global rank vector is safe: each vertex belongs to exactly one level.
    rank = torch.empty(n, dtype=torch.long)
    levels = tuple(torch.nonzero(height == level).flatten()
        for level in range(int(height.max().item()) + 1))
    for nodes in levels:
        rank[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
    children = tuple(branches[height[p[branches]] == level]
        for level in range(len(levels)))
    parent_ranks = tuple(rank[p[nodes]] for nodes in children)
    return BranchLevelLayoutV1(p, sn, sid, branches, roots, levels, children, parent_ranks)


def _inputs(layout, genotype, allele_count, sample_ids):
    g = _integer_cpu(genotype, "genotype", ndim=2)
    k = _integer_cpu(allele_count, "allele_count")
    ids = _integer_cpu(sample_ids, "sample_ids")
    _require(k.shape == (g.shape[0],) and ids.shape == (g.shape[1],)
        and ids.numel() == layout.sample_id.numel() and ids.unique().numel() == ids.numel()
        and bool(((k >= 1) & (k <= 4)).all()),
        "Original categorical allele axes1..4, missing=-1 and complete unique roster")
    sorted_ids, order = ids.to(torch.long).sort()
    where = torch.searchsorted(sorted_ids, layout.sample_id)
    _require(bool((where < ids.numel()).all()), "All native sampled identities must be present")
    _require(torch.equal(sorted_ids[where], layout.sample_id), "All and only original sampled identities")
    if layout.roots.numel() > 1:
        _require(g.shape[0] == 0, "H060 all-isolated compatibility remains zero-SNP only")
    return g, k.to(torch.long), order[where]


def descendant_allele_chunks_v1(layout, genotype, allele_count, sample_ids, *,
        marker_chunk_size=32, max_count_elements=8388608):
    """Yield bounded (start, stop, counts[N,C,A], total[C,A], called[C], proper[C,A]).

    A is max(allele_count, initial=1) over this original call, never a renumbered
    or globally substituted allele vocabulary. Invalid padded allele slots are
    excluded by proper. Every original marker is visited; chunking never drops
    a marker or sample. max_count_elements bounds the main int64 count tensor,
    not total RSS or allocator/PyG scratch; the caller retains existing caps.
    """
    _require(isinstance(layout, BranchLevelLayoutV1), "Prepared original native layout required")
    _require(type(marker_chunk_size) is int and marker_chunk_size > 0
        and type(max_count_elements) is int and max_count_elements > 0,
        "Positive bounded marker/count allocation controls")
    g, k, columns = _inputs(layout, genotype, allele_count, sample_ids)
    n = layout.parent.numel()
    if not g.shape[0]:
        return
    width = int(k.max().item())
    capacity = max_count_elements // (n * width)
    _require(capacity > 0, "One marker exceeds count-tensor bound; no pruning or fallback")
    chunk_size = min(marker_chunk_size, capacity)
    allele = torch.arange(width, dtype=torch.long)
    for start in range(0, g.shape[0], chunk_size):
        stop = min(start + chunk_size, g.shape[0])
        value = g[start:stop, columns].to(torch.long)
        _require(bool(((value >= -1) & (value < k[start:stop, None])).all()),
            "Original categorical calls must fit each original site allele axis")
        called_mask = value >= 0
        # Native sampled ancestors retain their own contribution as well as
        # descendants; unique sample_node makes seed index_copy unambiguous.
        seed = (value[:, :, None] == allele) & called_mask[:, :, None]
        total = seed.sum(dim=1, dtype=torch.long)
        called = called_mask.sum(dim=1, dtype=torch.long)
        counts = torch.zeros((n, stop-start, width), dtype=torch.long)
        counts.index_copy_(0, layout.sample_node, seed.permute(1, 0, 2).to(torch.long))
        for nodes, children, ranks in zip(layout.levels, layout.children, layout.parent_ranks):
            if children.numel():
                aggregate = scatter(counts.index_select(0, children), ranks,
                    dim=0, dim_size=nodes.numel(), reduce="sum")
                counts.index_copy_(0, nodes, counts.index_select(0, nodes) + aggregate)
        proper = ((total > 0) & (total < called[:, None])
            & (allele[None] < k[start:stop, None]) & (called[:, None] >= 2))
        del seed, value, called_mask
        yield start, stop, counts, total, called, proper
        del counts, total, called, proper


@torch.no_grad()
def level_batched_branch_support_v1(layout, genotype, allele_count, sample_ids, *,
        marker_chunk_size=32, max_count_elements=8388608):
    """The original five branch-concordance fields; no inferred mutation labels.

    Integer descendant counts, mismatch minima and all tied real branches are
    exact. Fractions/reductions use FP32, so old float64-then-cast canonical
    bytes are not promised. Returned normal tensors may feed learned modules;
    inference_mode tensors are deliberately not created.
    """
    _require(isinstance(layout, BranchLevelLayoutV1), "Prepared original native layout required")
    out = torch.zeros((layout.parent.numel(), 5), dtype=torch.float32)
    branches = layout.branches
    splits = 0
    for _, _, counts, total, called, proper in descendant_allele_chunks_v1(
            layout, genotype, allele_count, sample_ids,
            marker_chunk_size=marker_chunk_size, max_count_elements=max_count_elements):
        if not branches.numel():
            del counts
            continue
        inside = counts.index_select(0, branches).permute(1, 0, 2)
        descendant_called = inside.sum(dim=2, keepdim=True)
        difference = descendant_called + total[:, None, :] - 2 * inside
        mismatch = torch.minimum(difference, called[:, None, None] - difference)
        minimum = mismatch.amin(dim=1)
        winners = (mismatch == minimum[:, None, :]) & proper[:, None, :]
        ties = winners.sum(dim=1).clamp_min(1).to(torch.float32)
        denominator = called.clamp_min(1).to(torch.float32)
        credit = winners.to(torch.float32) / ties[:, None, :]
        out[branches, 0] += (credit * (minimum == 0)[:, None, :]).sum(dim=(0, 2))
        quality = 1. - minimum.to(torch.float32) / denominator[:, None]
        out[branches, 1] += (credit * quality[:, None, :]).sum(dim=(0, 2))
        discrepancy = mismatch.to(torch.float32) / denominator[:, None, None]
        out[branches, 2] += (discrepancy * proper[:, None, :]).sum(dim=(0, 2))
        splits += int(proper.sum().item())
        del counts, inside, descendant_called, difference, mismatch, minimum
        del winners, ties, denominator, credit, quality, discrepancy
    if splits:
        out[branches, :3] /= splits
        out[branches, 3] = splits
    else:
        out[branches, 4] = 1.
    out[layout.roots, 4] = 1.
    return out
