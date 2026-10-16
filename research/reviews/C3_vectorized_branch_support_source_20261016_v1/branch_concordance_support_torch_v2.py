"""Source-only CPU vectorization of observed branch concordance.

This additive file is not imported by the live C3 encoder. It is not execution,
qualification, admission or integration authority. It accepts observed genotypes
of all sampled haplotypes, including queries, but no query ancestry labels.

Integer decisions preserve the original feature equations. Floating support
accumulation is FP32 with reordered site/chunk/allele reductions, so this has a
new implementation identity and does not promise bitwise identity to the
original float64-accumulate, float32-return implementation.
"""
from __future__ import annotations

FEATURE_IMPLEMENTATION_ID = "branch_concordance_torch_int64_levels_fp32_v2"
REFERENCE_SOURCE_PATH = "src/arggnn/mutation_routed_tree_observations_v1.py"
REFERENCE_SOURCE_SHA256 = (
    "922df286e595370ade23595420a325014e4df64aac88060e8a1a846684eab068"
)
DEFAULT_SITE_CHUNK_SIZE = 32
MAX_SITE_CHUNK_SIZE = 256
SUPPORT_FIELDS = (
    "exact_split_support",
    "closest_weighted_support",
    "mean_discrepancy",
    "callable_split_count",
    "empty",
)


def _chunk_summary(torch, value, declared, sample_node, branches, levels, nodes, alleles):
    """Return FP32 unnormalized branch sums and an exact integer split count.

    All tensors are CPU. value follows the tree's sampled-node order. levels
    contains (child_node_ids, parent_node_ids) in ascending child height.
    """
    batch = value.shape[0]
    count_branches = branches.numel()
    result = torch.zeros((count_branches, 3), dtype=torch.float32, device="cpu")
    called = value >= 0
    ncalled = called.sum(dim=1, dtype=torch.int64)
    totals = torch.zeros((batch, alleles), dtype=torch.int64, device="cpu")
    totals.scatter_add_(1, value.clamp_min(0), called.to(torch.int64))
    allele_axis = torch.arange(alleles, dtype=torch.int64, device="cpu")
    proper = (
        (allele_axis[None, :] < declared[:, None])
        & (totals > 0)
        & (totals < ncalled[:, None])
        & (ncalled[:, None] >= 2)
    )
    active = proper.any(dim=1)
    split_count = int(proper.sum(dtype=torch.int64).item())
    if split_count == 0:
        return result, 0
    # A valid connected tree with >=2 called sampled nodes has a nonroot edge.
    # Avoid a reduction over an empty branch dimension even for malformed input.
    if count_branches == 0:
        raise ValueError("A proper observed split requires a nonroot branch")

    value = value[active]
    called = called[active]
    ncalled = ncalled[active]
    totals = totals[active]
    proper = proper[active]
    batch = value.shape[0]

    descendant = torch.zeros(
        (nodes, batch, alleles), dtype=torch.int64, device="cpu"
    )
    # Unique sampled nodes make these destinations distinct. Missing calls write
    # zero to the safe allele-0 location. Sampled internals receive their own seed
    # before their children's counts arrive, exactly as in the original routine.
    site = torch.arange(batch, dtype=torch.int64, device="cpu")
    descendant[
        sample_node[:, None],
        site[None, :],
        value.transpose(0, 1).clamp_min(0),
    ] = called.transpose(0, 1).to(torch.int64)

    for child, parent in levels:
        # Every edge appears once. All children of an edge have smaller height
        # than its parent, so each gathered source already contains descendants.
        # index_select gives a separate source buffer for in-place index_add_.
        descendant.index_add_(0, parent, descendant.index_select(0, child))

    descendant_called = descendant.sum(dim=2, dtype=torch.int64).index_select(
        0, branches
    )
    ncalled_fp32 = ncalled.to(torch.float32)
    for allele in range(alleles):  # At most four iterations, never a site loop.
        valid = proper[:, allele]
        if not bool(valid.any()):
            continue
        descendant_allele = descendant[:, :, allele].index_select(0, branches)
        difference = (
            descendant_called
            + totals[:, allele][None, :]
            - 2 * descendant_allele
        )
        mismatch = torch.minimum(difference, ncalled[None, :] - difference)
        minimum = mismatch.amin(dim=0)
        winners = (mismatch == minimum[None, :]) & valid[None, :]
        ties = winners.sum(dim=0, dtype=torch.int64)
        # Invalid alleles have no winners and contribute zero. The clamp only
        # makes their unused division defined, never changes a valid tie count.
        share = valid.to(torch.float32) / ties.clamp_min(1).to(torch.float32)
        winner_fp32 = winners.to(torch.float32)
        exact_share = torch.where(
            minimum == 0, share, torch.zeros_like(share)
        )
        closest_share = (
            1.0 - minimum.to(torch.float32) / ncalled_fp32
        ) * share
        result[:, 0].add_(
            (winner_fp32 * exact_share[None, :]).sum(
                dim=1, dtype=torch.float32
            )
        )
        result[:, 1].add_(
            (winner_fp32 * closest_share[None, :]).sum(
                dim=1, dtype=torch.float32
            )
        )
        result[:, 2].add_(
            (
                mismatch.to(torch.float32)
                / ncalled_fp32[None, :]
                * valid.to(torch.float32)[None, :]
            ).sum(dim=1, dtype=torch.float32)
        )
    return result, split_count


def branch_concordance_support_torch_v2(
    tree,
    genotype,
    allele_count,
    sample_ids,
    *,
    site_chunk_size=DEFAULT_SITE_CHUNK_SIZE,
):
    """Return a fresh NumPy float32 [N,5] support array for one connected tree.

    Positional inputs have the same meaning as branch_concordance_support_v1.
    sample_ids explicitly joins genotype columns to tree.sample_id. The original
    _validate_alleles, _integer and _layout checks remain authoritative. Donor
    frequencies, reference classes, sample ages and alpha are not arguments and
    are neither recomputed nor changed.

    Only CPU tensors are created. This function does not set Torch thread
    counts, environment variables, CUDA visibility or resource policies.
    Caller-owned CPU budgets and the existing worker RSS cap still apply.
    Importing this source file alone does not import NumPy, Torch or PyG.
    """
    if (
        type(site_chunk_size) is not int
        or not 1 <= site_chunk_size <= MAX_SITE_CHUNK_SIZE
    ):
        raise ValueError("site_chunk_size must be an integer in 1..256")

    import numpy as np
    from arggnn.mutation_routed_tree_observations_v1 import (
        _integer,
        _layout,
        _validate_alleles,
    )

    g, k = _validate_alleles(genotype, allele_count)
    input_ids = _integer(sample_ids, "sample_ids")
    p, _, sn, sid, _, _, height, _, _ = _layout(tree)
    if (
        g.shape[1] != len(input_ids)
        or len(set(input_ids.tolist())) != len(input_ids)
        or set(input_ids.tolist()) != set(sid.tolist())
    ):
        raise ValueError(
            "All and only actual sampled haplotypes must join genotype columns"
        )
    column_by_id = {
        identity: at for at, identity in enumerate(input_ids.tolist())
    }
    sample_column = np.asarray(
        [column_by_id[identity] for identity in sid], dtype=np.int64
    )
    nodes = len(p)
    branch_array = np.flatnonzero(p >= 0).astype(np.int64, copy=False)

    # Keep the C3 encoder's all-isolated-forest bypass unchanged. _layout above
    # deliberately retains the original connected-tree contract.
    if len(g) == 0 or len(branch_array) == 0:
        out = np.zeros((nodes, 5), dtype=np.float32)
        out[:, 4] = 1.0
        return out

    # Group once by child height, avoiding a full node mask per height. This is
    # O(N log N) topology preparation and O(N) retained schedule storage. A node
    # can be sampled and internal. No binary-tree or leaf-only assumption enters.
    order = branch_array[
        np.argsort(height[branch_array], kind="stable")
    ]
    boundaries = np.flatnonzero(np.diff(height[order]) != 0) + 1
    child_groups = np.split(order, boundaries)

    # Lazy import: writing, parsing or inspecting this module creates no Torch
    # runtime and cannot create a CUDA context.
    import torch

    with torch.inference_mode():
        branches = torch.from_numpy(branch_array.copy())
        sample_node = torch.from_numpy(sn.copy())
        levels = tuple(
            (
                torch.from_numpy(child.copy()),
                torch.from_numpy(p[child].copy()),
            )
            for child in child_groups
        )
        if any(
            bool(np.any(height[p[child]] <= height[child]))
            for child in child_groups
        ):
            raise ValueError("Every parent must have greater height than its child")
        accumulated = torch.zeros(
            (len(branch_array), 3), dtype=torch.float32, device="cpu"
        )
        splits = 0
        alleles = int(k.max(initial=1))
        for start in range(0, len(g), site_chunk_size):
            stop = min(start + site_chunk_size, len(g))
            value = torch.as_tensor(
                np.ascontiguousarray(g[start:stop, sample_column]),
                dtype=torch.int64,
                device="cpu",
            )
            declared = torch.as_tensor(
                np.ascontiguousarray(k[start:stop]),
                dtype=torch.int64,
                device="cpu",
            )
            chunk_sum, chunk_splits = _chunk_summary(
                torch,
                value,
                declared,
                sample_node,
                branches,
                levels,
                nodes,
                alleles,
            )
            accumulated.add_(chunk_sum)
            splits += chunk_splits

        out = torch.zeros((nodes, 5), dtype=torch.float32, device="cpu")
        if splits:
            # One global normalization. Never average per-chunk averages.
            normalizer = torch.tensor(splits, dtype=torch.float32, device="cpu")
            out[branches, :3] = accumulated / normalizer
            out[branches, 3] = normalizer
        else:
            out[branches, 4] = 1.0
        root = torch.from_numpy((p < 0).copy())
        out[root, 4] = 1.0
        return out.numpy().copy()
