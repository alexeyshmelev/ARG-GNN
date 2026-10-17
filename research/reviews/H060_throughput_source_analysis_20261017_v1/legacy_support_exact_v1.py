"""Inactive exact legacy support candidate: batched integer counts, original sums.

No file loader, truth API, model step, launcher, admission or persistent cache.
Float64 below preserves the old observed-feature recipe, not model precision.
"""
from __future__ import annotations
import numpy as np
import torch
from arggnn.mutation_routed_tree_observations_v1 import _layout, _validate_alleles, _integer
from research.reviews.level_batched_observed_preparation_source_20261017_v1.level_batched_support_v1 import (
    prepare_branch_level_layout_v1, descendant_allele_chunks_v1)


@torch.no_grad()
def branch_concordance_support_exact_v1(tree, genotype, allele_count, sample_ids, *,
        marker_chunk_size=32, max_count_elements=8388608):
    """Keep legacy site/allele order and NumPy float64 accumulation exactly.

    Only the inner per-site Python postorder sweep is replaced. Integer PyG
    level sums visit every native edge/sample/site. Main count allocation has
    the existing 8,388,608-element failstop; scratch/total RSS are not proved.
    No FP32 summary reduction, site reordering, split pruning or truth input.
    """
    g, k = _validate_alleles(genotype, allele_count)
    sample_ids = _integer(sample_ids, "sample_ids")
    raw_layout = _layout(tree)
    p, _, sn, sid, postorder, _, _, _, _ = raw_layout
    if (g.shape[1] != len(sample_ids) or len(set(sample_ids.tolist())) != len(sample_ids)
            or set(sample_ids.tolist()) != set(sid.tolist())):
        raise ValueError("All and only actual sampled haplotypes must join genotype columns")
    layout = prepare_branch_level_layout_v1(raw_layout)
    n = len(p); branches = np.flatnonzero(p >= 0)
    out = np.zeros((n, 5), dtype=np.float64)
    splits = 0
    for start, stop, counts, total, called, proper in descendant_allele_chunks_v1(
            layout, g, k, sample_ids, marker_chunk_size=marker_chunk_size,
            max_count_elements=max_count_elements):
        counts, total, called, proper = (x.numpy() for x in (counts, total, called, proper))
        for at in range(stop-start):
            ncalled = int(called[at])
            if ncalled < 2:
                continue
            alleles = np.flatnonzero(proper[at])
            if not len(alleles):
                continue
            # Padded allele slots are zero, so this equals legacy sum(1).
            descendant_alleles = counts[:, at, :]
            d = descendant_alleles.sum(1)[branches]
            for allele in alleles:
                difference = d + int(total[at, allele]) - 2 * descendant_alleles[branches, allele]
                mismatch = np.minimum(difference, ncalled - difference)
                discrepancy = mismatch.astype(np.float64) / ncalled
                minimum = int(mismatch.min())
                winners = branches[mismatch == minimum]
                if minimum == 0:
                    out[winners, 0] += 1. / len(winners)
                out[winners, 1] += (1. - minimum / ncalled) / len(winners)
                out[branches, 2] += discrepancy
                splits += 1
    if splits:
        out[branches, :3] /= splits
        out[branches, 3] = splits
    else:
        out[branches, 4] = 1.
    out[p < 0, 4] = 1.
    return out.astype(np.float32)
