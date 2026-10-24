"""Isolated categorical execution of the existing TreeLoom site MLP.

The MLP input depends on site and observed allele, not haplotype identity.
No learned value is cached: every call evaluates current parameters, then
gathers into the original site/haplotype order before unchanged reductions.
This module does not load files, patch a model, launch work, or update weights.
"""
import math
import torch


def categorical_features(genotype, donor, donor_mask, panel, entropy):
    """Return a finite [site, allele, 2*channels+3] feature table.

    Unused allele slots have no observations and zero downstream gradient.
    Their panel probability is replaced by one before log, then their features
    are set to zero. This avoids generating log(0) or NaN in unused padding.
    Every observed called slot has exactly the current MLP's input features.
    Missing haplotypes do not contribute to sum, max, count or MLP gradients.
    """
    called = genotype >= 0
    chosen = genotype.long().clamp_min(0)
    sites, alleles, channels = donor.shape
    usage = torch.zeros((sites, alleles), device=genotype.device, dtype=torch.int32)
    usage = usage.scatter_add(1, chosen, called.to(torch.int32))
    used = usage > 0
    available = donor_mask[:, None, :].expand(-1, alleles, -1)
    logfreq = torch.where(available, donor, torch.zeros_like(donor))
    panel_log = torch.where(used, panel, torch.ones_like(panel)).log()
    ent = entropy[:, None].expand(-1, alleles)
    features = torch.cat((logfreq, available.to(donor.dtype),
                          panel_log[..., None], ent[..., None],
                          torch.ones_like(panel_log)[..., None]), dim=-1)
    features = torch.where(used[..., None], features, torch.zeros_like(features))
    return features, chosen, called


def categorical_site_chunk(model, genotype, donor, donor_mask, panel,
                           entropy, inverse, count_holds):
    """Same FP32 site/chunk sums, maxima and counts using categorical MLP work.

    This preserves original site order, haplotype order, hold indices, max ties
    and gather-backward multiplicity. It does not change chunk boundaries.
    The caller must supply the same authenticated observation tensors and
    global256-site chunk partition as the existing executor.
    """
    h, d = genotype.shape[1], model.hidden_dim
    features, chosen, called = categorical_features(
        genotype, donor, donor_mask, panel, entropy)
    category_encoded = model.site_mlp(features)
    encoded = category_encoded.gather(1, chosen[..., None].expand(-1, -1, d))
    index = (inverse[:, None]*h + torch.arange(h, device=genotype.device)[None]).flatten()
    values = torch.where(called[..., None], encoded,
                         torch.zeros_like(encoded)).reshape(-1, d)
    sums = values.new_zeros(count_holds*h, d).index_add(0, index, values)
    counts = values.new_zeros(count_holds*h, 1).index_add(
        0, index, called.to(values.dtype).reshape(-1, 1))
    max_values = torch.where(called[..., None], encoded,
                             torch.full_like(encoded, -torch.inf)).reshape(-1, d)
    maxima = values.new_full((count_holds*h, d), -torch.inf).scatter_reduce(
        0, index[:, None].expand_as(max_values), max_values,
        reduce="amax", include_self=True)
    maxima = torch.where(counts > 0, maxima, torch.zeros_like(maxima))
    return (sums.reshape(count_holds, h, d),
            maxima.reshape(count_holds, h, d),
            counts.reshape(count_holds, h, 1))


def categorical_encode_sites(model, bank):
    """Diagnostic exact-source pooling adapter with original256-site chunks."""
    t, h = bank.sample_index.shape
    d = model.hidden_dim
    pool_holds, pool_sums, pool_max, pool_count = [], [], [], []
    for start in range(0, len(bank.genotype), model.site_chunk_size):
        end = min(start+model.site_chunk_size, len(bank.genotype))
        holds, inverse = torch.unique_consecutive(
            bank.site_hold[start:end], return_inverse=True)
        args = (bank.genotype[start:end], bank.log_donor_allele_frequency[start:end],
                bank.donor_call_mask[start:end], bank.panel_allele_frequency[start:end],
                bank.site_entropy[start:end], inverse, len(holds))
        sums, maximum, count = categorical_site_chunk(model, *args)
        pool_holds.append(holds)
        pool_sums.append(sums)
        pool_max.append(maximum)
        pool_count.append(count)
    sums = bank.spans.new_zeros(t, h, d)
    maxima = bank.spans.new_full((t, h, d), -torch.inf)
    counts = bank.spans.new_zeros(t, h, 1)
    if pool_holds:
        index = torch.cat(pool_holds)
        sums = sums.index_add(0, index, torch.cat(pool_sums))
        counts = counts.index_add(0, index, torch.cat(pool_count))
        chunk_max = torch.cat(pool_max)
        chunk_count = torch.cat(pool_count)
        chunk_max = torch.where(chunk_count > 0, chunk_max,
                                torch.full_like(chunk_max, -torch.inf))
        maxima = maxima.scatter_reduce(
            0, index[:, None, None].expand_as(chunk_max), chunk_max,
            reduce="amax", include_self=True)
    present = counts > 0
    means = sums/counts.clamp_min(1)
    maxima = torch.where(present, maxima, torch.zeros_like(maxima))
    logcount = counts.log1p()/math.log(100001.)
    empty = (~present).to(sums.dtype)
    span = (bank.spans.log1p()/math.log(2000001.))[:, None, None].expand(t, h, 1)
    return model.site_pool_projection(torch.cat(
        (means, maxima, logcount, empty, span), -1))
