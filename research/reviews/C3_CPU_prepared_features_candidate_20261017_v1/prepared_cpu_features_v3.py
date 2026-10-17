"""Inactive CPU feature-preparation candidate. No loader, launcher or authority.

The caller must supply exact immutable observation/plan/tree keys. This module
caches only observed CPU features, never learned embeddings or query labels.
No existing module globals or files are mutated. Adoption needs separate review.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import fields, replace
import hashlib
import json
import types

import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from arggnn import american_c3_gnn_observations_isolated_v2 as original
from arggnn.mutation_routed_tree_observations_v1 import _validate_alleles


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _key(key):
    require(isinstance(key, tuple) and 1 <= len(key) <= 8
            and all(type(v) in (str, int) for v in key)
            and all(not isinstance(v, str) or len(v) <= 256 for v in key),
            "Explicit bounded immutable source/plan/tree identity tuple")
    return key


def _digest_values(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        if value is None:
            digest.update(b"NONE")
            continue
        array = np.asarray(value)
        require(array.dtype.kind in "biuf", "Numeric CPU observation values only")
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _tree_digest(tree):
    return _digest_values((name, getattr(tree, name)) for name in
                          ("parent", "node_time", "sample_node", "sample_id"))


def _propagation_digest(propagation):
    return _digest_values((str(i)+":"+str(j), value.numpy())
        for i, level in enumerate(propagation) for j, value in enumerate(level))


def _layout_digest(layout):
    return _digest_values((str(i), value) for i, value in enumerate(layout))


class _CountPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

    def forward(self, source_counts, edge_index, recipients):
        return self.propagate(edge_index, x=(source_counts, None),
                              size=(source_counts.shape[0], recipients))

    def message(self, x_j):
        return x_j


class ImmutableTreeLayoutCacheV1:
    """Exact-key CPU LRU. Input and cached layout digests are checked on reuse."""
    def __init__(self, *, max_bytes, max_entries):
        require(type(max_bytes) is int and max_bytes > 0
                and type(max_entries) is int and max_entries > 0, "Finite caps")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.values = OrderedDict()
        self.bytes = self.builds = self.hits = 0

    def get(self, tree, source_key):
        key = _key(source_key)
        p = np.asarray(tree.parent)
        sn = np.asarray(tree.sample_node)
        require(p.ndim == sn.ndim == 1 and len(p) > 0 and len(sn) > 0
                and 128*len(p)+32*len(sn) <= self.max_bytes,
                "Prospective complete layout cache allowance")
        supplied = _tree_digest(tree)
        if key in self.values:
            record = self.values[key]
            require(record["input_digest"] == supplied
                    and record["layout_digest"] == _layout_digest(record["layout"])
                    and record["propagation_digest"] == _propagation_digest(record["propagation"]),
                    "Immutable tree identity/content drift")
            self.values.move_to_end(key)
            self.hits += 1
            return record
        # Original layout policy includes the exact isolated-sample C3 exception.
        raw = original._layout_with_isolated_samples(tree)
        layout = tuple(np.array(value, copy=True) for value in raw)
        for value in layout:
            value.flags.writeable = False
        p, _, _, _, _, _, height, _, _ = layout
        n = len(p)
        child = np.flatnonzero(p >= 0)
        rank = np.empty(n, dtype=np.int64)
        levels = original._level_arrays(height)
        for nodes in levels:
            rank[nodes] = np.arange(len(nodes))
        propagation = []
        for level, nodes in enumerate(levels):
            selected = child[height[p[child]] == level]
            if len(selected):
                edge = np.stack((np.arange(len(selected)), rank[p[selected]]))
                propagation.append((torch.tensor(nodes), torch.tensor(selected),
                                    torch.tensor(edge)))
        size = sum(value.nbytes for value in layout) + sum(
            value.numel() * value.element_size()
            for level in propagation for value in level)
        require(size <= self.max_bytes, "One complete tree layout exceeds cap")
        while self.values and (len(self.values) >= self.max_entries
                               or self.bytes + size > self.max_bytes):
            _, old = self.values.popitem(last=False)
            self.bytes -= old["bytes"]
        record = dict(input_digest=supplied, layout=layout,
                      layout_digest=_layout_digest(layout),
                      propagation=tuple(propagation),
                      propagation_digest=_propagation_digest(propagation), bytes=size)
        self.values[key] = record
        self.bytes += size
        self.builds += 1
        return record


def _compensated_add_(output, correction, branches, channel, increment):
    """Bounded FP32 Kahan update across site-tile/allele partial sums."""
    previous = output[branches, channel]
    residual = increment - correction[branches, channel]
    updated = previous + residual
    correction[branches, channel] = (updated - previous) - residual
    output[branches, channel] = updated


def branch_support_batched_v1(tree, genotype, allele_count, sample_ids, *,
                              layout_cache, tree_key, site_tile=32,
                              max_working_bytes=64 * 2**20):
    """Same branch summaries with integer PyG count propagation for site tiles.

    All observed sites, alleles, branches and minimum-discrepancy ties remain.
    Compensated FP32 aggregation replaces the original FP64 accumulator.
    This is an explicit numerical adaptation, tested at ordinary FP32 tolerance.
    """
    require(type(site_tile) is int and 0 < site_tile <= 256
            and type(max_working_bytes) is int and max_working_bytes > 0,
            "Explicit finite tiled CPU work allowance")
    g, cardinality = _validate_alleles(genotype, allele_count)
    supplied_ids = np.asarray(sample_ids)
    require(supplied_ids.ndim == 1 and supplied_ids.dtype.kind in "iu",
            "Original integer sample-identity axis")
    record = layout_cache.get(tree, tree_key)
    p, _, sn, sid, _, _, _, _, _ = record["layout"]
    require(g.shape[1] == len(supplied_ids)
            and len(set(supplied_ids.tolist())) == len(supplied_ids)
            and set(supplied_ids.tolist()) == set(sid.tolist()),
            "All and only original phased sample identities")
    if len(p) > 1 and np.all(p == -1):
        result = np.zeros((len(p), 5), dtype=np.float32)
        result[:, 4] = 1
        return result
    columns = {identity: at for at, identity in enumerate(supplied_ids.tolist())}
    order = np.asarray([columns[x] for x in sid], dtype=np.int64)
    branches = torch.tensor(np.flatnonzero(p >= 0))
    sample_nodes = torch.tensor(np.array(sn, copy=True))
    n, h = len(p), len(sn)
    # Conservative named tensor working set allowance, excluding bounded layout,
    # original caller-owned inputs and PyTorch allocator overhead.
    tile = min(site_tile, max(1, len(g)))
    allowance = 512 * (n + h + 1) * tile + 96 * n
    require(allowance <= max_working_bytes,
            "Prospective complete site-tile tensor working allowance")
    output = torch.zeros((n, 5), dtype=torch.float32)
    correction = torch.zeros((n, 3), dtype=torch.float32)
    splits = 0
    propagation = _CountPropagation()
    with torch.inference_mode():
        for start in range(0, len(g), site_tile):
            gt = torch.tensor(np.array(g[start:start+site_tile, order], copy=True),
                              dtype=torch.long)
            b = len(gt)
            called = gt >= 0
            ncalled = called.sum(1)
            chosen = gt.clamp_min(0)
            onehot = torch.nn.functional.one_hot(chosen, num_classes=4)
            onehot *= called[..., None]
            total = onehot.sum(1)
            counts = torch.zeros((n, b, 4), dtype=torch.long)
            counts[sample_nodes] = onehot.permute(1, 0, 2)
            for nodes, children, edge in record["propagation"]:
                aggregate = propagation(counts[children], edge, len(nodes))
                counts.index_add_(0, nodes, aggregate)
            descendant = counts[branches]
            called_descendant = descendant.sum(-1)
            denominator = ncalled.clamp_min(1).to(torch.float32)
            for allele in range(4):
                eligible = ((total[:, allele] > 0)
                            & (total[:, allele] < ncalled) & (ncalled >= 2))
                if not bool(eligible.any()):
                    continue
                difference = (called_descendant + total[None, :, allele]
                              - 2 * descendant[:, :, allele])
                mismatch = torch.minimum(difference, ncalled[None] - difference)
                minimum = mismatch.min(0).values
                winners = mismatch == minimum[None]
                winner_count = winners.sum(0).clamp_min(1).to(torch.float32)
                weight = winners.to(torch.float32) * eligible[None] / winner_count[None]
                _compensated_add_(output, correction, branches, 0,
                    (weight * (minimum == 0)[None]).sum(1))
                quality = 1 - minimum.to(torch.float32) / denominator
                _compensated_add_(output, correction, branches, 1,
                    (weight * quality[None]).sum(1))
                discrepancy = mismatch.to(torch.float32) / denominator[None]
                _compensated_add_(output, correction, branches, 2,
                    (discrepancy * eligible[None]).sum(1))
                splits += int(eligible.sum())
        if splits:
            output[branches, :3] /= splits
            output[branches, 3] = splits
        else:
            output[branches, 4] = 1
        output[torch.tensor(np.flatnonzero(p < 0)), 4] = 1
    return output.numpy()


def encode_block_prepared_v1(trees, intervals, genotype, allele_count,
                             site_positions, reference_class, sample_age, *,
                             tree_keys, layout_cache, site_tile=32,
                             max_working_bytes=64 * 2**20, **kwargs):
    """Original block encoder with only local layout/support function bindings."""
    require(len(trees) == len(tree_keys) and len({id(t) for t in trees}) == len(trees),
            "Explicit distinct original tree objects and immutable identities")
    identity = {id(tree): key for tree, key in zip(trees, tree_keys)}
    def layout(tree):
        return layout_cache.get(tree, identity[id(tree)])["layout"]
    def support(tree, gt, k, sample_ids):
        return branch_support_batched_v1(tree, gt, k, sample_ids,
            layout_cache=layout_cache, tree_key=identity[id(tree)],
            site_tile=site_tile, max_working_bytes=max_working_bytes)
    namespace = dict(original.encode_block.__globals__)
    namespace.update(_layout_with_isolated_samples=layout,
                     branch_concordance_support_v1=support)
    encoder = types.FunctionType(original.encode_block.__code__, namespace,
                                 "encode_block_with_prepared_CPU_features",
                                 original.encode_block.__defaults__)
    encoder.__kwdefaults__ = dict(original.encode_block.__kwdefaults__ or {})
    return encoder(trees, intervals, genotype, allele_count, site_positions,
                   reference_class, sample_age, **kwargs)


def _bank_items(bank):
    for field in fields(bank):
        value = getattr(bank, field.name)
        if isinstance(value, tuple):
            yield from ((field.name+":"+str(i), x) for i, x in enumerate(value))
        else:
            yield field.name, value


def _bank_digest(bank):
    return _digest_values((name, tensor.detach().numpy())
                          for name, tensor in _bank_items(bank))


class ImmutableBlockFeatureCacheV1:
    """Bounded direction-independent CPU banks. Never caches learned embeddings."""
    def __init__(self, *, layout_cache, max_tensor_bytes, max_entries):
        require(type(max_tensor_bytes) is int and max_tensor_bytes > 0
                and type(max_entries) is int and max_entries > 0, "Finite cache caps")
        self.layouts, self.max_tensor_bytes = layout_cache, max_tensor_bytes
        self.max_entries, self.values = max_entries, OrderedDict()
        self.bytes = self.builds = self.hits = 0

    def get_or_prepare(self, block_key, trees, intervals, genotype, allele_count,
                       site_positions, reference_class, sample_age, *,
                       tree_keys, **kwargs):
        key = _key(block_key)
        observed_genotype = np.asarray(genotype)
        n = sum(len(t.parent) for t in trees)
        require(observed_genotype.ndim == 2, "Complete site/sample matrix")
        sites, h = observed_genotype.shape
        require(0 < len(trees) <= 32 and 0 < h <= 1600 and n <= 250000
                and sites <= 4096 and kwargs.get("num_classes") == 3,
                "Original C3 whole block dimension limits and three classes")
        forecast = 256*n + 32*len(trees)*h + (h+256)*sites
        require(forecast <= self.max_tensor_bytes,
                "Prospective complete feature tensor allowance")
        values = [("intervals", intervals), ("genotype", genotype),
                  ("allele_count", allele_count), ("site_positions", site_positions),
                  ("reference_class", reference_class), ("sample_age", sample_age)]
        if "query_columns" in kwargs:
            values.append(("query_columns", kwargs["query_columns"]))
        supplied = _digest_values(values)
        controls = (tuple(tree_keys), tuple(_tree_digest(t) for t in trees),
                    supplied, json.dumps({k:v for k,v in kwargs.items()
                        if k != "query_columns"}, sort_keys=True))
        if key in self.values:
            record = self.values[key]
            require(record["controls"] == controls
                    and record["bank_digest"] == _bank_digest(record["bank"]),
                    "Immutable block source/config/feature drift")
            self.hits += 1
            self.values.move_to_end(key)
            return record["bank"], dict(record["receipt"])
        bank, receipt = encode_block_prepared_v1(trees, intervals, genotype,
            allele_count, site_positions, reference_class, sample_age,
            tree_keys=tree_keys, layout_cache=self.layouts, **kwargs)
        # Cache owns independent storage so caller-owned input-array mutation
        # cannot silently mutate a prepared bank. Reuse also checks all bytes.
        bank = replace(bank, **{field.name:
            (tuple(x.clone() for x in getattr(bank, field.name))
             if isinstance(getattr(bank, field.name), tuple)
             else getattr(bank, field.name).clone()) for field in fields(bank)})
        size = sum(x.numel()*x.element_size() for _, x in _bank_items(bank))
        require(size <= self.max_tensor_bytes, "One complete CPU feature bank exceeds cap")
        while self.values and (len(self.values) >= self.max_entries
                               or self.bytes+size > self.max_tensor_bytes):
            _, old = self.values.popitem(last=False)
            self.bytes -= old["bytes"]
        self.values[key] = dict(controls=controls, bank=bank, receipt=dict(receipt),
                                bank_digest=_bank_digest(bank), bytes=size)
        self.bytes += size
        self.builds += 1
        return bank, dict(receipt)
