"""Stable level grouping in linear/radix work instead of repeated full scans."""
import numpy as np


def bucket_rows(keys, rows=None, *, levels=None):
    keys = np.asarray(keys)
    if keys.ndim != 1 or keys.dtype.kind not in "iu" or np.any(keys < 0):
        raise ValueError("Nonnegative integer level keys required")
    if rows is None:
        rows = np.arange(len(keys), dtype=np.int64)
    if len(rows) != len(keys):
        raise ValueError("One level key per row required")
    count = int(keys.max()) + 1 if len(keys) else 0
    levels = count if levels is None else levels
    if levels < count:
        raise ValueError("Declared level count omits rows")
    order = np.argsort(keys, kind="stable")
    grouped = rows[order]
    ends = np.cumsum(np.bincount(keys, minlength=levels), dtype=np.int64)
    return tuple(np.split(grouped, ends[:-1])) if levels else ()


def block_indices(parent, height, depth):
    """Preserve exact ascending-node order within every original tree level."""
    levels = bucket_rows(height)
    rank = np.empty(len(parent), dtype=np.int64)
    for nodes in levels:
        rank[nodes] = np.arange(len(nodes))
    child = np.flatnonzero(parent >= 0)
    children = bucket_rows(height[parent[child]], child, levels=len(levels))
    ranks = tuple(rank[parent[nodes]] for nodes in children)
    descendants = np.flatnonzero(depth > 0)
    downward = bucket_rows(depth[descendants]-1, descendants,
                           levels=int(depth.max()) if len(depth) else 0)
    return levels, children, ranks, downward
