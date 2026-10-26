"""Exact replacement structural primitives. No simulation or model changes.

C++ source is built explicitly with build.sh. No runtime compiler or generated
source is executed. ctypes releases the GIL while these integer kernels run.
"""
from pathlib import Path
import ctypes
import sys
import numpy as np
from arggnn.mutation_routed_tree_observations_v1 import _integer
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.observations_v1 import (
    _layout_with_isolated_samples as _original_layout)
from research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.structural_index_v1 import (
    classify_sample_roots, validate_root_partition, isolated_clades)

_LIB = ctypes.CDLL(str(Path(__file__).with_name('libstructural.so')))
_I64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
_I32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
_U64 = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=2, flags='C_CONTIGUOUS')
_L = ctypes.c_int64
_LIB.tree_layout.argtypes = [_I64, _L, _I64, _L, _I64, _I64, _I64, _I64]
_LIB.forest_index.argtypes = [_I64, _L, _I32, _I32, _I32, _I64]
_LIB.tree_clades.argtypes = [_L, _I64, _L, _I32, _I32, _L, _I64, _U64, _I64]
for _name in ('tree_layout', 'forest_index', 'tree_clades'):
    getattr(_LIB, _name).restype = ctypes.c_int


def _layout_with_isolated_samples(tree):
    raw = np.asarray(tree.parent)
    if raw.ndim == 1 and raw.dtype.kind in 'iu' and len(raw) > 1 and np.all(raw == -1):
        return _original_layout(tree)
    p = _integer(tree.parent, 'parent').astype(np.int64, copy=False)
    t = np.asarray(tree.node_time, dtype=np.float64)
    sn = _integer(tree.sample_node, 'sample_node').astype(np.int64, copy=False)
    sid = _integer(tree.sample_id, 'sample_id').astype(np.int64, copy=False)
    n = len(p)
    if (not n or t.shape != (n,) or not np.isfinite(t).all() or np.any(t < 0)
            or np.any(p < -1) or np.any(p >= n) or np.count_nonzero(p == -1) != 1
            or not len(sn) or sid.shape != sn.shape or np.any(sn < 0) or np.any(sn >= n)
            or np.any(sid < 0) or len(set(sn.tolist())) != len(sn) or len(set(sid.tolist())) != len(sid)):
        raise ValueError('One complete observed native tree and unique sampled identities required')
    edge = np.flatnonzero(p >= 0)
    duration = np.zeros(n, dtype=np.float64)
    duration[edge] = t[p[edge]] - t[edge]
    if np.any(duration[edge] <= 0):
        raise ValueError('Strictly positive native endpoint durations required')
    post = np.empty(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    height = np.zeros(n, dtype=np.int64)
    desc = np.zeros(n, dtype=np.int64)
    status = _LIB.tree_layout(np.ascontiguousarray(p), n, np.ascontiguousarray(sn), len(sn), post, depth, height, desc)
    if status == 1:
        raise ValueError('Disconnected or cyclic administrative tree rows are forbidden')
    if status == 2:
        raise ValueError('Only the complete sampled-ancestor union may be retained')
    if status:
        raise RuntimeError(f'CPU structural kernel failed with status {status}')
    return p, t, sn, sid, post, depth, height, desc, duration


def _level_arrays(level):
    """Stable vectorized bucketing, retaining ascending native indexes per level."""
    level = np.asarray(level)
    count = int(level.max(initial=0)) + 1
    order = np.argsort(level, kind='stable').astype(np.int64, copy=False)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(level, minlength=count))))
    return tuple(order[lo:hi].copy() for lo, hi in zip(offsets[:-1], offsets[1:]))


def child_levels(parent, height):
    """Same ascending child index in each parent-height bucket."""
    child = np.flatnonzero(parent >= 0)
    rank = np.empty(len(parent), dtype=np.int64)
    levels = _level_arrays(height)
    for nodes in levels:
        rank[nodes] = np.arange(len(nodes), dtype=np.int64)
    order = np.argsort(height[parent[child]], kind='stable')
    grouped = child[order]
    offsets = np.concatenate(([0], np.cumsum(np.bincount(height[parent[child]], minlength=len(levels)))))
    children = tuple(grouped[lo:hi].copy() for lo, hi in zip(offsets[:-1], offsets[1:]))
    return levels, children, tuple(rank[parent[nodes]] for nodes in children)


def _forest_index(parent, samples, *, isolated_hold_indices=()):
    parent = np.ascontiguousarray(parent, dtype=np.int64)
    n = len(parent)
    if parent.ndim != 1 or n <= 0 or n >= 2**31 or np.any(parent < -1) or np.any(parent >= n):
        raise ValueError('Bounded parent forest required for compiled kernel')
    head = np.full(n, -1, dtype=np.int32)
    sibling = np.full(n, -1, dtype=np.int32)
    roots = np.full(n, -2, dtype=np.int32)
    steps = np.zeros(1, dtype=np.int64)
    status = _LIB.forest_index(parent, n, head, sibling, roots, steps)
    if status == 1:
        raise ValueError('Acyclic observed parent forest required')
    if status:
        raise RuntimeError(f'CPU forest kernel failed with status {status}')
    ordered = [classify_sample_roots(parent, head, roots, row) for row in samples]
    validate_root_partition(ordered, np.flatnonzero(parent < 0), isolated_hold_indices)
    return head, sibling, ordered, int(steps[0])


def _tree_clades(root, row, parent, head, sibling):
    if isinstance(root, tuple):
        return isolated_clades(root, row, parent, head)
    row = np.ascontiguousarray(row, dtype=np.int64)
    h = len(row)
    if (row.ndim!=1 or h<1 or not 0<=root<len(parent)
            or np.asarray(head).shape!=(len(parent),)
            or np.asarray(sibling).shape!=(len(parent),)):
        raise ValueError('Exact bounded tree root and adjacency shapes required')
    # A rooted tree with H leaves has at most H-2 distinct proper clades.
    # True unary vertices remain visited, but reuse their deepest representative.
    nodes = np.empty(h, dtype=np.int64)
    masks = np.empty((h, (h + 63) // 64), dtype=np.uint64)
    result = np.empty(3, dtype=np.int64)
    status = _LIB.tree_clades(root, row, h, np.ascontiguousarray(head, dtype=np.int32),
        np.ascontiguousarray(sibling, dtype=np.int32), len(parent), nodes, masks, result)
    errors = {1: 'Distinct sampled leaves in every native tree required',
              2: 'Acyclic observed tree required',
              3: 'Every native terminal node must be in the sampled panel',
              4: 'Sampled-interior cases are unsupported',
              5: 'Each native tree must contain exactly the full sampled panel'}
    if status in errors:
        raise ValueError(errors[status])
    if status:
        raise RuntimeError(f'CPU clade kernel failed with status {status}')
    count, visited, unary = map(int, result)
    clades = {int.from_bytes(masks[i].tobytes(), sys.byteorder): int(nodes[i]) for i in range(count)}
    return clades, visited, unary

_F32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
_U8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS')
_LIB.cut_index.argtypes = [_I64, _F32, _L, _I64, _L, _I64, _I64, _I32, _I64]
_LIB.cut_groups.argtypes = [_I64, _F32, _I64, _L, _I64, _U8, _L, _L, _I64, _I64, _I64, _I64, _U8, _I64]
_LIB.direction_layouts.argtypes = [_I64, _L, _L, ctypes.c_int, _I64, _I64, _I64, _F32]
for _name in ('cut_index', 'cut_groups', 'direction_layouts'):
    getattr(_LIB, _name).restype = ctypes.c_int


def cut_index(parent, times, roots):
    n = len(parent)
    if (np.asarray(parent).shape!=np.asarray(times).shape or np.asarray(parent).ndim!=1
            or np.asarray(roots).ndim!=1):
        raise ValueError('Exact cut node/time/root shapes required')
    head = np.full(n, -1, dtype=np.int64)
    sibling = np.full(n, -1, dtype=np.int64)
    owner = np.full(n, -1, dtype=np.int32)
    cut = np.full(n, -1, dtype=np.int64)
    status = _LIB.cut_index(np.ascontiguousarray(parent), np.ascontiguousarray(times), n,
        np.ascontiguousarray(roots), len(roots), head, sibling, owner, cut)
    if status:
        raise ValueError(f'Complete acyclic rooted cut traversal required, status {status}')
    return head, sibling, owner, cut


def cut_groups(parent, times, cut, samples, above):
    t, h = samples.shape
    if (np.asarray(parent).shape!=np.asarray(times).shape or np.asarray(parent).shape!=np.asarray(cut).shape
            or np.asarray(above).shape!=(h,) or np.asarray(parent).ndim!=1):
        raise ValueError('Exact cut parent/time/group shapes required')
    local = np.empty((t, h), dtype=np.int64)
    global_groups = np.empty_like(local)
    reps = np.empty(t*h, dtype=np.int64)
    counts = np.empty(t*h, dtype=np.int64)
    flags = np.empty(t*h, dtype=np.uint8)
    offsets = np.empty(t+1, dtype=np.int64)
    status = _LIB.cut_groups(np.ascontiguousarray(parent), np.ascontiguousarray(times),
        np.ascontiguousarray(cut), len(parent), np.ascontiguousarray(samples).reshape(-1),
        np.ascontiguousarray(above, dtype=np.uint8), t, h, local.reshape(-1),
        global_groups.reshape(-1), reps, counts, flags, offsets)
    if status:
        raise RuntimeError(f'CPU cut-group kernel failed with status {status}')
    total = int(offsets[-1])
    return reps[:total].copy(), counts[:total].copy(), flags[:total].astype(bool), offsets.tolist(), local, global_groups


def _direction_layouts(groups, order, h):
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    g = len(groups)
    if groups.shape!=(g,h) or g<1 or h<1:
        raise ValueError('Exact nonempty direction-layout group shapes required')
    order = tuple(order)
    if order not in (tuple(range(g)), tuple(range(g-1, -1, -1))):
        raise ValueError('Original complete forward or reverse episode order required')
    cells = np.empty((g, h), dtype=np.int64)
    current = np.zeros((g, h), dtype=np.int64)
    old = np.zeros((g, h), dtype=np.int64)
    counts = np.zeros((g, h), dtype=np.float32)
    status = _LIB.direction_layouts(groups.reshape(-1), g, h, int(order[0] != 0),
        cells.reshape(-1), current.reshape(-1), old.reshape(-1), counts.reshape(-1))
    if status:
        raise RuntimeError(f'CPU direction-layout kernel failed with status {status}')
    return cells, current, old, counts



def validate_bank_levels(bank, parent):
    """Preserve the current recovery runtime's supplied-level rejection guards."""
    roots=np.flatnonzero(parent<0)
    owner=np.full(len(parent),-1,dtype=np.int32)
    owner[roots]=np.arange(len(roots),dtype=np.int32)
    for tensor in bank.downward_levels:
        nodes=tensor.detach().numpy()
        if not ((parent[nodes]>=0).all() and (owner[parent[nodes]]>=0).all()):
            raise ValueError("Existing depth levels must retain parent-before-child order")
        owner[nodes]=owner[parent[nodes]]
    if not (owner>=0).all():
        raise ValueError("All original nodes must be rooted and covered")
