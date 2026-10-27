"""CPU algebra proof for a changed TreeLoom research operator.

No observed files, truth, GPU, training, or live model imports. Pointer jumping
is a deliberately simple O(N d log depth) reference schedule, not a fast kernel.
A future fused contraction implementation requires independent qualification.
"""
from __future__ import annotations
import torch


def parent_jumps(parent):
    """Topology-only immutable routing, permitting forests and arbitrary IDs."""
    if parent.device.type != 'cpu' or parent.dtype != torch.long:
        raise ValueError('CPU signed parent routing required')
    n = parent.numel()
    if n == 0 or bool(((parent < -1) | (parent >= n)).any()):
        raise ValueError('Nonempty bounded parent forest required')
    jumps = []
    p = parent.clone()
    for _ in range(n.bit_length() + 1):
        if not bool((p >= 0).any()):
            return tuple(jumps)
        jumps.append(p)
        valid = p >= 0
        p = torch.where(valid, p[p.clamp_min(0)], -torch.ones_like(p))
    raise ValueError('Parent routing contains a cycle')


def tree_transport(parent, edge_decay, values, jumps=None):
    """Return K values, where K_uv is the channelwise path-product kernel.

    edge_decay[v] is the decay on (v,parent[v]); root values are ignored.
    Every node contributes and every original tree edge remains active. The
    recursion is linear in values but nonlinear in observation-driven gates.
    """
    if edge_decay.shape != values.shape or values.ndim != 2:
        raise ValueError('Matching [nodes, channels] decay and value arrays required')
    jumps = parent_jumps(parent) if jumps is None else jumps
    up, a = values, edge_decay
    for p in jumps:
        valid = p >= 0
        ids = valid.nonzero(as_tuple=True)[0]
        up = up.index_add(0, p[ids], a[ids] * up[ids])
        safe = p.clamp_min(0)
        a = a * a[safe] * valid[:, None]
    nonroot = parent >= 0
    safe_parent = parent.clamp_min(0)
    # Recipient-excluded rerooting, so each source is counted exactly once.
    down = edge_decay * (up[safe_parent] - edge_decay * up)
    down = down * nonroot[:, None]
    a = edge_decay
    for p in jumps:
        valid = p >= 0
        safe = p.clamp_min(0)
        down = down + a * down[safe] * valid[:, None]
        a = a * a[safe] * valid[:, None]
    return up + down


def normalized_tree_transport(parent, edge_decay, values, jumps=None):
    numerator = tree_transport(parent, edge_decay, values, jumps)
    denominator = tree_transport(parent, edge_decay, torch.ones_like(values), jumps)
    return numerator / denominator


def explicit_tree_kernel(parent, edge_decay):
    """Small differentiable all-pairs reference, not a production algorithm."""
    n, d = edge_decay.shape
    adjacency = [[] for _ in range(n)]
    for child, par in enumerate(parent.tolist()):
        if par >= 0:
            adjacency[par].append((child, child))
            adjacency[child].append((par, child))
    rows = []
    for source in range(n):
        weights = {source: torch.ones(d, dtype=edge_decay.dtype)}
        stack = [(source, -1)]
        while stack:
            node, previous = stack.pop()
            for other, edge in adjacency[node]:
                if other == previous:
                    continue
                weights[other] = weights[node] * edge_decay[edge]
                stack.append((other, node))
        rows.append(torch.stack([weights.get(v, torch.zeros(d, dtype=edge_decay.dtype)) for v in range(n)]))
    return torch.stack(rows)


def affine_scan(alpha, beta, initial=None):
    """Inclusive parallel composition (a2,b2)o(a1,b1)=(a2*a1,b2+a2*b1)."""
    if alpha.shape != beta.shape or beta.ndim != 3:
        raise ValueError('Matching [intervals, samples, channels] arrays required')
    a, b = alpha, beta
    offset = 1
    while offset < len(beta):
        old_a, old_b = a, b
        a = torch.cat((old_a[:offset], old_a[offset:] * old_a[:-offset]), 0)
        b = torch.cat((old_b[:offset], old_b[offset:] + old_a[offset:] * old_b[:-offset]), 0)
        offset *= 2
    return b if initial is None else b + a * initial


def distance_scan(values, widths, rate, initial=None):
    """Exact piecewise-constant forcing for dh/dx = rate * (value - h)."""
    exponent = -widths[:, None, None] * rate
    alpha = torch.exp(exponent).expand_as(values)
    beta = -torch.expm1(exponent) * values
    return affine_scan(alpha, beta, initial)
