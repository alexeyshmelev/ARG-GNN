"""Synthetic-input-only FP32 path-kernel feasibility prototype.

No dataset/file readers, training, optimizer, CUDA, model or scientific inference.
Actual H058 source is authenticated by the separate synthetic check script.
A small-tree reference implementation; no production complexity/speed claim.
"""
from dataclasses import dataclass
import torch

F32 = torch.float32
G, A, MISSING = 8, 4, 4

@dataclass(frozen=True)
class ToyTree:
    parent: tuple
    sample_node: tuple
    reference_class: tuple
    sample_age: tuple
    query_column: tuple
    route_permutation: tuple

    def __post_init__(self):
        n, h = len(self.parent), len(self.sample_node)
        assert self.parent[0] == -1
        assert all(0 <= self.parent[v] < v for v in range(1, n))
        assert len(set(self.sample_node)) == h
        assert len(self.reference_class) == len(self.sample_age) == len(self.route_permutation) == h
        assert sorted(self.route_permutation) == list(range(h))
        assert all(self.reference_class[q] == -1 for q in self.query_column)
        for j, c in enumerate(self.reference_class):
            k = self.route_permutation[j]
            assert c in range(-1, 4)
            if c < 0:
                assert j == k
            else:
                assert self.reference_class[k] == c
                assert (self.sample_age[k] >= 350) == (self.sample_age[j] >= 350)

    @property
    def n(self): return len(self.parent)

    @property
    def children(self):
        return tuple(tuple(v for v, p in enumerate(self.parent) if p == u) for u in range(self.n))

    @property
    def references(self):
        return tuple(j for j, c in enumerate(self.reference_class) if c >= 0)

    def group(self, sample):
        return self.reference_class[sample] + 4 * int(self.sample_age[sample] >= 350)

    def inside(self, node, ancestor):
        while node >= 0:
            if node == ancestor: return True
            node = self.parent[node]
        return False


def fp32(x):
    assert x.dtype == F32 and x.device.type == 'cpu'


def logsum(parts, like):
    """All-empty-safe, differentiable positive log-domain sum."""
    if not parts: return torch.full_like(like, -torch.inf)
    x = torch.stack(parts)
    active = torch.isfinite(x).any(0)
    safe = torch.where(active.unsqueeze(0), x, torch.zeros_like(x))
    value = torch.logsumexp(safe, 0)
    return torch.where(active, value, torch.full_like(value, -torch.inf))


def pathlog(tree, lg, left, right):
    """Explicit small-tree path, avoiding root-prefix subtraction."""
    ancestors, u = set(), left
    while u >= 0:
        ancestors.add(u); u = tree.parent[u]
    v = right
    while v not in ancestors: v = tree.parent[v]
    lca, edges = v, []
    for node in (left, right):
        while node != lca:
            edges.append(lg[node]); node = tree.parent[node]
    return torch.stack(edges).sum() if edges else lg.new_zeros(())


def assigned_reference(tree, genotype):
    assert genotype.dtype == torch.int64 and genotype.ndim == 2
    assert genotype.shape[1] == len(tree.sample_node)
    assert bool(((genotype >= -1) & (genotype < A)).all())
    return genotype[:, list(tree.route_permutation)][:, list(tree.references)]


def group_fields(tree, lg):
    """Site-independent positive group masses, all source/age references."""
    fp32(lg)
    lg = lg.clamp(-16., -.0001)
    empty = lg.new_full((G,), -torch.inf)
    own = [empty.clone() for _ in tree.parent]
    for ref in tree.references:
        node, group = tree.sample_node[ref], tree.group(ref)
        own[node] = torch.where(torch.arange(G) == group, lg.new_zeros((G,)), empty)
    children = tree.children
    up = [None] * tree.n
    for v in reversed(range(tree.n)):
        up[v] = logsum([own[v]] + [lg[u] + up[u] for u in children[v]], empty)
    outside = [empty]
    for u in range(1, tree.n):
        v = tree.parent[u]
        cavity = [own[v], outside[v]] + [lg[s] + up[s] for s in children[v] if s != u]
        outside.append(lg[u] + logsum(cavity, empty))
    return lg, own, up, outside


def cut_log_positive(tree, fields, query, root):
    """Stable C(q,v): references below v but outside the query subtree.

    Strict ancestor case sums positive sibling/local contributions along q->v.
    This explicit prototype walks the path; binary-lift acceleration is absent.
    """
    lg, own, up, outside = fields
    empty = lg.new_full((G,), -torch.inf)
    if tree.inside(root, query): return empty
    if not tree.inside(query, root): return pathlog(tree, lg, query, root) + up[root]
    parts, child = [], query
    while child != root:
        parent = tree.parent[child]
        local = [own[parent]] + [lg[s] + up[s] for s in tree.children[parent] if s != child]
        parts.append(pathlog(tree, lg, query, parent) + logsum(local, empty))
        child = parent
    return logsum(parts, empty)


def dense_log_fields(tree, lg, genotype):
    """Independent positive log-message oracle matching H058 outside support."""
    fp32(lg)
    lg = lg.clamp(-16., -.0001)
    refallele = assigned_reference(tree, genotype)
    b = genotype.shape[0]
    empty = lg.new_full((b, G, A), -torch.inf)
    own = [empty.clone() for _ in tree.parent]
    for col, ref in enumerate(tree.references):
        allele = refallele[:, col]
        hot = torch.nn.functional.one_hot(allele.clamp_min(0), A).bool() & (allele >= 0)[:, None]
        mask = hot[:, None, :] & (torch.arange(G) == tree.group(ref))[None, :, None]
        own[tree.sample_node[ref]] = torch.where(mask, lg.new_zeros(()), empty)
    children, up = tree.children, [None] * tree.n
    for v in reversed(range(tree.n)):
        up[v] = logsum([own[v]] + [lg[u] + up[u] for u in children[v]], empty)
    outside = [empty]
    for u in range(1, tree.n):
        v = tree.parent[u]
        cavity = [own[v], outside[v]] + [lg[s] + up[s] for s in children[v] if s != u]
        outside.append(lg[u] + logsum(cavity, empty))
    return torch.stack([outside[tree.sample_node[q]] for q in tree.query_column], 1)


def features_h058(tree, log_n, genotype):
    """Preserves all-reference counts, original query genotype and empty-D quirk."""
    fp32(log_n)
    refs = assigned_reference(tree, genotype)
    counts = log_n.new_zeros((genotype.shape[0], G, A))
    for col, ref in enumerate(tree.references):
        allele = refs[:, col]
        hot = torch.nn.functional.one_hot(allele.clamp_min(0), A).to(F32) * (allele >= 0)[:, None]
        group_hot = torch.nn.functional.one_hot(torch.tensor(tree.group(ref)), G).to(F32)
        counts = counts + hot[:, None, :] * group_hot[None, :, None]
    called = counts.sum(-1)
    log_d = logsum([log_n[..., a] for a in range(A)], log_n[..., 0])
    query = genotype[:, list(tree.query_column)]
    choose = query.clamp_min(0)[:, :, None, None].expand(-1, -1, G, 1)
    match = log_n.gather(-1, choose).squeeze(-1)
    valid = (query >= 0)[:, :, None] & (called > 0)[:, None, :]
    safe_d = torch.where(torch.isfinite(log_d), log_d, torch.zeros_like(log_d))
    matching = (match - safe_d).exp()
    frequency = counts / called.clamp_min(1)[..., None]
    ordinary = frequency[:, None].expand(-1, query.shape[1], -1, -1).gather(-1, choose).squeeze(-1)
    contrast = torch.where(valid, matching - ordinary, torch.zeros_like(matching))
    mass = (safe_d - called.clamp_min(1).log()[:, None]).clamp_max(0)
    mass = torch.where(valid, mass, torch.zeros_like(mass))
    features = torch.cat((contrast, mass, valid.to(F32)), -1)
    mask = torch.cat((valid, valid, (query >= 0)[:, :, None].expand_as(valid)), -1)
    return features, mask


def signed_event_fields(tree, lg, genotype, symbols):
    """Diagnostic original signed formula. Does not claim FP32 stability."""
    fields = group_fields(tree, lg)
    effective, _, up, outside = fields
    assert len(symbols) == genotype.shape[0]
    result, abs_sums, event_counts = [], [], []
    basis = torch.cat((torch.eye(A, dtype=F32), torch.zeros((1, A), dtype=F32)))
    for row, labels in enumerate(symbols):
        assert len(labels) == tree.n and all(a in range(5) for a in labels)
        for ref in tree.references:
            expected = genotype[row, tree.route_permutation[ref]].item()
            assert labels[tree.sample_node[ref]] == (expected if expected >= 0 else MISSING)
        events = [v for v in range(1, tree.n) if labels[v] != labels[tree.parent[v]]]
        values, absolute = [], []
        for qcol in tree.query_column:
            q = tree.sample_node[qcol]
            terms = [outside[q].exp()[:, None] * basis[labels[0]][None, :]]
            for v in events:
                if tree.inside(v, q):
                    c = effective.new_zeros((G,))
                elif tree.inside(q, v):
                    c = outside[q].exp() - (pathlog(tree, effective, q, v) + outside[v]).exp()
                else:
                    c = (pathlog(tree, effective, q, v) + up[v]).exp()
                delta = basis[labels[v]] - basis[labels[tree.parent[v]]]
                terms.append(c[:, None] * delta[None, :])
            stacked = torch.stack(terms)
            values.append(stacked.sum(0)); absolute.append(stacked.abs().sum(0))
        result.append(torch.stack(values)); abs_sums.append(torch.stack(absolute)); event_counts.append(len(events))
    return torch.stack(result), torch.stack(abs_sums), event_counts


def positive_cover(tree, genotype, row):
    """Exact positive cover of reference calls; no query-label/genotype input.

    Maximal homogeneous reference subtrees plus singleton internal references.
    Producing this from raw calls is O(N) per site in this toy implementation.
    """
    labels = {}
    for ref in tree.references:
        symbol = genotype[row, tree.route_permutation[ref]].item()
        labels[tree.sample_node[ref]] = symbol if symbol >= 0 else MISSING
    subsets = [set() for _ in tree.parent]
    for v in reversed(range(tree.n)):
        if v in labels: subsets[v].add(labels[v])
        for u in tree.children[v]: subsets[v].update(subsets[u])
    pieces = []
    def visit(v):
        if not subsets[v]: return
        if len(subsets[v]) == 1:
            symbol = next(iter(subsets[v]))
            if symbol != MISSING: pieces.append(('clade', v, symbol))
            return
        if v in labels and labels[v] != MISSING: pieces.append(('atom', v, labels[v]))
        for u in tree.children[v]: visit(u)
    visit(0)
    return tuple(pieces)


def positive_cover_fields(tree, lg, genotype, covers=None):
    """Stable sparse correction: positive log sums, never signed allele totals."""
    fields = group_fields(tree, lg)
    effective, own, _, _ = fields
    covers = tuple(positive_cover(tree, genotype, s) for s in range(genotype.shape[0])) if covers is None else covers
    sites = []
    empty = lg.new_full((G,), -torch.inf)
    for pieces in covers:
        queries = []
        for qcol in tree.query_column:
            q, terms = tree.sample_node[qcol], [[] for _ in range(A)]
            for kind, v, symbol in pieces:
                if kind == 'clade':
                    contribution = cut_log_positive(tree, fields, q, v)
                else:
                    contribution = empty if tree.inside(v, q) else pathlog(tree, effective, q, v) + own[v]
                terms[symbol].append(contribution)
            queries.append(torch.stack([logsum(part, empty) for part in terms], -1))
        sites.append(torch.stack(queries))
    return torch.stack(sites), covers


def hybrid_fields(tree, lg, genotype):
    """Explicit structural cost fallback; no speculative performance guarantee.

    Conservative toy operation proxy only. Real implementation needs profiling.
    Choosing a branch from observed cover topology does not detach gate gradients.
    """
    covers = tuple(positive_cover(tree, genotype, s) for s in range(genotype.shape[0]))
    depth = []
    for qcol in tree.query_column:
        q, d = tree.sample_node[qcol], 0
        while q >= 0: d += 1; q = tree.parent[q]
        depth.append(d)
    fallback, sites = [], []
    for s, pieces in enumerate(covers):
        use_dense = len(pieces) * sum(depth) >= tree.n
        if use_dense:
            value = dense_log_fields(tree, lg, genotype[s:s+1])
        else:
            value, _ = positive_cover_fields(tree, lg, genotype[s:s+1], (pieces,))
        fallback.append(use_dense); sites.append(value[0])
    return torch.stack(sites), {'dense_fallback_sites': fallback, 'positive_piece_counts': [len(x) for x in covers], 'proxy_only': True}
