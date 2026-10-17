"""Observation-only positive-cover plans and FP32 differentiable evidence.

No file readers, targets, training, resource policies or execution entrypoint.
This source has not been imported or evaluated on scientific/numerical inputs.
Positive real-arithmetic semantics follow the separately tested FP32 prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import (
    PackedParentCPUV2, PackedHoldCPUV2, safe_logadd, safe_logsum,
    segmented_exclusive_logs, stamp, require,
)

G, A = 8, 4


def positive_logsum(parts, like):
    if not parts:
        return torch.full_like(like, -torch.inf)
    return safe_logsum(torch.stack(parts), 0)


@dataclass(frozen=True)
class PositiveHoldPlanV2:
    native_stamp: tuple
    parent: tuple[int, ...]
    query_node: tuple[int, ...]
    site_start: int
    site_stop: int
    # Per site: (kind0=clade/1=singleton-reference, local node, called allele).
    covers: tuple[tuple[tuple[int, int, int], ...], ...]
    called_allele_count: torch.Tensor  # CPU int64 [S_hold,8,4]
    count_stamp: tuple
    prefer_dense: tuple[bool, ...]

    def inside(self, node, ancestor):
        while node >= 0:
            if node == ancestor:
                return True
            node = self.parent[node]
        return False

    def edge_path(self, left, right):
        seen, x = set(), left
        while x >= 0:
            seen.add(x)
            x = self.parent[x]
        ancestor = right
        while ancestor >= 0 and ancestor not in seen:
            ancestor = self.parent[ancestor]
        require(ancestor >= 0, 'A connected original native tree is required')
        edges = []
        for x in (left, right):
            while x != ancestor:
                edges.append(x)
                x = self.parent[x]
        return tuple(edges)

    def validate(self, hold):
        hold.verify_immutable()
        require(isinstance(hold, PackedHoldCPUV2)
            and stamp(hold.native_node) == self.native_stamp
            and hold.site_start == self.site_start and hold.site_stop == self.site_stop
            and stamp(self.called_allele_count) == self.count_stamp,
            'Positive cover plan no longer matches its exact immutable native hold')


@dataclass(frozen=True)
class PositiveParentPlanV2:
    route_identity: int
    route_source_stamps: tuple
    holds: tuple[PositiveHoldPlanV2, ...]

    def validate(self, route):
        require(isinstance(route, PackedParentCPUV2) and id(route) == self.route_identity
            and route.source_stamps == self.route_source_stamps
            and len(self.holds) == len(route.holds), 'Positive plans require their exact bound route')
        for plan, hold in zip(self.holds, route.holds):
            plan.validate(hold)


def prepare_positive_parent_v2(route, cpu_bank, *, runtime_check):
    """Future supplied-observation preparation only; never called at authoring.

    No values are synthesized, inferred truth is never accepted, every original
    site/reference is retained. CPU genotype reads and O(S*N) cover construction
    are additional costs. Caller authenticates the existing original bank first.
    """
    require(isinstance(route, PackedParentCPUV2) and cpu_bank.genotype.device.type == 'cpu',
        'Original CPU bank and existing exact route are required')
    require(callable(runtime_check), 'Existing ROOT boundary callback required')
    source_stamps = dict(route.source_stamps)
    for name in ('parent', 'sample_index', 'site_hold', 'reference_class', 'sample_age', 'query_column', 'genotype'):
        require(source_stamps.get(name) == stamp(getattr(cpu_bank, name)),
            'Positive plan attempted to use a different original CPU observation: '+name)
    plans = []
    for hold in route.holds:
        runtime_check()
        hold.verify_immutable()
        parent = tuple(hold.parent.tolist())
        n = len(parent)
        child_lists = [[] for _ in range(n)]
        for v, parent_v in enumerate(parent):
            if parent_v >= 0:
                child_lists[parent_v].append(v)
        children = tuple(tuple(row) for row in child_lists)
        roots = tuple(v for v, p in enumerate(parent) if p < 0)
        require(len(roots) == 1, 'No synthetic reconnection of a native tree is permitted')
        order = []
        stack = list(roots)
        depths = [0]*n
        while stack:
            v = stack.pop()
            order.append(v)
            for u in children[v]:
                depths[u] = depths[v]+1
                stack.append(u)
        require(len(order) == n, 'Complete connected native tree required')
        refs = tuple(hold.reference_column.tolist())
        refnodes = tuple(hold.reference_node.tolist())
        refgroups = tuple(hold.reference_group.tolist())
        query = tuple(hold.query_node.tolist())
        permutation = hold.route_permutation.tolist()
        # Only existing known references seed labels. Query genotypes are not
        # used to choose covers and remain in their original columns at readout.
        counts = torch.zeros((hold.site_stop-hold.site_start,G,A),dtype=torch.int64)
        covers, preferred = [], []
        for local, s in enumerate(range(hold.site_start,hold.site_stop)):
            if local % 16 == 0:runtime_check()
            labels = {}
            for ref, node, group in zip(refs,refnodes,refgroups):
                allele = int(cpu_bank.genotype[s,permutation[ref]])
                require(-1 <= allele < A and 0 <= group < G, 'Observed call/group bounds differ')
                labels[node] = allele
                if allele >= 0:
                    counts[local,group,allele] += 1
            symbols = [set() for _ in range(n)]
            for v in reversed(order):
                if v in labels:
                    symbols[v].add(labels[v])
                for u in children[v]:
                    symbols[v].update(symbols[u])
            pieces = []
            stack = list(roots)
            while stack:
                v = stack.pop()
                if not symbols[v]:
                    continue
                if len(symbols[v]) == 1:
                    allele = next(iter(symbols[v]))
                    if allele >= 0:
                        pieces.append((0,v,allele))
                    continue
                if v in labels and labels[v] >= 0:
                    pieces.append((1,v,labels[v]))
                stack.extend(children[v])
            # Explicit structural work proxy; not a measurement/speed guarantee.
            use_dense = len(pieces)*sum(depths[q]+1 for q in query) >= n
            # A forced-dense site uses every original call directly. Its unused
            # cover is not retained; this drops no observation or neural step.
            covers.append(() if use_dense else tuple(pieces))
            preferred.append(use_dense)
        plans.append(PositiveHoldPlanV2(stamp(hold.native_node),parent,query,hold.site_start,hold.site_stop,
            tuple(covers),counts,stamp(counts),tuple(preferred)))
    return PositiveParentPlanV2(id(route),route.source_stamps,tuple(plans))


class PositiveMarkerBackendV2:
    """Parameter-free stable kernel; dense branch reuses actual H058 PyG maths.

    Structural loops are correctness-oriented, not an efficient dynamic-tree
    implementation. Dense fallback keeps every call and uses the identical gate.
    """
    @staticmethod
    def _group_fields(log_gamma, view, dense):
        # Include all references, even calls later missing at a particular SNP.
        seed = log_gamma.new_full((view.nodes,G),-torch.inf)
        ref = torch.where(view.group_onehot.bool(),log_gamma.new_zeros(()),
            log_gamma.new_full((),-torch.inf))
        seed = seed.index_copy(0,view.reference_node,ref)
        up = torch.full_like(seed,-torch.inf)
        for ns,ne,es,ee,_degree in view.level_slices:
            nodes = view.level_node[ns:ne]
            child,rank = view.level_child[es:ee],view.level_parent_rank[es:ee]
            values = up.index_select(0,child)+log_gamma.index_select(0,child)[:,None]
            aggregate = dense.segment_logsum(values,rank,ne-ns)
            up = up.index_copy(0,nodes,safe_logadd(seed.index_select(0,nodes),aggregate))
        side = torch.full_like(seed,-torch.inf)
        for ns,ne,es,ee,degree in view.level_slices:
            if es == ee:
                continue
            nodes = view.level_node[ns:ne]
            child,rank = view.level_child[es:ee],view.level_parent_rank[es:ee]
            values = up.index_select(0,child)+log_gamma.index_select(0,child)[:,None]
            left,right = segmented_exclusive_logs(values,rank,degree)
            siblings = safe_logadd(left,right)
            local = seed.index_select(0,nodes).index_select(0,rank)
            side = side.index_copy(0,child,safe_logadd(local,siblings))
        return seed,up,side

    @staticmethod
    def _path(log_gamma, plan, q, v):
        indices = plan.edge_path(q,v)
        if not indices:
            return log_gamma.new_zeros(())
        index = torch.tensor(indices,dtype=torch.long,device=log_gamma.device)
        return log_gamma.index_select(0,index).sum()

    def _cut(self,log_gamma,plan,up,side,q,v):
        empty = log_gamma.new_full((G,),-torch.inf)
        if plan.inside(v,q):
            return empty
        if not plan.inside(q,v):
            return self._path(log_gamma,plan,q,v)+up[v]
        # Positive affine path composition; no outside-mass subtraction.
        child = q
        distance = log_gamma.new_zeros(())
        parts = []
        while child != v:
            distance = distance+log_gamma[child]
            parts.append(distance+side[child])
            child = plan.parent[child]
        return positive_logsum(parts,empty)

    def _positive_log_fields(self,log_gamma,plan,fields,first,last):
        seed,up,side = fields
        empty = log_gamma.new_full((G,),-torch.inf)
        sites = []
        for pieces in plan.covers[first:last]:
            queries = []
            for q in plan.query_node:
                terms = [[] for _ in range(A)]
                for kind,v,allele in pieces:
                    if kind == 0:
                        value = self._cut(log_gamma,plan,up,side,q,v)
                    else:
                        value = empty if plan.inside(v,q) else self._path(log_gamma,plan,q,v)+seed[v]
                    terms[allele].append(value)
                queries.append(torch.stack([positive_logsum(x,empty) for x in terms],-1))
            sites.append(torch.stack(queries))
        return torch.stack(sites)

    @staticmethod
    def _features(log_n,query,counts):
        called = counts.sum(-1)
        log_d = safe_logsum(log_n,-1)
        choose = query.long().clamp_min(0)[:,:,None,None].expand(-1,-1,G,1)
        match = log_n.gather(-1,choose).squeeze(-1)
        valid = (query >= 0)[:,:,None] & (called > 0)[:,None,:]
        safe_d = torch.where(torch.isfinite(log_d),log_d,torch.zeros_like(log_d))
        matching = (match-safe_d).exp()
        frequency = counts/called.clamp_min(1)[...,None]
        ordinary = frequency[:,None].expand(-1,query.shape[1],-1,-1).gather(-1,choose).squeeze(-1)
        contrast = torch.where(valid,matching-ordinary,torch.zeros_like(matching))
        mass = (safe_d-called.clamp_min(1).log()[:,None]).clamp_max(0)
        mass = torch.where(valid,mass,torch.zeros_like(mass))
        feature = torch.cat((contrast,mass,valid.to(log_n.dtype)),-1)
        mask = torch.cat((valid,valid,(query >= 0)[:,:,None].expand_as(valid)),-1)
        return feature,mask

    def marker_block_v2(self,log_gamma,genotype,hold,plan,dense,*,first,last):
        """ONE original block only; caller checkpoints feature+consumer together.

        No marker sequence or learned group fields are retained across blocks.
        Dense fallback uses the actual original H058 feature algebra. Its full
        N*B*G*A graph is constructed inside the caller's <=16-marker checkpoint.
        """
        require(log_gamma.dtype == torch.float32
            and 0 <= first < last <= len(plan.covers) and last-first <= 16
            and genotype.shape == (last-first,hold.haplotypes),
            'Exact FP32 gate and complete original marker block required')
        plan.validate(hold)
        effective = log_gamma.clamp(-16.,-.0001)
        if any(plan.prefer_dense[first:last]):
            return dense.marker_features_v2(effective,genotype,hold)
        view = hold._transient_device_view(log_gamma.device)
        fields = self._group_fields(effective,view,dense)
        log_n = self._positive_log_fields(effective,plan,fields,first,last)
        counts = plan.called_allele_count[first:last].to(device=log_n.device,dtype=torch.float32)
        query = genotype.index_select(1,view.query_column)
        return self._features(log_n,query,counts)
