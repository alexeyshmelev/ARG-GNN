"""CPU-qualified research operator; no input files, labels or training entrypoint.

Complete native trees, finite branches, exact shared-leaf/age overlap, and
synchronous PyG convolution. Integer internal IDs never align different trees.
"""
from dataclasses import dataclass
from collections import defaultdict
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing


@dataclass(frozen=True)
class Tree:
    parent: np.ndarray
    age: np.ndarray
    leaves: np.ndarray  # Native sample nodes ordered by sample_ids.
    sample_ids: tuple  # Stable observed identities used only for routing validation.

    def validate(self):
        p, a, leaves = self.parent, self.age, self.leaves
        if p.ndim != 1 or a.shape != p.shape or not np.issubdtype(p.dtype, np.integer):
            raise ValueError('Native parent and age vectors required')
        n=len(p)
        if not n or not np.isfinite(a).all() or (a<0).any() or ((p < -1) | (p >= n)).any():
            raise ValueError('Invalid parent or age values')
        if leaves.ndim != 1 or not np.issubdtype(leaves.dtype,np.integer) or len(set(leaves)) != len(leaves):
            raise ValueError('Unique integer terminal samples required')
        if not isinstance(self.sample_ids,tuple) or len(self.sample_ids)!=len(leaves) or len(set(self.sample_ids))!=len(leaves):
            raise ValueError('Explicit unique stable sample IDs required')
        if ((leaves<0)|(leaves>=n)).any():
            raise ValueError('Sample outside node range')
        terminals=set(range(n))-set(p[p>=0])
        if terminals != set(leaves):
            raise ValueError('All and only terminal nodes must be declared samples')
        for u in range(n):
            seen=set()
            while u>=0:
                if u in seen: raise ValueError('Cycle in native parent graph')
                seen.add(u)
                v=int(p[u])
                if v>=0 and a[v]<a[u]: raise ValueError('Parent younger than child')
                u=v
        return self

    def paths(self):
        result=[]
        for leaf in self.leaves:
            path=[];u=int(leaf)
            while self.parent[u]>=0:
                path.append(u);u=int(self.parent[u])
            result.append(path)
        return result

    def masses(self):
        mass=np.zeros(len(self.parent),dtype=np.float64)
        for path in self.paths():
            for u in path: mass[u]+=self.age[self.parent[u]]-self.age[u]
        return mass


def branch_overlap(target, source):
    """Return sparse unnormalized M[target_node,source_node] by exact path merge."""
    if target.sample_ids!=source.sample_ids or not np.array_equal(target.age[target.leaves],source.age[source.leaves]):
        raise ValueError('Shared sample roster and sampling ages required')
    out=defaultdict(float)
    for pa,pb in zip(target.paths(),source.paths()):
        i=j=0
        while i<len(pa) and j<len(pb):
            u,v=pa[i],pb[j]
            end_a=target.age[target.parent[u]];end_b=source.age[source.parent[v]]
            width=min(end_a,end_b)-max(target.age[u],source.age[v])
            if width>0: out[(u,v)]+=float(width)
            if end_a<=end_b: i+=1
            if end_b<=end_a: j+=1
    return dict(out)


@dataclass(frozen=True)
class Relation:
    edge_index: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class JointGraph:
    relations: tuple
    offsets: np.ndarray
    node_count: int
    branch_coverage: torch.Tensor  # [N,2], left/right available branch mass.

    def to(self, device):
        return JointGraph(tuple(Relation(r.edge_index.to(device),r.weight.to(device)) for r in self.relations),
                          self.offsets,self.node_count,self.branch_coverage.to(device))


def build_graph(trees, coordinates, radius, weights=None):
    """One finite coordinate window, no cross-window edges; six fixed relations.

    Relations: native up/down, leaf left/right, branch left/right.
    Every supplied tree stays complete. Distinct coordinates are required.
    """
    trees=tuple(t.validate() for t in trees)
    x=np.asarray(coordinates,dtype=np.float64)
    k=len(trees)
    w=np.ones(k) if weights is None else np.asarray(weights,dtype=np.float64)
    if not k or x.shape!=(k,) or w.shape!=(k,) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError('Finite coordinate and quadrature vectors required')
    if len(set(x))!=k or (w<0).any() or not math.isfinite(radius) or radius<=0:
        raise ValueError('Distinct coordinates, nonnegative weights and positive radius required')
    for tree in trees:
        if tree.sample_ids!=trees[0].sample_ids or not np.array_equal(tree.age[tree.leaves],trees[0].age[trees[0].leaves]):
            raise ValueError('Shared sample roster and sampling ages required')
    offsets=np.r_[0,np.cumsum([len(t.parent) for t in trees])]
    sources=[[] for _ in range(6)];targets=[[] for _ in range(6)];values=[[] for _ in range(6)]
    def add(rel,src,dst,value):
        sources[rel].append(int(src));targets[rel].append(int(dst));values[rel].append(float(value))
    for t,tree in enumerate(trees):
        p=tree.parent;off=offsets[t]
        degree=np.bincount(p[p>=0],minlength=len(p))
        for child,parent in enumerate(p):
            if parent>=0:
                add(0,off+child,off+parent,1./degree[parent]);add(1,off+parent,off+child,1.)
    masses=[t.masses() for t in trees]
    coverage=np.zeros((offsets[-1],2),dtype=np.float64)
    for t,target in enumerate(trees):
        delta=x-x[t]
        for side in range(2):
            select=(delta<0) if side==0 else (delta>0)
            kernel=np.maximum(0.,1.-np.abs(delta)/radius)*select*w
            denominator=kernel.sum()
            if denominator<=0: continue
            for s in np.flatnonzero(kernel>0):
                alpha=kernel[s]/denominator;source=trees[s]
                for u,v in zip(target.leaves,source.leaves):
                    add(2+side,offsets[s]+v,offsets[t]+u,alpha)
                for (u,v),overlap in branch_overlap(target,source).items():
                    value=alpha*overlap/masses[t][u]
                    add(4+side,offsets[s]+v,offsets[t]+u,value)
                    coverage[offsets[t]+u,side]+=value
    relations=tuple(Relation(torch.tensor([sources[r],targets[r]],dtype=torch.long),
                             torch.tensor(values[r],dtype=torch.float64)) for r in range(6))
    return JointGraph(relations,offsets,int(offsets[-1]),torch.from_numpy(coverage))


def disjoint_batch(graphs):
    """Combine already-defined independent windows without cross-example mixing."""
    cursor=0;idx=[[] for _ in range(6)];weight=[[] for _ in range(6)];coverage=[];offsets=[0]
    for graph in graphs:
        for r,relation in enumerate(graph.relations):
            idx[r].append(relation.edge_index+cursor);weight[r].append(relation.weight)
        coverage.append(graph.branch_coverage)
        offsets.extend((graph.offsets[1:]+cursor).tolist());cursor+=graph.node_count
    if not coverage: raise ValueError('Nonempty graphs required')
    return JointGraph(tuple(Relation(torch.cat(idx[r],1),torch.cat(weight[r])) for r in range(6)),
                      np.asarray(offsets),cursor,torch.cat(coverage))


class WeightedSum(MessagePassing):
    def __init__(self): super().__init__(aggr='add',node_dim=0)
    def forward(self,x,relation):
        return self.propagate(relation.edge_index,x=x,weight=relation.weight.to(x.dtype),size=(len(x),len(x)))
    def message(self,x_j,weight): return x_j*weight[:,None]


class BranchJointConv(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.self_map=nn.Linear(width,width)
        self.maps=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(6))
        self.norm=nn.LayerNorm(width)
        self.aggregate=WeightedSum()
    def forward(self,x,graph):
        if x.ndim!=2 or len(x)!=graph.node_count: raise ValueError('Native node feature matrix required')
        z=self.self_map(x)
        for linear,relation in zip(self.maps,graph.relations):
            z=z+self.aggregate(linear(x),relation)
        return self.norm(x+F.gelu(z))
