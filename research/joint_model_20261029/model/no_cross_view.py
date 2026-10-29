"""Exact immutable CPU view for the no_cross control only.

Discarding nonfocal tree occurrences is exact because no_cross has no inter-tree
messages, no batch normalization, and no trainable query marker. Construct once
per cached site before collation; do not reconstruct inside a device forward.
"""
from dataclasses import replace
import torch
try:
    from .joint_model_v1 import Batch, require, csr
except ImportError:
    from joint_model_v1 import Batch, require, csr


def _subset(matrix, rows, columns):
    """Filter an immutable sparse map, preserving numeric values and row order."""
    row_map=torch.full((matrix.size(0),),-1,dtype=torch.long)
    col_map=torch.full((matrix.size(1),),-1,dtype=torch.long)
    row_map[rows]=torch.arange(len(rows));col_map[columns]=torch.arange(len(columns))
    coo=matrix.to_sparse_coo();indices=coo.indices()
    rr=row_map[indices[0]];cc=col_map[indices[1]];keep=(rr>=0)&(cc>=0)
    return torch.sparse_coo_tensor(torch.stack((rr[keep],cc[keep])),coo.values()[keep],
                                  (len(rows),len(columns))).coalesce().to_sparse_csr()


def focal_only_no_cross(batch):
    """Return a complete-focal-tree Batch, valid exclusively with mode=no_cross.

    Focal observations and canonical target order are unchanged. All ancestors,
    leaves and SNPs of each distinct focal occurrence are retained. This helper
    never accepts ancestry labels and does not alter the pinned core model.
    """
    require(type(batch) is Batch and batch.parent.device.type=='cpu','CPU prepared Batch required')
    trees=torch.unique(batch.focal_tree,sorted=True)
    require(len(trees)>0,'At least one focal tree required')
    tree_map=torch.full((len(batch.leaves),),-1,dtype=torch.long)
    tree_map[trees]=torch.arange(len(trees))
    nodes=torch.nonzero(tree_map[batch.node_tree]>=0,as_tuple=False).flatten()
    node_map=torch.full((len(batch.parent),),-1,dtype=torch.long)
    node_map[nodes]=torch.arange(len(nodes))
    h=batch.leaves.shape[1]
    leaf_rows=(trees[:,None]*h+torch.arange(h)[None,:]).reshape(-1)
    sites=tree_map[batch.site_tree]>=0
    parent=batch.parent[nodes]
    parent=torch.where(parent>=0,node_map[parent.clamp_min(0)],parent)
    descendant=_subset(batch.descendant,nodes,leaf_rows)
    ancestor=_subset(batch.ancestor,leaf_rows,nodes)
    native=tuple(_subset(m,nodes,nodes) for m in batch.native)
    empty=csr([],[],[],(len(nodes),len(nodes)))
    cross=tuple(empty for _ in range(4))
    matrices=(descendant,ancestor)+native+cross
    sparse_bytes=sum(m.crow_indices().numel()*8+m.col_indices().numel()*8+m.values().numel()*4 for m in matrices)
    statistics={'view':'focal_only_no_cross_v1','valid_mode':'no_cross','nodes':len(nodes),
        'trees':len(trees),'samples':h,'windows':batch.window_count,
        'site_rows':int(sites.sum()),'nnz':[m._nnz() for m in matrices],
        'sparse_bytes':sparse_bytes,'original_statistics':batch.statistics}
    return replace(batch,parent=parent,node_time=batch.node_time[nodes],
        node_tree=tree_map[batch.node_tree[nodes]],leaves=node_map[batch.leaves[trees]],
        tree_window=batch.tree_window[trees],coordinate=batch.coordinate[trees],
        coordinate_weight=batch.coordinate_weight[trees],site_tree=tree_map[batch.site_tree[sites]],
        site_genotype=batch.site_genotype[sites],site_alleles=batch.site_alleles[sites],
        focal_tree=tree_map[batch.focal_tree],geometry=batch.geometry[nodes],
        descendant=descendant,ancestor=ancestor,native=native,cross=cross,
        coupling_mass=torch.zeros((len(nodes),4),dtype=batch.coupling_mass.dtype),statistics=statistics)
