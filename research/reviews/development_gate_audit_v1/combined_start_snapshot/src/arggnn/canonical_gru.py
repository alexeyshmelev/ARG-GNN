"""Canonical observed-tree GNN plus conventional leaf-aligned GRU comparator.

Established preprocessing and ordinary neural modules, not novel invariance.
The CPU preprocessing interface accepts observed sequences only, never labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import GraphSequence, slice_genomic_window
from .models import SharedGraphEncoder, query_node_index, validate_model_input
from .reference_encoder import ReferenceFeatureAdapter
from .tree_canonicalization import CanonicalizedObservedTrees, canonicalize_observed_trees


@dataclass(frozen=True)
class CanonicalPreparedParent:
    sequence: GraphSequence
    correspondence: CanonicalizedObservedTrees
    adapter_provenance: dict

    def as_dict(self) -> dict:
        return {"sequence":self.sequence.as_dict(),"correspondence":self.correspondence.as_dict(),
                "adapter_provenance":self.adapter_provenance,"schema_version":1}


@torch.no_grad()
def prepare_canonical_parent(sequence: GraphSequence, reference_adapter: Mapping[str,Any], *,
        base_feature_count: int = 8, source_provenance: Mapping | None = None,
        original_feature_names: list[str] | None = None,
        source_tensor_sha256: str | None = None) -> CanonicalPreparedParent:
    """Strip stale summaries, canonicalize complete observed parent, re-augment.

    Exact adjacent duplicate merging is required for this comparator. External
    scoring positions/query IDs are retained and target_graph is remapped by
    explicit canonical correspondence. No label or parent dataset object enters.
    """
    options = dict(reference_adapter)
    canonical = canonicalize_observed_trees(sequence,time_column=options["time_column"],
        parent_branch_column=options["branch_column"],base_feature_count=base_feature_count,
        time_scale=options.get("time_scale",10000.),merge_identical_adjacent=True,
        source_provenance=source_provenance)
    adapter = ReferenceFeatureAdapter(base_feature_count,**options)
    augmented = adapter.precompute(canonical.sequence)
    provenance = adapter.manifest_provenance(original_feature_names,source_tensor_sha256)
    provenance["canonicalization_stage"] = "observed base prefix before full-parent adapter recomputation and fixed tiling"
    if not torch.equal(augmented.x[:,:base_feature_count],canonical.sequence.x):
        raise RuntimeError("Fresh adapter changed canonical base prefix")
    return CanonicalPreparedParent(augmented,canonical,provenance)


@dataclass(frozen=True)
class CanonicalGNNInput:
    sequence: GraphSequence
    query_index: Tensor          # [G,Q], stable ID correspondence, not features
    chronological: Tensor        # [G], physical ordering
    inverse_order: Tensor         # [G]
    distances: Tensor             # [G,2], physical width and midpoint gap

    def to(self,device: torch.device | str) -> CanonicalGNNInput:
        return CanonicalGNNInput(self.sequence.to(device),self.query_index.to(device),
            self.chronological.to(device),self.inverse_order.to(device),self.distances.to(device))


@torch.no_grad()
def prepare_canonical_gnn_input(sequence: GraphSequence) -> CanonicalGNNInput:
    if sequence.x.device.type != "cpu":
        raise ValueError("Prepare leaf correspondence on CPU")
    validate_model_input(sequence)
    order = torch.argsort(sequence.intervals[:,0])
    intervals = sequence.intervals[order].double()
    midpoint = intervals.mean(-1)
    gap = torch.cat((midpoint.new_zeros(1),midpoint[1:]-midpoint[:-1]))
    distances = torch.stack((intervals[:,1]-intervals[:,0],gap),dim=-1)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order))
    return CanonicalGNNInput(sequence,query_node_index(sequence),order,inverse,distances)


@dataclass(frozen=True)
class CanonicalTile:
    model_input: CanonicalGNNInput
    parent_position_indices: Tensor
    window_index: int


def canonical_fixed_tiles(parent: CanonicalPreparedParent,windows: int = 8) -> list[CanonicalTile]:
    """Tile observed inputs only and retain original parent scoring indexes."""
    if isinstance(windows,bool) or not isinstance(windows,int) or windows<1:
        raise ValueError("windows must be a positive integer")
    seq = parent.sequence
    coverage = seq.window_bounds if seq.window_bounds is not None else seq.intervals.new_tensor([0.,1.])
    tiles = []
    for index in range(windows):
        left,right = max(index/windows,float(coverage[0])),min((index+1)/windows,float(coverage[1]))
        if left>=right or not bool(((seq.positions>=left)&(seq.positions<right)).any()):
            continue
        tile,positions = slice_genomic_window(seq,left,right)
        tiles.append(CanonicalTile(prepare_canonical_gnn_input(tile),positions,index))
    covered = torch.cat([tile.parent_position_indices for tile in tiles]) if tiles else torch.empty(0,dtype=torch.long)
    if not torch.equal(covered.sort().values,torch.arange(seq.positions.numel())):
        raise RuntimeError("Fixed tiles must retain every parent coordinate exactly once")
    return tiles


class CanonicalTreeGNNGRU(nn.Module):
    """Full canonical-tree mean MPNN, then standard bidirectional query GRU.

    Dropout applies after input projection and every graph layer, within the
    stacked GRU, and before the output head. Distances are ordinary observed
    width/midpoint-gap inputs. Fixed readouts gather their containing graph.
    """
    def __init__(self,input_dim: int = 48,num_classes: int = 4,hidden_dim: int = 32,
                 num_layers: int = 2,temporal_layers: int = 2,dropout: float = 0.):
        super().__init__()
        if min(input_dim,num_classes,hidden_dim,num_layers,temporal_layers)<1 or num_classes<2 or not 0<=dropout<1:
            raise ValueError("Require positive dimensions/depths, at least2 classes and dropout in[0,1)")
        self.input_dim,self.num_classes,self.hidden_dim = input_dim,num_classes,hidden_dim
        self.num_layers,self.temporal_layers,self.dropout_probability = num_layers,temporal_layers,float(dropout)
        self.encoder = SharedGraphEncoder(input_dim,hidden_dim,num_layers)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim+2,hidden_dim,num_layers=temporal_layers,batch_first=True,
            bidirectional=True,dropout=dropout if temporal_layers>1 else 0.)
        self.head = nn.Linear(2*hidden_dim,num_classes)

    def forward(self,prepared: CanonicalGNNInput) -> Tensor:
        seq = prepared.sequence
        weight = self.encoder.input_linear.weight
        if seq.x.shape[-1]!=self.input_dim or seq.x.dtype!=weight.dtype or seq.x.device!=weight.device:
            raise ValueError("Prepared canonical feature dimension/device/dtype differs from model")
        h = self.dropout(F.gelu(self.encoder.input_linear(seq.x)))
        for layer in self.encoder.layers:
            h = self.dropout(layer(h,seq.edge_index))
        leaf = h[prepared.query_index[prepared.chronological]].transpose(0,1)
        distances = prepared.distances.to(leaf.dtype)[None].expand(leaf.shape[0],-1,-1)
        encoded,_ = self.gru(torch.cat((leaf,distances),dim=-1))
        output = encoded[:,prepared.inverse_order[seq.target_graph]].transpose(0,1)
        return self.head(self.dropout(output))

    def configuration_manifest(self) -> dict[str,Any]:
        result = {"kind":"canonical_observed_tree_GNN_GRU_comparator","input_dim":self.input_dim,
            "num_outputs":self.num_classes,"hidden_dim":self.hidden_dim,"GNN_depth":self.num_layers,
            "GRU_depth":self.temporal_layers,"bidirectional":True,"dropout":self.dropout_probability,
            "source_scope":"all canonical observed tree nodes, fresh F48 reference adapter",
            "node_correspondence":"stable sampled identity lookup, no learned ID embeddings",
            "GRU_inputs":"query graph-local embedding, physical global-coordinate width and midpoint gap",
            "readout":"containing canonical graph gathered in original position/query order",
            "canonicalization":"anonymous unary suppression and exact adjacent duplicate merging before fresh adapter",
            "parameters":sum(p.numel() for p in self.parameters()),
            "scientific_status":"ordinary preprocessing comparator; no novelty or predictive claim"}
        result["configuration_sha256"] = hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result
