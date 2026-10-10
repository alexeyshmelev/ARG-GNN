"""Shared tensor contract for complete ordered marginal-tree sequences.

Ancestry targets are intentionally absent from GraphSequence. Integer node IDs
identify extant leaves within a replicate and must never be embedded as features.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch
from torch import Tensor


@dataclass
class GraphSequence:
    x: Tensor                    # [N,F], observed features only
    edge_index: Tensor           # [2,E], directed parent/child edges
    node_graph: Tensor           # [N], graph index
    node_ids: Tensor             # [N], stable leaf ID or -1 for internal nodes
    clades: Tensor               # [N,C], descendant reference landmarks
    intervals: Tensor            # [G,2], full normalized genomic partition
    query_node_ids: Tensor        # [Q], stable IDs, external target ordering
    positions: Tensor            # [P], normalized fixed scoring coordinates
    target_graph: Tensor         # [P], containing marginal-tree index
    window_bounds: Tensor | None = None  # [2], global chromosome tile bounds

    @property
    def num_graphs(self) -> int:
        return int(self.intervals.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.query_node_ids.numel())

    def to(self, device: torch.device | str) -> GraphSequence:
        return GraphSequence(**{f.name: (None if getattr(self, f.name) is None else getattr(self, f.name).to(device)) for f in fields(self)})

    def as_dict(self) -> dict[str, Tensor | None]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Tensor | None]) -> GraphSequence:
        result = cls(**{f.name: values.get(f.name) if f.name == "window_bounds" else values[f.name] for f in fields(cls)})
        result.validate()
        return result

    def validate(self) -> None:
        """Check shape, partition, leaf identity, and target-coordinate invariants."""
        if self.x.ndim != 2 or self.x.shape[0] == 0:
            raise ValueError("x must be nonempty [N,F]")
        n = self.x.shape[0]
        for name in ("node_graph", "node_ids"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must be [N]")
        if self.clades.ndim != 2 or self.clades.shape[0] != n:
            raise ValueError("clades must be [N,C]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        for name in ("edge_index", "node_graph", "node_ids", "query_node_ids", "target_graph"):
            if getattr(self, name).dtype != torch.long:
                raise ValueError(f"{name} must have dtype torch.long")
        if self.intervals.ndim != 2 or self.intervals.shape[1] != 2 or self.num_graphs == 0:
            raise ValueError("intervals must be nonempty [G,2]")
        if self.positions.ndim != 1 or self.target_graph.shape != self.positions.shape:
            raise ValueError("positions and target_graph must be [P]")
        if self.query_node_ids.ndim != 1 or self.num_queries == 0:
            raise ValueError("query_node_ids must be nonempty [Q]")
        if torch.unique(self.query_node_ids).numel() != self.num_queries or bool((self.query_node_ids < 0).any()):
            raise ValueError("query IDs must be unique nonnegative leaf IDs")
        for name in ("x", "clades", "intervals", "positions"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} must be finite")
        if bool((self.clades < 0).any()):
            raise ValueError("clades must be nonnegative")
        if bool((self.node_ids < -1).any()):
            raise ValueError("node_ids must be stable leaf IDs or -1")
        if bool((self.node_graph < 0).any()) or bool((self.node_graph >= self.num_graphs).any()):
            raise ValueError("node_graph out of range")
        if torch.unique(self.node_graph).numel() != self.num_graphs:
            raise ValueError("every marginal graph must contain nodes")
        if self.edge_index.numel():
            if bool((self.edge_index < 0).any()) or bool((self.edge_index >= n).any()):
                raise ValueError("edge endpoint out of range")
            if not bool((self.node_graph[self.edge_index[0]] == self.node_graph[self.edge_index[1]]).all()):
                raise ValueError("vertical edges cannot cross marginal graphs")
        ordered = self.intervals[torch.argsort(self.intervals[:, 0])]
        if bool((ordered[:, 1] <= ordered[:, 0]).any()):
            raise ValueError("intervals must have positive widths")
        bounds = self.window_bounds.to(device=ordered.device, dtype=ordered.dtype) if self.window_bounds is not None else ordered.new_tensor([0.0, 1.0])
        if bounds.shape != (2,) or not bool(torch.isfinite(bounds).all()) or not 0 <= float(bounds[0]) < float(bounds[1]) <= 1:
            raise ValueError("window_bounds must be a positive interval inside [0,1]")
        if not bool(torch.isclose(ordered[0, 0], bounds[0], atol=1e-8, rtol=0)) or not bool(torch.isclose(ordered[-1, 1], bounds[1], atol=1e-8, rtol=0)):
            raise ValueError("intervals must cover the declared genomic window")
        if len(ordered) > 1 and not bool(torch.allclose(ordered[:-1, 1], ordered[1:, 0], atol=1e-8, rtol=0)):
            raise ValueError("intervals must form a contiguous partition")
        if bool((self.positions < 0).any()) or bool((self.positions >= 1).any()):
            raise ValueError("positions must lie in [0,1)")
        if bool((self.target_graph < 0).any()) or bool((self.target_graph >= self.num_graphs).any()):
            raise ValueError("target_graph out of range")
        selected = self.intervals[self.target_graph]
        if not bool(((self.positions >= selected[:, 0]) & (self.positions < selected[:, 1])).all()):
            raise ValueError("target_graph does not contain its scoring position")
        # Every query must occur exactly once in every complete marginal tree.
        for graph in range(self.num_graphs):
            leaves = self.node_ids[self.node_graph == graph]
            stable = leaves[leaves >= 0]
            if stable.numel() != torch.unique(stable).numel():
                raise ValueError("duplicate leaf ID within a marginal graph")
            if not bool(torch.isin(self.query_node_ids, stable).all()):
                raise ValueError("a queried leaf is absent from a marginal graph")


def slice_genomic_window(sequence: GraphSequence, left: float, right: float) -> tuple[GraphSequence, Tensor]:
    """Clip a fixed genomic tile, retaining every marginal boundary within it.

    Coordinates remain normalized to the parent chromosome. Returns scoring
    position indexes so external labels and comparators can be sliced identically.
    This does not change the parent replicate split or create new subjects.
    """
    sequence.validate()
    if not 0 <= left < right <= 1:
        raise ValueError("require 0 <= left < right <= 1")
    previous = sequence.window_bounds if sequence.window_bounds is not None else sequence.intervals.new_tensor([0., 1.])
    if left < float(previous[0]) or right > float(previous[1]):
        raise ValueError("requested tile exceeds its parent sequence")
    graph_keep = (sequence.intervals[:, 1] > left) & (sequence.intervals[:, 0] < right)
    graphs = torch.nonzero(graph_keep, as_tuple=False).flatten()
    graph_map = torch.full((sequence.num_graphs,), -1, dtype=torch.long, device=sequence.node_graph.device)
    graph_map[graphs] = torch.arange(len(graphs), device=graph_map.device)
    node_keep = graph_keep[sequence.node_graph]
    nodes = torch.nonzero(node_keep, as_tuple=False).flatten()
    node_map = torch.full((len(sequence.x),), -1, dtype=torch.long, device=sequence.node_graph.device)
    node_map[nodes] = torch.arange(len(nodes), device=node_map.device)
    edges = sequence.edge_index[:, node_keep[sequence.edge_index[0]] & node_keep[sequence.edge_index[1]]]
    indexes = torch.nonzero((sequence.positions >= left) & (sequence.positions < right), as_tuple=False).flatten()
    if len(indexes) == 0:
        raise ValueError("a genomic tile must contain at least one fixed scoring coordinate")
    intervals = sequence.intervals[graphs].clone()
    intervals[:, 0].clamp_(min=left)
    intervals[:, 1].clamp_(max=right)
    result = GraphSequence(
        x=sequence.x[nodes], edge_index=node_map[edges], node_graph=graph_map[sequence.node_graph[nodes]],
        node_ids=sequence.node_ids[nodes], clades=sequence.clades[nodes], intervals=intervals,
        query_node_ids=sequence.query_node_ids, positions=sequence.positions[indexes],
        target_graph=graph_map[sequence.target_graph[indexes]], window_bounds=intervals.new_tensor([left, right]),
    )
    result.validate()
    return result, indexes
