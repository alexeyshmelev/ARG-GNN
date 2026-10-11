"""PyG implementation of the frozen H036 whole-tree cavity message operator.

The learned modules, site encoder, full-parent GRU, classifier and checkpoint
keys are inherited unchanged. Only native tree message aggregation/routing
uses parameter-free PyG MessagePassing and SumAggregation modules.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation

from .mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeGNNV1, MutationRoutedWholeTreeInputsV1,
)


class _GatedUpwardMessagePassingV1(MessagePassing):
    """Compact bipartite child-to-parent level, with original edge messages."""
    def __init__(self):
        super().__init__(aggr=SumAggregation(), flow="source_to_target", node_dim=0)
        # Ephemeral return channel, not a registered buffer or parameter.
        self._edge_messages = None

    def forward(self, child_state: Tensor, child_gate: Tensor,
                parent_degree: Tensor, parent_rank: Tensor,
                parent_count: int, weight: Tensor):
        edge_count = parent_rank.numel()
        edge_index = torch.stack((
            torch.arange(edge_count, device=parent_rank.device, dtype=torch.long),
            parent_rank,
        ))
        self._edge_messages = None
        try:
            aggregate = self.propagate(
                edge_index, x=(child_state, None), gate=child_gate,
                denominator=parent_degree, weight=weight,
                size=(edge_count, parent_count),
            )
            edge_messages = self._edge_messages
            if edge_messages is None:
                raise RuntimeError("PyG did not evaluate upward edge messages")
            return aggregate, edge_messages
        finally:
            self._edge_messages = None

    def message(self, x_j: Tensor, gate: Tensor, denominator: Tensor,
                weight: Tensor) -> Tensor:
        message = F.linear(x_j, weight) * gate
        message = message / denominator[:, None]
        # Downward cavity exclusion needs these exact transformed edge values.
        self._edge_messages = message
        return message


class _ExcludedCavityMessagePassingV1(MessagePassing):
    """Parent-to-child level, excluding the recipient's upward contribution."""
    def __init__(self):
        super().__init__(aggr=SumAggregation(), flow="source_to_target", node_dim=0)

    def forward(self, parent_context: Tensor, recipient_message: Tensor,
                child_gate: Tensor, weight: Tensor) -> Tensor:
        count = parent_context.shape[0]
        local = torch.arange(count, device=parent_context.device, dtype=torch.long)
        # The compact source row is the recipient-specific parent context.
        # Each child receives one native parent edge, so sum has one summand.
        edge_index = torch.stack((local, local))
        return self.propagate(
            edge_index, x=(parent_context, None), excluded=recipient_message,
            gate=child_gate, weight=weight, size=(count, count),
        )

    def message(self, x_j: Tensor, excluded: Tensor, gate: Tensor,
                weight: Tensor) -> Tensor:
        cavity = x_j - excluded
        return F.linear(cavity, weight) * gate


class MutationRoutedPyGGNNV1(MutationRoutedWholeTreeGNNV1):
    """H036-equivalent graph computation implemented with PyG propagation.

    Use the original observation dataclass. A cached-parent PyG Data item can
    supply it with item.to_model_inputs(). No registered learned layer is
    aliased under a MessagePassing module. Consequently strict legacy state
    loading and parameter order remain unchanged.
    """
    def __init__(self, hidden_dim=32, *, expected_haplotypes=280,
                 site_chunk_size=256, checkpoint_sites=True):
        super().__init__(
            hidden_dim=hidden_dim, expected_haplotypes=expected_haplotypes,
            site_chunk_size=site_chunk_size, checkpoint_sites=checkpoint_sites,
        )
        self.pyg_upward = _GatedUpwardMessagePassingV1()
        self.pyg_downward = _ExcludedCavityMessagePassingV1()

    def encode_trees(self, bank: MutationRoutedWholeTreeInputsV1, *, control):
        t, h = bank.sample_index.shape
        n, d = len(bank.parent), self.hidden_dim
        site_leaf = self.encode_sites(bank)
        labels = bank.reference_class
        onehot = F.one_hot(labels.clamp_min(0), 4).to(site_leaf.dtype)
        onehot = onehot * (labels >= 0)[:, None]
        unknown = (labels < 0).to(site_leaf.dtype)[:, None]
        age = (bank.sample_age.log1p()/math.log(10001.))[:, None]
        observed_leaf = torch.cat((onehot, unknown, age), -1)[None].expand(t, h, 6)
        leaf_base = self.leaf_projection(torch.cat((site_leaf, observed_leaf), -1))
        sampled = bank.node_time.new_zeros(n)
        sampled.index_fill_(0, bank.sample_index.flatten(), 1.)
        root = (bank.parent < 0).to(sampled.dtype)
        node_age = bank.node_time.log1p()/math.log(10001.)
        base = self.node_projection(torch.stack((node_age, root, sampled), -1))
        # This is unique sampled-leaf feature injection, not graph messaging.
        base = base.index_add(0, bank.sample_index.flatten(), leaf_base.reshape(-1, d))
        raw = bank.branch_support_true if control == "aligned" else bank.branch_support_sham
        support = torch.cat((raw[:, :3], raw[:, 3:4].log1p()/math.log(100001.), raw[:, 4:5]), -1)
        gate_input = torch.cat((bank.edge_geometry, support), -1)
        ug, dg = self.up_gate(gate_input).sigmoid(), self.down_gate(gate_input).sigmoid()
        nonroot = bank.parent >= 0
        degree = torch.bincount(bank.parent[nonroot], minlength=n).to(base.dtype).clamp_min(1)
        upward = base.new_zeros(n, d)
        message_sum = base.new_zeros(n, d)
        child_message = base.new_zeros(n, d)
        for nodes, children, ranks in zip(
                bank.upward_levels, bank.upward_children, bank.upward_parent_rank):
            if children.numel():
                aggregate, message = self.pyg_upward(
                    upward[children], ug[children],
                    degree[bank.parent[children]], ranks, len(nodes),
                    self.up_message.weight,
                )
                child_message.index_copy_(0, children, message)
            else:
                aggregate = base.new_zeros(len(nodes), d)
            message_sum.index_copy_(0, nodes, aggregate)
            value = self.up_norm(
                base[nodes] +
                self.up_update(torch.cat((base[nodes], aggregate), -1)).tanh()
            )
            upward.index_copy_(0, nodes, value)
        outside = base.new_zeros(n, d)
        for nodes in bank.downward_levels:
            parents = bank.parent[nodes]
            parent_context = outside[parents] + base[parents] + message_sum[parents]
            value = self.pyg_downward(
                parent_context, child_message[nodes], dg[nodes],
                self.down_message.weight,
            )
            value = self.down_norm(self.down_update(value).tanh())
            outside.index_copy_(0, nodes, value)
        return self.leaf_context(torch.cat((
            upward[bank.sample_index], outside[bank.sample_index],
        ), -1))
