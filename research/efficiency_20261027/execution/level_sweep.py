"""Exact functional tree sweep with level-local autograd states.

Only immutable integer routing is reusable. Every learned state is recomputed
on every forward. Original node order, PyG edge order, reductions, nonlinearities,
recipient exclusion, root outside state, and model parameters are unchanged.
This module has no launcher, loader, global patch, or production activation.
"""
from dataclasses import dataclass
import numpy as np
import torch


def need(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class Gather:
    sources: tuple
    inverse: torch.Tensor

    def apply(self, values):
        pieces = [values[level].index_select(0, rows) for level, rows in self.sources]
        return torch.cat(pieces, dim=0).index_select(0, self.inverse)

    def to(self, device):
        return Gather(tuple((level, rows.to(device)) for level, rows in self.sources),
                      self.inverse.to(device))


def gather_plan(requested, level, rank):
    groups = []
    positions = []
    for source in np.unique(level[requested]):
        pos = np.flatnonzero(level[requested] == source)
        groups.append((int(source), torch.from_numpy(rank[requested[pos]].copy())))
        positions.append(pos)
    order = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    return Gather(tuple(groups), torch.from_numpy(np.argsort(order).astype(np.int64)))


@dataclass(frozen=True)
class LevelSweepPlan:
    node_count: int
    upward_nodes: tuple
    child_nodes: tuple
    child_parent_ranks: tuple
    child_sources: tuple
    upward_inverse: torch.Tensor
    roots: torch.Tensor
    message_inverse: torch.Tensor
    downward_nodes: tuple
    downward_parent_rows: tuple
    downward_inverse: torch.Tensor
    transport: object = None

    @classmethod
    def build(cls, bank):
        """Build on CPU from authenticated topology, before CUDA execution."""
        def array(tensor):
            need(tensor.device.type == "cpu" and tensor.dtype == torch.long,
                 "Plan construction requires original CPU int64 topology")
            return tensor.detach().numpy()
        parent = array(bank.parent)
        n = len(parent)
        need(n > 0 and np.all((parent >= -1) & (parent < n)), "Valid nonempty forest")
        levels = tuple(array(x) for x in bank.upward_levels)
        children = tuple(array(x) for x in bank.upward_children)
        ranks = tuple(array(x) for x in bank.upward_parent_rank)
        downward = tuple(array(x) for x in bank.downward_levels)
        need(len(levels) == len(children) == len(ranks), "Aligned upward recipes")
        all_nodes = np.concatenate(levels)
        need(np.array_equal(np.sort(all_nodes), np.arange(n)), "Upward levels partition every node once")
        height, rank = np.empty(n, np.int64), np.empty(n, np.int64)
        for k, nodes in enumerate(levels):
            height[nodes] = k
            rank[nodes] = np.arange(len(nodes))
        sources = []
        all_children = []
        for k, (nodes, child, parent_rank) in enumerate(zip(levels, children, ranks)):
            need(len(child) == len(parent_rank), "Aligned edge ranks")
            need(np.all((child >= 0) & (child < n)), "Valid child rows")
            need(np.all(parent[child] >= 0) and np.all(height[child] < k), "Upward dependencies strictly earlier")
            need(np.array_equal(nodes[parent_rank], parent[child]), "Original parent ranks retained")
            sources.append(gather_plan(child, height, rank) if len(child) else None)
            all_children.append(child)
        roots = np.flatnonzero(parent < 0)
        message_order = np.concatenate((roots, *all_children))
        need(np.array_equal(np.sort(message_order), np.arange(n)), "Every nonroot has exactly one outgoing message")
        depth_order = np.concatenate((roots, *downward))
        need(np.array_equal(np.sort(depth_order), np.arange(n)), "Downward levels partition nonroots")
        previous = roots
        down_rank = np.full(n, -1, np.int64)
        previous_rank = []
        for nodes in downward:
            down_rank.fill(-1)
            down_rank[previous] = np.arange(len(previous))
            need(np.all(parent[nodes] >= 0) and np.all(down_rank[parent[nodes]] >= 0),
                 "Downward parent must occur in the immediately preceding depth")
            previous_rank.append(down_rank[parent[nodes]].copy())
            previous = nodes
        def tensor(x):
            return torch.from_numpy(np.asarray(x, dtype=np.int64).copy())
        return cls(n, tuple(tensor(x) for x in levels), tuple(tensor(x) for x in children),
                   tuple(tensor(x) for x in ranks), tuple(sources), tensor(np.argsort(all_nodes)),
                   tensor(roots), tensor(np.argsort(message_order)), tuple(tensor(x) for x in downward),
                   tuple(tensor(x) for x in previous_rank), tensor(np.argsort(depth_order))).packed()

    def _buffers(self):
        arrays = [self.upward_inverse, self.roots, self.message_inverse, self.downward_inverse]
        for rows in (self.upward_nodes, self.child_nodes, self.child_parent_ranks,
                     self.downward_nodes, self.downward_parent_rows):
            arrays.extend(rows)
        for item in self.child_sources:
            if item is not None:
                arrays.append(item.inverse)
                arrays.extend(rows for _, rows in item.sources)
        return arrays

    def _from_flat(self, flat):
        cursor = 0
        def view(original):
            nonlocal cursor
            result = flat[cursor:cursor+original.numel()].view(original.shape)
            cursor += original.numel()
            return result
        inverse, roots, message_inverse, downward_inverse = [view(x) for x in
            (self.upward_inverse, self.roots, self.message_inverse, self.downward_inverse)]
        up, children, ranks, down, parent_rows = [tuple(view(x) for x in rows) for rows in
            (self.upward_nodes, self.child_nodes, self.child_parent_ranks,
             self.downward_nodes, self.downward_parent_rows)]
        sources = []
        for item in self.child_sources:
            if item is None:
                sources.append(None)
            else:
                order = view(item.inverse)
                sources.append(Gather(tuple((level, view(rows)) for level, rows in item.sources), order))
        need(cursor == flat.numel(), "Packed route partition must cover every integer")
        return LevelSweepPlan(self.node_count, up, children, ranks, tuple(sources),
            inverse, roots, message_inverse, down, parent_rows, downward_inverse, flat)

    def packed(self):
        return self if self.transport is not None else self._from_flat(torch.cat(self._buffers()))

    def to(self, device):
        plan = self.packed()
        target = plan.transport.to(device)
        # One contiguous transfer, then metadata-only views. This avoids hundreds
        # of tiny host-to-device transfers per recomputed block.
        return plan if target is plan.transport else plan._from_flat(target)

    def tensor_bytes(self):
        return sum(x.numel() * x.element_size() for x in self._buffers())


def sweep_tree_state(model, bank, base, up_gate, down_gate, degree, *, plan):
    """Drop-in numerical kernel. Return original node order and connected grads."""
    need(base.ndim == 2 and len(base) == plan.node_count and bank.parent.device == base.device
         and plan.upward_inverse.device == base.device, "Aligned plan, bank, and learned inputs")
    width = base.shape[1]
    upward, aggregates, messages = [], [], [base.new_zeros(len(plan.roots), width)]
    for nodes, children, ranks, source in zip(plan.upward_nodes, plan.child_nodes,
                                             plan.child_parent_ranks, plan.child_sources):
        if children.numel():
            aggregate, message = model.pyg_upward(source.apply(upward), up_gate[children],
                degree[bank.parent[children]], ranks, len(nodes), model.up_message.weight)
        else:
            aggregate = base.new_zeros(len(nodes), width)
            message = base.new_zeros(0, width)
        # Identical original operation order; only the destination buffer changes.
        value = model.up_norm(base[nodes] + model.up_update(torch.cat((base[nodes], aggregate), -1)).tanh())
        upward.append(value)
        aggregates.append(aggregate)
        messages.append(message)
    up = torch.cat(upward).index_select(0, plan.upward_inverse)
    message_sum = torch.cat(aggregates).index_select(0, plan.upward_inverse)
    child_message = torch.cat(messages).index_select(0, plan.message_inverse)
    outside_levels = [base.new_zeros(len(plan.roots), width)]
    for nodes, parent_rows in zip(plan.downward_nodes, plan.downward_parent_rows):
        parents = bank.parent[nodes]
        parent_context = outside_levels[-1][parent_rows] + base[parents] + message_sum[parents]
        value = model.pyg_downward(parent_context, child_message[nodes], down_gate[nodes], model.down_message.weight)
        outside_levels.append(model.down_norm(model.down_update(value).tanh()))
    outside = torch.cat(outside_levels).index_select(0, plan.downward_inverse)
    return up, outside
