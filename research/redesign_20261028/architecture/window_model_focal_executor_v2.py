"""Separate, source-compatible focal executor candidate for frozen v1.

The learned modules and state_dict are inherited unchanged. Preparation caches
only immutable routing. Every source occurrence remains present, including
repeated coordinate draws. No learned activation survives an optimizer update.
"""
from dataclasses import dataclass, fields
import torch
from . import window_model_frozen_v1 as reference

OBS_KEYS = reference.OBS_KEYS
MODEL_CONTEXT_KEYS = reference.MODEL_CONTEXT_KEYS
SOURCE_CONTEXT_KEYS = reference.SOURCE_CONTEXT_KEYS
load_observations = reference.load_observations
collate_observations = reference.collate_observations
coordinate_observations = reference.coordinate_observations


@dataclass(frozen=True)
class WindowRoute:
    sources: torch.Tensor
    targets: torch.Tensor
    output_rows: torch.Tensor
    coordinate: bool

    def to(self, device):
        return WindowRoute(self.sources.to(device), self.targets.to(device),
                           self.output_rows.to(device), self.coordinate)


@dataclass(frozen=True)
class Batch:
    source: reference.Batch
    focal: reference.Batch | None
    trees: torch.Tensor
    nodes: torch.Tensor
    cut_indices: tuple
    windows: tuple
    preserve_edge_zero_gradients: bool = False

    @property
    def core_focal_mask(self):
        return self.source.core_focal_mask

    def to(self, device):
        return Batch(self.source.to(device), None if self.focal is None else self.focal.to(device),
                     self.trees.to(device), self.nodes.to(device),
                     tuple(x.to(device) for x in self.cut_indices),
                     tuple(x.to(device) for x in self.windows), self.preserve_edge_zero_gradients)


def prepare_executor(source):
    """Select complete focal trees and rebuild their immutable routes on CPU.

    Shared metadata keeps its original window indexing. The all-source batch
    remains unchanged. Multiple focal rows on one tree share a second sweep.
    """
    reference.require(type(source) is reference.Batch, 'Frozen v1 Batch required')
    reference.require(source.parent.device.type == 'cpu', 'Prepare routes on CPU before .to(device)')
    trees = torch.unique(source.focal_tree, sorted=True)
    if not len(trees):
        return Batch(source, None, trees, torch.empty(0, dtype=torch.long), (), ())
    tree_map = torch.full((len(source.leaves),), -1, dtype=torch.long)
    tree_map[trees] = torch.arange(len(trees))
    nodes = torch.nonzero(tree_map[source.node_tree] >= 0, as_tuple=False).flatten()
    node_map = torch.full((len(source.parent),), -1, dtype=torch.long)
    node_map[nodes] = torch.arange(len(nodes))
    parent = source.parent[nodes]
    parent = torch.where(parent >= 0, node_map[parent.clamp_min(0)], -1)
    levels, children, ranks = [], [], []
    for old_nodes, old_children in zip(source.up_levels, source.up_children):
        selected = old_nodes[node_map[old_nodes] >= 0]
        if not len(selected):
            continue
        selected_children = old_children[node_map[old_children] >= 0]
        rank_map = torch.full((len(source.parent),), -1, dtype=torch.long)
        rank_map[selected] = torch.arange(len(selected))
        levels.append(node_map[selected])
        children.append(node_map[selected_children])
        ranks.append(rank_map[source.parent[selected_children]])
    down = tuple(node_map[level[node_map[level] >= 0]] for level in source.down_levels
                 if bool((node_map[level] >= 0).any()))
    cuts, cut_indices = [], []
    for cut in source.cuts:
        groups = cut.groups[trees]
        selected = torch.unique(groups, sorted=True)
        group_map = torch.full((len(cut.counts),), -1, dtype=torch.long)
        group_map[selected] = torch.arange(len(selected))
        cuts.append(reference.Cut(node_map[cut.representatives[selected]], group_map[groups],
                                  cut.counts[selected], cut.valid_time[selected], cut.root_saturated[selected]))
        cut_indices.append(selected)
    sites = torch.nonzero(tree_map[source.site_tree] >= 0, as_tuple=False).flatten()
    values = {field.name: getattr(source, field.name) for field in fields(source)}
    values.update(parent=parent, node_time=source.node_time[nodes], node_tree=tree_map[source.node_tree[nodes]],
                  leaves=node_map[source.leaves[trees]], intervals=source.intervals[trees],
                  tree_window=source.tree_window[trees], core=source.core[trees],
                  coordinate=source.coordinate[trees], coordinate_weight=source.coordinate_weight[trees],
                  site_tree=tree_map[source.site_tree[sites]], site_genotype=source.site_genotype[sites],
                  site_alleles=source.site_alleles[sites], focal_tree=tree_map[source.focal_tree],
                  up_levels=tuple(levels), up_children=tuple(children), up_ranks=tuple(ranks),
                  down_levels=down, cuts=tuple(cuts))
    focal = reference.Batch(**values)
    windows = []
    for win in torch.unique(source.tree_window[trees], sorted=True):
        source_index = torch.nonzero(source.tree_window == win, as_tuple=False).flatten()
        output_rows = torch.nonzero(source.tree_window[trees] == win, as_tuple=False).flatten()
        windows.append(WindowRoute(source_index, trees[output_rows], output_rows,
                                   bool(source.context_kind[win] == 1)))
    return Batch(source, focal, trees, nodes, tuple(cut_indices), tuple(windows),
                 bool((source.parent >= 0).any()) and not bool((focal.parent >= 0).any()))


def prepare_observations(arrays, cut_times=(100., 350., 1000.)):
    return prepare_executor(reference.prepare_observations(arrays, cut_times=cut_times))


def focal_context_kernel(batch, ell):
    """Only focal target rows, with every source occurrence and source mass."""
    source = batch.source
    blocks = []
    for route in batch.windows:
        if route.coordinate:
            distance = (source.coordinate[route.targets, None] - source.coordinate[route.sources][None, :]).abs()
            kernel = torch.exp(-distance / ell.double()) * source.coordinate_weight[route.sources][None, :]
        else:
            target_bounds = source.intervals[route.targets].double()
            source_bounds = source.intervals[route.sources].double()
            scale = ell.double()
            target_ratio = (target_bounds[:, 1] - target_bounds[:, 0]) / scale
            source_ratio = (source_bounds[:, 1] - source_bounds[:, 0]) / scale
            target_mass = -torch.expm1(-target_ratio)
            source_mass = -torch.expm1(-source_ratio)
            gap = (torch.maximum(target_bounds[:, None, 0], source_bounds[None, :, 0]) -
                   torch.minimum(target_bounds[:, None, 1], source_bounds[None, :, 1])).clamp_min(0)
            kernel = scale.square() * target_mass[:, None] * source_mass[None, :] * torch.exp(-gap / scale)
            ordinary = 2 * scale.square() * (target_ratio + torch.expm1(-target_ratio))
            series = scale.square() * (target_ratio.square() - target_ratio.pow(3) / 3 +
                                       target_ratio.pow(4) / 12 - target_ratio.pow(5) / 60)
            diagonal = torch.where(target_ratio < 1e-3, series, ordinary)
            kernel = torch.where(route.targets[:, None] == route.sources[None, :], diagonal[:, None], kernel)
        blocks.append((route, kernel))
    return tuple(blocks)


def focal_temporal_mean(values, kernel, target_count):
    output = values.new_zeros((target_count,) + values.shape[1:])
    for route, block in kernel:
        weights = (block / block.sum(-1, keepdim=True)).to(values.dtype)
        result = torch.einsum('tu,uhd->thd', weights, values.index_select(0, route.sources))
        output = output.index_copy(0, route.output_rows, result)
    return output


class WindowTreeLoom(reference.WindowTreeLoom):
    """Same parameters and input observations; omit work with no path to loss."""
    def forward(self, batch, *, mode='structure'):
        reference.require(type(batch) is Batch, 'Prepared focal executor Batch required')
        reference.require(mode in ('structure', 'leaf_only', 'no_cross'), 'Declared matched communication control')
        source, focal_batch = batch.source, batch.focal
        reference.require(len(source.cuts) == self.scales, 'Model and observed cut count agree')
        if focal_batch is None:
            return super().forward(source, mode=mode)
        if mode == 'no_cross':
            base, gates, degree = self.bases(focal_batch)
            _, _, second = self.sweep(focal_batch, base, gates, degree)
        else:
            base, gates, degree = self.bases(source)
            up, outside, leaf = self.sweep(source, base, gates, degree)
            focal_base = base.index_select(0, batch.nodes)
            injection = torch.zeros_like(focal_base)
            if mode == 'structure':
                for si, (cut, focal_cut, cut_index) in enumerate(zip(source.cuts, focal_batch.cuts, batch.cut_indices)):
                    z = self.cut_project[si](torch.cat((up[cut.representatives] + outside[cut.representatives],
                                                       cut.valid_time[:, None], cut.root_saturated[:, None]), -1))
                    kernel = focal_context_kernel(batch, self.log_lengths[si].exp())
                    transported = focal_temporal_mean(z[cut.groups], kernel, len(batch.trees))
                    delta = reference.mean_to_cut(transported, focal_cut) - z[cut_index]
                    residual = self.cut_output[si](delta) * self.residual_gate[si].sigmoid() / self.scales
                    injection = injection.index_add(0, focal_cut.representatives, residual)
            kernel = focal_context_kernel(batch, self.log_lengths[-1].exp())
            delta = focal_temporal_mean(leaf, kernel, len(batch.trees)) - leaf[batch.trees]
            residual = self.leaf_output(delta) * self.residual_gate[-1].sigmoid()
            injection = injection.index_add(0, focal_batch.leaves.reshape(-1), residual.reshape(-1, self.width))
            focal_gates = tuple(gate.index_select(0, batch.nodes) for gate in gates)
            _, _, second = self.sweep(focal_batch, focal_base + injection, focal_gates, degree[batch.nodes])
        focal = self.encode_snps(source.focal_genotype, source.focal_alleles,
                                 source.tree_window[source.focal_tree], source)
        row = torch.arange(len(source.focal_tree), device=source.parent.device)
        context = second[focal_batch.focal_tree, source.focal_sample]
        snp = focal[row, source.focal_sample]
        metadata = source.metadata[source.tree_window[source.focal_tree], source.focal_sample]
        logits = self.classifier(torch.cat((context, snp, metadata), -1))
        if mode == 'no_cross' and batch.preserve_edge_zero_gradients:
            # Frozen v1 evaluates omitted edged trees. Their zero gradients
            # still participate in AdamW decay/momentum, unlike grad=None.
            modules = (self.up_gate, self.down_gate, self.up_message, self.down_message,
                       self.down_update, self.down_norm)
            anchor = sum(parameter.sum() * 0 for module in modules for parameter in module.parameters())
            logits = logits + anchor
        return logits
