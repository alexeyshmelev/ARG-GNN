"""Fixed observed-reference augmentation for every marginal-tree node.

Domain input adapter/control, not a novelty claim. Precompute once on CPU,
then pass the same augmented GraphSequence to every comparison family.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from .data import GraphSequence
from .genealogy_baselines import _observed_trees
from .models import GraphSequenceInput, IndependentGNN, IntervalLineageGNN


NODE_SUMMARY_FIELDS = (
    'descendant_reference_fraction',
    'panel_reference_fraction',
    'minimum_encoded_tmrca',
    'mean_encoded_tmrca',
    'std_encoded_tmrca',
    'source_minimum_reference_fraction',
    'global_nearest_reference_vote',
)


@dataclass
class ReferenceNodeStatistics:
    features: Tensor                    # [N,R,7+S], source-major order
    descendant_reference_counts: Tensor # [N,R]
    total_reference_counts: Tensor      # [N,R]
    minimum_encoded_tmrca: Tensor        # [N,R]
    nearest_votes: Tensor               # [N,R]
    kernel_evidence: Tensor             # [N,R,S]


class ReferenceFeatureAdapter(nn.Module):
    """Append exact fixed-kernel source summaries, retaining original x first.

    R is the number of observed reference categories, and may differ from the
    downstream ancestry class count. All selected panel labels are one-hot.
    A reference descendant of node u has MRCA(u,reference)=u, including self
    at a labeled sample leaf. Outside references are accumulated in a downward
    pass over sibling shells. No truth fields or arbitrary ID embeddings exist.
    """
    def __init__(self, input_dim: int, *, reference_columns: Sequence[int],
                 time_column: int, branch_column: int | None = None,
                 time_scale: float = 10000.0,
                 kernel_scales: Sequence[float] = (0.05, 0.1, 0.2),
                 validate: bool = True):
        super().__init__()
        if input_dim < 1:
            raise ValueError('input_dim must be positive')
        self.input_dim = int(input_dim)
        self.reference_columns = tuple(int(c) for c in reference_columns)
        self.time_column = int(time_column)
        self.branch_column = branch_column
        if len(self.reference_columns) < 2 or len(set(self.reference_columns)) != len(self.reference_columns):
            raise ValueError('Declare at least two unique observed reference categories')
        if any(c < 0 or c >= input_dim for c in self.reference_columns) or not 0 <= time_column < input_dim:
            raise ValueError('Feature column is outside the original input dimension')
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError('time_scale must be finite and positive')
        self.time_scale = float(time_scale)
        self.kernel_scales = tuple(float(s) for s in kernel_scales)
        if any(not math.isfinite(s) or s < 1e-6 for s in self.kernel_scales):
            raise ValueError('Fixed kernel scales must be finite and >= 1e-6 in encoded-time units')
        self.validate_inputs = bool(validate)

    @property
    def num_reference_categories(self) -> int:
        return len(self.reference_columns)

    @property
    def summary_fields(self) -> tuple[str, ...]:
        return NODE_SUMMARY_FIELDS + tuple(f'kernel_evidence_scale_{s:g}' for s in self.kernel_scales)

    @property
    def added_dim(self) -> int:
        return self.num_reference_categories * len(self.summary_fields)

    @property
    def output_dim(self) -> int:
        return self.input_dim + self.added_dim

    def node_statistics(self, seq: GraphSequenceInput) -> ReferenceNodeStatistics:
        if seq.x.device.type != 'cpu':
            raise ValueError('Precompute observed reference features on CPU before moving sequences to GPU')
        if seq.x.shape[1] != self.input_dim:
            raise ValueError('Adapter expects original x only. Do not augment an already augmented sequence')
        _, reference, times, trees = _observed_trees(seq, self.reference_columns,
                                                   self.time_column, self.branch_column,
                                                   self.validate_inputs)
        n, r = reference.shape
        s = len(self.kernel_scales)
        subtree = reference.copy()
        for tree in trees:
            for u in reversed(tree.preorder):
                p = tree.parent[u]
                if p != -1:
                    subtree[p] += subtree[u]
        outside = np.zeros((n, r), dtype=np.float64)
        first = np.zeros_like(outside)
        second = np.zeros_like(outside)
        outside_minimum = np.full_like(outside, np.inf)
        outside_minimum_count = np.zeros_like(outside)
        outside_log_kernel = np.full((n, r, s), -np.inf, dtype=np.float64)
        scales = np.asarray(self.kernel_scales)
        with np.errstate(divide='ignore'):
            for tree in trees:
                for u in tree.preorder:
                    p = tree.parent[u]
                    if p == -1:
                        continue
                    shell = subtree[p] - subtree[u]
                    outside[u] = outside[p] + shell
                    first[u] = first[p] + shell * times[p]
                    second[u] = second[p] + shell * times[p] ** 2
                    if s:
                        log_shell = np.log(shell)[:, None] - times[p] / scales[None, :]
                        outside_log_kernel[u] = np.logaddexp(outside_log_kernel[p], log_shell)
                    candidate = np.where(shell > 0, times[p], np.inf)
                    smaller = candidate < outside_minimum[p]
                    equal = candidate == outside_minimum[p]
                    outside_minimum[u] = np.minimum(candidate, outside_minimum[p])
                    outside_minimum_count[u] = np.where(smaller, shell,
                        np.where(equal, shell + outside_minimum_count[p], outside_minimum_count[p]))
            total = outside + subtree
            if (total <= 0).any():
                raise ValueError('Every selected observed reference category must occur in every tree')
            inside_minimum = np.where(subtree > 0, times[:, None], np.inf)
            minimum = np.minimum(inside_minimum, outside_minimum)
            smaller = inside_minimum < outside_minimum
            equal = inside_minimum == outside_minimum
            minimum_count = np.where(smaller, subtree,
                np.where(equal, subtree + outside_minimum_count, outside_minimum_count))
            mean = (first + subtree * times[:, None]) / total
            mean_square = (second + subtree * times[:, None] ** 2) / total
            std = np.sqrt(np.maximum(mean_square - mean ** 2, 0))
            closest_time = minimum.min(axis=1, keepdims=True)
            closest_count = np.where(minimum == closest_time, minimum_count, 0)
            nearest_votes = closest_count / closest_count.sum(axis=1, keepdims=True)
            components = [subtree / total, total / total.sum(axis=1, keepdims=True),
                          minimum, mean, std, minimum_count / total, nearest_votes]
            if s:
                inside_log_kernel = np.log(subtree)[:, :, None] - times[:, None, None] / scales[None, None, :]
                combined_log_kernel = np.logaddexp(outside_log_kernel, inside_log_kernel)
                class_log_density = combined_log_kernel - np.log(total)[:, :, None]
                normalizer = np.logaddexp.reduce(class_log_density, axis=1, keepdims=True)
                evidence = np.exp(class_log_density - normalizer)
                components.extend(evidence[:, :, i] for i in range(s))
            else:
                evidence = np.empty((n, r, 0), dtype=np.float64)
        feature_array = np.stack(components, axis=-1)
        if not np.isfinite(feature_array).all():
            raise ValueError('Nonfinite observed reference statistics')
        def tensor(a):
            return torch.as_tensor(a, dtype=seq.x.dtype, device='cpu')
        return ReferenceNodeStatistics(tensor(feature_array), tensor(subtree), tensor(total),
                                       tensor(minimum), tensor(nearest_votes), tensor(evidence))

    def precompute(self, seq: GraphSequenceInput) -> GraphSequence:
        stats = self.node_statistics(seq)
        augmented_x = torch.cat((seq.x, stats.features.flatten(start_dim=1)), dim=1)
        return GraphSequence(x=augmented_x, edge_index=seq.edge_index,
                             node_graph=seq.node_graph, node_ids=seq.node_ids, clades=seq.clades,
                             intervals=seq.intervals, query_node_ids=seq.query_node_ids,
                             positions=seq.positions, target_graph=seq.target_graph,
                             window_bounds=getattr(seq, 'window_bounds', None))

    def forward(self, seq: GraphSequenceInput) -> GraphSequence:
        return self.precompute(seq)

    def manifest_provenance(self, original_feature_names: Sequence[str] | None = None,
                            source_tensor_sha256: str | None = None) -> dict:
        """Return serializable feature provenance, without writing a manifest."""
        original = list(original_feature_names) if original_feature_names is not None else [f'x_{i}' for i in range(self.input_dim)]
        if len(original) != self.input_dim:
            raise ValueError('Original feature names must cover the original input dimension')
        source_names = [original[c] for c in self.reference_columns]
        added_names = [f'{name}.{field}' for name in source_names for field in self.summary_fields]
        configuration = {
            'schema_version': 1,
            'kind': 'fixed_observed_reference_node_summary',
            'scientific_status': 'domain_input_adapter_control_no_novelty_claim',
            'input_dim': self.input_dim, 'output_dim': self.output_dim,
            'original_feature_names': original, 'added_feature_names': added_names,
            'reference_columns': list(self.reference_columns),
            'reference_category_names': source_names,
            'time_column': self.time_column, 'branch_column': self.branch_column,
            'time_encoding': 'log1p(generations)/log1p(time_scale)',
            'time_scale': self.time_scale, 'kernel_scales_encoded_time': list(self.kernel_scales),
            'kernel_definition': 'mean_reference_exp(-encoded_tmrca/scale), normalized across reference categories',
            'kernel_numerics': 'logaddexp accumulation and category log normalization',
            'feature_order': 'original_x_first_then_reference_category_major_summary_fields',
            'summary_fields': list(self.summary_fields),
            'node_scope': 'all nodes independently within each observed complete tree',
            'reference_self_relation': 'MRCA(node,descendant_reference)=node, including reference self',
            'truth_annotations_used': [], 'learned_arbitrary_id_embeddings': False,
            'learnable_adapter_parameters': 0,
            'tree_requirement': 'connected acyclic rooted tree with monotone observed node times',
            'cache_stage': 'CPU before window sampling and GPU transfer',
        }
        canonical = json.dumps(configuration, sort_keys=True, separators=(',', ':')).encode()
        configuration['configuration_sha256'] = hashlib.sha256(canonical).hexdigest()
        configuration['implementation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        dependency = Path(__file__).with_name('genealogy_baselines.py')
        configuration['topology_helper_dependency_sha256'] = hashlib.sha256(dependency.read_bytes()).hexdigest()
        if source_tensor_sha256 is not None:
            if len(source_tensor_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_tensor_sha256):
                raise ValueError('source_tensor_sha256 must be a lowercase SHA256 hex digest')
            configuration['source_tensor_sha256'] = source_tensor_sha256
        return configuration


class _PreparedReferenceModel(nn.Module):
    def prepare_sequence(self, raw_cpu_sequence: GraphSequenceInput) -> GraphSequence:
        return self.adapter.precompute(raw_cpu_sequence)

    def forward(self, prepared_sequence: GraphSequenceInput) -> Tensor:
        if prepared_sequence.x.shape[1] != self.adapter.output_dim:
            raise ValueError('Call prepare_sequence(raw CPU sequence) once before moving/caching inputs. Forward requires augmented x')
        return self.model(prepared_sequence)


class ReferenceGNN(_PreparedReferenceModel):
    """IndependentGNN on the common precomputed observed-reference adapter."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, *, reference_columns: Sequence[int],
                 time_column: int, branch_column: int | None = None,
                 time_scale: float = 10000.0,
                 kernel_scales: Sequence[float] = (0.05, 0.1, 0.2), validate: bool = True):
        super().__init__()
        self.adapter = ReferenceFeatureAdapter(input_dim, reference_columns=reference_columns,
            time_column=time_column, branch_column=branch_column, time_scale=time_scale,
            kernel_scales=kernel_scales, validate=validate)
        self.model = IndependentGNN(self.adapter.output_dim, num_classes,
                                    hidden_dim=hidden_dim, num_layers=num_layers, validate=validate)


class ReferenceIntervalGNN(_PreparedReferenceModel):
    """IntervalLineageGNN receiving the identical all-node observed augmentation."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64,
                 num_layers: int = 2, *, reference_columns: Sequence[int],
                 time_column: int, branch_column: int | None = None,
                 time_scale: float = 10000.0,
                 kernel_scales: Sequence[float] = (0.05, 0.1, 0.2), validate: bool = True,
                 **interval_options):
        super().__init__()
        self.adapter = ReferenceFeatureAdapter(input_dim, reference_columns=reference_columns,
            time_column=time_column, branch_column=branch_column, time_scale=time_scale,
            kernel_scales=kernel_scales, validate=validate)
        self.model = IntervalLineageGNN(self.adapter.output_dim, num_classes,
            hidden_dim=hidden_dim, num_layers=num_layers, validate=validate, **interval_options)
