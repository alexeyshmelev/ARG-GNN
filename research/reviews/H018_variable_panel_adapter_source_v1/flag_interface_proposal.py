"""Inactive synthetic-qualified H042 count-interface proposal. No reader, weights or authority."""
from dataclasses import fields
import math
import torch
from arggnn.anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagControlGNNV1, RankEightCladeFlagAdapterV1,
    AnonymousCladeFlagsV1, build_clade_flag_arrays_v1, _cpu_index_array)
from arggnn.mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from arggnn.mutation_routed_whole_tree_gnn_v1 import (
    MutationRoutedWholeTreeInputsV1, _FLOAT_FIELDS, _LEVEL_FIELDS)

def build_variable_panel_flags_v1(bank, *, haplotypes=40, queries=24):
    if type(bank) is not MutationRoutedWholeTreeInputsV1:
        raise TypeError("Exact original observation-only bank required")
    if bank.sample_index.shape[1]!=haplotypes or bank.query_column.shape!=(queries,):
        raise ValueError("Exact declared panel/query counts required")
    parent=_cpu_index_array(bank.parent);samples=_cpu_index_array(bank.sample_index)
    arrays,metadata=build_clade_flag_arrays_v1(parent,samples)
    return AnonymousCladeFlagsV1.from_verified_arrays(arrays,metadata),metadata

class VariablePanelFlagInterfaceV1(AnonymousCladeFlagControlGNNV1):
    """Original H042 learned operators/state layout with explicit40/Q24 count gates."""
    def __init__(self, hidden_dim=32, *, expected_haplotypes=40, expected_queries=24,
                 site_chunk_size=256, checkpoint_sites=True):
        if hidden_dim!=32 or expected_haplotypes!=40 or expected_queries!=24:
            raise ValueError("Only source-reviewed width32/H40/Q24 proposal")
        MutationRoutedFullPanelPyGGNNV1.__init__(
            self,hidden_dim=hidden_dim,expected_haplotypes=expected_haplotypes,
            site_chunk_size=site_chunk_size,checkpoint_sites=checkpoint_sites)
        self.expected_queries=expected_queries
        self.clade_coupling=RankEightCladeFlagAdapterV1()

    def forward_holds(self, bank, *, flags, control="aligned"):
            parameter = next(self.parameters())
            t, h = bank.sample_index.shape
            if (parameter.dtype != torch.float32 or h != self.expected_haplotypes
                    or bank.query_column.numel() != self.expected_queries or not t
                    or bank.parent.device != parameter.device
                    or flags.native_holds != t or flags.haplotypes != h):
                raise ValueError("Complete FP32 declared-panel bank and matching flags required")
            for name in _FLOAT_FIELDS:
                value = getattr(bank, name)
                if value.dtype != torch.float32 or value.device != parameter.device:
                    raise ValueError("Original floating observations/device differ")
            for f in fields(bank):
                value = getattr(bank, f.name)
                values = value if f.name in _LEVEL_FIELDS else (value,)
                if any(v.device != parameter.device for v in values):
                    raise ValueError("Complete observation bank device differs")
            context = self.encode_trees(bank, flags=flags, control=control)
            span = bank.spans.log1p() / math.log(2000001.)
            prev = torch.cat((
                bank.spans.new_zeros(1), .5 * (bank.spans[1:] + bank.spans[:-1]),
            ))
            nxt = torch.cat((
                .5 * (bank.spans[1:] + bank.spans[:-1]), bank.spans.new_zeros(1),
            ))
            geometry = torch.stack((
                span, prev.log1p() / math.log(2000001.),
                nxt.log1p() / math.log(2000001.),
            ), -1)
            sequence = torch.cat((context, geometry[:, None].expand(t, h, 3)), -1)
            history, _ = self.genomic_gru(sequence)
            return self.classifier(torch.cat((context, history), -1))
