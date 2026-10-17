"""Inactive, source-only minimal ordered-marker learning integration.

Main pair: identical ordered evidence and parameters, residual before versus
immediately after the second full-tree sweep. Native-hold readout is unchanged.
No target/file reader, optimizer, fit, launcher, guard or resource authority.
No model instantiation, forward or numerical campaign occurred at authoring.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.model_v1 import H058FullParentGNNV1
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.integration_hooks_v3 import PackedSameMarkerPayloadV3
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.observation_binding_v3 import PackedObservationBindingV3
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import LOG_MIN, LOG_MAX, require
from .positive_marker_backend_v1 import PositiveParentPlanV1, PositiveMarkerBackendV1

ARMS = ('ordered_marker_tree_feedback','ordered_marker_sample_readout')
ADDED_ORDER_PARAMETERS = 1440  # Analytic only, runtime check when later authorized.
TOTAL_PARAMETERS = 33966      # 31812 H045 +714 H058 +1440 ordered module.


@dataclass(frozen=True)
class OrderedMarkerPayloadV1(PackedSameMarkerPayloadV3):
    positive_plan: PositiveParentPlanV1


def attach_positive_plan_v1(baseline_payload,positive_plan):
    """No data reader; preserve the actual existing bank/roster/site binding."""
    require(isinstance(baseline_payload,PackedSameMarkerPayloadV3)
        and isinstance(positive_plan,PositiveParentPlanV1), 'Exact supplied original payload/plan required')
    positive_plan.validate(baseline_payload.route_cpu)
    return OrderedMarkerPayloadV1(episodes=baseline_payload.episodes,
        route_cpu=baseline_payload.route_cpu,binding=baseline_payload.binding,
        positive_plan=positive_plan)


class OrderedWithinHoldResidualV1(nn.Module):
    """[S_hold,Q,24] -> [Q,32], preserving order before hold reduction.

    Physical gaps are not inputs. Every observed marker in its existing exact
    row order is retained. The GRU resets at each native hold by design; the
    original chromosome-wide sampled-haplotype BiGRU still crosses all holds.
    """
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(24,8,bias=False)       #192
        self.marker_gru = nn.GRU(8,8,bidirectional=True) #864
        self.rank_input = nn.Linear(16,8,bias=False) #128
        self.output = nn.Linear(8,32,bias=False)     #256
        nn.init.zeros_(self.output.weight)
        self.float()

    def forward(self,features):
        require(features.dtype == torch.float32 and features.ndim == 3 and features.shape[-1] == 24,
            'Unpooled FP32 marker/query24 evidence required')
        q = features.shape[1]
        if features.shape[0] == 0:
            return features.new_zeros((q,32))
        sequence = self.input(features).tanh()
        _history,ends = self.marker_gru(sequence)
        # PyTorch state order: forward final, reverse final for one GRU layer.
        summary = torch.cat((ends[0],ends[1]),-1)
        residual = self.output(self.rank_input(summary).tanh())
        available = (features[...,16:] > 0).any(0).any(-1)
        return residual*available[:,None].to(residual.dtype)


class OrderedMarkerLearningGNNV1(H058FullParentGNNV1):
    """Complete unchanged H058 body plus identical1440 weights in either arm.

    Existing H058 mean/max injection remains common. Only the placement of the
    extra ordered residual differs. All trees and all952 sampled identities
    traverse both tree sweeps and the original full genomic BiGRU.
    """
    def __init__(self,*,arm,checkpoint_backbone=True,checkpoint_added=True):
        require(arm in ARMS and type(checkpoint_added) is bool,'Explicit new matched arm required')
        super().__init__(checkpoint_backbone=checkpoint_backbone)
        self.arm = arm
        self.checkpoint_added = checkpoint_added
        self.ordered_marker = OrderedWithinHoldResidualV1()
        self.positive_kernel = PositiveMarkerBackendV1()  # No learned state.
        self.float()

    def _hold_residuals(self,hold,plan,upward,geometry,support,genotype):
        view = hold._transient_device_view(upward.device)
        up = upward.index_select(0,view.native_node)
        edge9 = torch.cat((geometry.index_select(0,view.native_node),support.index_select(0,view.native_node)),-1)
        parent_up = up.index_select(0,view.parent.clamp_min(0))
        score = self.same_marker.gate.projection(torch.cat((edge9,up,parent_up),-1)).squeeze(-1)
        log_gamma = LOG_MAX+(LOG_MIN-LOG_MAX)*score.sigmoid()
        calls = genotype[hold.site_start:hold.site_stop]
        feature,mask = self.positive_kernel.marker_sequence(log_gamma,calls,hold,plan,
            self.same_marker.evidence,marker_block=16)
        # Same H058 reduction semantics, including its availability-channel mask.
        if feature.shape[0] == 0:
            pooled = feature.new_zeros((len(plan.query_node),48))
        else:
            count = mask.to(feature.dtype).sum(0)
            mean = torch.where(mask,feature,torch.zeros_like(feature)).sum(0)/count.clamp_min(1)
            maximum = torch.where(mask,feature,torch.full_like(feature,-torch.inf)).amax(0)
            maximum = torch.where(count > 0,maximum,torch.zeros_like(maximum))
            pooled = torch.cat((mean,maximum),-1)
        common = self.same_marker.output(self.same_marker.input(pooled).tanh())
        ordered = self.ordered_marker(feature)
        return common,ordered

    def encode_trees(self,bank,*,flags,control='aligned'):
        require(isinstance(flags,OrderedMarkerPayloadV1) and control == 'aligned'
            and isinstance(flags.binding,PackedObservationBindingV3),
            'Original aligned H058 payload plus exact observation-only positive plan required')
        episodes,route = flags.episodes,flags.route_cpu
        flags.binding.validate(bank,episodes,route)
        flags.positive_plan.validate(route)
        require(len(route.holds) == bank.sample_index.shape[0]
            and torch.equal(bank.spans,episodes.hold_span), 'Original complete hold joins required')
        base,ug,dg,degree = self.prepare_tree_base_v1(bank,control=control)
        upward,_outside = self.sweep_tree_state_v1(bank,base,ug,dg,degree)
        injection = self.clade_coupling(upward,episodes.flags)+self.episode_coupling(upward,episodes,arm='own_state')
        common_rows,ordered_rows = [],[]
        for hold,plan in zip(route.holds,flags.positive_plan.holds):
            def reduce_hold(up,geo,sup,gt,hold=hold,plan=plan):
                return self._hold_residuals(hold,plan,up,geo,sup,gt)
            args = (upward,bank.edge_geometry,bank.branch_support_true,bank.genotype)
            if self.checkpoint_added and self.training and torch.is_grad_enabled():
                common,ordered = checkpoint(reduce_hold,*args,use_reentrant=False,preserve_rng_state=True)
            else:
                common,ordered = reduce_hold(*args)
            common_rows.append(common)
            ordered_rows.append(ordered)
        flags.binding.validate(bank,episodes,route)
        flags.positive_plan.validate(route)
        rows = route.global_query_node.to(device=upward.device,dtype=torch.long)
        common = torch.cat(common_rows,0)
        ordered = torch.cat(ordered_rows,0)
        added = common+ordered if self.arm == ARMS[0] else common
        injection = injection.index_add(0,rows,added)
        second_up,second_out = self.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        context = self.leaf_context_v1(bank,second_up,second_out) # [T,952,32]
        if self.arm == ARMS[1]:
            # Same sampled query identities, after full-tree messages. The
            # original BiGRU/readout follows through inherited forward_holds.
            t,h,d = context.shape
            query = bank.query_column
            flat_rows = (torch.arange(t,device=context.device)[:,None]*h+query[None,:]).flatten()
            context = context.reshape(t*h,d).index_add(0,flat_rows,ordered).reshape(t,h,d)
        return context


def initialize_supplied_h045_v1(model,state):
    """Future supplied-state check only. No checkpoint reader or optimizer."""
    require(isinstance(model,OrderedMarkerLearningGNNV1) and isinstance(state,Mapping),
        'Exact new model and externally authenticated original H045 state required')
    expected = model.state_dict()
    added = {key for key in expected if key.startswith(('same_marker.','ordered_marker.'))}
    require(set(state) == set(expected)-added, 'Every original H045 state key required')
    for key,value in state.items():
        require(isinstance(value,torch.Tensor) and value.dtype == torch.float32
            and value.device.type == 'cpu' and value.shape == expected[key].shape
            and bool(torch.isfinite(value).all()), 'Original H045 tensor differs: '+key)
    require(sum(p.numel() for p in model.parameters()) == TOTAL_PARAMETERS
        and not bool(model.same_marker.output.weight.count_nonzero())
        and not bool(model.ordered_marker.output.weight.count_nonzero()),
        'Analytic33966 capacity and both zero residual outputs required')
    result = model.load_state_dict(state,strict=False)
    require(set(result.missing_keys) == added and not result.unexpected_keys,
        'Only new H058/ordered module weights may be absent')
    return dict(allocated_parameters=TOTAL_PARAMETERS,added_order_parameters=ADDED_ORDER_PARAMETERS,
        initial_function_equality_unverified=True,optimizer_constructed=False)


def clone_matched_pair_v1(initialized):
    """Independent storage, identical initialization; caller owns RNG/provenance."""
    require(isinstance(initialized,OrderedMarkerLearningGNNV1), 'Supplied initialized new model required')
    left = copy.deepcopy(initialized)
    right = copy.deepcopy(initialized)
    left.arm,right.arm = ARMS
    return {ARMS[0]:left,ARMS[1]:right}
