"""Inactive source-only complete ordered-marker pair with streaming evidence.

No reader, target resolver, execution wrapper, guard or admission is defined.
All dense/positive evidence and marker recurrent history stay differentiable.
The scientific producer in this package supplies the complete TRAIN/VAL caller.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.model_v1 import (
    H058FullParentGNNV1, validate_supplied_warmstart_v1)
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.integration_hooks_v3 import PackedSameMarkerPayloadV3
from research.reviews.same_marker_full_parent_binding_successor_source_20261016_v1.observation_binding_v3 import PackedObservationBindingV3
from research.reviews.biological_operator_next_fresh_20261016_v1.packed_csr_v2.same_marker_packed_csr_v2 import LOG_MIN,LOG_MAX,require
from .positive_marker_backend_v2 import PositiveParentPlanV2,PositiveMarkerBackendV2
from .protocol_v2 import ARMS,PARAMETERS,WARM_CHECKPOINT_REF,WARM_COMPLETION_REF

ADDED_ORDER_PARAMETERS=1440
TOTAL_PARAMETERS=33966


@dataclass(frozen=True)
class OrderedMarkerPayloadV2(PackedSameMarkerPayloadV3):
    positive_plan: PositiveParentPlanV2


def attach_positive_plan_v2(baseline_payload,positive_plan):
    require(isinstance(baseline_payload,PackedSameMarkerPayloadV3)
        and isinstance(positive_plan,PositiveParentPlanV2),'Exact original payload and positive plan required')
    positive_plan.validate(baseline_payload.route_cpu)
    return OrderedMarkerPayloadV2(episodes=baseline_payload.episodes,route_cpu=baseline_payload.route_cpu,
        binding=baseline_payload.binding,positive_plan=positive_plan)


class OrderedWithinHoldResidualV2(nn.Module):
    """Same1440-parameter bidirectional width8 function, streamed by direction.

    Two public unidirectional GRUs implement the two directions of the original
    bidirectional GRU. Their parameter names differ from inactive v1. Both new
    arms share this exact initialization; no v1 numerical checkpoint is reused.
    Hidden states cross blocks without detach and reset only at a native hold.
    """
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(24,8,bias=False)
        self.marker_forward=nn.GRU(8,8)
        self.marker_reverse=nn.GRU(8,8)
        self.rank_input=nn.Linear(16,8,bias=False)
        self.output=nn.Linear(8,32,bias=False)
        nn.init.zeros_(self.output.weight)
        self.float()

    def stream(self,log_gamma,calls,hold,plan,dense,kernel):
        require(log_gamma.dtype==torch.float32 and calls.ndim==2
            and calls.shape==(plan.site_stop-plan.site_start,hold.haplotypes),
            'All original observed hold markers and FP32 gate required')
        q=len(plan.query_node)
        sums=log_gamma.new_zeros((q,24));count=torch.zeros_like(sums)
        maximum=log_gamma.new_full((q,24),-torch.inf)
        available=torch.zeros((q,),dtype=torch.bool,device=log_gamma.device)
        hf=log_gamma.new_zeros((1,q,8));hr=torch.zeros_like(hf)
        starts=tuple(range(0,len(plan.covers),16))
        for first in starts:
            last=min(first+16,len(plan.covers))
            # Bind integer offsets per closure: later backward must not observe
            # the loop's final offsets. Only immutable hold/plan are captured.
            def forward_block(lg,gt,state,first=first,last=last):
                feature,mask=kernel.marker_block_v2(lg,gt,hold,plan,dense,first=first,last=last)
                _history,hidden=self.marker_forward(self.input(feature).tanh(),state)
                add=torch.where(mask,feature,torch.zeros_like(feature)).sum(0)
                high=torch.where(mask,feature,-torch.inf).amax(0)
                number=mask.to(feature.dtype).sum(0)
                valid=(feature[...,16:]>0).any(0).any(-1)
                return hidden,add,high,number,valid
            args=(log_gamma,calls[first:last],hf)
            if torch.is_grad_enabled():
                hf,add,high,number,valid=checkpoint(forward_block,*args,
                    use_reentrant=False,preserve_rng_state=False)
            else:
                hf,add,high,number,valid=forward_block(*args)
            # Exact existing H058 block statistics/reduction order.
            sums,count,maximum=sums+add,count+number,torch.maximum(maximum,high)
            available=available|valid
        for first in reversed(starts):
            last=min(first+16,len(plan.covers))
            def reverse_block(lg,gt,state,first=first,last=last):
                feature,_mask=kernel.marker_block_v2(lg,gt,hold,plan,dense,first=first,last=last)
                _history,hidden=self.marker_reverse(self.input(feature.flip(0)).tanh(),state)
                return hidden
            args=(log_gamma,calls[first:last],hr)
            if torch.is_grad_enabled():
                hr=checkpoint(reverse_block,*args,use_reentrant=False,preserve_rng_state=False)
            else:
                hr=reverse_block(*args)
        pooled=torch.cat((sums/count.clamp_min(1),
            torch.where(count>0,maximum,torch.zeros_like(maximum))),-1)
        if not starts:
            ordered=log_gamma.new_zeros((q,32))
        else:
            summary=torch.cat((hf[0],hr[0]),-1)
            ordered=self.output(self.rank_input(summary).tanh())
            ordered=ordered*available[:,None].to(ordered.dtype)
        return pooled,ordered


class OrderedMarkerLearningGNNV2(H058FullParentGNNV1):
    """All original body/714 weights plus identical1440 ordered weights.

    Only ordered-residual placement differs between the two registered names.
    Both original complete tree sweeps, all952 sampled contexts and the genomic
    BiGRU/classifier remain. Block checkpointing is mandatory when grad is on.
    """
    def __init__(self,*,arm,checkpoint_backbone=True,checkpoint_added=True):
        require(arm in ARMS and checkpoint_backbone is True and checkpoint_added is True,
            'Explicit matched arm and original nested checkpoint boundaries required')
        super().__init__(checkpoint_backbone=True)
        self.arm=arm
        self.ordered_marker=OrderedWithinHoldResidualV2()
        self.positive_kernel=PositiveMarkerBackendV2()
        self.float()

    def _hold_residuals(self,hold,plan,upward,geometry,support,genotype):
        view=hold._transient_device_view(upward.device)
        up=upward.index_select(0,view.native_node)
        edge9=torch.cat((geometry.index_select(0,view.native_node),support.index_select(0,view.native_node)),-1)
        parent_up=up.index_select(0,view.parent.clamp_min(0))
        score=self.same_marker.gate.projection(torch.cat((edge9,up,parent_up),-1)).squeeze(-1)
        log_gamma=LOG_MAX+(LOG_MIN-LOG_MAX)*score.sigmoid()
        calls=genotype[hold.site_start:hold.site_stop]
        pooled,ordered=self.ordered_marker.stream(log_gamma,calls,hold,plan,
            self.same_marker.evidence,self.positive_kernel)
        common=self.same_marker.output(self.same_marker.input(pooled).tanh())
        return common,ordered

    def encode_trees(self,bank,*,flags,control='aligned'):
        require(isinstance(flags,OrderedMarkerPayloadV2) and control=='aligned'
            and isinstance(flags.binding,PackedObservationBindingV3),
            'Original aligned H058 payload plus exact positive plan required')
        episodes,route=flags.episodes,flags.route_cpu
        flags.binding.validate(bank,episodes,route);flags.positive_plan.validate(route)
        require(len(route.holds)==bank.sample_index.shape[0]
            and torch.equal(bank.spans,episodes.hold_span),'Original complete native hold joins required')
        base,ug,dg,degree=self.prepare_tree_base_v1(bank,control=control)
        upward,_outside=self.sweep_tree_state_v1(bank,base,ug,dg,degree)
        injection=self.clade_coupling(upward,episodes.flags)+self.episode_coupling(upward,episodes,arm='own_state')
        common_rows,ordered_rows=[],[]
        for hold,plan in zip(route.holds,flags.positive_plan.holds,strict=True):
            def reduce_hold(up,geo,sup,gt,hold=hold,plan=plan):
                return self._hold_residuals(hold,plan,up,geo,sup,gt)
            args=(upward,bank.edge_geometry,bank.branch_support_true,bank.genotype)
            if torch.is_grad_enabled():
                common,ordered=checkpoint(reduce_hold,*args,use_reentrant=False,preserve_rng_state=True)
            else:
                common,ordered=reduce_hold(*args)
            common_rows.append(common);ordered_rows.append(ordered)
        flags.binding.validate(bank,episodes,route);flags.positive_plan.validate(route)
        rows=route.global_query_node.to(device=upward.device,dtype=torch.long)
        common=torch.cat(common_rows,0);ordered=torch.cat(ordered_rows,0)
        added=common+ordered if self.arm==ARMS[0] else common
        injection=injection.index_add(0,rows,added)
        second_up,second_out=self.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        context=self.leaf_context_v1(bank,second_up,second_out)
        if self.arm==ARMS[1]:
            t,h,d=context.shape
            flat_rows=(torch.arange(t,device=context.device)[:,None]*h+bank.query_column[None,:]).flatten()
            context=context.reshape(t*h,d).index_add(0,flat_rows,ordered).reshape(t,h,d)
        return context


def validate_trainable_fp32_v2(model):
    require(isinstance(model,OrderedMarkerLearningGNNV2)
        and sum(p.numel() for p in model.parameters())==TOTAL_PARAMETERS
        and all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters()),
        'Every33966 allocated parameter must remain trainable FP32')


def initialize_supplied_h045_v2(model,state):
    require(isinstance(state,Mapping),'Externally authenticated original H045 state required')
    validate_trainable_fp32_v2(model)
    expected=model.state_dict()
    added={key for key in expected if key.startswith(('same_marker.','ordered_marker.'))}
    require(set(state)==set(expected)-added,'Every original H045 state key required')
    for key,value in state.items():
        require(isinstance(value,torch.Tensor) and value.dtype==torch.float32
            and value.device.type=='cpu' and value.shape==expected[key].shape
            and bool(torch.isfinite(value).all()),'Original H045 tensor differs: '+key)
    require(not bool(model.same_marker.output.weight.count_nonzero())
        and not bool(model.ordered_marker.output.weight.count_nonzero()),
        'Both new residual outputs start exactly zero')
    result=model.load_state_dict(state,strict=False)
    require(set(result.missing_keys)==added and not result.unexpected_keys,
        'Only new H058/ordered weights may be absent')
    validate_trainable_fp32_v2(model)
    return dict(allocated_parameters=TOTAL_PARAMETERS,added_order_parameters=ADDED_ORDER_PARAMETERS,
        all_parameters_trainable_FP32=True,initial_function_equality_unverified=True)


def build_family_v2(payload,completion,*,checkpoint_ref,completion_ref,device,
                    arms=ARMS,checkpoint_backbone=True):
    require(tuple(arms)==ARMS and checkpoint_backbone is True and torch.device(device).type=='cpu',
        'Full matched CPU pair with original checkpoint boundaries required')
    state=validate_supplied_warmstart_v1(payload,completion,
        checkpoint_ref=checkpoint_ref,completion_ref=completion_ref)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        first=OrderedMarkerLearningGNNV2(arm=ARMS[0])
        initialization=initialize_supplied_h045_v2(first,state)
        second=copy.deepcopy(first);second.arm=ARMS[1]
    models={ARMS[0]:first,ARMS[1]:second}
    initial=first.state_dict()
    for arm,value in models.items():
        validate_trainable_fp32_v2(value)
        require(all(torch.equal(value.state_dict()[k],v) for k,v in state.items()),
            'Every original H045 tensor retained')
        require(all(torch.equal(value.state_dict()[k],v) for k,v in initial.items()),
            'Both placement arms start from identical learned tensors')
        value.to(device=device,dtype=torch.float32)
    pointers=[{p.data_ptr() for p in value.parameters()} for value in models.values()]
    require(not pointers[0]&pointers[1],'Independent placement-arm learned storage')
    optimizers={arm:torch.optim.AdamW(value.parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8) for arm,value in models.items()}
    require(all(not value.state for value in optimizers.values()),'Fresh independent AdamW moments')
    return models,optimizers,dict(initialization,parameters=PARAMETERS,seed=101,
        warmstart_checkpoint_ref=WARM_CHECKPOINT_REF,warmstart_completion_ref=WARM_COMPLETION_REF,
        independent_reset_AdamW=True,initial_predictor_equality_requires_scheduled_forward=True)
