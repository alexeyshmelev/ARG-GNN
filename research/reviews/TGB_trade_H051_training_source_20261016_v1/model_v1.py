"""Shared full-tree encoder, country-leaf GRU and readout; carry is sole arm change."""
from __future__ import annotations
import torch
from torch import nn
from research.reviews.TGB_trade_PyG_dataset_source_20261016_v1.whole_country_tree_encoder_v1 import WholeCountryTreeEncoderV1
from research.reviews.H051_conditional_innovation_design_20261016_v1.conditional_innovation_v1 import ConditionalInnovationTransportV1

PRIMARY_ARMS=("tree_only","first_order","conditional_triple")

class TradeOrderedModelV1(nn.Module):
    def __init__(self,arm,*,width=64,destinations=255):
        super().__init__()
        if arm not in PRIMARY_ARMS:raise ValueError("Frozen primary arm")
        self.arm=arm;self.width=width;self.destinations=destinations
        self.encoder=WholeCountryTreeEncoderV1(num_destinations=destinations,hidden_channels=width)
        self.temporal_cell=nn.GRUCell(width,width)
        self.readout=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,destinations))
        self.prior_epsilon=1e-8
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        # Parameter-free; every arm has identical learned parameter names/shapes.
        self.transport=ConditionalInnovationTransportV1(gate="fixed",
            alpha=1. if arm=="conditional_triple" else 0.,backend="pyg")

    def forward_step(self,prepared,state):
        forcing=self.encoder(prepared.graph,schedule=prepared.schedule).leaf_state
        if state is None or self.arm=="tree_only":carry=torch.zeros_like(forcing)
        elif prepared.event_plan is None:carry=state
        else:
            cells=self.transport(state,prepared.event_plan,basis="leaf")
            carry=cells.index_select(0,prepared.event_plan.leaf_bc)
        state=self.temporal_cell(forcing,carry)
        residual=self.readout(torch.cat((forcing,state),dim=1))
        # Fixed causal persistence prior is never compressed through hidden width.
        profile=prepared.graph.leaf_profile.to(dtype=residual.dtype)
        logits=torch.log(profile+self.prior_epsilon)+residual
        return logits,state

def parameter_report_v1(model):
    total=sum(p.numel() for p in model.parameters())
    inactive=(model.temporal_cell.weight_hh.numel() if model.arm=="tree_only" else 0)
    return dict(total_learned_parameters=total,structurally_unused_carry_parameters=inactive,
        structurally_unused_names=["temporal_cell.weight_hh"] if inactive else [],
        potentially_active_parameters=total-inactive,
        note="Potentially active excludes structural zero-carry weights only; actual data can yield other zero gradients",
        retained_temporal_state_rows=255,retained_temporal_state_width=model.width)
