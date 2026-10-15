"""Unregistered marker-grid reset-chain prototype. No loader, launcher or authority.

H039 actually completed its native-hold CRF experiment. RFMix and neural CTMCs
are prior art. This module claims neither operator priority nor predictive gain.
All learned/evidence/chain arithmetic is FP32. Original coordinate byte bindings
are preserved, then integral physical bp are represented as int64 for binning.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
import copy
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1, build_clade_flags_v1
from .marker_resolution_readout_v1 import (
    H044BoundMarkerBankV1, MarkerEvidenceBankV1, MarkerResolutionReadoutV1,
    tensor_identity_v1, H042_PASS30_REF, H042_COMPLETION_REF, original_grid_span_nll_v1,
)
from .directed_clade_exchange_crf_v1 import chain_log_marginals_v1

SOURCE = "src/arggnn/marker_grid_reset_chain_gnn_v1.py"
ARMS = ("graph_to_transition", "graph_to_own_confidence_fixed_distance_chain")
MODEL = dict(hidden_dim=32, expected_haplotypes=952, site_chunk_size=256, checkpoint_sites=True)
TOTAL_PARAMETERS = 31933


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer_bp(value):
    _require(value.device.type == "cpu" and not value.requires_grad, "CPU physical coordinates")
    result = value.to(dtype=torch.long)
    _require(torch.equal(value, result), "Physical coordinates must be exact integral bp, no truncation")
    return result.contiguous()


def disjoint_bin_features_v1(markers, bin_edges_bp, *, query_block=32, site_block=8192):
    """One FP32 bin reduction per SNP/query, no long-prefix cancellation.

    A site on an interior edge belongs to its right bin. Original observed tails
    outside scored coverage stay in the tree body but not this new channel.
    No target argument, FP64 accumulator, marker recurrence or disk cache.
    """
    _require(isinstance(markers, MarkerEvidenceBankV1) and markers.genotype.device.type == "cpu"
             and markers.log_evidence.dtype == torch.float32
             and type(query_block) is int and query_block > 0
             and type(site_block) is int and site_block > 0, "Original CPU categorical observations")
    edges, positions, centers = (_integer_bp(x) for x in
        (bin_edges_bp, markers.positions_bp0, markers.centers_bp))
    p, q, s = centers.numel(), markers.query_column.numel(), positions.numel()
    _require(0 < p <= 20000 and 0 < q <= 686 and s <= 2000000
             and edges.shape == (p+1,) and bool((edges[1:] > edges[:-1]).all())
             and bool((edges[1:] - edges[:-1] <= 100).all())
             and bool((centers >= edges[:-1]).all() and (centers < edges[1:]).all()),
             "Original positive disjoint scoring bins of at most100bp")
    site_bin = torch.searchsorted(edges, positions, right=True)-1
    retained = (site_bin >= 0) & (site_bin < p)
    usable_reference = markers.reference_called.all(-1)
    features = torch.empty((p,q,5), dtype=torch.float32)
    for first in range(0,q,query_block):
        last = min(first+query_block,q)
        sums = torch.zeros((p,last-first,5), dtype=torch.float32)
        columns = markers.query_column[first:last]
        for lo in range(0,s,site_block):
            hi = min(lo+site_block,s)
            take = retained[lo:hi]
            index = site_bin[lo:hi][take]
            if not index.numel():
                continue
            calls = markers.genotype[lo:hi].index_select(1,columns)[take].long()
            active = (calls >= 0) & usable_reference[lo:hi][take,None]
            evidence = markers.log_evidence[lo:hi][take].gather(
                2,calls.clamp_min(0)[:,None,:].expand(-1,4,-1)).permute(0,2,1)
            centered = evidence-evidence.mean(-1,keepdim=True)
            joined = torch.cat((centered.masked_fill(~active[...,None],0.),
                                active.to(torch.float32)[...,None]),-1)
            sums.index_add_(0,index,joined)
        count = sums[...,4:5]
        features[:,first:last,:4] = sums[...,:4]/count.clamp_min(1.)
        features[:,first:last,4:5] = count.log1p()
    return features


def _stamps(tensors):
    return tuple((id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,str(t.device)) for t in tensors)


@dataclass(frozen=True)
class MarkerGridResetBundleV1:
    original: H044BoundMarkerBankV1
    flags: object
    features_cpu: torch.Tensor
    grid_hold: torch.Tensor
    delta_bp: torch.Tensor
    pair_left: torch.Tensor
    pair_right: torch.Tensor
    edge_pair: torch.Tensor
    stamps: tuple

    @property
    def bank(self):
        return self.original.bank

    @property
    def source_bank(self):
        return self.original.source_bank

    @property
    def markers(self):
        return self.original.markers

    def tensors(self):
        return (self.flags.node_flags,self.features_cpu,self.grid_hold,self.delta_bp,
                self.pair_left,self.pair_right,self.edge_pair)

    def validate(self):
        bank, markers = self.original.validate()
        _require(self.stamps == _stamps(self.tensors()), "Prepared original grid/flags changed")
        return bank, markers

    def to(self,device):
        self.validate()
        result = type(self)(self.original.to(device),self.flags.to(device),self.features_cpu,
            *(getattr(self,k).to(device) for k in
              ("grid_hold","delta_bp","pair_left","pair_right","edge_pair")),stamps=())
        return type(self)(result.original,result.flags,result.features_cpu,result.grid_hold,
            result.delta_bp,result.pair_left,result.pair_right,result.edge_pair,_stamps(result.tensors()))


def prepare_marker_grid_reset_v1(original, bin_edges_bp):
    """Root supplies an existing H044-bound CPU observation bundle and edges."""
    _require(isinstance(original,H044BoundMarkerBankV1), "Original bound marker observations")
    bank,markers = original.validate()
    _require(bank.parent.device.type == "cpu" and markers.original_binding is not None
             and bank.sample_index.shape[1] == 952 and bank.query_column.shape == (686,),
             "Original complete CPU952/686 panel")
    _require(tensor_identity_v1(bin_edges_bp) ==
             markers.original_binding["array_identities"]["bin_edges_bp"], "Exact original physical scoring edges")
    centers = _integer_bp(markers.centers_bp)
    _require(centers.numel() > 0 and bool((centers[1:]-centers[:-1] == 100).all()),
             "Original complete100bp centers, no reset/subsample")
    features = disjoint_bin_features_v1(markers,bin_edges_bp)
    flags, _ = build_clade_flags_v1(bank)
    holds = markers.readout_hold.clone()
    pairs = torch.stack((holds[:-1],holds[1:]),-1)
    unique,inverse = torch.unique_consecutive(pairs,return_inverse=True,dim=0)
    result = MarkerGridResetBundleV1(original,flags,features,holds,
        (centers[1:]-centers[:-1]).to(torch.float32),unique[:,0].contiguous(),
        unique[:,1].contiguous(),inverse.contiguous(),())
    return type(result)(original,flags,features,holds,result.delta_bp,result.pair_left,
                        result.pair_right,result.edge_pair,_stamps(result.tensors()))


def reset_log_transition_v1(rate, delta_bp, log_pi):
    """Exact rank-one reset generator, FP32 stable even for tiny/large gaps.

    Q=lambda*(1*pi^T-I), P=exp(-lambda*delta)I+(1-exp(-lambda*delta))1*pi^T.
    No matrix exponential, approximate ODE solve or probability floor.
    """
    _require(rate.dtype == delta_bp.dtype == log_pi.dtype == torch.float32
             and rate.ndim == 2 and delta_bp.shape == (rate.shape[0],)
             and log_pi.shape == (4,) and rate.device == delta_bp.device == log_pi.device,
             "FP32 edge/query rates, physical gaps and four-class log simplex")
    log_stay = -rate*delta_bp[:,None]
    log_change = (-torch.expm1(log_stay)).log()
    off = log_change[...,None,None]+log_pi
    diagonal = torch.logaddexp(log_stay[...,None],log_change[...,None]+log_pi)
    eye = torch.eye(4,dtype=torch.bool,device=rate.device)
    return torch.where(eye,diagonal[...,None,:],off.expand(*rate.shape,4,4))


def reset_chain_log_marginals_v1(emission, rate, delta_bp, log_pi):
    """Existing H039 balanced exact-carry scan, with an explicit initial pi.

    Every query chunk spans the entire chromosome grid. No position chunk resets
    or detached carry. Balanced composition has O(P*B*4^3) arithmetic and O(logP)
    tensor stages. Checkpoint the entire query block to bound saved scan tables.
    """
    initial = torch.cat((emission[:1]+log_pi,emission[1:]),0)
    return chain_log_marginals_v1(initial,reset_log_transition_v1(rate,delta_bp,log_pi))


class MarkerGridResetChainGNNV1(nn.Module):
    """Complete H042 PyG body, common bin emissions, exactly31,933 weights/arm."""
    backbone_state_v1 = MarkerResolutionReadoutV1.backbone_state_v1

    def __init__(self,*,arm=ARMS[0],query_block=32,checkpoint_chain=True):
        super().__init__()
        _require(arm in ARMS and type(query_block) is int and 1 <= query_block <= 686,
                 "Registered prospective arm and query chunk")
        self.arm,self.query_block,self.checkpoint_chain = arm,query_block,bool(checkpoint_chain)
        self.backbone = AnonymousCladeFlagControlGNNV1(**MODEL)
        self.marker_head = nn.Sequential(nn.Linear(5,8),nn.GELU(),nn.Linear(8,4))
        self.rate_head = nn.Sequential(nn.Linear(64,16),nn.Tanh(),nn.Linear(16,1))
        theta = math.log((1./50000.-1e-6)/(1e-3-1./50000.))
        self.rate_theta = nn.Parameter(torch.tensor(theta,dtype=torch.float32))
        self.pi_logits = nn.Parameter(torch.zeros(3,dtype=torch.float32))
        for layer in (self.marker_head[-1],self.rate_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.float()
        _require(sum(p.numel() for p in self.parameters()) == TOTAL_PARAMETERS,
                 "Exact30788+84+1057+4 paired capacity")

    @property
    def base_rate(self):
        return 1e-6+(1e-3-1e-6)*self.rate_theta.sigmoid()

    @property
    def log_pi(self):
        return torch.cat((self.pi_logits,self.pi_logits.new_zeros(1))).log_softmax(-1)

    def decode_block_v1(self,base,context,features_cpu,grid_hold,delta_bp,pair_left,pair_right,edge_pair):
        _require(base.dtype == context.dtype == torch.float32 and base.ndim == context.ndim == 3
                 and base.shape[:2] == context.shape[:2] and base.shape[-1] == 4
                 and context.shape[-1] == 32, "FP32 complete-hold tree states and logits")
        feature = features_cpu.to(device=base.device,dtype=torch.float32)
        local = base.index_select(0,grid_hold)
        marker = self.marker_head(feature)*(feature[...,4:5] > 0).to(torch.float32)
        left,right = context.index_select(0,pair_left),context.index_select(0,pair_right)
        joined = torch.cat((.5*(left+right),(right-left).abs()),-1)
        pair_g = 2.*self.rate_head(joined).tanh().squeeze(-1)
        g = pair_g.index_select(0,edge_pair)
        if self.arm == ARMS[0]:
            rate,emission = self.base_rate*g.exp(),local+marker
        else:
            if local.shape[0] == 1:
                mean_g = local.new_zeros((*local.shape[:2],1))
            else:
                mean_g = torch.cat((g[:1],.5*(g[:-1]+g[1:]),g[-1:]),0)[...,None]
            rate = self.base_rate.expand_as(g)
            emission = mean_g.exp()*local+marker
        return reset_chain_log_marginals_v1(emission,rate,delta_bp,self.log_pi)

    def forward(self,bundle):
        _require(isinstance(bundle,MarkerGridResetBundleV1), "Bound prepared marker-grid observations")
        bank,markers = bundle.validate()
        state = self.backbone_state_v1(bank,bundle.flags) # exact inherited H04296-channel state
        base = self.backbone.classifier(state).index_select(1,bank.query_column)
        context = state[...,:32].index_select(1,bank.query_column)
        blocks = []
        for first in range(0,base.shape[1],self.query_block):
            last = min(first+self.query_block,base.shape[1])
            args = (base[:,first:last],context[:,first:last],bundle.features_cpu[:,first:last],
                    bundle.grid_hold,bundle.delta_bp,bundle.pair_left,bundle.pair_right,bundle.edge_pair)
            if self.checkpoint_chain and self.training and torch.is_grad_enabled():
                value = checkpoint(self.decode_block_v1,*args,use_reentrant=False)
            else:
                value = self.decode_block_v1(*args)
            blocks.append(value)
        return torch.cat(blocks,1) # normalized log marginals on all original[P,686,4] bins


def load_h042_grid_reset_warmstart_v1(model,payload,*,checkpoint_ref,completion_ref):
    """Source-only loader API. Root supplies already authenticated CPU payload."""
    _require(isinstance(model,MarkerGridResetChainGNNV1) and isinstance(payload,Mapping)
             and checkpoint_ref == H042_PASS30_REF and completion_ref == H042_COMPLETION_REF
             and payload.get("schema") == "anonymous_clade_flag_checkpoint_v1"
             and payload.get("hypothesis_id") == "H042" and payload.get("arm") == "exact_clade_flags"
             and payload.get("control_id") == "H040_exact_clade_flag_control_v1"
             and payload.get("pass_index") == 30 and payload.get("additional_optimizer_steps") == 960
             and payload.get("parameter_count") == 30788 and payload.get("seed") == 101
             and payload.get("teacher_used") is False and payload.get("model_kwargs") == MODEL,
             "Exact original completed H042 pass30, no optimizer resume")
    state,expected = payload.get("model_state"),model.backbone.state_dict()
    _require(isinstance(state,Mapping) and set(state) == set(expected), "Complete exact backbone state")
    for key,value in state.items():
        _require(isinstance(value,torch.Tensor) and value.device.type == "cpu"
                 and value.dtype == torch.float32 and value.shape == expected[key].shape
                 and bool(torch.isfinite(value).all()), "Original finite FP32 state: "+key)
    _require(all(p.requires_grad for p in model.parameters())
             and all(not bool(t.count_nonzero()) for layer in
                     (model.marker_head[-1],model.rate_head[-1])
                     for t in (layer.weight,layer.bias)),
             "All parameters trainable and both new output heads initially zero")
    model.backbone.load_state_dict(state,strict=True)
    return dict(base_parameters=30788,added_parameters=1145,total_parameters=TOTAL_PARAMETERS,
                all_backbone_parameters_trainable=all(p.requires_grad for p in model.parameters()),
                initial_pair_identical=True,original_H042_emissions_retained=True,
                initial_chain_is_new_in_both_arms=True,optimizer_moments_must_reset=True)


def make_matched_grid_reset_family_v1(payload,*,checkpoint_ref,completion_ref):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        transition = MarkerGridResetChainGNNV1()
        receipt = load_h042_grid_reset_warmstart_v1(transition,payload,
            checkpoint_ref=checkpoint_ref,completion_ref=completion_ref)
        confidence = copy.deepcopy(transition)
        confidence.arm = ARMS[1]
    models = {ARMS[0]:transition,ARMS[1]:confidence}
    optimizers = {a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8) for a,m in models.items()}
    return models,optimizers,receipt
