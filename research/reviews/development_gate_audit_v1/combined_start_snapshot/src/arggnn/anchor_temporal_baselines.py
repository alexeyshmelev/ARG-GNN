"""Independent temporal comparators on the same observed fixed-anchor sources."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .anchor_kernel import AnchorKernelBank
from .boundary_memory import BoundaryResponseGNN


def _positive(initial: float | Sequence[float], scales: int) -> Tensor:
    values=[initial]*scales if isinstance(initial,(int,float)) else list(initial)
    if len(values)!=scales or any(not math.isfinite(x) or x<=0 for x in values):
        raise ValueError('Require one finite positive source coupling per branch')
    return torch.tensor(values,dtype=torch.float32)[:,None]


def _coordinate_features(points: Tensor, cell_width: Tensor, dtype: torch.dtype) -> Tensor:
    # Positions are bp/1e6 globally. Do not reset the coordinate scale per tile.
    return torch.stack((points.double(),torch.ones_like(points.double())*cell_width.double()),-1).to(dtype)


def interpolate_anchor_values(anchors: Tensor, values: Tensor, positions: Tensor) -> Tensor:
    """Physical linear interpolation with constant endpoint extrapolation.

    Readouts are evaluated independently and never inserted into a recurrence.
    Coordinates stay in float64 until interpolation fractions are formed.
    """
    if anchors.ndim!=1 or positions.ndim!=1 or values.shape[0]!=anchors.numel() or anchors.numel()<1:
        raise ValueError('Require aligned fixed anchors and arbitrary coordinate vector')
    if not bool(torch.isfinite(anchors).all()) or not bool(torch.isfinite(positions).all()) or (len(anchors)>1 and not bool((anchors[1:]>anchors[:-1]).all())):
        raise ValueError('Require strictly increasing finite physical anchors')
    if len(anchors)==1: return values[0].expand(len(positions),*values.shape[1:])
    x,y=positions.double(),anchors.double()
    high=torch.searchsorted(y,x,right=True).clamp(min=1,max=len(y)-1)
    low=high-1
    fraction=((x-y[low])/(y[high]-y[low])).clamp(0,1).to(values.dtype)
    return values[low]+fraction.reshape(-1,*([1]*(values.ndim-1)))*(values[high]-values[low])


@dataclass(frozen=True)
class TemporalLayerSources:
    sources: Tensor                  # [A,S,Q,L], candidate's point source rule
    mass_sources: Tensor             # [A,S,Q,L], candidate's matching ones rule
    normalized: Tensor               # [A,Q,L], pointwise branch mixture / mass


@dataclass(frozen=True)
class TemporalAnchorState:
    anchors: Tensor                  # [A,Q,H], contextual anchor embeddings
    local_targets: Tensor            # [P,Q,H], independent plain graph embeddings
    references: Tensor               # [A,R,H], clamped plain observed sources
    layers: tuple[tuple[TemporalLayerSources,Tensor,Tensor], ...]
    # Each layer stores source diagnostics, encoded temporal bank, anchor message.


@dataclass(frozen=True)
class AnchorTemporalContext:
    anchors: Tensor
    targets: Tensor
    layers: tuple[TemporalLayerSources, ...]


class ReferenceConditionedTemporalLayer(nn.Module):
    """Same q/r maps, plain Cqr, rates, ones mass and residual gate as candidate."""
    def __init__(self,hidden_dim,latent_dim,num_scales,initial_coupling):
        super().__init__()
        self.query_projection=nn.Linear(hidden_dim,latent_dim)
        self.reference_projection=nn.Linear(hidden_dim,latent_dim)
        self.message_projection=nn.Linear(latent_dim,hidden_dim,bias=False)
        self.gate=nn.Linear(hidden_dim+latent_dim,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)
        self.register_buffer('initial_coupling',_positive(initial_coupling,num_scales))
        self.raw_coupling=nn.Parameter(torch.zeros(num_scales,latent_dim))
        self.mixture_logits=nn.Parameter(torch.zeros(num_scales,latent_dim))

    @property
    def rates(self): return self.initial_coupling*torch.exp(self.raw_coupling)

    @property
    def mixture(self): return self.mixture_logits.softmax(dim=0)

    def conditioned_sources(self,prepared,query,reference):
        rates=self.rates
        if not bool(torch.isfinite(rates).all()) or bool((rates<=0).any()):
            raise ValueError('Source coupling rates must be finite and positive')
        cqr=prepared.operators['plain'].cqr[prepared.layout.anchor_graph]
        sources=self.query_projection(query).tanh()[:,None]-(cqr@self.reference_projection(reference).tanh()[:,None])*rates[None,:,None,:]
        mass=torch.ones_like(sources)-cqr.sum(-1)[...,None]*rates[None,:,None,:]
        mixed=(sources*self.mixture[None,:,None,:]).sum(1)
        denominator=(mass*self.mixture[None,:,None,:]).sum(1)
        if not bool(torch.isfinite(mixed).all()) or not bool(torch.isfinite(denominator).all()) or bool((denominator<=0).any()):
            raise ValueError('Point source mass is nonpositive/nonfinite')
        return TemporalLayerSources(sources,mass,mixed/denominator)

    def update(self,local,message):
        return self.norm(local+torch.sigmoid(self.gate(torch.cat((local,message),-1)))*self.message_projection(message))


class _GRULayer(ReferenceConditionedTemporalLayer):
    def __init__(self,hidden_dim,latent_dim,num_scales,initial_coupling,dropout):
        super().__init__(hidden_dim,latent_dim,num_scales,initial_coupling)
        self.input_dropout=nn.Dropout(dropout)
        # Standard BiGRU has H state per direction, hence 2H active recurrent
        # channels. Its width/parameter difference is explicitly reported.
        self.temporal=nn.GRU(latent_dim+2,hidden_dim,batch_first=True,bidirectional=True)
        self.output_projection=nn.Linear(2*hidden_dim,latent_dim)

    def encode(self,sources,query,layout):
        coordinate=_coordinate_features(layout.anchors,layout.weights[0],sources.dtype)
        tokens=torch.cat((sources,coordinate[:,None].expand(-1,sources.shape[1],-1)),-1)
        encoded,_=self.temporal(self.input_dropout(tokens.transpose(0,1)))
        return encoded.transpose(0,1)

    def messages(self,encoded,local,points,layout):
        message=self.output_projection(encoded)
        return interpolate_anchor_values(layout.anchors,message,points)


class _TransformerLayer(ReferenceConditionedTemporalLayer):
    def __init__(self,hidden_dim,latent_dim,num_scales,initial_coupling,dropout,heads,max_attention_score_bytes):
        super().__init__(hidden_dim,latent_dim,num_scales,initial_coupling)
        self.heads,self.max_attention_score_bytes=heads,max_attention_score_bytes
        self.token_projection=nn.Linear(latent_dim+2,hidden_dim)
        self.input_dropout=nn.Dropout(dropout)
        self.temporal=nn.TransformerEncoderLayer(hidden_dim,heads,4*hidden_dim,dropout,
            activation='gelu',batch_first=True,norm_first=True)
        self.readout_query=nn.Linear(hidden_dim+2,hidden_dim)
        self.readout_attention=nn.MultiheadAttention(hidden_dim,heads,dropout=dropout,batch_first=True)
        self.output_projection=nn.Linear(hidden_dim,latent_dim)

    def encode(self,sources,query,layout):
        coordinate=_coordinate_features(layout.anchors,layout.weights[0],sources.dtype)
        count=sources.shape[0]*sources.shape[1]
        estimated_score_bytes=self.heads*count*count*sources.element_size()
        if estimated_score_bytes>self.max_attention_score_bytes:
            raise ValueError(f'Joint anchor attention dense-score estimate {estimated_score_bytes} bytes exceeds declared bound {self.max_attention_score_bytes}')
        tokens=self.token_projection(torch.cat((sources,coordinate[:,None].expand(-1,sources.shape[1],-1)),-1))
        # Joint anchor/query self-attention provides temporal cross-query
        # exchange. There are no learned sample-ID or numeric-ID embeddings.
        encoded=self.temporal(self.input_dropout(tokens.flatten(0,1)[None]))
        return encoded[0].reshape(tokens.shape)

    def messages(self,encoded,local,points,layout):
        coordinate=_coordinate_features(points,layout.weights[0],local.dtype)
        query=self.readout_query(torch.cat((local,coordinate[:,None].expand(-1,local.shape[1],-1)),-1))
        # Q independent decoder batches; each coordinate is an independent
        # attention query into that sample's globally contextual anchor bank.
        value,_=self.readout_attention(query.transpose(0,1),encoded.transpose(0,1),
            encoded.transpose(0,1),need_weights=False)
        return self.output_projection(value.transpose(0,1))


class _AnchorTemporalModel(nn.Module):
    family='abstract'
    def __init__(self,input_dim=47,num_classes=4,hidden_dim=32,num_layers=2,latent_dim=8,
        temporal_depth=2,num_scales=3,num_anchors=32,initial_coupling=(5.,20.,80.),
        dropout=0.,transformer_heads=4,max_attention_score_bytes=268435456):
        super().__init__()
        for value in (input_dim,num_classes,hidden_dim,num_layers,latent_dim,temporal_depth,num_scales,num_anchors,transformer_heads,max_attention_score_bytes):
            if type(value) is not int or value<1: raise ValueError('Require positive integer model dimensions')
        if not math.isfinite(dropout) or not 0<=dropout<1: raise ValueError('dropout must be in [0,1)')
        if self.family=='transformer' and hidden_dim%transformer_heads:
            raise ValueError('Transformer width must divide by its attention head count')
        self.input_dim,self.num_classes,self.hidden_dim=input_dim,num_classes,hidden_dim
        self.num_layers,self.latent_dim,self.temporal_depth=num_layers,latent_dim,temporal_depth
        self.num_scales,self.num_anchors,self.dropout=num_scales,num_anchors,float(dropout)
        self.transformer_heads=transformer_heads
        self.max_attention_score_bytes=max_attention_score_bytes
        self.encoder=BoundaryResponseGNN(input_dim,hidden_dim,num_layers)
        constructor=_GRULayer if self.family=='gru' else _TransformerLayer
        kwargs={} if self.family=='gru' else {'heads':transformer_heads,'max_attention_score_bytes':max_attention_score_bytes}
        self.layers=nn.ModuleList(constructor(hidden_dim,latent_dim,num_scales,initial_coupling,dropout,**kwargs) for _ in range(temporal_depth))
        self.head=nn.Linear(hidden_dim,num_classes)

    def rate_and_mixture_parameters(self):
        for layer in self.layers: yield from (layer.raw_coupling,layer.mixture_logits)

    def encode_anchors(self,prepared: AnchorKernelBank) -> TemporalAnchorState:
        bank,layout=prepared.bank,prepared.layout
        if prepared.num_scales!=self.num_scales or len(layout.anchors)!=self.num_anchors or 'plain' not in prepared.operators:
            raise ValueError('Prepared branch/anchor schema differs from temporal model')
        parameter=self.encoder.input_linear.weight
        if bank.query_cache is None or bank.leaf_x.shape[-1]!=self.input_dim or bank.leaf_x.device!=parameter.device or bank.leaf_x.dtype!=parameter.dtype:
            raise ValueError('Shared features must match input dimension/device/dtype')
        if not bool(torch.isfinite(bank.leaf_x).all()): raise ValueError('Observed features must be finite')
        h=self.encoder(bank.leaf_x,bank.query_cache.normalized_response)
        q,r=h[:,bank.query_cache.query_indices],h[:,bank.query_cache.reference_indices]
        query=q[layout.anchor_graph]
        reference=r[layout.anchor_graph]
        diagnostics=[]
        # All source/encoder anchor work finishes before ANY target decoding.
        # Adding targets cannot change even training-mode anchor dropout draws.
        for layer in self.layers:
            sources=layer.conditioned_sources(prepared,query,reference)
            encoded=layer.encode(sources.normalized,query,layout)
            message=layer.messages(encoded,query,layout.anchors,layout)
            query=layer.update(query,message)
            diagnostics.append((sources,encoded,message))
        return TemporalAnchorState(query,q[bank.target_graph],reference,tuple(diagnostics))

    def encode_context(self,prepared):
        state=self.encode_anchors(prepared)
        targets=state.local_targets
        for layer,(_,encoded,_) in zip(self.layers,state.layers):
            message=layer.messages(encoded,targets,prepared.bank.positions,prepared.layout)
            targets=layer.update(targets,message)
        return AnchorTemporalContext(state.anchors,targets,tuple(item[0] for item in state.layers))

    def forward(self,prepared): return self.head(self.encode_context(prepared).targets)

    def configuration_manifest(self,prepared=None):
        result={'kind':'same_source_fixed_anchor_temporal_'+self.family,
            'input_dim':self.input_dim,'num_classes':self.num_classes,'hidden_dim':self.hidden_dim,
            'source_GNN_depth':self.num_layers,'latent_dim':self.latent_dim,'temporal_depth':self.temporal_depth,
            'num_scales':self.num_scales,'num_anchors':self.num_anchors,'dropout':self.dropout,
            'transformer_heads':self.transformer_heads if self.family=='transformer' else None,
            'coordinate_units':'bp/1e6, original global coordinates; cell width in same units',
            'source_encoder':'identical independent plain BoundaryResponseGNN on same observed47 fields',
            'source_conditioning':'candidate tanh q/r, plain -Cqr reference term, positive global coupling and matching ones mass; pointwise branch mixture before temporal encoder',
            'source_rate_parameterization':'init*exp(raw), zero rate/mixture weight decay',
            'temporal_exchange':'bidirectional querywise GRU' if self.family=='gru' else 'joint anchor/query self-attention, then querywise cross-attention readout',
            'readout_rule':'physical linear interpolation, constant endpoint extrapolation' if self.family=='gru' else 'independent coordinate cross-attention into globally contextual same-query anchor keys/values',
            'references':'clamped independently encoded observed source embeddings',
            'causality':'noncausal known observed ARG context permitted',
            'numeric_ID_features':False,'labels_in_input':False,'online_eigensolve':False,
            'parameters':sum(p.numel() for p in self.parameters()),
            'encoder_parameters':sum(p.numel() for p in self.encoder.parameters()),
            'context_parameters':sum(p.numel() for p in self.layers.parameters()),
            'head_parameters':sum(p.numel() for p in self.head.parameters()),
            'active_recurrent_channels_per_query':2*self.hidden_dim if self.family=='gru' else None,
            'attention_token_count':'A*Q joint tokens' if self.family=='transformer' else None,
            'max_attention_dense_score_bytes_per_layer':self.max_attention_score_bytes if self.family=='transformer' else None,
            'attention_dense_score_estimate_rule':'heads*(A*Q)^2*dtype_bytes; guard before encode, excludes activations/backward/workspaces; efficient attention can use less' if self.family=='transformer' else None,
            'current_source_rates':[layer.rates.detach().tolist() for layer in self.layers],
            'current_source_mixtures':[layer.mixture.detach().tolist() for layer in self.layers],
            'limitations':'different temporal capacity/decoder from kernel; no PSD-kernel, anchor-refinement, prediction or novelty claim'}
        if prepared is not None: result.update(feature_columns=list(prepared.bank.feature_columns))
        result['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['configuration_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return result


class AnchorGRUModel(_AnchorTemporalModel):
    family='gru'


class AnchorTemporalTransformerModel(_AnchorTemporalModel):
    family='transformer'


class ObservedReferenceCoordinateSmoother(nn.Module):
    """No-fit observed role evidence from -Cqr, smoothed over the same anchors."""
    def __init__(self,num_classes=4,num_anchors=32,bandwidth_bp=30000.,pseudocount=.001,
                 reference_columns=(0,1,2,3)):
        super().__init__()
        if type(num_classes) is not int or num_classes<1 or type(num_anchors) is not int or num_anchors<1 or len(reference_columns)!=num_classes or len(set(reference_columns))!=num_classes:
            raise ValueError('Declare positive class/anchor count and distinct observed role columns')
        if not math.isfinite(bandwidth_bp) or bandwidth_bp<=0 or not math.isfinite(pseudocount) or pseudocount<=0:
            raise ValueError('Require positive finite bandwidth and pseudocount')
        self.num_classes,self.num_anchors=num_classes,num_anchors
        self.bandwidth_bp,self.pseudocount=float(bandwidth_bp),float(pseudocount)
        self.reference_columns=tuple(reference_columns)

    def anchor_evidence(self,prepared):
        bank,layout=prepared.bank,prepared.layout
        if len(layout.anchors)!=self.num_anchors: raise ValueError('Anchor count mismatch')
        cache=bank.query_cache
        if cache is None or cache.reference_ids.numel()==0: raise ValueError('Observed reference smoother requires a reference panel')
        lookup={column:index for index,column in enumerate(bank.feature_columns)}
        if any(column not in lookup for column in self.reference_columns):
            raise ValueError('Observed reference role columns absent from selected leaf features')
        roles=bank.leaf_x[:,cache.reference_indices][:,:,[lookup[column] for column in self.reference_columns]]
        if not bool(((roles==0)|(roles==1)).all()) or not bool((roles.sum(-1)==1).all()):
            raise ValueError('Use observed one-hot reference roles, without query targets')
        conductance=-cache.cqr[layout.anchor_graph]
        tolerance=64*torch.finfo(conductance.dtype).eps*conductance.abs().amax()
        if bool((conductance < -tolerance).any()): raise ValueError('Plain reference conductance must be nonnegative')
        evidence=conductance@roles[layout.anchor_graph]
        if not bool(torch.isfinite(evidence).all()) or bool((evidence<0).any()):
            raise ValueError('Observed evidence must be finite and nonnegative')
        return evidence

    def forward(self,prepared):
        evidence=self.anchor_evidence(prepared)
        distance=(prepared.bank.positions.double()[:,None]-prepared.layout.anchors.double()[None]).abs()
        log_mass=-distance/(self.bandwidth_bp/1e6)+prepared.layout.weights.double().log()[None]
        weight=log_mass.softmax(-1).to(evidence.dtype)
        score=torch.einsum('pa,aqk->pqk',weight,evidence)+self.pseudocount
        return score.log()-score.sum(-1,keepdim=True).log()

    def rate_and_mixture_parameters(self): return iter(())

    def configuration_manifest(self,prepared=None):
        return {'kind':'observed_reference_fixed_anchor_coordinate_smoother','parameters':0,
            'source_encoder':'none; uses observed reference-role onehots and plain -Cqr affinities',
            'common_input_available':'same47 observed fields and same plain boundary bank; only role/geometry subset used',
            'num_anchors':self.num_anchors,'bandwidth_bp':self.bandwidth_bp,'pseudocount':self.pseudocount,
            'coordinate_units':'global bp/1e6','coordinate_kernel':'exp(-abs(x-y)/(bandwidth_bp/1e6)) with anchor-cell mass; stable row normalization',
            'fitting':'none; bandwidth/pseudocount selected on development validation grid only',
            'labels_in_input':False,'numeric_ID_features':False,
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
