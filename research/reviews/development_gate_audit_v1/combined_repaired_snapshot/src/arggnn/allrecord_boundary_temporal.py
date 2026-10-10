"""Observed boundary47 controls with temporal access to every source record.

These standard all-record temporal models are not subdivision invariant.
Coordinates decode an already encoded source bank and never create tokens.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor,nn
import torch.nn.functional as F

from .boundary_memory import BoundaryResponseGNN
from .kron_interface import LeafInterfaceBank

FEATURE_COLUMNS=tuple(i for i in range(48) if i!=7)
INPUT_SCOPE='every observed hold in the sealed boundary47 bank; plain normalized full-boundary response, query/reference roles, no target-created source anchors'
AGGREGATION={
    'allrecord_gru':'two-layer bidirectional querywise GRU over every ordered observed hold',
    'allrecord_transformer':'two-layer querywise temporal self-attention over every ordered observed hold',
}


def interval_features(intervals: Tensor,dtype: torch.dtype) -> Tensor:
    """Use original global bp/1e6 coordinates; form differences before casting."""
    left,right=intervals.double().unbind(-1)
    return torch.stack((left,right,(left+right)/2,right-left),-1).to(dtype)


def interpolate_record_states(midpoints: Tensor,values: Tensor,positions: Tensor) -> Tensor:
    """Decode each coordinate independently, with constant endpoint extrapolation."""
    if midpoints.ndim!=1 or positions.ndim!=1 or len(midpoints)<1 or values.shape[0]!=len(midpoints):
        raise ValueError('Aligned observed record states and coordinate vector required')
    if not bool(torch.isfinite(positions).all()) or not bool(torch.isfinite(midpoints).all()) or (len(midpoints)>1 and not bool((midpoints[1:]>midpoints[:-1]).all())):
        raise ValueError('Finite strictly ordered record midpoint coordinates required')
    if len(midpoints)==1:return values[0].expand(len(positions),*values.shape[1:])
    knots=midpoints.double();points=positions.double()
    high=torch.searchsorted(knots,points,right=True).clamp(1,len(knots)-1);low=high-1
    fraction=((points-knots[low])/(knots[high]-knots[low])).clamp(0,1).to(values.dtype)
    return values[low]+fraction.reshape(-1,*([1]*(values.ndim-1)))*(values[high]-values[low])


@dataclass(frozen=True)
class AllRecordContext:
    local_query: Tensor               # [G,Q,H], original graph serialization
    reference_context: Tensor         # [G,Q,H], observed plain -Cqr weighted R
    reference_role_evidence: Tensor   # [G,Q,K], observed reference roles only
    order: Tensor                     # [G], genomic order to stored graph index
    tokens: Tensor                    # [G,Q,H], every record in genomic order
    memory: Tensor                    # [G,Q,2H] GRU or [G,Q,H] Transformer
    midpoints: Tensor                 # [G], original float64 global coordinates


class _AllRecordBoundaryTemporal(nn.Module):
    family='abstract'
    def __init__(self,input_dim=47,num_classes=4,hidden_dim=32,source_depth=2,
                 temporal_depth=2,dropout=0.,transformer_heads=4,
                 max_attention_score_bytes=268435456):
        super().__init__()
        for value in (input_dim,num_classes,hidden_dim,source_depth,temporal_depth,transformer_heads,max_attention_score_bytes):
            if type(value) is not int or value<1:raise ValueError('Positive integer dimensions required')
        if input_dim!=47 or num_classes!=4 or source_depth!=2 or temporal_depth!=2:
            raise ValueError('Prescribed boundary47/four-class source2/temporal2 schema required')
        if isinstance(dropout,bool) or not isinstance(dropout,(int,float)) or not math.isfinite(dropout) or not 0<=dropout<1:
            raise ValueError('Finite dropout in [0,1) required')
        if self.family=='allrecord_transformer' and hidden_dim%transformer_heads:
            raise ValueError('Transformer hidden width must divide its head count')
        self.input_dim,self.num_classes,self.hidden_dim=input_dim,num_classes,hidden_dim
        self.source_depth,self.temporal_depth=source_depth,temporal_depth
        self.dropout,self.transformer_heads=float(dropout),transformer_heads
        self.max_attention_score_bytes=max_attention_score_bytes
        self.encoder=BoundaryResponseGNN(input_dim,hidden_dim,source_depth)
        # Graph/query embeddings, weighted reference embedding, observed roles,
        # and physical intervals are all observed; query targets are absent.
        self.token=nn.Sequential(nn.Linear(3*hidden_dim+num_classes+4,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim))
        self.input_dropout=nn.Dropout(dropout)
        if self.family=='allrecord_gru':
            self.temporal=nn.GRU(hidden_dim,hidden_dim,num_layers=2,dropout=dropout,
                                 batch_first=True,bidirectional=True)
            self.head=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,num_classes))
        elif self.family=='allrecord_transformer':
            layer=nn.TransformerEncoderLayer(hidden_dim,transformer_heads,4*hidden_dim,
                dropout,activation='gelu',batch_first=True,norm_first=True)
            self.temporal=nn.TransformerEncoder(layer,2,norm=nn.LayerNorm(hidden_dim),enable_nested_tensor=False)
            self.coordinate=nn.Sequential(nn.Linear(3,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
            self.readout_query=nn.Linear(2*hidden_dim,hidden_dim)
            # Dropout belongs to source encoding. The decoder has no stochastic
            # mask whose shape depends on the number of requested coordinates.
            self.readout_attention=nn.MultiheadAttention(hidden_dim,transformer_heads,dropout=0.,batch_first=True)
            self.readout_norm=nn.LayerNorm(hidden_dim)
            self.head=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,num_classes))
        else:raise ValueError('Concrete all-record temporal family required')

    def _validate_source(self,bank):
        parameter=self.encoder.input_linear.weight;cache=bank.query_cache
        if cache is None or tuple(bank.feature_columns)!=FEATURE_COLUMNS or bank.leaf_x.ndim!=3 or bank.leaf_x.shape[-1]!=47:
            raise ValueError('Sealed branch-excluded boundary47 features and query/reference cache required')
        g,b,_=bank.leaf_x.shape
        if g<1 or cache.reference_indices.numel()<1 or cache.query_indices.numel()<1:
            raise ValueError('Nonempty observed records and query/reference panel required')
        tensors=(bank.leaf_x,cache.normalized_response,cache.cqr)
        if any(t.device!=parameter.device or t.dtype!=parameter.dtype or not bool(torch.isfinite(t).all()) for t in tensors):
            raise ValueError('Finite source features/Cplain must match model device/dtype')
        if cache.normalized_response.shape!=(g,b,b) or cache.cqr.shape!=(g,len(cache.query_indices),len(cache.reference_indices)):
            raise ValueError('Plain response/query/reference shape mismatch')
        if not torch.equal(cache.query_ids,bank.query_node_ids) or not torch.equal(bank.sample_ids[cache.query_indices],cache.query_ids) or not torch.equal(bank.sample_ids[cache.reference_indices],cache.reference_ids):
            raise ValueError('Stable query/reference correspondence differs from observed bank')
        intervals=bank.intervals
        if intervals.device!=parameter.device or intervals.shape!=(g,2) or not intervals.is_floating_point() or not bool(torch.isfinite(intervals).all()):
            raise ValueError('Observed physical interval coordinates required')
        order=torch.argsort(intervals[:,0],stable=True);ordered=intervals[order].double()
        if not bool((ordered[:,1]>ordered[:,0]).all()) or (g>1 and not bool((ordered[1:,0]>=ordered[:-1,1]).all())):
            raise ValueError('Positive nonoverlapping observed source intervals required')
        return order

    def encode_records(self,bank: LeafInterfaceBank) -> AllRecordContext:
        """Encode every source hold without reading positions or target_graph."""
        order=self._validate_source(bank);cache=bank.query_cache
        h=self.encoder(bank.leaf_x,cache.normalized_response)
        query=h[:,cache.query_indices];reference=h[:,cache.reference_indices]
        lookup={column:i for i,column in enumerate(bank.feature_columns)}
        roles=bank.leaf_x[:,cache.reference_indices][:,:,[lookup[i] for i in range(4)]]
        if not bool(((roles==0)|(roles==1)).all()) or not bool((roles.sum(-1)==1).all()):
            raise ValueError('Observed reference roles must be one-hot four-class inputs')
        conductance=-cache.cqr
        tolerance=64*torch.finfo(conductance.dtype).eps*conductance.abs().amax()
        if bool((conductance < -tolerance).any()):raise ValueError('Plain reference conductance must be nonnegative')
        conductance=torch.where(conductance<0,torch.zeros_like(conductance),conductance)
        mass=conductance.sum(-1,keepdim=True)
        denominator=torch.where(mass>0,mass,torch.ones_like(mass))
        ref_context=(conductance@reference)/denominator
        role_evidence=(conductance@roles)/denominator
        pooled=h.mean(1)[:,None].expand_as(query)
        physical=interval_features(bank.intervals,query.dtype)[:,None].expand(-1,len(cache.query_indices),-1)
        tokens=self.token(torch.cat((query,pooled,ref_context,role_evidence,physical),-1))[order]
        if self.family=='allrecord_transformer':
            estimate=self.transformer_heads*len(cache.query_indices)*len(order)**2*tokens.element_size()
            if estimate>self.max_attention_score_bytes:
                raise ValueError(f'Querywise all-record attention dense-score estimate {estimate} exceeds declared bound {self.max_attention_score_bytes}')
        temporal_input=self.input_dropout(tokens.transpose(0,1))
        encoded=self.temporal(temporal_input)
        memory=encoded[0] if self.family=='allrecord_gru' else encoded
        intervals=bank.intervals[order].double()
        return AllRecordContext(query,ref_context,role_evidence,order,tokens,memory.transpose(0,1).contiguous(),intervals.mean(-1))

    def decode(self,bank: LeafInterfaceBank,context: AllRecordContext) -> Tensor:
        points=bank.positions;mapping=bank.target_graph
        if points.ndim!=1 or not points.is_floating_point() or not bool(torch.isfinite(points).all()) or points.device!=context.local_query.device or mapping.dtype!=torch.long or mapping.ndim!=1 or len(mapping)!=len(points) or mapping.device!=points.device or bool(((mapping<0)|(mapping>=len(bank.intervals))).any()):
            raise ValueError('Finite fixed-coordinate readout with valid observed graph mapping required')
        if len(points) and not bool(((points>=bank.intervals[mapping,0])&(points<bank.intervals[mapping,1])).all()):
            raise ValueError('Readout coordinates must lie in their declared observed hold')
        local=context.local_query[mapping]
        if self.family=='allrecord_gru':
            memory=interpolate_record_states(context.midpoints,context.memory,points)
            return self.head(torch.cat((local,memory),-1))
        p=points.double();coordinates=torch.stack((p,torch.sin(2*math.pi*p),torch.cos(2*math.pi*p)),-1).to(local.dtype)
        position=self.coordinate(coordinates)[:,None].expand_as(local)
        query=self.readout_query(torch.cat((local,position),-1)).transpose(0,1)
        memory=context.memory.transpose(0,1)
        value,_=self.readout_attention(query,memory,memory,need_weights=False)
        value=self.readout_norm(query+value).transpose(0,1)
        return self.head(torch.cat((local,value),-1))

    def forward(self,bank):return self.decode(bank,self.encode_records(bank))

    def configuration_manifest(self,bank=None):
        result={'kind':self.family+'_boundary47_control','input_dim':47,'num_classes':4,
            'hidden_dim':self.hidden_dim,'source_GNN_depth':2,'temporal_depth':2,'dropout':self.dropout,
            'input_scope':INPUT_SCOPE,'cross_tree_aggregation':AGGREGATION[self.family],
            'source_encoder':'identical graph-local plain BoundaryResponseGNN, branch-excluded observed47 fields',
            'source_token':'query embedding, all-boundary mean, plain -Cqr normalized reference embedding/observed role evidence, physical interval features',
            'reference_conditioning':'row-normalized nonnegative plain -Cqr; disconnected rows produce zero reference context',
            'coordinate_units':'original global bp/1e6; left/right/midpoint/span formed in float64 before feature cast',
            'readout_rule':'independent physical midpoint interpolation with constant endpoint extrapolation' if self.family=='allrecord_gru' else 'independent coordinate cross-attention into same-query all-record keys/values',
            'causality':'noncausal observed context allowed','readout_created_source_tokens':False,
            'record_subdivision_invariant':False,'redundancy_preprocessing':'none; every sealed record remains a token',
            'numeric_ID_features':False,'labels_in_input':False,'online_eigensolve':False,
            'transformer_heads':self.transformer_heads if self.family=='allrecord_transformer' else None,
            'max_attention_score_bytes':self.max_attention_score_bytes if self.family=='allrecord_transformer' else None,
            'attention_estimate':'heads*queries*records^2*dtype_bytes; forward score estimate only, excludes backward/workspaces' if self.family=='allrecord_transformer' else None,
            'active_recurrent_channels_per_query':2*self.hidden_dim if self.family=='allrecord_gru' else None,
            'parameters':sum(p.numel() for p in self.parameters()),
            'encoder_parameters':sum(p.numel() for p in self.encoder.parameters()),
            'interpretation':'independent all-record standard temporal controls; no native reproduction, independent confirmation, superiority or novelty claim',
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if bank is not None:result['observed_records']=len(bank.intervals)
        result['configuration_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return result


class AllRecordBoundaryGRU(_AllRecordBoundaryTemporal):
    family='allrecord_gru'


class AllRecordBoundaryTransformer(_AllRecordBoundaryTemporal):
    family='allrecord_transformer'
