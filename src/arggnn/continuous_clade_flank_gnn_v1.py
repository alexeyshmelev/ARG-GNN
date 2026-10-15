"""H046 continuous source-native clade-flank context. Source-only Torch/PyG API.

The exact invariance contract fixes the learned field, original focal supports
and episode membership. Numerical integration cells may refine; original bank
holds, SNP pooling and the genomic BiGRU are not refinement invariant.
No reader, target input, launcher, admission or execution authority is provided.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping
import copy
import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .anonymous_clade_flag_control_v1 import (
    AnonymousCladeFlagControlGNNV1, AnonymousCladeFlagsV1,
    _cpu_index_array, _tree_clades,
)
from .anonymous_exact_clade_transport_v1 import _forest_index
from .marker_resolution_readout_v1 import (
    H044BoundMarkerBankV1, MarkerEvidenceBankV1, tensor_identity_v1,
    H042_PASS30_REF, H042_COMPLETION_REF, original_grid_span_nll_v1,
)
from .mutation_routed_whole_tree_gnn_v1 import _FLOAT_FIELDS, _LEVEL_FIELDS

SOURCE = "src/arggnn/continuous_clade_flank_gnn_v1.py"
SCHEMA = "continuous_source_native_clade_flanks_v1"
ARMS = ("foreign_flank", "own_state")
MODEL = dict(hidden_dim=32, expected_haplotypes=952,
             site_chunk_size=256, checkpoint_sites=True)
READOUT = dict(query_block=32, center_block=1024, checkpoint_readout=True, checkpoint_scan=True)
ARRAY_FIELDS = (
    "node_flags", "member_node", "member_hold", "member_episode", "episode_size",
    "episode_start", "episode_end", "node_member", "hold_span", "grid_hold",
    "distance_left", "distance_right", "window_index",
)
DEFAULT_PAYLOAD_CAP = 1006632960
SITE_BLOCK = 1024


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_clade_flank_arrays_v1(parent, sample_index, spans, readout_hold,
        positions_bp0, centers_bp, intervals_bp, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
    """Topology/physical coordinates only. Exact masks never escape two trees."""
    parent, samples, spans, grid, positions, centers, intervals = (
        np.asarray(x) for x in (parent, sample_index, spans, readout_hold,
                               positions_bp0, centers_bp, intervals_bp))
    _require(parent.ndim == 1 and 0 < parent.size < 2**31 and parent.dtype.kind == "i"
             and np.all((parent >= -1) & (parent < parent.size)), "Bounded signed parent forest")
    _require(samples.ndim == 2 and samples.dtype.kind == "i"
             and 0 < samples.shape[0] < 2**31 and samples.shape[1] >= 3
             and np.all((samples >= 0) & (samples < parent.size)), "Complete stable sampled leaves")
    n, (t, h), p = parent.size, samples.shape, centers.size
    _require(spans.dtype == np.float32 and spans.shape == (t,)
             and np.isfinite(spans).all() and np.all(spans > 0), "Finite FP32 native spans")
    _require(intervals.dtype == positions.dtype == centers.dtype == np.float64
             and intervals.shape == (t, 2) and positions.ndim == centers.ndim == 1
             and 0 < p <= 20000 and positions.size <= 2000000
             and np.isfinite(intervals).all() and np.isfinite(positions).all() and np.isfinite(centers).all()
             and np.all(intervals[:, 1] > intervals[:, 0])
             and np.all(intervals[1:, 0] == intervals[:-1, 1])
             and np.allclose((intervals[:, 1]-intervals[:, 0]).astype(np.float32), spans,
                             rtol=1e-6, atol=1e-5),
             "Exact continuous original physical interval/position arrays")
    _require(grid.dtype.kind == "i" and grid.shape == (p,) and np.all((grid >= 0) & (grid < t))
             and np.all(positions[1:] > positions[:-1]) and np.all(positions == np.floor(positions))
             and np.all(centers[1:] > centers[:-1])
             and np.all((positions >= intervals[0, 0]) & (positions < intervals[-1, 1]))
             and np.array_equal(np.searchsorted(intervals[:, 1], centers, side="right"), grid),
             "Original integer SNP positions and source-native grid membership")
    _require(type(max_payload_bytes) is int and 24*n+4*t+40*p <= max_payload_bytes,
             "Conservative complete payload cap 24*N+4*T+40*P required")
    head, sibling, roots, parent_steps = _forest_index(parent, samples)
    flags = np.zeros((n, 2), dtype=np.uint8)
    nodes, holds, episode = (np.empty(n, dtype=np.int32) for _ in range(3))
    sizes = np.zeros(n, dtype=np.int32)
    previous, previous_ids = None, {}
    m = e = visited = unary_total = 0
    for hold, (root, row) in enumerate(zip(roots, samples)):
        current, count, unary = _tree_clades(root, row, parent, head, sibling)
        visited += count
        unary_total += unary
        if previous is not None:
            for mask, node in previous.items():
                if mask not in current:
                    flags[node, 1] = 1
            for mask, node in current.items():
                if mask not in previous:
                    flags[node, 0] = 1
        ids = {}
        for mask, node in current.items():
            _require(m < n, "One proper-clade membership per native representative")
            key = previous_ids.get(mask)
            if key is None:
                key = e
                e += 1
            nodes[m], holds[m], episode[m] = node, hold, key
            sizes[key] += 1
            ids[mask] = key
            m += 1
        previous, previous_ids = current, ids
    _require(visited == n, "All original vertices including unaries must be retained")
    del previous, previous_ids, current, ids
    for value in (nodes, holds, episode):
        value.resize((m,), refcheck=False)
    sizes.resize((e,), refcheck=False)
    # Stable integer grouping; no episode-by-longest-run padded matrix.
    order = np.argsort(episode, kind="stable")
    nodes, holds, episode = (np.ascontiguousarray(a[order]) for a in (nodes, holds, episode))
    start = np.r_[True, episode[1:] != episode[:-1]] if m else np.empty(0, dtype=np.bool_)
    end = np.r_[episode[1:] != episode[:-1], True] if m else np.empty(0, dtype=np.bool_)
    _require(np.all(start[1:] | (holds[1:] == holds[:-1]+1)), "Consecutive native members")
    representative = np.full(n, -1, dtype=np.int32)
    eligible = sizes[episode] >= 2
    representative[nodes[eligible]] = np.flatnonzero(eligible).astype(np.int32)
    # A complete root-to-leaf traversal carries the deepest eligible clade.
    for root in roots:
        stack = [(root, -1)]
        while stack:
            node, inherited = stack.pop()
            selected = int(representative[node])
            if selected < 0:
                selected = inherited
            representative[node] = selected
            child = int(head[node])
            while child >= 0:
                stack.append((child, selected))
                child = int(sibling[child])
    left = np.maximum(centers-1000., intervals[grid, 0])
    right = np.minimum(centers+1000., intervals[grid, 1])
    windows = np.stack((np.searchsorted(positions, left, side="left"),
                        np.searchsorted(positions, centers, side="left"),
                        np.searchsorted(positions, right, side="left")), -1).astype(np.int64)
    _require(np.all(windows[:, 1:] >= windows[:, :-1])
             and np.all(np.diff(windows, axis=1) <= 1000),
             "Each half-open local side has at most1000 unique integer SNPs")
    arrays = dict(
        node_flags=flags, member_node=nodes, member_hold=holds, member_episode=episode,
        episode_size=sizes, episode_start=np.ascontiguousarray(start),
        episode_end=np.ascontiguousarray(end), node_member=representative,
        hold_span=np.ascontiguousarray(spans.copy()), grid_hold=grid.astype(np.int64, copy=True),
        distance_left=(centers-intervals[grid, 0]).astype(np.float32),
        distance_right=(intervals[grid, 1]-centers).astype(np.float32),
        window_index=np.ascontiguousarray(windows),
    )
    flag_meta = dict(schema="anonymous_clade_node_flags_v1",
        channels=["gained_from_left", "lost_to_right"], native_nodes=int(n),
        native_holds=int(t), haplotypes=int(h), node_flags_sha256=_sha(flags),
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        labels_times_targets_or_outcomes_used=False)
    meta = dict(schema=SCHEMA, native_nodes=int(n), native_holds=int(t), haplotypes=int(h),
        grid_centers=int(p), sites=int(positions.size), member_count=int(m), episode_count=int(e),
        eligible_members=int(eligible.sum()), unary_duplicate_count=int(unary_total),
        payload_bytes=int(sum(v.nbytes for v in arrays.values())),
        payload_upper_bound_bytes=int(24*n+4*t+40*p), max_payload_bytes=max_payload_bytes,
        topology_sha256=_sha(parent), sample_index_sha256=_sha(samples), spans_sha256=_sha(spans),
        positions_sha256=_sha(positions), centers_sha256=_sha(centers), intervals_sha256=_sha(intervals),
        array_sha256={k:_sha(v) for k,v in arrays.items()}, flag_meta=flag_meta,
        exact_bitsets_retained=False, internal_ancestor_correspondence_claimed=False,
        observation_only=True, source_native_supports_fixed=True,
        all_original_vertices_preserved=True, parent_links_followed=int(parent_steps))
    return arrays, meta


@dataclass(frozen=True)
class ContinuousCladeFlanksV1:
    flags: AnonymousCladeFlagsV1
    member_node: torch.Tensor
    member_hold: torch.Tensor
    member_episode: torch.Tensor
    episode_size: torch.Tensor
    episode_start: torch.Tensor
    episode_end: torch.Tensor
    node_member: torch.Tensor
    hold_span: torch.Tensor
    grid_hold: torch.Tensor
    distance_left: torch.Tensor
    distance_right: torch.Tensor
    window_index: torch.Tensor  # CPU only, physical observation lookup

    @classmethod
    def from_arrays(cls, arrays, meta):
        _require(set(arrays) == set(ARRAY_FIELDS) and meta.get("schema") == SCHEMA
                 and meta.get("exact_bitsets_retained") is False
                 and meta.get("observation_only") is True, "Builder-produced compact structural arrays")
        n, t, m, e, p = (meta[k] for k in
            ("native_nodes", "native_holds", "member_count", "episode_count", "grid_centers"))
        shapes = dict(node_flags=(n,2), member_node=(m,), member_hold=(m,), member_episode=(m,),
            episode_size=(e,), episode_start=(m,), episode_end=(m,), node_member=(n,),
            hold_span=(t,), grid_hold=(p,), distance_left=(p,), distance_right=(p,), window_index=(p,3))
        for name, value in arrays.items():
            dtype = (np.uint8 if name == "node_flags" else np.bool_ if name in ("episode_start","episode_end")
                     else np.float32 if name in ("hold_span","distance_left","distance_right")
                     else np.int64 if name in ("grid_hold","window_index") else np.int32)
            _require(value.shape == shapes[name] and value.dtype == dtype and value.flags.c_contiguous
                     and _sha(value) == meta["array_sha256"][name], "Exact H046 array shape/type/hash: "+name)
        _require(np.all((arrays["member_node"]>=0)&(arrays["member_node"]<n))
                 and np.unique(arrays["member_node"]).size == m
                 and np.all((arrays["member_hold"]>=0)&(arrays["member_hold"]<t))
                 and np.all((arrays["node_member"]>=-1)&(arrays["node_member"]<m))
                 and np.all(arrays["hold_span"]>0) and np.isfinite(arrays["hold_span"]).all()
                 and np.all(arrays["distance_left"]>=0) and np.all(arrays["distance_right"]>0),
                 "Bounded original native lookup and physical spans")
        _require(meta["payload_bytes"] == sum(v.nbytes for v in arrays.values())
                 <= meta["payload_upper_bound_bytes"] <= meta["max_payload_bytes"],
                 "Complete compact payload bound")
        flags = AnonymousCladeFlagsV1.from_verified_arrays(arrays["node_flags"], meta["flag_meta"])
        return cls(AnonymousCladeFlagsV1(flags.node_flags.clone(),flags.native_holds,
                    flags.haplotypes,flags.native_nodes),
                   *(torch.from_numpy(arrays[k].copy()) for k in ARRAY_FIELDS[1:]))

    def to(self, device):
        return type(self)(self.flags.to(device), *(getattr(self,k) if k=="window_index"
            else getattr(self,k).to(device) for k in ARRAY_FIELDS[1:]))

    def tensor_bytes(self):
        return self.flags.tensor_bytes()+sum(getattr(self,k).numel()*getattr(self,k).element_size()
                                             for k in ARRAY_FIELDS[1:])

    def device_bytes(self):
        return self.tensor_bytes()-self.window_index.numel()*self.window_index.element_size()


def _flank_stamps(value):
    tensors = [value.flags.node_flags]+[getattr(value,k) for k in ARRAY_FIELDS[1:]]
    return tuple((id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,str(t.device)) for t in tensors)


@dataclass(frozen=True)
class H046BoundCladeBankV1:
    marker_bundle: H044BoundMarkerBankV1
    flanks: ContinuousCladeFlanksV1
    flank_stamps: tuple

    @property
    def bank(self):
        return self.marker_bundle.bank

    @property
    def source_bank(self):
        return self.marker_bundle.source_bank

    @property
    def markers(self):
        return self.marker_bundle.markers

    def validate(self):
        bank, markers = self.marker_bundle.validate()
        _require(self.flank_stamps == _flank_stamps(self.flanks), "Immutable H046 preparation changed")
        return bank, markers, self.flanks

    def to(self, device):
        self.validate()
        flanks = self.flanks.to(device)
        return type(self)(self.marker_bundle.to(device),flanks,_flank_stamps(flanks))


def prepare_continuous_clade_bank_v1(bundle, intervals_bp, *, max_payload_bytes=DEFAULT_PAYLOAD_CAP):
    _require(isinstance(bundle,H044BoundMarkerBankV1),"Original H044-bound observation bundle")
    bank, markers = bundle.validate()
    _require(bank.parent.device.type=="cpu" and markers.original_binding is not None,
             "Original complete CPU observations")
    _require(tensor_identity_v1(intervals_bp)==markers.original_binding["array_identities"]["intervals_bp"],
             "Exact original source-native intervals from the bound sidecar")
    arrays, meta = build_clade_flank_arrays_v1(
        _cpu_index_array(bank.parent),_cpu_index_array(bank.sample_index),bank.spans.numpy(),
        _cpu_index_array(bank.readout_hold),markers.positions_bp0.numpy(),markers.centers_bp.numpy(),
        intervals_bp.numpy(),max_payload_bytes=max_payload_bytes)
    flank = ContinuousCladeFlanksV1.from_arrays(arrays,meta)
    return H046BoundCladeBankV1(bundle,flank,_flank_stamps(flank)),meta


def _affine_inclusive_scan(a, b):
    """Work-efficient flat Blelloch-style scan: O(M*K), O(log M) depth.

    An element is z -> a*z+b, ordered left-to-right. Recursive pair reduction
    halves the work at every level; reconstruction never uses an M-by-M table.
    """
    n = a.shape[0]
    if n <= 1:
        return a,b
    pairs = n//2
    ap = a[1:2*pairs:2]*a[:2*pairs:2]
    bp = b[1:2*pairs:2]+a[1:2*pairs:2,None]*b[:2*pairs:2]
    pa,pb = _affine_inclusive_scan(ap,bp)
    evens = (n+1)//2
    prev_a = torch.cat((a.new_ones(1),pa),0)[:evens]
    prev_b = torch.cat((b.new_zeros((1,b.shape[1])),pb),0)[:evens]
    ea = a[::2]*prev_a
    eb = b[::2]+a[::2,None]*prev_b
    if n%2:
        pa = torch.cat((pa,a.new_zeros(1)),0)
        pb = torch.cat((pb,b.new_zeros((1,b.shape[1]))),0)
    return (torch.stack((ea,pa),1).reshape(-1)[:n],
            torch.stack((eb,pb),1).reshape(-1,b.shape[1])[:n])


def segmented_flank_integrals_v1(projected, widths, starts, ends, tau, *, checkpoint_scan=True):
    """Two differentiable exclusive scans, normalized by tau, never by mass."""
    _require(projected.dtype==widths.dtype==tau.dtype==torch.float32
             and projected.ndim==2 and projected.shape[1]==4
             and widths.shape==starts.shape==ends.shape==(len(projected),)
             and starts.dtype==ends.dtype==torch.bool and tau.numel()==1
             and projected.device==widths.device==starts.device==ends.device==tau.device,
             "Compact same-device FP32 segmented integral inputs")
    n = len(projected)
    if not n:
        empty = projected.new_zeros((0,5))
        return empty,empty
    coefficient = torch.exp(-widths/tau)
    weight = -torch.expm1(-widths/tau)
    values = torch.cat((projected,projected.new_ones((n,1))),-1)
    def scan(a,b):
        return _affine_inclusive_scan(a,b)[1]
    def direction(a,b,boundary):
        reset = torch.where(boundary,torch.zeros_like(a),a)
        if checkpoint_scan and torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
            inclusive = checkpoint(scan,reset,b,use_reentrant=False)
        else:
            inclusive = scan(reset,b)
        exclusive = torch.cat((b.new_zeros((1,5)),inclusive[:-1]),0)
        return exclusive.masked_fill(boundary[:,None],0.)
    forward = direction(coefficient,weight[:,None]*values,starts)
    backward = direction(coefficient.flip(0),(weight[:,None]*values).flip(0),ends.flip(0)).flip(0)
    return forward,backward


class LocalReferencePrefixV1:
    """FP32 observation prefixes rebased every1024 sites.

    Every physical1000bp side contains <=1000 unique integer SNPs, so a range
    spans at most two blocks. No chromosome-wide FP32 subtraction or FP64
    evidence arithmetic is used. Tables are CPU-only and freed per query block.
    """
    def __init__(self, markers, query_start, query_end):
        _require(isinstance(markers,MarkerEvidenceBankV1)
                 and 0<=query_start<query_end<=len(markers.query_column),
                 "Original categorical marker evidence and bounded query block")
        q = markers.genotype.index_select(1,markers.query_column[query_start:query_end]).long()
        s,b = q.shape
        self.s,self.b = s,b
        if not s:
            self.prefix = torch.zeros((1,SITE_BLOCK+1,b,5),dtype=torch.float32)
            return
        active = (q>=0)&markers.reference_called.all(-1)[:,None]
        evidence = markers.log_evidence.gather(2,q.clamp_min(0)[:,None,:].expand(-1,4,-1))
        values = evidence.permute(0,2,1).contiguous()
        values = (values-values.mean(-1,keepdim=True)).masked_fill(~active[:,:,None],0.)
        joined = torch.cat((values,active.to(torch.float32)[:,:,None]),-1)
        count = (s+SITE_BLOCK-1)//SITE_BLOCK
        pad = count*SITE_BLOCK-s
        if pad:
            joined = torch.cat((joined,joined.new_zeros((pad,b,5))),0)
        block = joined.reshape(count,SITE_BLOCK,b,5)
        self.prefix = torch.cat((block.new_zeros((count,1,b,5)),block.cumsum(1)),1)

    def ranges(self, left, right):
        _require(left.device.type==right.device.type=="cpu" and left.dtype==right.dtype==torch.long
                 and left.shape==right.shape,"CPU physical range lookup")
        if not self.s:
            return self.prefix.new_zeros((*left.shape,self.b,5))
        # For an empty terminal range use the last existing block and mask.
        lo = left.clamp_max(self.s-1)
        hi = (right-1).clamp_min(0).clamp_max(self.s-1)
        lb,rb = lo//SITE_BLOCK,hi//SITE_BLOCK
        li,ri = left-lb*SITE_BLOCK,right-rb*SITE_BLOCK
        first = self.prefix[lb,li]
        last = self.prefix[rb,ri]
        tail = self.prefix[lb,SITE_BLOCK]
        result = torch.where((lb==rb)[...,None,None],last-first,tail-first+last)
        return result.masked_fill((left==right)[...,None,None],0.)

    def features(self, window_index):
        left = self.ranges(window_index[:,0],window_index[:,1])
        right = self.ranges(window_index[:,1],window_index[:,2])
        nl,nr = left[...,4],right[...,4]
        ell_l = left[...,:4]/nl.clamp_min(1.)[:,:,None]
        ell_r = right[...,:4]/nr.clamp_min(1.)[:,:,None]
        difference = ell_r-ell_l
        norm = difference.square().sum(-1)
        gate = (norm/(norm+1.))*((nl>0)&(nr>0)).to(torch.float32)
        callability = torch.minimum(nl,nr).log1p()/math.log(100001.)
        return torch.cat((.5*(ell_l+ell_r),difference,callability[:,:,None],gate[:,:,None]),-1)


class ContinuousCladeFlankGNNV1(nn.Module):
    def __init__(self, *, arm="foreign_flank", query_block=32, center_block=1024,
                 checkpoint_readout=True, checkpoint_scan=True):
        super().__init__()
        _require(arm in ARMS and type(query_block) is int and 1<=query_block<=686
                 and type(center_block) is int and 1<=center_block<=20000,"Registered arm/bounded chunks")
        self.arm,self.query_block,self.center_block = arm,query_block,center_block
        self.checkpoint_readout,self.checkpoint_scan = bool(checkpoint_readout),bool(checkpoint_scan)
        self.backbone = AnonymousCladeFlagControlGNNV1(**MODEL)
        self.A = nn.Linear(32,4,bias=False,dtype=torch.float32)
        self.H = nn.Linear(27,8,bias=False,dtype=torch.float32)
        self.O = nn.Linear(8,4,bias=False,dtype=torch.float32)
        theta = math.log((5000.-100.)/(100000.-5000.))
        self.theta = nn.Parameter(torch.tensor(theta,dtype=torch.float32))
        nn.init.zeros_(self.O.weight)
        self.float()
        _require(sum(p.numel() for p in self.parameters())==31165,"Exact30788+377 learned parameters")

    @property
    def tau(self):
        return 100.+(100000.-100.)*self.theta.sigmoid()

    def backbone_state_v1(self,bank,flags):
        parameter = next(self.parameters())
        t,h = bank.sample_index.shape
        _require(parameter.dtype==torch.float32 and h==952 and bank.query_column.numel()==686
                 and flags.native_holds==t and flags.haplotypes==h,"Complete original FP32 H042 body")
        for name in _FLOAT_FIELDS:
            value = getattr(bank,name)
            _require(value.dtype==torch.float32 and value.device==parameter.device,"Original FP32 observations")
        for f in fields(bank):
            value = getattr(bank,f.name)
            values = value if f.name in _LEVEL_FIELDS else (value,)
            _require(all(v.device==parameter.device for v in values),"Complete original observation device")
        base,ug,dg,degree = self.backbone.prepare_tree_base_v1(bank,control="aligned")
        upward,_ = self.backbone.sweep_tree_state_v1(bank,base,ug,dg,degree)
        injection = self.backbone.clade_coupling(upward,flags)
        second, outside = self.backbone.sweep_tree_state_v1(bank,base+injection,ug,dg,degree)
        context = self.backbone.leaf_context_v1(bank,second,outside)
        span = bank.spans.log1p()/math.log(2000001.)
        previous = torch.cat((bank.spans.new_zeros(1),.5*(bank.spans[1:]+bank.spans[:-1])))
        following = torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]),bank.spans.new_zeros(1)))
        geometry = torch.stack((span,previous.log1p()/math.log(2000001.),
                                 following.log1p()/math.log(2000001.)),-1)
        sequence = torch.cat((context,geometry[:,None].expand(t,h,3)),-1)
        history,_ = self.backbone.genomic_gru(sequence)
        return self.backbone.classifier(torch.cat((context,history),-1)),upward

    def _readout_block(self,base,projected,left,right,node_member,samples,query,holds,
                       distance_left,distance_right,local_cpu,tau):
        device = base.device
        selected_base = base[holds[:,None],query[None,:]]
        native = samples[holds[:,None],query[None,:]]
        member = node_member[native].long()
        eligible = member>=0
        safe = member.clamp_min(0).flatten()
        shape = (*member.shape,5)
        l = left.index_select(0,safe).reshape(shape)*torch.exp(-distance_left/tau)[:,None,None]
        r = right.index_select(0,safe).reshape(shape)*torch.exp(-distance_right/tau)[:,None,None]
        vl,vr,cl,cr = l[...,:4],r[...,:4],l[...,4:5],r[...,4:5]
        if self.arm=="own_state":
            current = projected.index_select(0,safe).reshape(*member.shape,4)
            vl,vr = cl*current+0.*vl,cr*current+0.*vr
        local = local_cpu.to(device=device,dtype=torch.float32)
        mean,difference,number,gate = local[...,:4],local[...,4:8],local[...,8:9],local[...,9:10]
        change = (vr-vl).abs()
        value = torch.cat((selected_base,mean,difference,vl,vr,change,cl,cr,number),-1)
        swapped = torch.cat((selected_base,mean,-difference,vr,vl,change,cr,cl,number),-1)
        hidden = .5*(self.H(value).tanh()+self.H(swapped).tanh())
        correction = gate*self.O(hidden).tanh()*eligible[:,:,None].to(torch.float32)
        return selected_base+correction

    def forward(self,bundle):
        _require(isinstance(bundle,H046BoundCladeBankV1),"Bound original H046 complete observation bundle")
        bank,markers,flanks = bundle.validate()
        base,upward = self.backbone_state_v1(bank,flanks.flags)
        _require(flanks.hold_span.device==base.device and flanks.member_node.device==base.device,
                 "Same-device complete clade payload")
        if not flanks.member_node.numel():
            return base.index_select(0,flanks.grid_hold).index_select(1,bank.query_column)
        projected = self.A(upward.index_select(0,flanks.member_node.long()))
        left,right = segmented_flank_integrals_v1(
            projected,flanks.hold_span.index_select(0,flanks.member_hold.long()),
            flanks.episode_start,flanks.episode_end,self.tau,checkpoint_scan=self.checkpoint_scan)
        blocks = []
        for start in range(0,len(bank.query_column),self.query_block):
            end = min(start+self.query_block,len(bank.query_column))
            prefix = LocalReferencePrefixV1(markers,start,end)
            pieces = []
            for first in range(0,len(markers.centers_bp),self.center_block):
                last = min(first+self.center_block,len(markers.centers_bp))
                local = prefix.features(flanks.window_index[first:last])
                args = (base,projected,left,right,flanks.node_member,bank.sample_index,
                        bank.query_column[start:end],flanks.grid_hold[first:last],
                        flanks.distance_left[first:last],flanks.distance_right[first:last],local,self.tau)
                if self.checkpoint_readout and self.training and torch.is_grad_enabled():
                    logits = checkpoint(self._readout_block,*args,use_reentrant=False)
                else:
                    logits = self._readout_block(*args)
                pieces.append(logits)
            del prefix
            blocks.append(torch.cat(pieces,0))
        return torch.cat(blocks,1)


def load_h042_flank_warmstart_v1(model,payload):
    _require(isinstance(model,ContinuousCladeFlankGNNV1) and isinstance(payload,Mapping)
             and payload.get("schema")=="anonymous_clade_flag_checkpoint_v1"
             and payload.get("hypothesis_id")=="H042" and payload.get("arm")=="exact_clade_flags"
             and payload.get("pass_index")==30 and payload.get("additional_optimizer_steps")==960
             and payload.get("parameter_count")==30788 and payload.get("seed")==101
             and payload.get("teacher_used") is False and payload.get("model_kwargs")==MODEL,
             "Original H042 seed101 pass30 learned payload")
    state = payload.get("model_state")
    expected = model.backbone.state_dict()
    _require(isinstance(state,Mapping) and set(state)==set(expected),"Complete H042 learned state")
    for key,value in state.items():
        _require(isinstance(value,torch.Tensor) and value.device.type=="cpu"
                 and value.dtype==torch.float32 and value.shape==expected[key].shape
                 and bool(torch.isfinite(value).all()),"Finite original FP32 H042 state: "+key)
    _require(not bool(model.O.weight.count_nonzero()) and all(p.requires_grad for p in model.parameters()),
             "New output exactly zero; full body trainable")
    model.backbone.load_state_dict(state,strict=True)
    return dict(H042_checkpoint_ref=H042_PASS30_REF,H042_completion_ref=H042_COMPLETION_REF,
        base_parameters=30788,added_parameters=377,total_parameters=31165,
        all_backbone_weights_preserved=True,initial_residual_exactly_zero=True,
        invariant_only_under_fixed_field_source_support_integration_mesh_refinement=True)


def make_matched_flank_family_v1(payload, **readout):
    _require(not readout or readout==READOUT,"Frozen numerical readout settings")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        prototype = ContinuousCladeFlankGNNV1(**(readout or READOUT))
        proof = load_h042_flank_warmstart_v1(prototype,payload)
        own = copy.deepcopy(prototype)
        own.arm = "own_state"
    models = {"foreign_flank":prototype,"own_state":own}
    optimizers = {a:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,
                                    betas=(.9,.999),eps=1e-8) for a,m in models.items()}
    return models,optimizers,proof


def logical_flank_bounds_v1(s,t,n,m,e,p,*,q=686,query_block=32,center_block=1024):
    """Named tensors, not a process RSS, autograd fit or GPU admission proof."""
    _require(all(type(v) is int and v>=0 for v in (s,t,n,m,e,p,q,query_block,center_block))
             and 0<t and 0<q and 0<p and m<=n and e<=m,"Actual bounded counts")
    b,c = min(q,query_block),min(p,center_block)
    padded = max(SITE_BLOCK,((s+SITE_BLOCK-1)//SITE_BLOCK)*SITE_BLOCK)
    return dict(
        parameters_per_arm=31165,added_parameters=377,FP32_parameter_bytes=124660,
        compact_flank_payload_bytes=6*n+14*m+4*e+4*t+40*p,
        compact_flank_payload_upper_bound_bytes=24*n+4*t+40*p,
        GPU_flank_payload_bytes=6*n+14*m+4*e+4*t+16*p,
        projected_member_table_bytes=16*m,
        two_integral_endpoint_table_bytes=40*m,
        # O(M) reductions/reconstruction and backward-save allowance; numerical
        # scans are checkpointed by default. Not total allocator/autograd memory.
        scan_named_tensor_allowance_bytes=1024*m,
        CPU_block_prefix_bytes=20*((padded//SITE_BLOCK)*(SITE_BLOCK+1))*b,
        CPU_prefix_construction_allowance_bytes=128*(s+SITE_BLOCK)*b,
        CPU_all_checkpoint_local_feature_bytes=40*p*q,
        GPU_one_head_block_named_allowance_bytes=1024*c*b,
        original_grid_logits_or_target_bytes=16*p*q,
        all96_original_grid_probability_bytes=96*16*p*q,
        original_backbone_autograd_and_allocator_peak_unmeasured=True,
        integration_scan_arithmetic="O(5*M), two segmented affine scans",
        no_per_grid_Python_loop=True,no_dense_tree_pair_matrix=True,
    )
