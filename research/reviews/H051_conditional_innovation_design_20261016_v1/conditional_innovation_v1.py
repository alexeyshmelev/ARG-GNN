"""Dimension-generic occupied-cell conditional innovation; no readers or jobs.

J=(Pi_BC-Pi_B)Pi_AB. Plans are built from caller-authenticated observed cut
memberships, not ancestry targets. Optional edge-continuity gates compare only
endpoint equality inside one authenticated inferred-tskit namespace.
No model integration, biological-ancestor claim, target resolver, file reader,
checkpoint reader, training loop, random initialization or implicit scan.
"""
from dataclasses import dataclass,fields,replace
import math,re
import torch
from torch import nn

_FIELDS=("leaf_ab","leaf_b","leaf_bc","ab_count","b_count","bc_count",
         "ab_to_b","bc_to_b","edge_ab","edge_bc","edge_count")

def require(ok,message):
    if not ok:raise ValueError(message)

def _digest(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{64}",value) is not None

def _stamp(tensors):
    return tuple((id(x),x.data_ptr(),x._version,tuple(x.shape),x.dtype,str(x.device)) for x in tensors)

def observed_contract_v1(record):
    expected={"schema","observed_trees_sha256","builder_sha256",
              "targets_or_outcomes_used","identity_namespace_sha256"}
    require(isinstance(record,dict) and set(record)==expected
        and record["schema"]=="H051_observation_only_cut_source_v1"
        and _digest(record["observed_trees_sha256"]) and _digest(record["builder_sha256"])
        and record["targets_or_outcomes_used"] is False
        and (record["identity_namespace_sha256"] is None or _digest(record["identity_namespace_sha256"])),
        "Explicit observation-only source contract, externally authenticated by caller")
    return tuple((key,record[key]) for key in sorted(record))

@dataclass(frozen=True)
class OccupiedPlanV1:
    leaf_ab:torch.Tensor
    leaf_b:torch.Tensor
    leaf_bc:torch.Tensor
    ab_count:torch.Tensor
    b_count:torch.Tensor
    bc_count:torch.Tensor
    ab_to_b:torch.Tensor
    bc_to_b:torch.Tensor
    edge_ab:torch.Tensor
    edge_bc:torch.Tensor
    edge_count:torch.Tensor
    event_sizes:tuple
    contracts:tuple
    stamps:tuple=()

    def tensors(self):
        return tuple(getattr(self,key) for key in _FIELDS)

    @property
    def h(self):return self.leaf_ab.numel()

    @property
    def old_cells(self):return self.ab_count.numel()

    @property
    def middle_cells(self):return self.b_count.numel()

    @property
    def new_cells(self):return self.bc_count.numel()

    @property
    def occupied_triples(self):return self.edge_count.numel()

    @property
    def device(self):return self.leaf_ab.device

    def validate(self):
        require(self.stamps==_stamp(self.tensors()),"Immutable occupied plan changed")
        return self

    def to(self,device):
        self.validate()
        result=replace(self,**{key:getattr(self,key).to(device) for key in _FIELDS},stamps=())
        return replace(result,stamps=_stamp(result.tensors()))

    def tensor_bytes(self):
        return sum(x.numel()*x.element_size() for x in self.tensors())

def make_plan_v1(a,b,c,*,provenance):
    """CPU metadata construction, O(H log H) sort work and O(H) stored cells."""
    contract=observed_contract_v1(provenance)
    xs=(a,b,c)
    require(all(isinstance(x,torch.Tensor) and x.ndim==1 and x.device.type=="cpu"
        and x.dtype in (torch.int32,torch.int64) for x in xs)
        and a.numel()>0 and a.shape==b.shape==c.shape,
        "Complete aligned persistent-leaf cut memberships; CPU integer vectors")
    a,b,c=(x.to(torch.long) for x in xs)
    unused,lb,nb=torch.unique(b,sorted=True,return_inverse=True,return_counts=True)
    unused,lab,nab=torch.unique(torch.stack((a,b),1),dim=0,sorted=True,return_inverse=True,return_counts=True)
    unused,lbc,nbc=torch.unique(torch.stack((b,c),1),dim=0,sorted=True,return_inverse=True,return_counts=True)
    ab_b=torch.empty(nab.numel(),dtype=torch.long).scatter_(0,lab,lb)
    bc_b=torch.empty(nbc.numel(),dtype=torch.long).scatter_(0,lbc,lb)
    edges,ne=torch.unique(torch.stack((lab,lbc),1),dim=0,sorted=True,return_counts=True)
    values=[lab,lb,lbc,nab,nb,nbc,ab_b,bc_b,edges[:,0].contiguous(),
            edges[:,1].contiguous(),ne]
    sizes=((a.numel(),nab.numel(),nb.numel(),nbc.numel(),ne.numel()),)
    result=OccupiedPlanV1(*values,sizes,(contract,))
    return replace(result,stamps=_stamp(result.tensors()))

def batch_plans_v1(plans):
    """Disjoint ragged events: no cross-parent cells or H-padding."""
    plans=tuple(plans)
    require(plans and all(isinstance(p,OccupiedPlanV1) for p in plans),"Nonempty occupied plans")
    for plan in plans:plan.validate()
    require(len({str(p.device) for p in plans})==1,"One batch device")
    offsets={"leaf_ab":0,"leaf_b":0,"leaf_bc":0,"ab_to_b":0,"bc_to_b":0,
             "edge_ab":0,"edge_bc":0}
    values={key:[] for key in _FIELDS}
    for p in plans:
        for key in _FIELDS:values[key].append(getattr(p,key)+offsets[key] if key in offsets else getattr(p,key))
        offsets["leaf_ab"]+=p.old_cells;offsets["edge_ab"]+=p.old_cells
        offsets["leaf_b"]+=p.middle_cells;offsets["ab_to_b"]+=p.middle_cells;offsets["bc_to_b"]+=p.middle_cells
        offsets["leaf_bc"]+=p.new_cells;offsets["edge_bc"]+=p.new_cells
    result=OccupiedPlanV1(*(torch.cat(values[key]) for key in _FIELDS),
        tuple(s for p in plans for s in p.event_sizes),tuple(c for p in plans for c in p.contracts))
    return replace(result,stamps=_stamp(result.tensors()))

def _accumulation_dtype(value):
    require(value.dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64),
        "Real floating states")
    return torch.float32 if value.dtype in (torch.float16,torch.bfloat16) else value.dtype

def _broadcast(value,ndim):
    return value.reshape((-1,)+(1,)*(ndim-1))

def _sum(value,index,rows,pyg_sum=None):
    if pyg_sum is not None:return pyg_sum(value,index=index,dim_size=rows,dim=0)
    return value.new_zeros((rows,)+tuple(value.shape[1:])).index_add(0,index,value)

def _mean(value,index,count,pyg_sum=None):
    return _sum(value,index,count.numel(),pyg_sum)/_broadcast(count.to(value.dtype),value.ndim)

@dataclass(frozen=True)
class EdgeWitnessV1:
    before:torch.Tensor
    after:torch.Tensor
    namespace_sha256:tuple

def edge_continuity_gate_v1(plan,witness,*,prior_count=16.):
    """alpha_B=stable valid boundary-edge leaves/(n_B+kappa), no learned IDs.

    Endpoints are (parent,child) global node keys, not edge row IDs. Missing/root
    records must be (-1,-1) and contribute no corroboration. Namespace identity
    and each edge's actual interval coverage require external source review.
    """
    plan.validate()
    require(isinstance(witness,EdgeWitnessV1) and math.isfinite(prior_count) and prior_count>=0,
        "Observed endpoint witness and nonnegative fixed prior count")
    expected=tuple(dict(c)["identity_namespace_sha256"] for c in plan.contracts)
    require(all(x is not None for x in expected) and witness.namespace_sha256==expected,
        "One exact authenticated inferred-tskit node namespace per event")
    x,y=witness.before,witness.after
    require(all(isinstance(z,torch.Tensor) and z.dtype==torch.long and z.shape==(plan.h,2)
        and z.device==plan.device and not z.requires_grad for z in (x,y)),
        "Complete persistent-leaf endpoint sidecars on plan device")
    def valid(z):
        present=(z>=0).all(1);missing=(z==-1).all(1)
        require(bool((present|missing).all()) and bool((~present|(z[:,0]!=z[:,1])).all()),
            "Real edge endpoints or explicit missing/root sentinel")
        return present
    vx,vy=valid(x),valid(y);eligible=vx&vy
    stable=eligible&(x==y).all(1)
    counts=torch.bincount(plan.leaf_b[stable],minlength=plan.middle_cells)
    coverage=torch.bincount(plan.leaf_b[eligible],minlength=plan.middle_cells)
    alpha=counts.to(torch.float64)/(plan.b_count.to(torch.float64)+prior_count)
    return alpha,dict(stable_count=counts,eligible_count=coverage,
        middle_count=plan.b_count,identity_is_inferred_bookkeeping_not_true_ancestor=True)

class ConditionalInnovationTransportV1(nn.Module):
    """Parameter-free, arbitrary feature trailing dimensions; optional PyG sums."""
    def __init__(self,*,gate="fixed",alpha=1.,prior_count=16.,backend="torch"):
        super().__init__()
        require(gate in ("fixed","count","edge_continuity") and backend in ("torch","pyg"),
            "Declared gate and aggregation backend")
        require(type(alpha) in (float,int) and math.isfinite(alpha) and 0<=alpha<=1
            and type(prior_count) in (float,int) and math.isfinite(prior_count) and prior_count>=0,
            "Fixed bounded alpha/nonnegative prior count")
        self.gate=gate;self.alpha=float(alpha);self.prior_count=float(prior_count)
        self.pyg_sum=None
        if backend=="pyg":
            from torch_geometric.nn.aggr import SumAggregation
            self.pyg_sum=SumAggregation()

    def components(self,state,plan,*,basis="ab"):
        plan.validate()
        require(isinstance(state,torch.Tensor) and state.ndim>=1 and state.device==plan.device
            and basis in ("ab","leaf"),"Same-device floating state and explicit input basis")
        dtype=_accumulation_dtype(state);value=state.to(dtype)
        require(state.shape[0]==(plan.old_cells if basis=="ab" else plan.h),"Complete state rows")
        if basis=="leaf":value=_mean(value,plan.leaf_ab,plan.ab_count,self.pyg_sum)
        triple=_mean(value.index_select(0,plan.edge_ab)*_broadcast(plan.edge_count.to(dtype),value.ndim),
            plan.edge_bc,plan.bc_count,self.pyg_sum)
        middle=_mean(value*_broadcast(plan.ab_count.to(dtype),value.ndim),
            plan.ab_to_b,plan.b_count,self.pyg_sum)
        marginal=middle.index_select(0,plan.bc_to_b)
        return dict(triple=triple,marginal=marginal,innovation=triple-marginal)

    def forward(self,state,plan,*,basis="ab",witness=None):
        parts=self.components(state,plan,basis=basis)
        if self.gate=="fixed":
            require(witness is None,"No hidden side input in fixed gate")
            alpha=parts["triple"].new_full((plan.middle_cells,),self.alpha)
        elif self.gate=="count":
            require(witness is None,"Count gate uses only observed cell counts")
            count=plan.b_count.to(parts["triple"].dtype);alpha=count/(count+self.prior_count)
        else:
            alpha,unused=edge_continuity_gate_v1(plan,witness,prior_count=self.prior_count)
            alpha=alpha.to(parts["triple"].dtype)
        gate=_broadcast(alpha.index_select(0,plan.bc_to_b),state.ndim)
        return (parts["marginal"]+gate*parts["innovation"]).to(state.dtype)

def relax_interval_v1(state,forcing,distance,length_scale,*,distance_unit,scale_unit):
    """Within a fixed carrier: exact exponential forcing, no event or hidden reset."""
    require(distance_unit==scale_unit and distance_unit in ("bp","morgan"),"Explicit identical genomic units")
    require(isinstance(state,torch.Tensor) and isinstance(forcing,torch.Tensor)
        and state.ndim>=1 and state.shape==forcing.shape and state.dtype==forcing.dtype and state.device==forcing.device,
        "Same complete carrier shape/dtype/device")
    dtype=_accumulation_dtype(state);s=state.to(dtype);f=forcing.to(dtype)
    distance=torch.as_tensor(distance,dtype=dtype,device=state.device)
    scale=torch.as_tensor(length_scale,dtype=dtype,device=state.device)
    require((distance.ndim==0 or distance.ndim==1 and distance.shape[0]==state.shape[0])
        and scale.ndim==0 and not distance.requires_grad
        and bool(torch.isfinite(distance).all()) and bool((distance>=0).all())
        and bool(torch.isfinite(scale)) and bool(scale>0),"Observed nonnegative scalar/per-cell distance; positive finite scale")
    ratio=distance/scale
    if ratio.ndim==1:ratio=_broadcast(ratio,state.ndim)
    rho=torch.exp(-ratio);weight=-torch.expm1(-ratio)
    return (rho*s+weight*f).to(state.dtype)

def new_cell_event_distances_v1(plan,event_distances):
    """Expand independent ragged-event distances to new cells, never to an H-by-H matrix."""
    plan.validate()
    value=torch.as_tensor(event_distances,dtype=torch.float64,device=plan.device)
    require(value.ndim==1 and len(value)==len(plan.event_sizes) and not value.requires_grad
        and bool(torch.isfinite(value).all()) and bool((value>=0).all()),"One observed distance per ragged event")
    counts=torch.tensor([s[3] for s in plan.event_sizes],dtype=torch.long,device=plan.device)
    return torch.repeat_interleave(value,counts)

def interval_morgan_distances_v1(map_bp,map_morgan,intervals_bp):
    """Integrate one fixed piecewise-linear cumulative map; no map/file reader."""
    xs=(map_bp,map_morgan,intervals_bp)
    require(all(isinstance(x,torch.Tensor) and x.device.type=="cpu" and x.dtype==torch.float64
        and not x.requires_grad and bool(torch.isfinite(x).all()) for x in xs),
        "Fixed observed CPU float64 map and intervals")
    require(map_bp.ndim==map_morgan.ndim==1 and map_bp.numel()==map_morgan.numel()>=2
        and bool((map_bp[1:]>map_bp[:-1]).all()) and bool((map_morgan[1:]>=map_morgan[:-1]).all())
        and intervals_bp.ndim==2 and intervals_bp.shape[1]==2
        and bool((intervals_bp[:,1]>=intervals_bp[:,0]).all())
        and bool((intervals_bp>=map_bp[0]).all()) and bool((intervals_bp<=map_bp[-1]).all()),
        "Monotone cumulative Morgan map and contained ordered intervals")
    z=intervals_bp.reshape(-1)
    at=torch.searchsorted(map_bp,z,right=True).sub(1).clamp(0,map_bp.numel()-2)
    fraction=(z-map_bp[at])/(map_bp[at+1]-map_bp[at])
    value=(map_morgan[at]+fraction*(map_morgan[at+1]-map_morgan[at])).reshape(-1,2)
    return value[:,1]-value[:,0]
