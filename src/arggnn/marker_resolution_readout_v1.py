"""H044 categorical SNP-resolved readout and exact tree-pooled matched control.

No I/O, guard, driver, runtime qualification or execution authority. Observation
preparation runs once on admitted CPU arrays. Original H042 source is unchanged.
All learned operations/gradients are FP32 and the complete genomic GRU remains.
"""
from __future__ import annotations
from dataclasses import dataclass,replace,fields
import copy,hashlib,math,re
from typing import Mapping
import torch
from torch import Tensor,nn
from torch.utils.checkpoint import checkpoint

ARMS=("localized","tree_pooled")
RADII_BP=(1000.,5000.,20000.)
CLASSES=4
MAX_ALLELES=4
ADDED_PARAMETERS=640
TOTAL_PARAMETERS=31428
ORIGINAL_LOADER_REF={'path': 'src/arggnn/mutation_routed_full_panel_observations_v1.py', 'sha256': 'a7184be9a735d95b80975f68666d51e9ed0316cc535d9f7b54baafea532a1107'}
H042_PASS30_REF={'path': 'experiments/results/H042_GPU4_clade_flags_train_seed101_v1/exact_clade_flags.pass_30.pt', 'sha256': '520fcffa1d5e85603620a7f4688b7fdf46669686fea7960fc311852ff2f7e366'}
H042_COMPLETION_REF={'path': 'experiments/results/H042_GPU4_clade_flags_train_seed101_v1/completed.json', 'sha256': '5b7d7adab0de14c18016db86bfc18ddc2b1d652ff530fadd101aaa73c90a6984'}

def _require(value,message):
    if not value:raise ValueError(message)

def _cpu(tensor,dtype=None):
    _require(isinstance(tensor,Tensor) and tensor.device.type=="cpu"
        and not tensor.requires_grad and tensor.is_contiguous(),"Contiguous non-gradient CPU observations required")
    if dtype is not None:_require(tensor.dtype==dtype,"Exact observation dtype required")

def tensor_identity_v1(tensor):
    """Stream exact ordered CPU bytes; no payload-sized bytes copy."""
    _cpu(tensor)
    view=memoryview(tensor.numpy()).cast("B");digest=hashlib.sha256()
    for start in range(0,len(view),8388608):digest.update(view[start:start+8388608])
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=digest.hexdigest())

def _ref(record):
    return (isinstance(record,dict) and set(record)=={"path","sha256"}
        and isinstance(record["path"],str)
        and not any(s in record["path"].lower() for s in ("truth","oracle"))
        and re.fullmatch("[0-9a-f]{64}",str(record["sha256"])) is not None)

@dataclass(frozen=True)
class MarkerCoordinatesV1:
    positions_bp0: Tensor          # [S] float64, exact original observed site order
    centers_bp: Tensor             # [P] float64, original50+100*k scoring centers
    bin_edges_bp: Tensor           # [P+1] float64, original clipped scoring edges
    intervals_bp: Tensor           # [T,2] float64, original complete native holds
    allele_cardinality: Tensor|None # [S] int64 from original observed loader, or explicit support rule
    binding: Mapping               # root-verified observed-entry/source/ordered-array identity

@dataclass(frozen=True)
class MarkerEvidenceBankV1:
    genotype: Tensor               # borrowed [S,H], original categorical calls
    query_column: Tensor           # borrowed [Q], original query order
    positions_bp0: Tensor           # [S] float64
    centers_bp: Tensor              # [P] float64
    bin_widths_bp: Tensor           # [P] float32
    readout_hold: Tensor            # [P] int64
    log_evidence: Tensor            # [S,4,4] float32; absent class exactlyzero
    reference_called: Tensor        # [S,4] bool
    window_left: Tensor             # [P,3] int64, computed once
    window_right: Tensor            # [P,3] int64
    pooled_hold_index: Tensor       # [P] int64, only original holds with scoring centers
    n_scored_holds: int
    n_holds: int
    support_rule: str
    original_binding: Mapping|None=None

    def tensor_bytes(self):
        """Charge borrowed genotype too; caller may deduplicate actual shared storage."""
        return sum(t.numel()*t.element_size() for t in (
            self.genotype,self.query_column,self.positions_bp0,self.centers_bp,
            self.bin_widths_bp,self.readout_hold,self.log_evidence,self.reference_called,
            self.window_left,self.window_right,self.pooled_hold_index))

    def features_for_queries(self,start,end):
        """O(S*B*C + P*B*C*R), two prefixes; no marker recurrence/affinity."""
        q=self.genotype.index_select(1,self.query_column[start:end]).long()
        _require(0<=start<end<=len(self.query_column),"Nonempty exact query block required")
        b=end-start
        active=(q>=0)[:,:,None]&self.reference_called[:,None,:]
        values=self.log_evidence.gather(2,q.clamp_min(0)[:,None,:].expand(-1,CLASSES,-1))
        values=values.permute(0,2,1).contiguous().masked_fill(~active,0.)
        zeros=values.new_zeros((1,b,CLASSES))
        # Observation-only accumulation: avoid cancelling weak late windows
        # against large chromosome-prefix totals. Learned features stay FP32.
        sums=torch.cat((zeros.to(torch.float64),values.to(torch.float64).cumsum(0)),0)
        counts=torch.cat((zeros,active.to(torch.float32).cumsum(0)),0)
        total=sums[self.window_right]-sums[self.window_left] # [P,R,B,C]
        number=counts[self.window_right]-counts[self.window_left]
        average=(total/number.to(torch.float64).clamp_min(1.)).to(torch.float32)
        # Counts are exactly represented FP32 integers for the physical2Mb unique-site bound.
        feature=torch.cat((average.permute(0,2,3,1),number.log1p().permute(0,2,3,1)),-1)
        return feature.contiguous() # [P,B,C,6], all missing/absent evidence and countszero

    def pool_within_trees(self,feature):
        """Pool these exact features, not SNPs/labels, before the identical decoder."""
        _require(feature.device.type=="cpu" and feature.dtype==torch.float32
            and feature.shape[0]==len(self.centers_bp) and feature.shape[2:]==(CLASSES,6),
            "Exact localized FP32 feature block required")
        shape=(self.n_scored_holds,*feature.shape[1:])
        sums=feature.new_zeros(shape)
        sums.index_add_(0,self.pooled_hold_index,feature*self.bin_widths_bp[:,None,None,None])
        widths=self.bin_widths_bp.new_zeros(self.n_scored_holds)
        widths.index_add_(0,self.pooled_hold_index,self.bin_widths_bp)
        pooled=sums/widths.clamp_min(1e-30)[:,None,None,None]
        return pooled.index_select(0,self.pooled_hold_index)

def build_marker_evidence_arrays_v1(genotype,reference_class,query_column,site_hold,readout_hold,
        positions_bp0,centers_bp,bin_edges_bp,intervals_bp,*,allele_cardinality=None):
    """Pure admitted-array algebra, with a generic tiny-panel API for finite tests.

    Actual H044 preparation must use prepare_marker_evidence_v1 below to verify
    original952/686 identity and sidecar binding. No rows/sites are dropped.
    """
    for x in (genotype,reference_class,query_column,site_hold,readout_hold):_cpu(x)
    for x in (positions_bp0,centers_bp,bin_edges_bp,intervals_bp):_cpu(x,torch.float64)
    _require(genotype.ndim==2 and genotype.dtype in (torch.int8,torch.int16,torch.int32,torch.int64),
        "Categorical [S,H] calls required")
    s,h=genotype.shape;p=len(centers_bp);t=len(intervals_bp)
    _require(h>0 and 0<=s<=2000000 and 0<p<=20000 and t>0,"Physical2Mb bounded complete representation")
    _require(reference_class.dtype==query_column.dtype==site_hold.dtype==readout_hold.dtype==torch.long,
        "Original int64 identity fields required")
    _require(reference_class.shape==(h,) and query_column.ndim==1 and len(query_column)>0
        and site_hold.shape==(s,) and readout_hold.shape==(p,)
        and positions_bp0.shape==(s,) and centers_bp.shape==(p,)
        and bin_edges_bp.shape==(p+1,) and intervals_bp.shape==(t,2),"Exact aligned shapes")
    _require(bool(torch.isfinite(positions_bp0).all() and torch.isfinite(centers_bp).all()
        and torch.isfinite(bin_edges_bp).all() and torch.isfinite(intervals_bp).all()),"Finite physical coordinates")
    _require(bool((genotype>=-1).all() and (genotype<MAX_ALLELES).all()
        and (reference_class>=-1).all() and (reference_class<CLASSES).all()),"All original categorical alleles/classes retained")
    _require(bool((query_column>=0).all() and (query_column<h).all())
        and query_column.unique().numel()==len(query_column)
        and bool((reference_class[query_column]==-1).all()),"Actual unknown-class queries only")
    _require(bool((intervals_bp[:,1]>intervals_bp[:,0]).all())
        and bool((intervals_bp[1:,0]==intervals_bp[:-1,1]).all())
        and 0<=float(intervals_bp[0,0])<float(intervals_bp[-1,1])<=2000000.,"Unchanged contiguous original holds")
    _require(bool((positions_bp0[1:]>positions_bp0[:-1]).all())
        and bool((positions_bp0==positions_bp0.floor()).all())
        and bool((positions_bp0>=intervals_bp[0,0]).all() and (positions_bp0<intervals_bp[-1,1]).all()),
        "All sites in exact unique original integer-position order")
    _require(bool((centers_bp[1:]-centers_bp[:-1]==100.).all())
        and bool(((centers_bp-50.)/100.==((centers_bp-50.)/100.).round()).all())
        and bool((centers_bp>=intervals_bp[0,0]).all() and (centers_bp<intervals_bp[-1,1]).all()),
        "Original complete100bp scoring centers, no support reduction")
    _require(bool((bin_edges_bp[1:]>bin_edges_bp[:-1]).all())
        and bool((centers_bp>=bin_edges_bp[:-1]).all() and (centers_bp<bin_edges_bp[1:]).all()),
        "Original positive clipped scoring bins")
    _require(torch.equal(torch.searchsorted(intervals_bp[:,1].contiguous(),positions_bp0,right=True),site_hold)
        and torch.equal(torch.searchsorted(intervals_bp[:,1].contiguous(),centers_bp,right=True),readout_hold),
        "SNP and scoring-center memberships match the unchanged native bank")
    if allele_cardinality is not None:
        _cpu(allele_cardinality,torch.long)
        _require(allele_cardinality.shape==(s,) and bool((allele_cardinality>=1).all()
            and (allele_cardinality<=MAX_ALLELES).all())
            and bool(((genotype<allele_cardinality[:,None])|(genotype<0)).all()),
            "Original per-site allele-cardinality/encoding must cover every valid call")
        size=allele_cardinality.to(torch.float32)
        support=torch.arange(MAX_ALLELES)[None,:]<allele_cardinality[:,None]
        rule="original_allele_cardinality"
    else:
        support=torch.stack([(genotype==a).any(1) for a in range(MAX_ALLELES)],-1)
        size=support.sum(-1).to(torch.float32).clamp_min(1.)
        rule="all_observed_panel_alleles"
    # Fixed four-allele count contractions; no learned emissions, per-SNP loop or S*H*C materialization.
    onehot=torch.nn.functional.one_hot(reference_class.clamp_min(0),CLASSES).to(torch.float32)
    onehot=onehot*(reference_class>=0)[:,None]
    count=torch.stack([(genotype==a).to(torch.float32)@onehot for a in range(MAX_ALLELES)],-1)
    total=count.sum(-1)
    called=total>0
    evidence=((count+.5)/(total[:,:,None]+.5*size[:,None,None])).log()+size.log()[:,None,None]
    evidence=evidence.masked_fill(~called[:,:,None]|~support[:,None,:],0.)
    _require(bool(torch.isfinite(evidence).all()),"Finite categorical Jeffreys evidence")
    radii=positions_bp0.new_tensor(RADII_BP)
    left=torch.searchsorted(positions_bp0,(centers_bp[:,None]-radii).contiguous(),right=False)
    right=torch.searchsorted(positions_bp0,(centers_bp[:,None]+radii).contiguous(),right=True)
    scored,pool_index=torch.unique_consecutive(readout_hold,return_inverse=True)
    return MarkerEvidenceBankV1(genotype,query_column,positions_bp0,centers_bp,
        (bin_edges_bp[1:]-bin_edges_bp[:-1]).to(torch.float32),readout_hold,
        evidence.contiguous(),called.contiguous(),left,right,pool_index,len(scored),t,rule)

def prepare_marker_evidence_v1(bank,sidecar:MarkerCoordinatesV1):
    """Verify the root-bound original loader sidecar, then prepare observation-only evidence."""
    from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    _require(type(bank) is MutationRoutedWholeTreeInputsV1 and isinstance(sidecar,MarkerCoordinatesV1),
        "Unchanged native bank and separately bound physical sidecar required")
    _require(bank.genotype.shape[1]==952 and bank.query_column.shape==(686,),"Original full952/686 panel")
    arrays=dict(genotype=bank.genotype,reference_class=bank.reference_class,query_column=bank.query_column,
        site_hold=bank.site_hold,readout_hold=bank.readout_hold,positions_bp0=sidecar.positions_bp0,
        centers_bp=sidecar.centers_bp,bin_edges_bp=sidecar.bin_edges_bp,intervals_bp=sidecar.intervals_bp)
    if sidecar.allele_cardinality is not None:arrays["allele_cardinality"]=sidecar.allele_cardinality
    binding=sidecar.binding
    _require(binding.get("schema")=="H044_original_observed_marker_coordinate_binding_v1"
        and binding.get("original_loader_source_ref")==ORIGINAL_LOADER_REF
        and _ref(binding.get("original_observed_entry_ref"))
        and binding.get("parent_id") in tuple("ap_expanded_train_%03d"%i for i in range(32))
            +tuple("ap_expanded_val_%03d"%i for i in range(8))
        and binding.get("array_identities")=={k:tensor_identity_v1(v) for k,v in arrays.items()},
        "Exact original entry/source/row/order/coordinate/class/query identities; no oracle bridge")
    data=build_marker_evidence_arrays_v1(**arrays,allele_cardinality=None) if sidecar.allele_cardinality is None else \
        build_marker_evidence_arrays_v1(**arrays)
    _require(data.n_holds==len(bank.spans)
        and torch.allclose(bank.spans,(sidecar.intervals_bp[:,1]-sidecar.intervals_bp[:,0]).to(torch.float32),
            rtol=1e-6,atol=1e-5),"Every original complete hold span retained")
    _require(binding.get("support_rule")==data.support_rule,"Support encoding rule fixed in root sidecar")
    return replace(data,original_binding=dict(binding))

def _tensor_stamps_v1(value):
    """O(number of tensors) mutation stamps; never hash GPU payloads per step."""
    tensors=[]
    if isinstance(value,MarkerEvidenceBankV1):
        tensors=[getattr(value,f.name) for f in fields(value) if isinstance(getattr(value,f.name),Tensor)]
    else:
        for f in fields(value):
            item=getattr(value,f.name)
            tensors.extend(item if isinstance(item,tuple) else (item,))
    return tuple((id(t),t.data_ptr(),t._version,tuple(t.shape),t.dtype,str(t.device))
        for t in tensors if isinstance(t,Tensor))

@dataclass(frozen=True)
class H044BoundMarkerBankV1:
    """Exact original CPU pairing carried unchanged through typed device transfer.

    Root first verifies the source/entry/sidecar closure. CPU identities are
    checked once in bind_marker_bank_v1. Stamps then reject accidental mutation,
    replacement or cross-parent pairing without repeated GPU hashing.
    """
    bank: object
    source_bank: object
    markers: MarkerEvidenceBankV1
    source_stamps: tuple
    device_stamps: tuple
    marker_stamps: tuple

    def validate(self):
        _require(self.source_stamps==_tensor_stamps_v1(self.source_bank)
            and self.device_stamps==_tensor_stamps_v1(self.bank)
            and self.marker_stamps==_tensor_stamps_v1(self.markers),
            "Immutable original bank/marker pairing was mutated or replaced")
        return self.bank,self.markers

    def to(self,device):
        self.validate()
        moved=self.source_bank.to(device)
        return H044BoundMarkerBankV1(moved,self.source_bank,self.markers,
            self.source_stamps,_tensor_stamps_v1(moved),self.marker_stamps)

def bind_marker_bank_v1(bank,markers):
    """One-time exact original CPU bank identity verification, not shape acceptance."""
    from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    _require(type(bank) is MutationRoutedWholeTreeInputsV1 and markers.original_binding is not None,
        "Verified original CPU bank and root-bound prepared marker payload required")
    binding=markers.original_binding
    for key in ("genotype","reference_class","query_column","site_hold","readout_hold"):
        _require(tensor_identity_v1(getattr(bank,key))==binding["array_identities"][key],
            "Actual original bank ordered identity differs from sidecar")
    _require(markers.genotype is bank.genotype and markers.query_column is bank.query_column
        and markers.readout_hold is bank.readout_hold,"Markers must borrow this actual same-parent CPU bank")
    stamp=_tensor_stamps_v1(bank)
    return H044BoundMarkerBankV1(bank,bank,markers,stamp,stamp,_tensor_stamps_v1(markers))

class SharedMarkerResidualV1(nn.Module):
    """640 parameters shared across every center/query/class; no class-ID embedding."""
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(38,16,bias=True,dtype=torch.float32)
        self.output=nn.Linear(16,1,bias=False,dtype=torch.float32)
        nn.init.zeros_(self.output.weight)

    def forward(self,leaf_context,feature):
        _require(leaf_context.dtype==feature.dtype==torch.float32
            and leaf_context.shape[-1]==32 and feature.shape==(*leaf_context.shape[:-1],CLASSES,6)
            and leaf_context.device==feature.device==self.input.weight.device,"Exact same-device FP32 readout fields")
        context=leaf_context.unsqueeze(-2).expand(*feature.shape[:-1],32)
        value=self.output(torch.nn.functional.silu(self.input(torch.cat((context,feature),-1)))).squeeze(-1)
        available=(feature[...,3:]>0).any(-1)
        return value*available.to(value.dtype)

class MarkerResolutionReadoutV1(nn.Module):
    """Original H042 two complete PyG sweeps+genomicGRU, with a matched SNP residual."""
    def __init__(self,*,arm,query_block=64,center_block=2048,checkpoint_readout=True):
        super().__init__()
        from .anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1
        _require(arm in ARMS and type(query_block) is int and 1<=query_block<=686
            and type(center_block) is int and 1<=center_block<=20000,"Fixed arm and bounded numerical-equivalent chunks")
        self.arm=arm;self.query_block=query_block;self.center_block=center_block
        self.checkpoint_readout=bool(checkpoint_readout)
        self.backbone=AnonymousCladeFlagControlGNNV1(hidden_dim=32,expected_haplotypes=952,
            site_chunk_size=256,checkpoint_sites=True)
        self.marker_decoder=SharedMarkerResidualV1()
        _require(sum(p.numel() for p in self.parameters())==TOTAL_PARAMETERS,"Exact30788+640 parameters")

    def backbone_state_v1(self,bank,flags):
        """Same frozen-source H042 readout calculation, exporting classifier input."""
        _require(bank.sample_index.shape[1]==952 and bank.query_column.shape==(686,)
            and flags.native_holds==len(bank.spans)
            and all(p.dtype==torch.float32 for p in self.parameters()),"Full original FP32 H042 panel")
        context=self.backbone.encode_trees(bank,flags=flags,control="aligned")
        t,h,_=context.shape
        span=bank.spans.log1p()/math.log(2000001.)
        prev=torch.cat((bank.spans.new_zeros(1),.5*(bank.spans[1:]+bank.spans[:-1])))
        nxt=torch.cat((.5*(bank.spans[1:]+bank.spans[:-1]),bank.spans.new_zeros(1)))
        geometry=torch.stack((span,prev.log1p()/math.log(2000001.),nxt.log1p()/math.log(2000001.)),-1)
        sequence=torch.cat((context,geometry[:,None].expand(t,h,3)),-1)
        history,_=self.backbone.genomic_gru(sequence)
        return torch.cat((context,history),-1) # complete history, no detach/reset/pruning

    def _emit(self,state,feature):
        return self.backbone.classifier(state)+self.marker_decoder(state[...,:32],feature)

    def forward(self,bundle:H044BoundMarkerBankV1,*,flags):
        _require(isinstance(bundle,H044BoundMarkerBankV1),"Verified immutable original bank-marker bundle required")
        bank,markers=bundle.validate()
        state=self.backbone_state_v1(bank,flags)
        blocks=[];device=state.device
        for start in range(0,686,self.query_block):
            end=min(start+self.query_block,686)
            feature=markers.features_for_queries(start,end)
            if self.arm=="tree_pooled":feature=markers.pool_within_trees(feature)
            pieces=[];query=bank.query_column[start:end]
            query_state=state.index_select(1,query)
            for first in range(0,len(markers.centers_bp),self.center_block):
                last=min(first+self.center_block,len(markers.centers_bp))
                holds=markers.readout_hold[first:last].to(device)
                selected=query_state.index_select(0,holds)
                observed=feature[first:last].to(device)
                if self.checkpoint_readout and self.training and torch.is_grad_enabled():
                    logits=checkpoint(self._emit,selected,observed,use_reentrant=False)
                else:logits=self._emit(selected,observed)
                pieces.append(logits)
            blocks.append(torch.cat(pieces,0))
        return torch.cat(blocks,1) # original[P,686,4] scoring grid for both arms

def load_h042_pass30_v1(model,payload,*,checkpoint_ref,completion_ref):
    """No checkpoint reader or optimizer resume; root supplies exact SHA-bound payload."""
    _require(isinstance(model,MarkerResolutionReadoutV1)
        and checkpoint_ref==H042_PASS30_REF and completion_ref==H042_COMPLETION_REF
        and payload.get("schema")=="anonymous_clade_flag_checkpoint_v1"
        and payload.get("hypothesis_id")=="H042" and payload.get("arm")=="exact_clade_flags"
        and payload.get("control_id")=="H040_exact_clade_flag_control_v1"
        and payload.get("pass_index")==30 and payload.get("additional_optimizer_steps")==960
        and payload.get("parameter_count")==30788 and payload.get("seed")==101
        and payload.get("teacher_used") is False
        and all(p.requires_grad for p in model.parameters()),"Exact completed H042pass30, all learned weights trainable")
    _require(all(isinstance(t,Tensor) and t.dtype==torch.float32 for t in payload["model_state"].values()),
        "Unmodified FP32 H042 learned state")
    model.backbone.load_state_dict(payload["model_state"],strict=True)
    _require(not bool(model.marker_decoder.output.weight.count_nonzero()),"New output projection must startzero")
    return dict(H042_checkpoint_ref=checkpoint_ref,completion_ref=completion_ref,
        base_parameters=30788,added_parameters=640,total_parameters=31428,
        all_backbone_parameters_trainable=all(p.requires_grad for p in model.backbone.parameters()),
        optimizer_moments_and_step_must_reset=True,initial_residual_exactly_zero=True)

def make_matched_pair_v1(payload,*,checkpoint_ref,completion_ref):
    """Identical initialization; independent arms/optimizers and no execution authority."""
    torch.manual_seed(101)
    localized=MarkerResolutionReadoutV1(arm="localized")
    receipt=load_h042_pass30_v1(localized,payload,
        checkpoint_ref=checkpoint_ref,completion_ref=completion_ref)
    pooled=copy.deepcopy(localized);pooled.arm="tree_pooled"
    _require(all(torch.equal(v,pooled.state_dict()[k]) for k,v in localized.state_dict().items()),
        "Every backbone/decoder tensor starts identically in both matched arms")
    return {"localized":localized,"tree_pooled":pooled},receipt

def original_grid_span_nll_v1(logits,target_mass,bin_widths):
    """External TRAIN loss only; target mass is never a model/feature input."""
    _require(logits.ndim==3 and logits.shape[-1]==CLASSES and target_mass.shape==logits.shape
        and logits.dtype==target_mass.dtype==bin_widths.dtype==torch.float32
        and logits.device==target_mass.device==bin_widths.device
        and bin_widths.shape==(logits.shape[0],)
        and bool(torch.isfinite(logits).all() and torch.isfinite(target_mass).all() and (target_mass>=0).all())
        and bool((bin_widths>0).all())
        and torch.allclose(target_mass.sum(-1),bin_widths[:,None].expand_as(target_mass[...,0]),
            rtol=1e-5,atol=1e-4),"Exact original complete100bp physical mass, not native-hold aggregation")
    return -(target_mass*logits.log_softmax(-1)).sum()/target_mass.sum()

def logical_bounds_v1(s,t,p,*,h=952,q=686,genotype_bytes=1,query_block=64):
    """Exact named tensor bytes, not an allocator/runtime/training-fit claim."""
    _require(all(type(x) is int and x>=0 for x in (s,t,p,h,q,genotype_bytes,query_block)),
        "Actual source/count bounds required")
    _require(s<=2000000 and t>=1 and 1<=p<=20000 and h>=q>0
        and genotype_bytes in (1,2,4,8) and query_block>0,"Actual positive counts/dtype/chunk within original bounds")
    b=min(q,query_block)
    return dict(parameters_per_arm=31428,FP32_parameter_bytes=125712,
        marker_cache_bytes=s*h*genotype_bytes+76*s+76*p+8*q,
        mixed_evidence_count_prefix_bytes=12*(s+1)*b*CLASSES,
        localized_feature_block_bytes=4*p*b*CLASSES*6,
        full_grid_logits_or_probabilities_bytes=4*p*q*CLASSES,
        all96_sealed_probability_bytes=96*4*p*q*CLASSES,
        one_FP32_target_mass_bytes=4*p*q*CLASSES,
        one_transient_FP64_target_mass_bytes=8*p*q*CLASSES,
        exported_H042_state_bytes=4*t*h*64,
        head_checkpoint_selected_state_and_feature_bytes=4*p*q*(64+CLASSES*6),
        original_backbone_autograd_and_allocator_peak_unmeasured=True)
