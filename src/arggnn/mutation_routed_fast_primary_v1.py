"""FP32 Present64 selection scoring on unchanged sealed physical-bin support.

Only NLL and accuracy are computed during selection. Full original diagnostics
remain a separately invoked final-selected check. No model/loss access to VAL
mass; target caches are process-local and created only after both VAL seals.
"""
from __future__ import annotations
from dataclasses import dataclass
import io
import numpy as np

SEAL_FIELDS = {"C4_probabilities","positions_bp","bin_edges_bp",
               "retained_fixed100bp_indexes","query_sample_indexes"}


@dataclass(frozen=True)
class PrimaryMassCacheV1:
    mass: np.ndarray                    # [R,64,4], FP32 score-only
    columns: np.ndarray                 # original query-output column indexes
    edges: np.ndarray                   # exact original physical edges
    denominator_query_bp: float         # original geometry, metadata only
    parent_id: str
    target_provenance: dict


def mass_cache_v1(parent, all_query_mass, *, target_provenance,
                  expected_present_queries=64):
    """Convert the unchanged external helper's exact class-span mass once."""
    columns=np.flatnonzero(np.asarray(parent["cohorts"])=="present_day")
    if len(columns)!=expected_present_queries or not len(columns):
        raise ValueError("Original complete Present64 query stratum required")
    mass=np.asarray(all_query_mass)
    if (mass.shape!=(len(parent["positions"]),len(parent["query_slots"]),4)
            or not np.isfinite(mass).all() or np.any(mass<0)):
        raise ValueError("Complete original class-span target mass required")
    selected=mass[:,columns]
    widths=np.diff(np.asarray(parent["edges"],dtype=np.float64))
    if (np.any(widths<=0) or not np.allclose(selected.sum(-1),widths[:,None],rtol=0,atol=1e-8)):
        raise ValueError("Each query/bin must retain exactly its original physical span")
    mass32=np.array(selected,dtype=np.float32,copy=True,order="C")
    if np.any((selected>0)&(mass32==0)):
        raise ValueError("Positive physical target mass cannot be silently dropped by FP32 conversion")
    mass32.setflags(write=False)
    columns=columns.copy();columns.setflags(write=False)
    edges=np.array(parent["edges"],dtype=np.float64,copy=True);edges.setflags(write=False)
    return PrimaryMassCacheV1(mass32,columns,edges,
        float((edges[-1]-edges[0])*len(columns)),parent["parent_id"],dict(target_provenance))


def verify_seal_v1(record,parent,reader):
    """Read verified probability bytes and preserve exact original support."""
    data=reader.read(record)
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        if set(archive.files)!=SEAL_FIELDS:
            raise ValueError("Original H036 probability-only seal schema required")
        checks={"positions_bp":parent["positions"],"bin_edges_bp":parent["edges"],
                "retained_fixed100bp_indexes":parent["retained_indexes"],
                "query_sample_indexes":parent["query_slots"]}
        if any(not np.array_equal(archive[k],v) for k,v in checks.items()):
            raise ValueError("Sealed original query/physical support changed")
        p=archive["C4_probabilities"].copy()
    if (p.dtype!=np.float32 or p.shape!=(len(parent["positions"]),len(parent["query_slots"]),4)
            or not np.isfinite(p).all() or np.any(p<0)
            or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5)):
        raise ValueError("Complete finite raw FP32 probabilities required; no floors/renormalization")
    return p


def score_primary_v1(probabilities,cache,*,backend="numpy",device=None,torch_mass=None):
    """Vectorized FP32 math only; zeros with positive true mass imply infinite NLL."""
    p=np.asarray(probabilities)
    if (p.dtype!=np.float32 or p.ndim!=3 or p.shape[0]!=cache.mass.shape[0]
            or p.shape[-1]!=4 or np.any(cache.columns>=p.shape[1])
            or not np.isfinite(p).all() or np.any(p<0)
            or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5)):
        raise ValueError("Raw complete FP32 probabilities required")
    primary=np.ascontiguousarray(p[:,cache.columns])
    denominator=np.float32(cache.denominator_query_bp)
    if denominator<=0 or not np.isfinite(denominator):
        raise ValueError("Positive original represented query-bp denominator required")
    if backend=="numpy":
        mass=cache.mass
        zero=(primary==0)&(mass>0)
        infinite_bp=float(np.sum(np.where(zero,mass,np.float32(0)),dtype=np.float32))
        # Replacing a zero by1 only prevents log(0) in the finite summand.
        # Every zero with true mass is separately retained as infinite NLL.
        logs=np.log(np.where(primary>0,primary,np.float32(1)))
        finite_nll=np.sum(-mass*logs,dtype=np.float32)/denominator
        predicted=primary.argmax(-1)  # original fixed class-order first tie
        correct=np.take_along_axis(mass,predicted[...,None],axis=-1)[...,0]
        accuracy=float(np.sum(correct,dtype=np.float32)/denominator)
    elif backend=="torch":
        import torch
        if device is None:raise ValueError("Explicit Torch score device required")
        value=torch.as_tensor(primary,dtype=torch.float32,device=device)
        mass=(torch.tensor(cache.mass,dtype=torch.float32,device=device)
              if torch_mass is None else torch_mass)
        if mass.dtype!=torch.float32 or tuple(mass.shape)!=tuple(cache.mass.shape) or mass.device!=value.device:
            raise ValueError("Explicit cached FP32 score-only mass/device must agree")
        with torch.no_grad():
            zero=(value==0)&(mass>0)
            infinite_bp=float(torch.where(zero,mass,torch.zeros_like(mass)).sum().cpu())
            logs=torch.where(value>0,value,torch.ones_like(value)).log()
            den=torch.tensor(denominator,dtype=torch.float32,device=device)
            finite_nll=(torch.sum(-mass*logs)/den).cpu().item()
            predicted=value.argmax(-1)
            accuracy=(mass.gather(-1,predicted[...,None])[...,0].sum()/den).cpu().item()
    else:raise ValueError("Backend must be numpy or torch")
    return dict(accuracy=accuracy,nll=None if infinite_bp else float(finite_nll),
        NLL_is_infinite=bool(infinite_bp),infinite_nll_query_bp=infinite_bp,
        denominator_query_bp=cache.denominator_query_bp,query_count=len(cache.columns),
        represented_bin_support_bp=[float(cache.edges[0]),float(cache.edges[-1])],
        probability_floor_used=False,posthoc_renormalization=False,
        score_arithmetic="float32",backend=backend,selection_metrics_only=True)


class FastPrimaryValidationV1:
    """Small score-only state for two sealed VAL parents, separate from observation banks."""
    def __init__(self,parents,bindings,reader,helpers,cfg,*,backend="numpy",device=None):
        self.parents={p:row for p,row in parents.items() if row["split"]=="val"}
        if len(self.parents)!=2:
            raise ValueError("Both original VAL parents required")
        self.bindings,self.reader,self.helpers,self.cfg=bindings,reader,helpers,cfg
        if backend not in ("numpy","torch") or (backend=="torch" and device is None):
            raise ValueError("Explicit valid score backend/device required")
        self.backend,self.device=backend,device
        self.target_caches={};self.torch_masses={};self.full_diagnostics_calls=0

    def score(self,seals):
        """Seal-verifies BOTH parents before first VAL truth access, then caches mass."""
        records=dict(seals)
        if len(seals)!=2 or set(records)!=set(self.parents):
            raise ValueError("Both current VAL probability seals required")
        # Complete this entire verification before constructing either target.
        probabilities={p:verify_seal_v1(records[p],parent,self.reader)
                       for p,parent in self.parents.items()}
        from . import census_tsinfer_fit_v1 as fit
        for parent_id,parent in self.parents.items():
            if parent_id not in self.target_caches:
                target=fit.ExternalTargetV1(parent,self.bindings["parents"][parent_id],
                    "VAL_after_sealed_prediction",self.reader,records[parent_id])
                full_mass=target.target_mass(parent,self.helpers)
                cache=mass_cache_v1(parent,full_mass,target_provenance=target.provenance)
                self.target_caches[parent_id]=cache
                if self.backend=="torch":
                    import torch
                    self.torch_masses[parent_id]=torch.tensor(cache.mass,dtype=torch.float32,device=self.device)
                del target,full_mass
        cells=[]
        for parent_id,parent in self.parents.items():
            group=score_primary_v1(probabilities[parent_id],self.target_caches[parent_id],
                backend=self.backend,device=self.device,torch_mass=self.torch_masses.get(parent_id))
            cells.append(dict(parent_id=parent_id,split="val",prediction_seal=records[parent_id],
                target_provenance=self.target_caches[parent_id].target_provenance,
                groups={"primary_present_day_64":group},development_only=True,TEST_parents=0))
        groups=[c["groups"]["primary_present_day_64"] for c in cells]
        infinite=any(g["NLL_is_infinite"] for g in groups)
        means={"primary_present_day_64":{
            "accuracy":float(np.mean(np.array([g["accuracy"] for g in groups],dtype=np.float32),dtype=np.float32)),
            "nll":None if infinite else float(np.mean(np.array([g["nll"] for g in groups],dtype=np.float32),dtype=np.float32)),
            "NLL_is_infinite":infinite}}
        return cells,means

    def full_selected_diagnostics(self,seals):
        """Unchanged complete diagnostics exactly once for final selected checkpoint."""
        if self.full_diagnostics_calls:
            raise ValueError("Full selected diagnostics already computed in this worker")
        records=dict(seals)
        if len(seals)!=2 or set(records)!=set(self.parents):
            raise ValueError("Both selected VAL seals required for final diagnostics")
        for parent_id,parent in self.parents.items():
            verify_seal_v1(records[parent_id],parent,self.reader)
        from . import train_persistence_trace_pilot_v1 as h025
        result=h025.score_v1(seals,self.parents,self.bindings,self.reader,self.helpers,self.cfg)
        self.full_diagnostics_calls+=1
        return result

    def receipt(self):
        return dict(primary_target_cache_parent_ids=list(self.target_caches),
            target_mass_cached_only_after_both_VAL_seals=True,
            target_mass_model_input=False,target_mass_VAL_loss=False,
            primary_arithmetic="float32",backend=self.backend,
            full_selected_diagnostics_calls=self.full_diagnostics_calls,
            primary_mass_cache_bytes=sum(c.mass.nbytes for c in self.target_caches.values()))
