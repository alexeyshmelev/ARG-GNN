"""H047 loss-only continuation helpers. No new learned module or parameter."""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F
from .anonymous_clade_flag_control_v1 import AnonymousCladeFlagControlGNNV1
from .biological_gst_control_v1 import exact_span_loss

ARMS=("original_risk","balanced_risk")
MODEL=dict(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True)
PARAMETERS=30788
COHORT_COUNTS={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26}
OBJECTIVES={"original_risk":"(182/686)*LP+(504/686)*LA",
            "balanced_risk":"0.5*LP+0.5*LA"}


def require(ok,message):
    if not ok:raise ValueError(message)


def cohort_columns_v1(cohorts):
    value=np.asarray(cohorts)
    require(value.shape==(686,) and value.dtype.kind in "US"
        and set(value.tolist())==set(COHORT_COUNTS)
        and all(int((value==name).sum())==count for name,count in COHORT_COUNTS.items()),
        "Original complete Present182/ancient504 observed cohort metadata")
    present=np.flatnonzero(value=="present_day").astype(np.int64)
    ancient=np.flatnonzero(value!="present_day").astype(np.int64)
    return present,ancient


def cohort_objective_v1(logits,mass,present_columns,*,arm):
    """Exact original risk or fixed equal-group risk, all in learned FP32.

    Mass is original physical class-span mass, optionally aggregated into native
    holds when logits are constant there. This aggregation remains loss/gradient
    equivalent to the complete original grid. No query or reference is removed.
    """
    require(arm in ARMS and logits.dtype==mass.dtype==torch.float32
        and logits.ndim==3 and logits.shape==mass.shape and logits.shape[-1]==4
        and logits.device==mass.device and bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(mass).all()) and bool((mass>=0).all()),
        "Complete same-device finite FP32 logits and external class-span mass")
    raw_columns=torch.as_tensor(present_columns)
    require(raw_columns.dtype in (torch.int8,torch.int16,torch.int32,torch.int64),
        "Observed cohort columns must be integer indexes")
    columns=raw_columns.to(dtype=torch.long,device=logits.device)
    q=logits.shape[1]
    require(columns.ndim==1 and 0<len(columns)<q and columns.unique().numel()==len(columns)
        and bool(((columns>=0)&(columns<q)).all()),"Unique nonempty observed present subset")
    present=torch.zeros(q,dtype=torch.bool,device=logits.device);present[columns]=True
    query_mass=mass.sum((0,2))
    require(bool((query_mass>0).all()),"Every query has positive represented physical span")
    logp=F.log_softmax(logits,dim=-1)
    query_nll=(-mass*logp).sum((0,2))
    lp=query_nll[present].sum()/query_mass[present].sum()
    la=query_nll[~present].sum()/query_mass[~present].sum()
    # The control calls the exact existing loss, preserving its FP32 reduction.
    original=exact_span_loss(logits,mass)
    loss=original if arm=="original_risk" else .5*lp+.5*la
    require(loss.dtype==lp.dtype==la.dtype==torch.float32 and bool(torch.isfinite(loss)),
        "Finite FP32 cohort objective")
    return loss,dict(original_all_query_NLL=original,Present_NLL=lp,ancient_NLL=la,
        actual_present_query_bp_weight=query_mass[present].sum()/query_mass.sum(),
        actual_ancient_query_bp_weight=query_mass[~present].sum()/query_mass.sum())


def make_matched_family_v1(payload,**model_kwargs):
    """Caller-supplied H042 pass30 state only; no checkpoint path/file reader."""
    require(model_kwargs==MODEL,"Unchanged complete H042 model configuration")
    require(payload.get("schema")=="anonymous_clade_flag_checkpoint_v1"
        and payload.get("hypothesis_id")=="H042" and payload.get("arm")=="exact_clade_flags"
        and payload.get("phase")=="train" and payload.get("pass_index")==30
        and payload.get("additional_optimizer_steps")==960 and payload.get("parameter_count")==PARAMETERS
        and payload.get("model_kwargs")==MODEL and payload.get("teacher_used") is False,
        "Actual original H042 pass30 checkpoint provenance")
    state=payload["model_state"]
    require(all(isinstance(v,torch.Tensor) and v.device.type=="cpu" and v.dtype==torch.float32
        and bool(torch.isfinite(v).all()) for v in state.values()),"Complete finite CPU FP32 H042 state")
    torch.manual_seed(101)
    family={}
    for arm in ARMS:
        model=AnonymousCladeFlagControlGNNV1(**MODEL)
        model.load_state_dict(state,strict=True)
        model.requires_grad_(True)
        require(sum(p.numel() for p in model.parameters())==PARAMETERS
            and all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters()),
            "All unchanged H042 parameters trainable; zero added parameters")
        family[arm]=model
    pointers=[{p.data_ptr() for p in family[a].parameters()} for a in ARMS]
    require(not pointers[0]&pointers[1],"Independent model storage")
    require(all(torch.equal(family[ARMS[0]].state_dict()[k],family[ARMS[1]].state_dict()[k])
        for k in state),"Identical complete initial state")
    return family,dict(parameters_per_arm=PARAMETERS,added_parameters=0,
        complete_H042_body_trainable=True,reference_inputs_unchanged=True,
        identical_initial_state=True,independent_model_storage=True,
        optimizer_state_inherited=False,checkpoint_provenance_checked=True)
