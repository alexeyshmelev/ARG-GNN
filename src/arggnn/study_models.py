"""Common-input controls for the next study, with explicit adaptations."""
from __future__ import annotations
import math
import torch
from torch import nn
from .models import (_BaseModel, build_model, query_node_index,
                     log_interval_kernel_mass, validate_model_input)


class QueryFeatureMLP(nn.Module):
    """Train on all observed summaries at the query node without graph edges."""
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2, validate=True):
        super().__init__()
        layers=[]
        width=input_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(width,hidden_dim),nn.GELU()))
            width=hidden_dim
        layers.append(nn.Linear(width,num_classes))
        self.head=nn.Sequential(*layers)
        self.validate_inputs=validate

    def forward(self,seq):
        if self.validate_inputs:
            validate_model_input(seq)
        indexes=query_node_index(seq)
        return self.head(seq.x[indexes[seq.target_graph]])


class LeafKernelGNN(_BaseModel):
    """Direct exact exponential smoothing control, no cross-node attention."""
    def __init__(self,*args,initial_tau=.05,**kwargs):
        super().__init__(*args,**kwargs)
        if initial_tau <= 1e-4:
            raise ValueError("initial_tau must exceed 1e-4")
        adjusted=initial_tau-1e-4
        self.raw_tau=nn.Parameter(torch.tensor(adjusted+math.log(-math.expm1(-adjusted))))
        self.head=nn.Linear(2*self.hidden_dim,self.num_classes)

    def forward(self,seq):
        banks,indexes=self.prepare(seq)
        leaf=banks[-1][indexes]
        tau=torch.nn.functional.softplus(self.raw_tau)+1e-4
        weights=log_interval_kernel_mass(seq.positions,seq.intervals,tau.to(seq.intervals.dtype)).softmax(-1).to(leaf.dtype)
        context=torch.einsum("pg,gqh->pqh",weights,leaf)
        local=leaf[seq.target_graph]
        return self.head(torch.cat((local,context),dim=-1))


def make_study_model(name,input_dim,num_classes,settings):
    common=dict(input_dim=input_dim,num_classes=num_classes,hidden_dim=settings["hidden_dim"],
                num_layers=settings["num_layers"],validate=False)
    if name.startswith("gst_"):
        from .gst_baseline import GSTAdaptation
        parts=name.split("_")
        if len(parts)!=3 or parts[1] not in {"ca","bc"} or parts[2] not in {"record","span"}:
            raise ValueError("Unknown GST adaptation")
        return GSTAdaptation(**common,routing="cross_attention" if parts[1]=="ca" else "broadcast",
                             token_weighting=parts[2],gate_variant="paper",dropout=settings.get("dropout",0.))
    if name.startswith("semigroup_"):
        from .semigroup_model import SemigroupGraphModel
        modes={"semigroup_full":("full","ordered"),"semigroup_diagonal":("diagonal","ordered"),
               "semigroup_set":("full","homogeneous_set")}
        if name not in modes:
            raise ValueError("Unknown semigroup control")
        kind,aggregation=modes[name]
        return SemigroupGraphModel(**common,matrix_kind=kind,aggregation=aggregation,
                                   latent_dim=8,generator_rank=2,initial_rate=20.)
    if name=="feature_mlp":
        return QueryFeatureMLP(**common)
    if name=="leaf_kernel":
        return LeafKernelGNN(**common)
    if name in {"late_transformer","supra_mpnn"}:
        from .temporal_baselines import LeafTemporalTransformer,LeafCorrespondenceSupraMPNN
        cls=LeafTemporalTransformer if name=="late_transformer" else LeafCorrespondenceSupraMPNN
        return cls(**common,dropout=settings.get("dropout",0.))
    if name in {"graphssm","graphssm_zoh"}:
        from .graphssm_baseline import GraphSSMAdaptation
        return GraphSSMAdaptation(**common,bidirectional=True,
                                 discretization="paper" if name=="graphssm" else "zoh",dropout=settings.get("dropout",0.))
    return build_model(name,**common)


def make_regression_model(name,input_dim,settings):
    """Replace the complete legacy classification head with one scalar output.

    Encoders retain their documented architectures. The temporary two-class
    head is discarded rather than leaving an unused output parameter in loss.
    """
    base=name.removesuffix("_no_geography")
    model=make_study_model(base,input_dim,2,settings)
    if isinstance(model.head,nn.Linear):
        model.head=nn.Linear(model.head.in_features,1)
    elif isinstance(model.head,nn.Sequential) and isinstance(model.head[-1],nn.Linear):
        model.head[-1]=nn.Linear(model.head[-1].in_features,1)
    else:
        raise TypeError("Unrecognized forecast-head architecture")
    model.num_classes=1
    if hasattr(model,"num_outputs"):
        model.num_outputs=1
    return model
