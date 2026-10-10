"""A regression adaptation must expose only its trained scalar output."""
import numpy as np
import pytest
import torch
from arggnn.chickenpox import ChickenpoxData,TrainingScaler,forecast_sequence
from arggnn.study_models import make_regression_model


@pytest.mark.parametrize("name",["feature_mlp","independent","set_pool","leaf_kernel","gru",
    "late_transformer","graphssm","span_set_lineage","interval_lineage",
    "gst_ca_record","gst_ca_span","semigroup_full","semigroup_diagonal","semigroup_set"])
def test_scalar_regression_shape_and_finite_encoder_gradients(name):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    counts=np.arange(24,dtype=float).reshape(8,3)%11
    data=ChickenpoxData(counts,tuple(str(i) for i in range(8)),("a","b","c"),
                       np.array([[0,1,1,2],[1,0,2,1]]),"synthetic")
    scaler=TrainingScaler.fit(counts,5)
    seq=forecast_sequence(data,scaler,5,history=4)
    model=make_regression_model(name,1,{"hidden_dim":8,"num_layers":2})
    prediction=model(seq)
    assert prediction.shape==(1,3,1)
    assert model.num_classes==1
    torch.nn.functional.mse_loss(prediction[0,:,0],torch.tensor([.1,.3,.7])).backward()
    head=model.head if isinstance(model.head,torch.nn.Linear) else model.head[-1]
    assert head.out_features==1
    assert head.weight.grad.abs().sum()>0
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
