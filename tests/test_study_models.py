"""Mechanism controls must preserve the intended input symmetries."""
import pytest
import torch
from test_models import example_sequence,clone,split_first_interval
from arggnn.study_models import make_study_model
from arggnn.train_study import grid_sequence_metrics


@pytest.mark.parametrize("name",["feature_mlp","leaf_kernel","span_set_lineage"])
def test_control_gradients_and_interval_subdivision(name):
    torch.manual_seed(29)
    torch.set_num_threads(1)
    model=make_study_model(name,3,3,{"hidden_dim":8,"num_layers":2}).double().eval()
    seq=example_sequence()
    expected=model(seq)
    assert expected.shape==(4,2,3)
    torch.testing.assert_close(model(split_first_interval(seq)),expected,atol=1e-10,rtol=1e-10)
    expected.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.head.weight.grad.abs().sum()>0 if hasattr(model.head,"weight") else model.head[-1].weight.grad.abs().sum()>0


def test_span_set_context_has_no_coordinate_dependence_with_same_local_graph():
    torch.manual_seed(30)
    seq=example_sequence()
    shifted=clone(seq,positions=torch.tensor([.3,.35,.5,.55],dtype=torch.float64))
    settings={"hidden_dim":8,"num_layers":2}
    model=make_study_model("span_set_lineage",3,3,settings).double().eval()
    torch.testing.assert_close(model(seq),model(shifted),atol=1e-10,rtol=1e-10)
    ordered=make_study_model("interval_lineage",3,3,settings).double().eval()
    assert (ordered(seq)-ordered(shifted)).abs().max()>1e-6


def test_grid_boundary_matching_does_not_double_count_or_miss_feasible_pair():
    # True boundaries after rows1,2. Predicted boundaries after rows0,1.
    labels=torch.tensor([[0],[0],[1],[0]])
    prediction=torch.tensor([[0],[1],[0],[0]])
    p=torch.nn.functional.one_hot(prediction,2).float()
    result=grid_sequence_metrics(p,labels)
    assert result["grid_true_switches"]==2
    assert result["grid_predicted_switches"]==2
    assert result["grid_boundary_matches_tolerance_one"]==2
    assert result["grid_boundary_f1"]==1
    one_boundary=torch.nn.functional.one_hot(torch.tensor([[0],[0],[1],[1]]),2).float()
    result=grid_sequence_metrics(one_boundary,labels)
    assert result["grid_boundary_matches_tolerance_one"]==1
    assert result["grid_boundary_recall"]==.5
