import pytest
import torch
from arggnn.metrics import classification_metrics,replicate_summary

def test_probabilistic_metrics_known_values():
    p=torch.tensor([[.8,.2],[.1,.9]],dtype=torch.double)
    y=torch.tensor([0,1])
    m=classification_metrics(p,y)
    assert m["accuracy_uniform_grid"]==1
    assert m["brier_uniform_grid"]==pytest.approx(.05)
    import math
    assert m["nll_uniform_grid"]==pytest.approx(-(math.log(.8)+math.log(.9))/2)
    assert m["confusion"]==[[1,0],[0,1]]

def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        classification_metrics(torch.tensor([[.3,.4]]),torch.tensor([0]))

def test_parent_replicate_summary_counts_replicates():
    r=replicate_summary([{"accuracy_uniform_grid":.8},{"accuracy_uniform_grid":1.0}],seed=4)
    assert r["independent_replicates"]==2
    assert r["mean_accuracy"]==pytest.approx(.9)
    assert r["bootstrap_unit"]=="independent simulated parent chromosome"
