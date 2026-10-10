"""Classical smoother limits and stationary-prior sanity checks."""
import torch
from arggnn.temporal_hmm import markov_smooth


def test_uninformative_classifier_preserves_training_prior():
    prior=torch.tensor([.2,.5,.3],dtype=torch.double)
    p=prior.expand(4,2,3).clone()
    x=torch.tensor([.01,.2,.55,.9],dtype=torch.double)
    for rate in (0.,3.,100.):
        torch.testing.assert_close(markov_smooth(p,x,prior,rate),p,atol=1e-12,rtol=1e-12)


def test_independent_limit_recovers_original_posteriors():
    p=torch.tensor([[[.7,.2,.1]],[[.1,.6,.3]],[[.3,.3,.4]]],dtype=torch.double)
    x=torch.tensor([.1,.5,.9],dtype=torch.double)
    prior=torch.tensor([.2,.5,.3],dtype=torch.double)
    torch.testing.assert_close(markov_smooth(p,x,prior,1e6),p,atol=1e-12,rtol=1e-12)


def test_zero_rate_assigns_one_constant_latent_state_with_finite_outputs():
    p=torch.tensor([[[.7,.2,.1]],[[.1,.6,.3]],[[.3,.3,.4]]],dtype=torch.double)
    x=torch.tensor([.1,.5,.9],dtype=torch.double)
    result=markov_smooth(p,x,torch.tensor([.2,.5,.3]),0.)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result,result[0].expand_as(result),atol=1e-12,rtol=1e-12)
