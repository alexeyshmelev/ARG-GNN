"""Classical spatial Markov smoothing control, fitted only on train/validation."""
from __future__ import annotations
import math
import torch


def markov_smooth(probabilities,positions,class_prior,rate,temperature=1.):
    """Forward/backward smoothing on declared fixed scoring coordinates.

    Transition P(d)=exp(-rate*d)I+(1-exp(-rate*d))*1*prior.
    This reset process has stationary prior and an exact continuous-time
    transition semigroup. Classifier posteriors are divided by training prior
    to form emissions. This is a baseline on fixed samples, not a novel kernel
    or an interval-refinement claim for the entire observation model.
    """
    p=probabilities.detach().cpu().double()
    x=positions.detach().cpu().double()
    prior=class_prior.detach().cpu().double()
    if p.ndim!=3 or x.shape!=(p.shape[0],) or prior.shape!=(p.shape[-1],):
        raise ValueError("Expected [P,Q,K], [P] and [K]")
    if not math.isfinite(rate) or rate<0 or not math.isfinite(temperature) or temperature<=0:
        raise ValueError("Rate must be nonnegative and temperature positive")
    if not torch.isfinite(p).all() or (p<0).any() or not torch.isfinite(x).all():
        raise ValueError("Finite nonnegative probabilities and finite coordinates required")
    if not (x[1:]>x[:-1]).all() or not (prior>0).all():
        raise ValueError("Coordinates must be strictly increasing and prior positive")
    if not torch.allclose(p.sum(-1),torch.ones_like(p[...,0]),atol=1e-5):
        raise ValueError("Input must be normalized posterior probabilities")
    prior=prior/prior.sum()
    k=p.shape[-1]
    emission=(p.clamp_min(1e-12).log()-prior.log())/temperature
    transitions=[]
    for delta in x[1:]-x[:-1]:
        survival=torch.exp(-rate*delta)
        transitions.append((survival*torch.eye(k,dtype=torch.double)+(1-survival)*prior[None,:]).log())
    forward=[emission[0]+prior.log()]
    forward[0]=forward[0]-torch.logsumexp(forward[0],dim=-1,keepdim=True)
    for t,transition in enumerate(transitions,1):
        alpha=emission[t]+torch.logsumexp(forward[-1][:,:,None]+transition[None,:,:],dim=1)
        forward.append(alpha-torch.logsumexp(alpha,dim=-1,keepdim=True))
    backward=[torch.zeros_like(emission[-1])]
    for t in range(len(transitions)-1,-1,-1):
        beta=torch.logsumexp(transitions[t][None,:,:]+emission[t+1][:,None,:]+backward[-1][:,None,:],dim=-1)
        backward.append(beta-torch.logsumexp(beta,dim=-1,keepdim=True))
    logp=torch.stack(forward)+torch.stack(list(reversed(backward)))
    return logp.softmax(-1).to(probabilities.dtype)
