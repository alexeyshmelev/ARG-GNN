"""H039 C4 row construction fix; no advanced-index/broadcast assignment."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .directed_clade_exchange_crf_v1 import (
 DirectedCladeExchangeCRFV1,CladeExchangeInputsV1,TRAIN_ARMS,CRF_ARMS,
 validate_witness_bank_v1,additive_projection_v1,potential_components_v1,
 endpoint_unary_equivalent_v1,chain_log_marginals_v1)
SOURCE="src/arggnn/directed_clade_exchange_crf_v2.py"


class DirectedCladeExchangeCRFV2(DirectedCladeExchangeCRFV1):
 """Same20 scalars/state keys; explicit dense C4 row assembly replaces index_put."""
 def log_transition_potentials(self,bank):
  validate_witness_bank_v1(bank)
  w=bank.witness_counts.reshape(*bank.witness_counts.shape[:-1],4,3)
  total=w.sum(-1,keepdim=True);denominator=torch.where(total>0,total,torch.ones_like(total))
  direct=w/denominator
  availability=bank.witness_donor_counts.to(w.dtype)
  off=availability[self._off_b].reshape(4,3)
  prior=off/off.sum(-1,keepdim=True)
  generic=(total>0).to(w.dtype)*prior
  delta=.5*(bank.spans[1:]+bank.spans[:-1])
  geometry=torch.log1p(delta/1000.)[:,None].expand_as(bank.turnover)
  feature=torch.stack((geometry,bank.turnover,bank.unresolved),-1)
  exit_score=self.exit_intercept+torch.matmul(feature,self.exit_coefficients)[...,None]
  log_exit=F.logsigmoid(exit_score);log_stay=F.logsigmoid(-exit_score)
  both=torch.stack((generic,direct),dim=0)
  offlog=log_exit[None,...,None]+F.log_softmax(
   self.off_diagonal_bias.reshape(4,3)+self.witness_scale*both,dim=-1)
  # offlog [2,T-1,Q,4,3]; diagonal [2,T-1,Q,4].
  # Explicitly expand the scientific two-potential axis, then insert each
  # source state's single diagonal at its C4 destination. No index_put.
  diagonal=log_stay.unsqueeze(0).expand(2,-1,-1,-1)
  rows=[torch.cat((offlog[...,a,:a],diagonal[...,a,None],offlog[...,a,a:]),dim=-1) for a in range(4)]
  pair=torch.stack(rows,dim=-2)
  expected=(2,bank.spans.numel()-1,686,4,4)
  if pair.shape!=expected:raise ValueError("Explicit complete generic/directed C4 potential shape differs")
  return pair[0],pair[1]
