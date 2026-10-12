"""H039 FP32 PyG backbone plus20-parameter CRF, inactive root-callable API."""
from __future__ import annotations
from dataclasses import dataclass,fields
import argparse,math,json,hashlib
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
SOURCE="src/arggnn/directed_clade_exchange_crf_v1.py"
TRAIN_ARMS=("plain_continuation","generic_unary","directed")
CRF_ARMS=("generic_unary","directed")
OFF_DIAGONAL=tuple((a,b) for a in range(4) for b in range(4) if a!=b)


@dataclass(frozen=True)
class CladeExchangeInputsV1(MutationRoutedWholeTreeInputsV1):
 witness_counts: Tensor             # [T-1,686,12], observed ordered-pair average
 turnover: Tensor                   # [T-1,686], resolved topology turnover
 unresolved: Tensor                 # [T-1,686], unresolved/ancestor fraction
 witness_donor_counts: Tensor       # [4], all266 observed pure proxies

 @classmethod
 def attach(cls,base,arrays,*,record):
  """Attach only a verified observation sidecar; shared old residency supports it."""
  if type(base) is not MutationRoutedWholeTreeInputsV1:raise TypeError("Original observation bank required")
  from .directed_clade_exchange_witness_v1 import ARRAY_FIELDS,_np
  if set(arrays)!=set(ARRAY_FIELDS):raise ValueError("Exact observation-only sidecar fields required")
  meta=record["metadata"]
  for name,key in (("parent","topology_sha256"),("sample_index","sample_index_sha256"),("query_column","query_column_sha256")):
   if hashlib.sha256(_np(getattr(base,name)).tobytes()).hexdigest()!=meta[key]:
    raise ValueError("Sidecar witness/local bank identity differs")
  if meta["native_holds"]!=base.sample_index.shape[0] or meta["queries"]!=686 or meta["truth_or_teacher_inputs"] is not False:
   raise ValueError("Exact complete observation-only sidecar metadata required")
  values={f.name:getattr(base,f.name) for f in fields(MutationRoutedWholeTreeInputsV1)}
  values.update(witness_counts=torch.from_numpy(arrays["witness_counts"]),
   turnover=torch.from_numpy(arrays["turnover"]),unresolved=torch.from_numpy(arrays["unresolved"]),
   witness_donor_counts=torch.from_numpy(arrays["donor_counts"]))
  bank=cls(**values);validate_witness_bank_v1(bank);return bank


def validate_witness_bank_v1(bank):
 if not isinstance(bank,CladeExchangeInputsV1):raise TypeError("Explicit observation-only witness bank required")
 t,h=bank.sample_index.shape;q=bank.query_column.numel()
 if h!=952 or q!=686 or bank.witness_counts.shape!=(t-1,q,12) or bank.turnover.shape!=(t-1,q) or bank.unresolved.shape!=(t-1,q):
  raise ValueError("Complete952/686/native-boundary witness bank required")
 if bank.witness_donor_counts.shape!=(4,) or bank.witness_donor_counts.dtype!=torch.long:
  raise ValueError("Actual observed source availability required")
 for value in (bank.witness_counts,bank.turnover,bank.unresolved):
  if value.dtype!=torch.float32 or value.requires_grad or value.device!=bank.parent.device:
   raise ValueError("Read-only FP32 observed witness fields on the bank device required")
 if bank.witness_donor_counts.device!=bank.parent.device:raise ValueError("Donor availability/device mismatch")
 return bank


def additive_projection_v1(potential):
 """Exact row+column projection; preserves LOG-POTENTIAL marginal sums."""
 if potential.shape[-2:]!=(4,4):raise ValueError("C4 state-pair potential required")
 return potential.mean(-1,keepdim=True)+potential.mean(-2,keepdim=True)-potential.mean((-2,-1),keepdim=True)


def potential_components_v1(generic,directed):
 """B, A, J with E=L-B, J=CEC and L=B+A+J."""
 if generic.shape!=directed.shape:raise ValueError("Same generic/directed potential shape required")
 residual=directed-generic;additive=additive_projection_v1(residual)
 return generic,additive,residual-additive


def additive_all_L_diagnostic_v1(directed):
 """Optional diagnostic only: removes generic persistence too, not a train arm."""
 return additive_projection_v1(directed)


def endpoint_unary_equivalent_v1(additive):
 """Return previous/current endpoint unaries for additive A, not an extra arm."""
 row=additive.mean(-1);column=additive.mean(-2)-additive.mean((-2,-1))[...,None]
 return row,column


def _matrix_normalize(matrix):
 return matrix-matrix.amax((-2,-1),keepdim=True)


def _vector_normalize(vector):
 return vector-torch.logsumexp(vector,-1,keepdim=True)


def _log_matrix_product(left,right):
 # [...,a,k,b], reduce intermediate C4 state k.
 return _matrix_normalize(torch.logsumexp(left[..., :, :,None]+right[...,None,:,:],dim=-2))


def _log_vector_matrix(vector,matrix):
 return _vector_normalize(torch.logsumexp(vector[..., :,None]+matrix,dim=-2))


def _log_matrix_vector(matrix,vector):
 return _vector_normalize(torch.logsumexp(matrix+vector[...,None,:],dim=-1))


def chain_log_marginals_v1(emissions,potential):
 """Work-efficient balanced C4 sum-product, no identity padding or adaptive solver.

 Upsweep composes chronological blocks. Downsweep carries forward/backward
 vectors. Irregular final blocks are carried unchanged, avoiding all-inf
 padding gradients. O(TQ*4^3) work, O(logT) sequential tensor stages.
 """
 if emissions.dtype!=torch.float32 or emissions.ndim!=3 or emissions.shape[-1]!=4:
  raise ValueError("FP32 [T,Q,C4] emissions required")
 t,q,_=emissions.shape
 if t<1 or potential.shape!=(t-1,q,4,4) or potential.dtype!=torch.float32 or potential.device!=emissions.device:
  raise ValueError("One complete C4 chain per native query required")
 if t==1:return emissions.log_softmax(-1)
 leaves=potential+emissions[1:,...,None,:]
 levels=[_matrix_normalize(leaves)]
 while levels[-1].shape[0]>1:
  current=levels[-1];pairs=current.shape[0]//2
  joined=_log_matrix_product(current[:2*pairs:2],current[1:2*pairs:2])
  if current.shape[0]%2:joined=torch.cat((joined,current[-1:]),dim=0)
  levels.append(joined)
 start=_vector_normalize(emissions[0]) # fixed uniform start adds a cancelling constant
 forward=start[None];backward=torch.zeros_like(forward)
 for current in reversed(levels[:-1]):
  pairs=current.shape[0]//2
  fleft=forward[:pairs];fright=_log_vector_matrix(fleft,current[:2*pairs:2])
  bright=backward[:pairs];bleft=_log_matrix_vector(current[1:2*pairs:2],bright)
  f=torch.stack((fleft,fright),dim=1).flatten(0,1)
  b=torch.stack((bleft,bright),dim=1).flatten(0,1)
  if current.shape[0]%2:
   f=torch.cat((f,forward[-1:]),dim=0);b=torch.cat((b,backward[-1:]),dim=0)
  forward,backward=f,b
 after=_log_vector_matrix(forward,levels[0])
 before=_log_matrix_vector(levels[0],backward)
 logits=torch.cat(((start+before[0])[None],after+backward),dim=0)
 return logits-torch.logsumexp(logits,-1,keepdim=True)


class DirectedCladeExchangeCRFV1(MutationRoutedFullPanelPyGGNNV1):
 """Unchanged PyG/mutation/SNP/bidirectional GRU body; exactly20 new scalars."""
 def __init__(self,hidden_dim=32,*,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True):
  if hidden_dim!=32 or expected_haplotypes!=952:raise ValueError("Frozen H038 width32/full952 backbone required")
  super().__init__(hidden_dim=hidden_dim,expected_haplotypes=expected_haplotypes,
   site_chunk_size=site_chunk_size,checkpoint_sites=checkpoint_sites)
  self.exit_intercept=nn.Parameter(torch.full((4,),math.log(3.)))
  self.exit_coefficients=nn.Parameter(torch.zeros(3))
  self.off_diagonal_bias=nn.Parameter(torch.zeros(12))
  self.witness_scale=nn.Parameter(torch.zeros(()))
  self.register_buffer("_off_a",torch.tensor([a for a,b in OFF_DIAGONAL],dtype=torch.long),persistent=False)
  self.register_buffer("_off_b",torch.tensor([b for a,b in OFF_DIAGONAL],dtype=torch.long),persistent=False)
  self.register_buffer("_diag",torch.arange(4,dtype=torch.long),persistent=False)

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
  offlog=log_exit[None,...,None]+F.log_softmax(self.off_diagonal_bias.reshape(4,3)+self.witness_scale*both,dim=-1)
  pair=w.new_zeros((2,*w.shape[:-2],4,4))
  pair[...,self._off_a,self._off_b]=offlog.flatten(-2)
  pair[...,self._diag,self._diag]=log_stay
  return pair[0],pair[1]

 def forward_holds(self,bank,*,control="directed"):
  if control not in CRF_ARMS:raise ValueError("Only registered generic_unary/directed train arms")
  raw=super().forward_holds(bank,control="aligned") # all952 tree and GRU states retained
  emission=raw.index_select(1,bank.query_column)
  generic,directed=self.log_transition_potentials(bank)
  if control=="generic_unary":
   base,additive,_=potential_components_v1(generic,directed);potential=base+additive
  else:potential=directed
  return chain_log_marginals_v1(emission,potential)

 def forward_train_holds(self,bank,*,control="directed"):
  return self.forward_holds(bank,control=control)

 def forward(self,bank,*,control="directed"):
  return self.forward_holds(bank,control=control).index_select(0,bank.readout_hold)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.parse_args()
 print(json.dumps(dict(status="source_only_H039_model",source=SOURCE,arms=list(TRAIN_ARMS),
  plain_parameters=29764,CRF_parameters=29784,additional_parameters=20,admission=False)))
if __name__=="__main__":main()
