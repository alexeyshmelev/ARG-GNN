"""H039 new dense-row CRF class; frozen optimization and verified v2 warmstart."""
from __future__ import annotations
import torch
from .directed_clade_exchange_crf_v2 import DirectedCladeExchangeCRFV2,TRAIN_ARMS
from .directed_clade_exchange_continuation_v2 import load_selected_warmstart_v1
from .directed_clade_exchange_continuation_v1 import (
 COUNTS,NEW_KEYS,TRAIN,VAL,train_shared_parent_v1,probability_v1,run_shared_continuations_v1)
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
SOURCE="src/arggnn/directed_clade_exchange_continuation_v3.py"


def build_family_v1(payload,model_kwargs,*,device,execution_guard):
 """Separate models/optimizers, one shared observation residency at caller."""
 if not callable(execution_guard):raise ValueError("Root execution/resource guard required")
 execution_guard()
 if (payload.get("arm")!="aligned" or payload.get("seed")!=101 or payload.get("teacher_used") is not False):
  raise ValueError("Same selected aligned seed101 model payload required")
 if model_kwargs!=dict(hidden_dim=32,expected_haplotypes=952,site_chunk_size=256,checkpoint_sites=True):
  raise ValueError("Exact H038 FP32 model settings required")
 state=payload["model_state"]
 if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float32 for v in state.values()):
  raise ValueError("Original raw FP32 model state required")
 torch.manual_seed(101)
 family={}
 for arm in TRAIN_ARMS:
  model=MutationRoutedFullPanelPyGGNNV1(**model_kwargs) if arm=="plain_continuation" else DirectedCladeExchangeCRFV2(**model_kwargs)
  if arm=="plain_continuation":model.load_state_dict(state,strict=True)
  else:
   result=model.load_state_dict(state,strict=False)
   if set(result.missing_keys)!=NEW_KEYS or result.unexpected_keys:raise ValueError("Only20 genuinely new head scalars may be missing")
  if sum(p.numel() for p in model.parameters())!=COUNTS[arm]:raise ValueError("Declared active learned capacity mismatch")
  model.to(device=device,dtype=torch.float32);family[arm]=model
 optimizers={arm:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
  for arm,model in family.items()}
 return family,optimizers
