"""H039 root-callable shared-bank continuation APIs. No admission or launches."""
from __future__ import annotations
import argparse,copy,io,json,random,time
import torch
from .directed_clade_exchange_crf_v1 import DirectedCladeExchangeCRFV1,TRAIN_ARMS,CRF_ARMS,validate_witness_bank_v1
from .directed_clade_exchange_witness_v1 import PARENTS
from .mutation_routed_full_panel_pyg_model_v1 import MutationRoutedFullPanelPyGGNNV1
from .biological_gst_control_v1 import exact_span_loss
from .mutation_routed_full_panel_primary_v1 import PRIMARY
SOURCE="src/arggnn/directed_clade_exchange_continuation_v1.py"
COUNTS=dict(plain_continuation=29764,generic_unary=29784,directed=29784)
NEW_KEYS={"exit_intercept","exit_coefficients","off_diagonal_bias","witness_scale"}
TRAIN=PARENTS[:32];VAL=PARENTS[32:]


def load_selected_warmstart_v1(completed_ref,reader):
 """Only the actual completed aligned H038 selected checkpoint, never predictions."""
 completed=json.loads(reader.read(completed_ref))
 if (completed.get("status")!="completed_H038_seed101_accuracy_selected_full_parent_run"
  or completed.get("arm")!="aligned" or completed.get("seed")!=101 or completed.get("outer_truth_read") is not False):
  raise ValueError("Completed aligned H038/seed101/no-outer warm start required")
 selected=completed["selected"];checkpoint=selected["checkpoint"]
 if checkpoint not in completed["retained_records"]:raise ValueError("Retained actual selected checkpoint required")
 payload=torch.load(io.BytesIO(reader.read(checkpoint)),map_location="cpu",weights_only=False)
 if (payload.get("arm")!="aligned" or payload.get("seed")!=101 or payload.get("teacher_used") is not False
  or payload.get("pass_index")!=selected["pass_index"] or payload.get("optimizer_steps")!=selected["optimizer_steps"]):
  raise ValueError("Selected checkpoint/training provenance mismatch")
 return payload,dict(completed_ref=completed_ref,checkpoint_ref=checkpoint,
  original_selected_pass=payload["pass_index"],original_optimizer_steps=payload["optimizer_steps"],
  optimizer_state_policy="reset_AdamW_moments_and_step_for_all_three_arms",
  teacher_predictions_opened=False)


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
  model=MutationRoutedFullPanelPyGGNNV1(**model_kwargs) if arm=="plain_continuation" else DirectedCladeExchangeCRFV1(**model_kwargs)
  if arm=="plain_continuation":model.load_state_dict(state,strict=True)
  else:
   result=model.load_state_dict(state,strict=False)
   if set(result.missing_keys)!=NEW_KEYS or result.unexpected_keys:raise ValueError("Only20 genuinely new head scalars may be missing")
  if sum(p.numel() for p in model.parameters())!=COUNTS[arm]:raise ValueError("Declared active learned capacity mismatch")
  model.to(device=device,dtype=torch.float32);family[arm]=model
 optimizers={arm:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
  for arm,model in family.items()}
 return family,optimizers


def train_shared_parent_v1(family,optimizers,bank,target_mass,*,execution_guard):
 """One same-parent update per arm, sequential graphs and no bank mutation."""
 if set(family)!=set(TRAIN_ARMS) or set(optimizers)!=set(TRAIN_ARMS):raise ValueError("All three registered continuation arms required")
 validate_witness_bank_v1(bank)
 if target_mass.shape!=(bank.spans.numel(),686,4) or target_mass.dtype!=torch.float32 or target_mass.requires_grad:
  raise ValueError("External exact native-hold TRAIN class-span mass required")
 records={}
 for arm in TRAIN_ARMS:
  execution_guard();model=family[arm];optimizer=optimizers[arm]
  model.train();optimizer.zero_grad(set_to_none=True)
  device=next(model.parameters()).device
  if device.type=="cuda":torch.cuda.synchronize(device)
  started=time.perf_counter()
  logits=model.forward_train_holds(bank,control="aligned" if arm=="plain_continuation" else arm)
  loss=exact_span_loss(logits,target_mass)
  if not bool(torch.isfinite(loss)):raise ValueError("Nonfinite continuation loss")
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
  if not bool(torch.isfinite(torch.cat([p.detach().reshape(-1) for p in model.parameters()])).all()):
   raise ValueError("Nonfinite continuation parameter")
  if device.type=="cuda":torch.cuda.synchronize(device)
  records[arm]=dict(loss=float(loss.detach()),update_seconds=time.perf_counter()-started)
  del logits,loss
  optimizer.zero_grad(set_to_none=True)
 return records


def probability_v1(model,bank,arm):
 if arm not in TRAIN_ARMS:raise ValueError("Registered arm required")
 model.eval()
 with torch.no_grad():logits=model(bank,control="aligned" if arm=="plain_continuation" else arm);probability=logits.softmax(-1)
 return probability


def run_shared_continuations_v1(config,family,optimizers,residency,*,execution_guard,seal,score,record,checkpoint):
 """Root-integrated schedule. Callbacks must be bound original storage/scorer APIs.

 Seal all24 VAL predictions before any score callback. Same fixed updates
 for every arm; no arm-specific early stopping. Callbacks own authoritative
 output pruning, actual source/runtime/lease admission and target boundary.
 """
 if (config.get("execution_guard_ready") is not True or config["arms"]!=list(TRAIN_ARMS)
  or config["seed"]!=101 or config["continuation_passes"]!=10 or config["evaluation_passes"]!=[5,10]
  or set(residency.banks)!=set(PARENTS) or set(residency.targets)!=set(TRAIN)):
  raise ValueError("Root-activated exact three-arm32TRAIN/8VAL/320-update continuation protocol required")
 if any(not callable(f) for f in (execution_guard,seal,score,record,checkpoint)):raise ValueError("Root-bound callbacks required")
 steps={arm:0 for arm in TRAIN_ARMS};history={arm:[] for arm in TRAIN_ARMS};selected={arm:None for arm in TRAIN_ARMS}
 best={arm:(float("inf"),float("inf"),float("inf")) for arm in TRAIN_ARMS}
 for pass_index in range(1,11):
  order=list(TRAIN);random.Random(101+pass_index).shuffle(order)
  summaries={arm:dict(loss=0.,model_seconds=0.) for arm in TRAIN_ARMS}
  pass_started=time.monotonic()
  for at,parent in enumerate(order):
   execution_guard();bank,mass=residency.get(parent)
   if at+1<len(order):residency.prefetch(order[at+1],protected=(parent,))
   rows=train_shared_parent_v1(family,optimizers,bank,mass,execution_guard=execution_guard)
   for arm,row in rows.items():
    steps[arm]+=1;summaries[arm]["loss"]+=row["loss"]/32;summaries[arm]["model_seconds"]+=row["update_seconds"]
   if (at+1)%8==0:
    print(json.dumps(dict(stage="H039_training_progress",pass_index=pass_index,parents_in_pass=at+1,
     steps=steps.copy(),elapsed_seconds=time.monotonic()-pass_started)),flush=True)
  if len(set(steps.values()))!=1:raise ValueError("Unequal optimization exposure")
  record("TRAIN_pass_%02d"%pass_index,dict(pass_index=pass_index,steps=steps.copy(),arms=summaries))
  if pass_index not in (5,10):continue
  seals={arm:[] for arm in TRAIN_ARMS}
  for parent in VAL:
   bank,_=residency.get(parent)
   for arm in TRAIN_ARMS:
    execution_guard();started=time.perf_counter();probability=probability_v1(family[arm],bank,arm)
    ref=seal(parent,arm,pass_index,probability)
    seals[arm].append((parent,ref));del probability
  record("pass_%02d_all24_VAL_sealed"%pass_index,dict(pass_index=pass_index,seals=seals,all_predictions_sealed_before_score=True))
  for arm in TRAIN_ARMS:
   cells,means=score(seals[arm])
   primary=means[PRIMARY];nll=float("inf") if primary["NLL_is_infinite"] else primary["nll"]
   key=(-primary["accuracy"],nll,pass_index)
   cp=checkpoint(arm,pass_index,family[arm],optimizers[arm],steps[arm])
   row=dict(arm=arm,pass_index=pass_index,optimizer_steps=steps[arm],checkpoint=cp,cells=cells,equal_parent_metrics=means)
   history[arm].append(row)
   previous=selected[arm]
   improved=key<best[arm]
   if improved:best[arm]=key;selected[arm]=row
   row["selected"]=improved
   row["prediction_seals"]=seals[arm]
   record("%s_pass_%02d_selection"%(arm,pass_index),dict(row=row,previous_selected=previous,
    selected=selected[arm],discard_current=not improved))
 if any(value!=320 for value in steps.values()):raise ValueError("Exact320 additional full-parent updates per arm required")
 return dict(hypothesis_id="H039",arms=list(TRAIN_ARMS),steps=steps,history=history,selected=selected,
  equal_extra_optimization_exposure=True,shared_read_only_residency=True,optimizer_state_reset_for_all=True,
  outer_evaluated=False,novelty_claim_established=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.parse_args()
 print(json.dumps(dict(status="source_only_root_callable_H039_continuation",arms=list(TRAIN_ARMS),counts=COUNTS,
  GPU_processes_proposed=1,admitted=False,fit_performed=False)))
if __name__=="__main__":main()
