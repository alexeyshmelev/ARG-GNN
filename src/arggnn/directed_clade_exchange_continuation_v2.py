"""H039 retained-checkpoint format fix; frozen v1 optimization/operator unchanged."""
from __future__ import annotations
import io,json
import torch
from .directed_clade_exchange_continuation_v1 import (
 COUNTS,NEW_KEYS,TRAIN,VAL,build_family_v1,train_shared_parent_v1,probability_v1,run_shared_continuations_v1)
SOURCE="src/arggnn/directed_clade_exchange_continuation_v2.py"
STATE_SHAPES={
 "site_mlp.0.weight":(32,19),"site_mlp.0.bias":(32,),"site_mlp.2.weight":(32,32),"site_mlp.2.bias":(32,),
 "site_pool_projection.0.weight":(32,67),"site_pool_projection.0.bias":(32,),
 "leaf_projection.0.weight":(32,38),"leaf_projection.0.bias":(32,),
 "node_projection.0.weight":(32,3),"node_projection.0.bias":(32,),
 "up_gate.weight":(32,9),"up_gate.bias":(32,),"down_gate.weight":(32,9),"down_gate.bias":(32,),
 "up_message.weight":(32,32),"up_update.weight":(32,64),"up_update.bias":(32,),
 "up_norm.weight":(32,),"up_norm.bias":(32,),
 "down_message.weight":(32,32),"down_update.weight":(32,32),"down_update.bias":(32,),
 "down_norm.weight":(32,),"down_norm.bias":(32,),
 "leaf_context.0.weight":(32,64),"leaf_context.0.bias":(32,),
 "genomic_gru.weight_ih_l0":(96,35),"genomic_gru.weight_hh_l0":(96,32),
 "genomic_gru.bias_ih_l0":(96,),"genomic_gru.bias_hh_l0":(96,),
 "genomic_gru.weight_ih_l0_reverse":(96,35),"genomic_gru.weight_hh_l0_reverse":(96,32),
 "genomic_gru.bias_ih_l0_reverse":(96,),"genomic_gru.bias_hh_l0_reverse":(96,),
 "classifier.0.weight":(32,96),"classifier.0.bias":(32,),"classifier.2.weight":(4,32),"classifier.2.bias":(4,)}


def load_selected_warmstart_v1(completed_ref,reader):
 """Exact path-indexed retained ref and actual nine-field H038 checkpoint."""
 completed=json.loads(reader.read(completed_ref))
 if (completed.get("status")!="completed_H038_seed101_accuracy_selected_full_parent_run"
  or completed.get("arm")!="aligned" or completed.get("seed")!=101 or completed.get("outer_truth_read") is not False
  or completed.get("outer_predictions_executed") is not False):
  raise ValueError("Completed aligned H038/seed101/no-outer warm start required")
 selected=completed["selected"];checkpoint=selected["checkpoint"];retained=completed["retained_records"]
 if (not isinstance(checkpoint,dict) or set(checkpoint)!={"path","sha256"} or not isinstance(retained,dict)
  or retained.get(checkpoint["path"])!=checkpoint):
  raise ValueError("Actual selected checkpoint must equal its retained path-indexed ref")
 payload=torch.load(io.BytesIO(reader.read(checkpoint)),map_location="cpu",weights_only=True)
 expected={"model_state","optimizer_state","pass_index","optimizer_steps","arm","seed","input_ref","initial_state_sha256","teacher_used"}
 if not isinstance(payload,dict) or set(payload)!=expected:
  raise ValueError("Exact actual H038 nine-field checkpoint payload required")
 if (payload["arm"]!="aligned" or payload["seed"]!=101 or payload["teacher_used"] is not False
  or payload["pass_index"]!=selected["pass_index"] or payload["optimizer_steps"]!=selected["optimizer_steps"]):
  raise ValueError("Selected checkpoint/training provenance mismatch")
 state=payload["model_state"]
 if not isinstance(state,dict) or set(state)!=set(STATE_SHAPES):
  raise ValueError("Original exact38 learned backbone state keys required")
 for name,shape in STATE_SHAPES.items():
  value=state[name]
  if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.device.type!="cpu" or tuple(value.shape)!=shape:
   raise ValueError("Original width32 FP32 checkpoint shape/dtype differs: "+name)
 if sum(v.numel() for v in state.values())!=29764:raise ValueError("Original29764-element backbone required")
 return payload,dict(completed_ref=completed_ref,checkpoint_ref=checkpoint,
  original_selected_pass=payload["pass_index"],original_optimizer_steps=payload["optimizer_steps"],
  retained_record_format="path_to_exact_ref_dict",model_state_keys=38,model_parameter_elements=29764,
  optimizer_state_policy="reset_AdamW_moments_and_step_for_all_three_arms",teacher_predictions_opened=False)
