"""Reproduce numerical summaries using only dedicated handmade CPU fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import torch
from arggnn.anchor_compact_actual_bank_v1 import compare_action_VJPs_v1, compare_model_step_v1

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
assert Path.cwd().resolve()==ROOT
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location("compact_adapter_handmade_tests",
    ROOT/"tests/test_anchor_compact_actual_bank_v1.py")
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
rows=[]
for dtype in (torch.float64,torch.float32):
 for coupling in ("full","scalar","diagonal","gain_matched_diagonal"):
  for stress in (False,True):
   native,compact,bank,view=fixtures.fixture(dtype,coupling,stress)
   model=compare_model_step_v1(native,compact,bank,view)
   action=compare_action_VJPs_v1(native,bank,view)
   checks={"model."+k:v for k,v in model["checks"].items()}
   checks.update({"action."+k:v for k,v in action["checks"].items()})
   worst=max(checks,key=lambda k:checks[k]["max_acceptance_units"])
   rows.append(dict(dtype=str(dtype),coupling=coupling,stress=stress,
     model_status=model["status"],action_status=action["status"],
     numeric_comparisons=len(checks),worst_family=worst,
     max_acceptance_units=checks[worst]["max_acceptance_units"],
     max_absolute_error=max(v["max_absolute_error"] for v in checks.values()),
     native_preclip_norm=model["preclip_native_norm"],
     compact_preclip_norm=model["preclip_compact_norm"],
     checks=checks))
payload=dict(kind="handmade_compact_actual_bank_adapter_parity_v1",
 actual_arrays_opened=False,CUDA_exercised=False,measured_efficiency=False,
 rows=rows,numeric_comparisons=sum(x["numeric_comparisons"] for x in rows),
 all_passed=all(x["model_status"]==x["action_status"]=="PASS" for x in rows),
 max_acceptance_units=max(x["max_acceptance_units"] for x in rows),
 max_absolute_error=max(x["max_absolute_error"] for x in rows))
output=ROOT/"research/reviews/anchor_compact_actual_bank_v1_author/handmade_parity_metrics.json"
output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in payload.items() if k!="rows"}))
