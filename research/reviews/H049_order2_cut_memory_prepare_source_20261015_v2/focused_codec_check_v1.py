"""One root-authorized handmade CPU codec range/roundtrip check, no model/fit."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import io,json,hashlib,ast
import numpy as np
import torch
from research.reviews.H049_order2_cut_memory_prepare_source_20261015_v2 import cut_recipe_codec_v2 as repaired
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import cut_recipe_codec_v1 as frozen
from research.reviews.order2_cut_memory_implementation_20261015_v1 import order2_cut_memory_v1 as core
HERE=Path(__file__).resolve().parent
assert not torch.cuda.is_initialized()
h=952
parent=torch.zeros(h+1,dtype=torch.long);parent[0]=-1
times=torch.zeros(h+1,dtype=torch.float32);times[0]=1000.;times[-1]=450.
bank=SimpleNamespace(parent=parent,node_time=times,sample_index=torch.arange(1,h+1).reshape(1,h),
 sample_age=times[1:].clone(),spans=torch.tensor([1234.],dtype=torch.float32))
plan,metadata=core.prepare_cut_plan_v1(bank)
arrays,receipt=repaired.encode_v1(plan,metadata)
with io.BytesIO() as buffer:
 np.savez_compressed(buffer,**arrays);payload=buffer.getvalue()
a=repaired.decode_v1(payload,receipt,bank)
b=frozen.decode_v1(payload,receipt,bank)
assert all(torch.equal(x,y) and torch.equal(x,z) for x,y,z in zip(plan.tensors(),a.tensors(),b.tensors()))
assert plan.group_offsets==a.group_offsets==b.group_offsets
assert plan.episode_of_hold==a.episode_of_hold==b.episode_of_hold
rejected=[]
for name,modified in [
 ("uint32_overflow",replace(plan,representatives=plan.representatives.clone())),
 ("uint16_overflow",replace(plan,episode_groups=plan.episode_groups.clone()))
]:
 if name=="uint32_overflow":modified.representatives[0]=int(np.iinfo(np.uint32).max)+1
 else:modified.episode_groups[0,0]=int(np.iinfo(np.uint16).max)+1
 try:repaired.encode_v1(modified,metadata)
 except ValueError as e:
  assert str(e)=="Lossless verified integer range";rejected.append(name)
 else:raise AssertionError("Integer overflow must be rejected before casting")
old=Path(frozen.__file__).read_text();new=Path(repaired.__file__).read_text()
assert new==old.replace("np.iinfo(dtype).max()","np.iinfo(dtype).max")
def function_text(s,name):
 n=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name)
 return ast.get_source_segment(s,n)
assert function_text(old,"decode_v1")==function_text(new,"decode_v1")
assert function_text(old,"compact_upper_v1")==function_text(new,"compact_upper_v1")
print(json.dumps({"schema":"H049_prep_v2_single_focused_handmade_codec_check_v1","status":"passed",
 "fixture":"one 953-node star tree, 952 persistent leaves, one older-than350 singleton, one 1234-bp native hold",
 "roundtrip_all_logical_tensors":True,"both_repaired_and_original_decode_equal":True,
 "uint16_uint32_overflow_rejected":rejected,"decoder_and_bounds_exact_source_text_preserved":True,
 "codec_only_diff":"np.iinfo(dtype).max() -> np.iinfo(dtype).max",
 "payload_bytes":len(payload),"synthetic_in_memory_only":True,"real_payloads_checkpoints_truth_read":False,
 "optimizer_updates":0,"GPU_initialized":torch.cuda.is_initialized(),"guard_or_admission":False}))
