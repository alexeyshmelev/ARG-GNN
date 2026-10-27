"""CPU forward/backward on the existing 128-tree observed TRAIN packet.

The objective is a numerical differentiability probe, not a training loss or
ancestry score. No targets, optimizer, checkpoint, or GPU are used.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time,resource
from pathlib import Path
from dataclasses import fields
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
from treeloom_transport import TreeLoomTransport,DiagonalTransport,prepare_transport_plan
from transport_algebra import explicit_tree_kernel

print("stage: imports complete", flush=True)
torch.set_num_threads(4)
torch.manual_seed(9027)
# Verify the actual PyG class, not only the independent algebra module.
p=torch.tensor([-1,0,0,1,1,2,2])
gate_logit=torch.randn(7,5,requires_grad=True)
x=torch.randn(7,5,requires_grad=True)
gates=gate_logit.sigmoid();plan=prepare_transport_plan(p,torch.zeros(7,2))
actual=DiagonalTransport()(x,gates,plan)
k=explicit_tree_kernel(p,gates)
expected=torch.einsum('uvd,vd->ud',k,x)/k.sum(1)
max_error=float((actual-expected).abs().max())
assert torch.allclose(actual,expected,atol=2e-5,rtol=2e-5)
a=torch.autograd.grad(actual.square().sum(),(x,gate_logit),retain_graph=True)
b=torch.autograd.grad(expected.square().sum(),(x,gate_logit))
grad_error=max(float((u-v).abs().max()) for u,v in zip(a,b))
assert all(torch.allclose(u,v,atol=2e-5,rtol=2e-5) for u,v in zip(a,b))

print('stage: PyG algebra verified', flush=True)
packet=Path('research/scalability_20261024/execution/representative_slice.npz')
packet_sha=hashlib.sha256(packet.read_bytes()).hexdigest()
with np.load(packet,allow_pickle=False) as loaded:
    arrays={}
    for field in fields(MutationRoutedWholeTreeInputsV1):
        key=field.name
        if key in ('upward_levels','upward_children','upward_parent_rank','downward_levels'):
            names=sorted((k for k in loaded.files if k.startswith(key+'_')),key=lambda k:int(k.rsplit('_',1)[1]))
            arrays[key]=tuple(loaded[k] for k in names)
        else:arrays[key]=loaded[key]
    offsets=loaded['node_offsets']
bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
isolated=tuple(i for i,(a,b) in enumerate(zip(offsets,offsets[1:])) if b-a==1600 and bool((bank.parent[int(a):int(b)]<0).all()))
episodes,episode_meta=build_episodes(bank,max_payload_bytes=2**30,isolated_hold_indices=isolated)
flag_meta=episode_meta['flag_metadata']
print('stage: packet and qualified flags prepared', flush=True)
plan=prepare_transport_plan(bank.parent,episodes.flags.node_flags)
model=TreeLoomTransport()
contract=model.contract()
print('stage: model and jump plan ready', contract, flush=True)
start=time.perf_counter()
logits=model(bank,plan)
print('stage: forward complete',time.perf_counter()-start,flush=True)
assert logits.shape==(128,1000,3)
assert torch.isfinite(logits).all()
# All query/interval outputs participate in the numerical derivative probe.
probe=logits.square().mean()
probe.backward()
print('stage: backward complete',time.perf_counter()-start,flush=True)
elapsed=time.perf_counter()-start
gradients=[dict(name=name,finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),nonzero=bool(p.grad is not None and p.grad.count_nonzero())) for name,p in model.named_parameters()]
assert all(x['finite'] for x in gradients)
assert all(x['nonzero'] for x in gradients)
receipt=dict(schema='TreeLoom_Transport_observed_CPU_smoke_v1',umbrella_hypothesis='treeloom_efficiency_redesign_20261027',status='passed',contract=contract,packet_path=str(packet),packet_sha256=packet_sha,trees=128,nodes=len(bank.parent),haplotypes=1600,queries=1000,sites=len(bank.genotype),output_shape=list(logits.shape),topology_jump_rounds=len(plan.jumps),forest_feature_metadata=flag_meta,isolated_hold_indices=list(isolated),feature_builder='research.repairs.benchmark_recovery_20261020.runtime.episodes.build_episodes',pyg_reference_max_abs_error=max_error,pyg_reference_gradient_max_abs_error=grad_error,finite_output=True,all_parameter_gradients_finite=True,all_parameter_gradients_nonzero=True,gradient_details=gradients,process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,diagnostic_forward_backward_seconds=elapsed,timing_is_not_benchmark=True,threads=4,torch_version=torch.__version__,source='existing_observation_only_TRAIN_packet',ancestry_targets_opened=False,optimizer_created=False,optimizer_updates=0,CUDA_initialized=False,genomic_context='The packet is a 128-tree prefix. Scan begins/ends at its diagnostic boundaries. This is not a full-parent update or a production context guarantee.',limitations=['No full-parent memory or GPU timing qualification.','No ancestry score or useful GPU activity measurement.','Fresh randomly initialized changed model.','Full all-parent training and chunk carry executor remain future work.'])
Path(__file__).with_name('observed_cpu_smoke.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='gradient_details'},indent=2))
