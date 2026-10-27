"""Small synthetic three-block FP32 forward/full-history backward comparison."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import time
from types import MethodType,SimpleNamespace
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution,_ExplicitTrunk
from .forward_fusion import ForwardFusedExecution,_ForwardFusionTrunk,first_with_forcing,finite_own_episode_from_max


def main():
    start=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(70027)
    h=1600;t=3;n=(h+1)*t
    parent=torch.arange(n)//(h+1)*(h+1)+h;parent[h::h+1]=-1
    samples=torch.stack([torch.arange(h)+i*(h+1) for i in range(t)])
    bank=SimpleNamespace(parent=parent,node_time=torch.tensor(([0.]*h+[1000.])*t),
        sample_index=samples,edge_geometry=torch.randn(n,4)*.1,branch_support_true=torch.rand(n,5),
        reference_class=torch.cat((torch.full((1000,),-1),torch.arange(3).repeat_interleave(200))),
        sample_age=torch.zeros(h),spans=torch.tensor([1.,4517.,25482.]))
    root_nodes=torch.arange(t)*(h+1)+h
    episodes=SimpleNamespace(hold_span=bank.spans,member_hold=torch.arange(t),member_episode=torch.zeros(t,dtype=torch.long),
        episode_size=torch.tensor([t]),member_node=root_nodes,
        flags=SimpleNamespace(node_flags=torch.randint(0,2,(n,2)).float()))
    cut=SimpleNamespace(group_offsets=tuple(i*h for i in range(t+1)),representatives=samples.flatten(),
        leaf_groups=torch.arange(t*h).reshape(t,h),group_counts=torch.ones(t*h),native_holds=t,haplotypes=h)
    bundle=SimpleNamespace(bank=bank,episodes=episodes,cut=cut)
    layout=[]
    for i in range(t):
        p=np.full(h+1,h,dtype=np.int64);p[-1]=-1
        leaves=np.arange(h,dtype=np.int64);empty=np.empty(0,dtype=np.int64)
        layout.append((i,i+1,i*(h+1),(i+1)*(h+1),p,(leaves,np.asarray([h])),
            (empty,leaves),(empty,np.zeros(h,dtype=np.int64)),(leaves,),i,i+1))
    features=torch.randn(t,h,67)*.1
    model=AmericanTreeLoomC3FullV1().float()
    def make(cls):
        executor=object.__new__(cls);executor.model=copy.deepcopy(model)
        executor.bank,executor.bundle,executor.device=bank,bundle,torch.device('cpu')
        executor.tick=lambda label:None;executor.profile=NullProfileV1()
        executor.layout=layout;executor.history_window=1;executor.hap_tile=32
        def sites(self,a,b):return self.model.site_pool_projection(features[a:b])
        executor._sites=MethodType(sites,executor)
        return executor
    original=make(ExplicitVJPExecution);fused=make(ForwardFusedExecution)
    boundary_errors=[]
    with torch.no_grad():
        reference_up=[];maxima=[]
        for z in layout:
            u,o=original._first(z);f=original._forcing(z,u,o)
            fu,fo,ff,m=first_with_forcing(fused,z)
            boundary_errors.append([float((u-fu).abs().max()),float((o-fo).abs().max()),float((f-ff).abs().max())])
            for a,b in [(u,fu),(o,fo),(f,ff)]:torch.testing.assert_close(a,b,atol=0,rtol=0)
            assert float(m)==float(u.abs().max())
            reference_up.append(u);maxima.append(m)
        assert torch.equal(original._finite_own_episode(reference_up),finite_own_episode_from_max(fused,torch.stack(maxima).amax()))
        for bad in [float('nan'),float('inf')]:
            try:finite_own_episode_from_max(fused,torch.tensor(bad))
            except ValueError:pass
            else:raise ValueError('Nonfinite proof scalar must fail closed')
    assert _ForwardFusionTrunk.backward is _ExplicitTrunk.backward
    results=[]
    def run(executor):
        for name,p in executor.model.named_parameters():
            if name.startswith(('forward_gru.','reverse_gru.','classifier.')):p.grad=torch.full_like(p,.125)
        at=time.monotonic();context=executor.context_forward()
        loss=(context.square()*torch.linspace(.1,1.,32)).mean()+.1*context.mean()
        loss.backward()
        result=dict(loss=float(loss.detach()),wall_seconds=time.monotonic()-at,receipt=executor.explicit_vjp_snapshot())
        gradients={name:None if p.grad is None else p.grad.detach().clone() for name,p in executor.model.named_parameters()}
        return context.detach(),gradients,result
    left=run(original);right=run(fused)
    torch.testing.assert_close(left[0],right[0],atol=1e-6,rtol=1e-5)
    errors={};disconnected=[]
    for name,a in left[1].items():
        b=right[1][name];assert (a is None)==(b is None),name
        if a is None:disconnected.append(name);continue
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-4,msg=name)
        errors[name]=float((a-b).abs().max())
        if name.startswith(('forward_gru.','reverse_gru.','classifier.')):assert bool((b==.125).all())
    assert left[2]['receipt']['retained_CPU_boundary_bytes']==right[2]['receipt']['retained_CPU_boundary_bytes']
    assert not torch.cuda.is_initialized()
    report=dict(schema='exact_forward_fusion_small_CPU_v1',passed=True,synthetic_holds=3,haplotypes=1600,nodes=n,
        tree_blocks=3,history_windows_per_direction=3,fresh_model_seed=70027,
        exact_first_forcing_boundary_errors=boundary_errors,absmax_equal=True,eligibility_equal=True,
        context_max_abs_error=float((left[0]-right[0]).abs().max()),parameter_gradient_max_abs_errors=errors,
        connected_or_preexisting_gradient_tensors=len(errors),unused_in_synthetic_probe=disconnected,
        original_backward_function_identity=True,retained_CPU_boundaries_equal=True,preexisting_decoder_gradients_preserved=True,
        original=left[2],fused=right[2],CUDA_initialized=False,scientific_files_read=False,optimizer_step=False,
        full_parent_or_GPU_speedup_claimed=False,
        scope='Synthetic star trees and synthetic pooled-site inputs; actual original base, PyG sweeps, episode/cut/history math, explicit full-history backward. Site MLP and focal decoder not exercised.',
        wall_seconds=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    output=Path(__file__).with_name('CPU_results_v1.json')
    with output.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('parameter_gradient_max_abs_errors','original','fused')},sort_keys=True))

if __name__=='__main__':main()
