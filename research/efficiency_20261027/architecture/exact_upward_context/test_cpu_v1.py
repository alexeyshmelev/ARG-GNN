"""CPU model-device composition parity; public large constructor retains CUDA-only admission."""
import hashlib,json,resource,signal,time,weakref
from pathlib import Path
from types import MethodType
from unittest.mock import patch
import torch
from research.efficiency_20261027.architecture.exact_upward_release import test_cpu_v2 as previous
from research.efficiency_20261027.architecture.exact_upward_residency.upward_residency import UpwardResidentExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import geometry_from_spans
from .upward_context_v1 import UpwardContextExecution
from .memory_v1 import future_context_bytes
from .decoder_v1 import decode_model_device


def make(t):
    ex=previous.make(UpwardContextExecution,t)
    base=ex.admit_residency
    ex.joint_stages=[]
    def admission(self,plan,stage,live,future):
        result=base(plan,stage,live,future)
        contexts=future_context_bytes(plan,self._context_ready,self._VJP_ready)
        self.joint_stages.append(dict(stage=stage,live_bank_bytes=live,future_bank_bytes=future,
            future_context_bytes=contexts,context_ready=self._context_ready,VJP_ready=self._VJP_ready))
        return dict(result,future_context_bytes=contexts)
    ex.admit_residency=MethodType(admission,ex)
    return ex


def coordinates(t):
    return (torch.tensor([0,0,t-1,t-1,0,t-1]),torch.tensor([0,0,15,66,66,1599]),torch.tensor([0,1,2,0,2,1]))


def run(t):
    ex=make(t);model=ex.model
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False,fused=False)
    rng=torch.random.get_rng_state().clone();context=ex.context_forward()
    owner=context.grad_fn.private_blocks
    old=[(weakref.ref(x),weakref.ref(x.untyped_storage())) for x in owner.blocks]
    assert context.is_contiguous() and context.untyped_storage().nbytes()==context.numel()*4
    assert len(context.grad_fn.saved_tensors)==3+49
    logits=[];hook=model.classifier.register_forward_hook(lambda module,args,value:logits.append(value.detach().clone()))
    loss,vjp,decoder=ex.decode_parent_backward(context,geometry_from_spans(ex.bank.spans),*coordinates(t),tile_haplotypes=32)
    hook.remove()
    assert vjp.device==context.device and vjp.shape==context.shape and vjp.is_contiguous()
    assert vjp.untyped_storage().data_ptr()!=context.untyped_storage().data_ptr()
    assert bool((vjp[:,32:64]==0).all()),'Empty focal tile must have zero VJP'
    context.backward(vjp)
    assert owner.state=='finished' and all(ref() is None for probes in old for ref in probes)
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    parameters={name:p.detach().clone() for name,p in model.named_parameters()}
    moments={str(i)+'/'+k:v.detach().clone() for i,state in optimizer.state_dict()['state'].items() for k,v in state.items()}
    assert len(gradients)==len(parameters)==49 and len(moments)==147 and torch.equal(rng,torch.random.get_rng_state())
    c=context.numel()*4
    assert ex.joint_stages[0]['future_context_bytes']==2*c
    after=[x for x in ex.joint_stages if x['stage']=='context_preallocated']
    before_vjp=[x for x in ex.joint_stages if x['stage']=='before_VJP_preallocation']
    after_vjp=[x for x in ex.joint_stages if x['stage']=='VJP_preallocated']
    assert len(after)==len(before_vjp)==len(after_vjp)==1
    assert after[0]['future_context_bytes']==before_vjp[0]['future_context_bytes']==c
    assert after_vjp[0]['future_context_bytes']==0
    assert ex.explicit_receipt['final_context_preallocated'] and not ex.explicit_receipt['full_context_concatenation']
    return dict(context=context.detach(),logits=torch.cat(logits),loss=loss,norm=norm,gradients=gradients,
        parameters=parameters,Adam=moments,receipt=ex.explicit_vjp_snapshot(),joint_stages=ex.joint_stages,
        exact_preallocated_context_and_VJP_bytes=2*c,decoder=decoder)


def parity(t):
    left=previous.run(UpwardResidentExecution,t);right=run(t);errors={}
    for name in ('context','logits','loss','norm','gradients','parameters','Adam'):
        pairs=left[name].items() if isinstance(left[name],dict) else [(name,left[name])]
        category=[]
        for key,a in pairs:
            b=right[name][key] if isinstance(right[name],dict) else right[name]
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5 if name in ('context','logits','loss') else 1e-4,msg=name+'/'+key)
            category.append(float((a-b).abs().max()))
        errors[name]=max(category)
    assert torch.equal(left['context'],right['context']) and torch.equal(left['logits'].argmax(-1),right['logits'].argmax(-1))
    life=right['receipt']['residency_lifetime'];bank=t*1605*32*4
    assert life['live_bank_bytes']==life['future_bank_bytes']==0 and life['peak_bank_bytes']==bank+1605*32*4
    return dict(holds=t,samples=1600,errors=errors,argmax_changes=0,gradient_tensors=49,updated_parameters=49,Adam_tensors=147,
        final_context_preallocated=True,full_context_concatenation=False,VJP_preallocated=True,
        lifetime=life,joint_admission_stages=right['joint_stages'],decoder=right['decoder'],
        exact_preallocated_context_and_VJP_bytes=right['exact_preallocated_context_and_VJP_bytes'])


def edges():
    # Reuse the exact sealed lifetime suite, replacing only its test-local class reference.
    with patch.object(previous,'UpwardReleaseExecution',UpwardContextExecution):cases=previous.edges()
    def reject(name,call):
        try:call()
        except (ValueError,RuntimeError,AssertionError):cases.append(name)
        else:raise AssertionError('Expected rejection '+name)
    ex=make(1);context=ex.context_forward();geo=geometry_from_spans(ex.bank.spans);coords=coordinates(1)
    reject('different_context_identity',lambda:ex.decode_parent_backward(context.clone(),geo,*coords,tile_haplotypes=32))
    reject('decoder_partition_not_in_plan',lambda:ex.decode_parent_backward(context,geo,*coords,tile_haplotypes=64))
    reject('VJP_aliases_context',lambda:decode_model_device(ex.model,context,context.detach(),geo,*coords,tile_haplotypes=32))
    loss,vjp,_=ex.decode_parent_backward(context,geo,*coords,tile_haplotypes=32)
    reject('duplicate_decoder',lambda:ex.decode_parent_backward(context,geo,*coords,tile_haplotypes=32))
    context.backward(vjp)
    reject('VJP_before_context_accounting',lambda:future_context_bytes({'complete_context_bytes':1024},False,True))
    return cases


def main():
    signal.alarm(120);started=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases=[parity(t) for t in (1,3,5)];edge=edges();assert not torch.cuda.is_initialized()
    result=dict(schema='TreeLoom_upward_context_composition_CPU_v1',passed=True,cases=cases,edge_cases=edge,
        CPU_device_is_test_only=True,public_factory_cuda_guard_unchanged=True,actual_GPU_peak_or_stream_behavior_proven=False,
        GPU_initialized=False,scientific_arrays_opened=False,production_modified=False,
        wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_refs=[dict(path=str(Path(__file__).with_name(name)),sha256=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
         for name in ('memory_v1.py','decoder_v1.py','upward_context_v1.py','large_context_executor_v1.py','test_cpu_v1.py')])
    path=Path(__file__).with_name('CPU_RESULTS_v1.json')
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(passed=True,edge_cases=len(edge),wall_seconds=result['wall_seconds'],peak_RSS_bytes=result['peak_RSS_bytes'],
        max_errors={key:max(c['errors'][key] for c in cases) for key in cases[0]['errors']})))

if __name__=='__main__':main()
