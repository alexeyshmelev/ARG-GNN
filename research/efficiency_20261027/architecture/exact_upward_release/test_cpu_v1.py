"""Synthetic CPU parity/lifetime checks, with no science payload or CUDA execution."""
import copy,gc,hashlib,json,resource,signal,time,weakref
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward,geometry_from_spans
from research.efficiency_20261027.architecture.exact_upward_residency.upward_residency import UpwardResidentExecution
from .upward_release_v1 import UpwardReleaseExecution
from .ownership_v1 import PrivateBlocks
from . import memory_v1 as memory

def fixture(t):
    started = time.monotonic()
    torch.manual_seed(70028+t)
    h, nodes_per_tree, groups_per_tree = 1600, 1605, 4
    n = t * nodes_per_tree
    parents, samples, layouts, assignments = [], [], [], []
    empty = np.empty(0, dtype=np.int64)
    leaves = np.arange(h, dtype=np.int64)
    middle = np.arange(h, h+4, dtype=np.int64)
    root = np.asarray([h+4], dtype=np.int64)
    for hold in range(t):
        offset = hold * nodes_per_tree
        membership = ((leaves + hold * 137) // 400) % 4
        p = np.concatenate((h + membership, np.full(4, h+4), np.asarray([-1]))).astype(np.int64)
        parents.append(torch.from_numpy(np.where(p >= 0, p+offset, p)))
        samples.append(torch.arange(h)+offset)
        assignments.append(torch.from_numpy(membership)+hold*4)
        layouts.append((hold, hold+1, offset, offset+nodes_per_tree, p,
            (leaves, middle, root), (empty, leaves, middle),
            (empty, membership, np.zeros(4, dtype=np.int64)),
            (middle, leaves), hold, hold+1))
    sites = 3 if t == 1 else 7
    genotype = torch.randint(0, 2, (sites, h), dtype=torch.int8)
    genotype[:, ::17] = -1
    genotype[1] = genotype[0]
    donor = torch.rand(sites, 2, 6).clamp_min(.1).log()
    donor[1] = donor[0]
    mask = torch.rand(sites, 6) > .2
    mask[1] = mask[0]
    panel = torch.rand(sites, 2).clamp_min(.1)
    panel = panel / panel.sum(dim=1, keepdim=True)
    panel[1] = panel[0]
    entropy = torch.rand(sites)
    entropy[1] = entropy[0]
    bank = SimpleNamespace(parent=torch.cat(parents),
        node_time=torch.tensor(([0.]*h+[300.]*4+[1000.])*t),
        sample_index=torch.stack(samples), edge_geometry=torch.randn(n,4)*.1,
        branch_support_true=torch.rand(n,5),
        reference_class=torch.cat((torch.arange(3).repeat_interleave(200),torch.full((1000,),-1))),
        sample_age=torch.zeros(h), spans=torch.tensor([1.,4517.,25482.,2.,3.][:t]),
        site_hold=torch.tensor([0,0,0] if t == 1 else [0,0,0,t-1,t-1,t-1,t-1]), genotype=genotype,
        log_donor_allele_frequency=donor, donor_call_mask=mask,
        panel_allele_frequency=panel, site_entropy=entropy)
    episodes = SimpleNamespace(hold_span=bank.spans, member_hold=torch.arange(t),
        member_episode=torch.zeros(t,dtype=torch.long), episode_size=torch.tensor([t]),
        member_node=torch.arange(t)*nodes_per_tree+h+4,
        flags=SimpleNamespace(node_flags=torch.randint(0,2,(n,2)).float()))
    cut = SimpleNamespace(group_offsets=tuple(i*4 for i in range(t+1)),
        representatives=torch.cat([torch.arange(h,h+4)+i*nodes_per_tree for i in range(t)]),
        leaf_groups=torch.stack(assignments), group_counts=torch.full((t*4,),400.),
        native_holds=t, haplotypes=h)
    bundle = SimpleNamespace(bank=bank, episodes=episodes, cut=cut)
    model = AmericanTreeLoomC3FullV1().float()

    return model,bank,bundle,layouts

def make(cls,t):
    model,bank,bundle,layouts=fixture(t)
    executor=object.__new__(cls)
    executor.model=model;executor.bank=bank;executor.bundle=bundle;executor.device=torch.device('cpu')
    executor.tick=lambda label:None;executor.profile=NullProfileV1()
    executor.layout=layouts;executor.history_window=1;executor.hap_tile=32;executor.capacity_stages=[]
    def capacity(self,plan,stage,live,future):
        assert self.device.type=='cpu' and 0<=live<=plan['maximum_simultaneous_bank_bytes']
        assert 0<=future<=plan['maximum_simultaneous_bank_bytes']-live
        self.capacity_stages.append(dict(stage=stage,live=live,future=future))
        return dict(CPU_test_only=True,GPU_capacity_checked=False)
    executor.admit_residency=MethodType(capacity,executor)
    return executor


def run(cls,t):
    executor=make(cls,t)
    model=executor.model
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,foreach=False,fused=False)
    rng=torch.random.get_rng_state().clone()
    context=executor.context_forward()
    if cls is UpwardReleaseExecution:
        owner=context.grad_fn.private_blocks
        old=[(weakref.ref(x),weakref.ref(x.untyped_storage())) for x in owner.blocks]
        assert len(context.grad_fn.saved_tensors)==3+49
    logits=[]
    hook=model.classifier.register_forward_hook(lambda module,args,value:logits.append(value.detach().clone()))
    hold=torch.tensor([0,0,t-1,t-1,0,t-1]);columns=torch.tensor([0,0,15,66,66,1599]);labels=torch.tensor([0,1,2,0,2,1])
    loss,vjp,decoder=decode_parent_backward(model,context,geometry_from_spans(executor.bank.spans),
        hold,columns,labels,tile_haplotypes=32)
    hook.remove();context.backward(vjp)
    if cls is UpwardReleaseExecution:
        assert owner.state=='finished' and all(ref() is None for probes in old for ref in probes)
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters()}
    assert len(gradients)==49 and all(torch.isfinite(x).all() for x in gradients.values())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    parameters={name:p.detach().clone() for name,p in model.named_parameters()}
    moments={str(i)+'/'+k:v.detach().clone() for i,state in optimizer.state_dict()['state'].items() for k,v in state.items()}
    assert len(moments)==147 and torch.equal(rng,torch.random.get_rng_state())
    return dict(context=context.detach(),logits=torch.cat(logits),loss=loss.detach(),norm=norm,
        gradients=gradients,parameters=parameters,Adam=moments,receipt=executor.explicit_vjp_snapshot(),
        capacity=executor.capacity_stages,decoder=decoder)


def parity_case(t):
    left=run(UpwardResidentExecution,t);right=run(UpwardReleaseExecution,t)
    errors={}
    for name in ('context','logits','loss','norm','gradients','parameters','Adam'):
        pairs=left[name].items() if isinstance(left[name],dict) else [(name,left[name])]
        category=[]
        for key,a in pairs:
            b=right[name][key] if isinstance(right[name],dict) else right[name]
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5 if name in ('context','logits','loss') else 1e-4,msg=name+'/'+key)
            category.append(float((a-b).abs().max()))
        errors[name]=max(category)
    assert torch.equal(left['context'],right['context'])
    assert torch.equal(left['logits'].argmax(-1),right['logits'].argmax(-1))
    block=1605*32*4;bank=t*block
    lifetime=right['receipt']['residency_lifetime']
    assert lifetime['peak_bank_bytes']==bank+block and lifetime['live_bank_bytes']==lifetime['future_bank_bytes']==0
    assert lifetime['upward_blocks_released']==lifetime['direct_blocks_released']==t
    history=[x for x in right['capacity'] if x['stage']=='explicit_complete_history_backward_begin']
    assert history==[dict(stage='explicit_complete_history_backward_begin',live=bank,future=0)]
    assert [x['live'] for x in right['capacity'] if x['stage']=='explicit_second_block_backward_released']==[bank]*t
    assert [x['live'] for x in right['capacity'] if x['stage']=='boundary_fused_first_block_released']==[bank-(i+1)*block for i in range(t)]
    return dict(holds=t,samples=1600,missing_max_ties_empty_holds_changing_cuts=True,full_decoder=True,
        gradient_tensors=49,updated_parameters=49,Adam_tensors=147,argmax_changes=0,errors=errors,
        lifetime=lifetime,original_peak_bank_bytes=left['receipt']['residency_lifetime']['peak_bank_bytes'],
        no_RNG_consumed=True,actual_tensor_and_storage_weakrefs_released_before_history=True)


def edges():
    cases=[]
    def reject(name,call):
        try:call()
        except (ValueError,RuntimeError,AssertionError):cases.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    owner=PrivateBlocks(torch.device('cpu'));x=torch.ones(4,32);owner.add(x);x.add_(1)
    reject('in_place_version_change',owner.begin);owner.abort();del x
    owner=PrivateBlocks(torch.device('cpu'));x=torch.ones(4,32);owner.add(x);alias=x.detach();del x
    owner.begin();reject('live_detached_storage_alias',lambda:owner.release(0));owner.abort();del alias
    owner=PrivateBlocks(torch.device('cpu'));owner.add(torch.ones(4,32));owner.begin();owner.release(0);owner.finish()
    reject('second_private_backward',owner.begin)
    reject('view_backing_storage',lambda:PrivateBlocks(torch.device('cpu')).add(torch.ones(8,32)[:4]))
    ex=make(UpwardReleaseExecution,3);context=ex.context_forward();owner=context.grad_fn.private_blocks
    reject('executor_reuse_while_context_live',ex.context_forward)
    context.sum().backward(retain_graph=True)
    reject('second_retained_graph_backward',lambda:context.sum().backward())
    another=ex.context_forward();another.sum().backward();cases.append('reuse_after_completed_backward')
    ex=make(UpwardReleaseExecution,1)
    with torch.autograd.graph.saved_tensors_hooks(lambda x:x,lambda x:x):
        reject('active_saved_tensor_hooks',ex.context_forward)
    ex=make(UpwardReleaseExecution,1);context=ex.context_forward();owner=context.grad_fn.private_blocks
    with torch.no_grad():next(ex.model.parameters()).add_(1)
    reject('saved_parameter_version_change',lambda:context.sum().backward())
    assert owner.state=='failed' and owner.blocks==[]
    ex=make(UpwardReleaseExecution,3);context=ex.context_forward();ref=weakref.ref(context.grad_fn.private_blocks)
    del context;gc.collect();assert ref() is None;cases.append('abandoned_context_released')
    ex=make(UpwardReleaseExecution,3);context=ex.context_forward();owner=context.grad_fn.private_blocks
    def fail(label):
        if label=='explicit_second_block_backward_released':raise ValueError('Injected bounded failure')
    ex.tick=fail
    reject('injected_backward_failure',lambda:context.sum().backward())
    assert owner.state=='failed' and owner.blocks==[];cases.append('failed_context_ownership_cleared')
    ex=make(UpwardReleaseExecution,1);context=ex.context_forward();owner=context.grad_fn.private_blocks
    owner.blocks[0].add_(1)
    reject('actual_context_version_change',lambda:context.sum().backward())
    assert owner.state=='failed' and owner.blocks==[]
    ex=make(UpwardReleaseExecution,1);context=ex.context_forward()
    reject('higher_order_backward',lambda:torch.autograd.grad(context.sum(),tuple(ex.model.parameters()),create_graph=True))
    plan=dict(upward_bank_bytes=30*32*4,maximum_simultaneous_bank_bytes=50*32*4,blocks=2)
    ledger=memory.ReplacementLedger(plan);a=torch.ones(10,32);b=torch.ones(20,32)
    ledger.add_upward(0,a);ledger.add_upward(1,b);ledger.begin_second()
    reject('live_storage_alias_in_ledger',lambda:ledger.add_direct(1,b))
    direct1=torch.zeros_like(b);ledger.add_direct(1,direct1);retired_key=ledger.upward[1][1];ledger.release_upward(1)
    direct0=torch.zeros_like(a)
    # Deterministic metadata simulation of a legal allocator address reuse.
    with patch.object(memory,'storage',return_value=(direct0.numel()*4,retired_key)):
        ledger.add_direct(0,direct0)
    ledger.release_upward(0);ledger.require_direct_complete();ledger.begin_first();ledger.release_direct(1,direct1)
    with patch.object(memory,'storage',return_value=(direct0.numel()*4,retired_key)):
        ledger.release_direct(0,direct0)
    ledger.require_released();assert ledger.peak<=plan['maximum_simultaneous_bank_bytes']
    cases.append('unequal_blocks_and_retired_pointer_reuse')
    return cases


def main():
    signal.alarm(120);started=time.monotonic();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=dict(schema='TreeLoom_upward_release_CPU_v1',passed=True,cases=[parity_case(t) for t in (1,3,5)],edge_cases=edges(),
        GPU_initialized=False,scientific_arrays_opened=False,production_modified=False,
        GPU_capacity_recorder_test_only=True,CUDA_stream_lifetime_not_tested_on_CPU=True)
    assert not torch.cuda.is_initialized()
    result.update(wall_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_refs=[dict(path=str(Path(__file__).with_name(name)),sha256=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
         for name in ('memory_v1.py','ownership_v1.py','upward_release_v1.py','large_release_executor_v1.py','test_cpu_v1.py')])
    path=Path(__file__).with_name('CPU_RESULTS_v1.json')
    with path.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(passed=True,edge_cases=len(result['edge_cases']),wall_seconds=result['wall_seconds'],
        peak_RSS_bytes=result['peak_RSS_bytes'],max_errors={key:max(c['errors'][key] for c in result['cases']) for key in result['cases'][0]['errors']})))

if __name__=='__main__':main()
