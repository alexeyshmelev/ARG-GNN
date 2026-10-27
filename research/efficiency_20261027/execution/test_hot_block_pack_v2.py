"""CPU exact-value, lifecycle, rejection and actual observed-trunk pack checks."""
import copy
from dataclasses import fields
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
from research.efficiency_20261027.execution.large_layout_plan import LargeLayoutPlan
from research.scalability_20261024.integration.executors import IntegratedStagedExecution
from research.benchmark_rebuild_20261024.american.treeloom_speed import static_model_v2 as frozen
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
from research.repairs.benchmark_recovery_20261020.runtime.cuts import prepare_cut_plan_v1
from research.efficiency_20261027.root.compare_execution import ROOT,CHECKPOINT,ref,write
from .hot_block_pack_v1 import BlockInputPack,tensors,block_bytes,admission_bounds
from .hot_block_executor_v1 import PackedBlockInputs,PackedLargeExplicitVJPExecution

BASE=ROOT/'research/efficiency_20261027/execution'


class CPUPacked(PackedBlockInputs,ExplicitVJPExecution):
    pass


def need(ok,message):
    if not ok:raise AssertionError(message)


def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected fail-closed rejection')


def load_fixture():
    packet=Path('research/scalability_20261024/execution/representative_slice.npz')
    metadata=json.loads(packet.with_suffix('.json').read_text())
    need(ref(packet)==metadata['packet_ref'],'Authenticated actual observed packet required')
    with np.load(packet,allow_pickle=False) as data:
        arrays={}
        for field in fields(MutationRoutedWholeTreeInputsV1):
            name=field.name
            arrays[name]=(tuple(data[name+'_'+str(i)] for i in range(metadata['level_tuple_lengths'][name]))
                if name in metadata['level_tuple_lengths'] else data[name])
        offsets=data['node_offsets']
        counts=np.diff(offsets).tolist()
        bank=MutationRoutedWholeTreeInputsV1.from_arrays(arrays)
    isolated=tuple(i for i,(a,b) in enumerate(zip(offsets,offsets[1:]))
        if b-a==1600 and bool((bank.parent[int(a):int(b)]<0).all()))
    episodes,_=build_episodes(bank,max_payload_bytes=2**30,isolated_hold_indices=isolated)
    cut,_=prepare_cut_plan_v1(bank,expected_haplotypes=1600,max_nodes=400000,max_holds=128,isolated_hold_indices=isolated)
    return bank,episodes,cut,counts,ref(packet)


def pack_checks(bank,layout):
    total=sum(block_bytes(bank,z) for z in layout)
    admissions=[]
    make=lambda:BlockInputPack(bank,layout,device='cpu',provenance='actual observed128hold packet',
        max_payload_bytes=total,admit=lambda remaining,stage:admissions.append((remaining,stage)),CPU_test_only=True)
    before=torch.get_rng_state().clone()
    cache=make()
    expected=object.__new__(IntegratedStagedExecution)
    expected.bank=bank
    expected.device=torch.device('cpu')
    started=time.perf_counter()
    for z in layout:
        original=expected._block(z)
        packed=cache.block(bank,z,provenance=cache.provenance)
        for old,new in zip(tensors(original),tensors(packed)):
            need(old.dtype==new.dtype and old.shape==new.shape and torch.equal(old,new),'Every packed value/dtype/order must be exact')
            if old.numel():need(old.data_ptr()!=new.data_ptr(),'CPU test must exercise independent immutable copies')
    build_seconds=time.perf_counter()-started
    del original,packed
    started=time.perf_counter()
    for order in (layout,tuple(reversed(layout)),tuple(reversed(layout))):
        for z in order:cache.block(bank,z,provenance=cache.provenance)
    lookup_seconds=time.perf_counter()-started
    snapshot=cache.snapshot()
    need(snapshot['block_copies']==len(layout) and snapshot['cache_hits']==3*len(layout),'Exactly one copy and three hits per original block')
    rejects(lambda:cache.block(bank,tuple(list(layout[0])),provenance=cache.provenance))
    rejects(lambda:cache.block(copy.copy(bank),layout[0],provenance=cache.provenance))
    rejects(lambda:cache.block(bank,layout[0],provenance='other parent'))
    errors=[]
    def other_thread():
        try:cache.block(bank,layout[0],provenance=cache.provenance)
        except ValueError:errors.append(True)
    thread=threading.Thread(target=other_thread)
    thread.start()
    thread.join()
    need(errors==[True],'Other-thread access must fail closed')
    packed=cache.block(bank,layout[0],provenance=cache.provenance)
    packed.node_time.add_(0)
    rejects(lambda:cache.block(bank,layout[0],provenance=cache.provenance))
    cache.abort()
    del packed
    rejects(lambda:cache.block(bank,layout[0],provenance=cache.provenance))
    cache=make()
    old=bank.node_time
    object.__setattr__(bank,'node_time',bank.node_time.clone())
    rejects(lambda:cache.block(bank,layout[0],provenance=cache.provenance))
    object.__setattr__(bank,'node_time',old)
    old.add_(0)
    rejects(lambda:cache.block(bank,layout[0],provenance=cache.provenance))
    cache.abort()
    rejects(lambda:BlockInputPack(bank,layout,device='cpu',provenance='test',max_payload_bytes=total-1,
        admit=lambda *args:None,CPU_test_only=True))
    rejects(lambda:BlockInputPack(bank,layout,device='cpu',provenance='test',max_payload_bytes=total,
        admit=lambda *args:None))
    cache=make()
    cache.begin_forward()
    rejects(cache.close)
    rejects(cache.begin_forward)
    cache.backward_complete()
    rejects(cache.begin_forward)
    cache.close()
    need(cache.snapshot()['retained_payload_bytes']==0 and not cache.layout,'Close drops both device storage and held layout references')
    rejects(lambda:cache.block(bank,layout[0],provenance=cache.provenance))
    need(torch.equal(before,torch.get_rng_state()),'Pack binding/build/use/release must not consume RNG')
    good=dict(remaining_payload_bytes=10,scratch_reserve_bytes=20,allocated_bytes=2,
        reserved_bytes=3,physical_free_bytes=40,soft_budget_bytes=40)
    admission_bounds(**good)
    rejects(lambda:admission_bounds(**dict(good,physical_free_bytes=29)))
    rejects(lambda:admission_bounds(**dict(good,soft_budget_bytes=32)))
    rejects(lambda:admission_bounds(**dict(good,reserved_bytes=1)))
    return dict(exact_values_dtypes_orders=True,independent_CPU_copy=True,full_four_pass_coverage=True,
        ownership_binding_mutation_provenance_partition_budget_lifetime_rejections=True,RNG_unchanged=True,
        snapshot=snapshot,CPU_copy_plus_validation_seconds=build_seconds,
        three_pass_lookup_seconds=lookup_seconds,CPU_timings_are_not_H2D_or_GPU_performance=True,
        admission_actual_reserved_and_physical_free_rejections=True)


def trunk_checks(bank,episodes,cut,layout):
    saved=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    template=frozen.AmericanTreeLoomC3FullV1()
    template.load_state_dict(saved['model_state'],strict=True)
    bundle=SimpleNamespace(bank=bank,episodes=episodes,cut=cut)
    def run(packed):
        model=copy.deepcopy(template)
        for name,parameter in model.named_parameters():
            parameter.grad=(torch.full_like(parameter,.125) if name.startswith(
                ('forward_gru.','reverse_gru.','classifier.')) else None)
        executor=object.__new__(CPUPacked if packed else ExplicitVJPExecution)
        executor.model,executor.bank,executor.bundle=model,bank,bundle
        executor.device=torch.device('cpu')
        executor.tick=lambda label:None
        executor.profile=NullProfileV1()
        executor.layout=layout
        executor.history_window=17
        executor.hap_tile=32
        if packed:
            executor.bind_hot_pack(provenance='actual128hold complete restricted slice',
                max_payload_bytes=sum(block_bytes(bank,z) for z in layout),
                admit=lambda *args:None,CPU_test_only=True)
        started=time.perf_counter()
        RNG=torch.get_rng_state().clone()
        context=executor.context_forward()
        if packed:
            rejects(executor.release_layout)
            need(executor.hot_pack_snapshot()['phase']=='forward_and_backward','Lease must span decoder boundary')
        coefficient=torch.linspace(.1,1.,32)
        loss=(context.square()*coefficient).mean()+.1*context.mean()
        loss.backward()
        need(torch.equal(RNG,torch.get_rng_state()),'Trunk RNG unchanged')
        gradients={name:None if value.grad is None else value.grad.detach().clone()
            for name,value in model.named_parameters()}
        proof=dict(kind='packed' if packed else 'original',seconds=time.perf_counter()-started,
            loss=float(loss.detach()),explicit_vjp=executor.explicit_vjp_snapshot())
        if packed:
            proof['pack']=executor.hot_pack_snapshot()
            need(proof['pack']['phase']=='backward_complete','Post-hook must occur after complete original custom backward')
            need(proof['pack']['block_copies']==len(layout) and proof['pack']['cache_hits']==3*len(layout),'All four trunk passes use one retained pack')
            rejects(executor.context_forward)
        output=context.detach().clone()
        del context,loss
        executor.release_layout()
        if packed:
            need(executor.hot_pack_snapshot()['retained_payload_bytes']==0,'Success releases strong input pack')
        print(json.dumps(proof),flush=True)
        return output,gradients,proof
    original=run(False)
    candidate=run(True)
    need(torch.equal(original[0],candidate[0]),'Actual complete restricted-slice contexts must be bit exact')
    errors={}
    for name,value in original[1].items():
        other=candidate[1][name]
        need((value is None)==(other is None),'Same connected parameter set')
        if value is not None:
            need(torch.equal(value,other),'Bit-exact original parameter gradient '+name)
            errors[name]=float((value-other).abs().max())
    for name in ('_first','_second','_forcing','_sites','_base','_finite_own_episode'):
        need(getattr(PackedLargeExplicitVJPExecution,name) is getattr(ExplicitVJPExecution,name),
             'Unchanged inherited computation '+name)
    return dict(actual_observed_native_holds=128,haplotypes=1600,blocks=len(layout),
        block_holds=65,ragged_final_block_holds=63,full_restricted_bidirectional_history=True,
        history_window=17,context_and_loss_bit_exact=True,parameter_gradient_max_abs_errors=errors,
        preexisting_decoder_gradients_preserved=True,RNG_unchanged=True,
        original_all_computation_methods_inherited_identically=True,runs=[original[2],candidate[2]],
        scope='Actual128hold observed packet/checkpoint241 trunk and nonlinear scalar probe; no full parent, focal decoder, optimizer, GPU or H2D speed claim.')


def main():
    need(not torch.cuda.is_initialized(),'CPU-only test required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bank,episodes,cut,counts,packet_ref=load_fixture()
    plan=LargeLayoutPlan.build(bank,counts,episodes.member_hold,holds_per_block=65)
    layout=plan.executor_layout()
    value_checks=pack_checks(bank,layout)
    learned=trunk_checks(bank,episodes,cut,layout)
    receipt=dict(schema='TreeLoom_hot_block_pack_CPU_v1',status='pass',
        source_refs=[ref(BASE/'hot_block_pack_v1.py'),ref(BASE/'hot_block_executor_v1.py'),ref(__file__),
            ref('research/efficiency_20261027/execution/explicit_vjp.py'),
            ref('research/scalability_20261024/integration/executors.py'),ref(CHECKPOINT),packet_ref],
        exact_inputs_and_lifecycle=value_checks,actual_trunk=learned,CUDA_initialized=torch.cuda.is_initialized(),
        production_modified=False,GPU_launches=0)
    write(BASE/'hot_block_pack_CPU_v2.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
