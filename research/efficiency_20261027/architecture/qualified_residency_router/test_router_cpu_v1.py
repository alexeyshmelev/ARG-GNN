"""Scalar boundary, factory-interface, and tiny real-model CPU parity tests.

Only the factory-interface test substitutes fake CUDA counters/constructors.
The numerical test directly instantiates the existing CPU test harness classes;
the production factory's CUDA/type guards are never bypassed for execution.
"""
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
from types import MethodType, SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import router_v1 as router


def rejected(call):
    try:
        call()
    except (ValueError, TypeError, KeyError):
        return True
    raise AssertionError('Expected rejection before any forward')


def scalar_tests():
    G = router.GIB
    base = dict(native_nodes=243269632, native_holds=78000,
        free_bytes=80*G, total_bytes=80*G, allocated_bytes=0, reserved_bytes=0,
        resident_soft_bytes=74*G)
    exact = router.select_capacity(**base)
    assert exact['selected_route'] == router.RESIDENT and exact['resident_soft_margin_bytes'] == 0
    large = router.select_capacity(**dict(base, native_nodes=250680038, native_holds=78362))
    assert large['selected_route'] == router.EXPLICIT
    node_over = router.select_capacity(**dict(base, native_nodes=base['native_nodes']+1))
    assert node_over['selected_route'] == router.EXPLICIT
    one_byte = router.select_capacity(**dict(base, resident_soft_bytes=74*G-1))
    assert one_byte['selected_route'] == router.EXPLICIT
    current = router.select_capacity(**dict(base, allocated_bytes=1, reserved_bytes=1, free_bytes=80*G-1))
    assert current['selected_route'] == router.EXPLICIT
    physical = dict(base, free_bytes=74*G, total_bytes=80*G)
    assert router.select_capacity(**physical)['resident_capacity_margin_bytes'] == 0
    assert router.select_capacity(**dict(physical, free_bytes=74*G-1))['selected_route'] == router.EXPLICIT
    cached = router.select_capacity(**dict(base, free_bytes=60*G, reserved_bytes=20*G))
    assert cached['selected_route'] == router.RESIDENT and cached['available_for_new_allocations_bytes'] == 80*G
    for update in (
        dict(native_nodes=True), dict(native_holds=0), dict(native_nodes=1599),
        dict(native_nodes=-1), dict(resident_soft_bytes=74*G+1),
        dict(resident_scratch_bytes=16*G-1), dict(resident_scratch_bytes=16*G+1),
        dict(allocated_bytes=1), dict(free_bytes=81*G), dict(reserved_bytes=1),
        dict(free_bytes=16*G-1), dict(resident_soft_bytes=16*G-1),
        dict(resident_soft_bytes=74.0*G)):
        rejected(lambda update=update: router.select_capacity(**dict(base, **update)))
    measured = router.select_capacity(**dict(base, native_nodes=205369272, native_holds=77472))
    assert measured['resident_required_additional_bytes'] == 69754402816
    assert measured['selected_route'] == router.RESIDENT
    return dict(passed=True, exact_soft_equality=True, exact_physical_equality=True,
        one_byte_or_one_node_excess_falls_back=True, current_allocation_counted=True,
        cached_capacity_counted_once=True, largest_all20_upper_bound_falls_back=True,
        invalid_and_insufficient_baseline_cases_rejected=13,
        TRAIN000_measured_counts_screen_passed=True)


def factory_tests():
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
    from research.efficiency_20261027.execution import large_block_executors
    from research.efficiency_20261027.architecture.exact_upward_residency import large_resident_executor_v1
    events = []
    class Lease:
        def release(self):
            events.append('release_probe_lease')
    class ProbeCache(LargeBlockLayoutCache):
        def __init__(self):
            pass
        def acquire(self, bundle, metadata, **kwargs):
            assert kwargs['holds_per_block'] == 512
            events.append('CPU_layout_ready')
            return Lease()
    class FakeCudaModel:
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device('cuda:0'), dtype=torch.float32)])
    bundle = SimpleNamespace(bank=SimpleNamespace(parent=torch.empty(1605)), native_holds=1,
        cut=SimpleNamespace(native_holds=1, haplotypes=1600), parent_id='synthetic')
    metadata = dict(episode_metadata=dict(native_nodes_per_tree=[1605]))
    def snapshot(device):
        assert device == torch.device('cuda:0') and events[-1] == 'resource_check'
        events.append('live_snapshot')
        return dict(free_bytes=80*router.GIB, total_bytes=80*router.GIB,
            allocated_bytes=0, reserved_bytes=0)
    def factory(name):
        def construct(model, actual_bundle, actual_metadata, **kwargs):
            assert actual_bundle is bundle and actual_metadata is metadata
            assert kwargs['holds_per_block'] == 512 and kwargs['haplotypes_per_gru'] == 64
            assert kwargs['history_holds_per_window'] == 1024
            if name == router.RESIDENT:
                assert kwargs['resident_scratch_bytes'] == 16*router.GIB
                assert 'resident_context' not in kwargs
            else:
                assert kwargs['resident_context'] is False and 'resident_scratch_bytes' not in kwargs
            events.append(name)
            return SimpleNamespace()
        return construct
    with patch.object(router, 'device_snapshot', snapshot), \
         patch.object(large_block_executors, 'build_executor', factory(router.EXPLICIT)), \
         patch.object(large_resident_executor_v1, 'build_executor', factory(router.RESIDENT)):
        for soft, expected in ((74*router.GIB, router.RESIDENT), (16*router.GIB, router.EXPLICIT)):
            events.clear()
            result = router.build_executor(FakeCudaModel(), bundle, metadata,
                resident_soft_bytes=soft, layout_cache=ProbeCache(),
                resource_check=lambda label: events.append('resource_check'))
            assert events == ['CPU_layout_ready', 'resource_check', 'live_snapshot', expected, 'release_probe_lease']
            assert result.capacity_route_receipt['selected_route'] == expected
            assert result.capacity_route_receipt['parent_id'] == 'synthetic'
        events.clear()
        with patch.object(large_resident_executor_v1, 'build_executor', side_effect=RuntimeError('test constructor failure')):
            try:
                router.build_executor(FakeCudaModel(), bundle, metadata,
                    resident_soft_bytes=74*router.GIB, layout_cache=ProbeCache(),
                    resource_check=lambda label: events.append('resource_check'))
            except RuntimeError as error:
                assert str(error) == 'test constructor failure'
            else:
                raise AssertionError('Constructor error must propagate')
        assert events == ['CPU_layout_ready', 'resource_check', 'live_snapshot', 'release_probe_lease']
    rejected(lambda: router.build_executor(torch.nn.Linear(1, 1), bundle, metadata,
        resident_soft_bytes=74*router.GIB, layout_cache=ProbeCache()))
    for bad in ([1604], [True], [], [1605, 1605]):
        rejected(lambda bad=bad: router.actual_counts(bundle, dict(episode_metadata=dict(native_nodes_per_tree=bad))))
    return dict(passed=True, routes_construct_original_factories=True,
        layout_precedes_live_snapshot=True, probe_lease_released=True,
        constructor_failure_propagates_without_fallback=True, public_CPU_model_rejected=True,
        incomplete_or_incoherent_prepared_counts_rejected=True,
        fake_counters_and_constructors_used_only_for_interface_test=True)


def fixture():
    from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1
    torch.manual_seed(71127)
    h, t, per = 1600, 3, 1605
    parents, samples, groups = [], [], []
    leaves, middles, roots, ranks = [], [], [], []
    for hold in range(t):
        offset = hold * per
        membership = ((np.arange(h) + hold*137)//400) % 4
        local = np.concatenate((h+membership, np.full(4,h+4),[-1])).astype(np.int64)
        parents.append(np.where(local >= 0, local+offset, local))
        samples.append(torch.arange(h)+offset)
        groups.append(torch.from_numpy(membership)+hold*4)
        leaves.append(np.arange(h)+offset)
        middles.append(np.arange(h,h+4)+offset)
        roots.append(np.asarray([h+4+offset]))
        ranks.append(membership+hold*4)
    parent = np.concatenate(parents)
    empty = np.empty(0,dtype=np.int64)
    all_leaves, all_middle, all_roots = map(np.concatenate,(leaves,middles,roots))
    layout = [(0,t,0,t*per,parent,(all_leaves,all_middle,all_roots),
        (empty,all_leaves,all_middle),(empty,np.concatenate(ranks),np.repeat(np.arange(t),4)),
        (all_middle,all_leaves),0,t)]
    genotype = torch.randint(0,2,(7,h),dtype=torch.int8)
    genotype[:,::17] = -1
    genotype[1] = genotype[0]
    panel = torch.rand(7,2).clamp_min(.1)
    panel /= panel.sum(dim=1,keepdim=True)
    bank = SimpleNamespace(parent=torch.from_numpy(parent),
        node_time=torch.tensor(([0.]*h+[300.]*4+[1000.])*t),
        sample_index=torch.stack(samples), edge_geometry=torch.randn(t*per,4)*.1,
        branch_support_true=torch.rand(t*per,5),
        reference_class=torch.cat((torch.arange(3).repeat_interleave(200),torch.full((1000,),-1))),
        sample_age=torch.zeros(h),spans=torch.tensor([1.,4517.,25482.]),
        site_hold=torch.tensor([0,0,0,2,2,2,2]),genotype=genotype,
        log_donor_allele_frequency=torch.rand(7,2,6).clamp_min(.1).log(),
        donor_call_mask=torch.rand(7,6)>.2,panel_allele_frequency=panel,site_entropy=torch.rand(7))
    episodes = SimpleNamespace(hold_span=bank.spans,member_hold=torch.arange(t),
        member_episode=torch.zeros(t,dtype=torch.long),episode_size=torch.tensor([t]),
        member_node=torch.arange(t)*per+h+4,
        flags=SimpleNamespace(node_flags=torch.randint(0,2,(t*per,2)).float()))
    cut = SimpleNamespace(group_offsets=tuple(i*4 for i in range(t+1)),
        representatives=torch.cat([torch.arange(h,h+4)+i*per for i in range(t)]),
        leaf_groups=torch.stack(groups),group_counts=torch.full((t*4,),400.),native_holds=t,haplotypes=h)
    bundle = SimpleNamespace(bank=bank,episodes=episodes,cut=cut,native_holds=t,parent_id='synthetic')
    return AmericanTreeLoomC3FullV1().float(),bundle,layout


def numerical_tests():
    from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward, geometry_from_spans
    from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
    from research.efficiency_20261027.architecture.exact_upward_residency.upward_residency import UpwardResidentExecution
    prototype,bundle,layout = fixture()
    focal_hold = torch.tensor([0,1,2,2,1,0])
    focal_column = torch.tensor([0,200,600,1000,1400,1599])
    labels = torch.tensor([0,1,2,1,0,2])
    def run(soft):
        receipt = router.select_capacity(native_nodes=bundle.bank.parent.numel(),native_holds=3,
            free_bytes=80*router.GIB,total_bytes=80*router.GIB,allocated_bytes=0,reserved_bytes=0,
            resident_soft_bytes=soft)
        cls = UpwardResidentExecution if receipt['selected_route']==router.RESIDENT else ExplicitVJPExecution
        executor = object.__new__(cls)
        executor.model = copy.deepcopy(prototype)
        executor.bank,executor.bundle,executor.device = bundle.bank,bundle,torch.device('cpu')
        executor.tick = lambda label: None
        executor.profile = NullProfileV1()
        executor.layout,executor.history_window,executor.hap_tile = layout,1024,64
        if cls is UpwardResidentExecution:
            def cpu_admission(self,plan,stage,live,future):
                assert self.device.type=='cpu' and live+future<=plan['maximum_simultaneous_bank_bytes']
                return dict(CPU_test_only=True,GPU_capacity_checked=False,stage=stage)
            executor.admit_residency = MethodType(cpu_admission,executor)
        model = executor.model
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
            betas=(.9,.999),eps=1e-8)
        optimizer.zero_grad(set_to_none=True)
        rng = torch.random.get_rng_state().clone()
        context = executor.context_forward()
        geometry = geometry_from_spans(bundle.bank.spans)
        with torch.no_grad():
            forward,_ = model.scan_context(context,geometry,direction='forward')
            reverse,_ = model.scan_context(context,geometry,direction='reverse')
            logits = model.classifier(torch.cat((context[focal_hold,focal_column],
                forward[focal_hold,focal_column],reverse[focal_hold,focal_column]),-1))
        loss,gradient,decoder = decode_parent_backward(model,context,geometry,
            focal_hold,focal_column,labels,tile_haplotypes=64)
        context.backward(gradient)
        gradients = {name:p.grad.detach().clone() for name,p in model.named_parameters()}
        assert len(gradients)==49 and all(bool(torch.isfinite(g).all()) for g in gradients.values())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        parameters = {name:p.detach().clone() for name,p in model.named_parameters()}
        adam = {(name,key):value.detach().clone() for name,p in model.named_parameters()
            for key,value in optimizer.state[p].items() if torch.is_tensor(value)}
        assert len(adam)==147 and torch.equal(rng,torch.random.get_rng_state())
        return dict(route=receipt['selected_route'],context=context.detach(),logits=logits,
            loss=loss.detach(),norm=norm.detach(),gradients=gradients,parameters=parameters,
            Adam=adam,decoder=decoder,receipt=executor.explicit_vjp_snapshot())
    baseline=run(16*router.GIB)
    resident=run(74*router.GIB)
    assert baseline['route']==router.EXPLICIT and resident['route']==router.RESIDENT
    errors={}
    for key in ('context','logits','loss','norm'):
        torch.testing.assert_close(baseline[key],resident[key],atol=1e-6,rtol=1e-5)
        errors[key]=float((baseline[key]-resident[key]).abs().max())
    for key in ('gradients','parameters','Adam'):
        assert baseline[key].keys()==resident[key].keys()
        errors[key]=0.
        for name,value in baseline[key].items():
            torch.testing.assert_close(value,resident[key][name],atol=1e-6,rtol=1e-4,msg=str(name))
            errors[key]=max(errors[key],float((value-resident[key][name]).abs().max()))
    assert torch.equal(baseline['logits'].argmax(-1),resident['logits'].argmax(-1))
    assert baseline['decoder']==resident['decoder'] and baseline['decoder']['native_focals']==6
    assert resident['receipt']['residency_plan']['native_nodes']==4815
    assert resident['receipt']['residency_plan']['largest_block_nodes']==4815
    assert baseline['receipt']['full_history_backward_calls']==resident['receipt']['full_history_backward_calls']==1
    return dict(passed=True,synthetic_holds=3,haplotypes=1600,native_nodes=4815,
        blocks=1,ragged512partition=True,haplotypes_per_gru=64,history_holds_per_window=1024,
        synthetic_targets=6,all49_gradients=True,all49_updated_parameters=True,all147_Adam_tensors=True,
        zero_argmax_mismatch=True,RNG_unchanged=True,max_abs_errors=errors,
        production_CUDA_guard_preserved=True,CPU_admission_override_is_test_only=True,
        full_parent_or_GPU_qualification=False)


def main():
    signal.alarm(120)
    start=time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result=dict(schema='TreeLoom_qualified_CPUcontext_capacity_router_CPU_v1',passed=True,
        scalar_boundaries=scalar_tests(),factory_interface=factory_tests(),tiny_model=numerical_tests(),
        scientific_payloads_opened=False,CUDA_initialized=torch.cuda.is_initialized(),
        production_changed=False,model_cuda_execution=False,
        wall_seconds=time.monotonic()-start,max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_refs=[dict(path=str(Path(__file__).with_name(name)),
            sha256=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            for name in ('router_v1.py','test_router_cpu_v1.py')])
    assert result['CUDA_initialized'] is False
    path=Path(__file__).with_name('CPU_RESULTS_v1.json')
    with path.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write('\n')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
