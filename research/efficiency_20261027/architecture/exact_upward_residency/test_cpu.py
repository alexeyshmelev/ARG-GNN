"""CPU edge/parity proof only; production GPU admission is never bypassed.

Synthetic CPU executors are built with object.__new__ and a local test-only
capacity recorder. The public factory keeps inherited CUDA and source guards.
"""
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
import weakref
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
from research.efficiency_20261027.architecture.exact_boundary_fusion.boundary_fusion import BoundaryFusedExecution, first_device, forcing_device
from .upward_residency import UpwardResidentExecution
from .memory_v1 import residency_plan, capacity_check, ResidencyLedger, DEFAULT_SCRATCH_BYTES


def parity_case(t):
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

    def make(cls):
        executor = object.__new__(cls)
        executor.model = copy.deepcopy(model)
        executor.bank, executor.bundle, executor.device = bank, bundle, torch.device('cpu')
        executor.tick = lambda label: None
        executor.profile = NullProfileV1()
        executor.layout, executor.history_window, executor.hap_tile = layouts, 1, 32
        executor.block_calls = 0
        executor.capacity_stages = []
        if cls is UpwardResidentExecution:
            def cpu_capacity(self, plan, stage, live, future):
                assert self.device.type == "cpu"
                assert 0 <= live <= plan["maximum_simultaneous_bank_bytes"]
                assert 0 <= future <= plan["maximum_simultaneous_bank_bytes"]-live
                self.capacity_stages.append(dict(stage=stage, live_bank_bytes=live, future_bank_bytes=future))
                return dict(GPU_capacity_checked=False, CPU_test_only=True, stage=stage)
            executor.admit_residency = MethodType(cpu_capacity, executor)
        inherited_block = executor._block
        def counted_block(self, z):
            self.block_calls += 1
            return inherited_block(z)
        executor._block = MethodType(counted_block, executor)
        return executor

    original, fused, resident = (make(cls) for cls in
        (ExplicitVJPExecution, BoundaryFusedExecution, UpwardResidentExecution))
    with torch.no_grad():
        for z in layouts:
            up, out = original._first(z)
            device_up, device_out = first_device(fused, z)
            force = original._forcing(z, up, out)
            device_force = forcing_device(fused, z, device_up, device_out)
            for left, right in [(up,device_up),(out,device_out),(force,device_force)]:
                torch.testing.assert_close(left,right,atol=0,rtol=0)
    original.block_calls = fused.block_calls = 0

    def run(executor):
        before = {name:p.detach().clone() for name,p in executor.model.named_parameters()}
        buffers = {name:p.detach().clone() for name,p in executor.model.named_buffers()}
        for name,p in executor.model.named_parameters():
            if name.startswith(('forward_gru.','reverse_gru.','classifier.')):
                p.grad = torch.full_like(p,.125)
        rng = torch.random.get_rng_state().clone()
        begin = time.monotonic()
        context = executor.context_forward()
        saved_up = []
        if isinstance(executor, UpwardResidentExecution):
            saved_up = [weakref.ref(tensor) for tensor in context.grad_fn.saved_tensors[3:3+t]]
            assert len(saved_up) == t and all(pointer() is not None for pointer in saved_up)
        loss = (context.square()*torch.linspace(.1,1.,32)).mean()+.1*context.mean()
        loss.backward()
        assert all(pointer() is None for pointer in saved_up), "Saved upward blocks must release with autograd"
        assert torch.equal(rng,torch.random.get_rng_state())
        for name,p in executor.model.named_parameters():
            assert torch.equal(p.detach(),before[name]),name
        for name,p in executor.model.named_buffers():
            assert torch.equal(p.detach(),buffers[name]),name
        gradients = {name:None if p.grad is None else p.grad.detach().clone()
                     for name,p in executor.model.named_parameters()}
        return context.detach(), gradients, dict(loss=float(loss.detach()),
            seconds=time.monotonic()-begin,block_calls=executor.block_calls,
            receipt=executor.explicit_vjp_snapshot())

    base, left, right = run(original), run(fused), run(resident)
    torch.testing.assert_close(base[0],left[0],atol=0,rtol=0)
    torch.testing.assert_close(left[0],right[0],atol=0,rtol=0)
    errors,unused = {},[]
    for name,value in left[1].items():
        other = right[1][name]
        baseline = base[1][name]
        assert (value is None)==(other is None)==(baseline is None),name
        if value is None:
            unused.append(name)
            continue
        torch.testing.assert_close(baseline,value,atol=1e-6,rtol=1e-4,msg=name)
        torch.testing.assert_close(value,other,atol=1e-6,rtol=1e-4,msg=name)
        errors[name] = float((value-other).abs().max())
        if name.startswith(('forward_gru.','reverse_gru.','classifier.')):
            assert bool((other==.125).all()),name
    saved = left[2]['receipt']['retained_CPU_boundary_bytes']-right[2]['receipt']['retained_CPU_boundary_bytes']
    assert saved==n*32*4
    assert base[2]['block_calls']==left[2]['block_calls']==right[2]['block_calls']==4*t
    lifetime = right[2]['receipt']['residency_lifetime']
    assert lifetime['peak_bank_bytes'] == 2*n*32*4
    assert lifetime['direct_gradient_bytes'] == lifetime['future_bank_bytes'] == 0
    history = [row for row in resident.capacity_stages if row['stage']=='explicit_complete_history_backward_begin']
    assert len(history)==1 and history[0]['live_bank_bytes']==2*n*32*4 and history[0]['future_bank_bytes']==0
    releases = [row['live_bank_bytes'] for row in resident.capacity_stages if row['stage']=='boundary_fused_first_block_released']
    assert releases == [2*n*32*4-(index+1)*nodes_per_tree*32*4 for index in range(t)]
    assert right[2]['receipt']['connected_trunk_parameters']==37
    assert not unused
    assert not torch.cuda.is_initialized()
    receipt = dict(schema='exact_upward_direct_residency_CPU_case_v1',passed=True,
        holds=t,haplotypes=h,nodes=n,tree_blocks=t,sites=sites,
        cut_groups=t*4,history_windows_per_direction=t,
        scope='Synthetic two-level trees with changed cuts across holds; all inherited categorical site MLP, mean/max/missing/empty-hold pooling, base, PyG sweeps, episode/cut and full bidirectional history paths.',
        exact_first_forcing_boundary_equality=True,
        context_max_abs_error=float((left[0]-right[0]).abs().max()),
        parameter_gradient_max_abs_errors=errors,connected_trunk_parameters=37,
        preserved_preexisting_decoder_gradient_tensors=12,unused_parameters=unused,
        RNG_parameters_and_buffers_unchanged=True,retained_CPU_bytes_saved=saved,
        block_call_counts_unchanged=True,original=base[2],boundary_fused=left[2],resident=right[2],
        capacity_stages=resident.capacity_stages,saved_upward_released_after_backward=True,
        GPU_initialized=False,scientific_payload_read=False,optimizer_step=False,
        full_parent_or_GPU_qualification=False,
        source_sha256=hashlib.sha256(Path(__file__).with_name('upward_residency.py').read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return receipt


def edge_cases():
    G = 2**30
    def rejected(function):
        try:
            function()
        except (ValueError, AssertionError):
            return True
        raise AssertionError("Expected explicit rejection")
    arguments = dict(free_bytes=80*G, total_bytes=100*G,
        allocated_bytes=10*G, reserved_bytes=20*G,
        future_bank_bytes=74*G, scratch_bytes=16*G)
    exact = capacity_check(**arguments)
    assert exact['capacity_margin_bytes'] == 0
    rejected(lambda: capacity_check(**dict(arguments, future_bank_bytes=74*G+1)))
    rejected(lambda: capacity_check(**dict(arguments, scratch_bytes=16*G-1)))
    rejected(lambda: capacity_check(**dict(arguments, allocated_bytes=21*G)))
    rejected(lambda: capacity_check(**dict(arguments, total_bytes=99*G)))
    tensor = torch.empty((10,32), dtype=torch.float32)
    ledger = ResidencyLedger(dict(upward_bank_bytes=tensor.numel()*4))
    ledger.add_upward(tensor)
    rejected(lambda: ledger.add_upward(tensor))
    rejected(lambda: ledger.add_direct(tensor))
    rejected(lambda: ledger.add_direct(torch.empty((20,32))[:10]))
    rejected(lambda: ledger.add_direct(torch.empty((10,32),dtype=torch.float64)))
    direct = torch.empty_like(tensor)
    ledger.add_direct(direct)
    ledger.require_both_complete()
    ledger.release_direct(direct)
    ledger.require_direct_released()
    rejected(lambda: ledger.release_direct(direct))
    model = torch.nn.Linear(1,1)
    executor = SimpleNamespace(bank=SimpleNamespace(parent=torch.arange(10)),
        bundle=SimpleNamespace(cut=SimpleNamespace(native_holds=2)),
        model=model, device=torch.device('cpu'), resident_scratch_bytes=16*G,
        layout=[(0,1,0,4),(1,2,4,10)])
    plan = residency_plan(executor)
    assert plan['upward_bank_bytes']==10*32*4
    assert plan['largest_block_nodes']==6
    executor.layout=[(0,1,0,4),(1,2,3,10)]
    rejected(lambda: residency_plan(executor))
    executor.layout=[(0,1,0,4)]
    rejected(lambda: residency_plan(executor))
    executor.layout=[(0,1,0,4),(1,2,4,10)]
    executor.resident_scratch_bytes=16*G-1
    rejected(lambda: residency_plan(executor))
    # Arithmetic only: no scientific array or GPU allocation.
    native_nodes=205369272
    assert native_nodes*32*4 == 26287266816
    assert 2*native_nodes*32*4+16*G == 69754402816
    return dict(passed=True, capacity_exact_boundary=True, insufficient_one_byte_rejected=True,
        scratch_reduction_rejected=True, incoherent_counters_rejected=True,
        duplicate_storage_rejected=True, hidden_larger_backing_storage_rejected=True,
        wrong_dtype_rejected=True, duplicate_release_rejected=True,
        incomplete_or_overlapping_native_partitions_rejected=True,
        unequal_block_sizes_supported=True, TRAIN000_bank_arithmetic_only=True)


def main():
    signal.alarm(120)
    started=time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    edges=edge_cases()
    cases=[parity_case(holds) for holds in (1,3,5)]
    assert not torch.cuda.is_initialized()
    report=dict(schema='exact_upward_direct_residency_CPU_v1',passed=True,
        edge_checks=edges,cases=cases,holds_tested=[1,3,5],
        singleton_episode_and_empty_holds_exercised=True,
        all37_trunk_gradients_and12_preserved_decoder_gradients=True,
        GPU_initialized=False,scientific_payload_read=False,optimizer_step=False,
        production_changed=False,full_parent_or_GPU_qualification=False,
        CPU_capacity_recorder_is_test_only=True,
        wall_seconds=time.monotonic()-started,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_refs=[dict(path=str(Path(__file__).with_name(name)),
            sha256=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest())
            for name in ['memory_v1.py','upward_residency.py','large_resident_executor_v1.py','test_cpu.py']])
    path=Path(__file__).with_name('CPU_results_v1.json')
    with path.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,cases=len(cases),wall_seconds=report['wall_seconds'],
        maxrss_bytes=report['maxrss_bytes'],context_max_abs_error=max(c['context_max_abs_error'] for c in cases),
        parameter_gradient_max_abs_error=max(max(c['parameter_gradient_max_abs_errors'].values()) for c in cases),
        GPU_initialized=False)))


if __name__=='__main__':
    main()
