"""Synthetic complete-history CPU comparison including the real site MLP path."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from research.benchmark_rebuild_20261024.american.treeloom_speed.static_model_v2 import AmericanTreeLoomC3FullV1
from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import NullProfileV1
from research.efficiency_20261027.execution.explicit_vjp import ExplicitVJPExecution
from .boundary_fusion import BoundaryFusedExecution, first_device, forcing_device


def main():
    signal.alarm(120)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(70028)
    h, t, nodes_per_tree, groups_per_tree = 1600, 3, 1605, 4
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
    sites = 7
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
        sample_age=torch.zeros(h), spans=torch.tensor([1.,4517.,25482.]),
        site_hold=torch.tensor([0,0,0,2,2,2,2]), genotype=genotype,
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
        inherited_block = executor._block
        def counted_block(self, z):
            self.block_calls += 1
            return inherited_block(z)
        executor._block = MethodType(counted_block, executor)
        return executor

    original, fused = make(ExplicitVJPExecution), make(BoundaryFusedExecution)
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
        loss = (context.square()*torch.linspace(.1,1.,32)).mean()+.1*context.mean()
        loss.backward()
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

    left, right = run(original), run(fused)
    torch.testing.assert_close(left[0],right[0],atol=0,rtol=0)
    errors,unused = {},[]
    for name,value in left[1].items():
        other = right[1][name]
        assert (value is None)==(other is None),name
        if value is None:
            unused.append(name)
            continue
        torch.testing.assert_close(value,other,atol=1e-6,rtol=1e-4,msg=name)
        errors[name] = float((value-other).abs().max())
        if name.startswith(('forward_gru.','reverse_gru.','classifier.')):
            assert bool((other==.125).all()),name
    saved = left[2]['receipt']['retained_CPU_boundary_bytes']-right[2]['receipt']['retained_CPU_boundary_bytes']
    assert saved==n*32*4
    assert left[2]['block_calls']==right[2]['block_calls']==4*t
    assert right[2]['receipt']['connected_trunk_parameters']==37
    assert not unused
    assert not torch.cuda.is_initialized()
    receipt = dict(schema='exact_device_boundary_fusion_CPU_v1',passed=True,
        holds=t,haplotypes=h,nodes=n,tree_blocks=t,sites=sites,
        cut_groups=12,history_windows_per_direction=t,
        scope='Synthetic two-level trees with changed cuts across holds; all inherited categorical site MLP, mean/max/missing/empty-hold pooling, base, PyG sweeps, episode/cut and full bidirectional history paths.',
        exact_first_forcing_boundary_equality=True,
        context_max_abs_error=float((left[0]-right[0]).abs().max()),
        parameter_gradient_max_abs_errors=errors,connected_trunk_parameters=37,
        preserved_preexisting_decoder_gradient_tensors=12,unused_parameters=unused,
        RNG_parameters_and_buffers_unchanged=True,retained_CPU_bytes_saved=saved,
        block_call_counts_unchanged=True,original=left[2],fused=right[2],
        GPU_initialized=False,scientific_payload_read=False,optimizer_step=False,
        full_parent_or_GPU_qualification=False,
        source_sha256=hashlib.sha256(Path(__file__).with_name('boundary_fusion.py').read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-started,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    path = Path(__file__).with_name('CPU_results_v1.json')
    with path.open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({key:value for key,value in receipt.items()
                     if key not in ('parameter_gradient_max_abs_errors','original','fused')}))


if __name__=='__main__':
    main()
