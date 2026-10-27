"""Observed-only CPU numerical diagnostic for the explicit Transport stage VJPs."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import hashlib
import json
import resource
import time
import traceback
from dataclasses import fields
from pathlib import Path
import numpy as np
import torch
from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
from research.repairs.benchmark_recovery_20261020.runtime.episodes import build_episodes
from research.efficiency_20261027.architecture.treeloom_transport import TreeLoomTransport, prepare_transport_plan
from research.efficiency_20261027.architecture.staged_transport import StagedTransport


def observed_bank(t=7):
    packet = Path('research/scalability_20261024/execution/representative_slice.npz')
    with np.load(packet, allow_pickle=False) as z:
        n = int(z['node_offsets'][t])
        arrays = {}
        selected = z['site_hold'] < t
        for field in fields(MutationRoutedWholeTreeInputsV1):
            key = field.name
            if key in ('upward_levels', 'upward_children', 'upward_parent_rank', 'downward_levels'):
                names = sorted((k for k in z.files if k.startswith(key + '_')), key=lambda k: int(k.rsplit('_', 1)[1]))
                values = []
                for name in names:
                    if key == 'upward_parent_rank':
                        children = z[name.replace('upward_parent_rank', 'upward_children')]
                        values.append(z[name][children < n])
                    else:
                        values.append(z[name][z[name] < n])
                arrays[key] = tuple(values)
            elif key in ('parent', 'node_time', 'edge_geometry', 'branch_support_true', 'branch_support_sham'):
                arrays[key] = z[key][:n]
            elif key in ('genotype', 'log_donor_allele_frequency', 'donor_call_mask', 'panel_allele_frequency', 'site_entropy', 'site_hold'):
                arrays[key] = z[key][selected]
            elif key in ('sample_index', 'spans', 'readout_hold'):
                arrays[key] = z[key][:t]
            else:
                arrays[key] = z[key]
        counts = np.diff(z['node_offsets'][:t + 1]).tolist()
    return MutationRoutedWholeTreeInputsV1.from_arrays(arrays), counts, packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.manual_seed(9030)
    base = Path('research/efficiency_20261027/architecture')
    started = time.perf_counter()
    result = dict(schema='TreeLoom_Transport_staged_CPU_parity_v1', status='running',
                  ancestry_targets_opened=False, optimizer_updates=0,
                  logits_atol=1e-6, logits_rtol=0., gradients_atol=1e-6, gradients_rtol=1e-4,
                  source_sha256={name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                                 for name in ('treeloom_transport.py', 'bounded_transport.py', 'staged_transport.py', 'test_staged_transport.py')},
                  trials=[], limitation='Seven original holds for candidate-vs-candidate numerical parity only; no ancestry accuracy, GPU performance or full-parent resource qualification.')
    try:
        bank, counts, packet = observed_bank()
        episodes, metadata = build_episodes(bank, max_payload_bytes=2**30, isolated_hold_indices=(0,))
        plan = prepare_transport_plan(bank.parent, episodes.flags.node_flags)
        model = TreeLoomTransport()
        # Include every haplotype tile, both sequence endpoints, and duplicate
        # coordinates. These are numerical probes, with no ancestry targets.
        columns = [0, 1, 127, 128, 255, 256, 383, 384, 511, 512, 639, 640,
                   767, 768, 895, 896, 1023, 1024, 1151, 1152, 1279, 1280,
                   1407, 1408, 1535, 1536, 1598, 1599]
        hold = torch.tensor([i % 7 for i in range(len(columns))] + [0, 6, 0, 6], dtype=torch.long)
        column = torch.tensor(columns + [0, 1599, 0, 1599], dtype=torch.long)
        weights = torch.randn(len(hold), 3)
        result.update(native_holds=7, native_nodes=len(bank.parent), sites=len(bank.genotype),
                      haplotypes=1600, focal_count=len(hold), isolated_holds=[0],
                      duplicate_focals=True, endpoint_columns=[0, 1599],
                      reference_focal_count=int((bank.reference_class[column] >= 0).sum()),
                      query_focal_count=int((bank.reference_class[column] < 0).sum()),
                      packet_sha256=hashlib.sha256(packet.read_bytes()).hexdigest())
        reference = model.forward_all_samples(bank, plan)[hold, column]
        reference_objective = (reference * weights).sum()
        reference_objective.backward()
        reference_gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
        reference = reference.detach()
        print('stage: monolithic observed CPU reference complete', flush=True)
        for cache in (False, True):
            for tree_block, scan_block in ((3, 3), (128, 1024)):
                candidate = copy.deepcopy(model)
                candidate.zero_grad(set_to_none=True)
                executor = StagedTransport(candidate, bank, episodes.flags.node_flags, counts,
                                           holds_per_block=tree_block, haplotypes_per_tile=128,
                                           scan_holds_per_block=scan_block, cache_first_nodes=cache)
                trial = dict(cache_first_nodes=cache, holds_per_block=tree_block,
                             scan_holds_per_block=scan_block, haplotypes_per_tile=128,
                             gradient_details=[], status='running')
                result['trials'].append(trial)
                before = time.perf_counter()
                actual = executor.forward_focals(hold, column)
                forward_seconds = time.perf_counter() - before
                objective = (actual * weights).sum()
                objective.backward()
                assert all(p.grad is None for p in candidate.parameters())
                before = time.perf_counter()
                executor.backward_focals(actual.grad)
                backward_seconds = time.perf_counter() - before
                trial.update(forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                             logits_max_abs_error=float((actual.detach() - reference).abs().max()),
                             argmax_equal=bool((actual.argmax(-1) == reference.argmax(-1)).all()),
                             objective=float(objective.detach()), reference_objective=float(reference_objective.detach()),
                             objective_abs_error=abs(float(objective.detach() - reference_objective.detach())),
                             named_host_array_accounting=executor.named_host_array_accounting())
                for name, p in candidate.named_parameters():
                    expected = reference_gradients[name]
                    row = dict(name=name, gradient_present=p.grad is not None)
                    if p.grad is not None:
                        row.update(max_abs_error=float((p.grad - expected).abs().max()), finite=bool(torch.isfinite(p.grad).all()))
                    trial['gradient_details'].append(row)
                trial['gradients_max_abs_error'] = max(row.get('max_abs_error', float('inf')) for row in trial['gradient_details'])
                trial['parameter_tensors'] = len(trial['gradient_details'])
                torch.testing.assert_close(actual, reference, atol=1e-6, rtol=0.)
                assert trial['argmax_equal'] and torch.isfinite(actual).all()
                for name, p in candidate.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    torch.testing.assert_close(p.grad, reference_gradients[name], atol=1e-6, rtol=1e-4, msg=name)
                assert executor.state is None
                trial['status'] = 'passed'
                print(json.dumps({k: v for k, v in trial.items() if k not in ('gradient_details', 'named_host_array_accounting')}), flush=True)
                del executor, candidate, actual, objective
        assert not torch.cuda.is_initialized()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', exception_type=type(exc).__name__, exception=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter() - started,
                      process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      CUDA_initialized=torch.cuda.is_initialized())
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
