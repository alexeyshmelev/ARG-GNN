"""FP32 qualification of a separately versioned executor; no ancestry fitting.

Each paired optimizer step uses a fixed synthetic linear/squared-logit probe,
not labels. Fresh matching models are used for each independent case/mode.
"""
import copy
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from . import window_model_frozen_v1 as old
from . import window_model_focal_executor_v2 as new
from .test_window_model import toy, permute_nodes

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FOCAL_KEYS = ('focal_tree_index', 'focal_sample_index', 'focal_position_local_bp',
              'focal_genotypes', 'focal_allele_count')


def select_focals(a, index):
    a = copy.deepcopy(a)
    for key in FOCAL_KEYS:
        a[key] = a[key][index]
    return a


def coordinate_toy():
    a = select_focals(toy(), [1])
    context = dict(context_coordinate_local_bp=np.array([120., 20., 20., 50., 250.]),
        context_tree_index=np.array([1, 0, 0, 0, 2]), context_window_index=np.zeros(5, dtype=np.int64),
        context_quadrature_weight=np.array([0., .25, .25, .25, .25]),
        context_proposal_density_per_bp=np.array([0., 1/400, 1/400, 1/400, 1/400]),
        context_forced_focal=np.array([True, False, False, False, False]),
        window_context_domain_local_bp=np.array([[0., 400.]]), window_context_kind=np.array([1]))
    return old.coordinate_observations(a, context)


def singleton():
    a = toy()
    a.update(parent=np.array([-1]), node_time=np.array([0.]), node_tree_index=np.array([0]),
        leaf_node_index=np.array([[0]]), leaf_features=a['leaf_features'][:1],
        tree_interval_local_bp=np.array([[0., 1.]]), tree_window_index=np.array([0]),
        tree_core_mask=np.array([True]), focal_tree_index=np.array([0]), focal_sample_index=np.array([0]),
        focal_position_local_bp=np.array([.5]), focal_genotypes=np.array([[-1]]), focal_allele_count=np.array([2]),
        reference_class=np.array([-1]), sample_age=np.array([0.]), observed_cohort=np.array([0]),
        query_mask=np.array([True]), site_tree_index=np.empty(0, dtype=np.int64),
        site_position_local_bp=np.empty(0), site_genotypes=np.empty((0, 1), dtype=np.int64),
        site_allele_count=np.empty(0, dtype=np.int64))
    return a


def edgeless_focal_with_edged_sources():
    a = select_focals(toy(), [1])
    # Tree one has four singleton roots; trees zero and two retain edges.
    nodes = np.flatnonzero(a['node_tree_index'] == 1)
    keep = np.ones(len(a['parent']), dtype=bool)
    keep[nodes[-1]] = False
    kept = np.flatnonzero(keep)
    mapping = np.full(len(keep), -1, dtype=np.int64)
    mapping[kept] = np.arange(len(kept))
    a['parent'][nodes] = -1
    a['parent'] = np.where(a['parent'][keep] < 0, -1, mapping[a['parent'][keep].clip(min=0)])
    for key in ('node_time', 'node_tree_index'):
        a[key] = a[key][keep]
    a['leaf_node_index'] = mapping[a['leaf_node_index']]
    return a


def four_real_packets():
    packet_root = WORK / 'diagnostic_packets'
    manifest = json.loads((packet_root / 'manifest.json').read_text())
    paths = [packet_root / row['packet_path'] for row in manifest['fit'][:4]]
    return old.collate_observations([old.load_observations(path) for path in paths]), paths


def max_error(left, right):
    return float((left - right).abs().max()) if left.numel() else 0.


def paired(name, arrays, mode, width=32):
    source = old.prepare_observations(arrays)
    plan = new.prepare_executor(source).to('cpu')
    torch.manual_seed(711)
    first = old.WindowTreeLoom(width=width)
    second = new.WindowTreeLoom(width=width)
    second.load_state_dict(first.state_dict(), strict=True)
    first_opt = torch.optim.AdamW(first.parameters(), lr=1e-3, weight_decay=.01, eps=1e-8)
    second_opt = torch.optim.AdamW(second.parameters(), lr=1e-3, weight_decay=.01, eps=1e-8)
    output = first(source, mode=mode)
    candidate = second(plan, mode=mode)
    torch.testing.assert_close(candidate, output, atol=2e-6, rtol=2e-5)
    coefficient = torch.linspace(-.7, 1.1, max(output.numel(), 1), dtype=output.dtype)[:output.numel()].reshape_as(output)
    # Scale the arbitrary numerical probe so clipping is exercised.
    loss = (output * coefficient).sum() * 10 + output.square().sum()
    candidate_loss = (candidate * coefficient).sum() * 10 + candidate.square().sum()
    loss.backward()
    candidate_loss.backward()
    max_grad = 0.
    max_update = 0.
    strict_update_parity = True
    update_failure = None
    gradient_none = []
    for (key, parameter), (key2, candidate_parameter) in zip(first.named_parameters(), second.named_parameters()):
        assert key == key2
        if (parameter.grad is None) != (candidate_parameter.grad is None):
            raise AssertionError(f'{name}/{mode}: None-vs-zero gradient mismatch: {key}')
        if parameter.grad is None:
            gradient_none.append(key)
            continue
        assert torch.isfinite(parameter.grad).all() and torch.isfinite(candidate_parameter.grad).all()
        torch.testing.assert_close(candidate_parameter.grad, parameter.grad, atol=2e-5, rtol=2e-4,
                                   msg=lambda message: f'{name}/{mode} gradient {key}: {message}')
        max_grad = max(max_grad, max_error(candidate_parameter.grad, parameter.grad))
    first_norm = torch.nn.utils.clip_grad_norm_(first.parameters(), .5)
    second_norm = torch.nn.utils.clip_grad_norm_(second.parameters(), .5)
    assert float(first_norm) > .5 or not output.numel()
    torch.testing.assert_close(second_norm, first_norm, atol=2e-5, rtol=2e-5)
    first_opt.step()
    second_opt.step()
    for (key, parameter), (_, candidate_parameter) in zip(first.named_parameters(), second.named_parameters()):
        strict_match = torch.allclose(candidate_parameter, parameter, atol=3e-6, rtol=2e-5)
        strict_update_parity = strict_update_parity and strict_match
        parameter_error = max_error(candidate_parameter, parameter)
        if parameter_error > max_update:
            max_update = parameter_error
            update_failure = key if not strict_match else update_failure
        first_state = first_opt.state[parameter]
        candidate_state = second_opt.state[candidate_parameter]
        assert first_state.keys() == candidate_state.keys()
        for state_key in first_state:
            torch.testing.assert_close(candidate_state[state_key], first_state[state_key], atol=2e-7, rtol=2e-4)
    with torch.no_grad():
        updated = first(source, mode=mode)
        updated_candidate = second(plan, mode=mode)
        torch.testing.assert_close(updated_candidate, updated, atol=3e-6, rtol=3e-5)
    return dict(case=name, mode=mode, passed=strict_update_parity, strict_update_parity=strict_update_parity,
                update_failure_parameter=update_failure, logits_max_abs_error=max_error(candidate, output),
                gradients_max_abs_error=max_grad, updated_parameters_max_abs_error=max_update,
                updated_logits_max_abs_error=max_error(updated_candidate, updated),
                updated_probe_loss_abs_error=float(abs(((updated_candidate * coefficient).sum() * 10 + updated_candidate.square().sum()) -
                                                       ((updated * coefficient).sum() * 10 + updated.square().sum()))),
                gradient_norm_before_clip=float(first_norm), none_gradient_parameters=gradient_none,
                source_trees=len(source.leaves), focal_trees=len(plan.trees), source_nodes=len(source.parent),
                focal_nodes=len(plan.nodes), focal_rows=len(source.focal_tree),
                temporal_source_columns=sum(len(route.sources) for route in plan.windows),
                temporal_target_rows=sum(len(route.targets) for route in plan.windows))


def main():
    torch.set_num_threads(2)
    coordinate = coordinate_toy()
    interval = select_focals(toy(), [1, 1, 2])
    repeated = copy.deepcopy(coordinate)
    for key in FOCAL_KEYS:
        repeated[key] = repeated[key][[0, 0]]
    cases = [('interval_all_targets', toy()), ('interval_subset_repeated_focal', interval),
             ('coordinate_duplicate_draws', coordinate), ('coordinate_multiple_rows_one_tree', repeated),
             ('mixed_multiple_windows', old.collate_observations([coordinate, interval])),
             ('singleton_no_sites_missing_focal', singleton()),
             ('edgeless_focal_with_edged_sources', edgeless_focal_with_edged_sources()),
             ('permuted_native_nodes', permute_nodes(coordinate, np.random.default_rng(9).permutation(len(coordinate['parent'])))),
             ('empty_focal_rows', select_focals(toy(), [])),
             ('real_uniform_packet', old.load_observations(WORK / 'data/train000_uniform_coordinate_window_v2.npz'))]
    four, paths = four_real_packets()
    cases.append(('four_distinct_real_fit_packets', four))
    report = dict(schema='focal_executor_v2_FP32_CPU_parity_v1', tests=[], source_hashes={},
                  CPU_threads=2, labels_used=False, ancestry_fitting=False, CUDA_initialized=False,
                  optimizer_updates_per_fresh_model=1, optimizer_probe='fixed label-free linear plus squared logits',
                  optimizer=dict(name='AdamW', lr=.001, weight_decay=.01, eps=1e-8, clip_norm=.5),
                  real_four_packet_paths=[str(path.relative_to(WORK)) for path in paths])
    for filename in ('window_model_frozen_v1.py', 'window_model_focal_executor_v2.py', 'test_focal_executor_v2.py'):
        report['source_hashes'][filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    assert report['source_hashes']['window_model_frozen_v1.py'] == 'e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12'
    started = time.perf_counter()
    output_path = ROOT / 'FOCAL_EXECUTOR_V2_PARITY.json'
    try:
        for name, arrays in cases:
            for mode in ('structure', 'leaf_only', 'no_cross'):
                try:
                    result = paired(name, arrays, mode)
                except Exception as error:
                    result = dict(case=name, mode=mode, passed=False, failure=f'{type(error).__name__}: {error}')
                report['tests'].append(result)
                print(json.dumps(result), flush=True)
        report['all_passed'] = all(result['passed'] for result in report['tests'])
    except Exception as error:
        report['all_passed'] = False
        report['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        report['peak_RSS_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['CUDA_initialized'] = torch.cuda.is_initialized()
        output_path.write_text(json.dumps(report, indent=2) + '\n')
    assert not report['CUDA_initialized']
    print(json.dumps({key: value for key, value in report.items() if key != 'tests'}, indent=2))


if __name__ == '__main__':
    main()
