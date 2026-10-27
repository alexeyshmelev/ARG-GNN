"""Tiny synthetic CPU checks for the actual European driver preprocessing route.

No real observation, checkpoint, target, fit, prediction, GPU or native executable
is opened or launched. Temporary synthetic observations are removed on exit.
"""
from collections import OrderedDict
from contextlib import ExitStack
import copy
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import tskit
from . import bindings_v1 as bridge

HERE = Path(__file__).parent
CHECKS = []


def check(name, ok=True):
    assert ok, name
    CHECKS.append(name)


def exact(a, b, name='value'):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert a.detach().numpy().tobytes() == b.detach().numpy().tobytes(), name
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    elif dataclasses.is_dataclass(a):
        assert type(a) is type(b), name
        for field in dataclasses.fields(a):
            if 'stamp' not in field.name:
                exact(getattr(a, field.name), getattr(b, field.name), name + '.' + field.name)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for key in a:
            exact(a[key], b[key], name + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), name
        for index, (left, right) in enumerate(zip(a, b)):
            exact(left, right, name + '.' + str(index))
    else:
        assert a == b, name


def fails(name, function):
    try:
        function()
    except (ValueError, AssertionError, KeyError):
        check(name)
    else:
        raise AssertionError(name + ' did not reject')


def fixture(folder):
    """952 samples, four donor classes, real age boundary and isolated native hold."""
    tables = tskit.TableCollection(40.)
    rows = []
    names = ('Ana', 'CHG', 'WHG', 'EHG')
    ages = (349., 350., 351., 700.)
    for index in range(952):
        query = index < 686
        cls = None if query else (index - 686) % 4
        age = 0. if query else ages[cls]
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=age)
        rows.append(dict(sample_index=index, observed_node_id=index,
            sample_id='synthetic_%d' % (index // 2), haplotype=index % 2,
            role='query' if query else 'pure_source_reference', observed_reference_class=cls,
            observed_cohort='present_day' if query else names[cls], age_generations=age))
    pairs = [tables.nodes.add_row(time=900.) for _ in range(476)]
    roots = [tables.nodes.add_row(time=1100.), tables.nodes.add_row(time=1200.)]
    for hold, root in ((0, roots[0]), (1, roots[1]), (3, roots[0])):
        for pair, node in enumerate(pairs):
            tables.edges.add_row(hold * 10., (hold + 1) * 10., root, node)
            for child in (pair * 2, pair * 2 + 1):
                tables.edges.add_row(hold * 10., (hold + 1) * 10., node, child)
    for position in (1., 11., 31.):
        tables.sites.add_row(position, 'A')
    tables.sort()
    ts = tables.tree_sequence()
    ts_path = folder / 'synthetic_inferred.trees'
    ts.dump(str(ts_path))
    def reference(path):
        return dict(path=str(path.relative_to(bridge.ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def save(name, value):
        path = folder / name
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        return reference(path)
    rows_ref = save('synthetic_samples.json', rows)
    vcf = folder / 'synthetic_observed.vcf'
    with vcf.open('w') as stream:
        stream.write('##fileformat=VCFv4.2\n')
        stream.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t')
        stream.write('\t'.join(row['sample_id'] for row in rows[::2]) + '\n')
        for site, position in enumerate((2, 12, 32)):
            calls = ['%s|%s' % ('.' if sample % 13 == 0 else sample % 4,
                                '.' if sample % 17 == 0 else (sample + site + 1) % 4)
                     for sample in range(476)]
            stream.write('1\t%d\t.\tA\tC,G,T\t.\tPASS\t.\tGT\t%s\n' % (position, '\t'.join(calls)))
    vcf_ref = reference(vcf)
    parent = 'synthetic_european_train_000'
    inference_ref = save('synthetic_inference.json', dict(
        status='post_five_iterations_inferred', parent_id=parent, split='train',
        sequence_length_bp=200000000, truth_or_oracle_read=False,
        input_observed_sha256={'genotypes.vcf': vcf_ref['sha256']},
        observed_sample_manifest=rows_ref, inferred_tree=reference(ts_path),
        post_five_iterations=dict(coverage_original_bp0=[0, 40], local_to_original_offset_bp=0,
            site_join=[dict(native_site_id=i, local_position_0based=p,
                original_position_0based=p, VCF_position_1based=p+1) for i, p in enumerate((1., 11., 31.))])))
    registry_ref = save('synthetic_registry.json', {'synthetic_only': True})
    records = dict(record_ids=np.array([0, 1, 2, 3], dtype=np.int64),
        sample_nodes=np.array([0, 685, 0, 0], dtype=np.int64),
        sample_indices=np.array([0, 685, 0, 0], dtype=np.int64),
        positions_bp0=np.array([1., 11., 31., 31.], dtype=np.float64),
        parent_id=np.asarray(parent),
        class_order=np.asarray(('Ana_nonBAA', 'CHG_nonBAA', 'WHG', 'EHG', 'Ana_viaBAA', 'CHG_viaBAA')))
    return parent, dict(inference_ref=inference_ref, observed_vcf_ref=vcf_ref,
        sampled_registry_ref=registry_ref), records


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check('CUDA_uninitialized_before', not torch.cuda.is_initialized())
    source_ref = bridge.file_ref(HERE / 'source_pins_v1.json')
    _, original = bridge.authenticate(source_ref)
    from research.efficiency_20261027.architecture.european_cohort_train import driver_v2
    from research.efficiency_20261027.architecture.european_cohort_candidate import observations as cohort
    from research.preprocessing_optimization_20261026.integration import hooks
    from research.preprocessing_optimization_20261026.tree import observations, process_observations, episodes, cut
    from . import observed_adapter_v1 as adapter
    owners = [importlib.import_module(module) for module, _ in bridge.OWNERS]
    originals = [getattr(owner, name) for owner, (_, name) in zip(owners, bridge.OWNERS)]
    kwargs = dict(source_manifest_ref=source_ref, max_rss_bytes=original['memory_envelope']['max_rss_bytes'],
        preparation_headroom_bytes=original['memory_envelope']['preparation_headroom_bytes'],
        train_inputs=original['train_inputs'], resource_check=lambda stage: None)
    fails('changed_manifest_hash_rejected', lambda: bridge.install(**dict(kwargs,
        source_manifest_ref=dict(source_ref, sha256='0' * 64))))
    fails('changed_memory_envelope_rejected', lambda: bridge.install(**dict(kwargs, max_rss_bytes=16*2**30)))
    fails('changed_TRAIN_refs_rejected', lambda: bridge.install(**dict(kwargs, train_inputs={})))
    with patch.object(owners[3], 'prepare_cut_plan_v1', lambda *args, **kw: None):
        fails('unexpected_last_owner_rejected_before_any_assignment', lambda: bridge.install(**kwargs))
        check('earlier_owners_unchanged_after_rejection', all(getattr(owner, name) is before
            for owner, (_, name), before in zip(owners[:3], bridge.OWNERS[:3], originals[:3])))
    with patch.object(owners[0], '_implementation', lambda *args: None):
        fails('changed_queue_helper_rejected', lambda: bridge.install(**kwargs))
    with patch.object(hooks, 'verify_bindings', side_effect=ValueError('synthetic verification failure')):
        fails('failed_post_assignment_verification_rolls_back', lambda: bridge.install(**kwargs))
    check('all_original_owners_restored_after_rollback', all(getattr(owner, name) is before
        for owner, (_, name), before in zip(owners, bridge.OWNERS, originals)))
    with tempfile.TemporaryDirectory(prefix='synthetic_route_', dir=HERE) as temporary:
        parent, item, records = fixture(Path(temporary))
        spec = copy.deepcopy(original)
        spec['train_inputs'] = {parent: item}
        events = []
        # Only the external registry reader is a test double. Real driver cache,
        # cohort guards, VCF loader, native tree conversion, encoder, episodes,
        # cut preparation and FrozenEuropeanBundle/CohortParent validators run.
        with patch.object(owners[0], 'sampled_records', return_value=records):
            load = driver_v2.observed_loader(spec, OrderedDict(), events.append)
            before = load(parent)
            receipt = bridge.install(**kwargs)
            check('installed_receipt_verifies', bridge.verify(receipt))
            code_names = {
                driver_v2.observed_loader.__code__: 'driver_observed_loader',
                cohort.load_relate_observation.__code__: 'cohort_load',
                cohort.prepare_observation.__code__: 'cohort_prepare',
                owners[0].prepare_observation.__code__: 'queue_prepare',
                adapter.load_relate_observation.__code__: 'qualified_queue_loader',
                hooks._process4_encode_block.__code__: 'qualified_process4_encoder',
                episodes.build_episodes.__code__: 'qualified_episodes',
                cut.prepare_cut_plan_v1.__code__: 'qualified_cut'}
            hits = set()
            def profile(frame, event, arg):
                if event == 'call' and frame.f_code in code_names:
                    hits.add(code_names[frame.f_code])
            previous_profile = sys.getprofile()
            try:
                sys.setprofile(profile)
                load = driver_v2.observed_loader(spec, OrderedDict(), events.append)
                after = load(parent)
            finally:
                sys.setprofile(previous_profile)
            check('actual_driver_route_hits_all_four_replacements', hits == set(code_names.values()))
            exact(before, after, 'actual_driver_complete_bundle_and_metadata')
            check('actual_952_sample_four_class_bundle_all_tensors_metadata_bitwise_equal')
            check('952_by8_cohort_channels_preserved', tuple(after[0].observed_cohort.shape) == (952, 8))
            check('all_four_native_holds_and_isolated_hold_preserved', after[0].native_holds == 4
                  and after[1]['observation_metadata']['isolated_zero_site_holds'] == [2])
            check('repeated_focal_draws_preserved', after[0].focal_hold.tolist() == [0, 1, 3, 3])
            check('original_resource_checks_and_process4_admission_run',
                  'TreeLoom_admit_process4_encoder' in events
                  and 'TreeLoom_after_complete_parent_preparation' in events)
            fails('duplicate_installation_rejected', lambda: bridge.install(**kwargs))
            changed = copy.deepcopy(receipt)
            changed['bindings'][1]['after']['partial_keywords']['max_rss_bytes'] -= 1
            fails('changed_binding_receipt_rejected', lambda: bridge.verify(changed))
            check('live_binding_receipt_still_verifies', bridge.verify(receipt))
    for owner, (_, name), before in zip(owners, bridge.OWNERS, originals):
        setattr(owner, name, before)
    check('all_process_local_bindings_restored_for_test_exit', all(getattr(owner, name) is before
        for owner, (_, name), before in zip(owners, bridge.OWNERS, originals)))
    check('CUDA_uninitialized_after', not torch.cuda.is_initialized())
    import psutil
    children = psutil.Process().children(recursive=True)
    trackers = [child for child in children if 'from multiprocessing.resource_tracker import main'
                in ' '.join(child.cmdline())]
    check('no_test_pool_workers_remain', len(children) == len(trackers) and len(trackers) <= 1)
    report = dict(schema='European_fit_preprocessing_CPU_contracts_v1', passed=True,
        source_manifest_ref=source_ref, checks=CHECKS, check_count=len(CHECKS),
        route_code_hits=sorted(hits), synthetic_samples=952, synthetic_sites=3, synthetic_holds=4,
        real_scientific_payloads_opened=False, targets_opened=False, GPU_initialized=False,
        full_parent_qualification_repeated=False, native_or_training_launches=0,
        qualification_limit='Tiny synthetic actual-route parity; prior full tree input remains American TRAIN000')
    path = HERE / 'CPU_TEST_RESULTS_v1.json'
    with path.open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps(dict(report_ref=bridge.file_ref(path), check_count=len(CHECKS), route_code_hits=sorted(hits))))


if __name__ == '__main__':
    main()
