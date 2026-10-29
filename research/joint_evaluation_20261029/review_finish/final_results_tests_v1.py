"""Bounded audit tests: synthetic truth only; original public prediction checks."""
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from research.joint_evaluation_20261029.review_finish import final_results_audit_v1 as a


def rejected(call, exception=ValueError):
    try:
        call()
    except exception:
        return
    raise AssertionError('Expected rejection did not occur')


def main():
    passed = []
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    p = np.eye(3, dtype=np.float32)[[0, 1, 1, 2, 2, 2]]
    result = a.metrics(p, y, a.C3)
    assert result['confusion'] == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert result['accuracy'] == 4/6 and result['Brier'] == 4/6
    a.near(result['macro_F1'], (2/3 + 0.5 + 0.8)/3, 'hand-computed F1')
    assert result['NLL'] is None and result['NLL_status'] == 'positive_infinity' and result['zero_truth_probability_count'] == 2
    passed.append('hand_calculated_confusion_accuracy_per_class_macro_F1_Brier_and_infinite_NLL')
    soft = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
    finite = a.metrics(soft, np.array([0, 1]), a.C3)
    a.near(finite['NLL'], -(math.log(float(soft[0, 0]))+math.log(float(soft[1, 1])))/2, 'hand finite NLL')
    assert finite['NLL_status'] == 'finite' and finite['per_class']['AFR']['F1'] == 0
    passed.append('finite_raw_NLL_and_fixed_class_absent_class_zero_F1')
    for bad in (np.full((2, 3), 0.2, dtype=np.float32), np.array([[1, 0, 0], [np.nan, 0, 0]], dtype=np.float32)):
        rejected(lambda: a.metrics(bad, np.array([0, 1]), a.C3))
    rejected(lambda: a.metrics(soft, np.array([0, 3]), a.C3))
    passed.append('invalid_probabilities_and_out_of_domain_labels_rejected')
    s = a.summary([1, 2, 3]);assert s == dict(mean=2., sample_SD=1., parents=3)
    rejected(lambda: a.check_summary(dict(mean=2., sample_SD=math.sqrt(2/3), parents=3), [1, 2, 3], 'wrong population SD'))
    passed.append('equal_parent_sample_SD_uses_ddof1')
    registry = dict(record_ids=np.arange(3, dtype=np.int64), sample_nodes=np.arange(3, dtype=np.int64),
        sample_indices=np.array([600, 600, 601], dtype=np.int64), positions_bp0=np.array([9.999, 10., 29999999.], dtype=np.float64),
        parent_id=np.asarray('synthetic'), class_order=np.asarray(a.C3))
    a.canonical(registry, 'synthetic', a.C3, 3)
    pred = dict(registry, probabilities=np.eye(3, dtype=np.float32))
    a.prediction_identity(pred, registry, a.C3)
    wrong = copy.deepcopy(pred);wrong['record_ids'] = wrong['record_ids'][::-1]
    rejected(lambda: a.prediction_identity(wrong, registry, a.C3))
    wrong = copy.deepcopy(pred);wrong['class_order'] = np.array(a.C3[::-1])
    rejected(lambda: a.prediction_identity(wrong, registry, a.C3))
    passed.append('permuted_keys_and_swapped_class_order_rejected')
    labels = np.zeros((2, 1000), dtype=np.uint8);labels[1, 0] = 1;labels[1, 1] = 2
    truth = dict(intervals_bp=np.array([[0., 10.], [10., 30000000.]]), query_sample_indices=np.arange(600, 1600, dtype=np.int64),
                 source_classes_0based=labels, parent_id=np.array('synthetic'), class_order=np.array(a.C3))
    assert np.array_equal(a.american_truth(registry, truth), [0, 1, 2])
    wrong = copy.deepcopy(truth);wrong['parent_id'] = np.array('different')
    rejected(lambda: a.american_truth(registry, wrong))
    wrong = copy.deepcopy(truth);wrong['intervals_bp'][1, 0] = 11
    rejected(lambda: a.american_truth(registry, wrong))
    wrong_registry = copy.deepcopy(registry);wrong_registry['positions_bp0'][2] = 30000000
    rejected(lambda: a.american_truth(wrong_registry, truth))
    wrong_registry = copy.deepcopy(registry);wrong_registry['sample_indices'][2] = 599
    rejected(lambda: a.american_truth(wrong_registry, truth))
    passed.append('independent_American_half_open_boundaries_raw_sample_join_and_wrong_parent_gap_extrapolation_rejection')
    import tskit, msprime
    tables = tskit.TableCollection(20.)
    for name in a.C4:tables.populations.add_row()
    for i in range(3222):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0., population=0)
    cen = [tables.nodes.add_row(flags=msprime.NODE_IS_CEN_EVENT, time=350., population=i) for i in range(4)]
    root = tables.nodes.add_row(time=400., population=0)
    for node in cen:tables.edges.add_row(0., 20., root, node)
    tables.edges.add_row(0., 10., cen[0], 0);tables.edges.add_row(10., 20., cen[1], 0)
    tables.edges.add_row(0., 20., cen[2], 1);tables.sort()
    seq = tables.tree_sequence()
    manifest = dict(census_time=350, sample_index_to_raw_node_id=list(range(3222)), population_id_to_name={str(i):name for i,name in enumerate(a.C4)})
    rows = [dict(sample_index='0', rebased_midpoint_bp='9.999'), dict(sample_index='0', rebased_midpoint_bp='10'), dict(sample_index='1', rebased_midpoint_bp='15')]
    assert np.array_equal(a.census_truth(seq, manifest, rows), [0, 1, 2])
    wrong = copy.deepcopy(rows);wrong[2]['rebased_midpoint_bp'] = '20'
    rejected(lambda: a.census_truth(seq, manifest, wrong))
    wrong = dict(manifest, census_time=349)
    rejected(lambda: a.census_truth(seq, wrong, rows))
    passed.append('independent_census350_join_on_synthetic_tskit_with_boundary_and_extrapolation_checks')
    with tempfile.TemporaryDirectory(prefix='final_results_fixture_', dir=a.REVIEW) as temp:
        root = Path(temp);audit = a.Audit(root)
        product = root/'product.json';product.write_text('{"complete": true}\n')
        waitfile = root/'actual_wait.json'
        rejected(lambda: audit.wait(waitfile, product), a.Pending)
        row = dict(actual_wait=True, actual_returncode=0, owned_process_group_absent=True, products_complete=True,
                   pid=2147483640, process_start_ticks='0', result_ref=audit.ref(product))
        waitfile.write_text(json.dumps(row));audit.wait(waitfile, product)
        for key, value in [('actual_wait', False), ('actual_returncode', 1), ('owned_process_group_absent', False), ('products_complete', False)]:
            broken = dict(row, **{key:value});waitfile.write_text(json.dumps(broken))
            rejected(lambda: a.Audit(root).wait(waitfile, product))
        waitfile.write_text(json.dumps(dict(row, pid=os.getsid(0))))
        rejected(lambda: a.Audit(root).wait(waitfile, product))
        waitfile.write_text(json.dumps(row))
        rejected(lambda: a.Audit(root).wait(waitfile, root/'other.json'))
        product.write_text('{"complete": false}\n')
        rejected(lambda: a.Audit(root).wait(waitfile, product))
        rejected(lambda: audit.arrays({'path':'never_opened_truth.npz','sha256':'0'*64}, truth=True))
        passed.append('genuine_wait_missing_failed_live_session_wrong_product_and_hash_tampering_fail_closed')
        passed.append('closed_truth_gate_rejects_before_private_file_open')
    with patch.object(a.Audit, 'gates', side_effect=a.Pending('mock pending')), patch.object(a.Audit, 'arrays', side_effect=AssertionError('Truth was opened')), contextlib.redirect_stdout(io.StringIO()) as stream:
        assert a.main(['--benchmark', 'all']) == 3
        report = json.loads(stream.getvalue())
        assert report['truth_opened'] is False and report['outputs_written'] is False
    passed.append('pending_main_exit3_cannot_open_truth_or_write_final_report')
    with patch.object(a.Audit, 'gates', return_value=[]), patch.object(a.Audit, 'arrays', side_effect=AssertionError('Readiness opened arrays')), contextlib.redirect_stdout(io.StringIO()) as stream:
        assert a.main(['--benchmark', 'all', '--check-ready']) == 0
        assert json.loads(stream.getvalue())['truth_opened'] is False
    passed.append('ready_only_mode_does_not_run_metric_or_truth_code')
    audit = a.Audit()
    plan = audit.json(a.BASE/'root/american_continue_plan_v1.json')
    parent = 'argformer_american_test_000';binding = plan['native_score_gate'][parent]
    focal = audit.json(binding['focal_preparation_ref']);r = audit.arrays(focal['focals_ref'])
    a.canonical(r, parent, a.C3, 10000)
    arg = audit.json(binding['argformer_completion_path']);flare = audit.json(binding['flare_ref'])
    for product in [arg, flare]:
        values = a.prediction_identity(audit.arrays(product['prediction_ref']), r, a.C3)
        assert ((values == 0) | (values == 1)).all()
    assert not audit.truth_reads
    passed.append('actual_retained_ARGformer_and_FLARE_TEST000_public_arrays_match_original10000_canonical_keys_without_truth')
    protocol = audit.json(a.BASE/'argmix/protocol_v1.json');recipe = audit.json(protocol['recipe_ref'])
    parent = 'published_recon_train_000'
    observation = audit.json(a.NATIVE/parent/'treeloom_observation/completed.json')
    rows, roster, bridge = a.argmix_support(audit, observation['support_ref'], protocol['recipe_ref'], parent, 'train', recipe)
    registry = audit.arrays(observation['registry_ref']);a.canonical(registry, parent, a.C4, 100000)
    assert np.array_equal(registry['record_ids'], np.arange(100000))
    assert np.array_equal(registry['positions_bp0'], [float(row['rebased_midpoint_bp']) for row in rows])
    assert np.array_equal(registry['sample_indices'], [int(row['sample_index']) for row in rows])
    assert np.array_equal(registry['sample_nodes'], [roster[int(row['sample_index'])]['inferred_tree_node_id'] for row in rows])
    assert not audit.truth_reads
    passed.append('actual_original_ARGMix_TRAIN000_support100000_semantic_keys_midpoints_and_registry_without_truth')
    _, _, american_protocol = audit.freeze('american')
    windows = []
    for parent, row in american_protocol['parents'].items():
        if row['split'] == 'train':
            data = audit.arrays(row['binding']['focal_registry_ref'])
            a.canonical(data, parent, a.C3, 10000)
            windows.append(len(np.unique(data['positions_bp0'])))
    assert len(windows) == 20 and sum(windows) == 158476 and 13*sum((n+15)//16 for n in windows) == 128869
    passed.append('actual_original_American20TRAIN_registries_prove_full_window_and_tail_update_counts_without_truth')
    audit.freeze('argmix')
    passed.append('both_final_scientific_freezes_and_all_bound_sources_authenticate')
    receipt = dict(schema='independent_final_results_audit_tests_v1', passed=True, recorded_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_ref=audit.ref(a.__file__), test_source_ref=audit.ref(__file__), tests=passed,
        production_GPU_or_training_launched=False, actual_heldout_truth_opened=False,
        test_truth='Only synthetic NumPy intervals and synthetic tskit census tables',
        successful_full_production_numerical_audit_pending_actual_scoring_waits=True)
    out = a.REVIEW/'final_results_tests_result_v2.json'
    with out.open('x') as f:json.dump(receipt, f, sort_keys=True, indent=2);f.write('\n')
    print(json.dumps(dict(passed=True, tests=len(passed), receipt_ref=audit.ref(out), source_ref=audit.ref(a.__file__)), sort_keys=True))


if __name__ == '__main__':main()
