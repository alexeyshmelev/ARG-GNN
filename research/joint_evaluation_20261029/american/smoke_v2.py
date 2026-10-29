"""Actual TRAIN CPU extraction/worker/train/predict/score integration smoke."""
import copy
from io import BytesIO
import json
import os
import resource
import sys
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from arggnn.american_c3_evaluate_v1 import verify_prediction, metrics
from research.joint_model_20261029.model.joint_model_v1 import (
    JointAncestryModel, prepare_observations, prepared_to_state, collate_prepared)
from .common_v2 import BASE, RESULTS, CLASSES, IDENTITY, path, ref, read, write, save_npz, load_npz, source_refs, require, now
from .weighting_v2 import analytic_tests, parent_batch_weight
from .state_codec_v2 import encode, decode
from .cache_v2 import initialize
from .observations import ObservationSource
from .targets import train_targets, join
from .pipeline_v2 import pool_for, stream, check_ram, check_disk


def exact(left, right, location='state'):
    if isinstance(left, torch.Tensor):
        require(left.shape == right.shape and left.layout == right.layout and left.dtype == right.dtype, 'Prepared shape/layout equality')
        if left.layout == torch.sparse_csr:
            exact(left.crow_indices(), right.crow_indices()); exact(left.col_indices(), right.col_indices()); exact(left.values(), right.values())
        else:
            require(torch.equal(left, right), 'Prepared tensor bitwise equality: ' + location)
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), 'Prepared dictionary structure')
        for key in left:
            exact(left[key], right[key], location + '.' + key)
    elif isinstance(left, (list, tuple)):
        require(len(left) == len(right), 'Prepared container size')
        for a, b in zip(left, right):
            exact(a, b)
    else:
        require(left == right, 'Prepared scalar equality')


def codec_boundary_tests():
    # Float values never undergo arithmetic; payload bits, signed zero and NaN payloads survive.
    bits = np.asarray([0, 2147483648, 1065353216, 2139095040, 4286578688, 2143289345, 2143289346], dtype=np.uint32)
    state = dict(floats=torch.from_numpy(bits.view(np.float32).copy()),
                 doubles=torch.tensor([0., -0., float('nan'), float('inf')], dtype=torch.float64),
                 ints=torch.tensor([0, -128, 127, -32768, 32767, -(2**31), 2**31-1, -(2**59), 2**59], dtype=torch.int64),
                 empty=torch.empty((0, 3), dtype=torch.int64), boolean=torch.tensor([True, False]))
    restored = decode(encode(state))
    for key, expected in state.items():
        actual = restored[key]
        require(expected.dtype == actual.dtype and expected.shape == actual.shape and expected.numpy().tobytes() == actual.numpy().tobytes(), 'Exact dtype, shape, bit pattern including NaN payload: ' + key)
    try:
        encode(dict(unsafe=torch.tensor([2**62], dtype=torch.int64)))
    except ValueError:
        pass
    else:
        raise AssertionError('Unsafe integer delta bound accepted')
    payload = bytearray(encode(state))
    payload[8:16] = (129 * 2**20).to_bytes(8, 'little')
    try:
        decode(payload)
    except ValueError:
        pass
    else:
        raise AssertionError('Oversized decoded payload accepted')


def main():
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    start = time.monotonic()
    protocol = json.loads((BASE / 'protocol_v2.json').read_text())
    parent = 'argformer_american_train_000'
    source = ObservationSource(parent, protocol['parents'][parent])
    plan = source.plan()
    total_windows = 0
    window_counts = {}
    for name, item in protocol['parents'].items():
        if item['split'] == 'train':
            records = load_npz(item['binding']['focal_registry_ref'])
            require(len(records['record_ids']) == 10000, 'All original TRAIN rows included in sizing')
            window_counts[name] = len(np.unique(records['positions_bp0']))
            total_windows += window_counts[name]
    selected = [plan[i] for i in (0, 1, len(plan) // 2, len(plan) - 1)]
    out = RESULTS / 'smoke_final_v2'
    out.mkdir(exist_ok=False)
    write(out / 'site_plan.json', dict(parent_id=parent, sites=selected))
    restored = json.loads((out / 'site_plan.json').read_text())['sites']
    expected, times, sizes = [], [], []
    for site, roundtrip in zip(selected, restored):
        before = time.monotonic()
        packet = source.extract(site)
        packet2 = source.extract(roundtrip)
        for key in packet:
            require(packet[key].dtype == packet2[key].dtype and np.array_equal(packet[key], packet2[key]), 'Full observation JSON roundtrip: ' + key)
        require((packet['reference_class'][packet['query_mask']] == -1).all(), 'Unknown query ancestry')
        for ti, interval in enumerate(packet['tree_interval_local_bp']):
            native = source.ts.at(float(interval.mean()))
            supplied_nodes = np.flatnonzero(packet['node_tree_index'] == ti)
            require(len(supplied_nodes) == len(list(native.nodes())), 'Every complete native marginal node retained')
            expected_positions = source.positions[(source.positions >= interval[0]) & (source.positions < interval[1])]
            require(np.array_equal(packet['site_position_local_bp'][packet['site_tree_index'] == ti], expected_positions),
                    'Every native interval SNP retained')
        expected.append(prepare_observations(packet))
        stream_bytes = BytesIO(); torch.save(prepared_to_state(expected[-1]), stream_bytes)
        sizes.append(len(stream_bytes.getvalue()))
        times.append(time.monotonic() - before)
    actual, transport = [], []
    initialize()
    with pool_for(protocol, 'build') as pool:
        for index, batch, metadata in stream(pool, parent, selected):
            reference_index = [s['window_index'] for s in selected].index(index)
            exact(prepared_to_state(expected[reference_index]), prepared_to_state(batch))
            actual.append(batch); transport.append(metadata)
        owned_rss = check_ram()
    with pool_for(protocol, 'read') as pool:
        for index, cached, metadata in stream(pool, parent, selected):
            require(metadata['cache_hit'] is True, 'A new worker pool reuses every persistent cached window')
            exact(prepared_to_state(expected[[s['window_index'] for s in selected].index(index)]), prepared_to_state(cached))
    weighting_tests = analytic_tests()
    codec_boundary_tests()
    target = train_targets(parent, protocol, source.records)
    require(target.shape == (10000,) and ((target >= 0) & (target < 3)).all(), 'All original TRAIN target labels')
    rows = np.asarray([i for site in selected for i in site['example_indices']], dtype=np.int64)
    batch = collate_prepared(actual)
    torch.manual_seed(protocol['initialization_seed'])
    model = JointAncestryModel(**protocol['model'])
    opt = protocol['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], betas=tuple(opt['betas']),
                                 eps=opt['epsilon'], weight_decay=opt['weight_decay'])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    y = torch.as_tensor(target[rows])
    losses = []
    for _ in range(2):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits = model(batch, mode='structure')
        loss = F.cross_entropy(logits, y, reduction='sum') * parent_batch_weight(len(plan), sum((w + 15) // 16 for w in window_counts.values()), len(selected), (len(plan) + 15) // 16, 200000)
        require(torch.isfinite(loss).item(), 'Finite actual TRAIN weighted loss')
        loss.backward()
        require(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])).item(), 'Finite actual TRAIN gradients')
        optimizer.step(); losses.append(float(loss.detach()))
    require(any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()), 'Optimizer actually changes learned state')
    checkpoint = out / 'cpu_smoke_state.pt'
    with checkpoint.open('xb') as destination:
        torch.save(model.state_dict(), destination)
    model.eval()
    reloaded = JointAncestryModel(**protocol['model']).eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    with torch.inference_mode():
        logits = model(batch, mode='structure')
        restored_logits = reloaded(collate_prepared(expected), mode='structure')
    require(torch.equal(logits, restored_logits), 'CPU checkpoint + spawned transport exact prediction parity')
    probabilities = torch.softmax(logits.double(), -1).float().numpy()
    records = {key: source.records[key][rows] if key not in ('parent_id', 'class_order') else source.records[key] for key in IDENTITY}
    prediction = dict(records, probabilities=probabilities)
    calls = verify_prediction(prediction, records)
    output_ref = save_npz(out / 'predictions.npz', prediction)
    require(np.array_equal(verify_prediction(load_npz(output_ref), records), calls), 'Original common evaluator prediction transport')
    confusion = np.zeros((3, 3), dtype=np.int64); np.add.at(confusion, (target[rows], calls), 1)
    scored = metrics(confusion)
    require(sum(sum(row) for row in scored['confusion']) == len(rows), 'Exact original native3 common score support')
    write(out / 'score.json', dict(parent_id=parent, split='train', result=scored, prediction_ref=output_ref))
    failures = []
    def rejected(name, action):
        try:
            action()
        except (ValueError, KeyError, IndexError):
            failures.append(name)
        else:
            raise AssertionError('Malformed input accepted: ' + name)
    held = next(p for p, item in protocol['parents'].items() if item['split'] != 'train')
    rejected('heldout_parent_rejected_before_private_TRAIN_read', lambda: train_targets(held, protocol, {}))
    bad_site = copy.deepcopy(selected[0]); bad_site['sample_indices'][0] += 1
    rejected('wrong_sample_focal_join', lambda: source.extract(bad_site))
    bad_site = copy.deepcopy(selected[0]); bad_site['example_indices'][0] = 10000
    rejected('missing_focal_ID', lambda: source.extract(bad_site))
    bad = {key: value.copy() for key, value in prediction.items()}; bad['record_ids'][0] += 1
    rejected('wrong_prediction_record_ID', lambda: verify_prediction(bad, records))
    bad = dict(prediction, probabilities=probabilities[:-1])
    rejected('missing_prediction_row', lambda: verify_prediction(bad, records))
    bad = dict(prediction, probabilities=np.full_like(probabilities, np.nan))
    rejected('nonfinite_prediction', lambda: verify_prediction(bad, records))
    require(not torch.cuda.is_initialized(), 'No CUDA initialized during CPU integration qualification')
    estimate = dict(TRAIN_parent_window_counts=window_counts, TRAIN_windows_per_epoch=total_windows,
          fixed_epochs=protocol['epochs'], total_window_presentations=total_windows * protocol['epochs'],
          actual_prepared_bytes=sizes, actual_CPU_prepare_seconds=times,
          hypothetical_all_TRAIN_prepared_cache_bytes=total_windows * float(np.mean(sizes)),
          persistent_prepared_cache_authorized=True, configured_worker_RAM_cache_bytes=0,
          configured_native_tree_cache_bytes=16 * 128 * 2**20, configured_max_pending_windows=32, minibatch_windows=16,
          observed_parent_plus16worker_RSS_bytes=owned_rss,
          one_time_ideal16worker_preparation_hours=total_windows * float(np.mean(times)) / 16 / 3600,
          estimate_limit='Four TRAIN windows only; CPU preparation projection excludes GPU compute, NFS contention, parent startup, and cache locality. A full-parent GPU qualification remains necessary before launch-time expectations.',
          total_new_disk_bytes=check_disk(), disk_allowance_bytes=2**30, owned_RAM_cap_bytes=64 * 2**30)
    resources_ref = write(RESULTS / 'smoke_resource_final_v2.json', estimate)
    receipt = dict(schema='american_joint_TRAIN_CPU_integration_v2', passed=True, utc=now(),
        source_refs=source_refs(), protocol_ref=ref(BASE / 'protocol_v2.json'), heldout_payloads_opened=False,
        TRAIN_truth_opened_only_for_original_registered_labels=True, CUDA_initialized=False,
        native_complete_nodes_and_SNPs_verified=True, JSON_packet_roundtrip_bitwise=True,
        worker_prepared_states_bitwise_equal=True, persistent_second_pool_cache_hit=True, weighting_tests=weighting_tests, codec_boundary_tests_passed=True, optimizer_steps=2, finite_losses=losses,
        CPU_checkpoint_predictions_bitwise_equal=True, common_native3_score_executed=True,
        canonical_raw_path_IDs_preserved=True, negative_cases_rejected=failures,
        sampled_rows=len(rows), sampled_windows=len(selected), spawned_worker_pids=sorted({v['worker_pid'] for v in transport}),
        resources_ref=resources_ref, elapsed_seconds=time.monotonic() - start)
    print(write(BASE / 'smoke_result_final_v2.json', receipt), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        raise
