"""Small real TRAIN-only successor integration test; no benchmark fit or held-out input."""
from io import BytesIO
import copy
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel, prepare_observations, prepared_to_state, collate_prepared
from research.joint_evaluation_20261029.american.weighting_v2 import analytic_tests, parent_batch_weight
from .runtime import BASE, RESULTS, CACHE, MODEL_ID, ref, read, write, save_torch, source_refs, require, now, tree_bytes, parent_item, train_targets, cache_admission
from .observations import ObservationSource
from .pipeline import pool_for, stream, check_ram
from .smoke import exact
from . import cache


def main():
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    protocol = read(ref(BASE / 'protocol_v1.json'))
    parents = ['published_recon_train_000', 'published_recon_train_001']
    items = {parent: parent_item(parent, protocol) for parent in parents}
    sources, selected, expected, labels = {}, {}, {}, {}
    counts, observed = {}, []
    for parent in parents:
        source = sources[parent] = ObservationSource(parent, items[parent]['observation_ref'])
        sites = source.plan()
        require(sorted(i for site in sites for i in site['example_indices']) == list(range(100000)), 'Every canonical TRAIN row exactly once')
        counts[parent] = len(sites)
        rng = np.random.default_rng(np.random.SeedSequence([20261029, source.parent_ordinal, 881]))
        selected[parent] = [sites[i] for i in np.sort(rng.choice(len(sites), 2, replace=False))]
        labels[parent], _ = train_targets(parent, protocol, source)
        for site in selected[parent]:
            begin = time.monotonic()
            packet = source.extract(site)
            batch = prepare_observations(packet)
            expected[(parent, site['window_index'])] = batch
            observed.append(dict(parent=parent, index=site['window_index'], targets=batch.target_count,
                                  preparation_seconds=time.monotonic() - begin, statistics=batch.statistics))
    cache.initialize(128 * 2**20)
    actual = {}
    with pool_for(protocol, items, 'build', 128 * 2**20) as pool:
        for parent in parents:
            for index, batch, metadata in stream(pool, parent, selected[parent]):
                exact(prepared_to_state(expected[(parent, index)]), prepared_to_state(batch))
                actual[(parent, index)] = batch
        owned_rss = check_ram()
    with pool_for(protocol, items, 'read') as pool:
        for parent in parents:
            for index, batch, metadata in stream(pool, parent, selected[parent]):
                require(metadata['cache_hit'] is True, 'Fresh worker pool uses persistent exact cache')
                exact(prepared_to_state(expected[(parent, index)]), prepared_to_state(batch))
    require(tree_bytes(CACHE) <= 128 * 2**20, 'Initial technical probe within128MiB allocation')
    parent = parents[0]
    chosen = selected[parent]
    rows = np.asarray([row for site in chosen for row in site['example_indices']])
    batch = collate_prepared([actual[(parent, site['window_index'])] for site in chosen])
    torch.manual_seed(protocol['initialization_seed'])
    model = JointAncestryModel(**protocol['model'])
    settings = protocol['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], betas=tuple(settings['betas']), eps=settings['epsilon'], weight_decay=settings['weight_decay'])
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch, mode='structure')
    # Diagnostic single-parent estimator only; no benchmark checkpoint or candidate selection.
    loss = F.cross_entropy(logits, torch.as_tensor(labels[parent][rows]), reduction='sum') * parent_batch_weight(counts[parent], (counts[parent]+15)//16, len(chosen), (counts[parent]+15)//16, 100000)
    require(torch.isfinite(loss).item(), 'Finite actual TRAIN objective')
    loss.backward()
    require(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)).item(), 'Finite all-parameter gradient norm')
    optimizer.step()
    require(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()), 'Real optimizer update occurred')
    out = RESULTS / 'technical_test'
    checkpoint_ref = save_torch(out / 'diagnostic_state.pt', model.state_dict())
    restored = JointAncestryModel(**protocol['model'])
    restored.load_state_dict(torch.load(ROOT_PATH(checkpoint_ref), weights_only=True, map_location='cpu'))
    model.eval(); restored.eval()
    with torch.inference_mode():
        a = model(batch, mode='structure')
        b = restored(collate_prepared([expected[(parent, site['window_index'])] for site in chosen]), mode='structure')
    require(torch.equal(a, b), 'Exact CPU cache/checkpoint prediction recovery')
    probabilities = torch.softmax(a.double(), dim=1).float().numpy()
    require(probabilities.shape == (len(rows), 4) and np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1., atol=1e-5), 'Four-class native probability schema')
    rejected = []
    def rejection(name, callback):
        try:
            callback()
        except (ValueError, KeyError, FileNotFoundError):
            rejected.append(name)
        else:
            raise AssertionError('Rejected boundary accepted: ' + name)
    heldout = next(p for p, item in protocol['parents'].items() if item['split'] == 'iid')
    rejection('heldout_label_read_blocked_before_IO', lambda: train_targets(heldout, protocol, None))
    rejection('existing8GiB_probe_allocation_is_not_production_admission', lambda: cache_admission(protocol['existing_probe_allocation_ref']['path'], protocol))
    bad = copy.deepcopy(chosen[0]); bad['sample_indices'][0] += 1
    rejection('wrong_canonical_sample_join', lambda: sources[parent].extract(bad))
    for source in sources.values():
        source.check_stats()
    require(not torch.cuda.is_initialized(), 'CPU-only technical test')
    receipt = dict(schema='argmix_joint_technical_test_v1', passed=True, benchmark_fit=False, recorded_utc=now(),
        model_id=MODEL_ID, protocol_ref=ref(BASE / 'protocol_v1.json'), source_refs=source_refs(),
        parents=parents, complete_target_rows_per_parent=100000, window_counts=counts, sampled_windows=observed,
        worker_and_persistent_cache_tensor_states_exact=True, fresh_pool_cache_hits=True,
        CPU_optimizer_steps=1, finite_loss=float(loss.detach()), checkpoint_prediction_exact=True,
        canonical_original_TRAIN_labels=True, original_four_class_probabilities=True,
        parent_tail_weighting_tests=analytic_tests(), rejected_boundaries=rejected,
        heldout_payloads_or_labels_opened=False, CUDA_initialized=False, owned_RSS_bytes=owned_rss,
        cache_scope_bytes=tree_bytes(CACHE), elapsed_seconds=time.monotonic() - start)
    print(write(out / 'completed.json', receipt), flush=True)


def ROOT_PATH(value):
    from .common import path
    return path(value)


if __name__ == '__main__':
    main()
