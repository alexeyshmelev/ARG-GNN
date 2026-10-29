"""Bounded train -> fixed-checkpoint prediction -> postseal native3 common score."""
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from arggnn.benchmarking.paper_argmix import paper_argmix_generator as native
from arggnn.benchmarking.paper_argmix import paper_argmix_score as native_score
from arggnn.benchmarking.paper_argmix.paper_argmix_features import read_targets
from research.joint_model_20261029.model.joint_model_v1 import (
    JointAncestryModel, prepare_observations, prepared_to_state, prepared_from_state, collate_prepared)
from .runtime import (BASE, ROOT, RESULTS, CACHE, CAMPAIGN, MODEL_ID, CLASSES, RECIPE_REF, IDENTITY,
    path, ref, read, verify, write, save_numpy, save_torch, frozen, require, now, output_guard, tree_bytes,
    parent_item, train_targets, require_wait, cache_admission)
from research.joint_evaluation_20261029.american.weighting_v2 import parent_batch_weight
from research.joint_evaluation_20261029.american.state_codec_v2 import encode, decode
from . import cache
from .observations import ObservationSource

WORKER_ITEMS = None
WORKER_SOURCE = None
WORKER_NAMESPACE = None
WORKER_MODE = None
RAM_CAP = 64 * 2**30


def worker_init(items, namespace, mode, cache_limit):
    global WORKER_ITEMS, WORKER_NAMESPACE, WORKER_MODE
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    WORKER_ITEMS, WORKER_NAMESPACE, WORKER_MODE = items, namespace, mode
    cache.CACHE_CAP = cache_limit


def prepare_job(parent, site):
    global WORKER_SOURCE
    begin = time.monotonic()
    item = WORKER_ITEMS[parent]
    if item['split'] == 'train':
        payload = cache.read_cache(WORKER_NAMESPACE, parent, item, site)
        if payload is not None:
            return site['window_index'], payload, dict(cache_hit=True, worker_pid=os.getpid())
        require(WORKER_MODE == 'build', 'Complete immutable TRAIN cache required; missing window is fatal')
    else:
        require(WORKER_MODE == 'heldout', 'Held-out observations only after fixed checkpoint gate')
    if WORKER_SOURCE is None or WORKER_SOURCE.parent_id != parent:
        WORKER_SOURCE = ObservationSource(parent, item['observation_ref'])
    state = prepared_to_state(prepare_observations(WORKER_SOURCE.extract(site)))
    payload = encode(state, level=9)
    if item['split'] == 'train':
        cache.write_cache(WORKER_NAMESPACE, parent, item, site, payload)
    return site['window_index'], payload, dict(cache_hit=False, preparation_seconds=time.monotonic() - begin,
            compressed_bytes=len(payload), worker_pid=os.getpid(), worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def stream(pool, parent, sites):
    source, pending = iter(sites), deque()
    for _ in range(min(32, len(sites))):
        pending.append(pool.submit(prepare_job, parent, next(source)))
    while pending:
        index, payload, metadata = pending.popleft().result()
        following = next(source, None)
        if following is not None:
            pending.append(pool.submit(prepare_job, parent, following))
        batch = prepared_from_state(decode(payload))
        yield index, batch, metadata


def pool_for(protocol, items, mode='read', cache_limit=0):
    # Workers receive public observation bindings only; no truth references.
    require(protocol['workers'] == 16 and mode in ('read', 'build', 'heldout'), 'Registered worker execution')
    return ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('spawn'), initializer=worker_init,
                               initargs=(items, protocol['cache_namespace'], mode, cache_limit))


def check_ram():
    import psutil
    process = psutil.Process()
    total = 0
    for child in [process] + process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    require(total <= RAM_CAP, 'Owned parent/worker RSS exceeds registered64GiB cap; no truncation or retry')
    return total


def prepare_cache(args, protocol, freeze_ref):
    import hashlib
    admission = cache_admission(args.admission, protocol)
    cache.initialize(admission['cache_max_bytes'])
    items = {parent: parent_item(parent, protocol) for parent, row in protocol['parents'].items() if row['split'] == 'train'}
    out = RESULTS / 'cache_build'
    out.mkdir(exist_ok=False)
    completed, start = {}, time.monotonic()
    largest = dict(nodes=[], SNP_rows=[], prepared_tensor_bytes=[])
    with pool_for(protocol, items, 'build', admission['cache_max_bytes']) as pool:
        for parent, item in items.items():
            if item['split'] != 'train':
                continue
            source = ObservationSource(parent, item['observation_ref'])
            sites = source.plan()
            seen = np.zeros(len(sites), dtype=bool)
            for index, batch, metadata in stream(pool, parent, sites):
                require(not seen[index] and np.array_equal(batch.focal_sample.numpy(), sites[index]['sample_indices']), 'Exact unique cached window targets')
                seen[index] = True
                def tensor_bytes(value):
                    if isinstance(value, torch.Tensor):
                        if value.layout == torch.sparse_csr:
                            return sum(tensor_bytes(t) for t in (value.crow_indices(), value.col_indices(), value.values()))
                        return value.numel() * value.element_size()
                    if isinstance(value, dict):
                        return sum(tensor_bytes(v) for v in value.values())
                    if isinstance(value, (list, tuple)):
                        return sum(tensor_bytes(v) for v in value)
                    return 0
                sizes = dict(nodes=batch.parent.numel(), SNP_rows=batch.site_genotype.shape[0], prepared_tensor_bytes=tensor_bytes(prepared_to_state(batch)))
                for key in largest:
                    largest[key].append(dict(parent_id=parent, window_index=index, **sizes))
                    largest[key] = sorted(largest[key], key=lambda row: (-row[key], row['parent_id'], row['window_index']))[:16]
                if int(seen.sum()) % 1000 == 0:
                    print(json.dumps(dict(stage='cache', parent=parent, windows=int(seen.sum()), owned_rss_bytes=check_ram(), elapsed_seconds=time.monotonic() - start)), flush=True)
            require(seen.all(), 'All parent cache windows completed')
            source.check_stats()
            completed[parent] = dict(windows=len(sites), targets=len(source.records['record_ids']),
                plan_sha256=hashlib.sha256(json.dumps(sites, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
            write(out / f'{parent}.json', completed[parent])
            del source
    require(len(completed) == 5 and sum(v['targets'] for v in completed.values()) == 500000, 'Complete original5-parent TRAIN cache coverage')
    actual = tree_bytes(CACHE)
    reserved = int(cache.COUNTER.read_text())
    require(actual <= reserved <= cache.CACHE_CAP, 'Actual completed cache within exact root-admitted cap')
    print(write(out / 'completed.json', dict(complete=True, freeze_ref=freeze_ref, cache_namespace=protocol['cache_namespace'],
        parents=completed, public_bindings=items, admission_ref=ref(args.admission), largest_TRAIN_windows=largest, actual_scope_bytes=actual, conservatively_reserved_bytes=reserved,
        query_ancestry_truth_opened=False, elapsed_seconds=time.monotonic() - start, completed_utc=now())), flush=True)


def device(index):
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    listing = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True)
    mapping = {int(row.split(',')[0]): row.split(',')[1].strip() for row in listing.stdout.splitlines()}
    require(index in (3, 4, 5) and os.environ.get('CUDA_VISIBLE_DEVICES') == mapping[index], 'Expose exactly the authorized physicalGPU UUID')
    require(torch.cuda.device_count() == 1, 'Exactly one visible assigned GPU')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(0)
    return 'cuda:0'


def train(args, protocol, freeze_ref):
    cache_ref, _ = require_wait(RESULTS / 'cache_build/completed.json', 'joint_argmix.train_cache')
    cache_product = read(cache_ref)
    require(cache_product['complete'] and cache_product['freeze_ref'] == freeze_ref and cache_product['cache_namespace'] == protocol['cache_namespace'], 'Complete frozen TRAIN cache gate')
    qualification_ref, _ = require_wait(RESULTS / 'gpu_qualification/completed.json', 'joint_argmix.gpu_qualification')
    qualification = read(qualification_ref)
    require(qualification['passed'] and qualification['freeze_ref'] == freeze_ref, 'Actual full16-window GPU qualification')
    items = cache_product['public_bindings']
    require(items == {parent: parent_item(parent, protocol) for parent, row in protocol['parents'].items() if row['split'] == 'train'}, 'Original observed bindings unchanged since cache build')
    out = RESULTS / 'fit'
    out.mkdir(exist_ok=False)
    selected = [p for p, row in protocol['parents'].items() if row['split'] == 'train']
    plans, labels, row_refs = {}, {}, {}
    # Public plans seal in RAM before same-parent TRAIN truth joins. Held-out inputs are untouched.
    for parent in selected:
        source = ObservationSource(parent, items[parent]['observation_ref'])
        plans[parent] = source.plan()
        labels[parent], label_binding = train_targets(parent, protocol, source)
        row_refs[parent] = dict(windows=len(plans[parent]), targets=len(labels[parent]),
                                focal_registry_ref=source.seal['registry_ref'], TRAIN_labels=label_binding)
        del source
    count = sum(len(v) for v in plans.values())
    targets = sum(len(v) for v in labels.values())
    total_batches = sum((len(v) + 15) // 16 for v in plans.values())
    require(count == sum(v['windows'] for v in cache_product['parents'].values()), 'Every original TRAIN window')
    require(targets == 500000, 'Every original ARGMix TRAIN target')
    plan_ref = write(out / 'training_plan.json', dict(parents=row_refs, windows=count, targets=targets,
                epochs=protocol['epochs'], planned_updates=protocol['epochs'] * sum((len(v) + 15) // 16 for v in plans.values()),
                canonical_target_presentations=targets * protocol['epochs']))
    torch.manual_seed(protocol['initialization_seed'])
    target_device = device(args.gpu)
    model = JointAncestryModel(**protocol['model']).to(target_device)
    opt = protocol['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['learning_rate'], betas=tuple(opt['betas']),
                                 eps=opt['epsilon'], weight_decay=opt['weight_decay'])
    updates, start = 0, time.monotonic()
    with pool_for(protocol, items) as pool, (out / 'progress.jsonl').open('x') as progress:
        for epoch in range(1, protocol['epochs'] + 1):
            rng = np.random.default_rng(np.random.SeedSequence([protocol['batch_seed'], epoch]))
            order = [selected[i] for i in rng.permutation(len(selected))]
            epoch_targets, epoch_ce, maximum_ram = 0, 0., 0
            for parent in order:
                shuffled = [plans[parent][i] for i in rng.permutation(len(plans[parent]))]
                minibatch, rows = [], []
                for position, (index, prepared, metadata) in enumerate(stream(pool, parent, shuffled)):
                    site = plans[parent][index]
                    require(np.array_equal(prepared.focal_sample.numpy(), site['sample_indices']), 'Exact prepared focal sample join')
                    minibatch.append(prepared)
                    rows.extend(site['example_indices'])
                    # Flush by parent stream position, never by randomized site index.
                    if len(minibatch) < 16 and position + 1 != len(shuffled):
                        continue
                    batch = collate_prepared(minibatch).pin_memory().to(target_device, non_blocking=True)
                    y = torch.as_tensor(labels[parent][rows], device=target_device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch, mode='structure')
                    ce = F.cross_entropy(prediction, y, reduction='sum')
                    loss = ce * parent_batch_weight(len(plans[parent]), total_batches, len(minibatch), (len(plans[parent]) + 15) // 16, targets)
                    require(torch.isfinite(loss).item(), 'Finite canonical-target weighted objective')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt['gradient_clip'])
                    require(torch.isfinite(norm).item(), 'Finite gradients')
                    optimizer.step()
                    updates += 1
                    epoch_targets += len(rows)
                    epoch_ce += float(ce.detach())
                    minibatch.clear(); rows.clear()
                    if updates % 25 == 0:
                        maximum_ram = max(maximum_ram, check_ram())
                        record = dict(epoch=epoch, updates=updates, epoch_targets=epoch_targets,
                                      elapsed_seconds=time.monotonic() - start, owned_rss_bytes=maximum_ram)
                        progress.write(json.dumps(record) + '\n'); progress.flush()
                        print(json.dumps(record), flush=True)
                require(not minibatch, 'All parent minibatches flushed')
            require(epoch_targets == targets, 'Every TRAIN target presented exactly once per epoch')
            epoch_checkpoint = save_torch(out / f'epoch_{epoch:02d}_state.pt', dict(epoch=epoch, updates=updates, model={key: value.detach().cpu() for key, value in model.state_dict().items()}, optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(), freeze_ref=freeze_ref))
            write(out / f'epoch_{epoch:02d}.json', dict(epoch=epoch, updates=updates, targets=epoch_targets,
                  mean_CE=epoch_ce / targets, checkpoint_ref=epoch_checkpoint, elapsed_seconds=time.monotonic() - start, owned_peak_observed_rss_bytes=maximum_ram))
            output_guard()
    checkpoint = out / 'final_state.pt'
    save_torch(checkpoint, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    product = dict(schema='argmix_joint_fixed_final_checkpoint_v1', model_id=MODEL_ID, complete=True,
        freeze_ref=freeze_ref, checkpoint_ref=ref(checkpoint), training_plan_ref=plan_ref, epochs=protocol['epochs'],
        updates=updates, target_presentations=targets * protocol['epochs'], heldout_observations_or_truth_opened=False,
        elapsed_seconds=time.monotonic() - start, completed_utc=now())
    print(write(out / 'completed.json', product), flush=True)


def predict(args, protocol, freeze_ref):
    parent = args.parent
    require(parent in protocol['parents'] and protocol['parents'][parent]['split'] != 'train', 'Original held-out parent only')
    fit_ref, _ = require_wait(RESULTS / 'fit/completed.json', 'joint_argmix.train')
    fit = read(fit_ref)
    require(fit['complete'] and fit['model_id'] == MODEL_ID and fit['freeze_ref'] == freeze_ref and
            fit['epochs'] == protocol['epochs'] and fit['target_presentations'] == protocol['training_target_presentations'], 'Fixed completed TRAIN checkpoint gate')
    checkpoint = verify(fit['checkpoint_ref'])
    target_device = device(args.gpu)
    model = JointAncestryModel(**protocol['model'])
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    model.eval().to(target_device)
    out = RESULTS / 'predictions' / parent
    out.parent.mkdir(exist_ok=True)
    out.mkdir(exist_ok=False)
    # Seal checkpoint before opening this parent's observation payloads.
    write(out / 'started.json', dict(freeze_ref=freeze_ref, checkpoint_ref=fit['checkpoint_ref'], parent_id=parent, utc=now()))
    items = {parent: parent_item(parent, protocol)}
    source = ObservationSource(parent, items[parent]['observation_ref'])
    sites = source.plan()
    plan_ref = write(out / 'site_plan.json', dict(parent_id=parent, rows=100000, sites=sites))
    logits = np.full((100000, 4), np.nan, dtype=np.float32)
    seen = np.zeros(100000, dtype=bool)
    batch, rows = [], []
    with pool_for(protocol, items, 'heldout') as pool:
        for index, prepared, metadata in stream(pool, parent, sites):
            site = sites[index]
            require(np.array_equal(prepared.focal_sample.numpy(), site['sample_indices']), 'Exact prediction sample join')
            batch.append(prepared); rows.extend(site['example_indices'])
            if len(batch) < 16 and index + 1 != len(sites):
                continue
            require(not seen[rows].any() and len(set(rows)) == len(rows), 'No duplicate prediction row')
            with torch.inference_mode():
                prediction = model(collate_prepared(batch).pin_memory().to(target_device, non_blocking=True), mode='structure')
            logits[rows] = prediction.cpu().numpy(); seen[rows] = True
            batch.clear(); rows.clear()
            check_ram()
            print(json.dumps(dict(parent_id=parent, completed_windows=index + 1, targets=int(seen.sum()))), flush=True)
    require(seen.all() and np.isfinite(logits).all(), 'All100000 canonical predictions complete')
    values = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    probabilities = (values / values.sum(1, keepdims=True)).astype(np.float32)
    require(probabilities.shape == (100000, 4) and np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1., atol=1e-5), 'Native common probability schema')
    source.check_stats()
    prediction_ref = save_numpy(out / 'probabilities.npy', probabilities)
    logits_ref = save_numpy(out / 'logits.npy', logits)
    product = dict(schema='paper_argmix_predictions_v1', model_id=MODEL_ID, complete=True, parent_id=parent,
        split=protocol['parents'][parent]['split'], config_ref=native.file_ref(path(RECIPE_REF)), examples=100000,
        class_order=list(CLASSES), probabilities_ref=prediction_ref, logits_ref=logits_ref,
        support_ref=source.seal['support_ref'], registry_ref=source.seal['registry_ref'], observation_ref=items[parent]['observation_ref'],
        freeze_ref=freeze_ref, checkpoint_ref=fit['checkpoint_ref'], site_plan_ref=plan_ref, truth_read=False,
        completed_utc=now())
    print(write(out / 'completed.json', product), flush=True)
    output_guard()


def validate_prediction(parent, protocol, freeze_ref, fit):
    prediction_path = RESULTS / 'predictions' / parent / 'completed.json'
    prediction_ref, _ = require_wait(prediction_path, f'joint_argmix.{parent}.prediction')
    product = read(prediction_ref)
    item = parent_item(parent, protocol)
    observation = read(item['observation_ref'])
    require(product['schema'] == 'paper_argmix_predictions_v1' and product['model_id'] == MODEL_ID and
            product['complete'] and product['parent_id'] == parent and product['split'] == protocol['parents'][parent]['split'] and
            product['config_ref'] == native.file_ref(path(RECIPE_REF)) and product['examples'] == 100000 and
            product['class_order'] == list(CLASSES) and product['truth_read'] is False and product['freeze_ref'] == freeze_ref and
            product['checkpoint_ref'] == fit['checkpoint_ref'] and product['observation_ref'] == item['observation_ref'] and
            product['support_ref'] == observation['support_ref'] and product['registry_ref'] == observation['registry_ref'],
            'Exact completed joint-model prediction on unchanged original support')
    probabilities = np.load(verify(product['probabilities_ref']), allow_pickle=False)
    require(probabilities.dtype == np.float32 and probabilities.shape == (100000, 4) and np.isfinite(probabilities).all() and
            (probabilities >= 0).all() and np.allclose(probabilities.sum(1), 1., atol=1e-5), 'All native probabilities valid before any held-out label read')
    return prediction_path, product


def prediction_barrier(args, protocol, freeze_ref):
    fit_ref, _ = require_wait(RESULTS / 'fit/completed.json', 'joint_argmix.train')
    fit = read(fit_ref)
    require(fit['complete'] and fit['freeze_ref'] == freeze_ref and fit['epochs'] == 5 and fit['target_presentations'] == 2500000,
            'Only complete fixed-final epoch5 fit')
    verify(fit['checkpoint_ref'])
    test_paths = []
    for parent, row in protocol['parents'].items():
        if row['split'] == 'train':
            continue
        prediction_path, product = validate_prediction(parent, protocol, freeze_ref, fit)
        if row['split'] in ('iid', 'misspec'):
            test_paths.append(prediction_path)
    require(len(test_paths) == 8, 'Three IID plus five misspec parents')
    # Preserve the native schema exactly; native verify_barrier regenerates it byte-for-field.
    barrier = native_score.prediction_barrier(path(RECIPE_REF), test_paths)
    print(write(RESULTS / 'prediction_barrier/barrier.json', barrier), flush=True)


def score(args, protocol, freeze_ref):
    barrier_path = RESULTS / 'prediction_barrier/barrier.json'
    require_wait(barrier_path, 'joint_argmix.prediction_barrier')
    native_score.verify_barrier(path(RECIPE_REF), barrier_path)
    paired_path = CAMPAIGN / 'paired_prediction_barrier/barrier.json'
    require_wait(paired_path, 'paper_argmix_paired.prediction_barrier')
    paired = native_score.verify_paired_barrier(path(RECIPE_REF), paired_path)
    require(native.file_ref(barrier_path) in [paired['first_barrier_ref'], paired['second_barrier_ref']], 'Original paired barrier includes this new model')
    labels = []
    for parent, row in protocol['parents'].items():
        if row['split'] in ('iid', 'misspec'):
            label_path = CAMPAIGN / parent / 'labels/completed.json'
            require_wait(label_path, f'paper_argmix_{parent}.labels')
            labels.append(label_path)
    comparator_path = CAMPAIGN / 'argmix_scores/completed.json'
    require_wait(comparator_path, 'paper_argmix_argmix.score')
    result = native_score.score(path(RECIPE_REF), barrier_path, labels, RESULTS / 'scores',
                                guard=output_guard, comparator_path=comparator_path)
    print(json.dumps(result, sort_keys=True), flush=True)


def validation_score(args, protocol, freeze_ref):
    require_wait(RESULTS / 'prediction_barrier/barrier.json', 'joint_argmix.prediction_barrier')
    fit_ref, _ = require_wait(RESULTS / 'fit/completed.json', 'joint_argmix.train')
    parent = next(parent for parent, item in protocol['parents'].items() if item['split'] == 'val')
    _, prediction = validate_prediction(parent, protocol, freeze_ref, read(fit_ref))
    label_ref, _ = require_wait(CAMPAIGN / parent / 'labels/completed.json', f'paper_argmix_{parent}.labels')
    label = read(label_ref)
    require(label['support_manifest_ref'] == prediction['support_ref'] and label['config_ref'] == prediction['config_ref'] and
            label['class_order'] == list(CLASSES) and label['examples'] == 100000, 'Exact original VAL support')
    probabilities = np.load(verify(prediction['probabilities_ref']), allow_pickle=False)
    truth = np.fromfile(verify(label['labels_ref']), dtype=np.uint8)
    require(truth.shape == (100000,) and (truth < 4).all(), 'Complete original validation targets')
    support = read(prediction['support_ref'])
    rows = read_targets(verify(support['targets_ref']))
    cohorts = np.asarray([row['cohort'] for row in rows])
    correct = probabilities.argmax(1) == truth
    result = dict(schema='argmix_joint_fixed_final_validation_v1', model_id=MODEL_ID, freeze_ref=freeze_ref,
        examples=100000, accuracy=float(correct.mean()), per_cohort={name: dict(examples=int((cohorts == name).sum()),
        accuracy=float(correct[cohorts == name].mean())) for name in sorted(set(cohorts))},
        selection_used=False, checkpoint_ref=prediction['checkpoint_ref'], label_ref=label_ref,
        support_ref=prediction['support_ref'], completed_utc=now())
    print(write(RESULTS / 'validation_score/completed.json', result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-cache', 'qualify-gpu', 'train', 'predict', 'barrier', 'score', 'validation-score'])
    parser.add_argument('--freeze', default=str(BASE / 'freeze_v1.json'))
    parser.add_argument('--gpu', type=int, choices=[3, 4, 5])
    parser.add_argument('--parent')
    parser.add_argument('--admission', default=str(ROOT / 'research/joint_evaluation_20261029/root/argmix_joint_production_admission_v1.json'))
    args = parser.parse_args()
    seal, protocol = frozen(args.freeze)
    require(args.stage not in ('train', 'predict', 'qualify-gpu') or args.gpu is not None, 'Assigned physical GPU required')
    require(args.stage != 'predict' or args.parent is not None, 'One exact held-out parent required')
    output_guard()
    if args.stage == 'qualify-gpu':
        from .gpu_qualification import qualify
        qualify(args, protocol, ref(args.freeze))
    else:
        {'prepare-cache': prepare_cache, 'train': train, 'predict': predict, 'barrier': prediction_barrier,
         'score': score, 'validation-score': validation_score}[args.stage](args, protocol, ref(args.freeze))


if __name__ == '__main__':
    main()
