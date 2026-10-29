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
from arggnn.american_c3_evaluate_v1 import metrics, verify_prediction
from research.joint_model_20261029.model.joint_model_v1 import (
    JointAncestryModel, prepare_observations, prepared_to_state, prepared_from_state, collate_prepared)
from .common_v2 import BASE, ROOT, RESULTS, CACHE, MODEL_ID, CLASSES, IDENTITY, path, ref, read, verify, write, load_npz, save_npz, save_torch, frozen, require, now, check_disk, tree_bytes
from .weighting_v2 import parent_batch_weight
from .state_codec_v2 import encode, decode
from . import cache_v2
from .observations import ObservationSource
from .targets import train_targets, join

WORKER_ITEMS = None
WORKER_SOURCE = None
WORKER_NAMESPACE = None
WORKER_MODE = None
RAM_CAP = 64 * 2**30


def worker_init(items, namespace, mode):
    global WORKER_ITEMS, WORKER_NAMESPACE, WORKER_MODE
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    WORKER_ITEMS, WORKER_NAMESPACE, WORKER_MODE = items, namespace, mode


def prepare_job(parent, site):
    global WORKER_SOURCE
    begin = time.monotonic()
    item = WORKER_ITEMS[parent]
    if item['split'] == 'train':
        payload = cache_v2.read_cache(WORKER_NAMESPACE, parent, item, site)
        if payload is not None:
            return site['window_index'], payload, dict(cache_hit=True, worker_pid=os.getpid())
        require(WORKER_MODE == 'build', 'Complete immutable TRAIN cache required; missing window is fatal')
    else:
        require(WORKER_MODE == 'heldout', 'Held-out observations only after fixed checkpoint gate')
    if WORKER_SOURCE is None or WORKER_SOURCE.parent_id != parent:
        WORKER_SOURCE = ObservationSource(parent, item)
    state = prepared_to_state(prepare_observations(WORKER_SOURCE.extract(site)))
    payload = encode(state, level=9)
    if item['split'] == 'train':
        cache_v2.write_cache(WORKER_NAMESPACE, parent, item, site, payload)
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


def pool_for(protocol, mode='read'):
    # Workers receive public observation bindings only; no truth references.
    require(protocol['workers'] == 16 and mode in ('read', 'build', 'heldout'), 'Registered worker execution')
    return ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('spawn'), initializer=worker_init,
                               initargs=(protocol['parents'], protocol['cache_namespace'], mode))


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
    cache_v2.initialize()
    out = RESULTS / 'cache_build_v2'
    out.mkdir(exist_ok=False)
    completed, start = {}, time.monotonic()
    largest = dict(nodes=[], SNP_rows=[], prepared_tensor_bytes=[])
    with pool_for(protocol, 'build') as pool:
        for parent, item in protocol['parents'].items():
            if item['split'] != 'train':
                continue
            source = ObservationSource(parent, item)
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
    require(len(completed) == 20 and sum(v['windows'] for v in completed.values()) == 158476 and sum(v['targets'] for v in completed.values()) == 200000,
            'Complete original20-parent TRAIN cache coverage')
    actual = tree_bytes(CACHE)
    reserved = int(cache_v2.COUNTER.read_text())
    require(actual <= reserved <= cache_v2.CACHE_CAP, 'Actual completed cache within charged63GiB cap')
    print(write(out / 'product.json', dict(complete=True, freeze_ref=freeze_ref, cache_namespace=protocol['cache_namespace'],
        parents=completed, largest_TRAIN_windows=largest, actual_scope_bytes=actual, conservatively_reserved_bytes=reserved,
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
    cache_product = read(ref(RESULTS / 'cache_build_v2/product.json'))
    require(cache_product['complete'] and cache_product['freeze_ref'] == freeze_ref and cache_product['cache_namespace'] == protocol['cache_namespace'], 'Complete frozen TRAIN cache gate')
    out = RESULTS / 'fit_v2'
    out.mkdir(exist_ok=False)
    selected = [p for p, row in protocol['parents'].items() if row['split'] == 'train']
    plans, labels, row_refs = {}, {}, {}
    # Public plans seal in RAM before same-parent TRAIN truth joins. Held-out inputs are untouched.
    for parent in selected:
        source = ObservationSource(parent, protocol['parents'][parent])
        plans[parent] = source.plan()
        labels[parent] = train_targets(parent, protocol, source.records)
        row_refs[parent] = dict(windows=len(plans[parent]), targets=len(labels[parent]),
                                focal_registry_ref=protocol['parents'][parent]['binding']['focal_registry_ref'])
        del source
    count = sum(len(v) for v in plans.values())
    targets = sum(len(v) for v in labels.values())
    total_batches = sum((len(v) + 15) // 16 for v in plans.values())
    require(count == 158476, 'Every original TRAIN window')
    require(targets == 200000, 'Every original American TRAIN target')
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
    with pool_for(protocol) as pool, (out / 'progress.jsonl').open('x') as progress:
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
            check_disk()
    checkpoint = out / 'final_state.pt'
    save_torch(checkpoint, {key: value.detach().cpu() for key, value in model.state_dict().items()})
    product = dict(schema='american_joint_fixed_final_checkpoint_v1', model_id=MODEL_ID, complete=True,
        freeze_ref=freeze_ref, checkpoint_ref=ref(checkpoint), training_plan_ref=plan_ref, epochs=protocol['epochs'],
        updates=updates, target_presentations=targets * protocol['epochs'], heldout_observations_or_truth_opened=False,
        elapsed_seconds=time.monotonic() - start, completed_utc=now())
    print(write(out / 'product.json', product), flush=True)


def predict(args, protocol, freeze_ref):
    parent = args.parent
    require(parent in protocol['parents'] and protocol['parents'][parent]['split'] != 'train', 'Original held-out parent only')
    fit = read(ref(RESULTS / 'fit_v2/product.json'))
    require(fit['complete'] and fit['model_id'] == MODEL_ID and fit['freeze_ref'] == freeze_ref and
            fit['epochs'] == protocol['epochs'] and fit['target_presentations'] == protocol['training_target_presentations'], 'Fixed completed TRAIN checkpoint gate')
    checkpoint = verify(fit['checkpoint_ref'])
    target_device = device(args.gpu)
    model = JointAncestryModel(**protocol['model'])
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'), strict=True)
    model.eval().to(target_device)
    out = RESULTS / 'predictions_v2' / parent
    out.parent.mkdir(exist_ok=True)
    out.mkdir(exist_ok=False)
    # Seal checkpoint before opening this parent's observation payloads.
    write(out / 'started.json', dict(freeze_ref=freeze_ref, checkpoint_ref=fit['checkpoint_ref'], parent_id=parent, utc=now()))
    source = ObservationSource(parent, protocol['parents'][parent])
    sites = source.plan()
    plan_ref = write(out / 'site_plan.json', dict(parent_id=parent, rows=10000, sites=sites))
    logits = np.full((10000, 3), np.nan, dtype=np.float32)
    seen = np.zeros(10000, dtype=bool)
    batch, rows = [], []
    with pool_for(protocol, 'heldout') as pool:
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
    require(seen.all() and np.isfinite(logits).all(), 'All10000 canonical predictions complete')
    values = np.exp(logits.astype(np.float64) - logits.max(1, keepdims=True))
    prediction = {key: source.records[key] for key in IDENTITY}
    prediction['probabilities'] = (values / values.sum(1, keepdims=True)).astype(np.float32)
    verify_prediction(prediction, source.records)
    source.check_stats()
    prediction_ref = save_npz(out / 'predictions.npz', prediction)
    logits_ref = save_npz(out / 'logits.npz', dict(logits=logits, record_ids=source.records['record_ids']))
    print(write(out / 'product.json', dict(schema='american_joint_native3_prediction_v1', complete=True, model_id=MODEL_ID,
         parent_id=parent, split=protocol['parents'][parent]['split'], rows=10000, class_order=list(CLASSES),
         prediction_ref=prediction_ref, logits_ref=logits_ref, focal_registry_ref=protocol['parents'][parent]['binding']['focal_registry_ref'],
         freeze_ref=freeze_ref, checkpoint_ref=fit['checkpoint_ref'], site_plan_ref=plan_ref,
         query_ancestry_truth_opened=False, completed_utc=now())), flush=True)
    check_disk()


def score(args, protocol, freeze_ref):
    parents = [p for p, row in protocol['parents'].items() if row['split'] != 'train']
    sealed = {}
    fit = read(ref(RESULTS / 'fit_v2/product.json'))
    verify(fit['checkpoint_ref'])
    require(fit['complete'] and fit['freeze_ref'] == freeze_ref, 'Exact completed fixed-final checkpoint')
    # Authenticate every15-parent prediction and original canonical identity BEFORE resolving any held-out truth.
    for parent in parents:
        product_ref = ref(RESULTS / 'predictions_v2' / parent / 'product.json')
        product = read(product_ref)
        require(product['model_id'] == MODEL_ID and product['freeze_ref'] == freeze_ref and product['complete'] and
                product['rows'] == 10000 and product['parent_id'] == parent and product['query_ancestry_truth_opened'] is False and
                product['checkpoint_ref'] == fit['checkpoint_ref'] and
                product['focal_registry_ref'] == protocol['parents'][parent]['binding']['focal_registry_ref'], 'Complete canonical new-model seal')
        records = load_npz(product['focal_registry_ref'])
        prediction = load_npz(product['prediction_ref'])
        verify_prediction(prediction, records)
        sealed[parent] = (records, prediction, product_ref)
    bindings = read(protocol['postseal_scoring_bindings_ref'])['parents']
    out = RESULTS / 'scores_v2'
    out.mkdir(exist_ok=False)
    write(out / 'all15_prediction_seal.json', dict(freeze_ref=freeze_ref, parents={p: v[2] for p, v in sealed.items()}, utc=now()))
    results = {}
    for parent in parents:
        completion = json.loads(path(bindings[parent]['private_completion_path']).read_text())
        generation = read(bindings[parent]['generation_ready_ref'])
        require(completion['parent_id'] == parent and completion['status'] == 'completed_C3_simulation_reconstruction' and
                completion['class_order'] == list(CLASSES) and completion['observation_ref'] == generation['observation_ref'],
                'Exact original same-parent private completion')
        truth_ref = completion['private_truth_ref']
        require(path(truth_ref).is_relative_to(path(bindings[parent]['private_completion_path']).parent / 'truth'),
                'Only original same-generator private truth may be resolved')
        records, prediction, product_ref = sealed[parent]
        true = join(records, load_npz(truth_ref, cap=8 * 2**30))
        calls = prediction['probabilities'].argmax(1)
        confusion = np.zeros((3, 3), dtype=np.int64)
        np.add.at(confusion, (true, calls), 1)
        result = metrics(confusion)
        chosen = prediction['probabilities'][np.arange(len(true)), true].astype(np.float64)
        result.update(NLL=None if (chosen == 0).any() else float(-np.log(chosen).mean()),
                      NLL_status='positive_infinity' if (chosen == 0).any() else 'finite',
                      Brier=float(((prediction['probabilities'].astype(np.float64) - np.eye(3)[true])**2).sum(1).mean()))
        results[parent] = dict(split=protocol['parents'][parent]['split'], result=result, truth_ref=truth_ref, prediction_product_ref=product_ref)
        write(out / f'{parent}.json', results[parent])
    summary = {}
    for split in ('validation', 'test'):
        selected = [v['result'] for v in results.values() if v['split'] == split]
        summary[split] = {key: dict(mean=float(np.mean([v[key] for v in selected])), sample_SD=float(np.std([v[key] for v in selected], ddof=1)))
                          for key in ('accuracy', 'macro_F1', 'Brier')}
    print(write(out / 'product.json', dict(schema='american_joint_common_native3_scores_v1', model_id=MODEL_ID,
               complete=True, parents=results, summary=summary, scoring_function='arggnn.american_c3_evaluate_v1.metrics',
               all15_predictions_sealed_before_truth=True, probabilities_modified=False, freeze_ref=freeze_ref)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare-cache', 'train', 'predict', 'score'])
    parser.add_argument('--freeze', default=str(BASE / 'freeze_v2.json'))
    parser.add_argument('--gpu', type=int, choices=[3, 4, 5])
    parser.add_argument('--parent')
    args = parser.parse_args()
    seal, protocol = frozen(args.freeze)
    require(args.stage in ('prepare-cache', 'score') or args.gpu is not None, 'Assigned physical GPU required')
    require(args.stage != 'predict' or args.parent is not None, 'One exact held-out parent required')
    check_disk()
    {'prepare-cache': prepare_cache, 'train': train, 'predict': predict, 'score': score}[args.stage](args, protocol, ref(args.freeze))


if __name__ == '__main__':
    main()
