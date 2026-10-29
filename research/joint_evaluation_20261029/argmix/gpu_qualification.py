"""Post-cache capacity probe on globally largest TRAIN windows, with no held-out reads."""
import argparse
from collections import defaultdict
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel, prepared_from_state, collate_prepared
from .runtime import BASE, RESULTS, ref, read, require, write, now, require_wait, train_targets
from .pipeline import device, check_ram
from .cache import read_cache
from research.joint_evaluation_20261029.american.state_codec_v2 import decode
from .observations import ObservationSource


def qualify(args, protocol, freeze_ref):
    cache_ref, _ = require_wait(RESULTS / 'cache_build/completed.json', 'joint_argmix.train_cache')
    cache = read(cache_ref)
    require(cache['complete'] and cache['freeze_ref'] == freeze_ref and cache['cache_namespace'] == protocol['cache_namespace'], 'Completed exact TRAIN cache before global capacity qualification')
    selections = cache['largest_TRAIN_windows']
    needed = defaultdict(set)
    for group in selections.values():
        require(len(group) == 16, 'Exactly16 complete windows in each capacity batch')
        for row in group:
            needed[row['parent_id']].add(row['window_index'])
    states, labels = {}, {}
    for parent, indices in needed.items():
        require(protocol['parents'][parent]['split'] == 'train', 'TRAIN-only capacity probe')
        source = ObservationSource(parent, cache['public_bindings'][parent]['observation_ref'])
        sites = source.plan()
        targets, _ = train_targets(parent, protocol, source)
        for index in indices:
            site = sites[index]
            payload = read_cache(protocol['cache_namespace'], parent, cache['public_bindings'][parent], site)
            require(payload is not None, 'Registered cached capacity window exists')
            batch = prepared_from_state(decode(payload))
            require(np.array_equal(batch.focal_sample.numpy(), site['sample_indices']), 'Exact capacity target identity')
            states[(parent, index)] = batch
            labels[(parent, index)] = targets[site['example_indices']]
        source.check_stats()
    target_device = device(args.gpu)
    torch.manual_seed(protocol['initialization_seed'])
    model = JointAncestryModel(**protocol['model']).to(target_device)
    settings = protocol['optimizer']
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], betas=tuple(settings['betas']),
                                 eps=settings['epsilon'], weight_decay=settings['weight_decay'])
    results = []
    for name, group in selections.items():
        keys = [(row['parent_id'], row['window_index']) for row in group]
        batch = collate_prepared([states[key] for key in keys]).pin_memory().to(target_device, non_blocking=True)
        target = torch.as_tensor(np.concatenate([labels[key] for key in keys]), device=target_device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch, mode='structure')
            # Diagnostic mean CE only; these temporary parameters never become a benchmark checkpoint.
            loss = F.cross_entropy(prediction, target)
            require(torch.isfinite(loss).item(), 'Finite capacity diagnostic loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
            require(torch.isfinite(norm).item(), 'Finite capacity diagnostic gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        results.append(dict(selection=name, windows=group, targets=len(target), losses=losses,
            two_updates_seconds=time.monotonic() - begin, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), owned_CPU_rss_bytes=check_ram()))
        del batch, prediction, loss, target
    receipt = dict(passed=True, recorded_utc=now(), physical_gpu=args.gpu, freeze_ref=freeze_ref,
        source_ref=ref(BASE / 'gpu_qualification.py'), complete_TRAIN_cache_ref=cache_ref,
        largest_across_all_registered_TRAIN_windows=True, selection_axes=list(selections), full_minibatch_windows=16,
        sampled_only=False, heldout_observations_or_truth_opened=False, benchmark_checkpoint_written=False,
        maximum_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in results),
        maximum_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in results), groups=results)
    print(write(RESULTS / 'gpu_qualification/completed.json', receipt), flush=True)
