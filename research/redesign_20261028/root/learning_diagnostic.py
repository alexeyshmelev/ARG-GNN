"""Registered bounded learning diagnostic, separate from published benchmarks.

CPU only. Fits on TRAIN000/001, checks internal development TRAIN002. The
observation loader whitelists model arrays and labels stay outside its API.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from research.redesign_20261028.architecture import window_model_frozen_v1 as wm

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
WORK = ROOT / 'research/redesign_20261028'
MODEL_SHA = 'e0528c8e0d01b69512fe588dc74ec966a3855489abf4160a8ef4b9111e5a4e12'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_order(count, updates, batch_size, seed):
    """Repeated fresh permutations, with a fixed final partial epoch."""
    if count % batch_size:
        raise ValueError('Registered support must divide into full batches')
    rng = np.random.default_rng(seed)
    batches = []
    while len(batches) < updates:
        order = rng.permutation(count)
        batches.extend(order[i:i + batch_size].tolist()
                       for i in range(0, count, batch_size))
    return batches[:updates]


def score(logits, targets, classes=6):
    if logits.shape != (len(targets), classes) or not torch.isfinite(logits).all():
        raise ValueError('Invalid logits')
    predictions = logits.argmax(-1)
    matrix = torch.bincount(targets * classes + predictions,
                            minlength=classes * classes).reshape(classes, classes)
    diagonal = matrix.diagonal().double()
    denominator = matrix.sum(0).double() + matrix.sum(1).double()
    f1 = torch.where(denominator > 0, 2 * diagonal / denominator,
                     torch.zeros_like(diagonal))
    return {
        'draws': len(targets),
        'cross_entropy': float(F.cross_entropy(logits, targets)),
        'accuracy': float(diagonal.sum() / len(targets)),
        'macro_f1_six_classes': float(f1.mean()),
        'class_f1': f1.tolist(),
        'confusion_true_rows_predicted_columns': matrix.tolist(),
        'predicted_class_counts': matrix.sum(0).tolist(),
        'true_class_counts': matrix.sum(1).tolist(),
    }


class Packets(Dataset):
    def __init__(self, rows, directory):
        self.rows = rows
        self.directory = Path(directory)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        started = time.perf_counter()
        row = self.rows[index]
        observations = wm.load_observations(self.directory / row['packet_path'])
        return observations, int(row['label']), int(index), time.perf_counter() - started


def worker_init(worker_id):
    torch.set_num_threads(1)


def collate(items):
    started = time.perf_counter()
    observations = wm.collate_observations([item[0] for item in items])
    batch = wm.prepare_observations(observations)
    targets = torch.tensor([item[1] for item in items], dtype=torch.long)
    indices = [item[2] for item in items]
    if len(batch.focal_tree) != len(items) or not batch.core_focal_mask.all():
        raise ValueError('Exactly one forced focal query required per packet')
    return batch, targets, indices, {
        'packet_read_worker_seconds': sum(item[3] for item in items),
        'collation_preparation_worker_seconds': time.perf_counter() - started,
        'nodes': len(batch.parent), 'trees': len(batch.leaves),
        'sites': len(batch.site_genotype),
    }


def loader(dataset, workers, batches=None, batch_size=4):
    options = dict(num_workers=workers, worker_init_fn=worker_init,
                   collate_fn=collate, multiprocessing_context='spawn',
                   persistent_workers=False, prefetch_factor=2)
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **options)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **options)


@torch.no_grad()
def evaluate(model, dataset, arm, workers, batch_size):
    model.eval()
    logits = []
    labels = []
    indices = []
    start = time.perf_counter()
    for batch, target, index, stats in loader(dataset, workers, batch_size=batch_size):
        result = model(batch, mode=arm)
        logits.append(result.cpu())
        labels.append(target)
        indices.extend(index)
    if indices != list(range(len(dataset))):
        raise ValueError('Evaluation support order changed')
    logits = torch.cat(logits)
    targets = torch.cat(labels)
    result = score(logits, targets)
    result['elapsed_seconds'] = time.perf_counter() - start
    return result, logits.numpy(), targets.numpy()


def verify_manifest(manifest_path, protocol):
    directory = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('protocol_sha256') != sha(WORK / 'root/pilot_protocol.json'):
        raise ValueError('Packet registration differs')
    fitting = manifest['fit']
    development = manifest['development']
    if len(fitting) != 128 or len(development) != 64:
        raise ValueError('Diagnostic support mismatch')
    for split, rows, allowed in (
        ('fit', fitting, set(protocol['fitting_parent_ids'])),
        ('development', development, {protocol['development_parent_id']}),
    ):
        for index, row in enumerate(rows):
            if row['draw_index'] != index or row['parent_id'] not in allowed:
                raise ValueError('Split or draw identity mismatch')
            label_path = (directory / row['label_path']).resolve()
            if not label_path.is_relative_to(directory.resolve()):
                raise ValueError('Label escapes registered directory')
            if sha(label_path) != row['label_sha256']:
                raise ValueError('Label file changed')
            target = np.load(label_path, allow_pickle=False)
            if target.dtype != np.int64 or target.size != 1:
                raise ValueError('One int64 target per packet')
            row['label'] = int(target.reshape(-1)[0])
            if not 0 <= row['label'] < 6:
                raise ValueError('Label range')
            path = (directory / row['packet_path']).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Packet escapes registered directory')
            if sha(path) != row['packet_sha256']:
                raise ValueError('Packet changed')
    return manifest, Packets(fitting, directory), Packets(development, directory)


def run(args):
    protocol_path = WORK / 'root/pilot_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if args.arm not in protocol['arms']:
        raise ValueError('Unregistered arm')
    if sha(Path(wm.__file__)) != MODEL_SHA:
        raise ValueError('Frozen model changed')
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_relative_to(WORK / 'diagnostic_packets'):
        raise ValueError('Unexpected packet source')
    manifest, fit, development = verify_manifest(manifest_path, protocol)
    config = protocol['training']
    output = WORK / 'root/learning_runs' / args.arm
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'started.json'
    if receipt.exists():
        raise FileExistsError('Never overwrite a previous diagnostic attempt')
    torch.set_num_threads(config['cpu_threads_per_fit'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(protocol['model']['model_seed'])
    np.random.seed(protocol['model']['model_seed'])
    model = wm.WindowTreeLoom(width=protocol['model']['width'],
                             classes=protocol['model']['classes'], scales=3)
    batches = make_order(len(fit), config['updates'], config['batch_size'],
                         config['batch_seed'])
    order_bytes = np.asarray(batches, dtype=np.int64).tobytes()
    initial_state = state_digest(model)
    started = time.perf_counter()
    metadata = {
        'arm': args.arm, 'pid': os.getpid(), 'device': 'cpu',
        'model_source_sha256': MODEL_SHA,
        'harness_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
        'manifest_sha256': sha(manifest_path),
        'initial_parameter_sha256': initial_state,
        'batch_order_sha256': hashlib.sha256(order_bytes).hexdigest(),
        'batch_order': batches, 'configuration': protocol,
        'torch_version': torch.__version__,
        'scope': 'small learnability diagnostic, no TEST access',
    }
    save_json(receipt, metadata)
    save_json(output / 'status.json', {'stage': 'initial_evaluation', 'updates': 0})
    initial = {}
    for split, dataset in (('fit', fit), ('development', development)):
        metrics, logits, labels = evaluate(model, dataset, args.arm,
                                          config['data_loader_workers'], config['batch_size'])
        initial[split] = metrics
        np.savez(output / ('initial_' + split + '.npz'), logits=logits, labels=labels)
        save_json(output / 'initial_metrics.json', initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  betas=tuple(config['betas']), eps=config['epsilon'],
                                  weight_decay=config['weight_decay'])
    train_loader = loader(fit, config['data_loader_workers'], batches=batches)
    model.train()
    timings = {key: 0.0 for key in ('loader_wait_seconds', 'forward_seconds',
                                  'backward_seconds', 'optimizer_seconds',
                                  'packet_read_worker_seconds',
                                  'collation_preparation_worker_seconds')}
    losses = []
    clipped = 0
    iterator = iter(train_loader)
    train_started = time.perf_counter()
    for update in range(config['updates']):
        start = time.perf_counter()
        batch, targets, indices, data_stats = next(iterator)
        waited = time.perf_counter() - start
        if indices != batches[update]:
            raise ValueError('Training draw order changed')
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        logits = model(batch, mode=args.arm)
        loss = F.cross_entropy(logits, targets)
        forward = time.perf_counter() - start
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite forward')
        start = time.perf_counter()
        loss.backward()
        backward = time.perf_counter() - start
        used = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.grad is not None]
        if not used or any(not torch.isfinite(parameter.grad).all()
                           for _, parameter in used):
            raise FloatingPointError('Nonfinite used gradient')
        if args.arm == 'structure' and len(used) != len(list(model.parameters())):
            raise ValueError('Disconnected structure parameter')
        start = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_norm_clip'],
                                             error_if_nonfinite=True)
        optimizer.step()
        optimizer_seconds = time.perf_counter() - start
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError('Nonfinite updated parameter')
        value = float(loss.detach())
        losses.append(value)
        clipped += int(norm > config['gradient_norm_clip'])
        measurements = dict(loader_wait_seconds=waited, forward_seconds=forward,
                            backward_seconds=backward, optimizer_seconds=optimizer_seconds,
                            packet_read_worker_seconds=data_stats['packet_read_worker_seconds'],
                            collation_preparation_worker_seconds=data_stats['collation_preparation_worker_seconds'])
        for key, value_time in measurements.items():
            timings[key] += value_time
        row = dict(update=update + 1, loss=value, gradient_norm_before_clip=float(norm),
                   active_parameter_tensors=len(used), indices=indices,
                   elapsed_train_seconds=time.perf_counter() - train_started,
                   **measurements, nodes=data_stats['nodes'], sites=data_stats['sites'])
        with (output / 'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if (update + 1) % 20 == 0:
            save_json(output / 'status.json', {'stage': 'training', **row,
                'learned_length_scales_bp': model.log_lengths.detach().exp().tolist(),
                'residual_gates': model.residual_gate.detach().sigmoid().tolist()})
            print(json.dumps({'arm': args.arm, 'update': update + 1,
                              'mean_recent_loss': float(np.mean(losses[-20:])),
                              'elapsed_train_seconds': row['elapsed_train_seconds']}), flush=True)
    # Consume exhaustion so both preparation workers shut down before evaluation.
    try:
        next(iterator)
        raise ValueError('More than registered updates')
    except StopIteration:
        pass
    training_seconds = time.perf_counter() - train_started
    save_json(output / 'status.json', {'stage': 'final_evaluation', 'updates': len(losses)})
    final = {}
    for split, dataset in (('fit', fit), ('development', development)):
        metrics, logits, labels = evaluate(model, dataset, args.arm,
                                          config['data_loader_workers'], config['batch_size'])
        final[split] = metrics
        np.savez(output / ('final_' + split + '.npz'), logits=logits, labels=labels)
    torch.save(model.state_dict(), output / 'final_state.pt')
    result = {key: value for key, value in metadata.items() if key != 'batch_order'}
    result.update(initial=initial, final=final, updates=len(losses),
                  training_elapsed_seconds=training_seconds, timings=timings,
                  first20_mean_loss=float(np.mean(losses[:20])),
                  last20_mean_loss=float(np.mean(losses[-20:])),
                  clipped_updates=clipped,
                  final_parameter_sha256=state_digest(model),
                  learned_length_scales_bp=model.log_lengths.detach().exp().tolist(),
                  residual_gates=model.residual_gate.detach().sigmoid().tolist(),
                  peak_main_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  elapsed_seconds=time.perf_counter() - started,
                  cuda_initialized=torch.cuda.is_initialized())
    if result['cuda_initialized']:
        raise ValueError('CPU diagnostic initialized CUDA')
    save_json(output / 'result.json', result)
    total = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if total > protocol['resource_scope']['max_run_output_bytes_per_arm']:
        raise ValueError('Registered output cap exceeded')
    save_json(output / 'status.json', {'stage': 'completed', 'updates': len(losses),
                                      'output_bytes': total})
    print(json.dumps({'arm': args.arm, 'completed': True, 'final': final}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['no_cross', 'structure'], required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        output = WORK / 'root/learning_runs' / args.arm
        output.mkdir(parents=True, exist_ok=True)
        failure = output / ('failure_' + str(os.getpid()) + '.txt')
        failure.write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
