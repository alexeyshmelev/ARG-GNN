"""Exact original loader/evaluation functions, isolated from optional NVML telemetry."""
import hashlib
import json
import time
import setproctitle
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from research.joint_model_20261029.data.site_dataset import collate_sites
from research.joint_model_20261029.data.loader_shutdown import register_worker_resource_cleanup


def initialize_worker(worker_id):
    torch.set_num_threads(1)
    setproctitle.setproctitle('AI Agent Research Data')
    register_worker_resource_cleanup()


def make_loader(dataset, batch_sites, workers, seed, shuffle):
    return DataLoader(dataset, batch_size=batch_sites, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=workers, persistent_workers=bool(workers),
                      prefetch_factor=2 if workers else None, pin_memory=True,
                      collate_fn=collate_sites, worker_init_fn=initialize_worker,
                      multiprocessing_context='spawn' if workers else None,
                      drop_last=False)


def scores(logits, labels, classes):
    loss = F.cross_entropy(logits, labels, reduction='sum').item()
    pred = logits.argmax(1)
    confusion = torch.bincount(labels * classes + pred, minlength=classes**2).reshape(classes, classes)
    c = confusion.to(torch.float64)
    denominator = c.sum(0) + c.sum(1)
    f1 = torch.where(denominator > 0, 2*c.diag()/denominator, torch.zeros_like(denominator))
    return dict(targets=len(labels), accuracy=float(c.diag().sum()/len(labels)),
                cross_entropy=loss/len(labels), macro_f1=float(f1.mean()),
                confusion=confusion.tolist(), true_counts=confusion.sum(1).tolist(),
                predicted_counts=confusion.sum(0).tolist())


@torch.no_grad()
def evaluate(model, loader, device, mode, expected_ids, classes, output=None):
    start = time.perf_counter()
    model.eval()
    logits, labels, identities = [], [], []
    for cpu in loader:
        batch = cpu.to(device, non_blocking=True)
        value = model(batch.observations, mode=mode)
        if not torch.isfinite(value).all().item():
            raise FloatingPointError('Nonfinite evaluation logits')
        logits.append(value.cpu())
        labels.append(batch.labels.cpu())
        identities.extend(cpu.canonical_ids)
    value = torch.cat(logits)
    truth = torch.cat(labels)
    assert len(truth) == len(expected_ids)
    assert len(set(identities)) == len(expected_ids)
    assert set(identities) == set(expected_ids), 'Evaluation IDs differ from registered canonical support'
    result = scores(value, truth, classes)
    result['seconds'] = time.perf_counter() - start
    result['canonical_identity_set_sha256'] = hashlib.sha256(json.dumps(sorted(identities), separators=(',',':')).encode()).hexdigest()
    if output is not None:
        np.savez_compressed(output, logits=value.numpy(), labels=truth.numpy(),
                            parent_id=np.array([x[0] for x in identities]),
                            example_index=np.array([x[1] for x in identities], dtype=np.int64))
    return result

