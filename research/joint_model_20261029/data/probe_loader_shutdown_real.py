"""Verify real sparse-site teardown and isolation from another owned test loader."""
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader,Subset
from .site_dataset import CachedSiteDataset,collate_sites
from .loader_shutdown import register_worker_resource_cleanup,shutdown_loaders
from .probe_loader_shutdown import Fixture

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')


def initialize_worker(_):
    torch.set_num_threads(1)
    register_worker_resource_cleanup()


def main():
    started=time.perf_counter();torch.set_num_threads(1)
    other=DataLoader(Fixture(),batch_size=4,num_workers=2,persistent_workers=True,
        multiprocessing_context='spawn',prefetch_factor=2,worker_init_fn=initialize_worker)
    other_iterator=iter(other);next(other_iterator)
    other_pids=[w.pid for w in other_iterator._workers]
    dataset=CachedSiteDataset(ROOT/'research/joint_model_20261029/data/cache_v1/manifest.json',mode='no_cross')
    target=DataLoader(Subset(dataset,range(16)),batch_size=4,num_workers=2,persistent_workers=True,
        multiprocessing_context='spawn',prefetch_factor=2,worker_init_fn=initialize_worker,collate_fn=collate_sites)
    iterator=iter(target);target_pids=[w.pid for w in iterator._workers];count=0;ids=[]
    try:
        for batch in iterator:
            count+=len(batch.labels);ids.extend(batch.canonical_ids)
        target_result=shutdown_loaders(target)
        assert all(w.is_alive() for w in other_iterator._workers),'Other loader remains alive'
        another=next(other_iterator)
        assert len(another['index'])==4,'Other loader still produces next data'
        assert all(record['worker_exitcodes']==[0,0] for record in target_result)
        print(json.dumps(dict(status='passed',real_sites=16,real_targets=count,unique_ids=len(set(ids)),
            target_worker_pids=target_pids,target_shutdown=target_result,
            unrelated_test_loader_pids=other_pids,other_loader_preserved_and_functional=True,
            wall_seconds=time.perf_counter()-started)),flush=True)
    finally:
        shutdown_loaders(target,other)


if __name__=='__main__':main()
