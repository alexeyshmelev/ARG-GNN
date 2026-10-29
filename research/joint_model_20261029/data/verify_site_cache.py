"""CPU cache integration checks, canonical bindings, and real loader throughput."""
from dataclasses import fields
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .site_dataset import CachedSiteDataset,collate_sites,require
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel

HERE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029/data')
MANIFEST=HERE/'cache_v1/manifest.json'


def initialize_worker(_):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def tensor_digest(value,hasher):
    if isinstance(value,torch.Tensor):
        hasher.update(str((value.dtype,tuple(value.shape),value.layout)).encode())
        if value.layout==torch.sparse_csr:
            for array in (value.crow_indices(),value.col_indices(),value.values()):tensor_digest(array,hasher)
        else:hasher.update(value.detach().contiguous().numpy().tobytes())
    elif isinstance(value,(list,tuple)):
        for child in value:tensor_digest(child,hasher)


def batch_digest(batch):
    value=hashlib.sha256()
    for field in fields(batch.observations):tensor_digest(getattr(batch.observations,field.name),value)
    tensor_digest(batch.labels,value)
    value.update(json.dumps(batch.canonical_ids).encode())
    return value.hexdigest()


def scan_loader(workers,epochs):
    ds=CachedSiteDataset(MANIFEST,split='fit')
    loader=DataLoader(ds,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(20261029),
        num_workers=workers,persistent_workers=bool(workers),prefetch_factor=2 if workers else None,
        pin_memory=False,collate_fn=collate_sites,worker_init_fn=initialize_worker,
        multiprocessing_context='spawn' if workers else None)
    receipts=[]
    for epoch in range(epochs):
        started=time.perf_counter();identities=[];signature=hashlib.sha256();batch_count=0;site_count=0
        for batch in loader:
            identities.extend(batch.canonical_ids);signature.update(batch_digest(batch).encode())
            require(batch.loss_scale==ds.site_count/(batch.site_count*ds.target_count),'Actual minibatch target loss scale')
            batch_count+=1;site_count+=batch.site_count
        require(len(identities)==ds.target_count and len(set(identities))==ds.target_count,'Every fit target once per epoch')
        require(set(identities)==set(ds.canonical_ids) and site_count==ds.site_count,'Exact registered fit coverage')
        receipts.append(dict(workers=workers,epoch=epoch,batches=batch_count,sites=site_count,targets=len(identities),
            tensor_label_order_sha256=signature.hexdigest(),wall_seconds=time.perf_counter()-started))
    return receipts


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261029)
    started=time.perf_counter();ds=CachedSiteDataset(MANIFEST,split='full_train')
    split_counts={}
    for split in ('fit','development','full_train'):
        selected=CachedSiteDataset(MANIFEST,split=split)
        split_counts[split]=dict(sites=selected.site_count,targets=selected.target_count,
            identity_sha256=hashlib.sha256(json.dumps(selected.canonical_ids).encode()).hexdigest())
    first=ds[0];again=ds[0]
    require(first is again and ds.cache_hits==1,'Worker-local LRU avoids repeated deserialization')
    second=ds[1];packed=collate_sites([first,second]);before=batch_digest(packed)
    require(packed.canonical_ids==first.canonical_ids+second.canonical_ids,'Collation canonical row order')
    require(batch_digest(packed.to('cpu'))==before,'SiteBatch CPU move preserves data')
    original_labels=first.labels.clone();packed.labels[0]=(packed.labels[0]+1)%6
    require(torch.equal(first.labels,original_labels),'Collation does not mutate immutable cached labels')
    packed=collate_sites([first,second]);model=JointAncestryModel(width=16,layers=2).eval();errors={}
    with torch.no_grad():
        for mode in ('no_cross','leaf_only','structure'):
            together=model(packed.observations,mode=mode)
            separate=torch.cat([model(item.observations,mode=mode) for item in (first,second)])
            difference=float((together-separate).abs().max())
            require(torch.allclose(together,separate,atol=2e-6,rtol=2e-5),'Cached collation model isolation: '+mode)
            errors[mode]=difference
    focal_ds=CachedSiteDataset(MANIFEST,split='full_train',mode='no_cross')
    focal_items=[focal_ds[0],focal_ds[1]];focal=collate_sites(focal_items)
    require(focal.canonical_ids==packed.canonical_ids and torch.equal(focal.labels,packed.labels),'Focal-only view preserves label/identity bindings')
    require(focal.observations.leaves.shape[0]==2,'No-cross retains one complete focal tree per site')
    require(focal_ds[0] is focal_items[0] and focal_ds.cache_hits==1,'Focal-only view cached once')
    with torch.no_grad():
        reduced=model(focal.observations,mode='no_cross');complete=model(packed.observations,mode='no_cross')
        require(torch.allclose(reduced,complete,atol=2e-6,rtol=2e-5),'Dataset focal-only view exact outputs')
        errors['focal_only_no_cross']=float((reduced-complete).abs().max())
    costs=[row['prepared_ref']['bytes']+row['labels_ref']['bytes'] for row in ds.windows[:2]]
    limited=CachedSiteDataset(MANIFEST,cache_bytes=max(costs));one=limited[0];limited[1];reloaded=limited[0]
    require(limited.cache_hits==0 and limited.cache_misses==3,'Bounded LRU evicts over-cap tensors')
    require(batch_digest(collate_sites([one]))==batch_digest(collate_sites([reloaded])),'LRU retrieval preserves complete tensors and labels')
    failures={}
    with tempfile.TemporaryDirectory(prefix='cache_integrity_',dir=HERE) as scratch_name:
        scratch=Path(scratch_name);row=dict(ds.windows[0])
        for name in ('prepared_ref','labels_ref'):
            ref=dict(row[name]);source=ds.cache_root/ref['path'];ref['path']=source.name
            (scratch/ref['path']).write_bytes(source.read_bytes());row[name]=ref
        probe=CachedSiteDataset(MANIFEST);probe.cache_root=scratch;probe.windows=(row,)
        probe[0]
        path=scratch/row['prepared_ref']['path'];raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        try:probe[0]
        except ValueError as error:failures['changed_cached_tensor']=str(error)
        else:raise AssertionError('Mutated cached tensor must fail SHA256')
        bad=json.loads(MANIFEST.read_text());bad['model_source_ref']['sha256']='0'*64
        invalid=scratch/'bad_model_manifest.json';invalid.write_text(json.dumps(bad))
        try:CachedSiteDataset(invalid)
        except ValueError as error:failures['model_source_mismatch']=str(error)
        else:raise AssertionError('Wrong model SHA must fail')
    per_site=[np.asarray([2.]),np.asarray([1.,5.]),np.asarray([2.,3.,4.,8.,9.])]
    n=sum(map(len,per_site));m=len(per_site);s=2
    estimators=[sum(per_site[i].sum() for i in choice)*m/(s*n) for choice in itertools.combinations(range(m),s)]
    expected=sum(x.sum() for x in per_site)/n
    require(np.isclose(np.mean(estimators),expected),'Unequal-size site sampling is unbiased for target mean')
    chosen=[per_site[0],per_site[2]];factor=m/(len(chosen)*n)
    require(np.isclose(sum(x.sum()*factor for x in chosen),sum(x.sum() for x in chosen)*factor),'Microbatch scale uses original site count')
    zero=scan_loader(0,1);four=scan_loader(4,2)
    require(zero[0]['tensor_label_order_sha256']==four[0]['tensor_label_order_sha256'],
            'Zero-worker and four-worker shuffled epochs have identical tensors, labels and row order')
    result=dict(status='passed',model_device='cpu',split_counts=split_counts,worker_cache_byte_cap=ds.cache_bytes,
        LRU_hit_reuses_immutable_state=True,LRU_eviction_preserves_tensors=True,mutation_rejections=failures,
        focal_only_no_cross_preserves_outputs_labels_ids=True,focal_only_cache_charge='Conservative full serialized bytes',
        collation_model_max_abs_errors=errors,unequal_site_target_mean_weighting=True,microbatch_factor_preserved=True,
        real_loader_epochs=zero+four,workers_zero_and_four_exact=True,TEST_payloads_opened=0,
        wall_seconds=time.perf_counter()-started)
    with (HERE/'SITE_CACHE_VERIFICATION.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
