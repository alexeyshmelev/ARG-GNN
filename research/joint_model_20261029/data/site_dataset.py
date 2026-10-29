"""Immutable site cache, canonical row identities, and unbiased target weighting.

An epoch reads only prepared CPU tensors and separate labels. It never opens
ARG/VCF sources or constructs graph routing. File snapshots are authenticated
once per worker and reauthenticated if their filesystem identity changes.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from research.joint_model_20261029.model import joint_model_v1 as model

PROTOCOL_SHA='82d42d8e597ac385bccadfba720350ff689b50be8e37c8ebcec819a88717f387'
PLAN_SHA='bfebc589dc0fb5a2f5a658f488d28dda9fd7b3baa2eb6fb03da76ed7f14c3a11'


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()


def safe_path(root,relative):
    path=(root/relative).resolve()
    require(path.is_relative_to(root.resolve()),'Cache reference escapes cache root')
    return path


def read_ref(root,ref,verified=None):
    path=safe_path(root,ref['path']);before=path.stat()
    snapshot=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
    require(before.st_size==ref['bytes'],'Cached file byte count mismatch: '+ref['path'])
    raw=path.read_bytes();after=path.stat()
    require(snapshot==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
            'Cache changed during read: '+ref['path'])
    key=(str(path),ref['sha256'])
    if verified is None or verified.get(key)!=snapshot:
        require(digest(raw)==ref['sha256'],'Cached file SHA256 mismatch: '+ref['path'])
        if verified is not None:verified[key]=snapshot
    return raw


@dataclass(frozen=True)
class SiteItem:
    observations:object
    labels:torch.Tensor
    canonical_ids:tuple
    site_key:tuple
    population_sites:int
    population_targets:int


@dataclass(frozen=True)
class SiteBatch:
    observations:object
    labels:torch.Tensor
    canonical_ids:tuple
    site_keys:tuple
    population_sites:int
    population_targets:int

    @property
    def site_count(self):return len(self.site_keys)
    @property
    def loss_scale(self):return self.population_sites/(self.site_count*self.population_targets)
    def to(self,device,non_blocking=False):
        return SiteBatch(self.observations.to(device,non_blocking=non_blocking),
            self.labels.to(device,non_blocking=non_blocking),self.canonical_ids,self.site_keys,
            self.population_sites,self.population_targets)
    def pin_memory(self):
        return SiteBatch(self.observations.pin_memory(),self.labels.pin_memory(),
            self.canonical_ids,self.site_keys,self.population_sites,self.population_targets)


class CachedSiteDataset(Dataset):
    """Registered fit/development/full_train support; deterministic row binding."""
    def __init__(self,manifest_path,split='fit',cache_bytes=2*2**30,mode='structure'):
        self.manifest_path=Path(manifest_path).resolve();self.cache_root=self.manifest_path.parent
        manifest=json.loads(self.manifest_path.read_bytes());self.manifest=manifest
        require(manifest['schema']=='joint_model_site_cache_v1' and manifest['complete'] is True,'Completed prepared site cache required')
        require(manifest['protocol_sha256']==PROTOCOL_SHA and manifest['plan_sha256']==PLAN_SHA,'Registered data policy and plan required')
        require(manifest['labels_separate'] is True and manifest['TEST_allowed'] is False,'Observation/target and TEST boundary')
        require(split in ('fit','development','full_train'),'Unknown dataset support')
        require(mode in ('no_cross','leaf_only','structure'),'Declared model arm required')
        self.mode=mode
        if mode=='no_cross':
            from research.joint_model_20261029.model import no_cross_view
            require(digest(Path(no_cross_view.__file__).read_bytes())=='0227f5cacbe4dc9a9fbd0bcca56a2f8582cd0d4a2c990b48a054e1a221f2b3e7','Qualified focal-only view source')
        require(digest(Path(model.__file__).read_bytes())==manifest['model_source_ref']['sha256'],'Runtime model source differs from prepared cache')
        read_ref(self.cache_root,manifest['model_source_ref'])
        for ref in manifest['source_refs'].values():read_ref(self.cache_root,ref)
        observations=json.loads(read_ref(self.cache_root,manifest['observations_manifest_ref']))
        require(observations['complete'] is True and observations['site_count']==1245 and observations['target_count']==50000,'Complete raw observation support')
        plan=json.loads(read_ref(self.cache_root,manifest['source_refs']['site_plan_v1.json']))
        require(len(manifest['windows'])==len(plan['sites'])==1245,'Exact full site support')
        seen=set()
        for row,registered,raw in zip(manifest['windows'],plan['sites'],observations['windows']):
            for field in ('window_index','parent_id','parent_ordinal','native_site_id','example_indices','sample_indices'):
                require(row[field]==registered[field]==raw[field],'Canonical plan binding mismatch: '+field)
            for field in ('observations_ref','labels_ref'):
                require(row[field]==raw[field],'Raw/ready cache reference mismatch: '+field)
            require(row['canonical_examples']==len(row['example_indices'])==len(row['sample_indices']),'Canonical row count')
            ids=[(row['parent_id'],int(i)) for i in row['example_indices']]
            require(len(set(ids))==len(ids) and not seen.intersection(ids),'Repeated canonical row')
            seen.update(ids)
        require(len(seen)==50000,'Full canonical target support')
        spec=manifest['support'][split];self.split=split
        self.windows=tuple(row for row in manifest['windows'] if row['parent_ordinal'] in spec['parent_ordinals'])
        self.site_keys=tuple((row['parent_id'],int(row['native_site_id'])) for row in self.windows)
        self.canonical_ids=tuple((row['parent_id'],int(i)) for row in self.windows for i in row['example_indices'])
        self.site_count=len(self.windows);self.target_count=len(self.canonical_ids)
        require(self.site_count==spec['sites'] and self.target_count==spec['targets'],'Registered split support counts')
        require(len(set(self.site_keys))==self.site_count,'Repeated site context')
        require(isinstance(cache_bytes,int) and cache_bytes>=0,'Nonnegative worker cache byte cap')
        self.cache_bytes=cache_bytes
        self._verified={};self._items=OrderedDict();self._item_bytes=0
        self.cache_hits=0;self.cache_misses=0

    def __getstate__(self):
        state=self.__dict__.copy()
        state.update(_verified={},_items=OrderedDict(),_item_bytes=0,cache_hits=0,cache_misses=0)
        return state

    def _fingerprint(self,row):
        result=[]
        for name in ('prepared_ref','labels_ref'):
            ref=row[name];path=safe_path(self.cache_root,ref['path']);value=path.stat()
            require(value.st_size==ref['bytes'],'Cached file byte count mismatch: '+ref['path'])
            result.append((value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns))
        return tuple(result)

    def __len__(self):return self.site_count

    def __getitem__(self,index):
        row=self.windows[index];key=row['window_index'];fingerprint=self._fingerprint(row)
        if key in self._items:
            prior,item,cost=self._items.pop(key)
            if prior==fingerprint:
                self._items[key]=(prior,item,cost);self.cache_hits+=1
                return item
            self._item_bytes-=cost
        self.cache_misses+=1
        state=torch.load(io.BytesIO(read_ref(self.cache_root,row['prepared_ref'],self._verified)),
                         map_location='cpu',weights_only=True)
        batch=model.prepared_from_state(state)
        require(batch.window_count==1 and batch.target_count==row['canonical_examples'],'Prepared site/target cardinality')
        require(torch.equal(batch.focal_sample,torch.tensor(row['sample_indices'],dtype=torch.long)),
                'Prepared target sample order differs from canonical registry')
        if self.mode=='no_cross':
            from research.joint_model_20261029.model.no_cross_view import focal_only_no_cross
            batch=focal_only_no_cross(batch)
        labels=np.load(io.BytesIO(read_ref(self.cache_root,row['labels_ref'],self._verified)),allow_pickle=False)
        require(labels.dtype==np.int64 and labels.shape==(batch.target_count,) and np.isin(labels,np.arange(6)).all(),
                'Canonical native six-class label vector')
        identities=tuple((row['parent_id'],int(i)) for i in row['example_indices'])
        item=SiteItem(batch,torch.from_numpy(labels.copy()),identities,
            (row['parent_id'],int(row['native_site_id'])),self.site_count,self.target_count)
        require(fingerprint==self._fingerprint(row),'Cached site changed during loading')
        cost=row['prepared_ref']['bytes']+row['labels_ref']['bytes']
        if cost<=self.cache_bytes:
            while self._items and self._item_bytes+cost>self.cache_bytes:
                _,(_,_,removed)=self._items.popitem(last=False);self._item_bytes-=removed
            self._items[key]=(fingerprint,item,cost);self._item_bytes+=cost
        return item


def collate_sites(items):
    require(len(items)>0,'Nonempty site batch')
    m=items[0].population_sites;n=items[0].population_targets
    require(all(item.population_sites==m and item.population_targets==n for item in items),'One population per site batch')
    identities=tuple(identity for item in items for identity in item.canonical_ids)
    keys=tuple(item.site_key for item in items)
    require(len(set(keys))==len(keys) and len(set(identities))==len(identities),'No duplicated sites or canonical rows in minibatch')
    batch=model.collate_prepared([item.observations for item in items])
    labels=torch.cat([item.labels for item in items])
    require(batch.target_count==len(labels)==len(identities),'Collated output/target alignment')
    return SiteBatch(batch,labels,identities,keys,m,n)
