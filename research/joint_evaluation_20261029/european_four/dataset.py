"""Observed-only focal selection over the unchanged complete TRAIN cache."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import selected_records
from arggnn.benchmarking.paper_ancestralpaths_four.data import four_path_targets
from arggnn.benchmarking.paper_ancestralpaths import data as original_data
from arggnn.benchmarking import european_native_models as io
from research.joint_model_20261029.data.site_dataset import CachedSiteDataset, SiteItem, collate_sites, require

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
MANIFEST=ROOT/'research/joint_model_20261029/data/cache_v1/manifest.json'
MANIFEST_SHA='a9a2839b450ee7bf12f90bc269a257bb759b79fdd6e8a93fa8e7eb4c0737e253'
PLAN_REF=dict(path='configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json',sha256='2d39761d0cc030af72ba1e313be63dfce3fa7e5a4e1f4cecbdf33a5109fc7f5b')


def focal_view(batch, positions):
    """Only readout routing changes; every observed context tensor is shared."""
    take=torch.as_tensor(positions,dtype=torch.long)
    require(take.ndim==1 and len(take)>0 and len(take.unique())==len(take),'Nonempty unique focal routing')
    require(int(take.min())>=0 and int(take.max())<batch.target_count,'Focal routing range')
    return replace(batch,focal_site=batch.focal_site[take],focal_sample=batch.focal_sample[take])


class FourPathDataset(Dataset):
    def __init__(self,cache_bytes=12*2**30):
        require(hashlib.sha256(MANIFEST.read_bytes()).hexdigest()==MANIFEST_SHA,'Frozen TRAIN cache manifest')
        manifest=json.loads(MANIFEST.read_bytes())
        plan=original_data.checked_plan(PLAN_REF)
        self.records={};self.cohort_counts=Counter();self.expected_ids=set()
        # Eligibility is finalized using only observed registry cohorts before any label file is opened.
        for parent in plan['train_parent_ids']:
            registry=io._read(manifest['source_bindings'][parent]['refs']['sampled_registry_ref'],observed=True)
            require(registry['parent_id']==parent and registry['split']=='train','Exact TRAIN registry')
            records,indices=selected_records(registry,plan)
            self.records[parent]={int(r['example_index']):r for r in records}
            self.expected_ids.update((parent,int(i)) for i in indices)
            self.cohort_counts.update(r['cohort'] for r in records)
        require(len(self.expected_ids)==40000 and dict(self.cohort_counts)=={c:10000 for c in ('present_day','Bronze_Age','Neo','Yam')},'Exact CNN four-path TRAIN support')
        self.base=CachedSiteDataset(MANIFEST,split='full_train',cache_bytes=cache_bytes,mode='structure')
        self.selection=[];self.windows=[]
        for index,row in enumerate(self.base.windows):
            positions=[i for i,e in enumerate(row['example_indices']) if int(e) in self.records[row['parent_id']]]
            if positions:
                self.selection.append((index,tuple(positions)));self.windows.append(row)
        self.site_count=len(self.selection);self.target_count=40000
        self.site_keys=tuple(self.base.site_keys[i] for i,_ in self.selection)
        self.canonical_ids=tuple((self.base.windows[i]['parent_id'],int(self.base.windows[i]['example_indices'][j])) for i,positions in self.selection for j in positions)
        require(len(self.canonical_ids)==len(set(self.canonical_ids))==40000 and set(self.canonical_ids)==self.expected_ids,'All original selected IDs exactly once')
        self.identity_sha256=hashlib.sha256(json.dumps(sorted(self.canonical_ids),separators=(',',':')).encode()).hexdigest()

    def __len__(self):return self.site_count

    def __getitem__(self,index):
        original,positions=self.selection[index]
        item=self.base[original]
        ids=tuple(item.canonical_ids[j] for j in positions)
        cohorts=[self.records[parent][number]['cohort'] for parent,number in ids]
        labels=four_path_targets(item.labels[list(positions)].numpy(),cohorts)
        batch=focal_view(item.observations,positions)
        require(batch.target_count==len(ids)==len(labels),'Focal label alignment')
        for offset,(parent,number) in enumerate(ids):
            require(int(batch.focal_sample[offset])==self.records[parent][number]['sample_index'],'Original canonical sample routing')
        return SiteItem(batch,torch.from_numpy(labels),ids,item.site_key,self.site_count,self.target_count)
