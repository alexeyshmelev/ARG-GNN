"""Effective saved600 provider/Dataset methods with a narrow local encoder route."""
import os,math
from collections import OrderedDict
import numpy as np
import torch
from arggnn import american_c3_gnn_dataset_isolated_v2 as _dataset
from arggnn import american_c3_gnn_stream_train_isolated_v2 as _stream
from .observations_legacy_exact_v1 import native_tree,encode_block
load_observation=_dataset.load_observation
AmericanC3BlockV1=_dataset.AmericanC3BlockV1
TRAIN=_stream.TRAIN
require=_stream.require
validate_plan_v1=_stream.validate_plan_v1
class AmericanC3WholeTreeDatasetV1(_dataset.AmericanC3WholeTreeDatasetV1):
    def __getitem__(self,index):
        if self._pid!=os.getpid():
            self._state=load_observation(self.plan['observation_ref'],self.plan['focal_registry_ref'],self.plan['config']);self._pid=os.getpid()
        state=self._state
        index=int(index) if self.direction=='forward' else len(self)-1-int(index)
        spec=self.plan['blocks'][index];a,b=spec['start_tree'],spec['stop_tree'];breaks=state['breaks']
        lo,hi=np.searchsorted(state['positions'],[breaks[a],breaks[b]])
        g=np.asarray(state['genotype'][lo:hi])
        require(self.plan['config']['allow_missing'] or (g>=0).all(),'Registered no-genotype-error reconstruction requires complete calls')
        tree=state['ts'].at_index(a);trees=[];intervals=[]
        for i in range(a,b):
            require(tree.index==i,'Every original tree retained in chronological order')
            trees.append(native_tree(tree,state['nodes']));intervals.append(tuple(tree.interval))
            if i+1<b:require(tree.next(),'Truncated original tree traversal')
        bank,receipt=encode_block(trees,intervals,g,state['allele_count'][lo:hi],state['positions'][lo:hi],
            state['labels'],state['ages'],num_classes=len(self.plan['class_order']),age_split=self.plan['config']['donor_age_split'],query_columns=state['query_columns'])
        require(receipt['nodes']==spec['nodes'] and receipt['sites']==spec['sites'],'Plan/source whole-tree count drift')
        spans=state['spans'];left=np.zeros(b-a);right=np.zeros(b-a)
        if a>0:left[0]=.5*(spans[a-1]+spans[a])
        if b<len(spans):right[-1]=.5*(spans[b-1]+spans[b])
        if b-a>1:
            left[1:]=.5*(spans[a+1:b]+spans[a:b-1]);right[:-1]=left[1:]
        geometry=np.stack((np.log1p(spans[a:b]),np.log1p(left),np.log1p(right)),-1)/math.log(2000001.)
        rows=np.flatnonzero((state['focal_tree']>=a)&(state['focal_tree']<b))
        return AmericanC3BlockV1(bank,parent_id=self.plan['parent_id'],plan_ref=self.plan_ref,block_index=index,
            start_tree=a,stop_tree=b,sequence_length_bp=state['ts'].sequence_length,complete_parent=a==0 and b==self.plan['trees'],
            genomic_geometry=torch.as_tensor(geometry,dtype=torch.float32),
            focal_rows=torch.as_tensor(rows,dtype=torch.long),
            focal_hold=torch.as_tensor(state['focal_tree'][rows]-a,dtype=torch.long),
            focal_sample_column=torch.as_tensor(state['focals']['sample_indices'][rows],dtype=torch.long),
            observations_receipt=receipt,worker_pid=os.getpid())

class SealedObservationProviderV1(_stream.SealedObservationProviderV1):
    def __init__(self,plan_ref,*,parent_id,worker_tmp_directory,cache_bytes):
        require(parent_id in TRAIN and type(cache_bytes) is int and cache_bytes>=0,
            "TRAIN-only bounded CPU observation provider")
        self.dataset=AmericanC3WholeTreeDatasetV1(plan_ref,direction="forward",
            worker_tmp_directory=worker_tmp_directory)
        self.plan=dict(self.dataset.plan,plan_ref=dict(self.dataset.plan_ref))
        validate_plan_v1(self.plan,parent_id=parent_id,length_bp=2000000)
        self.cache_bytes=cache_bytes;self.cache=OrderedDict();self.used=0;self.seal_record=None

