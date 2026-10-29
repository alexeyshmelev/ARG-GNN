"""CPU-only exact-support, cached-routing, optimizer and prediction-format integration."""
from collections import Counter
from dataclasses import fields
from io import BytesIO
import json
import time
import numpy as np
import torch
from torch.utils.data import Subset
from arggnn.benchmarking import european_native_models as io
from arggnn.benchmarking.paper_ancestralpaths_four.data import four_path_targets
from research.joint_model_20261029.data.site_dataset import read_ref,require
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel
from .loader import make_loader
from research.joint_model_20261029.data.loader_shutdown import shutdown_loaders
from .dataset import FourPathDataset,collate_sites,PLAN_REF
from .common import BASE,ref,read,write,check_ram
from .train import step,optimizer_for
from .predict import filtered_sites,canonical_arrays


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    p=read(ref(BASE/'protocol_v1.json'));begin=time.monotonic()
    ds=FourPathDataset(cache_bytes=0)
    expected={};counts=Counter()
    for parent,records in ds.records.items():
        binding=ds.base.manifest['source_bindings'][parent]
        product=io._read(binding['refs']['training_targets_ref'])
        require(product['registry_ref']==binding['refs']['sampled_registry_ref'] and product['split']=='train' and product['no_label_dependent_selection'],'Same original TRAIN target binding')
        labels=io._arrays(product['arrays_ref'],cap=2*2**20)['native_labels_0based']
        indices=sorted(records);selected=four_path_targets(labels[indices],[records[i]['cohort'] for i in indices])
        expected.update(((parent,i),int(y)) for i,y in zip(indices,selected));counts.update(map(int,selected))
    matched=0
    for original,positions in ds.selection:
        row=ds.base.windows[original]
        labels=np.load(BytesIO(read_ref(ds.base.cache_root,row['labels_ref'])),allow_pickle=False)
        for j in positions:
            identity=(row['parent_id'],int(row['example_indices'][j]))
            require(int(labels[j])==expected[identity],'Every reused cached label exactly equals original CNN4 target')
            matched+=1
    require(matched==40000 and ds.site_count==531,'Complete direct4 support')
    torch.manual_seed(p['model_seed']);model=JointAncestryModel(**p['model']);optimizer=optimizer_for(model,p)
    routing_checks=[];items=[]
    for index in (0,177,354):
        original,positions=ds.selection[index];full=ds.base[original];item=ds[index];items.append(item)
        for field in fields(full.observations):
            if field.name not in ('focal_site','focal_sample'):
                # Both loads authenticate identical cache bytes. Compare sparse and dense representations exactly.
                left=getattr(full.observations,field.name);right=getattr(item.observations,field.name)
                def equal(a,b):
                    if isinstance(a,torch.Tensor):
                        if a.layout==torch.sparse_csr:return torch.equal(a.crow_indices(),b.crow_indices()) and torch.equal(a.col_indices(),b.col_indices()) and torch.equal(a.values(),b.values())
                        return torch.equal(a,b)
                    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                    return a==b
                require(equal(left,right),'Unchanged observed context: '+field.name)
        model.eval()
        with torch.no_grad():
            full_logits=model(full.observations,mode='structure')[list(positions)]
            filtered_logits=model(item.observations,mode='structure')
        require(torch.equal(full_logits,filtered_logits),'Direct4 focal routing logits bitwise equal')
        require((item.observations.reference[item.observations.query]==-1).all(),'All query ancestry remains unknown')
        routing_checks.append(dict(dataset_index=index,targets=len(item.labels),native_site=item.site_key[1]))
    loader=make_loader(Subset(ds,[0,177,354]),2,2,p['batch_seed'],False)
    losses=[];worker_results=[]
    try:
        for epoch in range(2):
            observed=[]
            for cpu in loader:
                observed.extend(cpu.canonical_ids)
                if epoch==0:model.train();losses.append(step(model,optimizer,cpu,p)[0])
            require(set(observed)==set(i for item in items for i in item.canonical_ids),'Spawned persistent workers retain exact focal support')
            worker_results.append(observed)
        require(worker_results[0]==worker_results[1],'Persistent second epoch order stable')
    finally:shutdown=shutdown_loaders(loader)
    model.eval();cpu=collate_sites(items)
    with torch.no_grad():before=model(cpu.observations,mode='structure')
    stream=BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=JointAncestryModel(**p['model']);restored.load_state_dict(torch.load(stream,weights_only=True));restored.eval()
    with torch.no_grad():after=restored(cpu.observations,mode='structure')
    require(torch.equal(before,after),'CPU learned checkpoint logits bitwise equal')
    parent=items[0].canonical_ids[0][0]
    chosen=np.array(sorted(i for pr,i in items[0].canonical_ids),dtype=np.int64)
    original_ids=[i for pr,i in items[0].canonical_ids]
    with torch.no_grad():logits=model(items[0].observations,mode='structure').numpy()
    arrays=canonical_arrays(logits,original_ids,parent,chosen)
    stream=BytesIO();np.savez(stream,**arrays);stream.seek(0)
    with np.load(stream,allow_pickle=False) as loaded:
        require(all(np.array_equal(loaded[k],v) for k,v in arrays.items()),'Prediction NPZ serialization parity')
    plan={'sites':[dict(window_index=0,example_indices=[0,1,2],sample_indices=[8,9,10],native_site_id=7,context={'unchanged':True}),dict(window_index=1,example_indices=[3],sample_indices=[11],native_site_id=9,context={'unchanged':True})]}
    selected=filtered_sites(plan,{1,2});require(len(selected)==1 and selected[0]['sample_indices']==[9,10] and selected[0]['context']==plan['sites'][0]['context'],'Observation-only prediction site filtering preserves context')
    rejected=[]
    for labels,cohorts in [([4],['Neo']),([0],['BAA'])]:
        try:four_path_targets(np.asarray(labels),cohorts)
        except ValueError:rejected.append(cohorts[0])
        else:raise ValueError('Four-path boundary accepted invalid labels')
    require(not torch.cuda.is_initialized(),'CPU-only qualification')
    result=dict(passed=True,source_refs=p['source_refs'],protocol_ref=ref(BASE/'protocol_v1.json'),TRAIN_sites=531,TRAIN_targets=40000,TRAIN_labels_all_exact=True,
        cohort_counts=dict(ds.cohort_counts),class_counts=dict(counts),identity_sha256=ds.identity_sha256,routing_checks=routing_checks,
        nonrouting_observation_fields_bitwise_equal=True,query_ancestry_unknown=True,checkpoint_logits_bitwise_equal=True,
        persistent_worker_epochs=2,loader_shutdown=shutdown,optimizer_steps=len(losses),finite_losses=losses,prediction_NPZ_roundtrip=True,
        prediction_site_selection_passed=True,rejected_cases=rejected,TEST_payloads_opened=False,CUDA_initialized=False,
        family_rss_bytes=check_ram(),elapsed_seconds=time.monotonic()-begin)
    print(json.dumps(write(BASE/'smoke_result_v1.json',result)),flush=True)


if __name__=='__main__':main()
