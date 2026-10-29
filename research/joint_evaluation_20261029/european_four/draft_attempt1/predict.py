"""Predict all canonical non-BAA TEST rows from the sealed final50 checkpoint."""
import argparse
from io import BytesIO
import json
import time
import numpy as np
import torch
from arggnn.benchmarking.paper_ancestralpaths_four.workflow import selected_records
from arggnn.benchmarking.paper_ancestralpaths import data as original_data
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel,collate_prepared
from research.joint_evaluation_20261029.prediction.observations import TestObservationSource,make_site_plan
from research.joint_evaluation_20261029.prediction.workers import prepared_stream
from research.joint_evaluation_20261029.prediction.freeze import check_inputs
from .dataset import PLAN_REF
from .common import BASE,ROOT,RESULTS,MODEL_ID,freeze,ref,read,write,write_bytes,check_ram,check_disk,device,require


def filtered_sites(plan,eligible):
    sites=[]
    for original in plan['sites']:
        positions=[i for i,e in enumerate(original['example_indices']) if int(e) in eligible]
        if positions:
            site=dict(original,window_index=len(sites))
            for key in ('example_indices','sample_indices'):site[key]=[original[key][i] for i in positions]
            sites.append(site)
    identities=[i for site in sites for i in site['example_indices']]
    require(len(identities)==len(set(identities))==len(eligible) and set(identities)==set(eligible),'Exact eligible prediction support')
    return sites


def canonical_arrays(logits,identities,parent,indices):
    logits=np.asarray(logits,dtype=np.float32);identities=np.asarray(identities,dtype=np.int64)
    require(logits.shape==(len(indices),4) and np.isfinite(logits).all(),'Finite direct4 predictions')
    order=np.argsort(identities);require(np.array_equal(identities[order],indices),'Canonical prediction row identities')
    logits=logits[order];wide=logits.astype(np.float64);wide-=wide.max(1,keepdims=True)
    probabilities=np.exp(wide);probabilities/=probabilities.sum(1,keepdims=True)
    return dict(logits=logits,probabilities=probabilities.astype(np.float32),example_indices=indices,
                example_ids=np.asarray([parent+':'+str(int(i)) for i in indices]))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',required=True);parser.add_argument('--gpu',type=int,choices=(3,4,5),required=True)
    args=parser.parse_args();_,p,frozen_ref=freeze()
    fit_ref=ref(RESULTS/'train/product.json');fit=read(fit_ref)
    require(fit['complete'] and fit['freeze_ref']==frozen_ref and fit['model_id']==MODEL_ID and fit['final_epoch']==50 and fit['target_presentations']==2000000 and fit['optimizer_updates']==1700,'Final50 complete direct4 fit before TEST observations')
    read(ref(RESULTS/'train/loader_shutdown.json'))
    require(ref(ROOT/fit['checkpoint_ref']['path'])==fit['checkpoint_ref'],'Sealed final checkpoint')
    metadata=read(p['prediction_metadata_ref']);require(args.parent in metadata['parents'],'Registered TEST parent')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);target=device(args.gpu)
    model=JointAncestryModel(**p['model']);model.load_state_dict(torch.load(ROOT/fit['checkpoint_ref']['path'],map_location='cpu',weights_only=True),strict=True);model.eval().to(target)
    check_inputs(metadata,args.parent)
    item=metadata['parents'][args.parent]
    source=TestObservationSource(args.parent,item['binding'])
    records,indices=selected_records(source.registry,original_data.checked_plan(PLAN_REF));require(len(records)==8000,'All8000 non-BAA TEST rows')
    sites=filtered_sites(make_site_plan(source,item['parent_ordinal']),set(map(int,indices)))
    out=RESULTS/'predictions'/args.parent;out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();values=[];identities=[];batch=[];batch_ids=[]
    write(out/'started.json',dict(freeze_ref=frozen_ref,fit_ref=fit_ref,parent_id=args.parent,sites=len(sites),rows=8000,gpu=args.gpu,query_truth_read=False))
    for index,prepared,metadata_row in prepared_stream(args.parent,item['binding'],sites):
        require(index<len(sites) and np.array_equal(prepared.focal_sample.numpy(),sites[index]['sample_indices']),'Exact prediction sample routing')
        batch.append(prepared);batch_ids.extend(sites[index]['example_indices'])
        if len(batch)==p['batch_sites'] or index+1==len(sites):
            assembled=collate_prepared(batch).pin_memory().to(target,non_blocking=True)
            with torch.inference_mode():predicted=model(assembled,mode='structure').detach().cpu().numpy()
            values.append(predicted);identities.extend(batch_ids);batch.clear();batch_ids.clear();del assembled
            check_ram();check_disk()
            print(json.dumps(dict(parent=args.parent,sites=index+1,total_sites=len(sites),targets=len(identities))),flush=True)
    arrays=canonical_arrays(np.concatenate(values),identities,args.parent,indices)
    check_inputs(metadata,args.parent);_,_,after=freeze();require(after==frozen_ref and ref(RESULTS/'train/product.json')==fit_ref,'Frozen source and fitted model remain unchanged')
    stream=BytesIO();np.savez(stream,**arrays);arrays_ref=write_bytes(out/'predictions.npz',stream.getvalue())
    product=dict(schema='ap_four_predictions_v1',condition=MODEL_ID,parent_id=args.parent,split='test',rows=8000,class_names=['Ana','CHG','WHG','EHG'],source_plan_ref=PLAN_REF,
        registry_ref=item['binding']['refs']['sampled_registry_ref'],fit_ref=fit_ref,checkpoint_ref=fit['checkpoint_ref'],freeze_ref=frozen_ref,predictions_ref=arrays_ref,
        query_truth_read=False,probability_merging=False,context_includes_unknown_BAA_queries=True,elapsed_seconds=time.monotonic()-started,complete=True)
    print(json.dumps(write(out/'product.json',product)),flush=True)


if __name__=='__main__':main()
