"""Real TRAIN boundary, identity, lossless storage and train/predict integration."""
import argparse
from dataclasses import fields
from io import BytesIO
import os
import resource
import subprocess
import time
import numpy as np
import torch
from torch.nn import functional as F
from research.joint_model_20261029.model.joint_model_v1 import (
    Batch, JointAncestryModel, prepare_observations, prepared_to_state,
    prepared_from_state, collate_prepared)
from research.joint_evaluation_20261029.american.state_codec_v2 import encode, decode
from .common import BASE, CAMPAIGN, CLASSES, MODEL_REF, RECIPE_REF, ref, read, verify, require, write
from .observations import ObservationSource


def exact(a,b):
    if isinstance(a,torch.Tensor):
        require(a.dtype==b.dtype and a.shape==b.shape and a.layout==b.layout,'Tensor schema roundtrip')
        if a.layout==torch.sparse_csr:
            for x,y in zip((a.crow_indices(),a.col_indices(),a.values()),(b.crow_indices(),b.col_indices(),b.values())):
                exact(x,y)
        else:
            require(torch.equal(a,b),'Exact dense tensor roundtrip')
    elif isinstance(a,(tuple,list)):
        require(len(a)==len(b),'List roundtrip')
        for x,y in zip(a,b):exact(x,y)
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),'Dictionary roundtrip')
        for k in a:exact(a[k],b[k])
    else:
        require(a==b,'Scalar roundtrip')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    require(args.gpu in (3,4,5),'Authorized GPU')
    listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    mapping={int(v[0]):(v[1].strip(),int(v[2])) for v in (line.split(',') for line in listing.splitlines())}
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==mapping[args.gpu][0] and torch.cuda.device_count()==1,'Exact physical GPU binding')
    require(mapping[args.gpu][1]>=32768,'32GiB free admission for isolated smoke')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    verify(MODEL_REF)
    start=time.monotonic()
    parent='published_recon_train_000'
    source=ObservationSource(parent,ref(CAMPAIGN/parent/'treeloom_observation/completed.json'))
    sites=source.plan()
    require(len(sites)==1616 and sum(len(s['example_indices']) for s in sites)==100000,'All original target support')
    require(sorted(i for s in sites for i in s['example_indices'])==list(range(100000)),'No duplicate or omitted target')
    selected=[sites[0],sites[len(sites)//2],sites[-1]]
    batches=[]
    records=[]
    for site in selected:
        begin=time.monotonic()
        packet=source.extract(site)
        require((packet['reference_class'][packet['query_mask']]==-1).all(),'No query ancestry features')
        prepared=prepare_observations(packet)
        preparation=time.monotonic()-begin
        begin=time.monotonic()
        payload=encode(prepared_to_state(prepared),level=9)
        encoding=time.monotonic()-begin
        begin=time.monotonic()
        restored=prepared_from_state(decode(payload))
        decoding=time.monotonic()-begin
        exact(prepared_to_state(prepared),prepared_to_state(restored))
        require(np.array_equal(restored.focal_sample.numpy(),site['sample_indices']),'Exact canonical focal sample')
        batches.append(restored)
        records.append(dict(window=site['window_index'],targets=len(site['example_indices']),
           preparation_seconds=preparation,encoding_seconds=encoding,decoding_seconds=decoding,
           compressed_bytes=len(payload),statistics=prepared.statistics))
        print(records[-1],flush=True)
    wrong=dict(selected[0],sample_indices=selected[0]['sample_indices'][::-1])
    rejected=False
    try:source.extract(wrong)
    except ValueError:rejected=True
    require(rejected,'Mismatched query routing rejected')
    label_ref=ref(CAMPAIGN/parent/'labels/completed.json')
    product=read(label_ref)
    require(product['split']=='train' and product['parent_id']==parent and
            product['support_manifest_ref']==source.seal['support_ref'] and product['class_order']==list(CLASSES),
            'Exact original TRAIN-only labels')
    labels=np.fromfile(verify(product['labels_ref']),dtype=np.uint8)
    require(labels.shape==(100000,) and (labels<4).all(),'All original TRAIN labels')
    rows=np.concatenate([s['example_indices'] for s in selected])
    torch.manual_seed(20261029)
    model=JointAncestryModel(width=64,layers=3,classes=4).cuda().eval()
    batch=collate_prepared(batches).to('cuda:0')
    with torch.inference_mode():
        together=model(batch,mode='structure')
        separate=torch.cat([model(b.to('cuda:0'),mode='structure') for b in batches])
    maximum=float((together-separate).abs().max())
    require(torch.allclose(together,separate,atol=2e-5,rtol=2e-5),'Independent windows preserve model predictions when batched')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    y=torch.as_tensor(labels[rows].astype(np.int64),device='cuda:0')
    losses=[]
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction=model(batch,mode='structure')
        loss=F.cross_entropy(prediction,y)
        require(torch.isfinite(loss),'Finite TRAIN loss')
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'All model parameters have finite connected gradients')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    stream=BytesIO()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},stream)
    stream.seek(0)
    restored=JointAncestryModel(width=64,layers=3,classes=4).cuda().eval()
    restored.load_state_dict(torch.load(stream,weights_only=True,map_location='cpu'))
    with torch.inference_mode():
        before=model(batch,mode='structure')
        after=restored(batch,mode='structure')
    require(torch.equal(before,after),'Checkpoint restored predictions exactly')
    source.check_stats()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak<64*2**30,'Within own RAM allocation')
    output=dict(schema='argmix_joint_train_smoke_v1',passed=True,benchmark_fit=False,
        model_ref=MODEL_REF,recipe_ref=RECIPE_REF,source_refs=[ref(BASE/n) for n in ['common.py','observations.py','smoke.py']],
        parent=parent,original_targets=100000,unique_windows=len(sites),windows=records,
        batched_logits_max_difference=maximum,adamw_updates=2,training_losses=losses,
        checkpoint_roundtrip_exact=True,all_tensor_cache_roundtrip_exact=True,
        max_cpu_rss_bytes=peak,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        physical_gpu=args.gpu,elapsed_seconds=time.monotonic()-start,
        private_heldout_inputs_or_truth_opened=False)
    print(write(BASE/'smoke_v1.json',output),flush=True)


if __name__=='__main__':main()
