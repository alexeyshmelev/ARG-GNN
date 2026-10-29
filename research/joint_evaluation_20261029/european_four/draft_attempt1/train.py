"""Fresh direct4 fixed-final50 fit; capacity qualification weights are discarded."""
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel
from research.joint_model_20261029.root.train import make_loader, evaluate
from research.joint_model_20261029.data.loader_shutdown import shutdown_loaders
from .dataset import FourPathDataset, collate_sites
from .common import BASE, RESULTS, MODEL_ID, freeze, ref, read, write, save_torch, check_ram, check_disk, device, require


def optimizer_for(model,p):
    return torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],betas=tuple(p['betas']),eps=p['epsilon'],weight_decay=p['weight_decay'])


def step(model,optimizer,batch,p):
    optimizer.zero_grad(set_to_none=True)
    logits=model(batch.observations,mode='structure')
    require(logits.shape==(len(batch.labels),4) and torch.isfinite(logits).all().item(),'Finite direct four-output logits')
    raw=F.cross_entropy(logits,batch.labels,reduction='sum');loss=raw*batch.loss_scale
    require(torch.isfinite(loss).item(),'Finite canonical-target objective')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    return float(raw.detach()),int((logits.argmax(1)==batch.labels).sum()),float(norm)


def qualify(p,frozen_ref,target,gpu):
    ds=FourPathDataset(cache_bytes=0)
    keys=dict(nodes=lambda r:r['nodes'],SNP_rows=lambda r:r['site_occurrences'],prepared_bytes=lambda r:r['prepared_ref']['bytes'])
    groups={name:sorted(range(len(ds)),key=lambda i:(-key(ds.windows[i]),i))[:p['batch_sites']] for name,key in keys.items()}
    model=JointAncestryModel(**p['model']).to(target);optimizer=optimizer_for(model,p)
    results=[]
    for name,indices in groups.items():
        batch=collate_sites([ds[i] for i in indices]).pin_memory().to(target,non_blocking=True)
        torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
        losses=[step(model,optimizer,batch,p)[0]/len(batch.labels) for _ in range(2)]
        torch.cuda.synchronize()
        results.append(dict(axis=name,indices=indices,targets=len(batch.labels),losses=losses,seconds=time.monotonic()-begin,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),family_rss_bytes=check_ram()))
        del batch
    result=dict(passed=True,freeze_ref=frozen_ref,physical_gpu=gpu,groups=results,complete_TRAIN_support_ranked=True,
                heldout_opened=False,benchmark_checkpoint_written=False)
    print(json.dumps(write(RESULTS/'gpu_qualification_v1.json',result)),flush=True)


def train(p,frozen_ref,target,gpu):
    qualified=read(ref(RESULTS/'gpu_qualification_v1.json'))
    require(qualified['passed'] and qualified['freeze_ref']==frozen_ref and qualified['physical_gpu']==gpu,'Same frozen source and GPU capacity qualification before fit')
    out=RESULTS/'train';out.mkdir(parents=True,exist_ok=False)
    ds=FourPathDataset(cache_bytes=p['worker_cache_bytes'])
    require(ds.site_count==p['training_sites'] and ds.target_count==40000 and ds.identity_sha256==p['training_identity_sha256'],'Frozen TRAIN support')
    loader=make_loader(ds,p['batch_sites'],p['workers'],p['batch_seed'],True)
    model=JointAncestryModel(**p['model']).to(target);optimizer=optimizer_for(model,p)
    initial=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
    started=dict(model_id=MODEL_ID,freeze_ref=frozen_ref,gpu=gpu,pid=os.getpid(),initialization_sha256=initial,model_parameters=sum(t.numel() for t in model.parameters()),
       training_sites=ds.site_count,training_targets=ds.target_count,planned_updates=p['epochs']*((len(ds)+p['batch_sites']-1)//p['batch_sites']),started_unix=time.time(),warmstart=False)
    write(out/'started.json',started)
    begin=time.monotonic();updates=0;presentations=0;records=[];order=hashlib.sha256();peak_ram=0
    try:
        for epoch in range(1,p['epochs']+1):
            model.train();seen=set();sites=set();ce=0.;hits=0;maximum_norm=0.;epoch_begin=time.monotonic()
            for cpu in loader:
                require(not seen.intersection(cpu.canonical_ids) and not sites.intersection(cpu.site_keys),'No repeated epoch target or site')
                seen.update(cpu.canonical_ids);sites.update(cpu.site_keys)
                order.update(json.dumps(cpu.canonical_ids,separators=(',',':')).encode())
                batch=cpu.to(target,non_blocking=True)
                raw,correct,norm=step(model,optimizer,batch,p)
                ce+=raw;hits+=correct;maximum_norm=max(maximum_norm,norm)
                updates+=1;presentations+=len(cpu.labels);peak_ram=max(peak_ram,check_ram());check_disk()
                if updates%25==0:print(json.dumps(dict(epoch=epoch,updates=updates,presentations=presentations,elapsed_seconds=time.monotonic()-begin)),flush=True)
                del batch
            require(seen==ds.expected_ids and sites==set(ds.site_keys),'Exact complete epoch support')
            record=dict(epoch=epoch,updates=updates,targets=len(seen),sites=len(sites),mean_CE=ce/len(seen),accuracy=hits/len(seen),maximum_preclip_norm=maximum_norm,seconds=time.monotonic()-epoch_begin)
            records.append(record)
            save_torch(out/'checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,freeze_ref=frozen_ref),replace=epoch>1)
            write(out/'epochs.json',records,replace=epoch>1)
        require(updates==started['planned_updates'] and presentations==2000000,'All50 fixed epochs complete')
        final_fit=evaluate(model,loader,target,'structure',ds.canonical_ids,4)
        checkpoint_ref=save_torch(out/'final_state.pt',model.state_dict())
        _,_,end_freeze=freeze();require(end_freeze==frozen_ref,'Source freeze stable throughout fit')
        write(out/'product.json',dict(complete=True,model_id=MODEL_ID,freeze_ref=frozen_ref,checkpoint_ref=checkpoint_ref,final_epoch=50,
            optimizer_updates=updates,target_presentations=presentations,target_order_sha256=order.hexdigest(),final_fit=final_fit,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),family_peak_observed_rss_bytes=peak_ram,
            elapsed_seconds=time.monotonic()-begin,TEST_opened=False,checkpoint_selection='fixed final epoch50',training_identity_sha256=ds.identity_sha256))
    except BaseException:
        write(out/'failure.json',dict(traceback=traceback.format_exc(),updates=updates,presentations=presentations));raise
    finally:
        write(out/'loader_shutdown.json',shutdown_loaders(loader))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('qualify','train'));parser.add_argument('--gpu',type=int,choices=(3,4,5),required=True)
    args=parser.parse_args();_,p,frozen_ref=freeze()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(p['model_seed']);np.random.seed(p['model_seed'])
    target=device(args.gpu)
    (qualify if args.stage=='qualify' else train)(p,frozen_ref,target,args.gpu)


if __name__=='__main__':main()
