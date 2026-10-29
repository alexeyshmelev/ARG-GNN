"""Paired CPU/CUDA complete-predictor qualification on registered observations."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
import numpy as np
import torch
from torch.nn import functional as F
import pynvml
from research.joint_model_20261029.model import joint_model_v1 as model_module

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'
GPU='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    output=BASE/'root/gpu_qualification_v1'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol=json.loads((BASE/'root/gpu_qualification_protocol.json').read_text())
    assert sha(model_module.__file__)==protocol['model_sha256']
    manifest_path=BASE/'data/cache_v1/manifest_observations.json'
    assert sha(manifest_path)==protocol['observations_manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    rows=[manifest['windows'][i] for i in protocol['window_indices']]
    prepared=[]; labels=[];source_refs=[]
    for row in rows:
        for key in ['observations_ref','labels_ref']:
            ref=row[key];path=manifest_path.parent/ref['path']
            assert sha(path)==ref['sha256']
            source_refs.append(ref)
        with np.load(manifest_path.parent/row['observations_ref']['path'],allow_pickle=False) as f:
            prepared.append(model_module.prepare_observations({k:f[k] for k in f.files}))
        labels.append(torch.from_numpy(np.load(manifest_path.parent/row['labels_ref']['path'],allow_pickle=False)))
    cpu=model_module.collate_prepared(prepared)
    truth=torch.cat(labels)
    pynvml.nvmlInit()
    handle=pynvml.nvmlDeviceGetHandleByUUID(GPU)
    foreign_before=[x.pid for x in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if x.pid!=os.getpid()]
    assert not foreign_before,'Preserve other processes and defer isolated qualification'
    gpu=cpu.to('cuda:0'); gpu_truth=truth.to('cuda:0')
    assert uuid.UUID(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-'))==uuid.UUID(GPU.removeprefix('GPU-'))
    name=pynvml.nvmlSystemGetProcessName(os.getpid())
    if isinstance(name,bytes):name=name.decode()
    assert name=='AI Auto Research Pipeline Test'
    comparisons=[]
    for mode in ['no_cross','leaf_only','structure']:
        torch.manual_seed(101)
        first=model_module.JointAncestryModel(width=64,layers=3,classes=6)
        second=model_module.JointAncestryModel(width=64,layers=3,classes=6).to('cuda:0')
        second.load_state_dict(first.state_dict())
        optimizers=[torch.optim.AdamW(m.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4) for m in [first,second]]
        values=[];gradients=[];presence=[];times=[];norms=[];post=[]
        for model,batch,y,opt in zip([first,second],[cpu,gpu],[truth,gpu_truth],optimizers):
            if y.is_cuda:torch.cuda.synchronize()
            start=time.perf_counter()
            value=model(batch,mode=mode)
            loss=F.cross_entropy(value,y)
            loss.backward()
            if y.is_cuda:torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            values.append(value.detach().cpu())
            presence.append([p.grad is not None for p in model.parameters()])
            gradients.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().cpu().reshape(-1) for p in model.parameters()]))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            norms.append(float(norm))
            opt.step()
            with torch.no_grad():post.append(model(batch,mode=mode).cpu())
        delta=gradients[0]-gradients[1]
        relative=float(delta.norm()/gradients[0].norm().clamp_min(1e-12))
        finite=all(torch.isfinite(x).all().item() for x in values+gradients+post)
        passes=dict(finite=finite,gradient_presence=presence[0]==presence[1],
                    logits=torch.allclose(values[0],values[1],atol=protocol['logit_atol'],rtol=protocol['logit_rtol']),
                    gradient_relative_l2=relative<=protocol['gradient_relative_l2'],
                    post_update_logits=torch.allclose(post[0],post[1],atol=protocol['post_update_logit_atol'],rtol=protocol['post_update_logit_rtol']))
        result=dict(mode=mode,passed=all(passes.values()),checks=passes,
                    logit_max_abs=float((values[0]-values[1]).abs().max()),
                    gradient_relative_l2=relative,gradient_max_abs=float(delta.abs().max()),
                    post_update_logit_max_abs=float((post[0]-post[1]).abs().max()),
                    cpu_gpu_preclip_norm=norms,cpu_gpu_forward_backward_seconds=times,
                    before_argmax_changes=int((values[0].argmax(1)!=values[1].argmax(1)).sum()),
                    after_argmax_changes=int((post[0].argmax(1)!=post[1].argmax(1)).sum()))
        np.savez_compressed(output/(mode+'.npz'),cpu_logits=values[0].numpy(),gpu_logits=values[1].numpy(),
                             cpu_gradients=gradients[0].numpy(),gpu_gradients=gradients[1].numpy(),
                             cpu_post=post[0].numpy(),gpu_post=post[1].numpy())
        comparisons.append(result)
        print(json.dumps(result),flush=True)
    foreign_after=[x.pid for x in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if x.pid!=os.getpid()]
    report=dict(model_sha256=protocol['model_sha256'],harness_sha256=sha(__file__),protocol_sha256=sha(BASE/'root/gpu_qualification_protocol.json'),
                observations_manifest_sha256=protocol['observations_manifest_sha256'],sources=source_refs,
                comparisons=comparisons,targets=len(truth),gpu_uuid=GPU,process_title=name,pid=os.getpid(),
                foreign_before=foreign_before,foreign_after=foreign_after,
                allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                TEST_opened=False,weights_retained_for_training=False,optimizer_updates_per_mode_per_device=1,
                passed=all(x['passed'] for x in comparisons) and not foreign_after)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    pynvml.nvmlShutdown()
    assert report['passed'],'Qualification failed, preserve receipt and investigate'

if __name__=='__main__':main()
