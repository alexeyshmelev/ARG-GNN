"""Qualify the focal-only execution view against the full no-cross predictor."""
from pathlib import Path
import hashlib,json,os
import setproctitle
setproctitle.setproctitle('AI Auto Research Pipeline Test')
import numpy as np
import torch
from torch.nn import functional as F
from research.joint_model_20261029.model import joint_model_v1 as m
from research.joint_model_20261029.model import no_cross_view as view
BASE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
    assert sha(m.__file__)=='ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4'
    assert sha(view.__file__)=='0227f5cacbe4dc9a9fbd0bcca56a2f8582cd0d4a2c990b48a054e1a221f2b3e7'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest_path=BASE/'data/cache_v1/manifest_observations.json'
    manifest=json.loads(manifest_path.read_text())
    batches=[];labels=[]
    for index in [0,207]:
        row=manifest['windows'][index]
        p=manifest_path.parent/row['observations_ref']['path']
        assert sha(p)==row['observations_ref']['sha256']
        with np.load(p,allow_pickle=False) as f:batches.append(m.prepare_observations({k:f[k] for k in f.files}))
        p=manifest_path.parent/row['labels_ref']['path']
        assert sha(p)==row['labels_ref']['sha256']
        labels.append(torch.from_numpy(np.load(p,allow_pickle=False)))
    full=m.collate_prepared(batches).to('cuda')
    focal=m.collate_prepared([view.focal_only_no_cross(x) for x in batches]).to('cuda')
    truth=torch.cat(labels).to('cuda')
    torch.manual_seed(101)
    model=m.JointAncestryModel(width=64,layers=3,classes=6).to('cuda')
    outputs=[];gradients=[];presence=[]
    for batch in [full,focal]:
        model.zero_grad(set_to_none=True)
        value=model(batch,mode='no_cross')
        F.cross_entropy(value,truth).backward()
        outputs.append(value.detach().cpu())
        presence.append(np.array([x.grad is not None for x in model.parameters()]))
        gradients.append(torch.cat([(x.grad if x.grad is not None else torch.zeros_like(x)).detach().cpu().flatten() for x in model.parameters()]))
    error=gradients[0]-gradients[1]
    relative=float(error.norm()/gradients[0].norm().clamp_min(1e-12))
    passed=(torch.allclose(outputs[0],outputs[1],atol=2e-5,rtol=2e-4) and relative<=1e-4 and np.array_equal(*presence) and all(torch.isfinite(x).all().item() for x in outputs+gradients))
    out=BASE/'root/no_cross_gpu_view_v1'
    out.mkdir(exist_ok=False)
    np.savez_compressed(out/'numerics.npz',full_logits=outputs[0].numpy(),focal_logits=outputs[1].numpy(),full_gradients=gradients[0].numpy(),focal_gradients=gradients[1].numpy(),full_presence=presence[0],focal_presence=presence[1])
    report=dict(passed=bool(passed),model_sha256=sha(m.__file__),view_sha256=sha(view.__file__),harness_sha256=sha(__file__),
                observation_manifest_sha256=sha(manifest_path),window_indices=[0,207],targets=len(truth),
                full_nodes=len(full.parent),focal_nodes=len(focal.parent),
                logit_max_abs=float((outputs[0]-outputs[1]).abs().max()),gradient_relative_l2=relative,
                argmax_changes=int((outputs[0].argmax(1)!=outputs[1].argmax(1)).sum()),
                thresholds=dict(logit_atol=2e-5,logit_rtol=2e-4,gradient_relative_l2=1e-4),
                optimizer_steps=0,accuracy_claimed=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert passed
if __name__=='__main__':main()
