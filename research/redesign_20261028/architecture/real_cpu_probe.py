"""Untrained bounded real-observation forward/VJP, no labels or optimizer."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import torch
from research.redesign_20261028.architecture.window_model import load_observations,prepare_observations,WindowTreeLoom

def main():
    root=Path('/workspace-SR003.nfs2/ARG-GNN/research/redesign_20261028')
    parser=argparse.ArgumentParser()
    parser.add_argument('--packet',default='train000_first_window.npz')
    parser.add_argument('--output',default='REAL_CPU_PROBE.json')
    args=parser.parse_args()
    assert '/' not in args.packet and '/' not in args.output
    torch.set_num_threads(2);torch.manual_seed(20261028)
    started=time.perf_counter();obs=load_observations(root/'data'/args.packet)
    batch=prepare_observations(obs);prepared=time.perf_counter()
    model=WindowTreeLoom(width=32)
    start=time.perf_counter();logits=model(batch);forward=time.perf_counter()-start
    # Numerical connectivity probe is not an ancestry loss or a learning step.
    start=time.perf_counter();logits.square().mean().backward();backward=time.perf_counter()-start
    gradients={n:p.grad for n,p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in gradients.values())
    assert torch.isfinite(logits).all() and not torch.cuda.is_initialized()
    with torch.no_grad():
        start=time.perf_counter();control=model(batch,mode='no_cross');control_seconds=time.perf_counter()-start
    receipt=dict(schema='bounded_real_observation_CPU_probe_v1',data='data/'+args.packet,
        model_source_sha256=hashlib.sha256((root/'architecture/window_model.py').read_bytes()).hexdigest(),
        context_kind=batch.context_kind.tolist(),
        classes=6,nodes=len(batch.parent),trees=len(batch.leaves),samples=batch.leaves.shape[1],
        sites=len(batch.site_genotype),focal_rows=len(logits),core_rows=int(batch.core_focal_mask.sum()),
        hidden_width=32,parameters=sum(p.numel() for p in model.parameters()),parameter_tensors=len(gradients),
        finite_gradient_tensors=sum(bool(torch.isfinite(g).all()) for g in gradients.values()),
        nonzero_gradient_tensors=sum(bool(g.abs().max()>0) for g in gradients.values()),
        preparation_seconds=prepared-started,forward_seconds=forward,backward_seconds=backward,
        no_cross_forward_seconds=control_seconds,structure_control_max_logit_difference=float((logits-control).abs().max()),
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_threads=2,cuda_initialized=False,labels_read=False,optimizer_updates=0,
        trained=False,ancestry_accuracy=None,representative_training_batch=False)
    (root/'architecture'/args.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
