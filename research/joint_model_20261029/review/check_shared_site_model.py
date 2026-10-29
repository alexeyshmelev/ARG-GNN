"""Independent CPU full-model checks of multiquery sharing and batch isolation.

No ancestry labels, optimizer, model fitting or GPU are used.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
import torch
from research.joint_model_20261029.model import joint_model_v1 as model

BASE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029')
PIN='ba87b3b7639db4ce07b424a26d1673a043ea1d9ddbe34914d7f6397177006aa4'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(model.__file__)==PIN
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
assert not torch.cuda.is_initialized()
torch.manual_seed(101)
start=time.perf_counter()
files=[BASE/'data/profile_observations_v1/site_000000.npz',BASE/'data/profile_observations_v1/site_000027.npz']
batches=[model.prepare_observations(model.load_observations(path)) for path in files]
original=model.JointAncestryModel(width=16,layers=3,classes=6)


def variable_batch(batch):
    return replace(batch,metadata=batch.metadata.detach().clone().requires_grad_(),
                   geometry=batch.geometry.detach().clone().requires_grad_())


def derivatives(network,batch,mode,coefficient):
    logits=network(batch,mode=mode)
    tensors=list(network.parameters())+[batch.metadata,batch.geometry]
    values=torch.autograd.grad((logits*coefficient).sum(),tensors,allow_unused=True)
    present=[value is not None for value in values]
    flat=torch.cat([(torch.zeros_like(tensor) if value is None else value).detach().reshape(-1)
                    for tensor,value in zip(tensors,values)]).double()
    assert torch.isfinite(logits).all() and torch.isfinite(flat).all()
    return logits.detach(),flat,present,values


def compare_logits(candidate,reference):
    difference=(candidate-reference).abs()
    budget=2e-5+2e-4*reference.abs()
    assert (difference<=budget).all()
    return float(difference.max())


def compare_gradient(candidate,reference):
    denominator=torch.linalg.vector_norm(reference)
    assert denominator>0
    error=float(torch.linalg.vector_norm(candidate-reference)/denominator)
    assert error<=1e-4
    return error


checks={}
first=batches[0]
assert first.target_count>=3
selected=torch.tensor([0,first.target_count//2,first.target_count-1])
for mode in ['no_cross','leaf_only','structure']:
    network=deepcopy(original)
    full=variable_batch(first)
    coefficient=torch.zeros((first.target_count,6))
    coefficient[selected]=torch.sin(torch.arange(18).reshape(3,6).float()+.3)
    output,gradient,presence,_=derivatives(network,full,mode,coefficient)
    summed=torch.zeros_like(gradient)
    singles=[]
    single_presence=None
    for target in selected.tolist():
        single=variable_batch(replace(first,focal_site=first.focal_site[target:target+1],
                                      focal_sample=first.focal_sample[target:target+1]))
        value,derivative,current_presence,_=derivatives(deepcopy(original),single,mode,coefficient[target:target+1])
        singles.append(value)
        summed+=derivative
        if single_presence is None:single_presence=current_presence
        else:assert single_presence==current_presence
    assert presence==single_presence
    shared_output_error=compare_logits(torch.cat(singles),output[selected])
    shared_gradient_error=compare_gradient(summed,gradient)

    # Only the first window contributes to the objective. This checks all common
    # parameter gradients and each window's input-gradient isolation.
    joined=variable_batch(model.collate_prepared(batches))
    joined_coeff=torch.cat([coefficient,torch.zeros((batches[1].target_count,6))])
    joined_output,_,joined_presence,joined_derivatives=derivatives(deepcopy(original),joined,mode,joined_coeff)
    assert joined_presence==presence
    tensors=list(original.parameters())+[full.metadata,full.geometry]
    parameter_count=len(list(original.parameters()))
    comparable=[]
    for index,(tensor,value) in enumerate(zip(tensors,joined_derivatives)):
        if index==parameter_count:
            assert torch.count_nonzero(value[first.window_count:])==0
            value=value[:first.window_count]
        if index==parameter_count+1:
            assert torch.count_nonzero(value[len(first.parent):])==0
            value=value[:len(first.parent)]
        comparable.append((torch.zeros_like(tensor) if value is None else value).detach().reshape(-1))
    joined_gradient=torch.cat(comparable).double()
    batch_output_error=compare_logits(joined_output[:first.target_count],output)
    batch_gradient_error=compare_gradient(joined_gradient,gradient)
    changed_second=replace(batches[1],metadata=batches[1].metadata+.5)
    with torch.no_grad():
        changed_output=deepcopy(original)(model.collate_prepared([first,changed_second]),mode=mode)
    perturbation_error=compare_logits(changed_output[:first.target_count],joined_output[:first.target_count])
    assert not torch.allclose(changed_output[first.target_count:],joined_output[first.target_count:],atol=1e-7,rtol=1e-7)
    checks[mode]={
        'shared_vs_separate_gather_max_logit_error':shared_output_error,
        'shared_vs_separate_gather_full_VJP_relative_l2':shared_gradient_error,
        'single_vs_disjoint_batch_max_logit_error':batch_output_error,
        'single_vs_disjoint_batch_full_VJP_relative_l2':batch_gradient_error,
        'other_window_metadata_perturbation_max_logit_error':perturbation_error,
        'other_window_input_VJPs_exactly_zero':True,
        'gradient_presence_identical':True,
    }
assert sha(model.__file__)==PIN and not torch.cuda.is_initialized()
receipt={
    'schema':'independent_shared_site_full_model_review_v1','verdict':'pass',
    'model_sha256':PIN,'audit_source_sha256':sha(__file__),
    'input_refs':[{'path':str(path),'sha256':sha(path)} for path in files],
    'configuration':{'width':16,'layers':3,'classes':6,'seed':101,'CPU_threads':2},
    'modes':checks,'scope':'Complete CPU model forward/input-and-parameter VJP using deterministic label-free probes; no optimizer.',
    'shared_site_queries':first.target_count,'joint_batch_queries':sum(batch.target_count for batch in batches),
    'elapsed_seconds':time.perf_counter()-start,
    'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'reviewer_GPU_runs':0,'CUDA_initialized':False,'labels_read':False,'TEST_read':False,
}
(BASE/'review/SHARED_SITE_MODEL_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
