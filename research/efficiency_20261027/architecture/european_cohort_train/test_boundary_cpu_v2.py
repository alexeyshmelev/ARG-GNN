"""Bounded synthetic952-sample full-model loss/gradient/Adam comparison; no data."""
import copy
import hashlib
import json
from pathlib import Path
import resource
import signal
import time
from types import SimpleNamespace
import numpy as np
import torch
from research.efficiency_20261027.architecture.european_cohort_candidate.model import build_model, need
from research.efficiency_20261027.architecture.european_cohort_candidate.execution import CohortStagedExecution
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_execution as original
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_gru import install_chunked_gru
from .boundary_v1 import EuropeanBoundaryExecution


def fixture(holds):
    h, per_tree = 952, 957
    leaves, middle, root = np.arange(h), np.arange(h,h+4), np.array([h+4])
    empty = np.empty(0,dtype=np.int64)
    parents, samples, layouts, assignments = [], [], [], []
    for hold in range(holds):
        offset = hold*per_tree
        membership = ((leaves+hold*137)//238)%4
        p = np.concatenate((h+membership,np.full(4,h+4),np.array([-1])))
        parents.append(torch.from_numpy(np.where(p>=0,p+offset,p)))
        samples.append(torch.arange(h)+offset)
        assignments.append(torch.from_numpy(membership)+hold*4)
        layouts.append((hold,hold+1,offset,offset+per_tree,p,(leaves,middle,root),
            (empty,leaves,middle),(empty,membership,np.zeros(4,dtype=np.int64)),(middle,leaves),hold,hold+1))
    sites = 7
    genotype = torch.randint(0,2,(sites,h),dtype=torch.int8)
    genotype[:,::17] = -1
    donor, mask = torch.rand(sites,2,8).clamp_min(.1).log(), torch.rand(sites,8)>.2
    panel = torch.rand(sites,2).clamp_min(.1); panel /= panel.sum(-1,keepdim=True)
    entropy = torch.rand(sites)
    for value in (genotype,donor,mask,panel,entropy): value[1]=value[0]
    labels = torch.full((h,),-1); labels[686:] = torch.arange(266)%4
    bank = SimpleNamespace(parent=torch.cat(parents),node_time=torch.tensor(([0.]*h+[300.]*4+[1000.])*holds),
        sample_index=torch.stack(samples),edge_geometry=torch.randn(holds*per_tree,4)*.1,
        branch_support_true=torch.rand(holds*per_tree,5),reference_class=labels,
        sample_age=torch.arange(h).float()%300,spans=torch.linspace(1.,30000.,holds),
        site_hold=torch.tensor([0,0,0,holds-1,holds-1,holds-1,holds-1]),genotype=genotype,
        log_donor_allele_frequency=donor,donor_call_mask=mask,panel_allele_frequency=panel,site_entropy=entropy)
    episodes = SimpleNamespace(hold_span=bank.spans,member_hold=torch.arange(holds),
        member_episode=torch.zeros(holds,dtype=torch.long),episode_size=torch.tensor([holds]),
        member_node=torch.arange(holds)*per_tree+h+4,flags=SimpleNamespace(node_flags=torch.randint(0,2,(holds*per_tree,2)).float()))
    cut = SimpleNamespace(group_offsets=tuple(i*4 for i in range(holds+1)),
        representatives=torch.cat([torch.arange(h,h+4)+i*per_tree for i in range(holds)]),
        leaf_groups=torch.stack(assignments),group_counts=torch.full((holds*4,),238.),native_holds=holds,haplotypes=h)
    return bank,SimpleNamespace(bank=bank,episodes=episodes,cut=cut),layouts


def main():
    signal.alarm(120); started=time.monotonic()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(70029)
    reports=[]
    for holds in (1,3,5):
        bank,bundle,layouts=fixture(holds)
        model=build_model()
        if holds>1:
            with torch.no_grad():model.leaf_projection[0].weight[:,38:].uniform_(-.02,.02)
        cohort=torch.zeros(952,8);cohort[torch.arange(800),torch.arange(800)%8]=1.
        focal_hold=torch.arange(36)%holds
        focal_column=(torch.arange(36)*19)%686
        focal_hold[-1]=focal_hold[0];focal_column[-1]=focal_column[0]
        labels=torch.arange(36)%6
        def run(cls):
            executor=object.__new__(cls)
            executor.model=install_chunked_gru(copy.deepcopy(model),chunk_holds=2)
            executor.bank,executor.bundle,executor.device=bank,bundle,torch.device('cpu')
            executor.cohort_device=cohort
            executor.tick=lambda label:None
            executor.profile=original.NullProfileV1()
            executor.layout,executor.history_window,executor.hap_tile=layouts,1,32
            optimizer=torch.optim.AdamW(executor.model.parameters(),lr=.001,weight_decay=.0001,foreach=False,fused=False)
            rng=torch.random.get_rng_state().clone()
            context=executor.context_forward()
            geometry=original.geometry_from_spans(bank.spans)
            loss,gradient=original.decode_parent_backward(executor.model,context,geometry,focal_hold,focal_column,labels,
                                                         tile_haplotypes=32)
            with torch.no_grad():
                logits=original._tile_logits(executor.model,context,geometry,focal_hold,focal_column)
            context.backward(gradient)
            parameters=list(executor.model.parameters())
            need(len(parameters)==49 and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'All49 full-model gradients')
            gradients={n:p.grad.clone() for n,p in executor.model.named_parameters()}
            need(bool((executor.model.leaf_projection[0].weight.grad[:,38:].abs().sum(0)>0).all()),'All8 cohort columns connected')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            optimizer.step()
            need(torch.equal(rng,torch.random.get_rng_state()),'No RNG consumption during numerical call')
            return dict(context=context.detach(),loss=loss.detach(),logits=logits,gradient=gradient,gradients=gradients,
                parameters={n:p.detach().clone() for n,p in executor.model.named_parameters()},adam=copy.deepcopy(optimizer.state_dict()),
                norm=norm.detach(),receipt=getattr(executor,'explicit_receipt',None))
        baseline,candidate=run(CohortStagedExecution),run(EuropeanBoundaryExecution)
        errors={}
        def compare(key,left,right):
            torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5 if key in ('context','logits','loss') else 1e-4,msg=key)
            errors[key]=float((left-right).abs().max())
        for key in ('context','loss','logits','gradient','norm'):compare(key,baseline[key],candidate[key])
        for category in ('gradients','parameters'):
            for key,value in baseline[category].items():compare(category+'/'+key,value,candidate[category][key])
        adam_count=0
        for key,value in baseline['adam']['state'].items():
            for name,tensor in value.items():
                compare('adam/'+str(key)+'/'+name,tensor,candidate['adam']['state'][key][name]);adam_count+=1
        need(torch.equal(baseline['logits'].argmax(-1),candidate['logits'].argmax(-1)),'No sampled argmax changes')
        need(adam_count==147 and candidate['receipt']['connected_trunk_parameters']==37,'Complete Adam and trunk gradients')
        reports.append(dict(holds=holds,sites=7,samples=952,rows=36,repeated_rows=True,parameters=49,adam_tensors=adam_count,
            maximum_absolute_error=max(errors.values()),errors=errors,receipt=candidate['receipt']))
    need(not torch.cuda.is_initialized(),'No CUDA')
    report=dict(schema='European_cohort8_boundary_full_model_CPU_v2',passed=True,tests=reports,
        scientific_arrays_read=False,GPU_initialized=False,full_parent_qualification=False,
        missing_calls_max_ties_empty_holds_changing_cuts=True,all8_cohort_gradients_nonzero=True,
        tolerances=dict(context_logits_and_loss=dict(atol=1e-6,rtol=1e-5),norm_gradients_parameters_Adam=dict(atol=1e-6,rtol=1e-4)),wall_seconds=time.monotonic()-started,
        maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        source_sha256=hashlib.sha256(Path(__file__).with_name('boundary_v1.py').read_bytes()).hexdigest())
    with Path(__file__).with_name('CPU_boundary_v2.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({key:value for key,value in report.items() if key!='tests'}))

if __name__=='__main__':main()
