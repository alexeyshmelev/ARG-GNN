"""Deferred one ordinaryFP32 query-axis/cache/gradient/AdamW comparison.
No CLI execution or numerical import at module import. Root owns actual invocation.
Not an accuracy experiment, fit benchmark or repeat of the prior H4/N2 fixture.
"""
from pathlib import Path
import hashlib,importlib.util
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1"
STEP_REF={'path': 'research/reviews/C3_efficient_query_GPU_cache_successor_source_20261016_v1/efficient_step_v1.py', 'sha256': '0df05b1957affc222ca1c11d9f34cc3277f5a3098834fb36fedee64bb242d2e8'}

def _load():
    p=ROOT/STEP_REF["path"]
    if hashlib.sha256(p.read_bytes()).hexdigest()!=STEP_REF["sha256"]:raise ValueError("Exact new step source")
    spec=importlib.util.spec_from_file_location("_C3_efficient_single_FP32_qualification",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def _blocks(torch):
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    from arggnn.american_c3_gnn_dataset_isolated_v2 import AmericanC3BlockV1
    f=lambda v:torch.tensor(v,dtype=torch.float32)
    l=lambda v:torch.tensor(v,dtype=torch.long)
    sizes=(2,1,2);out=[];tree_start=0;focal_start=0
    parent0=[8,8,9,9,10,10,11,11,12,12,13,13,14,14,-1]
    for index,count in enumerate(sizes):
        parents=[];times=[];sample=[];support=[];geom=[]
        for j in range(count):
            off=15*j
            parents.extend([p+off if p>=0 else -1 for p in parent0])
            times.extend([0.]*8+[2.1,2.2,2.3,2.4,4.5,4.6,7.+.1*(tree_start+j)])
            sample.append([off+k for k in range(8)])
            support.extend([[.2,.3,.4,2.,0.]]*14+[[0.,0.,0.,0.,1.]])
            geom.extend([[.1+.01*(tree_start+j),.2,1.,.3]]*14+[[0.,0.,0.,0.]])
        up=[list(range(0,8)),list(range(8,12)),[12,13],[14]]
        children=[[],list(range(8)),list(range(8,12)),[12,13]]
        ranks=[[],[0,0,1,1,2,2,3,3],[0,0,1,1],[0,0]]
        upall=[];childall=[];rankall=[]
        for level in range(4):
            upall.append(l([15*j+x for j in range(count) for x in up[level]]))
            childall.append(l([15*j+x for j in range(count) for x in children[level]]))
            rankall.append(l([len(up[level])*j+x for j in range(count) for x in ranks[level]]))
        down=tuple(l([15*j+x for j in range(count) for x in level]) for level in ([12,13],[8,9,10,11],list(range(8))))
        sites=count*3
        genotype=torch.tensor([[(s+k+index)%2 for k in range(8)] for s in range(sites)],dtype=torch.int8)
        donors=f([[[.55+.01*(s%3)]*6,[.45-.01*(s%3)]*6] for s in range(sites)]).log()
        holds=[j for j in range(count) for _ in range(3)]
        bank=MutationRoutedWholeTreeInputsV1(parent=l(parents),node_time=f(times),sample_index=l(sample),
            upward_levels=tuple(upall),upward_children=tuple(childall),upward_parent_rank=tuple(rankall),
            downward_levels=down,edge_geometry=f(geom),branch_support_true=f(support),branch_support_sham=f(support),
            genotype=genotype,log_donor_allele_frequency=donors,donor_call_mask=torch.ones((sites,6),dtype=torch.bool),
            panel_allele_frequency=f([[.5,.5]]*sites),site_entropy=f([.69]*sites),site_hold=l(holds),
            spans=f([1.+.1*(tree_start+j) for j in range(count)]),readout_hold=l(list(range(count))),
            query_column=l([6,7]),reference_class=l([0,0,1,1,2,2,-1,-1]),sample_age=f([0.]*8))
        focal_hold=[j for j in range(count) for _ in range(4)]
        focal_col=[6,7,6,7]*count;N=len(focal_hold)
        out.append(AmericanC3BlockV1(bank,parent_id="qualification_only",block_index=index,
            start_tree=tree_start,stop_tree=tree_start+count,sequence_length_bp=6.,complete_parent=False,
            genomic_geometry=f([[.11+.01*(tree_start+j),.2,.3] for j in range(count)]),
            focal_rows=l(list(range(focal_start,focal_start+N))),
            focal_hold=l(focal_hold),focal_sample_column=l(focal_col)))
        tree_start+=count;focal_start+=N
    return out

def _original_reference(model,blocks,labels,torch,device,check):
    """Original fullH block checkpoint equations, only small fixture dimensions."""
    from torch.utils.checkpoint import checkpoint
    from torch.nn import functional as F
    dummy=torch.zeros((),device=device,dtype=torch.float32,requires_grad=True);n=labels.numel()
    def scan(direction):
        state=torch.zeros((1,8,32),device=device,dtype=torch.float32)
        rows=[];histories=[];locals_=[]
        order=range(len(blocks)) if direction=="forward" else range(len(blocks)-1,-1,-1)
        for index in order:
            def evaluate(_dummy,state,physical=index,scan_direction=direction):
                check("qualification_original_block")
                block=blocks[physical].clone().to(device)
                context=model.encode_bank(block.to_model_inputs())
                history,next_state=model.scan_context(context,block.genomic_geometry,direction=scan_direction,hidden=state)
                local=context[block.focal_hold,block.focal_sample_column] if scan_direction=="forward" else context.new_empty((0,32))
                return next_state,local,history[block.focal_hold,block.focal_sample_column]
            state,local,history=checkpoint(evaluate,dummy,state,use_reentrant=False,preserve_rng_state=True)
            rows.append(blocks[index].focal_rows);histories.append(history)
            if direction=="forward":locals_.append(local)
        order=torch.argsort(torch.cat(rows)).to(device)
        return (torch.cat(locals_).index_select(0,order) if locals_ else None,torch.cat(histories).index_select(0,order))
    local,forward=scan("forward");_,reverse=scan("reverse");losses=[];debug=[]
    for start in range(0,n,4):
        def classify(_dummy,x,f,r,y):
            return F.cross_entropy(model.classify_context(x,f,r),y,reduction="sum").to("cpu")
        x,f,r=local[start:start+4],forward[start:start+4],reverse[start:start+4]
        losses.append(checkpoint(classify,dummy,x,f,r,labels[start:start+4].to(device),
                                 use_reentrant=False,preserve_rng_state=True))
        with torch.no_grad():debug.append(model.classify_context(x,f,r).cpu())
    loss=torch.stack(losses).sum()/n;loss.backward()
    return loss.detach(),torch.cat(debug)

def run_once_v1(*,resource_check,device="cpu",atol=5e-5,rtol=5e-4):
    if not callable(resource_check):raise ValueError("Root actual prior guard/resource callback required")
    resource_check("before_new_FP32_qualification_numerical_import")
    import torch
    from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
    step=_load();actual=torch.device(device)
    if actual.type not in ("cpu","cuda") or actual.type=="cuda" and actual.index is None:
        raise ValueError("One explicit actual root-admitted device")
    prior=torch.random.get_rng_state().clone()
    cuda_prior=torch.cuda.get_rng_state(actual).cpu().clone() if actual.type=="cuda" else None
    try:
        torch.random.default_generator.manual_seed(7000102)
        if actual.type=="cuda":
            with torch.cuda.device(actual):torch.cuda.manual_seed(7000102)
        models=[AmericanC3WholeTreeGNNV1(hidden_dim=32,expected_haplotypes=8,donor_cohorts=2,
            site_chunk_size=256,checkpoint_sites=True).to(actual).train() for _ in range(2)]
        original,candidate=models;candidate.load_state_dict(original.state_dict(),strict=True)
        step._base().require_supported_model_v1(original,actual)
        blocks=_blocks(torch);labels=torch.tensor([(j*2+1)%3 for j in range(20)],dtype=torch.long)
        plan=dict(trees=5,blocks=[dict(start_tree=b.start_tree,stop_tree=b.stop_tree,
             nodes=len(b.parent),sites=len(b.genotype)) for b in blocks])
        class Provider:
            def __init__(self):self.plan=plan;self.calls=0
            def __len__(self):return len(blocks)
            def get(self,index):self.calls+=1;return blocks[index]
        provider=Provider()
        forecast=step.forecast_v1(plan,queries=2,hidden=32,focals=20,haplotypes=8)
        caps=dict(max_parent_trees=5,max_parent_blocks=3,max_active_device_feature_bytes=8*2**20,
            max_block_feature_bytes=4*2**20,max_query_context_device_bytes=8*2**20,
            max_recurrent_metadata_device_bytes=8*2**20,max_progress_record_bytes=8192,
            max_successor_cuda_reserved_bytes=24*2**30)
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001,
            betas=(.9,.999),eps=1e-8,foreach=False,fused=False) for m in models]
        loss,logits=_original_reference(original,blocks,labels,torch,actual,resource_check)
        result=step._backward_v1(candidate,provider,labels,device=actual,resource_check=resource_check,
            caps=caps,focal_chunk_size=4,expected_queries=2,capture_logits=True)
        torch.testing.assert_close(result["loss"],loss,atol=atol,rtol=rtol)
        torch.testing.assert_close(result["logits"],logits,atol=atol,rtol=rtol)
        errors={}
        for (name,p),(other,q) in zip(original.named_parameters(),candidate.named_parameters()):
            if name!=other or p.grad is None or q.grad is None:raise AssertionError("Complete parametergradient mapping")
            torch.testing.assert_close(q.grad,p.grad,atol=atol,rtol=rtol)
            errors[name]=float((q.grad-p.grad).abs().max())
        norms=[torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True) for m in models]
        torch.testing.assert_close(norms[1],norms[0],atol=atol,rtol=rtol)
        for opt in optimizers:opt.step()
        updates={};moment_error=0.
        for (name,p),(other,q) in zip(original.named_parameters(),candidate.named_parameters()):
            torch.testing.assert_close(q,p,atol=atol,rtol=rtol);updates[name]=float((q-p).abs().max())
            a,b=optimizers[0].state[p],optimizers[1].state[q]
            for key in ("exp_avg","exp_avg_sq"):
                torch.testing.assert_close(b[key],a[key],atol=atol,rtol=rtol)
                moment_error=max(moment_error,float((b[key]-a[key]).abs().max()))
            if int(a["step"].item())!=1 or int(b["step"].item())!=1:raise AssertionError("Exactly one actualAdamWupdate")
        if provider.calls!=3 or result["counts"]["encoder_value_forwards"]!=3 or result["counts"]["encoder_VJP_forwards"]!=3:
            raise AssertionError("B rawbanktransfers and2B encoderentries")
        return dict(schema="C3_efficient_query_cache_single_FP32_comparison_v1",passed=True,
            device=str(actual),precision="fp32",seed=7000102,trees=5,physical_blocks=3,
            haplotypes=8,reference_columns=6,query_columns=2,focals=20,optimizer_steps=1,
            fullgraph_encoder_preserved=True,allfocal_labels_preserved=True,atol=atol,rtol=rtol,
            logits_max_abs_error=float((result["logits"]-logits).abs().max()),
            loss_abs_error=float((result["loss"]-loss).abs()),
            parameter_gradient_max_abs_error=max(errors.values()),parameter_update_max_abs_error=max(updates.values()),
            optimizer_moment_max_abs_error=moment_error,source_counts=result["counts"],
            full_training_fit_or_accuracy_claimed=False,old_qualified_fixture_rerun=False)
    finally:
        torch.random.set_rng_state(prior)
        if cuda_prior is not None:torch.cuda.set_rng_state(cuda_prior,actual)
