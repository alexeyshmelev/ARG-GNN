"""Future admitted numerical qualification on explicit synthetic observations."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_frozen_v1 as frozen
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import american_full30mb_prepare_v1 as preparation
from .bounded_execution_v1 import BoundedExecutionV1,need

def synthetic_parent_v1():
    import tskit
    h=1600;spans=[8000.,17000.,2.,11000.,23000.,31000.,29909998.]
    tables=tskit.TableCollection(sequence_length=30000000.)
    for _ in range(h):tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=0.)
    starts=np.concatenate(([0.],np.cumsum(spans)))
    for hold,(left,right)in enumerate(zip(starts[:-1],starts[1:])):
        if hold==2:continue
        leaves=list(range(h))
        if hold%2:leaves[0],leaves[4]=leaves[4],leaves[0]
        if hold>=3:leaves[7],leaves[15]=leaves[15],leaves[7]
        level=leaves;depth=1
        while len(level)>1:
            next_level=[]
            for at in range(0,len(level),2):
                if at+1==len(level):next_level.append(level[at]);continue
                parent=tables.nodes.add_row(time=float(depth*70))
                tables.edges.add_row(left,right,parent,level[at]);tables.edges.add_row(left,right,parent,level[at+1])
                next_level.append(parent)
            level=next_level;depth+=1
    positions=[]
    for at,(left,right)in enumerate(zip(starts[:-1],starts[1:])):
        if at!=2:positions.extend(np.linspace(left+.25,right-.25,100))
    positions=np.asarray(sorted(positions),dtype=np.float64)
    for pos in positions:tables.sites.add_row(pos,ancestral_state="A")
    tables.sort();ts=tables.tree_sequence()
    # Every site deliberately has equal features, making cross-chunk amax ties exact.
    genotype=np.zeros((len(positions),h),dtype=np.int8)
    nodes=np.arange(h,dtype=np.int64)
    rng=np.random.default_rng(79001)
    sample=rng.integers(600,1600,size=10000,dtype=np.int64)
    focal_hold=np.arange(10000,dtype=np.int64)%len(spans)
    focal_positions=(starts[focal_hold]+starts[focal_hold+1])*.5
    focals=dict(record_ids=np.arange(10000,dtype=np.int64),sample_nodes=nodes[sample],
        sample_indices=sample,positions_bp0=focal_positions,parent_id=np.asarray("american_c3_train_000"),
        class_order=np.asarray(["EUR","EAS","AFR"]))
    state=dict(manifest=dict(schema="american_c3_observed_v1",genealogy_kind=frozen.GENEALOGY,
        parent_id="american_c3_train_000",query_ancestry_truth_in_inputs=False,class_order=["EUR","EAS","AFR"]),
        ts=ts,labels=np.asarray([2]*200+[0]*200+[1]*200+[-1]*1000,dtype=np.int64),
        query_columns=np.arange(600,1600),nodes=nodes,focals=focals,genotype=genotype,positions=positions,
        spans=np.asarray(spans),ages=np.zeros(h),allele_count=np.ones(len(positions),dtype=np.int8),focal_tree=focal_hold)
    fixture=lambda name:dict(path="SYNTHETIC_RAM_ONLY/"+name,sha256="0"*64)
    bundle,metadata=preparation.prepare_from_loaded_observation_v1(state,split="train",observation_ref=fixture("observations"),
        focal_registry_ref=fixture("focals"),native_preparation_ref=fixture("native"),
        max_nodes=100000,max_holds=100,max_episode_payload_bytes=8*2**20)
    return bundle,metadata

def run_parity_v1(*,resource_check):
    """Compare exact frozen eager CUDA path against bounded path, including AdamW."""
    resource_check("before_synthetic_nonzero_residual_parity")
    bundle,metadata=synthetic_parent_v1()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(79002);a=frozen.AmericanTreeLoomC3FullV1()
        for adapter in(a.clade_coupling,a.episode_coupling,a.memory):
            torch.nn.init.normal_(adapter.O.weight,mean=0.,std=.02)
    b=copy.deepcopy(a);a=a.to("cuda:0");b=b.to("cuda:0")
    opt=lambda model:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
        maximize=False,amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)
    oa,ob=opt(a),opt(b);a.train();b.train()
    gpu=bundle.to("cuda:0")
    with torch.autocast(device_type="cuda",enabled=False):
        left=a(gpu)
        right=BoundedExecutionV1(b,bundle,metadata,holds_per_block=2,haplotypes_per_gru=8,resource_check=resource_check).forward()
        labels=(torch.arange(10000,device="cuda:0")%3).long()
        la,lb=F.cross_entropy(left,labels,reduction="mean"),F.cross_entropy(right,labels,reduction="mean")
    torch.testing.assert_close(left,right,rtol=2e-4,atol=2e-6)
    torch.testing.assert_close(la,lb,rtol=2e-4,atol=2e-6)
    la.backward();lb.backward()
    names=dict(a.named_parameters());other=dict(b.named_parameters());need(len(names)==49,"All49 original tensors")
    max_gradient_error=0.
    for name,p in names.items():
        q=other[name];need(p.grad is not None and q.grad is not None,"Connected gradient "+name)
        torch.testing.assert_close(p.grad,q.grad,rtol=4e-4,atol=4e-6)
        max_gradient_error=max(max_gradient_error,float((p.grad-q.grad).abs().max()))
    memory_nonzero=any(bool(names[name].grad.abs().sum()>0)for name in names if name.startswith("memory."))
    need(memory_nonzero,"Nonzero memory-path gradient, not zero-O-only parity")
    na=torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True)
    nb=torch.nn.utils.clip_grad_norm_(b.parameters(),1.,error_if_nonfinite=True)
    torch.testing.assert_close(na,nb,rtol=4e-4,atol=4e-6)
    for name,p in names.items():torch.testing.assert_close(p.grad,other[name].grad,rtol=4e-4,atol=4e-6)
    oa.step();ob.step()
    for name,p in names.items():
        q=other[name];torch.testing.assert_close(p,q,rtol=4e-4,atol=4e-6)
        for key in("step","exp_avg","exp_avg_sq"):
            torch.testing.assert_close(oa.state[p][key],ob.state[q][key],rtol=4e-4,atol=4e-6)
    torch.cuda.synchronize(0)
    return dict(schema="frozen_TreeLoom_bounded_synthetic_parity_v1",passed=True,synthetic_only=True,
        real_TRAIN_or_VAL_or_TEST_truth_read=False,all49_raw_and_clipped_gradients_and_AdamW_parameters_moments_compared=True,
        nonzero_memory_gradient=True,nonzero_residual_O_initialization=True,all1600sample_inputs=True,
        original_global256_site_boundaries=True,tied_amax_gradients=True,isolated_zero_site_hold=True,
        irregular_spans=True,hold_block_size=2,GRU_haplotype_tile=8,
        max_logit_absolute_error=float((left-right).abs().max()),max_gradient_absolute_error=max_gradient_error,
        loss_absolute_error=float((la-lb).abs()),actual_GPU_fit_of_real30Mb_parent_claimed=False)
