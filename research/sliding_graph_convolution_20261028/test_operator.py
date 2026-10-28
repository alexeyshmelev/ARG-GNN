"""Meaningful toy checks of correspondence, isolation, halos and derivatives."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
import platform
import resource
import numpy as np
import torch
import torch_geometric
from branch_joint_conv import Tree, branch_overlap, build_graph, disjoint_batch, BranchJointConv

ROOT=Path(__file__).resolve().parent
checks=[]

def check(name, condition, **detail):
    if not bool(condition): raise AssertionError(name+': '+str(detail))
    checks.append({'name':name,'status':'pass',**detail})

def small(changed=False):
    parent=np.array([4,5,4,5,6,6,-1]) if changed else np.array([4,4,5,5,6,6,-1])
    return Tree(parent,np.array([0.,0.,0.,0.,2.,3.,8.]),np.arange(4),tuple(range(4))).validate()

def dense_overlap(a,b):
    da=[set() for _ in a.parent];db=[set() for _ in b.parent]
    for j,path in enumerate(a.paths()):
        for u in path: da[u].add(j)
    for j,path in enumerate(b.paths()):
        for v in path: db[v].add(j)
    out=np.zeros((len(a.parent),len(b.parent)))
    for u in range(len(a.parent)):
        for v in range(len(b.parent)):
            if a.parent[u]>=0 and b.parent[v]>=0:
                width=max(0.,min(a.age[a.parent[u]],b.age[b.parent[v]])-max(a.age[u],b.age[v]))
                out[u,v]=len(da[u]&db[v])*width
    return out

def sparse_dense(a,b):
    out=np.zeros((len(a.parent),len(b.parent)))
    for (u,v),value in branch_overlap(a,b).items():out[u,v]=value
    return out

def permutation(tree, order):
    inverse=np.argsort(order);parent=tree.parent[order]
    return Tree(np.where(parent>=0,inverse[parent.clip(min=0)],-1),tree.age[order],inverse[tree.leaves],tree.sample_ids).validate()

def balanced(h, seed):
    random=np.random.default_rng(seed);nodes=list(random.permutation(h));parent=[-1]*h;ages=[0.]*h;level=1
    while len(nodes)>1:
        following=[]
        for start in range(0,len(nodes),2):
            pair=nodes[start:start+2]
            if len(pair)==1: following+=pair;continue
            ancestor=len(parent);parent.append(-1);ages.append(float(20*level))
            for child in pair:parent[child]=ancestor
            following.append(ancestor)
        nodes=following;level+=1
    return Tree(np.array(parent),np.array(ages),np.arange(h),tuple(range(h)))

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(83)
    a,b=small(),small(True)
    mismatch=Tree(a.parent,a.age,a.leaves,(0,2,1,3))
    rejected=False
    try:build_graph([a,mismatch],[0.,1.],1.5)
    except ValueError:rejected=True
    check('Misaligned same-age sample rosters are rejected',rejected)
    exact=sparse_dense(a,b)
    check('Sparse path overlap equals independent dense descendant intersection',np.array_equal(exact,dense_overlap(a,b)))
    check('Cross-tree raw coupling transpose identity',np.array_equal(exact,sparse_dense(b,a).T))
    check('Identical tree coupling is diagonal finite-branch mass',np.array_equal(sparse_dense(a,a),np.diag(a.masses())))
    graph=build_graph([a,b,a],[0.,1.,2.],1.5)
    check('Branch coverage lies in [0,1]',torch.all((graph.branch_coverage>=0)&(graph.branch_coverage<=1+1e-12)),maximum=float(graph.branch_coverage.max()))
    ancient=Tree(a.parent,np.array([1.,1.,1.,1.,2.,3.,8.]),a.leaves,a.sample_ids).validate()
    zero=Tree(a.parent,np.array([0.,0.,0.,0.,0.,3.,8.]),a.leaves,a.sample_ids).validate()
    check('Ancient sample mass starts at sampling age',np.array_equal(ancient.masses()[:4],np.array([1.,1.,2.,2.])))
    zero_graph=build_graph([zero,a],[0.,1.],1.5)
    check('Zero branches remain finite without epsilon division',torch.isfinite(zero_graph.branch_coverage).all() and zero_graph.branch_coverage[0].sum()==0)
    short_root=Tree(a.parent,np.array([0.,0.,0.,0.,2.,3.,4.]),a.leaves,a.sample_ids)
    missing=build_graph([a,short_root],[0.,1.],1.5)
    check('Missing old-age support is not redistributed',0<float(missing.branch_coverage[4,1])<1,coverage=float(missing.branch_coverage[4,1]))
    model=BranchJointConv(5).double();features=torch.randn(graph.node_count,5,dtype=torch.double)
    order=np.array([6,2,4,1,5,0,3]);renamed=build_graph([permutation(t,order) for t in [a,b,a]],[0.,1.,2.],1.5)
    global_order=np.concatenate([order+j*7 for j in range(3)])
    output=model(features,graph);rename_output=model(features[global_order],renamed)
    error=float((rename_output-output[global_order]).abs().max())
    check('Internal node relabelling equivariance',error<1e-12,maximum_error=error)
    sample_order=np.array([2,0,3,1])
    reordered=build_graph([Tree(t.parent,t.age,t.leaves[sample_order],tuple(t.sample_ids[j] for j in sample_order)) for t in [a,b,a]],[0.,1.,2.],1.5)
    sample_error=float((model(features,reordered)-output).abs().max())
    check('Shared sample roster permutation leaves native outputs unchanged',sample_error<1e-12,maximum_error=sample_error)
    joined=disjoint_batch([graph,graph]);other=features*2+.3
    batch_error=float((model(torch.cat([features,other]),joined)-torch.cat([output,model(other,graph)])).abs().max())
    check('Disjoint minibatch equals separate evaluation',batch_error<1e-12,maximum_error=batch_error)
    changed=torch.cat([features,other+100])
    check('No foreign window influence',torch.equal(model(changed,joined)[:graph.node_count],model(torch.cat([features,other]),joined)[:graph.node_count]))
    # An interval split changes no sampled coordinate or attributed tree at that coordinate.
    intervals=[(0.,2.,a),(2.,4.,b)];split=[(0.,.4,a),(.4,2.,a),(2.,3.2,b),(3.2,4.,b)]
    coords=[.3,1.6,2.5]
    def select(records):return [next(tree for left,right,tree in records if left<=x<right) for x in coords]
    full_interval=build_graph(select(intervals),coords,2.);split_interval=build_graph(select(split),coords,2.)
    check('Fixed coordinates invariant to pure attributed-interval subdivision',torch.equal(model(features,full_interval),model(features,split_interval)))
    whole=build_graph([a,b,a,b,a,b,a,b,a],np.arange(9),1.1)
    halo=build_graph([a,b,a,b,a],np.arange(2,7),1.1)
    whole_x=torch.randn(whole.node_count,5,dtype=torch.double,requires_grad=True)
    halo_x=whole_x.detach()[14:49].clone().requires_grad_()
    layers=torch.nn.ModuleList([BranchJointConv(5).double(),BranchJointConv(5).double()])
    def stack(x,g):
        for layer in layers:x=layer(x,g)
        return x
    out_all=stack(whole_x,whole)[28:35];out_halo=stack(halo_x,halo)[14:21]
    err=float((out_all-out_halo).abs().max())
    check('Two-layer core equals exact two-grid-step halo',err<1e-12,maximum_error=err)
    probe=torch.randn_like(out_all)
    params=tuple(layers.parameters())
    grad_all=torch.autograd.grad((out_all*probe).sum(),(whole_x,)+params)
    grad_halo=torch.autograd.grad((out_halo*probe).sum(),(halo_x,)+params)
    grad_err=max(float((ga-gh).abs().max()) for ga,gh in zip((grad_all[0][14:49],)+grad_all[1:],grad_halo))
    outside=torch.cat([grad_all[0][:14],grad_all[0][49:]])
    check('Core halo input and parameter gradients agree',grad_err<1e-11 and torch.count_nonzero(outside)==0,maximum_error=grad_err)
    finite_features=torch.randn(14,2,dtype=torch.double,requires_grad=True)
    finite_model=BranchJointConv(2).double();finite_graph=build_graph([a,b],[0.,1.],1.5)
    passed=torch.autograd.gradcheck(lambda x:finite_model(x,finite_graph),(finite_features,),eps=1e-6,atol=1e-5,rtol=1e-3)
    check('Autograd agrees with finite differences on joint layer',passed)
    # Full same binary topology but changed branch dates between any cuts at 0 and 8.
    changed_age=Tree(a.parent,np.array([0.,0.,0.,0.,2.5,3.,8.]),a.leaves,a.sample_ids)
    check('Branch overlap detects age changes missed by endpoint cuts',not np.array_equal(sparse_dense(a,a),sparse_dense(a,changed_age)))
    # Representative synthetic size, unrelated random balanced trees, no real observations.
    start=time.perf_counter();large=build_graph([balanced(952,j) for j in range(5)],np.arange(5),2.1);prep=time.perf_counter()-start
    big_layers=torch.nn.ModuleList([BranchJointConv(32) for _ in range(3)])
    big_x=torch.randn(large.node_count,32,requires_grad=True)
    def run():
        result=big_x
        for layer in big_layers:result=layer(result,large)
        return result
    for _ in range(2):
        with torch.no_grad():run()
    start=time.perf_counter();big_out=run();forward=time.perf_counter()-start
    start=time.perf_counter();(big_out*torch.randn_like(big_out)).mean().backward();backward=time.perf_counter()-start
    check('Representative synthetic backward finite',torch.isfinite(big_x.grad).all() and all(p.grad is not None and torch.isfinite(p.grad).all() for p in big_layers.parameters()))
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Synthetic CPU operator checks only. No labels, fit, real data, GPU or baseline mutation.',
      'environment':{'python':platform.python_version(),'torch':torch.__version__,'pyg':torch_geometric.__version__,'threads':torch.get_num_threads(),'cuda_visible_devices':'empty'},
      'checks':checks,'synthetic_timing':{'trees':5,'leaves_per_tree':952,'nodes':large.node_count,'edges_by_relation':[r.edge_index.shape[1] for r in large.relations],
      'width':32,'layers':3,'preparation_seconds':prep,'forward_seconds':forward,'backward_seconds':backward,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'limitation':'One measured invocation after two warmups; random balanced toy trees. Not real loader or training throughput.'},
      'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in [ROOT/'branch_joint_conv.py',Path(__file__)]}}
    (ROOT/'test_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
