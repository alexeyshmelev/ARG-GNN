"""CPU-only scientific contract and full-predictor qualification; never fits labels."""
from pathlib import Path
from dataclasses import replace,fields
from hashlib import sha256
from datetime import datetime,timezone
import copy,json,io,time,resource,sys
import numpy as np
import torch
from joint_model_v1 import *

ROOT=Path(__file__).resolve().parent
checks=[]

def check(name,condition,**evidence):
    if not bool(condition):raise AssertionError(name+': '+str(evidence))
    checks.append({'name':name,'pass':True,**evidence});print(name,flush=True)


def toy(parent=None,ages=None,leaves=None,targets=None):
    parent=np.array([4,4,5,5,6,6,-1]) if parent is None else np.asarray(parent)
    ages=np.array([0.,0.,0.,0.,2.,3.,8.]) if ages is None else np.asarray(ages,dtype=float)
    leaves=np.arange(4) if leaves is None else np.asarray(leaves)
    h=len(leaves);n=len(parent);k=5
    metadata=np.random.default_rng(6).normal(size=(h,17)).astype(np.float32)
    reference=np.full(h,-1,dtype=np.int64);reference[0]=0
    query=reference<0
    selected=np.flatnonzero(query) if targets is None else np.asarray(targets)
    genotype=(np.arange(h)%2)[None,:]
    ids=np.array([f'parent:sample_{j}:hap0' for j in range(h)])
    coords=np.array([0.,-1000.,1000.,-3000.,3000.])
    return dict(parent=np.concatenate([np.where(parent>=0,parent+j*n,-1) for j in range(k)]),
        node_time=np.tile(ages,k),node_tree_index=np.repeat(np.arange(k),n),leaf_node_index=np.stack([leaves+j*n for j in range(k)]),
        leaf_features=metadata,tree_interval_local_bp=np.tile([[-100000.,100000.]],(k,1)),tree_window_index=np.zeros(k,dtype=np.int64),
        tree_core_mask=np.array([True,False,False,False,False]),focal_tree_index=np.array([0]),focal_sample_index=selected,
        focal_position_local_bp=np.array([0.]),focal_genotypes=genotype,focal_allele_count=np.array([2]),reference_class=reference,
        sample_age=ages[leaves],observed_cohort=np.arange(h)%9,query_mask=query,site_tree_index=np.arange(k),
        site_position_local_bp=coords.copy(),site_genotypes=np.repeat(genotype,k,axis=0),site_allele_count=np.full(k,2),
        focal_site_index=np.zeros(len(selected),dtype=np.int64),sample_id=ids,tree_sample_id=np.tile(ids,(k,1)),
        model_coordinate_bp=coords,model_coordinate_weight=np.array([0.,.25,.25,.25,.25]),model_context_kind=np.array([1]))


def reject(packet):
    try:prepare_observations(packet)
    except ValueError:return True
    return False


class ScatterReference(SparseSum):
    def message_and_aggregate(self,adj_t,x):
        coo=adj_t.to_sparse_coo();indices=coo.indices();values=coo.values()
        return x.new_zeros((adj_t.size(0),x.size(1))).index_add(0,indices[0],x[indices[1]]*values[:,None])


def install_reference(module):
    for name,child in list(module.named_children()):
        if isinstance(child,SparseSum):setattr(module,name,ScatterReference())
        else:install_reference(child)


def parity(batch,width=16,layers=2):
    torch.manual_seed(101);model=JointAncestryModel(width=width,layers=layers)
    reference=copy.deepcopy(model);install_reference(reference)
    results={}
    for mode in ('no_cross','leaf_only','structure'):
        model.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
        actual=model(batch,mode);expected=reference(batch,mode);probe=torch.randn_like(actual)
        (actual*probe).sum().backward();(expected*probe).sum().backward()
        max_error=float((actual-expected).abs().max());grad_error=0.;active=0
        for p,q in zip(model.parameters(),reference.parameters()):
            check_equal_none=(p.grad is None)==(q.grad is None)
            if not check_equal_none:raise AssertionError('Gradient presence mismatch')
            if p.grad is not None:
                active+=1;grad_error=max(grad_error,float((p.grad-q.grad).abs().max()))
                torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        results[mode]={'logit_max_error':max_error,'parameter_vjp_max_error':grad_error,'active_parameter_tensors':active}
    return results


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(101)
    a=toy();b=prepare_observations(a)
    bad=copy.deepcopy(a);bad['tree_sample_id'][1]=bad['tree_sample_id'][1,::-1]
    check('Misaligned same-age sample IDs rejected',reject(bad))
    bad=dict(a,labels=np.ones(len(a['focal_sample_index'])))
    check('Ancestry labels rejected from observation API',reject(bad))
    model=JointAncestryModel(width=16,layers=2);model.eval()
    base=model(b)
    keep=torch.tensor([2,0]);subset=replace(b,focal_site=b.focal_site[keep],focal_sample=b.focal_sample[keep])
    check('Query selection only gathers outputs; shared site encoding valid',torch.equal(model(subset),base[keep]))
    # Dense independent tree membership oracle.
    dense=b.descendant.to_dense().numpy();ancestor=b.ancestor.to_dense().numpy()
    oracle=np.zeros_like(dense)
    for ti,row in enumerate(a['leaf_node_index']):
        for si,leaf in enumerate(row):
            node=int(leaf)
            while node>=0:oracle[node,ti*4+si]=1;node=int(a['parent'][node])
    oracle=oracle/oracle.sum(1,keepdims=True)
    check('Descendant map equals independent membership oracle',np.allclose(dense,oracle,atol=1e-7))
    check('Full ancestor readout normalized and includes roots',np.allclose(ancestor.sum(1),1.,atol=1e-7) and np.all(ancestor[0,[0,4,6]]>0))
    order=np.random.default_rng(17).permutation(len(a['parent']));inv=np.argsort(order);permuted=copy.deepcopy(a)
    oldp=a['parent'][order];permuted['parent']=np.where(oldp>=0,inv[oldp.clip(min=0)],-1)
    for key in ('node_time','node_tree_index'):permuted[key]=a[key][order]
    permuted['leaf_node_index']=inv[a['leaf_node_index']]
    err=float((model(prepare_observations(permuted))-base).abs().max())
    check('Internal node relabeling preserves full logits',err<3e-6,error=err)
    roster=np.array([2,0,3,1]);inverse=np.argsort(roster);q=copy.deepcopy(a)
    for key in ('leaf_features','reference_class','sample_age','observed_cohort','query_mask','sample_id'):q[key]=q[key][roster]
    for key in ('leaf_node_index','tree_sample_id','site_genotypes','focal_genotypes'):q[key]=q[key][:,roster]
    q['focal_sample_index']=inverse[q['focal_sample_index']]
    err=float((model(prepare_observations(q))-base).abs().max())
    check('Shared sample permutation preserves full logits',err<3e-6,error=err)
    second=prepare_observations(toy());joined=collate_prepared([b,second])
    err=float((model(joined)-torch.cat([model(b),model(second)])).abs().max())
    check('Collated site windows preserve outputs/order and isolation',err<3e-6,error=err)
    altered=replace(joined,metadata=joined.metadata.clone());altered.metadata[1]+=100
    err=float((model(altered)[:b.target_count]-model(joined)[:b.target_count]).abs().max())
    check('Foreign batch-window observations cannot change target logits',err==0.,error=err)
    buffer=io.BytesIO();torch.save(prepared_to_state(b),buffer);buffer.seek(0)
    restored=prepared_from_state(torch.load(buffer,weights_only=True))
    check('Weights-only cache roundtrip preserves full predictor',torch.equal(model(restored),base))
    q=copy.deepcopy(a);q['model_coordinate_bp'][2]=q['model_coordinate_bp'][1]
    duplicate=prepare_observations(q)
    check('Exact duplicate coordinates retained with source quadrature mass',len(duplicate.leaves)==5 and torch.isfinite(model(duplicate)).all())
    q=copy.deepcopy(a);q['model_coordinate_bp'][1]=-49000
    far=prepare_observations(q)
    near_value=float(b.cross[0].to_dense()[0,7]);far_value=float(far.cross[0].to_dense()[0,7])
    check('Absolute distance attenuation survives sparse neighbors',far_value<near_value/10,near=near_value,far=far_value)
    const=torch.ones((len(b.parent),16));residuals=[]
    for j,matrix in enumerate(b.cross):residuals.append(float((torch.sparse.mm(matrix,const)-b.coupling_mass[:,j,None]*const).abs().max()))
    check('Constant states have zero cross residual from exact adjacency mass',max(residuals)<1e-7,errors=residuals)
    p=np.array([4,4,5,5,6,6,7,-1]);age=np.array([0,0,0,0,2,3,8,10.]);q=toy(p,age,np.arange(4));q['node_time'][8:16]=[0,0,0,0,.3,.6,1.,10.]
    specificity=prepare_observations(q);value=float(specificity.cross[2].to_dense()[4,8+6]);expected=.25*(1-1000/50000)*.5
    check('Jaccard reduces broad-source containment weight',abs(value-expected)<1e-7,value=value,expected=expected)
    q=toy(ages=[0,0,0,0,8,3,8]);zero=prepare_observations(q)
    check('Zero-duration internal branch has no branch message',float(zero.coupling_mass[4,2:].sum())==0 and torch.isfinite(model(zero)).all())
    huge=1e16;next_age=np.nextafter(huge,np.inf);tiny=prepare_observations(toy(ages=[0,0,0,0,huge,huge,next_age]))
    check('Tiny positive branch at large age stays finite and coupled',torch.isfinite(tiny.coupling_mass).all() and float(tiny.coupling_mass[4,2:].sum())>0)
    singleton=prepare_observations(toy(parent=[-1,-1],ages=[0,0],leaves=[0,1]))
    check('Forest singleton roots are valid leaf/root atoms',torch.equal(singleton.ancestor.to_dense(),torch.eye(10)) and torch.isfinite(model(singleton)).all())
    # A 160-edge unary chain above a complete binary tree is intentionally retained.
    parent=[4,4,5,5,6,6,7];ages=[0.,0.,0.,0.,2.,3.,8.]
    for u in range(7,167):parent.append(u+1 if u<166 else -1);ages.append(float(u+2))
    tall_packet=toy(parent,ages,np.arange(4),targets=[2]);tall=prepare_observations(tall_packet)
    rejected=False
    try:prepare_observations(tall_packet,max_path_entries=100)
    except ValueError as e:rejected='no truncation' in str(e)
    check('Tall tree retained exactly and explicit resource bound fails without truncation',tall.statistics['max_path_nodes']>160 and rejected,statistics=tall.statistics)
    # Isolate foreign deep branch content. No leaf or native channel can carry it.
    torch.manual_seed(9);deep_model=JointAncestryModel(width=8,layers=1)
    for linear in deep_model.layers[0].native_maps:linear.weight.data.zero_()
    h0=torch.randn((len(tall.parent),8),requires_grad=True)
    deep_model.initialize_nodes=lambda batch:h0
    deep_source=167+165
    true=deep_model(tall,'structure')[0,0];g_true=torch.autograd.grad(true,h0,retain_graph=True)[0][deep_source]
    leaf=deep_model(tall,'leaf_only')[0,0];g_leaf=torch.autograd.grad(leaf,h0)[0][deep_source]
    check('All-depth readout exposes foreign deep branch gradient beyond layer hop count',float(g_true.norm())>1e-9 and float(g_leaf.norm())==0.,branch_gradient=float(g_true.norm()),leaf_only_gradient=float(g_leaf.norm()),native_hops=161)
    toy_parity=parity(b)
    check('Full toy FP32 SpMM vs independent scatter logits and all parameter VJPs',True,arms=toy_parity)
    real_path=ROOT.parent/'data/profile_observations_v1/site_000000.npz'
    start=time.perf_counter();real=prepare_observations(load_observations(real_path));prepare_seconds=time.perf_counter()-start
    real_parity=parity(real)
    check('Full real-observation FP32 SpMM vs scatter logits and all parameter VJPs',True,arms=real_parity)
    # Registered default-size full model, all targets, no label loss or optimizer.
    torch.manual_seed(101);full=JointAncestryModel(width=64,layers=3)
    timing={}
    for mode in ('no_cross','leaf_only','structure'):
        with torch.no_grad():full(real,mode)
        full.zero_grad(set_to_none=True);start=time.perf_counter();out=full(real,mode);forward=time.perf_counter()-start
        start=time.perf_counter();(out*torch.randn_like(out)).sum().backward();backward=time.perf_counter()-start
        check('Finite complete real forward/backward '+mode,torch.isfinite(out).all() and all(p.grad is None or torch.isfinite(p.grad).all() for p in full.parameters()))
        timing[mode]={'forward_seconds':forward,'backward_seconds':backward,'targets':len(out),'parameters':sum(p.numel() for p in full.parameters())}
    # Source comparison uses the SAME observed coordinates and unique focal rows.
    sys.path.insert(0,str(ROOT.parents[2]/'redesign_20261028/architecture'))
    import window_model_frozen_v1 as old
    expanded=expand_coordinates(load_observations(real_path));old_obs={k:expanded[k] for k in old.OBS_KEYS+old.MODEL_CONTEXT_KEYS}
    # The old API requires one focal genotype row per query, so explicitly expand only that interface.
    index=expanded['focal_site_index']
    for key in ('focal_tree_index','focal_position_local_bp','focal_genotypes','focal_allele_count'):old_obs[key]=old_obs[key][index]
    start=time.perf_counter();old_batch=old.prepare_observations(old_obs);old_prep=time.perf_counter()-start
    old_model=old.WindowTreeLoom(width=64)
    with torch.no_grad():old_model(old_batch)
    start=time.perf_counter();old_out=old_model(old_batch);old_fwd=time.perf_counter()-start
    start=time.perf_counter();(old_out*torch.randn_like(old_out)).sum().backward();old_back=time.perf_counter()-start
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU qualification, original TRAIN observations only, no ancestry labels, fit or GPU',
      'source_sha256':sha256((ROOT/'joint_model_v1.py').read_bytes()).hexdigest(),'test_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
      'checks':checks,'real_packet':str(real_path),'real_statistics':real.statistics,'new_prepare_seconds':prepare_seconds,
      'new_full_model_timing':timing,'old_bounded_v1_same_observations':{'prepare_seconds':old_prep,'forward_seconds':old_fwd,'backward_seconds':old_back,'parameters':sum(p.numel() for p in old_model.parameters()),'targets':len(old_out)},
      'environment':{'torch':torch.__version__,'threads':torch.get_num_threads(),'device':'cpu'},'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'timing_limitation':'One invocation per mode after warmup, width64 L3 new vs original frozen width64. Different architectures and parameters; no accuracy claim. Old API duplicates focal SNP encoding across target rows.'}
    (ROOT/'qualification_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
