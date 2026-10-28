"""Small CPU mathematical/contract checks. No ancestry learning or CUDA."""
import copy
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from research.redesign_20261028.architecture.window_model import (
    OBS_KEYS,WindowTreeLoom,prepare_observations,collate_observations,
    load_observations,interval_kernel,mean_to_cut,temporal_mean,coordinate_observations,context_kernel)

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN/research/redesign_20261028/architecture')


def toy():
    parents=[np.array([4,4,5,5,5,-1]),np.array([4,4,-1,-1,-1]),np.array([4,4,4,4,-1])]
    times=[np.array([0.,0.,200.,500.,100.,700.]),np.array([0.,0.,200.,500.,50.]),np.array([0.,0.,200.,500.,800.])]
    p=[];nt=[];leaves=[];offset=0
    for ti,part in enumerate(parents):
        p.extend(np.where(part>=0,part+offset,-1));nt.extend([ti]*len(part))
        leaves.append(np.arange(4)+offset);offset+=len(part)
    cohort=np.array([0,2,5,8]);ref=np.array([-1,-1,2,3]);age=np.array([0.,0.,200.,500.]);query=ref<0
    meta=np.concatenate((np.eye(9)[cohort],np.eye(5)[np.where(ref>=0,ref,4)],
                         (np.log1p(age)/np.log(10001))[:,None],np.stack((query,~query),-1)),axis=-1)
    genotype=np.array([[0,1,0,1],[1,0,1,0],[0,1,1,1],[1,1,0,-1],[1,0,0,1],[0,0,1,1]])
    return dict(parent=np.array(p),node_time=np.concatenate(times),node_tree_index=np.array(nt),
        leaf_node_index=np.stack(leaves),leaf_features=meta.astype(np.float32),
        tree_interval_local_bp=np.array([[0.,100.],[100.,200.],[200.,400.]]),
        tree_window_index=np.zeros(3,dtype=np.int64),tree_core_mask=np.array([False,True,False]),
        focal_tree_index=np.array([0,1,2]),focal_sample_index=np.array([0,1,0]),
        focal_position_local_bp=np.array([10.,120.,250.]),focal_genotypes=genotype[[0,2,4]],
        focal_allele_count=np.full(3,2),reference_class=ref,sample_age=age,
        observed_cohort=cohort,query_mask=query,site_tree_index=np.repeat(np.arange(3),2),
        site_position_local_bp=np.array([10.,40.,120.,160.,250.,300.]),
        site_genotypes=genotype,site_allele_count=np.full(6,2))


def permute_leaves(a,permutation):
    a=copy.deepcopy(a);inverse=np.argsort(permutation)
    for key in ('leaf_features','reference_class','sample_age','observed_cohort','query_mask'):
        a[key]=a[key][permutation]
    for key in ('leaf_node_index','site_genotypes','focal_genotypes'):
        a[key]=a[key][:,permutation]
    a['focal_sample_index']=inverse[a['focal_sample_index']]
    return a


def permute_nodes(a,permutation):
    a=copy.deepcopy(a);inverse=np.argsort(permutation);old=a['parent'][permutation]
    a['parent']=np.where(old>=0,inverse[old.clip(min=0)],-1)
    for key in ('node_time','node_tree_index'):a[key]=a[key][permutation]
    a['leaf_node_index']=inverse[a['leaf_node_index']]
    return a


def main():
    torch.set_num_threads(2);torch.manual_seed(20261028)
    report=dict(schema='bounded_treeloom_cpu_contract_v1',tests={},optimizer_updates=0,
                accuracy_training=False,query_truth_read=False,cuda_initialized=False)
    a=toy();batch=prepare_observations(a);model=WindowTreeLoom(width=16)
    output=model(batch)
    require=lambda value,message:None if value else (_ for _ in ()).throw(AssertionError(message))
    def close(name,observed,expected,tolerance=2e-5):
        error=float((observed-expected).abs().max()) if observed.numel() else 0.
        require(torch.allclose(observed,expected,atol=tolerance,rtol=tolerance),name)
        report['tests'][name]=dict(passed=True,max_abs_error=error)
    require(output.shape==(3,6) and torch.isfinite(output).all(),'forward')
    output.square().mean().backward()
    grads={name:p.grad for name,p in model.named_parameters()}
    require(all(g is not None and torch.isfinite(g).all() for g in grads.values()),'finite backward')
    report['tests']['finite_forward_backward']=dict(passed=True,parameter_tensors=len(grads),
        nonzero_gradient_tensors=sum(bool(g.abs().max()>0) for g in grads.values()))
    with torch.no_grad():
        expected=model(batch)
        close('leaf_permutation',model(prepare_observations(permute_leaves(a,np.array([2,0,3,1])))),expected)
        rng=np.random.default_rng(14)
        close('internal_and_leaf_node_renumbering',model(prepare_observations(permute_nodes(a,rng.permutation(len(a['parent']))))),expected)
        other=copy.deepcopy(a);other['site_genotypes']=(other['site_genotypes']+1)%2
        joined=prepare_observations(collate_observations([a,other]))
        both=model(joined)
        close('independent_example_isolation',both[:3],expected)
        reverse=model(prepare_observations(collate_observations([other,a])))
        close('batch_example_permutation',reverse[3:],expected)
        close('second_example_matches_separate',both[3:],model(prepare_observations(other)))
        shared=collate_observations([a,a])
        for key in ('leaf_features','reference_class','sample_age','observed_cohort','query_mask'):
            shared[key]=shared[key][0]
        shared_batch=prepare_observations(shared)
        close('shared_parent_multiple_windows',model(shared_batch),torch.cat((expected,expected)))
        shared_collated=prepare_observations(collate_observations([shared,other]))
        close('collated_multiwindow_shared_panel',model(shared_collated),torch.cat((expected,expected,model(prepare_observations(other)))))
        require(model.core_logits(batch).shape==(1,6),'core selection')
        close('core_selection',model.core_logits(batch),expected[1:2])
        neighbor=copy.deepcopy(a);neighbor['site_genotypes'][0,0]=1-neighbor['site_genotypes'][0,0]
        neighbor_batch=prepare_observations(neighbor)
        close('no_cross_excludes_neighbor',model(neighbor_batch,mode='no_cross')[1],model(batch,mode='no_cross')[1],1e-6)
        change=float((model(neighbor_batch)[1]-expected[1]).abs().max())
        require(change>1e-8,'structured communication is connected')
        report['tests']['structure_receives_neighbor']=dict(passed=True,max_logit_change=change)
        focal=copy.deepcopy(a);focal['focal_genotypes'][1,1]=1-focal['focal_genotypes'][1,1]
        focal_change=float((model(prepare_observations(focal))[1]-expected[1]).abs().max())
        require(focal_change>1e-8,'direct focal SNP channel')
        report['tests']['direct_focal_SNP_connected']=dict(passed=True,max_logit_change=focal_change)
        altered=copy.deepcopy(a);altered['leaf_features'][1,0:9]=np.eye(9)[4]
        cohort_change=float((model(prepare_observations(altered))[1]-expected[1]).abs().max())
        require(cohort_change>1e-8,'direct cohort channel')
        report['tests']['cohort_connected']=dict(passed=True,max_logit_change=cohort_change)
    for cut in batch.cuts:
        constants=torch.full((len(cut.counts),3),2.75)
        close('cut_constant_'+str(len(cut.counts)),mean_to_cut(constants[cut.groups],cut),constants)
        source_values=torch.randn(len(cut.counts),3)
        leaf=source_values[cut.groups[0]]
        target_groups=cut.groups[1];target_unique=torch.unique(target_groups)
        target=torch.zeros((len(cut.counts),3)).index_add(0,target_groups,leaf)/cut.counts[:,None]
        source_unique=torch.unique(cut.groups[0])
        source_mass=(source_values[source_unique]*cut.counts[source_unique,None]).sum(0)
        target_mass=(target[target_unique]*cut.counts[target_unique,None]).sum(0)
        close('cut_mass_'+str(len(cut.counts)),target_mass,source_mass)
    require(any((cut.valid_time==0).any() for cut in batch.cuts),'ancient singleton mask')
    report['tests']['forest_polytomy_ancient_singleton']=dict(passed=True,roots=int((batch.parent<0).sum()))
    intervals=torch.tensor([[0.,2.],[2.,5.]],dtype=torch.float64)
    refined=torch.tensor([[0.,.5],[.5,2.],[2.,5.]],dtype=torch.float64)
    k=interval_kernel(intervals,torch.zeros(2,dtype=torch.long),torch.tensor(1.7))
    r=interval_kernel(refined,torch.zeros(3,dtype=torch.long),torch.tensor(1.7))
    merge=torch.tensor([[1.,1.,0.],[0.,0.,1.]],dtype=torch.float64)
    close('integral_kernel_additivity_only',merge@r@merge.T,k,1e-10)
    require(torch.linalg.eigvalsh(k).min()>0,'positive kernel')
    report['tests']['kernel_PSD']=dict(passed=True,min_eigenvalue=float(torch.linalg.eigvalsh(k).min()))
    bad=copy.deepcopy(a);bad['tree_interval_local_bp'][1,1]=100
    try:prepare_observations(bad)
    except ValueError:report['tests']['zero_span_rejected']=dict(passed=True)
    else:raise AssertionError('zero span accepted')
    try:prepare_observations(dict(a,labels=np.array([1,2,3])))
    except ValueError:report['tests']['labels_rejected_by_whitelist']=dict(passed=True)
    else:raise AssertionError('labels accepted')
    # A genuine one-leaf forest with no edges and no observed SNPs.
    single={k:copy.deepcopy(v) for k,v in a.items()}
    single.update(parent=np.array([-1]),node_time=np.array([0.]),node_tree_index=np.array([0]),
        leaf_node_index=np.array([[0]]),leaf_features=a['leaf_features'][:1],tree_interval_local_bp=np.array([[0.,1.]]),
        tree_window_index=np.array([0]),tree_core_mask=np.array([True]),focal_tree_index=np.array([0]),
        focal_sample_index=np.array([0]),focal_position_local_bp=np.array([.5]),focal_genotypes=np.array([[-1]]),
        focal_allele_count=np.array([2]),reference_class=np.array([-1]),sample_age=np.array([0.]),
        observed_cohort=np.array([0]),query_mask=np.array([True]),site_tree_index=np.empty(0,dtype=np.int64),
        site_position_local_bp=np.empty(0),site_genotypes=np.empty((0,1),dtype=np.int64),site_allele_count=np.empty(0,dtype=np.int64))
    singleton=model(prepare_observations(single))
    require(singleton.shape==(1,6) and torch.isfinite(singleton).all(),'singleton')
    report['tests']['singleton_empty_sites_missing_focal']=dict(passed=True)
    # Coordinate tokens preserve repeated draws, with no extra span factor.
    focal_only=copy.deepcopy(a)
    for key in ('focal_tree_index','focal_sample_index','focal_position_local_bp','focal_genotypes','focal_allele_count'):
        focal_only[key]=focal_only[key][1:2]
    context=dict(context_coordinate_local_bp=np.array([120.,20.,50.,250.]),
        context_tree_index=np.array([1,0,0,2]),context_window_index=np.zeros(4,dtype=np.int64),
        context_quadrature_weight=np.array([0.,1/3,1/3,1/3]),
        context_proposal_density_per_bp=np.array([0.,1/400,1/400,1/400]),
        context_forced_focal=np.array([True,False,False,False]),
        window_context_domain_local_bp=np.array([[0.,400.]]),window_context_kind=np.array([1]))
    coordinate=coordinate_observations(focal_only,context)
    cb=prepare_observations(coordinate)
    require(len(cb.leaves)==4 and len(cb.parent)==22,'Repeated tree draws were collapsed')
    ck=context_kernel(cb,torch.tensor(100.))[0][1]
    direct=torch.exp(-(cb.coordinate[:,None]-cb.coordinate[None,:]).abs()/100.)*cb.coordinate_weight[None,:]
    close('coordinate_weights_without_tree_span',ck,direct,1e-12)
    with torch.no_grad():
        co=model(cb)
        order=np.array([3,2,0,1]);permuted=copy.deepcopy(context)
        for key in ('context_coordinate_local_bp','context_tree_index','context_window_index',
                    'context_quadrature_weight','context_proposal_density_per_bp','context_forced_focal'):
            permuted[key]=permuted[key][order]
        close('coordinate_draw_permutation',model(prepare_observations(coordinate_observations(focal_only,permuted))),co)
        cc=prepare_observations(collate_observations([coordinate,coordinate]))
        close('coordinate_batch_isolation',model(cc),torch.cat((co,co)))
        close('coordinate_no_cross_focal_matches_original',model(cb,mode='no_cross'),model(prepare_observations(focal_only),mode='no_cross'))
    # Qualify efficient controls against the preserved pre-skip implementation.
    from research.redesign_20261028.architecture import window_model_v3 as previous
    parity={}
    for mode in ('no_cross','leaf_only'):
        old=previous.WindowTreeLoom(width=16);old.load_state_dict(model.state_dict())
        new=WindowTreeLoom(width=16);new.load_state_dict(model.state_dict())
        old_out=old(previous.prepare_observations(a),mode=mode)
        new_out=new(prepare_observations(a),mode=mode)
        close('efficient_'+mode+'_logits',new_out,old_out,1e-6)
        old_out.square().sum().backward();new_out.square().sum().backward()
        max_error=0.
        for (name,op),(new_name,np_) in zip(old.named_parameters(),new.named_parameters()):
            require(name==new_name,'Matched parameter names')
            og=torch.zeros_like(op) if op.grad is None else op.grad
            ng=torch.zeros_like(np_) if np_.grad is None else np_.grad
            require(torch.allclose(ng,og,atol=2e-6,rtol=2e-5),'Efficient control gradient: '+name)
            max_error=max(max_error,float((ng-og).abs().max()))
        parity[mode]=dict(max_gradient_abs_error=max_error,
            connected_parameter_tensors=sum(p.grad is not None for p in new.parameters()),
            connected_parameter_elements=sum(p.numel() for p in new.parameters() if p.grad is not None))
    report['efficient_controls']=parity
    report['cuda_initialized']=torch.cuda.is_initialized()
    require(not report['cuda_initialized'],'CUDA unexpectedly initialized')
    report['peak_RSS_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['all_passed']=True
    (ROOT/'CPU_TESTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
