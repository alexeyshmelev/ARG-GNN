"""Root-launcher APIs for exact bounded C3 inference and short-parent training.

No CLI launch, resource authority or CUDA initialization exists at import.
Training labels are separate arguments and only accepted for registered TRAIN.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .american_c3_common_v1 import require,bound
from .american_c3_evaluate_v1 import registry,verify_prediction
from .american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
from .american_c3_gnn_dataset_isolated_v2 import AmericanC3WholeTreeDatasetV1,shutdown_dataloader


@torch.inference_mode()
def _streamed_focal_predictions_impl(model,forward_loader,reverse_loader,*,device):
    """Exact full-parent bidirectional recurrence; CPU storage only O(focals*D).

    Trees are encoded twice in eval mode. Forward stores focal forward states.
    Reverse recomputes local tree contexts, carries reverse state, and seals
    focal posteriors. Block boundaries never reset recurrence or omit trees.
    Mathematical equivalence to complete recurrence permits FP32 kernel
    roundoff differences; it is not a finite-halo approximation.
    """
    require(isinstance(model,AmericanC3WholeTreeGNNV1),'Fresh C3 model required')
    fd,rd=forward_loader.dataset,reverse_loader.dataset
    require(isinstance(fd,AmericanC3WholeTreeDatasetV1) and isinstance(rd,AmericanC3WholeTreeDatasetV1)
        and fd.direction=='forward' and rd.direction=='reverse' and fd.plan_ref==rd.plan_ref,
        'Exact same admitted plan in complete opposite loader orders required')
    require(model.class_order==tuple(fd.plan['class_order']),'Fresh checkpoint class contract must equal plan')
    records=registry(fd.plan['focal_registry_ref']);n=len(records['record_ids']);d=model.hidden_dim
    model.eval();actual=next(model.parameters()).device;requested=torch.device(device)
    require(actual.type==requested.type and (requested.index is None or actual.index==requested.index),'Root transfers model before streaming')
    device=actual
    saved=torch.empty((n,d),dtype=torch.float32,device='cpu')
    forward_seen=np.zeros(n,dtype=bool);reverse_seen=np.zeros(n,dtype=bool)
    probabilities=np.empty((n,model.num_classes),dtype=np.float32)
    expected_tree=0;hidden=None;forward_blocks=0
    for block in forward_loader:
        require(block.parent_id==fd.plan['parent_id'] and block.plan_ref==fd.plan_ref and block.start_tree==expected_tree,
            'Forward blocks must cover each original tree exactly once, without gaps')
        expected_tree=block.stop_tree;rows=block.focal_rows.numpy().copy()
        require(not forward_seen[rows].any(),'Duplicate focal forward state')
        block=block.to(device,non_blocking=True);context=model.encode_bank(block.to_model_inputs())
        history,hidden=model.scan_context(context,block.genomic_geometry,direction='forward',hidden=hidden)
        if len(rows):saved[rows]=history[block.focal_hold,block.focal_sample_column].cpu()
        forward_seen[rows]=True;forward_blocks+=1
        del block,context,history
    require(expected_tree==fd.plan['trees'] and forward_blocks==len(fd) and forward_seen.all(),'Complete forward scan required before reverse')
    drain=shutdown_dataloader(forward_loader)
    require(drain['drained'],'Drain forward workers before starting reverse workers')
    expected_tree=fd.plan['trees'];hidden=None;reverse_blocks=0
    for block in reverse_loader:
        require(block.parent_id==fd.plan['parent_id'] and block.plan_ref==fd.plan_ref and block.stop_tree==expected_tree,
            'Reverse blocks must cover every original tree exactly once, without gaps')
        expected_tree=block.start_tree;rows=block.focal_rows.numpy().copy()
        require(not reverse_seen[rows].any(),'Duplicate focal reverse state')
        block=block.to(device,non_blocking=True);context=model.encode_bank(block.to_model_inputs())
        history,hidden=model.scan_context(context,block.genomic_geometry,direction='reverse',hidden=hidden)
        if len(rows):
            local=context[block.focal_hold,block.focal_sample_column]
            reverse=history[block.focal_hold,block.focal_sample_column]
            forward=saved[rows].to(device,non_blocking=True)
            logits=model.classify_context(local,forward,reverse)
            probabilities[rows]=logits.softmax(-1).cpu().numpy()
        reverse_seen[rows]=True;reverse_blocks+=1
        del block,context,history
    require(expected_tree==0 and reverse_blocks==len(rd) and reverse_seen.all(),'Complete reverse scan and every exact focal required')
    prediction={k:records[k] for k in ('record_ids','sample_nodes','sample_indices','positions_bp0','parent_id','class_order')}
    prediction['probabilities']=probabilities;verify_prediction(prediction,records)
    return prediction,dict(schema='american_c3_exact_streaming_focal_prediction_v1',parent_id=fd.plan['parent_id'],
        class_order=list(model.class_order),block_index_ref=fd.plan_ref,focal_registry_ref=fd.plan['focal_registry_ref'],
        observation_ref=fd.plan['observation_ref'],focal_records=n,trees=fd.plan['trees'],
        forward_blocks=forward_blocks,reverse_blocks=reverse_blocks,tree_encoding_passes=2,
        recurrent_state_reset_only_at_true_chromosome_endpoints=True,
        geometry_uses_global_neighbor_spans=True,finite_halo_approximation=False,
        CPU_focal_state_bytes=n*d*4,full_parent_GPU_graph_or_history=False,private_truth_opened=False,
        checkpoint_contract=model.checkpoint_contract(),author_Table1_exact_reproduction=False)


def streamed_focal_predictions(model,forward_loader,reverse_loader,*,device,checkpoint_ref):
    """Owned worker lifecycle: forward workers drain before reverse starts.

    Both loaders drain on success or failure. DataLoader objects may be
    constructed together, but callers must not pre-start the reverse iterator.
    """
    require(getattr(reverse_loader,'_iterator',None) is None,'Reverse workers must not pre-start')
    try:
        checkpoint_metadata=load_c3_checkpoint(checkpoint_ref,model)
        require(forward_loader.dataset.plan['parent_id'] not in checkpoint_metadata['trained_parent_ids'],
            'Independent held-out observation parent required for transfer inference')
        prediction,receipt=_streamed_focal_predictions_impl(model,forward_loader,reverse_loader,device=device)
    finally:
        forward_drain=shutdown_dataloader(forward_loader);reverse_drain=shutdown_dataloader(reverse_loader)
    require(forward_drain['drained'] and reverse_drain['drained'],'All owned workers must drain before sealing')
    receipt['worker_drain']=dict(forward=forward_drain,reverse=reverse_drain)
    receipt['checkpoint_ref']=checkpoint_ref;receipt['training_metadata']=checkpoint_metadata
    receipt['evaluation_length_bp']=forward_loader.dataset.plan['sequence_length_bp']
    receipt['training_condition_differs_from_paper_native']=True
    return prediction,receipt


def train_complete_parent_step(model,block,targets,optimizer,*,split,registered_train_parent_ids,max_training_length_bp,gradient_clip=1.):
    """Explicit supervised TRAIN condition for a complete independent short parent.

    Targets align to block's prespecified query focal rows and are never fields
    of the observation bank. No VAL/TEST truth path or reader is provided.
    Multi-block truncated BPTT is not implemented or silently substituted.
    """
    require(split=='train' and block.parent_id in registered_train_parent_ids and block.complete_parent is True
        and type(max_training_length_bp) is int and 0<block.sequence_length_bp<=max_training_length_bp<30000000,
        'Only registered complete independent TRAIN parents accepted')
    require(targets.dtype==torch.long and targets.shape==block.focal_rows.shape and targets.numel()>0
        and targets.device==next(model.parameters()).device and bool(((targets>=0)&(targets<model.num_classes)).all())
        and gradient_clip>0,'Separate exact TRAIN target/class/device contract required')
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=model(block);loss=F.cross_entropy(logits,targets)
    require(bool(torch.isfinite(loss)),'Finite supervised training loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm),records=targets.numel(),
        condition='fresh_C3_supervised_short_independent_complete_parents',
        paper_native_training_equivalent=False,block_reset_or_truncated_BPTT=False,training_length_bp=block.sequence_length_bp)


def checkpoint_payload(model,*,model_seed,training_condition,trained_parent_ids,train_sequence_lengths_bp,optimizer=None):
    require(type(model_seed) is int and training_condition=='fresh_C3_supervised_short_independent_complete_parents'
        and trained_parent_ids and len(set(trained_parent_ids))==len(trained_parent_ids)
        and len(train_sequence_lengths_bp)==len(trained_parent_ids) and all(0<x<30000000 for x in train_sequence_lengths_bp),'Explicit fresh model seed/TRAIN condition/parents required')
    out=dict(schema='american_c3_fresh_whole_tree_GNN_checkpoint_v1',contract=model.checkpoint_contract(),
        model_seed=model_seed,training_condition=training_condition,trained_parent_ids=list(trained_parent_ids),train_sequence_lengths_bp=list(train_sequence_lengths_bp),
        model_state=model.state_dict(),C4_initialization_or_head_slicing=False)
    if optimizer is not None:out['optimizer_state']=optimizer.state_dict()
    return out


def load_c3_checkpoint(record,model):
    """Strict new-family checkpoint loading. Legacy C4 files have no accepted schema."""
    value=torch.load(bound(record),map_location='cpu',weights_only=True,mmap=True)
    require(value.get('schema')=='american_c3_fresh_whole_tree_GNN_checkpoint_v1'
        and value.get('contract')==model.checkpoint_contract() and value.get('C4_initialization_or_head_slicing') is False
        and value.get('training_condition')=='fresh_C3_supervised_short_independent_complete_parents',
        'Only exact fresh C3-family checkpoints accepted; no C4 relabel/slicing or permissive load')
    model.load_state_dict(value['model_state'],strict=True)
    return {k:v for k,v in value.items() if k not in ('model_state','optimizer_state')}



def qualify_handmade_c3_cpu(*,seed=70001,atol=1e-5):
    """One tiny ROOT-authorized CPU qualifier, without scientific data/targets."""
    import math,os
    from .american_c3_gnn_observations_isolated_v2 import encode_block
    from .american_c3_gnn_dataset_isolated_v2 import AmericanC3BlockV1
    from .mutation_routed_tree_observations_v1 import NativeObservedTreeV1
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Qualifier requires explicit CPU-only environment')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=AmericanC3WholeTreeGNNV1(hidden_dim=4,expected_haplotypes=6,site_chunk_size=2)
        ordinary=torch.nn.GRU(7,4,num_layers=1,bidirectional=True)
    # Six actual leaves, a retained unary vertex, one root. Three handmade holds.
    parent=np.asarray([6,7,7,7,7,7,7,-1],dtype=np.int64)
    clocks=np.asarray([0,0,0,0,0,0,5,20],dtype=np.float64)
    tree=NativeObservedTreeV1(parent,clocks,np.arange(6,dtype=np.int64),np.arange(6,dtype=np.int64))
    trees=[tree,tree,tree];intervals=np.asarray([[0.,3.],[3.,7.],[7.,10.]])
    genotype=np.asarray([[0,0,1,1,0,1],[1,0,1,0,1,0]],dtype=np.int8)
    alleles=np.asarray([2,2],dtype=np.int8);positions=np.asarray([1.,5.])
    # The known EAS reference is deliberately masked; it remains a reference,
    # while only original query columns3,4,5 are readouts.
    reference=np.asarray([0,-1,2,-1,-1,-1],dtype=np.int64);ages=np.zeros(6)
    queries=np.asarray([3,4,5],dtype=np.int64)
    bank,encoding_receipt=encode_block(trees,intervals,genotype,alleles,positions,reference,ages,
        num_classes=3,query_columns=queries)
    require(len(bank.parent)==24 and bank.sample_index.shape==(3,6)
        and torch.equal(bank.query_column,torch.tensor([3,4,5])) and bank.log_donor_allele_frequency.shape==(2,2,6)
        and not bool(bank.donor_call_mask[:,1].any()) and not bool(bank.donor_call_mask[:,3:].any())
        and model.site_mlp[0].in_features==15 and model.leaf_projection[0].in_features==9
        and model.classifier[-1].out_features==3,'Generic C3 feature/mask/whole-node contract')
    span=np.asarray([3.,4.,3.]);previous=np.asarray([0.,3.5,3.5]);following=np.asarray([3.5,3.5,0.])
    geometry=torch.tensor(np.stack((np.log1p(span),np.log1p(previous),np.log1p(following)),-1)/math.log(2000001.),dtype=torch.float32)
    block=AmericanC3BlockV1(bank,parent_id='handmade_CPU_only',start_tree=0,stop_tree=3,sequence_length_bp=10.,
        complete_parent=True,genomic_geometry=geometry,focal_rows=torch.arange(3),focal_hold=torch.arange(3),
        focal_sample_column=torch.tensor([3,4,5]))
    model.train();logits=model.forward_complete(block)
    objective=logits.square().mean();objective.backward()
    gradients=[p.grad for p in model.parameters()]
    require(logits.shape==(3,6,3) and torch.isfinite(objective)
        and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),'Finite full-parent FP32 gradients through all learned modules')
    require(any(bool((g!=0).any()) for g in gradients),'Nonzero differentiability check')
    model.zero_grad(set_to_none=True);model.eval()
    with torch.inference_mode():
        context=model.encode_bank(bank)
        full_f,_=model.scan_context(context,geometry,direction='forward')
        full_r,_=model.scan_context(context,geometry,direction='reverse')
        for name in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'):
            getattr(ordinary,name).copy_(getattr(model.forward_gru,name))
            getattr(ordinary,name+'_reverse').copy_(getattr(model.reverse_gru,name))
        sequence=torch.cat((context,geometry[:,None].expand(-1,6,-1)),-1)
        standard,_=ordinary(sequence)
        require(torch.allclose(standard,torch.cat((full_f,full_r),-1),atol=atol,rtol=1e-5),
            'Two-direction modules must match ordinary whole-sequence bidirectional GRU equations')
        # Re-encode whole trees separately, retaining unary vertices and empty hold.
        contexts=[]
        for j in range(3):
            keep=(positions>=intervals[j,0])&(positions<intervals[j,1])
            one,_=encode_block([tree],intervals[j:j+1],genotype[keep],alleles[keep],positions[keep],
                reference,ages,num_classes=3,query_columns=queries)
            contexts.append(model.encode_bank(one))
        block_context=torch.cat(contexts)
        require(torch.allclose(context,block_context,atol=atol,rtol=1e-5),'Independent complete-tree encoding must not depend on block grouping')
        differences=[]
        for bounds in [[(0,1),(1,3)],[(0,1),(1,2),(2,3)]]:
            pieces=[];hidden=None
            for a,b in bounds:
                out,hidden=model.scan_context(block_context[a:b],geometry[a:b],direction='forward',hidden=hidden);pieces.append(out)
            split_f=torch.cat(pieces);pieces=[];hidden=None
            for a,b in reversed(bounds):
                out,hidden=model.scan_context(block_context[a:b],geometry[a:b],direction='reverse',hidden=hidden);pieces.insert(0,out)
            split_r=torch.cat(pieces)
            require(torch.allclose(full_f,split_f,atol=atol,rtol=1e-5)
                and torch.allclose(full_r,split_r,atol=atol,rtol=1e-5),'Non-reset state carry across2/3 blocks')
            dense=model.classify_context(context,full_f,full_r)
            streamed=model.classify_context(block_context,split_f,split_r)
            require(torch.allclose(dense,streamed,atol=atol,rtol=1e-5),'Identical complete/focal context join within ordinary FP32 tolerance')
            differences.append(dict(blocks=len(bounds),max_forward_difference=float((full_f-split_f).abs().max()),
                max_reverse_difference=float((full_r-split_r).abs().max()),max_logit_difference=float((dense-streamed).abs().max())))
    return dict(status='passed',kind='one_handmade_FP32_CPU_source_qualifier',seed=seed,atol=atol,
        features=dict(classes=3,donor_channels=6,site_width=15,observed_leaf_width=5,
            masked_reference_class_remains_nonquery=True,older_absent_cohort_masked=True),
        trees=3,nodes=24,haplotypes=6,sites=2,unary_nodes_retained=True,empty_hold_retained=True,
        full_parent_gradients_finite=True,learned_parameters_with_gradients=len(gradients),
        ordinary_bidirectional_max_difference=float((standard-torch.cat((full_f,full_r),-1)).abs().max()),
        complete_vs_block_tree_context_max_difference=float((context-block_context).abs().max()),
        directional_block_results=differences,no_ancestry_targets=True,scientific_or_public_data=False,
        GPU=False,model_accuracy_claim=False,scale_qualification=False)
