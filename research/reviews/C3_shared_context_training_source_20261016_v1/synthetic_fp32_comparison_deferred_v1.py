"""Deferred single CPU FP32 synthetic comparison. Default does not import Torch.

This is one two-tree/four-haplotype comparison, not a training campaign.
No data paths, guard, admission, optimizer step or GPU execution is used.
"""
from pathlib import Path
import argparse,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/C3_shared_context_training_source_20261016_v1"

def _helper():
    p=ROOT/HERE/"shared_context_step_v1.py"
    spec=importlib.util.spec_from_file_location("_C3_shared_context_fixture_step",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def _synthetic_blocks(torch):
    from arggnn.mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1
    from arggnn.american_c3_gnn_dataset_isolated_v2 import AmericanC3BlockV1
    blocks=[]
    for index in range(2):
        f=lambda value:torch.tensor(value,dtype=torch.float32)
        l=lambda value:torch.tensor(value,dtype=torch.long)
        support=f([[.2,.3,.4,2.,0.]]*6+[[0.,0.,0.,0.,1.]])
        donor=f([[[.6]*6,[.4]*6],[[.3]*6,[.7]*6]]).log()
        bank=MutationRoutedWholeTreeInputsV1(
            parent=l([4,4,5,5,6,6,-1]),node_time=f([0,0,0,0,2,3,5]),
            sample_index=l([[0,1,2,3]]),
            upward_levels=(l([0,1,2,3]),l([4,5]),l([6])),
            upward_children=(l([]),l([0,1,2,3]),l([4,5])),
            upward_parent_rank=(l([]),l([0,0,1,1]),l([0,0])),
            downward_levels=(l([4,5]),l([0,1,2,3])),
            edge_geometry=f([[.1,.2,1.,.3]]*6+[[0.,0.,0.,0.]]),
            branch_support_true=support,branch_support_sham=support.clone(),
            genotype=torch.tensor([[0,0,1,1],[1,index,1,0]],dtype=torch.int8),
            log_donor_allele_frequency=donor,donor_call_mask=torch.ones((2,6),dtype=torch.bool),
            panel_allele_frequency=f([[.5,.5],[.5,.5]]),site_entropy=f([.69,.69]),
            site_hold=l([0,0]),spans=f([1.+index]),readout_hold=l([0]),
            query_column=l([3]),reference_class=l([0,1,2,-1]),sample_age=f([0,0,0,0]))
        blocks.append(AmericanC3BlockV1(bank,parent_id="synthetic_only",block_index=index,
            start_tree=index,stop_tree=index+1,sequence_length_bp=3.,complete_parent=False,
            genomic_geometry=f([[.1+index*.1,.2,.3]]),focal_rows=l([index]),focal_hold=l([0]),
            focal_sample_column=l([3])))
    return blocks

def _original_reference(model,blocks,labels,torch):
    """Original isolated step equations, replacing only fixed H/N with tiny H4/N2.

    Matches original lines191–228: duplicate encoder per direction, original
    checkpointed GRU carries/classifier and one mean loss.backward; no update.
    """
    from torch.utils.checkpoint import checkpoint
    from torch.nn import functional as F
    dummy=torch.zeros((),dtype=torch.float32,requires_grad=True);n=labels.numel()
    def scan(direction):
        hidden=torch.zeros((1,4,32),dtype=torch.float32)
        rows=[];histories=[];locals_=[]
        order=range(2) if direction=="forward" else range(1,-1,-1)
        for index in order:
            row=blocks[index].focal_rows.clone()
            def evaluate(_dummy,state,physical=index,scan_direction=direction):
                block=blocks[physical].clone()
                context=model.encode_bank(block.to_model_inputs())
                history,next_state=model.scan_context(context,block.genomic_geometry,
                    direction=scan_direction,hidden=state)
                local=context[block.focal_hold,block.focal_sample_column] if scan_direction=="forward" else context.new_empty((0,32))
                return next_state,local,history[block.focal_hold,block.focal_sample_column]
            hidden,local,history=checkpoint(evaluate,dummy,hidden,use_reentrant=False,preserve_rng_state=True)
            rows.append(row);histories.append(history)
            if direction=="forward":locals_.append(local)
        permutation=torch.argsort(torch.cat(rows))
        return (torch.cat(locals_).index_select(0,permutation) if locals_ else None,
            torch.cat(histories).index_select(0,permutation))
    local,forward=scan("forward");_,reverse=scan("reverse")
    losses=[];logits=[]
    for start in range(0,n,1):
        def classify(_dummy,x,f,r,y):
            return F.cross_entropy(model.classify_context(x,f,r),y,reduction="sum")
        x,f,r=local[start:start+1],forward[start:start+1],reverse[start:start+1]
        losses.append(checkpoint(classify,dummy,x,f,r,labels[start:start+1],
            use_reentrant=False,preserve_rng_state=True))
        with torch.no_grad():logits.append(model.classify_context(x,f,r))
    loss=torch.stack(losses).sum()/n;loss.backward()
    return loss.detach(),torch.cat(logits)

def run_once_v1(*,atol=5e-5,rtol=5e-4):
    """Root-deferred, one fixed ordinary-FP32 comparison; no retries/tolerance search."""
    import torch
    from arggnn.american_c3_gnn_model_v1 import AmericanC3WholeTreeGNNV1
    helper=_helper()
    if torch.get_default_dtype()!=torch.float32 or torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Fixture requires ordinary CPU FP32, no autocast")
    prior_rng=torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(7000101)
        reference=AmericanC3WholeTreeGNNV1(hidden_dim=32,expected_haplotypes=4,donor_cohorts=2,
            site_chunk_size=256,checkpoint_sites=True).cpu().train()
        shared=AmericanC3WholeTreeGNNV1(hidden_dim=32,expected_haplotypes=4,donor_cohorts=2,
            site_chunk_size=256,checkpoint_sites=True).cpu().train()
        shared.load_state_dict(reference.state_dict(),strict=True)
        helper.require_supported_model_v1(reference,torch.device("cpu"))
        helper.require_supported_model_v1(shared,torch.device("cpu"))
        blocks=_synthetic_blocks(torch);labels=torch.tensor([0,2],dtype=torch.long)
        plan=dict(trees=2,blocks=[dict(start_tree=i,stop_tree=i+1,nodes=7,sites=2) for i in range(2)])
        forecast=helper.memory_forecast_v1(plan,haplotypes=4,hidden_dim=32,focals=2)
        caps={key:forecast[key] for key in helper.CAP_KEYS}
        reference_loss,reference_logits=_original_reference(reference,blocks,labels,torch)
        result=helper._two_phase_backward_v1(shared,lambda i:blocks[i],plan,labels,device=torch.device("cpu"),
            resource_check=lambda event:None,focal_chunk_size=1,memory_caps=caps,capture_logits=True)
        torch.testing.assert_close(result["logits"],reference_logits,atol=atol,rtol=rtol)
        torch.testing.assert_close(result["loss"],reference_loss,atol=atol,rtol=rtol)
        reference_parameters=dict(reference.named_parameters());errors={}
        for name,param in shared.named_parameters():
            original=reference_parameters[name]
            if param.grad is None or original.grad is None:raise AssertionError("Missing gradient: "+name)
            if param.grad.dtype!=torch.float32 or original.grad.dtype!=torch.float32:raise AssertionError("Non-FP32 gradient")
            torch.testing.assert_close(param.grad,original.grad,atol=atol,rtol=rtol,msg=lambda message:name+": "+message)
            errors[name]=float((param.grad-original.grad).abs().max())
        if result["counts"]["encoder_value_forwards"]!=2 or result["counts"]["encoder_VJP_forwards"]!=2:
            raise AssertionError("Expected exactly2B encoder entries")
        return dict(schema="C3_shared_context_single_synthetic_FP32_comparison_v1",passed=True,device="cpu",
            seed=7000101,trees=2,physical_blocks=2,haplotypes=4,focals=2,hidden_dim=32,optimizer_steps=0,
            atol=atol,rtol=rtol,logits_max_abs_error=float((result["logits"]-reference_logits).abs().max()),
            loss_abs_error=float((result["loss"]-reference_loss).abs()),
            parameter_gradient_max_abs_error=max(errors.values()),per_parameter_gradient_max_abs_errors=errors,
            source_counts=result["counts"],no_full_parent_or_data_or_F64_or_GPU=True,
            reference="original_isolated_step_equations_with_only_H4_N2_cardinality_substitution",
            bitwise_equivalence_claimed=False,full_training_memory_runtime_or_scientific_success_not_claimed=True)
    finally:torch.random.set_rng_state(prior_rng)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-once",action="store_true");args=p.parse_args(argv)
    if not args.run_once:
        print(json.dumps(dict(status="source_only_synthetic_fixture_execution_deferred",Torch_imported=False)));return 0
    print(json.dumps(run_once_v1(),sort_keys=True),flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
