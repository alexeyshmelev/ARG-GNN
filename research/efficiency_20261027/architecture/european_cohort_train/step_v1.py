"""One original full-parent native6 sampled update with typed executor selection."""
import torch
from arggnn.benchmarking.targets import SampledFocalTargets
from research.efficiency_20261027.architecture.european_cohort_candidate import execution as candidate
from research.efficiency_20261027.architecture.european_cohort_candidate.model import CONDITION, PARAMETERS, need
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_execution as original
from .boundary_v1 import EuropeanBoundaryExecution


def train_sampled_step(model,bundle,targets,optimizer,metadata,*,resource_check,
                       executor_type=EuropeanBoundaryExecution,capture=False,context_compare=None):
    candidate.binding(model,bundle,metadata,resource_check)
    need(executor_type in (EuropeanBoundaryExecution,candidate.CohortStagedExecution), 'Two explicit reviewed European executors')
    need(bundle.split=='train' and isinstance(targets,SampledFocalTargets)
         and targets.class_order==tuple(model.output_classes) and type(optimizer) is torch.optim.AdamW
         and len(optimizer.param_groups)==1, 'Original sampled native6 TRAIN and AdamW')
    targets.validate_binding(parent_id=bundle.parent_id,split='train',sampled_registry_ref=bundle.sampled_registry_ref)
    rows=torch.tensor(targets.readout_rows,dtype=torch.long)
    columns=torch.tensor(targets.query_columns,dtype=torch.long)
    need(torch.equal(rows,bundle.focal_hold)
         and bool(((columns>=0)&(columns<bundle.bank.query_column.numel())).all())
         and torch.equal(bundle.bank.query_column[columns],bundle.focal_sample_column),'Exact repeated sampled rows')
    parameters=list(model.parameters());group=optimizer.param_groups[0]
    need(len(parameters)==49 and sum(p.numel() for p in parameters)==PARAMETERS
         and group['lr']==.001 and group['weight_decay']==.0001 and tuple(group['betas'])==(.9,.999)
         and group['eps']==1e-8 and group.get('maximize',False) is False and group.get('amsgrad',False) is False
         and group.get('foreach') is False and group.get('fused') is False
         and len(group['params'])==49 and {id(p) for p in group['params']}=={id(p) for p in parameters},
         'All49 parameters and original explicit FP32 AdamW settings')
    before=candidate.optimizer_step_counts(optimizer,parameters,allow_empty=True)
    model.train();optimizer.zero_grad(set_to_none=True)
    executor=executor_type(model,bundle,metadata,resource_check=resource_check)
    with torch.autocast(device_type='cuda',enabled=False):
        context=executor.context_forward()
        if context_compare is not None:context_compare(context)
        geometry=original.geometry_from_spans(bundle.bank.spans)
        loss,gradient=original.decode_parent_backward(model,context,geometry,bundle.focal_hold,
            bundle.focal_sample_column,torch.tensor(targets.labels,dtype=torch.long),
            tile_haplotypes=executor.hap_tile,resource_check=resource_check)
        captured=None
        if capture:
            # A small all-row prediction receipt; no extra saved context gradient.
            logits=torch.empty((len(targets.labels),6),dtype=torch.float32)
            with torch.no_grad():
                for first in range(0,952,executor.hap_tile):
                    last=min(first+executor.hap_tile,952)
                    selected=torch.nonzero((bundle.focal_sample_column>=first)&(bundle.focal_sample_column<last),as_tuple=False).flatten()
                    if not len(selected):continue
                    resource_check('European_qualification_all_row_logits')
                    value=original._tile_logits(model,context[:,first:last].to(executor.device).contiguous(),
                        geometry.to(executor.device),bundle.focal_hold[selected].to(executor.device),
                        (bundle.focal_sample_column[selected]-first).to(executor.device))
                    logits[selected]=value.cpu()
            captured=dict(context=context.detach(),logits=logits)
        resource_check('European_cohort_complete_context_backward')
        context.backward(gradient)
        del gradient
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), 'All49 finite connected full-model gradients')
    cohort_gradient=model.leaf_projection[0].weight.grad[:,38:]
    cohort_channel_norms=cohort_gradient.norm(dim=0).detach().cpu().tolist()
    if capture:captured['gradients']={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    resource_check('European_complete_parent_optimizer')
    optimizer.step()
    need(all(bool(torch.isfinite(p).all()) for p in parameters),'Finite full-parent optimizer result')
    after=candidate.optimizer_step_counts(optimizer,parameters,allow_empty=False)
    need(after==before+1,'Exactly one original complete-parent update')
    receipt=dict(scientific_condition=CONDITION,parent_id=bundle.parent_id,sampled_records=len(targets.labels),
        loss=float(loss.detach()),gradient_norm_before_clip=float(norm),cohort_channel_gradient_norms=cohort_channel_norms,
        optimizer_step_before=before,optimizer_step_after=after,finite_parameter_tensors=49,
        one_complete_parent_optimizer_update=True,all_native_trees_nodes_sites_samples_retained=True,
        private_truth_opened_by_observation_or_executor=False,detached_history=False,artificial_block_boundaries=False,
        executor=executor_type.__name__,explicit_vjp=getattr(executor,'explicit_receipt',None))
    if capture:
        from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
        captured['parameters']={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        captured['optimizer']=_cpu_snapshot(optimizer.state_dict())
        return receipt,captured
    return receipt
