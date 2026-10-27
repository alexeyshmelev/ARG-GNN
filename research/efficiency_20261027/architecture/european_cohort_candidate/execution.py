"""Explicit cohort-aware successor execution; original complete context retained.

No launcher, truth reader, checkpoint selection or TEST prediction. ROOT owns
resource admission and supervised execution. Historical modules stay untouched.
"""
from types import SimpleNamespace
import torch
from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_execution as original
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_observations as observed
from .model import EuropeanCohortTreeLoom, CONDITION, PANEL_BINDING, PARAMETERS, need
from .observations import CohortParent


class CohortStagedExecution(original.EuropeanStagedExecutionV1):
    def __init__(self, model, bundle, metadata, **kwargs):
        need(type(model) is EuropeanCohortTreeLoom and type(bundle) is CohortParent,
             'Explicit successor model and all-panel cohort bundle')
        bundle.validate()
        need(model.scientific_condition == metadata.get('scientific_condition') == CONDITION,
             'Distinct scientific condition on model and prepared observations')
        self.cohort_parent = bundle
        # The original executor already exposes this typed module injection.
        # No monkeypatch, inherited global mutation or compatibility alias.
        module = SimpleNamespace(FrozenEuropeanTreeLoomV1=EuropeanCohortTreeLoom,
                                 _bank_tensors=observed.implementation._bank_tensors)
        super().__init__(model, bundle.original, metadata, frozen_module=module, **kwargs)
        self.cohort_device = bundle.observed_cohort.to(self.device)

    def _base(self, bank, site_leaf):
        return self.model.base_from_sites(bank, site_leaf, self.cohort_device)


def binding(model, bundle, metadata, resource_check):
    need(type(model) is EuropeanCohortTreeLoom and type(bundle) is CohortParent
         and model.scientific_condition == bundle.scientific_condition == CONDITION
         and model.condition == bundle.original.original.condition == PANEL_BINDING
         and callable(resource_check), 'Exact new condition and unchanged physical panel')
    gpu_binding.validate_env()
    parameter = next(model.parameters())
    need(parameter.device.type == 'cuda' and parameter.device.index in (None, 0)
         and all(p.dtype == torch.float32 for p in model.parameters()),
         'Supervised assigned physical GPU and FP32 required for real data')
    need(metadata.get('scientific_condition') == CONDITION
         and metadata.get('parent_id') == bundle.parent_id
         and metadata.get('sampled_registry_ref') == bundle.sampled_registry_ref
         and metadata.get('focal_identity_sha256') == bundle.focal_identity_sha256
         and metadata.get('observed_cohort_manifest_ref') == bundle.sample_manifest_ref
         and metadata.get('private_truth_opened') is False,
         'Exact immutable observation, cohort and registered focal support binding')
    bundle.validate()
    resource_check('European_cohort_complete_parent')


def optimizer_step_counts(optimizer, parameters, *, allow_empty):
    counts=[]
    for p in parameters:
        state=optimizer.state.get(p,{})
        if not state:
            need(allow_empty, 'Every parameter must have AdamW state after its update')
            counts.append(0)
            continue
        need(set(state) == {'step','exp_avg','exp_avg_sq'}
             and state['exp_avg'].shape == state['exp_avg_sq'].shape == p.shape
             and bool(torch.isfinite(state['exp_avg']).all())
             and bool(torch.isfinite(state['exp_avg_sq']).all()), 'Finite exact AdamW moments')
        step=float(state['step'])
        need(step >= 0 and step.is_integer(), 'Integral AdamW update counter')
        counts.append(int(step))
    need(len(set(counts)) == 1, 'All49 tensors have one common update count')
    return counts[0]


def train_sampled_step(model, bundle, targets, optimizer, metadata, *, resource_check):
    binding(model,bundle,metadata,resource_check)
    from arggnn.benchmarking.targets import SampledFocalTargets
    need(bundle.split == 'train' and isinstance(targets,SampledFocalTargets)
         and targets.class_order == tuple(model.output_classes)
         and type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
         'Original native-six TRAIN supervision and AdamW')
    targets.validate_binding(parent_id=bundle.parent_id,split='train',sampled_registry_ref=bundle.sampled_registry_ref)
    rows=torch.tensor(targets.readout_rows,dtype=torch.long)
    columns=torch.tensor(targets.query_columns,dtype=torch.long)
    need(torch.equal(rows,bundle.focal_hold)
         and bool(((columns>=0)&(columns<bundle.bank.query_column.numel())).all())
         and torch.equal(bundle.bank.query_column[columns],bundle.focal_sample_column),
         'Every original focal row and repeated draw retained')
    group=optimizer.param_groups[0];parameters=list(model.parameters())
    need(len(parameters)==49 and sum(p.numel() for p in parameters)==PARAMETERS
         and group['lr']==.001 and group['weight_decay']==.0001
         and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
         and group.get('maximize',False) is False and group.get('amsgrad',False) is False
         and len(group['params'])==len(parameters)
         and {id(p) for p in group['params']}=={id(p) for p in parameters},
         'Original optimizer settings with all successor parameters')
    before=optimizer_step_counts(optimizer,parameters,allow_empty=True)
    model.train();optimizer.zero_grad(set_to_none=True)
    executor=CohortStagedExecution(model,bundle,metadata,resource_check=resource_check)
    with torch.autocast(device_type='cuda',enabled=False):
        context=executor.context_forward()
        loss,gradient=original.decode_parent_backward(model,context,original.geometry_from_spans(bundle.bank.spans),
            bundle.focal_hold,bundle.focal_sample_column,torch.tensor(targets.labels,dtype=torch.long),
            tile_haplotypes=executor.hap_tile,resource_check=resource_check)
        resource_check('European_cohort_complete_context_backward')
        context.backward(gradient)
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
         'All49 parameters have connected finite gradients')
    cohort_grad=float(model.leaf_projection[0].weight.grad[:,38:].norm())
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    resource_check('European_cohort_complete_parent_optimizer')
    optimizer.step()
    need(all(bool(torch.isfinite(p).all()) for p in parameters), 'All updated parameters finite')
    after=optimizer_step_counts(optimizer,parameters,allow_empty=False)
    need(after==before+1, 'Exactly one complete-parent update')
    return dict(scientific_condition=CONDITION,parent_id=bundle.parent_id,sampled_records=len(targets.labels),
        loss=float(loss.detach()),gradient_norm_before_clip=float(norm),cohort_weight_gradient_norm=cohort_grad,
        optimizer_step_before=before,optimizer_step_after=after,finite_parameter_tensors=49,
        one_complete_parent_optimizer_update=True,private_truth_opened_by_observation_or_executor=False,
        all_native_trees_nodes_sites_samples_retained=True,detached_history=False,artificial_block_boundaries=False)


def predict_validation(model,bundle,metadata,*,resource_check):
    binding(model,bundle,metadata,resource_check)
    need(bundle.split=='val_prediction', 'Only independently registered VAL observations')
    model.eval();device=next(model.parameters()).device
    with torch.no_grad():
        executor=CohortStagedExecution(model,bundle,metadata,resource_check=resource_check)
        context=executor.context_forward();geometry=original.geometry_from_spans(bundle.bank.spans).to(device)
        output=torch.empty((bundle.focal_hold.numel(),6),dtype=torch.float32)
        seen=0
        for first in range(0,context.shape[1],executor.hap_tile):
            last=min(first+executor.hap_tile,context.shape[1])
            rows=torch.nonzero((bundle.focal_sample_column>=first)&(bundle.focal_sample_column<last),as_tuple=False).flatten()
            if not len(rows):continue
            resource_check('European_cohort_full_sequence_validation_tile')
            logits=original._tile_logits(model,context[:,first:last].to(device).contiguous(),geometry,
                bundle.focal_hold[rows].to(device),(bundle.focal_sample_column[rows]-first).to(device))
            output.index_copy_(0,rows,logits.cpu());seen+=len(rows)
        need(seen==len(output) and bool(torch.isfinite(output).all()), 'All independent VAL rows predicted')
        return output
