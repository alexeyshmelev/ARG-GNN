"""Authenticate a supplied completed AM checkpoint against the original context.

This CPU metadata preflight accepts any complete original update from 1 to 600.
The unchanged original load_resume validator later checks all model shapes,
AdamW moments and random states while restoring the admitted training process.
"""
from pathlib import Path
from research.repairs.benchmark_recovery_20261020.training.candidate_common import (
    ROOT, need, file_ref, read_ref,
)
from research.repairs.benchmark_recovery_20261020.training.candidate_worker import metadata_bindings
from research.repairs.benchmark_resume_20261021.training import checkpoint_v1 as checkpoints_io


def inspect_completed_checkpoint(ref, *, job_ref, barrier_ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'}
        and type(ref['path']) is str and type(ref['sha256']) is str and len(ref['sha256'])==64,
        'An explicit path and SHA256 completed checkpoint reference is required')
    path=Path(ref['path'])
    path=path if path.is_absolute() else ROOT/path
    need(path.resolve()==path and path.is_relative_to(ROOT/'experiments/results')
        and path.suffix=='.pt' and path.is_file() and not path.is_symlink()
        and 0<path.stat().st_size<=8*2**20 and file_ref(path)==ref,
        'Bounded ordinary completed checkpoint must authenticate before binding')
    import torch
    need(not torch.cuda.is_initialized(), 'Resume factory preflight must remain on CPU')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    fresh,original,registry,plans=metadata_bindings(read_ref(job_ref),read_ref(barrier_ref))
    schedule=[(epoch,parent) for epoch in range(1,31) for parent in registry.epoch_order(epoch)]
    need(len(schedule)==600, 'Original complete 30-pass 20-parent schedule required')
    steps=saved.get('complete_parent_updates')
    need(type(steps) is int and 1<=steps<=600
        and saved.get('schema')=='treeloom_fresh_full_checkpoint_v1'
        and saved.get('resume_schema')=='treeloom_exact_complete_update_resume_v1'
        and saved.get('epoch')==schedule[steps-1][0]
        and saved.get('last_completed_parent')==schedule[steps-1][1]
        and saved.get('next_schedule_index')==steps
        and saved.get('schedule_identity')==checkpoints_io.schedule_identity(schedule)
        and saved.get('parent_registry_ref')==registry.reference
        and saved.get('observation_barrier_ref')==barrier_ref
        and saved.get('settings')==original['settings']
        and saved.get('model_seed')==70001
        and saved.get('checkpoint_selection')=='fixed_final_epoch30'
        and saved.get('VAL_truth_read') is False and saved.get('TEST_truth_read') is False,
        'Complete checkpoint must retain original registry, barrier, settings and exact schedule position')
    optimizer=saved['optimizer_state'];groups=optimizer['param_groups'];states=optimizer['state']
    need(len(groups)==1 and len(states)==49 and len(groups[0]['params'])==49
        and len(set(groups[0]['params']))==49 and set(states)==set(groups[0]['params']),
        'All49 completed original AdamW states are required')
    group=groups[0]
    need(group['lr']==.001 and group['weight_decay']==.0001
        and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8
        and group['maximize'] is False and group['amsgrad'] is False,
        'Original AdamW settings are required')
    for state in states.values():
        step=state['step']
        need(torch.is_tensor(step) and step.device.type=='cpu' and step.numel()==1
            and bool(torch.isfinite(step).all()) and step.item()==steps,
            'Every saved AdamW counter must equal the completed update')
    for prior in saved['checkpoint_event_refs'].values():
        need(file_ref(prior['path'])==prior, 'Earlier scheduled checkpoints must remain authenticated')
    need(file_ref(path)==ref, 'Supplied checkpoint must remain unchanged during CPU validation')
    return dict(checkpoint_ref=ref,complete_parent_updates=steps,original_update_budget=600,
        updates_remaining=600-steps,next_optimizer_update=steps+1 if steps<600 else None,
        next_parent_id=schedule[steps][1] if steps<600 else None,optimizer_parameter_states=49,
        completed_updates_replayed=0,schedule_identity=checkpoints_io.schedule_identity(schedule),
        original_full_state_validator='research.repairs.benchmark_resume_20261021.training.checkpoint_v1.load_resume')
