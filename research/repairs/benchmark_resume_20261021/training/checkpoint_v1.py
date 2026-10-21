"""Atomic complete-update checkpoints with exact schedule and random states."""
import hashlib
import json
import os
import random
import numpy as np
import torch
from research.repairs.benchmark_recovery_20261020.training.candidate_common import ROOT, need, file_ref
from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
from arggnn.benchmarking.treeloom_training import _optimizer_counts


def schedule_identity(schedule):
    return hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest()


def random_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=state[0],keys=state[1].tolist(),
        position=state[2],has_gauss=state[3],cached_gaussian=state[4]),
        torch_CPU=torch.get_rng_state(),
        torch_CUDA=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_random_state(saved):
    random.setstate(saved['python'])
    state=saved['numpy']
    np.random.set_state((state['name'],np.asarray(state['keys'],dtype=np.uint32),
        state['position'],state['has_gauss'],state['cached_gaussian']))
    torch.set_rng_state(saved['torch_CPU'])
    if saved['torch_CUDA']:
        need(torch.cuda.is_initialized() and len(saved['torch_CUDA']) == 1,
             'One admitted logical CUDA device for random-state restoration')
        torch.cuda.set_rng_state_all(saved['torch_CUDA'])


def payload(model,optimizer,*,steps,schedule,registry_ref,barrier_ref,settings,checkpoints,resume_ref):
    need(1 <= steps <= 600 and len(_optimizer_counts(optimizer)) == 49
         and set(_optimizer_counts(optimizer)) == {steps}, 'Checkpoint only complete exact optimizer updates')
    return dict(schema='treeloom_fresh_full_checkpoint_v1',epoch=schedule[steps-1][0],
        complete_parent_updates=steps,model_seed=70001,parent_registry_ref=registry_ref,
        observation_barrier_ref=barrier_ref,settings=settings,contract=model.checkpoint_contract(),
        model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
        VAL_truth_read=False,TEST_truth_read=False,checkpoint_selection='fixed_final_epoch30',
        resume_schema='treeloom_exact_complete_update_resume_v1',schedule_identity=schedule_identity(schedule),
        last_completed_parent=schedule[steps-1][1],next_schedule_index=steps,
        checkpoint_event_refs=dict(checkpoints),random_state=random_state(),resumed_from_checkpoint_ref=resume_ref)


def write_checkpoint(path,value,*,replace=False):
    temporary=path.with_name(path.name+'.next') if replace else path
    need(not path.is_symlink() and not temporary.exists(), 'Ordinary fresh checkpoint staging path')
    with temporary.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    need(temporary.stat().st_size <= 8*2**20, 'Bounded full optimizer checkpoint')
    if replace:
        os.replace(temporary,path)
    return file_ref(path)


def load_resume(ref,*,model,optimizer,schedule,registry_ref,barrier_ref,settings):
    need(file_ref(ref['path']) == ref, 'Exact immutable resume checkpoint reference')
    saved=torch.load(ROOT/ref['path'],map_location='cpu',weights_only=True)
    steps=saved.get('complete_parent_updates')
    need(type(steps) is int and 1 <= steps <= 600
         and saved.get('schema') == 'treeloom_fresh_full_checkpoint_v1'
         and saved.get('resume_schema') == 'treeloom_exact_complete_update_resume_v1'
         and saved.get('epoch') == schedule[steps-1][0]
         and saved.get('last_completed_parent') == schedule[steps-1][1]
         and saved.get('next_schedule_index') == steps
         and saved.get('schedule_identity') == schedule_identity(schedule)
         and saved.get('parent_registry_ref') == registry_ref
         and saved.get('observation_barrier_ref') == barrier_ref
         and saved.get('settings') == settings
         and saved.get('contract') == model.checkpoint_contract()
         and saved.get('model_seed') == 70001
         and saved.get('VAL_truth_read') is False and saved.get('TEST_truth_read') is False,
         'Exact original full-parent schedule and scientific checkpoint identity required')
    actual,expected=saved['model_state'],model.state_dict()
    need(set(actual) == set(expected) and all(value.dtype == expected[name].dtype == torch.float32
        and value.shape == expected[name].shape and bool(torch.isfinite(value).all())
        for name,value in actual.items()), 'Finite complete original FP32 model state')
    state=saved['optimizer_state'];group=state['param_groups'][0]
    need(len(state['param_groups']) == 1 and len(state['state']) == 49
         and len(group['params']) == 49 and set(state['state']) == set(group['params'])
         and group['lr'] == .001 and group['weight_decay'] == .0001
         and tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8
         and group['maximize'] is False and group['amsgrad'] is False
         and all(int(row['step'].item()) == steps for row in state['state'].values()),
         'All 49 original AdamW states at exactly the completed update')
    for key,param in zip(group['params'],model.parameters()):
        for name in ('exp_avg','exp_avg_sq'):
            value=state['state'][key][name]
            need(value.shape == param.shape and value.dtype == torch.float32
                 and bool(torch.isfinite(value).all()), 'Exact finite AdamW moment shape/dtype')
    model.load_state_dict(actual,strict=True)
    optimizer.load_state_dict(state)
    need(set(_optimizer_counts(optimizer)) == {steps}, 'Actual restored optimizer counters')
    checkpoints=saved['checkpoint_event_refs']
    expected_epochs={str(e) for e in (5,10,15,20,25,30) if e*20 < steps}
    need(expected_epochs <= set(checkpoints), 'Every earlier scheduled checkpoint remains referenced')
    for prior in checkpoints.values():
        need(file_ref(prior['path']) == prior, 'Retained scheduled checkpoint must authenticate')
    checkpoints=dict(checkpoints)
    if steps % 100 == 0:
        checkpoints.setdefault(str(saved['epoch']),ref)
    restore_random_state(saved['random_state'])
    return steps,checkpoints
