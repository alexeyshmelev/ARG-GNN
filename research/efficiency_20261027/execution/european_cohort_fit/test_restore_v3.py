"""Real CPU AdamW interrupted/serialized/restored equivalence on all49 tensors.

Uses the actual32647-parameter cohort model and stochastic synthetic gradients;
no native trees, targets, predictions, CUDA or scientific training is executed.
"""
import copy
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from research.efficiency_20261027.architecture.european_cohort_train import driver_v3 as driver
from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot

HERE=Path(__file__).parent
PARENTS=('TRAIN000','TRAIN001','TRAIN002')

class Interrupted(Exception):pass


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b);return
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for left,right in zip(a,b):same(left,right)
    else:assert a==b


def model_optimizer():
    model=driver.v2.build_model()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
        maximize=False,amsgrad=False,foreach=False,fused=False)
    return model,optimizer


def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)


def callbacks(model,optimizer,trace,checkpoints,box,stop=None):
    def step(parent,targets,metadata):
        optimizer.zero_grad(set_to_none=True)
        noise=random.random()+float(np.random.random())+float(torch.rand(()))
        scale=(PARENTS.index(parent)+1)*.01+noise*.001
        loss=sum(((parameter*(index+1)*.0001+scale)**2).sum() for index,parameter in enumerate(model.parameters()))
        loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters()),1.,error_if_nonfinite=True);optimizer.step()
        return dict(parent_id=parent,one_complete_parent_optimizer_update=True,loss=float(loss.detach()))
    def event(value):trace.append(copy.deepcopy(value));return dict(path='event/'+str(value['optimizer_updates']),sha256='a'*64)
    def checkpoint(epoch,updates):
        ref=dict(path='checkpoint/'+str(epoch),sha256='b'*64);checkpoints[str(epoch)]=ref;return ref
    def save_resume(epoch,updates,refs,events):
        value=dict(model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
            rng_state=driver.capture_rng(cuda=False),cursor=driver.cursor(PARENTS,updates),checkpoint_event_refs=dict(refs),
            update_event_refs=list(events),optimizer_updates=updates,epoch=epoch)
        buffer=io.BytesIO();torch.save(value,buffer);data=buffer.getvalue()
        assert len(data)<=8*2**20
        box.update(data=data,bytes=len(data),updates=updates)
        if updates==stop:raise Interrupted()
        return dict(path='resume/'+str(updates),sha256='c'*64)
    return dict(parents=PARENTS,load_parent=lambda parent:(parent,{}),join_targets=lambda *args:None,step=step,
        optimizer_counts=lambda:driver.memory._optimizer_counts(optimizer),checkpoint=checkpoint,
        check=lambda stage:None,event=event,save_resume=save_resume)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    seed(211);baseline,opt=model_optimizer();trace=[];epochs={};box={}
    original=callbacks(baseline,opt,trace,epochs,box)
    original.pop('save_resume')
    result=driver.v2.original.fixed_training_loop(**original)
    expected_model=_cpu_snapshot(baseline.state_dict());expected_optimizer=_cpu_snapshot(opt.state_dict())
    expected_rng=driver.capture_rng(cuda=False)
    seed(211);partial,partial_opt=model_optimizer();partial_trace=[];partial_epochs={};partial_box={}
    interrupted=callbacks(partial,partial_opt,partial_trace,partial_epochs,partial_box,stop=17)
    try:driver.fixed_training_loop(**interrupted)
    except Interrupted:pass
    else:raise AssertionError('Actual simulated interruption required')
    assert partial_box['updates']==17 and len(partial_trace)==17
    saved=torch.load(io.BytesIO(partial_box['data']),map_location='cpu',weights_only=True)
    assert saved['cursor']==driver.cursor(PARENTS,17)
    seed(991);resumed,resumed_opt=model_optimizer()
    driver.restore_states(resumed,resumed_opt,saved,17);driver.restore_rng(saved['rng_state'],cuda=False)
    continued=callbacks(resumed,resumed_opt,partial_trace,partial_epochs,partial_box)
    resumed_result=driver.fixed_training_loop(**continued,start_update=17,
        checkpoint_refs=saved['checkpoint_event_refs'],update_refs=saved['update_event_refs'])
    same(trace,partial_trace);same(expected_model,_cpu_snapshot(resumed.state_dict()))
    same(expected_optimizer,_cpu_snapshot(resumed_opt.state_dict()));same(expected_rng,driver.capture_rng(cuda=False))
    assert result['checkpoint_event_refs']==resumed_result['checkpoint_event_refs'] and resumed_result['optimizer_updates']==90
    saved90=torch.load(io.BytesIO(partial_box['data']),map_location='cpu',weights_only=True)
    prior_count=len(partial_trace)
    terminal=driver.fixed_training_loop(**continued,start_update=90,
        checkpoint_refs=saved90['checkpoint_event_refs'],update_refs=saved90['update_event_refs'])
    assert len(partial_trace)==prior_count==90 and terminal['resume_checkpoint_ref'] is None
    # Mutated counters, model shapes, optimizer settings and budget must reject.
    rejections=[]
    for name,mutate in [('counter',lambda item:item['optimizer_state']['state'][0].update(step=torch.tensor(16.))),
        ('shape',lambda item:item['model_state'].update({next(iter(item['model_state'])):torch.zeros(1)})),
        ('optimizer',lambda item:item['optimizer_state']['param_groups'][0].update(lr=.01))]:
        item=copy.deepcopy(saved);mutate(item)
        try:driver.restore_states(resumed,resumed_opt,item,17)
        except ValueError:rejections.append(name)
        else:raise AssertionError('Restore mutation accepted: '+name)
    for count in (-1,91):
        try:driver.cursor(PARENTS,count)
        except ValueError:rejections.append('budget'+str(count))
        else:raise AssertionError('Bad budget accepted')
    assert all(driver.cursor(PARENTS,index)['next_parent_id']==driver.schedule(PARENTS)[index][1] for index in range(90))
    assert driver.cursor(PARENTS,90)['next_parent_id'] is None
    assert not torch.cuda.is_initialized()
    evidence=dict(schema='European_cohort8_resume_CPU_equivalence_v3',passed=True,
        baseline='Original fixed_training_loop;actual cohort model49tensors32647parameters;originalAdamW;synthetic stochastic gradient',
        interruption_after_updates=17,resumed_updates=73,final_updates=90,all49_model_tensors_bitwise_equal=True,
        all147_Adam_tensors_bitwise_equal=True,complete90_update_trace_bitwise_equal=True,
        python_numpy_torchCPU_RNG_equal=True,six_epoch_checkpoint_schedule_equal=True,terminal90_restore_runs_no_update=True,
        all90_next_parent_cursors_verified=True,checkpoint_serialization_bytes=partial_box['bytes'],mutation_rejections=rejections,
        CUDA_RNG_execution_tested=False,CUDA_initialized=False,scientific_arrays_opened=False,native_or_GPU_launch=False,
        source_refs=[driver.v2.contracts.file_ref(HERE/'test_restore_v3.py'),driver.v2.contracts.file_ref(Path(driver.__file__))])
    path=HERE/'RESTORE_EQUIVALENCE_CPU_v3.json';path.write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(passed=True,evidence=driver.v2.contracts.file_ref(path),checkpoint_bytes=partial_box['bytes'])))

if __name__=='__main__':main()
