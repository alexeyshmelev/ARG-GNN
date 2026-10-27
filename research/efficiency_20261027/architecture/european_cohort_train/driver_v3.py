"""Minimal source-bound continuation of the reviewed fixed90-update v2 fit.

The v2 numerical implementation, original three200Mb parents, seed101, optimizer,
30 epochs and six epoch checkpoints are unchanged. Each completed update also
gets an immutable RNG/optimizer/cursor checkpoint. ROOT launches and reaps.
"""
import copy
import io
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from . import driver_v2 as v2

HERE=Path(__file__).parent
memory=v2.memory
ORIGINAL_JOB_REF=v2.ORIGINAL_JOB_REF
read_small=v2.read_small
need=v2.need
RESUME_SCHEMA='European_cohort8_TRAIN_resume_checkpoint_v1'


def source_manifest_ref():return v2.contracts.file_ref(HERE/'source_pins_v3.json')


def verify_sources():
    ref=source_manifest_ref();manifest=read_small(ref)
    need(manifest.get('schema')=='European_cohort8_TRAIN_resume_sources_v3','Exact isolated resume source manifest')
    for entry in manifest['source_refs']:
        need(v2.contracts.file_ref(entry['path'])==entry,'Pinned resume source changed: '+entry['path'])
    v2.verify_sources()
    return ref


def build_spec(*,resume_checkpoint_ref=None,**kwargs):
    need(kwargs.get('mode')=='fit','v3 extends fixed FIT only;qualification remains v2')
    value=v2.build_spec(**kwargs)
    value.update(resume_driver_source_manifest_ref=source_manifest_ref(),resume_checkpoint_ref=resume_checkpoint_ref)
    return value


def validate(spec):
    base={key:value for key,value in spec.items() if key not in ('resume_driver_source_manifest_ref','resume_checkpoint_ref')}
    need(spec.get('resume_driver_source_manifest_ref')==verify_sources(),'Exact reviewed resume implementation')
    ref=spec.get('resume_checkpoint_ref')
    if ref is not None:
        path=v2.contracts.contained_path(ref['path'])
        need(path.is_relative_to(v2.RUN_ROOT) and path.parent.name=='resume'
             and path.parent.parent.name=='data' and path.parent.parent.parent.parent==v2.RUN_ROOT
             and path.stat().st_size<=8*2**20 and v2.contracts.file_ref(path)==ref,'Exact bounded immutable ROOT resume checkpoint')
    scientific,plan=v2.validate(base)
    return base,scientific,plan


def schedule(parents):
    need(len(parents)==3 and len(set(parents))==3,'Original three distinct TRAIN parents')
    result=[]
    for epoch in range(1,31):
        order=list(parents);random.Random(101+epoch).shuffle(order)
        result.extend((epoch,parent) for parent in order)
    return result


def cursor(parents,updates):
    need(type(updates) is int and 0<=updates<=90,'Fixed90 total continuation budget')
    order=schedule(parents)
    return dict(completed_updates=updates,next_epoch=None if updates==90 else order[updates][0],
        next_parent_id=None if updates==90 else order[updates][1],next_parent_offset=None if updates==90 else updates%3,
        fixed_total_updates=90)


def capture_rng(*,cuda=True):
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4])],
        torch_cpu=torch.random.get_rng_state().clone(),torch_cuda=None if not cuda else torch.cuda.get_rng_state(0).cpu().clone())


def restore_rng(value,*,cuda=True):
    need(set(value)=={'python','numpy','torch_cpu','torch_cuda'},'All declared RNG streams required')
    cpu=value['torch_cpu']
    need(torch.is_tensor(cpu) and cpu.device.type=='cpu' and cpu.dtype==torch.uint8 and cpu.ndim==1,'Exact CPU RNG byte tensor')
    state=value['numpy'];need(len(state)==5 and state[0]=='MT19937' and len(state[1])==624,'Exact NumPy RNG state')
    need((value['torch_cuda'] is not None)==cuda,'Exact single-device RNG restore mode')
    if cuda:
        device=value['torch_cuda'];need(torch.is_tensor(device) and device.device.type=='cpu' and device.dtype==torch.uint8 and device.ndim==1,'Exact CUDA RNG byte tensor')
    random.setstate(value['python'])
    np.random.set_state((state[0],np.array(state[1],dtype=np.uint32),state[2],state[3],state[4]))
    torch.random.set_rng_state(cpu)
    if cuda:torch.cuda.set_rng_state(value['torch_cuda'],0)


def restore_states(model,optimizer,saved,updates):
    state=saved['model_state'];wanted=model.state_dict()
    need(set(state)==set(wanted) and len(state)==49,'All49 exact model tensors')
    for name,value in state.items():
        need(torch.is_tensor(value) and value.device.type=='cpu' and value.dtype==torch.float32
             and value.shape==wanted[name].shape and bool(torch.isfinite(value).all()),'Finite exact model tensor '+name)
    old=saved['optimizer_state'];fresh=optimizer.state_dict()
    need(set(old)==set(fresh) and len(old['param_groups'])==1,'Exact AdamW state structure')
    group=old['param_groups'][0];expected=fresh['param_groups'][0]
    need(group==expected,'All original optimizer settings and parameter order unchanged')
    ids=group['params'];parameters=list(model.parameters());states=old['state']
    need(len(ids)==len(parameters)==len(states)==49 and set(ids)==set(states),'Every optimizer state present')
    for index,parameter in zip(ids,parameters):
        item=states[index];need(set(item)=={'step','exp_avg','exp_avg_sq'},'Exact plain Adam state')
        step=item['step'];need(torch.is_tensor(step) and step.numel()==1 and step.item()==updates,'Exact completed Adam counter')
        for key in ('exp_avg','exp_avg_sq'):
            value=item[key]
            need(torch.is_tensor(value) and value.device.type=='cpu' and value.dtype==torch.float32
                 and value.shape==parameter.shape and bool(torch.isfinite(value).all()),'Exact finite Adam moment')
        need(bool((item['exp_avg_sq']>=0).all()),'Nonnegative Adam second moment')
    model.load_state_dict(state,strict=True);optimizer.load_state_dict(old)


def fixed_training_loop(*,parents,load_parent,join_targets,step,optimizer_counts,checkpoint,check,event,save_resume,
                        start_update=0,checkpoint_refs=None,update_refs=None):
    order=schedule(parents);cursor(parents,start_update)
    refs=dict(checkpoint_refs or {});events=list(update_refs or [])
    need(len(events)==start_update and set(refs)=={str(epoch) for epoch in (5,10,15,20,25,30) if epoch*3<=start_update},
         'Complete immutable prior update and six-epoch checkpoint history')
    last_resume=None
    for index in range(start_update,90):
        epoch,parent=order[index]
        check('European_TreeLoom_before_complete_parent')
        bundle,metadata=load_parent(parent);targets=join_targets(parent,bundle,metadata)
        result=step(bundle,targets,metadata);updates=index+1;counts=optimizer_counts()
        need(len(counts)==49 and set(counts)=={updates} and result.get('parent_id')==parent
             and result.get('one_complete_parent_optimizer_update') is True,'One full-parent update and all49 actual Adam counters')
        events.append(event(dict(epoch=epoch,optimizer_updates=updates,**result)))
        del bundle,metadata,targets
        if updates%3==0 and epoch in v2.original.SETTINGS['checkpoint_epochs']:refs[str(epoch)]=checkpoint(epoch,updates)
        last_resume=save_resume(epoch,updates,refs,events)
    need(set(refs)=={'5','10','15','20','25','30'} and len(events)==90,'Complete fixed30-pass90-update history')
    return dict(optimizer_updates=90,completed_epochs=30,selected_epoch=30,checkpoint_ref=refs['30'],
        checkpoint_event_refs=refs,update_event_refs=events,resume_checkpoint_ref=last_resume)


def run_fit(spec,*,resource_check,acquire_gpu0):
    base,_,_=validate(spec)
    scientific,plan,check,writer,load,join,barrier_ref,seals,cache=v2.setup(base,resource_check,acquire_gpu0)
    model,optimizer=v2.fresh_model()
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    from arggnn.benchmarking.american_preparation import _Writer
    # Resume control artifacts are separately bounded under the same ROOT2GiB
    # whole-run guard. The scientific writer retains its original64MiB limit.
    resume_writer=_Writer(writer.path/'resume',768*2**20,check)
    resume_ref=spec['resume_checkpoint_ref'];start=0;previous={};events=[]
    def payload(epoch,updates):
        return dict(schema='European_cohort8_TRAIN_checkpoint_v1',scientific_condition=v2.CONDITION,
            source_manifest_ref=base['source_manifest_ref'],protocol_id=scientific['protocol_id'],
            sampling_plan_ref=scientific['sampling_plan_ref'],observation_barrier_ref=barrier_ref,
            epoch=epoch,optimizer_updates=updates,settings=copy.deepcopy(v2.original.SETTINGS),class_order=list(v2.CLASSES),
            model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
            checkpoint_selection='fixed_final_epoch30',TEST_truth_read=False)
    def save(target,name,value):
        stream=io.BytesIO();torch.save(value,stream);data=stream.getvalue()
        need(len(data)<=8*2**20,'Original8MiB per-checkpoint bound')
        path=target.prepare(name,len(data))
        with path.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        return target.finish(path,len(data))
    if resume_ref is not None:
        saved=torch.load(v2.contracts.contained_path(resume_ref['path']),map_location='cpu',weights_only=True)
        start=saved.get('optimizer_updates')
        need(saved.get('schema')==RESUME_SCHEMA and saved.get('scientific_condition')==v2.CONDITION
             and saved.get('source_manifest_ref')==base['source_manifest_ref']
             and saved.get('resume_driver_source_manifest_ref')==spec['resume_driver_source_manifest_ref']
             and saved.get('protocol_id')==scientific['protocol_id'] and saved.get('sampling_plan_ref')==scientific['sampling_plan_ref']
             and saved.get('settings')==v2.original.SETTINGS and saved.get('class_order')==list(v2.CLASSES)
             and saved.get('TEST_truth_read') is False and saved.get('checkpoint_selection')=='fixed_final_epoch30'
             and saved.get('cursor')==cursor(plan['train_parent_ids'],start) and 1<=start<=90
             and saved.get('epoch')==(start-1)//3+1,'Exact reviewed checkpoint,scientific protocol,and next-parent cursor')
        old_barrier=read_small(saved['observation_barrier_ref']);new_barrier=read_small(barrier_ref)
        need(old_barrier==new_barrier,'Every original observed parent and sampled registry revalidated before restore')
        previous=saved['checkpoint_event_refs'];events=saved['update_event_refs']
        for ref in [*previous.values(),*events]:need(v2.contracts.file_ref(ref['path'])==ref,'Immutable prior training evidence')
        restore_states(model,optimizer,saved,start)
        restore_rng(saved['rng_state'])
        writer.json('fit_resume_origin.json',dict(resume_checkpoint_ref=resume_ref,start_update=start,
            cursor=saved['cursor'],checkpoint_event_refs=previous,update_event_refs=events,
            source_manifest_ref=base['source_manifest_ref'],resume_driver_source_manifest_ref=spec['resume_driver_source_manifest_ref']))
        del saved
    def checkpoint(epoch,updates):return save(writer,'checkpoints/epoch_%02d.pt'%epoch,payload(epoch,updates))
    def record(value):return writer.json('updates/update_%03d.json'%value['optimizer_updates'],value)
    def save_resume(epoch,updates,refs,update_refs):
        value=payload(epoch,updates)
        value.update(schema=RESUME_SCHEMA,resume_driver_source_manifest_ref=spec['resume_driver_source_manifest_ref'],
            rng_state=capture_rng(),cursor=cursor(plan['train_parent_ids'],updates),
            checkpoint_event_refs=dict(refs),update_event_refs=list(update_refs),resumed_from_ref=resume_ref)
        return save(resume_writer,'update_%03d.pt'%updates,value)
    result=fixed_training_loop(parents=tuple(plan['train_parent_ids']),load_parent=load,
        join_targets=lambda parent,bundle,metadata:join(parent,bundle,metadata,model),
        step=lambda bundle,targets,metadata:v2.train_sampled_step(model,bundle,targets,optimizer,metadata,resource_check=check),
        optimizer_counts=lambda:memory._optimizer_counts(optimizer),checkpoint=checkpoint,check=check,event=record,
        save_resume=save_resume,start_update=start,checkpoint_refs=previous,update_refs=events)
    if result['resume_checkpoint_ref'] is None:result['resume_checkpoint_ref']=resume_ref
    saved=torch.load(v2.contracts.contained_path(result['checkpoint_ref']['path']),map_location='cpu',weights_only=True)
    need(saved['schema']=='European_cohort8_TRAIN_checkpoint_v1' and saved['scientific_condition']==v2.CONDITION
         and saved['source_manifest_ref']==base['source_manifest_ref'],'Distinct final cohort8 checkpoint binding')
    compatibility=dict(saved,schema='european_treeloom_full_checkpoint_v1')
    # A resumed-at90 run keeps its already sealed final checkpoint and barrier.
    v2.original._validate_saved(compatibility,v2.build_model(),protocol_id=scientific['protocol_id'],
        sampling_plan_ref=scientific['sampling_plan_ref'],updates=90,barrier_ref=saved['observation_barrier_ref'])
    torch.cuda.synchronize(0);check('European_cohort8_fit_complete')
    result.update(schema='European_cohort8_TRAIN_completion_v1',status='completed_fixed90updates',scientific_condition=v2.CONDITION,
        source_manifest_ref=base['source_manifest_ref'],observation_barrier_ref=barrier_ref,
        resume_driver_source_manifest_ref=spec['resume_driver_source_manifest_ref'],resumed_from_ref=resume_ref,
        resumed_after_updates=start,checkpoint_selection='fixed_final_epoch30',class_order=list(v2.CLASSES),TEST_truth_read=False,
        independent_validation_still_required_for_architecture_search_or_comparative_claims=True,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    ref=writer.json('training_completed.json',result)
    need(not writer.pending and not resume_writer.pending,'Every bounded artifact finished')
    return dict(result,completion_ref=ref)
