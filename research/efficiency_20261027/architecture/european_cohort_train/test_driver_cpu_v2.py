"""Metadata-only exact schedule and source gate checks; no observations or CUDA."""
import copy
import hashlib
import json
from pathlib import Path
import random
import torch
from . import driver_v2 as driver


def main():
    source_ref=driver.verify_sources()
    assert driver.TOLERANCES['logits']==driver.TOLERANCES['loss']==dict(atol=1e-6,rtol=1e-5)
    assert driver.TOLERANCES['gradients']==driver.TOLERANCES['parameters']==driver.TOLERANCES['Adam']==dict(atol=1e-6,rtol=1e-4)
    job=driver.read_small(driver.ORIGINAL_JOB_REF)['call']['spec']['scientific_spec']
    plan=driver.original._parent_plan(job['sampling_plan_ref'],job['protocol_id'])
    parents=tuple(plan['train_parent_ids'])
    counts=[0]*49;seen=[];events=[];checkpoints=[]
    rng_before=random.getstate()
    def step(parent,targets,metadata):
        seen.append(parent)
        for index in range(49):counts[index]+=1
        return dict(parent_id=parent,one_complete_parent_optimizer_update=True)
    def checkpoint(epoch,updates):
        checkpoints.append((epoch,updates));return dict(epoch=epoch,updates=updates)
    result=driver.original.fixed_training_loop(parents=parents,load_parent=lambda parent:(parent,{}),
        join_targets=lambda parent,bundle,metadata:None,step=step,optimizer_counts=lambda:counts,
        checkpoint=checkpoint,check=lambda label:None,event=events.append)
    expected=[]
    for epoch in range(1,31):
        order=list(parents);random.Random(101+epoch).shuffle(order);expected.extend(order)
    assert seen==expected and len(seen)==90 and counts==[90]*49
    assert checkpoints==[(5,15),(10,30),(15,45),(20,60),(25,75),(30,90)]
    assert result['selected_epoch']==30 and result['optimizer_updates']==90
    assert random.getstate()==rng_before
    missing=dict(path='research/efficiency_20261027/architecture/european_cohort_train/MISSING_REVIEW.json',sha256='0'*64)
    out=driver.RUN_ROOT/'metadata_test_never_created'/'data'
    spec=driver.build_spec(mode='qualification',output_directory=str(out),physical_gpu=job['physical_gpu'],
        gpu_uuid=job['gpu_uuid'],source_review_ref=missing)
    assert spec['independent_validation_required_for_fixed_TRAIN'] is False
    assert spec['scientific_spec']['settings']==job['settings']
    assert spec['scientific_spec']['train_inputs']==job['train_inputs']
    blocked=False
    try:driver.validate(spec)
    except ValueError:blocked=True
    assert blocked and not out.exists() and not torch.cuda.is_initialized()
    altered=copy.deepcopy(spec);altered['scientific_spec']['settings']['epochs']=1
    rejected=False
    try:driver.validate(altered)
    except ValueError:rejected=True
    assert rejected
    report=dict(schema='European_cohort8_TRAIN_metadata_CPU_v2',passed=True,source_manifest_ref=source_ref,
        updates=90,parameter_counters=49,each_parent_passes=30,original_seeded_order_exact=True,
        checkpoints=checkpoints,missing_source_review_fails_closed=True,altered_science_fails_closed=True,
        output_directory_created=False,scientific_arrays_read=False,GPU_initialized=False,
        global_python_RNG_preserved=True,full_parent_qualification=False)
    with Path(__file__).with_name('CPU_driver_v2.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
