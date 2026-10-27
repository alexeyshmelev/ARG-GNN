"""Selected-full release metadata and real inert specification contracts."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
from arggnn.benchmarking import contracts as c
from . import entrypoint_v3 as entry
from . import training_v3 as training
from .qualification_gate_v2 import RUNNER_REF,validate_full_qualification
from .configuration_v1 import configuration

BASE=c.ROOT/'research/efficiency_20261027/integration_candidate'
JOB='.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt5.json'


def fails(call):
    try:call()
    except (ValueError,KeyError):return
    raise AssertionError('Expected strict release rejection')


def selected(block):
    return configuration(dict(executor_kind='explicit_vjp',CPU_threads=8,holds_per_block=block,
        haplotypes_per_gru=64,history_holds_per_window=1024,context_residency='CPU',
        layout_cache_max_bytes=80*2**30,layout_cache_policy='actual_RSS_evict_unleased'))


def fixture(block):
    categories={name:dict(passed=True,tensors=count) for name,count in
        (('logits',1),('loss',1),('gradients',49),('parameters',49),('moments',147))}
    comparison=dict(passed=True,argmax_disagreements=0,categories=categories)
    records=[dict(kind=kind,phase='full_first_call',repeat=0,order_position=index,
        CPU_threads=cpu,holds_per_block=size,comparison=copy.deepcopy(comparison))
        for index,(kind,cpu,size) in enumerate((('original',1,128),('explicit_cpu_vjp',8,block)))]
    records.append(dict(records[0],phase='full_warm_repeat',repeat=1,order_position=2))
    return dict(schema='TreeLoom_selected_large_VJP_compare_v1',status='completed',FP32=True,
        source_refs=[RUNNER_REF],CPU_threads_by_kind=dict(original=1,explicit_cpu_vjp=8),
        execution_partitions=dict(original_holds_per_block=128,selected_holds_per_block=block,
            haplotypes_per_gru=64,history_window=1024),records=records,
        scope=dict(mode='full',full_parent_claim=True,native_holds=77472,
            supervised_original_focals=10000,haplotypes=1600,heldout_truth_opened=False))


def main():
    assert not torch.cuda.is_initialized()
    specs=[]
    old=c.read_json(JOB)['call']['spec']['arguments']
    for block in (128,512,1024):
        output='experiments/results/paper_american_20261024/treeloom_fit/native/inert_v3_spec_'+str(block)
        spec=training.build_spec(**dict(old,output_directory=output,execution_configuration=selected(block)))
        training._validate_spec(spec)
        assert spec['settings']==training.SETTINGS and spec['requirements']['CPU_Torch_threads']==8
        assert spec['requirements']['GNN_holds_per_block']==block and not (c.ROOT/output).exists()
        specs.append(dict(block=block,source_ref_count=len(spec['requirements']['source_refs']),output_created=False))
    valid=fixture(512)
    validate_full_qualification(valid,selected(512))
    rejected=[]
    def reject(name,change):
        value=copy.deepcopy(valid)
        change(value)
        fails(lambda:validate_full_qualification(value,selected(512)))
        rejected.append(name)
    reject('missing_original',lambda v:v.update(records=v['records'][1:2]))
    reject('misthreaded_original',lambda v:v['records'][0].update(CPU_threads=8))
    reject('first_not_original',lambda v:v['records'].reverse())
    reject('first_wrong_order',lambda v:v['records'][0].update(order_position=1))
    reject('first_wrong_phase',lambda v:v['records'][0].update(phase='slice_timed'))
    reject('first_wrong_repeat',lambda v:v['records'][0].update(repeat=1))
    reject('missing_source_refs',lambda v:v.update(source_refs=[]))
    reject('wrong_runner_digest',lambda v:v.update(source_refs=[dict(RUNNER_REF,sha256='0'*64)]))
    reject('wrong_original_thread_table',lambda v:v['CPU_threads_by_kind'].update(original=8))
    reject('wrong_selected_partition',lambda v:v['execution_partitions'].update(selected_holds_per_block=128))
    reject('wrong_GRU_partition',lambda v:v['execution_partitions'].update(haplotypes_per_gru=32))
    reject('wrong_history_partition',lambda v:v['execution_partitions'].update(history_window=128))
    reject('missing_selected',lambda v:v.update(records=[v['records'][0]]))
    reject('wrong_selected_row_partition',lambda v:v['records'][1].update(holds_per_block=128))
    reject('wrong_selected_row_threads',lambda v:v['records'][1].update(CPU_threads=1))
    reject('failed_comparison',lambda v:v['records'][1]['comparison'].update(passed=False))
    reject('argmax_disagreement',lambda v:v['records'][1]['comparison'].update(argmax_disagreements=1))
    for category,count in (('gradients',49),('parameters',49),('moments',147)):
        reject('incomplete_'+category,lambda v,k=category,n=count:v['records'][1]['comparison']['categories'][k].update(tensors=n-1))
    for category in ('logits','loss','gradients','parameters','moments'):
        reject('failed_'+category,lambda v,k=category:v['records'][1]['comparison']['categories'][k].update(passed=False))
    historical='research/efficiency_20261027/root/runs/explicit_full_gpu5_v1/data/completed.json'
    fails(lambda:validate_full_qualification(c.read_json(historical),selected(128)))
    pins=c.file_ref(BASE/'source_pins_v3.json')
    run='research/efficiency_20261027/integration_candidate/synthetic_run'
    refs={name:dict(path=path,sha256='0'*64) for name,path in (
        ('full',run+'/data/completed.json'),('wait',run+'/actual_wait.json'),
        ('launch',run+'/launch.json'),('review','synthetic_review'),('release','synthetic_release'))}
    wait=dict(pid=1234,process_start_ticks='987',actual_wait=True,owned_group_absent=True,actual_returncode=0)
    launch=dict(pid=1234,process_start_ticks='987')
    review=dict(status='pass',candidate_source_pins_ref=pins)
    release=dict(schema='TreeLoom_execution_successor_release_v1',status='independently_approved',
        configuration=selected(512),source_pins_ref=pins,full_parent_qualification_ref=refs['full'],
        independent_integration_review_ref=refs['review'],full_parent_actual_wait_ref=refs['wait'],
        full_parent_launch_ref=refs['launch'])
    values=dict(full=valid,wait=wait,launch=launch,review=review,release=release)
    def read(reference):
        for key,ref in refs.items():
            if ref==reference:return copy.deepcopy(values[key])
        raise AssertionError('Unexpected synthetic metadata read')
    with patch.object(entry,'read',side_effect=read):
        entry.validate_release(refs['release'])
        values['full']=c.read_json(historical)
        fails(lambda:entry.validate_release(refs['release']))
        values['full']=valid
        review['status']='pending'
        fails(lambda:entry.validate_release(refs['release']))
        review['status']='pass'
        wait['actual_wait']=False
        fails(lambda:entry.validate_release(refs['release']))
        wait['actual_wait']=True
        release['configuration']=selected(128)
        fails(lambda:entry.validate_release(refs['release']))
    receipt=dict(status='passed',CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),
        source_pins_ref=pins,test_ref=c.file_ref(Path(__file__).resolve()),
        actual_runner_ref=RUNNER_REF,actual_old_receipt_rejected=c.file_ref(historical),
        valid_synthetic_selected_full_accepted=True,synthetic_receipt_is_not_numerical_evidence=True,
        original_warm_repeat_may_have_nonzero_order=True,rejected_cases=rejected,
        original_core_settings_and_memory_preserved=True,real_inert_specs=specs,
        entrypoint_gate_plumbing_passed=True,production_modified=False,jobs_started=0,
        launchable_release_created=False)
    with (BASE/'spec_release_contracts_v3.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
