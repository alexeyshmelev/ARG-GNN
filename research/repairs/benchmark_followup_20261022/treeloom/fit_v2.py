"""Original 600-update TreeLoom fit with upfront RAM admission and exact resume.

The queue owns physical GPU5 admission, title, guards and genuine process waits.
Static original numerical modules remain unchanged. No gates or child launchers.
The 32 GiB allocator cap is provisional until a genuine full backward succeeds;
the queue must require at least 40 GiB free on physical GPU5 before admission.
"""
from collections import Counter
from pathlib import Path
import gc
import json
import os
import subprocess
import time
from research.repairs.benchmark_recovery_20261020.training.candidate_common import (
    ROOT,GIB,need,read_ref,file_ref,write_json,rss_bytes,scope_bytes,
)
from research.repairs.benchmark_recovery_20261020.training.candidate_worker import metadata_bindings
from research.repairs.benchmark_recovery_20261020.training.candidate_fit import create_pool
from research.repairs.benchmark_resume_20261021.training import checkpoint_v1 as checkpoints_io

MODULE='research.repairs.benchmark_followup_20261022.treeloom.fit_v2'
GPU_UUID='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
JOB=dict(path='.runtime/benchmarking/workflow_20261019_v8/jobs/AM_treeloom_fit.json',
    sha256='187e3177a9ccf587e1038ad4f477d6beaccb5128333977083bf06bd5084fe4a3')
BARRIER=dict(path='experiments/results/benchmark_campaign_20261018_workflow/release_v8/american/shared/AM_treeloom_fit/native/observation_barrier.json',
    sha256='4d68b86e133fc6ff547fbe2742557e844f18a27433e3a0f22f16846427ac0fc8')
PREVIOUS_MANIFEST=dict(path='research/repairs/benchmark_recovery_20261020/training/candidate_fit_source_manifest_v1.json',
    sha256='f70e550407886f740e9fffe8eb29c4ab145dcf55b800a45d7a55d73094cf4873')


def source_refs():
    refs=read_ref(PREVIOUS_MANIFEST)['source_refs']
    for ref in refs:
        need(file_ref(ref['path']) == ref, 'Unchanged qualified numerical and RAM-pool source')
    return refs+[file_ref(__file__),file_ref(checkpoints_io.__file__)]


def fit_factory(*,preparation_refs,generation_refs,output_directory,max_rss_bytes,qualification_ref,
                resume_checkpoint_ref=None):
    from arggnn.benchmarking import american_workflow as american
    job=read_ref(JOB);barrier=read_ref(BARRIER)
    fresh,original,registry,plans=metadata_bindings(job,barrier)
    inherited=job['call']['spec']
    qualification=read_ref(qualification_ref)
    first='argformer_american_train_000'
    need(qualification_ref == inherited['qualification_ref']
         and qualification.get('schema') == 'one_full_AMtrain000_exact_compact_CPU_qualification_v2'
         and qualification.get('status') == 'passed_complete_parent_exact_preparation_and_two_reuses'
         and qualification.get('parent_id') == first
         and qualification.get('observed_input_refs') == {key:original['train_inputs'][first][key] for key in fresh.INPUT_REFS}
         and qualification.get('candidate_source_refs') == inherited['candidate_source_refs']
         and qualification.get('private_truth_opened') is False
         and qualification.get('models_executed') is False and qualification.get('CUDA_called') is False
         and qualification.get('full_parent_preparation_count') == 1 and qualification.get('exact_restores') == 2
         and qualification.get('all_original_tensor_and_metadata_signatures_equal') is True
         and qualification.get('original_complete_parent_retained') is True,
         'Retain the genuine original full-parent qualification and observed identities')
    for ref in qualification['candidate_source_refs']:
        need(file_ref(ref['path']) == ref, 'Historical qualification source must remain unchanged')
    parents=set(registry.ids('train'))
    need(set(preparation_refs) == set(generation_refs) == parents
         and max_rss_bytes == 512*GIB, 'All 20 original parents and upfront 512 GiB admission')
    for parent in parents:
        need(american.train_binding(parent,preparation_refs[parent],generation_refs[parent])
             == original['train_inputs'][parent], 'Exact original observed and target references')
    out=ROOT/output_directory
    need(out.resolve() == out and out.is_relative_to(ROOT/'experiments/results') and not out.exists(),
         'Fresh output scope required')
    if resume_checkpoint_ref is not None:
        need(file_ref(resume_checkpoint_ref['path']) == resume_checkpoint_ref, 'Exact completed checkpoint')
    spec=dict(schema='treeloom_upfront_RAM_resume_fit_v2',job_ref=JOB,observation_barrier_ref=BARRIER,
        output_directory=str(out.relative_to(ROOT)),max_rss_bytes=max_rss_bytes,
        observation_cache_bytes=352*GIB,training_headroom_bytes=128*GIB,
        max_cuda_reserved_bytes=32*GIB,max_output_bytes=64*2**20,physical_gpu=5,gpu_uuid=GPU_UUID,
        resume_checkpoint_ref=resume_checkpoint_ref,qualification_ref=qualification_ref,source_refs=source_refs())
    return dict(module=MODULE,function='run_fit',spec=spec,python='.runtime/env/bin/python',
        source_refs=spec['source_refs'])


def enable_gpu(torch,spec,output,resource_check):
    import setproctitle
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') == '5'
         and os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID
         and spec['gpu_uuid'] == GPU_UUID and spec['physical_gpu'] == 5
         and setproctitle.getproctitle() == 'AI Auto Research Pipeline Test',
         'Admitted physical GPU5 UUID and prescribed title required before CUDA')
    need(not torch.cuda.is_initialized(), 'No CUDA before explicit physical GPU5 proof')
    command=['nvidia-smi','-i','5','--query-gpu=uuid','--format=csv,noheader,nounits']
    result=subprocess.run(command,check=True,text=True,capture_output=True,timeout=15)
    need(result.stdout.strip() == GPU_UUID, 'Actual physical GPU5 UUID must match')
    resource_check('TreeLoom_before_GPU5_CUDA')
    need(int(os.environ.get('ARGGNN_MAX_CUDA_RESERVED_BYTES','0')) == spec['max_cuda_reserved_bytes'],
         'Queue and fit must share the provisional 32 GiB allocator cap')
    need(torch.cuda.device_count() == 1, 'Exactly one admitted visible device')
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(spec['max_cuda_reserved_bytes']/total,0)
    torch.cuda.reset_peak_memory_stats(0)
    write_json(output/'gpu_binding.json',dict(physical_gpu=5,gpu_uuid=GPU_UUID,logical_device='cuda:0',
        process_title=setproctitle.getproctitle(),max_cuda_reserved_bytes=spec['max_cuda_reserved_bytes']))


def run_fit(spec,*,resource_check):
    import torch
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.benchmarking.treeloom_training import _optimizer_counts,_target_binding
    from arggnn.benchmarking.treeloom_prediction import validate_saved_checkpoint
    from research.repairs.benchmark_recovery_20261020.runtime.binding import ReadableTreeLoom
    from research.repairs.benchmark_recovery_20261020.training.candidate_gru import install_chunked_gru
    from research.repairs.benchmark_recovery_20261020.training.progress import StageProgress
    from research.repairs.benchmark_recovery_20261020.training.step import install_execution
    need(spec['schema'] == 'treeloom_upfront_RAM_resume_fit_v2' and spec['job_ref'] == JOB
         and spec['observation_barrier_ref'] == BARRIER and spec['max_rss_bytes'] == 512*GIB
         and spec['observation_cache_bytes'] == 352*GIB and spec['training_headroom_bytes'] == 128*GIB
         and spec['max_cuda_reserved_bytes'] == 32*GIB and spec['max_output_bytes'] == 64*2**20
         and spec['source_refs'] == source_refs(), 'Exact source-pinned upfront admission specification')
    need(callable(resource_check), 'Actual queue resource guard required')
    output=ROOT/spec['output_directory']
    need(output.resolve() == output and output.is_relative_to(ROOT/'experiments/results')
         and not output.exists(), 'Fresh fit output scope')
    output.mkdir();(output/'checkpoints').mkdir()
    def check(label):
        resource_check(label)
        need(rss_bytes() <= spec['max_rss_bytes'], 'Complete fit RSS cap exceeded')
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved(0) <= spec['max_cuda_reserved_bytes'], 'Fit CUDA allocator cap exceeded')
    check('TreeLoom_upfront_full_RAM_admission')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fresh,original,registry,plans=metadata_bindings(read_ref(JOB),read_ref(BARRIER))
    barrier=read_ref(BARRIER)
    runtime=ReadableTreeLoom(registry)
    install_execution(runtime,status_path=output/'training_live.json',holds_per_block=128,haplotypes_per_gru=32)
    parents=tuple(registry.ids('train'));schedule=[]
    need(len(parents) == len(set(parents)) == 20, 'Exactly 20 original TRAIN parents')
    for epoch in range(1,31):
        order=tuple(registry.epoch_order(epoch))
        need(len(order) == 20 and set(order) == set(parents), 'Every parent exactly once per original epoch')
        schedule.extend((epoch,parent) for parent in order)
    write_json(output/'schedule.json',dict(schedule=schedule,settings=original['settings'],
        observation_barrier_ref=BARRIER,resume_checkpoint_ref=spec['resume_checkpoint_ref'],
        max_rss_bytes=512*GIB,observation_disk_cache=False))
    model=install_chunked_gru(runtime.build_model(),chunk_holds=1024)
    enable_gpu(torch,spec,output,check)
    model=model.to(device='cuda:0',dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
        betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False,
        capturable=False,differentiable=False)
    start,checkpoints=0,{}
    if spec['resume_checkpoint_ref'] is not None:
        start,checkpoints=checkpoints_io.load_resume(spec['resume_checkpoint_ref'],model=model,optimizer=optimizer,
            schedule=schedule,registry_ref=registry.reference,barrier_ref=BARRIER,settings=original['settings'])
    need(start or not optimizer.state, 'Fresh optimizer required when no completed checkpoint exists')
    write_json(output/'resume_state.json',dict(completed_updates_restored=start,next_optimizer_update=start+1,
        original_budget=600,updates_remaining=600-start,completed_updates_replayed=0,
        resume_checkpoint_ref=spec['resume_checkpoint_ref']))
    active_profile=[None]
    def event(name,value):
        check('TreeLoom_progress')
        raw=(json.dumps(dict(event=name,**value),sort_keys=True,allow_nan=False)+'\n').encode()
        path=output/'progress.jsonl'
        need(len(raw) <= 16384 and (path.stat().st_size if path.exists() else 0)+len(raw) <= 8*2**20,
             'Bounded progress metadata')
        with path.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    def tick(label):
        check(label)
        if active_profile[0] is not None:
            active_profile[0].mark('complete_parent_work',preparation_stage=label)
    pool=create_pool(spec,original,registry,fresh,None,event,tick)
    def resolve_truth(parent):
        need(registry.split_for(parent) == 'train', 'TRAIN truth only')
        return original['train_inputs'][parent]['truth_ref']
    joiner=fresh.FreshTrainTargetJoin(registry,resolve_truth,train_inputs=original['train_inputs'],
        max_truth_uncompressed_bytes=original['max_truth_uncompressed_bytes'])
    def prepare(parent):
        item=original['train_inputs'][parent]
        config=dict(native_preparation_ref=item['native_preparation_ref'],class_order=list(fresh.CLASSES),
            expected_haplotypes=1600,reference_counts=[200,200,200],
            input_reference_class_mask=None,input_reference_sample_indices=None)
        state=load_observation(item['observation_ref'],item['focal_registry_ref'],config)
        bundle,metadata=runtime.prepare_observation(state,split='train',resource_check=tick,
            **{key:item[key] for key in fresh.INPUT_REFS},**original['preparation_limits'])
        del state
        return bundle,metadata
    for index in range(start,600):
        epoch,parent=schedule[index];steps=index+1
        check('TreeLoom_before_complete_parent')
        profile=StageProgress(output/'preparation_live.json',parent_id=parent)
        active_profile[0]=profile
        try:
            profile.mark('parent_load_begin',epoch=epoch,next_optimizer_update=steps);profile.flush()
            with profile.stage('prepare_once_or_restore_complete_parent'):
                bundle,metadata=pool.load(parent,lambda:prepare(parent))
        finally:
            active_profile[0]=None;profile.close()
        seal=barrier['parents'][parent]
        need(bundle.parent_id == parent and bundle.split == 'train' and bundle.sequence_length_bp == 30000000
             and metadata.get('parent_id') == parent and metadata.get('declared_split') == 'train'
             and metadata.get('parent_registry_ref') == registry.reference
             and metadata.get('all_native_trees_nodes_sites_samples_retained') is True
             and metadata.get('private_truth_opened') is False
             and bundle.focal_identity_sha256 == seal['focal_identity_sha256']
             and bundle.native_holds == plans[parent]['native_holds']
             and bundle.original_sites == plans[parent]['original_sites']
             and all(getattr(bundle,key) == seal[key] for key in fresh.INPUT_REFS), 'Original complete observed parent')
        bundle.validate()
        target=joiner.join(parent,split='train',plan=plans[parent],observation_seal=seal)
        _target_binding(target,parent,bundle,registry,original['train_inputs'])
        started=time.monotonic()
        result=runtime.train_step(model,bundle,target,optimizer,metadata,resource_check=check)
        torch.cuda.synchronize()
        need(result['one_complete_parent_optimizer_update'] is True and result['native_focals'] == 10000
             and result['parent_id'] == parent and len(_optimizer_counts(optimizer)) == 49
             and set(_optimizer_counts(optimizer)) == {steps}, 'One original full-parent update at the exact schedule position')
        value=checkpoints_io.payload(model,optimizer,steps=steps,schedule=schedule,
            registry_ref=registry.reference,barrier_ref=BARRIER,settings=original['settings'],
            checkpoints=checkpoints,resume_ref=spec['resume_checkpoint_ref'])
        if steps % 20 == 0 and epoch in (5,10,15,20,25,30):
            checkpoints[str(epoch)]=checkpoints_io.write_checkpoint(output/'checkpoints'/('epoch_%02d.pt'%epoch),value)
            value['checkpoint_event_refs']=dict(checkpoints)
        latest=checkpoints_io.write_checkpoint(output/'checkpoints'/'latest.pt',value,replace=True)
        write_json(output/'latest_checkpoint.json',dict(checkpoint_ref=latest,complete_parent_updates=steps,
            epoch=epoch,parent_id=parent,completed_updates_replayed=0),replace=True)
        event('complete_parent_update',dict(epoch=epoch,optimizer_updates=steps,
            elapsed_seconds=time.monotonic()-started,checkpoint_ref=latest,
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),**result))
        del bundle,metadata,target,value
        gc.collect();torch.cuda.empty_cache()
        need(scope_bytes(output) <= spec['max_output_bytes'], 'Bounded complete fit output scope')
    expected_counts=Counter(parent for _,parent in schedule[start:])
    need(pool.preparations == {parent:1 for parent in expected_counts}
         and pool.restorations == {parent:count-1 for parent,count in expected_counts.items() if count>1},
         'Each needed observed parent prepares once with exact remaining reuse count')
    # A resume directly from a scheduled checkpoint may lack its self-reference.
    if start == 600 and '30' not in checkpoints:
        checkpoints['30']=spec['resume_checkpoint_ref']
    need(set(checkpoints) == {'5','10','15','20','25','30'}, 'All scheduled checkpoints remain available')
    completion=dict(schema='treeloom_fresh_full_training_completion_v1',
        status='completed_full20parents30passes600updates',parent_registry_ref=registry.reference,
        observation_barrier_ref=BARRIER,completed_epochs=30,complete_parent_updates=600,
        checkpoint_ref=checkpoints['30'],checkpoint_event_refs=checkpoints,selected_epoch=30,
        checkpoint_selection='fixed_final_epoch30',final_saved_optimizer_parameter_states=49,
        final_saved_optimizer_steps=[600],VAL_truth_read=False,TEST_truth_read=False,ancestry_accuracy=None,
        resumed_from_updates=start,updates_executed_this_process=600-start,completed_updates_replayed=0,
        compact_pool_receipt=pool.receipt(),observation_disk_cache=False,max_rss_bytes=512*GIB,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    saved=torch.load(ROOT/checkpoints['30']['path'],map_location='cpu',weights_only=True)
    validate_saved_checkpoint(saved,model,registry,completion)
    ref=write_json(output/'training_completed.json',completion)
    return dict(completion,completion_ref=ref)
