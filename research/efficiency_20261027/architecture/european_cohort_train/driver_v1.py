"""Frozen TRAIN-only cohort8 fit and separately supervised full-parent qualification.

ROOT owns launch, source-review receipts, live admission, process-family guards
and actual waits. No TEST prediction API, scheduler, resume, simulation or launch.
All90 fit updates start independently from seed101 after qualification release.
"""
from collections import OrderedDict
import copy
import gc
import io
import json
import os
import resource
from pathlib import Path
import time
import torch
from arggnn.benchmarking import contracts as contracts
from arggnn.benchmarking import treeloom_training as memory
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import common
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_stage as original
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_gru import install_chunked_gru
from research.repairs.benchmark_full_vram_20261022.training import full_vram_v2 as full_vram
from research.efficiency_20261027.architecture.european_cohort_candidate import observations
from research.efficiency_20261027.architecture.european_cohort_candidate.model import CONDITION, CLASSES, build_model, need
from research.efficiency_20261027.architecture.european_cohort_candidate.execution import CohortStagedExecution
from .boundary_v1 import EuropeanBoundaryExecution
from .step_v1 import train_sampled_step

HERE=Path(__file__).parent
ORIGINAL_JOB_REF=dict(path='.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json',
    sha256='fecd670f6f26332ab165648f4d65ad51131650d195e86df2a39a5545f2b724c1')
POLICY_REF=dict(path='research/efficiency_20261027/fidelity/EUROPEAN_COHORT8_TRAIN_VAL_GATE_RECOMMENDATION.json',
    sha256='277fd48fefaeb07a13f522b8b36cb3d266520a3e23b138d07d82e54998cda138')
RUN_ROOT=contracts.ROOT/'research/efficiency_20261027/root/runs'
TOLERANCE=dict(atol=1e-6,rtol=1e-4)


def read_small(ref):
    path=contracts.contained_path(ref['path'])
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*2**20
         and contracts.file_ref(path)==ref,'Exact bounded immutable metadata reference')
    return json.loads(path.read_text())


def source_manifest_ref():
    return contracts.file_ref(HERE/'source_pins_v1.json')


def verify_sources():
    ref=source_manifest_ref();manifest=read_small(ref)
    need(manifest['schema']=='European_cohort8_TRAIN_sources_v1','Exact source-manifest schema')
    for entry in manifest['source_refs']:
        need(contracts.file_ref(entry['path'])==entry,'Pinned numerical/driver source changed: '+entry['path'])
    read_small(POLICY_REF)
    return ref


def build_spec(*,mode,output_directory,physical_gpu,gpu_uuid,source_review_ref,
               qualification_ref=None,fit_release_ref=None):
    need(mode in ('qualification','fit'),'Only full TRAIN qualification or fixed TRAIN fit')
    scientific=copy.deepcopy(read_small(ORIGINAL_JOB_REF)['call']['spec']['scientific_spec'])
    output=contracts.contained_path(output_directory)
    need(output.is_relative_to(RUN_ROOT) and output.parent!=RUN_ROOT and not output.exists(),
         'Fresh ROOT-owned run output directory')
    scientific.update(output_directory=str(output.relative_to(contracts.ROOT)),physical_gpu=physical_gpu,gpu_uuid=gpu_uuid)
    return dict(schema='European_cohort8_TRAIN_driver_v1',mode=mode,scientific_condition=CONDITION,
        original_job_ref=ORIGINAL_JOB_REF,policy_ref=POLICY_REF,source_manifest_ref=source_manifest_ref(),
        scientific_spec=scientific,source_review_ref=source_review_ref,qualification_ref=qualification_ref,
        fit_release_ref=fit_release_ref,gpu_memory_policy=full_vram.POLICY,
        full_device_vram_permission=full_vram.permission(physical_gpu,gpu_uuid),
        independent_validation_required_for_fixed_TRAIN=False,TEST_prediction_or_tuning=False)


def validate(spec):
    scientific=spec['scientific_spec']
    expected=build_spec(mode=spec['mode'],output_directory=scientific['output_directory'],
        physical_gpu=scientific['physical_gpu'],gpu_uuid=scientific['gpu_uuid'],
        source_review_ref=spec['source_review_ref'],qualification_ref=spec['qualification_ref'],fit_release_ref=spec['fit_release_ref'])
    need(spec==expected and verify_sources()==spec['source_manifest_ref'],'Exact frozen driver/spec/source binding')
    review=read_small(spec['source_review_ref'])
    need(review.get('schema')=='European_cohort8_TRAIN_source_review_v1'
         and review.get('source_manifest_ref')==spec['source_manifest_ref'] and review.get('passed') is True
         and review.get('all_observation_barrier_retained') is True
         and review.get('fixed_TRAIN_independent_of_new_VAL') is True,'Independent exact driver/source and policy review required')
    plan=original._parent_plan(scientific['sampling_plan_ref'],scientific['protocol_id'])
    need(scientific['settings']==original.SETTINGS and scientific['optimizer_updates']==90
         and len(plan['train_parent_ids'])==3 and plan['training_total']==50000,'Original30pass90update schedule')
    if spec['mode']=='qualification':
        need(spec['qualification_ref'] is None and spec['fit_release_ref'] is None,'Qualification cannot masquerade as fit release')
    else:
        qualification=read_small(spec['qualification_ref']);release=read_small(spec['fit_release_ref'])
        need(qualification.get('schema')=='European_cohort8_full_parent_qualification_v1'
             and qualification.get('passed') is True and qualification.get('source_manifest_ref')==spec['source_manifest_ref']
             and qualification.get('scientific_condition')==CONDITION and qualification.get('all_original_rows_used') is True
             and qualification.get('qualified_parent')==plan['train_parent_ids'][0]
             and qualification.get('prepared_parents')==plan['train_parent_ids']
             and qualification.get('tolerance')==TOLERANCE and qualification.get('gradient_tensors')==49
             and qualification.get('parameter_tensors')==49 and qualification.get('Adam_tensors')==147
             and qualification.get('argmax_changes')==0
             and 0<qualification.get('worker_peak_RSS_bytes',0)<=scientific['memory_envelope']['max_rss_bytes'],'Source-bound complete200Mb TRAIN qualification required')
        need(release.get('schema')=='European_cohort8_TRAIN_release_v1' and release.get('passed') is True
             and release.get('source_manifest_ref')==spec['source_manifest_ref']
             and release.get('qualification_ref')==spec['qualification_ref']
             and release.get('source_review_ref')==spec['source_review_ref']
             and release.get('qualification_actual_wait_returncode')==0
             and release.get('qualification_process_family_absent') is True
             and release.get('live_process_family_host_GPU_storage_admission_required') is True,
             'ROOT exact qualification completion and reaped-family release required; live fit admission remains mandatory')
    return scientific,plan


def observed_loader(spec,cache,check):
    def load(parent):
        item=spec['train_inputs'][parent]
        def prepare():
            state=observations.load_relate_observation(item['inference_ref'],item['observed_vcf_ref'],
                max_sites=spec['max_sites'],max_metadata_bytes=spec['max_metadata_bytes'],resource_check=check)
            return observations.prepare_observation(state,parent_id=parent,split='train',source_refs=state['source_refs'],
                sampled_registry_ref=item['sampled_registry_ref'],resource_check=check,
                max_sampled_metadata_bytes=spec['max_metadata_bytes'],**spec['preparation_limits'])
        return memory._load_cached_parent(parent,cache,prepare,observation_cache_bytes=spec['observation_cache_bytes'],
            memory_envelope=spec['memory_envelope'],rss_bytes=memory._current_rss_bytes,
            tensor_bytes=lambda value:memory._cpu_bytes(value,torch),resource_check=check,event=lambda *args:None)
    return load


def setup(spec,resource_check,acquire_gpu0):
    scientific,plan=validate(spec)
    need(callable(resource_check) and callable(acquire_gpu0),'Actual supervised guards and late GPU acquisition')
    runtime,check,writer=common._cpu_setup(scientific,resource_check)
    cache,seals=OrderedDict(),{}
    load=observed_loader(scientific,cache,check)
    for parent in plan['train_parent_ids']:
        bundle,metadata=load(parent)
        seals[parent]=dict(parent_id=parent,**{key:scientific['train_inputs'][parent][key] for key in original.INPUTS},
            focal_identity_sha256=bundle.focal_identity_sha256,native_holds=bundle.native_holds,
            native_nodes=bundle.bank.parent.numel(),sites=bundle.bank.genotype.shape[0],samples=952,
            observed_cohort_manifest_ref=bundle.sample_manifest_ref,all_native_trees_nodes_sites_samples_retained=True,
            private_truth_opened=False)
        del bundle,metadata
    barrier_ref=writer.json('observation_barrier.json',dict(schema='European_cohort8_all_TRAIN_observation_barrier_v1',
        scientific_condition=CONDITION,sampling_plan_ref=scientific['sampling_plan_ref'],parents=seals,
        all_original_TRAIN_observations_sealed_before_targets=True,private_truth_opened=False))
    acquire_gpu0()
    full_vram.enable_gpu(runtime,dict(**scientific,gpu_memory_policy=spec['gpu_memory_policy'],
        full_device_vram_permission=spec['full_device_vram_permission']),writer.path,check)
    def join(parent,bundle,metadata,model):
        need(bundle.split=='train' and bundle.parent_id==parent
             and bundle.focal_identity_sha256==seals[parent]['focal_identity_sha256'], 'Exact observation seal before TRAIN targets')
        memory._admit_parent_headroom(cache,memory_envelope=scientific['memory_envelope'],stage='training',
            rss_bytes=memory._current_rss_bytes,resource_check=check,event=lambda *args:None,keep_parent=parent)
        return original._training_targets(scientific['train_inputs'][parent]['training_targets_ref'],parent,bundle,model,
                                          max_metadata_bytes=scientific['max_metadata_bytes'])
    return scientific,plan,check,writer,load,join,barrier_ref,seals,cache


def fresh_model():
    model=install_chunked_gru(build_model().to(device='cuda:0',dtype=torch.float32))
    need(bool((model.leaf_projection[0].weight[:,38:]==0).all()),'Fresh zero cohort extension')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8,
                               maximize=False,amsgrad=False,foreach=False,fused=False)
    need(not optimizer.state,'Independent fresh optimizer; no qualification checkpoint reuse')
    return model,optimizer


def run_fit(spec,*,resource_check,acquire_gpu0):
    need(spec['mode']=='fit','Fixed TRAIN fit mode')
    scientific,plan,check,writer,load,join,barrier_ref,seals,cache=setup(spec,resource_check,acquire_gpu0)
    model,optimizer=fresh_model()
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    def checkpoint(epoch,updates):
        payload=dict(schema='European_cohort8_TRAIN_checkpoint_v1',scientific_condition=CONDITION,
            source_manifest_ref=spec['source_manifest_ref'],protocol_id=scientific['protocol_id'],
            sampling_plan_ref=scientific['sampling_plan_ref'],observation_barrier_ref=barrier_ref,
            epoch=epoch,optimizer_updates=updates,settings=copy.deepcopy(original.SETTINGS),class_order=list(CLASSES),
            model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
            checkpoint_selection='fixed_final_epoch30',TEST_truth_read=False)
        # Serialize the small model into bounded RAM before reserving exact disk bytes.
        stream=io.BytesIO();torch.save(payload,stream);data=stream.getvalue()
        need(len(data)<=8*2**20,'Original8MiB per-checkpoint bound')
        path=writer.prepare('checkpoints/epoch_%02d.pt'%epoch,len(data))
        with path.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
        return writer.finish(path,len(data))
    def record(value):writer.json('updates/update_%03d.json'%value['optimizer_updates'],value)
    result=original.fixed_training_loop(parents=tuple(plan['train_parent_ids']),load_parent=load,
        join_targets=lambda parent,bundle,metadata:join(parent,bundle,metadata,model),
        step=lambda bundle,targets,metadata:train_sampled_step(model,bundle,targets,optimizer,metadata,resource_check=check),
        optimizer_counts=lambda:memory._optimizer_counts(optimizer),checkpoint=checkpoint,check=check,event=record)
    saved=torch.load(contracts.contained_path(result['checkpoint_ref']['path']),map_location='cpu',weights_only=True)
    need(saved['schema']=='European_cohort8_TRAIN_checkpoint_v1' and saved['scientific_condition']==CONDITION
         and saved['source_manifest_ref']==spec['source_manifest_ref'],'Distinct final cohort8 checkpoint binding')
    compatibility=dict(saved,schema='european_treeloom_full_checkpoint_v1')
    original._validate_saved(compatibility,build_model(),protocol_id=scientific['protocol_id'],
        sampling_plan_ref=scientific['sampling_plan_ref'],updates=90,barrier_ref=barrier_ref)
    torch.cuda.synchronize(0);check('European_cohort8_fit_complete')
    result.update(schema='European_cohort8_TRAIN_completion_v1',status='completed_fixed90updates',scientific_condition=CONDITION,
        source_manifest_ref=spec['source_manifest_ref'],observation_barrier_ref=barrier_ref,
        checkpoint_selection='fixed_final_epoch30',class_order=list(CLASSES),TEST_truth_read=False,
        independent_validation_still_required_for_architecture_search_or_comparative_claims=True,
        actual_worker_wait_and_owned_cleanup_still_required=True)
    ref=writer.json('training_completed.json',result)
    need(not writer.pending,'Every bounded artifact finished')
    return dict(result,completion_ref=ref)


def run_qualification(spec,*,resource_check,acquire_gpu0):
    need(spec['mode']=='qualification','Full-parent qualification mode')
    scientific,plan,check,writer,load,join,barrier_ref,seals,cache=setup(spec,resource_check,acquire_gpu0)
    parent=plan['train_parent_ids'][0]
    # Retain no other prepared bank while the complete CPU context reference is live.
    for other in list(cache):
        if other!=parent:memory._drop_cache_entry(cache,other)
    bundle,metadata=load(parent)
    model,optimizer=fresh_model();targets=join(parent,bundle,metadata,model)
    cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(0).clone()
    torch.cuda.reset_peak_memory_stats(0);started=time.monotonic()
    baseline,left=train_sampled_step(model,bundle,targets,optimizer,metadata,resource_check=check,
        executor_type=CohortStagedExecution,capture=True)
    torch.cuda.synchronize(0)
    baseline['wall_seconds']=time.monotonic()-started
    baseline['max_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(0)
    baseline['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(0)
    need(torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(0)),
         'Qualification baseline consumed no RNG')
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    model,optimizer=fresh_model()
    torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,0)
    context_error=[0.]
    def compare_context(context):
        reference=left.pop('context')
        need(context.shape==reference.shape,'Complete context shape unchanged')
        for first in range(0,len(context),128):
            check('European_full_context_numerical_comparison')
            a,b=reference[first:first+128],context[first:first+128]
            torch.testing.assert_close(a,b,**TOLERANCE)
            context_error[0]=max(context_error[0],float((a-b).abs().max()))
        del reference
    torch.cuda.reset_peak_memory_stats(0);started=time.monotonic()
    candidate,right=train_sampled_step(model,bundle,targets,optimizer,metadata,resource_check=check,capture=True,
        context_compare=compare_context)
    torch.cuda.synchronize(0)
    candidate['wall_seconds']=time.monotonic()-started
    candidate['max_cuda_reserved_bytes']=torch.cuda.max_memory_reserved(0)
    candidate['max_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(0)
    del right['context']
    errors={'context':context_error[0]}
    def compare(name,a,b):
        torch.testing.assert_close(a,b,**TOLERANCE,msg=name)
        errors[name]=float((a-b).abs().max())
    compare('loss',torch.tensor(baseline['loss']),torch.tensor(candidate['loss']))
    compare('norm',torch.tensor(baseline['gradient_norm_before_clip']),torch.tensor(candidate['gradient_norm_before_clip']))
    compare('logits',left['logits'],right['logits'])
    argmax_changes=int((left['logits'].argmax(-1)!=right['logits'].argmax(-1)).sum())
    need(argmax_changes==0,'All original sampled row argmax values preserved')
    for category in ('gradients','parameters'):
        need(len(left[category])==len(right[category])==49,'Every model tensor compared')
        for name,value in left[category].items():compare(category+'/'+name,value,right[category][name])
    adam_tensors=0
    for index,state in left['optimizer']['state'].items():
        for name,value in state.items():compare('Adam/'+str(index)+'/'+name,value,right['optimizer']['state'][index][name]);adam_tensors+=1
    need(adam_tensors==147 and all(value>0 for value in candidate['cohort_channel_gradient_norms']),
         'All147 Adam tensors and all8 nonzero cohort gradients')
    need(torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(0)),
         'Qualification candidate consumed no RNG')
    check('European_full_parent_qualification_complete')
    result=dict(schema='European_cohort8_full_parent_qualification_v1',passed=True,scientific_condition=CONDITION,
        source_manifest_ref=spec['source_manifest_ref'],observation_barrier_ref=barrier_ref,
        qualified_parent=parent,prepared_parents=plan['train_parent_ids'],observation_seals=seals,
        all_original_rows_used=True,sampled_rows=len(targets.labels),tolerance=TOLERANCE,
        gradient_tensors=49,parameter_tensors=49,Adam_tensors=147,argmax_changes=argmax_changes,
        errors=errors,baseline=baseline,candidate=candidate,
        worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        no_qualification_checkpoint_used_for_fit=True,TEST_truth_read=False,
        extra_logits_passes_for_qualification=True,steady_state_performance_claimed=False,
        other_parent_execution_peaks_measured=False,actual_worker_wait_and_owned_cleanup_still_required=True)
    ref=writer.json('qualification.json',result)
    need(not writer.pending,'Every qualification artifact finished')
    return dict(result,completion_ref=ref)
