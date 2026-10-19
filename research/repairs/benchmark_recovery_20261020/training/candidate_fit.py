"""First real update at 128 GiB, then explicitly admitted prepare-once RAM reuse."""
import gc
import json
import os
import resource
import time
from .candidate_common import (
    ROOT, GIB, need, read_ref, file_ref, write_json, identity, rss_bytes,
    source_identity, continuation_allocation,
)
from .candidate_worker import configure_torch, metadata_bindings, enable_gpu

BOUNDS_PATH = 'research/reviews/workflow_progress_diagnosis_20261018/treeloom/storage_bound_v2.json'
POOL_SOURCE_PATHS = (
    'research/reviews/workflow_progress_diagnosis_20261018/treeloom/resident_pool_v2.py',
    'research/reviews/workflow_progress_diagnosis_20261018/treeloom/cache_v1.py',
    'research/reviews/workflow_progress_diagnosis_20261018/treeloom/indices_v1.py',
    BOUNDS_PATH,
)


def create_pool(request,spec,registry,fresh,guard,event,resource_check):
    """Ordinary imported readonly RAM codec; no legacy source transformation."""
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom.resident_pool_v2 import CompactParentPool
    refs = {row['path']:row for row in request['source_refs']}
    need(set(POOL_SOURCE_PATHS) <= set(refs), "Pinned readable compact-pool source closure required")
    for name in POOL_SOURCE_PATHS:
        need(file_ref(name) == refs[name], "Compact-pool source changed")
    bounds = read_ref(refs[BOUNDS_PATH])
    parent_bounds = {row['parent_id']:row['complete_compact_cache_upper_bound_bytes']+160000
                     for row in bounds['parents']}
    parents = set(registry.ids('train'))
    need(set(parent_bounds) == parents, "All 20 original parent bounds required")
    inputs = {parent:{key:spec['train_inputs'][parent][key] for key in fresh.INPUT_REFS}
              for parent in parents}
    external = {parent:{'bank.genotype':read_ref(inputs[parent]['observation_ref'])['genotypes_ref']}
                for parent in parents}
    return CompactParentPool(parent_bounds=parent_bounds,input_refs=inputs,external_arrays=external,
        max_cache_bytes=352*GIB,max_rss_bytes=512*GIB,preparation_headroom_bytes=128*GIB,
        rss_bytes=rss_bytes,resource_check=resource_check,event=event)


def fit(request,request_ref,job,barrier,output,guard):
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from arggnn.american_c3_gnn_train_v1 import _cpu_snapshot
    from arggnn.benchmarking.treeloom_training import _optimizer_counts, _target_binding
    from ..runtime.binding import ReadableTreeLoom
    from .candidate_gru import install_chunked_gru
    from .progress import StageProgress
    from .step import install_execution
    torch = configure_torch()
    fresh,spec,registry,plans = metadata_bindings(job,barrier)
    expected_holds = max(plan['native_holds'] for plan in plans.values())
    need(read_ref(request['preflight_ref'])['native_holds'] >= expected_holds,
         "GPU compatibility receipt must cover every complete TRAIN sequence length")
    runtime = ReadableTreeLoom(registry)
    install_execution(runtime,status_path=output/'training_live.json',holds_per_block=128,haplotypes_per_gru=32)
    parents = tuple(registry.ids('train'))
    need(len(parents) == 20 and len(set(parents)) == 20, 'Exactly 20 original TRAIN parents required')
    schedule = []
    for epoch in range(1,31):
        order = tuple(registry.epoch_order(epoch))
        need(len(order) == 20 and set(order) == set(parents), 'Each epoch visits all 20 parents exactly once')
        schedule.extend((epoch,parent) for parent in order)
    need(len(schedule) == 600, "Fixed complete-parent schedule required")
    write_json(output/'schedule.json',dict(settings=spec['settings'],schedule=schedule,
        observation_barrier_ref=request['observation_barrier_ref'],first_update_retained=True,
        observation_disk_cache=False,continuation_requires_512GiB_CPU_reservation=True))
    model = install_chunked_gru(runtime.build_model(),chunk_holds=1024)
    optimizer, pool = None, None
    active_profile = [None]
    allocation_ref = None
    def resolve_truth(parent):
        need(registry.split_for(parent) == 'train', "Only TRAIN truth may be resolved")
        return spec['train_inputs'][parent]['truth_ref']
    joiner = fresh.FreshTrainTargetJoin(registry,resolve_truth,train_inputs=spec['train_inputs'],
        max_truth_uncompressed_bytes=spec['max_truth_uncompressed_bytes'])
    checkpoints = {}
    def event(name,value):
        guard.check('candidate_progress')
        raw = (json.dumps(dict(event=name,**value),sort_keys=True,allow_nan=False)+'\n').encode()
        need(len(raw) <= 16384, "Compact progress event required")
        path = output/'progress.jsonl'
        need((path.stat().st_size if path.exists() else 0)+len(raw) <= 8*2**20,
             "Bounded complete schedule progress")
        with path.open('ab') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    def pool_tick(label):
        guard.check(label)
        if active_profile[0] is not None:
            # The original codec has many tensor names. Keep one bounded mark
            # with the latest exact name, rather than an unbounded mark family.
            active_profile[0].mark('compact_pool_work',codec_stage=label)
    def prepare(parent,profile):
        item = spec['train_inputs'][parent]
        def tick(label):
            guard.check(label)
            profile.mark('observed_preparation_work',preparation_stage=label)
        config = dict(native_preparation_ref=item['native_preparation_ref'],class_order=list(fresh.CLASSES),
            expected_haplotypes=1600,reference_counts=[200,200,200],
            input_reference_class_mask=None,input_reference_sample_indices=None)
        with profile.stage('load_original_observed_parent'):
            state = load_observation(item['observation_ref'],item['focal_registry_ref'],config)
        with profile.stage('prepare_complete_observed_parent'):
            bundle,metadata = runtime.prepare_observation(state,split='train',resource_check=tick,
                **{key:item[key] for key in fresh.INPUT_REFS},**spec['preparation_limits'])
        del state
        return bundle,metadata
    def checkpoint(name,epoch,steps):
        guard.check('candidate_checkpoint')
        path = output/'checkpoints'/name
        payload = dict(schema='treeloom_fresh_full_checkpoint_v1',epoch=epoch,complete_parent_updates=steps,
            model_seed=70001,
            parent_registry_ref=registry.reference,observation_barrier_ref=request['observation_barrier_ref'],
            settings=spec['settings'],contract=model.checkpoint_contract(),
            model_state=_cpu_snapshot(model.state_dict()),optimizer_state=_cpu_snapshot(optimizer.state_dict()),
            VAL_truth_read=False,TEST_truth_read=False,checkpoint_selection='fixed_final_epoch30',
            candidate_request_ref=request_ref,qualification_step_preserved=True,
            artifact_retention='all_six_scheduled_checkpoints_plus_first_update_within_64MiB')
        with path.open('xb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        need(path.stat().st_size <= 8*2**20, "Bounded full checkpoint required")
        return file_ref(path)
    for at,(epoch,parent) in enumerate(schedule,1):
        guard.check('before_parent_load')
        profile = StageProgress(output/'preparation_live.json',parent_id=parent)
        active_profile[0] = profile
        try:
            profile.mark('parent_load_begin',epoch=epoch,next_optimizer_update=at)
            profile.flush()
            if at == 1:
                bundle,metadata = prepare(parent,profile)
                prepared_now = True
            else:
                need(pool is not None, "Explicit continuation and compact pool required")
                prepared_now = parent not in pool.records
                with profile.stage('compact_pool_load'):
                    bundle,metadata = pool.load(parent,lambda:prepare(parent,profile))
            seal = barrier['parents'][parent]
            need(registry.split_for(parent) == 'train' and bundle.parent_id == parent
                 and bundle.split == 'train' and bundle.sequence_length_bp == 30000000
                 and metadata.get('parent_id') == parent and metadata.get('declared_split') == 'train'
                 and metadata.get('parent_registry_ref') == registry.reference
                 and metadata.get('all_native_trees_nodes_sites_samples_retained') is True
                 and metadata.get('private_truth_opened') is False
                 and bundle.focal_identity_sha256 == seal['focal_identity_sha256']
                 and bundle.native_holds == plans[parent]['native_holds']
                 and bundle.original_sites == plans[parent]['original_sites']
                 and all(getattr(bundle,key) == seal[key] for key in fresh.INPUT_REFS),
                 "Full observations must match the inherited immutable seal")
            bundle.validate()
            profile.mark('parent_load_complete',native_holds=bundle.native_holds,prepared_now=prepared_now)
            event('complete_parent_prepared' if prepared_now else 'complete_parent_restored',
                dict(epoch=epoch,parent_id=parent,next_optimizer_update=at,
                     stage_timings=profile.snapshot()['stage_timings'],observation_disk_cache=False))
            if at == 1:
                need(not torch.cuda.is_initialized(), "No CUDA before first-parent GPU gate")
                write_json(output/'ready_for_gpu.json',dict(request_ref=request_ref,worker_identity=identity(),
                    phase='fit',parent_id=parent,native_holds=bundle.native_holds,CUDA_initialized=False))
                enable_gpu(torch,guard,request_ref,profile)
                model = model.to(device='cuda:0',dtype=torch.float32)
                optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
                    betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,fused=False,
                    capturable=False,differentiable=False)
                need(not optimizer.state, "Fresh independent optimizer required")
        finally:
            active_profile[0] = None
            profile.close()
        with_target = joiner.join(parent,split='train',plan=plans[parent],observation_seal=seal)
        _target_binding(with_target,parent,bundle,registry,spec['train_inputs'])
        started = time.monotonic()
        result = runtime.train_step(model,bundle,with_target,optimizer,metadata,resource_check=guard.check)
        torch.cuda.synchronize()
        need(result['one_complete_parent_optimizer_update'] is True
             and result['native_focals'] == 10000 and result['parent_id'] == parent,
             "A complete original full-parent update must finish")
        counts = _optimizer_counts(optimizer)
        need(len(counts) == 49 and set(counts) == {at}, "All 49 optimizer counters must match actual schedule")
        event('complete_parent_update',dict(epoch=epoch,optimizer_updates=at,elapsed_seconds=time.monotonic()-started,
            CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),**result))
        del with_target
        gc.collect()
        torch.cuda.empty_cache()
        if at == 1:
            # Keep this prepared bundle while ROOT reviews the genuine update.
            # Its first compact copy is made only after the 512 GiB reservation.
            saved = checkpoint('qualification_step_001.pt',epoch,at)
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            bounded = peak_rss <= 128*GIB
            qualification = dict(schema='treeloom_complete_parent_candidate_v1',
                status='completed_first_update' if bounded else 'first_update_RSS_bound_failed',
                request_ref=request_ref,worker_identity=identity(),first_parent=parent,epoch=epoch,
                optimizer_updates=1,all_49_Adam_counters=counts,checkpoint_ref=saved,
                original_schedule_position_preserved=True,preflight_ref=request['preflight_ref'],
                result=result,CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                peak_RSS_bytes=peak_rss,initial_RSS_bound_passed=bounded,
                source_identity=source_identity(request['source_refs']),ancestry_accuracy=None)
            ref = write_json(output/'qualification.json',qualification)
            need(bounded, "Observed first-update RSS peak exceeded 128 GiB; checkpoint preserved")
            write_json(output/'waiting_for_continuation.json',dict(request_ref=request_ref,
                worker_identity=identity(),qualification_ref=ref,optimizer_updates=1,
                next_parent=schedule[1][1],first_prepared_parent_retained_in_RAM=True))
            gate_ref = guard.wait_gate('continue_gate.json',request_ref,qualification_ref=ref)
            allocation = continuation_allocation(output,request_ref,identity())
            guard.request['max_rss_bytes'] = allocation['max_rss_bytes']
            allocation_ref = write_json(output/'continuation_resource_activation.json',dict(
                request_ref=request_ref,worker_identity=identity(),gate_ref=gate_ref,
                cpu_admission_ref=allocation['cpu_admission_ref'],max_rss_bytes=512*GIB,
                observation_cache_bytes=352*GIB,working_reserve_bytes=128*GIB,
                optimizer_updates_preserved=1,checkpoint_ref=saved))
            pool = create_pool(request,spec,registry,fresh,guard,event,pool_tick)
            profile = StageProgress(output/'compact_pool_live.json',parent_id=parent)
            active_profile[0] = profile
            try:
                with profile.stage('retain_qualified_parent_in_compact_pool'):
                    pool.load(parent,lambda:(bundle,metadata))
            finally:
                active_profile[0] = None
                profile.close()
            event('continuation_activated',dict(optimizer_updates=1,cpu_admission_ref=allocation['cpu_admission_ref'],
                compact_pool=pool.receipt(),cold_repreparations=0))
        del bundle,metadata
        gc.collect()
        if at == 20:
            pool.require_all_prepared()
            event('all20_compact_parents_retained',pool.receipt())
        if at % 20 == 0 and epoch in (5,10,15,20,25,30):
            checkpoints[str(epoch)] = checkpoint('epoch_%02d.pt'%epoch,epoch,at)
            event('checkpoint_saved',dict(epoch=epoch,optimizer_updates=at,checkpoint_ref=checkpoints[str(epoch)]))
    need(set(checkpoints) == {'5','10','15','20','25','30'}, "All fixed checkpoint epochs required")
    pool.require_all_prepared()
    need(set(pool.restorations) == set(registry.ids('train'))
         and set(pool.restorations.values()) == {29}
         and sum(pool.restorations.values()) == 580,
         "All 20 parents prepared once and exactly restored for the remaining 29 epochs")
    saved = torch.load(ROOT/checkpoints['30']['path'],map_location='cpu',weights_only=True)
    need(saved['complete_parent_updates'] == 600 and saved['epoch'] == 30
         and len(saved['optimizer_state']['state']) == 49
         and all(int(value['step'].item()) == 600 for value in saved['optimizer_state']['state'].values()),
         "Actual saved final full optimizer state must contain all 600 updates")
    write_json(output/'training_completed.json',dict(schema='treeloom_fresh_full_training_completion_v1',
        status='completed_full20parents30passes600updates',parent_registry_ref=registry.reference,
        observation_barrier_ref=request['observation_barrier_ref'],
        final_saved_optimizer_parameter_states=49,final_saved_optimizer_steps=[600],
        actual_worker_wait_and_owned_cleanup_still_required=True,
        completed_epochs=30,complete_parent_updates=600,checkpoint_ref=checkpoints['30'],
        checkpoint_event_refs=checkpoints,selected_epoch=30,checkpoint_selection='fixed_final_epoch30',
        request_ref=request_ref,VAL_truth_read=False,TEST_truth_read=False,ancestry_accuracy=None,
        qualification_step_retained_in_schedule=True,observation_disk_cache=False,
        compact_pool_receipt=pool.receipt(),continuation_resource_activation_ref=allocation_ref))
