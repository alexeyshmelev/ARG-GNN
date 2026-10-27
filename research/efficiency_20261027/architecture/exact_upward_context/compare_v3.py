"""One prefix/full pair: qualified resident CPU context versus released GPU context.

ROOT launches and admits this standalone worker. No production checkpoint,
observation disk cache, repeated benchmark, TEST access, or scheduler mutation.
"""
import argparse
import copy
import fcntl
import gc
import json
import os
import resource
import stat
import sys
import time
from types import SimpleNamespace
from . import protocol_v3 as protocol
from .protocol_v3 import ROOT,RUNS,GIB,GPU_UUID,KINDS,need,contained,reference,read


def run_update(kind,args,torch,rt,saved,bundle,metadata,labels,frozen,cache,check,output,context_reference):
    from research.efficiency_20261027.root.compare_selected_full import MemoryProfile,failure_record
    from research.efficiency_20261027.root.compare_host_parallelism import rss,write
    from research.efficiency_20261027.architecture.exact_upward_residency.large_resident_executor_v1 import build_executor as reference_factory
    from research.efficiency_20261027.architecture.exact_upward_residency.memory_v1 import residency_plan as reference_plan
    from .large_context_executor_v1 import build_executor as candidate_factory
    from .memory_v1 import residency_plan as candidate_plan,DEFAULT_SCRATCH_BYTES
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward,geometry_from_spans
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    limits=protocol.bounds(args.mode)
    candidate=kind==KINDS[1]
    need(kind in KINDS and torch.get_num_threads()==8,'Same CPU8 and original512 partitions')
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    profile=MemoryProfile(torch,check,journal=output/'stage_memory.jsonl',
        identity=dict(kind=kind,phase=args.mode+'_first_call',repeat=0),GPU_budget_bytes=limits['device_soft_bytes'])
    model=optimizer=executor=hook=context=vjp=None
    logits=[];failed=False;diagnostic_seconds=0.;context_comparison=None
    started=time.perf_counter();wall_start=time.time()
    try:
        with profile.stage('profile_restore'):
            model=rt.build_model();model.load_state_dict(saved['model_state'],strict=True)
            model=model.to(device='cuda:0',dtype=torch.float32)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,
                betas=(.9,.999),eps=1e-8,maximize=False,amsgrad=False,foreach=False,
                fused=False,capturable=False,differentiable=False)
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state']))
            restore_random_state(saved['random_state'])
        hook=model.classifier.register_forward_hook(lambda module,inputs,value:logits.append(value.detach().cpu()))
        model.train();optimizer.zero_grad(set_to_none=True)
        with profile.stage('profile_layout'):
            kwargs=dict(layout_cache=cache,frozen_module=frozen,holds_per_block=512,
                haplotypes_per_gru=64,history_holds_per_window=1024,resource_check=check,profile=profile)
            if candidate:kwargs['resident_soft_bytes']=limits['device_soft_bytes']
            executor=(candidate_factory if candidate else reference_factory)(model,bundle,metadata,**kwargs)
            plan=(candidate_plan if candidate else reference_plan)(executor)
            need(plan['native_nodes']==bundle.bank.parent.numel() and plan['scratch_reserve_bytes']==DEFAULT_SCRATCH_BYTES
                 and torch.cuda.memory_allocated()+plan['initial_additional_capacity_bytes']<=limits['device_soft_bytes'],
                 'Actual full bank/context lifetime plus unchanged16GiB scratch fits worker budget')
        with torch.autocast(device_type='cuda',enabled=False):
            with profile.stage('profile_context_forward'):
                context=executor.context_forward()
                need(context.shape==(limits['expected_holds'],1600,32)
                     and context.device.type==('cuda' if candidate else 'cpu'),'Actual selected complete context device')
                context_boundary=dict(device=str(context.device),bytes=context.numel()*context.element_size())
            if not candidate:
                # Retain the already allocated CPU storage, without cloning or its autograd graph.
                context_reference['value']=context.detach()
            else:
                # This diagnostic is outside synchronized allocator stage intervals.
                # Each bounded copy is released before the next, then the reference
                # is released before the decoder. No context values are serialized.
                from .context_diagnostic_v1 import compare_context
                diagnostic_started=time.perf_counter()
                reference_context=context_reference.pop('value')
                context_comparison=compare_context(torch,context,reference_context,check)
                del reference_context
                diagnostic_seconds=time.perf_counter()-diagnostic_started
                context_comparison.update(diagnostic_seconds=diagnostic_seconds,reference_released_before_decoder=True)
                write(output/'context_diagnostic.json',context_comparison)
                need(context_comparison['practical_context_passed'],
                     'Prospective fixed practical context envelope5e-6/1e-5 failed; no tolerance escalation')
            with profile.stage('profile_decoder_forward_backward'):
                geometry=geometry_from_spans(bundle.bank.spans)
                kwargs=dict(tile_haplotypes=64,profile=profile,resource_check=check)
                if candidate:
                    loss,vjp,decoder=executor.decode_parent_backward(context,geometry,bundle.focal_hold,
                        bundle.focal_sample_column,labels,**kwargs)
                else:
                    loss,vjp,decoder=decode_parent_backward(model,context,geometry,bundle.focal_hold,
                        bundle.focal_sample_column,labels,**kwargs)
                need(vjp.shape==context.shape and vjp.device==context.device,'Actual matching complete context VJP')
                vjp_boundary=dict(device=str(vjp.device),bytes=vjp.numel()*vjp.element_size())
            with profile.stage('profile_trunk_backward'):context.backward(vjp)
        with profile.stage('profile_gradient_capture'):
            gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()}
            need(len(gradients)==49 and all(bool(torch.isfinite(v).all()) for v in gradients.values()),'All49 finite original gradients')
        with profile.stage('profile_clip_optimizer'):
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started;wall_end=time.time()
        check('combined_update_complete')
        with profile.stage('profile_state_capture'):
            parameters={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
            moments={str(i):{key:value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key,value in row.items()} for i,row in optimizer.state_dict()['state'].items()}
            need(len(parameters)==len(moments)==49 and all(int(row['step'].item())==242 for row in moments.values()),
                 'All original parameters and Adam counters at242')
            numerical=dict(logits=torch.cat(logits),loss=loss.detach().cpu(),gradients=gradients,parameters=parameters,moments=moments)
            need(numerical['logits'].shape==(len(labels),3),'Every original selected target captured once')
        row=dict(kind=kind,update_seconds=elapsed-diagnostic_seconds,update_elapsed_including_diagnostic_seconds=elapsed,
            diagnostic_context_seconds=diagnostic_seconds,context_comparison=context_comparison,
            update_wall_start_unix=wall_start,update_wall_end_unix=wall_end,
            CPU_threads=8,holds_per_block=512,loss=float(loss),gradient_norm=float(norm),decoder=decoder,
            context_boundary=context_boundary,VJP_boundary=vjp_boundary,stages=profile.snapshot(),
            torch_peak_allocated_bytes=profile.run_allocated_peak,torch_peak_reserved_bytes=profile.run_reserved_peak,
            RSS_bytes=rss(),process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cache_snapshot=cache.snapshot(),explicit_vjp=executor.explicit_vjp_snapshot(),resident_bank_plan=plan,
            full_history=args.mode=='full',CPU_context=not candidate,hotpack=False,
            native_nodes=bundle.bank.parent.numel(),device_soft_budget_bytes=limits['device_soft_bytes'],parameter_count=49,optimizer_step=242)
        return row,numerical
    except BaseException as error:
        failed=True
        write(output/('failed_'+kind+'.json'),failure_record(error,kind=kind,
            elapsed_seconds=time.perf_counter()-started,stages=profile.snapshot()))
        raise
    finally:
        errors=[]
        for name,action in [('remove_hook',None if hook is None else hook.remove),
            ('clear_gradients',None if optimizer is None else lambda:optimizer.zero_grad(set_to_none=True)),
            ('release_layout',None if executor is None else executor.release_layout)]:
            if action is not None:
                try:action()
                except BaseException as error:errors.append(dict(action=name,error=str(error)))
        model=optimizer=executor=hook=context=vjp=action=None
        try:gc.collect();torch.cuda.empty_cache()
        except BaseException as error:errors.append(dict(action='allocator_cleanup',error=str(error)))
        if errors:
            write(output/('cleanup_'+kind+'.json'),dict(errors=errors,primary_failure_preserved=failed))
            if not failed:raise RuntimeError('Update cleanup failed; see cleanup receipt')


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode',choices=('prefix',),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--source-review-path',required=True)
    parser.add_argument('--source-review-sha256',required=True)
    parser.add_argument('--family-cap-bytes',type=int,required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--gpu-wait-seconds',type=int,required=True)
    parser.add_argument('--european-expected-returncode',type=int,default=0)
    parser.add_argument('--required-peer-directory',action='append',default=[])
    parser.add_argument('--prefix-run-directory')
    parser.add_argument('--prefix-completed-sha256')
    args=parser.parse_args(argv);limits=protocol.bounds(args.mode)
    pins=protocol.verify_sources(args.source_pins_sha256,args.source_review_path,args.source_review_sha256)
    prerequisites=dict(original_full_residency=protocol.scientific_predecessor())
    if args.mode=='full':
        need(args.prefix_run_directory and args.prefix_completed_sha256,'Full mode requires exact successful/reaped combined prefix')
        prerequisites['combined_prefix']=protocol.prefix_release(args.prefix_run_directory,args.prefix_completed_sha256,pins['source_manifest_ref'])
    else:need(args.prefix_run_directory is None and args.prefix_completed_sha256 is None,'Prefix has no combined-prefix predecessor')
    output=contained(args.output);directory=output.parent
    need(directory.parent==RUNS and output.name=='data' and not output.exists() and directory.is_dir(),
         'Fresh data in exact existing ROOT guardian directory')
    from research.efficiency_20261027.root import run_guarded_profile_with_peer as guardian
    started=time.monotonic()
    while not (directory/'launch.json').exists():
        need(time.monotonic()-started<60,'Actual ROOT launch receipt required');time.sleep(.1)
    request,request_ref=read(directory/'request.json');launch,launch_ref=read(directory/'launch.json')
    pid,parent=os.getpid(),os.getppid();ticks,parent_ticks=guardian.identity(pid),guardian.identity(parent)
    protocol.validate_guardian(request,launch,args,argv,pid,ticks,parent,parent_ticks)
    need(os.getsid(pid)==os.getpgid(pid)==pid,'Actual own ROOT process session')
    need(os.environ.get('ARGGNN_PHYSICAL_GPU')=='5' and os.environ.get('CUDA_VISIBLE_DEVICES')==GPU_UUID
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device'
         and 'ARGGNN_READABLE_GPU_FD' not in os.environ and 'ARGGNN_MAX_CUDA_RESERVED_BYTES' not in os.environ,
         'Exact ROOT single full-device environment without inherited lease')
    import torch
    need(not torch.cuda.is_initialized(),'No CUDA before exact metadata admission')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from research.efficiency_20261027.root.compare_host_parallelism import rss,write,prepare
    from research.efficiency_20261027.root.compare_selected_full import failure_record
    from research.efficiency_20261027.execution.compare_full_context import compare_state
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
    from arggnn.benchmarking.paper_campaign.processes import pss
    from research.repairs.benchmark_multigpu_20261022.runtime.queue_gpu_v1 import physical_path
    output.mkdir();lease={'physical':None};last_check=[-float('inf')];peak_family=[0]
    def check(stage):
        elapsed=time.monotonic()-started
        need(elapsed<=args.max_seconds,'Whole-worker wall bound at '+stage)
        need(rss()<=limits['worker_RSS_bytes'],'Worker RSS bound at '+stage)
        if torch.cuda.is_initialized():
            need(lease['physical'] is not None and torch.cuda.memory_reserved()<=limits['device_soft_bytes'],
                 'Acquired physical lease and finite reserved-memory bound')
        if elapsed-last_check[0]<5:return
        last_check[0]=elapsed
        need(os.getppid()==parent and guardian.identity(parent)==parent_ticks,'Actual ROOT guardian remains alive')
        need(reference(directory/'request.json')==request_ref and reference(directory/'launch.json')==launch_ref,
             'Actual ROOT request and launch unchanged')
        value=pss(pid);peak_family[0]=max(peak_family[0],value)
        need(value<=args.family_cap_bytes,'Complete owned-family PSS bound')
        need(guardian.scope_bytes(directory)<=limits['output_bytes'],'Whole ROOT output bound')
        if lease['physical'] is not None:
            held=os.fstat(lease['physical']);named=physical_path(5).stat()
            need((held.st_dev,held.st_ino)==(named.st_dev,named.st_ino),'Actual held physical GPU5 inode unchanged')
    try:
        write(output/'configuration.json',dict(arguments=vars(args),**pins,prerequisites=prerequisites,limits=limits,
            checkpoint_ref=protocol.CHECKPOINT_REF,request_ref=request_ref,launch_ref=launch_ref,
            numerical_calls_per_arm=1,profiler=False,steady_throughput_claim=False,observation_disk_cache=False))
        from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh,training_v2 as validation
        need(reference(protocol.CHECKPOINT_REF['path'])==protocol.CHECKPOINT_REF,'Immutable original checkpoint241')
        saved=torch.load(ROOT/protocol.CHECKPOINT_REF['path'],map_location='cpu',weights_only=True,mmap=True)
        registry=fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref']);rt=fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved,rt.build_model(),registry,expected_steps=241)
        need(not torch.cuda.is_initialized(),'CPU observations prepared before CUDA')
        if args.mode=='full':
            from research.preprocessing_optimization_20261026.integration.hooks import install_tree
            install_tree(preparation_workers=4,max_rss_bytes=limits['worker_RSS_bytes'])
            bundle,metadata,labels,frozen,scope=prepare(SimpleNamespace(mode='full'),rt,saved,check)
        else:
            from research.efficiency_20261027.execution.prepare_large_slice_v2 import prepare_large_prefix
            bundle,metadata,labels,frozen,scope=prepare_large_prefix(SimpleNamespace(slice_holds=1153),rt,saved,check)
        need(not torch.cuda.is_initialized() and bundle.native_holds==limits['expected_holds']
             and bundle.bank.parent.numel()==sum(metadata['episode_metadata']['native_nodes_per_tree']),
             'Complete selected native observations and no preparation CUDA')
        scope=dict(scope,native_nodes=bundle.bank.parent.numel())
        if args.mode=='full':need(bundle.bank.parent.numel()==205369272 and bundle.original_sites==227474 and len(labels)==10000,
                                  'Exact complete30Mb TRAIN000 observation and all10000 targets')
        write(output/'scope.json',scope)
        waiting=time.monotonic();last_print=-float('inf');required=limits['device_soft_bytes']+2*GIB
        while True:
            check('wait_for_European_resource_terminal')
            gate=protocol.european_resource_release(args.european_expected_returncode)
            free=guardian.gpu_free(5) if gate is not None else None
            if gate is not None and free>=required:
                fd=os.open(physical_path(5),os.O_RDWR|os.O_NOFOLLOW)
                try:fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:os.close(fd)
                else:lease['physical']=fd;break
            elapsed=time.monotonic()-waiting
            if elapsed-last_print>=30:
                print(json.dumps(dict(event='combined_GPU5_deferred',seconds=elapsed,resource_predecessor_reaped=gate is not None,
                    free_GPU_bytes=free,required_free_GPU_bytes=required,CUDA_initialized=False)),flush=True);last_print=elapsed
            need(elapsed<args.gpu_wait_seconds,'Exact resource predecessor/capacity/physical lease wait expired');time.sleep(5)
        need(stat.S_ISREG(os.fstat(lease['physical']).st_mode),'Existing ordinary physical GPU5 lease')
        need(protocol.european_resource_release(args.european_expected_returncode)==gate and guardian.gpu_free(5)>=required,
             'Resource predecessor and capacity rechecked under actual shared physical lease')
        need(protocol.scientific_predecessor()==prerequisites['original_full_residency'],'Original scientific prerequisite unchanged')
        if args.mode=='full':need(protocol.prefix_release(args.prefix_run_directory,args.prefix_completed_sha256,pins['source_manifest_ref'])
                                 ==prerequisites['combined_prefix'],'Exact combined prefix unchanged')
        write(output/'gpu_resource_gate.json',dict(**gate,physical_gpu=5,gpu_uuid=GPU_UUID,shared_physical_lease=True,
            campaign_reservation_created=False,minimum_free_bytes=required,observed_free_bytes=free))
        rows=[];baseline=None;context_reference={}
        for kind in KINDS:
            protocol.verify_sources(args.source_pins_sha256,args.source_review_path,args.source_review_sha256)
            cache=LargeBlockLayoutCache(max_bytes=limits['cache_bytes'],admit=lambda amount,stage:
                need(rss()+amount<=limits['worker_RSS_bytes'],'Layout working reserve at '+stage))
            row,numerical=run_update(kind,args,torch,rt,saved,bundle,metadata,labels,frozen,cache,check,output,context_reference)
            if baseline is None:baseline=numerical
            row['comparison']=compare_state(baseline,numerical)
            write(output/(kind+'.json'),row);protocol.require_comparison(row['comparison']);rows.append(row)
            del cache;gc.collect()
            print(json.dumps(dict(event='combined_update_complete',kind=kind,seconds=row['update_seconds'],comparison_passed=True)),flush=True)
        need(not context_reference,'Complete reference context released before completion')
        norm_comparison=protocol.compare_norm(rows[0]['gradient_norm'],rows[1]['gradient_norm'])
        need(norm_comparison['passed'],'Original strict gradient norm parity')
        protocol.verify_sources(args.source_pins_sha256,args.source_review_path,args.source_review_sha256)
        check('combined_complete')
        result=dict(schema='TreeLoom_upward_context_prefix_diagnostic_v1',mode=args.mode,status='diagnostic_completed',**pins,
            checkpoint_ref=protocol.CHECKPOINT_REF,scope=scope,records=rows,gradient_norm_comparison=norm_comparison,prerequisites=prerequisites,
            European_resource_gate=gate,all_strict_comparisons_passed=rows[1]['context_comparison']['strict_context_passed'],
            strict_downstream_passed=True,strict_context_passed=rows[1]['context_comparison']['strict_context_passed'],
            practical_context_passed=True,eligible_for_full_qualification=False,limits=limits,
            measured_peak_family_PSS_bytes=peak_family[0],physical_gpu='5',visible_UUID=GPU_UUID,
            total_seconds=time.monotonic()-started,own_actual_wait_and_reaping_still_required=True,
            limitations=['One fixed-order call per arm; no stable throughput or sustained utilization claim.',
                'Every arm restores original241 model, AdamW and RNG; exactly one update to242.',
                'Prefix retains complete restricted-prefix history; only full mode qualifies the original full parent.',
                'Original strict context status is reported separately from the fixed5e-6/1e-5 practical diagnostic envelope.',
                'Logits/loss/norm/all49 gradients/parameters and147 Adam tensors retain original strict tolerances;zeroargmax.',
                'This prefix-only diagnostic cannot release a full comparison or production migration.',
                'Reported update time excludes separately recorded bounded context diagnostic; raw elapsed/full wall are retained.',
                'No held-out accuracy score, production migration, or saved learned checkpoint.'])
        protocol.require_completed(result,args.mode,pins['source_manifest_ref'])
        write(output/'completed.json',result)
        print(json.dumps(dict(event='combined_comparison_completed',completion_ref=reference(output/'completed.json'))),flush=True)
    except BaseException as error:
        write(output/'failed.json',failure_record(error,elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if lease['physical'] is not None:os.close(lease['physical']);lease['physical']=None


if __name__=='__main__':main()
