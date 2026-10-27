"""Exactly one missing fused profiler-only learned call; no baseline rerun or Adam step.

The completed v1 prefix numerical qualification is consumed as evidence and is
not repeated. Corrected attribution excludes native API user annotations explicitly.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, CHECKPOINT, GIB, need, ref, rss, write
from .compare_boundary_timeline_v1 import KINDS, baseline_factory, fused_factory
from .bounded_timeline_v1 import DiagnosticProfile
from .prepare_large_slice_v2 import prepare_large_prefix
from .large_layout_cache import LargeBlockLayoutCache
from .deferred_gpu_gate import wait_before_cuda
from .direct_activity_ledger_v3 import export_API

BASE=ROOT/'research/efficiency_20261027/execution'
QUALIFICATION=ROOT/'research/efficiency_20261027/root/runs/boundary_timeline_gpu5_v1/data/completed.json'


def verify_pins(digest):
    path=BASE/'source_pins_boundary_activity_v3.json'
    need(ref(path)['sha256']==digest,'Exact reviewed profiler-only source manifest')
    sources=json.loads(path.read_text())['source_refs']
    need(all(ref(item['path'])==item for item in sources),'Every immutable source/data pin remains exact')
    return dict(source_manifest_ref=ref(path),source_refs=sources)


def failed_v2_cleanup():
    from .deferred_gpu_gate import owned_group_members
    from research.efficiency_20261027.root.run_guarded_profile_with_peer import identity
    directory=ROOT/'research/efficiency_20261027/root/runs/boundary_activity_gpu5_v2'
    launch=json.loads((directory/'launch.json').read_text())
    wait=json.loads((directory/'actual_wait.json').read_text())
    need(wait['actual_wait'] and wait['owned_group_absent'] and wait['actual_returncode']==-15
         and wait['pid']==launch['pid']==1185817
         and wait['process_start_ticks']==launch['process_start_ticks']=='574723450'
         and identity(launch['pid']) != launch['process_start_ticks']
         and not owned_group_members(launch['pid']), 'Exact failed v2 worker must remain reaped and absent')
    return ref(directory/'actual_wait.json')


def run_profile(kind,rt,saved,bundle,metadata,labels,frozen,check,output):
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward,geometry_from_spans
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    model=executor=hook=context=vjp=capture=None
    logits=[]
    stages=DiagnosticProfile(True)
    cache=LargeBlockLayoutCache(max_bytes=256*2**20,admit=lambda amount,stage:
        need(rss()+amount<=32*GIB,'Profiler layout host reserve at '+stage))
    try:
        model=rt.build_model()
        model.load_state_dict(saved['model_state'],strict=True)
        model=model.to(device='cuda:0',dtype=torch.float32)
        restore_random_state(saved['random_state'])
        model.train()
        model.zero_grad(set_to_none=True)
        # Preserve the same classifier CPU-copy hook used by v1 learned clocks.
        hook=model.classifier.register_forward_hook(lambda module,inputs,value:logits.append(value.detach().cpu()))
        executor=(baseline_factory if kind==KINDS[0] else fused_factory)(model,bundle,metadata,
            layout_cache=cache,frozen_module=frozen,holds_per_block=512,haplotypes_per_gru=64,
            history_holds_per_window=1024,resource_check=check,profile=stages)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False) as capture:
            with torch.profiler.record_function('diagnostic_learned_update'),torch.autocast(device_type='cuda',enabled=False):
                with stages.stage('context_forward'):
                    context=executor.context_forward()
                    need(context.shape==(1153,1600,32) and context.device.type=='cpu','Exact original prefix CPU context')
                with stages.stage('decoder_forward_backward'):
                    loss,vjp,decoder=decode_parent_backward(model,context,geometry_from_spans(bundle.bank.spans),
                        bundle.focal_hold,bundle.focal_sample_column,labels,tile_haplotypes=64,
                        profile=stages,resource_check=check)
                with stages.stage('trunk_backward'):
                    context.backward(vjp)
                torch.cuda.synchronize()
        peak_allocated=torch.cuda.max_memory_allocated()
        peak_reserved=torch.cuda.max_memory_reserved()
        connected=sum(parameter.grad is not None for parameter in model.parameters())
        need(connected==49,'Complete original learned graph remains connected')
        check('profiler_learned_call_complete')
        analysis=export_API(capture,output/kind)
        need(analysis['physical_event_counts'].get('kernel',0)>0
             and analysis['physical_event_counts'].get('H2D',0)>0
             and analysis['physical_event_counts'].get('D2H',0)>0,'Actual native physical activity recorded')
        need(analysis['excluded_category_counts'].get('gpu_user_annotation',0)>0,'GPU annotations excluded explicitly')
        check('profiler_native_export_complete')
        return dict(kind=kind,profiled_calls=1,analysis=analysis,
            ledger_ref=ref(output/(kind+'_physical_ledger.json')),
            activity_summary_ref=ref(output/(kind+'_activity_summary.json')),
            stages=stages.snapshot(),torch_peak_allocated_bytes=peak_allocated,torch_peak_reserved_bytes=peak_reserved,
            connected_parameters=connected,explicit_vjp=executor.explicit_vjp_snapshot(),
            optimizer_constructed=False,optimizer_step_executed=False,numerical_comparison_repeated=False,
            checkpoint_parameters_updated=False,hotpack=False,profiled_clocks_are_not_speed_samples=True)
    finally:
        if hook is not None:hook.remove()
        if model is not None:model.zero_grad(set_to_none=True)
        if executor is not None:executor.release_layout()
        context=vjp=model=executor=hook=capture=None
        logits.clear()
        cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--gpu-predecessor-directory',required=True)
    parser.add_argument('--category-probe-sha256',required=True)
    parser.add_argument('--category-probe-directory',required=True)
    args=parser.parse_args()
    args.slice_holds=1153
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3','4','5')
         and os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-')
         and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
         and os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device','ROOT-admitted single GPU')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    pins=verify_pins(args.source_pins_sha256)
    failed_cleanup_ref=failed_v2_cleanup()
    qualification=json.loads(QUALIFICATION.read_text())
    need(qualification['status']=='completed' and len(qualification['records'])==8
         and all(row['comparison']['passed'] and row['parameter_count']==49 and row['optimizer_step']==242
             for row in qualification['records']),'Existing exact eight-call numerical qualification must have passed')
    probe_path=(ROOT/args.category_probe_directory/'data/completed.json').resolve()
    need(probe_path.is_relative_to(ROOT/'research/efficiency_20261027/root/runs')
         and ref(probe_path)['sha256']==args.category_probe_sha256,'Exact reviewed completed category-probe evidence')
    probe=json.loads(probe_path.read_text())
    need(probe['status']=='passed' and probe['native_physical_category_allowlist_verified']
         and probe['deliberate_CPU_gap_preserved'],'Actual CUDA category probe must precede learned profiling')
    recovered=BASE/'recovered_boundary_baseline_v2/RECOVERY.json'
    recovered_data=json.loads(recovered.read_text())
    need(recovered_data['status']=='passed' and recovered_data['complete_native_JSON_validated']
         and recovered_data['learned_calls_executed']==0, 'Complete recovered baseline is mandatory; never rerun it')
    output=Path(args.output).resolve()
    need(output.is_relative_to(ROOT/'research/efficiency_20261027/root/runs') and not output.exists(),'Fresh ROOT profiler output')
    output.mkdir(parents=True)
    started=time.perf_counter()
    def check(stage):
        need(time.perf_counter()-started<=600 and rss()<=32*GIB,'Profiler600second/32GiB host cap at '+stage)
        if torch.cuda.is_initialized():need(torch.cuda.memory_reserved()<=24*GIB,'Profiler24GiB CUDA cap at '+stage)
    write(output/'configuration.json',dict(arguments=vars(args),**pins,qualification_ref=ref(QUALIFICATION),
        category_probe_ref=ref(probe_path),calls=list(KINDS[1:]),host_cap_bytes=32*GIB,device_soft_cap_bytes=24*GIB,
        max_worker_seconds=600,complete_ledger_cap_each_bytes=8*2**20,numerical_requalification=False))
    try:
        from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh,training_v2 as validation
        need(ref(CHECKPOINT)['sha256']=='12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24','Exact checkpoint241')
        saved=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
        registry=fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref'])
        rt=fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved,rt.build_model(),registry,expected_steps=241)
        bundle,metadata,labels,frozen,scope=prepare_large_prefix(args,rt,saved,check)
        write(output/'scope.json',scope)
        failed_v2_cleanup()
        gate=wait_before_cuda(torch,check,predecessor_directory=args.gpu_predecessor_directory,output=output,
            gpu_index=int(os.environ['ARGGNN_PHYSICAL_GPU']),gpu_budget_bytes=24*GIB,timeout_seconds=30,poll_seconds=5)
        rows=[]
        for kind in KINDS[1:]:
            verify_pins(args.source_pins_sha256)
            row=run_profile(kind,rt,saved,bundle,metadata,labels,frozen,check,output)
            rows.append(row)
            write(output/(kind+'.json'),row)
        verify_pins(args.source_pins_sha256)
        write(output/'completed.json',dict(status='completed',schema='TreeLoom_boundary_activity_fused_only_v3',
            **pins,qualification_ref=ref(QUALIFICATION),category_probe_ref=ref(probe_path),scope=scope,
            records=rows,failed_v2_cleanup_ref=failed_cleanup_ref,recovered_baseline_ref=ref(recovered),gpu_predecessor_gate=gate,total_seconds=time.perf_counter()-started,
            physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'],numerical_qualification_repeated=False,
            optimizer_step_executed=False,production_changed=False,
            limitations=['Exactly one fused profiled call; baseline is recovered from preserved v2 trace; no new speed estimate or numerical qualification.',
                'Complete physical CUDA ledger uses native API annotation flags and excludes GPU annotations from union.',
                'This1153hold profile is process-specific and instrumented, not full-parent or device-wide duty evidence.',
                'The preserved classifier CPU-copy hook contributes to this learned recording.']))
    except BaseException as error:
        write(output/'failed.json',dict(error_type=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    main()
