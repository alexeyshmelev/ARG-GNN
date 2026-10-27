"""Root-launched128-hold TRAIN numerical qualification, never ancestry validation.

Both arms are the same fresh Transport model: monolithic versus device-blocked
execution. Uses the original22 frozen TRAIN focal targets on this diagnostic
window. No optimizer, baseline accuracy comparison or full-parent claim.
"""
import argparse,gc,json,os,time
from pathlib import Path
from types import SimpleNamespace
from .paired_training import ROOT,OWNED,need,reference


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-rss-gib',type=float,default=32)
    parser.add_argument('--max-gpu-reserved-gib',type=float,default=24)
    parser.add_argument('--max-seconds',type=int,default=1200)
    parser.add_argument('--warm-repeats',type=int,choices=(0,1),default=1)
    parser.add_argument('--tree-block',type=int,default=16)
    parser.add_argument('--haplotype-tile',type=int,default=16)
    parser.add_argument('--scan-block',type=int,default=32)
    args=parser.parse_args()
    need(os.environ.get('ARGGNN_PHYSICAL_GPU') in ('3','4','5') and
         os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-') and
         ',' not in os.environ['CUDA_VISIBLE_DEVICES'] and
         os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device',
         'Root admission must select one GPU3/4/5 UUID with the existing full-device policy')
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh
    from research.scalability_20261024.profiling.run_profile import prepare
    from .treeloom_transport import TreeLoomTransport,prepare_transport_plan
    from .bounded_transport import BoundedTransport
    from torch.nn import functional as F
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    need(torch.cuda.device_count()==1,'Exactly one root-selected logical CUDA device required')
    output=(ROOT/args.output).resolve()
    need(output.is_relative_to(OWNED) and not output.exists(),'Fresh architecture-owned diagnostic output required')
    output.mkdir(parents=True);start=time.monotonic();deadline=start+args.max_seconds
    def rss():return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    def check(stage):
        need(time.monotonic()<deadline,'Diagnostic wall budget exceeded')
        need(rss()<=args.max_rss_gib*2**30,'Diagnostic RSS budget exceeded at '+stage)
        if torch.cuda.is_initialized():need(torch.cuda.memory_reserved()<=args.max_gpu_reserved_gib*2**30,'Diagnostic GPU reserve budget exceeded at '+stage)
    def write(name,value):
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        return reference(output/name)
    pins=json.loads((OWNED/'gpu_128_qualification_sources.json').read_text())['source_refs']
    own_ref=reference(__file__)
    for pin in pins:need(reference(pin['path'])==pin,'Qualified dependency changed before diagnostic')
    write('invocation.json',dict(arguments=vars(args),source_ref=own_ref,source_refs=pins,
        physical_gpu=os.environ['ARGGNN_PHYSICAL_GPU'],visible_UUID=os.environ['CUDA_VISIBLE_DEVICES'],
        external_resource_guardian_required=True,hard_allocator_fraction_cap=False,optimizer_updates=0))
    try:
        checkpoint='research/repairs/TreeLoom_resume94_RAM_headroom_v1/checkpoint.pt'
        need(reference(checkpoint)['sha256']=='5ceebcd75ae978689f987a702ea5a157d6ddc2bd7d057e6aded56627bbf45276','Original94 lineage checkpoint required')
        saved=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=True,mmap=True)
        registry=fresh.FreshParentRegistry.from_ref(saved['parent_registry_ref']);runtime=fresh.FreshTreeLoom(registry)
        bundle,metadata,labels,_,scope=prepare(SimpleNamespace(mode='slice'),runtime,saved,check)
        del saved
        need(bundle.native_holds==128 and bundle.bank.sample_index.shape==(128,1600) and
             labels.shape==bundle.focal_hold.shape==bundle.focal_sample_column.shape==(22,),
             'Exactly original observed128-hold window and its22 frozen TRAIN focals required')
        cpu_plan=prepare_transport_plan(bundle.bank.parent,bundle.episodes.flags.node_flags)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70001);initial=TreeLoomTransport()
        weights={k:v.detach().clone() for k,v in initial.state_dict().items()};del initial
        reference_numerical=None;records=[]
        schedule=[('first_call',0,('monolithic','bounded'))]
        if args.warm_repeats:schedule.append(('warm_libraries',1,('bounded','monolithic')))
        for phase,repeat,kinds in schedule:
            for kind in kinds:
                check('before_'+kind);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                model=TreeLoomTransport();model.load_state_dict(weights,strict=True);model=model.to('cuda:0');model.train()
                torch.cuda.synchronize();transfer_seconds=time.perf_counter()-tick
                if kind=='monolithic':
                    tick=time.perf_counter();device_bank=bundle.bank.to('cuda:0');device_plan=cpu_plan.to('cuda:0')
                    torch.cuda.synchronize();transfer_seconds+=time.perf_counter()-tick
                else:
                    setup_start=time.perf_counter()
                    execution=BoundedTransport(model,bundle.bank,bundle.episodes.flags.node_flags,
                        metadata['episode_metadata']['native_nodes_per_tree'],holds_per_block=args.tree_block,
                        haplotypes_per_tile=args.haplotype_tile,scan_holds_per_block=args.scan_block,resource_check=check)
                setup_seconds=0. if kind=='monolithic' else time.perf_counter()-setup_start
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.autocast(device_type='cuda',enabled=False):
                    if kind=='monolithic':
                        logits=model.focal_logits(device_bank,device_plan,bundle.focal_hold.to('cuda:0'),bundle.focal_sample_column.to('cuda:0'))
                    else:logits=execution.forward_focals(bundle.focal_hold,bundle.focal_sample_column)
                    loss=F.cross_entropy(logits,labels.to('cuda:0'))
                torch.cuda.synchronize();forward_seconds=time.perf_counter()-tick
                check(kind+'_forward_complete');tick=time.perf_counter();loss.backward();torch.cuda.synchronize()
                backward_seconds=time.perf_counter()-tick;check(kind+'_backward_complete')
                need(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)) and
                     all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                     'All55 finite connected candidate parameter gradients required')
                numerical=dict(logits=logits.detach().cpu(),gradients={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()})
                need(len(numerical['gradients'])==55,'Exactly55 candidate parameter tensors required')
                need(all(torch.equal(model.state_dict()[name].detach().cpu(),value) for name,value in weights.items()),'No optimizer or weight modification permitted')
                if reference_numerical is None:
                    reference_numerical=numerical;comparison=dict(reference=True)
                else:
                    # Forward absolute1e-6, gradient elementwise atol1e-6/rtol1e-4.
                    torch.testing.assert_close(numerical['logits'],reference_numerical['logits'],atol=1e-6,rtol=0.)
                    gradient_rows=[]
                    for name,value in numerical['gradients'].items():
                        expected=reference_numerical['gradients'][name]
                        torch.testing.assert_close(value,expected,atol=1e-6,rtol=1e-4)
                        gradient_rows.append(dict(name=name,max_abs_error=float((value-expected).abs().max())))
                    comparison=dict(passed=True,forward_atol=1e-6,forward_rtol=0.,gradient_atol=1e-6,gradient_rtol=1e-4,
                        logits_max_abs_error=float((numerical['logits']-reference_numerical['logits']).abs().max()),
                        gradient_max_abs_error=max(r['max_abs_error'] for r in gradient_rows),per_parameter=gradient_rows)
                record=dict(kind=kind,phase=phase,repeat=repeat,loss=float(loss),source_and_model_transfer_seconds=transfer_seconds,
                    execution_setup_seconds=setup_seconds,forward_seconds=forward_seconds,backward_seconds=backward_seconds,
                    GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    process_RSS_bytes=rss(),comparison=comparison,all55_gradients_finite=True,optimizer_updates=0)
                records.append(record);write(f'{phase}_{repeat}_{kind}.json',record)
                print(json.dumps({k:v for k,v in record.items() if k!='comparison'}),flush=True)
                if kind=='monolithic':del device_bank,device_plan
                else:del execution
                del model,logits,loss,numerical;gc.collect();torch.cuda.empty_cache();check('after_'+kind)
        for pin in pins:need(reference(pin['path'])==pin,'Qualified dependency changed during diagnostic')
        need(reference(__file__)==own_ref,'Diagnostic source changed during execution')
        result=dict(schema='TreeLoom_Transport_observed128_GPU_numerical_qualification_v1',status='passed',scope=scope,
            source_ref=own_ref,records=records,elapsed_seconds=time.monotonic()-start,
            seed=70001,fresh_Transport_parameters=29315,strict_numerical_gates_passed=True,
            original_TRAIN_focals=22,TRAIN_targets_used=True,optimizer_created=False,optimizer_updates=0,
            VAL_truth_opened=False,TEST_opened=False,full_original_parent=False,
            baseline_accuracy_comparison=False,ancestry_predictive_quality_established=False,useful_GPU_duty_measured=False,
            limitations=['Same randomly initialized candidate, two execution routes; not current TreeLoom versus Transport.',
                         '128-tree prefix has truncated external genomic context and only22 original frozen focal targets.',
                         'No full-parent200M-node resource proof, optimization trajectory or accuracy-preservation claim.',
                         'One first/warm diagnostic timing is not an end-to-end efficiency benchmark.'])
        write('completed.json',result);print(json.dumps(dict(status='passed',output=str(output)),flush=True))
    except BaseException as error:
        write('failed.json',dict(status='failed',error_type=type(error).__name__,message=str(error),elapsed_seconds=time.monotonic()-start))
        raise


if __name__=='__main__':main()
