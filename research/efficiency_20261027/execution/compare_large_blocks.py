"""ROOT-admitted observed-prefix update qualification for larger exact blocks.

Uses the complete declared 1024/1153-hold slice history in every arm.
All context boundaries remain CPU to isolate partition effects. No launcher,
production binding, dependency mutation or observation packet cache.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import gc
import json
import os
import resource
import statistics
import time
from research.efficiency_20261027.root.compare_host_parallelism import (
    ROOT, CHECKPOINT, GIB, need, ref, rss, write)
from research.efficiency_20261027.execution.compare_full_context import (
    compare_state, source_names as full_source_names)
from research.efficiency_20261027.execution.compare_explicit_resident_vjp import (
    MemoryProfile, failure_record)
from research.efficiency_20261027.execution.prepare_large_slice import prepare_large_prefix

KINDS=("explicit_128","explicit_512","explicit_1024")
BLOCK_SIZES=dict(explicit_128=128,explicit_512=512,explicit_1024=1024)


def schedule(timed_repeats):
    need(timed_repeats in (0,6),"One qualification call per arm and optional six balanced rounds")
    rows=[("qualification",0,KINDS)]
    if timed_repeats:
        a,b,c=KINDS
        orders=((a,b,c),(c,b,a),(b,c,a),(a,c,b),(c,a,b),(b,a,c))
        rows += [("timed",repeat,order) for repeat,order in enumerate(orders,start=1)]
    return rows


def source_names():
    return list(dict.fromkeys(full_source_names()+[
        "research/efficiency_20261027/execution/"+name for name in (
            "explicit_vjp.py","explicit_resident_vjp.py","compare_explicit_resident_vjp.py",
            "large_layout_plan.py","large_layout_cache.py","large_block_executors.py",
            "prepare_large_slice.py","compare_large_blocks.py")]+[
        "research/preprocessing_optimization_20261026/tree/kernels.py",
        "research/repairs/benchmark_recovery_20261020/runtime/episodes.py",
        "research/repairs/benchmark_recovery_20261020/runtime/cuts.py",
        "src/arggnn/american_c3_gnn_dataset_v1.py"]))

def run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    import torch
    from research.efficiency_20261027.execution.large_block_executors import build_executor as large_factory
    from research.efficiency_20261027.execution.explicit_vjp import build_executor as explicit_factory
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
        decode_parent_backward, geometry_from_spans)
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS and torch.get_num_threads() == 8, "Explicit 128/512/1024 hold CPU8 comparison")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    identity = dict(kind=kind, phase=phase, repeat=repeat)
    profile = MemoryProfile(torch, check, journal=output/"stage_memory.jsonl", identity=identity,
        GPU_budget_bytes=int(args.gpu_memory_gib*GIB))
    full_context_bytes = bundle.native_holds*1600*32*4
    free, total = torch.cuda.mem_get_info()
    update_wall_start = time.time()
    started = time.perf_counter()
    model = optimizer = executor = hook = context = vjp = None
    logits = []
    admission = None
    failed = False
    try:
        with profile.stage("profile_model_optimizer_transfer"):
            model = rt.build_model()
            model.load_state_dict(saved["model_state"], strict=True)
            model = model.to(device="cuda:0", dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                betas=(.9, .999), eps=1e-8, maximize=False, amsgrad=False, foreach=False,
                fused=False, capturable=False, differentiable=False)
            optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
            restore_random_state(saved["random_state"])
        hook = model.classifier.register_forward_hook(lambda module, inputs, value: logits.append(value.detach().cpu()))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with profile.stage("profile_layout"):
            kwargs = dict(layout_cache=cache, frozen_module=frozen, holds_per_block=BLOCK_SIZES[kind],
                haplotypes_per_gru=args.haplotypes_per_gru, history_holds_per_window=1024,
                resource_check=check, profile=profile)
            executor = (explicit_factory(model, bundle, metadata, **kwargs) if kind == "explicit_128"
                else large_factory(model, bundle, metadata, **kwargs))
        with torch.autocast(device_type="cuda", enabled=False):
            with profile.stage("profile_context_forward"):
                context = executor.context_forward()
                need(context.shape == (bundle.native_holds, 1600, 32), "Complete original context shape")
            context_boundary = dict(device=str(context.device), bytes=context.numel()*context.element_size())
            with profile.stage("profile_decoder_forward_backward"):
                decode = decode_parent_backward
                loss, vjp, decoder = decode(model, context, geometry_from_spans(bundle.bank.spans),
                    bundle.focal_hold, bundle.focal_sample_column, labels,
                    tile_haplotypes=args.haplotypes_per_gru, profile=profile, resource_check=check)
                need(vjp.shape == context.shape and vjp.device == context.device,
                     "Complete context VJP on the chosen boundary device")
            VJP_boundary = dict(device=str(vjp.device), bytes=vjp.numel()*vjp.element_size())
            with profile.stage("profile_trunk_backward"):
                context.backward(vjp)
        with profile.stage("profile_gradient_capture"):
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
            need(len(gradients) == 49 and all(bool(torch.isfinite(g).all()) for g in gradients.values()),
                 "All49 finite connected original parameter gradients")
        with profile.stage("profile_clip_optimizer"):
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        check("full_comparison_update_complete")
        torch.cuda.synchronize()
        update_seconds = time.perf_counter()-started
        update_wall_end = time.time()
        with profile.stage("profile_state_capture"):
            parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            moments = {str(i): {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in row.items()} for i, row in optimizer.state_dict()["state"].items()}
            need(len(moments) == 49 and all(int(row["step"].item()) == 242 for row in moments.values()),
                 "Every original AdamW step must become242")
            numerical = dict(logits=torch.cat(logits), loss=loss.detach().cpu(), gradients=gradients,
                parameters=parameters, moments=moments)
            need(numerical["logits"].shape == (len(labels), 3), "Every original target logit captured once")
        row = dict(**identity, CPU_threads=8, update_seconds=update_seconds,
            update_wall_start_unix=update_wall_start, update_wall_end_unix=update_wall_end,
            protocol_wall_end_unix=time.time(),
            protocol_seconds=time.perf_counter()-started, loss=float(loss), gradient_norm=float(norm),
            decoder=decoder, context_boundary=context_boundary, VJP_boundary=VJP_boundary,
            holds_per_block=BLOCK_SIZES[kind],
            stages=profile.snapshot(), torch_peak_allocated_bytes=profile.run_allocated_peak,
            torch_peak_reserved_bytes=profile.run_reserved_peak, RSS_bytes=rss(),
            process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            initial_physical_free_bytes=free, physical_total_bytes=total, cache_snapshot=cache.snapshot(),
            explicit_vjp=executor.explicit_vjp_snapshot() if hasattr(executor, "explicit_vjp_snapshot") else None)
        return row, numerical
    except BaseException as error:
        failed = True
        write(output/("failed_update_"+kind+"_"+phase+"_"+str(repeat)+".json"),
            failure_record(error, identity=identity, stages=profile.snapshot(),
                holds_per_block=BLOCK_SIZES[kind], elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        cleanup_errors = []
        for label, action in (("remove_hook", None if hook is None else hook.remove),
            ("clear_gradients", None if optimizer is None else lambda: optimizer.zero_grad(set_to_none=True)),
            ("release_layout", None if executor is None else getattr(executor, "release_layout", None))):
            if action is not None:
                try:
                    action()
                except BaseException as error:
                    cleanup_errors.append(dict(action=label, error=str(error)))
        context = vjp = executor = model = optimizer = hook = action = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except BaseException as error:
            cleanup_errors.append(dict(action="allocator_cleanup", error=str(error)))
        if cleanup_errors:
            write(output/("cleanup_"+kind+"_"+phase+"_"+str(repeat)+".json"),
                dict(identity=identity, errors=cleanup_errors, primary_failure_preserved=failed))
            if not failed:
                raise RuntimeError("Update cleanup failed; see cleanup receipt")


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--slice-holds",type=int,choices=(1024,1153),default=1024)
    parser.add_argument("--haplotypes-per-gru",type=int,default=64)
    parser.add_argument("--timed-repeats",type=int,choices=(0,6),default=0)
    parser.add_argument("--gpu-memory-gib",type=float,default=16)
    parser.add_argument("--max-rss-gib",type=float,default=24)
    args=parser.parse_args()
    need(1<=args.haplotypes_per_gru<=128 and 0<args.gpu_memory_gib<=64
         and args.max_rss_gib>0,"Explicit bounded decoder and memory budgets")
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3","4","5")
         and os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-")
         and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
         and os.environ.get("ARGGNN_GPU_MEMORY_POLICY")=="full_device",
         "ROOT-selected one admitted GPU UUID and full-device policy required")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh, training_v2 as validation
    from research.scalability_20261024.integration.cache import LayoutCache
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    output=Path(args.output).resolve()
    need(output.is_relative_to(ROOT/"research/efficiency_20261027") and not output.exists(),
         "Fresh isolated efficiency output required")
    output.mkdir(parents=True)
    def check(stage):
        need(rss()<=args.max_rss_gib*GIB,"RSS guard exceeded at "+stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved()<=args.gpu_memory_gib*GIB,"GPU soft budget exceeded at "+stage)
    def make_cache(kind):
        cls=LayoutCache if kind=="explicit_128" else LargeBlockLayoutCache
        return cls(max_bytes=256*2**20,admit=lambda amount,stage:
            need(rss()+amount<=args.max_rss_gib*GIB,"Layout working reserve exceeded at "+stage))
    names=source_names()
    sources=[ref(name) for name in names]
    started=time.perf_counter()
    write(output/"configuration.json",dict(schema="TreeLoom_larger_blocks_config_v1",
        arguments=vars(args),source_refs=sources,checkpoint_ref=ref(CHECKPOINT),
        CPU_threads=8,context_residency="CPU",external_guardian_required=True,
        preparation_child_and_IPC_reserve_bytes=8*GIB,
        hard_allocator_cap=False,automatic_smaller_block_fallback=False))
    try:
        need(ref(CHECKPOINT)["sha256"]=="12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24",
             "Exact checkpoint 241 required")
        saved=torch.load(ROOT/CHECKPOINT,map_location="cpu",weights_only=True,mmap=True)
        registry=fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
        rt=fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved,rt.build_model(),registry,expected_steps=241)
        bundle,metadata,labels,frozen,scope=prepare_large_prefix(args,rt,saved,check)
        write(output/"scope.json",scope)
        counts=metadata["episode_metadata"]["native_nodes_per_tree"]
        write(output/"partition_sizes.json",{kind:dict(holds_per_block=size,
            block_count=(len(counts)+size-1)//size,
            maximum_nodes=max(sum(counts[a:a+size]) for a in range(0,len(counts),size)))
            for kind,size in BLOCK_SIZES.items()})
        caches={kind:make_cache(kind) for kind in KINDS}
        rows=[];baseline=None
        for phase,repeat,order in schedule(args.timed_repeats):
            for position,kind in enumerate(order):
                need([ref(name) for name in names]==sources,"Source changed before update")
                row,numerical=run_update(kind,phase,repeat,args,rt,saved,
                    bundle,metadata,labels,frozen,caches[kind],check,output)
                if baseline is None:
                    baseline=numerical
                row["comparison"]=compare_state(baseline,numerical)
                row["order_position"]=position
                rows.append(row)
                write(output/(kind+"_"+phase+"_"+str(repeat)+".json"),row)
                need(row["comparison"]["passed"],"Strict numerical or zero-argmax qualification failed")
        need([ref(name) for name in names]==sources,"Source changed during qualification")
        summary={}
        for kind in KINDS:
            subset=[r for r in rows if r["kind"]==kind]
            timed=[r for r in subset if r["phase"]=="timed"]
            summary[kind]=dict(calls=len(subset),
                qualification_update_seconds=subset[0]["update_seconds"],
                timed_median_update_seconds=None if not timed else statistics.median(r["update_seconds"] for r in timed),
                max_allocated_bytes=max(r["torch_peak_allocated_bytes"] for r in subset),
                max_reserved_bytes=max(r["torch_peak_reserved_bytes"] for r in subset))
        result=dict(schema="TreeLoom_larger_blocks_GPU_qualification_v1",status="completed",
            source_refs=sources,checkpoint_ref=ref(CHECKPOINT),scope=scope,
            records=rows,summaries=summary,total_seconds=time.perf_counter()-started,
            physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"],visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
            context_residency="CPU",FP32=True,CPU_threads=8,original_site_chunk_rows=256,
            original_complete_slice_history=True,external_guardian_required=True,
            soft_GPU_budget_bytes=int(args.gpu_memory_gib*GIB),max_RSS_bytes=int(args.max_rss_gib*GIB),
            caveats=["Every arm uses the identical observed prefix and original selected TRAIN labels; full parent history outside the prefix is absent.",
                "Baseline is frozen explicit VJP at 128 holds; larger arms change only immutable layout grouping and the admitted partition guard.",
                "Every call restores checkpoint 241, AdamW and RNG; original decoder, full trunk VJP, 49 gradients, single clip and AdamW step 242 are checked.",
                "One qualification call per arm measures actual activation peaks and parity, not steady throughput. Optional six timed permutations balance order positions.",
                "GPU peaks from this prefix do not admit full-parent residency or a different topology/hold count. No automatic smaller-workload fallback."])
        write(output/"completed.json",result)
        print(json.dumps(dict(status="completed",completion_ref=ref(output/"completed.json"),summaries=summary),indent=2))
    except BaseException as error:
        write(output/"failed.json",failure_record(error,elapsed_seconds=time.perf_counter()-started,
            source_refs_at_start=sources))
        raise


if __name__=="__main__":
    main()
