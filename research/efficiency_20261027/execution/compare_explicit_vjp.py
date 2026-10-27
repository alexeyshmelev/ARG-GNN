"""ROOT-admitted integrated versus explicit-VJP slice/full update comparison.

Reuses reviewed full-run memory accounting and the authenticated preparation
path. No launcher, checkpoint writer, live-source mutation or disk cache.
"""
from contextlib import contextmanager
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
    ROOT, CHECKPOINT, GIB, need, ref, rss, write, prepare)
from research.efficiency_20261027.execution.compare_full_context import (
    MemoryProfile as AllocatorProfile, compare_state, source_names as full_source_names)

KINDS = ("integrated", "explicit_vjp")


class MemoryProfile(AllocatorProfile):
    """Add wall-clock intervals for alignment with ROOT device telemetry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_stages = []

    @contextmanager
    def stage(self, name):
        top = name.startswith("profile_")
        started_unix = time.time() if top else None
        try:
            with super().stage(name):
                yield
        finally:
            if top:
                ended_unix = time.time()
                row = dict(identity=self.identity, stage=name,
                    wall_start_unix=started_unix, wall_end_unix=ended_unix,
                    wall_elapsed_seconds=ended_unix-started_unix)
                self.wall_stages.append(row)
                with (self.journal.parent/"stage_clock.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                print(json.dumps(dict(event="stage_clock", **row)), flush=True)

    def snapshot(self):
        result = super().snapshot()
        result["stage_wall_clock"] = self.wall_stages
        result["clock_scope"] = "Unix wall-clock intervals include top-level synchronization and receipt emission; monotonic stage timings are recorded separately."
        return result

def run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    import torch
    from research.scalability_20261024.integration.executors import build_executor
    from research.efficiency_20261027.execution.explicit_vjp import build_executor as explicit_factory
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
        decode_parent_backward, geometry_from_spans)
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS and torch.get_num_threads() == 8, "Explicit integrated/explicitVJP CPU8 comparison")
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
            kwargs = dict(layout_cache=cache, frozen_module=frozen, holds_per_block=args.holds_per_block,
                haplotypes_per_gru=args.haplotypes_per_gru, history_holds_per_window=1024,
                resource_check=check, profile=profile)
            executor = (explicit_factory(model, bundle, metadata, **kwargs) if kind == "explicit_vjp"
                else build_executor("integrated", model, bundle, metadata, execution="staged", **kwargs))
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
            stages=profile.snapshot(), torch_peak_allocated_bytes=profile.run_allocated_peak,
            torch_peak_reserved_bytes=profile.run_reserved_peak, RSS_bytes=rss(),
            process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            initial_physical_free_bytes=free, physical_total_bytes=total, cache_snapshot=cache.snapshot(),
            explicit_vjp=executor.explicit_vjp_snapshot() if hasattr(executor, "explicit_vjp_snapshot") else None)
        return row, numerical
    finally:
        if hook is not None:
            hook.remove()
        context = vjp = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if executor is not None:
            executor.release_layout()
        executor = model = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("slice", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holds-per-block", type=int, default=128)
    parser.add_argument("--haplotypes-per-gru", type=int, default=64)
    parser.add_argument("--timed-repeats", type=int, default=6)
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=8)
    parser.add_argument("--max-rss-gib", type=float, default=24)
    args = parser.parse_args()
    need(1 <= args.holds_per_block <= 256 and 1 <= args.haplotypes_per_gru <= 128,
         "Original bounded partitions required")
    need(6 <= args.timed_repeats <= 12 and args.timed_repeats % 2 == 0,
         "Six to twelve balanced slice timing rounds")
    need(0 <= args.warm_repeats <= 2 and 0 < args.gpu_memory_gib <= 64,
         "Bounded full repeats and at most64GiB soft device budget")
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3", "4", "5")
        and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
        and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
        and os.environ.get("ARGGNN_GPU_MEMORY_POLICY") == "full_device",
        "ROOT-selected one GPU3/4/5 UUID and full-device policy required")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh, training_v2 as validation
    from research.scalability_20261024.integration.cache import LayoutCache
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT/"research/efficiency_20261027") and not output.exists(),
         "Fresh efficiency comparison output required")
    output.mkdir(parents=True)
    def check(stage):
        need(rss() <= args.max_rss_gib*GIB, "RSS guard exceeded at "+stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= args.gpu_memory_gib*GIB, "GPU soft budget exceeded at "+stage)
    def make_cache(cap):
        return LayoutCache(max_bytes=cap, admit=lambda amount, stage:
            need(rss()+amount <= args.max_rss_gib*GIB, "Layout working reserve exceeded at "+stage))
    names = full_source_names()+["research/efficiency_20261027/execution/explicit_vjp.py",
        "research/efficiency_20261027/execution/compare_explicit_vjp.py"]
    sources = [ref(name) for name in names]
    started = time.perf_counter()
    write(output/"configuration.json", dict(schema="TreeLoom_explicitVJP_compare_config_v1",
        arguments=vars(args), CPU_threads=8, source_refs=sources, checkpoint_ref=ref(CHECKPOINT),
        external_guardian_required=True, hard_allocator_cap=False))
    try:
        need(ref(CHECKPOINT)["sha256"] == "12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24", "Exact241 checkpoint")
        saved = torch.load(ROOT/CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        registry = fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
        rt = fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved, rt.build_model(), registry, expected_steps=241)
        if args.mode == "full":
            from research.preprocessing_optimization_20261026.integration.hooks import install_tree
            install_tree(preparation_workers=4, max_rss_bytes=int(args.max_rss_gib*GIB))
        need(not torch.cuda.is_initialized(), "Prepare before CUDA initialization")
        bundle, metadata, labels, frozen, scope = prepare(args, rt, saved, check)
        if args.mode == "full":
            need(bundle.native_holds == 77472 and len(labels) == 10000,
                 "Exact full TRAIN000 support and original targets")
        write(output/"scope.json", scope)
        warmup_rows = []
        if args.mode == "full":
            slice_args = SimpleNamespace(**dict(vars(args), mode="slice"))
            sb, sm, sl, sf, ss = prepare(slice_args, rt, saved, check)
            slice_baseline = None
            for kind in KINDS:
                cache = make_cache(64*2**20)
                row, numerical = run_update(kind, "slice_warmup", 0, args, rt, saved,
                    sb, sm, sl, sf, cache, check, output)
                if slice_baseline is None:
                    slice_baseline = numerical
                row["comparison"] = compare_state(slice_baseline, numerical)
                warmup_rows.append(row)
                write(output/(kind+"_slice_warmup.json"), row)
                need(row["comparison"]["passed"], "Slice warmup parity failed")
                del cache
            del sb, sm, sl, sf, slice_baseline, numerical
            gc.collect()
            torch.cuda.empty_cache()
            schedule = [("full_first_call", 0, KINDS)] + [("full_warm_repeat", r,
                tuple(reversed(KINDS)) if r % 2 else KINDS) for r in range(1, args.warm_repeats+1)]
        else:
            schedule = [("slice_warmup", 0, KINDS), ("slice_warmup", 1, tuple(reversed(KINDS)))]
            schedule += [("slice_timed", r, KINDS if r % 2 else tuple(reversed(KINDS)))
                for r in range(1, args.timed_repeats+1)]
        caches = {kind: make_cache(8*GIB if args.mode == "full" else 64*2**20) for kind in KINDS}
        rows, baseline = [], None
        for phase, repeat, order in schedule:
            for position, kind in enumerate(order):
                need([ref(name) for name in names] == sources, "Source changed before update")
                row, numerical = run_update(kind, phase, repeat, args, rt, saved,
                    bundle, metadata, labels, frozen, caches[kind], check, output)
                if baseline is None:
                    baseline = numerical
                row["comparison"] = compare_state(baseline, numerical)
                row["order_position"] = position
                rows.append(row)
                write(output/(kind+"_"+phase+"_"+str(repeat)+".json"), row)
                need(row["comparison"]["passed"], "Numerical or argmax parity failed")
        need([ref(name) for name in names] == sources, "Source changed during qualification")
        summaries = {}
        for kind in KINDS:
            timed = [row for row in rows if row["kind"] == kind and row["phase"] != "slice_warmup"]
            summaries[kind] = dict(calls=len(timed), median_update_seconds=statistics.median(row["update_seconds"] for row in timed),
                max_allocated_bytes=max(row["torch_peak_allocated_bytes"] for row in timed),
                max_reserved_bytes=max(row["torch_peak_reserved_bytes"] for row in timed))
        result = dict(schema="TreeLoom_explicitVJP_compare_v1", status="completed", scope=scope,
            source_refs=sources, checkpoint_ref=ref(CHECKPOINT), FP32=True, CPU_threads=8,
            warmup_records=warmup_rows, records=rows, summaries=summaries,
            physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"], visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
            total_seconds=time.perf_counter()-started, external_guardian_required=True,
            soft_GPU_budget_bytes=args.gpu_memory_gib*GIB, max_RSS_bytes=args.max_rss_gib*GIB,
            execution_partitions=dict(holds_per_block=args.holds_per_block,
                haplotypes_per_gru=args.haplotypes_per_gru, history_window=1024),
            caveats=["Every call restores241 parameters, AdamW and RNG; one clip and optimizer step after original decoder and complete trunk VJPs.",
                "CPU context is retained. ExplicitVJP reorders FP32 parameter accumulation and preserves the full mathematical derivative.",
                "Slice timing uses two warmup rounds and balanced timed order; restricted128hold history is not fullparent evidence.",
                "Full mode uses one slice warmup per arm and independent cold full layout caches; one full call per arm is qualification, not stable throughput statistics.",
                "Stage allocator peaks are exact intervals; boundary GPU allocations in explicit_vjp are sparse samples, not true peaks.",
                "Production checkpoint, observed inputs and held-out truth are unchanged."])
        write(output/"completed.json", result)
        print(json.dumps(dict(status="completed", completion_ref=ref(output/"completed.json"), summaries=summaries), indent=2))
    except BaseException as error:
        write(output/"failed.json", dict(status="failed", error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.perf_counter()-started, source_refs_at_start=sources, partial_receipts_preserved=True))
        raise


if __name__ == "__main__":
    main()
