"""ROOT-admitted original and selected large-block explicit VJP full update comparison.

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

KINDS = ("original", "explicit_cpu_vjp", "explicit_resident_vjp")


def resident_admission(native_holds, free_bytes, soft_budget_bytes, reserve_gib):
    context_bytes = native_holds*1600*32*4
    full = native_holds == 77472
    context_budget = max(2*context_bytes, 32*GIB if full else 0)
    need(reserve_gib >= (8 if full else 1), "Explicit resident scratch reserve required")
    required = context_budget + int(reserve_gib*GIB)
    result = dict(context_bytes=context_bytes, two_context_buffers_bytes=2*context_bytes,
        rounded_context_budget_bytes=context_budget, scratch_reserve_bytes=int(reserve_gib*GIB),
        required_free_and_soft_budget_bytes=required, physical_free_bytes=free_bytes,
        soft_budget_bytes=soft_budget_bytes, allocation_guarantee=False)
    need(required <= soft_budget_bytes, "Resident context and scratch exceed soft budget")
    need(required <= free_bytes, "Resident context and scratch exceed physical free memory")
    return result


def slice_schedule(timed_repeats):
    need(timed_repeats in (6, 12),
         "Six or twelve position-balanced slice timing rounds")
    a, b, c = KINDS
    orders = ((a,b,c), (c,b,a), (b,c,a), (a,c,b), (c,a,b), (b,a,c))
    return [("slice_warmup", 0, KINDS), ("slice_warmup", 1, tuple(reversed(KINDS)))] + [
        ("slice_timed", repeat, orders[(repeat-1) % 6])
        for repeat in range(1, timed_repeats+1)]


def failure_record(error, **values):
    return dict(status="failed", error_type=type(error).__name__, error=str(error),
        CUDA_OOM=type(error).__name__ == "OutOfMemoryError", wall_time_unix=time.time(),
        partial_receipts_preserved=True, retry_attempted=False, **values)


class MemoryProfile(AllocatorProfile):
    """Exact stage peaks and clocks; preserve primary failures if telemetry fails."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wall_stages = []

    @contextmanager
    def stage(self, name):
        if not name.startswith("profile_"):
            with super().stage(name):
                yield
            return
        need(not self.top_active, "Top-level memory intervals must not nest")
        cuda = self.torch.cuda
        before = after = None
        primary = telemetry_error = None
        success = False
        started_unix, started = time.time(), time.perf_counter()
        self.top_active = True
        try:
            cuda.synchronize()
            before = self._sample()
            self._retain(before)
            cuda.reset_peak_memory_stats()
            print(json.dumps(dict(event="stage_begin", stage=name)), flush=True)
            yield
            success = True
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                cuda.synchronize()
                after = self._sample()
                self._retain(after)
            except BaseException as error:
                telemetry_error = error
            elapsed = time.perf_counter()-started
            ended_unix = time.time()
            timing = self.timings.setdefault(name, dict(calls=0, total_seconds=0., max_seconds=0.))
            timing["calls"] += 1
            timing["total_seconds"] += elapsed
            timing["max_seconds"] = max(timing["max_seconds"], elapsed)
            row = dict(identity=self.identity, stage=name, completed=success,
                before=before, after=after,
                interval_peak_allocated_bytes=None if after is None else after["peak_allocated_bytes"],
                interval_peak_reserved_bytes=None if after is None else after["peak_reserved_bytes"],
                RSS_bytes=rss(), error=None if primary is None else failure_record(primary),
                telemetry_error=None if telemetry_error is None else str(telemetry_error))
            clock = dict(identity=self.identity, stage=name, wall_start_unix=started_unix,
                wall_end_unix=ended_unix, wall_elapsed_seconds=ended_unix-started_unix,
                completed=success)
            self.memory_stages.append(row)
            self.wall_stages.append(clock)
            self.top_active = False
            try:
                with self.journal.open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                with (self.journal.parent/"stage_clock.jsonl").open("a") as stream:
                    stream.write(json.dumps(clock, allow_nan=False)+"\n")
                print(json.dumps(dict(event="stage_memory", **row)), flush=True)
                print(json.dumps(dict(event="stage_clock", **clock)), flush=True)
            except BaseException:
                if primary is None:
                    raise
            if primary is None:
                if telemetry_error is not None:
                    raise telemetry_error
                need(after["peak_reserved_bytes"] <= self.GPU_budget_bytes,
                     "Stage transient reserved-memory peak exceeded soft budget at "+name)

    def snapshot(self):
        result = super().snapshot()
        result["stage_wall_clock"] = self.wall_stages
        result["clock_scope"] = "Unix stage intervals include synchronization and exclude subsequent journal emission. Failed telemetry is explicit and cannot replace the primary execution failure."
        return result


def run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    import torch
    from research.scalability_20261024.integration.executors import build_executor
    from research.efficiency_20261027.execution.large_block_executors import build_executor as explicit_factory
    from research.efficiency_20261027.root.context_residency import decode_resident_parent_backward
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
        decode_parent_backward, geometry_from_spans)
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS, "Original128 and selected CPU/resident explicit VJP comparison")
    cpu_threads = 1 if kind == "original" else 8
    torch.set_num_threads(cpu_threads)
    need(torch.get_num_threads() == cpu_threads, "Recorded per-arm CPU threads required")
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
        if kind == "explicit_resident_vjp":
            with profile.stage("profile_resident_admission"):
                admission = resident_admission(bundle.native_holds, free,
                    int(args.gpu_memory_gib*GIB), args.minimum_free_reserve_gib)
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
            block_size = 128 if kind == "original" else args.holds_per_block
            kwargs = dict(layout_cache=cache, frozen_module=frozen, holds_per_block=block_size,
                haplotypes_per_gru=args.haplotypes_per_gru, history_holds_per_window=1024,
                resource_check=check, profile=profile)
            executor = (build_executor("original", model, bundle, metadata, execution="staged", **kwargs)
                if kind == "original" else explicit_factory(model, bundle, metadata,
                    resident_context=kind == "explicit_resident_vjp", **kwargs))
        with torch.autocast(device_type="cuda", enabled=False):
            with profile.stage("profile_context_forward"):
                context = executor.context_forward()
                need(context.shape == (bundle.native_holds, 1600, 32), "Complete original context shape")
            context_boundary = dict(device=str(context.device), bytes=context.numel()*context.element_size())
            with profile.stage("profile_decoder_forward_backward"):
                decode = decode_resident_parent_backward if kind == "explicit_resident_vjp" else decode_parent_backward
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
        row = dict(**identity, CPU_threads=cpu_threads, update_seconds=update_seconds,
            update_wall_start_unix=update_wall_start, update_wall_end_unix=update_wall_end,
            protocol_wall_end_unix=time.time(),
            protocol_seconds=time.perf_counter()-started, loss=float(loss), gradient_norm=float(norm),
            decoder=decoder, context_boundary=context_boundary, VJP_boundary=VJP_boundary,
            resident_admission=admission, holds_per_block=block_size,
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
                resident_admission=admission, elapsed_seconds=time.perf_counter()-started))
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
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("full",), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holds-per-block", type=int)
    parser.add_argument("--haplotypes-per-gru", type=int, default=64)
    parser.add_argument("--timed-repeats", type=int, default=6)
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=8)
    parser.add_argument("--max-rss-gib", type=float, default=24)
    parser.add_argument("--minimum-free-reserve-gib", type=float)
    args = parser.parse_args()
    if args.holds_per_block is None:
        args.holds_per_block = 32 if args.mode == "slice" else 128
    if args.minimum_free_reserve_gib is None:
        args.minimum_free_reserve_gib = 1 if args.mode == "slice" else 8
    if args.mode == "full":
        resident_admission(77472, 80*GIB, int(args.gpu_memory_gib*GIB), args.minimum_free_reserve_gib)
    need(args.holds_per_block in (128, 512, 1024) and args.haplotypes_per_gru == 64,
         "Explicitly qualified block size and unchanged64 decoder tile required")
    need(args.timed_repeats in (6, 12),
         "Six or twelve position-balanced slice timing rounds")
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
    from research.efficiency_20261027.execution.large_layout_cache import LargeBlockLayoutCache
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
    def make_cache(cap, kind):
        cls = LayoutCache if kind == "original" else LargeBlockLayoutCache
        return cls(max_bytes=cap, admit=lambda amount, stage:
            need(rss()+amount <= args.max_rss_gib*GIB, "Layout working reserve exceeded at "+stage))
    names = list(dict.fromkeys(full_source_names()+["research/efficiency_20261027/execution/"+name
        for name in ("explicit_vjp.py", "explicit_resident_vjp.py", "large_layout_plan.py",
            "large_layout_cache.py", "large_block_executors.py", "compare_explicit_resident_vjp.py")]+[
            "research/efficiency_20261027/root/compare_selected_full.py"]))
    attempt_job = ".runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt5.json"
    attempt = json.loads((ROOT / attempt_job).read_text())
    original_pins = attempt["source_refs"]
    need(len(original_pins) == 157, "Exact active attempt5 dependency closure required")
    for pin in original_pins:
        need(ref(pin["path"]) == pin, "Frozen production dependency changed: " + pin["path"])
    names = list(dict.fromkeys(names + [attempt_job] + [pin["path"] for pin in original_pins]))
    sources = [ref(name) for name in names]
    started = time.perf_counter()
    write(output/"configuration.json", dict(schema="TreeLoom_selected_large_VJP_compare_config_v1",
        arguments=vars(args), CPU_threads_by_kind=dict(original=1, explicit_cpu_vjp=8, explicit_resident_vjp=8), source_refs=sources, checkpoint_ref=ref(CHECKPOINT),
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
                cache = make_cache(64*2**20, kind)
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
            schedule = slice_schedule(args.timed_repeats)
        caches = {kind: make_cache(8*GIB, kind) for kind in KINDS}
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
        result = dict(schema="TreeLoom_selected_large_VJP_compare_v1", status="completed", scope=scope,
            source_refs=sources, checkpoint_ref=ref(CHECKPOINT), FP32=True, CPU_threads_by_kind=dict(original=1, explicit_cpu_vjp=8, explicit_resident_vjp=8),
            warmup_records=warmup_rows, records=rows, summaries=summaries,
            physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"], visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
            total_seconds=time.perf_counter()-started, external_guardian_required=True,
            soft_GPU_budget_bytes=args.gpu_memory_gib*GIB, max_RSS_bytes=args.max_rss_gib*GIB,
            minimum_resident_scratch_reserve_bytes=int(args.minimum_free_reserve_gib*GIB),
            execution_partitions=dict(original_holds_per_block=128, selected_holds_per_block=args.holds_per_block,
                haplotypes_per_gru=args.haplotypes_per_gru, history_window=1024),
            caveats=["Every call restores241 parameters, AdamW and RNG; one clip and optimizer step after original decoder and complete trunk VJPs.",
                "Both selected candidates use the same admitted larger block size. Original execution retains128-hold grouping. No genomic history or supervised targets are removed.",
                "Resident candidate keeps complete context and VJP on CUDA, first-state/history boundaries on CPU. Explicit VJP changes FP32 parameter accumulation order.",
                "Original128 usesCPU1 to match the active production setting. Both selected candidates useCPU8. Every timing states its own thread count.",
                "Full mode uses one slice warmup per arm and independent cold full layout caches; one full call per arm is qualification, not stable throughput statistics.",
                "Stage allocator peaks are exact intervals; boundary GPU allocations in explicit_vjp are sparse samples, not true peaks.",
                "Production checkpoint, observed inputs and held-out truth are unchanged."])
        write(output/"completed.json", result)
        print(json.dumps(dict(status="completed", completion_ref=ref(output/"completed.json"), summaries=summaries), indent=2))
    except BaseException as error:
        write(output/"failed.json", failure_record(error, elapsed_seconds=time.perf_counter()-started, source_refs_at_start=sources))
        raise


if __name__ == "__main__":
    main()
