"""ROOT-admitted full TRAIN000 integrated versus resident-context qualification.

No launcher, production binding or checkpoint writer. Prepare once with the
qualified four-process CPU encoder; warm CUDA libraries on the authenticated
128-hold slice; compare full updates with independent per-arm layout caches.
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
import time

from research.efficiency_20261027.root.compare_host_parallelism import (
    ROOT, CHECKPOINT, GIB, Profile, need, ref, rss, write, prepare)

KINDS = ("integrated", "context_gpu")


def full_schedule(warm_repeats=0):
    need(type(warm_repeats) is int and 0 <= warm_repeats <= 2,
         "At most two explicit warm full-parent repeats")
    return [("full_first_call", 0, KINDS)] + [
        ("full_warm_repeat", repeat, tuple(reversed(KINDS)) if repeat % 2 else KINDS)
        for repeat in range(1, warm_repeats+1)]


def compare_state(reference, candidate):
    """Return exhaustive strict metrics; caller writes them before rejecting."""
    import torch
    result = dict(passed=True, categories={})
    for category in ("logits", "loss", "gradients", "parameters", "moments"):
        atol, rtol = 1e-6, 1e-5 if category in ("logits", "loss") else 1e-4
        summary = dict(atol=atol, rtol=rtol, max_abs_error=0., tensors=0,
            elements=0, mismatched_elements=0, mismatched_paths=[], structural_errors=[])
        def visit(a, b, path):
            if isinstance(a, dict):
                if not isinstance(b, dict) or set(a) != set(b):
                    summary["structural_errors"].append(path+": dictionary keys")
                    return
                for key in a:
                    visit(a[key], b[key], path+"/"+str(key))
            elif torch.is_tensor(a):
                if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                    summary["structural_errors"].append(path+": tensor shape or dtype")
                    return
                summary["tensors"] += 1
                summary["elements"] += a.numel()
                mismatch = int((~torch.isclose(a, b, atol=atol, rtol=rtol)).sum())
                if a.numel():
                    difference = (a-b).abs()
                    if bool(torch.isfinite(difference).all()):
                        summary["max_abs_error"] = max(summary["max_abs_error"], float(difference.max()))
                    else:
                        summary["structural_errors"].append(path+": nonfinite difference")
                summary["mismatched_elements"] += mismatch
                if mismatch:
                    summary["mismatched_paths"].append(path)
            elif a != b:
                summary["structural_errors"].append(path+": scalar mismatch")
        visit(reference[category], candidate[category], category)
        summary["passed"] = not summary["mismatched_elements"] and not summary["structural_errors"]
        result["passed"] = result["passed"] and summary["passed"]
        result["categories"][category] = summary
    result["loss_delta"] = float(candidate["loss"]-reference["loss"])
    result["argmax_disagreements"] = int((candidate["logits"].argmax(-1) != reference["logits"].argmax(-1)).sum())
    result["argmax_equal"] = result["argmax_disagreements"] == 0
    result["passed"] = result["passed"] and result["argmax_equal"]
    return result


class MemoryProfile(Profile):
    """Exact allocator high-water intervals for sequential top-level stages.

    Nested operator timers retain their original behavior. Only profile_* stage
    boundaries synchronize and reset CUDA allocator peaks. The run maximum is
    the maximum over these intervals, including between-stage allocations.
    """
    def __init__(self, torch, check, *, journal, identity, GPU_budget_bytes):
        super().__init__(torch, check)
        self.journal, self.identity = journal, identity
        self.GPU_budget_bytes = GPU_budget_bytes
        self.memory_stages = []
        self.run_allocated_peak = self.run_reserved_peak = 0
        self.top_active = False

    def _sample(self):
        cuda = self.torch.cuda
        free, total = cuda.mem_get_info()
        return dict(allocated_bytes=cuda.memory_allocated(), reserved_bytes=cuda.memory_reserved(),
            peak_allocated_bytes=cuda.max_memory_allocated(), peak_reserved_bytes=cuda.max_memory_reserved(),
            physical_free_bytes=free, physical_total_bytes=total)

    def _retain(self, row):
        self.run_allocated_peak = max(self.run_allocated_peak, row["peak_allocated_bytes"])
        self.run_reserved_peak = max(self.run_reserved_peak, row["peak_reserved_bytes"])

    @contextmanager
    def stage(self, name):
        if not name.startswith("profile_"):
            with super().stage(name):
                yield
            return
        need(not self.top_active, "Top-level memory intervals must not nest")
        cuda = self.torch.cuda
        cuda.synchronize()
        before = self._sample()
        self._retain(before)
        cuda.reset_peak_memory_stats()
        self.top_active = True
        success = False
        try:
            with super().stage(name):
                yield
            success = True
        finally:
            cuda.synchronize()
            after = self._sample()
            self._retain(after)
            row = dict(identity=self.identity, stage=name, completed=success,
                before=before, after=after, interval_peak_allocated_bytes=after["peak_allocated_bytes"],
                interval_peak_reserved_bytes=after["peak_reserved_bytes"], RSS_bytes=rss())
            self.memory_stages.append(row)
            self.top_active = False
            with self.journal.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            print(json.dumps(dict(event="stage_memory", **row)), flush=True)
            if success:
                need(after["peak_reserved_bytes"] <= self.GPU_budget_bytes,
                     "Stage transient reserved-memory peak exceeded soft budget at "+name)

    def snapshot(self):
        result = super().snapshot()
        result.update(stage_memory=self.memory_stages,
            run_peak_allocated_bytes=self.run_allocated_peak,
            run_peak_reserved_bytes=self.run_reserved_peak,
            memory_scope="Exact synchronized top-level allocator intervals; nested timers have no separate peak resets. Physical free-memory values are boundary snapshots, not NVML per-process peaks.")
        return result


def run_update(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check, output):
    import torch
    from research.scalability_20261024.integration.executors import build_executor
    from research.efficiency_20261027.root.context_residency import (
        build_executor as resident_factory, decode_resident_parent_backward)
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import (
        decode_parent_backward, geometry_from_spans)
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    need(kind in KINDS and torch.get_num_threads() == 8, "Explicit integrated/resident CPU8 comparison")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    identity = dict(kind=kind, phase=phase, repeat=repeat)
    profile = MemoryProfile(torch, check, journal=output/"stage_memory.jsonl", identity=identity,
        GPU_budget_bytes=int(args.gpu_memory_gib*GIB))
    full_context_bytes = bundle.native_holds*1600*32*4
    free, total = torch.cuda.mem_get_info()
    if kind == "context_gpu":
        need(2*full_context_bytes+args.minimum_free_reserve_gib*GIB <= free,
             "Resident context/VJP plus explicit scratch reserve exceeds current free device memory")
        need(2*full_context_bytes+args.minimum_free_reserve_gib*GIB <= args.gpu_memory_gib*GIB,
             "Resident context/VJP plus explicit scratch reserve exceeds soft budget")
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
            executor = (resident_factory(model, bundle, metadata, **kwargs) if kind == "context_gpu"
                else build_executor("integrated", model, bundle, metadata, execution="staged", **kwargs))
        with torch.autocast(device_type="cuda", enabled=False):
            with profile.stage("profile_context_forward"):
                context = executor.context_forward()
                need(context.shape == (bundle.native_holds, 1600, 32), "Complete original context shape")
            context_boundary = dict(device=str(context.device), bytes=context.numel()*context.element_size())
            with profile.stage("profile_decoder_forward_backward"):
                decode = decode_resident_parent_backward if kind == "context_gpu" else decode_parent_backward
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
            protocol_seconds=time.perf_counter()-started, loss=float(loss), gradient_norm=float(norm),
            decoder=decoder, context_boundary=context_boundary, VJP_boundary=VJP_boundary,
            stages=profile.snapshot(), torch_peak_allocated_bytes=profile.run_allocated_peak,
            torch_peak_reserved_bytes=profile.run_reserved_peak, RSS_bytes=rss(),
            process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            initial_physical_free_bytes=free, physical_total_bytes=total, cache_snapshot=cache.snapshot())
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


def source_names():
    names = ["research/efficiency_20261027/execution/compare_full_context.py",
        "research/efficiency_20261027/root/compare_host_parallelism.py",
        "research/efficiency_20261027/root/context_residency.py",
        "research/scalability_20261024/integration/executors.py",
        "research/scalability_20261024/integration/cache.py",
        "research/scalability_20261024/execution/layout_plan.py",
        "research/scalability_20261024/categorical_site/categorical_site.py",
        "research/preprocessing_optimization_20261026/integration/hooks.py",
        "research/preprocessing_optimization_20261026/tree/process_observations.py",
        "research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/training.py"]
    names += ["research/benchmark_rebuild_20261024/american/treeloom_speed/"+name for name in (
        "source_pins.json", "runtime_v2.py", "training_v2.py", "static_model_v2.py", "staged_execution_v2.py",
        "decoder_v2.py", "gru_v2.py", "stable_leaf_history_v2.py", "parallel_history_window.py")]
    return names


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holds-per-block", type=int, default=128)
    parser.add_argument("--haplotypes-per-gru", type=int, default=32)
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--gpu-memory-gib", type=float, default=64)
    parser.add_argument("--max-rss-gib", type=float, default=256)
    parser.add_argument("--minimum-free-reserve-gib", type=float, default=8)
    args = parser.parse_args()
    args.mode = "full"
    schedule = full_schedule(args.warm_repeats)
    need(1 <= args.holds_per_block <= 256 and 1 <= args.haplotypes_per_gru <= 128,
         "Original bounded partitions required")
    need(args.minimum_free_reserve_gib >= 8 and args.gpu_memory_gib <= 64,
         "At least8GiB scratch reserve and at most64GiB soft device budget")
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3", "4", "5")
        and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
        and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
        and os.environ.get("ARGGNN_GPU_MEMORY_POLICY") == "full_device",
        "ROOT-selected one GPU3/4/5 UUID and full-device policy required")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as fresh, training_v2 as validation
    from research.preprocessing_optimization_20261026.integration.hooks import install_tree
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
    names = source_names()
    sources = [ref(name) for name in names]
    started = time.perf_counter()
    write(output/"configuration.json", dict(schema="TreeLoom_full_context_compare_config_v1",
        arguments=vars(args), CPU_threads=8, preparation_processes=4, source_refs=sources,
        checkpoint_ref=ref(CHECKPOINT), schedule=schedule,
        warmup="One actual128hold update per arm, excluded from full timing; full prepared bank retained on CPU.",
        cache_policy="Independent per-arm8GiB in-memory full layout caches; both first full calls start cold.",
        external_guardian_required=True, hard_allocator_cap=False))
    try:
        need(ref(CHECKPOINT)["sha256"] == "12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24", "Exact241 checkpoint")
        saved = torch.load(ROOT/CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        registry = fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
        rt = fresh.FreshTreeLoom(registry)
        validation.validate_saved_state(saved, rt.build_model(), registry, expected_steps=241)
        install_tree(preparation_workers=4, max_rss_bytes=int(args.max_rss_gib*GIB))
        need(not torch.cuda.is_initialized(), "Prepare full observations before CUDA initialization")
        bundle, metadata, labels, frozen, scope = prepare(args, rt, saved, check)
        need(bundle.native_holds == 77472 and len(labels) == 10000 and bundle.bank.sample_index.shape == (77472, 1600),
             "Exact full TRAIN000 native support and all10000original targets")
        write(output/"full_scope.json", scope)
        slice_args = SimpleNamespace(**dict(vars(args), mode="slice"))
        sb, sm, sl, sf, ss = prepare(slice_args, rt, saved, check)
        def make_cache(cap):
            return LayoutCache(max_bytes=cap, admit=lambda amount, stage:
                need(rss()+amount <= args.max_rss_gib*GIB, "Layout working reserve exceeded at "+stage))
        slice_baseline = None
        warmup_rows = []
        for kind in KINDS:
            cache = make_cache(64*2**20)
            row, numerical = run_update(kind, "slice_warmup", 0, args, rt, saved, sb, sm, sl, sf, cache, check, output)
            if slice_baseline is None:
                slice_baseline = numerical
            row["comparison"] = compare_state(slice_baseline, numerical)
            warmup_rows.append(row)
            write(output/(kind+"_slice_warmup.json"), row)
            need(row["comparison"]["passed"], "Slice warmup numerical or argmax parity failed")
            del cache
        del sb, sm, sl, sf, slice_baseline, numerical
        gc.collect()
        torch.cuda.empty_cache()
        caches = {kind: make_cache(8*GIB) for kind in KINDS}
        rows, baseline = [], None
        for phase, repeat, order in schedule:
            for position, kind in enumerate(order):
                need([ref(name) for name in names] == sources, "Source changed before full call")
                row, numerical = run_update(kind, phase, repeat, args, rt, saved,
                    bundle, metadata, labels, frozen, caches[kind], check, output)
                if baseline is None:
                    baseline = numerical
                row["comparison"] = compare_state(baseline, numerical)
                row["order_position"] = position
                rows.append(row)
                write(output/(kind+"_"+phase+"_"+str(repeat)+".json"), row)
                need(row["comparison"]["passed"], "Full-parent numerical or argmax parity failed")
        need([ref(name) for name in names] == sources, "Source changed during qualification")
        result = dict(schema="TreeLoom_full_context_compare_v1", status="completed", scope=scope,
            slice_warmup_scope=ss, source_refs=sources, checkpoint_ref=ref(CHECKPOINT), FP32=True,
            CPU_threads=8, preparation_processes=4, warmup_records=warmup_rows, records=rows,
            physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"], visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
            total_seconds=time.perf_counter()-started, external_guardian_required=True,
            soft_GPU_budget_bytes=args.gpu_memory_gib*GIB, max_RSS_bytes=args.max_rss_gib*GIB,
            execution_partitions=dict(holds_per_block=args.holds_per_block,
                haplotypes_per_gru=args.haplotypes_per_gru, history_window=1024),
            caveats=["All calls restore241 weights, AdamW and RNG; no diagnostic update is committed to training.",
                "One128hold warmup per arm warms shared libraries but not full-size algorithms or full layout caches.",
                "Each arm owns a separate full layout cache; first full calls are both cold, later calls reuse only that arm's cache.",
                "Default one full call per arm is correctness/memory qualification and an initial timing observation, not stable throughput statistics.",
                "Top-level stages synchronize and record allocator peaks; physical-memory boundary snapshots are not continuous process memory monitoring.",
                "Standalone RSS guard is cheaper than production PSS/receipt checking; production throughput must be confirmed separately.",
                "No production checkpoint, observed input or held-out truth is changed."])
        write(output/"completed.json", result)
        print(json.dumps(dict(status="completed", completion_ref=ref(output/"completed.json")), indent=2))
    except BaseException as error:
        write(output/"failed.json", dict(status="failed", error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.perf_counter()-started,
            source_refs_at_start=sources, partial_receipts_preserved=True))
        raise


if __name__ == "__main__":
    main()
