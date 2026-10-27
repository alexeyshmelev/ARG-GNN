"""ROOT-admitted 128-hold comparison; no launcher, production write or disk cache.

Warmup and timed updates use uninstrumented executors. A separate final pass
counts actual block calls and tensor payloads, with optional transfer profiling.
Every call restores checkpoint241 and performs the same true TRAIN update242.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import copy
import gc
import json
import os
import resource
import statistics
import time

from research.efficiency_20261027.root.compare_execution import (
    ROOT, CHECKPOINT, GIB, Profile, need, ref, rss, write, prepare)

KINDS = ("original", "integrated", "level_local_v2", "context_gpu")


def schedule(repeats=8):
    need(type(repeats) is int and repeats in (8, 16),
         "Eight or sixteen timed rounds provide balanced four-arm order")
    forward = tuple(KINDS[i:]+KINDS[:i] for i in range(len(KINDS)))
    orders = forward + tuple(tuple(reversed(order)) for order in forward)
    return tuple((round_number + 1, orders[round_number % len(orders)])
                 for round_number in range(repeats))


def tensor_inventory(block):
    """Count actual returned tensors, including empty transfer API calls.

    Verified _block recipes source every returned field from CPU. This observes
    API payload, not a driver memcpy count; empty tensors may issue no memcpy.
    """
    import torch
    fields = {}
    for name, value in vars(block).items():
        tensors = value if isinstance(value, tuple) else (value,)
        need(all(torch.is_tensor(x) for x in tensors), "Tensor-only block fields")
        fields[name] = dict(tensor_calls=len(tensors),
            nonempty_tensor_calls=sum(int(x.numel() > 0) for x in tensors),
            payload_bytes=sum(x.numel() * x.element_size() for x in tensors),
            destination_devices=sorted({str(x.device) for x in tensors}))
    return fields


class BlockAudit:
    """Mixin active only in the diagnostic pass; does not patch live classes."""
    def _block(self, z):
        import torch
        if not hasattr(self, "block_audit"):
            self.block_audit = []
            self.block_events = []
        start = end = None
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        tick = time.perf_counter()
        block = super()._block(z)
        elapsed = time.perf_counter() - tick
        if end is not None:
            end.record()
            self.block_events.append((start, end))
        self.block_audit.append(dict(hold_range=[int(z[0]), int(z[1])],
            fields=tensor_inventory(block), CPU_wall_seconds=elapsed))
        return block

    def transfer_snapshot(self):
        calls = getattr(self, "block_audit", [])
        events = getattr(self, "block_events", [])
        fields = {}
        for call in calls:
            for name, row in call["fields"].items():
                total = fields.setdefault(name, dict(tensor_calls=0,
                    nonempty_tensor_calls=0, payload_bytes=0))
                for key in total:
                    total[key] += row[key]
        counts = {}
        for call in calls:
            key = ":".join(map(str, call["hold_range"]))
            counts[key] = counts.get(key, 0) + 1
        return dict(observed_block_invocations=len(calls),
            observed_visits_by_hold_range=counts, fields=fields,
            tensor_calls=sum(row["tensor_calls"] for row in fields.values()),
            nonempty_tensor_calls=sum(row["nonempty_tensor_calls"] for row in fields.values()),
            payload_bytes=sum(row["payload_bytes"] for row in fields.values()),
            CPU_wall_seconds=sum(call["CPU_wall_seconds"] for call in calls),
            CUDA_event_interval_ms=sum(start.elapsed_time(end) for start, end in events),
            destination=str(self.device),
            interpretation="Observed explicit CPU-source _block transfer API outputs. CUDA event intervals include host submission gaps and pageable staging; not isolated copy-engine time. Empty API calls can issue no memcpy.")


def executor_factory(kind, diagnostic, *args, **kwargs):
    from research.efficiency_20261027.execution.executors_v2 import (
        build_executor, MinimalTransferLevelExecution)
    from research.efficiency_20261027.root.context_residency import (
        build_executor as resident_factory, ResidentContextExecution)
    if not diagnostic:
        if kind == "context_gpu":
            kwargs.pop("execution")
            return resident_factory(*args, **kwargs)
        return build_executor(kind, *args, **kwargs)
    from research.benchmark_rebuild_20261024.american.treeloom_speed.staged_execution_v2 import StagedExecutionV1
    from research.scalability_20261024.integration.executors import IntegratedStagedExecution
    from research.efficiency_20261027.execution.level_sweep import sweep_tree_state
    class AuditedOriginal(BlockAudit, StagedExecutionV1):
        pass
    class AuditedIntegrated(BlockAudit, IntegratedStagedExecution):
        pass
    class AuditedResident(BlockAudit, ResidentContextExecution):
        pass
    class AuditedMinimal(BlockAudit, MinimalTransferLevelExecution):
        def _sweep(self, z, bank, base, ug, dg, degree):
            import torch
            if not hasattr(self, "route_events"):
                self.route_events = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            plan = self.sweep_plans[z[0]].to(self.device)
            end.record()
            self.route_events.append((start, end, self.sweep_plans[z[0]].tensor_bytes()))
            return sweep_tree_state(self.model, bank, base, ug, dg, degree, plan=plan)
        def transfer_snapshot(self):
            result = super().transfer_snapshot()
            events = getattr(self, "route_events", [])
            result["packed_route"] = dict(observed_transfer_calls=len(events),
                payload_bytes=sum(row[2] for row in events),
                CUDA_event_interval_ms=sum(row[0].elapsed_time(row[1]) for row in events))
            return result
    kwargs.pop("execution")
    if kind == "original":
        kwargs.pop("layout_cache")
    if kind != "level_local_v2":
        kwargs.pop("max_sweep_plan_bytes", None)
        kwargs.pop("sweep_plan_admit", None)
    return {"original": AuditedOriginal, "integrated": AuditedIntegrated,
            "level_local_v2": AuditedMinimal, "context_gpu": AuditedResident}[kind](*args, **kwargs)


def compare(reference, candidate):
    import torch
    errors = {}
    for category in ("logits", "loss", "gradients", "parameters", "moments"):
        atol, rtol = 1e-6, (1e-5 if category in ("logits", "loss") else 1e-4)
        maximum = 0.
        def visit(a, b):
            nonlocal maximum
            if isinstance(a, dict):
                need(set(a) == set(b), "Identical numerical state keys")
                for key in a:
                    visit(a[key], b[key])
            elif torch.is_tensor(a):
                torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
                if a.numel():
                    maximum = max(maximum, float((a-b).abs().max()))
            else:
                need(a == b, "Identical scalar numerical state")
        visit(reference[category], candidate[category])
        errors[category] = dict(max_abs_error=maximum, atol=atol, rtol=rtol)
    return dict(passed=True, categories=errors,
        loss_delta=float(candidate["loss"]-reference["loss"]),
        argmax_disagreements=int((reference["logits"].argmax(-1) != candidate["logits"].argmax(-1)).sum()))


def run_one(kind, phase, repeat, args, rt, saved, bundle, metadata, labels, frozen, cache, check):
    import torch
    from research.benchmark_rebuild_20261024.american.treeloom_speed.decoder_v2 import decode_parent_backward, geometry_from_spans
    from research.repairs.TreeLoom_lazy_compact_prefetch_20261024_v1.training import restore_random_state
    from research.efficiency_20261027.root.context_residency import decode_resident_parent_backward
    decode = decode_resident_parent_backward if kind == "context_gpu" else decode_parent_backward
    diagnostic = phase == "transfer_diagnostic"
    profile = Profile(torch, check)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    logits, executor = [], None
    with profile.stage("profile_model_optimizer_transfer"):
        model = rt.build_model()
        model.load_state_dict(saved["model_state"], strict=True)
        model = model.to(device="cuda:0", dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
            betas=(.9, .999), eps=1e-8, maximize=False, amsgrad=False, foreach=False,
            fused=False, capturable=False, differentiable=False)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
        restore_random_state(saved["random_state"])
    hook = model.classifier.register_forward_hook(lambda module, inputs, output: logits.append(output.detach().cpu()))
    trace_enabled = diagnostic and args.trace_transfers
    trace = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
        with_stack=False) if trace_enabled else nullcontext()
    try:
        with trace as captured:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with profile.stage("profile_layout"):
                executor = executor_factory(kind, diagnostic, model, bundle, metadata,
                    execution="staged", layout_cache=cache, frozen_module=frozen,
                    holds_per_block=args.holds_per_block, haplotypes_per_gru=args.haplotypes_per_gru,
                    history_holds_per_window=1024, resource_check=check, profile=profile,
                    **(dict(max_sweep_plan_bytes=64*2**20,
                        sweep_plan_admit=lambda amount, stage: need(rss()+amount <= args.max_rss_gib*GIB,
                        "Sweep plan host reserve exceeded at "+stage)) if kind == "level_local_v2" else {}))
            with torch.autocast(device_type="cuda", enabled=False):
                with profile.stage("profile_context_forward"):
                    context = executor.context_forward()
                with profile.stage("profile_decoder_forward_backward"):
                    loss, vjp, decoder = decode(model, context,
                        geometry_from_spans(bundle.bank.spans), bundle.focal_hold,
                        bundle.focal_sample_column, labels, tile_haplotypes=args.haplotypes_per_gru,
                        profile=profile, resource_check=check)
                with profile.stage("profile_trunk_backward"):
                    context.backward(vjp)
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
            need(len(gradients) == 49 and all(bool(torch.isfinite(g).all()) for g in gradients.values()),
                 "All49 finite connected actual gradients")
            with profile.stage("profile_clip_optimizer"):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            check("profile_update_complete")
            torch.cuda.synchronize()
            update_seconds = time.perf_counter()-started
        parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        moments = {str(i): {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
            for k, v in row.items()} for i, row in optimizer.state_dict()["state"].items()}
        need(len(moments) == 49 and all(int(row["step"].item()) == 242 for row in moments.values()),
             "All49 Adam counters must be242")
        copies = []
        if trace_enabled:
            for row in captured.key_averages():
                if "memcpy" in row.key.lower() or row.key in ("aten::_to_copy", "aten::copy_"):
                    copies.append(dict(name=row.key, calls=row.count,
                        cpu_total_us=row.cpu_time_total, device_total_us=getattr(row, "device_time_total", 0.)))
        row = dict(kind=kind, phase=phase, repeat=repeat, seconds=time.perf_counter()-started,
            update_seconds=update_seconds, loss=float(loss), gradient_norm=float(norm),
            decoder=decoder, stages=profile.snapshot(), transfer_diagnostic=diagnostic,
            transfer_trace_enabled=trace_enabled, copy_events=copies,
            block_transfer_audit=executor.transfer_snapshot() if diagnostic else None,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(), RSS_bytes=rss(),
            process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cache_snapshot=cache.snapshot() if cache is not None else None,
            sweep_plan_snapshot=executor.sweep_plan_snapshot() if hasattr(executor, "sweep_plan_snapshot") else None)
        return row, dict(logits=torch.cat(logits), loss=loss.detach().cpu(),
            gradients=gradients, parameters=parameters, moments=moments)
    finally:
        hook.remove()
        if "context" in locals():
            del context
        if "vjp" in locals():
            del vjp
        optimizer.zero_grad(set_to_none=True)
        if executor is not None and hasattr(executor, "release_layout"):
            executor.release_layout()
        del executor, model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holds-per-block", type=int, default=128)
    parser.add_argument("--haplotypes-per-gru", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timed-repeats", type=int, default=8)
    parser.add_argument("--gpu-memory-gib", type=float, default=8)
    parser.add_argument("--max-rss-gib", type=float, default=24)
    parser.add_argument("--trace-transfers", action="store_true")
    args = parser.parse_args()
    args.mode = "slice"
    timed_schedule = schedule(args.timed_repeats)
    need(args.cpu_threads in (1, 4, 8), "Explicit bounded parent CPU thread count")
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
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT/"research/efficiency_20261027") and not output.exists(),
         "Fresh efficiency diagnostic output required")
    output.mkdir(parents=True)
    def check(stage):
        need(rss() <= args.max_rss_gib*GIB, "RSS guard exceeded at "+stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= args.gpu_memory_gib*GIB, "GPU soft budget exceeded at "+stage)
    # Pin the actual imported paths, not a broad mutable directory glob.
    names = ["research/efficiency_20261027/root/compare_execution.py",
        "research/efficiency_20261027/root/context_residency.py",
        "research/efficiency_20261027/execution/compare_minimal_transfer.py",
        "research/efficiency_20261027/execution/executors.py",
        "research/efficiency_20261027/execution/executors_v2.py",
        "research/efficiency_20261027/execution/level_sweep.py",
        "research/scalability_20261024/integration/executors.py",
        "research/scalability_20261024/integration/cache.py",
        "research/scalability_20261024/execution/layout_plan.py",
        "research/scalability_20261024/categorical_site/categorical_site.py"]
    names += ["research/benchmark_rebuild_20261024/american/treeloom_speed/"+name
        for name in ("source_pins.json", "runtime_v2.py", "static_model_v2.py", "staged_execution_v2.py",
                     "decoder_v2.py", "gru_v2.py", "stable_leaf_history_v2.py", "parallel_history_window.py")]
    source_refs = [ref(name) for name in names]
    need(ref(CHECKPOINT)["sha256"] == "12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24", "Exact241 checkpoint")
    started = time.perf_counter()
    saved = torch.load(ROOT/CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    registry = fresh.FreshParentRegistry.from_ref(saved["parent_registry_ref"])
    rt = fresh.FreshTreeLoom(registry)
    validation.validate_saved_state(saved, rt.build_model(), registry, expected_steps=241)
    bundle, metadata, labels, frozen, scope = prepare(args, rt, saved, check)
    cache = LayoutCache(max_bytes=64*2**20, admit=lambda amount, stage:
        need(rss()+amount <= args.max_rss_gib*GIB, "Layout cache reserve exceeded at "+stage))
    records, baseline = [], None
    rounds = [("warmup", 1, KINDS), ("warmup", 2, tuple(reversed(KINDS)))]
    rounds += [("timed", repeat, order) for repeat, order in timed_schedule]
    rounds += [("transfer_diagnostic", 1, KINDS)]
    for phase, repeat, order in rounds:
        for position, kind in enumerate(order):
            row, numerical = run_one(kind, phase, repeat, args, rt, saved, bundle, metadata,
                labels, frozen, cache if kind != "original" else None, check)
            if baseline is None:
                baseline = numerical
            row["comparison"] = compare(baseline, numerical)
            row["order_position"] = position
            records.append(row)
            write(output/(kind+"_"+phase+"_"+str(repeat)+".json"), row)
    need([ref(name) for name in names] == source_refs, "Imported source changed during comparison")
    summaries = {}
    for kind in KINDS:
        rows = [row for row in records if row["phase"] == "timed" and row["kind"] == kind]
        summaries[kind] = dict(repeats=len(rows),
            median_update_seconds=statistics.median(row["update_seconds"] for row in rows),
            min_update_seconds=min(row["update_seconds"] for row in rows),
            max_update_seconds=max(row["update_seconds"] for row in rows),
            median_protocol_seconds=statistics.median(row["seconds"] for row in rows))
    result = dict(schema="TreeLoom_minimal_transfer_comparison_v1", status="completed",
        scope=scope, checkpoint_ref=ref(CHECKPOINT), source_refs=source_refs, records=records,
        timed_summaries=summaries, total_seconds=time.perf_counter()-started,
        CPU_threads=args.cpu_threads, physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"],
        visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"], FP32=True,
        execution_partitions=dict(holds_per_block=args.holds_per_block,
            haplotypes_per_gru=args.haplotypes_per_gru, history_window=1024),
        external_guardian_required=True, soft_GPU_budget_bytes=args.gpu_memory_gib*GIB,
        max_RSS_bytes=args.max_rss_gib*GIB, hard_allocator_cap=False,
        caveats=["Two warmup rounds and separate transfer diagnostics are excluded from timing summaries.",
            "All timed rounds reset241 weights, AdamW and RNG; observation/layout CPU caches are reused only in memory.",
            "The128hold restricted history and22actual TRAIN focals do not establish full-parent speed or accuracy.",
            "Timed protocol retains original gradient/logit capture and stage synchronizations; update_seconds stops before post-update state capture and excludes cleanup.",
            "CUDA event transfer intervals include host launch gaps. Profiler copy rows may overlap and are not additive phases.",
            "Observed block inventories exclude site/history/decoder transfers; optional profiler records additional operations.",
            "Production, checkpoints, observation files and held-out truth are untouched."])
    write(output/"completed.json", result)
    print(json.dumps(dict(status="completed", completion_ref=ref(output/"completed.json"),
                         timed_summaries=summaries), indent=2))


if __name__ == "__main__":
    main()
