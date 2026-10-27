"""Isolated original block-copy versus fresh immutable-pack transfer benchmark.

This measures input movement only. No model, learned arithmetic, targets,
optimizer, production output, or cross-round device input cache is used.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import time
from types import SimpleNamespace
import torch
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, CHECKPOINT, GIB, need, ref, rss, write
from research.scalability_20261024.integration.executors import _IntegratedLayoutAndSites
from .hot_block_pack_v2 import BlockInputPack, CUDAPackAdmission, tensors, block_bytes
from .prepare_hot_block_transfer_v1 import prepare_prefix
from .deferred_gpu_gate import wait_before_cuda

BASE = ROOT / "research/efficiency_20261027/execution"
KINDS = ("original", "hotpack")


def schedule():
    return [("warmup", 0, KINDS)] + [("timed", i+1, KINDS if i % 2 == 0 else tuple(reversed(KINDS))) for i in range(6)]


def pass_orders(layout):
    return (layout, layout, tuple(reversed(layout)), tuple(reversed(layout)))


def original_block(bank, z, device):
    return _IntegratedLayoutAndSites._block(SimpleNamespace(bank=bank, device=torch.device(device)), z)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def allocator(device):
    if torch.device(device).type != "cuda":
        return dict(allocated_bytes=None, reserved_bytes=None, peak_allocated_bytes=None, peak_reserved_bytes=None)
    return dict(allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=torch.cuda.memory_reserved(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def verify_values(bank, layout, *, device, provenance, cap, admit, CPU_test_only=False):
    """Untimed exact input equality and actual output payload measurement."""
    cache = BlockInputPack(bank, layout, device=device, provenance=provenance,
        max_payload_bytes=cap, admit=admit, CPU_test_only=CPU_test_only)
    payloads, tensor_counts = [], []
    try:
        for z in layout:
            expected = original_block(bank, z, "cpu")
            baseline = original_block(bank, z, device)
            candidate = cache.block(bank, z, provenance=provenance)
            old, plain, hot = tensors(expected), tensors(baseline), tensors(candidate)
            need(len(old) == len(plain) == len(hot), "Exact original tensor count")
            for a, b, c in zip(old, plain, hot):
                need(a.dtype == b.dtype == c.dtype and a.shape == b.shape == c.shape
                     and torch.equal(a, b.cpu()) and torch.equal(a, c.cpu()), "Exact copied values, dtypes and row order")
            actual = sum(t.numel()*t.element_size() for t in hot)
            need(actual == block_bytes(bank, z), "Actual copied output payload agrees with accounting")
            payloads.append(actual)
            tensor_counts.append(len(hot))
            del expected, baseline, candidate, old, plain, hot, a, b, c
        synchronize(device)
        return dict(exact_values_dtypes_shapes_order=True, actual_block_payload_bytes=payloads,
            actual_block_tensor_counts=tensor_counts, total_payload_bytes=sum(payloads),
            payload_definition="Sum of actual copied output tensor numel times element_size; not measured bus traffic.")
    finally:
        cache.close()


def run_round(kind, bank, layout, *, device, provenance, cap, admit, check, CPU_test_only=False):
    need(kind in KINDS, "Explicit baseline or candidate")
    gc.collect()
    synchronize(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = allocator(device)
    before_rss = rss()
    check("transfer_round_before_timing")
    cache = None
    block = None
    calls = 0
    started = time.perf_counter()
    try:
        if kind == "hotpack":
            cache = BlockInputPack(bank, layout, device=device, provenance=provenance,
                max_payload_bytes=cap, admit=admit, CPU_test_only=CPU_test_only)
        bound = time.perf_counter()
        for pass_index, order in enumerate(pass_orders(layout)):
            for z in order:
                block = (original_block(bank, z, device) if cache is None else cache.block(bank, z, provenance=provenance))
                calls += 1
                del block
                block = None
            if pass_index == 0:
                synchronize(device)
                cold_finished = time.perf_counter()
        synchronize(device)
        finished = time.perf_counter()
        peaks = allocator(device)
        pack = None if cache is None else cache.snapshot()
        need(calls == 4*len(layout), "Complete four-pass F,F,R,R call sequence")
        if pack is not None:
            need(pack["block_copies"] == len(layout) and pack["cache_hits"] == 3*len(layout), "One cold copy and three reuse hits per block")
        check("transfer_round_after_timing")
        result = dict(kind=kind, binding_host_seconds=bound-started,
            binding_plus_cold_first_pass_seconds=cold_finished-started,
            remaining_three_pass_seconds=finished-cold_finished,
            all_four_pass_seconds=finished-started, block_calls=calls, pack=pack,
            initial_allocator=initial, final_and_peak_allocator=peaks,
            RSS_before_bytes=before_rss, RSS_after_bytes=rss(),
            process_lifetime_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            synchronization_boundaries=["before timing", "after first complete pass", "after fourth complete pass"],
            CPU_test_only=CPU_test_only)
    finally:
        block = None
        if cache is not None:
            # This transfer-only pack remains bound: no graph or fake backward is created.
            cache.close()
        synchronize(device)
    result["after_release_allocator"] = allocator(device)
    result["released_pack"] = None if cache is None else cache.snapshot()
    return result


def verify_pins(expected_digest):
    path = BASE / "source_pins_hot_transfer_v1.json"
    need(ref(path)["sha256"] == expected_digest, "Exact reviewed transfer source manifest")
    manifest = json.loads(path.read_text())
    sources = manifest["source_refs"]
    need(len(sources) == len({item["path"] for item in sources}), "Unique source pins")
    need(all(ref(item["path"]) == item for item in sources), "Every reviewed source pin must remain exact")
    return dict(manifest_ref=ref(path), source_refs=sources)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-pins-sha256", required=True)
    parser.add_argument("--gpu-predecessor-directory", required=True)
    args = parser.parse_args()
    need(os.environ.get("ARGGNN_PHYSICAL_GPU") in ("3", "4", "5")
         and os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
         and "," not in os.environ["CUDA_VISIBLE_DEVICES"]
         and os.environ.get("ARGGNN_GPU_MEMORY_POLICY") == "full_device", "ROOT-admitted single physical GPU required")
    import setproctitle
    setproctitle.setproctitle("AI Auto Research Pipeline Test")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized(), "No CUDA before authenticated CPU preparation and predecessor gate")
    output = Path(args.output).resolve()
    need(output.is_relative_to(ROOT / "research/efficiency_20261027/root/runs") and not output.exists(), "Fresh ROOT-owned output directory")
    pins = verify_pins(args.source_pins_sha256)
    output.mkdir(parents=True)
    started = time.perf_counter()
    def check(stage, additional_bytes=0):
        need(time.perf_counter()-started <= 600, "Finite600second complete worker wall guard at " + stage)
        need(rss()+additional_bytes <= 24*GIB, "24GiB host RSS/reserve guard at " + stage)
        if torch.cuda.is_initialized():
            need(torch.cuda.memory_reserved() <= 24*GIB, "24GiB device reserved soft guard at " + stage)
    write(output / "configuration.json", dict(arguments=vars(args), **pins,
        payload_cap_bytes=512*2**20, GPU_soft_budget_bytes=24*GIB, scratch_reserve_bytes=16*GIB,
        host_cap_bytes=24*GIB, max_worker_seconds=600, CPU_threads=8, protocol=schedule(),
        no_allocator_flush=True, mathematical_training_not_executed=True))
    try:
        checkpoint_ref = ref(CHECKPOINT)
        need(checkpoint_ref["sha256"] == "12b8fbdd4bfa164ca3aab28d967d1154a3ce389cfc49aefca0fbdf9dc143de24", "Exact checkpoint241 barrier source")
        saved = torch.load(ROOT / CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        bank, layout, provenance, scope = prepare_prefix(saved, check)
        del saved
        gc.collect()
        write(output / "scope.json", scope)
        need(sum(block_bytes(bank, z) for z in layout) <= 512*2**20, "Entire three-block payload fits512MiB")
        gate = wait_before_cuda(torch, check, predecessor_directory=args.gpu_predecessor_directory,
            output=output, gpu_index=int(os.environ["ARGGNN_PHYSICAL_GPU"]), gpu_budget_bytes=24*GIB,
            timeout_seconds=60, poll_seconds=5)
        write(output / "gpu_predecessor_gate.json", gate)
        torch.cuda.init()
        admit = CUDAPackAdmission(device="cuda:0", soft_budget_bytes=24*GIB, scratch_reserve_bytes=16*GIB, resource_check=check)
        proof = verify_values(bank, layout, device="cuda:0", provenance=provenance,
            cap=512*2**20, admit=admit)
        write(output / "exact_values.json", proof)
        rows = []
        for phase, repeat, order in schedule():
            verify_pins(args.source_pins_sha256)
            for position, kind in enumerate(order):
                row = run_round(kind, bank, layout, device="cuda:0", provenance=provenance,
                    cap=512*2**20, admit=admit, check=check)
                row.update(phase=phase, repeat=repeat, order_position=position,
                    actual_copied_payload_bytes=proof["total_payload_bytes"]*(4 if kind == "original" else 1),
                    actual_copy_tensor_count=sum(proof["actual_block_tensor_counts"])*(4 if kind == "original" else 1))
                rows.append(row)
                write(output / (phase+"_"+str(repeat)+"_"+kind+".json"), row)
                print(json.dumps(dict(event="transfer_round_complete", kind=kind, phase=phase,
                    repeat=repeat, all_four_pass_seconds=row["all_four_pass_seconds"])), flush=True)
        verify_pins(args.source_pins_sha256)
        pairs = []
        for repeat in range(1, 7):
            pair = {r["kind"]: r for r in rows if r["phase"] == "timed" and r["repeat"] == repeat}
            a, b = pair["original"]["all_four_pass_seconds"], pair["hotpack"]["all_four_pass_seconds"]
            pairs.append(dict(repeat=repeat, original_seconds=a, hotpack_seconds=b,
                saved_seconds=a-b, original_over_hotpack=a/b))
        write(output / "completed.json", dict(status="completed", schema="TreeLoom_hot_transfer_GPU_microbenchmark_v1",
            **pins, checkpoint_ref=checkpoint_ref, scope=scope, exact_values=proof, records=rows, paired_results=pairs,
            timed_medians={kind: statistics.median(r["all_four_pass_seconds"] for r in rows
                if r["phase"] == "timed" and r["kind"] == kind) for kind in KINDS},
            median_paired_saved_seconds=statistics.median(p["saved_seconds"] for p in pairs),
            admissions=admit.records, total_seconds=time.perf_counter()-started,
            gpu_predecessor_gate=gate, physical_gpu=os.environ["ARGGNN_PHYSICAL_GPU"],
            visible_UUID=os.environ["CUDA_VISIBLE_DEVICES"],
            limitations=["Transfer-only prefix measurement; no end-to-end training speed or full-parent peak claim.",
                "Every candidate round includes binding, cold pack and all four passes. No pack survives a round.",
                "Original imported block conversion is unchanged. No learned forward, decoder, backward, loss or optimizer executes.",
                "Exact device input comparison is outside timing. Payload bytes are actual tensor sizes, not bus traffic.",
                "Allocator remains warm across rounds; cold means new pack and cold input copies, not cold CUDA context.",
                "No per-tensor synchronization; both arms synchronize after pass1 and pass4.",
                "RSS lifetime peak includes preparation; per-round fields are boundary RSS, not sampled round peaks."]))
    except BaseException as error:
        write(output / "failed.json", dict(error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter()-started))
        raise


if __name__ == "__main__":
    main()
