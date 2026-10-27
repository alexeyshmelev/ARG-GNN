"""CPU-only protocol proof using the authenticated observed128hold packet."""
import ast
import json
from types import SimpleNamespace
import torch
from research.efficiency_20261027.execution import compare_hot_block_transfer_v1 as runner
from research.efficiency_20261027.execution.test_hot_block_pack_v3 import load_fixture
from research.efficiency_20261027.execution.large_layout_plan import LargeLayoutPlan
from research.efficiency_20261027.execution.launch_hot_block_transfer_v1 import argv_for
from research.efficiency_20261027.root.compare_host_parallelism import ROOT, ref, write, need

BASE = ROOT / "research/efficiency_20261027/execution"


def main():
    need(not torch.cuda.is_initialized(), "CPU-only protocol test")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    bank, episodes, cut, counts, packet_ref = load_fixture()
    layout = LargeLayoutPlan.build(bank, counts, episodes.member_hold, holds_per_block=65).executor_layout()
    need([(z[0], z[1]) for z in layout] == [(0, 65), (65, 128)], "Actual ragged diagnostic layout")
    rng = torch.get_rng_state().clone()
    admitted = []
    admit = lambda remaining, stage: admitted.append(dict(remaining_bytes=remaining, stage=stage))
    proof = runner.verify_values(bank, layout, device="cpu", provenance="CPU observed packet",
        cap=512*2**20, admit=admit, CPU_test_only=True)
    rows = []
    schedules = runner.schedule()
    need(len(schedules) == 7 and schedules[0] == ("warmup", 0, runner.KINDS), "One complete warmup pair")
    for position in (0, 1):
        need(sum(order[position] == "original" for phase, repeat, order in schedules if phase == "timed") == 3,
             "Each timed arm appears first and second equally")
    actual_orders = []
    original_block = runner.original_block
    def traced(bank, z, device):
        actual_orders.append((z[0], z[1]))
        return original_block(bank, z, device)
    runner.original_block = traced
    try:
        for phase, repeat, order in schedules:
            for kind in order:
                actual_orders.clear()
                row = runner.run_round(kind, bank, layout, device="cpu", provenance="CPU observed packet",
                    cap=512*2**20, admit=admit, check=lambda stage: None, CPU_test_only=True)
                if kind == "original":
                    need(actual_orders == [(z[0], z[1]) for group in runner.pass_orders(layout) for z in group],
                         "Original exact F,F,R,R pass order")
                else:
                    need(row["pack"]["block_copies"] == 2 and row["pack"]["cache_hits"] == 6,
                         "Fresh two-copy six-hit pack every round")
                    need(row["released_pack"]["phase"] == "closed" and row["released_pack"]["retained_payload_bytes"] == 0,
                         "Each round releases exact pack accounting")
                need(0 <= row["binding_host_seconds"] <= row["binding_plus_cold_first_pass_seconds"] <= row["all_four_pass_seconds"],
                     "Nested cold and all-four timing boundaries")
                need(abs(row["all_four_pass_seconds"]-row["binding_plus_cold_first_pass_seconds"]-row["remaining_three_pass_seconds"]) < 1e-9,
                     "All4 includes cold pass and three reuse passes")
                need(row["final_and_peak_allocator"]["peak_allocated_bytes"] is None, "No invented CUDA peak in CPU proof")
                rows.append(dict(phase=phase, repeat=repeat, **row))
    finally:
        runner.original_block = original_block
    args = SimpleNamespace(directory="research/efficiency_20261027/root/runs/hot_transfer_review_placeholder_v1",
        gpu=5, predecessor_directory="research/efficiency_20261027/root/runs/selected_full_gpu5_v1",
        peer_directory=["research/efficiency_20261027/root/runs/selected_full_gpu5_v1"], source_pins_sha256="0"*64)
    argv = argv_for(args)
    need(argv[:3] == [".runtime/env/bin/python3.11", "-B", "research/efficiency_20261027/root/run_guarded_profile_with_peer.py"],
         "Visible project Python and reviewed guardian")
    need(argv[argv.index("--run-seconds")+1] == "600" and argv[argv.index("--host-gib")+1] == "24"
         and argv[argv.index("--device-gib")+1] == "24", "Exact bounded guarded launch")
    need(not (ROOT / args.directory).exists(), "Reviewing argv creates no output or launched process")
    sources = [BASE / name for name in ("prepare_hot_block_transfer_v1.py", "compare_hot_block_transfer_v1.py",
        "launch_hot_block_transfer_v1.py", "test_hot_block_transfer_v1.py")]
    for source in sources:
        tree = ast.parse(source.read_text())
        need(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("eval", "exec", "compile")
             for n in ast.walk(tree)), "Readable source without dynamic code execution")
    preparation = sources[0].read_text()
    need("FreshTrainTargetJoin" not in preparation and 'item["truth_ref"]' not in preparation,
         "Observed-only preparer contains no target join or truth input access")
    need(torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), "No RNG consumption or CUDA initialization")
    result = dict(status="passed", schema="TreeLoom_hot_transfer_CPU_protocol_v1", source_refs=[ref(p) for p in sources],
        observed_packet_ref=packet_ref, exact_values=proof, records=rows,
        one_warmup_pair_and_six_balanced_timed_pairs=True, original_pass_order="forward,forward,reverse,reverse",
        fresh_pack_per_round=True, preserved_RNG=True, CUDA_initialized=False, gpu_launched=False,
        launcher_argv_checked_without_subprocess=True, actual1153hold_preparation_executed=False,
        no_GPU_speed_or_peak_claim=True)
    write(BASE / "hot_block_transfer_CPU_v1.json", result)
    print(json.dumps(dict(status="passed", result_ref=ref(BASE / "hot_block_transfer_CPU_v1.json"),
        rounds=len(rows), actual_payload_bytes=proof["total_payload_bytes"], CUDA_initialized=False)), flush=True)


if __name__ == "__main__":
    main()
