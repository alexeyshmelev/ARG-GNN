"""CPU-only checks of strict comparison and stage high-water accounting."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import ast
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from research.efficiency_20261027.execution.compare_full_context import (
    MemoryProfile, compare_state, full_schedule, source_names, ref, write)


class FakeCUDA:
    def __init__(self):
        self.allocated = self.reserved = self.peak_allocated = self.peak_reserved = 0
        self.resets = 0
    def is_initialized(self):
        return True
    def synchronize(self):
        pass
    def mem_get_info(self):
        return 8192-self.reserved, 8192
    def memory_allocated(self):
        return self.allocated
    def memory_reserved(self):
        return self.reserved
    def max_memory_allocated(self):
        return self.peak_allocated
    def max_memory_reserved(self):
        return self.peak_reserved
    def reset_peak_memory_stats(self):
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved
        self.resets += 1
    def set_usage(self, allocated, reserved):
        assert allocated <= reserved
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    here = Path("research/efficiency_20261027/execution")
    ast.parse((here/"compare_full_context.py").read_text())
    ast.parse(Path(__file__).read_text())
    sources = [ref(name) for name in source_names()]
    assert full_schedule(0) == [("full_first_call", 0, ("integrated", "context_gpu"))]
    assert full_schedule(1)[1] == ("full_warm_repeat", 1, ("context_gpu", "integrated"))
    probe = dict(logits=torch.tensor([[.1, .2, .3]]), loss=torch.tensor(.7),
        gradients={str(i): torch.tensor([.1]) for i in range(49)},
        parameters={str(i): torch.tensor([.2]) for i in range(49)},
        moments={str(i): dict(step=torch.tensor(242.), exp_avg=torch.tensor([.01]),
            exp_avg_sq=torch.tensor([.001])) for i in range(49)})
    same = compare_state(probe, probe)
    assert same["passed"] and same["categories"]["gradients"]["tensors"] == 49
    assert same["categories"]["moments"]["tensors"] == 147
    wrong_gradient = dict(probe, gradients=dict(probe["gradients"], **{"0": torch.tensor([.1002])}))
    assert not compare_state(probe, wrong_gradient)["passed"]
    assert compare_state(probe, wrong_gradient)["categories"]["gradients"]["mismatched_elements"] == 1
    tie = dict(probe, logits=torch.tensor([[.1, .2, .2]]))
    changed_argmax = dict(tie, logits=torch.tensor([[.1, .2, .2000001]]))
    report = compare_state(tie, changed_argmax)
    assert report["categories"]["logits"]["passed"]
    assert report["argmax_disagreements"] == 1 and not report["passed"]
    fake = FakeCUDA()
    journal = here/"full_runner_CPU_stage_test.jsonl"
    assert not journal.exists()
    profile = MemoryProfile(SimpleNamespace(cuda=fake), lambda stage: None,
        journal=journal, identity=dict(test="CPU fake allocator"), GPU_budget_bytes=1024)
    with profile.stage("profile_context_forward"):
        fake.set_usage(100, 200)
        with profile.stage("_second"):
            fake.set_usage(250, 400)
        fake.set_usage(80, 150)
    with profile.stage("profile_decoder_forward_backward"):
        fake.set_usage(200, 300)
        fake.set_usage(180, 250)
    snapshot = profile.snapshot()
    assert fake.resets == 2
    assert snapshot["run_peak_allocated_bytes"] == 250
    assert snapshot["run_peak_reserved_bytes"] == 400
    assert snapshot["stage_memory"][0]["interval_peak_reserved_bytes"] == 400
    assert snapshot["stage_memory"][1]["interval_peak_reserved_bytes"] == 300
    rejected = False
    try:
        with profile.stage("profile_peak_budget_test"):
            fake.set_usage(500, 1200)
            fake.set_usage(100, 200)
    except ValueError:
        rejected = True
    assert rejected and not profile.top_active
    assert profile.run_reserved_peak == 1200
    assert len(journal.read_text().splitlines()) == 3
    assert not torch.cuda.is_initialized()
    receipt = dict(schema="full_context_runner_CPU_semantics_v1", status="pass", CUDA_initialized=False,
        source_refs=sources+[ref(__file__)], strict_49_gradient_and_147_moment_tensor_comparison=True,
        gradient_mismatch_rejected=True, within_tolerance_argmax_change_rejected=True,
        per_stage_reset_and_run_max_preserved=True, transient_soft_budget_excess_rejected=True,
        full_schedule=full_schedule(1), scope="CPU comparator and fake-allocator bookkeeping only; no real device allocation, observations, labels, training or checkpoint loaded.")
    write(here/"full_runner_cpu_semantics.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
