"""Tiny synthetic CPU checks only; no real model, data or GPU execution."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "optional_grouped_update", HERE/"grouped_finite_update_candidate_v1.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.1, .2, .3, .4]))
    def forward(self, bundle):
        return self.weight[None, None, :].expand(2, 1, 4)

class InjectedAdamW(torch.optim.AdamW):
    def __init__(self, params, injection=None):
        super().__init__(params, lr=.001, weight_decay=.0001,
                         betas=(.9, .999), eps=1e-8)
        self.injection = injection
    def step(self, closure=None):
        value = super().step(closure)
        p = next(iter(self.state))
        with torch.no_grad():
            if self.injection == "parameter_nonfinite":
                p.fill_(float("inf"))
            elif self.injection == "moment_nonfinite":
                self.state[p]["exp_avg"].fill_(float("nan"))
            elif self.injection == "CPU_step_nonfinite":
                self.state[p]["step"].fill_(float("nan"))
            elif self.injection == "CPU_step_malformed_dtype":
                self.state[p]["step"] = torch.tensor(1, dtype=torch.int64)
        return value

def case(name, *, expected=None, bad_target=False, bad_gradient=False, injection=None):
    model = TinyModel()
    optimizer = InjectedAdamW(model.parameters(), injection)
    bundle = SimpleNamespace(
        bank=SimpleNamespace(parent=torch.zeros(1, dtype=torch.long)),
        markers=SimpleNamespace(bin_widths_bp=torch.tensor([100., 100.])))
    mass = torch.tensor([[[100., 0., 0., 0.]], [[0., 100., 0., 0.]]])
    if bad_target:
        mass[0, 0, 0] = -100.
    if bad_gradient:
        model.weight.register_hook(lambda g: torch.full_like(g, float("nan")))
    counts = dict(guards=0, steps=0)
    def guard():
        counts["guards"] += 1
    def after():
        counts["steps"] += 1
    message = None
    try:
        row = candidate.update_v1(model, optimizer, bundle, mass, guard,
                                  qualify=True, after_optimizer_step=after)
        assert expected is None, (name, "unexpected success")
        assert row["finite_FP32"] is True
        assert row["memory"] == dict(real_GPU_qualification=False, device_type="cpu")
        assert counts == dict(guards=2, steps=1)
        assert optimizer.state[model.weight]["step"].device.type == "cpu"
        assert optimizer.state[model.weight]["step"].dtype == torch.float32
    except ValueError as error:
        message = str(error)
        assert expected is not None and expected in message, (name, message)
        assert counts["guards"] == 1
        assert counts["steps"] == (1 if injection else 0)
        if not injection:
            assert not optimizer.state, (name, "optimizer ran before rejection")
    assert all(p.grad is None for p in model.parameters())
    return dict(name=name, passed=True, failure_message=message, **counts)

def main():
    assert candidate.all_finite_grouped_v1([])
    assert candidate.all_finite_grouped_v1([torch.tensor([0., 1.])])
    assert not candidate.all_finite_grouped_v1([torch.tensor([float("inf")])])
    results = [
        case("finite_FP32_success"),
        case("malformed_physical_mass", bad_target=True,
             expected="Exact original complete100bp physical mass"),
        case("nonfinite_gradient", bad_gradient=True,
             expected="Finite complete-body FP32 gradients"),
        case("post_step_parameter_nonfinite", injection="parameter_nonfinite",
             expected="Finite updated learned state"),
        case("post_step_moment_nonfinite", injection="moment_nonfinite",
             expected="Finite FP32 independent AdamW"),
        case("CPU_step_nonfinite", injection="CPU_step_nonfinite",
             expected="Finite FP32 independent AdamW"),
        case("CPU_step_malformed_dtype", injection="CPU_step_malformed_dtype",
             expected="Finite FP32 independent AdamW"),
    ]
    ref = lambda p: dict(path=str(p.relative_to(HERE)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    report = dict(
        schema="H048_optional_grouped_finite_synthetic_CPU_checks_v1",
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
        device="cpu", synthetic_only=True, real_model_or_payload_used=False,
        GPU_execution=False, benchmark=False, FP64_used=False,
        candidate_ref=ref(HERE/"grouped_finite_update_candidate_v1.py"),
        test_source_ref=ref(Path(__file__)), cases=results,
        all_passed=all(x["passed"] for x in results),
        limitations="CUDA grouping path is source-inspected only; no real-model parity or speed claim.")
    (HERE/"synthetic_CPU_checks_actual_v1.json").write_text(
        json.dumps(report, sort_keys=True, indent=2)+"\n")
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    main()
