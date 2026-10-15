"""Optional future H048-style update; inert and not admitted or wired to H048.

Only per-tensor CUDA finite-result host reads are grouped. The original FP32
objective, clip, optimizer call, callback and rejection stages are retained.
No payload access, model construction, launch or CUDA action at import time.
"""
from __future__ import annotations
import time
import torch
from arggnn.marker_resolution_readout_v1 import original_grid_span_nll_v1

ORIGINAL_UPDATE_REF = {
    "path": "src/arggnn/train_continuous_clade_flank_v1.py",
    "sha256": "eda198ef04192d978433c64f921b865aa89a9304adcc76893fa2852a14d72ae4",
    "lines": [208, 244],
}
OBJECTIVE_REF = {
    "path": "src/arggnn/marker_resolution_readout_v1.py",
    "sha256": "27ecd30683024def2d408bf16250102b3252d0ae112687b9664fca086b141310",
    "lines": [369, 379],
}

def _require(ok, message):
    if not ok:
        raise ValueError(message)

def all_finite_grouped_v1(values):
    """Keep CPU checks individual; one finite-result host read per other device.

    Inputs must be tensors. Each device reduction is the logical conjunction of
    the same per-tensor isfinite(...).all() predicates as the inherited update.
    No tensor is changed, detached from training, or omitted.
    """
    groups = {}
    for value in values:
        groups.setdefault(value.device, []).append(value)
    for device, tensors in groups.items():
        if device.type == "cpu":
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                return False
        else:
            flags = [torch.isfinite(value).all() for value in tensors]
            if not bool(torch.stack(flags).all()):
                return False
    return True

def update_v1(model, optimizer, bundle, mass, guard, *, qualify=False,
              after_optimizer_step=None):
    """Future replacement for the inherited H046 update, never live-patched."""
    guard()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device = bundle.bank.parent.device
    if qualify and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    try:
        with torch.autocast(device_type=device.type, enabled=False):
            logits = model(bundle)
            loss = original_grid_span_nll_v1(
                logits, mass, bundle.markers.bin_widths_bp.to(device))
        _require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)),
                 "Finite complete-parent FP32 grid loss")
        loss.backward()

        # Preserve the pre-clip/pre-step rejection gate.
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        _require(all(g.dtype == torch.float32 for g in gradients)
                 and all_finite_grouped_v1(gradients),
                 "Finite complete-body FP32 gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if after_optimizer_step is not None:
            after_optimizer_step()

        # Preserve separate post-step parameter and optimizer rejection gates.
        _require(all_finite_grouped_v1(list(model.parameters())),
                 "Finite updated learned state")
        states = [v for state in optimizer.state.values() for v in state.values()
                  if isinstance(v, torch.Tensor)]
        _require(all(v.dtype == torch.float32 for v in states)
                 and all_finite_grouped_v1(states),
                 "Finite FP32 independent AdamW")
        row = dict(loss=float(loss.detach()), gradient_norm=float(norm),
                   seconds=time.monotonic()-started, finite_FP32=True)
        if qualify:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                row["memory"] = dict(
                    real_GPU_qualification=True,
                    allocated_bytes=int(torch.cuda.memory_allocated(device)),
                    reserved_bytes=int(torch.cuda.memory_reserved(device)),
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
            else:
                row["memory"] = dict(real_GPU_qualification=False,
                                     device_type=device.type)
        del logits, loss, norm
    finally:
        optimizer.zero_grad(set_to_none=True)
    _require(all(p.grad is None for p in model.parameters()),
             "End gradient lifetime before next arm")
    guard()
    return row
