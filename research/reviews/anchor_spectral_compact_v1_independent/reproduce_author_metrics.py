"""Reproducible CPU handmade parity measurements, no actual data arrays."""
from pathlib import Path
import hashlib
import json
import os
import runpy
import torch

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OUT = ROOT / "research/reviews/anchor_spectral_compact_v1_independent"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1)
t = runpy.run_path(str(ROOT / "tests/test_anchor_spectral_compact_v1.py"))

def error(actual, expected, dtype, gradient=False):
    if dtype == torch.float64:
        rtol, atol = (1e-8, 1e-10) if gradient else (1e-10, 1e-12)
    else:
        rtol, atol = (1e-3, 1e-5) if gradient else (2e-4, 2e-5)
    delta = (actual - expected).abs().detach()
    bound = atol + rtol * expected.abs().detach()
    return {"max_absolute_error": float(delta.max()),
            "max_acceptance_units": float((delta / bound).max()),
            "rtol": rtol, "atol": atol}

rows = []
for dtype in (torch.float64, torch.float32):
    for coupling in ("full", "scalar", "diagonal", "gain_matched_diagonal"):
        bank, native, layout, operators, geometry = t["fixture"](dtype, reorder=[2, 0, 1])
        values = t["inputs"](layout, dtype)
        query, reference, rates, damping, mixture = t["parameters"](values)
        sources = t["source_bundle"](layout, operators, query, reference, rates)
        actual = t["anchor_spectral_compact_action_v1"](layout, geometry, sources, rates, damping,
                                                         coupling=coupling, readout_chunk_size=2)
        original = t["anchor_kernel_action"](native, operators, sources, rates, damping, coupling=coupling)
        oracle = t["dense_action"](bank, layout, operators, sources, rates, damping, coupling)
        parts = [t["mixed_messages"](x, mixture) for x in (actual, original, oracle)]
        grads = [torch.autograd.grad(t["objective"](p[2], p[1]), values, retain_graph=True) for p in parts]
        row = {"fixture": "noncommuting_serialized_boundary_collision", "dtype": str(dtype),
               "coupling": coupling, "raw_vs_original": error(actual, original, dtype),
               "raw_vs_dense_oracle": error(actual, oracle, dtype), "quantities": {}}
        for i, name in enumerate(("numerator", "mass", "normalized")):
            row["quantities"][name] = {"vs_original": error(parts[0][i], parts[1][i], dtype),
                                       "vs_oracle": error(parts[0][i], parts[2][i], dtype)}
        row["gradient_families"] = {}
        for i, name in enumerate(("query_signal", "reference_signal", "raw_log_rate", "raw_log_damping", "mixture_logits")):
            row["gradient_families"][name] = {"vs_original": error(grads[0][i], grads[1][i], dtype, True),
                                             "vs_oracle": error(grads[0][i], grads[2][i], dtype, True)}
        rows.append(row)
    positions = torch.linspace(0., .999, 1025, dtype=torch.float64)
    _, native, layout, operators, geometry = t["fixture"](dtype, intervals=[[0., 1.]], positions=positions.tolist())
    values = t["inputs"](layout, dtype)
    query, reference, rates, damping, mixture = t["parameters"](values)
    sources = t["source_bundle"](layout, operators, query, reference, rates)
    actual = t["anchor_spectral_compact_action_v1"](layout, geometry, sources, rates, damping, readout_chunk_size=17)
    original = t["anchor_kernel_action"](native, operators, sources, rates, damping)
    g1 = torch.autograd.grad(actual.square().sum(), values[2:4], retain_graph=True)
    g2 = torch.autograd.grad(original.square().sum(), values[2:4])
    rows.append({"fixture": "1025_readouts_constant_hold", "dtype": str(dtype), "coupling": "full",
                 "compact_segments": layout.num_segments, "original_segments": len(native.segment_graph),
                 "raw_vs_original": error(actual, original, dtype),
                 "raw_log_rate_gradient_vs_original": error(g1[0], g2[0], dtype, True),
                 "raw_log_damping_gradient_vs_original": error(g1[1], g2[1], dtype, True)})

def leaves(value):
    if isinstance(value, dict):
        if "max_absolute_error" in value:
            yield value
        else:
            for child in value.values():
                yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)
summary = {"max_absolute_error_across_reported_quantities": max(x["max_absolute_error"] for x in leaves(rows)),
           "max_acceptance_units_across_reported_quantities": max(x["max_acceptance_units"] for x in leaves(rows))}
receipt = {"schema": "compact_handmade_parity_metrics_v1", "torch": str(torch.__version__),
           "device": "cpu", "scope": "explicit synthetic tensors only", "rows": rows,
           "summary": summary, "note": "No performance benchmark or scientific predictive result. Gradients on the dense-readout fixture sum over all outputs, so their absolute scale is larger than small-fixture VJPs."}
(OUT / "parity_metrics.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
print(json.dumps({"summary": summary, "rows": len(rows), "dense_readout_rows": rows[-1]}, indent=2))
