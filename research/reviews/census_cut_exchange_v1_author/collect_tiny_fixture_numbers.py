"""Collect tiny-fixture parity numbers only, never performance or project data."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, platform
import torch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1)
root = Path(__file__).resolve().parents[3]
assert Path.cwd() == root
dest = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h021_handmade_tests", root / "tests/test_census_cut_exchange_v1.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

def discrepancy(actual, expected, dtype, gradient=False):
    if dtype == torch.float64:
        rt, at = (1e-8, 1e-10) if gradient else (1e-10, 1e-12)
    else:
        rt, at = (1e-3, 1e-5) if gradient else (2e-4, 2e-5)
    difference = (actual.detach() - expected.detach()).abs()
    return {"max_absolute_error": float(difference.max()),
            "max_tolerance_fraction": float((difference / (at + rt * expected.detach().abs())).max())}

rows = []
for dtype in (torch.float64, torch.float32):
    for control in ("proposed", "eta0", "permuted_groups", "raw_diagonal"):
        b = t.bank(dtype)
        if control == "permuted_groups":
            b = t.permuted_group_control_v1(b, torch.tensor([2, 3, 1, 0]))
        values = t.leaves(b, dtype)
        query, reference, exchange, damping, eta, mixture = t.params(values)
        sources = t.bundle(b, query, reference, exchange)
        actual = t.census_cut_action_v1(b, sources, exchange, damping, eta, control=control, readout_chunk_size=2)
        expected = t.oracle_action(b, t.INTERVALS, sources, exchange, damping, eta, control)
        ap, ep = t.parts(actual, mixture), t.parts(expected, mixture)
        gradients = [torch.autograd.grad(t.objective(x[2], x[1]), values, retain_graph=True, allow_unused=True) for x in (ap, ep)]
        metrics = {name: discrepancy(x, y, dtype) for name, x, y in zip(("numerator", "mass", "normalized"), ap, ep)}
        metrics["raw_action"] = discrepancy(actual, expected, dtype)
        gradient_metrics = {name: ("inactive" if x is None else discrepancy(x, y, dtype, True))
                            for name, x, y in zip(("query", "reference", "raw_exchange", "raw_damping", "raw_eta", "mixture_logits"), *gradients)}
        assert all(item["max_tolerance_fraction"] <= 1 for item in metrics.values())
        assert all(item == "inactive" or item["max_tolerance_fraction"] <= 1 for item in gradient_metrics.values())
        rows.append({"dtype": str(dtype), "control": control, "outputs": metrics, "VJP": gradient_metrics,
                     "minimum_mass": float(ap[1].detach().min()), "maximum_abs_normalized": float(ap[2].detach().abs().max())})

b = t.bank()
values = t.leaves(b)
query, reference, exchange, damping, eta, mixture = t.params(values)
points = [0., .2, .5, .7, .999]
gram = torch.cat([torch.cat([t.oracle_path(b, t.INTERVALS, exchange, damping, eta, x, y, 0, 0, "proposed") for y in points], 1) for x in points], 0)
p0, p1 = (t.dense_projection(x, torch.float64) for x in b.partitions)
rho = t.clade_turnover_rho_v1(*b.partitions)
jump = torch.diag(rho ** eta[0, 0])

def evaluate(inputs):
    q, r, ex, dam, et, mix = t.params(inputs)
    msg = t.census_cut_messages_v1(b, q, r, ex, dam, et, mix)
    return t.objective(msg.targets, msg.mass)
grad = torch.autograd.grad(evaluate(values), values)
generator = torch.Generator().manual_seed(951)
directions = tuple(torch.randn(x.shape, dtype=x.dtype, generator=generator) for x in values)
exact = sum((g * d).sum() for g, d in zip(grad, directions))
step = 2e-4
f = {k: evaluate(tuple(x.detach() + k * step * d for x, d in zip(values, directions))) for k in (-2, -1, 1, 2)}
finite_difference = (f[-2] - 8*f[-1] + 8*f[1] - f[2]) / (12*step)

receipt = {"schema": "h021_census_cut_exchange_v1_tiny_fixture_numbers", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Handmade CPU float64/float32 fixtures only, Q4/R2/G2/A3/S3/H2; no actual data, fit, CUDA, or performance benchmark.",
    "torch_version": torch.__version__, "python_version": platform.python_version(),
    "source_sha256": {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ("src/arggnn/census_cut_exchange_v1.py", "tests/test_census_cut_exchange_v1.py")},
    "oracle": "Independent dense full physical E/J path, reverse transpose, torch.matrix_exp with one unrelated 1x1 constant padding block; no production eigensolve or dense matrix exponential.",
    "comparisons": rows,
    "ordering_fixture": {"projection_commutator_frobenius": float(torch.linalg.matrix_norm(p0@p1-p1@p0)), "projection_jump_commutator_frobenius": float(torch.linalg.matrix_norm(p0@jump-jump@p0)), "sample_aligned_rho": rho.tolist()},
    "completed_kernel_fixture": {"points": points, "branch": 0, "channel": 0, "symmetry_max_absolute_error": float((gram-gram.T).abs().detach().max()), "minimum_eigenvalue": float(torch.linalg.eigvalsh(gram).detach().min()), "certificate_scope": "One fixed-geometry unnormalized Q4 toy kernel only, not the normalized network or a cross-input kernel."},
    "five_point_directional_derivative": {"step": step, "autograd": float(exact.detach()), "finite_difference": float(finite_difference.detach()), "absolute_error": float((exact-finite_difference).abs().detach()), "rtol": 2e-8, "atol": 1e-9}}
(dest/"tiny_fixture_numbers.json").write_text(json.dumps(receipt, indent=2)+"\n")
for dtype in ("torch.float64", "torch.float32"):
    group = [x for x in rows if x["dtype"] == dtype]
    print(json.dumps({"dtype": dtype, "maximum_output_absolute_error": max(v["max_absolute_error"] for x in group for v in x["outputs"].values()), "maximum_VJP_absolute_error": max(v["max_absolute_error"] for x in group for v in x["VJP"].values() if isinstance(v, dict)), "maximum_output_tolerance_fraction": max(v["max_tolerance_fraction"] for x in group for v in x["outputs"].values()), "maximum_VJP_tolerance_fraction": max(v["max_tolerance_fraction"] for x in group for v in x["VJP"].values() if isinstance(v, dict))}))
print(json.dumps({k: receipt[k] for k in ("ordering_fixture", "completed_kernel_fixture", "five_point_directional_derivative")}, indent=2))
