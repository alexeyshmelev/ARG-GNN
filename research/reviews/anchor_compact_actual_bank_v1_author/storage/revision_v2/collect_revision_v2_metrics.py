"""Versioned handmade-only positive equivalence and I1 negative replay."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from dataclasses import replace
import torch
import arggnn.anchor_compact_actual_bank_v1 as current

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
assert Path.cwd().resolve() == ROOT and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1)
OUT = ROOT/"research/reviews/anchor_compact_actual_bank_v1_author/storage/revision_v2"
BEFORE = OUT/"before"
old_source = BEFORE/"src/arggnn/anchor_compact_actual_bank_v1.py"
assert hashlib.sha256(old_source.read_bytes()).hexdigest() == "5830a3471ce3e7145db1731a71ca58b3fa13161f6855fe805fb2e29d9827c33f"
name = "arggnn._compact_actual_revision_v2_archived_source"
spec = importlib.util.spec_from_file_location(name, old_source)
legacy = importlib.util.module_from_spec(spec)
sys.modules[name] = legacy
spec.loader.exec_module(legacy)
spec = importlib.util.spec_from_file_location("compact_actual_revision_v2_fixtures",
    ROOT/"tests/test_anchor_compact_actual_bank_v1.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

def function_ast(path, name):
    tree = ast.parse(path.read_text())
    item = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.dump(item, include_attributes=False)

assert function_ast(old_source, "comparison_v1") == function_ast(
    ROOT/"src/arggnn/anchor_compact_actual_bank_v1.py", "comparison_v1")
assert function_ast(old_source, "qualification_objective_v1") == function_ast(
    ROOT/"src/arggnn/anchor_compact_actual_bank_v1.py", "qualification_objective_v1")
assert function_ast(old_source, "optimizer_v1") == function_ast(
    ROOT/"src/arggnn/anchor_compact_actual_bank_v1.py", "optimizer_v1")

def capture_action(module, model, bank, view):
    saved_native = module.anchor_kernel_action
    saved_compact = module.anchor_spectral_compact_action_v1
    seen, arrays = [], {}
    def native(*args, **kwargs):
        result = saved_native(*args, **kwargs)
        seen.append(("native", kwargs["coupling"]))
        arrays["native"] = result.detach().clone()
        return result
    def compact(*args, **kwargs):
        result = saved_compact(*args, **kwargs)
        seen.append(("compact", kwargs["coupling"]))
        arrays["compact"] = result.detach().clone()
        return result
    module.anchor_kernel_action, module.anchor_spectral_compact_action_v1 = native, compact
    try:
        result = module.compare_action_VJPs_v1(model, bank, view)
    finally:
        module.anchor_kernel_action, module.anchor_spectral_compact_action_v1 = saved_native, saved_compact
    return result, seen, arrays

positives = []
for dtype in (torch.float64, torch.float32):
    for coupling in current.COUPLINGS:
        for stress in (False, True):
            model, compact_model, bank, view = fixtures.fixture(dtype, coupling, stress)
            old_action, old_seen, old_arrays = capture_action(legacy, model, bank, view)
            new_action, new_seen, new_arrays = capture_action(current, model, bank, view)
            old_step = legacy.compare_model_step_v1(model, compact_model, bank, view)
            new_step = current.compare_model_step_v1(model, compact_model, bank, view)
            expected_seen = [("native", coupling), ("compact", coupling)]
            assert old_seen == new_seen == expected_seen
            assert old_action == {k:v for k,v in new_action.items() if k != "cell_identity"}
            assert old_step == {k:v for k,v in new_step.items() if k != "cell_identity"}
            assert old_action["status"] == new_action["status"] == "PASS"
            assert old_step["status"] == new_step["status"] == "PASS"
            assert all(torch.equal(old_arrays[k], new_arrays[k]) for k in old_arrays)
            positives.append(dict(dtype=str(dtype), coupling=coupling, stress=stress,
                performed_cell_identity=new_action["cell_identity"],
                observed_dispatch=new_seen, old_new_raw_outputs_bitwise_equal=True,
                old_new_action_numeric_receipts_identical=True,
                old_new_full_step_numeric_receipts_identical=True,
                action_comparisons=len(new_action["checks"]),
                model_comparisons=len(new_step["checks"])))

negatives = []
for requested in (*current.COUPLINGS[1:], "mislabeled_geometry"):
    model, _, bank, view = fixtures.fixture(torch.float64)
    if requested == "mislabeled_geometry":
        view = replace(view, geometry_name="plain")
    else:
        view = current.derive_compact_actual_bank_v1(bank, "age", requested)
    old_result, old_seen, _ = capture_action(legacy, model, bank, view)
    assert old_result["status"] == "PASS" and old_seen == [("native", "full"), ("compact", "full")]
    calls = []
    original_encoder = model.encoder.forward
    original_native = current.anchor_kernel_action
    original_compact = current.anchor_spectral_compact_action_v1
    def blocked(*args, **kwargs):
        calls.append("meaningful_work")
        raise AssertionError("New mismatch reached encoder/action")
    model.encoder.forward = blocked
    current.anchor_kernel_action = current.anchor_spectral_compact_action_v1 = blocked
    error = None
    try:
        current.compare_action_VJPs_v1(model, bank, view)
    except ValueError as exc:
        error = str(exc)
    finally:
        model.encoder.forward = original_encoder
        current.anchor_kernel_action, current.anchor_spectral_compact_action_v1 = original_native, original_compact
    assert error is not None and not calls
    negatives.append(dict(request=requested, model_geometry=model.geometry,
        view_geometry=view.geometry_name, model_coupling=model.coupling, view_coupling=view.coupling,
        archived_false_PASS=old_result["status"], archived_actual_dispatch=old_seen,
        repaired_status="rejected_before_encoder_and_action", error=error,
        encoder_or_action_calls=0))

# Existing positive metric collector should preserve its entire numerical JSON.
runpy.run_path(str(ROOT/"research/reviews/anchor_compact_actual_bank_v1_author/collect_handmade_metrics.py"),
               run_name="__main__")
old_metrics = BEFORE/"research/reviews/anchor_compact_actual_bank_v1_author/handmade_parity_metrics.json"
new_metrics = ROOT/"research/reviews/anchor_compact_actual_bank_v1_author/handmade_parity_metrics.json"
assert old_metrics.read_bytes() == new_metrics.read_bytes()
assert not torch.cuda.is_initialized()
receipt = dict(kind="compact_actual_bank_source_revision_v2_handmade_equivalence",
    actual_arrays_opened=False, CUDA_context_initialized=False, measured_efficiency=False,
    archived_source_sha256=hashlib.sha256(old_source.read_bytes()).hexdigest(),
    amended_source_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
    mathematical_tolerance_objective_optimizer_AST_unchanged=True,
    positives=positives, negative_I1_replays=negatives,
    positive_cells=len(positives), I1_negative_cells=len(negatives),
    old_new_raw_outputs_bitwise_equal=True,
    old_new_numeric_receipts_identical=True,
    original_3488_numeric_metric_JSON_byte_identical=True,
    metric_JSON_sha256=hashlib.sha256(new_metrics.read_bytes()).hexdigest(),
    initial_FAIL_is_preserved_not_reclassified=True)
(OUT/"revision_v2_equivalence.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in receipt.items() if k not in ("positives","negative_I1_replays")}))
