"""Bounded stdlib-only source/metadata checks; never import scientific modules."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
HERE = ROOT / "research/reviews/C3_interleaved_prediction_successor_source_20261017_v1"
BRIDGE = ROOT / "research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v1/prediction_bridge_v1.py"
SCAN = ROOT / "research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1/six_candidate_scan_v1.py"


def require(value, message):
    if not value:
        raise ValueError(message)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def rejects(node, namespace):
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<pure-metadata-guard>", "exec"), namespace)
    except ValueError:
        return True
    return False


def run():
    checks = []
    mapping = json.loads((HERE / "source_map.v1.json").read_text())
    for ref in mapping["source_refs"]:
        path = ROOT / ref["path"]
        require(path.suffix in (".py", ".json"), "Source/compact JSON references only")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"], "Pinned source SHA: " + ref["path"])
    checks.append("sealed_audit_and_all_recorded_source_hashes")

    for path in sorted(HERE.glob("*.py")):
        compile(ast.parse(path.read_text()), str(path), "exec")
    checks.append("all_successor_Python_AST_and_compile_without_import")

    scan = (HERE / "six_candidate_scan_v1.py").read_bytes()
    require(scan == SCAN.read_bytes(), "Existing interleaved scheduler must remain byte-identical")
    require(hashlib.sha256(scan).hexdigest() == "4da2db9c217b7e4cfe25f0b9c9cf6caeaf228b36e8ced978d7d7299af3607255",
            "Exact audited scheduler")
    checks.append("byte_identical_existing_VAL003_004_interleaved_scheduler")

    original = ast.parse(BRIDGE.read_text())
    successor = ast.parse((HERE / "prediction_adapter_v1.py").read_text())
    old_predict = function(original, "predict_v1")
    predict = function(successor, "predict_v1")
    require(ast.dump(function(original, "new_candidate_models_v1")) ==
            ast.dump(function(successor, "new_candidate_models_v1")),
            "Original model construction and RNG isolation unchanged")
    phase_gate, contract_gate = predict.body[1:3]
    require(isinstance(phase_gate, ast.Expr) and isinstance(contract_gate, ast.Expr),
            "Explicit guards are the first statements after predictor docstring")
    phases = ["predict_val_%03d" % i for i in range(3)]
    for phase in phases:
        require(not rejects(phase_gate, {"req": {"phase": phase}, "require": require}),
                "Original supported phase accepted")
    for phase in ["predict_val_003", "predict_val_004", "predict_test_000", "matrix", "score", "train", "", None]:
        require(rejects(phase_gate, {"req": {"phase": phase}, "require": require}), "Outside scope blocked")
    class Contract:
        VAL = tuple("american_c3_val_%03d" % i for i in range(5))
    require(not rejects(contract_gate, {"contract": Contract, "require": require}),
            "Original five-parent contract order accepted")
    for invalid in [Contract.VAL[::-1], Contract.VAL[:3], ("american_c3_val_003",) + Contract.VAL[1:]]:
        class WrongContract:
            VAL = invalid
        require(rejects(contract_gate, {"contract": WrongContract, "require": require}), "Changed contract blocked")
    checks.append("pure_metadata_VAL000_002_scope_and_original_five_parent_order")

    candidate_gate = next(node for node in predict.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and len(node.value.args) == 2
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == "Every original epoch, in the original candidate order")
    candidates = [{"epoch": epoch} for epoch in [5, 10, 15, 20, 25, 30]]
    require(not rejects(candidate_gate, {"candidates": candidates, "require": require}), "All six epochs accepted")
    for invalid in [candidates[:5], candidates[::-1], candidates + [candidates[-1]], [candidates[0]] * 6]:
        require(rejects(candidate_gate, {"candidates": invalid, "require": require}), "Changed candidates blocked")
    support_gate = next(node for node in predict.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and len(node.value.args) == 2
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == "Original full panel, length and focal support")
    namespace = {"records": {"record_ids": list(range(10000))},
        "plan": {"haplotypes": 1600, "sequence_length_bp": 30000000}, "require": require}
    require(not rejects(support_gate, namespace), "Full original support accepted")
    for field, value in [("haplotypes", 1000), ("sequence_length_bp", 2000000)]:
        changed = dict(namespace, plan=dict(namespace["plan"], **{field: value}))
        require(rejects(support_gate, changed), "Reduced support blocked")
    require(rejects(support_gate, dict(namespace, records={"record_ids": list(range(9999))})),
            "Missing focal blocked")
    checks.append("pure_metadata_six_epoch_order_and_full1600_30Mb_10000_support")

    # Normalize exactly the disclosed non-scientific edits, then compare every
    # remaining AST node to the effective VAL003/004 bridge.
    successor.body[0] = original.body[0]
    old_import = next(n for n in original.body if isinstance(n, ast.ImportFrom)
        and n.module.endswith(".six_candidate_scan_v1"))
    new_import_index = next(i for i, n in enumerate(successor.body) if isinstance(n, ast.ImportFrom)
        and n.module.endswith(".six_candidate_scan_v1"))
    successor.body[new_import_index] = old_import
    predict.body[1] = old_predict.body[1]
    del predict.body[2]
    require(ast.dump(successor) == ast.dump(original), "Only four documented adapter edits permitted")
    checks.append("remaining_adapter_AST_identical_including_nested_metadata_and_sealing")

    audit = json.loads((ROOT / "research/reviews/C3_prediction_efficiency_audit_20261017_v1/audit.v1.json").read_text())
    routes = audit["routes"]
    require([r["parent_id"] for r in routes] == ["american_c3_val_%03d" % i for i in range(5)],
            "All five original parents")
    serial_raw = sum(r["full_successful_route_static_counts"]["raw_feature_block_constructions"] for r in routes[:3])
    future_raw = sum(2 * r["blocks"] for r in routes[:3])
    require((serial_raw, future_raw) == (86700, 14450), "Static first-three raw constructions")
    require(serial_raw == 6 * future_raw, "Six weights share each raw bank")
    require(sum(r["full_successful_route_static_counts"]["learned_encoder_block_calls"] for r in routes) == 144264,
            "Learned encoders remain independent, unchanged")
    require(sum(r["full_successful_route_static_counts"]["raw_feature_block_constructions"] for r in routes) == 96294
        and sum(2 * r["blocks"] for r in routes) == 24044, "Static all-five totals")
    checks.append("sealed_audit_static_counts_only_no_observed_progress_or_speedup")

    selection = ast.parse((ROOT / "src/arggnn/american_c3_gnn_selection_isolated_v2.py").read_text())
    matrix_fn = function(selection, "validation_matrix_v1")
    matrix_gate = next(node for node in matrix_fn.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and len(node.value.args) == 2
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == "Complete original5VAL x six-checkpoint prediction barrier before truth resolution")
    epochs = (5, 10, 15, 20, 25, 30)
    parents = tuple("american_c3_val_%03d" % i for i in range(5))
    matrix = {"status": "sealed_C3_all30_original_VAL_predictions",
        "protocol_sha256": "pure-metadata-only", "validation_parent_ids": list(parents),
        "checkpoint_epochs": list(epochs), "all_predictions_sealed_before_VAL_truth": True,
        "metrics": "original_C3_external_evaluator_unchanged", "cells": [None] * 30}
    namespace = {"matrix": matrix, "training": {"protocol_sha256": "pure-metadata-only"},
        "VAL": parents, "CHECKPOINT_EPOCHS": epochs, "require": require}
    require(not rejects(matrix_gate, namespace), "Original all30 first barrier accepts complete metadata")
    for field, value in [("cells", [None] * 29), ("cells", [None] * 31),
                         ("all_predictions_sealed_before_VAL_truth", False),
                         ("validation_parent_ids", list(parents[:3]))]:
        require(rejects(matrix_gate, dict(namespace, matrix=dict(matrix, **{field: value}))),
                "Partial or changed matrix cannot pass original first truth barrier")
    score = function(selection, "score_validation_matrix_v1")
    authenticated = next(i for i, node in enumerate(score.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "validation_matrix_v1")
    resolver = next(i for i, node in enumerate(score.body)
        if any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
               and child.func.id == "resolve_val_truth" for child in ast.walk(node)))
    require(authenticated < resolver, "Full original matrix is authenticated before first truth resolution")
    adapter_tree = ast.parse((HERE / "prediction_adapter_v1.py").read_text())
    forbidden = {"score_validation_matrix_v1", "original_score", "resolve_val_truth",
                 "select_original_val_checkpoint_v1", "seal_validation_matrix_v1"}
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in forbidden for n in ast.walk(adapter_tree)),
            "No truth, matrix, scorer or selection callable introduced by adapter")
    checks.append("original_all30_pretruth_barrier_and_scoring_order_unchanged")

    for name in ("torch", "numpy", "torch_geometric", "tskit"):
        require(name not in sys.modules, "No numerical imports during static tests")
    checks.append("no_numerical_import_or_model_data_checkpoint_truth_execution")
    return {"schema": "C3_interleaved_successor_static_checks_v1",
        "status": "STATIC_SOURCE_AND_PURE_METADATA_PASS",
        "checks": checks, "checks_count": len(checks),
        "numerical_equivalence_tested": False, "scientific_inference_executed": False,
        "resource_or_lifecycle_checks_executed": False, "launch_or_admission_authority": False}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
