from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OWNED = ROOT / "research/reviews/census_native_qualification_runner_v1_author"
SOURCE = "src/arggnn/census_native_qualification_runner_v1.py"
CONFIG = "configs/census_native_qualification_runner_v1.json"
DOCS = "docs/census_native_qualification_runner_v1.md"
TESTS = "tests/test_census_native_qualification_runner_v1.py"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


config = json.loads((ROOT / CONFIG).read_text())
artifact_sha256 = {p: sha256(ROOT / p) for p in (SOURCE, CONFIG, DOCS, TESTS)}
dependencies = config["source_sha256"]
require(len(dependencies) == 25, "Expected 25 dependencies")
require(set(dependencies) == set(config["required_source_paths"]), "Dependency closure differs")
for p, expected in dependencies.items():
    require(sha256(ROOT / p) == expected, f"Dependency changed: {p}")
executed_source_sha256 = {**dependencies, SOURCE: artifact_sha256[SOURCE]}

final_test_path = OWNED / "pytest_attempt_09.json"
final_test = json.loads(final_test_path.read_text())
require(final_test["exit_code"] == 0, "Final tests did not pass")
require(final_test["before"] == final_test["after"], "Files changed during final test")
for p, expected in final_test["after"].items():
    require(artifact_sha256[p] == expected, f"Final test no longer binds {p}")
require(sha256(OWNED / "pytest_attempt_09.txt") == final_test["output_sha256"], "Final output drift")
require("21 passed in 23.09s" in (OWNED / "pytest_attempt_09.txt").read_text(), "Unexpected final test output")

attempts = []
for p in sorted(OWNED.glob("pytest_attempt_*.json")):
    record = json.loads(p.read_text())
    output = p.with_suffix(".txt")
    require(sha256(output) == record["output_sha256"], f"Attempt output drift: {output}")
    attempts.append({"receipt_ref": ref(p), "output_ref": ref(output), "exit_code": record["exit_code"], "seconds": record["seconds"]})
require(len(attempts) == 9, "Expected all nine preserved test attempts")

metrics_path = OWNED / "handmade_metrics.json"
metrics = json.loads(metrics_path.read_text())
inspection_ref = metrics["inspection_evidence_ref"]
inspection_path = ROOT / inspection_ref["path"]
require(sha256(inspection_path) == inspection_ref["sha256"], "Inspection evidence drift")
require(inspection_path.stat().st_size == inspection_ref["bytes"], "Inspection evidence size drift")
inspection = json.loads(inspection_path.read_text())
require(inspection["source_sha256"] == executed_source_sha256, "Refreshed inspection source closure differs")
require(inspection["fixture_mode"] is True, "Author evidence must be fixture mode")
require(inspection["status"] == "inspection_complete_pending_root_freeze", "Incomplete fixture inspection")
require(len(inspection["cells"]) == 32 and not inspection["failed_cells"], "Incomplete fixture tile coverage")
require(inspection["verified_file_bytes"][SOURCE]["sha256"] == artifact_sha256[SOURCE], "Inspection runner pin differs")
require(set(metrics["actual_runtime_code_file_sha256"]) == set(config["runtime_files"]), "Runtime closure differs")
for p, expected in metrics["actual_runtime_code_file_sha256"].items():
    require(sha256(ROOT / p) == expected, f"Installed runtime drift: {p}")
for k in ("real_small_cell_checks", "full280_single_tree_checks"):
    require(metrics[k]["status"] == "qualified" and len(metrics[k]["checks"]) == 10, f"Missing real handmade checks: {k}")
    require(set(map(tuple, metrics[k]["performed_controls"])) == {(d, c) for d in config["dtypes"] for c in config["controls"]}, f"Handmade control coverage differs: {k}")
for k in ("actual_observation_model_target_arrays_opened", "actual_admission_created", "GPU_used", "fitting_performed"):
    require(metrics[k] is False, f"Unexpected author scope: {k}")

for p in sorted(OWNED.iterdir()):
    if p.is_file() and p.name != "receipt.json" and p.suffix in {".py", ".md", ".json", ".txt"}:
        artifact_sha256[str(p.relative_to(ROOT))] = sha256(p)
for p in sorted(OWNED.glob("collector_attempt_*/inspect/evidence.json")):
    artifact_sha256[str(p.relative_to(ROOT))] = sha256(p)
old_evidence = OWNED / "collector_tmp/inspect/evidence.json"
if old_evidence.is_file():
    artifact_sha256[str(old_evidence.relative_to(ROOT))] = sha256(old_evidence)

receipt = {
    "schema": "census_native_qualification_runner_v1_author_seal",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "author_source_handmade_CPU_qualified_pending_fresh_independent_review_and_root_actual_admission",
    "hypothesis_id": "H021",
    "role": "runner_author_not_independent_reviewer_of_this_runner",
    "actual_admission": False,
    "root": str(ROOT),
    "artifact_sha256": artifact_sha256,
    "dependency_sha256": dependencies,
    "executed_source_sha256_for_actual_admission": executed_source_sha256,
    "all_25_config_dependencies_verified": True,
    "final_test_receipt_ref": ref(final_test_path),
    "final_test_output_ref": ref(OWNED / "pytest_attempt_09.txt"),
    "final_test_bound_hashes": final_test["after"],
    "final_test_result": {"passed": 21, "pytest_seconds": 23.09, "wrapper_seconds": final_test["seconds"], "exit_code": 0},
    "preserved_test_attempts": attempts,
    "handmade_metrics_ref": ref(metrics_path),
    "current_fixture_inspection_evidence_ref": inspection_ref,
    "current_fixture_inspection_binds_exact_runner_and_25_dependencies": True,
    "installed_runtime_code_file_sha256": metrics["actual_runtime_code_file_sha256"],
    "runtime_identity": metrics["runtime_identity"],
    "default_shipped_model_initial_state_sha256_by_dtype": metrics["default_shipped_model_initial_state_sha256_by_dtype"],
    "handmade_support": metrics["handmade_TS"],
    "event_collision_counts": metrics["event_collision_counts"],
    "real_handmade_numerical_cells": {
        "small_two_tree_dtype_control_checks": len(metrics["real_small_cell_checks"]["checks"]),
        "full280_single_tree_dtype_control_checks": len(metrics["full280_single_tree_checks"]["checks"]),
        "small_two_tree_donor_range": metrics["real_small_cell_donor_range"],
        "maximum_small_two_tree_float64_directional_absolute_error": max(c["directional_float64"]["absolute_error"] for c in metrics["real_small_cell_checks"]["checks"] if c["directional_float64"]),
        "all32_phase_coverage_uses_fixture_callback_probes_not_32_actual_model_checks": True,
        "synthetic_objective": "fixed coefficients on mean logits, no biological targets",
        "disposable_AdamW_is_unit_qualification_not_fitting": True,
    },
    "preserved_corrections": [
        "Attempt 01 failed on a test expecting unrelated-parent decoders not to run after a corrupted parent. Global closure preflight was strengthened and the expected global failure corrected.",
        "Source audit corrected structurally inactive eta handling for a one-tree cell with no turnover boundary and added a real full280 single-tree qualification test.",
        "No numerical tolerance was relaxed."
    ],
    "fixture_limits": [
        "Mock admissions, reviews, resources and runtime pins inside fixture evidence are fixture data, not actual root admission.",
        "The author receipt cannot substitute for a fresh independent runner review.",
        "The default shipped state hashes and installed runtime hashes require independent root verification and freezing.",
        "Earlier collector_tmp evidence is preserved and does not establish the current source seal.",
        "Actual native joins, all32 actual numerical checks, actual resource behavior and predictive utility remain unrun."
    ],
    "work_not_performed": [
        "Actual project observation/model/target arrays",
        "Original-paper data execution or reproduction",
        "Genuine comparator experiments",
        "Native inference or simulations",
        "Fitting, GPU jobs or package installation",
        "Git operations or ledger/decision edits",
        "Global storage scans"
    ],
    "prior_sealed_dependencies_unchanged": True,
    "minimum_admission_contract": {
        "schema": "census_native_qualification_admission_v1",
        "status": "active",
        "fields": ["schema", "status", "job_id", "phase", "created_at_utc", "expires_at_utc", "refs", "source_sha256", "observed_artifact_sha256", "output_directory", "expected_certificates", "inspection_manifest", "initial_state_sha256_by_dtype", "group_permutation"],
        "exact_refs": ["runner_config", "runtime", "storage", "launcher", "native_manifest", "prepared_manifest", "feature_lineage", "runner_review"],
        "source_sha256": "Exactly executed_source_sha256_for_actual_admission, containing runner plus25 dependencies",
        "observed_artifact_sha256": "Exact original byte closure, normally52 files, no targets/truth/oracles",
        "inspection_numerical_fields": "expected_certificates {}, inspection_manifest null, initial_state_sha256_by_dtype {}, group_permutation []",
        "numerical_requires": "New active job/storage/launcher, externally hashed complete successful inspection,32 expected certificates keyed parent_id:tile_id, both initial-state hashes and frozen complete184-entry Q permutation",
        "fresh_review_required": "Full fresh independent receipt and review binding on exact runner/config, fixture reviews forbidden",
        "limits": "Admission age/duration <=24h; storage snapshot <=1h; used+new+concurrent below450GB; reserved new bytes >=1GB; output cap1GB; CPU main interpreter and one thread",
        "details_path": DOCS,
        "command_shapes_path": "research/reviews/census_native_qualification_runner_v1_author/README.md",
    },
}
receipt_path = OWNED / "receipt.json"
with receipt_path.open("x") as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({
    "status": receipt["status"],
    "artifact_sha256": {p: artifact_sha256[p] for p in (SOURCE, CONFIG, DOCS, TESTS)},
    "receipt_ref": ref(receipt_path),
    "final_test_receipt_ref": receipt["final_test_receipt_ref"],
    "final_test_output_ref": receipt["final_test_output_ref"],
    "metrics_ref": receipt["handmade_metrics_ref"],
    "inspection_ref": inspection_ref,
    "dependencies_verified": len(dependencies),
}, indent=2))
