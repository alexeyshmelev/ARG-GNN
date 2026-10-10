"""Seal final H021 core author artifacts without touching the root registry."""
from pathlib import Path
import datetime, hashlib, json
root = Path(__file__).resolve().parents[3]
assert Path.cwd() == root
dest = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256((root / path).read_bytes()).hexdigest()
core = ["src/arggnn/census_cut_exchange_v1.py", "tests/test_census_cut_exchange_v1.py", "docs/census_cut_exchange_v1.md"]
final = json.loads((dest/"pytest_attempt_03.json").read_text())
assert final["exit_code"] == 0
assert "36 passed in 4.23s" in (dest/"pytest_attempt_03.txt").read_text()
assert all(sha(path) == value for path,value in final["source_sha256"].items())
metrics = json.loads((dest/"tiny_fixture_numbers.json").read_text())
assert all(sha(path) == value for path,value in metrics["source_sha256"].items())
inventory = json.loads((root/"research/reviews/biological_candidate_efficiency_v1/source_audit_receipt.json").read_text())["source_sha256"]
inventory.update({"src/arggnn/anchor_spectral_compact_v1.py": "35cd11e92499a997dd9623f937797f2c482886698e2a79b48d0a3eb8aa87bf9c", "tests/test_anchor_spectral_compact_v1.py": "4b5394572ab3bef233652a9322711eb13bbd1d80330107a1abdc93a4c971701a"})
frozen = {path: {"expected_sha256": expected, "actual_sha256": sha(path), "matches": sha(path)==expected} for path,expected in inventory.items()}
assert all(item["matches"] for item in frozen.values())
registry = json.loads((root/"research/experiment_decisions.json").read_text())
registration = next(item for item in registry["decisions"] if item["id"] == "H021")
assert any(item["id"] == "census_cut_exchange_proto_v1_source_only" for item in registration["registered_runs"])
names = ["README.md", "run_pytest.py", "collect_tiny_fixture_numbers.py", "seal_author_core.py", "tiny_fixture_numbers.json"]
names += [f"pytest_attempt_{attempt:02d}.{suffix}" for attempt in (1,2,3) for suffix in ("json","txt")]
artifacts = {path: sha(path) for path in core}
artifacts.update({str((dest/name).relative_to(root)): sha((dest/name).relative_to(root)) for name in names})
receipt = {"schema": "h021_census_cut_exchange_v1_author_receipt", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "project_root": str(root), "condition": "census_cut_exchange_proto_v1_source_only", "hypothesis": "H021", "status": "source_only_core_handmade_CPU_qualified_pending_independent_review", "registered_record_observed": registration, "registered_record_canonical_sha256": hashlib.sha256(json.dumps(registration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "artifact_sha256": artifacts, "frozen_source_integrity": frozen, "final_test": final, "tiny_fixture_numbers": "research/reviews/census_cut_exchange_v1_author/tiny_fixture_numbers.json", "actions": {"handmade_CPU_tests": True, "tiny_fixture_parity_metrics": True, "actual_data_or_holdout_arrays": False, "prepared_bank_arrays": False, "native_inference": False, "model_fit": False, "GPU_or_CUDA_work": False, "profiling_or_performance_benchmark": False, "packages_installed": False, "git_used": False, "registry_modified_by_author": False, "frozen_source_or_config_modified": False, "recursive_or_storage_scan": False, "Mac_project_writes": False}, "limitations": ["Whole-tree encoder inputs and gather interface only in this core bundle; neural encoder and actual-bank/scaler/feature joins require separate qualification.", "No scientific benefit, measured performance, actual support statistics, native parity, model-wide PSD or first/novel generic kernel claim.", "Strict scored/context observed date <350; older/equal pure references are proxy evidence only.", "Root alone owns independent review, ledger integration, Git and subsequent training admission."]}
(dest/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
binding = {"schema": "h021_census_cut_exchange_v1_core_binding", "receipt_sha256": sha((dest/"receipt.json").relative_to(root)), "core_sha256": {path:sha(path) for path in core}, "artifact_sha256": artifacts}
(dest/"core_binding.json").write_text(json.dumps(binding, indent=2)+"\n")
print(json.dumps({"receipt_sha256": binding["receipt_sha256"], "core_sha256": binding["core_sha256"], "core_binding_sha256": sha((dest/"core_binding.json").relative_to(root)), "frozen_source_count": len(frozen), "all_frozen_sources_match": True}, indent=2))
