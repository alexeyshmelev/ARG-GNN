"""Seal the separate H021 inherited encoder, checking previously sealed core."""
from pathlib import Path
import datetime, hashlib, json
root=Path(__file__).resolve().parents[3]
assert Path.cwd()==root
dest=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256((root/path).read_bytes()).hexdigest()
core_dir=Path("research/reviews/census_cut_exchange_v1_author")
core_binding=json.loads((root/core_dir/"core_binding.json").read_text())
assert sha(core_dir/"core_binding.json")=="94bb947cadce54b559be5182405e2e5953fdd4eaf005d276b369bbf48a890021"
assert sha(core_dir/"receipt.json")==core_binding["receipt_sha256"]
assert all(sha(path)==value for path,value in core_binding["artifact_sha256"].items())
core_receipt=json.loads((root/core_dir/"receipt.json").read_text())
frozen={path:{"expected_sha256":item["expected_sha256"],"actual_sha256":sha(path),"matches":sha(path)==item["expected_sha256"]} for path,item in core_receipt["frozen_source_integrity"].items()}
assert all(item["matches"] for item in frozen.values())
final=json.loads((dest/"pytest_attempt_05.json").read_text())
assert final["exit_code"]==0 and "30 passed in 3.36s" in (dest/"pytest_attempt_05.txt").read_text()
assert all(sha(path)==value for path,value in final["source_sha256"].items())
numbers=json.loads((dest/"tiny_fixture_numbers.json").read_text())
assert all(sha(path)==value for path,value in numbers["source_sha256"].items())
for attempt in range(1,6):
    item=json.loads((dest/f"pytest_attempt_{attempt:02d}.json").read_text())
    assert sha((dest/f"pytest_attempt_{attempt:02d}.txt").relative_to(root))==item["output_sha256"]
for attempt in range(1,4):
    item=json.loads((dest/f"collector_attempt_{attempt:02d}.json").read_text())
    assert sha((dest/f"collector_attempt_{attempt:02d}.txt").relative_to(root))==item["output_sha256"]
assert json.loads((dest/"collector_attempt_01.json").read_text())["collector_sha256"]==sha((dest/"collect_tiny_fixture_numbers_attempt_01.py").relative_to(root))
assert json.loads((dest/"collector_attempt_03.json").read_text())["exit_code"]==0
core=["src/arggnn/census_tree_encoder_v1.py","tests/test_census_tree_encoder_v1.py","docs/census_tree_encoder_v1.md"]
names=["README.md","run_pytest.py","collect_tiny_fixture_numbers.py","collect_tiny_fixture_numbers_attempt_01.py","seal_author_encoder.py","tiny_fixture_numbers.json","tiny_fixture_numbers_attempt_02.json"]
names += [f"pytest_attempt_{attempt:02d}.{suffix}" for attempt in range(1,6) for suffix in ("json","txt")]
names += [f"collector_attempt_{attempt:02d}.{suffix}" for attempt in range(1,4) for suffix in ("json","txt")]
artifacts={path:sha(path) for path in core}
artifacts.update({str((dest/name).relative_to(root)):sha((dest/name).relative_to(root)) for name in names})
registry=json.loads((root/"research/experiment_decisions.json").read_text())
registration=next(item for item in registry["decisions"] if item["id"]=="H021")
assert any(item["id"]=="census_cut_exchange_proto_v1_source_only" for item in registration["registered_runs"])
receipt={"schema":"h021_census_tree_encoder_v1_author_receipt","created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"project_root":str(root),"hypothesis":"H021","condition":"census_cut_exchange_proto_v1_source_only","status":"separate_inherited_forest_GNN_source_and_handmade_CPU_qualified_pending_independent_review","registered_record_observed":registration,"artifact_sha256":artifacts,"sealed_core":{"binding_path":str(core_dir/"core_binding.json"),"binding_sha256":sha(core_dir/"core_binding.json"),"receipt_sha256":core_binding["receipt_sha256"],"all_prior_artifact_hashes_match":True},"frozen_source_integrity":frozen,"final_test":final,"tiny_fixture_numbers_path":str((dest/"tiny_fixture_numbers.json").relative_to(root)),"actions":{"handmade_CPU_unit_forwards_backwards":True,"handmade_tskit_TableCollection_fixture":True,"actual_data_or_holdout_arrays":False,"prepared_bank_arrays":False,"native_inference":False,"stochastic_genealogy_simulation":False,"model_fit":False,"GPU_or_CUDA_work":False,"profiling_or_performance_benchmark":False,"packages_installed":False,"git_used":False,"registry_modified_by_author":False,"sealed_core_or_frozen_source_or_config_modified":False,"recursive_or_storage_scan":False,"Mac_project_writes":False},"limitations":["Inherited whole-tree backbone, no new scientific novelty or predictive benefit claim.","Source neural batching is by checked forest height/depth, actual speed/utilization/autograd memory and10x fitting gate unmeasured.","No forward Python Q-by-R MRCA recomputation; donor banks must be prequalified during preparation.","Native data/scaler/feature/donor and file-array joins remain unqualified. Only handmade tensor/TS fixtures executed.","Sampled interiors explicitly rejected under preserved all-and-only-sampled-leaves representation.","Root owns fresh independent review, integration, ledger updates, fitting admission, scheduling and Git."]}
(dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
binding={"schema":"h021_census_tree_encoder_v1_binding","receipt_sha256":sha((dest/"receipt.json").relative_to(root)),"core_sha256":{path:sha(path) for path in core},"artifact_sha256":artifacts,"prior_core_binding_sha256":sha(core_dir/"core_binding.json")}
(dest/"encoder_binding.json").write_text(json.dumps(binding,indent=2)+"\n")
print(json.dumps({"receipt_sha256":binding["receipt_sha256"],"encoder_binding_sha256":sha((dest/"encoder_binding.json").relative_to(root)),"core_sha256":binding["core_sha256"],"sealed_exchange_core_unchanged":True,"frozen_source_count":len(frozen),"all_frozen_sources_match":True},indent=2))
