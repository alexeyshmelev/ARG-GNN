"""Final source-only H021 observation-adapter author seal; no array loading."""
from pathlib import Path
import datetime,hashlib,json
root=Path(__file__).resolve().parents[3];assert Path.cwd()==root
dest=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256((root/path).read_bytes()).hexdigest()
config=json.loads((root/'configs/census_biological_bank_v1.json').read_text())
assert config['fixture_mode'] is False
source_checks={path:{'expected_sha256':expected,'actual_sha256':sha(path),'matches':sha(path)==expected} for path,expected in config['source_sha256'].items()}
assert all(item['matches'] for item in source_checks.values())
assert source_checks['research/reviews/census_cut_encoder_v1_independent/receipt.json']['actual_sha256']=='73827b371263cdf648442a6f474161bb3c5122df31ad9a639a201d1cf1c6fe57'
pair_bindings={}
for directory,name,expected in [('census_cut_exchange_v1_author','core_binding.json','94bb947cadce54b559be5182405e2e5953fdd4eaf005d276b369bbf48a890021'),('census_tree_encoder_v1_author','encoder_binding.json','7aeb53323d98aa5309fd36a1d92f82ea2f3414bd66e414edbc46681f3b6b4afa')]:
    path=Path('research/reviews')/directory/name;assert sha(path)==expected
    binding=json.loads((root/path).read_text());assert all(sha(p)==value for p,value in binding['artifact_sha256'].items())
    assert sha(path.parent/'receipt.json')==binding['receipt_sha256']
    pair_bindings[str(path)]={'sha256':expected,'all_prior_artifacts_match':True}
independent=json.loads((root/'research/reviews/census_cut_encoder_v1_independent/receipt.json').read_text())
frozen={path:{'expected_sha256':item['expected_sha256'],'actual_sha256':sha(path),'matches':sha(path)==item['expected_sha256']} for path,item in independent['frozen_source_integrity'].items()}
assert all(item['matches'] for item in frozen.values())
final=json.loads((dest/'pytest_attempt_04.json').read_text())
assert final['exit_code']==0 and '30 passed in 6.23s' in (dest/'pytest_attempt_04.txt').read_text()
assert all(sha(path)==value for path,value in final['source_sha256'].items())
metrics=json.loads((dest/'handmade_join_metrics.json').read_text());assert all(sha(path)==value for path,value in metrics['source_sha256'].items())
for attempt in range(1,5):
    record=json.loads((dest/f'pytest_attempt_{attempt:02d}.json').read_text());assert sha((dest/f'pytest_attempt_{attempt:02d}.txt').relative_to(root))==record['output_sha256']
collector=json.loads((dest/'collector_attempt_01.json').read_text());assert collector['exit_code']==0
assert sha((dest/'collector_attempt_01.txt').relative_to(root))==collector['output_sha256']
assert sha((dest/'collect_handmade_join_metrics.py').relative_to(root))==collector['script_sha256']
core=['src/arggnn/census_biological_bank_v1.py','tests/test_census_biological_bank_v1.py','configs/census_biological_bank_v1.json','docs/census_biological_bank_v1.md']
names=['README.md','run_pytest.py','collect_handmade_join_metrics.py','handmade_join_metrics.json','collector_attempt_01.json','collector_attempt_01.txt','seal_author.py']
names += [f'pytest_attempt_{attempt:02d}.{suffix}' for attempt in range(1,5) for suffix in ('json','txt')]
artifacts={path:sha(path) for path in core};artifacts.update({str((dest/name).relative_to(root)):sha((dest/name).relative_to(root)) for name in names})
registry=json.loads((root/'research/experiment_decisions.json').read_text());registration=next(row for row in registry['decisions'] if row['id']=='H021')
receipt={'schema':'census_biological_bank_v1_author_receipt','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'project_root':str(root),'hypothesis':'H021','status':'source_and_handmade_TS_CPU_qualified_pending_independent_review_and_root_actual_join_admission','registered_H021_record_observed':registration,'artifact_sha256':artifacts,'source_protocol_and_dependency_integrity':source_checks,'reviewed_pair_bindings_unchanged':pair_bindings,'frozen_source_integrity':frozen,'final_test':final,'handmade_metrics':'research/reviews/census_biological_bank_v1_author/handmade_join_metrics.json','actions':{'handmade_tskit_TS_and_pinned_producer_replay':True,'exact_handmade_Torch_serialization_roundtrip':True,'same_weight_handmade_GNN_forwards_VJPs':True,'synthetic_clipped_AdamW_unit_steps':5,'actual_project_holdout_or_prepared_bank_arrays':False,'actual32_tile_support_or_TRAIN_minmax_numeric_execution':False,'actual_source_file_byte_reader_admission':False,'native_rerun_or_inference':False,'stochastic_genealogy_simulation':False,'model_fitting_loop':False,'GPU_or_CUDA_work':False,'profile_or_performance_benchmark':False,'install':False,'git':False,'shared_ledger_or_decisions_edits':False,'reviewed_modules_or_sealed_bundles_modified':False,'storage_scan':False,'Mac_project_writes':False},'limitations':['Supplied observed source paths/hashes must be actually verified by the root reader before objects are passed. No actual file/source-to-roster byte admission was executed.','No actual32-tile support or TRAIN min/max/all-tile numerical qualification, native fitting admission, reviewer verdict or scientific/efficiency benefit.','Full CPU certificate hashing before wrapper forward and level-state/autograd memory remain unmeasured.','Physical anchor collisions with unequal frozen normalized arithmetic reject without epsilon; actual tiles require root qualification.','F47 reuses original float32 age/reference summary semantics and deletes only original branch7. Direct native float64 controls date/cut/donor construction.','Anonymous unary suppression is qualified only after exact pinned transformation parity; sampled interiors or independent unary observations need a new representation.','Root owns admission, runner, actual storage/runtime/data access, independent review, integration, fitting, ledger and Git.']}
(dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
binding={'schema':'census_biological_bank_v1_author_binding','receipt_sha256':sha((dest/'receipt.json').relative_to(root)),'core_sha256':{path:sha(path) for path in core},'artifact_sha256':artifacts,'reviewed_pair_bindings_unchanged':pair_bindings}
(dest/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps({'receipt_sha256':binding['receipt_sha256'],'binding_sha256':sha((dest/'binding.json').relative_to(root)),'core_sha256':binding['core_sha256'],'all_reviewed_pair_and_source_pins_match':True,'frozen_source_count':len(frozen)},indent=2))
