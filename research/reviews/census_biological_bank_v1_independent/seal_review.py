from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path('/workspace-SR003.nfs2/ARG-GNN');owned=root/'research/reviews/census_biological_bank_v1_independent'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={
'src/arggnn/census_biological_bank_v1.py':'1e89e93a23827e15439eceaabea505c6fae088fc407b62e1fd06c71dd26460c6',
'tests/test_census_biological_bank_v1.py':'402822d492c27cede525608f10787be0ac211f71ffd49d26d39cd3f7d02fd3fb',
'configs/census_biological_bank_v1.json':'048655813c60285e7e0934c76bc5ae986902455a998c3abb42b21f4e4feb88c9',
'docs/census_biological_bank_v1.md':'dffb20712530988c09e0d50f76958e356f411860f627b96b9055da39adf5c4c4',
'research/reviews/census_biological_bank_v1_author/receipt.json':'ccc759654b3e098311170f40190dcf7bef83c9a615a56daed69355fd4681f3f1'}
config=json.loads((root/'configs/census_biological_bank_v1.json').read_text());expected.update(config['source_sha256'])
integrity={p:{'expected_sha256':h,'actual_sha256':sha(root/p),'matches':sha(root/p)==h} for p,h in expected.items()}
assert all(r['matches'] for r in integrity.values())
attempts=[json.loads(p.read_text()) for p in sorted(owned.glob('pytest_attempt_*.json'))];last=attempts[-1]
assert last['exit_code']==0 and all(r['all_pinned_source_unchanged'] for r in attempts)
metrics=json.loads((owned/'handmade_metrics.json').read_text())
artifacts={str(p.relative_to(root)):sha(p) for p in sorted(owned.iterdir()) if p.is_file() and p.name!='receipt.json'}
record={
'schema':'census_biological_bank_v1_independent_receipt',
'created_at_utc':datetime.now(timezone.utc).isoformat(),
'project_root':str(root),'hypothesis_id':'H021',
'status':'qualified_pinned_source_and_handmade_CPU_with_signed_zero_caveat_actual_admission_blocked',
'verdict_scope':'No blocking source defect found for value-exact observed producer parity, direct native geometry and an independently frozen certificate. Actual source-reader/data/structural/numerical/fitting admission remains blocked and unrun.',
'independence':{'fresh_context':True,'development_history_received':False,'desired_verdict_supplied':False,'own_fixture_recipe':True,'author_handmade_suite_rerun':True,'reviewed_source_modified':False},
'source_protocol_and_dependency_integrity':integrity,
'environment_source_sha256':{p:sha(root/p) for p in ['AGENTS.md','scripts/project_env.sh']},
'artifact_sha256':artifacts,
'final_test':dict(last,total_passed=64,author_passed=30,independent_passed=34,pytest_reported_seconds=8.52),
'attempt_dispositions':[
{'attempt':'write01','exit_code':127,'reason':'Reviewer SSH/heredoc quoting failure before Python launch. Preserved malformed owned texts, then used literal bash heredoc.'},
{'attempt':'pytest01','passed':61,'failed':1,'reason':'Reviewer unary comparison incorrectly assumed equal retained row order. Corrected explicit known cross-fixture native ID correspondence; no source edit.'},
{'attempt':'pytest02','passed':62,'failed':0,'reason':'Corrected unary alignment passes.'},
{'attempt':'pytest03','passed':62,'failed':1,'reason':'New reviewer node-table reversal fixture supplied int64 edge indexes to tskit requiring int32. Corrected only handmade fixture index dtype.'},
{'attempt':'pytest04','passed':63,'failed':0,'reason':'Corrected native ID reversal passes all controls.'},
{'attempt':'pytest05','passed':64,'failed':0,'reason':'Added unresolved independent native Jaccard/right-hold-span oracle; final source sealed.'}],
'independent_evidence':{
'handmade_native_shape':{k:metrics[k] for k in ['native_ts_nodes','native_ts_samples','native_ts_trees','stored_x48_shape','connected_node_counts','retained_node_counts','excluded_nonancestor_rows_by_tree']},
'direct_tskit_MRCA_donor_max_absolute_errors':metrics['donor_oracle_max_abs_errors'],
'no_readout_or_anchor_hold_kept':True,
'exact_boundary_readout_in_right_hold':True,
'all_native_postorder_rows_ancestor_union_samples_edges_F47_endpoints_grid_checked':True,
'independent_x48_reference_summary_semantics_enumerated':True,
'anonymous_unary_native_endpoint_additivity_after_value_parity':True,
'injected_unary_feature_rejected':True,
'native_350_vs_350_0000001_identical_x48_different_cut_and_certificate':True,
'actual_ID_relabel_all_five_controls_exact_logits':True,
'native_node_table_reversal_all_five_controls_logits_rtol':1e-10,
'native_node_table_reversal_all_five_controls_logits_atol':1e-12,
'forest_vs_separate_output_and_parameter_VJP_qualified':True,
'full_GNN_central_directional_derivative_qualified':True,
'all_five_control_gradient_checks':metrics['all_five_control_gradient_checks'],
'all_GraphSequence_fields_mutation_rejected':True,
'missing_wrong_or_stale_expected_digest_rejected_before_sentinel_model_call':True,
'physical_equal_anchor_readout_unequal_frozen_IEEE_rejected_without_epsilon':True,
'native_cut_group_and_Jaccard_rho_oracles':True,
'physical_span_group_size_and_right_hold_rho_oracles':True,
'handmade_all32_TRAIN_min_max_tie_order_and_no_outcome_checks':True},
'nonblocking_source_caveat':{'id':'C1','line':205,'claim':'Producer replay parity is numerical value equality with dtype/shape/stride, not byte equality. +0/-0 byte changes are admitted, then receive distinct certificate hashes. No nonzero feature drift acceptance observed.','resolution':'Document value-exact replay or strengthen the bytes check before byte-identical replay claims.'},
'content_and_source_trust_boundaries':[
'Source bindings have role/path/hash syntax validation only. Actual byte verification and decoded-object observed lineage are external reader obligations.',
'Roster role/class/date-to-observed-source lineage, existing scaler semantics and target-column ordering require root actual reader qualification. Handmade role/type/date counts pass only their deterministic contract.',
'A new certificate hash supplied after mutated content may validate. Only an independently frozen expected digest protects the original admitted content. Resealing against the original expected digest rejects.',
'Current adapter/config/reader execution source must be independently pinned by the root job. Adapter source is explicitly pinned in this receipt; config dependency pins do not self-pin the adapter.',
'F47 retains original float32 encoded-age and reference TMRCA summaries. Deleting original branch7 does not establish an age-free feature condition.',
'Anonymous unary suppression is qualified only after the deterministic pinned observation transformation passes value parity.',
'IDs are provenance/gather keys. Native and actual ID relabel tests qualify nonembedding behavior for the inspected shared model.'
],
'blocked_unrun_admission_gates':[
'Actual observed source-file-byte hashes and source-to-roster/grid lineage reader.',
'Actual role reader, scaler/feature provenance, original sample/readout/target-column ordering.',
'Actual32 tile structural/collision qualification and support opportunity survey.',
'Prespecified TRAIN minimum/maximum numeric qualification then actual all-tile bank/control/VJP/clipped-AdamW pass.',
'Frozen actual fit protocol, fitting, storage/runtime/GPU admission and predictive/speed/memory claims.'
],
'actions':{'source_read':True,'handmade_tskit_Tables_only':True,'handmade_pinned_producer_replay':True,'handmade_model_forward_and_VJP':True,'author_synthetic_clipped_AdamW_unit_steps_rerun':5,'actual_observation_model_or_target_array_access':False,'native_inference_or_rerun':False,'stochastic_simulation':False,'CUDA_GPU_jobs':False,'installation':False,'Git':False,'shared_ledger_or_decisions_edit':False,'source_or_author_artifact_edit':False,'bulk_storage_scan':False,'Mac_project_file_writes':False},
'owned_directory_only':str(owned)}
(owned/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'receipt':str(owned/'receipt.json'),'sha256':sha(owned/'receipt.json'),'status':record['status'],'passed':64,'all_pinned_source_matches':True},indent=2))
