"""Seal the bounded independent gate audit only after the final frozen retest."""
from pathlib import Path
import csv,datetime,hashlib,json,re
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/development_gate_audit_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def check_map(base,mapping):
    changed=[n for n,h in mapping.items() if not (base/n).is_file() or sha(base/n)!=h]
    assert not changed,changed
    return {'checked':len(mapping),'changed':[]}
start=read('secondary_timing_repair_start_hashes.json')
sources=start['source_sha256']
check_map(ROOT,sources)
auxiliary=read('auxiliary_fixture_hashes.json')['source_sha256'];check_map(ROOT,auxiliary)
write('final_scope_hashes.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'contract_source_sha256':sources,'auxiliary_fixture_sha256':auxiliary,
    'executable_sources_unchanged_during_final_independent_retest':True})
preserved={}
for inv,snap in [('combined_start_hashes.json','combined_start_snapshot'),('combined_repair_end_hashes.json','combined_repaired_snapshot'),('secondary_start_hashes.json','secondary_frozen_snapshot'),('secondary_repair_start_hashes.json','secondary_repaired_snapshot'),('secondary_timing_repair_start_hashes.json','secondary_timing_repaired_snapshot'),('auxiliary_fixture_hashes.json','auxiliary_fixture_snapshot')]:
    preserved[snap]=check_map(OUT/snap,read(inv)['source_sha256'])
old=ROOT/'research/reviews/development_producer_audit_v1';older=json.loads((old/'receipt.json').read_text())
oldreceipt=sha(old/'receipt.json');assert oldreceipt=='64d792fde7bdb0ae696c46789229ba114dd79a83b662663c8a90c683e97a912e'
preserved['older_audit_artifacts']=check_map(old,older['review_artifact_sha256'])
preserved['older_synthetic_repro']=check_map(ROOT,json.loads((old/'synthetic_repro_artifact_hashes.json').read_text())['artifact_sha256'])
preserved['older_receipt_sha256']=oldreceipt
write('preserved_evidence_final_verification.json',preserved)
phases=[]
for name,label in [('pytest_combined.log','original_combined'),('pytest_combined_post.log','producer_settings_repaired_combined'),('pytest_secondary.log','original_secondary'),('pytest_secondary_post.log','atomic_publication_repaired_secondary'),('pytest_secondary_timing_post.log','runtime_timing_repaired_secondary')]:
    text=(OUT/name).read_text();match=re.search(r'(\d+) passed in ([\d.]+)s',text)
    assert match is not None,name
    phases.append({'phase':label,'passed':int(match[1]),'seconds':float(match[2]),'log':name})
runtime=read('secondary_runtime_replay_post_timing_repair.json')
for key in ['validation_policy','validation_settings','reload_policy','reload_settings','replay_validation_policy','replay_validation_settings']:
    row=runtime['checks'][key]
    assert row['outcome']['rejected'] and row['success_returned'] is False
    assert row['artifact_writes_after_mutation']==0
    assert row['prediction_exists_after_failed_fit'] is False
    if key.startswith('validation'):assert row['checkpoint_exists_after_failed_fit'] is False
    assert row['primary_candidate_unchanged'] and row['primary_baseline_unchanged']
pub=read('secondary_partial_write_post_repair_fixed.json')
assert len(pub['checks'])==8
assert all(row['pending_files']==0 for row in pub['checks'].values())
combined=read('combined_replay_post_repair.json')
assert combined['checks']['pure_ten_producer_assembly_no_raw_reads']=={'raw_artifacts':240,'source_snapshot_artifacts':241,'validated':10}
for key in ['candidate_wrong_class_count','candidate_wrong_width','gru_wrong_temporal_depth','candidate_wrong_heldout_cache_path']:
    assert combined['checks'][key]['combined_gate']['rejected']
review=(OUT/'review_draft.md').read_text()
review=review.replace('Status: final runner timing repair and independent final retest are pending. This draft is not a final certification or cohort admission.',
    'Status: the bounded synthetic CPU audit passed after owner repairs. This certifies the reviewed contracts and rejection behavior within the explicit mock limits below; it does not establish genuine cohort completion or phase promotion.')
review=review.replace('Its earlier-check repair is pending the final freeze.',
    'The repaired runner now checks admission and active numerical policy immediately after epoch validation and checkpoint-replay validation, before subsequent checkpoint/prediction writes. All six independent active drift probes now reject with zero artifact writes after mutation.')
final=phases[-1]
review=review.replace('| Final runtime-timing repaired suite | pending | pending |',
    f"| Final runtime-timing repaired suite | {final['passed']} | {final['seconds']} |")
review=review.replace('Failed earlier fixtures/artifacts are preserved; final source and artifact inventories will be sealed after the last repair is frozen and independently retested.',
    'Failed earlier fixtures/artifacts are preserved. Final source/artifact inventories are sealed after the last repair freeze and independent retest.')
review+='\n## Final timing repair and provenance\n\n'
review+=('The post-repair replay preserves the 48-epoch production cap and uses the original four active settings/numerical-policy mutations plus two direct checkpoint-replay validation mutations on real tiny matched models. Validation mutations reject before the first checkpoint; reload mutations retain only checkpoints created while admission and policy were still valid. No prediction is emitted and no save is attempted after mutation in all six cases. Earlier failed checkpoint/prediction artifacts remain unchanged as historical evidence. The synthetic two-LR complete/partial replay still produces 12/partial cells without primary reselection or final promotion.\n\n')
review+=(f"The final contract source inventory contains {len(sources)} files plus {len(auxiliary)} auxiliary fixture/configuration pins captured in a later explicit phase. The repaired combined snapshot incorporates the documented end-phase documentation-only amendment; its executable bytes were stable during the 129-test retest. All preserved snapshots match their applicable inventories. The older producer audit remains unchanged: 84 audit artifacts and 2531 ignored synthetic repro files match its sealed receipt {oldreceipt}. receipt.json binds final source, review, test logs, reproduction scripts, snapshots and ignored synthetic artifact inventories. Live seal/verification logs are excluded to avoid self-changing hashes.\n")
(OUT/'review.md').write_text(review)
rows=[
 {'id':'DG001','severity':'high','status':'resolved','component':'combined producer settings admission','finding':'Gate accepted four settings/path mutations rejected by actual producer guards','resolution':'Exact typed ten-family producer assembly and mandatory canonical artifact/source snapshot validation','evidence':'combined_replay.json; combined_replay_post_repair.json; pytest_combined_post.log','limit':'Positive completion admission mocked; original raw catalog was incomplete and preserved honestly'},
 {'id':'DG002','severity':'high','status':'resolved','component':'secondary public publication','finding':'Controlled write failure left 17 bytes of invalid output; synthesis shared unguarded publication','resolution':'Shared staged owned-inode fsync admission recheck and exclusive hard-link publication','evidence':'secondary_partial_write_repro.json; secondary_partial_write_post_repair_fixed.json; pytest_secondary_post.log','limit':'Eight isolated public emission probes use non-executable positive admission stubs'},
 {'id':'DG003','severity':'medium','status':'resolved','component':'secondary runtime guard timing','finding':'Policy/settings mutations were rejected before success but checkpoint/prediction artifacts could already be written','resolution':'Immediate live admission and numerical-policy guards after epoch validation and checkpoint-replay validation before writes','evidence':'secondary_runtime_replay.json; secondary_runtime_replay_post_timing_repair.json; pytest_secondary_timing_post.log','limit':'Real tiny CPU models under explicit mocked positive cohort, producer and reviewed dependency contracts; prior valid checkpoints remain after reload faults'},
 {'id':'LIMIT01','severity':'scope','status':'limitation','component':'real completion and production selection','finding':'No genuine cohort or production three-seed matrix was audited','resolution':'No phase promotion claimed; synthetic mocks and serialized seed clones identified explicitly','evidence':'review.md; secondary_runtime_replay_post_timing_repair.json','limit':'No GPUs, simulations, actual held-out data, real winner, performance or confirmatory authorization'}]
with (OUT/'concerns.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
repros={}
for file in sorted((ROOT/'.runtime/tmp/development_gate_audit_v1').rglob('*')):
    if file.is_file() and not file.is_symlink():repros[str(file.relative_to(ROOT))]=sha(file)
write('synthetic_repro_artifact_hashes.json',{'no_actual_cohort_data':True,'artifact_sha256':repros})
artifacts={}
excluded={'receipt.json','seal_report.log','receipt_verification.log'}
for file in sorted(OUT.rglob('*')):
    if file.is_file() and not file.is_symlink() and str(file.relative_to(OUT)) not in excluded:
        artifacts[str(file.relative_to(OUT))]=sha(file)
receipt={'kind':'independent_development_gate_audit_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'bounded_pass_after_owner_repairs','resolved_finding_ids':['DG001','DG002','DG003'],
    'actual_fresh_cohort_admitted':False,'GPU_used':False,'simulations_run':False,'actual_final_test_access':False,
    'active_executable_config_emitted':False,
    'synthetic_executable_shaped_input_config_under_mock_admission':True,
    'promotes_final_test_phase':False,'authorizes_confirmatory_generation':False,'primary_selection_changed':False,
    'independent_test_phases':phases,'phase_counts_not_summed':True,
    'combined_positive_completion_mocked':True,'secondary_positive_completion_producer_and_reviewed_dependency_mocked':True,
    'secondary_public_emission_probes':8,'secondary_active_runtime_drift_probes':6,
    'secondary_artifact_writes_after_mutation':0,
    'secondary_synthesis_actual_toy_template_fits':4,'secondary_synthesis_seed_clone_cells':12,
    'secondary_synthesis_production_independent_seed_fit_claimed':False,
    'source_sha256':sources,'auxiliary_fixture_sha256':auxiliary,
    'source_inventory_sha256':sha(OUT/'final_scope_hashes.json'),
    'review_artifact_sha256':artifacts,'synthetic_repro_files':len(repros),
    'synthetic_repro_inventory_sha256':sha(OUT/'synthetic_repro_artifact_hashes.json'),
    'older_sealed_producer_receipt_sha256':oldreceipt,'older_sealed_producer_evidence_unchanged':True,
    'limitations':['Synthetic completion positives only; no genuine cohort admission',
        'Original combined fixture raw catalog was incomplete and retained',
        'Secondary main matrix handmade; four toy templates cloned across three seed metadata rows',
        'No GPU/performance/convergence/real winner/final-test/confirmatory certification']}
write('receipt.json',receipt)
check_map(ROOT,sources);check_map(ROOT,auxiliary);check_map(OUT,artifacts);check_map(ROOT,repros)
print(json.dumps({'receipt_sha256':sha(OUT/'receipt.json'),'source_files':len(sources),
    'auxiliary_pins':len(auxiliary),'review_artifacts':len(artifacts),'synthetic_repro_files':len(repros),
    'independent_final_phase':final,'all_receipt_hashes_verified':True},sort_keys=True))

