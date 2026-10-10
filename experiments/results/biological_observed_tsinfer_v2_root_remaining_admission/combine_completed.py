from pathlib import Path
import json,hashlib,datetime,importlib.util,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
source=ROOT/'scripts/infer_biological_paths_tsinfer_observed_v2.py'
config=ROOT/'configs/biological_paths_tsinfer_observed_v2.json'
assert sha(source)=='9904dbbed05d38d2e1c3e03c7bc07f84376df309fff8fe4852ddeec229aac7b8'
assert sha(config)=='450a502066b74231b81f364ce47f4f927c1c28bf22be20f6d7a9abf945bd0953'
spec=importlib.util.spec_from_file_location('root_combined_observed_v2',source)
wrapper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=wrapper;spec.loader.exec_module(wrapper)
base=wrapper.load_producer();c=json.loads(config.read_text());bio,lock=wrapper.validate_config(base,c)
paths={phase:ROOT/c['output_root']/phase/'manifest.json' for phase in ['canary','remaining']}
manifests={phase:json.loads(path.read_text()) for phase,path in paths.items()}
assert manifests['canary']['status']=='passed_train000_observed_canary1000'
assert manifests['remaining']['status']=='remaining_three_observed_parents_inferred1000'
admission=json.loads((ROOT/'experiments/results/biological_observed_tsinfer_v2_root_remaining_admission/admission.json').read_text())
assert wrapper.canary_gate(base,c,lock,admission)==manifests['canary']
rows=[];metrics=[]
for phase in ['canary','remaining']:
 control=ROOT/f'experiments/results/biological_observed_tsinfer_v2_root_{phase}_admission/finished.json'
 finished=json.loads(control.read_text());assert finished['exit_code']==0
 for role in ['child','supervisor']:
  proc=Path('/proc')/str(finished[role+'_pid'])/'stat'
  if proc.exists():
   start=int(proc.read_text().rsplit(')',1)[1].split()[19])
   assert start!=finished[role+'_start_ticks'], 'native process not terminal'
 m=manifests[phase];assert m['scope_parents']==c['phases'][phase]
 for row in m['replicates']:
  expected=row['parent_id']
  actual=wrapper.parent_adequacy(base,c,lock,row['frozen_helper_result'],expected_parent=expected,phase=phase)
  assert actual==row['adequacy']
  wrapper.wrapper_join(base,c,row,expected_parent=expected,phase=phase)
  v=json.loads(base.checked_ref(actual['native_verification']).read_text())
  j=json.loads(base.checked_ref(actual['stage_journal']).read_text())
  assert len(j)==11 and all(stage['status']=='complete' for stage in j)
  assert v['exact_observed_genotypes_and_ages'] and v['strict_native_times']
  assert not v['source_labels_passed_to_inference'] and not v['truth_or_oracle_tree_read']
  rows.append(row)
  metrics.append(dict(parent_id=expected,phase=phase,original_sites_preserved=v['all_original_sites_preserved'],native_connected_identity=v['native_connected_identity'],eleven_native_stage_seconds_sum=sum(stage['elapsed_seconds'] for stage in j),strict_dates_and_genotypes_preserved=True))
assert [row['parent_id'] for row in rows]==list(base.PARENTS)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
combined=dict(kind='biological_paths_tsinfer_observed_v2',schema_version=1,status='four_observed_development_parents_inferred1000',condition=c['condition'],source_helper_condition=base.CONDITION,created_utc=now,scope_parents=list(base.PARENTS),replicates=rows,implementation=ref(source),config=ref(config),producer=c['producer'],biological_config=c['biological_config'],runtime=c['runtime'],observed_asset_pins=c['observed_asset_pins'],phase_manifests={phase:ref(path) for phase,path in paths.items()},rescaling_intervals=1000,truth_or_oracle_read=False,heldout_test_or_publication_predictions=False,native_paper_reproduction_claimed=False,source_labels_passed_to_inference=False,global_internal_ids_approximate_inferred_ancestry_not_ground_truth=True)
p=ROOT/c['output_root']/'combined_observed_manifest.json';assert not p.exists()
p.write_text(json.dumps(combined,indent=2)+'\n')
out=ROOT/'experiments/results/biological_observed_tsinfer_v2_summary';out.mkdir(parents=True,exist_ok=False)
summary=dict(kind='completed_four_parent_observed_tsinfer_tsdate_v2_summary',utc=now,status='all_four_development_observation_parents_passed_native1000',combined_observed_manifest=ref(p),parents=metrics,independent_TRAIN_parents=2,independent_VAL_parents=2,TEST_parents=0,current_inference_family_task_transfer_not_original_ARGformer_paper_reproduction=True,ancestry_accuracy_measured=False,native_identity_estimation_is_approximate=True,ARGformer_prediction_comparison_still_pending=True,storage_high_water_before_combined_manifest=max(m['storage_before_final_manifest']['local_high_water_bytes'] for m in manifests.values()),storage_cap_bytes=20000000000)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'combined_manifest.json').write_bytes(p.read_bytes())
print(json.dumps({'manifest':ref(p),'summary':ref(out/'summary.json'),'tree_counts':[x['native_connected_identity']['trees'] for x in metrics],'reuse_occurrences':[x['native_connected_identity']['internal_reuse_occurrences'] for x in metrics]}))
