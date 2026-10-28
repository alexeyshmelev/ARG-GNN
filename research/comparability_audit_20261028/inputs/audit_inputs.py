"""Read-only actual-artifact audit; writes only concise evidence in this folder."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import h5py

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/comparability_audit_20261028/inputs'

def ref(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

verified={}
def read(reference):
    if isinstance(reference,str):reference=ref(reference)
    assert ref(reference['path'])==reference,reference['path']
    verified[reference['path']]=reference
    return json.loads((ROOT/reference['path']).read_text())

def arrays(reference):
    assert ref(reference['path'])==reference
    verified[reference['path']]=reference
    with np.load(ROOT/reference['path'],allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}

def main():
    comparison_path='experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json'
    comparison=read(comparison_path)
    job=read('.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json')
    spec=job['call']['spec']['scientific_spec']
    cnn=read('experiments/results/paper_ancestralpaths_cnn_20261024/paper_ap_cnn__all_train__fit/native/fit.json')
    first_tree=read(comparison['predictions']['treeloom']['ancestralpaths_european_test_000'])
    completion=read(first_tree['fit_completion_ref']);barrier=read(completion['observation_barrier_ref'])
    plan=read(cnn['plan_ref'])
    cnn_fit_job=read('.runtime/benchmarking/paper_ap_cnn_20261024/jobs/paper_ap_cnn__all_train__fit.json')
    cnn_feature_job=read('.runtime/benchmarking/workflow_20261019_v8/jobs/AP__ancestralpaths_european_train_000__ap_features.json')
    assert ref(cnn['checkpoint_ref']['path'])==cnn['checkpoint_ref']
    with h5py.File(ROOT/cnn['checkpoint_ref']['path'],'r') as file:
        cnn_saved_optimizer_steps=int(file['optimizer_weights/iter:0'][()])
    assert cnn_saved_optimizer_steps==cnn['optimizer_updates']==8335
    assert cnn['plan_ref']==completion['sampling_plan_ref']==spec['sampling_plan_ref']
    cnn_ids=arrays(cnn['training_example_ids_ref'])['example_ids'].tolist()
    expected_ids=[];train_rows=[];labels_total=np.zeros(6,dtype=np.int64);labels_by_cohort=collections.defaultdict(lambda:np.zeros(6,dtype=np.int64))
    all_panels={};feature_counts=np.zeros(8,dtype=np.int64)
    for parent in cnn['training_parent_ids']:
        tree=spec['train_inputs'][parent];product=cnn['training_products'][parent]
        features=read(product['features_ref']);target=read(product['targets_ref']);registry=read(tree['sampled_registry_ref'])
        assert tree['sampled_registry_ref']==features['registry_ref']==target['registry_ref']==barrier['parents'][parent]['sampled_registry_ref']
        assert tree['training_targets_ref']==product['targets_ref']
        assert tree['inference_ref']==features['inference_ref']==barrier['parents'][parent]['inference_ref']
        feature_arrays=arrays(features['arrays_ref']);target_arrays=arrays(target['arrays_ref'])
        n=registry['examples'];indices=feature_arrays['example_indices'];labels=target_arrays['native_labels_0based']
        assert np.array_equal(indices,np.arange(n)) and len(labels)==n
        expected_ids.extend(f'{parent}:{int(i)}' for i in indices)
        for record in registry['records']:labels_by_cohort[record['cohort']][labels[record['example_index']]]+=1
        counts=np.bincount(labels,minlength=6);labels_total+=counts
        x=feature_arrays['features'];valid=np.any(x!=-15,axis=-1)
        nonzero=((x[:,:,:8]>0)&valid[:,:,None]).sum(axis=(0,1));feature_counts+=nonzero
        inference=read(tree['inference_ref']);panel=read(inference['observed_sample_manifest']);all_panels[parent]=panel
        train_rows.append(dict(parent_id=parent,rows=n,cohort_quotas=registry['cohort_quotas'],native_label_counts=counts.tolist(),
            sampled_registry_ref=tree['sampled_registry_ref'],training_targets_ref=tree['training_targets_ref'],inference_ref=tree['inference_ref'],
            CNN_and_TreeLoom_exact_registry_and_target_refs_equal=True,
            CNN_example_indices_match_every_registry_draw=True,CNN_nonzero_feature_entries_per_cohort_channel=nonzero.tolist()))
    assert cnn_ids==expected_ids and len(cnn_ids)==50000
    tests=[]
    test_parent_ids=sorted(comparison['predictions']['treeloom'])
    assert len(test_parent_ids)==comparison['test_parents']==5
    assert set(test_parent_ids)==set(comparison['predictions']['ancestralpaths_historical_cnn_native6'])
    for parent in test_parent_ids:
        tree=read(comparison['predictions']['treeloom'][parent]);ap=read(comparison['predictions']['ancestralpaths_historical_cnn_native6'][parent])
        features=read(ap['features_ref']);registry=read(tree['sampled_registry_ref'])
        assert tree['fit_completion_ref']==first_tree['fit_completion_ref'] and tree['checkpoint_ref']==first_tree['checkpoint_ref']
        assert ap['fit_ref']==ref('experiments/results/paper_ancestralpaths_cnn_20261024/paper_ap_cnn__all_train__fit/native/fit.json')
        assert tree['query_ancestry_truth_opened'] is False and ap['query_truth_read'] is False
        for key in ('prediction_wrapper_source_ref','recurrent_execution_source_ref','static_executor_source_ref'):
            assert ref(tree[key]['path'])==tree[key]
            verified[tree[key]['path']]=tree[key]
        assert tree['sampled_registry_ref']==ap['registry_ref']==features['registry_ref']
        assert tree['inference_ref']==features['inference_ref'] and tree['native_path_names']==ap['native_path_names']
        inference=read(tree['inference_ref']);all_panels[parent]=read(inference['observed_sample_manifest'])
        tests.append(dict(parent_id=parent,rows=registry['examples'],cohort_quotas=registry['cohort_quotas'],
            exact_registry_equal=True,exact_native_inference_equal=True,class_order_equal=True,
            registry_ref=tree['sampled_registry_ref'],inference_ref=tree['inference_ref']))
    reference=next(iter(all_panels.values()))
    panel_summary=[]
    for cohort in ('present_day','Bronze_Age','BAA','Yam','Neo','WHG','EHG','Ana','CHG'):
        rows=[r for r in reference if r['observed_cohort']==cohort]
        panel_summary.append(dict(cohort=cohort,haplotypes=len(rows),role=rows[0]['role'],
            observed_reference_class=rows[0]['observed_reference_class'],
            min_age_generations=min(r['age_generations'] for r in rows),max_age_generations=max(r['age_generations'] for r in rows)))
    compact=lambda rows:[(r['sample_index'],r['sample_id'],r['haplotype'],r['observed_cohort'],r['role'],r['observed_reference_class'],r['age_generations']) for r in rows]
    assert all(compact(panel)==compact(reference) for panel in all_panels.values())
    donor_counts=[sum(r['observed_reference_class']==cls and (r['age_generations']<350)==young for r in reference)
                  for young in (True,False) for cls in range(4)]
    checkpoint=ROOT/first_tree['checkpoint_ref']['path']
    assert ref(checkpoint)==first_tree['checkpoint_ref']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    weight=saved['model_state']['leaf_projection.0.weight']
    progress=read(str(checkpoint.parent.parent.relative_to(ROOT)/'progress.json'))['events']
    assert len(progress)==90 and sum(e['sampled_records'] for e in progress)==1500000
    step_counts=sorted({int(v['step']) for v in saved['optimizer_state']['state'].values()})
    source_paths=[
        'research/repairs/benchmark_full_vram_20261022/training/ap_fit_v2.py',
        'research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_stage.py',
        'research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_observations.py',
        'research/repairs/benchmark_multigpu_20261022/scientific/queue_adapters/european_execution.py',
        'src/arggnn/benchmarking/european_treeloom.py',
        'src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py',
        'src/arggnn/ancestralpaths_native_bridge.py',
        'src/arggnn/benchmarking/european_execution.py',
        'research/reviews/H060_isolated_hold_compatibility_source_20261017_v1/observations_v1.py',
        'research/reviews/author_repository_deep_audit_20261023_v1/ancestralpaths/history_source/previous_training.py']
    # Actual fit job carries hash bindings; report whether inspected fit sources
    # are covered by that executed job, rather than assuming current files match.
    def references(value):
        out=[]
        if isinstance(value,dict):
            if set(('path','sha256')).issubset(value):out.append(dict(path=value['path'],sha256=value['sha256']))
            for child in value.values():out.extend(references(child))
        elif isinstance(value,list):
            for child in value:out.extend(references(child))
        return out
    pinned={r['path']:r for r in references(job)}
    feature_pins={r['path']:r for r in references(cnn_feature_job)}
    cnn_fit_pins={r['path']:r for r in references(cnn_fit_job)}
    for source_pin in cnn_fit_pins.values():
        assert ref(source_pin['path'])==source_pin
    assert all(ref(source_pin['path'])==source_pin for source_pin in feature_pins.values() if source_pin['path'].endswith('.py'))
    source_evidence=[]
    for name in source_paths:
        actual=ref(name);pin=pinned.get(name)
        if pin is not None:assert actual==pin
        feature_pin=feature_pins.get(name)
        if feature_pin is not None:assert actual==feature_pin
        source_evidence.append(dict(**actual,pinned_in_actual_TreeLoom_fit_job=pin is not None,
            pinned_in_original_CNN_train000_feature_job=feature_pin is not None))
    report=dict(schema='European_actual_input_comparability_audit_v1',comparison_ref=ref(comparison_path),
        actual_treeloom_job_ref=ref('.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json'),
        actual_CNN_fit_job_ref=ref('.runtime/benchmarking/paper_ap_cnn_20261024/jobs/paper_ap_cnn__all_train__fit.json'),
        original_CNN_feature_job_ref=ref('.runtime/benchmarking/workflow_20261019_v8/jobs/AP__ancestralpaths_european_train_000__ap_features.json'),
        all_five_TEST_predictions_use_same_fit_and_checkpoint=True,
        same_training_parent_ids=cnn['training_parent_ids'],same_training_rows=50000,
        same_training_example_id_sequence=True,same_training_targets=True,train_parents=train_rows,test_parents=tests,
        native_class_order=saved['class_order'],training_class_counts=labels_total.tolist(),
        training_labels_by_cohort={k:v.tolist() for k,v in labels_by_cohort.items()},
        all_eight_parent_observed_panel_memberships_and_dates_equal=True,panel=panel_summary,
        TreeLoom=dict(pure_reference_haplotypes=266,unknown_query_haplotypes=686,explicit_cohort_channels=0,
            leaf_metadata_channels=['Ana_pure','CHG_pure','WHG_pure','EHG_pure','unknown','log1p_sample_age/log10001'],
            donor_group_order=['Ana_age_lt350','CHG_age_lt350','WHG_age_lt350','EHG_age_lt350','Ana_age_ge350','CHG_age_ge350','WHG_age_ge350','EHG_age_ge350'],
            donor_group_counts=donor_counts,epochs=30,optimizer_updates=90,label_exposures=1500000,
            actual_optimizer_step_counts=step_counts,checkpoint_ref=first_tree['checkpoint_ref'],checkpoint_selection='fixed_final_epoch30',
            metadata_weight_column_norms=weight[:,-6:].norm(dim=0).tolist(),age_input_weight_nonzero=bool(weight[:,-1].count_nonzero()),
            learned_causal_effect_of_age_not_measured=True),
        CNN=dict(ancient_cohort_haplotypes=770,present_day_excluded_haplotypes=182,
            features=['Bronze_Age','BAA','Yam','Neo','WHG','EHG','Ana','CHG','ancestor_age_div1500'],
            feature_shape=[5,9],ancient_focal_self_included=True,
            actual_training_nonzero_feature_entries_per_cohort_channel=feature_counts.tolist(),
            epochs=5,optimizer_updates=8335,checkpoint_optimizer_iterations=cnn_saved_optimizer_steps,batch_size=30,label_exposures=250000,
            checkpoint_ref=cnn['checkpoint_ref'],checkpoint_selection='final_epoch5'),
        accuracy={method:{cohort:{endpoint:metric[endpoint]['accuracy']['mean'] for endpoint in ('native6','derived_C4')}
            for cohort,metric in comparison['summary'][method].items()} for method in ('treeloom','ancestralpaths_historical_cnn_native6')},
        source_evidence=source_evidence,verified_metadata_and_small_array_refs=list(verified.values()),
        no_training_or_predictions_performed=True,no_scientific_data_mutated=True)
    destination=HERE/'inputs_audit.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report=ref(destination),same_rows=50000,same_example_ids=True,same_targets=True,
        train_class_counts=labels_total.tolist(),donor_group_counts=donor_counts,
        actual_optimization_steps=step_counts,source_evidence=source_evidence)))

if __name__=='__main__':main()
