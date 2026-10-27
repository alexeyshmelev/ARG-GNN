"""Small synthetic metadata/math tests; no real checkpoint,targets or inference."""
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import contracts_v1 as c, predictions_v1 as prediction, features_v1 as features
from . import prior_v1 as prior, metrics_v1 as metrics, evaluation_v1 as evaluation

CHECKS=[]


def rejects(name,callback):
    try:callback()
    except (ValueError,KeyError):CHECKS.append(name);return
    raise AssertionError('Expected rejection: '+name)


def pure_math():
    fitted=prior.count_prior(np.asarray(['BAA','BAA','Neo','Neo','Neo']),np.array([5,4,0,0,1]))
    assert fitted['class_calls_0based']['BAA']==4
    assert fitted['class_calls_0based']['present_day']==0
    assert fitted['probabilities']['BAA']==[0,0,0,0,.5,.5]
    CHECKS.append('TRAIN_prior_native_order_tie_and_pooled_missing_cohort_fallback')
    rejects('prior_reject_unknown_public_cohort',lambda:prior.count_prior(np.array(['invented']),np.array([1])))
    result=metrics.classification(np.array([4,5]),np.array([0,2]),c.CLASSES)
    assert result['accuracy']==0 and result['per_class']['Ana_nonBAA']['recall'] is None
    assert result['macro_recall_all_classes'] is None and result['macro_recall_present_classes']==0
    assert np.asarray(result['confusion']).shape==(6,6)
    CHECKS.append('six_class_distinction_absent_supports_and_undefined_recalls')
    result=metrics.baa_ana_chg(np.array([4,5,4]),np.array([0,2,5]))
    assert result['conditional_support']==3 and result['errors']==2 and result['off_pair_predictions_counted_as_errors']==1
    assert result['accuracy']==1/3 and result['confusion_true_Ana_CHG_by_predicted_Ana_CHG_other']==[[1,1,0],[0,0,1]]
    CHECKS.append('BAA_native_errors_and_derived_Ana_CHG_off_pair_denominator')
    result=metrics.paired(np.array([0,1,2,3]),np.array([0,0,2,2]),np.array([1,1,2,2]))
    assert [result[key] for key in ('left_correct_only','right_correct_only','both_correct','both_wrong')]==[1,1,1,1]
    assert result['accuracy_difference_left_minus_right']==0
    CHECKS.append('paired_wins_losses_denominators')
    y=np.arange(10)%6;cohorts=np.asarray(list(c.COHORTS)*2)
    values={method:np.eye(6,dtype=np.float32)[y] for method in c.METHODS}
    report=metrics.report(y,values,cohorts)
    assert len(report['paired_native6_correctness'])==10 and set(report['native6']['cohort8_treeloom'])=={'overall',*c.COHORTS}
    assert report['independent_parent_count']==1 and report['across_parent_variance'] is None
    CHECKS.append('all_five_conditions_all_cohorts_no_row_independence_claim')


def source_and_truth():
    original=ast.parse((c.ROOT/'src/arggnn/benchmarking/european_execution.py').read_text())
    candidate=ast.parse((c.HERE/'features_v1.py').read_text())
    left=next(node for node in original.body if isinstance(node,ast.FunctionDef) and node.name=='iter_native_sampled_features')
    right=next(node for node in candidate.body if isinstance(node,ast.FunctionDef) and node.name==left.name)
    def scientific_tail(fn):
        start=next(i for i,node in enumerate(fn.body) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='queries' for target in node.targets))
        return ast.dump(ast.Module(body=fn.body[start:],type_ignores=[]),include_attributes=False)
    assert scientific_tail(left)==scientific_tail(right)
    original=ast.parse((c.ROOT/'src/arggnn/benchmarking/european_sampling.py').read_text())
    candidate=ast.parse((c.HERE/'evaluation_v1.py').read_text())
    def function(tree):return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='native_truth_labels')
    assert ast.dump(function(original),include_attributes=False)==ast.dump(function(candidate),include_attributes=False)
    CHECKS.append('native_feature_loop_and_truth_join_AST_identical_to_original_bodies')
    parent='synthetic';queries=[dict(sample_id='q'+str(i),haplotype=0,sample_index=i) for i in range(686)]
    records=[dict(example_index=0,sample_id='q0',haplotype=0,position_original_bp=0,cohort='BAA'),
        dict(example_index=1,sample_id='q1',haplotype=0,position_original_bp=9999999,cohort='Neo'),
        dict(example_index=2,sample_id='q0',haplotype=0,position_original_bp=5000000,cohort='BAA')]
    registry=dict(schema='ap_paper_sampled_registry_v1',parent_id=parent,split='validation',examples=3,query_identities=queries,records=records)
    schema=dict(parent_id=parent,split='validation',native_paths_1based=[1,2,3,4,5,6],query_sample_indices=list(range(686)),
        query_keys=[['q'+str(i),0] for i in range(686)])
    paths=np.ones((2,686),dtype=np.int64);paths[:,0]=[5,6]
    truth=dict(intervals_bp=np.array([[0,5000000],[5000000,10000000]]),native_paths_1based=paths,query_raw_node_ids=np.arange(686))
    assert evaluation.native_truth_labels(registry,schema,truth).tolist()==[4,0,5]
    rejects('original200Mb_truth_guard_still_rejects10Mb',lambda:c.sampling.native_truth_labels(registry,schema,truth))
    bad=copy.deepcopy(truth);bad['intervals_bp'][-1,1]=200000000
    rejects('isolated10Mb_truth_guard_rejects200Mb',lambda:evaluation.native_truth_labels(registry,schema,bad))
    CHECKS.append('synthetic_truth_identity_join_half_open_endpoint_boundary')


def checkpoint_metadata():
    definition=c.io._read(c.definition_ref(),observed=True)
    good=dict(schema='European_cohort8_TRAIN_completion_v1',status='completed_fixed90updates',completed_epochs=30,
        selected_epoch=30,optimizer_updates=90,checkpoint_selection='fixed_final_epoch30',TEST_truth_read=False,
        scientific_condition=c.TRAIN_CONDITION,source_manifest_ref=definition['cohort8_TRAIN_source_ref'],
        resume_driver_source_manifest_ref=definition['cohort8_resume_source_ref'],class_order=list(c.CLASSES))
    prediction.validate_fit_metadata('cohort8_treeloom',good,definition)
    for key,value in [('optimizer_updates',89),('selected_epoch',25),('source_manifest_ref',{}),('scientific_condition','old'),('TEST_truth_read',True)]:
        bad=dict(good,**{key:value})
        rejects('checkpoint_reject_'+key,lambda:prediction.validate_fit_metadata('cohort8_treeloom',bad,definition))
    rejects('cohort_checkpoint_cannot_alias_original',lambda:prediction.validate_fit_metadata('original_treeloom',good,definition))
    CHECKS.append('corrected_checkpoint_fixed_final90_source_acceptance')


def barrier_boundaries():
    source={'path':'source','sha256':'s'};registry_ref={'path':'registry','sha256':'r'};inference_ref={'path':'inference','sha256':'i'}
    definition_ref={'path':'definition','sha256':'d'};review_ref={'path':'review','sha256':'v'}
    refs={method:{'path':method,'sha256':method} for method in c.METHODS}
    products={};arrays={}
    for method in c.METHODS:
        provenance=dict(fit_completion_ref={'path':'fit','sha256':'f'},checkpoint_ref={'path':'weight','sha256':'w'},
            checkpoint_acceptance_ref={'path':'acceptance','sha256':'a'},fitted_prior_ref={'path':'prior','sha256':'p'})
        products[method]=dict(schema='European_short_VAL_prediction_v1',method=method,parent_id='fixture',split='validation',rows=10000,
            registry_ref=registry_ref,inference_ref=inference_ref,definition_ref=definition_ref,source_manifest_ref=source,
            class_order=list(c.CLASSES),VAL_truth_opened=False,TEST_accessed=False,provenance=provenance,
            predictions_ref={'path':method+'array','sha256':'a'})
        arrays[method+'array']=dict(example_indices=np.arange(10000),probabilities=np.full((10000,6),1/6,dtype=np.float64))
    barrier=dict(schema='European_short_VAL_all_predictions_barrier_v1',methods=list(c.METHODS),prediction_refs=refs,
        registry_ref=registry_ref,inference_ref=inference_ref,source_manifest_ref=source,definition_ref=definition_ref,
        source_review_ref=review_ref,all_predictions_and_TRAIN_prior_sealed_before_VAL_truth=True,VAL_truth_opened=False,TEST_accessed=False)
    reads=[]
    def read(ref,**kwargs):
        reads.append(ref['path'])
        if ref['path']=='barrier':return barrier
        if ref['path']=='definition':return {}
        if ref['path']=='fit':return {'checkpoint_ref':{'path':'weight','sha256':'w'}}
        return products[ref['path']]
    class NoTruth:
        def __iter__(self):raise AssertionError('Truth reference mapping accessed before prediction rejection')
    with patch.object(c,'registry',lambda *a:({'parent_id':'fixture'},[])),patch.object(c,'definition_ref',lambda:definition_ref),\
         patch.object(c.io,'_read',read),patch.object(c.io,'_arrays',lambda ref,**kw:arrays[ref['path']]),\
         patch.object(c,'acceptance',lambda *a:None),patch.object(prediction,'validate_fit_metadata',lambda method,value,definition:value),\
         patch.object(prior,'load_prior',lambda *a:{}),patch.object(c,'ready',lambda *a:source):
        assert len(evaluation.validated_predictions(refs,registry_ref,inference_ref,source)[2])==5
        CHECKS.append('complete_all_five_same_registry_prediction_barrier')
        del barrier['prediction_refs']['released_dense']
        rejects('missing_method_rejected_before_any_truth_mapping_access',lambda:evaluation.score(
            {'path':'barrier'},NoTruth(),'unused',source_review_ref=review_ref,guard=lambda amount:None,
            execution_check=lambda:None,write_headroom_bytes=1,max_truth_uncompressed_bytes=1))
        barrier['prediction_refs']['released_dense']={'path':'released_dense','sha256':'released_dense'}
        products['released_dense']['registry_ref']={'path':'different','sha256':'x'}
        rejects('changed_registry_rejected_before_any_truth_mapping_access',lambda:evaluation.score(
            {'path':'barrier'},NoTruth(),'unused',source_review_ref=review_ref,guard=lambda amount:None,
            execution_check=lambda:None,write_headroom_bytes=1,max_truth_uncompressed_bytes=1))
        products['released_dense']['registry_ref']=registry_ref
        arrays['released_densearray']['example_indices'][1]=0
        rejects('duplicate_canonical_rows_rejected',lambda:evaluation.validated_predictions(refs,registry_ref,inference_ref,source))
        arrays['released_densearray']['example_indices']=np.arange(10000)
        arrays['released_densearray']['probabilities'][0,0]=np.nan
        rejects('nonfinite_predictions_rejected',lambda:evaluation.validated_predictions(refs,registry_ref,inference_ref,source))
    assert all('truth' not in path.lower() for path in reads)


def main():
    for path in sorted(c.HERE.glob('*.py')):ast.parse(path.read_text(),filename=str(path))
    pure_math();source_and_truth();checkpoint_metadata();barrier_boundaries()
    result=dict(schema='European_short_VAL_evaluation_CPU_tests_v2',passed=True,checks=CHECKS,
        source_refs=[c.io._ref(path) for path in sorted(c.HERE.glob('*.py'))],definition_ref=c.definition_ref(),
        real_checkpoint_payloads_read=False,real_TRAIN_labels_read=False,VAL_truth_read=False,
        model_forward_or_training_run=False,scientific_prediction_or_scoring_run=False,
        fixtures='In-memory tiny labels and full10000-row synthetic probability/identity arrays only')
    path=c.HERE/'CONTRACTS_CPU_v2.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(passed=True,checks=len(CHECKS),result_ref=c.io._ref(path)),sort_keys=True))

if __name__=='__main__':main()
