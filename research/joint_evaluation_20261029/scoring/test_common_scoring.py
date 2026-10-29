"""Synthetic CPU checks; no real TEST prediction, target-schema or truth reads."""
import json
from pathlib import Path
import tempfile
import numpy as np
import common_scoring as scorer


def rejects(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('Malformed input was accepted')


def main():
    checks=[]
    p=np.eye(6,dtype=np.float32);y=np.arange(6,dtype=np.int64)
    result=scorer.native_metrics.metrics(p,y,scorer.CLASSES)
    assert result['accuracy']==result['macro_F1']==1. and result['Brier']==0 and result['raw_NLL']==0
    assert result['confusion']==np.eye(6,dtype=int).tolist();checks.append('all_six_classes_perfect')
    q=p.copy();q[0]=p[1]
    result=scorer.native_metrics.metrics(q,y,scorer.CLASSES)
    assert result['raw_NLL'] is None and result['raw_NLL_is_infinite'] and result['zero_probability_truth_count']==1
    assert result['confusion'][0][1]==1 and result['source_style_confusion_predicted_rows_truth_columns'][1][0]==1
    assert np.isclose(result['Brier'],2/6) and np.isclose(result['accuracy'],5/6);checks.append('zero_probability_NLL_and_confusion_orientation')
    result=scorer.native_metrics.metrics(p[:1],y[:1],scorer.CLASSES)
    assert result['macro_F1']==1/6 and result['per_class']['WHG']['F1']==0;checks.append('absent_class_zero_macro_policy')
    summary=scorer.native_metrics.parent_summary([0.,.25,.5,.75,1.])
    assert summary['mean']==.5 and np.isclose(summary['sample_SD'],np.std([0.,.25,.5,.75,1.],ddof=1))
    assert scorer.native_metrics.parent_summary([None,1.,1.,1.,1.])['mean'] is None;checks.append('equal_parent_sample_SD_and_infinite_parent')
    records=[dict(sample_index=i,native_site_id=i,position_original_bp=float(i)) for i in range(6)]
    logits=np.zeros((6,6),np.float32);probs=np.full((6,6),1/6,np.float32)
    arrays=dict(logits=logits,probabilities=probs,example_indices=np.arange(6,dtype=np.int64),sample_indices=np.arange(6,dtype=np.int64),native_site_ids=np.arange(6,dtype=np.int64),positions_original_bp=np.arange(6,dtype=np.float64))
    scorer.validate_arrays(arrays,records,'synthetic_parent',True)
    for field in ['example_indices','sample_indices','native_site_ids','positions_original_bp']:
        altered={key:value.copy() for key,value in arrays.items()};altered[field][0]=altered[field][1]
        rejects(lambda:scorer.validate_arrays(altered,records,'synthetic_parent',True))
    altered={key:value.copy() for key,value in arrays.items()};altered['probabilities'][0]=p[0]
    rejects(lambda:scorer.validate_arrays(altered,records,'synthetic_parent',True));checks.append('identity_and_unmodified_softmax_rejections')
    bad=probs.copy();bad[0,0]=np.nan;rejects(lambda:scorer.validate_probabilities(bad,6))
    bad=probs.copy();bad[0]*=.5;rejects(lambda:scorer.validate_probabilities(bad,6));checks.append('nonfinite_and_unnormalized_rejections')
    raw=np.arange(686,dtype=np.int64);keys=[['sample_'+str(i),0] for i in raw]
    identities=[dict(sample_id=key[0],haplotype=0,sample_index=int(i)) for key,i in zip(keys,raw)]
    paths=np.ones((2,686),dtype=np.int8);paths[0,0]=2;paths[1,0]=3
    registry=dict(schema='ap_paper_sampled_registry_v1',parent_id='synthetic_parent',split='test',examples=2,
        query_identities=identities,records=[dict(example_index=0,sample_id='sample_0',haplotype=0,sample_index=0,position_original_bp=99999999.,cohort='Neo'),dict(example_index=1,sample_id='sample_0',haplotype=0,sample_index=0,position_original_bp=100000000.,cohort='Neo')])
    schema=dict(parent_id='synthetic_parent',split='test',native_paths_1based=[1,2,3,4,5,6],query_sample_indices=raw.tolist(),query_keys=keys)
    truth=dict(intervals_bp=np.array([[0.,100000000.],[100000000.,200000000.]]),native_paths_1based=paths,query_raw_node_ids=raw)
    assert np.array_equal(scorer.sampling.native_truth_labels(registry,schema,truth),[1,2]);checks.append('original_native_join_half_open_boundary')
    broken=dict(schema,query_keys=keys[::-1]);rejects(lambda:scorer.sampling.native_truth_labels(registry,broken,truth));checks.append('semantic_sample_haplotype_join_rejection')
    rejects(lambda:scorer.checked({'path':'research/joint_evaluation_20261029/scoring/truth/absent.npz','sha256':'0'*64},{}));checks.append('private_path_denied_before_read')
    with tempfile.TemporaryDirectory(dir=scorer.HERE,prefix='synthetic_') as directory:
        root=Path(directory);file=root/'plain.txt';file.write_text('original')
        frozen=scorer.ref(file);file.write_text('changed');rejects(lambda:scorer.checked(frozen,{}))
        seal=root/'seal.json';seal.write_text(json.dumps(dict(schema='joint_native6_prediction_barrier_v1',verdict='pass',payload={'sealed':1})))
        original=scorer.prediction_barrier
        scorer.prediction_barrier=lambda _:({'sealed':2},{},{},{},{})
        try:rejects(lambda:scorer.score('unused',str(seal),str(root/'must_not_exist.json')))
        finally:scorer.prediction_barrier=original
        assert not (root/'must_not_exist.json').exists()
    checks.append('checksum_mutation_and_changed_seal_fail_before_truth')
    receipt=dict(schema='joint_native6_scorer_synthetic_tests_v1',verdict='pass',checks=checks,scorer_sha256=scorer.sha(Path(scorer.__file__)),test_source_sha256=scorer.sha(Path(__file__)),real_TEST_truth_reads=0,optimizer_updates=0,GPU_runs=0)
    scorer.save(scorer.HERE/'SCORER_CPU_TESTS_v3.json',receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
