"""Independent saved-result audit; no inference, fitting or optimizer execution."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/redesign_20261028'
EXPORT=BASE/'diagnostic_packets'
manifest=json.loads((EXPORT/'manifest.json').read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(logits,labels):
    assert logits.shape==(len(labels),6) and np.isfinite(logits).all()
    pred=logits.argmax(axis=1)
    matrix=np.zeros((6,6),dtype=np.int64)
    np.add.at(matrix,(labels,pred),1)
    den=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*np.diag(matrix),den,out=np.zeros(6),where=den>0)
    x=logits.astype(np.float64)
    m=x.max(1)
    ce=np.mean(m+np.log(np.exp(x-m[:,None]).sum(1))-x[np.arange(len(x)),labels])
    return dict(cross_entropy=float(ce),accuracy=float(np.trace(matrix)/len(labels)),
        macro_f1_six_classes=float(f1.mean()),class_f1=f1.tolist(),
        confusion_true_rows_predicted_columns=matrix.tolist(),
        true_class_counts=matrix.sum(1).tolist(),predicted_class_counts=matrix.sum(0).tolist()),pred

source_labels={}
for parent,binding in manifest['source_bindings'].items():
    ref=binding['training_target_arrays_ref'];path=ROOT/ref['path']
    assert sha(path)==ref['sha256']
    with np.load(path,allow_pickle=False) as a:source_labels[parent]=a['native_labels_0based'].copy()
expected={split:np.array([source_labels[r['parent_id']][r['example_index']] for r in manifest[split]],dtype=np.int64)
          for split in ['fit','development']}
results={};predictions={};starts={};audits={}
for arm in ['no_cross','structure']:
    folder=BASE/'root/learning_runs'/arm
    result=json.loads((folder/'result.json').read_text());results[arm]=result
    started=json.loads((folder/'started.json').read_text());starts[arm]=started
    status=json.loads((folder/'status.json').read_text())
    assert status['stage']=='completed' and status['updates']==400 and result['updates']==400
    rows=[json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
    assert len(rows)==400
    for i,row in enumerate(rows):
        assert row['update']==i+1 and row['indices']==started['batch_order'][i]
        assert len(row['indices'])==4
        for key in ['loss','gradient_norm_before_clip','elapsed_train_seconds','loader_wait_seconds',
                    'forward_seconds','backward_seconds','optimizer_seconds',
                    'packet_read_worker_seconds','collation_preparation_worker_seconds']:
            assert np.isfinite(row[key]) and row[key]>=0
    for key,total in result['timings'].items():
        assert np.isclose(total,sum(row[key] for row in rows),rtol=1e-10,atol=1e-8)
    assert result['model_source_sha256']==sha(BASE/'architecture/window_model_frozen_v1.py')
    assert result['harness_sha256']==sha(BASE/'root/learning_diagnostic.py')
    assert result['protocol_sha256']==sha(BASE/'root/pilot_protocol.json')
    assert result['manifest_sha256']==sha(EXPORT/'manifest.json')
    assert result['cuda_initialized'] is False
    checked={}
    for stage in ['initial','final']:
        for split in ['fit','development']:
            with np.load(folder/f'{stage}_{split}.npz',allow_pickle=False) as a:
                logits=a['logits'];labels=a['labels']
            np.testing.assert_array_equal(labels,expected[split])
            independent,pred=metrics(logits,labels)
            saved=result[stage][split]
            for key in ['cross_entropy','accuracy','macro_f1_six_classes']:
                assert np.isclose(independent[key],saved[key],rtol=1e-6,atol=2e-7),(arm,stage,split,key)
            for key in ['confusion_true_rows_predicted_columns','true_class_counts','predicted_class_counts']:
                assert independent[key]==saved[key]
            checked[f'{stage}_{split}']=independent
            if stage=='final' and split=='development':predictions[arm]=pred
    audits[arm]=dict(finite_training_rows=400,target_presentations=1600,
        result_sha256=sha(folder/'result.json'),verified_metrics=checked,
        training_elapsed_seconds=result['training_elapsed_seconds'])
for key in ['initial_parameter_sha256','batch_order_sha256','protocol_sha256','manifest_sha256','model_source_sha256','harness_sha256']:
    assert starts['no_cross'][key]==starts['structure'][key],key
np.testing.assert_array_equal(starts['no_cross']['batch_order'],starts['structure']['batch_order'])
y=expected['development'];correct0=predictions['no_cross']==y;correct1=predictions['structure']==y
receipt=dict(schema='independent_learning_result_review_v1',verdict='pass',arms=audits,
    matching_initialization_order_protocol_manifest_model_harness=True,
    all_saved_labels_join_canonical_prepared_TRAIN_targets=True,
    paired_development=dict(draws=len(y),no_cross_correct=int(correct0.sum()),structure_correct=int(correct1.sum()),
        structure_only_correct=int((correct1&~correct0).sum()),no_cross_only_correct=int((correct0&~correct1).sum()),
        difference_correct=int(correct1.sum()-correct0.sum()),
        structure_minus_no_cross_NLL=audits['structure']['verified_metrics']['final_development']['cross_entropy']-
            audits['no_cross']['verified_metrics']['final_development']['cross_entropy']),
    conclusion='Both small models learn. Structure gives one more correct development draw out of64 but worse NLL and higher cost. No accuracy superiority or clade-specific benefit is established.',
    training_by_reviewer=False,TEST_read=False)
(BASE/'review/LEARNING_RESULT_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='arms'},indent=2))
print(json.dumps({arm:dict(final_development=a['verified_metrics']['final_development'],
    training_elapsed_seconds=a['training_elapsed_seconds']) for arm,a in audits.items()},indent=2))
