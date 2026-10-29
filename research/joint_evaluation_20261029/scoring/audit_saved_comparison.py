"""Independent saved-result arithmetic/provenance audit, without private truth I/O."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,time
import numpy as np

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/joint_evaluation_20261029/scoring'
CLASSES=['Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA']
COHORTS=['present_day','Bronze_Age','BAA','Neo','Yam']
BASELINES=['ancestralpaths_historical_cnn_native6','ancestralpaths_dense_control','gst_lai_8_native_graphsetconv','treeloom']
NEW='treeloom_joint_structure_v1'
METRICS=['accuracy','macro_F1','Brier','raw_NLL','mean_predictive_entropy_nats','mean_max_probability','ECE_10_equal_width']

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def path(value):
    p=(ROOT/value).resolve();need(p.is_relative_to(ROOT),'Repository path required');return p

def ref(value):
    p=path(value);digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):digest.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=digest.hexdigest(),bytes=p.stat().st_size)

def checked(record):
    need(not any('truth' in part.lower() for part in Path(record['path']).parts),'This audit never opens private truth')
    actual=ref(record['path']);need(actual['sha256']==record['sha256'],'Changed source/artifact '+record['path'])
    if 'bytes' in record:need(actual['bytes']==record['bytes'],'Changed artifact byte count')
    return path(record['path'])

def close(a,b,label,atol=1e-12):
    need(a is None and b is None or a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=atol),'Arithmetic mismatch '+label)

def summary(values,saved,label):
    need(saved['independent_parents']==5 and saved['weighting']=='equal parent' and saved['ddof']==1,'Parent summary convention')
    finite=all(v is not None for v in values)
    close(saved['mean'],statistics.mean(values) if finite else None,label+' mean')
    close(saved['sample_SD'],statistics.stdev(values) if finite else None,label+' sample SD')

def group(row,name):return row['overall'] if name=='overall' else row['cohorts'][name]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comparison',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    report=json.loads(path(args.comparison).read_text());protocol=json.loads(checked(report['protocol_ref']).read_text())
    seal=json.loads(checked(report['prediction_barrier_ref']).read_text());payload=seal['payload']
    historical=json.loads(checked(protocol['historical_comparison_ref']).read_text())
    need(report['schema']=='joint_native6_external_comparison_v1' and seal['verdict']=='pass','Saved comparison and prediction barrier')
    need(report['methods']==BASELINES+[NEW] and report['native_class_order']==CLASSES,'Exact methods and declared-six classes')
    parents=['ancestralpaths_european_test_'+str(i).zfill(3) for i in range(5)]
    need(report['parents']==protocol['parents']==payload['parents']==parents,'Five frozen TEST parents')
    need(payload['methods']==report['methods'] and payload['complete_product_count']==25 and payload['canonical_targets_per_method']==50000,'All25 original-size prediction seal')
    need(payload['protocol_ref']==report['protocol_ref'] and payload['private_target_payloads_opened'] is False,'Pre-truth seal identity')
    need(report['selection_after_TEST'] is False and report['all_25_predictions_authenticated_before_private_truth'] is True,'Frozen selection and truth ordering')
    for item in payload['authenticated_files'].values():checked(item)
    freeze=json.loads(checked(protocol['prediction_freeze_ref']).read_text())
    need(freeze['checkpoint_ref']['sha256']=='651afd90ef9ccb51436c5b5816569b77d883109c474db7c81a84ce8faf068a9b','Frozen model checkpoint')
    baseline_groups=0;groups_checked=0;probability_groups=0;infinite=[]
    for parent in parents:
        truth_counts=report['scoring_truth_evidence'][parent]['class_counts']
        need(report['scoring_truth_evidence'][parent]['refs']==historical['test_targets'][parent],'Original truth refs unchanged; payload not reread')
        for method in report['methods']:
            row=report['rows'][method][parent];evidence=payload['predictions'][method][parent]
            product=json.loads(checked(evidence['product_ref']).read_text())
            registry=json.loads(checked(evidence['registry_ref']).read_text());records=sorted(registry['records'],key=lambda x:x['example_index'])
            cohorts=np.asarray([x['cohort'] for x in records])
            with np.load(checked(evidence['predictions_ref']),allow_pickle=False) as z:
                probabilities=z['probabilities'].copy();ids=z['example_indices'].copy()
                if method==NEW:
                    logits=z['logits'].copy()
                    need(logits.dtype==np.float32 and logits.shape==(10000,6) and np.isfinite(logits).all(),'Raw frozen model logits')
                    v=logits.astype(np.float64);v=np.exp(v-v.max(1,keepdims=True));v/=v.sum(1,keepdims=True)
                    need(np.allclose(v,probabilities,atol=2e-7,rtol=2e-5),'Raw new softmax; no masking or calibration')
                    for field,key in [('sample_indices','sample_index'),('native_site_ids','native_site_id'),('positions_original_bp','position_original_bp')]:
                        need(np.array_equal(z[field],[r[key] for r in records]),'Exact canonical identity '+field)
                    need(product['freeze_ref']==protocol['prediction_freeze_ref'],'Identical repaired freeze across parents')
            need(probabilities.dtype==np.float32 and probabilities.shape==(10000,6) and np.isfinite(probabilities).all(),'Native-six probabilities')
            need(np.all((probabilities>=0)&(probabilities<=1)) and np.allclose(probabilities.sum(1),1,atol=1e-6,rtol=1e-5),'Finite raw normalized probabilities')
            need(ids.dtype==np.int64 and np.array_equal(ids,np.arange(10000)),'Every original target once')
            need(np.array_equal(sum(np.asarray(row['cohorts'][c]['confusion']) for c in COHORTS),row['overall']['confusion']),'Cohort confusion sums to parent')
            for name in ['overall']+COHORTS:
                metric=group(row,name);n=10000 if name=='overall' else 2000
                cm=np.asarray(metric['confusion']);need(cm.shape==(6,6) and cm.dtype.kind in 'iu' and (cm>=0).all() and cm.sum()==n,'Full group confusion/count')
                need(metric['records']==n and metric['confusion_convention']=='truth rows, prediction columns','Native row count/convention')
                need(np.array_equal(cm.T,metric['source_style_confusion_predicted_rows_truth_columns']),'Transposed confusion convention')
                if name=='overall':need(np.array_equal(cm.sum(1),truth_counts),'Original target class counts')
                precision=[];recall=[];f1=[]
                for i,label in enumerate(CLASSES):
                    actual=int(cm[i].sum());predicted=int(cm[:,i].sum());tp=int(cm[i,i]);pc=metric['per_class'][label]
                    values=dict(precision=tp/predicted if predicted else 0.,recall=tp/actual if actual else 0.,F1=2*tp/(actual+predicted) if actual+predicted else 0.)
                    need(pc['actual_count']==actual and pc['predicted_count']==predicted,'Per-class support')
                    for key,value in values.items():close(pc[key],value,label+'/'+key)
                    f1.append(values['F1'])
                close(metric['accuracy'],int(cm.trace())/n,'Accuracy');close(metric['macro_F1'],statistics.mean(f1),'Declared-six macro F1')
                need(metric['raw_NLL_is_infinite']==(metric['zero_probability_truth_count']>0),'Raw NLL infinite flag')
                need((metric['raw_NLL'] is None)==metric['raw_NLL_is_infinite'],'Raw NLL null convention')
                if metric['raw_NLL_is_infinite']:infinite.append(method+'/'+parent+'/'+name)
                p=probabilities if name=='overall' else probabilities[cohorts==name]
                need(len(p)==n and np.array_equal(np.bincount(p.argmax(1),minlength=6),cm.sum(0)),'Predicted class marginal agrees with saved probabilities')
                confidence=p.max(1);entropy=-np.sum(np.where(p>0,p*np.log(np.where(p>0,p,1.)),0),axis=1)
                close(metric['mean_predictive_entropy_nats'],float(entropy.mean()),'Raw entropy');close(metric['mean_max_probability'],float(confidence.mean()),'Raw confidence')
                bins=metric['reliability_bins'];need(len(bins)==10,'Ten calibration bins')
                ece=0;correct_from_bins=0
                for i,b in enumerate(bins):
                    mask=(confidence>=i/10)&((confidence<(i+1)/10) if i<9 else confidence<=1);count=int(mask.sum())
                    need(b['lower']==i/10 and b['upper']==(i+1)/10 and b['records']==count,'Unchanged confidence bin membership')
                    close(b['mean_confidence'],float(confidence[mask].mean()) if count else None,'Bin confidence')
                    if count:
                        correct_from_bins+=b['accuracy']*count;ece+=count/n*abs(b['mean_confidence']-b['accuracy'])
                close(correct_from_bins,float(cm.trace()),'Reliability correct count');close(ece,metric['ECE_10_equal_width'],'ECE')
                if method in BASELINES:
                    old=historical['rows'][method][parent]
                    need(metric==(old['overall']['native6'] if name=='overall' else old['cohorts'][name]['native6']),'Historical metric exact parity');baseline_groups+=1
                groups_checked+=1;probability_groups+=1
        for method in report['methods']:
            row=report['rows'][method][parent]
            for key in ['Brier','raw_NLL','mean_predictive_entropy_nats','mean_max_probability']:
                values=[row['cohorts'][c][key] for c in COHORTS]
                close(row['overall'][key],statistics.mean(values) if all(v is not None for v in values) else None,'Cohort additive '+key,atol=2e-6)
    for method in report['methods']:
        for name in ['overall']+COHORTS:
            rows=[group(report['rows'][method][parent],name) for parent in parents];saved=report['summary'][method][name]
            for key in METRICS:summary([r[key] for r in rows],saved[key],method+'/'+name+'/'+key)
            for label in CLASSES:
                for key in ['precision','recall','F1']:summary([r['per_class'][label][key] for r in rows],saved['per_class'][label][key],label+'/'+key)
            need(np.array_equal(sum(np.asarray(r['confusion']) for r in rows),saved['aggregate_confusion']),'Aggregate parent confusion')
    for baseline in BASELINES:
        for name in ['overall']+COHORTS:
            for key in ['accuracy','macro_F1','Brier','raw_NLL']:
                pairs=[(group(report['rows'][NEW][p],name)[key],group(report['rows'][baseline][p],name)[key]) for p in parents]
                deltas=[None if a is None or b is None else a-b for a,b in pairs];saved=report['paired_differences'][baseline][name][key]
                need(saved['per_parent']==deltas and saved['direction']==NEW+' minus '+baseline,'Paired parent deltas')
                summary(deltas,saved,baseline+'/'+name+'/'+key)
    result=dict(schema='independent_saved_comparison_audit_v1',verdict='pass',created_unix=time.time(),comparison_ref=ref(args.comparison),audit_source_ref=ref(__file__),protocol_ref=report['protocol_ref'],prediction_barrier_ref=report['prediction_barrier_ref'],methods=report['methods'],parents=parents,targets_per_method=50000,metric_groups_checked=groups_checked,probability_groups_checked=probability_groups,historical_metric_groups_exactly_equal=baseline_groups,infinite_raw_NLL_groups=infinite,all_sealed_sources_and_predictions_reauthenticated=True,private_truth_payloads_opened=False,optimizer_or_model_execution=False,summary=report['summary'],paired_differences=report['paired_differences'],limitations=['No repeated private truth read. New target-dependent per-record NLL/Brier rely on the reviewed common scorer and exact historical metric parity; confusion/class/calibration/additive arithmetic and raw prediction-derived statistics were independently recomputed.','Five equal-weight parents with sample SD do not establish target-level independent uncertainty or a general superiority claim.'])
    target=path(args.output);need(target.is_relative_to(HERE),'Audit output owned scoring directory')
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(verdict='pass',audit_ref=ref(target),metric_groups=groups_checked,historical_exact_groups=baseline_groups,overall_accuracy={m:report['summary'][m]['overall']['accuracy'] for m in report['methods']}),indent=2))

if __name__=='__main__':main()
