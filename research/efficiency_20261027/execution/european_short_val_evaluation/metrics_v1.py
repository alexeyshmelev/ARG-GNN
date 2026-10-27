"""Pure fixed native6 and explicitly named derived development endpoints."""
import itertools
import numpy as np
from . import contracts_v1 as c


def classification(labels,calls,names):
    labels=np.asarray(labels);calls=np.asarray(calls);k=len(names)
    c.need(labels.shape==calls.shape and labels.ndim==1 and labels.dtype.kind in 'iu' and calls.dtype.kind in 'iu'
        and ((labels>=0)&(labels<k)).all() and ((calls>=0)&(calls<k)).all(),'Complete integer class labels and calls')
    matrix=np.bincount(labels*k+calls,minlength=k*k).reshape(k,k)
    per_class={};recalls=[];f1s=[]
    for index,name in enumerate(names):
        actual=int(matrix[index].sum());predicted=int(matrix[:,index].sum());tp=int(matrix[index,index])
        recall=tp/actual if actual else None
        f1=2*tp/(actual+predicted) if actual+predicted else 0.
        per_class[name]=dict(support=actual,predicted=predicted,true_positive=tp,
            recall=recall,precision=tp/predicted if predicted else None,F1=f1,
            recall_undefined_due_to_absent_true_class=not bool(actual))
        if recall is not None:recalls.append(recall)
        f1s.append(f1)
    return dict(records=len(labels),correct=int((labels==calls).sum()),
        accuracy=float((labels==calls).mean()) if len(labels) else None,
        macro_F1=float(np.mean(f1s)),macro_F1_policy='All declared classes; zero when true and predicted supports are both zero',
        macro_recall_present_classes=float(np.mean(recalls)) if recalls else None,
        macro_recall_all_classes=float(np.mean(recalls)) if len(recalls)==k else None,
        absent_true_classes=[name for name,item in per_class.items() if item['support']==0],
        per_class=per_class,confusion=matrix.tolist(),confusion_convention='True rows; predicted columns',class_order=list(names))


def paired(labels,left,right):
    a=np.asarray(left)==labels;b=np.asarray(right)==labels;n=len(labels)
    wins=int((a&~b).sum());losses=int((~a&b).sum())
    return dict(records=n,left_correct_only=wins,right_correct_only=losses,
        both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()),
        accuracy_difference_left_minus_right=(wins-losses)/n if n else None,
        no_independent_row_uncertainty_claim=True)


def baa_ana_chg(labels,calls):
    mapping=np.asarray(c.NATIVE_TO_C4);truth=mapping[labels];predicted=mapping[calls]
    keep=np.isin(truth,[0,1]);truth=truth[keep];predicted=predicted[keep]
    # Every off-pair call remains a wrong prediction in the original denominator.
    categorized=np.where(np.isin(predicted,[0,1]),predicted,2)
    table=np.bincount(truth*3+categorized,minlength=6).reshape(2,3)
    return dict(endpoint='Derived Ana-versus-CHG within public BAA and true-derived-Ana-or-CHG subset',
        original_BAA_rows=len(labels),conditional_support=len(truth),excluded_true_other=len(labels)-len(truth),
        true_Ana_support=int((truth==0).sum()),true_CHG_support=int((truth==1).sum()),
        correct=int((truth==categorized).sum()),errors=int((truth!=categorized).sum()),
        off_pair_predictions_counted_as_errors=int((categorized==2).sum()),
        accuracy=float((truth==categorized).mean()) if len(truth) else None,
        confusion_true_Ana_CHG_by_predicted_Ana_CHG_other=table.tolist(),
        native_to_derived_C4=list(c.NATIVE_TO_C4),native6_class_distinctions_retained_in_primary=True)


def report(labels,probabilities,cohorts):
    labels=np.asarray(labels);cohorts=np.asarray(cohorts)
    c.need(labels.ndim==cohorts.ndim==1 and len(labels)>0 and len(labels)==len(cohorts)
        and set(probabilities)==set(c.METHODS) and np.isin(cohorts,c.COHORTS).all(),'Every declared condition on the same rows')
    calls={method:c.probability_array(value,len(labels)).argmax(1) for method,value in probabilities.items()}
    groups={'overall':np.ones(len(labels),dtype=bool),**{cohort:cohorts==cohort for cohort in c.COHORTS}}
    native={method:{name:classification(labels[mask],value[mask],c.CLASSES) for name,mask in groups.items()}
            for method,value in calls.items()}
    comparisons={left+' minus '+right:{name:paired(labels[mask],calls[left][mask],calls[right][mask])
                    for name,mask in groups.items()} for left,right in itertools.combinations(c.METHODS,2)}
    mapping=np.asarray(c.NATIVE_TO_C4)
    derived={method:{name:classification(mapping[labels[mask]],mapping[value[mask]],c.C4)
                    for name,mask in groups.items()} for method,value in calls.items()}
    baa=cohorts=='BAA'
    return dict(native6=native,paired_native6_correctness=comparisons,
        within_public_BAA_native6={method:native[method]['BAA'] for method in c.METHODS},
        within_public_BAA_derived_Ana_CHG={method:baa_ana_chg(labels[baa],value[baa]) for method,value in calls.items()},
        derived_C4_from_native_argmax_calls=derived,
        derived_C4_rule='Map each native6 argmax call via[0,1,2,3,0,1]; separate from native6 and from probability-sum argmax',
        independent_parent_count=1,development_evidence_only=True,independent_chromosome_replication=False,
        across_parent_variance=None,row_level_independence_claimed=False,
        cohort_prior_interpretation='Public-cohort metadata baseline; a positive gap is development evidence beyond this prior,not architectural novelty.')
