"""One fixed TRAIN-only public-cohort frequency control; no model search."""
from . import contracts_v1 as c


def count_prior(cohorts,labels):
    import numpy as np
    cohorts=np.asarray(cohorts);labels=np.asarray(labels)
    c.need(cohorts.ndim==labels.ndim==1 and len(labels)>0 and len(cohorts)==len(labels)
        and labels.dtype.kind in 'iu' and ((labels>=0)&(labels<6)).all()
        and np.isin(cohorts,c.COHORTS).all(),'Only observed query cohorts and six-class TRAIN labels')
    counts={cohort:np.bincount(labels[cohorts==cohort],minlength=6).tolist() for cohort in c.COHORTS}
    pooled=np.bincount(labels,minlength=6).tolist()
    vectors={};calls={}
    for cohort in c.COHORTS:
        chosen=counts[cohort] if sum(counts[cohort]) else pooled
        vectors[cohort]=[value/sum(chosen) for value in chosen]
        calls[cohort]=int(np.argmax(chosen))
    return dict(counts=counts,pooled_counts=pooled,probabilities=vectors,class_calls_0based=calls,
        tie_break='First maximum in frozen native6 class order',missing_cohort_fallback='Pooled original TRAIN counts',
        training_rows=len(labels),features_used=['public observed query cohort'],TRAIN_passes=1)


def fit_TRAIN_prior(output_directory,*,source_review_ref,guard,execution_check,write_headroom_bytes):
    import numpy as np
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    spec,plan=c.train_contract();cohorts=[];labels=[];bindings={}
    for parent in plan['train_parent_ids']:
        execution_check();guard(0);item=spec['train_inputs'][parent]
        registry=c.io._read(item['sampled_registry_ref'],observed=True)
        target=c.io._read(item['training_targets_ref'])
        c.need(registry.get('parent_id')==target.get('parent_id')==parent
            and registry.get('split')==target.get('split')=='train'
            and registry.get('labels_present') is False and registry.get('query_truth_read') is False
            and registry.get('cohort_quotas')==plan['parents'][parent]['cohort_quotas']
            and target.get('schema')=='ap_paper_TRAIN_native_targets_v1'
            and target.get('registry_ref')==item['sampled_registry_ref']
            and target.get('no_label_dependent_selection') is True
            and target.get('native_path_names')==list(c.CLASSES),'Exact original frozen TRAIN targets and registry')
        c.sampling.registry_arrays(registry)
        rows=sorted(registry['records'],key=lambda row:row['example_index'])
        values=c.io._arrays(target['arrays_ref'],cap=2*2**20)['native_labels_0based']
        c.need(values.shape==(len(rows),) and target['rows']==registry['examples']==len(rows),
            'Preserve every original TRAIN focal row,including repeated sample/site draws')
        cohorts.extend(row['cohort'] for row in rows);labels.append(values)
        bindings[parent]={key:item[key] for key in ('sampled_registry_ref','training_targets_ref')}
    c.need(len(cohorts)==50000 and all(cohorts.count(cohort)==10000 for cohort in c.COHORTS),
        'Exactly original50000 TRAIN rows and10000 per public query cohort')
    result=count_prior(np.asarray(cohorts),np.concatenate(labels))
    output=c.io._new_output(output_directory)
    return c.io._json(output/'prior.json',dict(schema='European_short_VAL_TRAIN_cohort_prior_v1',
        source_manifest_ref=source,definition_ref=c.definition_ref(),training_plan_ref=c.TRAIN_PLAN_REF,
        train_bindings=bindings,class_order=list(c.CLASSES),VAL_truth_opened=False,TEST_accessed=False,**result),guard)


def load_prior(prior_ref,source_manifest_ref):
    import numpy as np
    value=c.io._read(prior_ref,observed=True);spec,plan=c.train_contract()
    expected={parent:{key:spec['train_inputs'][parent][key] for key in ('sampled_registry_ref','training_targets_ref')}
              for parent in plan['train_parent_ids']}
    c.need(value.get('schema')=='European_short_VAL_TRAIN_cohort_prior_v1'
        and value.get('source_manifest_ref')==source_manifest_ref and value.get('definition_ref')==c.definition_ref()
        and value.get('training_plan_ref')==c.TRAIN_PLAN_REF and value.get('train_bindings')==expected
        and value.get('class_order')==list(c.CLASSES) and value.get('training_rows')==50000
        and value.get('TRAIN_passes')==1 and value.get('VAL_truth_opened') is False and value.get('TEST_accessed') is False
        and value.get('features_used')==['public observed query cohort']
        and value.get('tie_break')=='First maximum in frozen native6 class order'
        and value.get('missing_cohort_fallback')=='Pooled original TRAIN counts','Exact frozen TRAIN-only prior')
    counts=np.asarray([value['counts'][cohort] for cohort in c.COHORTS])
    pooled=np.asarray(value['pooled_counts'])
    c.need(counts.shape==(5,6) and counts.dtype.kind in 'iu' and (counts>=0).all()
        and np.array_equal(counts.sum(0),pooled) and np.all(counts.sum(1)==10000),'Exact complete TRAIN prior counts')
    for cohort,row in zip(c.COHORTS,counts):
        c.need(value['probabilities'][cohort]==(row/row.sum()).tolist()
            and value['class_calls_0based'][cohort]==int(row.argmax()),'Frozen frequencies and deterministic tie rule')
    return value


def predict_prior(prior_ref,registry_ref,inference_ref,output_directory,*,source_review_ref,
                  guard,execution_check,write_headroom_bytes):
    import numpy as np
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    fitted=load_prior(prior_ref,source);_,rows=c.registry(registry_ref,inference_ref)
    probabilities=np.asarray([fitted['probabilities'][row['cohort']] for row in rows],dtype=np.float64)
    return c.seal_predictions('TRAIN_cohort_prior',probabilities,registry_ref,inference_ref,output_directory,
        source_manifest_ref=source,provenance=dict(fitted_prior_ref=prior_ref),guard=guard,execution_check=execution_check)
