"""Frozen one-parent evaluation definitions and exact observed/source bindings."""
import json
from pathlib import Path
from arggnn.benchmarking import european_native_models as io, european_sampling as sampling
from research.efficiency_20261027.integration_candidate.european_short_development_val import short_plan_v1 as short

ROOT=short.ROOT
HERE=Path(__file__).parent
METHODS=('cohort8_treeloom','original_treeloom','historical_cnn','released_dense','TRAIN_cohort_prior')
CLASSES=tuple(sampling.NATIVE_PATHS)
COHORTS=tuple(sampling.COHORTS)
C4=('Ana','CHG','WHG','EHG')
NATIVE_TO_C4=(0,1,2,3,0,1)
TRAIN_JOB_REF=dict(path='.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json',sha256='fecd670f6f26332ab165648f4d65ad51131650d195e86df2a39a5545f2b724c1')
TRAIN_PLAN_REF=dict(path='configs/benchmarks/recipes/ap_paper_sampled_plan_v1.json',sha256='2d39761d0cc030af72ba1e313be63dfce3fa7e5a4e1f4cecbdf33a5109fc7f5b')
TRAIN_PROTOCOL='ancestralpaths_paper200mb_native6'
SHORT_PROTOCOL='ancestralpaths_short10mb_development_native6'
TRAIN_CONDITION='AP_paper_scale_200Mb_native6_observed_cohort8_v1'
need=io._need


def definition_ref():return io._ref(HERE/'definition_v1.json')


def verify_sources(review_ref):
    manifest_ref=io._ref(HERE/'source_pins_v1.json');manifest=io._read(manifest_ref)
    need(manifest.get('schema')=='European_short_VAL_evaluation_sources_v1','Exact evaluation source manifest')
    for entry in manifest['source_refs']:need(io._ref(entry['path'])==entry,'Pinned evaluation source changed: '+entry['path'])
    review=io._read(review_ref,observed=True)
    need(review.get('schema')=='European_short_VAL_evaluation_source_review_v1' and review.get('passed') is True
         and review.get('source_manifest_ref')==manifest_ref,'Exact independently reviewed evaluation implementation')
    definition=io._read(definition_ref(),observed=True)
    need(definition['methods']==list(METHODS) and definition['class_order']==list(CLASSES)
         and definition['short_plan_ref']==short.PLAN_REF,'Frozen methods,endpoint and one-parent definition')
    short.validate_plan()
    return manifest_ref


def ready(review_ref,guard,execution_check,headroom):
    io._ready(guard,execution_check,headroom)
    return verify_sources(review_ref)


def train_contract():
    spec=io._read(TRAIN_JOB_REF,observed=True)['call']['spec']['scientific_spec']
    plan=io._read(TRAIN_PLAN_REF,observed=True)
    need(spec['protocol_id']==TRAIN_PROTOCOL and spec['sampling_plan_ref']==TRAIN_PLAN_REF
         and plan==sampling.paper_sampling_plan(plan['train_parent_ids'],plan['test_parent_ids'],seed=plan['seed'])
         and len(plan['train_parent_ids'])==3 and plan['training_total']==50000,
         'Original three200Mb TRAIN parents and50000 unchanged focal rows')
    return spec,plan


def registry(ref,inference_ref):
    value=io._read(ref,observed=True);frozen=short.validate_plan();parent=frozen['registration']['parent_id']
    need(value.get('schema')=='ap_paper_sampled_registry_v1' and value.get('short_development_plan_ref')==short.PLAN_REF
         and value.get('parent_id')==parent and value.get('split')=='validation'
         and value.get('sequence_length_bp')==10000000 and value.get('observation_ref')==inference_ref
         and value.get('development_model_selection_only') is True and value.get('examples')==10000
         and value.get('cohort_quotas')=={cohort:2000 for cohort in COHORTS}
         and value.get('labels_present') is False and value.get('query_truth_read') is False,
         'Exactly the frozen10000 observed rows of the single10Mb VAL parent')
    sampling.registry_arrays(value)
    rows=sorted(value['records'],key=lambda row:row['example_index'])
    need(all(sum(row['cohort']==cohort for row in rows)==2000 for cohort in COHORTS),
         'All five observed query-cohort denominators preserved')
    return value,rows


def probability_array(value,n):
    import numpy as np
    value=np.asarray(value)
    need(value.shape==(n,6) and value.dtype.kind=='f' and np.isfinite(value).all()
         and ((value>=0)&(value<=1)).all() and np.allclose(value.sum(1),1.,atol=1e-6,rtol=1e-5),
         'Complete finite unmodified six-class probability rows')
    return value


def acceptance(method,fit_ref,checkpoint_ref,acceptance_ref,source_manifest_ref):
    value=io._read(acceptance_ref,observed=True)
    need(value.get('schema')=='European_short_VAL_checkpoint_acceptance_v1' and value.get('passed') is True
         and value.get('method')==method and value.get('fit_completion_ref')==fit_ref
         and value.get('checkpoint_ref')==checkpoint_ref and value.get('evaluation_source_manifest_ref')==source_manifest_ref
         and value.get('completed_fit_and_genuine_terminal_wait_verified') is True
         and value.get('fit_process_family_absent') is True and value.get('checkpoint_selected_without_VAL_truth') is True,
         'ROOT binds exact completed,reaped,fixed checkpoint before VAL truth')
    need(type(value.get('reviewed_fit_source_refs')) is list and value['reviewed_fit_source_refs'],
         'Explicit reviewed fit sources required')
    for entry in value['reviewed_fit_source_refs']:io._checked(entry,observed=True)
    io._checked(checkpoint_ref,observed=True)
    return value


def seal_predictions(method,probabilities,registry_ref,inference_ref,output_directory,*,
                     source_manifest_ref,provenance,guard,execution_check):
    import numpy as np
    need(method in METHODS,'Declared prediction condition only')
    value,_=registry(registry_ref,inference_ref)
    probabilities=probability_array(probabilities,10000);execution_check();guard(0)
    output=io._new_output(output_directory)
    arrays_ref=io._npz(output/'predictions.npz',dict(probabilities=probabilities,
        example_indices=np.arange(10000,dtype=np.int64)),guard)
    return io._json(output/'predictions.json',dict(schema='European_short_VAL_prediction_v1',method=method,
        parent_id=value['parent_id'],split='validation',rows=10000,class_order=list(CLASSES),
        registry_ref=registry_ref,inference_ref=inference_ref,predictions_ref=arrays_ref,
        definition_ref=definition_ref(),source_manifest_ref=source_manifest_ref,provenance=provenance,
        VAL_truth_opened=False,TEST_accessed=False,calibration_claimed=False),guard)
