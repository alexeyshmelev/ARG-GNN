"""10 Mb typed consumers of fixed TRAIN checkpoints; no launch or truth reader."""
from types import SimpleNamespace
from . import contracts_v1 as c


def validate_fit_metadata(method,complete,definition):
    c.need(method in c.METHODS[:4],'Learned checkpoint condition')
    if method in ('cohort8_treeloom','original_treeloom'):
        cohort=method=='cohort8_treeloom'
        c.need(complete.get('schema')==('European_cohort8_TRAIN_completion_v1' if cohort else 'european_treeloom_fit_completion_v1')
            and complete.get('status')==('completed_fixed90updates' if cohort else 'completed_full_parent_sampled_fit')
            and complete.get('completed_epochs')==complete.get('selected_epoch')==30
            and complete.get('optimizer_updates')==90 and complete.get('checkpoint_selection')=='fixed_final_epoch30'
            and complete.get('TEST_truth_read') is False,'Fixed finalepoch30 and all90 TRAIN updates')
        if cohort:
            c.need(complete.get('scientific_condition')==c.TRAIN_CONDITION
                and complete.get('source_manifest_ref')==definition['cohort8_TRAIN_source_ref']
                and complete.get('resume_driver_source_manifest_ref')==definition['cohort8_resume_source_ref']
                and complete.get('class_order')==list(c.CLASSES),'Exact reviewed corrected cohort8 fit sources')
        else:c.need(complete.get('protocol_id')==c.TRAIN_PROTOCOL and complete.get('sampling_plan_ref')==c.TRAIN_PLAN_REF,
            'Original native6 TreeLoom TRAIN protocol')
    else:
        cnn=method=='historical_cnn'
        c.need(complete.get('schema')==('paper_ancestralpaths_historical_cnn_fit_v1' if cnn else 'ap_paper_native_dense_fit_v1')
            and complete.get('parameters')==(19078 if cnn else 26630)
            and complete.get('optimizer_updates')==(8335 if cnn else 7815)
            and complete.get('batch_size')==(30 if cnn else 32) and complete.get('epochs')==5
            and complete.get('checkpoint_selection')=='final_epoch5' and complete.get('validation_selection') is False
            and complete.get('training_rows')==50000 and complete.get('plan_ref')==c.TRAIN_PLAN_REF
            and complete.get('native_path_names')==list(c.CLASSES),'Unchanged fixed TRAIN CNN/Dense checkpoint')
        _,plan=c.train_contract()
        c.need(complete.get('training_parent_ids')==plan['train_parent_ids'],'Only original three TRAIN parents')
        if cnn:
            from arggnn.benchmarking.paper_ancestralpaths import CONDITION
            c.need(complete.get('condition')==CONDITION,'HistoricalCNN cannot alias releasedDense')
    return complete


def load_treeloom(method,fit_ref,acceptance_ref,source_manifest_ref):
    import torch
    from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_stage as train
    from research.efficiency_20261027.integration_candidate.european_short_development_val import model_binding_v1 as original
    from research.efficiency_20261027.integration_candidate.european_short_development_val import cohort_model_binding_v1 as cohort
    definition=c.io._read(c.definition_ref(),observed=True)
    complete=validate_fit_metadata(method,c.io._read(fit_ref,observed=True),definition)
    c.acceptance(method,fit_ref,complete['checkpoint_ref'],acceptance_ref,source_manifest_ref)
    barrier=c.io._read(complete['observation_barrier_ref'],observed=True)
    _,plan=c.train_contract()
    c.need(barrier.get('sampling_plan_ref')==c.TRAIN_PLAN_REF
        and set(barrier.get('parents',{}))==set(plan['train_parent_ids'])
        and barrier.get('all_original_TRAIN_observations_sealed_before_targets') is True
        and barrier.get('private_truth_opened') is False,'Original TRAIN observation-before-target barrier')
    path=c.io._checked(complete['checkpoint_ref'],observed=True)
    c.need(path.stat().st_size<=8*2**20,'Finite reviewed checkpoint')
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model=cohort.build_model() if method=='cohort8_treeloom' else original.ShortDevelopmentTreeLoomV1(condition='AP_short10Mb_development_native6')
    if method=='cohort8_treeloom':
        c.need(saved.get('schema')=='European_cohort8_TRAIN_checkpoint_v1'
            and saved.get('scientific_condition')==c.TRAIN_CONDITION
            and saved.get('source_manifest_ref')==definition['cohort8_TRAIN_source_ref'],
            'Corrected200Mb TRAIN weights accepted only into the reviewed10Mb cohort consumer')
        saved=dict(saved,schema='european_treeloom_full_checkpoint_v1')
    else:c.need(saved.get('scientific_condition') is None,'Original model has no cohort8 identity')
    train._validate_saved(saved,model,protocol_id=c.TRAIN_PROTOCOL,sampling_plan_ref=c.TRAIN_PLAN_REF,
        updates=90,barrier_ref=complete['observation_barrier_ref'])
    c.io._checked(complete['checkpoint_ref'],observed=True)
    return model,complete


def predict_original(model,bundle,metadata,*,resource_check):
    """Original prediction loop with only exact short-model/bundle construction."""
    import torch
    from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
    from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_execution as original
    from research.efficiency_20261027.integration_candidate.european_short_development_val import observations_v1 as obs
    from research.efficiency_20261027.integration_candidate.european_short_development_val import model_binding_v1 as binding
    from research.efficiency_20261027.integration_candidate.european_short_development_val.consumer_execution_binding_v1 import ShortDevelopmentStagedExecution
    c.need(type(model) is binding.ShortDevelopmentTreeLoomV1 and type(bundle) is obs.ShortDevelopmentWholeParent
        and bundle.split=='val_prediction' and metadata['private_truth_opened'] is False
        and metadata['parent_id']==bundle.parent_id and metadata['sampled_registry_ref']==bundle.sampled_registry_ref
        and metadata['focal_identity_sha256']==bundle.focal_identity_sha256,'Exact original short observed consumer')
    gpu_binding.validate_env();bundle.validate();model.eval();device=next(model.parameters()).device
    c.need(device.type=='cuda' and device.index in (None,0),'ROOT-admitted prediction device')
    frozen=SimpleNamespace(FrozenEuropeanTreeLoomV1=binding.ShortDevelopmentTreeLoomV1,_bank_tensors=binding._bank_tensors)
    with torch.no_grad():
        executor=ShortDevelopmentStagedExecution(model,bundle,metadata,resource_check=resource_check,frozen_module=frozen)
        context=executor.context_forward();geometry=original.geometry_from_spans(bundle.bank.spans).to(device)
        output=torch.empty((bundle.focal_hold.numel(),6),dtype=torch.float32);seen=0
        for first in range(0,context.shape[1],executor.hap_tile):
            last=min(first+executor.hap_tile,context.shape[1])
            rows=torch.nonzero((bundle.focal_sample_column>=first)&(bundle.focal_sample_column<last),as_tuple=False).flatten()
            if not len(rows):continue
            resource_check('European_full_sequence_prediction_tile')
            logits=original._tile_logits(model,context[:,first:last].to(device).contiguous(),geometry,
                bundle.focal_hold[rows].to(device),(bundle.focal_sample_column[rows]-first).to(device))
            output.index_copy_(0,rows,logits.to('cpu'));seen+=len(rows)
        c.need(seen==len(output) and bool(torch.isfinite(output).all()),'Every original sampled row receives finite logits')
        return output


def predict_treeloom(method,fit_ref,acceptance_ref,registry_ref,inference_ref,observed_vcf_ref,output_directory,*,
                     source_review_ref,guard,execution_check,resource_check,write_headroom_bytes,
                     max_sites,max_metadata_bytes,preparation_limits):
    c.need(method in ('cohort8_treeloom','original_treeloom') and callable(resource_check),'Explicit model and ROOT callback')
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    c.registry(registry_ref,inference_ref)
    from research.efficiency_20261027.integration_candidate.european_short_development_val import cohort_observations_v1 as cohort,observations_v1 as original
    from research.efficiency_20261027.integration_candidate.european_short_development_val.consumer_execution_binding_v1 import predict_validation
    from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters.european_gru import install_chunked_gru
    from research.repairs.benchmark_multigpu_20261022.scientific import gpu_binding
    adapter=cohort if method=='cohort8_treeloom' else original
    binding={} if method=='cohort8_treeloom' else {'protocol_id':c.SHORT_PROTOCOL}
    state=adapter.load_relate_observation(inference_ref,observed_vcf_ref,max_sites=max_sites,
        max_metadata_bytes=max_metadata_bytes,resource_check=resource_check,**binding)
    bundle,metadata=adapter.prepare_observation(state,parent_id=c.short.frozen_parent()['parent_id'],split='validation',
        source_refs=state['source_refs'],sampled_registry_ref=registry_ref,max_sampled_metadata_bytes=max_metadata_bytes,
        resource_check=resource_check,**preparation_limits,**binding)
    del state
    execution_check();gpu_binding.validate_env()
    model,complete=load_treeloom(method,fit_ref,acceptance_ref,source)
    model=install_chunked_gru(model.to(device='cuda:0'))
    predict=predict_validation if method=='cohort8_treeloom' else predict_original
    logits=predict(model,bundle,metadata,resource_check=resource_check)
    probabilities=logits.softmax(-1).cpu().numpy()
    return c.seal_predictions(method,probabilities,registry_ref,inference_ref,output_directory,
        source_manifest_ref=source,provenance=dict(fit_completion_ref=fit_ref,checkpoint_ref=complete['checkpoint_ref'],
        checkpoint_acceptance_ref=acceptance_ref,observed_vcf_ref=observed_vcf_ref,
        focal_identity_sha256=bundle.focal_identity_sha256,all_observed_trees_nodes_sites_samples_retained=True),
        guard=guard,execution_check=execution_check)


def predict_keras(method,fit_ref,acceptance_ref,features_ref,output_directory,*,tensorflow,
                  source_review_ref,guard,execution_check,write_headroom_bytes):
    import numpy as np
    from arggnn import ancestralpaths_native_bridge as bridge
    from .features_v1 import load_features
    c.need(method in ('historical_cnn','released_dense'),'Unchanged author prediction condition')
    source=c.ready(source_review_ref,guard,execution_check,write_headroom_bytes)
    fitted=validate_fit_metadata(method,c.io._read(fit_ref,observed=True),c.io._read(c.definition_ref(),observed=True))
    c.acceptance(method,fit_ref,fitted['checkpoint_ref'],acceptance_ref,source)
    import importlib.metadata
    c.need(tensorflow.__version__=='2.15.1' and importlib.metadata.version('keras')=='2.15.0',
        'Original qualified TensorFlow2.15.1 and Keras2.15.0 runtime')
    feature,registry,x=load_features(features_ref,source_manifest_ref=source)
    network=tensorflow.keras.models.load_model(c.io._checked(fitted['checkpoint_ref'],observed=True))
    c.need(network.count_params()==(19078 if method=='historical_cnn' else 26630)
        and network.input_shape==((None,1,5,9) if method=='historical_cnn' else (None,5,9))
        and network.output_shape==(None,6),'Exact saved CNN/Dense topology; no rebuilding or fitting')
    if method=='historical_cnn':
        probabilities=np.empty((len(x),6),dtype=np.float32)
        for start in range(0,len(x),128):
            execution_check();guard(0)
            probabilities[start:start+128]=network(x[start:start+128,None,:,:],training=False).numpy()
    else:
        execution_check();guard(0)
        probabilities=bridge.predict_native_probabilities(network,x,batch_size=128).astype(np.float32)
        execution_check();guard(0)
    c.io._checked(fitted['checkpoint_ref'],observed=True)
    return c.seal_predictions(method,probabilities,feature['registry_ref'],feature['inference_ref'],output_directory,
        source_manifest_ref=source,provenance=dict(fit_completion_ref=fit_ref,checkpoint_ref=fitted['checkpoint_ref'],
        checkpoint_acceptance_ref=acceptance_ref,features_ref=features_ref,author_inference_unchanged=True),
        guard=guard,execution_check=execution_check)
