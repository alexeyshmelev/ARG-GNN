"""Complete AP native6 saved-fit prediction through the plaintext executor.

Retains european_training final-fit validation, original all-observation barrier,
full-parent loader, exact sampled identities and prediction product schemas.
Only setup uses the current campaign GPU3/4/5 UUID binding and prediction calls
our static executor. Original modules are not modified or dynamically rewritten.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import os
import tempfile
from arggnn.benchmarking import european_training as original
from arggnn.benchmarking import european_treeloom as european
from arggnn.benchmarking import treeloom_training as memory
from arggnn.benchmarking import american_preparation as generation
from arggnn.benchmarking import contracts as c
from . import paper_argmix_generator as g
from .paper_argmix_european_api import predict_logits
from .paper_argmix_gpu import validate_gpu_binding,ALLOWED_PHYSICAL_GPUS
from .paper_argmix_treeloom_history import SPEED_SOURCE_REFS

SCHEMA='AP_native6_plaintext_campaign_prediction_spec_v1'
PROTOCOL='ancestralpaths_paper200mb_native6'


def requirements():
    return dict(original=original.requirements(),static_executor=True,runtime_source_generation=False,
      source_refs=[c.file_ref(__file__),c.file_ref('src/arggnn/benchmarking/paper_argmix/paper_argmix_european_api.py'),
                   c.file_ref('src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py'),
                   c.file_ref('src/arggnn/benchmarking/paper_argmix/paper_argmix_gpu.py'),
                   c.file_ref('src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_history.py'),
                   *[dict(row) for row in SPEED_SOURCE_REFS]])


def build_spec(*,physical_gpu=None,gpu_uuid_map=None,**prediction_kwargs):
    g.need(physical_gpu is None or (type(physical_gpu) is int and physical_gpu in ALLOWED_PHYSICAL_GPUS),
           'Dynamic campaign GPU or explicit physical3/4/5 required')
    if gpu_uuid_map is not None:
        g.need(set(gpu_uuid_map)==set(map(str,ALLOWED_PHYSICAL_GPUS)) and
               all(isinstance(value,str) and value.startswith('GPU-') for value in gpu_uuid_map.values()) and
               len(set(gpu_uuid_map.values()))==3,'Explicit distinct campaign GPU3/4/5 UUID map required')
    prediction_kwargs=dict(prediction_kwargs)
    prediction_kwargs.setdefault('protocol_id',PROTOCOL)
    g.need(prediction_kwargs['protocol_id']==PROTOCOL,'AP native6 protocol only')
    prediction=original.build_prediction_spec(**prediction_kwargs)
    return dict(schema=SCHEMA,prediction_spec=prediction,physical_gpu=physical_gpu,
                gpu_uuid_map=copy.deepcopy(gpu_uuid_map),gpu_uuid=None if physical_gpu is None or gpu_uuid_map is None else gpu_uuid_map[str(physical_gpu)],requirements=requirements())


def setup(spec,resource_check):
    import torch
    g.need(callable(resource_check),'Actual campaign resource callback required')
    mapping=spec['gpu_uuid_map']
    environment_map=os.environ.get('ARGGNN_GPU_UUID_MAP')
    if mapping is None:
        g.need(environment_map is not None,'Runtime immutable campaign UUID mapping required')
        mapping=json.loads(environment_map)
    elif environment_map is not None:
        g.need(json.loads(environment_map)==mapping,'Environment and pinned campaign UUID maps differ')
    g.need(set(mapping)==set(map(str,ALLOWED_PHYSICAL_GPUS)) and len(set(mapping.values()))==3,
           'Complete distinct campaign GPU3/4/5 UUID map required')
    binding=validate_gpu_binding(gpu_uuid=spec['gpu_uuid'],gpu_uuid_map=mapping)
    g.need(spec['physical_gpu'] is None or binding['physical_gpu']==spec['physical_gpu'],
           'Campaign physical GPU differs from specification')
    prediction=spec['prediction_spec']
    def check(*args,**kwargs):
        resource_check(*args,**kwargs)
        g.need(memory._current_rss_bytes()<=prediction['memory_envelope']['max_rss_bytes'],'AP prediction RSS cap exceeded')
    check('AP_plaintext_prediction_before_setup')
    out=c.contained_path(prediction['output_directory'])
    writer=generation._Writer(out,prediction['max_output_bytes'],check)
    for name in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH'):
        path=out/('.tmp' if name in ('TMPDIR','TMP','TEMP') else '.cache/'+name.lower())
        path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    tempfile.tempdir=str(out/'.tmp');os.environ['CUDA_CACHE_MAXSIZE']=str(16*2**20)
    g.need(torch.cuda.device_count()==1,'One campaign UUID must be visible as logical cuda0')
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    g.need(os.environ.get('ARGGNN_GPU_MEMORY_POLICY')=='full_device',
           'Current campaign full-device GPU memory policy required')
    # The campaign leases the complete device. No per-process allocator ceiling
    # or artificial fraction is imposed; Root still supervises actual usage.
    return torch,check,writer,binding


def run_predictions(spec,*,resource_check):
    g.need(spec['schema']==SCHEMA and spec['requirements']==requirements() and
           spec['prediction_spec']['protocol_id']==PROTOCOL,'Exact source-pinned AP plaintext specification required')
    prediction=spec['prediction_spec']
    plan=original._validate(prediction,fit=False)
    torch,check,writer,binding=setup(spec,resource_check)
    complete=original._fit_completion(prediction['fit_completion_ref'],protocol_id=PROTOCOL,
       sampling_plan_ref=prediction['sampling_plan_ref'],train_count=len(plan['train_parent_ids']))
    barrier=c.read_ref(complete['observation_barrier_ref'])
    g.need(barrier.get('sampling_plan_ref')==prediction['sampling_plan_ref'] and
           set(barrier.get('parents',{}))==set(plan['train_parent_ids']) and
           barrier.get('all_original_TRAIN_observations_sealed_before_targets') is True and
           barrier.get('private_truth_opened') is False,'All original TRAIN observations must precede saved fit')
    path=european._bound(complete['checkpoint_ref'])
    g.need(path.stat().st_size<=8*2**20,'Finite complete final checkpoint required')
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model=european.build_model(PROTOCOL)
    original._validate_saved(saved,model,protocol_id=PROTOCOL,sampling_plan_ref=prediction['sampling_plan_ref'],
       updates=30*len(plan['train_parent_ids']),barrier_ref=complete['observation_barrier_ref'])
    del saved
    model=model.to(device='cuda:0',dtype=torch.float32)
    cache,predictions=OrderedDict(),{}
    load=original._observed_loader(prediction,cache,check,fit=False)
    for parent in plan['test_parent_ids']:
        if parent not in prediction['heldout_inputs']:
            continue
        bundle,metadata=load(parent)
        memory._admit_parent_headroom(cache,memory_envelope=prediction['memory_envelope'],stage='prediction',
            rss_bytes=memory._current_rss_bytes,resource_check=check,event=lambda *args:None,keep_parent=parent)
        logits=predict_logits(model,bundle,metadata,resource_check=check,gpu_uuid=binding['gpu_uuid'],gpu_uuid_map=spec['gpu_uuid_map'])
        n,classes=bundle.focal_hold.numel(),len(model.output_classes)
        g.need(logits.shape==(n,classes) and logits.dtype==torch.float32 and torch.isfinite(logits).all().item(),
               'Every original registered held-out logit required')
        probabilities=logits.softmax(-1).to('cpu').numpy()
        item=prediction['heldout_inputs'][parent]
        records=european._sampled_records(item['sampled_registry_ref'],observation_ref=item['inference_ref'],parent_id=parent,split='test',
                                         max_metadata_bytes=prediction['max_metadata_bytes'])
        prediction_ref=writer.npz('predictions/'+parent+'.npz',example_indices=records['record_ids'],probabilities=probabilities)
        product=dict(parent_id=parent,split='test',registry_ref=item['sampled_registry_ref'],predictions_ref=prediction_ref,rows=n,
           native_path_names=list(model.output_classes),schema='european_treeloom_prediction_product_v1',method='treeloom',
           **item,checkpoint_ref=complete['checkpoint_ref'],fit_completion_ref=prediction['fit_completion_ref'],
           focal_identity_sha256=bundle.focal_identity_sha256,query_ancestry_truth_opened=False,
           all_native_trees_nodes_sites_samples_retained=True,static_executor_source_ref=c.file_ref('src/arggnn/benchmarking/paper_argmix/paper_argmix_treeloom_executor.py'),
           campaign_gpu_binding=binding)
        predictions[parent]=writer.json('products/'+parent+'.json',product)
        del logits,probabilities,records,bundle,metadata
        check('AP_plaintext_heldout_prediction_sealed')
    torch.cuda.synchronize(0)
    result=dict(schema='european_treeloom_prediction_completion_v1',status='completed_all_requested_sampled_predictions',
       protocol_id=PROTOCOL,sampling_plan_ref=prediction['sampling_plan_ref'],fit_completion_ref=prediction['fit_completion_ref'],
       predictions=predictions,query_ancestry_truth_opened=False,actual_worker_wait_and_owned_cleanup_still_required=True,
       campaign_gpu_binding=binding,static_executor=True)
    ref=writer.json('predictions_completed.json',result)
    g.need(not writer.pending and set(predictions)==set(prediction['heldout_inputs']),'Every requested AP parent must finish')
    return dict(result,completion_ref=ref)


def prediction_factory(**kwargs):
    spec=build_spec(**kwargs)
    return dict(module='arggnn.benchmarking.paper_argmix.paper_argmix_ap_prediction',function='run_predictions',
                spec=spec,python='.runtime/env/bin/python',source_refs=spec['requirements']['source_refs'])
