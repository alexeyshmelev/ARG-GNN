"""Unchanged author export/inference/refinement plus exact-target output join.
Nothing launches or imports numerical packages at import. Root owns every actual job.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=Path(__file__).resolve().parent
PRIMARY_REF={'path':'research/reviews/ARGMix_prospective1611_primary_native_source_20261016_v1/primary_native_v1.py','sha256':'9e56b970d9f16b614ceebcb6175ff1e8fc12445af578d22e1ace6c6e117228de'}

def primary():
    p=ROOT/PRIMARY_REF['path']
    if hashlib.sha256(p.read_bytes()).hexdigest()!=PRIMARY_REF['sha256']:
        raise ValueError('Exact sealed primary adapter source')
    spec=importlib.util.spec_from_file_location('_ARGMix_primary_native_sealed',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def _runtime(p,qualification_record,copy_record):
    q=p.metadata(p.checked(qualification_record))
    p.need(q['source_copy_ref']==copy_record and q['actual_native_import_success'] is True
        and q['author_sources_unchanged'] is True,'Actual root-qualified untouched author native runtime')
    extension=p.checked(q['native_extension_ref'])
    p.need(p.path(p.metadata(p.checked(copy_record))['source_copy']) in extension.parents,
           'Qualified extension inside same untouched owned author source copy')
    return q,extension

def prepare_inference(native_job_record,parent_plan_record,checkpoint_record,runtime_qualification_record,output,*,guard):
    """Write exact author stage argv and counted scope. No export/inference or label read."""
    p=primary();p.contract();job=p.metadata(p.checked(native_job_record));parent=p.metadata(p.checked(parent_plan_record))
    p.need(job['schema']=='ARGMix_prospective1611_primary_native_job_plan_v1' and
        parent['schema']=='ARGMix_prospective1611_primary_observed_parent_v1','Exact native primary source plans')
    p.need(job['source_ref']==parent['source_ref']==p.ref(ROOT/PRIMARY_REF['path']),'Exact sealed primary source')
    source,copy=p._copy(job['untouched_copy_ref'])
    _runtime(p,runtime_qualification_record,job['untouched_copy_ref'])
    checkpoint=p.checked(checkpoint_record)
    native_job_dir=p.path(native_job_record['path']).parent
    p.need(native_job_dir/'checkpoints' in checkpoint.parents and checkpoint.name=='best_model.pt',
        'Actual selected unchanged author checkpoint in the corresponding native job output')
    config_path=p.checked(job['backend_config_ref'])
    p.need(config_path.stat().st_size<=8*2**20,'Bounded native backend config')
    original_cfg=json.loads(config_path.read_text())
    p.need(original_cfg['processing']['exclude_edge_trees'] is True and
        original_cfg['processing']['subgraph_merge']['enabled'] is False,'Native default support without merging')
    output=p.fresh(output,guard)
    for name in ['artifacts','raw','joint']:(output/name).mkdir()
    pairs=[dict(rel=str(p.checked(parent['native_tree_ref'])),id=parent['parent_id'],
                references=str(p.checked(parent['references_ref'])))]
    pair_ref=p.write_bytes(output/'pairs.yaml',(json.dumps(pairs,indent=2)+'\n').encode(),guard)
    cfg=json.loads(json.dumps(original_cfg))
    cfg['paths']=dict(references=str(p.checked(parent['references_ref'])),val_pairs=str(p.checked(pair_ref)))
    cfg['relate']['recomb_map']=str(p.checked(parent['native_map_ref']))
    cfg['runtime']['load_checkpoint']=str(checkpoint)
    cfg['inference'].update(device='cuda',max_count=0,max_predict_haplotypes=0)
    cfg['inference']['h5_features'].update(enabled=True,source='final_layer',include_probs=False,
        write_pred_conf=True,final_layer_dim=16)
    p.need('infer_min_bp' not in cfg['processing'] and 'infer_max_bp' not in cfg['processing'],
        'No silently restricted native context')
    cfg_ref=p.write_bytes(output/'backend_config.yaml',(json.dumps(cfg,indent=2)+'\n').encode(),guard)
    reference_pops=cfg['populations']['reference'];predict_pops=cfg['populations']['predict']
    p.need([cfg['populations']['class_labels'][str(i)] for i in range(4)]==p.CLASSES,
        'Exact native checkpoint/config output class order')
    python=str(ROOT/p.contract()['native_python'])
    ts_model=output/'artifacts/model.ts.pt'
    raw_h5=output/'raw/inference_raw.h5';joint_h5=output/'joint/lai.h5'
    commands=[
        [str(source/'src/export_torchscript.py'),'--model_checkpoint',str(checkpoint),
         '--config',str(p.checked(cfg_ref)),'--output',str(ts_model),'--export_mode','logits_and_final',
         '--ref_pops',*reference_pops,'--predict_pops',*predict_pops,'--device','cpu'],
        [str(source/'src/run_inference.py'),'--config',str(p.checked(cfg_ref)),
         '--val_pairs',str(p.checked(pair_ref)),'--torchscript_model',str(ts_model),
         '--vocab_file',str(ts_model).replace('.pt','_vocab.json'),
         '--checkpoint',str(checkpoint),'--output_dir',str(output/'raw'),'--output_name','inference_raw',
         '--write_tsv','false','--probs_h5',str(raw_h5),'--h5_feature_source','final_layer',
         '--h5_include_probs','false','--h5_write_pred_conf','true','--device','cuda'],
        [str(source/'src/run_joint_inference.py'),'--config',str(p.checked(cfg_ref)),
         '--checkpoint',str(checkpoint),'--input_h5',str(raw_h5),'--output_dir',str(output/'joint'),
         '--output_prefix','lai','--device','cuda']]
    stages=[]
    for stage,argv in zip(['export','inference','refinement'],commands):
        request=dict(schema='ARGMix_prospective1611_primary_native_stage_v1',stage=stage,
            source_ref=p.ref(__file__),primary_source_ref=PRIMARY_REF,source_copy_ref=job['untouched_copy_ref'],
            native_runtime_qualification_ref=runtime_qualification_record,checkpoint_ref=checkpoint_record,
            argv=argv,CUDA_VISIBLE_DEVICES='' if stage=='export' else '0')
        r=p.save_json(output/(stage+'_request.json'),request,guard)
        stages.append(dict(stage=stage,request_ref=r,
            argv=[python,str(HERE/'native_stage_launcher_v1.py'),str(p.checked(r))],
            env=dict(CUDA_VISIBLE_DEVICES=request['CUDA_VISIBLE_DEVICES'],
                TMPDIR=str(p.path(copy['source_copy'])/'owned_build/tmp'))))
    native_rows=parent['query_count']*max(0,parent['native_tree_count']-2)
    # Native HDF5 payload: ids/intervals/label/conf32 + refs180 + final_layer32 =244.
    raw_payload=244*native_rows;joint_payload=268*native_rows
    value=dict(schema='ARGMix_prospective1611_primary_native_inference_plan_v1',source_ref=p.ref(__file__),
        primary_source_ref=PRIMARY_REF,native_job_ref=native_job_record,parent_plan_ref=parent_plan_record,
        checkpoint_ref=checkpoint_record,source_copy_ref=job['untouched_copy_ref'],
        native_runtime_qualification_ref=runtime_qualification_record,config_ref=cfg_ref,pairs_ref=pair_ref,
        stages=stages,expected_raw_h5=str(raw_h5.relative_to(ROOT)),expected_joint_h5=str(joint_h5.relative_to(ROOT)),
        class_order=p.CLASSES,expected_full_native_rows=native_rows,
        capacity=dict(raw_H5_array_payload_bytes=raw_payload,joint_H5_array_payload_bytes=joint_payload,
            conservative_two_file_scope_ceiling_bytes=2*(raw_payload+joint_payload)+256*2**20,
            final_target_seal_ceiling_bytes=8*2**20,
            native_joint_dense_array_payload_bytes=64*native_rows,
            dense_array_estimate_excludes_Python_group_buffers_and_allocator_overhead=True,
            root_existing_guard_and_actual_supervisor_required=True),
        native_default_postprocessing='unchanged author run_joint_refinement_on_hdf5, including its own overlap/tail handling',
        native_features='author final_layer HDF5 fp16_bits representation and FP32 native BiGRU',
        native_reference_order='training native RustHashMap, inference unmodified Python metadata insertion order',
        focal_diploid_exclusion=False,native_inference_nearest_drop=0,
        outer_infer_CLI_bypass_reason='outer CLI requires all donor population categories in output source labels, unsuitable for NF/Yam dual-role panel',
        externally_supplied_config_differences=['six observed donor category labels and four output ancestry labels',
            'four observed query cohort labels','native raw-tree/map/sample observed external panel paths'],
        decoded_outputs_modified=False,ancestry_truth_or_labels_read=False,
        numerical_export_inference_refinement_pending=True)
    p.save_json(output/'inference_plan.json',value,guard);return value

def launch_native_stage(request_path):
    """Root invokes in a separate process. Executes exact untouched author main."""
    p=primary();request=p.metadata(p.path(request_path))
    p.need(request['schema']=='ARGMix_prospective1611_primary_native_stage_v1' and
        request['source_ref']==p.ref(__file__) and request['primary_source_ref']==PRIMARY_REF,'Exact prepared native stage')
    p.need(os.environ.get('CUDA_VISIBLE_DEVICES')==request['CUDA_VISIBLE_DEVICES'],'Prepared physical GPU scope')
    source,_=p._copy(request['source_copy_ref'])
    _,extension=_runtime(p,request['native_runtime_qualification_ref'],request['source_copy_ref'])
    p.checked(request['checkpoint_ref'])
    script=p.path(request['argv'][0])
    expected=dict(export='export_torchscript.py',inference='run_inference.py',refinement='run_joint_inference.py')
    p.need(script==source/'src'/expected[request['stage']],'Exact unmodified author entry point')
    from setproctitle import setproctitle
    setproctitle('AI Auto Research Pipeline Test')
    import sys
    import torch
    p.need('ancpath' not in sys.modules,'Fresh process for exact qualified native extension')
    spec=importlib.util.spec_from_file_location('ancpath',extension)
    native=importlib.util.module_from_spec(spec);sys.modules['ancpath']=native;spec.loader.exec_module(native)
    p.need(hasattr(native,'ShardPoolIterator'),'Current qualified author Torch inference extension')
    sys.path.insert(0,str(source/'src'));sys.argv=request['argv']
    import runpy
    runpy.run_path(str(script),run_name='__main__')

def seal_native_targets(inference_plan_record,terminal_record,output,*,guard,resource_check):
    """Read native final predictions, copy exact supported rows. No decoder or label access."""
    p=primary();plan=p.metadata(p.checked(inference_plan_record));terminal=p.metadata(p.checked(terminal_record))
    p.need(plan['schema']=='ARGMix_prospective1611_primary_native_inference_plan_v1'
        and plan['source_ref']==p.ref(__file__),'Exact source-bound native inference plan')
    p.need(terminal['inference_plan_ref']==inference_plan_record and terminal['actual_wait_consumed'] is True
        and terminal['returncodes']==[0,0,0] and
        terminal['stage_request_refs']==[r['request_ref'] for r in plan['stages']],
        'Actual root success waits for export, native inference and native refinement')
    parent=p.metadata(p.checked(plan['parent_plan_ref']))
    support=p.metadata(p.checked(parent['original_shared_support_ref']));targets=p._target_rows(support)
    import numpy as np
    import h5py
    import torch
    import tskit
    checkpoint=torch.load(p.checked(plan['checkpoint_ref']),map_location='cpu',weights_only=False)
    classes={int(k):v for k,v in checkpoint['joint_model']['class_labels'].items()}
    p.need([classes[i] for i in range(4)]==plan['class_order']==p.CLASSES,'Native checkpoint output order, never sorted donor categories')
    mask=np.fromfile(p.checked(parent['native_supported_ref']),dtype=np.uint8)
    target_trees=np.fromfile(p.checked(parent['native_target_index_ref']),dtype='<i8')
    p.need(mask.shape==(100000,) and target_trees.shape==(100000,) and np.all(mask<=1),'Exact observed-only support artifacts')
    supported=np.flatnonzero(mask);p.need(len(supported)==parent['common_support']['supported_targets'],'Frozen coverage count')
    nodes=np.asarray([int(t['sample_index']) for t in targets],dtype=np.int64)
    keys=target_trees[supported]*3222+nodes[supported];order=np.argsort(keys)
    sorted_keys=keys[order];p.need(len(np.unique(sorted_keys))==len(keys),'Unique exact supported native targets')
    source=tskit.load(p.checked(parent['native_tree_ref']))
    starts=np.asarray([t.interval.left for t in source.trees()])
    ends=np.asarray([t.interval.right for t in source.trees()])
    rows=np.zeros((len(supported),4),dtype=np.float32)
    labels=np.full(len(supported),-1,dtype=np.int32);conf=np.zeros(len(supported),dtype=np.float32)
    seen=np.zeros(len(supported),dtype=np.bool_)
    raw=p.path(plan['expected_raw_h5']);joint=p.path(plan['expected_joint_h5'])
    p.need(raw.is_file() and joint.is_file() and not raw.is_symlink() and not joint.is_symlink(),'Native output files')
    with h5py.File(raw,'r') as original,h5py.File(joint,'r') as native:
        n=plan['expected_full_native_rows']
        p.need(native['node_id'].shape==(n,) and original['node_id'].shape==(n,)
            and native['refined_probs'].shape==(n,4) and native['refined_label'].shape==(n,)
            and native['refined_conf'].shape==(n,),'Complete full native inference outputs')
        for first in range(0,n,131072):
            resource_check('before_native_prediction_target_join')
            end=min(n,first+131072)
            for name in ['shard_id','node_id','tree_start','tree_end']:
                p.need(np.array_equal(original[name][first:end],native[name][first:end]),'Native postprocessor preserved raw row identity')
            shard=np.asarray(native['shard_id'][first:end]);node=np.asarray(native['node_id'][first:end],dtype=np.int64)
            left=np.asarray(native['tree_start'][first:end]);right=np.asarray(native['tree_end'][first:end])
            ti=np.searchsorted(starts,left,side='left')
            p.need(np.all(shard==0) and np.all((node>=0)&(node<3222))
                and np.all((ti>0)&(ti<len(starts)-1)),'Single native shard, observed sample IDs and default edge support')
            p.need(np.all(starts[ti]==left) and np.all(ends[ti]==right),'Exact native tree intervals, no custom projection')
            search_keys=ti*3222+node;where=np.searchsorted(sorted_keys,search_keys)
            in_range=where<len(sorted_keys);hit=np.flatnonzero(in_range)
            hit=hit[sorted_keys[where[hit]]==search_keys[hit]]
            if len(hit):
                destination=order[where[hit]]
                p.need(len(np.unique(destination))==len(destination) and not np.any(seen[destination]),'No repeated native target output')
                # Read exact author values. Never re-normalize, smooth, argmax or re-run GRU.
                prob=np.asarray(native['refined_probs'][first:end],dtype=np.float32)[hit]
                lab=np.asarray(native['refined_label'][first:end],dtype=np.int32)[hit]
                confidence=np.asarray(native['refined_conf'][first:end],dtype=np.float32)[hit]
                p.need(np.all(np.isfinite(prob)) and np.all(prob>=0) and
                    np.allclose(prob.sum(1),1,rtol=0,atol=1e-5) and np.all((lab>=0)&(lab<4))
                    and np.all(np.isfinite(confidence)),'Finite native predictions')
                rows[destination]=prob;labels[destination]=lab;conf[destination]=confidence;seen[destination]=True
    p.need(np.all(seen),'Every predeclared supported target has one unmodified native output')
    output=p.fresh(output,guard)
    probs_ref=p.write_bytes(output/'probabilities.f4',rows.astype('<f4').tobytes(),guard)
    labels_ref=p.write_bytes(output/'native_predicted_classes.i4',labels.astype('<i4').tobytes(),guard)
    confidence_ref=p.write_bytes(output/'native_confidence.f4',conf.astype('<f4').tobytes(),guard)
    target_rows_ref=p.write_bytes(output/'original_target_rows.u4',supported.astype('<u4').tobytes(),guard)
    result=dict(schema='ARGMix_prospective1611_primary_native_prediction_seal_v1',source_ref=p.ref(__file__),
        inference_plan_ref=inference_plan_record,actual_terminal_ref=terminal_record,parent_id=parent['parent_id'],split=parent['split'],
        native_raw_H5_ref=p.ref(raw),native_joint_H5_ref=p.ref(joint),checkpoint_ref=plan['checkpoint_ref'],
        original_shared_support_ref=parent['original_shared_support_ref'],native_support_mask_ref=parent['native_supported_ref'],
        original_target_rows_ref=target_rows_ref,probabilities_ref=probs_ref,native_predicted_classes_ref=labels_ref,
        native_confidence_ref=confidence_ref,shape=[len(supported),4],class_order=p.CLASSES,
        coverage=parent['common_support'],unsupported_targets_imputed=False,original100k_sensitivity_plan_replaced=False,
        probabilities_and_native_hard_calls_copied_without_modification=True,
        native_postprocessor='original author run_joint_inference.py, full emitted sequence and native overlap/tail handling',
        ancestry_truth_or_labels_read=False,optimizer_steps=0,scientific_accuracy_claimed=False)
    p.save_json(output/'prediction_seal.json',result,guard);return result
