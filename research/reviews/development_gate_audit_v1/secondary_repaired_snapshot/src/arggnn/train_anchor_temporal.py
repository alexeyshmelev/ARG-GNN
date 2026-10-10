"""Dedicated train/validation runner for same-source fixed-anchor comparators."""
from __future__ import annotations
from .resources import ROOT,project_path,require_storage,set_research_title
set_research_title()
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from .anchor_kernel import prepare_anchor_kernel_bank
from .anchor_temporal_baselines import AnchorGRUModel,AnchorTemporalTransformerModel,ObservedReferenceCoordinateSmoother
from .boundary_dataset import load_boundary_dataset
from .confirmatory_metrics import logit_metrics
from .train import set_seed,dump_json
from .train_anchor import require_device
from .train_study import sha


def checked_temporal_protocol(path):
    settings=json.loads(Path(path).read_text())
    if settings.get('splits',{}).get('test',0)!=0 or settings.get('selection')!='minimum mean parent-validation uncalibrated logit NLL':
        raise RuntimeError('Require train/validation-only raw-logit parent-NLL selection')
    for key in ('boundary_dataset_manifest_sha256','boundary_preparation_sha256','source_dataset_manifest_sha256'):
        value=settings.get(key)
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('Frozen source/boundary hashes required')
    if settings.get('source_rate_weight_decay')!=0:
        raise RuntimeError('Source coupling and mixture weight decay must be zero')
    for key in ('hidden_dim','num_layers','latent_dim','num_anchors','num_scales','num_classes','epochs','patience','max_attention_score_bytes'):
        value=settings.get(key)
        if type(value) is not int or value<1: raise RuntimeError('Require positive frozen dimensions/budgets')
    for key in ('learning_rate','weight_decay'):
        value=settings.get(key)
        if not isinstance(value,(int,float)) or not np.isfinite(value) or (value<=0 if key=='learning_rate' else value<0):
            raise RuntimeError('Invalid frozen optimizer setting')
    models,specs=settings.get('models'),settings.get('model_specs')
    if isinstance(models, list) and any(not isinstance(name, str) or
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None for name in models):
        raise RuntimeError("Model IDs must be safe file-name components")
    if not isinstance(models,list) or not models or len(set(models))!=len(models) or not isinstance(specs,dict) or set(models)!=set(specs):
        raise RuntimeError('Require distinct frozen model IDs/specifications')
    for spec in specs.values():
        family=spec.get('family')
        keys={'family','temporal_depth','dropout','transformer_heads'} if family in ('gru','transformer') else {'family','bandwidth_bp','pseudocount'}
        if set(spec)!=keys or family not in ('gru','transformer','reference_smoother'):
            raise RuntimeError('Unsupported comparator specification')
        if family in ('gru','transformer'):
            if type(spec['temporal_depth']) is not int or spec['temporal_depth']<1 or not isinstance(spec['dropout'],(int,float)) or not 0<=spec['dropout']<1:
                raise RuntimeError('Invalid temporal depth/dropout')
    if settings.get('num_anchors')!=32:
        raise RuntimeError('This fixed pilot comparator protocol requires32 physical anchors')
    return settings


def make_temporal_model(name,input_dim,settings):
    spec=dict(settings['model_specs'][name]); family=spec.pop('family')
    if family=='reference_smoother':
        return ObservedReferenceCoordinateSmoother(num_classes=settings['num_classes'],num_anchors=settings['num_anchors'],**spec)
    constructor=AnchorGRUModel if family=='gru' else AnchorTemporalTransformerModel
    return constructor(input_dim,settings['num_classes'],hidden_dim=settings['hidden_dim'],
        num_layers=settings['num_layers'],latent_dim=settings['latent_dim'],num_scales=settings['num_scales'],
        num_anchors=settings['num_anchors'],initial_coupling=settings['initial_coupling'],
        max_attention_score_bytes=settings['max_attention_score_bytes'],**spec)


def optimizer_groups(model,weight_decay):
    rate=list(model.rate_and_mixture_parameters()); ids={id(p) for p in rate}
    if len(ids)!=len(rate): raise RuntimeError('Duplicate source-rate parameter group')
    return [{'params':[p for p in model.parameters() if id(p) not in ids],'weight_decay':weight_decay},
            {'params':rate,'weight_decay':0.}]


def evaluate_temporal(model,cases,device_banks):
    """Join external validation targets only AFTER independent model forward."""
    model.eval(); records=[]; parent_logits=[]
    with torch.no_grad():
        for case in cases:
            output=None; seen=torch.zeros(len(case.positions),dtype=torch.bool)
            for tile in case.tiles:
                logits=model(device_banks[id(tile)]).detach().cpu()
                indexes=tile.parent_position_indices
                if indexes.dtype!=torch.long or indexes.ndim!=1 or len(indexes)!=len(logits) or torch.unique(indexes).numel()!=len(indexes) or bool(((indexes<0)|(indexes>=len(case.positions))).any()):
                    raise RuntimeError('Invalid parent target-coordinate correspondence')
                if not torch.equal(tile.bank.positions,case.positions[indexes]) or not torch.equal(tile.bank.query_node_ids,case.query_node_ids) or bool(seen[indexes].any()):
                    raise RuntimeError('Tile predictions fail the fixed parent coordinate/query contract')
                if output is None: output=logits.new_empty((len(case.positions),len(case.query_node_ids),logits.shape[-1]))
                output[indexes]=logits; seen[indexes]=True
            if output is None or not bool(seen.all()) or output.shape[:2]!=case.labels.shape:
                raise RuntimeError('Incomplete parent predictions or external-label shape mismatch')
            metrics=logit_metrics(output.double().numpy(),case.labels.numpy())
            records.append({'parent_id':case.parent_id,'replicate':case.metadata['replicate'],**metrics})
            parent_logits.append(output)
    return records,parent_logits


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--data')
    parser.add_argument('--output');parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--models',nargs='+');parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args()
    config_path=project_path(args.config)
    config_bytes=config_path.read_bytes()
    config_sha=hashlib.sha256(config_bytes).hexdigest()
    settings=checked_temporal_protocol(config_path)
    if sha(config_path)!=config_sha:
        raise RuntimeError('Frozen configuration changed while being read')
    if args.validate_config:
        for name in settings['models']:
            model=make_temporal_model(name,47,settings)
            print(json.dumps({'event':'configuration_valid','model':name,'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
        return
    if not args.data or not args.output: parser.error('--data and --output are required for a run')
    torch.set_num_threads(4);require_device(args.device)
    directory=project_path(args.data)
    dataset=load_boundary_dataset(directory,expected_manifest_sha256=settings['boundary_dataset_manifest_sha256'],
        expected_preparation_sha256=settings['boundary_preparation_sha256'],
        expected_source_manifest_sha256=settings['source_dataset_manifest_sha256'])
    cases={split:dataset.cases(split) for split in ('train','val')}
    for split in cases:
        if len(cases[split])!=settings['splits'][split] or any(case.metadata['num_classes']!=settings['num_classes'] for case in cases[split]):
            raise RuntimeError('Frozen parent/class counts differ')
    requested=args.models or settings['models']
    if len(set(requested))!=len(requested) or any(name not in settings['models'] for name in requested):
        raise RuntimeError('Choose distinct frozen comparator models')
    output=project_path(args.output)
    if output.exists(): raise RuntimeError('Choose a fresh run directory')
    require_storage(headroom_bytes=200_000_000);output.mkdir(parents=True)
    (output/'protocol.json').write_bytes(config_bytes)
    (output/'dataset_manifest.json').write_text((directory/'manifest.json').read_text())
    paths=['train_anchor_temporal.py','anchor_temporal_baselines.py','anchor_kernel.py','boundary_memory.py',
        'boundary_dataset.py','kron_interface.py','models.py','data.py','reference_encoder.py',
        'genealogy_baselines.py','simulate.py','train_study.py','train_anchor.py','confirmatory_metrics.py',
        'train.py','metrics.py','resources.py']
    snapshot={name:sha(ROOT/'src/arggnn'/name) for name in paths}
    device=torch.device(args.device)
    prepared={id(tile):prepare_anchor_kernel_bank(tile.bank,num_anchors=32,num_scales=settings['num_scales']).to(device,torch.float32)
        for group in cases.values() for case in group for tile in case.tiles}
    tiles=[tile for case in cases['train'] for tile in case.tiles]
    results={'stage':settings['stage'],'protocol_sha256':config_sha,'implementation_sha256':snapshot,
        'boundary_dataset_manifest_sha256':dataset.manifest_sha256,'training_seed':args.seed,
        'selection':settings['selection'],'evaluation_split':'validation_only','models':{},
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device)},
        'physical_gpu':os.environ.get('ARGGNN_PHYSICAL_GPU'),'interpretation':settings['interpretation']}
    for name in requested:
        set_seed(args.seed);model=make_temporal_model(name,47,settings).to(device)
        parameters=list(model.parameters());started=time.monotonic();history=[];checkpoint=None
        if parameters:
            optimizer=torch.optim.AdamW(optimizer_groups(model,settings['weight_decay']),lr=settings['learning_rate'])
            best=float('inf');stale=0;best_epoch=-1;checkpoint=output/(name+'_best.pt')
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            for epoch in range(settings['epochs']):
                model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
                for index in order:
                    tile=tiles[index];logits=model(prepared[id(tile)])
                    labels=tile.labels.to(device)
                    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite comparator loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                        raise RuntimeError('Nonfinite comparator gradient')
                    torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();losses.append(float(loss.detach()))
                validation,_=evaluate_temporal(model,cases['val'],prepared)
                nll=float(np.mean([record['nll'] for record in validation]))
                item={'epoch':epoch,'train_nll':float(np.mean(losses)),'val_nll':nll,'seconds':time.monotonic()-started}
                history.append(item);print(json.dumps({'event':'epoch','model':name,**item}),flush=True)
                if nll<best:
                    best,best_epoch,stale=nll,epoch,0
                    torch.save({'model':model.state_dict(),'epoch':epoch,'protocol_sha256':config_sha,
                                'boundary_manifest_sha256':dataset.manifest_sha256,'validation_nll':best},checkpoint)
                else:
                    stale+=1
                    if stale>=settings['patience']:break
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model'])
        else:
            best_epoch=None
        validation,logits=evaluate_temporal(model,cases['val'],prepared)
        prediction_path=output/(name+'_validation_logits.pt')
        torch.save({'logits':logits,'parent_ids':[case.parent_id for case in cases['val']],
            'positions':[case.positions for case in cases['val']],'query_node_ids':[case.query_node_ids for case in cases['val']],
            'protocol_sha256':config_sha},prediction_path)
        if sha(config_path)!=config_sha:
            raise RuntimeError('Frozen configuration changed during training')
        if snapshot!={path:sha(ROOT/'src/arggnn'/path) for path in paths}:
            raise RuntimeError('Active comparator implementation/dependencies changed')
        result={'parameters':sum(p.numel() for p in parameters),'best_epoch':best_epoch,
            'validation_nll':float(np.mean([record['nll'] for record in validation])),
            'history':history,'per_parent':validation,'seconds':time.monotonic()-started,
            'model_configuration':model.configuration_manifest(),
            'checkpoint':None if checkpoint is None else str(checkpoint),
            'checkpoint_sha256':None if checkpoint is None else sha(checkpoint),
            'validation_logits':str(prediction_path),'validation_logits_sha256':sha(prediction_path),
            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated() if parameters and device.type=='cuda' else None}
        results['models'][name]=result;dump_json(output/(name+'.json'),result);dump_json(output/'progress.json',results)
        print(json.dumps({'event':'model_complete','model':name,'validation_nll':result['validation_nll']}),flush=True)
        del model
        if parameters:del optimizer
        if device.type=='cuda':torch.cuda.empty_cache()
    dump_json(output/'results.json',results)


if __name__=='__main__':main()
