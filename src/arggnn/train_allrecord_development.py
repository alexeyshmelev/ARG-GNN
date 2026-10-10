"""Fresh64/16/TEST0 all-record temporal-control runner; no preparation jobs."""
from __future__ import annotations
from .development_execution import configure_deterministic_development, DETERMINISTIC_POLICY, deterministic_development_manifest
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from .allrecord_boundary_temporal import AllRecordBoundaryGRU,AllRecordBoundaryTransformer
from .allrecord_development_bindings import (ROOT,SEEDS,RAW_NLL,capture_settings,
    admit_captured_run,check_live,development_path,implementation_hashes)
from .confirmatory_metrics import logit_metrics
from .resources import require_storage,set_research_title
from .train import set_seed
from .train_anchor_development import require_device

set_research_title()
_IMPORTED_PINS=implementation_hashes()


def factory(name,settings):
    if name not in settings['models']:raise RuntimeError('Model outside frozen all-record grid')
    spec=dict(settings['model_specs'][name]);family=spec.pop('family')
    cls=AllRecordBoundaryGRU if family=='allrecord_gru' else AllRecordBoundaryTransformer
    if family not in ('allrecord_gru','allrecord_transformer'):raise RuntimeError('Unsupported all-record family')
    return cls(47,4,max_attention_score_bytes=settings['max_attention_score_bytes'],**spec)


def write_json(path,value,*,exclusive=False):
    content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb' if exclusive else 'wb') as handle:handle.write(content)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def checkpoint_directory(output,config_sha,seed):
    identity=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    return development_path(Path('checkpoints/allrecord_development_v1')/f'{output.name}_{identity}_{config_sha[:12]}_s{seed}')


def evaluate_allrecord(model,cases,device_banks):
    """Join separately stored labels only after source-only model prediction."""
    model.eval();records=[];parent_logits=[]
    with torch.no_grad():
        for case in cases:
            output=None;seen=torch.zeros(len(case.positions),dtype=torch.bool)
            for tile in case.tiles:
                logits=model(device_banks[id(tile)]).detach().cpu().double()
                indexes=tile.parent_position_indices
                if indexes.dtype!=torch.long or indexes.ndim!=1 or len(indexes)!=len(logits) or torch.unique(indexes).numel()!=len(indexes) or bool(((indexes<0)|(indexes>=len(case.positions))).any()):
                    raise RuntimeError('Invalid parent readout-coordinate correspondence')
                if not torch.equal(tile.bank.positions,case.positions[indexes]) or not torch.equal(tile.bank.query_node_ids,case.query_node_ids) or bool(seen[indexes].any()):
                    raise RuntimeError('Tile/parent fixed-coordinate/query correspondence changed')
                if output is None:output=logits.new_empty((len(case.positions),len(case.query_node_ids),4))
                if logits.shape!=(len(indexes),len(case.query_node_ids),4) or not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Invalid finite all-record logits')
                output[indexes]=logits;seen[indexes]=True
            if output is None or not bool(seen.all()) or output.shape[:2]!=case.labels.shape:
                raise RuntimeError('Incomplete parent logits or external-label shape mismatch')
            metrics=logit_metrics(output.numpy(),case.labels.numpy())
            records.append({'parent_id':case.parent_id,'replicate':case.metadata['replicate'],**metrics})
            parent_logits.append(output)
    return records,parent_logits


def parent_mean(records):
    if not records or any(not math.isfinite(row['nll']) or row['nll']<0 for row in records):
        raise RuntimeError('Finite equal-parent raw-logit validation scores required')
    return math.fsum(row['nll'] for row in records)/len(records)


def train_configuration(model,settings,name,seed,cases,device_banks,checkpoint,binding,*,integrity_check,callback=None):
    """Fixed48/patience8 selected checkpoint; labels never enter the model."""
    deterministic_development_manifest()
    parameters=list(model.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=settings['model_learning_rates'][name],weight_decay=.0001)
    tiles=[tile for case in cases['train'] for tile in case.tiles]
    history=[];best=float('inf');best_epoch=None;stale=0;started=time.monotonic()
    for epoch in range(48):
        deterministic_development_manifest()
        integrity_check(full=False)
        model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
        for index in order:
            tile=tiles[index];logits=model(device_banks[id(tile)])
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),tile.labels.to(logits.device).flatten())
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite all-record training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):raise RuntimeError('Nonfinite all-record gradient')
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            losses.append(float(loss.detach()))
        validation,_=evaluate_allrecord(model,cases['val'],device_banks);nll=parent_mean(validation)
        history.append({'epoch':epoch,'train_nll':math.fsum(losses)/len(losses),
            'val_nll':nll,'elapsed_seconds':time.monotonic()-started})
        if callback:callback(history[-1])
        deterministic_development_manifest()
        integrity_check(full=False)
        if nll<best:
            best,best_epoch,stale=nll,epoch,0
            torch.save({'model':model.state_dict(),'epoch':epoch,'seed':seed,'validation_nll':best,
                'configuration_id':name,**binding},checkpoint)
        else:
            stale+=1
            if stale>=8:break
    if best_epoch is None:raise RuntimeError('No finite selected checkpoint')
    deterministic_development_manifest()
    integrity_check(full=True)
    saved=torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=True)
    if saved.get('seed')!=seed or saved.get('configuration_id')!=name or any(saved.get(key)!=value for key,value in binding.items()) or saved.get('epoch')!=best_epoch or saved.get('validation_nll')!=best:
        raise RuntimeError('Selected checkpoint source/config/seed/dependency binding differs')
    model.load_state_dict(saved['model'])
    validation,logits=evaluate_allrecord(model,cases['val'],device_banks)
    deterministic_development_manifest()
    integrity_check(full=True)
    selected=parent_mean(validation)
    if selected!=best:raise RuntimeError('Reloaded selected checkpoint validation NLL differs')
    result={'parameters':sum(p.numel() for p in parameters),'best_epoch':best_epoch,
        'learning_rate':settings['model_learning_rates'][name],'validation_nll':selected,
        'original_selected_validation_nll':best,'reloaded_validation_nll':selected,
        'checkpoint_replay_nll_delta':selected-best,
        'per_parent':validation,'history':history,'training_seconds':time.monotonic()-started,
        'model_configuration':model.configuration_manifest(),'checkpoint':str(checkpoint),
        'checkpoint_sha256':file_sha(checkpoint),'smoother_fitting':None}
    return result,logits


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--output');parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--models',nargs='+');parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    captured=capture_settings(args.config,args.expected_config_sha256);settings=captured.values
    if args.seed not in SEEDS:raise RuntimeError('Seed outside fixed101/202/303 development matrix')
    names=args.models or settings['models']
    if len(names)!=len(set(names)) or any(name not in settings['models'] for name in names):raise RuntimeError('Select distinct models from frozen eight-configuration family')
    initial=settings['bound_implementation_sha256']
    if initial!=_IMPORTED_PINS or initial!=implementation_hashes():raise RuntimeError('Bound/imported all-record implementation hashes differ')
    if not args.validate_config and not args.output:parser.error('--output is required for fitting')
    completion,dataset,ages,lock=admit_captured_run(captured,imported_pins=_IMPORTED_PINS)
    del completion,ages
    execution_policy = configure_deterministic_development()
    if not args.validate_config:require_device(args.device)
    if args.validate_config:
        for name in names:
            net=factory(name,settings)
            print(json.dumps({'event':'bound_allrecord_config_valid','model':name,
                'config_sha256':captured.sha256,'parameters':sum(p.numel() for p in net.parameters()),
                'input_scope':settings['input_scope'],'test_access':False}),flush=True)
        check_live(settings,captured,lock,full=True)
        return
    output=development_path(args.output)
    if output.exists():raise RuntimeError('Fresh all-record run directory required')
    checkpoints=checkpoint_directory(output,captured.sha256,args.seed)
    if checkpoints.exists():raise RuntimeError('Fresh all-record checkpoint namespace required')
    require_storage(headroom_bytes=200_000_000)
    output.mkdir(parents=True);checkpoints.mkdir(parents=True)
    with (output/'protocol.json').open('xb') as handle:handle.write(captured.content)
    cases={split:dataset.cases(split) for split in ('train','val')}
    write_json(output/'dataset_binding.json',{'completion_receipt_sha256':settings['completion_receipt_sha256'],
        'source_dataset_manifest_sha256':settings['source_dataset_manifest_sha256'],
        'boundary_manifest_sha256':dataset.manifest_sha256,'boundary_preparation_sha256':dataset.preparation_sha256,
        'parents':{split:[case.parent_id for case in group] for split,group in cases.items()},
        'source_snapshot':lock.manifest(),'test_parents':0},exclusive=True)
    device=torch.device(args.device)
    binding={'protocol_sha256':captured.sha256,'completion_receipt_sha256':settings['completion_receipt_sha256'],
        'boundary_manifest_sha256':settings['boundary_dataset_manifest_sha256'],'implementation_sha256':initial}
    results={'deterministic_execution_policy':execution_policy,'stage':settings['stage'],'family':settings['family'],'development_namespace':'development-v1',
        'evaluation_split':'fresh_development_validation_only','training_seeds':SEEDS,'training_seed':args.seed,
        'protocol_sha256':captured.sha256,'completion_receipt_sha256':settings['completion_receipt_sha256'],
        'boundary_dataset_manifest_sha256':settings['boundary_dataset_manifest_sha256'],
        'initial_implementation_sha256':initial,'nll_policy':RAW_NLL,'selection':settings['selection'],
        'input_scope':settings['input_scope'],'cross_tree_aggregation':settings['cross_tree_aggregation'],
        'record_subdivision_invariant':False,'models':{},'physical_gpu':os.environ.get('ARGGNN_PHYSICAL_GPU'),
        'interpretation':settings['interpretation'],'test_access':False,
        'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device)}}
    active=None
    def integrity_check(*,full=False):
        require_storage(headroom_bytes=100_000_000)
        return check_live(settings,captured,lock,full=full)
    try:
        device_banks={id(tile):tile.bank.to(device,torch.float32) for group in cases.values() for case in group for tile in case.tiles}
        for name in names:
            active=name;integrity_check(full=True);set_seed(args.seed)
            model=factory(name,settings).to(device)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            def callback(row):print(json.dumps({'event':'epoch','model':name,'seed':args.seed,**row}),flush=True)
            checkpoint=checkpoints/(name+'.pt')
            result,logits=train_configuration(model,settings,name,args.seed,cases,device_banks,checkpoint,binding,
                integrity_check=integrity_check,callback=callback)
            prediction=checkpoints/(name+'_validation_logits.pt')
            torch.save({'logits':logits,'parent_ids':[case.parent_id for case in cases['val']],
                'positions':[case.positions for case in cases['val']],
                'query_node_ids':[case.query_node_ids for case in cases['val']],
                'protocol_sha256':captured.sha256,'seed':args.seed,'nll_policy':RAW_NLL,
                'completion_receipt_sha256':settings['completion_receipt_sha256']},prediction)
            ending=integrity_check(full=True)
            result.update(validation_logits=str(prediction),validation_logits_sha256=file_sha(prediction),
                ending_config_sha256=captured.sha256,ending_implementation_sha256=ending,
                peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
            results['models'][name]=result
            write_json(output/(name+'.json'),result,exclusive=True);write_json(output/'progress.json',results)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        results.update(ending_config_sha256=captured.sha256,ending_implementation_sha256=integrity_check(full=True),
            ending_input_artifact_sha256=lock.hashes)
        write_json(output/'results.json',results,exclusive=True)
    except Exception as error:
        write_json(output/'failure.json',{'family':settings['family'],'configuration_id':active,'seed':args.seed,
            'protocol_sha256':captured.sha256,'initial_implementation_sha256':initial,
            'error_type':type(error).__name__,'reason':str(error),'phase':'fresh_development_validation_only',
            'test_access':False},exclusive=True)
        raise


if __name__=='__main__':main()
