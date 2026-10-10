"""Fresh development for existing GST-span and GraphSSM-ZOH adaptations."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from .priorart_development_bindings import (ROOT,FAMILIES,SEEDS,RAW_NLL,STATUS,
    capture_training_snapshot,implementation_hashes)
from .argmix_development_bindings import development_path,safe_id,sha
from .train_argmix_development import evaluate,check_tile_correspondence,require_device
from .study_models import make_study_model
from .resources import require_storage
from .train import set_seed
from .development_execution import (DETERMINISTIC_POLICY,configure_deterministic_development,deterministic_development_manifest)

_IMPORTED_PINS=implementation_hashes()


def factory(configuration,settings):
    safe_id(configuration['id']);family=settings['family']
    if family not in FAMILIES or type(configuration['hidden_dim']) is not int or isinstance(configuration['dropout'],bool):raise RuntimeError('Supported typed prior-art configuration required')
    options={'hidden_dim':configuration['hidden_dim'],'dropout':configuration['dropout'],
        'num_layers':settings['num_layers'],'reference_adapter':settings['reference_adapter']}
    model=make_study_model(family,48,4,options)
    model_configuration(model,family)
    return model


def model_configuration(model,family):
    if family=='gst_ca_span':
        from .gst_baseline import GSTAdaptation
        if type(model) is not GSTAdaptation:raise RuntimeError('Existing GST factory class differs')
        values=model.configuration_provenance()
        fixed={'control_name':'gst_ca_span','input_dim':48,'num_classes':4,'num_layers':2,'num_heads':4,
            'ffn_multiplier':2,'self_loop_weight':2.,'routing':'cross_attention','token_weighting':'span',
            'gate_variant':'paper','stochastic_depth':0.,'position_encodings':False,'learned_numerical_id_embeddings':False}
        if any(values.get(key)!=value for key,value in fixed.items()):raise RuntimeError('Existing GST equation adaptation differs from frozen specification')
        values.update(source_layers=2,temporal_layers=2,
            cross_tree_aggregation='unordered span-weighted graph-set attention inside one scored tile',
            readout='containing-tree query-leaf MLP; coordinates only select graph',
            exact_zoh=False,record_subdivision_invariant='no whole-model claim; span keys alone do not establish invariance')
    elif family=='graphssm_zoh':
        from .graphssm_baseline import GraphSSMAdaptation,OFFICIAL_GRAPHSSM_COMMIT
        if type(model) is not GraphSSMAdaptation or len(model.blocks)!=2 or model.discretization!='zoh' or model.bidirectional is not True or model.blocks[0].diffusion.root.in_features!=48 or model.num_classes!=4 or any(block.cell.state_dim!=16 or block.cell.discretization!='zoh' for block in model.blocks) or model.blocks[0].mixer is None or model.blocks[1].mixer is not None:
            raise RuntimeError('Existing GraphSSM factory differs from frozen ZOH adaptation')
        values={'control_name':family,'paper':'https://arxiv.org/abs/2406.00943',
            'inspected_source_commit':OFFICIAL_GRAPHSSM_COMMIT,'input_dim':48,'num_classes':4,
            'hidden_dim':model.hidden_dim,'num_layers':2,'source_layers':2,'temporal_layers':2,
            'state_dim':16,'dropout':model.blocks[0].dropout.p,'discretization':'zoh','bidirectional':True,
            'directions_share_parameters':True,'spatial_receptive_field':'two local GraphSAGE diffusion blocks; distant raw-node perturbations need not reach every query, verified F40 summaries also carry global reference information','representation_mixing':'first block only, previous/current observed representations',
            'memory_alignment':'all nonnegative persistent query/reference leaf identities; anonymous internal nodes reset each hold',
            'duration':'original physical interval right-left, cast only after subtraction; global bp/1e6',
            'cross_tree_aggregation':'ordered exact-ZOH forward/reverse scans inside one scored tile',
            'readout':'containing-tree snapshot-end query representation; no fractional within-hold decoder',
            'exact_zoh':True,'record_subdivision_invariant':'no whole-model claim; representation mixing, stacked nonlinear blocks and snapshot-end readout remain',
            'learned_numerical_id_embeddings':False}
    else:raise RuntimeError('Unsupported prior-art family')
    values.update(scientific_status=STATUS,external_pretraining=False,truth_features=[],
        input_scope='full raw observed trees and verified F48; no canonicalization; no cross-tile aggregation',
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        limitation='Independent existing common-input adaptation; not exact native reproduction, independent confirmation or novelty proof.')
    return values


def parent_mean(rows):
    if not rows or any(not math.isfinite(row['nll']) or row['nll']<0 for row in rows):raise RuntimeError('Finite equally weighted parent raw-NLL scores required')
    return math.fsum(row['nll'] for row in rows)/len(rows)


def train_configuration(configuration,settings,parents,device,seed,checkpoint,binding,*,integrity_check,callback=None):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('Fixed101/202/303 seeds required')
    if settings.get('deterministic_policy')!=DETERMINISTIC_POLICY:raise RuntimeError('Frozen strict deterministic policy required')
    deterministic_development_manifest()
    integrity_check(full_sources=True);set_seed(seed)
    model=factory(configuration,settings).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=configuration['learning_rate'],weight_decay=settings['weight_decay'])
    tiles=[(parent,tile) for parent in parents['train'] for tile in parent.tiles]
    if not tiles or not parents['val']:raise RuntimeError('Nonempty train/validation parents required')
    history=[];best=float('inf');best_epoch=None;stale=0;started=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(settings['epochs']):
        deterministic_development_manifest()
        integrity_check(full_sources=False);model.train();order=list(range(len(tiles)));random.shuffle(order);losses=[]
        for index in order:
            parent,tile=tiles[index]
            indices=check_tile_correspondence(parent,tile,torch.zeros(len(parent.sequence.positions),dtype=torch.bool))
            logits=model(tile.sequence.to(device));labels=parent.labels[indices].to(device)
            if logits.shape!=(*labels.shape,4):raise RuntimeError('Prior-art training logits/target alignment differs')
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite prior-art training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError('Nonfinite prior-art gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm']);optimizer.step();losses.append(float(loss.detach()))
        validation,_=evaluate(model,parents['val'],device);nll=parent_mean(validation)
        history.append({'epoch':epoch,'train_nll':math.fsum(losses)/len(losses),'val_nll':nll,'elapsed_seconds':time.monotonic()-started})
        if callback:callback(history[-1])
        integrity_check(full_sources=False)
        if nll<best:
            best,best_epoch,stale=nll,epoch,0
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation_nll':best,'seed':seed,
                'configuration_id':configuration['id'],'binding':binding,**binding},checkpoint)
        else:
            stale+=1
            if stale>=settings['patience']:break
    if best_epoch is None:raise RuntimeError('No finite selected prior-art checkpoint')
    integrity_check(full_sources=True)
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    if chosen.get('binding')!=binding or any(chosen.get(key)!=value for key,value in binding.items()) or chosen.get('seed')!=seed or chosen.get('configuration_id')!=configuration['id'] or chosen.get('epoch')!=best_epoch or chosen.get('validation_nll')!=best:raise RuntimeError('Selected prior-art checkpoint binding differs')
    model.load_state_dict(chosen['model'])
    validation,predictions=evaluate(model,parents['val'],device);selected=parent_mean(validation)
    if selected!=best:raise RuntimeError('Reloaded selected raw-logit NLL differs')
    integrity_check(full_sources=True)
    return {'configuration_id':configuration['id'],'configuration':configuration,'training_seed':seed,
        'parameters':sum(p.numel() for p in model.parameters()),'validation_nll':selected,
        'historical_validation_nll':best,'reloaded_validation_nll':selected,'checkpoint_replay_nll_delta':selected-best,
        'deterministic_policy':deterministic_development_manifest(),'per_parent':validation,
        'best_epoch':best_epoch,'history':history,'model_configuration':model_configuration(model,settings['family']),
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'training_seconds':time.monotonic()-started,
        'ending_config_sha256':binding['protocol_sha256'],'ending_implementation_sha256':binding['implementation_sha256'],
        'smoother_fitting':None,'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None},predictions


def dump_json(path,value,*,exclusive=False):
    content=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with Path(path).open('xb' if exclusive else 'wb') as handle:handle.write(content)


def execute_run(settings,parents,snapshot,config_path,config_sha,output,seed,names,device):
    applied=deterministic_development_manifest()
    if settings.get('deterministic_policy')!=applied:raise RuntimeError('Applied deterministic policy differs from bound config')
    snapshot.check(full_sources=True)
    identity=hashlib.sha256(str(output.relative_to(ROOT)).encode()).hexdigest()[:16]
    checkpoints=development_path(Path('checkpoints/priorart_development_v1')/f'{output.name}_{identity}_{config_sha[:12]}_s{seed}')
    if output.exists() or checkpoints.exists():raise RuntimeError('Fresh prior-art output/checkpoint namespaces required')
    require_storage(headroom_bytes=500_000_000);output.mkdir(parents=True);checkpoints.mkdir(parents=True)
    with (output/'protocol.json').open('xb') as handle:handle.write(snapshot.bytes_for(config_path))
    binding={'protocol_sha256':config_sha,'completion_receipt_sha256':settings['completion_receipt_sha256'],
        'boundary_manifest_sha256':settings['boundary_dataset_manifest_sha256'],
        'implementation_sha256':snapshot.implementation_sha256,'deterministic_policy':applied}
    results={'stage':settings['stage'],'family':settings['family'],'development_namespace':'development-v1',
        'evaluation_split':'fresh_development_validation_only','training_seeds':SEEDS,'training_seed':seed,
        'nll_policy':RAW_NLL,'deterministic_policy':applied,'protocol_sha256':config_sha,'completion_receipt_sha256':settings['completion_receipt_sha256'],
        'boundary_dataset_manifest_sha256':settings['boundary_dataset_manifest_sha256'],
        'initial_implementation_sha256':snapshot.implementation_sha256,'input_scope':settings['input_scope'],
        'cross_tree_aggregation':settings['cross_tree_aggregation'],'scope_amendment':settings['scope_amendment'],
        'models':{},'test_access':False,'scientific_status':STATUS,'source_artifact_sha256':snapshot.sources,
        'interpretation':settings['interpretation'],'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device)},
        'physical_gpu':os.environ.get('ARGGNN_PHYSICAL_GPU')}
    active=None
    def integrity_check(**kwargs):
        deterministic_development_manifest()
        snapshot.check(**kwargs)
    try:
        for configuration in settings['configurations']:
            if configuration['id'] not in names:continue
            active=configuration['id'];snapshot.check(full_sources=True)
            def callback(row):
                require_storage(headroom_bytes=100_000_000)
                print(json.dumps({'event':'epoch','family':settings['family'],'configuration_id':active,'seed':seed,**row}),flush=True)
            result,logits=train_configuration(configuration,settings,parents,device,seed,checkpoints/(active+'.pt'),binding,
                integrity_check=integrity_check,callback=callback)
            snapshot.check(full_sources=True);prediction=checkpoints/(active+'_validation_logits.pt')
            torch.save({'logits':logits,'parent_ids':[parent.parent_id for parent in parents['val']],
                'positions':[parent.sequence.positions for parent in parents['val']],
                'query_node_ids':[parent.sequence.query_node_ids for parent in parents['val']],
                'seed':seed,'protocol_sha256':config_sha,'completion_receipt_sha256':settings['completion_receipt_sha256'],'nll_policy':RAW_NLL},prediction)
            result.update(validation_logits=str(prediction),validation_logits_sha256=sha(prediction))
            results['models'][active]=result
            dump_json(output/(active+'.json'),result,exclusive=True);dump_json(output/'progress.json',results)
        snapshot.check(full_sources=True)
        results.update(ending_config_sha256=config_sha,ending_implementation_sha256=snapshot.implementation_sha256)
        dump_json(output/'results.json',results,exclusive=True)
    except Exception as error:
        dump_json(output/'failure.json',{'family':settings['family'],'configuration_id':active,'training_seed':seed,
            'protocol_sha256':config_sha,'completion_receipt_sha256':settings['completion_receipt_sha256'],
            'initial_implementation_sha256':snapshot.implementation_sha256,'reason':str(error),
            'exception_type':type(error).__name__,'phase':'fresh_development_validation_only','test_access':False},exclusive=True)
        raise
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--expected-config-sha256',required=True)
    parser.add_argument('--output');parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--configurations',nargs='+');parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--validate-config',action='store_true')
    args=parser.parse_args(argv);torch.set_num_threads(4)
    if args.seed not in SEEDS:raise ValueError('Fixed training seeds required')
    if args.configurations:
        for name in args.configurations:safe_id(name)
    output=None if args.output is None else development_path(args.output)
    if not args.validate_config and output is None:parser.error('--output is required for fitting')
    if output is not None and output.exists():raise RuntimeError('Fresh prior-art output directory required')
    config_path=development_path(args.config)
    settings,parents,snapshot=capture_training_snapshot(config_path,args.expected_config_sha256)
    if snapshot.implementation_sha256!=_IMPORTED_PINS:raise RuntimeError('Prior-art source changed after module imports')
    names=args.configurations or [configuration['id'] for configuration in settings['configurations']]
    if len(names)!=len(set(names)) or not set(names).issubset({configuration['id'] for configuration in settings['configurations']}):raise RuntimeError('Distinct IDs from frozen prior-art grid required')
    snapshot.check(full_sources=True)
    applied=configure_deterministic_development()
    if settings.get('deterministic_policy')!=applied:raise RuntimeError('Applied deterministic policy differs from frozen config')
    if args.validate_config:
        for configuration in settings['configurations']:
            if configuration['id'] in names:
                model=factory(configuration,settings)
                print(json.dumps({'event':'priorart_bound_config_valid','configuration_id':configuration['id'],
                    'parameters':sum(p.numel() for p in model.parameters()),'input_scope':settings['input_scope'],'test_access':False}),flush=True)
        snapshot.check(full_sources=True);return
    # Full completion, source bytes, prefix/target joins and own/imported code
    # have all passed before the first device query or model device transfer.
    require_device(args.device)
    execute_run(settings,parents,snapshot,config_path,args.expected_config_sha256,output,args.seed,names,torch.device(args.device))


if __name__=='__main__':main()
