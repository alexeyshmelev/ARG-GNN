"""Whole-tree TreeLoom bridge on the immutable1611 observed panel and rows.

The existing frozen32325-parameter European model, preparation equations,
30-pass optimizer recipe and complete-parent mathematical context are retained.
Only observation/target I/O and a plaintext executor binding are new.
"""
from __future__ import annotations

import gc
import json
import os
import random
import resource
import time
import numpy as np
import torch
from . import paper_argmix_generator as g
from .paper_argmix_features import read_targets
from arggnn.benchmarking import european_treeloom as european
from arggnn.benchmarking import treeloom
from .paper_argmix_gpu import validate_gpu_binding
from .paper_argmix_european_api import model_binding

PROTOCOL='european_c4_prospective1611'


def legacy_ref(path):
    reference=g.file_ref(path)
    return {key:reference[key] for key in ('path','sha256')}


def observation_seal(config_path,parent_id,parent_output,inferred_output,support_output,output,*,guard):
    config,parent,obs,observations,rows=g.observed_assets(config_path,parent_id,parent_output)
    import tskit
    bridge_path=g.confined(inferred_output)/'completed.json'
    support_path=g.confined(support_output)/'completed.json'
    bridge,support=g.metadata(bridge_path),g.metadata(support_path)
    g.need(bridge['parent_id']==support['parent_id']==parent_id and
           bridge['config_ref']==support['config_ref']==g.file_ref(config_path) and
           support['inferred_bridge_ref']==g.file_ref(bridge_path) and support['truth_or_labels_read'] is False,
           'Same observed parent and immutable sampled support required')
    ts=tskit.load(g.verify_ref(bridge['tree_ref']))
    roster=g.metadata(g.verify_ref(bridge['roster_ref']))
    positions=np.fromfile(obs/'positions.u4',dtype='<u4')
    g.need(ts.num_samples==3222 and len(roster)==3222 and ts.num_sites==len(positions) and
           np.array_equal(ts.sites_position+bridge['local_to_rebased_offset_bp'],positions) and
           [r['inferred_tree_node_id'] for r in roster]==list(map(int,ts.samples())) and
           np.array_equal(ts.nodes_time[ts.samples()],[r['age_generations'] for r in roster]),
           'Every native site/sample/date must join the original observed panel')
    targets=read_targets(g.verify_ref(support['targets_ref']))
    g.need(len(targets)==100000 and support['examples']==100000,'Complete paper sampled rows required')
    limits=config['treeloom']['preparation_limits']
    node_count=sum(tree.num_edges+tree.num_roots for tree in ts.trees())
    g.need(node_count<=limits['max_nodes'] and ts.num_trees<=limits['max_holds'] and
           18*node_count+4*ts.num_trees<=limits['max_episode_payload_bytes'],
           'Complete native context exceeds a finite TreeLoom capacity cap; never crop')
    output=g._fresh_directory(output,guard)
    registry=dict(record_ids=np.arange(len(targets),dtype=np.int64),
                  sample_nodes=np.asarray([roster[int(r['sample_index'])]['inferred_tree_node_id'] for r in targets],dtype=np.int64),
                  sample_indices=np.asarray([int(r['sample_index']) for r in targets],dtype=np.int64),
                  positions_bp0=np.asarray([float(r['rebased_midpoint_bp']) for r in targets],dtype=np.float64),
                  parent_id=np.asarray(parent_id),class_order=np.asarray(g.CLASSES))
    guard(4*1024**2)
    np.savez(output/'sampled_registry.npz',**registry)
    receipt=dict(schema='paper_treeloom1611_observation_seal_v1',parent_id=parent_id,split=parent['split'],
                 config_ref=g.file_ref(config_path),source_ref=g.file_ref(__file__),
                 observed_manifest_ref=g.file_ref(obs/'observation_manifest.json'),
                 bridge_ref=g.file_ref(bridge_path),support_ref=g.file_ref(support_path),
                 registry_ref=legacy_ref(output/'sampled_registry.npz'),original_sites=ts.num_sites,
                 native_holds=ts.num_trees,native_node_occurrences=node_count,haplotypes=3222,queries=2972,
                 sampled_rows=100000,original_observed_vcf_ref=observations.get('external_observed_vcf_ref',observations['observed_files'].get('genotypes.vcf.gz')),
                 private_truth_opened=False,all_native_trees_nodes_sites_samples_retained=True,
                 bank_persisted=False,bank_prepared_during_fit_or_prediction=True,
                 observation_ancestry='Four pure-founder reference classes only; NF/Yam and later queries retain unknown ancestry')
    g.Writes(1024**2,guard).json(output/'completed.json',receipt)
    return receipt


def load_observation(config_path,seal_path,*,resource_check):
    import tskit
    config=g.metadata(config_path)
    seal=g.metadata(seal_path)
    g.need(seal['schema']=='paper_treeloom1611_observation_seal_v1' and
           seal['config_ref']==g.file_ref(config_path) and seal['private_truth_opened'] is False,'Observed TreeLoom seal required')
    bridge=g.metadata(g.verify_ref(seal['bridge_ref']))
    observations=g.metadata(g.verify_ref(seal['observed_manifest_ref']))
    ts=tskit.load(g.verify_ref(bridge['tree_ref']))
    rows=g.metadata(g.verify_ref(bridge['roster_ref']))
    original_rows=g.metadata(g.verify_ref(observations['observed_files']['sample_manifest.json']))
    g.need(len(rows)==len(original_rows)==3222 and all(
        all(r[key]==old[key] for key in old) for r,old in zip(rows,original_rows)), 'Full semantic roster differs')
    positions=np.fromfile(g.verify_ref(observations['observed_files']['positions.u4']),dtype='<u4')
    n,h=observations['sites'],3222
    g.need(ts.num_sites==n and len(positions)==n and np.array_equal(ts.sites_position+bridge['local_to_rebased_offset_bp'],positions),
           'Every native site must join packed original genotypes')
    width=(h+7)//8
    packed_path=g.verify_ref(observations['observed_files']['genotypes.packbits'])
    g.need(packed_path.stat().st_size==n*width,'Full packed genotype dimensions differ')
    packed=np.memmap(packed_path,dtype=np.uint8,mode='r',shape=(n,width))
    genotype=np.empty((n,h),dtype=np.int8)
    for start in range(0,n,4096):
        resource_check('TreeLoom_unpack_original_observed_genotypes')
        genotype[start:start+4096]=np.unpackbits(packed[start:start+4096],axis=1,bitorder='little',count=h)
    labels=np.asarray([-1 if row['observed_reference_class'] is None else row['observed_reference_class'] for row in rows],dtype=np.int64)
    queries=np.asarray([row['sample_index'] for row in rows if row['query_eligible']],dtype=np.int64)
    g.need(len(queries)==2972 and np.all(labels[queries]==-1) and np.count_nonzero(labels>=0)==250,
           'Only250 pure founder donor labels; all2972 queries remain unknown')
    source_refs=dict(inference_ref=legacy_ref(g.verify_ref(seal['bridge_ref'])),
                     observed_vcf_ref={key:seal['original_observed_vcf_ref'][key] for key in ('path','sha256')},
                     sample_manifest_ref={key:bridge['roster_ref'][key] for key in ('path','sha256')},
                     packed_genotypes_ref={key:observations['observed_files']['genotypes.packbits'][key] for key in ('path','sha256')})
    manifest=dict(schema='paper_treeloom1611_observed_state_v1',parent_id=seal['parent_id'],split=seal['split'],
                  sequence_length_bp=config['map']['length_bp'],truth_or_oracle_read=False,
                  coverage_original_bp0=bridge['observed_coverage_bp0'],native_pipeline='Relate1.2.4 bounded staged; no extra size-estimation iterations')
    return dict(manifest=manifest,ts=ts,rows=rows,nodes=np.asarray(ts.samples(),dtype=np.int64),genotype=genotype,
                positions=np.asarray(ts.sites_position,dtype=np.float64),allele_count=np.full(n,2,dtype=np.int8),
                labels=labels,ages=np.asarray([r['age_generations'] for r in rows],dtype=np.float64),query_columns=queries,
                coordinate_offset_bp=bridge['local_to_rebased_offset_bp'],source_refs=source_refs)


def prepare_bundle(config_path,seal_path,*,resource_check):
    config=g.metadata(config_path)
    seal=g.metadata(seal_path)
    state=load_observation(config_path,seal_path,resource_check=resource_check)
    split={'train':'train','val':'validation','iid':'evaluation','misspec':'evaluation'}[seal['split']]
    # The public preparer records the manifest split verbatim before mapping it.
    state['manifest']['split']=split
    bundle,metadata=european.prepare_observation(state,protocol_id=PROTOCOL,parent_id=seal['parent_id'],split=split,
       source_refs=state['source_refs'],sampled_registry_ref=seal['registry_ref'],
       resource_check=resource_check,**config['treeloom']['preparation_limits'])
    return bundle,metadata


def tensor_bytes(bundle):
    values=treeloom._european_module()._bank_tensors(bundle.bank)+treeloom._european_module()._episode_tensors(bundle.episodes)+bundle.cut.tensors()+(bundle.focal_hold,bundle.focal_sample_column)
    seen=set();total=0
    for value in values:
        storage=value.untyped_storage()
        key=(storage.data_ptr(),storage.nbytes())
        if key not in seen:
            seen.add(key);total+=storage.nbytes()
    return total


def make_targets(label_path,seal_path,bundle):
    from arggnn.benchmarking.targets import SampledFocalTargets
    seal,label=g.metadata(seal_path),g.metadata(label_path)
    g.need(label['parent_id']==bundle.parent_id and label['split']=='train' and
           label['support_manifest_ref']==seal['support_ref'] and label['class_order']==list(g.CLASSES),
           'TRAIN labels must join identical100000 rows')
    labels=np.fromfile(g.verify_ref(label['labels_ref']),dtype=np.uint8)
    g.need(labels.shape==(100000,) and np.all(labels<4),'Complete100000 TRAIN labels required')
    lookup={int(value):index for index,value in enumerate(bundle.bank.query_column)}
    return SampledFocalTargets(readout_rows=tuple(map(int,bundle.focal_hold.tolist())),
       query_columns=tuple(lookup[int(value)] for value in bundle.focal_sample_column),labels=tuple(map(int,labels)),
       class_order=g.CLASSES,parent_id=bundle.parent_id,split='train',sampled_registry_ref=bundle.sampled_registry_ref)


def train(config_path,stage_root,output,*,device_name,cpu_threads,guard):
    from .paper_argmix_treeloom_executor import PaperEuropeanBoundedExecutionV1
    config=g.metadata(config_path)
    torch.set_num_threads(cpu_threads)
    validate_gpu_binding()
    g.need(device_name=='cuda:0' and torch.cuda.device_count()==1,
           'Root must expose one admitted GPU3/4/5 UUID as logical cuda0')
    model=european.build_model(PROTOCOL).to(device_name)
    g.need(sum(p.numel() for p in model.parameters())==32325 and len(list(model.parameters()))==49,'Frozen European architecture required')
    settings=config['treeloom']['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,betas=(.9,.999),eps=1e-8)
    output=g._fresh_directory(output,guard)
    stage_root=g.confined(stage_root)
    parents=[p['parent_id'] for p in config['parents'] if p['split']=='train']
    source_seals={p:g.file_ref(stage_root/p/'treeloom_observation/completed.json') for p in parents}
    # All complete observations are authenticated before any TRAIN truth is joined.
    for reference in source_seals.values():
        seal=g.metadata(g.verify_ref(reference));g.need(seal['private_truth_opened'] is False,'Observation-first barrier')
    def check(label):
        guard(0)
        g.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['treeloom']['max_rss_bytes'],'TreeLoom finite RSS ceiling exceeded')
    updates=0;events=[];checkpoints={};cached=None;started=time.monotonic()
    with (output/'progress.jsonl').open('x') as progress:
        for epoch in range(1,31):
            order=list(parents);random.Random(101+epoch).shuffle(order)
            for parent in order:
                seal_path=stage_root/parent/'treeloom_observation/completed.json'
                if cached is not None and cached[0]==parent:
                    bundle,metadata=cached[1:]
                else:
                    cached=None;gc.collect()
                    bundle,metadata=prepare_bundle(config_path,seal_path,resource_check=check)
                    payload=tensor_bytes(bundle)
                    if payload<=config['treeloom']['cache_bytes']:
                        cached=(parent,bundle,metadata)
                targets=make_targets(stage_root/parent/'labels/completed.json',seal_path,bundle)
                model_binding(model,bundle,metadata,check)
                model.train();optimizer.zero_grad(set_to_none=True)
                logits=PaperEuropeanBoundedExecutionV1(model,bundle,metadata,resource_check=check).forward()
                loss=torch.nn.functional.cross_entropy(logits,torch.tensor(targets.labels,device=device_name),reduction='mean')
                g.need(torch.isfinite(loss).item(),'Finite complete-parent mean CE required')
                loss.backward()
                parameters=list(model.parameters())
                g.need(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters),'All49 finite connected gradients required')
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();updates+=1
                counters=[int(optimizer.state[p]['step'].item()) for p in parameters]
                g.need(len(counters)==49 and set(counters)=={updates},'All49 actual Adam counters must advance once')
                event=dict(epoch=epoch,parent_id=parent,optimizer_updates=updates,loss=float(loss.detach()),gradient_norm=float(norm),
                           sampled_records=100000,native_holds=bundle.native_holds,native_node_occurrences=bundle.bank.parent.numel(),
                           prepared_tensor_bytes=tensor_bytes(bundle),elapsed_seconds=time.monotonic()-started)
                text=json.dumps(event,sort_keys=True)+'\n';guard(len(text.encode()));progress.write(text);progress.flush();print(text,end='',flush=True)
                events.append(event);del targets,logits,loss,bundle,metadata;check('after_TreeLoom_parent_update')
            if epoch in (5,10,15,20,25,30):
                path=output/f'epoch{epoch:02d}.pt';guard(8*1024**2)
                torch.save(dict(schema='paper_treeloom1611_checkpoint_v1',model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   optimizer_state=optimizer.state_dict(),config_ref=g.file_ref(config_path),epoch=epoch,optimizer_updates=updates,
                   source_observation_seals=source_seals,class_order=list(g.CLASSES),settings=settings,
                   checkpoint_selection='fixed_final_epoch30',test_truth_read=False),path)
                checkpoints[str(epoch)]=g.file_ref(path)
    g.need(updates==150,'Thirty full passes over five TRAIN parents required')
    receipt=dict(schema='paper_treeloom1611_selected_model_v1',config_ref=g.file_ref(config_path),source_ref=g.file_ref(__file__),
       checkpoint_ref=checkpoints['30'],checkpoint_events=checkpoints,selected_epoch=30,optimizer_updates=150,
       train_examples_per_parent=100000,train_parents=parents,source_observation_seals=source_seals,
       class_order=list(g.CLASSES),model_parameters=32325,trainable_tensors=49,settings=settings,
       validation_used_for_optimization=False,test_or_ood_read=False,seconds=time.monotonic()-started,
       prepared_tensor_bytes_by_parent={p:max(e['prepared_tensor_bytes'] for e in events if e['parent_id']==p) for p in parents},
       max_rss_bytes_observed=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    g.Writes(8*1024**2,guard).json(output/'completed.json',receipt)
    return receipt


def predict(config_path,selected_path,seal_path,output,*,device_name,cpu_threads,guard):
    from .paper_argmix_treeloom_executor import PaperEuropeanBoundedExecutionV1
    config,selected,seal=g.metadata(config_path),g.metadata(selected_path),g.metadata(seal_path)
    g.need(selected['config_ref']==seal['config_ref']==g.file_ref(config_path) and seal['split'] in ('iid','misspec') and
           selected['optimizer_updates']==150 and selected['selected_epoch']==30,'Same selected whole-parent TreeLoom model required')
    validate_gpu_binding()
    g.need(device_name=='cuda:0' and torch.cuda.device_count()==1,'Only one admitted campaign UUID as logical cuda0')
    torch.set_num_threads(cpu_threads)
    def check(label):
        guard(0)
        g.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=config['treeloom']['max_rss_bytes'],'TreeLoom prediction RSS ceiling exceeded')
    model=european.build_model(PROTOCOL)
    saved=torch.load(g.verify_ref(selected['checkpoint_ref']),map_location='cpu',weights_only=True)
    g.need(saved['config_ref']==g.file_ref(config_path) and saved['class_order']==list(g.CLASSES) and saved['epoch']==30 and saved['optimizer_updates']==150,'Checkpoint lineage differs')
    model.load_state_dict(saved['model_state'],strict=True);model.to(device_name).eval()
    bundle,metadata=prepare_bundle(config_path,seal_path,resource_check=check)
    model_binding(model,bundle,metadata,check)
    with torch.no_grad():
        logits=PaperEuropeanBoundedExecutionV1(model,bundle,metadata,resource_check=check).forward()
        probabilities=logits.float().softmax(-1).cpu().numpy()
    g.need(probabilities.shape==(100000,4) and np.all(np.isfinite(probabilities)),'Complete100000 shared predictions required')
    output=g._fresh_directory(output,guard);guard(probabilities.nbytes+1024)
    np.save(output/'probabilities.npy',probabilities,allow_pickle=False)
    receipt=dict(schema='paper_argmix_predictions_v1',method='TreeLoom_frozen1611_32325',parent_id=seal['parent_id'],split=seal['split'],
                 config_ref=g.file_ref(config_path),selected_model_ref=g.file_ref(selected_path),observation_seal_ref=g.file_ref(seal_path),
                 support_ref=seal['support_ref'],probabilities_ref=g.file_ref(output/'probabilities.npy'),examples=100000,
                 class_order=list(g.CLASSES),truth_read=False,full_native_context=True,prepared_tensor_bytes=tensor_bytes(bundle))
    g.Writes(1024**2,guard).json(output/'completed.json',receipt)
    return receipt
