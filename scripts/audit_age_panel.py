#!/usr/bin/env python3
"""Exhaustive, label-free, parent-weighted diagnostics of sealed age sidecars."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from arggnn.resources import ROOT, project_path
from arggnn.boundary_dataset import load_boundary_dataset, sha256_file
from arggnn.age_dataset import load_age_caches
from arggnn.anchor_kernel import prepare_anchor_kernel_bank
from arggnn.data import GraphSequence

AGE_MANIFEST = '23a0903548c09f0a8f803e67140815c2497b487f25470745cad1e8e0b8b04871'
AGE_PREPARATION = 'ec50f55989271ee93a6fe3174ecd90e4899c82de5cdac2f55d6ae87f8b0a71d7'
BOUNDARY_MANIFEST = '7f25874806535897d819b99d20c8f0da1b47f4fe10dde04280e7c42f6e1b9e09'
BOUNDARY_PREPARATION = 'c1749a89c2776debadb16a814516a1998057f380fada0c9efe8acafaed284642'
SOURCE_MANIFEST = '611c7e32973c22bd35115b835279884f7b0c866136e91ae2c5f2a95daefd6fe7'
DEPENDENCIES = ('age_dataset.py','age_interface.py','kron_interface.py','boundary_dataset.py',
    'models.py','simulate.py','train_study.py','data.py','reference_encoder.py',
    'genealogy_baselines.py','anchor_kernel.py','boundary_memory.py')


def event(kind, **values):
    print(json.dumps({'event':kind,**values},sort_keys=True),flush=True)


def source_hashes():
    result={str(Path('src/arggnn')/name):sha256_file(ROOT/'src/arggnn'/name)
            for name in DEPENDENCIES}
    result['scripts/audit_age_panel.py']=sha256_file(Path(__file__))
    return result


def relative(numerator, denominator):
    if bool(((denominator==0)&(numerator!=0)).any()):
        raise RuntimeError('Nonzero residual has an exactly zero reference norm')
    return torch.where(denominator==0,torch.zeros_like(numerator),
        numerator/torch.where(denominator==0,torch.ones_like(denominator),denominator))


def norm(value):
    return torch.linalg.matrix_norm(value,ord='fro')


def best_scalar(matrix, plain):
    numerator=(matrix*plain).sum(dim=(-2,-1))
    denominator=plain.square().sum(dim=(-2,-1))
    beta=relative(numerator,denominator)
    return beta,relative(norm(matrix-beta[...,None,None]*plain),norm(matrix))


def quality(matrix, values, vectors):
    residual=relative(norm((vectors*values.unsqueeze(-2))@vectors.transpose(-1,-2)-matrix),norm(matrix))
    identity=torch.eye(matrix.shape[-1],dtype=matrix.dtype)
    orthogonality=(vectors.transpose(-1,-2)@vectors-identity).abs().amax(dim=(-2,-1))
    return residual,orthogonality


def weighted_summary(value, widths):
    if value.ndim!=1 or widths.shape!=value.shape or not bool(torch.isfinite(value).all()):
        raise RuntimeError('Invalid graph metric')
    weight=widths/widths.sum()
    return {'physical_span_weighted_mean':float((value*weight).sum()),
            'minimum_graph':float(value.min()),'maximum_graph':float(value.max())}


def aggregate(parents):
    result={'parent_count':len(parents)}
    if not parents: return result
    for section in ('operators','horizons'):
        keys=parents[0][section]
        result[section]={}
        for operator in keys:
            result[section][operator]={}
            for metric in keys[operator]:
                means=[p[section][operator][metric]['physical_span_weighted_mean'] for p in parents]
                result[section][operator][metric]={
                    'equal_parent_mean':sum(means)/len(means),
                    'minimum_parent_mean':min(means),'maximum_parent_mean':max(means),
                    'minimum_graph_across_parents':min(p[section][operator][metric]['minimum_graph'] for p in parents),
                    'maximum_graph_across_parents':max(p[section][operator][metric]['maximum_graph'] for p in parents)}
    result['anchor_source_coverage']={}
    for key in ('clipped_record_fraction','clipped_physical_span_fraction',
                'original_record_fraction','original_physical_span_fraction'):
        values=[p['anchor_source_coverage'][key] for p in parents]
        result['anchor_source_coverage'][key]={'equal_parent_mean':sum(values)/len(values),
            'minimum_parent':min(values),'maximum_parent':max(values)}
    return result


def components(sequence, graphs, edges, times, query_ids, reference_ids, horizon):
    metric={key:[] for key in (
        'query_multiquery_fraction','query_nonsingle_sample_fraction',
        'query_reference_grounded_fraction','query_conflicting_reference_component_fraction',
        'reference_singleton_fraction','reference_older_or_equal_fraction',
        'conflicting_reference_component_count','query_component_query_size_mean')}
    for nodes,pairs in zip(graphs,edges):
        leader={node:node for node in nodes}
        def find(node):
            while leader[node]!=node:
                leader[node]=leader[leader[node]]; node=leader[node]
            return node
        for u,v in pairs:
            if times[u]<horizon and times[v]<horizon: leader[find(u)]=find(v)
        sample={int(sequence.node_ids[node]):node for node in nodes if int(sequence.node_ids[node])>=0}
        sets={}
        for identity in query_ids:
            node=sample[identity]; sets.setdefault(find(node),{'q':[],'r':[],'categories':set()})['q'].append(node)
        for identity in reference_ids:
            node=sample[identity]; comp=sets.setdefault(find(node),{'q':[],'r':[],'categories':set()})
            comp['r'].append(node)
            roles=sequence.x[node,:4]
            if not bool(((roles==0)|(roles==1)).all()) or float(roles.sum())!=1.:
                raise RuntimeError('Reference role must be observed one-hot input')
            comp['categories'].add(int(roles.argmax()))
        q,r=len(query_ids),len(reference_ids)
        metric['query_multiquery_fraction'].append(sum(len(c['q']) for c in sets.values() if len(c['q'])>1)/q)
        metric['query_nonsingle_sample_fraction'].append(sum(len(c['q']) for c in sets.values() if len(c['q'])+len(c['r'])>1)/q)
        metric['query_reference_grounded_fraction'].append(sum(len(c['q']) for c in sets.values() if c['r'])/q)
        metric['query_conflicting_reference_component_fraction'].append(sum(len(c['q']) for c in sets.values() if len(c['categories'])>1)/q)
        metric['reference_singleton_fraction'].append(sum(len(c['r']) for c in sets.values() if len(c['q'])+len(c['r'])==1)/r)
        metric['reference_older_or_equal_fraction'].append(sum(times[sample[identity]]>=horizon for identity in reference_ids)/r)
        metric['conflicting_reference_component_count'].append(sum(len(c['categories'])>1 for c in sets.values()))
        metric['query_component_query_size_mean'].append(sum(len(c['q'])**2 for c in sets.values())/q)
    return {key:torch.tensor(value,dtype=torch.float64) for key,value in metric.items()}


def observed_source(record):
    path=project_path(record['prepared_tensor_file'])
    if sha256_file(path)!=record['prepared_tensor_sha256']:
        raise RuntimeError('Observed prepared tensor changed after verified loading')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['adapter']['source_tensor_sha256']!=record['tensor_sha256']:
        raise RuntimeError('Prepared observed-source binding mismatch')
    sequence=GraphSequence.from_dict(payload['sequence'])
    groups=[[] for _ in range(sequence.num_graphs)]
    edge_groups=[set() for _ in range(sequence.num_graphs)]
    graph_ids=sequence.node_graph.tolist()
    for node,graph in enumerate(graph_ids): groups[graph].append(node)
    for u,v in sequence.edge_index.T.tolist(): edge_groups[graph_ids[u]].add((min(u,v),max(u,v)))
    return sequence,groups,[sorted(pairs) for pairs in edge_groups]


def self_test():
    widths=torch.tensor([.1,.9],dtype=torch.float64)
    mean=weighted_summary(torch.tensor([0.,1.],dtype=torch.float64),widths)
    assert math.isclose(mean['physical_span_weighted_mean'],.9)
    plain=torch.tensor([[[2.,-1.],[-1.,2.]]],dtype=torch.float64)
    beta,departure=best_scalar(plain*.3,plain)
    torch.testing.assert_close(beta,torch.tensor([.3],dtype=torch.float64),rtol=1e-15,atol=0)
    assert departure.max()<1e-15
    values,vectors=torch.linalg.eigh(plain)
    reconstruction,orthogonality=quality(plain,values,vectors)
    assert reconstruction.max()<1e-15 and orthogonality.max()<1e-15
    # Exact horizon ties separate the two below-cut leaves even when an
    # anonymous unary root exists on the threshold.
    class Fixture:
        node_ids=torch.tensor([10,20,30,-1])
        x=torch.zeros(4,8,dtype=torch.float64)
    fixture=Fixture(); fixture.x[2,0]=1.
    checked=components(fixture,[[0,1,2,3]],[[(0,3),(1,3),(2,3)]],[0.,0.,250.,350.],
                       [10,20],[30],350.)
    assert checked['query_multiquery_fraction'].item()==0
    assert checked['query_reference_grounded_fraction'].item()==0
    merged=components(fixture,[[0,1,2,3]],[[(0,3),(1,3),(2,3)]],[0.,0.,250.,350.],
                      [10,20],[30],600.)
    assert merged['query_multiquery_fraction'].item()==1
    assert merged['query_reference_grounded_fraction'].item()==1
    event('self_test_passed',checks=['span_weighting','scalar_residual','spectral_reconstruction','strict_horizon_tie'])


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--age-data',default='data/age_inferred_cohort64_pilot_v1')
    parser.add_argument('--output',default='research/reviews/age_panel_geometry_v1.json')
    parser.add_argument('--threads',type=int,default=4,choices=(1,2,3,4))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU diagnostics require CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(args.threads)
    if args.self_test: self_test(); return
    output=project_path(args.output)
    if output.exists(): raise RuntimeError('Choose a fresh audit output')
    started=time.perf_counter()
    age_dir=project_path(args.age_data); age_path=age_dir/'manifest.json'
    if sha256_file(age_path)!=AGE_MANIFEST: raise RuntimeError('Unexpected sealed age manifest')
    age_manifest=json.loads(age_path.read_text())
    if age_manifest['boundary_manifest_sha256']!=BOUNDARY_MANIFEST or age_manifest['boundary_preparation_sha256']!=BOUNDARY_PREPARATION or age_manifest['source_manifest_sha256']!=SOURCE_MANIFEST:
        raise RuntimeError('Unexpected age/base/source binding')
    snapshot=source_hashes()
    event('start',pid=os.getpid(),threads=args.threads,age_manifest_sha256=AGE_MANIFEST)
    dataset=load_boundary_dataset(age_manifest['boundary_directory'],
        expected_manifest_sha256=BOUNDARY_MANIFEST,
        expected_preparation_sha256=BOUNDARY_PREPARATION,
        expected_source_manifest_sha256=SOURCE_MANIFEST)
    caches=load_age_caches(age_dir,dataset,expected_manifest_sha256=AGE_MANIFEST,
                           expected_preparation_sha256=AGE_PREPARATION)
    event('integrity_verified',parents=len(dataset.parent_cases),tiles=len(caches),
          elapsed_seconds=time.perf_counter()-started)
    source_path=Path(dataset.manifest['source']['manifest_path'])
    source_manifest=json.loads(source_path.read_text())
    source_records={(record['split'],record['replicate']):record for record in source_manifest['replicates']}
    parents=[]
    for parent in dataset.parent_cases:
        _,split,replicate=parent.parent_id.split(':'); replicate=int(replicate)
        source_record=source_records[(split,replicate)]
        sequence,groups,source_edges=observed_source(source_record)
        source_widths=(sequence.intervals[:,1]-sequence.intervals[:,0]).double()
        source_span=float(source_widths.sum())
        chronological=torch.argsort(sequence.intervals[:,0]); source_starts=sequence.intervals[chronological,0].contiguous()
        original_touched=torch.zeros(sequence.num_graphs,dtype=torch.bool)
        metrics={label:{} for label in ('plain','age350','age1400')}
        all_widths=[]; tiles=[]; clipped_touched=clipped_count=0; touched_span=0.; anchors_total=0
        expected_queries=parent.query_node_ids
        references=None; boundary_count=None
        for tile in parent.tiles:
            identity=tile.provenance['tile_id']; bank=tile.bank; cache=caches[identity]
            if not torch.equal(expected_queries,cache.query_ids): raise RuntimeError('Parent query order changed')
            if references is None: references=cache.reference_ids; boundary_count=cache.sample_ids.numel()
            elif not torch.equal(references,cache.reference_ids): raise RuntimeError('Reference order changed within parent')
            width=(bank.intervals[:,1]-bank.intervals[:,0]).double()
            layout=prepare_anchor_kernel_bank(bank,cache,num_anchors=32).layout
            touches=torch.bincount(layout.anchor_graph,minlength=len(width))
            touched=touches>0
            clipped_touched+=int(touched.sum()); clipped_count+=len(width)
            touched_span+=float(width[touched].sum()); anchors_total+=len(layout.anchors)
            mid=bank.intervals.mean(-1)
            original=chronological[torch.searchsorted(source_starts,mid,right=True)-1]
            if not bool(((bank.intervals[:,0]>=sequence.intervals[original,0])&(bank.intervals[:,1]<=sequence.intervals[original,1])).all()):
                raise RuntimeError('Clipped hold does not belong to its original observed graph')
            original_touched[original[touched]]=True
            tiles.append({'tile_id':identity,'clipped_graph_count':len(width),
                'physical_span':float(width.sum()),'source_graph_indices':original.tolist(),
                'anchor_count':len(layout.anchors),'source_graph_records_touched':int(touched.sum()),
                'source_record_fraction':float(touched.double().mean()),
                'source_physical_span_fraction':float(width[touched].sum()/width.sum())})
            raw=cache.raw_response; plain=raw[:,0]
            query_plain=torch.cat((cache.cqq[:,0],cache.cqr[:,0]),dim=-1)
            for index,label in enumerate(cache.operator_labels):
                current=raw[:,index]; beta,departure=best_scalar(current,plain)
                scalar=plain*cache.mean_diagonal_ratios[:,index,None,None]
                query=torch.cat((cache.cqq[:,index],cache.cqr[:,index]),dim=-1)
                scalar_query=torch.cat((cache.scalar_cqq[:,index],cache.scalar_cqr[:,index]),dim=-1)
                query_beta,query_departure=best_scalar(query,query_plain)
                reconstruction,orthogonality=quality(cache.cqq[:,index],cache.eigenvalues[:,index],cache.eigenvectors[:,index])
                scalar_reconstruction,_=quality(cache.scalar_cqq[:,index],cache.scalar_eigenvalues[:,index],cache.scalar_eigenvectors[:,index])
                diag=[row[index] for row in cache.diagnostics]
                values={
                    'best_scalar_full_boundary':beta,
                    'relative_departure_best_scalar_full_boundary':departure,
                    'relative_departure_mean_matched_full_boundary':relative(norm(current-scalar),norm(current)),
                    'relative_departure_best_scalar_query_rows':query_departure,
                    'best_scalar_query_rows':query_beta,
                    'relative_departure_matched_scalar_cache_query_rows':relative(norm(query-scalar_query),norm(query)),
                    'relative_departure_matched_scalar_cache_cqq':relative(norm(cache.cqq[:,index]-cache.scalar_cqq[:,index]),norm(cache.cqq[:,index])),
                    'relative_departure_matched_scalar_cache_cqr':relative(norm(cache.cqr[:,index]-cache.scalar_cqr[:,index]),norm(cache.cqr[:,index])),
                    'mean_diagonal_attenuation':cache.mean_diagonal_ratios[:,index],
                    'minimum_query_eigenvalue':cache.eigenvalues[:,index].min(-1).values,
                    'maximum_query_eigenvalue':cache.eigenvalues[:,index].max(-1).values,
                    'query_spectral_reconstruction_relative':reconstruction,
                    'scalar_query_spectral_reconstruction_relative':scalar_reconstruction,
                    'query_spectral_orthogonality_max':orthogonality,
                    'reference_grounding_row_sum_mean':(-cache.cqr[:,index].sum(-1)).mean(-1),
                    'source_cache_spectral_reconstruction_relative':cache.spectral_reconstruction_relative[:,index],
                }
                for name in ('interior_condition_number','relative_solve_residual','minimum_response_eigenvalue',
                             'response_row_sum_relative_to_source','response_row_sum_relative_to_response',
                             'response_symmetry_relative_to_source','maximum_response_offdiagonal'):
                    values['raw_schur_'+name]=torch.tensor([getattr(item,name) for item in diag],dtype=torch.float64)
                for threshold in (.01,.05,.1):
                    values[f'physical_fraction_departure_best_scalar_gt_{threshold:g}']=(departure>threshold).double()
                for name,value in values.items(): metrics[label].setdefault(name,[]).append(value)
            all_widths.append(width)
        widths=torch.cat(all_widths)
        if not math.isclose(float(widths.sum()),source_span,rel_tol=0.,abs_tol=1e-12):
            raise RuntimeError('Tile spans fail to partition the observed parent')
        summaries={label:{name:weighted_summary(torch.cat(values),widths) for name,values in table.items()}
                   for label,table in metrics.items()}
        times=torch.expm1(sequence.x[:,6].double()*math.log1p(10000.)).tolist()
        horizon_stats={}
        for horizon in (350.,600.):
            values=components(sequence,groups,source_edges,times,expected_queries.tolist(),references.tolist(),horizon)
            horizon_stats[str(int(horizon))]={name:weighted_summary(value,source_widths) for name,value in values.items()}
        result={'parent_id':parent.parent_id,'split':split,'replicate':replicate,
            'query_count':len(expected_queries),'reference_count':len(references),'boundary_count':boundary_count,
            'tile_count':len(parent.tiles),'source_graph_count':sequence.num_graphs,
            'clipped_graph_count':clipped_count,'physical_span':source_span,
            'source_prepared_tensor_sha256':source_record['prepared_tensor_sha256'],
            'source_tensor_sha256':source_record['tensor_sha256'],
            'adapter_configuration_sha256':source_record['adapter_configuration_sha256'],
            'operators':summaries,'horizons':horizon_stats,'tiles':tiles,
            'anchor_source_coverage':{'anchors_per_tile':32,'anchor_count':anchors_total,
                'clipped_records_touched':clipped_touched,'clipped_record_fraction':clipped_touched/clipped_count,
                'clipped_physical_span_fraction':touched_span/source_span,
                'original_records_touched':int(original_touched.sum()),
                'original_record_fraction':float(original_touched.double().mean()),
                'original_physical_span_fraction':float(source_widths[original_touched].sum()/source_widths.sum())}}
        parents.append(result)
        event('parent_done',split=split,replicate=replicate,source_graphs=sequence.num_graphs,
              clipped_graphs=clipped_count,elapsed_seconds=time.perf_counter()-started)
    if snapshot!=source_hashes(): raise RuntimeError('Diagnostic source/dependencies changed during audit')
    if sha256_file(age_path)!=AGE_MANIFEST or sha256_file(dataset.directory/'manifest.json')!=BOUNDARY_MANIFEST:
        raise RuntimeError('Sealed manifests changed during audit')
    report={'schema_version':1,'kind':'exhaustive_label_free_sealed_inferred_age_panel_geometry',
        'created_utc':datetime.now(timezone.utc).isoformat(),'device':'cpu','threads':args.threads,
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__)},
        'elapsed_seconds':time.perf_counter()-started,'pid':os.getpid(),
        'age_manifest_sha256':AGE_MANIFEST,'age_preparation_sha256':AGE_PREPARATION,
        'boundary_manifest_sha256':BOUNDARY_MANIFEST,'boundary_preparation_sha256':BOUNDARY_PREPARATION,
        'observed_source_manifest_sha256':SOURCE_MANIFEST,'observed_source_protocol_sha256':dataset.manifest['source']['protocol_sha256'],
        'source_sha256':snapshot,'operator_normalization':'one shared plain full-boundary mean diagonal per graph',
        'aggregation':'Graph widths normalized by total observed physical span within each parent; equal-parent mean and parent ranges, split-specific then pooled descriptive summaries. No weighting by record count or ancestry labels.',
        'best_scalar':'Frobenius least-squares beta per graph: <Cs,C0>/||C0||_F^2; residual ||Cs-beta*C0||_F/||Cs||_F',
        'matched_scalar':'Full-B alpha*C0 uses cached age/plain mean-diagonal ratio; direct cached scalar QQ/QR blocks are also compared independently.',
        'horizon_rule':'Strict decoded observed age < tau, endpoints on tau excluded; crossing edges terminate at separate unlabeled virtual roots.',
        'anchor_coverage_rule':'32 fixed physical cell midpoints per verified tile using actual consumer layout; covered span means union of complete graph holds containing >=1 anchor. All holds still contribute generator propagation.',
        'verification_scope':'Both frozen verified loaders checked all source/sidecar hashes and bindings before metrics. Their external supervision integrity checks are mandatory. No labels, baseline probabilities, population truth, oracle tree, or query-label agreement enter any diagnostic calculation.',
        'interpretation':'Geometry, spectrum, observed component coverage, and anchor-source sampling diagnostics only. No inferred biological correctness or predictive benefit is established.',
        'parents':parents,
        'split_summaries':{split:aggregate([p for p in parents if p['split']==split]) for split in ('train','val')},
        'pooled_descriptive_summary':aggregate(parents)}
    with output.open('x') as handle: json.dump(report,handle,indent=2,sort_keys=True,allow_nan=False); handle.write('\n')
    event('complete',output=str(output),sha256=sha256_file(output),bytes=output.stat().st_size,
          parents=len(parents),tiles=len(caches),elapsed_seconds=report['elapsed_seconds'])


if __name__=='__main__': main()
