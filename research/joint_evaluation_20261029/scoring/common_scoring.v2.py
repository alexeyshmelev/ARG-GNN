"""Externally sealed native-six scoring; no model execution or competitor writes."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
from arggnn.benchmarking import european_sampling as sampling
from arggnn.benchmarking.paper_ancestralpaths import metrics as native_metrics

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/joint_evaluation_20261029/scoring'
CLASSES=['Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA']
COHORTS=['present_day','Bronze_Age','BAA','Neo','Yam']
BASELINES=['ancestralpaths_historical_cnn_native6','ancestralpaths_dense_control','gst_lai_8_native_graphsetconv','treeloom']
NEW_METHOD='treeloom_joint_structure_v1'
METRICS=['accuracy','macro_F1','Brier','raw_NLL','mean_predictive_entropy_nats','mean_max_probability','ECE_10_equal_width']


def require(ok,message):
    if not bool(ok):raise ValueError(message)

def path_in_repo(value):
    p=(ROOT/value).resolve()
    require(p.is_relative_to(ROOT),'Reference outside repository')
    return p

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def ref(path):
    p=path_in_repo(path)
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)

def same_ref(a,b):return path_in_repo(a['path'])==path_in_repo(b['path']) and a['sha256']==b['sha256']

def checked(record,checked_files,private=False):
    p=path_in_repo(record['path'])
    require(private or not any('truth' in part.lower() for part in p.parts),'Private truth opened before prediction barrier')
    actual=ref(p)
    require(actual['sha256']==record['sha256'],'SHA mismatch: '+str(p))
    require('bytes' not in record or actual['bytes']==record['bytes'],'Byte-size mismatch')
    checked_files[actual['path']]=actual
    return p

def read_ref(record,checked_files,private=False):return json.loads(checked(record,checked_files,private).read_text())

def save(path,value):
    p=path_in_repo(path)
    require(p.is_relative_to(HERE),'Scorer outputs must remain in owned scoring directory')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def load_arrays(record,checked_files,private=False,maximum_bytes=16*2**20,members=None):
    p=checked(record,checked_files,private)
    with zipfile.ZipFile(p) as z:
        names=z.namelist() if members is None else [name+'.npy' for name in members]
        require(sum(z.getinfo(name).file_size for name in names)<=maximum_bytes,'Array allocation bound exceeded')
    with np.load(p,allow_pickle=False) as archive:
        keys=archive.files if members is None else members
        return {key:archive[key].copy() for key in keys}

def validate_probabilities(probabilities,rows):
    p=np.asarray(probabilities)
    require(p.dtype==np.float32 and p.shape==(rows,6),'Float32 native-six probability shape')
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Finite probabilities in [0,1]')
    require(np.allclose(p.sum(1),1.,atol=1e-6,rtol=1e-5),'Raw probability sums; no repair')

def validate_arrays(arrays,records,parent,is_new):
    n=len(records);p=arrays['probabilities'];validate_probabilities(p,n)
    require(arrays['example_indices'].dtype==np.int64 and np.array_equal(arrays['example_indices'],np.arange(n)),'Every canonical example exactly once in original order')
    if 'example_ids' in arrays:require(np.array_equal(arrays['example_ids'],[parent+':'+str(i) for i in range(n)]),'Global parent/example identities')
    diagnostics=dict(max_probability_sum_error=float(np.abs(p.sum(1)-1).max()))
    if is_new:
        require(set(arrays)=={'logits','probabilities','example_indices','sample_indices','native_site_ids','positions_original_bp'},'Exact label-free new prediction array schema')
        for field,key,dtype in [('sample_indices','sample_index',np.int64),('native_site_ids','native_site_id',np.int64),('positions_original_bp','position_original_bp',np.float64)]:
            require(arrays[field].dtype==dtype and np.array_equal(arrays[field],[row[key] for row in records]),'Canonical identity mismatch: '+field)
        logits=arrays['logits']
        require(logits.dtype==np.float32 and logits.shape==(n,6) and np.isfinite(logits).all(),'Finite raw six logits')
        z=logits.astype(np.float64);z-=z.max(1,keepdims=True);softmax=np.exp(z);softmax/=softmax.sum(1,keepdims=True)
        require(np.allclose(p,softmax,atol=2e-7,rtol=2e-5),'Probabilities must be unmodified raw six-class softmax')
        diagnostics['max_softmax_absolute_error']=float(np.abs(p-softmax).max())
    return diagnostics

def load_protocol(path):
    protocol=json.loads(path_in_repo(path).read_text());files={}
    require(protocol['schema']=='joint_native6_scoring_protocol_v1','Registered scorer schema')
    assessment=read_ref(protocol['assessment_ref'],files)
    require(assessment['method_id']==protocol['new_method_id']==NEW_METHOD,'Distinct new model identity')
    require(assessment['model_selection_after_TEST'] is False and assessment['selected_arm']=='structure','Frozen pre-TEST model')
    require(assessment['classes']==CLASSES and assessment['targets_per_parent']==10000,'Native-six endpoint')
    require(protocol['expected_methods']==BASELINES+[NEW_METHOD],'Exactly four original baselines and the new method')
    require(protocol['parents']==assessment['parents'] and len(protocol['parents'])==5,'Original five TEST parents')
    require(set(protocol['new_product_paths'])==set(protocol['parents']),'All five new product paths registered')
    for item in protocol['source_refs']:checked(item,files)
    checked(assessment['checkpoint'],files);checked(assessment['model_source'],files)
    return protocol,assessment,files

def prediction_barrier(protocol_path):
    protocol,assessment,files=load_protocol(protocol_path)
    historical=read_ref(protocol['historical_comparison_ref'],files)
    require(set(historical['expected_methods'])==set(BASELINES),'Unchanged historical comparison method set')
    plan=read_ref(historical['plan_ref'],files)
    parents=protocol['parents'];require(plan['test_parent_ids']==parents,'Frozen original TEST registry plan')
    require(not set(parents).intersection(plan['train_parent_ids']),'Disjoint TRAIN and TEST parents')
    registries={};probabilities={};evidence={};products={m:{} for m in protocol['expected_methods']}
    for parent in parents:
        products[NEW_METHOD][parent]=ref(path_in_repo(protocol['new_product_paths'][parent]))
        for method in BASELINES:products[method][parent]=historical['predictions'][method][parent]
        parsed={method:read_ref(products[method][parent],files) for method in protocol['expected_methods']}
        registry_ref=parsed[BASELINES[0]]['registry_ref'];registry=read_ref(registry_ref,files)
        require(registry['parent_id']==parent and registry['split']=='test' and registry['examples']==10000,'Original registry support')
        require(registry['labels_present'] is False and registry['query_truth_read'] is False and registry['native_path_names']==CLASSES,'Label-free canonical registry')
        require(registry['cohort_quotas']==plan['parents'][parent]['cohort_quotas'],'Frozen cohort quantities')
        sampling.registry_arrays(registry)
        records=sorted(registry['records'],key=lambda row:row['example_index'])
        require([row['example_index'] for row in records]==list(range(10000)),'Exact ordered canonical rows')
        require(len({(row['sample_id'],row['haplotype'],row['position_original_bp']) for row in records})==10000,'Unique biological target identities')
        require(all(sum(row['cohort']==c for row in records)==2000 for c in COHORTS),'Each original cohort has 2000 targets')
        registries[parent]=registry
        for method,product in parsed.items():
            require(product['parent_id']==parent and product['split']=='test' and product['rows']==10000 and product['native_path_names']==CLASSES,'Complete prediction product')
            require(product.get('query_truth_read',product.get('query_ancestry_truth_opened')) is False,'Truth-free product declaration')
            require(same_ref(product['registry_ref'],registry_ref),'Exactly matched registry across methods')
            if method==NEW_METHOD:
                require(product['schema']=='joint_model_native6_prediction_v1' and product['complete'] is True and product['model_id']==NEW_METHOD,'Explicit new prediction identity')
                require(same_ref(product['frozen_checkpoint_ref'],assessment['checkpoint']),'Pre-TEST frozen checkpoint lineage')
                checked(product['frozen_checkpoint_ref'],files);checked(product['freeze_ref'],files)
            arrays=load_arrays(product['predictions_ref'],files)
            diagnostics=validate_arrays(arrays,records,parent,method==NEW_METHOD)
            probabilities[method,parent]=arrays['probabilities']
            evidence.setdefault(method,{})[parent]=dict(product_ref=products[method][parent],predictions_ref=product['predictions_ref'],registry_ref=registry_ref,**diagnostics)
    payload=dict(protocol_ref=ref(protocol_path),assessment_ref=protocol['assessment_ref'],methods=protocol['expected_methods'],parents=parents,
        complete_product_count=25,predictions=evidence,authenticated_files=files,private_target_payloads_opened=False,
        unchanged_historical_product_count=20,canonical_targets_per_method=50000,probability_policy='Raw native-six probabilities; no clipping, masking, recalibration or renormalization.')
    return payload,protocol,historical,registries,probabilities

def seal(protocol_path,output):
    payload,*_=prediction_barrier(protocol_path)
    save(output,dict(schema='joint_native6_prediction_barrier_v1',verdict='pass',payload=payload))
    print(json.dumps(dict(status='sealed',barrier_ref=ref(output),private_truth_opened=False)))

def score(protocol_path,seal_path,output):
    sealed=json.loads(path_in_repo(seal_path).read_text())
    require(sealed['schema']=='joint_native6_prediction_barrier_v1' and sealed['verdict']=='pass','Successful saved all-model/all-parent barrier required')
    payload,protocol,historical,registries,probabilities=prediction_barrier(protocol_path)
    require(payload==sealed['payload'],'Prediction seal changed; no truth read permitted')
    # Only after the complete revalidated seal may private target files be read.
    parents=protocol['parents'];methods=protocol['expected_methods'];rows={m:{} for m in methods};truth_evidence={};truth_files={}
    for parent in parents:
        refs=historical['test_targets'][parent]
        generated=read_ref(refs['generation_ref'],truth_files,private=True)
        schema=read_ref(refs['target_schema_ref'],truth_files,private=True)
        require(generated['parent_id']==schema['parent_id']==parent and generated['split']==schema['split']=='test','Same-parent TEST truth provenance')
        base=path_in_repo(refs['generation_ref']['path']).parent
        for filename,key in [('truth/full_truth_intervals.npz','truth_ref'),('truth/target_schema.json','target_schema_ref')]:
            require(generated['files'][filename]['sha256']==refs[key]['sha256'] and path_in_repo(refs[key]['path'])==base/filename,'Original generated truth binding')
        truth=load_arrays(refs['truth_ref'],truth_files,private=True,maximum_bytes=16*2**30,members=['intervals_bp','native_paths_1based','query_raw_node_ids'])
        y=sampling.native_truth_labels(registries[parent],schema,truth);del truth
        records=sorted(registries[parent]['records'],key=lambda row:row['example_index'])
        cohorts=np.asarray([row['cohort'] for row in records])
        truth_evidence[parent]=dict(refs=refs,labels_sha256=hashlib.sha256(y.astype('<i8').tobytes()).hexdigest(),class_counts=np.bincount(y,minlength=6).tolist())
        for method in methods:
            p=probabilities[method,parent]
            groups={'overall':native_metrics.metrics(p,y,CLASSES)}
            groups.update({c:native_metrics.metrics(p[cohorts==c],y[cohorts==c],CLASSES) for c in COHORTS})
            rows[method][parent]=dict(overall=groups['overall'],cohorts={c:groups[c] for c in COHORTS})
            if method in BASELINES:
                saved=historical['rows'][method][parent]
                for group,actual in groups.items():
                    expected=saved['overall']['native6'] if group=='overall' else saved['cohorts'][group]['native6']
                    require(actual==expected,'Historical native metric changed: '+method+'/'+parent+'/'+group)
        print('Scored complete original parent '+parent,flush=True)
    summary={};paired={}
    for method in methods:
        summary[method]={}
        for group in ['overall']+COHORTS:
            values=[rows[method][parent]['overall'] if group=='overall' else rows[method][parent]['cohorts'][group] for parent in parents]
            summary[method][group]={name:native_metrics.parent_summary([v[name] for v in values]) for name in METRICS}
            summary[method][group]['per_class']={label:{name:native_metrics.parent_summary([v['per_class'][label][name] for v in values]) for name in ['precision','recall','F1']} for label in CLASSES}
            summary[method][group]['aggregate_confusion']=sum(np.asarray(v['confusion']) for v in values).tolist()
    for baseline in BASELINES:
        paired[baseline]={}
        for group in ['overall']+COHORTS:
            paired[baseline][group]={}
            for name in ['accuracy','macro_F1','Brier','raw_NLL']:
                pairs=[(rows[NEW_METHOD][parent]['overall'],rows[baseline][parent]['overall']) if group=='overall' else (rows[NEW_METHOD][parent]['cohorts'][group],rows[baseline][parent]['cohorts'][group]) for parent in parents]
                delta=[None if a[name] is None or b[name] is None else a[name]-b[name] for a,b in pairs]
                paired[baseline][group][name]=dict(direction=NEW_METHOD+' minus '+baseline,per_parent=delta,**native_metrics.parent_summary(delta))
    # Preserve all historical model products byte-for-byte; detect intervening changes.
    for item in payload['authenticated_files'].values():require(sha(path_in_repo(item['path']))==item['sha256'],'Sealed source or prediction changed during scoring')
    report=dict(schema='joint_native6_external_comparison_v1',protocol_ref=ref(protocol_path),prediction_barrier_ref=ref(seal_path),
        primary_endpoint='native6',native_class_order=CLASSES,methods=methods,parents=parents,targets_per_parent=10000,
        targets_per_cohort_parent=2000,all_25_predictions_authenticated_before_private_truth=True,historical_baselines_unchanged=True,
        original_native_metrics_reproduced_exactly=True,scoring_truth_evidence=truth_evidence,truth_files=truth_files,rows=rows,summary=summary,
        paired_differences=paired,source_refs=protocol['source_refs'],selection_after_TEST=False,
        uncertainty='Equal independent-parent mean and sample SD, ddof=1. No target-level independent-replicate claim.',
        scope='New separately registered native-six comparison. Existing workflows, competitors and barriers remain unchanged; other domains/endpoints are outside this comparison.',
        probability_convention='Raw native6 softmax/probabilities, no clipping or renormalization. raw_NLL null with infinite flag means at least one exact zero true-class probability.',
        published_native_reproduction_claim=False)
    save(output,report)
    print(json.dumps(dict(status='scored',comparison_ref=ref(output),summary={m:summary[m]['overall']['accuracy'] for m in methods}),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['seal','score']);parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);parser.add_argument('--seal')
    args=parser.parse_args();require(Path.cwd()==ROOT,'Run inside repository')
    if args.stage=='seal':seal(args.protocol,args.output)
    else:
        require(args.seal is not None,'Saved seal path required');score(args.protocol,args.seal,args.output)

if __name__=='__main__':main()
