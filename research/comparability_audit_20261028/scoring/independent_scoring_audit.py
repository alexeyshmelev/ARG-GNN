"""Read-only independent arithmetic audit; only audit evidence is written.

The source scorer and its truth-join helpers are intentionally not imported.
Truth NPZ members are streamed in bounded buffers to extract selected rows.
"""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

OUT = Path('research/comparability_audit_20261028/scoring')
COMPARISON = Path('experiments/results/paper_gst_lai_8_20261024/paper_gst_lai_8__all_methods_all_test__score/native/comparison.json')
CLASSES = ['Ana_nonBAA', 'CHG_nonBAA', 'WHG', 'EHG', 'Ana_viaBAA', 'CHG_viaBAA']
COLLAPSE = np.array([0, 1, 2, 3, 0, 1])
COHORTS = ['present_day', 'Bronze_Age', 'BAA', 'Neo', 'Yam']
checked = {}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def authenticate(ref):
    p = Path(ref['path'])
    if str(p) not in checked:
        actual = digest(p)
        assert actual == ref['sha256'], ('SHA mismatch', str(p))
        checked[str(p)] = dict(sha256=actual, bytes=p.stat().st_size)
    assert checked[str(p)]['sha256'] == ref['sha256']
    return p

def read_json(ref):
    return json.loads(authenticate(ref).read_text())

def selected_truth(path, rows, cols):
    # A bounded stream avoids materializing the approximately 680 MB native array.
    with zipfile.ZipFile(path) as z:
        info = z.getinfo('native_paths_1based.npy')
        assert info.file_size < 1024**3
        with z.open(info) as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                assert version == (2, 0)
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            assert not fortran and len(shape) == 2 and shape[1] == 686
            assert dtype.kind in 'iu' and rows.max() < shape[0]
            row_bytes = shape[1] * dtype.itemsize
            cursor = 0
            labels = np.empty(len(rows), dtype=np.int64)
            for row in np.unique(rows):
                skip = int(row) * row_bytes - cursor
                while skip:
                    data = f.read(min(skip, 2**20))
                    assert data
                    skip -= len(data)
                    cursor += len(data)
                values = np.frombuffer(f.read(row_bytes), dtype=dtype)
                assert values.shape == (686,)
                selection = rows == row
                labels[selection] = values[cols[selection]] - 1
                cursor += row_bytes
    return labels, dict(native_array_shape=list(shape), selected_interval_rows=int(len(np.unique(rows))), max_discard_buffer_bytes=2**20)

def metric(p, y):
    prediction = np.argmax(p, axis=1)
    n = p.shape[1]
    conf = np.bincount(y*n+prediction, minlength=n*n).reshape(n,n)
    denom = conf.sum(axis=0) + conf.sum(axis=1)
    f1 = np.divide(2*np.diag(conf), denom, out=np.zeros(n), where=denom!=0)
    return dict(records=len(y), accuracy=float(np.mean(prediction==y)), macro_F1=float(f1.mean()), confusion=conf.tolist())

c = json.loads(COMPARISON.read_text())
plan = read_json(c['plan_ref'])
parents = plan['test_parent_ids']
assert len(parents) == 5 and len(plan['train_parent_ids']) == 3
assert not set(parents) & set(plan['train_parent_ids'])
assert set(c['test_targets']) == set(parents)
assert set(c['predictions']) == set(c['expected_methods'])
registries, probabilities, metadata = {}, {}, {}
for parent in parents:
    ref = None
    for method, items in c['predictions'].items():
        assert set(items) == set(parents)
        product = read_json(items[parent])
        assert product['parent_id'] == parent and product['split'] == 'test'
        assert product['native_path_names'] == CLASSES and product['rows'] == 10000
        assert product.get('query_truth_read',product.get('query_ancestry_truth_opened')) is False
        assert ref is None or product['registry_ref'] == ref
        ref = product['registry_ref']
        with np.load(authenticate(product['predictions_ref']), allow_pickle=False) as z:
            assert np.array_equal(z['example_indices'], np.arange(10000))
            ids_present = 'example_ids' in z.files
            if ids_present:
                assert np.array_equal(z['example_ids'], [parent+':'+str(i) for i in range(10000)])
            p = z['probabilities']
        assert p.shape == (10000,6) and np.isfinite(p).all()
        assert np.all(p>=0) and np.all(p<=1) and np.allclose(p.sum(axis=1),1,atol=1e-5,rtol=1e-5)
        probabilities[method,parent] = p
        metadata[method,parent] = dict(prediction_product_ref=items[parent], predictions_ref=product['predictions_ref'], registry_ref=ref, example_indices_exact=True, global_example_ids_present=ids_present, max_probability_sum_error=float(np.abs(p.sum(axis=1)-1).max()))
    registry = read_json(ref)
    records = sorted(registry['records'], key=lambda r:r['example_index'])
    assert registry['parent_id'] == parent and registry['split'] == 'test'
    assert registry['labels_present'] is False and registry['query_truth_read'] is False
    assert registry['examples'] == len(records) == 10000
    assert [r['example_index'] for r in records] == list(range(10000))
    assert registry['native_path_names'] == CLASSES
    assert registry['cohort_quotas'] == plan['parents'][parent]['cohort_quotas']
    assert len({(r['sample_id'],r['haplotype'],r['position_original_bp']) for r in records}) == 10000
    assert all(sum(r['cohort']==cohort for r in records)==2000 for cohort in COHORTS)
    registries[parent] = (registry,records)

# All twenty prediction products are authenticated before any truth file is read.
recomputed = {method:{} for method in c['predictions']}
truth_evidence = {}
for parent in parents:
    registry, records = registries[parent]
    refs = c['test_targets'][parent]
    generated = read_json(refs['generation_ref'])
    schema = read_json(refs['target_schema_ref'])
    truth_path = authenticate(refs['truth_ref'])
    assert generated['parent_id'] == schema['parent_id'] == parent
    assert generated['split'] == schema['split'] == 'test'
    for name,key in [('truth/full_truth_intervals.npz','truth_ref'),('truth/target_schema.json','target_schema_ref')]:
        assert generated['files'][name]['sha256'] == refs[key]['sha256']
        assert Path(refs[key]['path']) == Path(refs['generation_ref']['path']).parent/name
    assert schema['native_paths_1based'] == [1,2,3,4,5,6]
    assert schema['native_to_C4'] == COLLAPSE.tolist()
    with np.load(truth_path, allow_pickle=False) as z:
        intervals = z['intervals_bp']
        raw = z['query_raw_node_ids']
    assert intervals.ndim == 2 and intervals.shape[1] == 2
    assert intervals[0,0] == 0 and intervals[-1,1] == 200000000
    assert np.isfinite(intervals).all() and np.all(intervals[:,1]>intervals[:,0])
    assert np.array_equal(intervals[:-1,1],intervals[1:,0])
    assert np.array_equal(raw,schema['query_sample_indices']) and len(set(raw.tolist())) == 686
    by_key = {tuple(key):(int(sample),col) for col,(key,sample) in enumerate(zip(schema['query_keys'],raw))}
    assert len(by_key) == len(registry['query_identities']) == 686
    assert all(by_key[(q['sample_id'],q['haplotype'])][0] == q['sample_index'] for q in registry['query_identities'])
    assert all(by_key[(r['sample_id'],r['haplotype'])][0] == r['sample_index'] for r in records)
    positions = np.asarray([r['position_original_bp'] for r in records])
    assert np.all(positions>=0) and np.all(positions<200000000)
    rows = np.searchsorted(intervals[:,1],positions,side='right')
    cols = np.asarray([by_key[(r['sample_id'],r['haplotype'])][1] for r in records])
    y, stream = selected_truth(truth_path,rows,cols)
    assert np.all((y>=0)&(y<6))
    cohorts = np.asarray([r['cohort'] for r in records])
    assert np.all(np.isin(y[cohorts=='BAA'],[4,5]))
    assert np.all(np.isin(y[cohorts!='BAA'],[0,1,2,3]))
    truth_evidence[parent] = dict(refs=refs,labels_sha256=hashlib.sha256(y.astype('<i8').tobytes()).hexdigest(),class_counts=np.bincount(y,minlength=6).tolist(),stream=stream)
    for method in c['predictions']:
        p = probabilities[method,parent]
        p4 = np.column_stack([p[:,0]+p[:,4],p[:,1]+p[:,5],p[:,2],p[:,3]])
        results = {}
        for group in ['overall']+COHORTS:
            mask = np.ones(len(y),dtype=bool) if group=='overall' else cohorts==group
            result = dict(native6=metric(p[mask],y[mask]),derived_C4=metric(p4[mask],COLLAPSE[y[mask]]))
            saved = c['rows'][method][parent]['overall'] if group=='overall' else c['rows'][method][parent]['cohorts'][group]
            for endpoint,values in result.items():
                assert values['confusion'] == saved[endpoint]['confusion'],(method,parent,group,endpoint,'confusion')
                assert values['records'] == saved[endpoint]['records']
                for name in ['accuracy','macro_F1']:
                    assert abs(values[name]-saved[endpoint][name])<1e-14,(method,parent,group,endpoint,name)
            results[group] = result
        recomputed[method][parent] = results
    print('Verified', parent, stream['selected_interval_rows'], 'selected truth intervals', flush=True)

summary = {}
for method in c['predictions']:
    summary[method] = {}
    for group in ['overall']+COHORTS:
        summary[method][group] = {}
        for endpoint in ['native6','derived_C4']:
            result = {}
            for name in ['accuracy','macro_F1']:
                values = [recomputed[method][parent][group][endpoint][name] for parent in parents]
                result[name] = dict(mean=float(np.mean(values)),sample_SD=float(np.std(values,ddof=1)))
                for stat in ['mean','sample_SD']:
                    assert abs(result[name][stat]-c['summary'][method][group][endpoint][name][stat])<1e-14
            result['aggregate_confusion'] = sum(np.asarray(recomputed[method][parent][group][endpoint]['confusion']) for parent in parents).tolist()
            summary[method][group][endpoint] = result
output = dict(schema='independent_european_scoring_audit_v1',comparison_ref=dict(path=str(COMPARISON),sha256=digest(COMPARISON)),audit_script_sha256=digest(__file__),native_class_order=CLASSES,train_parent_ids=plan['train_parent_ids'],test_parent_ids=parents,predictions_authenticated_before_truth=True,checked_files=checked,prediction_checks={method:{parent:metadata[method,parent] for parent in parents} for method in c['predictions']},truth_evidence=truth_evidence,recomputed_summary=summary,all_240_confusion_accuracy_and_macro_F1_groups_match=True,limitations=['The actual prediction computations were not rerun. This verifies their saved identity contracts and arithmetic, not every internal feature-to-row transformation.','Only selected native truth values were streamed; the unrelated source_classes truth member was never opened.','Truth-free metadata and a scorer barrier do not constitute an operating-system file-access trace of fitting or prediction.'])
path = OUT/'independent_scoring_evidence.json'
path.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
print('PASS',path,'sha256',digest(path))
print(json.dumps({method:{endpoint:summary[method]['overall'][endpoint]['accuracy'] for endpoint in ['native6','derived_C4']} for method in summary},indent=2))
