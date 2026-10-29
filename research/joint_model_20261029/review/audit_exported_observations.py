"""Independent all-packet metadata/target audit; no fitting or TEST access."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from research.redesign_20261028.data.window_data import COHORTS

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'
CACHE=BASE/'data/cache_v1'
MANIFEST= CACHE/'manifest_observations.json'
MODEL_KEYS={'parent','node_time','node_tree_index','leaf_node_index','leaf_features',
 'tree_interval_local_bp','tree_window_index','tree_core_mask','focal_tree_index',
 'focal_sample_index','focal_site_index','focal_position_local_bp','focal_genotypes',
 'focal_allele_count','reference_class','sample_age','observed_cohort','query_mask',
 'site_tree_index','site_position_local_bp','site_genotypes','site_allele_count',
 'sample_id','tree_sample_id','context_coordinate_local_bp','context_tree_index',
 'context_window_index','context_quadrature_weight','context_proposal_density_per_bp',
 'context_forced_focal','window_context_domain_local_bp','window_context_kind'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

started=time.perf_counter()
assert sha(MANIFEST)=='43edc132a24c1c57d41e34d82315c4754a2185991c1e443a0b7fe3e4aa95b54b'
manifest=read(MANIFEST)
plan=read(BASE/'data/site_plan_v1.json')['sites']
assert manifest['complete'] and manifest['labels_separate'] and manifest['TEST_allowed'] is False
assert len(manifest['windows'])==len(plan)==1245
assert sha(BASE/'data/site_plan_v1.json')==manifest['plan_sha256']
assert sha(BASE/'data/DATA_PROTOCOL_v1.json')==manifest['protocol_sha256']
for ref in manifest['source_refs'].values():
    assert sha(CACHE/ref['path'])==ref['sha256']
old_helper=ROOT/'research/redesign_20261028/data/window_data.py'
assert sha(old_helper)=='a1788ea0a33335450bfb43c689aa3c598070f07b28c40930be3fb086f42aa309'

panels={};truth={};expected_meta={}
for parent,source in manifest['source_bindings'].items():
    assert parent in [f'ancestralpaths_european_train_{i:03d}' for i in range(3)]
    panel_path=ROOT/source['panel_ref']['path']
    assert sha(panel_path)==source['panel_ref']['sha256']
    panel=read(panel_path);panels[parent]=panel
    labels_path=ROOT/source['target_array_ref']['path']
    assert sha(labels_path)==source['target_array_ref']['sha256']
    with np.load(labels_path,allow_pickle=False) as labels:
        truth[parent]=labels['native_labels_0based'].copy()
    assert truth[parent].shape==(source['canonical_examples'],)
    metadata=np.zeros((len(panel),17),dtype=np.float32)
    for sample,row in enumerate(panel):
        metadata[sample,COHORTS.index(row['observed_cohort'])]=1.
        metadata[sample,9+(4 if row['observed_reference_class'] is None else row['observed_reference_class'])]=1.
        metadata[sample,14]=np.log1p(row['age_generations'])/np.log(10001.)
        metadata[sample,15]=float(row['role']=='query')
        metadata[sample,16]=float(row['role']!='query')
    expected_meta[parent]=metadata

all_ids=[];class_counts=Counter();outside_coordinate_domain=0;raw_bytes=0
for row,site in zip(manifest['windows'],plan):
    for key in ['window_index','parent_id','parent_ordinal','native_site_id','example_indices','sample_indices']:
        assert row[key]==site[key]
    parent=site['parent_id'];panel=panels[parent]
    assert row['canonical_examples']==len(site['example_indices'])
    for kind in ['observations_ref','labels_ref']:
        ref=row[kind];path=CACHE/ref['path']
        assert path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256']
        raw_bytes+=ref['bytes']
    with np.load(CACHE/row['observations_ref']['path'],allow_pickle=False) as archive:
        assert set(archive.files)==MODEL_KEYS
        a={key:archive[key].copy() for key in archive.files}
    labels=np.load(CACHE/row['labels_ref']['path'],allow_pickle=False)
    np.testing.assert_array_equal(labels,truth[parent][site['example_indices']])
    assert labels.dtype==np.int64 and labels.shape==(row['canonical_examples'],)
    all_ids.extend((parent,index) for index in site['example_indices'])
    class_counts.update(labels.tolist())
    np.testing.assert_array_equal(a['leaf_features'],expected_meta[parent])
    expected_ids=np.asarray([f"{parent}:{record['sample_id']}:{record['haplotype']}" for record in panel])
    np.testing.assert_array_equal(a['sample_id'],expected_ids)
    np.testing.assert_array_equal(a['tree_sample_id'],np.broadcast_to(expected_ids,a['leaf_node_index'].shape))
    np.testing.assert_array_equal(a['sample_age'],[record['age_generations'] for record in panel])
    np.testing.assert_array_equal(a['reference_class'],[-1 if record['observed_reference_class'] is None else record['observed_reference_class'] for record in panel])
    np.testing.assert_array_equal(a['query_mask'],[record['role']=='query' for record in panel])
    np.testing.assert_array_equal(a['observed_cohort'],[COHORTS.index(record['observed_cohort']) for record in panel])
    assert np.all(a['reference_class'][a['query_mask']]==-1)
    np.testing.assert_array_equal(a['focal_sample_index'],site['sample_indices'])
    np.testing.assert_array_equal(a['focal_site_index'],np.zeros(len(labels),dtype=np.int64))
    np.testing.assert_array_equal(a['focal_position_local_bp'],[site['position_local_bp']])
    mapping={'coordinates':'context_coordinate_local_bp','weights':'context_quadrature_weight',
             'density':'context_proposal_density_per_bp','forced':'context_forced_focal'}
    for field,array_key in mapping.items():np.testing.assert_array_equal(a[array_key],site['context'][field])
    np.testing.assert_array_equal(a['window_context_domain_local_bp'],[site['context']['domain']])
    np.testing.assert_array_equal(a['window_context_kind'],[site['context']['kind']])
    assert np.all(a['context_window_index']==0) and np.all(a['tree_window_index']==0)
    t,h=a['leaf_node_index'].shape
    assert h==952 and t==row['unique_native_trees'] and len(a['parent'])==row['nodes']
    assert np.all(a['node_tree_index'][a['leaf_node_index']]==np.arange(t)[:,None])
    np.testing.assert_array_equal(a['node_time'][a['leaf_node_index']],np.broadcast_to(a['sample_age'],(t,h)))
    branch=a['parent']>=0
    assert np.all(a['node_tree_index'][a['parent'][branch]]==a['node_tree_index'][branch])
    assert np.all(a['node_time'][a['parent'][branch]]>=a['node_time'][branch])
    intervals=a['tree_interval_local_bp'];context_tree=a['context_tree_index']
    x=a['context_coordinate_local_bp']
    assert np.all((x>=intervals[context_tree,0])&(x<intervals[context_tree,1]))
    st=a['site_tree_index'];sp=a['site_position_local_bp']
    assert len(st)==row['site_occurrences']
    assert np.all((sp>=intervals[st,0])&(sp<intervals[st,1]))
    ft=int(a['focal_tree_index'][0])
    assert ft==context_tree[0]
    matching=np.flatnonzero((st==ft)&(sp==site['position_local_bp']))
    assert len(matching)==1
    np.testing.assert_array_equal(a['focal_genotypes'][0],a['site_genotypes'][matching[0]])
    np.testing.assert_array_equal(a['focal_allele_count'][0],a['site_allele_count'][matching[0]])
    low,high=site['context']['domain']
    outside_coordinate_domain+=int(np.any((sp<low)|(sp>=high)))
    if (row['window_index']+1)%400==0:print('verified_packets',row['window_index']+1,flush=True)
assert len(all_ids)==len(set(all_ids))==50000
for parent,labels in truth.items():
    assert sorted(index for name,index in all_ids if name==parent)==list(range(len(labels)))
receipt={
    'schema':'independent_all_exported_observations_review_v1','verdict':'pass',
    'audit_source_sha256':sha(__file__),'manifest_sha256':sha(MANIFEST),
    'authenticated_packets':1245,'authenticated_labels':50000,
    'all_target_ids_unique_and_complete':True,'all_targets_match_original_canonical_labels':True,
    'all_observed_metadata_and_haplotype_ids_match_source_panels':True,
    'query_ancestry_inputs_all_unknown':True,'focal_SNP_gathers_match_native_interval_pool':True,
    'all_registered_coordinates_and_target_gathers_match':True,
    'native_label_counts':[class_counts[i] for i in range(6)],
    'windows_with_SNPs_outside_coordinate_domain':outside_coordinate_domain,
    'raw_packet_and_label_bytes':raw_bytes,
    'old_reader_helper_sha256':sha(old_helper),
    'source_provenance_note':'Old reader helper hash verified against prior independent review; data agent will archive it in the additive prepared manifest without rewriting raw export evidence.',
    'seconds':time.perf_counter()-started,'TEST_read':False,'reviewer_GPU_runs':0,'reviewer_fitting':False,
    'limits':'No full rescans of large tree/VCF inputs. Original extractor joins are inspected; old tree/VCF reader correctness was previously qualified. Prepared matrices and dataset/trainer remain separate gates.',
}
(BASE/'review/EXPORTED_OBSERVATIONS_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
