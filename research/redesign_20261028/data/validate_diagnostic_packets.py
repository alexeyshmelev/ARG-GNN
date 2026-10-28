"""Fresh readback of every exported packet, label, identity and context draw."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from research.redesign_20261028.data.window_data import small_json,KastoreArrays,path_for
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/redesign_20261028'
OUT=HERE/'diagnostic_packets'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def need(condition,message):
    if not condition:raise ValueError(message)


def main():
    started=time.perf_counter();manifest_raw=(OUT/'manifest.json').read_bytes()
    manifest=json.loads(manifest_raw);completion=json.loads((OUT/'completion.json').read_text())
    need(completion['complete'] and completion['manifest_sha256']==sha(manifest_raw),'Completion manifest binding')
    need(manifest['protocol_sha256']==sha((HERE/'root/pilot_protocol.json').read_bytes()),'Protocol binding')
    need(manifest['status']=='complete' and manifest['observation_NPZ_contains_labels'] is False,'Completed observations only')
    for ref in manifest['source_refs'].values():
        need(sha((OUT/ref['archived_path']).read_bytes())==ref['sha256'],'Archived source checksum')
    sources={}
    for parent,binding in manifest['source_bindings'].items():
        registry=small_json(binding['refs']['sampled_registry_ref'])
        panel=small_json(binding['panel_ref'])
        target_meta=small_json(binding['refs']['training_targets_ref'])
        target_raw=path_for(target_meta['arrays_ref']).read_bytes()
        need(sha(target_raw)==target_meta['arrays_ref']['sha256'],'Original target checksum')
        with np.load(path_for(target_meta['arrays_ref']),allow_pickle=False) as z:
            labels=z['native_labels_0based'].copy()
        sources[parent]=dict(records={r['example_index']:r for r in registry['records']},panel=panel,
            labels=labels,sequence_length=binding['inferred_sequence_length_bp'],
            columns=KastoreArrays(path_for(binding['tree_ref'])))
    verified=[]
    specifications=[('fit',128,20261028,0,33334),('development',64,20261029,1,16666)]
    for split,count,seed,code,total in specifications:
        rows=manifest[split];need(len(rows)==count,'Exact declared split draw count')
        ids=np.random.default_rng(seed).integers(0,total,size=count)
        for i,(r,gid) in enumerate(zip(rows,ids)):
            if split=='fit':
                expected_parent=f'ancestralpaths_european_train_{int(gid>=16667):03d}'
                expected_index=int(gid%16667)
            else:expected_parent='ancestralpaths_european_train_002';expected_index=int(gid)
            need((r['draw_index'],r['parent_id'],r['example_index'])==(i,expected_parent,expected_index),
                 'Deterministic target sampling or duplicate multiplicity mismatch')
            packet_path=OUT/r['packet_path'];label_path=OUT/r['label_path']
            need(packet_path.resolve().is_relative_to(OUT) and label_path.resolve().is_relative_to(OUT),'Output path scope')
            need(sha(packet_path.read_bytes())==r['packet_sha256'],'Packet hash')
            need(sha(label_path.read_bytes())==r['label_sha256'],'Label hash')
            source=sources[r['parent_id']];record=source['records'][expected_index]
            label=np.load(label_path,allow_pickle=False)
            need(label.shape==() and label.dtype==np.int64 and int(label)==r['label']==int(source['labels'][expected_index]),
                 'Separate scalar label join')
            with np.load(packet_path,allow_pickle=False) as z:
                need('labels' not in z.files and not any('truth' in key for key in z.files),'Truth in observation NPZ')
                p={k:z[k] for k in z.files}
            need(p['focal_example_index'].tolist()==[expected_index],'Focal ID')
            need(p['focal_sample_index'].tolist()==[record['sample_index']],'Focal sample')
            need(p['focal_native_site_id'].tolist()==[record['native_site_id']],'Focal native site')
            need(p['focal_position_local_bp'].tolist()==[record['position_local_bp']],'Focal local coordinate')
            need(p['focal_position_original_bp'].tolist()==[record['position_original_bp']],'Focal original coordinate')
            focal=float(record['position_local_bp']);left=max(0.,focal-50000);right=min(source['sequence_length'],focal+50000)
            rng=np.random.default_rng(np.random.SeedSequence([20261030,code,i]))
            expected_coords=np.r_[focal,rng.uniform(left,right,size=4)]
            np.testing.assert_array_equal(p['context_coordinate_local_bp'],expected_coords)
            np.testing.assert_array_equal(p['context_quadrature_weight'],[0,.25,.25,.25,.25])
            np.testing.assert_array_equal(p['context_proposal_density_per_bp'],[0]+[1/(right-left)]*4)
            np.testing.assert_array_equal(p['context_forced_focal'],[True,False,False,False,False])
            np.testing.assert_array_equal(p['window_context_domain_local_bp'],[[left,right]])
            need(np.all(p['context_window_index']==0) and p['window_context_kind'].tolist()==[1], 'One coordinate window')
            np.testing.assert_array_equal(p['tree_interval_original_bp'],p['tree_interval_local_bp']+p['coordinate_offset_bp'])
            np.testing.assert_array_equal(p['site_position_original_bp'],p['site_position_local_bp']+p['coordinate_offset_bp'])
            intervals=p['tree_interval_local_bp'];mapping=p['context_tree_index']
            need(np.all(intervals[mapping,0]<=expected_coords) and np.all(expected_coords<intervals[mapping,1]),
                 'Containing native intervals')
            need(int(p['tree_core_mask'].sum())==1 and p['tree_core_mask'][p['focal_tree_index'][0]],'Focal core mask')
            need(np.all(p['tree_window_index']==0),'One forest context per draw')
            h=len(source['panel']);sample_ids=np.array([row['observed_node_id'] for row in source['panel']])
            need(p['leaf_features'].shape==(h,17) and p['focal_genotypes'].shape==(1,h),'Complete observed panel')
            np.testing.assert_array_equal(p['reference_class'],[-1 if r0['observed_reference_class'] is None else r0['observed_reference_class'] for r0 in source['panel']])
            np.testing.assert_array_equal(p['sample_age'],[r0['age_generations'] for r0 in source['panel']])
            need(np.all(p['reference_class'][p['query_mask']]==-1),'Query ancestry exposed')
            children=np.flatnonzero(p['parent']>=0);parents=p['parent'][children]
            np.testing.assert_array_equal(p['edge_index'],np.stack([children,parents]))
            np.testing.assert_array_equal(p['node_tree_index'][children],p['node_tree_index'][parents])
            need(np.all(p['node_time'][parents]>p['node_time'][children]),'Tree times')
            native_sites=source['columns']['sites/position']
            for tid,(a,b) in enumerate(intervals):
                leaves=p['leaf_node_index'][tid]
                np.testing.assert_array_equal(p['node_sample_index'][leaves],np.arange(h))
                np.testing.assert_array_equal(p['node_original_id'][leaves],sample_ids)
                begin,end=np.searchsorted(native_sites,[a,b],side='left')
                entries=np.flatnonzero(p['site_tree_index']==tid)
                np.testing.assert_array_equal(p['site_native_id'][entries],np.arange(begin,end))
                np.testing.assert_array_equal(p['site_position_local_bp'][entries],native_sites[begin:end])
            entries=np.flatnonzero((p['site_native_id']==record['native_site_id']) &
                                   (p['site_tree_index']==p['focal_tree_index'][0]))
            need(len(entries)==1,'Exactly one focal observed site in core')
            np.testing.assert_array_equal(p['focal_genotypes'][0],p['site_genotypes'][entries[0]])
            verified.append((split,i,r['parent_id'],expected_index,int(label)))
    size=sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())
    need(size<=2**30,'Output exceeds 1 GiB')
    report=dict(schema='diagnostic_packet_full_readback_v1',status='pass',manifest_sha256=sha(manifest_raw),
        packets_checked=len(verified),labels_checked=len(verified),all_target_ids_and_context_seeds_match=True,
        all_observation_files_exclude_labels=True,all_retained_tree_sites_checked=True,
        source_tree_topology_algorithm='independently qualified bounded replacement reader; no repeated whole-source hash',
        total_output_bytes=size,elapsed_seconds=time.perf_counter()-started,
        fit_class_counts=np.bincount([r[4] for r in verified if r[0]=='fit'],minlength=6).tolist(),
        development_class_counts=np.bincount([r[4] for r in verified if r[0]=='development'],minlength=6).tolist())
    destination=HERE/'data/diagnostic_packet_validation.json'
    with destination.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
