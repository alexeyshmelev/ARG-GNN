"""Full canonical label/observation audit plus frozen-reader parity examples."""
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from .observations import MODEL_KEYS,load_observations
from .targets import TargetTable
from .site_dataset import read_ref,require
from research.redesign_20261028.data.window_data import TrainWindowDataset

HERE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029/data')


def main():
    started=time.perf_counter();cache=HERE/'cache_v1'
    manifest=json.loads((cache/'manifest_observations.json').read_text())
    plan=json.loads((HERE/'site_plan_v1.json').read_text())
    assessment=json.loads((HERE/'SOURCE_ASSESSMENT.json').read_text())
    truth={p:TargetTable(p,b) for p,b in assessment['parents'].items()};seen=set();counts={};parity=[]
    for row,site in zip(manifest['windows'],plan['sites']):
        for field in ('window_index','parent_id','parent_ordinal','native_site_id','example_indices','sample_indices'):
            require(row[field]==site[field],'Registered canonical mapping: '+field)
        labels=np.load(io.BytesIO(read_ref(cache,row['labels_ref'])),allow_pickle=False)
        require(np.array_equal(labels,truth[row['parent_id']].select(row['example_indices'])),'Exact unchanged canonical native labels')
        raw=read_ref(cache,row['observations_ref'])
        with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
            require(set(archive.files)==set(MODEL_KEYS),'Strict observation-only whitelist')
            packet={key:archive[key].copy() for key in archive.files}
        require(packet['focal_genotypes'].shape==(1,952),'Compact shared focal SNP')
        require(np.array_equal(packet['focal_sample_index'],row['sample_indices']),'Exact focal sample row order')
        require(np.array_equal(packet['focal_site_index'],np.zeros(len(labels),dtype=np.int64)),'All canonical rows gather same focal SNP')
        require(np.array_equal(packet['context_coordinate_local_bp'],site['context']['coordinates']),'Registered context occurrences')
        require(len(packet['context_tree_index'])==5 and packet['context_forced_focal'].sum()==1,'Five occurrences retained')
        require(np.array_equal(packet['context_quadrature_weight'],site['context']['weights']),'Registered source mass')
        ids={(row['parent_id'],int(i)) for i in row['example_indices']}
        require(not seen.intersection(ids),'Canonical row appears once');seen.update(ids)
        counts[row['parent_id']]=counts.get(row['parent_id'],0)+len(labels)
        if row['window_index'] in (0,415,830):
            old=TrainWindowDataset(row['parent_id'],assessment['parents'][row['parent_id']]['refs'])
            indices=row['example_indices']
            reference=old.batch(indices,context_specs={index:site['context'] for index in indices})
            compared=[]
            for key,value in packet.items():
                if key in ('focal_site_index','sample_id','tree_sample_id'):continue
                expected=reference[key]
                actual=value
                if key in ('focal_tree_index','focal_position_local_bp','focal_genotypes','focal_allele_count'):
                    actual=value[np.zeros(len(indices),dtype=np.int64)]
                require(np.array_equal(actual,expected),'Frozen reader parity: '+key)
                compared.append(key)
            require(np.array_equal(labels,reference['labels']),'Frozen reader label alignment')
            parity.append(dict(window_index=row['window_index'],parent_id=row['parent_id'],canonical_targets=len(indices),compared_fields=compared))
    require(len(seen)==50000 and len(manifest['windows'])==1245,'Complete canonical support')
    result=dict(status='passed',site_count=1245,target_count=len(seen),targets_by_parent=counts,
        all_labels_exact=True,all_observations_authenticated=True,strict_observation_whitelist=True,
        full_context_occurrences_verified=True,frozen_reader_parity=parity,TEST_payloads_opened=0,
        wall_seconds=time.perf_counter()-started)
    path=HERE/'OBSERVATION_VERIFICATION.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
