"""Freeze full canonical support and label-independent shared-site contexts."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from research.redesign_20261028.data.window_data import small_json, uniform_coordinate_context

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
HERE = ROOT / 'research/joint_model_20261029/data'


def main():
    assessment_path = HERE / 'SOURCE_ASSESSMENT.json'
    assessment = json.loads(assessment_path.read_text())
    policy = dict(schema='joint_model_shared_site_data_policy_v1', registered_utc=datetime.now(timezone.utc).isoformat(),
        source_assessment_sha256=hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
        root_authorization='2026-10-29: full canonical support, shared-site K5/50kb/seed20261029, initial16TRAIN000site profile; full export awaits measured size approval',
        immutable_original_simulations=True, immutable_original_labels=True, original_registry_splits_unchanged=True,
        classes=['Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA'],
        support=dict(fit=dict(parent_ordinals=[0,1],targets=33334,sites=830),
                     development=dict(parent_ordinals=[2],targets=16666,sites=415),
                     full_train=dict(parent_ordinals=[0,1,2],targets=50000,sites=1245)),
        context=dict(draws=5,radius_bp=50000.,seed=20261029,
                     seed_derivation='numpy.SeedSequence([20261029, parent_ordinal, native_site_id])',
                     parent_ordinal='TRAIN000=0, TRAIN001=1, TRAIN002=2',
                     random_proposal='four IID uniform physical coordinates in the clipped focal +/-50kb domain',
                     random_mass=.25,forced_focal_mass=0.,duplicate_draws='retain separate coordinate occurrences',
                     targets_at_same_site='one identical context and feature graph; sample index used only for output gathering',
                     SNP_extent='all SNPs in complete selected native tree intervals, possibly extending beyond50kb'),
        sampling=dict(target_subsampling=False,site_epoch='uniform random permutation of all sites in selected support',
                      fixed_site_minibatch=True,
                      training_loss='M/(S*N) times sum of target losses, where M is support sites,N support canonical rows,S actual minibatch sites',
                      microbatch_rule='Keep original minibatch M/(S*N) across all microbatches and perform one optimizer update',
                      evaluation='sum losses and counts over all canonical rows, then divide by N; no equal-site mean'),
        profile=dict(parent_ordinal=0,site_count=16,selection='16 evenly spaced indices in sorted native-site roster including endpoints'),
        full_export_authorized=False,isolated_project_cap_bytes=16*2**30, TEST_allowed=False)
    sites=[]
    for ordinal,parent in enumerate(sorted(assessment['parents'])):
        info=assessment['parents'][parent]
        registry=small_json(info['refs']['sampled_registry_ref'])
        groups=defaultdict(list)
        for row in registry['records']:groups[int(row['native_site_id'])].append(row)
        profile_indices=set(np.linspace(0,len(groups)-1,16,dtype=np.int64).tolist()) if ordinal==0 else set()
        for site_order,(site_id,rows) in enumerate(sorted(groups.items())):
            rows=sorted(rows,key=lambda r:r['example_index'])
            position=float(rows[0]['position_local_bp'])
            assert all(row['position_local_bp']==position for row in rows)
            spec=uniform_coordinate_context(np.random.default_rng(np.random.SeedSequence([20261029,ordinal,site_id])),
                    position,50000.,5,info['sequence_length_bp'])
            sites.append(dict(window_index=len(sites),parent_id=parent,parent_ordinal=ordinal,native_site_id=site_id,
                position_local_bp=position,position_original_bp=position+info['coordinate_offset_bp'],
                example_indices=[int(row['example_index']) for row in rows],
                sample_indices=[int(row['sample_index']) for row in rows],
                context={key:np.asarray(value).tolist() if isinstance(value,np.ndarray) else value for key,value in spec.items()},
                bounded_profile=site_order in profile_indices))
    assert len(sites)==1245 and sum(len(site['example_indices']) for site in sites)==50000
    plan=dict(schema='joint_model_shared_site_plan_v1',sites=sites)
    for name,value in [('DATA_PROTOCOL_v1.json',policy),('site_plan_v1.json',plan)]:
        path=HERE/name
        with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
        print(name,hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_size)


if __name__=='__main__':main()
