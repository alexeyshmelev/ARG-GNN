"""Independently reconstruct every frozen shared-site target and coordinate plan."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

assessment=read(BASE/'data/SOURCE_ASSESSMENT.json')
protocol=read(BASE/'data/DATA_PROTOCOL_v1.json')
plan=read(BASE/'data/site_plan_v1.json')
assert protocol['source_assessment_sha256']==sha(BASE/'data/SOURCE_ASSESSMENT.json')
assert protocol['context']['seed']==20261029
assert protocol['context']['draws']==5 and protocol['context']['radius_bp']==50000.
assert protocol['sampling']['fixed_site_minibatch'] is True
assert protocol['sampling']['target_subsampling'] is False
assert protocol['TEST_allowed'] is False
expected=[]
for ordinal,parent in enumerate(sorted(assessment['parents'])):
    source=assessment['parents'][parent]
    registry=read(ROOT/source['refs']['sampled_registry_ref']['path'])
    groups=defaultdict(list)
    for record in registry['records']:
        groups[record['native_site_id']].append(record)
    profile_indices=set(np.linspace(0,len(groups)-1,16,dtype=np.int64).tolist()) if ordinal==0 else set()
    for site_order,(native_site_id,records) in enumerate(sorted(groups.items())):
        records.sort(key=lambda record:record['example_index'])
        position=records[0]['position_local_bp']
        low=max(0.,position-50000.)
        high=min(source['sequence_length_bp'],position+50000.)
        rng=np.random.default_rng(np.random.SeedSequence([20261029,ordinal,native_site_id]))
        coordinates=[float(position)]+rng.uniform(low,high,size=4).tolist()
        expected.append({
            'window_index':len(expected),'parent_id':parent,'parent_ordinal':ordinal,
            'native_site_id':native_site_id,'position_local_bp':position,
            'position_original_bp':position+source['coordinate_offset_bp'],
            'example_indices':[record['example_index'] for record in records],
            'sample_indices':[record['sample_index'] for record in records],
            'context':{'coordinates':coordinates,'weights':[0.,.25,.25,.25,.25],
                       'density':[0.]+[1./(high-low)]*4,'forced':[True,False,False,False,False],
                       'domain':[low,high],'kind':1},
            'bounded_profile':site_order in profile_indices,
        })
assert expected==plan['sites']
assert len(expected)==1245 and sum(len(row['example_indices']) for row in expected)==50000
assert sum(row['bounded_profile'] for row in expected)==16
assert protocol['support']=={
    'fit':{'parent_ordinals':[0,1],'targets':33334,'sites':830},
    'development':{'parent_ordinals':[2],'targets':16666,'sites':415},
    'full_train':{'parent_ordinals':[0,1,2],'targets':50000,'sites':1245},
}
receipt={
    'schema':'independent_shared_site_plan_review_v1','verdict':'pass',
    'audit_source_sha256':sha(Path(__file__)),
    'protocol_sha256':sha(BASE/'data/DATA_PROTOCOL_v1.json'),
    'site_plan_sha256':sha(BASE/'data/site_plan_v1.json'),
    'reconstructed_sites':len(expected),'reconstructed_target_ids':50000,
    'exact_target_order_sample_routing_and_all_context_draws_match':True,
    'independent_parent_internal_split':True,'label_dependent_context_selection':False,
    'SNP_extent_disclosure':'Complete selected native tree intervals can extend beyond the 50 kb coordinate radius.',
    'sampling_review':'Uniform fixed-site groups with M/(S*N) summed-target objective; preserve group denominator under microbatch splitting. Metrics use total target denominator.',
    'scope':'Only immutable registry/panel/plan metadata read. Adapter and exported packets remain to be reviewed.',
    'labels_read':False,'TEST_read':False,'reviewer_GPU_runs':0,
}
(BASE/'review/SITE_PLAN_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))

# The proposed log-age measure has a stable equivalent formula. Record a new
# draft-preparation failure without rerunning the old operator tests.
lower=1000.
upper=math.nextafter(lower,math.inf)
naive=math.log1p(upper/100.)-math.log1p(lower/100.)
stable=math.log1p((upper-lower)/(100.+lower))
assert naive==0. and stable>0.
finding={
    'schema':'independent_log_age_numerical_finding_v1',
    'lower':lower,'upper':upper,'positive_physical_duration':upper-lower,
    'subtractive_log_age_mass':naive,'stable_log_age_mass':stable,
    'issue':'Positive physical overlap with subtractive mass zero can cause 0/0 in branch normalization.',
    'required_fix':'Use log1p((upper-lower)/(AGE_SCALE+lower)) for both branch mass and overlap.',
    'model_agent_notified':True,'status':'Fix requested; final source qualification pending.',
}
(BASE/'review/LOG_AGE_NUMERICAL_FINDING.json').write_text(json.dumps(finding,indent=2)+'\n')
print(json.dumps(finding,indent=2))
