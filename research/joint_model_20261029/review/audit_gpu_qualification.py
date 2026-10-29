"""Read-only CPU audit of saved CPU/GPU complete-predictor comparisons."""
from pathlib import Path
import hashlib
import json
import numpy as np

BASE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029')
OUT=BASE/'root/gpu_qualification_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

protocol_path=BASE/'root/gpu_qualification_protocol.json'
protocol=read(protocol_path)
report=read(OUT/'result.json')
manifest_path=BASE/'data/cache_v1/manifest_observations.json'
manifest=read(manifest_path)
assert protocol['model_sha256']==report['model_sha256']==sha(BASE/'model/joint_model_v1.py')
assert report['harness_sha256']==sha(BASE/'root/qualify_gpu.py')
assert report['protocol_sha256']==sha(protocol_path)
assert protocol['observations_manifest_sha256']==report['observations_manifest_sha256']==sha(manifest_path)
assert protocol['model']=={'width':64,'layers':3,'classes':6}
assert [protocol[key] for key in ['logit_atol','logit_rtol','gradient_relative_l2','post_update_logit_atol','post_update_logit_rtol']]==[2e-5,2e-4,1e-4,3e-4,3e-4]
assert protocol['physical_gpu']==5
assert report['gpu_uuid']=='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
assert report['process_title']=='AI Auto Research Pipeline Test'
assert report['foreign_before']==report['foreign_after']==[]
assert report['TEST_opened'] is False and report['weights_retained_for_training'] is False
assert report['optimizer_updates_per_mode_per_device']==1 and report['passed'] is True
rows=[manifest['windows'][index] for index in protocol['window_indices']]
refs=[row[key] for row in rows for key in ['observations_ref','labels_ref']]
assert refs==report['sources']
assert sum(row['canonical_examples'] for row in rows)==report['targets']==399
for ref in refs:
    path=manifest_path.parent/ref['path']
    assert sha(path)==ref['sha256'] and path.stat().st_size==ref['bytes']
assert all(row['parent_ordinal'] in [0,1] for row in rows)

comparisons={}
for comparison in report['comparisons']:
    mode=comparison['mode']
    with np.load(OUT/(mode+'.npz'),allow_pickle=False) as archive:
        assert set(archive.files)=={'cpu_logits','gpu_logits','cpu_gradients','gpu_gradients','cpu_post','gpu_post'}
        saved={key:archive[key].copy() for key in archive.files}
    assert all(value.dtype==np.float32 and np.isfinite(value).all() for value in saved.values())
    assert all(saved[key].shape==(399,6) for key in ['cpu_logits','gpu_logits','cpu_post','gpu_post'])
    assert saved['cpu_gradients'].shape==saved['gpu_gradients'].shape==(133202,)
    initial_difference=saved['gpu_logits'].astype(np.float64)-saved['cpu_logits'].astype(np.float64)
    post_difference=saved['gpu_post'].astype(np.float64)-saved['cpu_post'].astype(np.float64)
    gradient_difference=saved['gpu_gradients'].astype(np.float64)-saved['cpu_gradients'].astype(np.float64)
    norm=float(np.linalg.norm(saved['cpu_gradients'].astype(np.float64)))
    assert norm>0
    relative=float(np.linalg.norm(gradient_difference)/norm)
    initial_pass=bool(np.all(np.abs(initial_difference)<=protocol['logit_atol']+protocol['logit_rtol']*np.abs(saved['cpu_logits'])))
    post_pass=bool(np.all(np.abs(post_difference)<=protocol['post_update_logit_atol']+protocol['post_update_logit_rtol']*np.abs(saved['cpu_post'])))
    gradient_pass=relative<=protocol['gradient_relative_l2']
    assert initial_pass and post_pass and gradient_pass
    assert np.isclose(relative,comparison['gradient_relative_l2'],rtol=1e-5,atol=5e-12)
    for key,value in [('logit_max_abs',np.abs(initial_difference).max()),
                      ('post_update_logit_max_abs',np.abs(post_difference).max()),
                      ('gradient_max_abs',np.abs(gradient_difference).max())]:
        assert value==comparison[key]
    for device,index in [('cpu',0),('gpu',1)]:
        measured_norm=float(np.linalg.norm(saved[device+'_gradients'].astype(np.float64)))
        assert np.isclose(measured_norm,comparison['cpu_gpu_preclip_norm'][index],rtol=1e-6,atol=1e-8)
    for prefix,field in [('', 'before_argmax_changes'),('_post','after_argmax_changes')]:
        left=saved['cpu_post' if prefix else 'cpu_logits']
        right=saved['gpu_post' if prefix else 'gpu_logits']
        changes=int((left.argmax(1)!=right.argmax(1)).sum())
        assert changes==comparison[field]==0
    assert comparison['passed'] and all(comparison['checks'].values())
    assert np.isfinite(comparison['cpu_gpu_forward_backward_seconds']).all()
    comparisons[mode]={
        'pass':True,'array_sha256':sha(OUT/(mode+'.npz')),
        'max_absolute_logit_difference':float(np.abs(initial_difference).max()),
        'full_gradient_relative_l2_float64_reduction':relative,
        'max_absolute_gradient_difference':float(np.abs(gradient_difference).max()),
        'post_update_max_absolute_logit_difference':float(np.abs(post_difference).max()),
        'initial_and_post_update_argmax_changes':0,
        'gradient_presence_runtime_check_passed':comparison['checks']['gradient_presence'],
    }
assert set(comparisons)=={'no_cross','leaf_only','structure'}
receipt={
    'schema':'independent_complete_gpu_qualification_review_v1','verdict':'pass',
    'audit_source_sha256':sha(__file__),'source_result_sha256':sha(OUT/'result.json'),
    'protocol_sha256':sha(protocol_path),'model_sha256':report['model_sha256'],
    'harness_sha256':report['harness_sha256'],'manifest_sha256':sha(manifest_path),
    'targets':399,'window_indices':protocol['window_indices'],'comparisons':comparisons,
    'largest_initial_logit_error':max(value['max_absolute_logit_difference'] for value in comparisons.values()),
    'largest_gradient_relative_l2':max(value['full_gradient_relative_l2_float64_reduction'] for value in comparisons.values()),
    'largest_post_update_logit_error':max(value['post_update_max_absolute_logit_difference'] for value in comparisons.values()),
    'all_saved_numerics_finite':True,'before_after_foreign_compute_empty':True,
    'limitations':[
        'Gradient presence arrays were not saved; identical presence is supported by the pinned harness runtime comparison and saved boolean, not an independent per-tensor array reconstruction.',
        'Post-update comparison concerns logits, not parameter equality or longer AdamW trajectory equivalence.',
        'This qualification tests the complete five-tree no_cross path; a new focal-only executor needs its own GPU parity qualification before use.',
        'Only before/after foreign-compute lists were retained; there is no continuous activity trace.',
        'One un-warmed forward/backward per mode is numerical evidence only, not throughput evidence.',
    ],
    'reviewer_GPU_runs':0,'reviewer_optimizer_updates':0,'TEST_read':False,
}
(BASE/'review/GPU_QUALIFICATION_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
