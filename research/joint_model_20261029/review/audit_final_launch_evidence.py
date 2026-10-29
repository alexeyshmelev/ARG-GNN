"""CPU-only audit of saved no-cross GPU arrays and complete prepared-cache metadata."""
from dataclasses import fields
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from research.joint_model_20261029.model import joint_model_v1 as model
from research.joint_model_20261029.data.site_dataset import CachedSiteDataset

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def authenticate(root,ref):
    path=root/ref['path']
    assert path.stat().st_size==ref['bytes'] and sha(path)==ref['sha256']

def same(a,b):
    assert type(a) is type(b)
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and a.layout==b.layout and a.shape==b.shape
        if a.layout==torch.sparse_csr:
            for name in ('crow_indices','col_indices','values'):same(getattr(a,name)(),getattr(b,name)())
        else:assert torch.equal(a,b)
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for left,right in zip(a,b):same(left,right)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    else:assert a==b


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    output=BASE/'root/no_cross_gpu_view_v1'
    result=read(output/'result.json')
    assert result['passed'] is True and result['optimizer_steps']==0
    assert result['model_sha256']==sha(BASE/'model/joint_model_v1.py')
    assert result['view_sha256']==sha(BASE/'model/no_cross_view.py')
    assert result['harness_sha256']==sha(BASE/'root/qualify_no_cross_gpu.py')
    assert result['observation_manifest_sha256']==sha(BASE/'data/cache_v1/manifest_observations.json')
    thresholds=result['thresholds']
    assert thresholds=={'logit_atol':2e-5,'logit_rtol':2e-4,'gradient_relative_l2':1e-4}
    with np.load(output/'numerics.npz',allow_pickle=False) as archive:
        saved={key:archive[key].copy() for key in archive.files}
    assert set(saved)=={'full_logits','focal_logits','full_gradients','focal_gradients','full_presence','focal_presence'}
    for name,value in saved.items():
        assert value.dtype==(np.bool_ if 'presence' in name else np.float32)
        assert np.isfinite(value).all()
        assert value.shape==((55,) if 'presence' in name else (133202,) if 'gradients' in name else (398,6))
    assert np.array_equal(saved['full_presence'],saved['focal_presence'])
    assert int(saved['full_presence'].sum())==40
    error=saved['focal_logits'].astype(np.float64)-saved['full_logits'].astype(np.float64)
    assert np.all(np.abs(error)<=thresholds['logit_atol']+thresholds['logit_rtol']*np.abs(saved['full_logits']))
    grad_error=saved['focal_gradients'].astype(np.float64)-saved['full_gradients'].astype(np.float64)
    relative=float(np.linalg.norm(grad_error)/np.linalg.norm(saved['full_gradients'].astype(np.float64)))
    assert relative<=thresholds['gradient_relative_l2']
    assert np.isclose(relative,result['gradient_relative_l2'],rtol=1e-5,atol=5e-12)
    assert float(np.abs(error).max())==result['logit_max_abs']
    assert int((saved['focal_logits'].argmax(1)!=saved['full_logits'].argmax(1)).sum())==result['argmax_changes']==0
    assert result['targets']==398 and result['window_indices']==[0,207]
    view_review=dict(verdict='pass',source_result_sha256=sha(output/'result.json'),array_sha256=sha(output/'numerics.npz'),
        harness_sha256=result['harness_sha256'],targets=398,logit_max_abs=float(np.abs(error).max()),
        full_gradient_relative_l2_float64_reduction=relative,gradient_presence_identical=True,
        active_parameter_tensors=40,total_parameter_tensors=55,argmax_changes=0,
        limits=['Paired same-GPU forward and CE-gradient comparison, no optimizer update.',
                'No activity trace or GPU UUID receipt; harness pins the expected CUDA visibility. Core CPU/GPU qualification is recorded separately.'])
    manifest_path=BASE/'data/cache_v1/manifest.json';manifest=read(manifest_path)
    assert sha(manifest_path)=='a9a2839b450ee7bf12f90bc269a257bb759b79fdd6e8a93fa8e7eb4c0737e253'
    assert manifest['complete'] and manifest['site_count']==1245 and manifest['target_count']==50000
    for ref in list(manifest['source_refs'].values())+[manifest['model_source_ref'],manifest['observations_manifest_ref']]:authenticate(manifest_path.parent,ref)
    raw=read(manifest_path.parent/manifest['observations_manifest_ref']['path'])
    plan=read(manifest_path.parent/manifest['source_refs']['site_plan_v1.json']['path'])
    seen=set();prepared_paths=set()
    for row,original,registered in zip(manifest['windows'],raw['windows'],plan['sites']):
        for name in ('window_index','parent_id','parent_ordinal','native_site_id','example_indices','sample_indices'):
            assert row[name]==original[name]==registered[name]
        for name in ('observations_ref','labels_ref'):assert row[name]==original[name]
        assert row['canonical_examples']==len(row['example_indices'])
        ids={(row['parent_id'],index) for index in row['example_indices']}
        assert len(ids)==row['canonical_examples'] and not seen.intersection(ids)
        seen.update(ids)
        ref=row['prepared_ref'];path=manifest_path.parent/ref['path']
        assert path.stat().st_size==ref['bytes'] and str(path) not in prepared_paths
        prepared_paths.add(str(path))
    assert len(seen)==50000 and len(prepared_paths)==1245
    assert sum(row['prepared_ref']['bytes'] for row in manifest['windows'])==manifest['prepared_bytes']==9011935132
    assert manifest['support']=={'fit':{'parent_ordinals':[0,1],'targets':33334,'sites':830},'development':{'parent_ordinals':[2],'targets':16666,'sites':415},'full_train':{'parent_ordinals':[0,1,2],'targets':50000,'sites':1245}}
    runtime=read(BASE/'data/RUNTIME_DATA_RECEIPT_v1.json')
    for ref in runtime['sources']+runtime['verification_refs']+[runtime['cache_manifest_ref']]:authenticate(ROOT,ref)
    verification=read(BASE/'data/SITE_CACHE_VERIFICATION.json')
    assert verification['status']=='passed' and verification['workers_zero_and_four_exact']
    assert verification['TEST_payloads_opened']==0
    records=verification['real_loader_epochs']
    assert all(row['sites']==830 and row['targets']==33334 and row['batches']==104 for row in records)
    assert records[0]['tensor_label_order_sha256']==records[1]['tensor_label_order_sha256']
    ds=CachedSiteDataset(manifest_path,split='full_train',cache_bytes=0)
    checked=[]
    for index in [0,207,1244]:
        row=manifest['windows'][index]
        authenticate(manifest_path.parent,row['observations_ref'])
        with np.load(manifest_path.parent/row['observations_ref']['path'],allow_pickle=False) as archive:
            recomputed=model.prepare_observations({key:archive[key].copy() for key in archive.files})
        restored=ds[index].observations
        for field in fields(restored):same(getattr(restored,field.name),getattr(recomputed,field.name))
        checked.append(dict(window_index=index,targets=row['canonical_examples'],all_batch_fields_exact=True))
    cache_review=dict(verdict='pass',manifest_sha256=sha(manifest_path),runtime_receipt_sha256=sha(BASE/'data/RUNTIME_DATA_RECEIPT_v1.json'),
        verification_source_sha256=sha(BASE/'data/verify_site_cache.py'),verification_receipt_sha256=sha(BASE/'data/SITE_CACHE_VERIFICATION.json'),
        sites=1245,targets=50000,prepared_bytes=manifest['prepared_bytes'],support=manifest['support'],
        all_metadata_and_source_bindings_authenticated=True,representative_raw_to_prepared_recomputations=checked,
        full_fit_worker_equivalence='Reviewed author source and authenticated complete 830-site epoch tensor/label/order digest equality; not rerun.',
        limits=['All prepared file paths/sizes and metadata checked, three tensors independently recomputed; avoided duplicate full 9 GB readback.',
               'Worker integration timing includes tensor hashing and uses 2 GiB per worker; training profiles separately use 12 GiB per worker.'])
    receipt=dict(schema='independent_final_numerical_data_review_v1',verdict='pass',audit_source_sha256=sha(__file__),
        no_cross_gpu_view=view_review,prepared_cache=cache_review,reviewer_GPU_runs=0,reviewer_optimizer_updates=0,TEST_read=False)
    path=BASE/'review/FINAL_NUMERICAL_DATA_REVIEW.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
