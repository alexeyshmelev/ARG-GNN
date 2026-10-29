"""CPU-only independent audit of three completed development fits.

Reads frozen TRAIN cache labels and saved predictions/checkpoints. Performs a
single restored-model CPU forward per arm, never an optimizer update or TEST read.
"""
import argparse
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from research.joint_model_20261029.model.joint_model_v1 import JointAncestryModel
from research.joint_model_20261029.data.site_dataset import CachedSiteDataset

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def state_sha(state):return hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in state.values())).hexdigest()
def close(left,right):assert math.isclose(float(left),float(right),rel_tol=2e-6,abs_tol=2e-6),(left,right)

def score(logits,labels):
    z=logits.astype(np.float64);z-=z.max(axis=1,keepdims=True)
    cross_entropy=float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(len(labels)),labels]))
    predicted=logits.argmax(1)
    confusion=np.bincount(labels*6+predicted,minlength=36).reshape(6,6)
    denom=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*confusion.diagonal(),denom,out=np.zeros(6,dtype=float),where=denom>0)
    true=confusion.sum(1);pred=confusion.sum(0)
    recall=np.divide(confusion.diagonal(),true,out=np.zeros(6,dtype=float),where=true>0)
    return dict(targets=len(labels),correct=int(confusion.diagonal().sum()),accuracy=float(confusion.diagonal().sum()/len(labels)),
        cross_entropy=cross_entropy,macro_f1=float(f1.mean()),confusion=confusion.tolist(),
        true_counts=true.tolist(),predicted_counts=pred.tolist(),per_class_recall=recall.tolist(),per_class_f1=f1.tolist())

def verify_score(actual,reported):
    for key in ['targets','confusion','true_counts','predicted_counts']:assert actual[key]==reported[key],key
    for key in ['accuracy','cross_entropy','macro_f1']:close(actual[key],reported[key])

def reference_labels(manifest,split):
    identities={}
    allowed=manifest['support'][split]['parent_ordinals']
    for row in manifest['windows']:
        if row['parent_ordinal'] not in allowed:continue
        path=Path(manifest['_root'])/row['labels_ref']['path']
        assert sha(path)==row['labels_ref']['sha256']
        values=np.load(path,allow_pickle=False)
        assert values.dtype==np.int64 and len(values)==row['canonical_examples']
        for index,label in zip(row['example_indices'],values):
            key=(row['parent_id'],index)
            assert key not in identities
            identities[key]=int(label)
    assert len(identities)==manifest['support'][split]['targets']
    return identities

def load_predictions(path,expected):
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)=={'logits','labels','parent_id','example_index'}
        arrays={key:archive[key].copy() for key in archive.files}
    logits=arrays['logits'];labels=arrays['labels'];ids=list(zip(arrays['parent_id'].tolist(),arrays['example_index'].tolist()))
    assert logits.dtype==np.float32 and labels.dtype==np.int64 and arrays['example_index'].dtype==np.int64
    assert logits.shape==(len(expected),6) and labels.shape==(len(expected),)
    assert len(ids)==len(set(ids))==len(expected) and set(ids)==set(expected)
    assert np.isfinite(logits).all()
    assert all(expected[key]==int(label) for key,label in zip(ids,labels))
    return arrays,ids,score(logits,labels)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--runs',nargs=3,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    protocol_path=Path(args.protocol);protocol=read(protocol_path)
    assert protocol['live_launch_allowed'] is True
    assert protocol['allowed_stages']==['development']
    assert protocol['benchmark_training_authorized'] is False
    assert protocol['test_used_for_selection'] is False
    assert protocol['model']=={'width':64,'layers':3,'classes':6}
    assert protocol['epochs']==50 and protocol['batch_sites']==16
    assert protocol['model_seed']==101 and protocol['batch_seed']==20261032
    assert protocol['workers']==4 and protocol['worker_cache_bytes']==12*2**30
    assert protocol['learning_rate']==.001 and protocol['betas']==[.9,.999]
    assert protocol['epsilon']==1e-8 and protocol['weight_decay']==1e-4 and protocol['gradient_clip']==1
    for ref in protocol['source_refs']:assert sha(ROOT/ref['path'])==ref['sha256']
    manifest_path=Path(protocol['manifest']);assert sha(manifest_path)==protocol['manifest_sha256']
    manifest=read(manifest_path);manifest['_root']=str(manifest_path.parent)
    fit_truth=reference_labels(manifest,'fit');dev_truth=reference_labels(manifest,'development')
    assert not set(fit_truth).intersection(dev_truth)
    torch.manual_seed(101)
    initial=JointAncestryModel(**protocol['model'])
    initial_hash=state_sha(initial.state_dict());initial_state={key:value.clone() for key,value in initial.state_dict().items()}
    summaries={};orders=set();starts=set()
    for name in args.runs:
        assert Path(name).name==name
        path=BASE/'runs'/name
        result=read(path/'result.json');started=read(path/'started.json');status=read(path/'status.json')
        assert status['status']=='completed' and not (path/'failure.json').exists()
        mode=result['arm'];assert mode in ['no_cross','leaf_only','structure'] and mode not in summaries
        assert result['stage']=='development'
        assert not (path/'loader_shutdown_failure.json').exists()
        shutdown=read(path/'loader_shutdown.json')
        assert len(shutdown)==2
        assert all(row['worker_exitcodes']==[0,0,0,0] and row['iterator_shutdown'] and not row['pin_thread_alive'] for row in shutdown)
        assert result['protocol_sha256']==sha(protocol_path) and result['manifest_sha256']==sha(manifest_path)
        assert result['harness_sha256']==sha(BASE/'root/train.py')
        assert result['initialization_sha256']==initial_hash
        assert result['batch_sites']==16 and result['workers']==4 and result['worker_cache_bytes']==12*2**30
        assert result['fit_sites']==830 and result['fit_targets']==33334
        assert result['planned_epochs']==50 and result['optimizer_updates']==result['planned_updates']==2600
        assert result['target_presentations']==1666700
        assert result['test_evaluated'] is False and result['benchmark_score_claimed'] is False
        assert result['process_title']=='AI Auto Research Pipeline Test'
        assert result['gpu_uuid'] in ['GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71','GPU-3db85505-d061-e45d-d154-e7e8c1cad497']
        assert result['model_parameters']==133202 and result['precision']=={'learned_dtype':'float32','tf32_matmul':False,'tf32_cudnn':False}
        assert all(started[key]==result[key] for key in started)
        epochs=result['epoch_records'];assert len(epochs)==50
        assert read(path/'epochs.json')==epochs
        for epoch,record in enumerate(epochs,1):
            assert record['epoch']==epoch and record['updates']==epoch*52
            assert record['targets']==33334 and record['sites']==830
            assert record['training_seconds']>0 and math.isfinite(record['online_training_cross_entropy'])
            assert 0<=record['online_training_accuracy']<=1
            assert ('development' in record)==(epoch%5==0)
            if 'development' in record:
                dev=record['development'];assert dev['targets']==16666
                assert math.isfinite(dev['cross_entropy']) and 0<=dev['accuracy']<=1
                assert np.asarray(dev['confusion']).sum()==16666
        progress=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()]
        assert progress[0]['update']==1 and progress[-1]['update']==2600
        assert all(row['update']<next_row['update'] for row,next_row in zip(progress,progress[1:]))
        assert all(math.isfinite(row['preclip_gradient_norm']) and math.isfinite(row['online_training_cross_entropy']) for row in progress)
        assert math.isfinite(result['max_preclip_gradient_norm']) and 0<=result['clipped_steps']<=2600
        arrays,fit_ids,fit_score=load_predictions(path/'final_fit.npz',fit_truth)
        dev_arrays,dev_ids,dev_score=load_predictions(path/'final_development.npz',dev_truth)
        verify_score(fit_score,result['final_fit']);verify_score(dev_score,epochs[-1]['development'])
        checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        state=torch.load(path/'final_state.pt',map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==50 and checkpoint['updates']==2600 and checkpoint['protocol_sha256']==sha(protocol_path)
        assert state.keys()==checkpoint['model'].keys()==initial_state.keys()
        assert all(torch.equal(value,checkpoint['model'][key]) and torch.isfinite(value).all() for key,value in state.items())
        changed=[key for key,value in state.items() if not torch.equal(value,initial_state[key])]
        assert changed
        restored=JointAncestryModel(**protocol['model']);restored.load_state_dict(state);restored.eval()
        optimizer=torch.optim.AdamW(restored.parameters(),lr=protocol['learning_rate'],betas=tuple(protocol['betas']),eps=protocol['epsilon'],weight_decay=protocol['weight_decay'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        assert len(optimizer.state)>0
        for group in optimizer.param_groups:
            assert group['lr']==.001 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==1e-4
        for entry in optimizer.state.values():
            assert float(entry['step'])==2600
            assert all(torch.isfinite(value).all() for value in entry.values() if isinstance(value,torch.Tensor))
        sample=CachedSiteDataset(manifest_path,split='development',cache_bytes=0,mode=mode)[0]
        with torch.no_grad():value=restored(sample.observations,mode=mode).numpy()
        lookup={identity:i for i,identity in enumerate(dev_ids)}
        saved=dev_arrays['logits'][[lookup[key] for key in sample.canonical_ids]]
        diff=np.abs(value.astype(np.float64)-saved.astype(np.float64))
        assert np.all(diff<=2e-5+2e-4*np.abs(saved))
        activity=read(path/'activity.json')
        assert activity['samples'] and activity['errors']==[]
        foreign=sorted({pid for sample in activity['samples'] for pid in sample['compute_pids'] if pid!=result['pid']})
        per_phase={}
        for label in sorted({row['phase'] for row in activity['samples']}):
            samples=[row for row in activity['samples'] if row['phase']==label]
            per_phase[label]=dict(samples=len(samples),mean_device_GPU_percent=float(np.mean([row['gpu_percent'] for row in samples])),
                foreign_compute_PIDs=sorted({pid for row in samples for pid in row['compute_pids'] if pid!=result['pid']}))
        for record in epochs:
            close(record['training_seconds'],record['epoch_end_monotonic']-record['epoch_start_monotonic'])
            assert 0<=record['iterator_seconds']<=record['first_batch_seconds']<=record['training_seconds']
        assert all(left['epoch_end_monotonic']<=right['epoch_start_monotonic'] for left,right in zip(epochs,epochs[1:]))
        warm_samples=[sample for sample in activity['samples'] if any(epoch['epoch_start_monotonic']<=sample['monotonic']<epoch['epoch_end_monotonic'] for epoch in epochs[1:])]
        assert warm_samples
        warm_seconds=sum(epoch['training_seconds'] for epoch in epochs[1:])
        efficiency=dict(first_epoch_seconds=epochs[0]['training_seconds'],first_iterator_seconds=epochs[0]['iterator_seconds'],
            first_batch_seconds=epochs[0]['first_batch_seconds'],warm_epochs=49,warm_training_seconds=warm_seconds,
            warm_useful_targets_per_second=49*33334/warm_seconds,warm_updates_per_second=49*52/warm_seconds,
            total_fit_presentations_per_run_elapsed_second=1666700/result['elapsed_seconds'],
            warm_interval_device_activity_samples=len(warm_samples),
            warm_interval_mean_device_GPU_percent=float(np.mean([row['gpu_percent'] for row in warm_samples])),
            sampled_device_attribution='shared_device' if foreign else 'exclusive_at_sample_times',
            per_phase_device_samples=per_phase,phase_seconds=result['phase_seconds'],
            peak_allocated_bytes=result['allocated_peak_bytes'],peak_reserved_bytes=result['reserved_peak_bytes'],
            parent_peak_RSS_bytes=result['parent_process_peak_rss_bytes'],
            caveat='Warm epoch intervals include loader time. Device utilization is sampled and device-wide. Shared GPU4 arms are not isolated speed comparisons; CPU wait and CUDA event times overlap.')
        orders.add(result['target_order_sha256']);starts.add(result['initialization_sha256'])
        summaries[mode]=dict(run=name,fit=fit_score,development=dev_score,optimizer_updates=2600,target_presentations=1666700,
            active_optimizer_parameter_tensors=len(optimizer.state),changed_parameter_tensors=len(changed),changed_parameter_names=changed,checkpoint_finite_and_exact=True,
            loader_shutdown_passed=True,efficiency=efficiency,
            restored_cpu_sample_targets=len(sample.labels),restored_cpu_saved_gpu_max_absolute_logit_difference=float(diff.max()),
            elapsed_seconds=result['elapsed_seconds'],epoch_training_seconds=sum(row['training_seconds'] for row in epochs),
            clipped_steps=result['clipped_steps'],maximum_preclip_norm=result['max_preclip_gradient_norm'],
            sampled_foreign_compute_PIDs=foreign,activity_samples=len(activity['samples']),
            result_sha256=sha(path/'result.json'),checkpoint_sha256=sha(path/'checkpoint.pt'),state_sha256=sha(path/'final_state.pt'),
            fit_predictions_sha256=sha(path/'final_fit.npz'),development_predictions_sha256=sha(path/'final_development.npz'),
            activity_sha256=sha(path/'activity.json'))
    assert set(summaries)=={'no_cross','leaf_only','structure'} and len(orders)==len(starts)==1
    ranking=sorted(summaries,key=lambda mode:(-summaries[mode]['development']['correct'],summaries[mode]['development']['cross_entropy']))
    receipt=dict(schema='independent_completed_development_review_v1',verdict='pass',selected_arm=ranking[0],ranking=ranking,
        selection_rule='Greatest final development canonical-target accuracy, then lower CE for an exact accuracy tie.',
        protocol_sha256=sha(protocol_path),audit_source_sha256=sha(__file__),trainer_sha256=sha(BASE/'root/train.py'),
        manifest_sha256=sha(manifest_path),common_initialization_sha256=starts.pop(),common_training_order_sha256=orders.pop(),
        modes=summaries,reviewer_GPU_runs=0,reviewer_optimizer_updates=0,TEST_read=False,
        limitations=['Selection uses one held-out TRAIN chromosome; correlated target rows do not provide independent population-level evidence.',
                    'Sampled process lists are reported without assuming exclusivity if foreign PIDs are present.',
                    'Restored final model and optimizer states were loaded on CPU; no resumed optimizer step was run.'])
    output=BASE/'review/COMPLETED_DEVELOPMENT_REVIEW.json'
    with output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
