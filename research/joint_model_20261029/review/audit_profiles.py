"""Audit saved complete two-epoch GPU throughput profiles without device execution."""
from pathlib import Path
import hashlib,json,math
BASE=Path('/workspace-SR003.nfs2/ARG-GNN/research/joint_model_20261029')
ROOT=BASE.parent.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
protocol_path=BASE/'root/profile_protocol_v1.json';protocol=read(protocol_path)
assert protocol['allowed_stages']==['profile'] and protocol['benchmark_training_authorized'] is False
assert protocol['profile_epochs']==2 and protocol['profile_measured_epoch']==2
assert protocol['worker_cache_bytes']==12*2**30 and protocol['workers']==4
assert protocol['live_launch_allowed'] is True
for ref in protocol['source_refs']:assert sha(ROOT/ref['path'])==ref['sha256']
results=[];initializations=set()
for batch in [16,8,32]:
    path=BASE/f'runs/profile_b{batch}_v1'
    r=read(path/'result.json');activity=read(path/'activity.json');status=read(path/'status.json')
    assert status['status']=='completed' and not (path/'failure.json').exists()
    assert r['stage']=='profile' and r['arm']=='structure' and r['batch_sites']==batch
    assert r['harness_sha256']==sha(BASE/'root/train.py') and r['protocol_sha256']==sha(protocol_path)
    assert r['manifest_sha256']==protocol['manifest_sha256']
    assert r['fit_sites']==830 and r['fit_targets']==33334 and r['planned_epochs']==2
    assert r['worker_cache_bytes']==12*2**30 and r['workers']==4
    assert r['target_presentations']==66668
    steps=math.ceil(830/batch)
    assert r['optimizer_updates']==r['planned_updates']==steps*2
    assert len(r['epoch_records'])==2 and not activity['errors'] and activity['samples']
    for epoch,record in enumerate(r['epoch_records'],1):
        assert record['epoch']==epoch and record['sites']==830 and record['targets']==33334
        assert record['updates']==epoch*steps and record['training_seconds']>0
        assert math.isfinite(record['online_training_cross_entropy']) and 0<=record['online_training_accuracy']<=1
    assert r['test_evaluated'] is False and r['benchmark_score_claimed'] is False
    assert r['gpu']==5 and r['gpu_uuid']=='GPU-3db85505-d061-e45d-d154-e7e8c1cad497'
    assert r['process_title']=='AI Auto Research Pipeline Test'
    assert all(pid==r['pid'] for sample in activity['samples'] for pid in sample['compute_pids'])
    assert math.isfinite(r['max_preclip_gradient_norm']) and r['clipped_steps']<=r['optimizer_updates']
    assert not list(path.glob('*.pt')) and not list(path.glob('*.npz'))
    initializations.add(r['initialization_sha256'])
    repeat=r['epoch_records'][1]['training_seconds']
    results.append(dict(batch_sites=batch,startup_seconds=r['startup_seconds'],cold_epoch_seconds=r['epoch_records'][0]['training_seconds'],
        repeat_epoch_seconds=repeat,repeat_targets_per_second=33334/repeat,repeat_updates_per_second=steps/repeat,
        optimizer_updates=r['optimizer_updates'],target_presentations=r['target_presentations'],
        peak_allocated_bytes=r['allocated_peak_bytes'],peak_reserved_bytes=r['reserved_peak_bytes'],
        parent_peak_RSS_bytes=r['parent_process_peak_rss_bytes'],activity_samples=len(activity['samples']),
        whole_profile_mean_device_utilization_percent=sum(s['gpu_percent'] for s in activity['samples'])/len(activity['samples']),
        sampled_foreign_compute_PIDs=[],result_sha256=sha(path/'result.json'),activity_sha256=sha(path/'activity.json')))
assert len(initializations)==1
best=max(x['repeat_targets_per_second'] for x in results)
selected=min(x['batch_sites'] for x in results if x['repeat_targets_per_second']>=0.9*best)
assert selected==16
receipt=dict(schema='independent_complete_pipeline_profile_review_v1',verdict='pass',selected_batch_sites=selected,
    selection='Smallest batch at or above 90 percent of the best observed repeat-epoch canonical-target throughput.',
    profile_execution_order=[16,8,32],protocol_sha256=sha(protocol_path),trainer_sha256=sha(BASE/'root/train.py'),
    audit_source_sha256=sha(__file__),common_initialization_sha256=initializations.pop(),profiles=results,
    limitations=['One warmed epoch per batch, fixed candidate order; launch envelope evidence, not statistical speedup.',
                'Different batch sizes imply different update counts. Profile online losses are not a batch-size accuracy selection criterion.',
                'Utilization samples cover both cold and warm work. No per-epoch device-activity boundary was saved.',
                'Host RSS is parent-only; four persistent worker LRU caps are 12 GiB each. Dataset caches immutable observations only.',
                'Offline preprocessing: 38.47 s raw export, 306.36 s prepared cache, 9,011,935,132 prepared bytes, already reported separately.',
                'Target-order digest includes minibatch serialization boundaries; differing digests across batch sizes are not evidence of differing site permutations.'],
    reviewer_GPU_runs=0,reviewer_optimizer_updates=0,TEST_read=False)
(BASE/'review/PROFILE_REVIEW.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
