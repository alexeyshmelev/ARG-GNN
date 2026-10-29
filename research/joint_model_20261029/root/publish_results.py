"""Publish reviewed development metrics and an explicit benchmark progress snapshot."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'

def main():
    assert Path.cwd()==ROOT
    review=json.loads((BASE/'review/COMPLETED_DEVELOPMENT_REVIEW.json').read_text())
    assert review['verdict']=='pass'
    selected=review['selected_arm']
    protocol=json.loads((BASE/'root/benchmark_protocol_v1.json').read_text())
    assert protocol['selected_arm']==selected
    run=BASE/'runs'/f'benchmark_{selected}_v1'
    started=json.loads((run/'started.json').read_text())
    status=json.loads((run/'status.json').read_text())
    epochs=json.loads((run/'epochs.json').read_text())
    assert started['fit_targets']==50000 and started['fit_sites']==1245
    assert started['initialization_sha256']==review['common_initialization_sha256']
    names={'no_cross':'No cross-tree connections','leaf_only':'Same-leaf connections','structure':'Branch-structure connections'}
    modes=review['modes']
    best_acc=max(v['development']['accuracy'] for v in modes.values())
    best_f1=max(v['development']['macro_f1'] for v in modes.values())
    best_ce=min(v['development']['cross_entropy'] for v in modes.values())
    rows=[]
    for arm in ('no_cross','leaf_only','structure'):
        score=modes[arm]['development']
        vals=[]
        for key,best,scale,digits in [('accuracy',best_acc,100,2),('macro_f1',best_f1,100,2),('cross_entropy',best_ce,1,4)]:
            value=f'{score[key]*scale:.{digits}f}'
            vals.append(f'**{value}**' if score[key]==best else value)
        rows.append('| '+names[arm]+' | '+' | '.join(vals)+' |')
    efficiency=modes['structure']['efficiency']
    snapshot=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_arm=selected,
        protocol_sha256=hashlib.sha256((BASE/'root/benchmark_protocol_v1.json').read_bytes()).hexdigest(),
        benchmark_run=str(run.relative_to(ROOT)),started=started,status=status,completed_epochs=len(epochs),
        last_complete_epoch=epochs[-1] if epochs else None,author_TEST_comparison_available=False)
    (BASE/'root/BENCHMARK_PROGRESS.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    updates=status.get('update',status.get('updates'))
    text=f'''<!-- joint-model-20261029:start -->
## TreeLoom joint model, 29 October

**The complete new model passed qualification and all three 50-epoch learning runs. Benchmark training is {'running' if status['status']=='training' else status['status']}.** The selected arm is **{names[selected].lower()}**. Its fresh fit uses all **50,000 original European six-class TRAIN targets** on GPU 5, with **{updates:,}/{started['planned_updates']:,} updates** recorded at this snapshot and **{len(epochs)}/50 epochs** complete. [Training snapshot](research/joint_model_20261029/root/BENCHMARK_PROGRESS.json).

The table is internal development, using 33,334 targets from two TRAIN chromosomes for fitting and 16,666 targets from the third for selection. Every arm completed 2,600 updates with the same initialization and target order. These are fixed final-epoch results, not the best interim checkpoints.

| Model arm | Development accuracy (%) | Development macro F1 (%) | Cross-entropy |
|---|---:|---:|---:|
{chr(10).join(rows)}

One held-out chromosome cannot establish population-level uncertainty or superiority over native methods. **New held-out benchmark scores remain pending.** The historical competitor TEST scores below are a different evaluation and must not be compared directly with this table.

The new PyTorch Geometric model applies three joint layers to complete native trees, passing information within and across trees without a recurrent network. All queries at the same SNP share one encoding. Preparing all 1,245 sites took 5.11 minutes with four workers and used about 9.01 GB. During the branch-structure arm's 49 warm training epochs, measured throughput was **{efficiency['warm_useful_targets_per_second']:,.0f} targets/s** and sampled GPU activity averaged **{efficiency['warm_interval_mean_device_GPU_percent']:.1f}%**, with {efficiency['warm_interval_device_activity_samples']} samples. Its first worker startup took {efficiency['first_iterator_seconds']:.1f} seconds. Total fit duration, evaluation, startup and shutdown costs remain separately reported in the audit. This is a new architecture and does not claim numerically equivalent acceleration of old TreeLoom.

Independent checks cover observations and labels, batch isolation, edge cases, gradients, CPU/GPU agreement, all saved prediction identities, restored checkpoints and worker cleanup. Native methods, simulations and historical results are preserved. Sliding joint graph windows have prior art, so no general new-kernel claim is made.

[Complete reviewed results](research/joint_model_20261029/review/COMPLETED_DEVELOPMENT_REVIEW.json) · [Plan](research/joint_model_20261029/PLAN.md) · [Architecture](research/joint_model_20261029/model/DESIGN.md) · [Data protocol](research/joint_model_20261029/data/DATA_PROTOCOL_v1.json) · [Frozen benchmark fit](research/joint_model_20261029/root/benchmark_protocol_v1.json)
<!-- joint-model-20261029:end -->'''
    p=ROOT/'README.md'
    content=p.read_text()
    start=content.index('<!-- joint-model-20261029:start -->')
    end=content.index('<!-- joint-model-20261029:end -->')+len('<!-- joint-model-20261029:end -->')
    p.write_text(content[:start]+text+content[end:])
    p=BASE/'PLAN.md'
    content=p.read_text()
    for prefix in ('Complete meaningful','Freeze the selected','Verify optimizer'):
        content=content.replace('- [ ] '+prefix,'- [x] '+prefix)
    content+='\n## Completed development and benchmark launch\n\nAll three 50-epoch development fits passed the independent saved-prediction, checkpoint, optimizer, target-order and cleanup audit. The fixed final rule selected '+selected+'. A fresh 50-epoch benchmark fit on all 50,000 original TRAIN targets is underway. Native TEST comparison, additional ancestry tasks and publication claims remain outstanding. This model does not retroactively replace historical scores.\n'
    p.write_text(content)
    p=ROOT/'research/experiment_decisions.json'
    decisions=json.loads(p.read_text())
    for item in decisions['decisions']:
        if item.get('id')=='treeloom_joint_model_complete_20261029':
            item['status']='benchmark_training_started_after_reviewed_development'
            item['conclusion']='All three 50-epoch fits passed independent review. Final-development selection chose '+selected+'. Fresh fit on all50000 original TRAIN targets is running. No TEST superiority claim.'
            item['development_metrics']={arm:entry['development'] for arm,entry in modes.items()}
            item['evidence']+=['research/joint_model_20261029/review/COMPLETED_DEVELOPMENT_REVIEW.json','research/joint_model_20261029/root/benchmark_protocol_v1.json','research/joint_model_20261029/root/BENCHMARK_PROGRESS.json']
    decisions['last_root_update_utc']=snapshot['recorded_utc']
    p.write_text(json.dumps(decisions,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,benchmark_updates=updates,completed_epochs=len(epochs),warm_efficiency=efficiency),indent=2))

if __name__=='__main__':
    main()
